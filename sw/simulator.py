import numpy as np
import larq as lq
import tensorflow as tf
from helper import make_kernels, convolve2D
import pandas as pd
import keras

import tensorflow_model_optimization as tfmot


def binary_quantization(x):
    """ Quantizes the input to -1 and +1 based on sign. """
    x[x > 0] = 1
    x[x < 0] = -1
    return x


class MyModel:
    def __init__(self, prune=None):
        self.num_pre_pruned_weights = 0
        self.num_pruned_weights = 0
        self.weight_info = {"name": [], "1-bit": [], "32-bit": []}
        self.prune = prune
        self.layers = []
        # self.outputs = []
        self.prediction = None
        self.output_size = None
        # NN Topology
        self.kwargs = dict(input_quantizer="ste_sign",
                           kernel_quantizer="ste_sign",
                           kernel_constraint="weight_clip",
                           use_bias=False)

        self.larq_model = keras.models.Sequential()

        self.structure = []

    def add(self, layer):
        self.larq_model.add(layer)

        if isinstance(layer, lq.layers.QuantConv2D):
            weights = layer.get_weights()
            self.layers.append(Conv2D(make_kernels(np.sign(weights)), layer.input_shape[1:]))
            struct = {"name": "layer_" + "conv_" + str(np.array(self.layers[-1].weights).shape),
                      "input_shape": layer.input_shape,
                      "output_shape": layer.output_shape,
                      "weights": [],
                      "type": "1-bit"}
            self.structure.append(struct)

        elif isinstance(layer, keras.layers.MaxPooling2D):
            input_shape = self.layers[-1].output_shape
            self.layers.append(Maxpool(layer.pool_size, input_shape))
            struct = {"name": "layer_" + "maxpool_" + str(layer.pool_size[0]) + "_" + str(layer.pool_size[1]),
                      "input_shape": layer.input_shape,
                      "output_shape": layer.output_shape,
                      "weights": None,
                      "type": None}
            self.structure.append(struct)

        elif isinstance(layer, keras.layers.BatchNormalization):
            weights = layer.get_weights()
            input_shape = self.layers[-1].output_shape
            self.layers.append(BatchNormalization(input_shape, weights))
            struct = {"name": "layer_" + "batchnorm",
                      "input_shape": layer.input_shape,
                      "output_shape": layer.output_shape,
                      "weights": [],
                      "type": "32-bit"}
            self.structure.append(struct)

        elif isinstance(layer, keras.layers.Flatten):
            if len(self.layers) == 0:
                input_shape = layer.input_shape
            else:
                input_shape = self.layers[-1].output_shape
            self.layers.append(Flatten(input_shape))
            struct = {"name": "layer_" + "flatten",
                      "input_shape": layer.input_shape,
                      "output_shape": layer.output_shape,
                      "weights": None,
                      "type": None}
            self.structure.append(struct)

        elif isinstance(layer, lq.layers.QuantDense):
            weights = layer.get_weights()
            input_shape = self.layers[-1].output_shape
            units = layer.units
            self.layers.append(Quantdense(input_shape, units, np.sign(weights)))

            struct = {"name": "layer_" + "fc",
                      "input_shape": layer.input_shape,
                      "output_shape": layer.output_shape,
                      "weights": [],
                      "type": "1-bit"}
            self.structure.append(struct)

        elif isinstance(layer, tf.keras.layers.Activation):
            input_shape = self.layers[-1].output_shape
            self.layers.append(Softmax(input_shape))
            struct = {"name": "layer_" + "softmax",
                      "input_shape": layer.input_shape,
                      "output_shape": layer.output_shape,
                      "weights": None,
                      "type": None}
            self.structure.append(struct)

    def predict(self, input):
        interm_input = [input]
        for n, layer in enumerate(self.layers):
            interm_input = layer.inference(interm_input)

        self.prediction = interm_input
        return self.prediction

    def compile(self, optimizer, loss, metrics):
        self.larq_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_images, train_labels, batch_size, epochs):
        self.larq_model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)
        self.update_weights()

    def update_weights(self):
        for i, struct in enumerate(self.structure):
            if struct["weights"] is not None:
                non_pruned_weights = np.array(self.larq_model.layers[i].get_weights())
                num_pruned_weights = non_pruned_weights.size
                # If pruning is enables; Set the insignificant weights to zero in keras and larq model.
                if self.prune is not None and struct["name"] == "layer_fc":
                    # Prune insignificant weights
                    pruned_weights = non_pruned_weights.copy()
                    pruned_weights[
                        (non_pruned_weights >= -self.prune) & (non_pruned_weights < self.prune)] = 0

                    # Update the weights in the larq model; This does not work currently
                    self.larq_model.layers[i].set_weights(pruned_weights)
                   
                    # Count the number of non zero weigths
                    num_pruned_weights = np.count_nonzero(pruned_weights)
                
                # Retreive the pruned weigths from the keras model and apply it it the costom model
                weights = self.larq_model.layers[i].get_weights()
                self.layers[i].set_weights(weights)
                self.structure[i]["weights"] = self.layers[i].weights
                
                # Get pruning info
                num_pruned_weights_1_bit = 0
                num_pruned_weights_32_bit = 0
                num_weights_1_bit = 0
                num_weights_32_bit = 0
                if self.structure[i]["type"] == "1-bit":
                    num_pruned_weights_1_bit =  num_pruned_weights
                    num_weights_1_bit =  np.array(self.structure[i]["weights"]).size
                elif self.structure[i]["type"] == "32-bit":
                    num_pruned_weights_32_bit =  num_pruned_weights
                    num_weights_32_bit =  np.array(self.structure[i]["weights"]).size
                
                self.weight_info["name"].append(self.structure[i]["name"])
                self.weight_info["1-bit"].append((num_weights_1_bit, num_pruned_weights_1_bit))
                self.weight_info["32-bit"].append((num_weights_32_bit, num_pruned_weights_32_bit))

    # Get the accuracy info of the keras model
    def evaluate(self, test_images, test_labels):
        return self.larq_model.evaluate(test_images, test_labels)

    # Perform single inference of keras model
    def predict_larq(self, input):
        return self.larq_model.predict(np.array([input]))

    # Test the accuracy of the costum model
    def test_accuracy(self, test_images, test_labels):
        number_of_tests = test_images.shape[0]

        n_correct = 0
        for i, test in enumerate(test_images):
            if np.argmax(self.predict(test)) == test_labels[i]:
                n_correct += 1
        return n_correct / number_of_tests

    def get_weight_info(self):
        weight_info = pd.DataFrame(self.weight_info)

        total = {"1-bit": (0,0), "32-bit": (0,0)}
        for bit_type in ["1-bit", '32-bit']: 
            for (non_pruned, pruned) in weight_info[bit_type]:
                total[bit_type] = (total[bit_type][0] + non_pruned, total[bit_type][1] + pruned)
        total = pd.DataFrame(total)

        reduction = (1-(total["1-bit"][1] /  total["1-bit"][0])), (1-(total["32-bit"][1] /  (total["32-bit"][0])))
        # if self.prune is not None:
        #     print("The total number of pre pruned weights are:" + str(self.num_pre_pruned_weights))
        #     print("The total number of post pruned weights are:" + str(self.num_pruned_weights))
        # else:
        #     pass
        return weight_info, total, reduction

    def simulate(self, inputs):
        results = []
        for input in inputs:
            my_model_result = self.predict(input)
            larq_model_result = self.predict_larq(input)

            # print("my model: " + str(my_model_result))
            # print("keras model: " + str(larq_model_result))

            result = np.all(np.isclose(np.array(my_model_result), np.array(larq_model_result), rtol=1e-02, atol=1e-08,
                                       equal_nan=False))
            results.append(result)
        valid = np.all(results)
        return valid, results


class MyLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = []
        self.output = []


"""
       channels is an 4-D Array with the following format:
           (input_channel_index, output_channel_index, kernel_row_index, kernel_column_index)
       weights is an 2-D Array with the following format:
           (row_index, column_index)
       output is an 3-D Array with the following format:
           (output_channel_index, kernel_row_index, kernel_column_index)

       The output equation is as follows:
           output[k] += convolve2D(channels[s], weights[k][s])

           or 

           output[k] = convolve2D(channels[0], weights[k][0]) + convolve2D(channels[1], weights[k][1]) + ...
"""


class Conv2D(MyLayer):
    def __init__(self, kernels, input_shape):
        MyLayer.__init__(self, input_shape)
        self.weights = kernels
        self.output_shape = (self.input_shape[0] - kernels.shape[2] + 1, self.input_shape[1] - kernels.shape[2] + 1)

    def inference(self, channels):
        temp = np.zeros((self.weights.shape[0], self.output_shape[0], self.output_shape[0]))
        channels = binary_quantization(channels)

        for k, channel_kernel in enumerate(self.weights):
            for s, _ in enumerate(channel_kernel):
                temp[k] += convolve2D(channels[s], self.weights[k][s])

        self.output = temp
        return self.output

    def set_weights(self, weights):
        self.weights = make_kernels(np.sign(weights))


class Flatten(MyLayer):
    def __init__(self, input_shape):
        MyLayer.__init__(self, input_shape)
        self.output_shape = tuple([np.array(input_shape).size])

    def inference(self, interm_input):
        self.output = []
        for i, row in enumerate(interm_input[0]):
            for j, col in enumerate(row):
                for m, image in enumerate(interm_input):
                    self.output.append(interm_input[m][i][j])
        return self.output


class Softmax(MyLayer):
    def __init__(self, input_shape):
        MyLayer.__init__(self, input_shape)
        self.output_shape = input_shape

    def inference(self, interm_input):
        self.output = tf.nn.softmax(interm_input).numpy()
        return self.output


class Maxpool(MyLayer):
    def __init__(self, kernel_shape, input_shape):
        MyLayer.__init__(self, input_shape)
        self.kernel_shape = kernel_shape
        self.output_shape = (self.input_shape[0] // self.kernel_shape[0], self.input_shape[1] // self.kernel_shape[1])

    def inference(self, interm_input):
        # Convert the input list of lists to a NumPy array
        n = self.kernel_shape[0]
        input_array = np.array(interm_input)
        self.output = []
        # Get the dimensions of the input array
        l, h, w = input_array.shape

        # Calculate the dimensions of the output array
        new_h = h // n
        new_w = w // n

        # Perform max pooling
        for m in range(l):
            # Initialize the output array
            output_array = np.zeros((new_h, new_w))
            for i in range(new_h):
                for j in range(new_w):
                    # Define the region of the input array for the current kernel position
                    h_start = i * n
                    h_end = h_start + n
                    w_start = j * n
                    w_end = w_start + n

                    # Extract the region
                    region = input_array[m][h_start:h_end, w_start:w_end]

                    # Compute the max value in the region and assign it to the output list
                    output_array[i, j] = np.max(region)
            self.output.append(output_array)
        return self.output


class Quantdense(MyLayer):
    def __init__(self, input_shape, units, weights):
        MyLayer.__init__(self, input_shape)
        self.units = units
        self.weights = weights
        self.output_shape = tuple([units])

    """
    input is an 1-D Array
    weights is an 2-D Array
    output is an 1-D array
    
    equation ->
    output[i] = input[i] dot weights[i]
    """

    def inference(self, interm_input):
        # Convert inputs to a numpy array
        input_array = np.array(interm_input)

        # Quantize the inputs
        quantized_inputs = binary_quantization(input_array)
        # Compute the dot product of quantized inputs and weights
        self.output = np.dot(np.resize(quantized_inputs, (quantized_inputs.shape[0],)), self.weights.squeeze(0))

        return self.output

    def set_weights(self, weights):
        self.weights = np.array(np.sign(weights))


"""
    Learnable parameters: Epsilon, Gamma
    Not learnable parameters: Mean, Variance
    input is an 1-D Array
    weights is an 2-D Array
    output is an 1-D array

    equation ->
    output[i] = input[i] dot weights[i]
"""


class BatchNormalization(MyLayer):
    def __init__(self, input_shape, weights):
        MyLayer.__init__(self, input_shape)
        # self.beta = weights[0]
        # self.mean = weights[1]
        # self.var = weights[2]
        self.weights = weights
        self.output_shape = input_shape

    def inference(self, interm_input):
        input = np.array(interm_input)
        self.output = []

        # 2d batchnorm:
        for n, image in enumerate(input):
            """
            output = self.weights[0][n] + image - self.weights[1][n]) / np.sqrt(self.weights[2][n]
            self.weights[0][n]- self.weights[1][n]) / np.sqrt(self.weights[2][n] + image > 0
            - self.weights[0][n] + self.weights[1][n]) / np.sqrt(self.weights[2][n] < image
            """

            x_normalized = (image - self.weights[1][n]) / np.sqrt(self.weights[2][n])
            self.output.append(x_normalized + self.weights[0][n])

        return self.output

    def set_weights(self, weights):
        self.weights = weights
