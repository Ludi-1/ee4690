import numpy as np
import larq as lq
import tensorflow as tf
from sw.helper import make_kernels, convolve2D
import keras


def binary_quantization(x):
    """ Quantizes the input to -1 and +1 based on sign. """
    return np.where(x >= 0, 1, -1)


class MyModel:
    def __init__(self):
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

    def add(self, layer):
        self.larq_model.add(layer)

        if isinstance(layer, lq.layers.QuantConv2D):
            weights = layer.get_weights()
            self.layers.append(Conv2D(np.sign(make_kernels(np.sign(weights))), layer.input_shape[1:]))

        elif isinstance(layer, keras.layers.MaxPooling2D):
            input_shape = self.layers[-1].output_shape
            self.layers.append(Maxpool(layer.pool_size, input_shape))

        elif isinstance(layer, keras.layers.BatchNormalization):
            weights = layer.get_weights()
            input_shape = self.layers[-1].output_shape
            self.layers.append(BatchNormalization(input_shape, weights))

        elif isinstance(layer, keras.layers.Flatten):
            input_shape = self.layers[-1].output_shape
            self.layers.append(Flatten(input_shape))

        elif isinstance(layer, lq.layers.QuantDense):
            weights = layer.get_weights()
            input_shape = self.layers[-1].output_shape
            units = layer.units
            self.layers.append(Quantdense(input_shape, units, np.sign(weights)))

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
        for i, layer in enumerate(self.layers):
            if layer.has_weights():
                weights = self.larq_model.layers[i].get_weights()
                layer.set_weigths(weights)

    def evaluate(self, test_images, test_labels):
        return self.larq_model.evaluate(test_images, test_labels)

    def get_weights(self):
        pass

    def predict_larq(self, input):
        return self.larq_model.predict(np.array([input]))

    def simulate(self, inputs):
        results = []
        for input in inputs:

            my_model_result = self.predict(input)
            larq_model_result = self.predict_larq(input)

            result = np.all(np.isclose(np.array(my_model_result), np.array(larq_model_result), rtol=1e-03, atol=1e-08, equal_nan=False))
            results.append(result)
        valid = np.all(results)
        return (valid, results)

class MyLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = []
        self.output = []


class Conv2D(MyLayer):
    def __init__(self, kernels, input_shape):
        MyLayer.__init__(self, input_shape)
        self.kernels = kernels
        self.output_shape = (self.input_shape[0] - kernels.shape[2] + 1, self.input_shape[1] - kernels.shape[2] + 1)

    def inference(self, channels):
        temp = np.zeros((self.kernels.shape[0], self.output_shape[0], self.output_shape[0]))
        channels = np.sign(channels)

        for k, channel_kernel in enumerate(self.kernels):
            for s, _ in enumerate(channel_kernel):
                temp[k] += convolve2D(channels[s], self.kernels[k][s])

        self.output = temp
        return self.output

    def set_weights(self, weights):
        self.kernels = make_kernels(weights)


class Flatten(MyLayer):
    def __init__(self, input_shape):
        MyLayer.__init__(self, input_shape)

    def inference(self, interm_input):
        self.output = []
        for i, row in enumerate(interm_input[0]):
            for j, col in enumerate(row):
                for m, image in enumerate(interm_input):
                    self.output.append(interm_input[m][i][j])
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
        self.output_shape = (1, units)

    def inference(self, interm_input):
        # Convert inputs to a numpy array
        input_array = np.array(interm_input)

        # Quantize the inputs
        quantized_inputs = binary_quantization(input_array)

        # Compute the dot product of quantized inputs and weights
        self.output = np.dot(quantized_inputs, self.weights.squeeze(0))

        return self.output

    def set_weights(self, weights):
        self.weights = np.array(weights)

# Learnable parameters: Epsilon, Gamma
# Not learnable parameters: Mean, Variance
class BatchNormalization(MyLayer):
    def __init__(self, input_shape, weights):
        MyLayer.__init__(self, input_shape)
        self.beta = weights[0]
        self.mean = weights[1]
        self.var = weights[2]
        self.output_shape = input_shape

    def inference(self, interm_input):
        input = np.array(interm_input)
        self.output = []

        # 2d batchnorm:
        for n, image in enumerate(input):
            X_normalized = (image - self.mean[n]) / np.sqrt(self.var[n])
            self.output.append(X_normalized + self.beta[n])

        return self.output

    def set_weights(self, weights):
        self.weights = weights
