import numpy as np
import larq as lq
import tensorflow as tf

from sw.helper import make_kernels


def binary_quantization(x):
    """ Quantizes the input to -1 and +1 based on sign. """
    return np.where(x >= 0, 1, -1)


class MyModel:
    def __init__(self):
        self.layers = []
        self.outputs = []
        self.prediction = None

        # NN Topology
        self.kwargs = dict(input_quantizer="ste_sign",
                      kernel_quantizer="ste_sign",
                      kernel_constraint="weight_clip",
                      use_bias=False)

        self.larq_model = tf.keras.models.Sequential()

    def add(self, layer):
        if isinstance(layer, lq.layers.QuantConv2D):
            self.larq_model.add(layer)
            self.layers.append(Conv2D(make_kernels(layer.get_weights()), layer.input_shape[1:]))
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            pass
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            pass
        elif isinstance(layer, tf.keras.layers.Flatten):
            pass
        elif isinstance(layer, lq.layers.QuantDense):
            pass


    def predict(self, input):
        interm_input = [input]
        for n, layer in enumerate(self.layers):
            interm_input = layer.inference(interm_input)

            self.outputs.append(interm_input)

        self.prediction = interm_input
        return self.prediction


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


class Flatten(MyLayer):
    def __init__(self, input_shape):
        MyLayer.__init__(self, input_shape)

    def inference(self, interm_input):
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

        # 2d batchnorm:
        for n, image in enumerate(input):
            X_normalized = (image - self.mean[n]) / np.sqrt(self.var[n])
            self.output.append(X_normalized + self.beta[n])

        return self.output


# https://github.com/lebrice/VHDL-CPU/blob/master/Final%20Deliverable/processor/decodeStage/decodeStage.vhd
def convolve2D(input_mat, kernel_mat):
    """
  Perform the 2-D convolution operation.

  :input_mat: the input matrix.
  :kernel_mat: the kernel matrix used for convolution.
  """

    # Ensure none of the inputs are empty.
    if input_mat.size == 0 or kernel_mat.size == 0:
        raise Exception("Error! Empty matrices found.")

    # Ensure the input is a square matrix.
    if input_mat.shape[0] != input_mat.shape[1]:
        raise Exception("Error! The input is not a square matrix.")

    # Ensure the kernel is a square matrix.
    if kernel_mat.shape[0] != kernel_mat.shape[1]:
        raise Exception("Error! The kernel is not a square matrix.")

    # Get the size of the input and kernel matrices.
    input_size = input_mat.shape[0]
    kernel_size = kernel_mat.shape[0]

    # Ensure the kernel is not larger than the input matrix.
    if input_size < kernel_size:
        raise Exception("Error! The kernel is larger than the input.")

    # Flip the kernel.
    kernel_mat = kernel_mat

    # Set up the output matrix.
    output_size = (input_size - kernel_size) + 1
    output_mat = np.zeros(shape=(output_size, output_size))

    row_offset = 0

    for output_row in range(output_size):
        col_offset = 0

        for output_col in range(output_size):
            kernel_row = 0

            for row in range(row_offset, row_offset + kernel_size):
                kernel_col = 0

                for col in range(col_offset, col_offset + kernel_size):
                    # Perform the convolution computation.
                    output_mat[output_row][output_col] += kernel_mat[kernel_row][kernel_col].item() * input_mat[row][
                        col].item()
                    kernel_col += 1

                kernel_row += 1

            col_offset += 1

        row_offset += 1

    return output_mat
