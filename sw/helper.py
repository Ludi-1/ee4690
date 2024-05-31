import keras
import matplotlib.pyplot as plt
import numpy as np
import larq as lq
# from tensorflow.python.estimator import keras

from sw.simulator import MyModel, Conv2D, Flatten, Quantdense, BatchNormalization, Maxpool


# Helper functions:
def print_image(image):
    # Squeeze the third dimension or you can use indexing to select the first slice
    image_2d = np.squeeze(image)

    # Plotting the image
    plt.imshow(image_2d, cmap='gray')  # Use the gray colormap for grayscale
    plt.colorbar()  # Optionally add a colorbar to see the intensity scale
    plt.show()


def get_filters(weights):
    weights = weights[0]
    filters = []
    for i in range(0, 2):
        filter = [[], [], []]
        for n, row in enumerate(weights):
            for m, col in enumerate(row):
                filter[n].append(col[0][i])
        filters.append(filter)
    return np.array(filters)


def get_output(output):
    output = output[0]
    outputs = []
    for i in range(0, output.shape[2]):
        new_output = []
        for n, row in enumerate(output):
            new_output.append([])
            for m, col in enumerate(row):
                new_output[n].append(col[i])
        outputs.append(new_output)
    return np.array(outputs)


def plot_intermediate_results(intermediate_output, title):
    length = len(intermediate_output)
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, length)  # 2 rows, 2 columns

    for i, result in enumerate(intermediate_output):
        # Squeeze the third dimension or you can use indexing to select the first slice
        image_2d = np.squeeze(result)

        # Plotting the image
        if length > 1:
            axs[i].imshow(image_2d, cmap='gray')  # Use the gray colormap for grayscale
            axs[i].set_title(f"{title}")  # Set title for each subplot
        else:
            axs.imshow(image_2d, cmap='gray')  # Use the gray colormap for grayscale
            axs.set_title(f"{title}")  # Set title for the single subplot

    # Adjust layout to prevent overlap
    fig.tight_layout()
    # Display the figure
    plt.show()


def plot_differences(list1, list2):
    # Ensure both lists have the same length by trimming the longer list
    # Calculate the differences
    print(f"larq: {list1}")
    # print(f"sim: {list2}")
    differences = [absolute_difference(element, list2[n]) for n, element in enumerate(list1[0])]

    # Plot the differences
    plt.figure()

    # plt.plot(list1[0], linestyle='-', marker='o', label = 'Larq')
    # plt.plot(list2, linestyle='-', marker='o', label = 'Simulation')
    plt.plot(differences, linestyle='-', marker='o', label='Difference')
    plt.title('Differences Between Larq and Simulation Outputs')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def absolute_difference(num1, num2):
    if num1 == num2:
        return 0
    elif num1 > num2:
        return num1 - num2
    else:
        return num2 - num1


def make_kernels(kernels):
    # Discard Bias
    kernels = kernels[0]
    (n_rows, n_columns, n_features, n_kernels) = kernels.shape

    # Channel, kernels, rows, columns
    weights = np.empty((n_kernels, n_features, n_rows, n_columns))

    for r, rows in enumerate(kernels):
        for c, column in enumerate(rows):
            for f, channel in enumerate(column):
                for e, element in enumerate(channel):
                    weights[e][f][r][c] = element
    return weights


def retrieve_weights(model):
    weights = []
    layers = []
    with lq.context.quantized_scope(True):
        model.save("binary_model.h1")  # save binary weights
        for layer in model.layers:
            layers.append(type(layer))
            weights.append(np.sign(layer.get_weights()))

    return layers, weights


def setup_sim(weights, layers, quantdense_sizes):
    my_model = MyModel()
    quantdense_layers = 0

    for n, layer in enumerate(layers):
        if layer == "cn":
            if n == 0:
                my_model.add(Conv2D(make_kernels(weights[n]), (28, 28)))
            else:
                input_shape = my_model.layers[n - 1].output_shape
                my_model.add(Conv2D(make_kernels(weights[n]), input_shape))
        elif layer == "bn":
            input_shape = my_model.layers[n - 1].output_shape
            my_model.add(BatchNormalization(input_shape, weights[n]))
        elif layer == "mp":
            input_shape = my_model.layers[n - 1].output_shape
            my_model.add(Maxpool((2, 2), input_shape))
        elif layer == "fc":
            input_shape = my_model.layers[n - 1].output_shape
            my_model.add(Quantdense(input_shape, quantdense_sizes[quantdense_layers], np.array(weights[n])))
            quantdense_layers += 1
        elif layer == "fl":
            input_shape = my_model.layers[n - 1].output_shape
            my_model.add(Flatten(input_shape))
    return my_model


def check_result(a, b):
    if np.all(a == b):
        print("OUTPUTS ARE THE SAME")
    else:
        print("ERROR WRONG ANSWER")
