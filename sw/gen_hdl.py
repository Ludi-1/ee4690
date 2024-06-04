"""
1. Define NN topology
2. Train NN
3. Generate HDL
"""

# Libraries
import larq as lq
import tensorflow as tf
import numpy as np
import os

import templates

# Image dataset
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

model = tf.keras.models.Sequential()

input_shape = (28, 28, 1) # Input img shape
filters_a = 32 # Number of output channels
kernel_three = (4, 4) # Kernel dimension

filters_b = 32 # Number of output channels
kernel_b = (3, 3) # Kernel dimension

# Prepare dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# print_image(train_images[0])
# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

model.add(lq.layers.QuantConv2D(filters_a, kernel_three,
                                input_quantizer="ste_sign",
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantConv2D(filters_b, kernel_b,
                                input_quantizer="ste_sign",
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Flatten())
model.add(lq.layers.QuantDense(128, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))

# model.add(tf.keras.layers.Flatten())
# # model.add(lq.layers.QuantDense(500, use_bias=False, **kwargs))
# model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
# model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

output_shapes = [layer.output_shape for layer in model.layers]
 
heights = []
widths = []
channels = []
 
for shape in output_shapes:
    if len(shape) == 4:  
        _, height, width, channel = shape
        heights.append(height)
        widths.append(width)
        channels.append(channel)
    elif len(shape) == 2:  
        _, channel = shape
        heights.append(None)
        widths.append(None)
        channels.append(channel)

model.fit(train_images, train_labels, batch_size=64, epochs=6)

test_loss, test_acc = model.evaluate(test_images, test_labels)
lq.models.summary(model)

if not os.path.exists("gen_hdl"):
    os.mkdir("gen_hdl")
  
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        beta, moving_mean, moving_variance = layer.get_weights()
        # print(f"Layer: {layer.name}")
        # print(f"  Beta (offset): {beta}")
        # print(f"Beta Length: {len(beta)}")
        # print(f"  Moving Mean: {moving_mean}")
        # print(f" Moving Mean Length: {len(moving_mean)}")
        # print(f"  Moving Variance: {moving_variance}")
        # print(f"  Moving Variance Length: {len(moving_variance)}")


### PARSE FC FUNC
betas = []
moving_means = []
moving_variances = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        beta, moving_mean, moving_variance = layer.get_weights()
        betas.append(beta)
        moving_means.append(moving_mean)
        moving_variances.append(moving_variance)
      
def parse_bn(beta, moving_mean, moving_variance, num: int):

    # thresholds = np.zeros(len(beta))
    compare = ""
    for output_neuron in range(len(beta)):
        # print(len(beta))
        threshold = moving_mean[output_neuron] - beta[output_neuron] * np.sqrt(moving_variance[output_neuron])
        compare += f"   assign o_data[{output_neuron}] = i_data[{output_neuron}] > {threshold} ? 1 : 0;\n"

    output_hdl = templates.BN_TEMPLATE \
        .replace("%DIM_DATA%", str(len(beta))) \
        .replace("%LAYER_NUM%", str(num)) \
        .replace("%COMPARE%", compare)
        
    with open(f"gen_hdl/bn_layer_{num}.v", "w") as f:
        f.write(output_hdl)

  
# for n in range(len(betas)):
#     parse_bn(betas[n], moving_means[n], moving_variances[n], n)

def parse_conv(conv_weights, num : int):

    kernel_size, kernel_size, input_channels, output_channels = conv_weights.shape
    conv_weights[conv_weights == -1] = 0
    conv_weight = np.reshape(conv_weights, (kernel_size**2, input_channels, output_channels))
    print(conv_weight.shape)
    buffer = ""
    xnor = ""
    for i in range(input_channels):
        buffer += f"""ibuf_conv #(
                        .img_width(INPUT_DIM),
                        .kernel_dim(KERNEL_DIM),
                    ) ibuf (
                        .clk(clk),
                        .i_we(i_we),
                        .i_data(i_data[{i}]),
                        .o_data(window[{i}]),
                    );\n"""

    for i in range(output_channels):
        for j in range(input_channels):
            for k in range(kernel_size**2):
                weight = conv_weight[k, j, i]
                if weight == 0:
                    xnor += f"assign temp[{i*output_channels+j*input_channels+k}] = ~window[{j}][{k}];\n" 
                elif weight == 1:
                    xnor += f"assign temp[{i*output_channels+j*input_channels+k}] = window[{j}][{k}];\n" 
                else:
                    raise Exception(f"neuron value not 0 or 1: {weight}") 
            
    
    output_hdl = templates.CONV_TEMPLATE \
        .replace("%BUFFER%", buffer) \
        .replace("%LAYER_NUM%", str(num)) \
        .replace("%XNOR%", xnor)
    with open(f"gen_hdl/conv_layer_{num}.sv", "w") as f:
        f.write(output_hdl)
    

# Extract weights
with lq.context.quantized_scope(True):
    weights = model.layers[3].get_weights()

parse_conv(weights[0], 3)