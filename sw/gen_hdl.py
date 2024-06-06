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

from sw import templates

# Image dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

# NN Topology
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(lq.layers.QuantDense(128*5, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(128*2, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=64, epochs=6)
test_loss, test_acc = model.evaluate(test_images, test_labels)
lq.models.summary(model)

if not os.path.exists("gen_hdl"):
    os.mkdir("gen_hdl")

### PARSE FC FUNC
def parse_fc(fc_weights, num: int):
    fc_weights[fc_weights == -1] = 0
    fc_weights = fc_weights.T
    xnor = ""
    weight_dim = fc_weights.shape
    for output_neuron in range(weight_dim[0]):
        for input_neuron in range(weight_dim[1]):
            weight = fc_weights[output_neuron][input_neuron]
            if weight == 0:
                xnor += f"assign xnor_result[{input_neuron}][{output_neuron}] = ~i_data[{input_neuron}];\n"
            elif weight == 1:
                xnor += f"assign xnor_result[{input_neuron}][{output_neuron}] = i_data[{input_neuron}];\n"
            else:
                raise Exception(f"neuron value not 0 or 1: {input_neuron}")

    output_hdl = templates.FC_TEMPLATE \
        .replace("%XNOR_GEN%", xnor) \
        .replace("%LAYER_NUM%", str(num)) \
        .replace("%INPUT_DIM%", str(weight_dim[1])) \
        .replace("%OUTPUT_DIM%", str(weight_dim[0]))
    with open(f"gen_hdl/L{num}_fc.v", "w") as f:
        f.write(output_hdl)
    return (weight_dim[1], weight_dim[0])

### PARSE CONV
heights = []
widths = []
channels = []
output_shapes = [layer.output_shape for layer in model.layers]

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

def parse_conv(conv_weights, num : int):

    kernel_size, kernel_size, input_channels, output_channels = conv_weights.shape
    conv_weights[conv_weights == -1] = 0
    conv_weight = np.reshape(conv_weights, (kernel_size**2, input_channels, output_channels))
    print(conv_weight.shape)
    buffer = ""
    xnor = ""

    for i in range(output_channels):
        for j in range(input_channels):
            for k in range(kernel_size**2):
                weight = conv_weight[k, j, i]
                if weight == 0:
                    xnor += f"assign xnor[{i}][{j}][{k}] = ~window[{j}][{k}];\n" 
                elif weight == 1:
                    xnor += f"assign xnor[{i}][{j}][{k}] = window[{j}][{k}];\n" 
                else:
                    raise Exception(f"neuron value not 0 or 1: {weight}") 
            
    
    output_hdl = templates.CONV_TEMPLATE \
        .replace("%BUFFER%", buffer) \
        .replace("%LAYER_NUM%", str(num)) \
        .replace("%XNOR%", xnor)
    with open(f"gen_hdl/L{num}_conv.v", "w") as f:
        f.write(output_hdl)

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
        compare += f"   assign o_data[{output_neuron}] = i_data[{output_neuron}] > {round(threshold)} ? 1 : 0;\n"

    output_hdl = templates.BN_TEMPLATE \
        .replace("%DIM_DATA%", str(len(beta))) \
        .replace("%LAYER_NUM%", str(num)) \
        .replace("%COMPARE%", compare)
        
    with open(f"gen_hdl/L{num}_bn.v", "w") as f:
        f.write(output_hdl)

# Extract weights
parameters = ""
signals = ""
modules = ""
ports = ""

n = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        beta, moving_mean, moving_variance = layer.get_weights()
        parse_bn(beta, moving_mean, moving_variance, n)
        signals += (
            f'wire [$clog2(L{n-1}_INPUT_DIM)-1:0] L{n}_i_data [L{n-1}_OUTPUT_DIM-1:0];\n'
        )
        modules += (
            f'layer_{n}_bn #(\n'
            f'\t.INPUT_DIM(L{n-1}_INPUT_DIM),\n'
            f'\t.OUTPUT_DIM(L{n-1}_OUTPUT_DIM)\n'
            f') L{n}_bn (\n'
            f'\t.i_data(L{n}_i_data),\n'
            f'\t.o_data(L{n+1}_i_data)\n);\n'
        )
        n += 1
    elif isinstance(layer, lq.layers.QuantDense):
        with lq.context.quantized_scope(True):
            weights = layer.get_weights()
            input_dim, output_dim = parse_fc(weights[0], n)
            parameters += (
                f'\tparameter L{n}_INPUT_DIM = {input_dim},\n'
                f'\tparameter L{n}_OUTPUT_DIM = {output_dim},\n'
                )
            signals += (
                f'wire [L{n}_INPUT_DIM-1:0] L{n}_i_data;\n'
            )
            modules += (
                f'layer_{n}_fc #(\n'
                f'\t.INPUT_DIM(L{n}_INPUT_DIM),\n'
                f'\t.OUTPUT_DIM(L{n}_OUTPUT_DIM)\n'
                f') L{n}_fc (\n'
                f'\t.i_data(L{n}_i_data),\n'
                f'\t.o_data(L{n+1}_i_data)\n);\n'
            )
            n += 1

signals += (
    f'wire [$clog2(L{n-1}_INPUT_DIM)-1:0] L{n}_i_data [L{n-1}_OUTPUT_DIM-1:0];\n'
    f'assign o_data = L{n}_i_data;\n'
)
ports += (
    f'\toutput [$clog2(L{n-1}_INPUT_DIM)-1:0] o_data [9:0],\n'
)

output_hdl = templates.TOP_TEMPLATE \
        .replace("%PARAMETERS%", parameters.rstrip(",\n")) \
        .replace("%PORTS%", ports) \
        .replace("%SIGNALS%", signals) \
        .replace("%MODULES%", modules)

with open(f"./gen_hdl/top.v", "w") as f:
    f.write(output_hdl)
