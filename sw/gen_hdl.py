"""
1. Define NN topology
2. Train NN
3. Generate HDL
"""

# Libraries
import larq as lq
import tensorflow as tf
import numpy as np

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
model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=64, epochs=6)
test_loss, test_acc = model.evaluate(test_images, test_labels)
lq.models.summary(model)

### PARSE FC FUNC
def parse_fc(fc_weights, num):
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

    output_hdl = templates.FC_TEMPLATE.replace("%XNOR_GEN%", xnor)
    with open(f"gen_hdl/fc_layer_{num}.sv", "w") as f:
        f.write(output_hdl)

# Extract weights
with lq.context.quantized_scope(True):
    weights = model.layers[1].get_weights()

parse_fc(weights[0], 0)