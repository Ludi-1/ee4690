"""
Python simulation models for the layers
    1. Train model
    2. Extract weights
    3. Emulate inference with weights
"""
# Libraries
import larq as lq
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from sw.simulator import MyModel

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

tf.keras.backend.set_floatx('float64')
from sw.helper import retrieve_weights, plot_differences, check_result, plot_intermediate_results, get_output

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Image dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

# NN Topology
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

model = MyModel(prune=0.01)

input_shape = (28, 28, 1)  # Input img shape
filters_a = 32  # Number of output channels
kernel_a = (5, 5)  # Kernel dimension

filters_b = 64  # Number of output channels
kernel_b = (3, 3)  # Kernel dimension

model.add(tf.keras.layers.Flatten(input_shape=input_shape))

model.add(lq.layers.QuantDense(128, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
#
model.add(lq.layers.QuantDense(128, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantDense(64, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))


model.add(lq.layers.QuantDense(10, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))

# Train NNy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=6)
test_loss, test_acc = model.evaluate(test_images, test_labels)

for struct in model.structure:
    # print(struct["name"])
    pass
valid, _ = model.simulate(test_images[0:1])

accuracy = model.test_accuracy(test_images, test_labels)
model.get_prune_info()

print("The accuracy is:" + str(accuracy))
if valid:
    print("The simulation test has passed")
else:
    print("ERROR: The simulation test has FAILED")

