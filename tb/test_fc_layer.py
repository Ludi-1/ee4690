import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random
import tensorflow as tf

@cocotb.test()
async def fc_layer_test(dut):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
    test_image = test_images[0]
    test_image[test_image < 0.0] = 0
    test_image[test_image > 0.0] = 1
    
    n = 0
    for row in test_image:
        for column in row:
            dut.i_data[n].value = int(column)
            n += 1
    await Timer(1, units='ns')
    for digit in range(10):
        print(f"Digit {digit}: {dut.o_data[digit].value.signed_integer}")
    await Timer(10e2, units='ns')
    