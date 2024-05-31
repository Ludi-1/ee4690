import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random
import tensorflow as tf

@cocotb.test()
async def ctrl_test(dut):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
    test_image = test_images[0]
    test_image[test_image < 0] == 0
    test_image[test_image >= 0] == 1


    clock = Clock(dut.clk, 10, units="ns")  # Create a 10ns clock period
    cocotb.start_soon(clock.start())  # Start the clock

    n = 0
    for row in test_image:
        for column in row:
            dut.i_data[n] = column
    await Timer(10e2, units='ns')
    