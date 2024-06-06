import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random
import tensorflow as tf

@cocotb.test()
async def ibuf_conv_test(dut):
    clock = Clock(dut.clk, 4, units="ns")  # Create a 4ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
    test_image = test_images[0]
    test_image[test_image < 0.0] = 0
    test_image[test_image > 0.0] = 1
    
    dut.i_write_enable.value = 1
    n = 0
    for row in test_image:
        for column in row:
            dut.i_data[0].value = int(column)
            n += 1
            await RisingEdge(dut.clk)
    dut.i_write_enable.value = 0
    await Timer(10e2, units='ns')
    