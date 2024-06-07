import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random
import tensorflow as tf

@cocotb.test()
async def top_test(dut):
    clock = Clock(dut.clk, 4, units="ns")  # Create a 4ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

    num = 0
    correct = 0
    await RisingEdge(dut.clk)
    for test_image in test_images[0:10]:
        dut.i_we.value = 1
        test_image[test_image < 0.0] = 0
        test_image[test_image > 0.0] = 1
        n = 0
        test_label = test_labels[num]
        num += 1
        for row in test_image:
            for column in row:
                dut.i_data.value = int(column)
                dut.i_addr.value = n
                n += 1
                await RisingEdge(dut.clk)
        dut.i_we.value = 0
        await RisingEdge(dut.clk)
        digit_list = []
        for digit in range(10):
            # print(f"Digit {digit}: {dut.o_data[digit].value.signed_integer}")
            digit_list.append(dut.o_data[digit].value.signed_integer)
        max_value = max(digit_list)
        max_index = digit_list.index(max_value)
        if max_index == test_label:
            correct += 1
        print(f"Actual: {test_label} | Prediction: {max_index} | Accuracy: {correct/num}")
        await RisingEdge(dut.clk)
    