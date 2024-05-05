import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def mac_test(dut):
    clock = Clock(dut.CLK, 10, units="ns")  # Create a 10ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    dut.rst.value = 1
    dut.enable.value = 0
    await RisingEdge(dut.CLK)
    dut.rst.value = 0
    await RisingEdge(dut.CLK)
    dut.enable.value = 1
    data = []
    for i in range(10):
        dut.in_1.value = i
        dut.in_2.value = i
        await RisingEdge(dut.CLK)
        data.append(int(dut.out.value))
    await RisingEdge(dut.CLK)
    dut.enable.value = 0
    acc = 0
    for i in range(10):
        assert(data[i] == acc)
        acc += i**2