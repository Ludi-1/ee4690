import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def counter_test(dut):
    clock = Clock(dut.CLK, 10, units="ns")  # Create a 10ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    dut.rst.value = 0
    dut.count_en.value = 0
    await RisingEdge(dut.CLK)
    dut.count_en.value = 1
    for count in range(64):
        await RisingEdge(dut.CLK)
    dut.count_en.value = 0
    await RisingEdge(dut.CLK)