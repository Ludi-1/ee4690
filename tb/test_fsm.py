import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def fsm_test(dut):
    clock = Clock(dut.CLK, 10, units="ns")  # Create a 10ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    dut.rst.value = 0
    await RisingEdge(dut.CLK)