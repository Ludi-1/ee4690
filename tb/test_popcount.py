import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def counter_test(dut):
    for i in range(255):
        dut.bitstring_in.value = i
        await Timer(100, units='ps')
        output = int(dut.count_out.value)
        golden = bin(i)[2:].count("1")
        # print(output, golden)
        assert(output == golden)