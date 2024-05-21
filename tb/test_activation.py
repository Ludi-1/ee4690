import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def counter_test(dut):
    for _ in range(255):
        in_val = random.getrandbits(32)
        popcnt = bin(in_val)[2:].count("1")
        subtracted = 2 * popcnt - 32
        sign = 1 if subtracted >= 0 else 0

        dut.xnor_in.value = in_val
        await Timer(100, units='ps')
        output = int(dut.activated_out.value)
        # print(output, golden)
        assert(output == sign)