import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def fsm_test(dut):
    clock = Clock(dut.CLK, 10, units="ns")  # Create a 10ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    dut.rst.value = 1
    dut.start.value = 0
    await RisingEdge(dut.CLK)
    dut.rst.value = 0
    # write regfile A (and B TODO)
    dut.we_a.value = 1
    dut.we_b.value = 1
    for addr in range(64):
        dut.addr_a_in.value = addr
        dut.addr_b_in.value = addr
        wr_data_a = random.getrandbits(8)
        wr_data_b = random.getrandbits(8)
        dut.data_a.value = wr_data_a
        dut.data_b.value = wr_data_b
        await RisingEdge(dut.CLK)
    dut.we_a.value = 0
    dut.we_b.value = 0

    await RisingEdge(dut.CLK)
    dut.start.value = 1
    await RisingEdge(dut.CLK)
    dut.start.value = 0
    for i in range(8*64):
        await RisingEdge(dut.CLK)

    await RisingEdge(dut.CLK)
    dut.start.value = 0
    for addr in range(64):
        dut.addr_c_in.value = addr
        await RisingEdge(dut.CLK)
        print(f"addr {addr} - {hex(dut.data_c.value)}")