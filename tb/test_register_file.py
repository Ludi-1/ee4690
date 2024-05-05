import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import random

@cocotb.test()
async def reg_file_test(dut):
    clock = Clock(dut.CLK, 10, units="ns")  # Create a 10ns clock period
    cocotb.start_soon(clock.start())  # Start the clock
    data = []
    await RisingEdge(dut.CLK)
    dut.we.value = 1
    for addr in range(64):
        dut.addr.value = addr
        wr_data = random.getrandbits(8)
        data.append(wr_data)
        dut.wr_data.value = wr_data
        await RisingEdge(dut.CLK)
        print(f"Write data {addr}: {wr_data}")
    dut.we.value = 0

    for addr in range(64):
        dut.addr.value = addr
        await RisingEdge(dut.CLK)
        assert data[addr] == dut.rd_data.value, \
            f"Assert failed: {data[addr]} != {dut.rd_data.value}"
        print(f"Read data {addr}: {int(dut.rd_data.value)}")