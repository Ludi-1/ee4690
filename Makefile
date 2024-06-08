SIM = verilator
PWD=$(shell pwd)
TOPLEVEL_LANG = verilog
WAVES=1
# EXTRA_ARGS += --trace-fst --trace-structs
# EXTRA_ARGS += --trace --trace-structs

TOPLEVEL ?= register_file
$(shell rm -rf sim_build)

ifeq ($(TOPLEVEL),ibuf_conv)
    VERILOG_SOURCES = $(shell pwd)/rtl/ibuf_conv.sv
    MODULE = tb.ibuf_conv_tb
else ifeq ($(TOPLEVEL),ibuf_fc)
    VERILOG_SOURCES = $(shell pwd)/ibuf_fc.v
    MODULE = tb.test_ibuf_fc
else ifeq ($(TOPLEVEL),layer_0_fc)
    VERILOG_SOURCES = $(shell pwd)/gen_hdl/fc_layer_0.sv
    COMPILE_ARGS += -Player_0_fc.CLASSIFIER=1
    MODULE = tb.test_fc_layer
else ifeq ($(TOPLEVEL),top)
    VERILOG_SOURCES = $(shell pwd)/sw/gen_hdl/*
    MODULE = tb.top_tb
else
    $(error Given TOPLEVEL '$(TOPLEVEL)' not supported)
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	@rm -rf sim_build
	@rm -rf dump.fst $(TOPLEVEL).fst