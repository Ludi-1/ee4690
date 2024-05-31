SIM = verilator
PWD=$(shell pwd)
TOPLEVEL_LANG = verilog
WAVES=1
EXTRA_ARGS += --trace-fst --trace-structs
EXTRA_ARGS += --trace --trace-structs

TOPLEVEL ?= register_file
$(shell rm -rf sim_build)

ifeq ($(TOPLEVEL),ibuf_conv)
    VERILOG_SOURCES = $(shell pwd)/hdl/ibuf_conv.v
    MODULE = tb.test_ibuf_conv
else ifeq ($(TOPLEVEL),ibuf_fc)
    VERILOG_SOURCES = $(shell pwd)/ibuf_fc.v
    MODULE = tb.test_ibuf_fc
else ifeq ($(TOPLEVEL),top)
    VERILOG_SOURCES = $(shell pwd)/hdl/ibuf_conv.v
    VERILOG_SOURCES += $(shell pwd)/hdl/ibuf_fc.v
    MODULE = tb.test_top
else ifeq ($(TOPLEVEL),layer_0_fc)
    VERILOG_SOURCES = $(shell pwd)/gen_hdl/fc_layer_0.sv
    MODULE = tb.test_fc_layer
else
    $(error Given TOPLEVEL '$(TOPLEVEL)' not supported)
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	@rm -rf sim_build
	@rm -rf dump.fst $(TOPLEVEL).fst