SIM = verilator
PWD=$(shell pwd)
TOPLEVEL_LANG = verilog
WAVES=1
EXTRA_ARGS += --trace-fst --trace-structs
EXTRA_ARGS += --trace --trace-structs

TOPLEVEL ?= register_file
$(shell rm -rf sim_build)

ifeq ($(TOPLEVEL),register_file)
    VERILOG_SOURCES = $(shell pwd)/hdl/register_file.v
    MODULE = tb.test_register_file
else ifeq ($(TOPLEVEL),counter)
    VERILOG_SOURCES = $(shell pwd)/hdl/counter.v
    MODULE = tb.test_counter
else ifeq ($(TOPLEVEL),fsm)
    VERILOG_SOURCES = $(shell pwd)/hdl/fsm.v
    VERILOG_SOURCES += $(shell pwd)/hdl/counter.v
    # VERILOG_SOURCES += $(shell pwd)/hdl/counter_2d.v
    MODULE = tb.test_fsm
else ifeq ($(TOPLEVEL),mac)
    VERILOG_SOURCES = $(shell pwd)/hdl/mac.v
    MODULE = tb.test_mac
else
    $(error Given TOPLEVEL '$(TOPLEVEL)' not supported)
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	@rm -rf sim_build
	@rm -rf dump.fst $(TOPLEVEL).fst