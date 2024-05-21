SIM = icarus
PWD=$(shell pwd)
TOPLEVEL_LANG = verilog
WAVES=1
# Use for verilator
# EXTRA_ARGS += --trace-fst --trace-structs
# EXTRA_ARGS += --trace --trace-structs

TOPLEVEL ?= popcount
$(shell rm -rf sim_build)

ifeq ($(TOPLEVEL),popcount)
    VERILOG_SOURCES = $(shell pwd)/hdl/popcount.v
    MODULE = tb.test_popcount
else ifeq ($(TOPLEVEL),activation)
    VERILOG_SOURCES = $(shell pwd)/hdl/activation.v
    VERILOG_SOURCES += $(shell pwd)/hdl/popcount.v
    MODULE = tb.test_activation
else
    $(error Given TOPLEVEL '$(TOPLEVEL)' not supported)
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	@rm -rf sim_build
	@rm -rf dump.fst $(TOPLEVEL).fst
