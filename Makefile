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
# else ifeq ($(TOPLEVEL),fc_ctrl)
#     VERILOG_SOURCES = $(shell pwd)/hdl/fc_ctrl.sv
#     MODULE = tb.test_ctrl
else
    $(error Given TOPLEVEL '$(TOPLEVEL)' not supported)
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

clean::
	@rm -rf sim_build
	@rm -rf dump.fst $(TOPLEVEL).fst