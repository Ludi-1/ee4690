"""
Templates for generating SV through Python
"""

# Template for the conv layer
# TODO: parameterize over datatype size
CONV_TEMPLATE = """module layer_%LAYER_NUM%_conv #(
    parameter INPUT_DIM = 784,
    parameter OUTPUT_DIM = 10,
    parameter KERNEL_DIM = 3,
    parameter INPUT_CHANNELS = 1,
    parameter OUTPUT_CHANNELS = 2,
    parameter DATATYPE_SIZE = 1
) (
    input clk,
    input reset,

    input i_we,
    input [DATATYPE_SIZE-1:0] i_data [INPUT_DIM-1:0],

    output [DATATYPE_SIZE-1:0] o_data [OUTPUT_CHANNELS-1:0],
    output o_we
);


endmodule"""