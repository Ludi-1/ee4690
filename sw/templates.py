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
    parameter INPUT_DATATYPE_SIZE = 1,
    parameter OUTPUT_DATATYPE_SIZE = 1
) (
    input clk,
    input reset,

    input i_we,
    input [INPUT_DATATYPE_SIZE-1:0] i_data [INPUT_CHANNELS-1:0],

    output [OUTPUT_DATATYPE_SIZE-1:0] o_data [OUTPUT_CHANNELS-1:0],
    output o_we
);

wire [INPUT_DATATYPE_SIZE-1:0] window [KERNEL_DIM][INPUT_CHANNELS-1:0];

genvar i;
generate
    for (i = 0; i < INPUT_CHANNELS; i++) begin
        ibuf_conv #(
            .img_width(INPUT_DIM),
            .kernel_dim(KERNEL_DIM)
        ) ibuf (
            .clk(clk),
            .i_write_enable(i_we),
            .i_data(i_data[i]),
            .o_data(window[i])
        )
    end
endgenerate

generate
    if(INPUT_DATATYPE_SIZE == 1)
        ????
    else
        ????
endgenerate

endmodule"""