"""
Templates for generating SV through Python
"""

# Template for the FClayer
# TODO: parameterize over datatype size
FC_TEMPLATE = """module layer_%LAYER_NUM%_fc #(
    parameter INPUT_DIM = 784,
    parameter OUTPUT_DIM = 10,
    parameter INPUT_CHANNELS = 1
) (
    input clk,
    input reset,

    input i_we,
    input i_data [INPUT_DIM-1:0][INPUT_CHANNELS-1:0],
    output o_data [OUTPUT_DIM-1:0],
    output o_we
);

wire xnor_result [INPUT_DIM-1:0][OUTPUT_DIM-1:0];
wire [$clog2(INPUT_DIM)-1:0] popcnt [OUTPUT_DIM-1:0];

%XNOR_GEN%

generate
    for (genvar i = 0; i < OUTPUT_DIM; i++) begin
        assign popcnt[i] = 0;
        for (genvar j = 0; j < INPUT_DIM) begin
            assign popcnt[i] += xnor_result[j][i];
        end
        assign o_data[i] = 2 * popcnt[i] - INPUT_DIM;
    end
endgenerate

endmodule"""

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