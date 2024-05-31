"""
Templates for generating SV through Python
"""

# Template for the FC layer
# TODO: parameterize over datatype size
FC_TEMPLATE = """module layer_%LAYER_NUM%_fc #(
    parameter INPUT_DIM = 784,
    parameter OUTPUT_DIM = 10
) (
    input i_data [INPUT_DIM-1:0],
    output reg [$clog2(INPUT_DIM)-1:0] o_data [OUTPUT_DIM-1:0]
);

wire xnor_result [INPUT_DIM-1:0][OUTPUT_DIM-1:0];
reg [$clog2(INPUT_DIM)-1:0] popcnt [OUTPUT_DIM-1:0];
reg [$clog2(INPUT_DIM)-1:0] act [OUTPUT_DIM-1:0];

%XNOR_GEN%

localparam CONCAT_BITS = $clog2(INPUT_DIM)-1;

always_comb begin
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        popcnt[i] = 0;
        for (int j = 0; j < INPUT_DIM; j++) begin
            popcnt[i] += { {CONCAT_BITS{1'b0}}, xnor_result[j][i]};
        end
        act[i] = popcnt[i] << 1 - INPUT_DIM;
        o_data[i] = popcnt[i];
        // if (act[i][$clog2(INPUT_DIM)-1]) begin
        //     o_data[i] = 0;
        // end else begin
        //     o_data[i] = 1;
        // end
    end
end

initial begin
    $dumpfile("dump_layer_%LAYER_NUM%_fc.vcd");
    $dumpvars(1, layer_%LAYER_NUM%_fc);
    for (int i = 0; i < INPUT_DIM; i++) begin
        $dumpvars(0, i_data[i]);
    end
end


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