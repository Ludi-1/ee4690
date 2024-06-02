"""
Templates for generating SV through Python
"""

# Template for the FC layer
# TODO: parameterize over datatype size
FC_TEMPLATE = """module layer_%LAYER_NUM%_fc #(
    parameter INPUT_DIM = 784,
    parameter OUTPUT_DIM = 10,
    parameter INPUT_CHANNELS = 1,
    parameter logic CLASSIFIER = 1'b%CLASSIFIER%,
    parameter DATATYPE_SIZE = CLASSIFIER ? $clog2(INPUT_DIM) - 1 : 1
) (
    input i_data [INPUT_CHANNELS-1:0],
    output reg [DATATYPE_SIZE-1:0] o_data [OUTPUT_DIM-1:0]
);

wire xnor_result [INPUT_DIM-1:0][OUTPUT_DIM-1:0];
reg [$clog2(INPUT_DIM)-1:0] popcnt [OUTPUT_DIM-1:0];
logic signed [$clog2(INPUT_DIM):0] shift [OUTPUT_DIM-1:0];
logic signed [$clog2(INPUT_DIM)-1:0] act [OUTPUT_DIM-1:0];

ibuf_fc #(
    .FIFO_LENGTH(INPUT_DIM),
    .INPUT_CHANNELS(INPUT_CHANNELS)
) ibuf (
    
)

%XNOR_GEN%

localparam CONCAT_BITS = $clog2(INPUT_DIM)-1;

always_comb begin
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        popcnt[i] = 0;
        for (int j = 0; j < INPUT_DIM; j++) begin
            popcnt[i] += { {CONCAT_BITS{1'b0}}, xnor_result[j][i]};
        end
        shift[i] = popcnt[i] << 1;
        act[i] = shift[i] - INPUT_DIM;
    end
end

generate 
    for (genvar i = 0; i < OUTPUT_DIM; i++) begin
        if (CLASSIFIER) begin
            assign o_data[i] = act[i]; // (no soft)max
        end else begin
            assign o_data[i] = ~act[i][$clog2(INPUT_DIM)-1]; // sign function
        end
    end
endgenerate

initial begin
    $dumpfile("dump_layer_%LAYER_NUM%_fc.vcd");
    for (int i = 0; i < INPUT_DIM; i++) begin
        $dumpvars(0, i_data[i]);
    end
    for (int j = 0; j < OUTPUT_DIM; j++) begin
        $dumpvars(0, o_data[j]);
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

BN_TEMPLATE = """module layer_%LAYER_NUM%_BN #(
    parameter DIM = %DIM_DATA%,
    parameter INPUT_DATA_SIZE = 32,
    parameter OUTPUT_DATA_SIZE = 1,
    parameter 
)(
    input clk,
    input reset,

    input [INPUT_DATA_SIZE - 1: 0] i_data [DIM -1 : 0],

    output o_data [DIM - 1 : 0],
);
%COMPARE%

endmodule
"""
