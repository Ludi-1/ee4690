"""
Templates for generating SV through Python
"""

# Template for the FC layer
FC_TEMPLATE = """module layer_%LAYER_NUM%_fc #(
    parameter INPUT_DIM = %INPUT_DIM%,
    parameter OUTPUT_DIM = %OUTPUT_DIM%
) (
    input i_data [INPUT_DIM-1:0],
    output reg [DATATYPE_SIZE-1:0] o_data [OUTPUT_DIM-1:0]
);

wire xnor_result [INPUT_DIM-1:0][OUTPUT_DIM-1:0];
reg [$clog2(INPUT_DIM)-1:0] popcnt [OUTPUT_DIM-1:0];
reg [$clog2(INPUT_DIM)-1:0] shift [OUTPUT_DIM-1:0];
reg [$clog2(INPUT_DIM):0] act [OUTPUT_DIM-1:0];

%XNOR_GEN%

localparam CONCAT_BITS = $clog2(INPUT_DIM)-1;

always_comb begin
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        popcnt[i] = 0;
        for (int j = 0; j < INPUT_DIM; j++) begin
            popcnt[i] += { {CONCAT_BITS{1'b0}}, xnor_result[j][i]};
        end
        shift[i] = popcnt[i] <<< 1;
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

always @(posedge clk) begin
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        o_data[i] = shift[i] - INPUT_DIM;
    end
end

endgenerate

initial begin
    $dumpfile("dump.fst");
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        $dumpvars(0, o_data[i]);
    end
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        $dumpvars(0, shift[i]);
    end
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        $dumpvars(0, popcnt[i]);
    end
end

endmodule"""

FC_TEMPLATE_OLD = """module layer_%LAYER_NUM%_fc #(
    parameter INPUT_DIM = %INPUT_DIM%,
    parameter OUTPUT_DIM = %OUTPUT_DIM%,
    parameter INPUT_CHANNELS = %INPUT_CHANNELS%,
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
            assign o_data[i] = ~act[i][$clog2(INPUT_DIM)]; // sign function
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
CONV_TEMPLATE = """module layer_%LAYER_NUM%_conv #(
    parameter INPUT_DIM = 28,
    parameter OUTPUT_DIM = 10,
    parameter KERNEL_DIM = 3,
    parameter INPUT_CHANNELS = 1,
    parameter OUTPUT_CHANNELS = 2,
    parameter DATATYPE_SIZE = 1
) (
    input clk,
    input reset,

    input i_we,
    input i_data [INPUT_CHANNELS-1:0],

    output o_data [OUTPUT_CHANNELS-1:0],
    output o_we
);

reg window [INPUT_CHANNELS-1:0][KERNEL_DIM**2-1:0];
reg xnor [OUTPUT_CHANNELS-1:0][INPUT_CHANNELS-1:0][KERNEL_DIM**2-1:0];

ibuf_conv #(
    .img_width(INPUT_DIM),
    .kernel_dim(KERNEL_DIM),
    .input_channels(INPUT_CHANNELS)
) ibuf (
    .clk(clk),
    .i_we(i_we),
    .i_data(i_data),
    .o_data(window)
);

%XNOR%

always @(*) begin
    for (int i = 0; i < OUTPUT_CHANNELS; i++) begin
        for (int j = 0; j < INPUT_CHANNELS; j++) begin
            for (int k = 0; k < KERNEL_DIM ** 2; k++) begin
            o[i] += xnor[i][j][k];
            end
        end
    end
end

endmodule"""

BN_TEMPLATE = """module layer_%LAYER_NUM%_bn #(
    parameter INPUT_DIM = 1,
    parameter OUTPUT_DIM = 1
)(
    input signed [$clog2(INPUT_DIM):0] i_data [OUTPUT_DIM-1:0],
    output [OUTPUT_DIM-1:0] o_data
);

%COMPARE%

endmodule
"""

TOP_TEMPLATE = """module top #(
    parameter IMG_DIM = 28,
%PARAMETERS%
) (
%PORTS%
    input clk,
    input i_we,
    input [$clog2(L0_INPUT_DIM)-1:0] i_addr,
    input i_data
);

%SIGNALS%

ibuf #(
    .IMG_DIM(IMG_DIM)
) ibuf1 (
    .clk(clk),
    .i_we(i_we),
    .i_addr(i_addr),
    .i_data(i_data),
    .o_data(L0_i_data)
);

%MODULES%

initial begin
    $dumpfile("dump.fst");
    for (int i = 0; i < L0_INPUT_DIM; i++) begin
        $dumpvars(0, L0_i_data[i]);
    end
    for (int i = 0; i < L1_INPUT_DIM; i++) begin
        $dumpvars(0, L1_i_data[i]);
    end
    for (int i = 0; i < 10; i++) begin
        $dumpvars(0, o_data[i]);
    end
end

endmodule"""