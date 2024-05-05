module top #(
    parameter MATRIX_DIM = 8,
    parameter INPUT_WIDTH = 8,
    parameter OUTPUT_WIDTH = 20,
    parameter ADDR_WIDTH = $clog2(MATRIX_DIM**2)
)
(
    input CLK,
    input rst,
    input wire start,
    input [ADDR_WIDTH-1:0] addr_a_in,
    input [INPUT_WIDTH-1:0] data_a,
    input wire we_a,
    input [ADDR_WIDTH-1:0] addr_b_in,
    input [INPUT_WIDTH-1:0] data_b,
    input wire we_b,
    input [ADDR_WIDTH-1:0] addr_c_in,
    input [OUTPUT_WIDTH-1:0] data_c
);

wire [ADDR_WIDTH-1:0] addr_a_fsm, addr_b_fsm, addr_c_fsm;
wire [ADDR_WIDTH-1:0] addr_a, addr_b, addr_c;
wire mac_en, we_c, state;
wire [INPUT_WIDTH-1:0] rd_data_a, rd_data_b;

assign addr_a = state ? addr_a_fsm : addr_a_in;
assign addr_b = state ? addr_b_fsm : addr_b_in;
assign addr_c = state ? addr_c_fsm : addr_c_in;

// FSM
fsm #(
    .MATRIX_DIM(MATRIX_DIM),
    .ADDR_WIDTH(ADDR_WIDTH)
) fsm1 (
    .CLK(CLK),
    .rst(rst),
    .start(start),
    .addr_a(addr_a_fsm),
    .addr_b(addr_b_fsm),
    .mac_enable(mac_en),
    .addr_c(addr_c_fsm),
    .we_c(we_c),
    .state(state)
);

// regfile A
register_file #(
    .DATATYPE_SIZE(INPUT_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) regfile_a (
    .CLK(CLK),
    .addr(addr_a),
    .wr_data(data_a),
    .we(we_a),
    .rd_data(rd_data_a)
);

// regfile B
register_file #(
    .DATATYPE_SIZE(INPUT_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
) regfile_b (
    .CLK(CLK),
    .addr(addr_b),
    .wr_data(data_b),
    .we(we_b),
    .rd_data(rd_data_b)
);

// mac

endmodule