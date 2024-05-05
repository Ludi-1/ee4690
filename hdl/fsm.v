module fsm #(
    parameter MATRIX_DIM = 8,
    parameter ADDR_WIDTH = 6
)
(
    input CLK,
    input rst,
    input start,

    output [ADDR_WIDTH-1:0] addr_a,
    output [ADDR_WIDTH-1:0] addr_b,
    output reg mac_enable,
    output reg [ADDR_WIDTH-1:0] addr_c
);

localparam WAIT = 2'd0; // reset state
localparam BUSY = 2'd1; // accumulation
localparam NEXT = 2'd2; // next MAC
localparam DONE = 2'd3; // MMM done

reg [1:0] state, nextstate;
reg count_enable;
reg count_en, count_c_en, count_rst;
wire [MATRIX_DIM-1:0] count;

// state
always @(posedge CLK, posedge rst) begin
    if  (rst) state <= WAIT;
    else      state <= nextstate;
end

// next state logic
always @(*) begin
    case(state)
        WAIT: if (start)                    nextstate = BUSY;
              else                          nextstate = WAIT;
        BUSY: if (count == MATRIX_DIM-1)    nextstate = DONE;
              else                          nextstate = BUSY;
        NEXT: if (count == 2**ADDR_WIDTH-1) nextstate = DONE;
              else                          nextstate = BUSY;
        DONE: if (start)                    nextstate = BUSY;
              else                          nextstate = DONE;
        default:                            nextstate = WAIT;
    endcase
end

// output logic
always @(*) begin
    if (state == WAIT) begin
        count_en = 1'b0;
        count_c_en = 1'b0;
        count_rst = 1'b1;
        mac_enable = 1'b0;
    end else if (state == BUSY) begin
        count_en = 1'b1;
        count_c_en = 1'b0;
        count_rst = 1'b0;
        mac_enable = 1'b1;
    end else if (state == NEXT) begin
        count_en = 1'b0;
        count_c_en = 1'b1;
        count_rst = 1'b0;
        mac_enable = 1'b0;
    end else if (state == DONE) begin
        count_en = 1'b0;
        count_c_en = 1'b0;
        count_rst = 1'b1;
        mac_enable = 1'b0;
    end
end

// counter FSM
counter #(
    .COUNT_MAX(MATRIX_DIM),
    .COUNT_WIDTH(3),
    .COUNT1_INC(1),
    .COUNT2_INC(0)
) counter_fsm (
    .CLK(CLK),
    .rst(rst | count_rst),
    .count_en(count_en),
    .count(count)
);

// counter A
counter #(
    .COUNT_MAX(8),
    .COUNT_WIDTH(ADDR_WIDTH),
    .COUNT1_INC(1),
    .COUNT2_INC(8)
) counter_a (
    .CLK(CLK),
    .rst(rst | count_rst),
    .count_en(count_en),
    .count(addr_a)
);

// counter B
counter #(
    .COUNT_MAX(8),
    .COUNT_WIDTH(ADDR_WIDTH),
    .COUNT1_INC(8),
    .COUNT2_INC(1)
) counter_b (
    .CLK(CLK),
    .rst(rst | count_rst),
    .count_en(count_en),
    .count(addr_b)
);

// counter C
counter #(
    .COUNT_MAX(8),
    .COUNT_WIDTH(ADDR_WIDTH),
    .COUNT1_INC(1),
    .COUNT2_INC(8)
) counter_c (
    .CLK(CLK),
    .rst(rst | count_rst),
    .count_en(count_en),
    .count(addr_c)
);

endmodule