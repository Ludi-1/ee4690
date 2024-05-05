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
    output reg [ADDR_WIDTH-1:0] addr_c,
    output reg we_c
);

localparam WAIT = 2'd0; // reset state
localparam BUSY = 2'd1; // accumulation
localparam DONE = 2'd3; // MMM done

localparam COUNT_WIDTH = $clog2(MATRIX_DIM);
localparam COUNT_FSM_MAX = MATRIX_DIM-1;

reg [1:0] state, nextstate;
reg count_enable;
reg count_en, count_c_en;
reg [COUNT_WIDTH-1:0] count;
reg [$clog2(MATRIX_DIM)-1:0] count_addr_a1, count_addr_b2;
reg [$clog2(MATRIX_DIM*(MATRIX_DIM-1))-1:0] count_addr_a2, count_addr_b1;

assign addr_a = {3'b0, count_addr_a1} + count_addr_a2;
assign addr_b = {3'b0, count_addr_b2} + count_addr_b1;

// state
always @(posedge CLK, posedge rst) begin
    if  (rst) state <= WAIT;
    else      state <= nextstate;
end

// next state logic
always @(*) begin
    case(state)
        WAIT: if (start)                               nextstate = BUSY;
              else                                     nextstate = WAIT;
        BUSY: if (addr_c == MATRIX_DIM**2-1)           nextstate = DONE;
              else                                     nextstate = BUSY;
        // BUSY: if (count ==
        //         COUNT_FSM_MAX[$clog2(MATRIX_DIM)-1:0]) nextstate = NEXT;
        //       else                                     nextstate = BUSY;
        // NEXT: if (addr_c == MATRIX_DIM**2-1)           nextstate = DONE;
        //       else                                     nextstate = BUSY;
        DONE: if (start)                               nextstate = BUSY;
              else                                     nextstate = DONE;
        default:                                       nextstate = WAIT;
    endcase
end

// output logic
always @(*) begin
    if (state == WAIT) begin
        count_en = 1'b0;
        count_c_en = 1'b0;
        mac_enable = 1'b0;
        we_c = 1'b0;
    end else if (state == BUSY) begin
        count_en = 1'b1;
        mac_enable = 1'b1;
        we_c = 1'b0;
        count_en = 1'b1;
        if (count == COUNT_FSM_MAX[$clog2(MATRIX_DIM)-1:0]) begin
            count_c_en = 1'b1;
        end else begin
            count_c_en = 1'b0;
        end
    // end else if (state == NEXT) begin
    //     count_en = 1'b1;
    //     count_c_en = 1'b1;
    //     count_rst = 1'b0;
    //     mac_enable = 1'b0;
    //     we_c = 1'b1;
    end else if (state == DONE) begin
        count_en = 1'b0;
        count_c_en = 1'b0;
        mac_enable = 1'b0;
        we_c = 1'b0;
    end else begin
        count_en = 1'b0;
        count_c_en = 1'b0;
        mac_enable = 1'b0;
        we_c = 1'b0;
    end
end

// counter FSM
counter #(
    .COUNT_MAX(MATRIX_DIM),
    .COUNT_INC(1)
) counter_fsm (
    .CLK(CLK),
    .rst(rst),
    .count_en(count_en),
    .count(count)
);

// counter C
counter #(
    .COUNT_MAX(MATRIX_DIM**2),
    .COUNT_INC(1)
) counter_c (
    .CLK(CLK),
    .rst(rst),
    .count_en(count_c_en),
    .count(addr_c)
);

// counter A
counter #(
    .COUNT_MAX(MATRIX_DIM),
    .COUNT_INC(1)
) counter_a1 (
    .CLK(CLK),
    .rst(rst),
    .count_en(count_en),
    .count(count_addr_a1)
);

counter #(
    .COUNT_MAX(MATRIX_DIM),
    .COUNT_INC(8)
) counter_a2 (
    .CLK(CLK),
    .rst(rst),
    .count_en(count_c_en),
    .count(count_addr_a2)
);

// counter B
counter #(
    .COUNT_MAX(MATRIX_DIM),
    .COUNT_INC(8)
) counter_b1 (
    .CLK(CLK),
    .rst(rst),
    .count_en(count_en),
    .count(count_addr_b1)
);

counter #(
    .COUNT_MAX(MATRIX_DIM),
    .COUNT_INC(1)
) counter_b2 (
    .CLK(CLK),
    .rst(rst),
    .count_en(count_c_en),
    .count(count_addr_b2)
);

endmodule