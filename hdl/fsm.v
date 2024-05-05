module fsm #(
    parameter ADDR_WIDTH = 6
)
(
    input CLK,
    input rst,
    input start,

    output [ADDR_WIDTH-1:0] addr_a,
    output [ADDR_WIDTH-1:0] addr_b,

    output wire mac_enable
);

localparam WAIT = 2'd0;
localparam BUSY = 2'd1;
localparam DONE = 2'd2;

reg [3:0] state, nextstate;
reg [ADDR_WIDTH-1:0] count;
reg count_enable;

// state
always @(posedge CLK, posedge rst) begin
    if  (rst) state <= WAIT;
    else      state <= nextstate;
end

// next state logic
always @(*) begin
    case(state)
        WAIT: if (start)                 nextstate = BUSY;
              else                       nextstate = WAIT;
        BUSY: if (count == ADDR_WIDTH-1) nextstate = DONE;
              else                       nextstate = BUSY;
        DONE: if (start)                 nextstate = BUSY;
              else                       nextstate = DONE;
        default:                         nextstate = WAIT;
    endcase
end

// // output logic
// always @(*) begin
//     if (state == WAIT) begin
//         count_en = 1'b0;
//         mac_enable = 1'b0;
//     end else if (state == BUSY) begin
//         count_en = 1'b1;
//     end else begin
//         count = 1'b0;
//         mac_enable = 1'b0;
//     end
// end

endmodule