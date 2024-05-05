module mac #(
    parameter IN_WIDTH = 8,
    parameter OUT_WIDTH = 20
)

(
    input CLK,
    input rst,
    input enable,
    input set_sum,

    input [IN_WIDTH-1:0] in_1,
    input [IN_WIDTH-1:0] in_2,

    output [OUT_WIDTH-1:0] out
);

wire [IN_WIDTH*2-1:0] product;
reg [OUT_WIDTH-1:0] sum;

assign product = in_1 * in_2;
assign out = set_sum ? {4'b0, product} : sum + {4'b0, product};

// always @(*) begin
//     if (set_sum) begin
//         out = {4'b0, product};
//     end else begin
//         out = sum + {4'b0, product};
//     end
// end

always @(posedge CLK, posedge rst) begin
    if (rst) begin 
        sum <= 'd0;
    end else if (enable) begin
        sum <= out;
    end
end

endmodule