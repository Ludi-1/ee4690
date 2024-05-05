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

    output reg [OUT_WIDTH-1:0] out
);

wire [IN_WIDTH*2-1:0] product;
wire [OUT_WIDTH-1:0] sum;

assign product = in_1 * in_2;
assign sum = out + {4'b0, product};

always @(posedge CLK, posedge rst) begin
    if (rst) begin 
        out <= '0;
    end else if (enable) begin
        if (set_sum) begin
            out <= product;
        end else begin
            out <= sum;
        end
    end
end

endmodule