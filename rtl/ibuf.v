module ibuf #(
    parameter IMG_DIM = 28,
    parameter IMG_SIZE = IMG_DIM**2,
    parameter ADDR_WIDTH = $clog2(IMG_SIZE)
) (
    input clk,
    input i_we,
    input [ADDR_WIDTH-1:0] i_addr,
    input i_data,
    output reg [IMG_SIZE-1:0] o_data
);

reg registers [2**ADDR_WIDTH-1:0];

always @(posedge clk) begin
    if (i_we) begin
        registers[i_addr] <= i_data;
    end
end

always @(*) begin
    for (int i = 0; i < IMG_SIZE; i++) begin
        o_data[i] = registers[i];
    end
end

// initial begin
//     $dumpfile("dump_flatten.fst");
//     for (int j = 0; j < IMG_SIZE; j++) begin
//         $dumpvars(0, o_data[j]);
//     end
// end

endmodule