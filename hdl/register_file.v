module register_file #(
    parameter DATATYPE_SIZE = 8,
    parameter ADDR_WIDTH = 6
)
(
    input CLK,

    input [ADDR_WIDTH-1:0] addr, // 8*8 = 64 entries -> log2(64) = 6
    input [DATATYPE_SIZE-1:0] wr_data, // 8-bit datatype size
    input we,
    output reg [DATATYPE_SIZE-1:0] rd_data
);

reg [DATATYPE_SIZE-1:0] registers [2**ADDR_WIDTH-1:0];

always @(posedge CLK) begin
    if (we) begin
        registers[addr] <= wr_data;
    end
end

always @(*) begin
    rd_data = registers[addr];
end

endmodule