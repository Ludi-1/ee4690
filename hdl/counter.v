module counter #(
    parameter COUNT_MAX = 8,
    parameter COUNT_WIDTH = 3,
    parameter COUNT1_INC = 1,
    parameter COUNT2_INC = 0
)

(
    input CLK,
    input rst,
    input count_en,

    output reg [COUNT_WIDTH-1:0] count
);

reg [COUNT_MAX-1:0] count1, count2;

assign count = count1 + count2;

always @(posedge CLK, posedge rst) begin
    if (rst) begin
        count1 <= '0;
        count2 <= '0;
    end else if (count_en) begin
        if (count1 == (COUNT_MAX-1)*COUNT1_INC) begin
            count1 <= 0;
            if (count2 == (COUNT_MAX-1)*COUNT2_INC) begin
                count2 <= 0;
            end else begin
                count2 <= count2 + COUNT2_INC;
            end
        end else begin
            count1 <= count1 + COUNT1_INC;
        end
    end
end

endmodule