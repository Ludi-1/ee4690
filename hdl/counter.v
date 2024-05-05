module counter #(
    parameter COUNT1_MAX = 8,
    parameter COUNT2_MAX = 8
)

(
    input CLK,
    input rst,
    input count_en,

    output reg [COUNT1_MAX-1:0] count1,
    output reg [COUNT2_MAX-1:0] count2
);

always @(posedge CLK, posedge rst) begin
    if (rst) begin
        count1 <= '0;
        count2 <= '0;
    end else if (count_en) begin
        if (count1 == COUNT1_MAX-1) begin
            count1 <= 0;
            if (count2 == COUNT2_MAX-1) begin
                count2 <= 0;
            end else begin
                count2 <= count2 + 1;
            end
        end else begin
            count1 <= count1 + 1;
        end
    end
end

endmodule