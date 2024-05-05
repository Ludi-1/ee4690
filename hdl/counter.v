module counter #(
    parameter COUNT_MAX = 8,
    parameter COUNT_INC = 1
)

(
    input CLK,
    input rst,
    input count_en,

    output reg [$clog2((COUNT_MAX-1)*COUNT_INC)-1:0] count
);

localparam MAX_COUNT = (COUNT_MAX-1)*COUNT_INC;

always @(posedge CLK, posedge rst) begin
    if (rst) begin
        count <= 'd0;
    end else if (count_en) begin
        if (count == MAX_COUNT[$clog2(COUNT_MAX*COUNT_INC)-1:0]) begin
            count <= 'd0;
        end else begin
            count <= count + COUNT_INC;
        end
    end
end

endmodule