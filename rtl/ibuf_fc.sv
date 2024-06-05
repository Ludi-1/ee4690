module ibuf_fc #(
    parameter img_width = 28, // INPUT IMAGE SIZE / AMUONT OF INPUT NEURONS
    parameter fifo_length = img_width**2
) (
    input clk, rst,
    input i_write_enable,
    input i_data,
    output reg [fifo_length-1:0] o_data
);

always @(posedge clk) begin
    if (rst) begin
        o_data <= 0;
    end else begin
        if (i_write_enable) begin
            for (int fifo_idx = 0; fifo_idx < fifo_length - 1; fifo_idx++) begin
                o_data[fifo_idx + 1] <= o_data[fifo_idx]; 
            end
            o_data[0] <= i_data;
        end
    end
end

endmodule