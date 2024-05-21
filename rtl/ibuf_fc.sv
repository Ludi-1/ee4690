module ibuf_fc #(
    parameter img_width = 28, // INPUT IMAGE SIZE / AMUONT OF INPUT NEURONS
    parameter fifo_length = img_width**2
) (
    input clk,
    input i_write_enable,
    input i_data,
    output reg o_data [fifo_length-1:0]
);

always @(posedge clk) begin
    if (i_write_enable) begin
        o_data[0] <= i_data;
        for (int fifo_idx = 0; fifo_idx < fifo_length - 1; fifo_idx++) begin
            o_data[fifo_idx + 1] <= o_data[fifo_idx]; 
        end
    end
end

endmodule