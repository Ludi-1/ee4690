module ibuf_fc #(
    parameter FIFO_LENGTH = 768,
    parameter INPUT_CHANNELS = 3
) (
    input clk,
    input i_write_enable,
    input i_data [INPUT_CHANNELS-1:0],
    output reg [FIFO_LENGTH-1:0] o_data [INPUT_CHANNELS-1:0]
);

always @(posedge clk) begin
    if (i_write_enable) begin
        for (int i = 0; i < INPUT_CHANNELS; i++) begin
            o_data[i][0] <= i_data[i];
            for (int fifo_idx = 0; fifo_idx < FIFO_LENGTH - 1; fifo_idx++) begin
                o_data[i][fifo_idx + 1] <= o_data[i][fifo_idx]; 
            end
        end
    end
end

endmodule