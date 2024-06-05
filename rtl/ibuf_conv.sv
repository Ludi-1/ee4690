module ibuf_conv #(
    parameter img_width = 28, // INPUT IMAGE SIZE
    parameter kernel_dim = 3,
    parameter INPUT_CHANNELS = 3,
) (
    input clk,
    input i_write_enable,
    input [INPUT_CHANNELS-1:0] i_data,
    output reg o_data [INPUT_CHANNELS-1:0][kernel_dim**2-1:0]
);

localparam fifo_length = img_width * (kernel_dim - 1) + kernel_dim;
reg fifo_data [INPUT_CHANNELS-1:0][fifo_length-1:0];

always @(posedge clk) begin
    if (i_write_enable) begin
        for (int i = 0; i < INPUT_CHANNELS; i++) begin
            fifo_data[i][0] <= i_data[i];
            for (int fifo_idx = 0; fifo_idx < fifo_length - 1; fifo_idx++) begin
                fifo_data[i][fifo_idx + 1] <= fifo_data[i][fifo_idx]; 
            end
        end
    end
end

always @(*) begin
    for (int i = 0; i < kernel_dim; i++) begin
        for (int j = 0; j < kernel_dim; j++) begin
            o_data[i + j*kernel_dim] = fifo_data[i*img_width + j];
        end
    end
end

endmodule