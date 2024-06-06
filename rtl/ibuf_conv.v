module ibuf_conv #(
    parameter IMG_DIM = 28, // INPUT IMAGE SIZE
    parameter KERNEL_DIM = 3,
    parameter INPUT_CHANNELS = 1
) (
    input clk,
    input i_write_enable,
    input i_data [INPUT_CHANNELS-1:0],
    output reg [KERNEL_DIM**2-1:0] o_data [INPUT_CHANNELS-1:0]
);

localparam FIFO_LENGTH = IMG_DIM * (KERNEL_DIM - 1) + KERNEL_DIM;
reg fifo_data [INPUT_CHANNELS-1:0][FIFO_LENGTH-1:0];

always @(posedge clk) begin
    if (i_write_enable) begin
        for (int i = 0; i < INPUT_CHANNELS; i++) begin
            fifo_data[i][0] <= i_data[i];
            for (int fifo_idx = 0; fifo_idx < FIFO_LENGTH - 1; fifo_idx++) begin
                fifo_data[i][fifo_idx + 1] <= fifo_data[i][fifo_idx]; 
            end
        end
    end
end

always @(*) begin
    for (int i = 0; i < INPUT_CHANNELS; i++) begin
        for (int j = 0; j < KERNEL_DIM; j++) begin
            for (int k = 0; k < KERNEL_DIM; k++) begin
                o_data[i][j+ k*KERNEL_DIM] = fifo_data[i][j*IMG_DIM+k];
            end
        end
    end
end

initial begin
    $dumpfile("dump_ibuf_conf.fst");
    for (int i = 0; i < INPUT_CHANNELS; i++) begin
        $dumpvars(0, i_data[i]);
        for (int j = 0; j < KERNEL_DIM**2; j++) begin
            $dumpvars(0, o_data[i]);
        end
    end
end

endmodule