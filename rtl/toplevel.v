module top (
    input clk, reset, w_e, data, start,
    output ready,
    output reg [$clog2(784+1)+1:0] preds [9:0]
);

// Enable for Icarus wavedump
initial begin
    $dumpfile("toplevel.dmp");
    $dumpvars();
    // $finish();
end

wire [28*28-1:0] fc0_in;
wire [$clog2(784+1)+1:0] fc0_out;
wire fc0_out_we;

wire reset_accel;

assign reset_accel = reset | start;

always_ff @(posedge(clk)) begin
    if (reset) begin
        for (int i = 0; i < 10; i++) begin
            preds[i] <= 0;
        end
    end else begin
        if (fc0_out_we) begin
            for (int i = 0; i < 10; i++) begin
                preds[i + 1] <= preds[i]; 
            end
            preds[0] <= fc0_out;
        end
    end
end


ibuf_fc #(28) buf0 (
    .clk(clk),
    .rst(reset),
    .i_write_enable(w_e),
    .i_data(data),
    .o_data(fc0_in)
);

classifier #(784, 10) fc0 (
    .clk(clk),
    .reset(reset_accel),
    .inbuffer_data(fc0_in),
    .outbuffer_data(fc0_out),
    .outbuffer_we(fc0_out_we),
    .ready(ready)
);


endmodule