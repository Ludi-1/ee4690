module conv_window #(
    parameter kernel_dim = 3, // Only supporting symmetric kernel sizes
    parameter input_channels = 1, // Currently, only one input channel is supported.
    parameter output_channels = 16 // The amount of filters in the layer to be run in parallel
) (
    input [kernel_dim**2-1:0] window,
    output [output_channels-1:0] channels
);

reg [kernel_dim**2:0] xnor_out [output_channels-1:0];

genvar i;
generate
    for (i = 0; i < output_channels; i += 1) begin
        activation #(kernel_dim**2) a0 (
            .xnor_in(xnor_out[i]),
            .activated_out(channels[i])
        );
    end
endgenerate

// xnor_out should be defined here.

endmodule