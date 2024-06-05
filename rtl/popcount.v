module popcount #(parameter WIDTH = 8) (
    input [WIDTH-1:0] bitstring_in,
    output [$clog2(WIDTH+1)-1:0] count_out
);
    // This pads the input bitstring to a power of 2 to enable the divide and conquer structure.
    // If it turns out that this structure is too large, we may want to replace it with a dadda / wallace tree

    localparam ACTUAL_WIDTH = 2 ** $clog2(WIDTH);
    localparam PADDING = ACTUAL_WIDTH - WIDTH;

    wire [ACTUAL_WIDTH-1:0] extended_in;

    assign extended_in = {{PADDING{1'b0}}, bitstring_in};

    genvar i;
    generate
        if (WIDTH == 1) begin
            assign count_out = bitstring_in;
        end else begin
            wire [$clog2((ACTUAL_WIDTH/2)+1)-1:0] sum0, sum1;
            popcount #(ACTUAL_WIDTH/2) u0 (
                .bitstring_in(extended_in[ACTUAL_WIDTH/2-1:0]),
                .count_out(sum0)
            );
            popcount #(ACTUAL_WIDTH/2) u1 (
                .bitstring_in(extended_in[ACTUAL_WIDTH-1:ACTUAL_WIDTH/2]),
                .count_out(sum1)
            );
            assign count_out = sum0 + sum1;
        end
    endgenerate

endmodule
