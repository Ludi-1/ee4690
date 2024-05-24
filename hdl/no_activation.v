// `include "popcount.v"

module no_activation #(parameter WIDTH = 32) (
    input [WIDTH-1:0] xnor_in,
    output [$clog2(WIDTH+1)+1:0] logit
);
    // One bit extra for multiplication and one bit extra for sign
    localparam INTERMEDIATE_WIDTH = $clog2(WIDTH+1)+1;
    
    wire [INTERMEDIATE_WIDTH:0] popcount_out;
    wire [INTERMEDIATE_WIDTH:0] shifted;

    assign popcount_out[INTERMEDIATE_WIDTH:INTERMEDIATE_WIDTH-1] = 2'b0;

    popcount #(WIDTH) p0 (
        .bitstring_in(xnor_in),
        .count_out(popcount_out[$clog2(WIDTH+1)-1:0])
    );

    assign shifted = popcount_out << 1; // Multiply by 2
    assign logit = shifted - WIDTH; // Subtract N bits

    // initial begin
    //     $dumpfile("no_activation.dmp");
    //     $dumpvars();
    //     // $finish();
    // end

endmodule