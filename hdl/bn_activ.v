// This module takes in a count value from popcount and returns a value after BN and Activation, to be used in next layer
module bn_activ #(
    parameter SIZE = 4,  // size of count_value
    parameter THRESHOLD = 0.5
) (
    input [SIZE-1:0] count_value,
    output activ_out
);
    assign activ_out = (count_value >= THRESHOLD) ? 1 : 0;

endmodule