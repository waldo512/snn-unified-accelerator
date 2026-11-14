`timescale 1ns/1ps
`default_nettype none

// Canonical FC W0 arithmetic for one H40 tile. Each valid lane supplies one
// complete INT5 weight row, so eight binary spike events update all neurons in
// one cycle. Addition intentionally wraps at ACC_BITS, matching the existing
// P3A/R19B accumulator contract.
module unified_h40_fc_compat_w0_delta #(
    parameter integer H_COUNT = 40,
    parameter integer LANES = 8,
    parameter integer W_BITS = 5,
    parameter integer ACC_BITS = 18,
    parameter integer INPUT_SHIFT = 6
) (
    input  logic [LANES-1:0]                         i_lane_valid,
    input  logic [LANES*H_COUNT*W_BITS-1:0]         i_weight_rows_flat,
    output logic [H_COUNT*ACC_BITS-1:0]              o_delta_flat
);
    logic signed [ACC_BITS-1:0] delta [0:H_COUNT-1];

    function automatic logic signed [ACC_BITS-1:0] scaled_weight(
        input logic valid,
        input logic [W_BITS-1:0] packed_weight
    );
        logic signed [W_BITS-1:0] signed_weight;
        logic signed [ACC_BITS-1:0] widened_weight;
        begin
            signed_weight = $signed(packed_weight);
            widened_weight = {{(ACC_BITS-W_BITS){signed_weight[W_BITS-1]}},
                              signed_weight};
            if (valid)
                scaled_weight = widened_weight <<< INPUT_SHIFT;
            else
                scaled_weight = '0;
        end
    endfunction

    integer hidden_i;
    integer lane_i;
    always_comb begin
        for (hidden_i = 0; hidden_i < H_COUNT; hidden_i = hidden_i + 1) begin
            delta[hidden_i] = '0;
            for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1)
                delta[hidden_i] = delta[hidden_i] + scaled_weight(
                    i_lane_valid[lane_i],
                    i_weight_rows_flat[
                        lane_i*H_COUNT*W_BITS + hidden_i*W_BITS +: W_BITS
                    ]
                );
            o_delta_flat[hidden_i*ACC_BITS +: ACC_BITS] = delta[hidden_i];
        end
    end

    initial begin
        if (ACC_BITS < W_BITS)
            $error("FC_COMPAT ACC_BITS must be at least W_BITS");
        if (INPUT_SHIFT >= ACC_BITS)
            $error("FC_COMPAT INPUT_SHIFT must be smaller than ACC_BITS");
    end
endmodule

`default_nettype wire
