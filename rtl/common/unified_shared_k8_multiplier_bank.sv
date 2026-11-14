`default_nettype none

module unified_shared_k8_multiplier_bank #(
    parameter int LANES = 8
) (
    input  logic [LANES*8-1:0]   i_lhs_flat,
    input  logic [LANES*8-1:0]   i_rhs_flat,
    output logic [LANES*16-1:0]  o_product_flat
);
    always_comb begin
        for (int lane = 0; lane < LANES; lane = lane + 1)
            o_product_flat[lane*16 +: 16] =
                $signed(i_lhs_flat[lane*8 +: 8]) *
                $signed(i_rhs_flat[lane*8 +: 8]);
    end
endmodule

`default_nettype wire
