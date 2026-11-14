`timescale 1ns/1ps
`default_nettype none

// Arithmetic-only lane for the homogeneous dual-role H40 tile. Transaction
// ownership is shared once at tile level; lane masking drives the DSP clock
// enable instead of clearing every operand bit through LUTs.
module unified_dual_role_dsp_lane_candidate_v2 (
    input  wire logic                    i_clk,
    input  wire logic                    i_fire,
    input  wire logic                    i_first,
    input  wire logic signed [24:0]      i_term_a,
    input  wire logic signed [24:0]      i_term_d,
    input  wire logic signed [17:0]      i_factor_b,
    output logic signed [47:0]           o_result
);
    (* use_dsp = "yes" *) logic signed [47:0] accumulator_q;
    logic signed [24:0] preadd_value;
    logic signed [42:0] product_value;
    logic signed [47:0] product_extended;
    logic signed [47:0] accumulate_value;

    always_comb begin
        preadd_value = $signed(i_term_a) + $signed(i_term_d);
        product_value = $signed(preadd_value) * $signed(i_factor_b);
        product_extended = {{5{product_value[42]}}, product_value};
        accumulate_value = i_first ? product_extended :
            accumulator_q + product_extended;
    end

    always_ff @(posedge i_clk) begin
        if (i_fire)
            accumulator_q <= accumulate_value;
    end

    assign o_result = accumulator_q;

endmodule

`default_nettype wire
