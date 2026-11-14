`timescale 1ns/1ps
`default_nettype none

// Two forty-lane DSP accumulator banks consume four signed topology terms per
// cycle. Each DSP48E1 pre-adder folds two independently enabled terms before
// the shared binary-feature factor. The tile-local sequencer merges the two
// held results only after the final contribution beat.
module unified_dual_role_h40_tile_p4all_dual_accum_candidate_v1 #(
    parameter int LANES = 40
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic                    i_first,
    input  wire logic                    i_last,
    input  wire logic                    i_use_secondary,
    input  wire logic [LANES-1:0]        i_lane_mask,
    input  wire logic [LANES-1:0]        i_enable0,
    input  wire logic [LANES-1:0]        i_enable1,
    input  wire logic [LANES-1:0]        i_enable2,
    input  wire logic [LANES-1:0]        i_enable3,
    input  wire logic [LANES*25-1:0]     i_term0_flat,
    input  wire logic [LANES*25-1:0]     i_term1_flat,
    input  wire logic [LANES*25-1:0]     i_term2_flat,
    input  wire logic [LANES*25-1:0]     i_term3_flat,
    input  wire logic [LANES*18-1:0]     i_factor_b_flat,
    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic                         o_result_has_secondary,
    output logic [LANES-1:0]             o_result_lane_mask,
    output logic [LANES*48-1:0]          o_result_primary_flat,
    output logic [LANES*48-1:0]          o_result_secondary_flat
);
    logic result_valid_q;
    logic result_has_secondary_q;
    logic [LANES-1:0] active_lane_mask_q;
    logic input_fire;

    assign o_ready = !i_abort && (!result_valid_q || i_result_ready);
    assign input_fire = i_valid && o_ready;
    assign o_result_valid = !i_abort && result_valid_q;
    assign o_result_has_secondary = result_has_secondary_q;
    assign o_result_lane_mask = active_lane_mask_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            unified_dual_role_dsp_lane_candidate_v3 u_primary (
                .i_clk(i_clk),
                .i_fire(input_fire && i_lane_mask[lane]),
                .i_first(i_first),
                .i_enable_a(i_enable0[lane]),
                .i_enable_d(i_enable1[lane]),
                .i_term_a($signed(i_term0_flat[lane*25 +: 25])),
                .i_term_d($signed(i_term1_flat[lane*25 +: 25])),
                .i_factor_b($signed(i_factor_b_flat[lane*18 +: 18])),
                .o_result(o_result_primary_flat[lane*48 +: 48])
            );

            unified_dual_role_dsp_lane_candidate_v3 u_secondary (
                .i_clk(i_clk),
                .i_fire(input_fire && i_use_secondary &&
                        i_lane_mask[lane]),
                .i_first(i_first),
                .i_enable_a(i_enable2[lane]),
                .i_enable_d(i_enable3[lane]),
                .i_term_a($signed(i_term2_flat[lane*25 +: 25])),
                .i_term_d($signed(i_term3_flat[lane*25 +: 25])),
                .i_factor_b($signed(i_factor_b_flat[lane*18 +: 18])),
                .o_result(o_result_secondary_flat[lane*48 +: 48])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            result_valid_q <= 1'b0;
            result_has_secondary_q <= 1'b0;
            active_lane_mask_q <= '0;
        end else if (i_abort) begin
            result_valid_q <= 1'b0;
            result_has_secondary_q <= 1'b0;
            active_lane_mask_q <= '0;
        end else begin
            if (result_valid_q && i_result_ready)
                result_valid_q <= 1'b0;
            if (input_fire) begin
                if (i_first) begin
                    result_has_secondary_q <= i_use_secondary;
                    active_lane_mask_q <= i_lane_mask;
                end
                result_valid_q <= i_last;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("P4-all dual accumulator requires 40 lanes");
    end
endmodule

`default_nettype wire
