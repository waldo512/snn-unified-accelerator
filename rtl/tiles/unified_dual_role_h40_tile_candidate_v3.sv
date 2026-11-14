`timescale 1ns/1ps
`default_nettype none

module unified_dual_role_h40_tile_candidate_v3 #(
    parameter int LANES = 40
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic                    i_first,
    input  wire logic                    i_last,
    input  wire logic [LANES-1:0]        i_lane_mask,
    input  wire logic [LANES-1:0]        i_enable_a,
    input  wire logic [LANES-1:0]        i_enable_d,
    input  wire logic [LANES*25-1:0]     i_term_a_flat,
    input  wire logic [LANES*25-1:0]     i_term_d_flat,
    input  wire logic [LANES*18-1:0]     i_factor_b_flat,
    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic [LANES-1:0]             o_result_lane_mask,
    output logic [LANES*48-1:0]          o_result_flat
);
    logic result_valid_q;
    logic [LANES-1:0] active_lane_mask_q;
    logic input_fire;

    assign o_ready = !i_abort && (!result_valid_q || i_result_ready);
    assign input_fire = i_valid && o_ready;
    assign o_result_valid = !i_abort && result_valid_q;
    assign o_result_lane_mask = active_lane_mask_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            unified_dual_role_dsp_lane_candidate_v3 u_lane (
                .i_clk(i_clk),
                .i_fire(input_fire && i_lane_mask[lane]),
                .i_first(i_first),
                .i_enable_a(i_enable_a[lane]),
                .i_enable_d(i_enable_d[lane]),
                .i_term_a($signed(i_term_a_flat[lane*25 +: 25])),
                .i_term_d($signed(i_term_d_flat[lane*25 +: 25])),
                .i_factor_b($signed(i_factor_b_flat[lane*18 +: 18])),
                .o_result(o_result_flat[lane*48 +: 48])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            result_valid_q <= 1'b0;
            active_lane_mask_q <= '0;
        end else if (i_abort) begin
            result_valid_q <= 1'b0;
            active_lane_mask_q <= '0;
        end else begin
            if (result_valid_q && i_result_ready)
                result_valid_q <= 1'b0;
            if (input_fire) begin
                if (i_first)
                    active_lane_mask_q <= i_lane_mask;
                result_valid_q <= i_last;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("dual-role H40 candidate requires exactly 40 lanes");
    end

endmodule

`default_nettype wire
