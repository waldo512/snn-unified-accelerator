`timescale 1ns/1ps
`default_nettype none

module unified_dual_role_h40_two_stage_dsp_tile_candidate_v1 #(
    parameter int LANES = 40
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_acc_valid,
    output logic                         o_acc_ready,
    input  wire logic [LANES-1:0]        i_acc_lane_mask,
    input  wire logic                    i_acc_clear,
    input  wire logic [LANES-1:0]        i_acc_enable_a,
    input  wire logic [LANES-1:0]        i_acc_enable_d,
    input  wire logic [LANES*25-1:0]     i_acc_term_a_flat,
    input  wire logic [LANES*25-1:0]     i_acc_term_d_flat,
    input  wire logic [LANES*18-1:0]     i_acc_factor_b_flat,
    input  wire logic                    i_lif_valid,
    output logic                         o_lif_ready,
    input  wire logic [1:0]              i_lif_op,
    input  wire logic [LANES-1:0]        i_lif_lane_mask,
    input  wire logic [LANES*25-1:0]     i_state_flat,
    input  wire logic [LANES*18-1:0]     i_leak_flat,
    input  wire logic [LANES*25-1:0]     i_threshold_flat,
    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic [LANES-1:0]             o_result_lane_mask,
    output logic [LANES-1:0]             o_spike,
    output logic [LANES*48-1:0]          o_state_result_flat
);
    localparam logic [1:0] LIF_FINALIZE = 2'd2;
    logic result_valid_q;
    logic acc_fire;
    logic lif_fire;

    assign o_acc_ready = !i_abort;
    assign o_lif_ready = !i_abort && (!result_valid_q || i_result_ready);
    assign acc_fire = i_acc_valid && o_acc_ready;
    assign lif_fire = i_lif_valid && o_lif_ready;
    assign o_result_valid = !i_abort && result_valid_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            unified_dual_role_two_stage_dsp_lane_candidate_v1 u_lane (
                .i_clk(i_clk),
                .i_acc_fire(acc_fire && i_acc_lane_mask[lane]),
                .i_acc_clear(i_acc_clear),
                .i_acc_enable_a(i_acc_enable_a[lane]),
                .i_acc_enable_d(i_acc_enable_d[lane]),
                .i_acc_term_a($signed(i_acc_term_a_flat[lane*25 +: 25])),
                .i_acc_term_d($signed(i_acc_term_d_flat[lane*25 +: 25])),
                .i_acc_factor_b($signed(i_acc_factor_b_flat[lane*18 +: 18])),
                .i_lif_fire(lif_fire && i_lif_lane_mask[lane]),
                .i_lif_op(i_lif_op),
                .i_state($signed(i_state_flat[lane*25 +: 25])),
                .i_leak($signed(i_leak_flat[lane*18 +: 18])),
                .i_threshold($signed(i_threshold_flat[lane*25 +: 25])),
                .o_spike(o_spike[lane]),
                .o_state_result(o_state_result_flat[lane*48 +: 48])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            result_valid_q <= 1'b0;
            o_result_lane_mask <= '0;
        end else if (i_abort) begin
            result_valid_q <= 1'b0;
            o_result_lane_mask <= '0;
        end else begin
            if (result_valid_q && i_result_ready)
                result_valid_q <= 1'b0;
            if (lif_fire && i_lif_op == LIF_FINALIZE) begin
                result_valid_q <= 1'b1;
                o_result_lane_mask <= i_lif_lane_mask;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("two-stage DSP tile requires exactly 40 lanes");
    end
endmodule

`default_nettype wire
