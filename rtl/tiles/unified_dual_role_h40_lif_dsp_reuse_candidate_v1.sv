`timescale 1ns/1ps
`default_nettype none

// H40 tile whose DSP feedback can finalize a threshold decision without a
// lane-wide LUT comparator/subtractor. Before i_finalize, P must contain
// membrane-threshold. Finalize clears spiking lanes and restores threshold
// to non-spiking lanes, yielding reset-or-retain state directly in P.
module unified_dual_role_h40_lif_dsp_reuse_candidate_v1 #(
    parameter int LANES = 40
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic                    i_last,
    input  wire logic                    i_finalize,
    input  wire logic [LANES-1:0]        i_lane_mask,
    input  wire logic [LANES-1:0]        i_clear_p_mask,
    input  wire logic [LANES-1:0]        i_enable_a,
    input  wire logic [LANES-1:0]        i_enable_d,
    input  wire logic [LANES*25-1:0]     i_term_a_flat,
    input  wire logic [LANES*25-1:0]     i_term_d_flat,
    input  wire logic [LANES*18-1:0]     i_factor_b_flat,
    input  wire logic [LANES*25-1:0]     i_threshold_flat,
    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic [LANES-1:0]             o_result_lane_mask,
    output logic [LANES-1:0]             o_spike,
    output logic [LANES*48-1:0]          o_result_flat
);
    logic result_valid_q;
    logic [LANES-1:0] active_lane_mask_q;
    logic [LANES-1:0] spike_q;
    logic input_fire;

    assign o_ready = !i_abort && (!result_valid_q || i_result_ready);
    assign input_fire = i_valid && o_ready;
    assign o_result_valid = !i_abort && result_valid_q;
    assign o_result_lane_mask = active_lane_mask_q;
    assign o_spike = spike_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            logic lane_is_spike;
            logic effective_clear_p;
            logic effective_enable_a;
            logic effective_enable_d;
            logic signed [24:0] effective_term_a;
            logic signed [24:0] effective_term_d;
            logic signed [17:0] effective_factor_b;

            assign lane_is_spike = !o_result_flat[lane*48 + 47];
            assign effective_clear_p = i_finalize ? lane_is_spike :
                i_clear_p_mask[lane];
            assign effective_enable_a = i_finalize ? !lane_is_spike :
                i_enable_a[lane];
            assign effective_enable_d = i_finalize ? 1'b0 : i_enable_d[lane];
            assign effective_term_a = i_finalize ?
                $signed(i_threshold_flat[lane*25 +: 25]) :
                $signed(i_term_a_flat[lane*25 +: 25]);
            assign effective_term_d = i_finalize ? 25'sd0 :
                $signed(i_term_d_flat[lane*25 +: 25]);
            assign effective_factor_b = i_finalize ? 18'sd1 :
                $signed(i_factor_b_flat[lane*18 +: 18]);

            unified_dual_role_dsp_lane_candidate_v4 u_lane (
                .i_clk(i_clk),
                .i_fire(input_fire && i_lane_mask[lane]),
                .i_clear_p(effective_clear_p),
                .i_enable_a(effective_enable_a),
                .i_enable_d(effective_enable_d),
                .i_term_a(effective_term_a),
                .i_term_d(effective_term_d),
                .i_factor_b(effective_factor_b),
                .o_result(o_result_flat[lane*48 +: 48])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            result_valid_q <= 1'b0;
            active_lane_mask_q <= '0;
            spike_q <= '0;
        end else if (i_abort) begin
            result_valid_q <= 1'b0;
            active_lane_mask_q <= '0;
            spike_q <= '0;
        end else begin
            if (result_valid_q && i_result_ready)
                result_valid_q <= 1'b0;
            if (input_fire) begin
                if (i_clear_p_mask == i_lane_mask)
                    active_lane_mask_q <= i_lane_mask;
                if (i_finalize)
                    for (integer lane = 0; lane < LANES; lane++)
                        if (i_lane_mask[lane])
                            spike_q[lane] <= !o_result_flat[lane*48 + 47];
                result_valid_q <= i_last;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("DSP-reuse LIF candidate requires exactly 40 lanes");
    end

endmodule

`default_nettype wire
