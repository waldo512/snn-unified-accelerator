`timescale 1ns/1ps
`default_nettype none

// Forty identical lanes form one physical tile. Topology adapters only choose
// the operands; they do not own arithmetic. FC/SRNN map lanes to destination
// neurons, while SCNN maps them to spatial positions of an output channel.
module unified_dual_role_h40_tile_candidate_v1 #(
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
    input  wire logic [LANES*25-1:0]     i_term_a_flat,
    input  wire logic [LANES*25-1:0]     i_term_d_flat,
    input  wire logic [LANES*18-1:0]     i_factor_b_flat,
    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic [LANES*48-1:0]          o_result_flat
);
    logic [LANES-1:0] lane_ready;
    logic [LANES-1:0] lane_result_valid;
    logic [LANES*48-1:0] lane_result_flat;

    assign o_ready = &lane_ready;
    assign o_result_valid = &lane_result_valid;
    assign o_result_flat = lane_result_flat;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
            logic signed [24:0] term_a;
            logic signed [24:0] term_d;
            logic signed [17:0] factor_b;

            assign term_a = i_lane_mask[lane] ? $signed(
                i_term_a_flat[lane*25 +: 25]) : 25'sd0;
            assign term_d = i_lane_mask[lane] ? $signed(
                i_term_d_flat[lane*25 +: 25]) : 25'sd0;
            assign factor_b = i_lane_mask[lane] ? $signed(
                i_factor_b_flat[lane*18 +: 18]) : 18'sd0;

            unified_dual_role_dsp_lane_candidate_v1 u_lane (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_abort(i_abort),
                .i_valid(i_valid),
                .o_ready(lane_ready[lane]),
                .i_first(i_first),
                .i_last(i_last),
                .i_term_a(term_a),
                .i_term_d(term_d),
                .i_factor_b(factor_b),
                .o_result_valid(lane_result_valid[lane]),
                .i_result_ready(i_result_ready),
                .o_result(lane_result_flat[lane*48 +: 48])
            );
        end
    endgenerate

    initial begin
        if (LANES != 40)
            $error("dual-role H40 candidate requires exactly 40 lanes");
    end

endmodule

`default_nettype wire
