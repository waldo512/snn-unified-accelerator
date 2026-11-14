`timescale 1ns/1ps
`default_nettype none

// FC result pipeline around the shared FC/SRNN state engine. W0 production,
// W2 reduction and output buffering retain the existing FC_COMPAT contract;
// only the neuron state/LIF implementation is shared with SRNN.
module unified_h40_fc_srnn_shared_compute_tile_candidate_v1 #(
    parameter integer TILE_BASE = 0,
    parameter integer H_COUNT = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer W0_BITS = 5,
    parameter integer FC_ACC_BITS = 18,
    parameter integer W0_INPUT_SHIFT = 6,
    parameter integer O_DIM = 10,
    parameter integer W2_BITS = 8,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer T_BITS = 9
) (
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         i_abort,
    input  logic                                         i_mode_fc,
    input  logic                                         i_fc_clear_state,
    input  logic                                         i_fc_step_start,
    output logic                                         o_fc_step_ready,
    input  logic [T_BITS-1:0]                            i_fc_timestep,
    input  logic                                         i_fc_row_valid,
    output logic                                         o_fc_row_ready,
    input  logic [SOURCE_LANES-1:0]                      i_fc_lane_valid,
    input  logic [SOURCE_LANES*H_COUNT*W0_BITS-1:0]      i_fc_w0_weight_rows_flat,
    input  logic                                         i_fc_step_close,
    input  logic [H_COUNT*O_DIM*W2_BITS-1:0]             i_fc_w2_weight_flat,
    output logic                                         o_fc_result_valid,
    input  logic                                         i_fc_result_ready,
    output logic [T_BITS-1:0]                            o_fc_result_timestep,
    output logic [31:0]                                  o_fc_result_events,
    output logic [H_COUNT-1:0]                           o_fc_result_spikes,
    output logic [O_DIM*OUT_ACC_BITS-1:0]                o_fc_partial_logits_flat,
    output logic [H_COUNT*FC_ACC_BITS-1:0]               o_fc_vmem_flat,
    output logic [H_COUNT*16-1:0]                        o_fc_trace_flat,

    input  logic                                         i_srnn_clear_spikes,
    input  logic                                         i_srnn_clear_word_valid,
    input  logic [9:0]                                   i_srnn_clear_word_base,
    input  logic                                         i_srnn_accumulate_valid,
    input  logic [9:0]                                   i_srnn_accumulate_base,
    input  logic [SOURCE_LANES-1:0]                      i_srnn_accumulate_lane_mask,
    input  logic [SOURCE_LANES*32-1:0]                   i_srnn_accumulate_contribution_flat,
    input  logic [9:0]                                   i_srnn_active_neurons,
    input  logic                                         i_srnn_lif_begin,
    input  logic                                         i_srnn_lif_word_valid,
    input  logic [9:0]                                   i_srnn_lif_word_base,
    input  logic signed [15:0]                           i_srnn_leak_q15,
    input  logic signed [31:0]                           i_srnn_threshold,
    input  logic [4:0]                                   i_srnn_current_left_shift,
    input  logic                                         i_srnn_reset_to_zero,
    input  logic                                         i_srnn_commit_timestep,
    output logic [H_COUNT-1:0]                           o_srnn_previous_spikes,
    output logic [H_COUNT-1:0]                           o_srnn_current_spikes,
    output logic [H_COUNT*32-1:0]                        o_srnn_accumulator_flat,
    output logic [H_COUNT*32-1:0]                        o_srnn_membrane_flat,

    output logic                                         o_busy,
    output logic                                         o_protocol_error
);
    logic state_result_ready;
    logic state_result_start;
    logic state_done;
    logic [T_BITS-1:0] state_timestep;
    logic [31:0] state_events;
    logic [H_COUNT-1:0] state_spikes;
    logic [15:0] state_spike_count;
    logic state_producer_active;
    logic state_consumer_active;
    logic [31:0] state_overlap_cycles;
    logic state_protocol_error;
    logic w2_done;
    logic [O_DIM*OUT_ACC_BITS-1:0] w2_partial_flat;
    logic w2_inflight_q;
    logic [T_BITS-1:0] pending_timestep_q;
    logic [31:0] pending_events_q;
    logic [H_COUNT-1:0] pending_spikes_q;
    logic local_protocol_error_q;

    assign state_result_ready = !w2_inflight_q &&
        (!o_fc_result_valid || i_fc_result_ready);
    assign o_busy = state_producer_active || state_consumer_active ||
        w2_inflight_q || o_fc_result_valid;
    assign o_protocol_error = state_protocol_error || local_protocol_error_q;

    unified_h40_fc_srnn_shared_state_engine_candidate_v1 #(
        .TILE_BASE(TILE_BASE),
        .H_COUNT(H_COUNT),
        .SOURCE_LANES(SOURCE_LANES),
        .W_BITS(W0_BITS),
        .FC_ACC_BITS(FC_ACC_BITS),
        .INPUT_SHIFT(W0_INPUT_SHIFT),
        .T_BITS(T_BITS)
    ) u_state (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_mode_fc(i_mode_fc),
        .i_fc_clear_state(i_fc_clear_state),
        .i_fc_step_start(i_fc_step_start),
        .o_fc_step_ready(o_fc_step_ready),
        .i_fc_timestep(i_fc_timestep),
        .i_fc_row_valid(i_fc_row_valid),
        .o_fc_row_ready(o_fc_row_ready),
        .i_fc_lane_valid(i_fc_lane_valid),
        .i_fc_weight_rows_flat(i_fc_w0_weight_rows_flat),
        .i_fc_step_close(i_fc_step_close),
        .i_fc_result_ready(state_result_ready),
        .o_fc_result_start(state_result_start),
        .o_fc_done(state_done),
        .o_fc_completed_timestep(state_timestep),
        .o_fc_completed_events(state_events),
        .o_fc_spikes(state_spikes),
        .o_fc_spike_count(state_spike_count),
        .o_fc_vmem_flat(o_fc_vmem_flat),
        .o_fc_trace_flat(o_fc_trace_flat),
        .o_fc_producer_active(state_producer_active),
        .o_fc_consumer_active(state_consumer_active),
        .o_fc_overlap_cycles(state_overlap_cycles),
        .i_srnn_clear_spikes(i_srnn_clear_spikes),
        .i_srnn_clear_word_valid(i_srnn_clear_word_valid),
        .i_srnn_clear_word_base(i_srnn_clear_word_base),
        .i_srnn_accumulate_valid(i_srnn_accumulate_valid),
        .i_srnn_accumulate_base(i_srnn_accumulate_base),
        .i_srnn_accumulate_lane_mask(i_srnn_accumulate_lane_mask),
        .i_srnn_accumulate_contribution_flat(
            i_srnn_accumulate_contribution_flat),
        .i_srnn_active_neurons(i_srnn_active_neurons),
        .i_srnn_lif_begin(i_srnn_lif_begin),
        .i_srnn_lif_word_valid(i_srnn_lif_word_valid),
        .i_srnn_lif_word_base(i_srnn_lif_word_base),
        .i_srnn_leak_q15(i_srnn_leak_q15),
        .i_srnn_threshold(i_srnn_threshold),
        .i_srnn_current_left_shift(i_srnn_current_left_shift),
        .i_srnn_reset_to_zero(i_srnn_reset_to_zero),
        .i_srnn_commit_timestep(i_srnn_commit_timestep),
        .o_srnn_previous_spikes(o_srnn_previous_spikes),
        .o_srnn_current_spikes(o_srnn_current_spikes),
        .o_srnn_accumulator_flat(o_srnn_accumulator_flat),
        .o_srnn_membrane_flat(o_srnn_membrane_flat),
        .o_protocol_error(state_protocol_error)
    );

    unified_h40_fc_compat_w2_partial #(
        .H_COUNT(H_COUNT),
        .O_DIM(O_DIM),
        .W_BITS(W2_BITS),
        .ACC_BITS(OUT_ACC_BITS)
    ) u_fc_w2 (
        .clk(clk),
        .rst_n(rst_n),
        .i_start(state_done),
        .i_spikes(state_spikes),
        .i_weight_flat(i_fc_w2_weight_flat),
        .o_done(w2_done),
        .o_partial_flat(w2_partial_flat)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w2_inflight_q <= 1'b0;
            pending_timestep_q <= '0;
            pending_events_q <= 32'd0;
            pending_spikes_q <= '0;
            o_fc_result_valid <= 1'b0;
            o_fc_result_timestep <= '0;
            o_fc_result_events <= 32'd0;
            o_fc_result_spikes <= '0;
            o_fc_partial_logits_flat <= '0;
            local_protocol_error_q <= 1'b0;
        end else if (i_abort || i_fc_clear_state) begin
            w2_inflight_q <= 1'b0;
            pending_timestep_q <= '0;
            pending_events_q <= 32'd0;
            pending_spikes_q <= '0;
            o_fc_result_valid <= 1'b0;
            o_fc_result_timestep <= '0;
            o_fc_result_events <= 32'd0;
            o_fc_result_spikes <= '0;
            o_fc_partial_logits_flat <= '0;
            local_protocol_error_q <= 1'b0;
        end else begin
            if (o_fc_result_valid && i_fc_result_ready)
                o_fc_result_valid <= 1'b0;

            if (state_done) begin
                if (w2_inflight_q)
                    local_protocol_error_q <= 1'b1;
                w2_inflight_q <= 1'b1;
                pending_timestep_q <= state_timestep;
                pending_events_q <= state_events;
                pending_spikes_q <= state_spikes;
            end

            if (w2_done && w2_inflight_q) begin
                if (o_fc_result_valid && !i_fc_result_ready)
                    local_protocol_error_q <= 1'b1;
                w2_inflight_q <= 1'b0;
                o_fc_result_valid <= 1'b1;
                o_fc_result_timestep <= pending_timestep_q;
                o_fc_result_events <= pending_events_q;
                o_fc_result_spikes <= pending_spikes_q;
                o_fc_partial_logits_flat <= w2_partial_flat;
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_state_result_start;
    logic [15:0] unused_state_spike_count;
    logic [31:0] unused_state_overlap_cycles;
    always_comb begin
        unused_state_result_start = state_result_start;
        unused_state_spike_count = state_spike_count;
        unused_state_overlap_cycles = state_overlap_cycles;
    end
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
