`timescale 1ns/1ps
`default_nettype none

// Production-facing actual-path service for SRNN M2 learning.  It tags the
// sample-major recurrent commit stream, stores one complete window, then
// exposes the timestep-major replay contract consumed by the coordinator.
module unified_srnn_m2_live_actual_replay_service #(
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int BACKBONE_MAX = 450,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [$clog2(BATCH_MAX+1)-1:0]            i_cfg_batch_count,
    input  logic [$clog2(TIMESTEP_MAX+1)-1:0]         i_cfg_timestep_count,
    input  logic [$clog2(BACKBONE_MAX+1)-1:0]         i_cfg_backbone_count,
    input  logic [VERSION_WIDTH-1:0]                  i_cfg_expected_version,
    input  logic                                      i_cfg_check_version,
    output logic                                      o_cfg_error,

    input  logic                                      i_capture_start,
    output logic                                      o_capture_start_ready,
    input  logic                                      i_abort,
    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_commit_version,
    input  logic                                      i_commit_sample_last,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_commit_spikes,
    output logic                                      o_capture_done,

    input  logic                                      i_replay_start,
    output logic                                      o_replay_start_ready,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_replay_timestep_id,
    output logic                                      o_replay_valid,
    input  logic                                      i_replay_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_replay_sample_id,
    output logic [SOURCE_VECTOR_MAX-1:0]              o_replay_spikes,
    output logic                                      o_replay_done,

    output logic                                      o_busy,
    output logic                                      o_configured,
    output logic                                      o_protocol_error,
    output logic                                      o_version_error,
    output logic                                      o_tail_error
);

    logic adapter_cfg_ready;
    logic adapter_cfg_error;
    logic adapter_window_start_ready;
    logic adapter_capture_valid;
    logic adapter_capture_ready;
    logic [$clog2(BATCH_MAX)-1:0] adapter_capture_sample;
    logic [$clog2(TIMESTEP_MAX)-1:0] adapter_capture_timestep;
    logic [SOURCE_VECTOR_MAX-1:0] adapter_capture_spikes;
    logic adapter_window_done;
    logic adapter_busy;
    logic adapter_configured;
    logic adapter_protocol_error;
    logic adapter_version_error;

    logic replay_cfg_ready;
    logic replay_capture_done;
    logic replay_busy;
    logic replay_configured;
    logic replay_protocol_error;
    logic replay_tail_error;

    logic cfg_fire;
    logic capture_start_fire;
    logic replay_start_fire;
    logic adapter_done_seen_q;
    logic replay_done_seen_q;
    logic capture_complete_q;

    always_comb begin
        o_cfg_ready = adapter_cfg_ready && replay_cfg_ready;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        o_capture_start_ready = adapter_window_start_ready &&
            replay_cfg_ready && adapter_configured && replay_configured;
        capture_start_fire = i_capture_start && o_capture_start_ready;

        o_replay_start_ready = capture_complete_q && replay_cfg_ready &&
            replay_configured && !adapter_busy;
        replay_start_fire = i_replay_start && o_replay_start_ready;

        o_cfg_error = adapter_cfg_error;
        o_busy = adapter_busy || replay_busy;
        o_configured = adapter_configured && replay_configured;
        o_protocol_error = adapter_protocol_error || replay_protocol_error;
        o_version_error = adapter_version_error;
        o_tail_error = replay_tail_error;
    end

    unified_srnn_m2_recurrent_commit_id_adapter #(
        .BATCH_MAX(BATCH_MAX),
        .TIMESTEP_MAX(TIMESTEP_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) commit_adapter (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire),
        .o_cfg_ready(adapter_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_timestep_count(i_cfg_timestep_count),
        .i_cfg_expected_version(i_cfg_expected_version),
        .i_cfg_check_version(i_cfg_check_version),
        .o_cfg_error(adapter_cfg_error),
        .i_window_start(capture_start_fire),
        .o_window_start_ready(adapter_window_start_ready),
        .i_abort_window(i_abort),
        .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready),
        .i_commit_version(i_commit_version),
        .i_commit_sample_last(i_commit_sample_last),
        .i_commit_spikes(i_commit_spikes),
        .o_capture_valid(adapter_capture_valid),
        .i_capture_ready(adapter_capture_ready),
        .o_capture_sample_id(adapter_capture_sample),
        .o_capture_timestep_id(adapter_capture_timestep),
        .o_capture_spikes(adapter_capture_spikes),
        .o_window_done(adapter_window_done),
        .o_busy(adapter_busy),
        .o_configured(adapter_configured),
        .o_protocol_error(adapter_protocol_error),
        .o_version_error(adapter_version_error)
    );

    unified_srnn_m2_actual_backbone_replay_buffer #(
        .BATCH_MAX(BATCH_MAX),
        .TIMESTEP_MAX(TIMESTEP_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX)
    ) replay_buffer (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire),
        .o_cfg_ready(replay_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_timestep_count(i_cfg_timestep_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_capture_start(capture_start_fire),
        .i_capture_valid(adapter_capture_valid),
        .o_capture_ready(adapter_capture_ready),
        .i_capture_sample_id(adapter_capture_sample),
        .i_capture_timestep_id(adapter_capture_timestep),
        .i_capture_spikes(adapter_capture_spikes),
        .o_capture_done(replay_capture_done),
        .i_replay_start(replay_start_fire),
        .i_replay_timestep_id(i_replay_timestep_id),
        .o_replay_valid(o_replay_valid),
        .i_replay_ready(i_replay_ready),
        .o_replay_sample_id(o_replay_sample_id),
        .o_replay_spikes(o_replay_spikes),
        .o_replay_done(o_replay_done),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .o_busy(replay_busy),
        .o_configured(replay_configured),
        .o_protocol_error(replay_protocol_error),
        .o_tail_error(replay_tail_error)
    );

    // Completion validity gates replay-buffer reads. Keep it on the same
    // synchronous reset boundary as the inferred replay BRAM controls.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            adapter_done_seen_q <= 1'b0;
            replay_done_seen_q <= 1'b0;
            capture_complete_q <= 1'b0;
            o_capture_done <= 1'b0;
        end else begin
            o_capture_done <= 1'b0;
            if (cfg_fire || i_abort || capture_start_fire) begin
                adapter_done_seen_q <= 1'b0;
                replay_done_seen_q <= 1'b0;
                capture_complete_q <= 1'b0;
            end else if ((adapter_done_seen_q || adapter_window_done) &&
                    (replay_done_seen_q || replay_capture_done)) begin
                adapter_done_seen_q <= 1'b0;
                replay_done_seen_q <= 1'b0;
                capture_complete_q <= 1'b1;
                o_capture_done <= 1'b1;
            end else begin
                if (adapter_window_done)
                    adapter_done_seen_q <= 1'b1;
                if (replay_capture_done)
                    replay_done_seen_q <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
