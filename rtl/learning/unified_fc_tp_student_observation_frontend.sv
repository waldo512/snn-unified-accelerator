`timescale 1ns/1ps
`default_nettype none

// Production FC student-observation boundary. It captures stable inference
// snapshots, replays one batch in sample-major order, and converts the native
// FC fixed-point state to the formats consumed by the shared TP engine.
module unified_fc_tp_student_observation_frontend #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 480,
    parameter int MAX_OUTPUTS = 10,
    parameter int MAX_EVENTS = 8192,
    parameter int FEATURE_INDEX_WIDTH = 12,
    parameter int VMEM_WIDTH = 18,
    parameter int LOGIT_WIDTH = 24,
    parameter int LOGIT_SCALE_WIDTH = 24,
    parameter int SAMPLE_INDEX_WIDTH =
        (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH =
        (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int OUTPUT_INDEX_WIDTH =
        (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [BATCH_COUNT_WIDTH-1:0]              i_cfg_batch_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]             i_cfg_output_count,
    input  logic signed [VMEM_WIDTH-1:0]              i_cfg_vth,
    input  logic [LOGIT_SCALE_WIDTH-1:0]              i_cfg_logit_to_q6_scale,
    input  logic                                      i_release,
    output logic                                      o_release_ready,

    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [FEATURE_INDEX_WIDTH-1:0]            i_event_feature_index,

    input  logic                                      i_snapshot_start,
    output logic                                      o_snapshot_start_ready,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]             i_snapshot_label,
    input  logic [MAX_HIDDEN-1:0]                     i_snapshot_spikes_flat,
    input  logic [MAX_HIDDEN*16-1:0]                  i_snapshot_trace_flat,
    input  logic [MAX_HIDDEN*VMEM_WIDTH-1:0]          i_snapshot_vmem_flat,
    input  logic [MAX_OUTPUTS*LOGIT_WIDTH-1:0]         i_snapshot_logits_flat,
    output logic                                      o_snapshot_busy,
    output logic                                      o_snapshot_done,

    input  logic                                      i_batch_seal,
    output logic                                      o_batch_seal_ready,

    output logic                                      o_student_trace_valid,
    input  logic                                      i_student_trace_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_student_trace_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_student_trace_hidden,
    output logic [16:0]                               o_student_trace_q016,

    output logic                                      o_surrogate_valid,
    input  logic                                      i_surrogate_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_surrogate_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_surrogate_hidden,
    output logic [16:0]                               o_surrogate_q016,

    output logic                                      o_replay_event_valid,
    input  logic                                      i_replay_event_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_replay_event_sample,
    output logic [FEATURE_INDEX_WIDTH-1:0]            o_replay_event_feature_index,

    output logic                                      o_logit_row_valid,
    input  logic                                      i_logit_row_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_logit_sample,
    output logic [OUTPUT_INDEX_WIDTH-1:0]             o_logit_label,
    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic signed [15:0]                        o_logit_q6,

    output logic                                      o_replay_done,
    input  logic                                      i_replay_done_ready,
    output logic                                      o_active,
    output logic                                      o_protocol_error
);
    logic hidden_valid;
    logic hidden_ready;
    logic [SAMPLE_INDEX_WIDTH-1:0] hidden_sample;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index;
    logic hidden_spike;
    logic [15:0] hidden_trace;
    logic signed [VMEM_WIDTH-1:0] hidden_vmem;
    logic raw_logit_valid;
    logic raw_logit_ready;
    logic signed [LOGIT_WIDTH-1:0] raw_logit;
    logic signed [VMEM_WIDTH-1:0] vth_q;
    logic [LOGIT_SCALE_WIDTH-1:0] logit_scale_q;
    logic cfg_fire;

    assign cfg_fire = i_cfg_valid && o_cfg_ready;

    unified_fc_tp_student_observation_replay #(
        .MAX_BATCH(MAX_BATCH), .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_OUTPUTS(MAX_OUTPUTS), .MAX_EVENTS(MAX_EVENTS),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH), .LOGIT_WIDTH(LOGIT_WIDTH),
        .SAMPLE_INDEX_WIDTH(SAMPLE_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_replay (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_cfg_valid(i_cfg_valid), .o_cfg_ready(o_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_output_count(i_cfg_output_count),
        .i_release(i_release), .o_release_ready(o_release_ready),
        .i_event_valid(i_event_valid), .o_event_ready(o_event_ready),
        .i_event_feature_index(i_event_feature_index),
        .i_snapshot_start(i_snapshot_start),
        .o_snapshot_start_ready(o_snapshot_start_ready),
        .i_snapshot_label(i_snapshot_label),
        .i_snapshot_spikes_flat(i_snapshot_spikes_flat),
        .i_snapshot_trace_flat(i_snapshot_trace_flat),
        .i_snapshot_vmem_flat(i_snapshot_vmem_flat),
        .i_snapshot_logits_flat(i_snapshot_logits_flat),
        .o_snapshot_busy(o_snapshot_busy), .o_snapshot_done(o_snapshot_done),
        .i_batch_seal(i_batch_seal),
        .o_batch_seal_ready(o_batch_seal_ready),
        .o_hidden_valid(hidden_valid), .i_hidden_ready(hidden_ready),
        .o_hidden_sample(hidden_sample), .o_hidden_index(hidden_index),
        .o_hidden_spike(hidden_spike), .o_hidden_trace(hidden_trace),
        .o_hidden_vmem(hidden_vmem),
        .o_replay_event_valid(o_replay_event_valid),
        .i_replay_event_ready(i_replay_event_ready),
        .o_replay_event_sample(o_replay_event_sample),
        .o_replay_event_feature_index(o_replay_event_feature_index),
        .o_logit_row_valid(o_logit_row_valid),
        .i_logit_row_ready(i_logit_row_ready),
        .o_logit_sample(o_logit_sample), .o_logit_label(o_logit_label),
        .o_logit_valid(raw_logit_valid), .i_logit_ready(raw_logit_ready),
        .o_logit(raw_logit), .o_replay_done(o_replay_done),
        .i_replay_done_ready(i_replay_done_ready), .o_active(o_active),
        .o_protocol_error(o_protocol_error)
    );

    unified_fc_tp_student_numeric_adapter #(
        .SAMPLE_ID_WIDTH(SAMPLE_INDEX_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH), .LOGIT_WIDTH(LOGIT_WIDTH),
        .LOGIT_SCALE_WIDTH(LOGIT_SCALE_WIDTH)
    ) u_numeric (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_hidden_valid(hidden_valid), .o_hidden_ready(hidden_ready),
        .i_hidden_sample(hidden_sample), .i_hidden_index(hidden_index),
        .i_hidden_spike(hidden_spike),
        .i_hidden_trace_q016(hidden_trace),
        .i_hidden_vmem_post(hidden_vmem), .i_vth(vth_q),
        .o_student_trace_valid(o_student_trace_valid),
        .i_student_trace_ready(i_student_trace_ready),
        .o_student_trace_sample(o_student_trace_sample),
        .o_student_trace_hidden(o_student_trace_hidden),
        .o_student_trace_q016(o_student_trace_q016),
        .o_surrogate_valid(o_surrogate_valid),
        .i_surrogate_ready(i_surrogate_ready),
        .o_surrogate_sample(o_surrogate_sample),
        .o_surrogate_hidden(o_surrogate_hidden),
        .o_surrogate_q016(o_surrogate_q016),
        .i_logit_valid(raw_logit_valid), .o_logit_ready(raw_logit_ready),
        .i_logit_raw(raw_logit), .i_logit_to_q6_scale(logit_scale_q),
        .o_logit_valid(o_logit_valid), .i_logit_ready(i_logit_ready),
        .o_logit_q6(o_logit_q6)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            vth_q <= '0;
            logit_scale_q <= '0;
        end else if (cfg_fire) begin
            vth_q <= i_cfg_vth;
            logit_scale_q <= i_cfg_logit_to_q6_scale;
        end
    end
endmodule

`default_nettype wire
