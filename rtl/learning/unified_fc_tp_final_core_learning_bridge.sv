`timescale 1ns/1ps
`default_nettype none

// FC topology adapter around the shared TP learning backend. It mirrors the
// production K8 ingress, sequences stable observation capture by sample and
// forwards all updates to the native resident weight service used by inference.
module unified_fc_tp_final_core_learning_bridge #(
    parameter bit ENABLE_W0_LEARNING = 1'b1,
    parameter bit USE_EXTERNAL_W2_LEARNER = 1'b0,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int LANES = 8,
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 480,
    parameter int MAX_FEATURES = 2312,
    parameter int MAX_TARGET_DIMENSION = 16,
    parameter int MAX_OUTPUTS = 10,
    parameter int MAX_EVENTS = 8192,
    parameter int MAX_WINDOWS_PER_SAMPLE = 512,
    parameter int TARGET_TRACE_WIDTH = 20,
    parameter int W2_MAX_GRADIENT_ELEMENTS = 32768,
    parameter int W0_PARAMETER_HIDDEN_STRIDE = 0,
    parameter int W2_PARAMETER_OUTPUT_STRIDE = 0,
    parameter int VMEM_WIDTH = 18,
    parameter int LOGIT_WIDTH = 24,
    parameter int LOGIT_SCALE_WIDTH = 24,
    parameter int SAMPLE_ID_WIDTH = 16,
    parameter int SAMPLE_INDEX_WIDTH =
        (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH =
        (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int FEATURE_INDEX_WIDTH =
        (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int TARGET_INDEX_WIDTH =
        (MAX_TARGET_DIMENSION <= 2) ? 1 : $clog2(MAX_TARGET_DIMENSION),
    parameter int TARGET_COUNT_WIDTH = $clog2(MAX_TARGET_DIMENSION + 1),
    parameter int OUTPUT_INDEX_WIDTH =
        (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1),
    parameter int WINDOW_COUNT_WIDTH = $clog2(MAX_WINDOWS_PER_SAMPLE + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_clear_error,

    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_cfg_w0_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_w0_parameter_base,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w0_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w0_vote_threshold,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_cfg_w2_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_w2_parameter_base,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w2_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w2_vote_threshold,
    input  logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]         i_cfg_hidden_count,
    input  logic [FEATURE_COUNT_WIDTH-1:0]        i_cfg_feature_count,
    input  logic [TARGET_COUNT_WIDTH-1:0]         i_cfg_target_dimension,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]         i_cfg_output_count,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w0_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w0_code_max,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w2_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w2_code_max,
    input  logic signed [VMEM_WIDTH-1:0]          i_cfg_vth,
    input  logic [LOGIT_SCALE_WIDTH-1:0]          i_cfg_logit_to_q6_scale,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  logic                                  i_release,
    output logic                                  o_release_ready,
    input  logic                                  i_learning_enable,

    input  logic                                  i_window_valid,
    output logic                                  o_window_ready,
    input  logic [LANES-1:0]                     i_record_valid,
    input  logic [SAMPLE_ID_WIDTH-1:0]           i_window_sample_id,
    input  logic [8:0]                            i_window_timestep,
    input  logic [LANES-1:0]                     i_boundary_mask,
    input  logic                                  i_window_sample_last,
    input  logic [LANES*FEATURE_INDEX_WIDTH-1:0] i_addr_flat,
    input  logic [LANES*8-1:0]                   i_data_flat,

    output logic                                  o_infer_window_valid,
    input  logic                                  i_infer_window_ready,
    output logic [LANES-1:0]                     o_infer_record_valid,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_infer_window_sample_id,
    output logic [8:0]                            o_infer_window_timestep,
    output logic [LANES-1:0]                     o_infer_boundary_mask,
    output logic                                  o_infer_window_sample_last,
    output logic [LANES*FEATURE_INDEX_WIDTH-1:0] o_infer_addr_flat,
    output logic [LANES*8-1:0]                   o_infer_data_flat,

    input  logic                                  i_sample_compute_done,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]         i_snapshot_label,
    input  logic [MAX_HIDDEN-1:0]                i_snapshot_spikes_flat,
    input  logic [MAX_HIDDEN*16-1:0]             i_snapshot_trace_flat,
    input  logic [MAX_HIDDEN*VMEM_WIDTH-1:0]     i_snapshot_vmem_flat,
    input  logic [MAX_OUTPUTS*LOGIT_WIDTH-1:0]    i_snapshot_logits_flat,

    input  logic                                  i_target_hidden_trace_valid,
    output logic                                  o_target_hidden_trace_ready,
    input  logic [SAMPLE_INDEX_WIDTH-1:0]         i_target_hidden_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_target_hidden_trace_hidden,
    input  logic [16:0]                           i_target_hidden_trace_q016,
    input  logic                                  i_target_input_trace_valid,
    output logic                                  o_target_input_trace_ready,
    input  logic [SAMPLE_INDEX_WIDTH-1:0]         i_target_input_trace_sample,
    input  logic [TARGET_INDEX_WIDTH-1:0]         i_target_input_trace_index,
    input  logic [TARGET_TRACE_WIDTH-1:0]         i_target_input_trace_q016,
    input  logic                                  i_target_done,
    output logic                                  o_target_done_ready,

    output logic                                  o_window_commit_valid,
    input  logic                                  i_window_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_window_commit_profile_id,
    output logic [31:0]                           o_window_commit_sequence,
    output logic                                  o_sample_learning_done,
    output logic                                  o_batch_learning_done,
    output logic                                  o_inference_hold,
    output logic                                  o_window_busy,

    output logic                                  o_native_read_valid,
    input  logic                                  i_native_read_ready,
    output logic [TENSOR_ID_WIDTH-1:0]            o_native_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_native_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_native_read_tag,
    input  logic                                  i_native_response_valid,
    output logic                                  o_native_response_ready,
    input  logic [TAG_WIDTH-1:0]                  i_native_response_tag,
    input  logic signed [CODE_WIDTH-1:0]          i_native_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]      i_native_response_residual,
    input  logic                                  i_native_response_error,
    output logic                                  o_native_write_valid,
    input  logic                                  i_native_write_ready,
    output logic [TENSOR_ID_WIDTH-1:0]            o_native_write_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_native_write_parameter_addr,
    output logic signed [CODE_WIDTH-1:0]          o_native_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]      o_native_write_residual,
    output logic                                  o_native_write_code_enable,
    output logic                                  o_native_write_residual_enable,
    output logic                                  o_dirty,
    output logic                                  o_saturated,

    // Normalized W2 observation client. In external mode the shared output
    // learner owns probability, outer-product and parameter-RMW arithmetic.
    output logic                                  o_external_cfg_valid,
    input  logic                                  i_external_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_external_cfg_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_external_cfg_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_external_cfg_parameter_base,
    output logic [BATCH_COUNT_WIDTH-1:0]          o_external_cfg_batch_count,
    output logic [HIDDEN_COUNT_WIDTH-1:0]         o_external_cfg_source_count,
    output logic [OUTPUT_COUNT_WIDTH-1:0]         o_external_cfg_output_count,
    output logic [VOTE_WIDTH-1:0]                 o_external_cfg_vote_increment,
    output logic [VOTE_WIDTH-1:0]                 o_external_cfg_vote_threshold,
    output logic signed [CODE_WIDTH-1:0]          o_external_cfg_code_min,
    output logic signed [CODE_WIDTH-1:0]          o_external_cfg_code_max,
    input  logic                                  i_external_cfg_error,
    input  logic                                  i_external_active,
    output logic                                  o_external_release,
    input  logic                                  i_external_release_ready,
    output logic                                  o_external_source_trace_valid,
    input  logic                                  i_external_source_trace_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]         o_external_source_trace_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]         o_external_source_trace_index,
    output logic [15:0]                           o_external_source_trace_q016,
    input  logic                                  i_external_source_trace_error,
    output logic                                  o_external_logit_row_valid,
    input  logic                                  i_external_logit_row_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]         o_external_logit_sample_id,
    output logic [OUTPUT_INDEX_WIDTH-1:0]         o_external_logit_label,
    output logic                                  o_external_logit_valid,
    input  logic                                  i_external_logit_ready,
    output logic signed [15:0]                    o_external_logit_q6,
    input  logic                                  i_external_logit_error,
    input  logic                                  i_external_logit_row_done,
    output logic                                  o_external_observation_done,
    input  logic                                  i_external_observation_done_ready,
    input  logic                                  i_external_window_done,
    input  logic                                  i_external_window_busy,
    input  logic                                  i_external_context_error,
    input  logic                                  i_external_gradient_overflow,
    input  logic                                  i_external_update_config_error,
    input  logic                                  i_external_request_error,
    input  logic                                  i_external_store_error,
    input  logic                                  i_external_internal_error,
    input  logic                                  i_external_dirty,
    input  logic                                  i_external_saturated,

    output logic [WINDOW_COUNT_WIDTH-1:0]         o_captured_window_count,
    output logic [BATCH_COUNT_WIDTH-1:0]          o_captured_sample_count,
    output logic                                  o_observation_error,
    output logic                                  o_context_error,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_parameter_error,
    output logic                                  o_protocol_error
);
    logic cfg_fire;
    logic subsystem_cfg_ready;
    logic coordinator_cfg_ready;
    logic subsystem_active;
    logic coordinator_active;
    logic release_fire;
    logic subsystem_release_ready;
    logic coordinator_release_ready;

    logic capture_event_valid;
    logic capture_event_ready;
    logic [FEATURE_INDEX_WIDTH-1:0] capture_event_feature;
    logic capture_drain_start;
    logic capture_drain_start_ready;
    logic capture_drain_done;
    logic capture_drain_done_ready;
    logic coordinator_drain_start_ready;
    logic coordinator_drain_done;
    logic capture_idle;
    logic capture_busy;
    logic capture_protocol_error;

    logic snapshot_start;
    logic snapshot_start_ready;
    logic snapshot_busy;
    logic snapshot_done;
    logic batch_seal;
    logic batch_seal_ready;
    logic subsystem_commit_valid;
    logic subsystem_commit_ready;
    logic subsystem_window_busy;
    logic subsystem_protocol_error;
    logic coordinator_commit_ready;
    logic coordinator_protocol_error;

    assign o_cfg_ready = subsystem_cfg_ready && coordinator_cfg_ready;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign o_release_ready = subsystem_release_ready &&
        coordinator_release_ready && capture_idle;
    assign release_fire = i_release && o_release_ready;
    assign o_active = subsystem_active && coordinator_active;

    assign o_window_commit_valid = subsystem_commit_valid;
    assign subsystem_commit_ready = coordinator_commit_ready;
    assign o_window_busy = subsystem_window_busy || capture_busy ||
        o_inference_hold;
    assign o_protocol_error = subsystem_protocol_error ||
        capture_protocol_error || coordinator_protocol_error;

    // W2-only learning has no input-event gradient to replay. Preserve the
    // coordinator's drain-before-snapshot ordering with an empty drain while
    // leaving the real capture/replay handshake unchanged for W0 learning.
    assign coordinator_drain_start_ready = ENABLE_W0_LEARNING ?
        capture_drain_start_ready : 1'b1;
    assign coordinator_drain_done = ENABLE_W0_LEARNING ?
        capture_drain_done : 1'b1;

    unified_fc_tp_k8_event_capture_replay #(
        .LANES(LANES), .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .MAX_WINDOWS_PER_SAMPLE(MAX_WINDOWS_PER_SAMPLE),
        .WINDOW_COUNT_WIDTH(WINDOW_COUNT_WIDTH)
    ) u_event_capture (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_learning_enable(i_learning_enable && o_active &&
                           ENABLE_W0_LEARNING),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid),
        .i_window_sample_id(i_window_sample_id),
        .i_window_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_window_sample_last(i_window_sample_last),
        .i_addr_flat(i_addr_flat), .i_data_flat(i_data_flat),
        .o_infer_window_valid(o_infer_window_valid),
        .i_infer_window_ready(i_infer_window_ready),
        .o_infer_record_valid(o_infer_record_valid),
        .o_infer_window_sample_id(o_infer_window_sample_id),
        .o_infer_window_timestep(o_infer_window_timestep),
        .o_infer_boundary_mask(o_infer_boundary_mask),
        .o_infer_window_sample_last(o_infer_window_sample_last),
        .o_infer_addr_flat(o_infer_addr_flat),
        .o_infer_data_flat(o_infer_data_flat),
        .i_drain_start(capture_drain_start),
        .o_drain_start_ready(capture_drain_start_ready),
        .o_event_valid(capture_event_valid),
        .i_event_ready(capture_event_ready),
        .o_event_feature_index(capture_event_feature),
        .o_drain_done(capture_drain_done),
        .i_drain_done_ready(capture_drain_done_ready),
        .o_capture_idle(capture_idle), .o_busy(capture_busy),
        .o_captured_window_count(o_captured_window_count),
        .o_protocol_error(capture_protocol_error)
    );

    unified_fc_tp_sample_batch_coordinator #(
        .MAX_BATCH(MAX_BATCH), .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH)
    ) u_sample_batch_coordinator (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(coordinator_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_release(release_fire),
        .o_release_ready(coordinator_release_ready),
        .o_active(coordinator_active),
        .i_learning_enable(i_learning_enable),
        .i_sample_compute_done(i_sample_compute_done),
        .o_event_drain_start(capture_drain_start),
        .i_event_drain_start_ready(coordinator_drain_start_ready),
        .i_event_drain_done(coordinator_drain_done),
        .o_event_drain_done_ready(capture_drain_done_ready),
        .o_snapshot_start(snapshot_start),
        .i_snapshot_start_ready(snapshot_start_ready),
        .i_snapshot_done(snapshot_done), .o_batch_seal(batch_seal),
        .i_batch_seal_ready(batch_seal_ready),
        .i_window_commit_valid(subsystem_commit_valid),
        .i_window_commit_accept_ready(i_window_commit_ready),
        .o_window_commit_ready(coordinator_commit_ready),
        .o_sample_learning_done(o_sample_learning_done),
        .o_batch_learning_done(o_batch_learning_done),
        .o_inference_hold(o_inference_hold),
        .o_captured_sample_count(o_captured_sample_count),
        .o_protocol_error(coordinator_protocol_error)
    );

    unified_fc_tp_native_online_learning_subsystem #(
        .ENABLE_W0_LEARNING(ENABLE_W0_LEARNING),
        .USE_EXTERNAL_W2_LEARNER(USE_EXTERNAL_W2_LEARNER),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN), .MAX_FEATURES(MAX_FEATURES),
        .MAX_TARGET_DIMENSION(MAX_TARGET_DIMENSION),
        .MAX_OUTPUTS(MAX_OUTPUTS), .MAX_EVENTS(MAX_EVENTS),
        .TARGET_TRACE_WIDTH(TARGET_TRACE_WIDTH),
        .W2_MAX_GRADIENT_ELEMENTS(W2_MAX_GRADIENT_ELEMENTS),
        .W0_PARAMETER_HIDDEN_STRIDE(W0_PARAMETER_HIDDEN_STRIDE),
        .W2_PARAMETER_OUTPUT_STRIDE(W2_PARAMETER_OUTPUT_STRIDE),
        .VMEM_WIDTH(VMEM_WIDTH), .LOGIT_WIDTH(LOGIT_WIDTH),
        .LOGIT_SCALE_WIDTH(LOGIT_SCALE_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .TARGET_INDEX_WIDTH(TARGET_INDEX_WIDTH),
        .TARGET_COUNT_WIDTH(TARGET_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_online_subsystem (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(subsystem_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_w0_tensor_id(i_cfg_w0_tensor_id),
        .i_cfg_w0_parameter_base(i_cfg_w0_parameter_base),
        .i_cfg_w0_vote_increment(i_cfg_w0_vote_increment),
        .i_cfg_w0_vote_threshold(i_cfg_w0_vote_threshold),
        .i_cfg_w2_tensor_id(i_cfg_w2_tensor_id),
        .i_cfg_w2_parameter_base(i_cfg_w2_parameter_base),
        .i_cfg_w2_vote_increment(i_cfg_w2_vote_increment),
        .i_cfg_w2_vote_threshold(i_cfg_w2_vote_threshold),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_feature_count(i_cfg_feature_count),
        .i_cfg_target_dimension(i_cfg_target_dimension),
        .i_cfg_output_count(i_cfg_output_count),
        .i_cfg_w0_code_min(i_cfg_w0_code_min),
        .i_cfg_w0_code_max(i_cfg_w0_code_max),
        .i_cfg_w2_code_min(i_cfg_w2_code_min),
        .i_cfg_w2_code_max(i_cfg_w2_code_max),
        .i_cfg_vth(i_cfg_vth),
        .i_cfg_logit_to_q6_scale(i_cfg_logit_to_q6_scale),
        .o_cfg_error(o_cfg_error), .o_active(subsystem_active),
        .i_release(release_fire),
        .o_release_ready(subsystem_release_ready),
        .i_event_valid(capture_event_valid),
        .o_event_ready(capture_event_ready),
        .i_event_feature_index(capture_event_feature),
        .i_snapshot_start(snapshot_start),
        .o_snapshot_start_ready(snapshot_start_ready),
        .i_snapshot_label(i_snapshot_label),
        .i_snapshot_spikes_flat(i_snapshot_spikes_flat),
        .i_snapshot_trace_flat(i_snapshot_trace_flat),
        .i_snapshot_vmem_flat(i_snapshot_vmem_flat),
        .i_snapshot_logits_flat(i_snapshot_logits_flat),
        .o_snapshot_busy(snapshot_busy), .o_snapshot_done(snapshot_done),
        .i_batch_seal(batch_seal),
        .o_batch_seal_ready(batch_seal_ready),
        .i_target_hidden_trace_valid(i_target_hidden_trace_valid),
        .o_target_hidden_trace_ready(o_target_hidden_trace_ready),
        .i_target_hidden_trace_sample(i_target_hidden_trace_sample),
        .i_target_hidden_trace_hidden(i_target_hidden_trace_hidden),
        .i_target_hidden_trace_q016(i_target_hidden_trace_q016),
        .i_target_input_trace_valid(i_target_input_trace_valid),
        .o_target_input_trace_ready(o_target_input_trace_ready),
        .i_target_input_trace_sample(i_target_input_trace_sample),
        .i_target_input_trace_index(i_target_input_trace_index),
        .i_target_input_trace_q016(i_target_input_trace_q016),
        .i_target_done(i_target_done),
        .o_target_done_ready(o_target_done_ready),
        .o_window_commit_valid(subsystem_commit_valid),
        .i_window_commit_ready(subsystem_commit_ready),
        .o_window_commit_profile_id(o_window_commit_profile_id),
        .o_window_commit_sequence(o_window_commit_sequence),
        .o_window_busy(subsystem_window_busy),
        .o_observation_error(o_observation_error),
        .o_context_error(o_context_error),
        .o_arithmetic_overflow(o_arithmetic_overflow),
        .o_parameter_error(o_parameter_error),
        .o_protocol_error(subsystem_protocol_error),
        .o_native_read_valid(o_native_read_valid),
        .i_native_read_ready(i_native_read_ready),
        .o_native_read_tensor_id(o_native_read_tensor_id),
        .o_native_read_parameter_addr(o_native_read_parameter_addr),
        .o_native_read_tag(o_native_read_tag),
        .i_native_response_valid(i_native_response_valid),
        .o_native_response_ready(o_native_response_ready),
        .i_native_response_tag(i_native_response_tag),
        .i_native_response_code(i_native_response_code),
        .i_native_response_residual(i_native_response_residual),
        .i_native_response_error(i_native_response_error),
        .o_native_write_valid(o_native_write_valid),
        .i_native_write_ready(i_native_write_ready),
        .o_native_write_tensor_id(o_native_write_tensor_id),
        .o_native_write_parameter_addr(o_native_write_parameter_addr),
        .o_native_write_code(o_native_write_code),
        .o_native_write_residual(o_native_write_residual),
        .o_native_write_code_enable(o_native_write_code_enable),
        .o_native_write_residual_enable(o_native_write_residual_enable),
        .o_dirty(o_dirty), .o_saturated(o_saturated),
        .o_external_cfg_valid(o_external_cfg_valid),
        .i_external_cfg_ready(i_external_cfg_ready),
        .o_external_cfg_profile_id(o_external_cfg_profile_id),
        .o_external_cfg_tensor_id(o_external_cfg_tensor_id),
        .o_external_cfg_parameter_base(o_external_cfg_parameter_base),
        .o_external_cfg_batch_count(o_external_cfg_batch_count),
        .o_external_cfg_source_count(o_external_cfg_source_count),
        .o_external_cfg_output_count(o_external_cfg_output_count),
        .o_external_cfg_vote_increment(o_external_cfg_vote_increment),
        .o_external_cfg_vote_threshold(o_external_cfg_vote_threshold),
        .o_external_cfg_code_min(o_external_cfg_code_min),
        .o_external_cfg_code_max(o_external_cfg_code_max),
        .i_external_cfg_error(i_external_cfg_error),
        .i_external_active(i_external_active),
        .o_external_release(o_external_release),
        .i_external_release_ready(i_external_release_ready),
        .o_external_source_trace_valid(o_external_source_trace_valid),
        .i_external_source_trace_ready(i_external_source_trace_ready),
        .o_external_source_trace_sample_id(
            o_external_source_trace_sample_id),
        .o_external_source_trace_index(o_external_source_trace_index),
        .o_external_source_trace_q016(o_external_source_trace_q016),
        .i_external_source_trace_error(i_external_source_trace_error),
        .o_external_logit_row_valid(o_external_logit_row_valid),
        .i_external_logit_row_ready(i_external_logit_row_ready),
        .o_external_logit_sample_id(o_external_logit_sample_id),
        .o_external_logit_label(o_external_logit_label),
        .o_external_logit_valid(o_external_logit_valid),
        .i_external_logit_ready(i_external_logit_ready),
        .o_external_logit_q6(o_external_logit_q6),
        .i_external_logit_error(i_external_logit_error),
        .i_external_logit_row_done(i_external_logit_row_done),
        .o_external_observation_done(o_external_observation_done),
        .i_external_observation_done_ready(
            i_external_observation_done_ready),
        .i_external_window_done(i_external_window_done),
        .i_external_window_busy(i_external_window_busy),
        .i_external_context_error(i_external_context_error),
        .i_external_gradient_overflow(i_external_gradient_overflow),
        .i_external_update_config_error(i_external_update_config_error),
        .i_external_request_error(i_external_request_error),
        .i_external_store_error(i_external_store_error),
        .i_external_internal_error(i_external_internal_error),
        .i_external_dirty(i_external_dirty),
        .i_external_saturated(i_external_saturated)
    );

    // Snapshot busy is intentionally internal: the coordinator waits for its
    // completion pulse before releasing the per-sample event buffer.
    logic unused_snapshot_busy;
    assign unused_snapshot_busy = snapshot_busy;

endmodule

`default_nettype wire
