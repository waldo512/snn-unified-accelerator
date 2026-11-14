`timescale 1ns/1ps
`default_nettype none

// Production FC online-learning sideband. Real inference observations and
// external target traces feed the shared TP engine; every RMW transaction is
// forwarded to the native resident FC weight image used by inference.
module unified_fc_tp_native_online_learning_subsystem #(
    parameter bit ENABLE_W0_LEARNING = 1'b1,
    parameter bit USE_EXTERNAL_W2_LEARNER = 1'b0,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 480,
    parameter int MAX_FEATURES = 2312,
    parameter int MAX_TARGET_DIMENSION = 16,
    parameter int MAX_OUTPUTS = 10,
    parameter int MAX_EVENTS = 8192,
    parameter int TARGET_TRACE_WIDTH = 20,
    parameter int W2_MAX_GRADIENT_ELEMENTS = 32768,
    parameter int W0_PARAMETER_HIDDEN_STRIDE = 0,
    parameter int W2_PARAMETER_OUTPUT_STRIDE = 0,
    parameter int VMEM_WIDTH = 18,
    parameter int LOGIT_WIDTH = 24,
    parameter int LOGIT_SCALE_WIDTH = 24,
    parameter int SAMPLE_ID_WIDTH =
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
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
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

    input  logic                                  i_event_valid,
    output logic                                  o_event_ready,
    input  logic [FEATURE_INDEX_WIDTH-1:0]        i_event_feature_index,
    input  logic                                  i_snapshot_start,
    output logic                                  o_snapshot_start_ready,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]         i_snapshot_label,
    input  logic [MAX_HIDDEN-1:0]                 i_snapshot_spikes_flat,
    input  logic [MAX_HIDDEN*16-1:0]              i_snapshot_trace_flat,
    input  logic [MAX_HIDDEN*VMEM_WIDTH-1:0]      i_snapshot_vmem_flat,
    input  logic [MAX_OUTPUTS*LOGIT_WIDTH-1:0]     i_snapshot_logits_flat,
    output logic                                  o_snapshot_busy,
    output logic                                  o_snapshot_done,
    input  logic                                  i_batch_seal,
    output logic                                  o_batch_seal_ready,

    input  logic                                  i_target_hidden_trace_valid,
    output logic                                  o_target_hidden_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_target_hidden_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_target_hidden_trace_hidden,
    input  logic [16:0]                           i_target_hidden_trace_q016,
    input  logic                                  i_target_input_trace_valid,
    output logic                                  o_target_input_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_target_input_trace_sample,
    input  logic [TARGET_INDEX_WIDTH-1:0]         i_target_input_trace_index,
    input  logic [TARGET_TRACE_WIDTH-1:0]         i_target_input_trace_q016,
    input  logic                                  i_target_done,
    output logic                                  o_target_done_ready,

    output logic                                  o_window_commit_valid,
    input  logic                                  i_window_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_window_commit_profile_id,
    output logic [31:0]                           o_window_commit_sequence,
    output logic                                  o_window_busy,
    output logic                                  o_observation_error,
    output logic                                  o_context_error,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_parameter_error,
    output logic                                  o_protocol_error,

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

    // Normalized FC client boundary for the single shared FC/SCNN output
    // learner. This boundary is legal only for the W2-only deployment profile.
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
    output logic [SAMPLE_ID_WIDTH-1:0]            o_external_source_trace_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]         o_external_source_trace_index,
    output logic [15:0]                           o_external_source_trace_q016,
    input  logic                                  i_external_source_trace_error,
    output logic                                  o_external_logit_row_valid,
    input  logic                                  i_external_logit_row_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]            o_external_logit_sample_id,
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
    input  logic                                  i_external_saturated
);
    logic frontend_cfg_ready;
    logic frontend_release_ready;
    logic frontend_active;
    logic frontend_protocol_error;
    logic student_valid;
    logic student_ready;
    logic [SAMPLE_ID_WIDTH-1:0] student_sample;
    logic [HIDDEN_INDEX_WIDTH-1:0] student_hidden;
    logic [16:0] student_trace;
    logic surrogate_valid;
    logic surrogate_ready;
    logic [SAMPLE_ID_WIDTH-1:0] surrogate_sample;
    logic [HIDDEN_INDEX_WIDTH-1:0] surrogate_hidden;
    logic [16:0] surrogate_value;
    logic replay_event_valid;
    logic replay_event_ready;
    logic [SAMPLE_ID_WIDTH-1:0] replay_event_sample;
    logic [FEATURE_INDEX_WIDTH-1:0] replay_event_feature;
    logic logit_row_valid;
    logic logit_row_ready;
    logic [SAMPLE_ID_WIDTH-1:0] logit_sample;
    logic [OUTPUT_INDEX_WIDTH-1:0] logit_label;
    logic logit_valid;
    logic logit_ready;
    logic signed [15:0] logit_q6;
    logic replay_done;
    logic replay_done_ready;

    logic engine_cfg_ready;
    logic engine_cfg_error;
    logic engine_active;
    logic engine_release_ready;
    logic engine_observation_done_ready;
    logic engine_protocol_error;
    logic engine_read_valid;
    logic engine_read_ready;
    logic [PROFILE_ID_WIDTH-1:0] engine_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] engine_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] engine_read_parameter_addr;
    logic [TAG_WIDTH-1:0] engine_read_tag;
    logic engine_result_valid;
    logic engine_result_ready;
    logic [PROFILE_ID_WIDTH-1:0] engine_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] engine_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] engine_result_parameter_addr;
    logic [TAG_WIDTH-1:0] engine_result_tag;
    logic signed [CODE_WIDTH-1:0] engine_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] engine_result_residual;
    logic engine_write_code_enable;
    logic engine_write_residual_enable;
    logic engine_write_code_fire;
    logic engine_write_residual_fire;
    logic engine_read_response_ready;
    logic engine_logit_row_done;
    logic active_profile_valid_q;
    logic [PROFILE_ID_WIDTH-1:0] active_profile_q;
    logic response_outstanding_q;
    logic [TAG_WIDTH-1:0] expected_response_tag_q;
    logic binding_error_q;
    logic cfg_fire;
    logic release_fire;
    logic native_read_fire;
    logic native_response_fire;
    logic native_write_fire;
    logic external_commit_valid_q;
    logic [31:0] external_commit_sequence_q;

    assign o_cfg_ready = frontend_cfg_ready && engine_cfg_ready;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign o_release_ready = frontend_release_ready && engine_release_ready;
    assign release_fire = i_release && o_release_ready;
    assign o_cfg_error = engine_cfg_error;
    assign o_active = frontend_active && engine_active;

    assign replay_done_ready = i_target_done &&
        engine_observation_done_ready;
    assign o_target_done_ready = replay_done &&
        engine_observation_done_ready;

    assign o_native_read_valid = engine_read_valid;
    assign engine_read_ready = i_native_read_ready;
    assign o_native_read_tensor_id = engine_read_tensor_id;
    assign o_native_read_parameter_addr = engine_read_parameter_addr;
    assign o_native_read_tag = engine_read_tag;
    assign o_native_response_ready = engine_read_response_ready;
    assign native_read_fire = o_native_read_valid && i_native_read_ready;
    assign native_response_fire = i_native_response_valid &&
        o_native_response_ready;

    assign o_native_write_valid = engine_result_valid;
    assign engine_result_ready = i_native_write_ready;
    assign o_native_write_tensor_id = engine_result_tensor_id;
    assign o_native_write_parameter_addr = engine_result_parameter_addr;
    assign o_native_write_code = engine_result_code;
    assign o_native_write_residual = engine_result_residual;
    assign o_native_write_code_enable = engine_write_code_enable;
    assign o_native_write_residual_enable = engine_write_residual_enable;
    assign native_write_fire = o_native_write_valid && i_native_write_ready;
    assign o_protocol_error = frontend_protocol_error ||
        engine_protocol_error || binding_error_q;

    assign o_external_cfg_valid = USE_EXTERNAL_W2_LEARNER ? cfg_fire : 1'b0;
    assign o_external_cfg_profile_id = i_cfg_profile_id;
    assign o_external_cfg_tensor_id = i_cfg_w2_tensor_id;
    assign o_external_cfg_parameter_base = i_cfg_w2_parameter_base;
    assign o_external_cfg_batch_count = i_cfg_batch_count;
    assign o_external_cfg_source_count = i_cfg_hidden_count;
    assign o_external_cfg_output_count = i_cfg_output_count;
    assign o_external_cfg_vote_increment = i_cfg_w2_vote_increment;
    assign o_external_cfg_vote_threshold = i_cfg_w2_vote_threshold;
    assign o_external_cfg_code_min = i_cfg_w2_code_min;
    assign o_external_cfg_code_max = i_cfg_w2_code_max;
    assign o_external_release = USE_EXTERNAL_W2_LEARNER ? release_fire : 1'b0;
    assign o_external_source_trace_valid = USE_EXTERNAL_W2_LEARNER ?
        student_valid : 1'b0;
    assign o_external_source_trace_sample_id = student_sample;
    assign o_external_source_trace_index = student_hidden;
    assign o_external_source_trace_q016 = student_trace[16] ?
        16'hffff : student_trace[15:0];
    assign o_external_logit_row_valid = USE_EXTERNAL_W2_LEARNER ?
        logit_row_valid : 1'b0;
    assign o_external_logit_sample_id = logit_sample;
    assign o_external_logit_label = logit_label;
    assign o_external_logit_valid = USE_EXTERNAL_W2_LEARNER ?
        logit_valid : 1'b0;
    assign o_external_logit_q6 = logit_q6;
    assign o_external_observation_done = USE_EXTERNAL_W2_LEARNER ?
        (replay_done && i_target_done) : 1'b0;

    unified_fc_tp_student_observation_frontend #(
        .MAX_BATCH(MAX_BATCH), .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_OUTPUTS(MAX_OUTPUTS), .MAX_EVENTS(MAX_EVENTS),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .VMEM_WIDTH(VMEM_WIDTH), .LOGIT_WIDTH(LOGIT_WIDTH),
        .LOGIT_SCALE_WIDTH(LOGIT_SCALE_WIDTH),
        .SAMPLE_INDEX_WIDTH(SAMPLE_ID_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_student_frontend (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(frontend_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_output_count(i_cfg_output_count),
        .i_cfg_vth(i_cfg_vth),
        .i_cfg_logit_to_q6_scale(i_cfg_logit_to_q6_scale),
        .i_release(release_fire), .o_release_ready(frontend_release_ready),
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
        .o_student_trace_valid(student_valid),
        .i_student_trace_ready(student_ready),
        .o_student_trace_sample(student_sample),
        .o_student_trace_hidden(student_hidden),
        .o_student_trace_q016(student_trace),
        .o_surrogate_valid(surrogate_valid),
        .i_surrogate_ready(surrogate_ready),
        .o_surrogate_sample(surrogate_sample),
        .o_surrogate_hidden(surrogate_hidden),
        .o_surrogate_q016(surrogate_value),
        .o_replay_event_valid(replay_event_valid),
        .i_replay_event_ready(replay_event_ready),
        .o_replay_event_sample(replay_event_sample),
        .o_replay_event_feature_index(replay_event_feature),
        .o_logit_row_valid(logit_row_valid),
        .i_logit_row_ready(logit_row_ready),
        .o_logit_sample(logit_sample), .o_logit_label(logit_label),
        .o_logit_valid(logit_valid), .i_logit_ready(logit_ready),
        .o_logit_q6(logit_q6), .o_replay_done(replay_done),
        .i_replay_done_ready(replay_done_ready), .o_active(frontend_active),
        .o_protocol_error(frontend_protocol_error)
    );

    generate
    if (!USE_EXTERNAL_W2_LEARNER) begin : gen_private_learning_engine
    unified_fc_tp_online_learning_engine #(
        .ENABLE_W0_LEARNING(ENABLE_W0_LEARNING),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN), .MAX_FEATURES(MAX_FEATURES),
        .MAX_TARGET_DIMENSION(MAX_TARGET_DIMENSION),
        .MAX_OUTPUTS(MAX_OUTPUTS),
        .TARGET_TRACE_WIDTH(TARGET_TRACE_WIDTH),
        .W2_MAX_GRADIENT_ELEMENTS(W2_MAX_GRADIENT_ELEMENTS),
        .W0_PARAMETER_HIDDEN_STRIDE(W0_PARAMETER_HIDDEN_STRIDE),
        .W2_PARAMETER_OUTPUT_STRIDE(W2_PARAMETER_OUTPUT_STRIDE),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .TARGET_INDEX_WIDTH(TARGET_INDEX_WIDTH),
        .TARGET_COUNT_WIDTH(TARGET_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_engine (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(engine_cfg_ready),
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
        .o_cfg_error(engine_cfg_error), .o_active(engine_active),
        .i_release(release_fire), .o_release_ready(engine_release_ready),
        .i_student_trace_valid(student_valid),
        .o_student_trace_ready(student_ready),
        .i_student_trace_sample(student_sample),
        .i_student_trace_hidden(student_hidden),
        .i_student_trace_q016(student_trace),
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
        .i_surrogate_valid(surrogate_valid),
        .o_surrogate_ready(surrogate_ready),
        .i_surrogate_sample(surrogate_sample),
        .i_surrogate_hidden(surrogate_hidden),
        .i_surrogate_q016(surrogate_value),
        .i_event_valid(replay_event_valid), .o_event_ready(replay_event_ready),
        .i_event_sample_id(replay_event_sample),
        .i_event_feature_index(replay_event_feature),
        .i_logit_row_valid(logit_row_valid),
        .o_logit_row_ready(logit_row_ready),
        .i_logit_sample_id(logit_sample), .i_logit_label(logit_label),
        .i_logit_valid(logit_valid), .o_logit_ready(logit_ready),
        .i_logit_q6(logit_q6), .o_logit_row_done(engine_logit_row_done),
        .i_observation_done(replay_done && i_target_done),
        .o_observation_done_ready(engine_observation_done_ready),
        .o_window_commit_valid(o_window_commit_valid),
        .i_window_commit_ready(i_window_commit_ready),
        .o_window_commit_profile_id(o_window_commit_profile_id),
        .o_window_commit_sequence(o_window_commit_sequence),
        .o_window_busy(o_window_busy),
        .o_observation_error(o_observation_error),
        .o_context_error(o_context_error),
        .o_arithmetic_overflow(o_arithmetic_overflow),
        .o_parameter_error(o_parameter_error),
        .o_protocol_error(engine_protocol_error),
        .o_read_valid(engine_read_valid), .i_read_ready(engine_read_ready),
        .o_read_profile_id(engine_read_profile_id),
        .o_read_tensor_id(engine_read_tensor_id),
        .o_read_parameter_addr(engine_read_parameter_addr),
        .o_read_tag(engine_read_tag),
        .i_read_response_valid(i_native_response_valid),
        .o_read_response_ready(engine_read_response_ready),
        .i_read_response_code(i_native_response_code),
        .i_read_response_residual(i_native_response_residual),
        .i_read_response_error(i_native_response_error),
        .o_result_valid(engine_result_valid),
        .i_result_ready(engine_result_ready),
        .o_result_profile_id(engine_result_profile_id),
        .o_result_tensor_id(engine_result_tensor_id),
        .o_result_parameter_addr(engine_result_parameter_addr),
        .o_result_tag(engine_result_tag),
        .o_result_code(engine_result_code),
        .o_result_residual(engine_result_residual),
        .o_write_code_enable(engine_write_code_enable),
        .o_write_residual_enable(engine_write_residual_enable),
        .o_write_code_fire(engine_write_code_fire),
        .o_write_residual_fire(engine_write_residual_fire),
        .o_dirty(o_dirty), .o_saturated(o_saturated)
    );
    end else begin : gen_external_w2_learner
        assign engine_cfg_ready = i_external_cfg_ready;
        assign engine_cfg_error = i_external_cfg_error || ENABLE_W0_LEARNING;
        assign engine_active = i_external_active;
        assign engine_release_ready = i_external_release_ready &&
            !external_commit_valid_q;
        assign engine_observation_done_ready =
            i_external_observation_done_ready;
        assign engine_protocol_error = 1'b0;
        assign engine_logit_row_done = i_external_logit_row_done;

        assign student_ready = i_external_source_trace_ready;
        assign surrogate_ready = 1'b1;
        assign replay_event_ready = 1'b1;
        assign logit_row_ready = i_external_logit_row_ready;
        assign logit_ready = i_external_logit_ready;

        assign o_target_hidden_trace_ready = engine_active;
        assign o_target_input_trace_ready = engine_active;
        assign o_window_commit_valid = external_commit_valid_q;
        assign o_window_commit_profile_id = active_profile_q;
        assign o_window_commit_sequence = external_commit_sequence_q;
        assign o_window_busy = i_external_window_busy || external_commit_valid_q;
        assign o_observation_error = i_external_source_trace_error ||
            i_external_logit_error;
        assign o_context_error = i_external_context_error;
        assign o_arithmetic_overflow = i_external_gradient_overflow;
        assign o_parameter_error = i_external_update_config_error ||
            i_external_request_error || i_external_store_error ||
            i_external_internal_error;

        assign engine_read_valid = 1'b0;
        assign engine_read_profile_id = '0;
        assign engine_read_tensor_id = '0;
        assign engine_read_parameter_addr = '0;
        assign engine_read_tag = '0;
        assign engine_read_response_ready = 1'b0;
        assign engine_result_valid = 1'b0;
        assign engine_result_profile_id = '0;
        assign engine_result_tensor_id = '0;
        assign engine_result_parameter_addr = '0;
        assign engine_result_tag = '0;
        assign engine_result_code = '0;
        assign engine_result_residual = '0;
        assign engine_write_code_enable = 1'b0;
        assign engine_write_residual_enable = 1'b0;
        assign engine_write_code_fire = 1'b0;
        assign engine_write_residual_fire = 1'b0;
        assign o_dirty = i_external_dirty;
        assign o_saturated = i_external_saturated;

        always_ff @(posedge i_clk or negedge i_rst_n) begin
            if (!i_rst_n) begin
                external_commit_valid_q <= 1'b0;
                external_commit_sequence_q <= '0;
            end else begin
                if (cfg_fire) begin
                    external_commit_valid_q <= 1'b0;
                    external_commit_sequence_q <= '0;
                end
                if (i_external_window_done)
                    external_commit_valid_q <= 1'b1;
                if (external_commit_valid_q && i_window_commit_ready) begin
                    external_commit_valid_q <= 1'b0;
                    external_commit_sequence_q <=
                        external_commit_sequence_q + 1'b1;
                end
                if (release_fire)
                    external_commit_valid_q <= 1'b0;
            end
        end
    end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_profile_valid_q <= 1'b0;
            active_profile_q <= '0;
            response_outstanding_q <= 1'b0;
            expected_response_tag_q <= '0;
            binding_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                binding_error_q <= 1'b0;
            if (cfg_fire) begin
                active_profile_valid_q <= 1'b1;
                active_profile_q <= i_cfg_profile_id;
            end
            if (release_fire) begin
                active_profile_valid_q <= 1'b0;
                active_profile_q <= '0;
            end
            case ({native_read_fire, native_response_fire})
                2'b10: response_outstanding_q <= 1'b1;
                2'b01: response_outstanding_q <= 1'b0;
                2'b11: response_outstanding_q <= 1'b1;
                default: response_outstanding_q <= response_outstanding_q;
            endcase
            if (native_read_fire) begin
                expected_response_tag_q <= engine_read_tag;
                if (!active_profile_valid_q ||
                    engine_read_profile_id != active_profile_q ||
                    (response_outstanding_q && !native_response_fire))
                    binding_error_q <= 1'b1;
            end
            if (native_response_fire) begin
                if (!response_outstanding_q ||
                    i_native_response_tag != expected_response_tag_q)
                    binding_error_q <= 1'b1;
            end
            if (native_write_fire &&
                (!active_profile_valid_q ||
                 engine_result_profile_id != active_profile_q ||
                 engine_result_tag != expected_response_tag_q))
                binding_error_q <= 1'b1;
            if (engine_write_code_fire !=
                (native_write_fire && engine_write_code_enable))
                binding_error_q <= 1'b1;
            if (engine_write_residual_fire !=
                (native_write_fire && engine_write_residual_enable))
                binding_error_q <= 1'b1;
            if (engine_logit_row_done && !active_profile_valid_q)
                binding_error_q <= 1'b1;
            if (release_fire && response_outstanding_q)
                binding_error_q <= 1'b1;
        end
    end

endmodule

`default_nettype wire
