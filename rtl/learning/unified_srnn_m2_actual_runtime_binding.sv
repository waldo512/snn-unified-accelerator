`timescale 1ns/1ps
`default_nettype none

// Production SRNN client-side composition. The common runtime owns profile,
// run and physical-core arbitration; this block captures real recurrent-core
// commits, replays them timestep-major, supplies class target trajectories and
// retires each runtime timestep only after the M2 forward/learn transaction.
module unified_srnn_m2_actual_runtime_binding #(
    parameter bit USE_INTERNAL_STORE = 1'b1,
    parameter int STORE_DEPTH = 120320,
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int CLASS_MAX = 20,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int TIMESTEP_ID_WIDTH = $clog2(TIMESTEP_MAX),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(TIMESTEP_MAX + 1),
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int CLASS_INDEX_WIDTH = 5,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int WINDOW_SEQUENCE_WIDTH = 32,
    parameter int TARGET_WORD_WIDTH = 32,
    parameter int TARGET_WORD_INDEX_WIDTH =
        (((SOURCE_VECTOR_MAX+TARGET_WORD_WIDTH-1)/TARGET_WORD_WIDTH) <= 1) ?
            1 : $clog2((SOURCE_VECTOR_MAX+TARGET_WORD_WIDTH-1)/
                TARGET_WORD_WIDTH)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wh1_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wh1_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wh1_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wh1_threshold,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_threshold,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_max,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [TIMESTEP_COUNT_WIDTH-1:0]           i_cfg_timestep_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_cfg_class_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    input  logic [VERSION_WIDTH-1:0]                  i_cfg_commit_version,
    input  logic                                      i_cfg_check_commit_version,
    output logic                                      o_configured,
    output logic                                      o_cfg_error,

    input  logic                                      i_parameter_load_valid,
    output logic                                      o_parameter_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_parameter_load_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_parameter_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_parameter_load_residual,

    input  logic                                      i_target_label_load_start,
    output logic                                      o_target_label_load_start_ready,
    input  logic                                      i_target_label_valid,
    output logic                                      o_target_label_ready,
    input  logic [$clog2(BATCH_MAX)-1:0]              i_target_label_sample_id,
    input  logic [$clog2(CLASS_MAX)-1:0]              i_target_label_class_id,
    input  logic                                      i_target_label_last,
    output logic                                      o_target_label_load_done,
    input  logic                                      i_target_spike_word_write_valid,
    output logic                                      o_target_spike_word_write_ready,
    input  logic [$clog2(CLASS_MAX)-1:0]              i_target_spike_word_class_id,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_target_spike_word_timestep_id,
    input  logic [TARGET_WORD_INDEX_WIDTH-1:0]        i_target_spike_word_id,
    input  logic [TARGET_WORD_WIDTH-1:0]              i_target_spike_word_data,
    input  logic                                      i_target_trace_write_valid,
    output logic                                      o_target_trace_write_ready,
    input  logic [$clog2(CLASS_MAX)-1:0]              i_target_trace_class_id,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_target_trace_timestep_id,
    input  logic [$clog2(BACKBONE_MAX)-1:0]           i_target_trace_backbone_id,
    input  logic signed [31:0]                        i_target_trace_q14,
    output logic                                      o_target_labels_loaded,

    input  logic                                      i_runtime_client_active,
    input  logic                                      i_runtime_timestep_start,
    output logic                                      o_runtime_timestep_start_ready,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_runtime_timestep_id,
    input  logic [VERSION_WIDTH-1:0]                  i_runtime_sequence_version,
    input  logic                                      i_runtime_learning_enable,
    input  logic                                      i_runtime_abort_timestep,
    output logic                                      o_runtime_abort_ready,
    output logic                                      o_runtime_timestep_done,
    output logic                                      o_runtime_timestep_aborted,
    output logic                                      o_runtime_busy,
    output logic                                      o_runtime_error,

    input  logic                                      i_backbone_sample_start_valid,
    output logic                                      o_backbone_sample_start_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_backbone_sample_id,
    output logic                                      o_core_start_valid,
    input  logic                                      i_core_start_ready,
    input  logic                                      i_core_done,
    input  logic                                      i_core_aborted,
    output logic                                      o_core_abort,
    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_commit_version,
    input  logic                                      i_commit_sample_last,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_commit_spikes,

    output logic                                      o_forward_code_valid,
    input  logic                                      i_forward_code_ready,
    output logic [TENSOR_ID_WIDTH-1:0]                o_forward_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_parameter_addr,
    output logic signed [CODE_WIDTH-1:0]              o_forward_code,
    output logic                                      o_forward_last,
    output logic                                      o_forward_parameter_scan_done,
    output logic                                      o_hidden_observation_valid,
    input  logic                                      i_hidden_observation_ready,
    output logic                                      o_hidden_observation_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_hidden_observation_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_hidden_observation_hidden_id,
    output logic signed [31:0]                        o_hidden_membrane_pre_q14,
    output logic signed [31:0]                        o_hidden_membrane_next_q14,
    output logic                                      o_hidden_spike,
    output logic signed [31:0]                        o_hidden_trace_q14,
    output logic signed [31:0]                        o_hidden_output_feature_q14,
    output logic                                      o_hidden_observation_last,
    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]              o_logit_class_id,
    output logic signed [63:0]                        o_logit_q28,
    output logic                                      o_logit_last,
    output logic                                      o_learning_commit_pulse,
    output logic                                      o_detach_pulse,
    output logic [VERSION_WIDTH-1:0]                  o_model_version,
    output logic [VERSION_WIDTH-1:0]                  o_adapt_version,
    output logic [VERSION_WIDTH-1:0]                  o_forward_model_version,
    output logic [VERSION_WIDTH-1:0]                  o_forward_adapt_version,
    output logic [PARAM_ADDR_WIDTH:0]                 o_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0]                 o_update_count,
    output logic                                      o_quiescent,

    output logic                                      o_store_load_valid,
    input  logic                                      i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0]              o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_store_load_residual,
    output logic                                      o_store_learning_read_valid,
    input  logic                                      i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_store_learning_read_tag,
    input  logic                                      i_store_learning_response_valid,
    output logic                                      o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0]              i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_store_learning_response_residual,
    input  logic                                      i_store_learning_response_error,
    output logic                                      o_store_learning_write_valid,
    input  logic                                      i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0]              o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_store_learning_write_residual,
    output logic                                      o_store_learning_write_code_enable,
    output logic                                      o_store_learning_write_residual_enable,
    output logic                                      o_store_forward_read_valid,
    input  logic                                      i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_store_forward_read_tag,
    input  logic                                      i_store_forward_response_valid,
    output logic                                      o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]              i_store_forward_response_code,
    input  logic                                      i_store_forward_response_error,
    input  logic                                      i_store_protocol_error
);
    logic bridge_cfg_ready;
    logic bridge_configured;
    logic bridge_error;
    logic bridge_capture_busy;
    logic bridge_quiescent;
    logic bridge_coordinator_start;
    logic target_cfg_ready;
    logic target_cfg_error;
    logic target_configured;
    logic target_busy;
    logic target_protocol_error;
    logic coordinator_cfg_ready;
    logic coordinator_cfg_error;
    logic coordinator_busy;
    logic coordinator_error;
    logic configured_q;
    logic error_q;
    logic cfg_fire;
    logic abort_transaction;

    logic actual_replay_start;
    logic actual_replay_start_ready;
    logic [TIMESTEP_ID_WIDTH-1:0] actual_replay_timestep;
    logic actual_replay_valid;
    logic actual_replay_ready;
    logic [$clog2(BATCH_MAX)-1:0] actual_replay_sample;
    logic [SOURCE_VECTOR_MAX-1:0] actual_replay_spikes;
    logic actual_replay_done;

    logic target_vector_start;
    logic target_vector_start_ready;
    logic [TIMESTEP_ID_WIDTH-1:0] target_vector_timestep;
    logic target_vector_valid;
    logic target_vector_ready;
    logic [$clog2(BATCH_MAX)-1:0] target_vector_sample;
    logic [SOURCE_VECTOR_MAX-1:0] target_vector_spikes;
    logic target_vector_done;
    logic target_trace_start;
    logic target_trace_start_ready;
    logic [TIMESTEP_ID_WIDTH-1:0] target_trace_timestep;
    logic target_trace_valid;
    logic target_trace_ready;
    logic [$clog2(BATCH_MAX)-1:0] target_trace_sample;
    logic [$clog2(BACKBONE_MAX)-1:0] target_trace_backbone;
    logic signed [31:0] target_trace_data;
    logic target_trace_last;
    logic target_trace_done;
    logic label_replay_start;
    logic label_replay_start_ready;
    logic label_replay_valid;
    logic label_replay_ready;
    logic [$clog2(BATCH_MAX)-1:0] label_replay_sample;
    logic [$clog2(CLASS_MAX)-1:0] label_replay_class;
    logic label_replay_last;
    logic label_replay_done;

    logic unused_target_vector_last;
    logic unused_target_label_replay_done;
    logic unused_restore_ready;
    logic unused_recovery_required;
    logic [VERSION_WIDTH-1:0] unused_learning_base_model_version;
    logic [VERSION_WIDTH-1:0] unused_learning_base_adapt_version;
    logic [WINDOW_SEQUENCE_WIDTH-1:0] unused_commit_window_sequence;
    logic unused_commit_code_dirty;
    logic unused_commit_optimizer_dirty;
    logic unused_learning_result_dirty;
    logic unused_learning_saturated;
    logic coordinator_timestep_start_ready;

    assign o_cfg_ready = bridge_cfg_ready && target_cfg_ready &&
        coordinator_cfg_ready && !i_runtime_client_active && !error_q;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign abort_transaction = i_abort || i_runtime_abort_timestep;
    assign o_configured = configured_q && bridge_configured &&
        target_configured && o_target_labels_loaded && !o_cfg_error;
    assign o_cfg_error = error_q || bridge_error || target_cfg_error ||
        target_protocol_error || coordinator_cfg_error;
    assign o_runtime_busy = bridge_capture_busy || coordinator_busy ||
        target_busy;
    assign o_runtime_error = o_cfg_error || coordinator_error;
    assign o_quiescent = bridge_quiescent && !coordinator_busy &&
        !target_busy && !i_runtime_client_active && !o_runtime_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            error_q <= 1'b0;
        end else begin
            if (cfg_fire)
                configured_q <= 1'b1;
            if (i_clear_error && !i_runtime_client_active &&
                !bridge_capture_busy && !coordinator_busy && !target_busy) begin
                error_q <= 1'b0;
            end else if (bridge_error || target_cfg_error ||
                         target_protocol_error || coordinator_cfg_error ||
                         coordinator_error) begin
                error_q <= 1'b1;
            end
            if (bridge_error || target_cfg_error || coordinator_cfg_error)
                configured_q <= 1'b0;
        end
    end

    unified_srnn_m2_live_runtime_capture_bridge #(
        .BATCH_MAX(BATCH_MAX), .TIMESTEP_MAX(TIMESTEP_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_actual_replay (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(abort_transaction),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(bridge_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count[$clog2(BATCH_MAX+1)-1:0]),
        .i_cfg_timestep_count(i_cfg_timestep_count),
        .i_cfg_backbone_count(
            i_cfg_backbone_count[$clog2(BACKBONE_MAX+1)-1:0]),
        .i_cfg_expected_version(i_cfg_commit_version),
        .i_cfg_check_version(i_cfg_check_commit_version),
        .o_runtime_configured(bridge_configured),
        .i_runtime_client_active(i_runtime_client_active),
        .i_runtime_timestep_start(i_runtime_timestep_start),
        .o_runtime_timestep_start_ready(o_runtime_timestep_start_ready),
        .o_coordinator_timestep_start(bridge_coordinator_start),
        .i_coordinator_timestep_start_ready(
            coordinator_timestep_start_ready),
        .i_backbone_sample_start_valid(i_backbone_sample_start_valid),
        .o_backbone_sample_start_ready(o_backbone_sample_start_ready),
        .o_backbone_sample_id(o_backbone_sample_id),
        .o_core_start_valid(o_core_start_valid),
        .i_core_start_ready(i_core_start_ready),
        .i_core_done(i_core_done), .i_core_aborted(i_core_aborted),
        .o_core_abort(o_core_abort),
        .i_commit_valid(i_commit_valid), .o_commit_ready(o_commit_ready),
        .i_commit_version(i_commit_version),
        .i_commit_sample_last(i_commit_sample_last),
        .i_commit_spikes(i_commit_spikes),
        .i_replay_start(actual_replay_start),
        .o_replay_start_ready(actual_replay_start_ready),
        .i_replay_timestep_id(actual_replay_timestep),
        .o_replay_valid(actual_replay_valid),
        .i_replay_ready(actual_replay_ready),
        .o_replay_sample_id(actual_replay_sample),
        .o_replay_spikes(actual_replay_spikes),
        .o_replay_done(actual_replay_done),
        .o_started_sample_count(), .o_completed_sample_count(),
        .o_capture_complete(), .o_capture_busy(bridge_capture_busy),
        .o_quiescent(bridge_quiescent), .o_error(bridge_error)
    );

    unified_srnn_m2_class_target_trajectory_provider #(
        .BATCH_MAX(BATCH_MAX), .CLASS_MAX(CLASS_MAX),
        .TIMESTEP_MAX(TIMESTEP_MAX), .BACKBONE_MAX(BACKBONE_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .WORD_WIDTH(TARGET_WORD_WIDTH),
        .WORD_INDEX_WIDTH(TARGET_WORD_INDEX_WIDTH)
    ) u_target_provider (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(target_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count[$clog2(BATCH_MAX+1)-1:0]),
        .i_cfg_class_count(i_cfg_class_count[$clog2(CLASS_MAX+1)-1:0]),
        .i_cfg_timestep_count(i_cfg_timestep_count),
        .i_cfg_backbone_count(
            i_cfg_backbone_count[$clog2(BACKBONE_MAX+1)-1:0]),
        .o_cfg_error(target_cfg_error),
        .i_label_load_start(i_target_label_load_start),
        .o_label_load_start_ready(o_target_label_load_start_ready),
        .i_label_valid(i_target_label_valid),
        .o_label_ready(o_target_label_ready),
        .i_label_sample_id(i_target_label_sample_id),
        .i_label_class_id(i_target_label_class_id),
        .i_label_last(i_target_label_last),
        .o_label_load_done(o_target_label_load_done),
        .i_label_replay_start(label_replay_start),
        .o_label_replay_start_ready(label_replay_start_ready),
        .o_label_replay_valid(label_replay_valid),
        .i_label_replay_ready(label_replay_ready),
        .o_label_replay_sample_id(label_replay_sample),
        .o_label_replay_class_id(label_replay_class),
        .o_label_replay_last(label_replay_last),
        .o_label_replay_done(label_replay_done),
        .i_spike_word_write_valid(i_target_spike_word_write_valid),
        .o_spike_word_write_ready(o_target_spike_word_write_ready),
        .i_spike_word_class_id(i_target_spike_word_class_id),
        .i_spike_word_timestep_id(i_target_spike_word_timestep_id),
        .i_spike_word_id(i_target_spike_word_id),
        .i_spike_word_data(i_target_spike_word_data),
        .i_trace_write_valid(i_target_trace_write_valid),
        .o_trace_write_ready(o_target_trace_write_ready),
        .i_trace_class_id(i_target_trace_class_id),
        .i_trace_timestep_id(i_target_trace_timestep_id),
        .i_trace_backbone_id(i_target_trace_backbone_id),
        .i_trace_q14(i_target_trace_q14),
        .i_vector_replay_start(target_vector_start),
        .o_vector_replay_start_ready(target_vector_start_ready),
        .i_vector_replay_timestep_id(target_vector_timestep),
        .o_vector_valid(target_vector_valid),
        .i_vector_ready(target_vector_ready),
        .o_vector_sample_id(target_vector_sample),
        .o_vector_spikes(target_vector_spikes),
        .o_vector_last(unused_target_vector_last),
        .o_vector_done(target_vector_done),
        .i_trace_replay_start(target_trace_start),
        .o_trace_replay_start_ready(target_trace_start_ready),
        .i_trace_replay_timestep_id(target_trace_timestep),
        .o_trace_valid(target_trace_valid),
        .i_trace_ready(target_trace_ready),
        .o_trace_sample_id(target_trace_sample),
        .o_trace_backbone_id(target_trace_backbone),
        .o_trace_q14(target_trace_data),
        .o_trace_last(target_trace_last),
        .o_trace_done(target_trace_done),
        .i_abort(abort_transaction), .o_busy(target_busy),
        .o_configured(target_configured),
        .o_labels_loaded(o_target_labels_loaded),
        .o_protocol_error(target_protocol_error)
    );

    unified_srnn_m2_timestep_train_infer_coordinator #(
        .USE_INTERNAL_STORE(USE_INTERNAL_STORE), .STORE_DEPTH(STORE_DEPTH),
        .BATCH_MAX(BATCH_MAX), .TIMESTEP_MAX(TIMESTEP_MAX),
        .HIDDEN_MAX(HIDDEN_MAX), .BACKBONE_MAX(BACKBONE_MAX),
        .CLASS_MAX(CLASS_MAX), .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .TIMESTEP_ID_WIDTH(TIMESTEP_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .WINDOW_SEQUENCE_WIDTH(WINDOW_SEQUENCE_WIDTH)
    ) u_coordinator (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(coordinator_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_wh1_tensor_id(i_cfg_wh1_tensor_id),
        .i_cfg_wh1_base(i_cfg_wh1_base),
        .i_cfg_wh1_increment(i_cfg_wh1_increment),
        .i_cfg_wh1_threshold(i_cfg_wh1_threshold),
        .i_cfg_wout_tensor_id(i_cfg_wout_tensor_id),
        .i_cfg_wout_base(i_cfg_wout_base),
        .i_cfg_wout_increment(i_cfg_wout_increment),
        .i_cfg_wout_threshold(i_cfg_wout_threshold),
        .i_cfg_code_min(i_cfg_code_min), .i_cfg_code_max(i_cfg_code_max),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_class_count(i_cfg_class_count),
        .i_cfg_wout_element_count(i_cfg_wout_element_count),
        .o_cfg_error(coordinator_cfg_error),
        .i_load_valid(i_parameter_load_valid),
        .o_load_ready(o_parameter_load_ready),
        .i_load_addr(i_parameter_load_addr),
        .i_load_code(i_parameter_load_code),
        .i_load_residual(i_parameter_load_residual),
        .i_timestep_start(bridge_coordinator_start),
        .o_timestep_start_ready(coordinator_timestep_start_ready),
        .i_timestep_id(i_runtime_timestep_id),
        .i_sequence_version(i_runtime_sequence_version),
        .i_learning_enable(i_runtime_learning_enable),
        .i_abort_timestep(abort_transaction),
        .o_abort_ready(o_runtime_abort_ready),
        .o_actual_replay_start(actual_replay_start),
        .i_actual_replay_start_ready(actual_replay_start_ready),
        .o_actual_replay_timestep_id(actual_replay_timestep),
        .i_actual_replay_valid(actual_replay_valid),
        .o_actual_replay_ready(actual_replay_ready),
        .i_actual_replay_sample_id(SAMPLE_ID_WIDTH'(actual_replay_sample)),
        .i_actual_replay_spikes(actual_replay_spikes),
        .i_actual_replay_done(actual_replay_done),
        .i_actual_replay_error(bridge_error),
        .o_target_vector_start(target_vector_start),
        .i_target_vector_start_ready(target_vector_start_ready),
        .o_target_vector_timestep_id(target_vector_timestep),
        .i_target_vector_valid(target_vector_valid),
        .o_target_vector_ready(target_vector_ready),
        .i_target_vector_sample_id(SAMPLE_ID_WIDTH'(target_vector_sample)),
        .i_target_vector_spikes(target_vector_spikes),
        .i_target_vector_done(target_vector_done),
        .i_target_vector_error(target_protocol_error),
        .i_external_backbone_valid(1'b0),
        .o_external_backbone_ready(),
        .i_external_backbone_path(1'b0),
        .i_external_backbone_sample_id('0),
        .i_external_backbone_id('0),
        .i_external_backbone_q14('0),
        .i_external_backbone_last(1'b0),
        .i_external_backbone_done(1'b0),
        .i_external_backbone_error(1'b0),
        .o_target_trace_start(target_trace_start),
        .i_target_trace_start_ready(target_trace_start_ready),
        .o_target_trace_timestep_id(target_trace_timestep),
        .i_target_trace_valid(target_trace_valid),
        .o_target_trace_ready(target_trace_ready),
        .i_target_trace_sample_id(SAMPLE_ID_WIDTH'(target_trace_sample)),
        .i_target_trace_backbone_id(
            BACKBONE_INDEX_WIDTH'(target_trace_backbone)),
        .i_target_trace_q14(target_trace_data),
        .i_target_trace_last(target_trace_last),
        .i_target_trace_done(target_trace_done),
        .i_target_trace_error(target_protocol_error),
        .o_target_trace_valid(), .i_target_trace_ready(1'b1),
        .o_target_trace_sample_id(), .o_target_trace_backbone_id(),
        .o_target_trace_q14(), .o_target_trace_last(),
        .o_target_trace_done(),
        .o_forward_code_valid(o_forward_code_valid),
        .i_forward_code_ready(i_forward_code_ready),
        .o_forward_tensor_id(o_forward_tensor_id),
        .o_forward_local_addr(o_forward_local_addr),
        .o_forward_parameter_addr(o_forward_parameter_addr),
        .o_forward_code(o_forward_code), .o_forward_last(o_forward_last),
        .o_forward_parameter_scan_done(o_forward_parameter_scan_done),
        .o_hidden_observation_valid(o_hidden_observation_valid),
        .i_hidden_observation_ready(i_hidden_observation_ready),
        .o_hidden_observation_path(o_hidden_observation_path),
        .o_hidden_observation_sample_id(o_hidden_observation_sample_id),
        .o_hidden_observation_hidden_id(o_hidden_observation_hidden_id),
        .o_hidden_membrane_pre_q14(o_hidden_membrane_pre_q14),
        .o_hidden_membrane_next_q14(o_hidden_membrane_next_q14),
        .o_hidden_spike(o_hidden_spike),
        .o_hidden_trace_q14(o_hidden_trace_q14),
        .o_hidden_output_feature_q14(o_hidden_output_feature_q14),
        .o_hidden_observation_last(o_hidden_observation_last),
        .o_logit_valid(o_logit_valid), .i_logit_ready(i_logit_ready),
        .o_logit_sample_id(o_logit_sample_id),
        .o_logit_class_id(o_logit_class_id), .o_logit_q28(o_logit_q28),
        .o_logit_last(o_logit_last),
        .o_label_replay_start(label_replay_start),
        .i_label_replay_start_ready(label_replay_start_ready),
        .i_label_valid(label_replay_valid),
        .o_label_ready(label_replay_ready),
        .i_label_sample_id(SAMPLE_ID_WIDTH'(label_replay_sample)),
        .i_label_class_id(CLASS_INDEX_WIDTH'(label_replay_class)),
        .i_label_last(label_replay_last),
        .i_restore_complete(1'b0),
        .o_restore_ready(unused_restore_ready),
        .o_recovery_required(unused_recovery_required),
        .o_learning_commit_pulse(o_learning_commit_pulse),
        .o_detach_pulse(o_detach_pulse),
        .o_model_version(o_model_version), .o_adapt_version(o_adapt_version),
        .o_forward_model_version(o_forward_model_version),
        .o_forward_adapt_version(o_forward_adapt_version),
        .o_learning_base_model_version(
            unused_learning_base_model_version),
        .o_learning_base_adapt_version(
            unused_learning_base_adapt_version),
        .o_learning_commit_window_sequence(unused_commit_window_sequence),
        .o_commit_code_dirty(unused_commit_code_dirty),
        .o_commit_optimizer_dirty(unused_commit_optimizer_dirty),
        .o_learning_result_dirty(unused_learning_result_dirty),
        .o_learning_saturated(unused_learning_saturated),
        .o_timestep_done(o_runtime_timestep_done),
        .o_timestep_aborted(o_runtime_timestep_aborted),
        .o_busy(coordinator_busy), .o_error(coordinator_error),
        .o_zero_gradient_count(o_zero_gradient_count),
        .o_update_count(o_update_count),
        .o_store_load_valid(o_store_load_valid),
        .i_store_load_ready(i_store_load_ready),
        .o_store_load_addr(o_store_load_addr),
        .o_store_load_code(o_store_load_code),
        .o_store_load_residual(o_store_load_residual),
        .o_store_learning_read_valid(o_store_learning_read_valid),
        .i_store_learning_read_ready(i_store_learning_read_ready),
        .o_store_learning_read_addr(o_store_learning_read_addr),
        .o_store_learning_read_tag(o_store_learning_read_tag),
        .i_store_learning_response_valid(i_store_learning_response_valid),
        .o_store_learning_response_ready(o_store_learning_response_ready),
        .i_store_learning_response_tag(i_store_learning_response_tag),
        .i_store_learning_response_code(i_store_learning_response_code),
        .i_store_learning_response_residual(
            i_store_learning_response_residual),
        .i_store_learning_response_error(i_store_learning_response_error),
        .o_store_learning_write_valid(o_store_learning_write_valid),
        .i_store_learning_write_ready(i_store_learning_write_ready),
        .o_store_learning_write_addr(o_store_learning_write_addr),
        .o_store_learning_write_code(o_store_learning_write_code),
        .o_store_learning_write_residual(o_store_learning_write_residual),
        .o_store_learning_write_code_enable(
            o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_store_forward_read_valid),
        .i_store_forward_read_ready(i_store_forward_read_ready),
        .o_store_forward_read_addr(o_store_forward_read_addr),
        .o_store_forward_read_tag(o_store_forward_read_tag),
        .i_store_forward_response_valid(i_store_forward_response_valid),
        .o_store_forward_response_ready(o_store_forward_response_ready),
        .i_store_forward_response_tag(i_store_forward_response_tag),
        .i_store_forward_response_code(i_store_forward_response_code),
        .i_store_forward_response_error(i_store_forward_response_error),
        .i_store_protocol_error(i_store_protocol_error)
    );

    assign unused_target_label_replay_done = label_replay_done;
endmodule

`default_nettype wire
