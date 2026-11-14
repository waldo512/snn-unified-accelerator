`timescale 1ns/1ps
`default_nettype none

// Production SCNN train/infer binding. One physical SCNN core serves normal
// student/classifier execution and the post-commit forward check. The selected
// Conv + classifier learner shares the external parameter backend and owns the
// model-version boundary; target and core datapaths remain explicit children.
module unified_scnn_tp_online_system_binding #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    // Physical Conv8 learning tile capacity. Full-layer coordinates and
    // observation depths are controlled independently by OBS_*_CHANNELS.
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int OBS_INPUT_CHANNELS = 512,
    parameter int OBS_OUTPUT_CHANNELS = 512,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int FRAME_WIDTH = 8,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH+1),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS+1),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int FEATURE_INDEX_WIDTH = MAX_FEATURES <= 2 ? 1 :
        $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES+1),
    parameter int CLASS_INDEX_WIDTH = MAX_CLASSES <= 2 ? 1 :
        $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES+1),
    parameter bit USE_SAMPLE_CONTEXT = 1'b1,
    parameter bit USE_CLASSIFIER_STUDENT_TRACE = 1'b1,
    parameter bit USE_INTERNAL_MODULATION = 1'b1,
    parameter bit USE_EXTERNAL_OUTPUT_LEARNER = 1'b0,
    parameter int INTERNAL_MODULATION_BATCH = 2,
    parameter bit INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR = 1'b0,
    parameter int SURROGATE_WAVE_CHANNELS = OBS_OUTPUT_CHANNELS,
    parameter int TRACE_WAVE_CHANNELS = OBS_OUTPUT_CHANNELS
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,

    input  logic                                i_profile_aligned,
    input  logic                                i_start_allowed,
    input  logic                                i_scnn_parameter_owner,
    input  logic                                i_scnn_learning_enable,
    input  logic [PROFILE_ID_WIDTH-1:0]         i_system_profile_id,
    input  logic                                i_run_start_valid,
    output logic                                o_run_start_ready,
    input  logic [BATCH_COUNT_WIDTH-1:0]        i_batch_count,
    input  logic [TIMESTEP_COUNT_WIDTH-1:0]     i_timestep_count,

    output logic                                o_target_start_valid,
    input  logic                                i_target_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_target_timestep,
    input  logic                                i_target_done,
    input  logic                                i_target_busy,
    input  logic                                i_target_error,

    output logic                                o_core_start_valid,
    input  logic                                i_core_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_core_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_core_timestep,
    output logic                                o_core_post_forward,
    output logic                                o_classifier_capture_enable,
    input  logic                                i_selected_conv_done,
    input  logic                                i_core_sample_done,
    input  logic                                i_core_quiescent,
    input  logic                                i_core_error,
    output logic                                o_core_abort,

    input  logic [PROFILE_ID_WIDTH-1:0]         i_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]          i_conv_tensor_id,
    input  logic [TENSOR_ID_WIDTH-1:0]          i_classifier_tensor_id,
    input  logic [CHANNEL_COUNT_WIDTH-1:0]      i_total_input_channels,
    input  logic [CHANNEL_COUNT_WIDTH-1:0]      i_total_output_channels,
    input  logic [PARAM_ADDR_WIDTH-1:0]         i_conv_parameter_base,
    input  logic [Q16_ADDR_WIDTH-1:0]           i_conv_weight_q16_base,
    input  logic [PARAM_ADDR_WIDTH-1:0]         i_classifier_parameter_base,
    input  logic [FEATURE_COUNT_WIDTH-1:0]      i_classifier_feature_count,
    input  logic [CLASS_COUNT_WIDTH-1:0]        i_classifier_class_count,
    input  logic                                i_classifier_trace_leak_valid,
    input  logic signed [15:0]                  i_classifier_trace_leak_q8_8,
    input  logic [VOTE_WIDTH-1:0]               i_conv_vote_increment,
    input  logic [VOTE_WIDTH-1:0]               i_conv_vote_threshold,
    input  logic [VOTE_WIDTH-1:0]               i_classifier_vote_increment,
    input  logic [VOTE_WIDTH-1:0]               i_classifier_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0]        i_code_min,
    input  logic signed [CODE_WIDTH-1:0]        i_code_max,
    input  logic [VERSION_WIDTH-1:0]            i_sample_version,

    input  logic                                i_context_load_valid,
    output logic                                o_context_load_ready,
    input  logic [VERSION_WIDTH-1:0]            i_context_model_version,
    input  logic [VERSION_WIDTH-1:0]            i_context_adapt_version,
    input  logic                                i_restore_complete,
    output logic                                o_restore_ready,
    output logic                                o_recovery_required,

    input  logic                                i_sample_context_valid,
    output logic                                o_sample_context_ready,
    input  logic                                i_sample_context_target,
    input  logic [SAMPLE_WIDTH-1:0]             i_sample_context_sample,
    input  logic [VERSION_WIDTH-1:0]            i_sample_context_version,

    input  logic                                i_input_valid,
    output logic                                o_input_ready,
    input  logic                                i_input_target,
    input  logic [SAMPLE_WIDTH-1:0]             i_input_sample,
    input  logic [TIMESTEP_WIDTH-1:0]           i_input_timestep,
    input  logic [VERSION_WIDTH-1:0]            i_input_sample_version,
    input  logic [CHANNEL_INDEX_WIDTH-1:0]      i_input_channel,
    input  logic [$clog2(SPATIAL*SPATIAL)-1:0] i_input_spatial,
    input  logic signed [15:0]                  i_input_q8_8,
    input  logic                                i_trace_valid,
    output logic                                o_trace_ready,
    input  logic                                i_trace_target,
    input  logic [SAMPLE_WIDTH-1:0]             i_trace_sample,
    input  logic [TIMESTEP_WIDTH-1:0]           i_trace_timestep,
    input  logic [VERSION_WIDTH-1:0]            i_trace_sample_version,
    input  logic [CHANNEL_INDEX_WIDTH-1:0]      i_trace_channel,
    input  logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0]
                                                i_trace_spatial,
    input  logic signed [15:0]                  i_trace_q8_8,
    input  logic                                i_surrogate_valid,
    output logic                                o_surrogate_ready,
    input  logic                                i_surrogate_target,
    input  logic [SAMPLE_WIDTH-1:0]             i_surrogate_sample,
    input  logic [TIMESTEP_WIDTH-1:0]           i_surrogate_timestep,
    input  logic [VERSION_WIDTH-1:0]            i_surrogate_sample_version,
    input  logic [CHANNEL_INDEX_WIDTH-1:0]      i_surrogate_channel,
    input  logic [$clog2(SPATIAL*SPATIAL)-1:0] i_surrogate_spatial,
    input  logic signed [15:0]                  i_surrogate_q8_8,

    input  logic                                i_modulation_valid,
    output logic                                o_modulation_ready,
    input  logic [SAMPLE_WIDTH-1:0]             i_modulation_row,
    input  logic [SAMPLE_WIDTH-1:0]             i_modulation_col,
    input  logic signed [15:0]                  i_modulation_q8_8,

    input  logic                                i_feature_trace_wave_valid,
    output logic                                o_feature_trace_wave_ready,
    input  logic [LANES-1:0]                    i_feature_trace_wave_mask,
    input  logic [FEATURE_INDEX_WIDTH-1:0]      i_feature_trace_wave_base,
    input  logic [LANES*16-1:0]                 i_feature_trace_wave_q8_8_flat,
    input  logic [VERSION_WIDTH-1:0]            i_feature_trace_wave_version,
    input  logic [FRAME_WIDTH-1:0]              i_feature_trace_wave_frame_index,
    input  logic                                i_feature_trace_wave_first,
    input  logic                                i_feature_trace_wave_last,
    input  logic                                i_feature_trace_wave_sample_reset,
    input  logic                                i_feature_trace_wave_sample_last,
    input  logic [CLASS_INDEX_WIDTH-1:0]        i_sample_label,
    input  logic                                i_logit_wave_valid,
    output logic                                o_logit_wave_ready,
    input  logic [ENGINES-1:0]                  i_logit_wave_mask,
    input  logic [CLASS_INDEX_WIDTH-1:0]        i_logit_wave_base,
    input  logic [CLASS_COUNT_WIDTH-1:0]        i_logit_wave_count,
    input  logic [ENGINES*16-1:0]               i_logit_wave_q8_8_flat,
    input  logic [VERSION_WIDTH-1:0]            i_logit_wave_version,
    input  logic [FRAME_WIDTH-1:0]              i_logit_wave_frame_index,
    input  logic                                i_logit_wave_frame_last,
    input  logic                                i_logit_wave_sample_last,

    output logic                                o_external_cfg_valid,
    input  logic                                i_external_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_external_cfg_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_external_cfg_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_external_cfg_parameter_base,
    output logic [BATCH_COUNT_WIDTH-1:0]        o_external_cfg_batch_count,
    output logic [FEATURE_COUNT_WIDTH-1:0]      o_external_cfg_source_count,
    output logic [CLASS_COUNT_WIDTH-1:0]        o_external_cfg_output_count,
    output logic [VOTE_WIDTH-1:0]               o_external_cfg_vote_increment,
    output logic [VOTE_WIDTH-1:0]               o_external_cfg_vote_threshold,
    output logic signed [CODE_WIDTH-1:0]        o_external_cfg_code_min,
    output logic signed [CODE_WIDTH-1:0]        o_external_cfg_code_max,
    input  logic                                i_external_cfg_error,
    input  logic                                i_external_active,
    output logic                                o_external_release,
    input  logic                                i_external_release_ready,
    output logic                                o_external_source_trace_valid,
    input  logic                                i_external_source_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_external_source_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0]      o_external_source_trace_index,
    output logic [15:0]                         o_external_source_trace_q016,
    input  logic                                i_external_source_trace_error,
    output logic                                o_external_logit_row_valid,
    input  logic                                i_external_logit_row_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_external_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]        o_external_logit_label,
    output logic                                o_external_logit_valid,
    input  logic                                i_external_logit_ready,
    output logic signed [15:0]                  o_external_logit_q6,
    input  logic                                i_external_logit_error,
    input  logic                                i_external_logit_row_done,
    output logic                                o_external_observation_done,
    input  logic                                i_external_observation_done_ready,
    input  logic                                i_external_window_done,
    input  logic                                i_external_window_busy,
    input  logic                                i_external_context_error,
    input  logic                                i_external_gradient_overflow,
    input  logic                                i_external_update_config_error,
    input  logic                                i_external_request_error,
    input  logic                                i_external_store_error,
    input  logic                                i_external_internal_error,

    output logic                                o_read_valid,
    input  logic                                i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_read_tag,
    input  logic                                i_read_response_valid,
    output logic                                o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]        i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_read_response_residual,
    input  logic                                i_read_response_error,
    output logic                                o_result_valid,
    input  logic                                i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_result_tag,
    output logic signed [CODE_WIDTH-1:0]        o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_result_residual,
    output logic                                o_write_code_enable,
    output logic                                o_write_residual_enable,

    output logic [VERSION_WIDTH-1:0]            o_model_version,
    output logic [VERSION_WIDTH-1:0]            o_adapt_version,
    output logic [VERSION_WIDTH-1:0]            o_core_model_version,
    output logic [VERSION_WIDTH-1:0]            o_core_adapt_version,
    output logic                                o_commit_pulse,
    output logic                                o_commit_code_dirty,
    output logic                                o_commit_optimizer_dirty,
    output logic                                o_detach_pulse,
    output logic                                o_inference_hold,
    output logic                                o_capture_complete,
    output logic                                o_learning_start_fire,
    output logic [SAMPLE_WIDTH-1:0]             o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_active_timestep,
    output logic                                o_run_done,
    output logic                                o_run_aborted,
    output logic                                o_abort_pending,
    output logic                                o_busy,
    output logic                                o_quiescent,
    output logic                                o_error
);
    logic learner_start_valid;
    logic learner_start_ready;
    logic [TIMESTEP_WIDTH-1:0] learner_timestep;
    logic learner_done;
    logic learner_busy;
    logic learner_quiescent;
    logic learner_error;
    logic normal_core_start_valid;
    logic normal_core_start_ready;
    logic [SAMPLE_WIDTH-1:0] normal_core_sample;
    logic [TIMESTEP_WIDTH-1:0] normal_core_timestep;
    logic control_core_abort;
    logic control_busy;
    logic control_quiescent;
    logic control_error;
    logic control_classifier_capture_enable;
    logic post_core_start_valid;
    logic post_core_start_ready;
    logic [TIMESTEP_WIDTH-1:0] post_core_timestep;
    logic runtime_forward_start_valid;
    logic runtime_forward_start_ready;
    logic runtime_forward_active;
    logic runtime_forward_done;
    logic core_active_q;
    logic core_post_q;
    logic core_abort_pending_q;
    logic binding_error_q;
    logic core_request_valid;
    logic core_request_post;
    logic core_start_fire;
    logic core_abort_complete;

    assign core_request_valid = normal_core_start_valid ||
        post_core_start_valid;
    assign core_request_post = post_core_start_valid;
    assign o_core_start_valid = core_request_valid &&
        runtime_forward_start_ready && !core_active_q && !binding_error_q;
    assign runtime_forward_start_valid = core_request_valid &&
        i_core_start_ready && !core_active_q && !binding_error_q;
    assign core_start_fire = o_core_start_valid && i_core_start_ready;
    assign normal_core_start_ready = !post_core_start_valid &&
        i_core_start_ready && runtime_forward_start_ready &&
        !core_active_q && !binding_error_q;
    assign post_core_start_ready = !normal_core_start_valid &&
        i_core_start_ready && runtime_forward_start_ready &&
        !core_active_q && !binding_error_q;
    assign o_core_sample = core_request_post ?
        o_active_sample : normal_core_sample;
    assign o_core_timestep = core_request_post ?
        post_core_timestep : normal_core_timestep;
    assign o_core_post_forward = core_request_post;
    // Post-commit forward is visibility verification, not another training
    // observation.  Re-entering the completed capture window would eventually
    // backpressure the classifier feature stream and prevent retirement.
    assign o_classifier_capture_enable =
        control_classifier_capture_enable && !core_post_q;
    assign o_core_abort = control_core_abort;
    assign core_abort_complete = core_active_q && core_abort_pending_q &&
        i_core_quiescent;
    assign runtime_forward_done = (core_active_q && i_core_sample_done) ||
        core_abort_complete;

    assign o_busy = control_busy || learner_busy || core_active_q ||
        runtime_forward_active;
    assign o_quiescent = control_quiescent && !core_active_q &&
        i_core_quiescent && learner_quiescent && !o_error;
    assign o_error = binding_error_q || learner_error || control_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            core_active_q <= 1'b0;
            core_post_q <= 1'b0;
            core_abort_pending_q <= 1'b0;
            binding_error_q <= 1'b0;
        end else begin
            if (i_clear_error && !core_active_q && i_core_quiescent)
                binding_error_q <= 1'b0;
            if (core_start_fire) begin
                core_active_q <= 1'b1;
                core_post_q <= core_request_post;
                core_abort_pending_q <= 1'b0;
            end
            if (control_core_abort && core_active_q)
                core_abort_pending_q <= 1'b1;
            if ((core_active_q && i_core_sample_done) ||
                core_abort_complete) begin
                core_active_q <= 1'b0;
                core_post_q <= 1'b0;
                core_abort_pending_q <= 1'b0;
            end
            if (normal_core_start_valid && post_core_start_valid)
                binding_error_q <= 1'b1;
            if (i_core_sample_done && !core_active_q)
                binding_error_q <= 1'b1;
            if (core_active_q && i_core_sample_done &&
                runtime_forward_active == 1'b0)
                binding_error_q <= 1'b1;
        end
    end

    unified_scnn_tp_final_system_control #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .BATCH_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_control (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(i_abort), .i_profile_aligned(i_profile_aligned),
        .i_start_allowed(i_start_allowed),
        .i_scnn_parameter_owner(i_scnn_parameter_owner),
        .i_scnn_learning_enable(i_scnn_learning_enable),
        .i_run_start_valid(i_run_start_valid),
        .o_run_start_ready(o_run_start_ready),
        .i_batch_count(i_batch_count), .i_timestep_count(i_timestep_count),
        .o_learner_start_valid(learner_start_valid),
        .i_learner_start_ready(learner_start_ready),
        .o_learner_timestep(learner_timestep),
        .i_learner_complete(learner_done), .i_learner_busy(learner_busy),
        .i_learner_quiescent(learner_quiescent),
        .i_learner_error(learner_error),
        .o_target_start_valid(o_target_start_valid),
        .i_target_start_ready(i_target_start_ready),
        .o_target_sample(o_target_sample),
        .o_target_timestep(o_target_timestep),
        .i_target_done(i_target_done), .i_target_busy(i_target_busy),
        .i_target_error(i_target_error),
        .o_core_start_valid(normal_core_start_valid),
        .i_core_start_ready(normal_core_start_ready),
        .o_core_sample(normal_core_sample),
        .o_core_timestep(normal_core_timestep),
        .o_classifier_capture_enable(control_classifier_capture_enable),
        .i_selected_conv_done(i_selected_conv_done && core_active_q &&
            !core_post_q),
        .i_core_sample_done(i_core_sample_done && core_active_q &&
            !core_post_q),
        .i_core_quiescent(i_core_quiescent), .i_core_error(i_core_error),
        .o_core_abort(control_core_abort),
        .o_post_forward_start_valid(post_core_start_valid),
        .i_post_forward_start_ready(post_core_start_ready),
        .o_post_forward_timestep(post_core_timestep),
        .i_post_forward_done(i_core_sample_done && core_active_q &&
            core_post_q),
        .i_post_forward_quiescent(i_core_quiescent),
        .i_post_forward_error(i_core_error), .o_post_forward_abort(),
        .o_active_sample(o_active_sample),
        .o_active_timestep(o_active_timestep), .o_run_done(o_run_done),
        .o_run_aborted(o_run_aborted), .o_abort_pending(o_abort_pending),
        .o_busy(control_busy), .o_quiescent(control_quiescent),
        .o_error(control_error)
    );

    unified_scnn_tp_selected_layers_runtime_client #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .OBS_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .OBS_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS), .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL), .KERNEL_SIZE(KERNEL_SIZE),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH), .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .MAX_FEATURES(MAX_FEATURES),
        .MAX_CLASSES(MAX_CLASSES), .LANES(LANES), .ENGINES(ENGINES),
        .FRAME_WIDTH(FRAME_WIDTH), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .USE_SAMPLE_CONTEXT(USE_SAMPLE_CONTEXT),
        .USE_CLASSIFIER_STUDENT_TRACE(USE_CLASSIFIER_STUDENT_TRACE),
        .USE_INTERNAL_MODULATION(USE_INTERNAL_MODULATION),
        .USE_EXTERNAL_OUTPUT_LEARNER(USE_EXTERNAL_OUTPUT_LEARNER),
        .INTERNAL_MODULATION_BATCH(INTERNAL_MODULATION_BATCH),
        .INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR(
            INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR),
        .SURROGATE_WAVE_CHANNELS(SURROGATE_WAVE_CHANNELS),
        .TRACE_WAVE_CHANNELS(TRACE_WAVE_CHANNELS)
    ) u_learner (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_system_parameter_owner(i_scnn_parameter_owner),
        .i_system_learning_enable(i_scnn_learning_enable),
        .i_system_profile_aligned(i_profile_aligned),
        .i_system_profile_id(i_system_profile_id),
        .i_run_valid(learner_start_valid), .o_run_ready(learner_start_ready),
        .i_run_timestep(learner_timestep),
        .i_run_timestep_count(i_timestep_count), .o_run_done(learner_done),
        .o_capture_complete(o_capture_complete),
        .o_learning_start_fire(o_learning_start_fire),
        .i_profile_id(i_profile_id), .i_conv_tensor_id(i_conv_tensor_id),
        .i_classifier_tensor_id(i_classifier_tensor_id),
        .i_batch_count(i_batch_count),
        .i_total_input_channels(i_total_input_channels),
        .i_total_output_channels(i_total_output_channels),
        .i_conv_parameter_base(i_conv_parameter_base),
        .i_conv_weight_q16_base(i_conv_weight_q16_base),
        .i_classifier_parameter_base(i_classifier_parameter_base),
        .i_classifier_feature_count(i_classifier_feature_count),
        .i_classifier_class_count(i_classifier_class_count),
        .i_classifier_trace_leak_valid(i_classifier_trace_leak_valid),
        .i_classifier_trace_leak_q8_8(i_classifier_trace_leak_q8_8),
        .i_conv_vote_increment(i_conv_vote_increment),
        .i_conv_vote_threshold(i_conv_vote_threshold),
        .i_classifier_vote_increment(i_classifier_vote_increment),
        .i_classifier_vote_threshold(i_classifier_vote_threshold),
        .i_code_min(i_code_min), .i_code_max(i_code_max),
        .i_sample_version(i_sample_version),
        .i_context_load_valid(i_context_load_valid),
        .o_context_load_ready(o_context_load_ready),
        .i_context_model_version(i_context_model_version),
        .i_context_adapt_version(i_context_adapt_version),
        .i_forward_start_valid(runtime_forward_start_valid),
        .o_forward_start_ready(runtime_forward_start_ready),
        .o_forward_start_fire(), .i_forward_done(runtime_forward_done),
        .o_forward_active(runtime_forward_active),
        .o_forward_model_version(o_core_model_version),
        .o_forward_adapt_version(o_core_adapt_version),
        .i_restore_complete(i_restore_complete),
        .o_restore_ready(o_restore_ready),
        .o_recovery_required(o_recovery_required),
        .i_sample_context_valid(i_sample_context_valid),
        .o_sample_context_ready(o_sample_context_ready),
        .i_sample_context_target(i_sample_context_target),
        .i_sample_context_sample(i_sample_context_sample),
        .i_sample_context_version(i_sample_context_version),
        .i_input_valid(i_input_valid), .o_input_ready(o_input_ready),
        .i_input_target(i_input_target), .i_input_sample(i_input_sample),
        .i_input_timestep(i_input_timestep),
        .i_input_sample_version(i_input_sample_version),
        .i_input_channel(i_input_channel),
        .i_input_spatial(i_input_spatial), .i_input_q8_8(i_input_q8_8),
        .i_trace_valid(i_trace_valid), .o_trace_ready(o_trace_ready),
        .i_trace_target(i_trace_target), .i_trace_sample(i_trace_sample),
        .i_trace_timestep(i_trace_timestep),
        .i_trace_sample_version(i_trace_sample_version),
        .i_trace_channel(i_trace_channel),
        .i_trace_spatial(i_trace_spatial), .i_trace_q8_8(i_trace_q8_8),
        .i_surrogate_valid(i_surrogate_valid),
        .o_surrogate_ready(o_surrogate_ready),
        .i_surrogate_target(i_surrogate_target),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_timestep(i_surrogate_timestep),
        .i_surrogate_sample_version(i_surrogate_sample_version),
        .i_surrogate_channel(i_surrogate_channel),
        .i_surrogate_spatial(i_surrogate_spatial),
        .i_surrogate_q8_8(i_surrogate_q8_8),
        .i_modulation_valid(i_modulation_valid),
        .o_modulation_ready(o_modulation_ready),
        .i_modulation_row(i_modulation_row),
        .i_modulation_col(i_modulation_col),
        .i_modulation_q8_8(i_modulation_q8_8),
        .i_feature_trace_wave_valid(i_feature_trace_wave_valid),
        .o_feature_trace_wave_ready(o_feature_trace_wave_ready),
        .i_feature_trace_wave_mask(i_feature_trace_wave_mask),
        .i_feature_trace_wave_base(i_feature_trace_wave_base),
        .i_feature_trace_wave_q8_8_flat(i_feature_trace_wave_q8_8_flat),
        .i_feature_trace_wave_version(i_feature_trace_wave_version),
        .i_feature_trace_wave_frame_index(i_feature_trace_wave_frame_index),
        .i_feature_trace_wave_first(i_feature_trace_wave_first),
        .i_feature_trace_wave_last(i_feature_trace_wave_last),
        .i_feature_trace_wave_sample_reset(
            i_feature_trace_wave_sample_reset),
        .i_feature_trace_wave_sample_last(i_feature_trace_wave_sample_last),
        .i_sample_label(i_sample_label),
        .i_logit_wave_valid(i_logit_wave_valid),
        .o_logit_wave_ready(o_logit_wave_ready),
        .i_logit_wave_mask(i_logit_wave_mask),
        .i_logit_wave_base(i_logit_wave_base),
        .i_logit_wave_count(i_logit_wave_count),
        .i_logit_wave_q8_8_flat(i_logit_wave_q8_8_flat),
        .i_logit_wave_version(i_logit_wave_version),
        .i_logit_wave_frame_index(i_logit_wave_frame_index),
        .i_logit_wave_frame_last(i_logit_wave_frame_last),
        .i_logit_wave_sample_last(i_logit_wave_sample_last),
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
        .o_read_valid(o_read_valid), .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr),
        .o_read_tag(o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(o_read_response_ready),
        .i_read_response_code(i_read_response_code),
        .i_read_response_residual(i_read_response_residual),
        .i_read_response_error(i_read_response_error),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_profile_id(o_result_profile_id),
        .o_result_tensor_id(o_result_tensor_id),
        .o_result_parameter_addr(o_result_parameter_addr),
        .o_result_tag(o_result_tag), .o_result_code(o_result_code),
        .o_result_residual(o_result_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_model_version(o_model_version),
        .o_adapt_version(o_adapt_version),
        .o_commit_pulse(o_commit_pulse),
        .o_commit_code_dirty(o_commit_code_dirty),
        .o_commit_optimizer_dirty(o_commit_optimizer_dirty),
        .o_detach_pulse(o_detach_pulse),
        .o_commit_pending(), .i_commit_publish_ready(1'b1),
        .o_inference_hold(o_inference_hold), .o_busy(learner_busy),
        .o_quiescent(learner_quiescent), .o_error(learner_error)
    );

    initial begin
        if (PROFILE_ID_WIDTH < 1 || MAX_BATCH < 1 || MAX_TIMESTEPS < 1)
            $error("SCNN online-system binding dimensions are invalid");
    end
endmodule

`default_nettype wire
