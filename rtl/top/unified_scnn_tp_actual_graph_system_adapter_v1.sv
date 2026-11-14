`timescale 1ns/1ps
`default_nettype none

// Binds the production target/student SCNN graph sequencer to the sole final
// core and to the existing selected-layer learner.  This adapter owns no
// convolution arithmetic, classifier arithmetic, or trainable parameters.
module unified_scnn_tp_actual_graph_system_adapter_v1 #(
    parameter int LAYER_WORDS = 12,
    parameter int ENGINES = 12,
    parameter int MAX_BATCH = 2,
    parameter int MAX_TIMESTEPS = 2,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int INPUT_SPATIAL_COUNT = 16,
    parameter int SURROGATE_SPATIAL_COUNT = 16,
    parameter int TARGET_DIM = 11,
    parameter int TARGET_OUTPUT_CHANNELS = 64,
    parameter int TARGET_OUTPUT_SPATIAL = 32,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH),
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int CLASSIFIER_FEATURE_COUNT = 2048,
    parameter int CLASSIFIER_CLASS_COUNT = TARGET_DIM,
    parameter int STUDENT_FIRST_LAYER = 0,
    parameter int STUDENT_LAST_LAYER = 7,
    parameter int SELECTED_LAYER = 7,
    parameter bit ALLOW_RECURRENT_TIMESTEPS = 1'b1,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int CLASS_WIDTH = TARGET_DIM <= 2 ? 1 : $clog2(TARGET_DIM)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_enable,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,

    input  logic                                      i_target_start_valid,
    output logic                                      o_target_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]                   i_target_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_target_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_target_state_version,
    input  logic [7:0]                                i_target_profile_id,
    input  logic [7:0]                                i_projector_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_projector_parameter_base,
    input  logic signed [15:0]                        i_target_membrane_leak_q8_8,
    input  logic signed [15:0]                        i_target_trace_leak_q8_8,
    input  logic signed [15:0]                        i_target_threshold_q8_8,
    input  logic                                      i_target_reset_to_zero,
    input  logic                                      i_observe_projector,
    input  logic                                      i_trace_state_reset,

    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [CLASS_WIDTH-1:0]                    i_label_class,
    input  logic signed [15:0]                        i_label_spike_q8_8,
    input  logic                                      i_label_last,

    output logic                                      o_target_feature_write_valid,
    input  logic                                      i_target_feature_write_ready,
    output logic                                      o_target_feature_write_slot,
    output logic [3:0]                                o_target_feature_write_bank,
    output logic [FEATURE_ADDR_WIDTH-1:0]             o_target_feature_write_addr,
    output logic signed [15:0]                        o_target_feature_write_data_q8_8,

    input  logic                                      i_core_start_valid,
    output logic                                      o_core_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]                   i_core_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_core_timestep,
    input  logic                                      i_core_post_forward,
    input  logic [7:0]                                i_student_profile_id,
    input  logic [VERSION_WIDTH-1:0]                  i_student_state_version,
    input  logic [VERSION_WIDTH-1:0]                  i_post_forward_state_version,

    output logic                                      o_student_input_start_valid,
    input  logic                                      i_student_input_start_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_student_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_student_input_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_student_input_state_version,
    input  logic                                      i_student_input_done,
    input  logic                                      i_student_input_error,

    output logic                                      o_descriptor_request_valid,
    input  logic                                      i_descriptor_request_ready,
    output logic                                      o_descriptor_request_target,
    output logic [7:0]                                o_descriptor_profile_id,
    output logic [3:0]                                o_descriptor_layer_index,
    input  logic                                      i_descriptor_valid,
    output logic                                      o_descriptor_ready,
    input  logic [7:0]                                i_descriptor_response_profile_id,
    input  logic [3:0]                                i_descriptor_response_layer_index,
    input  logic [LAYER_WORDS*32-1:0]                 i_descriptor_words,
    input  logic                                      i_descriptor_error,

    input  logic [PARAM_ADDR_WIDTH-1:0]               i_classifier_parameter_base,
    input  logic [$clog2(CLASSIFIER_FEATURE_COUNT+1)-1:0]
                                                       i_classifier_feature_count,
    input  logic [$clog2(CLASSIFIER_CLASS_COUNT+1)-1:0]
                                                       i_classifier_class_count,
    input  logic                                      i_classifier_input_slot,
    input  logic                                      i_classifier_sample_reset,
    input  logic                                      i_classifier_sample_last,
    input  logic                                      i_classifier_observation_enable,
    input  logic                                      i_classifier_input_prescan_enable,

    output logic                                      o_core_session_active,
    output logic                                      o_core_layer_valid,
    input  logic                                      i_core_layer_ready,
    output logic [3:0]                                o_core_layer_index,
    output logic [LAYER_WORDS*32-1:0]                 o_core_layer_words,
    output logic                                      o_core_input_slot,
    output logic                                      o_core_output_slot,
    output logic [VERSION_WIDTH-1:0]                  o_core_state_version,
    output logic                                      o_core_sample_reset,
    output logic                                      o_core_sample_last,
    output logic                                      o_core_observation_enable,
    output logic                                      o_core_input_prescan_enable,
    input  logic                                      i_core_layer_done,
    input  logic                                      i_core_layer_busy,
    input  logic                                      i_core_layer_error,

    input  logic                                      i_core_weight_read_valid,
    output logic                                      o_core_weight_read_ready,
    input  logic [Q64_ADDR_WIDTH-1:0]                 i_core_weight_read_addr0,
    input  logic [Q64_ADDR_WIDTH-1:0]                 i_core_weight_read_addr1,
    output logic                                      o_core_weight_result_valid,
    input  logic                                      i_core_weight_result_ready,
    output logic [63:0]                               o_core_weight_data0,
    output logic [63:0]                               o_core_weight_data1,
    output logic                                      o_core_weight_response_error,

    input  logic                                      i_core_prescan_valid,
    output logic                                      o_core_prescan_ready,
    input  logic [15:0]                               i_core_prescan_channel,
    input  logic [15:0]                               i_core_prescan_spatial,
    input  logic [3:0]                                i_core_prescan_layer,
    input  logic [VERSION_WIDTH-1:0]                  i_core_prescan_version,
    input  logic signed [15:0]                        i_core_prescan_data_q8_8,
    input  logic                                      i_core_lif_valid,
    output logic                                      o_core_lif_ready,
    input  logic [3:0]                                i_core_lif_layer,
    input  logic [15:0]                               i_core_lif_channel,
    input  logic [15:0]                               i_core_lif_spatial,
    input  logic [VERSION_WIDTH-1:0]                  i_core_lif_version,
    input  logic [15:0]                               i_core_lif_surrogate_q8_8,

    input  logic                                      i_core_feature_write_valid,
    output logic                                      o_core_feature_write_ready,
    input  logic                                      i_core_feature_write_slot,
    input  logic [3:0]                                i_core_feature_write_bank,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_core_feature_write_addr,
    input  logic signed [15:0]                        i_core_feature_write_data_q8_8,

    output logic                                      o_sample_context_valid,
    input  logic                                      i_sample_context_ready,
    output logic                                      o_sample_context_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_sample_context_sample,
    output logic [VERSION_WIDTH-1:0]                  o_sample_context_version,
    output logic                                      o_input_valid,
    input  logic                                      i_input_ready,
    output logic                                      o_input_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_input_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_input_version,
    output logic [8:0]                                o_input_channel,
    output logic [3:0]                                o_input_spatial,
    output logic signed [15:0]                        o_input_q8_8,
    output logic                                      o_trace_valid,
    input  logic                                      i_trace_ready,
    output logic                                      o_trace_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_trace_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_trace_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_trace_version,
    output logic [8:0]                                o_trace_channel,
    output logic [1:0]                                o_trace_spatial,
    output logic signed [15:0]                        o_trace_q8_8,
    output logic                                      o_surrogate_valid,
    input  logic                                      i_surrogate_ready,
    output logic                                      o_surrogate_target,
    output logic [SAMPLE_WIDTH-1:0]                   o_surrogate_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_surrogate_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_surrogate_version,
    output logic [8:0]                                o_surrogate_channel,
    output logic [3:0]                                o_surrogate_spatial,
    output logic signed [15:0]                        o_surrogate_q8_8,

    output logic                                      o_forward_read_valid,
    input  logic                                      i_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_forward_read_tag,
    input  logic                                      i_forward_response_valid,
    output logic                                      o_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_forward_response_tag,
    input  logic signed [15:0]                        i_forward_response_code,
    input  logic                                      i_forward_response_error,

    output logic                                      o_selected_conv_done,
    output logic                                      o_core_sample_done,
    output logic                                      o_core_aborted,
    output logic                                      o_target_done,
    output logic                                      o_target_busy,
    output logic                                      o_busy,
    output logic                                      o_quiescent,
    output logic                                      o_protocol_error
);
    logic runtime_target_start_ready;
    logic runtime_core_start_ready;
    logic target_start_fire;
    logic core_start_fire;
    logic [SAMPLE_WIDTH-1:0] target_sample_q;
    logic [TIMESTEP_WIDTH-1:0] target_timestep_q;
    logic [VERSION_WIDTH-1:0] target_version_q;

    logic layer_active_target, layer_active_student;
    logic [SAMPLE_WIDTH-1:0] student_active_sample;
    logic [TIMESTEP_WIDTH-1:0] student_active_timestep;
    logic final_classifier_session_active;
    logic final_layer_valid;
    logic final_layer_ready;
    logic [3:0] final_layer_index;
    logic [LAYER_WORDS*32-1:0] final_layer_words;
    logic final_input_slot, final_output_slot;
    logic [VERSION_WIDTH-1:0] final_state_version;
    logic final_sample_reset, final_sample_last;
    logic final_observation_enable, final_input_prescan_enable;
    logic final_layer_done;

    logic fork_cfg_ready, fork_write_ready;
    logic fork_feature_write_valid;
    logic fork_feature_write_slot;
    logic [3:0] fork_feature_write_bank;
    logic [FEATURE_ADDR_WIDTH-1:0] fork_feature_write_addr;
    logic signed [15:0] fork_feature_write_data;
    logic fork_observation_valid, fork_observation_ready;
    logic [SAMPLE_WIDTH-1:0] fork_observation_sample;
    logic fork_observation_timestep;
    logic [VERSION_WIDTH-1:0] fork_observation_version;
    logic [$clog2(MAX_OUTPUT_CHANNELS)-1:0] fork_observation_channel;
    logic [1:0] fork_observation_spatial;
    logic signed [15:0] fork_observation_trace;
    logic fork_observation_last, fork_busy, fork_error;

    logic projector_read_valid, projector_read_ready;
    logic [7:0] projector_read_profile, projector_read_tensor;
    logic [PARAM_ADDR_WIDTH-1:0] projector_read_addr;
    logic [TAG_WIDTH-1:0] projector_read_tag;
    logic projector_response_valid, projector_response_ready;
    logic [TAG_WIDTH-1:0] projector_response_tag;
    logic signed [15:0] projector_response_code;
    logic projector_response_error;

    logic q64_request_valid, q64_request_ready;
    logic [Q64_ADDR_WIDTH-1:0] q64_request_addr0, q64_request_addr1;
    logic q64_response_valid, q64_response_ready;
    logic [63:0] q64_response_data0, q64_response_data1;
    logic q64_scalar_valid, q64_scalar_ready;
    logic [PARAM_ADDR_WIDTH-1:0] q64_scalar_addr;
    logic [TAG_WIDTH-1:0] q64_scalar_tag;
    logic q64_scalar_response_valid, q64_scalar_response_ready;
    logic [TAG_WIDTH-1:0] q64_scalar_response_tag;
    logic signed [15:0] q64_scalar_response_code;
    logic q64_scalar_response_error;
    logic q64_busy, q64_error, arbiter_busy, arbiter_error;
    logic runtime_busy, runtime_quiescent, runtime_error;
    logic observation_error;

    logic [31:0] selected_word6;
    logic [15:0] selected_pooled_width, selected_pooled_height;
    logic selected_layer_handshake;
    logic context_required;

    assign context_required = i_target_start_valid ||
        (i_core_start_valid && !i_core_post_forward);
    assign o_target_start_ready = i_enable && runtime_target_start_ready &&
        i_sample_context_ready && !i_core_start_valid;
    assign o_core_start_ready = i_enable && runtime_core_start_ready &&
        (!context_required || i_core_post_forward || i_sample_context_ready) &&
        !i_target_start_valid;
    assign target_start_fire = i_target_start_valid && o_target_start_ready;
    assign core_start_fire = i_core_start_valid && o_core_start_ready;
    assign o_sample_context_valid = i_enable && !i_target_start_valid &&
        i_core_start_valid && !i_core_post_forward && runtime_core_start_ready ||
        i_enable && i_target_start_valid && !i_core_start_valid &&
        runtime_target_start_ready;
    assign o_sample_context_target = i_target_start_valid;
    assign o_sample_context_sample = i_target_start_valid ? i_target_sample :
        i_core_sample;
    assign o_sample_context_version = i_target_start_valid ?
        i_target_state_version : i_student_state_version;

    assign selected_word6 = final_layer_words[6*32 +: 32];
    assign selected_pooled_width = selected_word6[23:22] == 0 ?
        {10'd0, selected_word6[5:0]} :
        ({10'd0, selected_word6[5:0]} >> 1);
    assign selected_pooled_height = selected_word6[23:22] == 0 ?
        {10'd0, selected_word6[11:6]} :
        ({10'd0, selected_word6[11:6]} >> 1);
    assign selected_layer_handshake = final_layer_valid && final_layer_ready &&
        final_layer_index == 4'(SELECTED_LAYER);

    assign o_core_session_active = i_enable &&
        (runtime_busy || final_layer_valid || final_classifier_session_active);
    assign o_core_layer_valid = i_enable && final_layer_valid &&
        (final_layer_index != 4'(SELECTED_LAYER) || fork_cfg_ready);
    assign final_layer_ready = i_enable && i_core_layer_ready &&
        (final_layer_index != 4'(SELECTED_LAYER) || fork_cfg_ready);
    assign o_core_layer_index = final_layer_index;
    assign o_core_layer_words = final_layer_words;
    assign o_core_input_slot = final_input_slot;
    assign o_core_output_slot = final_output_slot;
    assign o_core_state_version = final_state_version;
    assign o_core_sample_reset = final_sample_reset;
    assign o_core_sample_last = final_sample_last;
    assign o_core_observation_enable = final_observation_enable;
    assign o_core_input_prescan_enable = final_input_prescan_enable;
    assign o_core_feature_write_ready = i_enable && fork_write_ready;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            target_sample_q <= '0;
            target_timestep_q <= '0;
            target_version_q <= '0;
        end else if (target_start_fire) begin
            target_sample_q <= i_target_sample;
            target_timestep_q <= i_target_timestep;
            target_version_q <= i_target_state_version;
        end
    end

    unified_scnn_tp_target_conv8_trace_fork_t1 #(
        .ENGINES(ENGINES), .MAX_BATCH(MAX_BATCH),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .PATH_CONTEXTS(2)
    ) u_selected_trace (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_state_reset(i_trace_state_reset),
        .i_cfg_valid(selected_layer_handshake), .o_cfg_ready(fork_cfg_ready),
        .i_cfg_output_channels(selected_word6[21:12]),
        .i_cfg_pooled_width(selected_pooled_width),
        .i_cfg_pooled_height(selected_pooled_height),
        .i_cfg_output_slot(final_output_slot),
        .i_cfg_path(layer_active_student),
        .i_cfg_sample(layer_active_target ? target_sample_q :
            student_active_sample),
        .i_cfg_timestep(layer_active_target ? target_timestep_q :
            student_active_timestep),
        .i_cfg_version(final_state_version),
        .i_cfg_trace_leak_q8_8(i_target_trace_leak_q8_8),
        .i_write_valid(i_enable && i_core_feature_write_valid),
        .o_write_ready(fork_write_ready),
        .i_write_slot(i_core_feature_write_slot),
        .i_write_bank(i_core_feature_write_bank),
        .i_write_addr(i_core_feature_write_addr),
        .i_write_spike_q8_8(i_core_feature_write_data_q8_8),
        .o_feature_write_valid(fork_feature_write_valid),
        .i_feature_write_ready(1'b1),
        .o_feature_write_slot(fork_feature_write_slot),
        .o_feature_write_bank(fork_feature_write_bank),
        .o_feature_write_addr(fork_feature_write_addr),
        .o_feature_write_data_q8_8(fork_feature_write_data),
        .i_layer_done(i_core_layer_done), .o_layer_done(final_layer_done),
        .o_observation_valid(fork_observation_valid),
        .i_observation_ready(fork_observation_ready),
        .o_observation_sample(fork_observation_sample),
        .o_observation_timestep(fork_observation_timestep),
        .o_observation_version(fork_observation_version),
        .o_observation_channel(fork_observation_channel),
        .o_observation_spatial(fork_observation_spatial),
        .o_observation_trace_q8_8(fork_observation_trace),
        .o_observation_last(fork_observation_last),
        .o_busy(fork_busy), .o_error(fork_error)
    );

    unified_scnn_tp_actual_graph_observation_binding #(
        .MAX_BATCH(MAX_BATCH), .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .INPUT_SPATIAL_COUNT(INPUT_SPATIAL_COUNT),
        .SURROGATE_SPATIAL_COUNT(SURROGATE_SPATIAL_COUNT),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_observations (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_enable(i_enable), .i_active_target(layer_active_target),
        .i_active_student(layer_active_student),
        .i_target_sample(target_sample_q),
        .i_target_timestep(target_timestep_q[0]),
        .i_target_version(target_version_q),
        .i_student_sample(student_active_sample),
        .i_student_timestep(student_active_timestep[0]),
        .i_student_version(i_student_state_version),
        .i_prescan_valid(i_core_prescan_valid),
        .o_prescan_ready(o_core_prescan_ready),
        .i_prescan_channel(i_core_prescan_channel),
        .i_prescan_spatial(i_core_prescan_spatial),
        .i_prescan_layer(i_core_prescan_layer),
        .i_prescan_version(i_core_prescan_version),
        .i_prescan_data_q8_8(i_core_prescan_data_q8_8),
        .i_lif_valid(i_core_lif_valid), .o_lif_ready(o_core_lif_ready),
        .i_lif_layer(i_core_lif_layer), .i_lif_channel(i_core_lif_channel),
        .i_lif_spatial(i_core_lif_spatial),
        .i_lif_version(i_core_lif_version),
        .i_lif_surrogate_q8_8(i_core_lif_surrogate_q8_8),
        .i_trace_valid(fork_observation_valid),
        .o_trace_source_ready(fork_observation_ready),
        .i_trace_sample(fork_observation_sample),
        .i_trace_timestep(fork_observation_timestep),
        .i_trace_version(fork_observation_version),
        .i_trace_channel(9'(fork_observation_channel)),
        .i_trace_spatial(fork_observation_spatial),
        .i_trace_q8_8(fork_observation_trace),
        .o_input_valid(o_input_valid), .i_input_ready(i_input_ready),
        .o_input_target(o_input_target), .o_input_sample(o_input_sample),
        .o_input_timestep(o_input_timestep), .o_input_version(o_input_version),
        .o_input_channel(o_input_channel), .o_input_spatial(o_input_spatial),
        .o_input_q8_8(o_input_q8_8),
        .o_surrogate_valid(o_surrogate_valid),
        .i_surrogate_ready(i_surrogate_ready),
        .o_surrogate_target(o_surrogate_target),
        .o_surrogate_sample(o_surrogate_sample),
        .o_surrogate_timestep(o_surrogate_timestep),
        .o_surrogate_version(o_surrogate_version),
        .o_surrogate_channel(o_surrogate_channel),
        .o_surrogate_spatial(o_surrogate_spatial),
        .o_surrogate_q8_8(o_surrogate_q8_8),
        .o_trace_valid(o_trace_valid), .i_trace_ready(i_trace_ready),
        .o_trace_target(o_trace_target), .o_trace_sample(o_trace_sample),
        .o_trace_timestep(o_trace_timestep), .o_trace_version(o_trace_version),
        .o_trace_channel(o_trace_channel), .o_trace_spatial(o_trace_spatial),
        .o_trace_q8_8(o_trace_q8_8), .o_protocol_error(observation_error)
    );

    unified_scnn_classifier_q64_forward_bridge #(
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(16)
    ) u_q64_bridge (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_request_valid(q64_request_valid), .o_request_ready(q64_request_ready),
        .i_request_addr0(q64_request_addr0),
        .i_request_addr1(q64_request_addr1),
        .o_response_valid(q64_response_valid),
        .i_response_ready(q64_response_ready),
        .o_response_data0(q64_response_data0),
        .o_response_data1(q64_response_data1),
        .o_forward_read_valid(q64_scalar_valid),
        .i_forward_read_ready(q64_scalar_ready),
        .o_forward_read_addr(q64_scalar_addr),
        .o_forward_read_tag(q64_scalar_tag),
        .i_forward_response_valid(q64_scalar_response_valid),
        .o_forward_response_ready(q64_scalar_response_ready),
        .i_forward_response_tag(q64_scalar_response_tag),
        .i_forward_response_code(q64_scalar_response_code),
        .i_forward_response_error(q64_scalar_response_error),
        .o_busy(q64_busy), .o_protocol_error(q64_error)
    );

    unified_tp_two_client_forward_read_arbiter #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(16)
    ) u_forward_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_c0_valid(projector_read_valid), .o_c0_ready(projector_read_ready),
        .i_c0_addr(projector_read_addr), .i_c0_tag(projector_read_tag),
        .o_c0_response_valid(projector_response_valid),
        .i_c0_response_ready(projector_response_ready),
        .o_c0_response_tag(projector_response_tag),
        .o_c0_response_code(projector_response_code),
        .o_c0_response_error(projector_response_error),
        .i_c1_valid(q64_scalar_valid), .o_c1_ready(q64_scalar_ready),
        .i_c1_addr(q64_scalar_addr), .i_c1_tag(q64_scalar_tag),
        .o_c1_response_valid(q64_scalar_response_valid),
        .i_c1_response_ready(q64_scalar_response_ready),
        .o_c1_response_tag(q64_scalar_response_tag),
        .o_c1_response_code(q64_scalar_response_code),
        .o_c1_response_error(q64_scalar_response_error),
        .o_store_valid(o_forward_read_valid),
        .i_store_ready(i_forward_read_ready),
        .o_store_addr(o_forward_read_addr), .o_store_tag(o_forward_read_tag),
        .i_store_response_valid(i_forward_response_valid),
        .o_store_response_ready(o_forward_response_ready),
        .i_store_response_tag(i_forward_response_tag),
        .i_store_response_code(i_forward_response_code),
        .i_store_response_error(i_forward_response_error),
        .o_busy(arbiter_busy), .o_protocol_error(arbiter_error)
    );

    unified_scnn_tp_actual_graph_runtime_binding #(
        .LAYER_WORDS(LAYER_WORDS), .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS), .TARGET_DIM(TARGET_DIM),
        .OUTPUT_CHANNELS(TARGET_OUTPUT_CHANNELS),
        .OUTPUT_SPATIAL(TARGET_OUTPUT_SPATIAL),
        .BANK_DEPTH(FEATURE_DEPTH), .BANK_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .ALLOW_RECURRENT_TIMESTEPS(ALLOW_RECURRENT_TIMESTEPS),
        .STUDENT_FIRST_LAYER(STUDENT_FIRST_LAYER),
        .STUDENT_LAST_LAYER(STUDENT_LAST_LAYER),
        .STUDENT_SELECTED_LAYER(SELECTED_LAYER),
        .ENABLE_FINAL_CORE_CLIENT(1'b1),
        .FINAL_CORE_Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .FINAL_CORE_FEATURE_COUNT(CLASSIFIER_FEATURE_COUNT),
        .FINAL_CORE_CLASS_COUNT(CLASSIFIER_CLASS_COUNT)
    ) u_runtime (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_target_start_valid(i_enable && i_target_start_valid &&
            i_sample_context_ready && !i_core_start_valid),
        .o_target_start_ready(runtime_target_start_ready),
        .i_target_profile_id(i_target_profile_id),
        .i_projector_tensor_id(i_projector_tensor_id),
        .i_projector_parameter_base(i_projector_parameter_base),
        .i_target_sample(i_target_sample), .i_target_timestep(i_target_timestep),
        .i_target_state_version(i_target_state_version),
        .i_membrane_leak_q8_8(i_target_membrane_leak_q8_8),
        .i_trace_leak_q8_8(i_target_trace_leak_q8_8),
        .i_threshold_q8_8(i_target_threshold_q8_8),
        .i_reset_to_zero(i_target_reset_to_zero),
        .i_observe_projector(i_observe_projector),
        .i_projector_state_reset(i_trace_state_reset),
        .i_label_valid(i_label_valid), .o_label_ready(o_label_ready),
        .i_label_class(i_label_class), .i_label_spike_q8_8(i_label_spike_q8_8),
        .i_label_last(i_label_last),
        .o_projector_read_valid(projector_read_valid),
        .i_projector_read_ready(projector_read_ready),
        .o_projector_read_profile_id(projector_read_profile),
        .o_projector_read_tensor_id(projector_read_tensor),
        .o_projector_read_parameter_addr(projector_read_addr),
        .o_projector_read_tag(projector_read_tag),
        .i_projector_response_valid(projector_response_valid),
        .o_projector_response_ready(projector_response_ready),
        .i_projector_response_tag(projector_response_tag),
        .i_projector_response_code(projector_response_code),
        .i_projector_response_error(projector_response_error),
        .o_target_feature_write_valid(o_target_feature_write_valid),
        .i_target_feature_write_ready(i_target_feature_write_ready),
        .o_target_feature_write_slot(o_target_feature_write_slot),
        .o_target_feature_write_bank(o_target_feature_write_bank),
        .o_target_feature_write_addr(o_target_feature_write_addr),
        .o_target_feature_write_data_q8_8(o_target_feature_write_data_q8_8),
        .o_projector_observation_valid(),
        .i_projector_observation_ready(1'b1),
        .o_projector_observation_sample(),
        .o_projector_observation_timestep(),
        .o_projector_observation_version(),
        .o_projector_observation_index(),
        .o_projector_observation_channel(),
        .o_projector_observation_spatial(),
        .o_projector_observation_current_q8_8(),
        .o_projector_observation_spike_q8_8(),
        .o_projector_observation_membrane_pre_q8_8(),
        .o_projector_observation_membrane_next_q8_8(),
        .o_projector_observation_trace_q8_8(),
        .o_projector_observation_surrogate_q8_8(),
        .o_projector_observation_last(),
        .i_core_start_valid(i_enable && i_core_start_valid &&
            !i_target_start_valid &&
            (i_core_post_forward || i_sample_context_ready)),
        .o_core_start_ready(runtime_core_start_ready),
        .i_core_sample(i_core_sample), .i_core_timestep(i_core_timestep),
        .i_core_post_forward(i_core_post_forward),
        .i_student_profile_id(i_student_profile_id),
        .i_student_state_version(i_student_state_version),
        .o_student_command_valid(), .o_student_command_ready(),
        .o_student_command_sample(), .o_student_command_timestep(),
        .o_student_input_start_valid(o_student_input_start_valid),
        .i_student_input_start_ready(i_student_input_start_ready),
        .o_student_input_sample(o_student_input_sample),
        .o_student_input_timestep(o_student_input_timestep),
        .o_student_input_state_version(o_student_input_state_version),
        .i_student_input_done(i_student_input_done),
        .i_student_input_error(i_student_input_error),
        .o_descriptor_request_valid(o_descriptor_request_valid),
        .i_descriptor_request_ready(i_descriptor_request_ready),
        .o_descriptor_request_target(o_descriptor_request_target),
        .o_descriptor_profile_id(o_descriptor_profile_id),
        .o_descriptor_layer_index(o_descriptor_layer_index),
        .i_descriptor_valid(i_descriptor_valid),
        .o_descriptor_ready(o_descriptor_ready),
        .i_descriptor_response_profile_id(i_descriptor_response_profile_id),
        .i_descriptor_response_layer_index(i_descriptor_response_layer_index),
        .i_descriptor_words(i_descriptor_words),
        .i_descriptor_error(i_descriptor_error),
        .o_layer_valid(), .i_layer_ready(1'b0), .o_layer_target(),
        .o_layer_index(), .o_layer_words(), .o_layer_input_slot(),
        .o_layer_output_slot(), .o_layer_state_version(),
        .i_layer_done(1'b0), .i_layer_error(1'b0),
        .o_classifier_start_valid(), .i_classifier_start_ready(1'b0),
        .o_classifier_sample(), .o_classifier_timestep(),
        .i_classifier_done(1'b0), .i_classifier_error(1'b0),
        .o_post_forward_start_valid(), .i_post_forward_start_ready(1'b0),
        .o_post_forward_sample(), .o_post_forward_timestep(),
        .i_post_forward_done(1'b0), .i_post_forward_error(1'b0),
        .o_target_observation_enable(),
        .o_target_input_prescan_enable(),
        .o_student_observation_enable(),
        .o_student_input_prescan_enable(),
        .o_layer_active_target(layer_active_target),
        .o_layer_active_student(layer_active_student),
        .o_student_active_sample(student_active_sample),
        .o_student_active_timestep(student_active_timestep),
        .o_student_state_invalidated(),
        .o_selected_conv_done(o_selected_conv_done),
        .o_core_sample_done(o_core_sample_done),
        .o_core_aborted(o_core_aborted),
        .o_target_done(o_target_done), .o_student_done(),
        .o_target_busy(o_target_busy), .o_student_busy(),
        .o_busy(runtime_busy), .o_quiescent(runtime_quiescent),
        .o_protocol_error(runtime_error),
        .i_final_core_sample_reset(1'b1),
        .i_final_core_sample_last(1'b1),
        .i_final_core_classifier_parameter_base(i_classifier_parameter_base),
        .i_final_core_classifier_feature_count(i_classifier_feature_count),
        .i_final_core_classifier_class_count(i_classifier_class_count),
        .i_final_core_classifier_state_version(i_student_state_version),
        .i_final_core_post_forward_state_version(
            i_post_forward_state_version),
        .i_final_core_classifier_input_slot(i_classifier_input_slot),
        .i_final_core_classifier_sample_reset(i_classifier_sample_reset),
        .i_final_core_classifier_sample_last(i_classifier_sample_last),
        .i_final_core_classifier_observation_enable(
            i_classifier_observation_enable),
        .i_final_core_classifier_input_prescan_enable(
            i_classifier_input_prescan_enable),
        .o_final_core_session_active(),
        .o_final_core_classifier_session_active(
            final_classifier_session_active),
        .o_final_core_layer_valid(final_layer_valid),
        .i_final_core_layer_ready(final_layer_ready),
        .o_final_core_layer_index(final_layer_index),
        .o_final_core_layer_words(final_layer_words),
        .o_final_core_input_slot(final_input_slot),
        .o_final_core_output_slot(final_output_slot),
        .o_final_core_state_version(final_state_version),
        .o_final_core_sample_reset(final_sample_reset),
        .o_final_core_sample_last(final_sample_last),
        .o_final_core_observation_enable(final_observation_enable),
        .o_final_core_input_prescan_enable(final_input_prescan_enable),
        .i_final_core_layer_done(final_layer_done),
        .i_final_core_layer_busy(i_core_layer_busy),
        .i_final_core_layer_error(i_core_layer_error || fork_error),
        .i_final_core_physical_weight_read_valid(i_core_weight_read_valid),
        .i_final_core_physical_weight_read_addr0(i_core_weight_read_addr0),
        .i_final_core_physical_weight_read_addr1(i_core_weight_read_addr1),
        .o_final_core_physical_weight_read_ready(o_core_weight_read_ready),
        .i_final_core_weight_result_valid(1'b0),
        .i_final_core_weight_result_ready(i_core_weight_result_ready),
        .o_final_core_weight_result_valid(o_core_weight_result_valid),
        .o_final_core_weight_data0(o_core_weight_data0),
        .o_final_core_weight_data1(o_core_weight_data1),
        .o_final_core_weight_response_error(o_core_weight_response_error),
        .i_final_core_weight_store_read_ready(q64_request_ready),
        .o_final_core_weight_store_read_valid(q64_request_valid),
        .o_final_core_weight_store_read_addr0(q64_request_addr0),
        .o_final_core_weight_store_read_addr1(q64_request_addr1),
        .i_final_core_weight_store_result_valid(q64_response_valid),
        .o_final_core_weight_store_result_ready(q64_response_ready),
        .i_final_core_weight_store_data0(q64_response_data0),
        .i_final_core_weight_store_data1(q64_response_data1),
        .i_final_core_weight_store_response_error(q64_error)
    );

    assign o_busy = runtime_busy || fork_busy || q64_busy || arbiter_busy;
    assign o_quiescent = runtime_quiescent && !fork_busy && !q64_busy &&
        !arbiter_busy;
    assign o_protocol_error = runtime_error || fork_error ||
        observation_error || q64_error || arbiter_error ||
        (i_enable && i_target_start_valid && i_core_start_valid);

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_forwarding;
    always_comb unused_forwarding = ^{projector_read_profile,
        projector_read_tensor, fork_feature_write_valid,
        fork_feature_write_slot, fork_feature_write_bank,
        fork_feature_write_addr, fork_feature_write_data,
        fork_observation_last};
    /* verilator lint_on UNUSEDSIGNAL */

    initial begin
        if (LAYER_WORDS < 7 || MAX_OUTPUT_CHANNELS < 2 ||
            FEATURE_ADDR_WIDTH < 1 || PARAM_ADDR_WIDTH != Q64_ADDR_WIDTH+2)
            $error("SCNN actual-graph system adapter parameter mismatch");
    end
endmodule

`default_nettype wire
