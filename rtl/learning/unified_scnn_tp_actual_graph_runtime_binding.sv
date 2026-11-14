`default_nettype none

// Production SCNN physical-command boundary. One accepted normal command
// executes the student graph followed by the classifier; a post-commit command
// executes only the classifier forward. The independent target command shares
// the same descriptor and layer services through the dual-branch graph adapter.
module unified_scnn_tp_actual_graph_runtime_binding #(
    parameter int LAYER_WORDS = 12,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int TARGET_DIM = 11,
    parameter int OUTPUT_CHANNELS = 64,
    parameter int OUTPUT_SPATIAL = 32,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int CLASS_WIDTH = TARGET_DIM <= 2 ? 1 : $clog2(TARGET_DIM),
    parameter int OUTPUT_COUNT = OUTPUT_CHANNELS*OUTPUT_SPATIAL*OUTPUT_SPATIAL,
    parameter int OUTPUT_INDEX_WIDTH = OUTPUT_COUNT <= 2 ? 1 :
        $clog2(OUTPUT_COUNT),
    parameter int CHANNEL_WIDTH = OUTPUT_CHANNELS <= 2 ? 1 :
        $clog2(OUTPUT_CHANNELS),
    parameter int SPATIAL_INDEX_WIDTH = OUTPUT_SPATIAL*OUTPUT_SPATIAL <= 2 ?
        1 : $clog2(OUTPUT_SPATIAL*OUTPUT_SPATIAL),
    parameter int BANK_DEPTH = 9216,
    parameter int BANK_ADDR_WIDTH = $clog2(BANK_DEPTH),
    parameter bit ALLOW_RECURRENT_TIMESTEPS = 1'b0,
    parameter int STUDENT_FIRST_LAYER = 0,
    parameter int STUDENT_LAST_LAYER = 7,
    parameter int STUDENT_SELECTED_LAYER = 7,
    parameter bit ENABLE_FINAL_CORE_CLIENT = 1'b0,
    parameter int FINAL_CORE_Q64_ADDR_WIDTH = 22,
    parameter int FINAL_CORE_FEATURE_COUNT = 2048,
    parameter int FINAL_CORE_CLASS_COUNT = TARGET_DIM,
    parameter TARGET_SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,

    input  logic                                      i_target_start_valid,
    output logic                                      o_target_start_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_target_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_projector_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_projector_parameter_base,
    input  logic [SAMPLE_WIDTH-1:0]                   i_target_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_target_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_target_state_version,
    input  logic signed [15:0]                        i_membrane_leak_q8_8,
    input  logic signed [15:0]                        i_trace_leak_q8_8,
    input  logic signed [15:0]                        i_threshold_q8_8,
    input  logic                                      i_reset_to_zero,
    input  logic                                      i_observe_projector,
    input  logic                                      i_projector_state_reset,

    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [CLASS_WIDTH-1:0]                    i_label_class,
    input  logic signed [15:0]                        i_label_spike_q8_8,
    input  logic                                      i_label_last,

    output logic                                      o_projector_read_valid,
    input  logic                                      i_projector_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_projector_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_projector_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_projector_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_projector_read_tag,
    input  logic                                      i_projector_response_valid,
    output logic                                      o_projector_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_projector_response_tag,
    input  logic signed [15:0]                        i_projector_response_code,
    input  logic                                      i_projector_response_error,

    output logic                                      o_target_feature_write_valid,
    input  logic                                      i_target_feature_write_ready,
    output logic                                      o_target_feature_write_slot,
    output logic [3:0]                                o_target_feature_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]                o_target_feature_write_addr,
    output logic [15:0]                               o_target_feature_write_data_q8_8,

    output logic                                      o_projector_observation_valid,
    input  logic                                      i_projector_observation_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_projector_observation_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_projector_observation_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_projector_observation_version,
    output logic [OUTPUT_INDEX_WIDTH-1:0]             o_projector_observation_index,
    output logic [CHANNEL_WIDTH-1:0]                  o_projector_observation_channel,
    output logic [SPATIAL_INDEX_WIDTH-1:0]            o_projector_observation_spatial,
    output logic signed [15:0]                        o_projector_observation_current_q8_8,
    output logic [15:0]                               o_projector_observation_spike_q8_8,
    output logic signed [15:0]                        o_projector_observation_membrane_pre_q8_8,
    output logic signed [15:0]                        o_projector_observation_membrane_next_q8_8,
    output logic signed [15:0]                        o_projector_observation_trace_q8_8,
    output logic [15:0]                               o_projector_observation_surrogate_q8_8,
    output logic                                      o_projector_observation_last,

    input  logic                                      i_core_start_valid,
    output logic                                      o_core_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]                   i_core_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_core_timestep,
    input  logic                                      i_core_post_forward,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_student_profile_id,
    input  logic [VERSION_WIDTH-1:0]                  i_student_state_version,
    output logic                                      o_student_command_valid,
    output logic                                      o_student_command_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_student_command_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_student_command_timestep,

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
    output logic [PROFILE_ID_WIDTH-1:0]               o_descriptor_profile_id,
    output logic [3:0]                                o_descriptor_layer_index,
    input  logic                                      i_descriptor_valid,
    output logic                                      o_descriptor_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_descriptor_response_profile_id,
    input  logic [3:0]                                i_descriptor_response_layer_index,
    input  logic [LAYER_WORDS*32-1:0]                 i_descriptor_words,
    input  logic                                      i_descriptor_error,

    output logic                                      o_layer_valid,
    input  logic                                      i_layer_ready,
    output logic                                      o_layer_target,
    output logic [3:0]                                o_layer_index,
    output logic [LAYER_WORDS*32-1:0]                 o_layer_words,
    output logic                                      o_layer_input_slot,
    output logic                                      o_layer_output_slot,
    output logic [VERSION_WIDTH-1:0]                  o_layer_state_version,
    input  logic                                      i_layer_done,
    input  logic                                      i_layer_error,

    output logic                                      o_classifier_start_valid,
    input  logic                                      i_classifier_start_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_classifier_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_classifier_timestep,
    input  logic                                      i_classifier_done,
    input  logic                                      i_classifier_error,
    output logic                                      o_post_forward_start_valid,
    input  logic                                      i_post_forward_start_ready,
    output logic [SAMPLE_WIDTH-1:0]                 o_post_forward_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_post_forward_timestep,
    input  logic                                      i_post_forward_done,
    input  logic                                      i_post_forward_error,

    output logic                                      o_target_observation_enable,
    output logic                                      o_target_input_prescan_enable,
    output logic                                      o_student_observation_enable,
    output logic                                      o_student_input_prescan_enable,
    output logic                                      o_layer_active_target,
    output logic                                      o_layer_active_student,
    output logic [SAMPLE_WIDTH-1:0]                   o_student_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_student_active_timestep,
    output logic                                      o_student_state_invalidated,
    output logic                                      o_selected_conv_done,
    output logic                                      o_core_sample_done,
    output logic                                      o_core_aborted,
    output logic                                      o_target_done,
    output logic                                      o_student_done,
    output logic                                      o_target_busy,
    output logic                                      o_student_busy,
    output logic                                      o_busy,
    output logic                                      o_quiescent,
    output logic                                      o_protocol_error
,

    // Optional direct connection to the production final core.  The default
    // is disabled so existing graph-runtime users retain their old executor
    // boundary; when enabled, the graph and classifier flows above use this
    // one shared layer/Q64 client.
    input  logic                                      i_final_core_sample_reset,
    input  logic                                      i_final_core_sample_last,
    input  logic [PARAM_ADDR_WIDTH-1:0]
                                                       i_final_core_classifier_parameter_base,
    input  logic [$clog2(FINAL_CORE_FEATURE_COUNT+1)-1:0]
                                                       i_final_core_classifier_feature_count,
    input  logic [$clog2(FINAL_CORE_CLASS_COUNT+1)-1:0]
                                                       i_final_core_classifier_class_count,
    input  logic [VERSION_WIDTH-1:0]
                                                       i_final_core_classifier_state_version,
    input  logic [VERSION_WIDTH-1:0]
                                                       i_final_core_post_forward_state_version,
    input  logic                                      i_final_core_classifier_input_slot,
    input  logic                                      i_final_core_classifier_sample_reset,
    input  logic                                      i_final_core_classifier_sample_last,
    input  logic                                      i_final_core_classifier_observation_enable,
    input  logic                                      i_final_core_classifier_input_prescan_enable,
    output logic                                      o_final_core_session_active,
    output logic                                      o_final_core_classifier_session_active,
    output logic                                      o_final_core_layer_valid,
    input  logic                                      i_final_core_layer_ready,
    output logic [3:0]                                o_final_core_layer_index,
    output logic [LAYER_WORDS*32-1:0]                 o_final_core_layer_words,
    output logic                                      o_final_core_input_slot,
    output logic                                      o_final_core_output_slot,
    output logic [VERSION_WIDTH-1:0]                  o_final_core_state_version,
    output logic                                      o_final_core_sample_reset,
    output logic                                      o_final_core_sample_last,
    output logic                                      o_final_core_observation_enable,
    output logic                                      o_final_core_input_prescan_enable,
    input  logic                                      i_final_core_layer_done,
    input  logic                                      i_final_core_layer_busy,
    input  logic                                      i_final_core_layer_error,
    input  logic                                      i_final_core_physical_weight_read_valid,
    input  logic [FINAL_CORE_Q64_ADDR_WIDTH-1:0]
                                                       i_final_core_physical_weight_read_addr0,
    input  logic [FINAL_CORE_Q64_ADDR_WIDTH-1:0]
                                                       i_final_core_physical_weight_read_addr1,
    output logic                                      o_final_core_physical_weight_read_ready,
    input  logic                                      i_final_core_weight_result_valid,
    input  logic                                      i_final_core_weight_result_ready,
    output logic                                      o_final_core_weight_result_valid,
    output logic [63:0]                               o_final_core_weight_data0,
    output logic [63:0]                               o_final_core_weight_data1,
    output logic                                      o_final_core_weight_response_error,
    input  logic                                      i_final_core_weight_store_read_ready,
    output logic                                      o_final_core_weight_store_read_valid,
    output logic [FINAL_CORE_Q64_ADDR_WIDTH-1:0]
                                                       o_final_core_weight_store_read_addr0,
    output logic [FINAL_CORE_Q64_ADDR_WIDTH-1:0]
                                                       o_final_core_weight_store_read_addr1,
    input  logic                                      i_final_core_weight_store_result_valid,
    output logic                                      o_final_core_weight_store_result_ready,
    input  logic [63:0]                               i_final_core_weight_store_data0,
    input  logic [63:0]                               i_final_core_weight_store_data1,
    input  logic                                      i_final_core_weight_store_response_error
);
    logic i_student_start_valid;
    logic o_student_start_ready;
    logic [SAMPLE_WIDTH-1:0] i_student_sample;
    logic [TIMESTEP_WIDTH-1:0] i_student_timestep;
    logic graph_busy;
    logic graph_error;
    logic execution_busy;
    logic execution_quiescent;
    logic execution_error;
    logic graph_layer_valid_i;
    logic graph_layer_target_i;
    logic [3:0] graph_layer_index_i;
    logic [LAYER_WORDS*32-1:0] graph_layer_words_i;
    logic graph_layer_input_slot_i;
    logic graph_layer_output_slot_i;
    logic [VERSION_WIDTH-1:0] graph_layer_state_version_i;
    logic graph_layer_ready_to_runtime;
    logic graph_layer_done_to_runtime;
    logic graph_layer_error_to_runtime;
    logic classifier_start_ready_to_execution;
    logic post_forward_start_ready_to_execution;
    logic classifier_done_to_execution;
    logic classifier_error_to_execution;
    logic post_forward_done_to_execution;
    logic post_forward_error_to_execution;
    logic final_graph_layer_done;
    logic final_graph_layer_error;
    logic final_classifier_start_ready;
    logic final_post_forward_start_ready;
    logic final_classifier_done;
    logic final_post_forward_done;
    logic final_classifier_error;
    logic final_post_forward_error;
    logic final_client_busy;
    logic final_client_error;
    logic final_client_session_active;
    logic final_client_classifier_session_active;
    logic final_client_layer_valid;
    logic [3:0] final_client_layer_index;
    logic [LAYER_WORDS*32-1:0] final_client_layer_words;
    logic final_client_input_slot;
    logic final_client_output_slot;
    logic [VERSION_WIDTH-1:0] final_client_state_version;
    logic final_client_sample_reset;
    logic final_client_sample_last;
    logic final_client_observation_enable;
    logic final_client_input_prescan_enable;
    logic final_client_weight_read_ready;
    logic final_client_weight_result_valid;
    logic [63:0] final_client_weight_data0;
    logic [63:0] final_client_weight_data1;
    logic final_client_weight_response_error;
    logic final_client_weight_store_read_valid;
    logic final_client_weight_store_result_ready;

    assign o_busy = graph_busy || execution_busy;
    assign o_quiescent = execution_quiescent && !graph_busy && !graph_error;
    assign o_protocol_error = graph_error || execution_error ||
        final_client_error;
    assign o_student_command_valid = i_student_start_valid;
    assign o_student_command_ready = o_student_start_ready;
    assign o_student_command_sample = i_student_sample;
    assign o_student_command_timestep = i_student_timestep;

    generate
        if (ENABLE_FINAL_CORE_CLIENT) begin : g_final_core_client
            unified_scnn_tp_final_core_scnn_client_v1 #(
                .LAYER_WORDS(LAYER_WORDS),
                .VERSION_WIDTH(VERSION_WIDTH),
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
                .Q64_ADDR_WIDTH(FINAL_CORE_Q64_ADDR_WIDTH),
                .SAMPLE_WIDTH(SAMPLE_WIDTH),
                .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
                .FEATURE_COUNT(FINAL_CORE_FEATURE_COUNT),
                .CLASS_COUNT(FINAL_CORE_CLASS_COUNT)
            ) u_final_core_client (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_clear_error(i_clear_error),
                .i_graph_busy(graph_busy),
                .i_graph_protocol_error(graph_error),
                .i_graph_layer_valid(graph_layer_valid_i),
                .o_graph_layer_ready(graph_layer_ready_to_runtime),
                .i_graph_layer_target(graph_layer_target_i),
                .i_graph_layer_index(graph_layer_index_i),
                .i_graph_layer_words(graph_layer_words_i),
                .i_graph_input_slot(graph_layer_input_slot_i),
                .i_graph_output_slot(graph_layer_output_slot_i),
                .i_graph_state_version(graph_layer_state_version_i),
                .i_graph_sample_reset(i_final_core_sample_reset),
                .i_graph_sample_last(i_final_core_sample_last),
                .i_graph_target_observation_enable(
                    o_target_observation_enable),
                .i_graph_target_input_prescan_enable(
                    o_target_input_prescan_enable),
                .i_graph_student_observation_enable(
                    o_student_observation_enable),
                .i_graph_student_input_prescan_enable(
                    o_student_input_prescan_enable),
                .o_graph_layer_done(final_graph_layer_done),
                .o_graph_layer_error(final_graph_layer_error),
                .i_classifier_start_valid(o_classifier_start_valid),
                .o_classifier_start_ready(final_classifier_start_ready),
                .i_classifier_sample(o_classifier_sample),
                .i_classifier_timestep(o_classifier_timestep),
                .i_post_forward_start_valid(o_post_forward_start_valid),
                .o_post_forward_start_ready(final_post_forward_start_ready),
                .i_post_forward_sample(o_post_forward_sample),
                .i_post_forward_timestep(o_post_forward_timestep),
                .i_classifier_parameter_base(
                    i_final_core_classifier_parameter_base),
                .i_classifier_feature_count(
                    i_final_core_classifier_feature_count),
                .i_classifier_class_count(i_final_core_classifier_class_count),
                .i_classifier_state_version(
                    i_final_core_classifier_state_version),
                .i_post_forward_state_version(
                    i_final_core_post_forward_state_version),
                .i_classifier_input_slot(i_final_core_classifier_input_slot),
                .i_classifier_sample_reset(
                    i_final_core_classifier_sample_reset),
                .i_classifier_sample_last(i_final_core_classifier_sample_last),
                .i_classifier_observation_enable(
                    i_final_core_classifier_observation_enable),
                .i_classifier_input_prescan_enable(
                    i_final_core_classifier_input_prescan_enable),
                .o_classifier_done(final_classifier_done),
                .o_post_forward_done(final_post_forward_done),
                .o_classifier_error(final_classifier_error),
                .o_post_forward_error(final_post_forward_error),
                .o_core_session_active(final_client_session_active),
                .o_classifier_session_active(
                    final_client_classifier_session_active),
                .o_core_layer_valid(final_client_layer_valid),
                .i_core_layer_ready(i_final_core_layer_ready),
                .o_core_layer_index(final_client_layer_index),
                .o_core_layer_words(final_client_layer_words),
                .o_core_input_slot(final_client_input_slot),
                .o_core_output_slot(final_client_output_slot),
                .o_core_state_version(final_client_state_version),
                .o_core_sample_reset(final_client_sample_reset),
                .o_core_sample_last(final_client_sample_last),
                .o_core_observation_enable(final_client_observation_enable),
                .o_core_input_prescan_enable(
                    final_client_input_prescan_enable),
                .i_core_layer_done(i_final_core_layer_done),
                .i_core_layer_busy(i_final_core_layer_busy),
                .i_core_layer_error(i_final_core_layer_error),
                .i_core_weight_read_valid(
                    i_final_core_physical_weight_read_valid),
                .o_core_weight_read_ready(final_client_weight_read_ready),
                .i_core_weight_read_addr0(
                    i_final_core_physical_weight_read_addr0),
                .i_core_weight_read_addr1(
                    i_final_core_physical_weight_read_addr1),
                .o_weight_read_valid(final_client_weight_store_read_valid),
                .i_weight_read_ready(i_final_core_weight_store_read_ready),
                .o_weight_read_addr0(o_final_core_weight_store_read_addr0),
                .o_weight_read_addr1(o_final_core_weight_store_read_addr1),
                .i_weight_result_valid(
                    i_final_core_weight_store_result_valid),
                .o_weight_result_ready(
                    final_client_weight_store_result_ready),
                .i_core_weight_result_ready(
                    i_final_core_weight_result_ready),
                .o_core_weight_result_valid(
                    final_client_weight_result_valid),
                .o_core_weight_data0(final_client_weight_data0),
                .o_core_weight_data1(final_client_weight_data1),
                .o_core_weight_response_error(
                    final_client_weight_response_error),
                .i_weight_data0(i_final_core_weight_store_data0),
                .i_weight_data1(i_final_core_weight_store_data1),
                .i_weight_response_error(
                    i_final_core_weight_store_response_error),
                .o_busy(final_client_busy),
                .o_protocol_error(final_client_error)
            );

            assign graph_layer_done_to_runtime = final_graph_layer_done;
            assign graph_layer_error_to_runtime = final_graph_layer_error;
            assign classifier_start_ready_to_execution =
                final_classifier_start_ready;
            assign post_forward_start_ready_to_execution =
                final_post_forward_start_ready;
            assign classifier_done_to_execution = final_classifier_done;
            assign classifier_error_to_execution = final_classifier_error;
            assign post_forward_done_to_execution = final_post_forward_done;
            assign post_forward_error_to_execution = final_post_forward_error;
            assign o_final_core_session_active = final_client_session_active;
            assign o_final_core_classifier_session_active =
                final_client_classifier_session_active;
            assign o_final_core_layer_valid = final_client_layer_valid;
            assign o_final_core_layer_index = final_client_layer_index;
            assign o_final_core_layer_words = final_client_layer_words;
            assign o_final_core_input_slot = final_client_input_slot;
            assign o_final_core_output_slot = final_client_output_slot;
            assign o_final_core_state_version = final_client_state_version;
            assign o_final_core_sample_reset = final_client_sample_reset;
            assign o_final_core_sample_last = final_client_sample_last;
            assign o_final_core_observation_enable =
                final_client_observation_enable;
            assign o_final_core_input_prescan_enable =
                final_client_input_prescan_enable;
            assign o_final_core_physical_weight_read_ready =
                final_client_weight_read_ready;
            assign o_final_core_weight_result_valid =
                final_client_weight_result_valid;
            assign o_final_core_weight_data0 = final_client_weight_data0;
            assign o_final_core_weight_data1 = final_client_weight_data1;
            assign o_final_core_weight_response_error =
                final_client_weight_response_error;
            assign o_final_core_weight_store_read_valid =
                final_client_weight_store_read_valid;
            assign o_final_core_weight_store_result_ready =
                final_client_weight_store_result_ready;
        end else begin : g_no_final_core_client
            assign graph_layer_ready_to_runtime = i_layer_ready;
            assign graph_layer_done_to_runtime = i_layer_done;
            assign graph_layer_error_to_runtime = i_layer_error;
            assign classifier_start_ready_to_execution =
                i_classifier_start_ready;
            assign post_forward_start_ready_to_execution =
                i_post_forward_start_ready;
            assign classifier_done_to_execution = i_classifier_done;
            assign classifier_error_to_execution = i_classifier_error;
            assign post_forward_done_to_execution = i_post_forward_done;
            assign post_forward_error_to_execution = i_post_forward_error;
            assign final_graph_layer_error = 1'b0;
            assign final_client_error = 1'b0;
            assign o_final_core_session_active = 1'b0;
            assign o_final_core_classifier_session_active = 1'b0;
            assign o_final_core_layer_valid = 1'b0;
            assign o_final_core_layer_index = '0;
            assign o_final_core_layer_words = '0;
            assign o_final_core_input_slot = 1'b0;
            assign o_final_core_output_slot = 1'b0;
            assign o_final_core_state_version = '0;
            assign o_final_core_sample_reset = 1'b0;
            assign o_final_core_sample_last = 1'b0;
            assign o_final_core_observation_enable = 1'b0;
            assign o_final_core_input_prescan_enable = 1'b0;
            assign o_final_core_physical_weight_read_ready = 1'b0;
            assign o_final_core_weight_result_valid = 1'b0;
            assign o_final_core_weight_data0 = '0;
            assign o_final_core_weight_data1 = '0;
            assign o_final_core_weight_response_error = 1'b0;
            assign o_final_core_weight_store_read_valid = 1'b0;
            assign o_final_core_weight_store_read_addr0 = '0;
            assign o_final_core_weight_store_read_addr1 = '0;
            assign o_final_core_weight_store_result_ready = 1'b0;
        end
    endgenerate

    unified_scnn_tp_graph_classifier_execution_adapter #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_execution (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_core_start_valid(i_core_start_valid),
        .o_core_start_ready(o_core_start_ready),
        .i_core_sample(i_core_sample), .i_core_timestep(i_core_timestep),
        .i_core_post_forward(i_core_post_forward),
        .o_student_start_valid(i_student_start_valid),
        .i_student_start_ready(o_student_start_ready),
        .o_student_sample(i_student_sample),
        .o_student_timestep(i_student_timestep),
        .i_student_done(o_student_done), .i_student_error(graph_error),
        .o_classifier_start_valid(o_classifier_start_valid),
        .i_classifier_start_ready(classifier_start_ready_to_execution),
        .o_classifier_sample(o_classifier_sample),
        .o_classifier_timestep(o_classifier_timestep),
        .i_classifier_done(classifier_done_to_execution),
        .i_classifier_error(classifier_error_to_execution),
        .o_post_forward_start_valid(o_post_forward_start_valid),
        .i_post_forward_start_ready(post_forward_start_ready_to_execution),
        .o_post_forward_sample(o_post_forward_sample),
        .o_post_forward_timestep(o_post_forward_timestep),
        .i_post_forward_done(post_forward_done_to_execution),
        .i_post_forward_error(post_forward_error_to_execution),
        .o_selected_conv_done(o_selected_conv_done),
        .o_core_sample_done(o_core_sample_done),
        .o_aborted(o_core_aborted), .o_busy(execution_busy),
        .o_quiescent(execution_quiescent),
        .o_protocol_error(execution_error)
    );

    unified_scnn_tp_dual_branch_graph_adapter #(
        .LAYER_WORDS(LAYER_WORDS),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS), .TARGET_DIM(TARGET_DIM),
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS), .OUTPUT_SPATIAL(OUTPUT_SPATIAL),
        .BANK_DEPTH(BANK_DEPTH), .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .ALLOW_RECURRENT_TIMESTEPS(ALLOW_RECURRENT_TIMESTEPS),
        .STUDENT_FIRST_LAYER(STUDENT_FIRST_LAYER),
        .STUDENT_LAST_LAYER(STUDENT_LAST_LAYER),
        .STUDENT_SELECTED_LAYER(STUDENT_SELECTED_LAYER),
        .TARGET_SURROGATE_ROM_FILE(TARGET_SURROGATE_ROM_FILE)
    ) u_graph (.*,
        .i_layer_ready(graph_layer_ready_to_runtime),
        .i_layer_done(graph_layer_done_to_runtime),
        .i_layer_error(graph_layer_error_to_runtime),
        .o_layer_valid(graph_layer_valid_i),
        .o_layer_target(graph_layer_target_i),
        .o_layer_index(graph_layer_index_i),
        .o_layer_words(graph_layer_words_i),
        .o_layer_input_slot(graph_layer_input_slot_i),
        .o_layer_output_slot(graph_layer_output_slot_i),
        .o_layer_state_version(graph_layer_state_version_i),
        .o_busy(graph_busy),
        .o_protocol_error(graph_error)
    );

    assign o_layer_valid = graph_layer_valid_i;
    assign o_layer_target = graph_layer_target_i;
    assign o_layer_index = graph_layer_index_i;
    assign o_layer_words = graph_layer_words_i;
    assign o_layer_input_slot = graph_layer_input_slot_i;
    assign o_layer_output_slot = graph_layer_output_slot_i;
    assign o_layer_state_version = graph_layer_state_version_i;
endmodule

`default_nettype wire
