`default_nettype none

// Production SCNN graph boundary. It composes the TP target graph and the
// student Conv1..Conv8 graph while leaving input loading, descriptor storage,
// parameter storage and the convolution/LIF/pool executor as shared services.
module unified_scnn_tp_dual_branch_graph_adapter #(
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

    input  logic                                      i_student_start_valid,
    output logic                                      o_student_start_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_student_profile_id,
    input  logic [SAMPLE_WIDTH-1:0]                   i_student_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_student_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_student_state_version,
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

    output logic                                      o_target_observation_enable,
    output logic                                      o_target_input_prescan_enable,
    output logic                                      o_student_observation_enable,
    output logic                                      o_student_input_prescan_enable,
    output logic                                      o_layer_active_target,
    output logic                                      o_layer_active_student,
    output logic [SAMPLE_WIDTH-1:0]                   o_student_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_student_active_timestep,
    output logic                                      o_student_state_invalidated,
    output logic                                      o_target_done,
    output logic                                      o_student_done,
    output logic                                      o_target_busy,
    output logic                                      o_student_busy,
    output logic                                      o_busy,
    output logic                                      o_protocol_error
);
    logic target_start_ready_i;
    logic student_start_ready_i;
    logic target_start_valid_i;
    logic student_start_valid_i;
    logic target_error;
    logic student_error;
    logic start_conflict_q;
    logic graph_idle;

    logic target_descriptor_request_valid;
    logic target_descriptor_request_ready;
    logic [PROFILE_ID_WIDTH-1:0] target_descriptor_profile_id;
    logic [3:0] target_descriptor_layer_index;
    logic target_descriptor_valid;
    logic target_descriptor_ready;
    logic [PROFILE_ID_WIDTH-1:0] target_descriptor_response_profile_id;
    logic [3:0] target_descriptor_response_layer_index;
    logic [LAYER_WORDS*32-1:0] target_descriptor_words;
    logic target_descriptor_error;

    logic student_descriptor_request_valid;
    logic student_descriptor_request_ready;
    logic [PROFILE_ID_WIDTH-1:0] student_descriptor_profile_id;
    logic [3:0] student_descriptor_layer_index;
    logic student_descriptor_valid;
    logic student_descriptor_ready;
    logic [PROFILE_ID_WIDTH-1:0] student_descriptor_response_profile_id;
    logic [3:0] student_descriptor_response_layer_index;
    logic [LAYER_WORDS*32-1:0] student_descriptor_words;
    logic student_descriptor_error;
    logic descriptor_busy;
    logic descriptor_arbiter_error;

    logic target_layer_valid;
    logic target_layer_ready;
    logic [3:0] target_layer_index;
    logic [LAYER_WORDS*32-1:0] target_layer_words;
    logic target_layer_input_slot;
    logic target_layer_output_slot;
    logic [VERSION_WIDTH-1:0] target_layer_state_version;
    logic target_layer_done;
    logic target_layer_error;

    logic student_layer_valid;
    logic student_layer_ready;
    logic [3:0] student_layer_index;
    logic [LAYER_WORDS*32-1:0] student_layer_words;
    logic student_layer_input_slot;
    logic student_layer_output_slot;
    logic [VERSION_WIDTH-1:0] student_layer_state_version;
    logic student_layer_done;
    logic student_layer_error;
    logic layer_active_target;
    logic layer_active_student;
    logic layer_busy;
    logic layer_arbiter_error;

    assign graph_idle = !o_target_busy && !o_student_busy &&
        !descriptor_busy && !layer_busy;
    assign target_start_valid_i = i_target_start_valid &&
        o_target_start_ready;
    assign student_start_valid_i = i_student_start_valid &&
        o_student_start_ready;
    assign o_target_start_ready = target_start_ready_i && graph_idle &&
        !i_student_start_valid && !o_protocol_error;
    assign o_student_start_ready = student_start_ready_i && graph_idle &&
        !i_target_start_valid && !o_protocol_error;
    assign target_layer_error = i_layer_error && layer_active_target;
    assign student_layer_error = i_layer_error && layer_active_student;
    assign o_layer_target = layer_active_target;
    assign o_layer_active_target = layer_active_target;
    assign o_layer_active_student = layer_active_student;
    assign o_busy = !graph_idle;
    assign o_protocol_error = start_conflict_q || target_error ||
        student_error || descriptor_arbiter_error || layer_arbiter_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            start_conflict_q <= 1'b0;
        end else begin
            if (i_clear_error)
                start_conflict_q <= 1'b0;
            if (graph_idle && i_target_start_valid &&
                i_student_start_valid)
                start_conflict_q <= 1'b1;
        end
    end

    unified_scnn_tp_target_branch_t1 #(
        .LAYER_WORDS(LAYER_WORDS),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS), .TARGET_DIM(TARGET_DIM),
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS),
        .OUTPUT_SPATIAL(OUTPUT_SPATIAL), .BANK_DEPTH(BANK_DEPTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .ALLOW_RECURRENT_TIMESTEPS(ALLOW_RECURRENT_TIMESTEPS),
        .TARGET_SURROGATE_ROM_FILE(TARGET_SURROGATE_ROM_FILE)
    ) u_target_branch (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_projector_state_reset(i_projector_state_reset),
        .i_start_valid(target_start_valid_i),
        .o_start_ready(target_start_ready_i),
        .i_profile_id(i_target_profile_id),
        .i_projector_tensor_id(i_projector_tensor_id),
        .i_projector_parameter_base(i_projector_parameter_base),
        .i_sample(i_target_sample), .i_timestep(i_target_timestep),
        .i_target_state_version(i_target_state_version),
        .i_membrane_leak_q8_8(i_membrane_leak_q8_8),
        .i_trace_leak_q8_8(i_trace_leak_q8_8),
        .i_threshold_q8_8(i_threshold_q8_8),
        .i_reset_to_zero(i_reset_to_zero),
        .i_observe_projector(i_observe_projector),
        .i_label_valid(i_label_valid), .o_label_ready(o_label_ready),
        .i_label_class(i_label_class),
        .i_label_spike_q8_8(i_label_spike_q8_8),
        .i_label_last(i_label_last),
        .o_read_valid(o_projector_read_valid),
        .i_read_ready(i_projector_read_ready),
        .o_read_profile_id(o_projector_read_profile_id),
        .o_read_tensor_id(o_projector_read_tensor_id),
        .o_read_parameter_addr(o_projector_read_parameter_addr),
        .o_read_tag(o_projector_read_tag),
        .i_read_response_valid(i_projector_response_valid),
        .o_read_response_ready(o_projector_response_ready),
        .i_read_response_tag(i_projector_response_tag),
        .i_read_response_code(i_projector_response_code),
        .i_read_response_error(i_projector_response_error),
        .o_feature_write_valid(o_target_feature_write_valid),
        .i_feature_write_ready(i_target_feature_write_ready),
        .o_feature_write_slot(o_target_feature_write_slot),
        .o_feature_write_bank(o_target_feature_write_bank),
        .o_feature_write_addr(o_target_feature_write_addr),
        .o_feature_write_data_q8_8(o_target_feature_write_data_q8_8),
        .o_projector_observation_valid(o_projector_observation_valid),
        .i_projector_observation_ready(i_projector_observation_ready),
        .o_projector_observation_sample(o_projector_observation_sample),
        .o_projector_observation_timestep(
            o_projector_observation_timestep),
        .o_projector_observation_version(o_projector_observation_version),
        .o_projector_observation_index(o_projector_observation_index),
        .o_projector_observation_channel(o_projector_observation_channel),
        .o_projector_observation_spatial(o_projector_observation_spatial),
        .o_projector_observation_current_q8_8(
            o_projector_observation_current_q8_8),
        .o_projector_observation_spike_q8_8(
            o_projector_observation_spike_q8_8),
        .o_projector_observation_membrane_pre_q8_8(
            o_projector_observation_membrane_pre_q8_8),
        .o_projector_observation_membrane_next_q8_8(
            o_projector_observation_membrane_next_q8_8),
        .o_projector_observation_trace_q8_8(
            o_projector_observation_trace_q8_8),
        .o_projector_observation_surrogate_q8_8(
            o_projector_observation_surrogate_q8_8),
        .o_projector_observation_last(o_projector_observation_last),
        .o_descriptor_request_valid(target_descriptor_request_valid),
        .i_descriptor_request_ready(target_descriptor_request_ready),
        .o_descriptor_profile_id(target_descriptor_profile_id),
        .o_descriptor_layer_index(target_descriptor_layer_index),
        .i_descriptor_valid(target_descriptor_valid),
        .o_descriptor_ready(target_descriptor_ready),
        .i_descriptor_profile_id(target_descriptor_response_profile_id),
        .i_descriptor_layer_index(target_descriptor_response_layer_index),
        .i_descriptor_words(target_descriptor_words),
        .i_descriptor_error(target_descriptor_error),
        .o_layer_valid(target_layer_valid),
        .i_layer_ready(target_layer_ready),
        .o_layer_index(target_layer_index),
        .o_layer_words(target_layer_words),
        .o_layer_input_slot(target_layer_input_slot),
        .o_layer_output_slot(target_layer_output_slot),
        .o_layer_state_version(target_layer_state_version),
        .i_layer_done(target_layer_done),
        .i_layer_error(target_layer_error),
        .o_target_observation_enable(o_target_observation_enable),
        .o_target_input_prescan_enable(o_target_input_prescan_enable),
        .o_student_state_invalidated(o_student_state_invalidated),
        .o_done(o_target_done), .o_busy(o_target_busy),
        .o_protocol_error(target_error)
    );

    unified_scnn_tp_student_branch_t1 #(
        .LAYER_WORDS(LAYER_WORDS),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .FIRST_LAYER(STUDENT_FIRST_LAYER),
        .LAST_LAYER(STUDENT_LAST_LAYER),
        .SELECTED_LAYER(STUDENT_SELECTED_LAYER)
    ) u_student_branch (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(student_start_valid_i),
        .o_start_ready(student_start_ready_i),
        .i_profile_id(i_student_profile_id),
        .i_sample(i_student_sample), .i_timestep(i_student_timestep),
        .i_state_version(i_student_state_version),
        .o_input_start_valid(o_student_input_start_valid),
        .i_input_start_ready(i_student_input_start_ready),
        .o_input_sample(o_student_input_sample),
        .o_input_timestep(o_student_input_timestep),
        .o_input_state_version(o_student_input_state_version),
        .i_input_done(i_student_input_done),
        .i_input_error(i_student_input_error),
        .o_descriptor_request_valid(student_descriptor_request_valid),
        .i_descriptor_request_ready(student_descriptor_request_ready),
        .o_descriptor_profile_id(student_descriptor_profile_id),
        .o_descriptor_layer_index(student_descriptor_layer_index),
        .i_descriptor_valid(student_descriptor_valid),
        .o_descriptor_ready(student_descriptor_ready),
        .i_descriptor_profile_id(student_descriptor_response_profile_id),
        .i_descriptor_layer_index(student_descriptor_response_layer_index),
        .i_descriptor_words(student_descriptor_words),
        .i_descriptor_error(student_descriptor_error),
        .o_layer_valid(student_layer_valid),
        .i_layer_ready(student_layer_ready),
        .o_layer_index(student_layer_index),
        .o_layer_words(student_layer_words),
        .o_layer_input_slot(student_layer_input_slot),
        .o_layer_output_slot(student_layer_output_slot),
        .o_layer_state_version(student_layer_state_version),
        .i_layer_done(student_layer_done),
        .i_layer_error(student_layer_error),
        .o_selected_observation_enable(o_student_observation_enable),
        .o_selected_input_prescan_enable(
            o_student_input_prescan_enable),
        .o_active_sample(o_student_active_sample),
        .o_active_timestep(o_student_active_timestep),
        .o_done(o_student_done), .o_busy(o_student_busy),
        .o_protocol_error(student_error)
    );

    unified_scnn_tp_target_student_descriptor_arbiter #(
        .LAYER_WORDS(LAYER_WORDS),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH)
    ) u_descriptor_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_target_request_valid(target_descriptor_request_valid),
        .o_target_request_ready(target_descriptor_request_ready),
        .i_target_profile_id(target_descriptor_profile_id),
        .i_target_layer_index(target_descriptor_layer_index),
        .o_target_response_valid(target_descriptor_valid),
        .i_target_response_ready(target_descriptor_ready),
        .o_target_response_profile_id(target_descriptor_response_profile_id),
        .o_target_response_layer_index(target_descriptor_response_layer_index),
        .o_target_response_words(target_descriptor_words),
        .o_target_response_error(target_descriptor_error),
        .i_student_request_valid(student_descriptor_request_valid),
        .o_student_request_ready(student_descriptor_request_ready),
        .i_student_profile_id(student_descriptor_profile_id),
        .i_student_layer_index(student_descriptor_layer_index),
        .o_student_response_valid(student_descriptor_valid),
        .i_student_response_ready(student_descriptor_ready),
        .o_student_response_profile_id(
            student_descriptor_response_profile_id),
        .o_student_response_layer_index(
            student_descriptor_response_layer_index),
        .o_student_response_words(student_descriptor_words),
        .o_student_response_error(student_descriptor_error),
        .o_common_request_valid(o_descriptor_request_valid),
        .i_common_request_ready(i_descriptor_request_ready),
        .o_common_request_target(o_descriptor_request_target),
        .o_common_profile_id(o_descriptor_profile_id),
        .o_common_layer_index(o_descriptor_layer_index),
        .i_common_response_valid(i_descriptor_valid),
        .o_common_response_ready(o_descriptor_ready),
        .i_common_response_profile_id(i_descriptor_response_profile_id),
        .i_common_response_layer_index(i_descriptor_response_layer_index),
        .i_common_response_words(i_descriptor_words),
        .i_common_response_error(i_descriptor_error),
        .o_busy(descriptor_busy),
        .o_protocol_error(descriptor_arbiter_error)
    );

    unified_scnn_tp_target_student_layer_arbiter #(
        .LAYER_WORDS(LAYER_WORDS), .VERSION_WIDTH(VERSION_WIDTH)
    ) u_layer_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_target_valid(target_layer_valid),
        .o_target_ready(target_layer_ready),
        .i_target_layer_index(target_layer_index),
        .i_target_layer_words(target_layer_words),
        .i_target_input_slot(target_layer_input_slot),
        .i_target_output_slot(target_layer_output_slot),
        .i_target_state_version(target_layer_state_version),
        .o_target_done(target_layer_done),
        .i_student_valid(student_layer_valid),
        .o_student_ready(student_layer_ready),
        .i_student_layer_index(student_layer_index),
        .i_student_layer_words(student_layer_words),
        .i_student_input_slot(student_layer_input_slot),
        .i_student_output_slot(student_layer_output_slot),
        .i_student_state_version(student_layer_state_version),
        .o_student_done(student_layer_done),
        .o_shared_valid(o_layer_valid), .i_shared_ready(i_layer_ready),
        .o_shared_layer_index(o_layer_index),
        .o_shared_layer_words(o_layer_words),
        .o_shared_input_slot(o_layer_input_slot),
        .o_shared_output_slot(o_layer_output_slot),
        .o_shared_state_version(o_layer_state_version),
        .i_shared_done(i_layer_done), .i_shared_error(i_layer_error),
        .o_active_target(layer_active_target),
        .o_active_student(layer_active_student), .o_busy(layer_busy),
        .o_protocol_error(layer_arbiter_error)
    );
endmodule

`default_nettype wire
