`default_nettype none

// Composes the real label projector, target feature writer and serialized
// Conv2..Conv8 sequencer. The convolution/LIF/pool engine and feature
// scratchpad remain external shared resources; this boundary only drives their
// existing descriptor, layer and write interfaces.
module unified_scnn_tp_target_branch_t1 #(
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
    parameter int SPATIAL_INDEX_WIDTH = OUTPUT_SPATIAL*OUTPUT_SPATIAL <= 2 ? 1 :
        $clog2(OUTPUT_SPATIAL*OUTPUT_SPATIAL),
    parameter int BANK_DEPTH = 9216,
    parameter int BANK_ADDR_WIDTH = $clog2(BANK_DEPTH),
    parameter bit ALLOW_RECURRENT_TIMESTEPS = 1'b0,
    parameter TARGET_SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_projector_state_reset,

    input  logic                                      i_start_valid,
    output logic                                      o_start_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_projector_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_projector_parameter_base,
    input  logic [SAMPLE_WIDTH-1:0]                   i_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_target_state_version,
    input  logic signed [15:0]                        i_membrane_leak_q8_8,
    input  logic signed [15:0]                        i_trace_leak_q8_8,
    input  logic signed [15:0]                        i_threshold_q8_8,
    input  logic                                      i_reset_to_zero,
    input  logic                                      i_observe_projector,

    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [CLASS_WIDTH-1:0]                    i_label_class,
    input  logic signed [15:0]                        i_label_spike_q8_8,
    input  logic                                      i_label_last,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_read_tag,
    input  logic                                      i_read_response_valid,
    output logic                                      o_read_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_read_response_tag,
    input  logic signed [15:0]                        i_read_response_code,
    input  logic                                      i_read_response_error,

    output logic                                      o_feature_write_valid,
    input  logic                                      i_feature_write_ready,
    output logic                                      o_feature_write_slot,
    output logic [3:0]                                o_feature_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]                o_feature_write_addr,
    output logic [15:0]                               o_feature_write_data_q8_8,

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

    output logic                                      o_descriptor_request_valid,
    input  logic                                      i_descriptor_request_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_descriptor_profile_id,
    output logic [3:0]                                o_descriptor_layer_index,
    input  logic                                      i_descriptor_valid,
    output logic                                      o_descriptor_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_descriptor_profile_id,
    input  logic [3:0]                                i_descriptor_layer_index,
    input  logic [LAYER_WORDS*32-1:0]                 i_descriptor_words,
    input  logic                                      i_descriptor_error,

    output logic                                      o_layer_valid,
    input  logic                                      i_layer_ready,
    output logic [3:0]                                o_layer_index,
    output logic [LAYER_WORDS*32-1:0]                 o_layer_words,
    output logic                                      o_layer_input_slot,
    output logic                                      o_layer_output_slot,
    output logic [VERSION_WIDTH-1:0]                  o_layer_state_version,
    input  logic                                      i_layer_done,
    input  logic                                      i_layer_error,
    output logic                                      o_target_observation_enable,
    output logic                                      o_target_input_prescan_enable,
    output logic                                      o_student_state_invalidated,

    output logic                                      o_done,
    output logic                                      o_busy,
    output logic                                      o_protocol_error
);
    logic projector_cfg_valid;
    logic projector_cfg_ready;
    logic [PROFILE_ID_WIDTH-1:0] projector_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] projector_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] projector_parameter_base;
    logic [SAMPLE_WIDTH-1:0] projector_sample;
    logic [TIMESTEP_WIDTH-1:0] projector_timestep;
    logic [VERSION_WIDTH-1:0] projector_state_version;
    logic projector_done;
    logic projector_busy;
    logic projector_error;
    logic projector_observation_valid;
    logic projector_observation_ready;
    logic projector_observation_last;
    logic [SAMPLE_WIDTH-1:0] projector_observation_sample;
    logic [TIMESTEP_WIDTH-1:0] projector_observation_timestep;
    logic [VERSION_WIDTH-1:0] projector_observation_version;
    logic [OUTPUT_INDEX_WIDTH-1:0] projector_observation_index;
    logic [CHANNEL_WIDTH-1:0] projector_observation_channel;
    logic [SPATIAL_INDEX_WIDTH-1:0] projector_observation_spatial;
    logic signed [15:0] projector_observation_current;
    logic [15:0] projector_observation_spike;
    logic signed [15:0] projector_observation_membrane_pre;
    logic signed [15:0] projector_observation_membrane_next;
    logic signed [15:0] projector_observation_trace;
    logic [15:0] projector_observation_surrogate;
    logic feature_writer_ready;
    logic feature_mapping_error;
    logic observe_projector_q;
    logic signed [15:0] membrane_leak_q;
    logic signed [15:0] trace_leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic start_fire;
    logic sequencer_busy;
    logic sequencer_error;

    assign start_fire = i_start_valid && o_start_ready;
    assign o_projector_observation_valid = observe_projector_q &&
        projector_observation_valid && feature_writer_ready;
    assign projector_observation_ready = feature_writer_ready &&
        (!observe_projector_q || i_projector_observation_ready);

    assign o_projector_observation_sample = projector_observation_sample;
    assign o_projector_observation_timestep = projector_observation_timestep;
    assign o_projector_observation_version = projector_observation_version;
    assign o_projector_observation_index = projector_observation_index;
    assign o_projector_observation_channel = projector_observation_channel;
    assign o_projector_observation_spatial = projector_observation_spatial;
    assign o_projector_observation_current_q8_8 = projector_observation_current;
    assign o_projector_observation_spike_q8_8 = projector_observation_spike;
    assign o_projector_observation_membrane_pre_q8_8 =
        projector_observation_membrane_pre;
    assign o_projector_observation_membrane_next_q8_8 =
        projector_observation_membrane_next;
    assign o_projector_observation_trace_q8_8 = projector_observation_trace;
    assign o_projector_observation_surrogate_q8_8 =
        projector_observation_surrogate;
    assign o_projector_observation_last = projector_observation_last;
    assign o_busy = sequencer_busy || projector_busy;
    assign o_protocol_error = sequencer_error || projector_error ||
        feature_mapping_error;

    unified_scnn_tp_target_branch_sequencer_t1 #(
        .LAYER_WORDS(LAYER_WORDS),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .ALLOW_RECURRENT_TIMESTEPS(ALLOW_RECURRENT_TIMESTEPS)
    ) u_sequencer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(i_start_valid), .o_start_ready(o_start_ready),
        .i_profile_id(i_profile_id),
        .i_projector_tensor_id(i_projector_tensor_id),
        .i_projector_parameter_base(i_projector_parameter_base),
        .i_sample(i_sample), .i_timestep(i_timestep),
        .i_target_state_version(i_target_state_version),
        .o_projector_cfg_valid(projector_cfg_valid),
        .i_projector_cfg_ready(projector_cfg_ready),
        .o_projector_profile_id(projector_profile_id),
        .o_projector_tensor_id(projector_tensor_id),
        .o_projector_parameter_base(projector_parameter_base),
        .o_projector_sample(projector_sample),
        .o_projector_timestep(projector_timestep),
        .o_projector_state_version(projector_state_version),
        .i_projector_done(projector_done),
        .i_projector_error(projector_error || feature_mapping_error),
        .o_descriptor_request_valid(o_descriptor_request_valid),
        .i_descriptor_request_ready(i_descriptor_request_ready),
        .o_descriptor_profile_id(o_descriptor_profile_id),
        .o_descriptor_layer_index(o_descriptor_layer_index),
        .i_descriptor_valid(i_descriptor_valid),
        .o_descriptor_ready(o_descriptor_ready),
        .i_descriptor_profile_id(i_descriptor_profile_id),
        .i_descriptor_layer_index(i_descriptor_layer_index),
        .i_descriptor_words(i_descriptor_words),
        .i_descriptor_error(i_descriptor_error),
        .o_layer_valid(o_layer_valid), .i_layer_ready(i_layer_ready),
        .o_layer_index(o_layer_index), .o_layer_words(o_layer_words),
        .o_layer_input_slot(o_layer_input_slot),
        .o_layer_output_slot(o_layer_output_slot),
        .o_layer_state_version(o_layer_state_version),
        .i_layer_done(i_layer_done), .i_layer_error(i_layer_error),
        .o_target_observation_enable(o_target_observation_enable),
        .o_target_input_prescan_enable(o_target_input_prescan_enable),
        .o_student_state_invalidated(o_student_state_invalidated),
        .o_done(o_done), .o_busy(sequencer_busy),
        .o_protocol_error(sequencer_error)
    );

    unified_scnn_tp_target_projector_q8_8 #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS), .TARGET_DIM(TARGET_DIM),
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS),
        .OUTPUT_SPATIAL(OUTPUT_SPATIAL),
        .TARGET_SURROGATE_ROM_FILE(TARGET_SURROGATE_ROM_FILE)
    ) u_projector (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_sample_state_reset(i_projector_state_reset),
        .i_cfg_valid(projector_cfg_valid), .o_cfg_ready(projector_cfg_ready),
        .i_cfg_profile_id(projector_profile_id),
        .i_cfg_tensor_id(projector_tensor_id),
        .i_cfg_parameter_base(projector_parameter_base),
        .i_cfg_sample(projector_sample),
        .i_cfg_timestep(projector_timestep),
        .i_cfg_sample_version(projector_state_version),
        .i_cfg_membrane_leak_q8_8(membrane_leak_q),
        .i_cfg_trace_leak_q8_8(trace_leak_q),
        .i_cfg_threshold_q8_8(threshold_q),
        .i_cfg_reset_to_zero(reset_to_zero_q),
        .i_label_valid(i_label_valid), .o_label_ready(o_label_ready),
        .i_label_class(i_label_class),
        .i_label_spike_q8_8(i_label_spike_q8_8),
        .i_label_last(i_label_last),
        .o_read_valid(o_read_valid), .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr),
        .o_read_tag(o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(o_read_response_ready),
        .i_read_response_tag(i_read_response_tag),
        .i_read_response_code(i_read_response_code),
        .i_read_response_error(i_read_response_error),
        .o_observation_valid(projector_observation_valid),
        .i_observation_ready(projector_observation_ready),
        .o_observation_sample(projector_observation_sample),
        .o_observation_timestep(projector_observation_timestep),
        .o_observation_sample_version(projector_observation_version),
        .o_observation_output_index(projector_observation_index),
        .o_observation_channel(projector_observation_channel),
        .o_observation_spatial(projector_observation_spatial),
        .o_observation_current_q8_8(projector_observation_current),
        .o_observation_spike_q8_8(projector_observation_spike),
        .o_observation_membrane_pre_q8_8(projector_observation_membrane_pre),
        .o_observation_membrane_next_q8_8(projector_observation_membrane_next),
        .o_observation_trace_q8_8(projector_observation_trace),
        .o_observation_surrogate_q8_8(projector_observation_surrogate),
        .o_observation_last(projector_observation_last),
        .o_done(projector_done), .o_busy(projector_busy),
        .o_protocol_error(projector_error)
    );

    unified_scnn_tp_target_feature_writer #(
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS),
        .OUTPUT_SPATIAL(OUTPUT_SPATIAL),
        .BANK_DEPTH(BANK_DEPTH), .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) u_feature_writer (
        .i_valid(projector_observation_valid &&
            (!observe_projector_q || i_projector_observation_ready)),
        .o_ready(feature_writer_ready),
        .i_output_channel(projector_observation_channel),
        .i_spatial_index(projector_observation_spatial),
        .i_output_slot(1'b1),
        .i_spike_q8_8(projector_observation_spike),
        .o_write_valid(o_feature_write_valid),
        .i_write_ready(i_feature_write_ready),
        .o_write_slot(o_feature_write_slot),
        .o_write_bank(o_feature_write_bank),
        .o_write_addr(o_feature_write_addr),
        .o_write_data_q8_8(o_feature_write_data_q8_8),
        .o_mapping_error(feature_mapping_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            observe_projector_q <= 1'b0;
            membrane_leak_q <= '0;
            trace_leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
        end else if (start_fire) begin
            observe_projector_q <= i_observe_projector;
            membrane_leak_q <= i_membrane_leak_q8_8;
            trace_leak_q <= i_trace_leak_q8_8;
            threshold_q <= i_threshold_q8_8;
            reset_to_zero_q <= i_reset_to_zero;
        end
    end
endmodule

`default_nettype wire
