`timescale 1ns/1ps
`default_nettype none

// One timestep of retained observations feeds all Conv8 tiles through one TP
// learner and one external RMW service. This block does not implement a model
// version commit: the parent must exclude forward reads until joint commit.
module unified_scnn_tp_conv8_learning_replay_path #(
    parameter bit ALLOW_UNORDERED_SURROGATE_CAPTURE = 1'b0,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int SAMPLE_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int INPUT_INDEX_WIDTH =
        (MAX_INPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_INPUT_CHANNELS*SPATIAL*SPATIAL),
    parameter int TRACE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL),
    parameter int SURROGATE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL),
    parameter int OBS_INPUT_CHANNELS = 512,
    parameter int OBS_OUTPUT_CHANNELS = 512,
    parameter int SURROGATE_WAVE_CHANNELS = 0,
    parameter int SURROGATE_SPATIAL_BLOCK_SIZE = 0,
    parameter int TRACE_WAVE_CHANNELS = 0,
    parameter int MAX_TIMESTEPS = 20,
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int VERSION_WIDTH = 16
) (
    input wire logic i_clk,
    input wire logic i_rst_n,
    input wire logic i_cfg_valid,
    output logic o_cfg_ready,
    input wire logic [PROFILE_ID_WIDTH-1:0] i_cfg_profile_id,
    input wire logic [TENSOR_ID_WIDTH-1:0] i_cfg_tensor_id,
    input wire logic [BATCH_COUNT_WIDTH-1:0] i_cfg_batch_count,
    input wire logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_total_input_channels,
    input wire logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_total_output_channels,
    input wire logic [PARAM_ADDR_WIDTH-1:0] i_cfg_parameter_base,
    input wire logic [Q16_ADDR_WIDTH-1:0] i_cfg_weight_q16_base,
    input wire logic [VOTE_WIDTH-1:0] i_cfg_vote_increment,
    input wire logic [VOTE_WIDTH-1:0] i_cfg_vote_threshold,
    input wire logic signed [CODE_WIDTH-1:0] i_cfg_code_min,
    input wire logic signed [CODE_WIDTH-1:0] i_cfg_code_max,
    output logic o_cfg_error,
    output logic o_active,
    output logic [PROFILE_ID_WIDTH-1:0] o_active_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0] o_active_tensor_id,
    input wire logic i_release,
    output logic o_release_ready,
    input wire logic i_modulation_valid,
    output logic o_modulation_ready,
    input wire logic [SAMPLE_WIDTH-1:0] i_modulation_row,
    input wire logic [SAMPLE_WIDTH-1:0] i_modulation_col,
    input wire logic signed [15:0] i_modulation_q8_8,
    output logic o_capture_complete,
    output logic o_window_done,
    output logic o_window_busy,
    output logic o_gradient_update_accepted,
    output logic o_gradient_zero_skipped,
    output logic o_gradient_valid,
    output logic o_gradient_ready,
    output logic signed [1:0] o_gradient_sign,
    output logic [PARAM_ADDR_WIDTH-1:0] o_gradient_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0] o_gradient_parameter_addr,
    output logic [Q16_ADDR_WIDTH-1:0] o_gradient_weight_q16_addr,
    output logic [Q64_ADDR_WIDTH-1:0] o_gradient_weight_q64_addr,
    output logic [1:0] o_gradient_weight_q64_lane,
    output logic o_gradient_tile_last,
    output logic o_read_valid,
    input wire logic i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0] o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0] o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0] o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0] o_read_tag,
    input wire logic i_read_response_valid,
    output logic o_read_response_ready,
    input wire logic signed [CODE_WIDTH-1:0] i_read_response_code,
    input wire logic signed [RESIDUAL_WIDTH-1:0] i_read_response_residual,
    input wire logic i_read_response_error,
    output logic o_result_valid,
    input wire logic i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0] o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0] o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0] o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0] o_result_tag,
    output logic signed [CODE_WIDTH-1:0] o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0] o_result_residual,
    output logic o_write_code_enable,
    output logic o_write_residual_enable,
    output logic o_write_code_fire,
    output logic o_write_residual_fire,
    output logic o_dirty,
    output logic o_saturated,
    output logic o_protocol_error,
    input wire logic i_input_valid,
    output logic o_input_ready,
    input wire logic i_input_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_input_sample,
    input wire logic [TIMESTEP_WIDTH-1:0] i_input_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_input_sample_version,
    input wire logic [CHANNEL_INDEX_WIDTH-1:0] i_input_channel,
    input wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_input_spatial,
    input wire logic signed [15:0] i_input_q8_8,
    input wire logic i_trace_valid,
    output logic o_trace_ready,
    input wire logic i_trace_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_trace_sample,
    input wire logic [TIMESTEP_WIDTH-1:0] i_trace_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_trace_sample_version,
    input wire logic [CHANNEL_INDEX_WIDTH-1:0] i_trace_channel,
    input wire logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0] i_trace_spatial,
    input wire logic signed [15:0] i_trace_q8_8,
    input wire logic i_surrogate_valid,
    output logic o_surrogate_ready,
    input wire logic i_surrogate_target,
    input wire logic [SAMPLE_WIDTH-1:0] i_surrogate_sample,
    input wire logic [TIMESTEP_WIDTH-1:0] i_surrogate_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_surrogate_sample_version,
    input wire logic [CHANNEL_INDEX_WIDTH-1:0] i_surrogate_channel,
    input wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_surrogate_spatial,
    input wire logic signed [15:0] i_surrogate_q8_8,
    input wire logic [TIMESTEP_WIDTH-1:0] i_cfg_timestep,
    input wire logic [VERSION_WIDTH-1:0] i_cfg_sample_version,
    output logic [12:0] o_tiles_completed
);
    logic st_o_cfg_ready;
    logic st_o_active;
    logic st_o_capture_complete;
    logic st_o_release_ready;
    logic st_o_protocol_error;
    logic st_o_input_ready;
    logic st_o_trace_ready;
    logic st_o_surrogate_ready;
    logic st_o_replay_ready;
    logic st_o_replay_busy;
    logic st_o_replay_done;
    logic st_o_replay_input_valid;
    logic st_o_replay_input_target;
    logic [SAMPLE_WIDTH-1:0] st_o_replay_input_sample;
    logic [TIMESTEP_WIDTH-1:0] st_o_replay_input_timestep;
    logic [VERSION_WIDTH-1:0] st_o_replay_input_sample_version;
    logic [CHANNEL_INDEX_WIDTH-1:0] st_o_replay_input_channel;
    logic [$clog2(SPATIAL*SPATIAL)-1:0] st_o_replay_input_spatial;
    logic signed [15:0] st_o_replay_input_q8_8;
    logic st_o_replay_trace_valid;
    logic st_o_replay_trace_target;
    logic [SAMPLE_WIDTH-1:0] st_o_replay_trace_sample;
    logic [TIMESTEP_WIDTH-1:0] st_o_replay_trace_timestep;
    logic [VERSION_WIDTH-1:0] st_o_replay_trace_sample_version;
    logic [CHANNEL_INDEX_WIDTH-1:0] st_o_replay_trace_channel;
    logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0] st_o_replay_trace_spatial;
    logic signed [15:0] st_o_replay_trace_q8_8;
    logic st_o_replay_surrogate_valid;
    logic st_o_replay_surrogate_target;
    logic [SAMPLE_WIDTH-1:0] st_o_replay_surrogate_sample;
    logic [TIMESTEP_WIDTH-1:0] st_o_replay_surrogate_timestep;
    logic [VERSION_WIDTH-1:0] st_o_replay_surrogate_sample_version;
    logic [CHANNEL_INDEX_WIDTH-1:0] st_o_replay_surrogate_channel;
    logic [$clog2(SPATIAL*SPATIAL)-1:0] st_o_replay_surrogate_spatial;
    logic signed [15:0] st_o_replay_surrogate_q8_8;
    logic ad_o_cfg_ready;
    logic ad_o_active;
    logic ad_o_release_ready;
    logic ad_o_capture_complete;
    logic ad_o_protocol_error;
    logic ad_o_input_ready;
    logic ad_o_trace_ready;
    logic ad_o_surrogate_ready;
    logic ad_o_student_input_valid;
    logic [SAMPLE_WIDTH-1:0] ad_o_student_input_sample;
    logic [INPUT_INDEX_WIDTH-1:0] ad_o_student_input_index;
    logic signed [15:0] ad_o_student_input_q8_8;
    logic ad_o_target_input_valid;
    logic [SAMPLE_WIDTH-1:0] ad_o_target_input_sample;
    logic [INPUT_INDEX_WIDTH-1:0] ad_o_target_input_index;
    logic signed [15:0] ad_o_target_input_q8_8;
    logic ad_o_student_trace_valid;
    logic [SAMPLE_WIDTH-1:0] ad_o_student_trace_sample;
    logic [TRACE_INDEX_WIDTH-1:0] ad_o_student_trace_index;
    logic signed [15:0] ad_o_student_trace_q8_8;
    logic ad_o_target_trace_valid;
    logic [SAMPLE_WIDTH-1:0] ad_o_target_trace_sample;
    logic [TRACE_INDEX_WIDTH-1:0] ad_o_target_trace_index;
    logic signed [15:0] ad_o_target_trace_q8_8;
    logic ad_o_student_surrogate_valid;
    logic [SAMPLE_WIDTH-1:0] ad_o_student_surrogate_sample;
    logic [SURROGATE_INDEX_WIDTH-1:0] ad_o_student_surrogate_index;
    logic signed [15:0] ad_o_student_surrogate_q8_8;
    logic ad_o_target_surrogate_valid;
    logic [SAMPLE_WIDTH-1:0] ad_o_target_surrogate_sample;
    logic [SURROGATE_INDEX_WIDTH-1:0] ad_o_target_surrogate_index;
    logic signed [15:0] ad_o_target_surrogate_q8_8;
    logic lr_o_cfg_ready;
    logic lr_o_cfg_error;
    logic lr_o_active;
    logic [PROFILE_ID_WIDTH-1:0] lr_o_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] lr_o_active_tensor_id;
    logic lr_o_release_ready;
    logic lr_o_student_input_ready;
    logic lr_o_target_input_ready;
    logic lr_o_student_trace_ready;
    logic lr_o_target_trace_ready;
    logic lr_o_student_surrogate_ready;
    logic lr_o_target_surrogate_ready;
    logic lr_o_modulation_ready;
    logic lr_o_capture_complete;
    logic lr_o_start_ready;
    logic lr_o_window_done;
    logic lr_o_window_busy;
    logic lr_o_gradient_update_accepted;
    logic lr_o_gradient_zero_skipped;
    logic lr_o_gradient_valid;
    logic lr_o_gradient_ready;
    logic signed [1:0] lr_o_gradient_sign;
    logic [PARAM_ADDR_WIDTH-1:0] lr_o_gradient_local_addr;
    logic [PARAM_ADDR_WIDTH-1:0] lr_o_gradient_parameter_addr;
    logic [Q16_ADDR_WIDTH-1:0] lr_o_gradient_weight_q16_addr;
    logic [Q64_ADDR_WIDTH-1:0] lr_o_gradient_weight_q64_addr;
    logic [1:0] lr_o_gradient_weight_q64_lane;
    logic lr_o_gradient_tile_last;
    logic lr_o_read_valid;
    logic [PROFILE_ID_WIDTH-1:0] lr_o_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] lr_o_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] lr_o_read_parameter_addr;
    logic [TAG_WIDTH-1:0] lr_o_read_tag;
    logic lr_o_read_response_ready;
    logic lr_o_result_valid;
    logic [PROFILE_ID_WIDTH-1:0] lr_o_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] lr_o_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] lr_o_result_parameter_addr;
    logic [TAG_WIDTH-1:0] lr_o_result_tag;
    logic signed [CODE_WIDTH-1:0] lr_o_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] lr_o_result_residual;
    logic lr_o_write_code_enable;
    logic lr_o_write_residual_enable;
    logic lr_o_write_code_fire;
    logic lr_o_write_residual_fire;
    logic lr_o_dirty;
    logic lr_o_saturated;
    logic lr_o_protocol_error;
    logic [PROFILE_ID_WIDTH-1:0] i_cfg_profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] i_cfg_tensor_id_q;
    logic [BATCH_COUNT_WIDTH-1:0] i_cfg_batch_count_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_total_input_channels_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_total_output_channels_q;
    logic [PARAM_ADDR_WIDTH-1:0] i_cfg_parameter_base_q;
    logic [Q16_ADDR_WIDTH-1:0] i_cfg_weight_q16_base_q;
    logic [VOTE_WIDTH-1:0] i_cfg_vote_increment_q;
    logic [VOTE_WIDTH-1:0] i_cfg_vote_threshold_q;
    logic signed [CODE_WIDTH-1:0] i_cfg_code_min_q;
    logic signed [CODE_WIDTH-1:0] i_cfg_code_max_q;
    logic [TIMESTEP_WIDTH-1:0] i_cfg_timestep_q;
    logic [VERSION_WIDTH-1:0] i_cfg_sample_version_q;
    logic ctl_cfg_ready, ctl_active, ctl_error;
    logic ctl_tile_cfg_valid, ctl_replay_valid, ctl_matrix_valid;
    logic ctl_start, ctl_release, ctl_done;
    logic [8:0] ctl_input_base, ctl_output_base;
    logic [9:0] ctl_input_count, ctl_output_count;
    logic [SAMPLE_WIDTH-1:0] ctl_row, ctl_col;
    logic signed [15:0] ctl_matrix;
    logic cfg_fire, tile_cfg_fire, tile_release_fire, release_fire;
    logic child_error, learner_cfg_error_q;
    // Configuration rejection depends on cfg_valid. Latch it before feeding
    // the sequencer's request inhibit to avoid a combinational error loop.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) learner_cfg_error_q <= 1'b0;
        else if (cfg_fire) learner_cfg_error_q <= 1'b0;
        else if (lr_o_cfg_error) learner_cfg_error_q <= 1'b1;
    end
    assign o_cfg_ready = st_o_cfg_ready && ctl_cfg_ready;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    assign tile_cfg_fire = ctl_tile_cfg_valid && lr_o_cfg_ready && ad_o_cfg_ready;
    assign tile_release_fire = ctl_release && lr_o_release_ready && ad_o_release_ready;
    assign o_release_ready = ctl_done && st_o_release_ready && !child_error && !ctl_error;
    assign release_fire = i_release && o_release_ready;
    assign child_error = st_o_protocol_error || ad_o_protocol_error ||
        lr_o_protocol_error || learner_cfg_error_q;
    assign o_protocol_error = child_error || ctl_error;
    assign o_cfg_error = st_o_protocol_error || ctl_error || lr_o_cfg_error;
    assign o_active = ctl_active || st_o_active;
    assign o_window_busy = ctl_active && !ctl_done;
    assign o_window_done = ctl_done;
    assign o_capture_complete = st_o_capture_complete;
    assign o_active_profile_id = i_cfg_profile_id_q;
    assign o_active_tensor_id = i_cfg_tensor_id_q;
    assign o_gradient_update_accepted = lr_o_gradient_update_accepted;
    assign o_gradient_zero_skipped = lr_o_gradient_zero_skipped;
    assign o_gradient_valid = lr_o_gradient_valid;
    assign o_gradient_ready = lr_o_gradient_ready;
    assign o_gradient_sign = lr_o_gradient_sign;
    assign o_gradient_local_addr = lr_o_gradient_local_addr;
    assign o_gradient_parameter_addr = lr_o_gradient_parameter_addr;
    assign o_gradient_weight_q16_addr = lr_o_gradient_weight_q16_addr;
    assign o_gradient_weight_q64_addr = lr_o_gradient_weight_q64_addr;
    assign o_gradient_weight_q64_lane = lr_o_gradient_weight_q64_lane;
    assign o_gradient_tile_last = lr_o_gradient_tile_last;
    assign o_read_valid = lr_o_read_valid;
    assign o_read_profile_id = lr_o_read_profile_id;
    assign o_read_tensor_id = lr_o_read_tensor_id;
    assign o_read_parameter_addr = lr_o_read_parameter_addr;
    assign o_read_tag = lr_o_read_tag;
    assign o_read_response_ready = lr_o_read_response_ready;
    assign o_result_valid = lr_o_result_valid;
    assign o_result_profile_id = lr_o_result_profile_id;
    assign o_result_tensor_id = lr_o_result_tensor_id;
    assign o_result_parameter_addr = lr_o_result_parameter_addr;
    assign o_result_tag = lr_o_result_tag;
    assign o_result_code = lr_o_result_code;
    assign o_result_residual = lr_o_result_residual;
    assign o_write_code_enable = lr_o_write_code_enable;
    assign o_write_residual_enable = lr_o_write_residual_enable;
    assign o_write_code_fire = lr_o_write_code_fire;
    assign o_write_residual_fire = lr_o_write_residual_fire;
    assign o_dirty = lr_o_dirty;
    assign o_saturated = lr_o_saturated;
    assign o_input_ready = st_o_input_ready;
    assign o_trace_ready = st_o_trace_ready;
    assign o_surrogate_ready = st_o_surrogate_ready;
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            i_cfg_profile_id_q <= '0;
            i_cfg_tensor_id_q <= '0;
            i_cfg_batch_count_q <= '0;
            i_cfg_total_input_channels_q <= '0;
            i_cfg_total_output_channels_q <= '0;
            i_cfg_parameter_base_q <= '0;
            i_cfg_weight_q16_base_q <= '0;
            i_cfg_vote_increment_q <= '0;
            i_cfg_vote_threshold_q <= '0;
            i_cfg_code_min_q <= '0;
            i_cfg_code_max_q <= '0;
            i_cfg_timestep_q <= '0;
            i_cfg_sample_version_q <= '0;
        end else if (cfg_fire) begin
            i_cfg_profile_id_q <= i_cfg_profile_id;
            i_cfg_tensor_id_q <= i_cfg_tensor_id;
            i_cfg_batch_count_q <= i_cfg_batch_count;
            i_cfg_total_input_channels_q <= i_cfg_total_input_channels;
            i_cfg_total_output_channels_q <= i_cfg_total_output_channels;
            i_cfg_parameter_base_q <= i_cfg_parameter_base;
            i_cfg_weight_q16_base_q <= i_cfg_weight_q16_base;
            i_cfg_vote_increment_q <= i_cfg_vote_increment;
            i_cfg_vote_threshold_q <= i_cfg_vote_threshold;
            i_cfg_code_min_q <= i_cfg_code_min;
            i_cfg_code_max_q <= i_cfg_code_max;
            i_cfg_timestep_q <= i_cfg_timestep;
            i_cfg_sample_version_q <= i_cfg_sample_version;
        end
    end
    unified_scnn_tp_conv8_tile_replay_controller #(
        .MAX_BATCH(MAX_BATCH), .TILE_INPUTS(MAX_INPUT_CHANNELS),
        .TILE_OUTPUTS(MAX_OUTPUT_CHANNELS), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH)
    ) u_controller (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_cfg_valid(cfg_fire),
        .o_cfg_ready(ctl_cfg_ready), .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_input_channels(i_cfg_total_input_channels),
        .i_cfg_output_channels(i_cfg_total_output_channels),
        .i_modulation_valid(i_modulation_valid), .o_modulation_ready(o_modulation_ready),
        .i_modulation_row(i_modulation_row), .i_modulation_col(i_modulation_col),
        .i_modulation_q8_8(i_modulation_q8_8),
        .i_store_capture_complete(st_o_capture_complete), .i_child_error(child_error),
        .o_tile_cfg_valid(ctl_tile_cfg_valid),
        .i_tile_cfg_ready(lr_o_cfg_ready && ad_o_cfg_ready),
        .o_input_channel_base(ctl_input_base), .o_output_channel_base(ctl_output_base),
        .o_input_channel_count(ctl_input_count), .o_output_channel_count(ctl_output_count),
        .o_replay_valid(ctl_replay_valid), .i_replay_ready(st_o_replay_ready),
        .i_replay_done(st_o_replay_done),
        .o_modulation_valid(ctl_matrix_valid), .i_modulation_ready(lr_o_modulation_ready),
        .o_modulation_row(ctl_row), .o_modulation_col(ctl_col),
        .o_modulation_q8_8(ctl_matrix),
        .i_tile_capture_complete(lr_o_capture_complete && ad_o_capture_complete),
        .o_tile_start(ctl_start), .i_tile_start_ready(lr_o_start_ready),
        .i_tile_done(lr_o_window_done), .o_tile_release(ctl_release),
        .i_tile_release_ready(lr_o_release_ready && ad_o_release_ready),
        .o_window_done(ctl_done), .i_window_done_ready(release_fire),
        .o_active(ctl_active), .o_protocol_error(ctl_error),
        .o_tiles_completed(o_tiles_completed)
    );
    unified_scnn_tp_conv8_observation_store #(
        .ALLOW_UNORDERED_SURROGATE_CAPTURE(
            ALLOW_UNORDERED_SURROGATE_CAPTURE),
        .SURROGATE_WAVE_CHANNELS(SURROGATE_WAVE_CHANNELS),
        .SURROGATE_SPATIAL_BLOCK_SIZE(SURROGATE_SPATIAL_BLOCK_SIZE),
        .TRACE_WAVE_CHANNELS(TRACE_WAVE_CHANNELS),
        .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .VERSION_WIDTH(VERSION_WIDTH),
        .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .MAX_INPUT_CHANNELS(OBS_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(OBS_OUTPUT_CHANNELS)
    ) u_observations (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire),
        .o_cfg_ready(st_o_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_timestep(i_cfg_timestep),
        .i_cfg_sample_version(i_cfg_sample_version),
        .i_cfg_input_channels(i_cfg_total_input_channels),
        .i_cfg_output_channels(i_cfg_total_output_channels),
        .o_active(st_o_active),
        .o_capture_complete(st_o_capture_complete),
        .i_release(release_fire),
        .o_release_ready(st_o_release_ready),
        .o_protocol_error(st_o_protocol_error),
        .i_input_valid(i_input_valid),
        .o_input_ready(st_o_input_ready),
        .i_input_target(i_input_target),
        .i_input_sample(i_input_sample),
        .i_input_timestep(i_input_timestep),
        .i_input_sample_version(i_input_sample_version),
        .i_input_channel(i_input_channel),
        .i_input_spatial(i_input_spatial),
        .i_input_q8_8(i_input_q8_8),
        .i_trace_valid(i_trace_valid),
        .o_trace_ready(st_o_trace_ready),
        .i_trace_target(i_trace_target),
        .i_trace_sample(i_trace_sample),
        .i_trace_timestep(i_trace_timestep),
        .i_trace_sample_version(i_trace_sample_version),
        .i_trace_channel(i_trace_channel),
        .i_trace_spatial(i_trace_spatial),
        .i_trace_q8_8(i_trace_q8_8),
        .i_surrogate_valid(i_surrogate_valid),
        .o_surrogate_ready(st_o_surrogate_ready),
        .i_surrogate_target(i_surrogate_target),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_timestep(i_surrogate_timestep),
        .i_surrogate_sample_version(i_surrogate_sample_version),
        .i_surrogate_channel(i_surrogate_channel),
        .i_surrogate_spatial(i_surrogate_spatial),
        .i_surrogate_q8_8(i_surrogate_q8_8),
        .i_replay_valid(ctl_replay_valid),
        .o_replay_ready(st_o_replay_ready),
        .i_replay_input_channel_base(ctl_input_base),
        .i_replay_input_channel_count(ctl_input_count),
        .i_replay_output_channel_base(ctl_output_base),
        .i_replay_output_channel_count(ctl_output_count),
        .o_replay_busy(st_o_replay_busy),
        .o_replay_done(st_o_replay_done),
        .o_replay_input_valid(st_o_replay_input_valid),
        .i_replay_input_ready(ad_o_input_ready),
        .o_replay_input_target(st_o_replay_input_target),
        .o_replay_input_sample(st_o_replay_input_sample),
        .o_replay_input_timestep(st_o_replay_input_timestep),
        .o_replay_input_sample_version(st_o_replay_input_sample_version),
        .o_replay_input_channel(st_o_replay_input_channel),
        .o_replay_input_spatial(st_o_replay_input_spatial),
        .o_replay_input_q8_8(st_o_replay_input_q8_8),
        .o_replay_trace_valid(st_o_replay_trace_valid),
        .i_replay_trace_ready(ad_o_trace_ready),
        .o_replay_trace_target(st_o_replay_trace_target),
        .o_replay_trace_sample(st_o_replay_trace_sample),
        .o_replay_trace_timestep(st_o_replay_trace_timestep),
        .o_replay_trace_sample_version(st_o_replay_trace_sample_version),
        .o_replay_trace_channel(st_o_replay_trace_channel),
        .o_replay_trace_spatial(st_o_replay_trace_spatial),
        .o_replay_trace_q8_8(st_o_replay_trace_q8_8),
        .o_replay_surrogate_valid(st_o_replay_surrogate_valid),
        .i_replay_surrogate_ready(ad_o_surrogate_ready),
        .o_replay_surrogate_target(st_o_replay_surrogate_target),
        .o_replay_surrogate_sample(st_o_replay_surrogate_sample),
        .o_replay_surrogate_timestep(st_o_replay_surrogate_timestep),
        .o_replay_surrogate_sample_version(st_o_replay_surrogate_sample_version),
        .o_replay_surrogate_channel(st_o_replay_surrogate_channel),
        .o_replay_surrogate_spatial(st_o_replay_surrogate_spatial),
        .o_replay_surrogate_q8_8(st_o_replay_surrogate_q8_8)
    );
    unified_scnn_tp_conv_tile_observation_adapter #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .VERSION_WIDTH(VERSION_WIDTH),
        .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .INPUT_INDEX_WIDTH(INPUT_INDEX_WIDTH),
        .TRACE_INDEX_WIDTH(TRACE_INDEX_WIDTH),
        .SURROGATE_INDEX_WIDTH(SURROGATE_INDEX_WIDTH)
    ) u_coordinates (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(tile_cfg_fire),
        .o_cfg_ready(ad_o_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count_q),
        .i_cfg_timestep(i_cfg_timestep_q),
        .i_cfg_sample_version(i_cfg_sample_version_q),
        .i_cfg_input_channel_base(ctl_input_base),
        .i_cfg_input_channel_count(ctl_input_count),
        .i_cfg_output_channel_base(ctl_output_base),
        .i_cfg_output_channel_count(ctl_output_count),
        .o_active(ad_o_active),
        .i_release(tile_release_fire),
        .o_release_ready(ad_o_release_ready),
        .o_capture_complete(ad_o_capture_complete),
        .o_protocol_error(ad_o_protocol_error),
        .i_input_valid(st_o_replay_input_valid),
        .o_input_ready(ad_o_input_ready),
        .i_input_target(st_o_replay_input_target),
        .i_input_sample(st_o_replay_input_sample),
        .i_input_timestep(st_o_replay_input_timestep),
        .i_input_sample_version(st_o_replay_input_sample_version),
        .i_input_channel(st_o_replay_input_channel),
        .i_input_spatial(st_o_replay_input_spatial),
        .i_input_q8_8(st_o_replay_input_q8_8),
        .i_trace_valid(st_o_replay_trace_valid),
        .o_trace_ready(ad_o_trace_ready),
        .i_trace_target(st_o_replay_trace_target),
        .i_trace_sample(st_o_replay_trace_sample),
        .i_trace_timestep(st_o_replay_trace_timestep),
        .i_trace_sample_version(st_o_replay_trace_sample_version),
        .i_trace_channel(st_o_replay_trace_channel),
        .i_trace_spatial(st_o_replay_trace_spatial),
        .i_trace_q8_8(st_o_replay_trace_q8_8),
        .i_surrogate_valid(st_o_replay_surrogate_valid),
        .o_surrogate_ready(ad_o_surrogate_ready),
        .i_surrogate_target(st_o_replay_surrogate_target),
        .i_surrogate_sample(st_o_replay_surrogate_sample),
        .i_surrogate_timestep(st_o_replay_surrogate_timestep),
        .i_surrogate_sample_version(st_o_replay_surrogate_sample_version),
        .i_surrogate_channel(st_o_replay_surrogate_channel),
        .i_surrogate_spatial(st_o_replay_surrogate_spatial),
        .i_surrogate_q8_8(st_o_replay_surrogate_q8_8),
        .o_student_input_valid(ad_o_student_input_valid),
        .i_student_input_ready(lr_o_student_input_ready),
        .o_student_input_sample(ad_o_student_input_sample),
        .o_student_input_index(ad_o_student_input_index),
        .o_student_input_q8_8(ad_o_student_input_q8_8),
        .o_target_input_valid(ad_o_target_input_valid),
        .i_target_input_ready(lr_o_target_input_ready),
        .o_target_input_sample(ad_o_target_input_sample),
        .o_target_input_index(ad_o_target_input_index),
        .o_target_input_q8_8(ad_o_target_input_q8_8),
        .o_student_trace_valid(ad_o_student_trace_valid),
        .i_student_trace_ready(lr_o_student_trace_ready),
        .o_student_trace_sample(ad_o_student_trace_sample),
        .o_student_trace_index(ad_o_student_trace_index),
        .o_student_trace_q8_8(ad_o_student_trace_q8_8),
        .o_target_trace_valid(ad_o_target_trace_valid),
        .i_target_trace_ready(lr_o_target_trace_ready),
        .o_target_trace_sample(ad_o_target_trace_sample),
        .o_target_trace_index(ad_o_target_trace_index),
        .o_target_trace_q8_8(ad_o_target_trace_q8_8),
        .o_student_surrogate_valid(ad_o_student_surrogate_valid),
        .i_student_surrogate_ready(lr_o_student_surrogate_ready),
        .o_student_surrogate_sample(ad_o_student_surrogate_sample),
        .o_student_surrogate_index(ad_o_student_surrogate_index),
        .o_student_surrogate_q8_8(ad_o_student_surrogate_q8_8),
        .o_target_surrogate_valid(ad_o_target_surrogate_valid),
        .i_target_surrogate_ready(lr_o_target_surrogate_ready),
        .o_target_surrogate_sample(ad_o_target_surrogate_sample),
        .o_target_surrogate_index(ad_o_target_surrogate_index),
        .o_target_surrogate_q8_8(ad_o_target_surrogate_q8_8)
    );
    unified_scnn_tp_conv_tile_online_learning_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .SPATIAL(SPATIAL),
        .POOLED_SPATIAL(POOLED_SPATIAL),
        .KERNEL_SIZE(KERNEL_SIZE),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .INPUT_INDEX_WIDTH(INPUT_INDEX_WIDTH),
        .TRACE_INDEX_WIDTH(TRACE_INDEX_WIDTH),
        .SURROGATE_INDEX_WIDTH(SURROGATE_INDEX_WIDTH)
    ) u_learner (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(tile_cfg_fire),
        .o_cfg_ready(lr_o_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id_q),
        .i_cfg_tensor_id(i_cfg_tensor_id_q),
        .i_cfg_batch_count(i_cfg_batch_count_q),
        .i_cfg_total_input_channels(i_cfg_total_input_channels_q),
        .i_cfg_total_output_channels(i_cfg_total_output_channels_q),
        .i_cfg_input_channel_count(ctl_input_count),
        .i_cfg_output_channel_count(ctl_output_count),
        .i_cfg_input_channel_base(ctl_input_base),
        .i_cfg_output_channel_base(ctl_output_base),
        .i_cfg_parameter_base(i_cfg_parameter_base_q),
        .i_cfg_weight_q16_base(i_cfg_weight_q16_base_q),
        .i_cfg_vote_increment(i_cfg_vote_increment_q),
        .i_cfg_vote_threshold(i_cfg_vote_threshold_q),
        .i_cfg_code_min(i_cfg_code_min_q),
        .i_cfg_code_max(i_cfg_code_max_q),
        .o_cfg_error(lr_o_cfg_error),
        .o_active(lr_o_active),
        .o_active_profile_id(lr_o_active_profile_id),
        .o_active_tensor_id(lr_o_active_tensor_id),
        .i_release(tile_release_fire),
        .o_release_ready(lr_o_release_ready),
        .i_student_input_valid(ad_o_student_input_valid),
        .o_student_input_ready(lr_o_student_input_ready),
        .i_student_input_sample(ad_o_student_input_sample),
        .i_student_input_index(ad_o_student_input_index),
        .i_student_input_q8_8(ad_o_student_input_q8_8),
        .i_target_input_valid(ad_o_target_input_valid),
        .o_target_input_ready(lr_o_target_input_ready),
        .i_target_input_sample(ad_o_target_input_sample),
        .i_target_input_index(ad_o_target_input_index),
        .i_target_input_q8_8(ad_o_target_input_q8_8),
        .i_student_trace_valid(ad_o_student_trace_valid),
        .o_student_trace_ready(lr_o_student_trace_ready),
        .i_student_trace_sample(ad_o_student_trace_sample),
        .i_student_trace_index(ad_o_student_trace_index),
        .i_student_trace_q8_8(ad_o_student_trace_q8_8),
        .i_target_trace_valid(ad_o_target_trace_valid),
        .o_target_trace_ready(lr_o_target_trace_ready),
        .i_target_trace_sample(ad_o_target_trace_sample),
        .i_target_trace_index(ad_o_target_trace_index),
        .i_target_trace_q8_8(ad_o_target_trace_q8_8),
        .i_student_surrogate_valid(ad_o_student_surrogate_valid),
        .o_student_surrogate_ready(lr_o_student_surrogate_ready),
        .i_student_surrogate_sample(ad_o_student_surrogate_sample),
        .i_student_surrogate_index(ad_o_student_surrogate_index),
        .i_student_surrogate_q8_8(ad_o_student_surrogate_q8_8),
        .i_target_surrogate_valid(ad_o_target_surrogate_valid),
        .o_target_surrogate_ready(lr_o_target_surrogate_ready),
        .i_target_surrogate_sample(ad_o_target_surrogate_sample),
        .i_target_surrogate_index(ad_o_target_surrogate_index),
        .i_target_surrogate_q8_8(ad_o_target_surrogate_q8_8),
        .i_modulation_valid(ctl_matrix_valid),
        .o_modulation_ready(lr_o_modulation_ready),
        .i_modulation_row(ctl_row),
        .i_modulation_col(ctl_col),
        .i_modulation_q8_8(ctl_matrix),
        .o_capture_complete(lr_o_capture_complete),
        .i_start(ctl_start),
        .o_start_ready(lr_o_start_ready),
        .o_window_done(lr_o_window_done),
        .o_window_busy(lr_o_window_busy),
        .o_gradient_update_accepted(lr_o_gradient_update_accepted),
        .o_gradient_zero_skipped(lr_o_gradient_zero_skipped),
        .o_gradient_valid(lr_o_gradient_valid),
        .o_gradient_ready(lr_o_gradient_ready),
        .o_gradient_sign(lr_o_gradient_sign),
        .o_gradient_local_addr(lr_o_gradient_local_addr),
        .o_gradient_parameter_addr(lr_o_gradient_parameter_addr),
        .o_gradient_weight_q16_addr(lr_o_gradient_weight_q16_addr),
        .o_gradient_weight_q64_addr(lr_o_gradient_weight_q64_addr),
        .o_gradient_weight_q64_lane(lr_o_gradient_weight_q64_lane),
        .o_gradient_tile_last(lr_o_gradient_tile_last),
        .o_read_valid(lr_o_read_valid),
        .i_read_ready(i_read_ready),
        .o_read_profile_id(lr_o_read_profile_id),
        .o_read_tensor_id(lr_o_read_tensor_id),
        .o_read_parameter_addr(lr_o_read_parameter_addr),
        .o_read_tag(lr_o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(lr_o_read_response_ready),
        .i_read_response_code(i_read_response_code),
        .i_read_response_residual(i_read_response_residual),
        .i_read_response_error(i_read_response_error),
        .o_result_valid(lr_o_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_profile_id(lr_o_result_profile_id),
        .o_result_tensor_id(lr_o_result_tensor_id),
        .o_result_parameter_addr(lr_o_result_parameter_addr),
        .o_result_tag(lr_o_result_tag),
        .o_result_code(lr_o_result_code),
        .o_result_residual(lr_o_result_residual),
        .o_write_code_enable(lr_o_write_code_enable),
        .o_write_residual_enable(lr_o_write_residual_enable),
        .o_write_code_fire(lr_o_write_code_fire),
        .o_write_residual_fire(lr_o_write_residual_fire),
        .o_dirty(lr_o_dirty),
        .o_saturated(lr_o_saturated),
        .o_protocol_error(lr_o_protocol_error)
    );
endmodule
`default_nettype wire
