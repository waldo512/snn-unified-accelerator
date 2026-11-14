`timescale 1ns/1ps
`default_nettype none

// SCNN-specific Conv8 observation/gradient adapter backed by the common
// direct-residual RMW service. One bounded tile is captured and replayed at a
// time; completed gradient signs are streamed directly into the shared store.
module unified_scnn_tp_conv_tile_online_learning_path #(
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
        $clog2(MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL)
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]         i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]          i_cfg_tensor_id,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]        i_cfg_batch_count,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_total_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_total_output_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_input_channel_count,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_output_channel_count,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_cfg_input_channel_base,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_cfg_output_channel_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_cfg_parameter_base,
    input  wire logic [Q16_ADDR_WIDTH-1:0]           i_cfg_weight_q16_base,
    input  wire logic [VOTE_WIDTH-1:0]               i_cfg_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]               i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]        i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]        i_cfg_code_max,
    output logic                                o_cfg_error,
    output logic                                o_active,
    output logic [PROFILE_ID_WIDTH-1:0]         o_active_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_active_tensor_id,
    input  wire logic                                i_release,
    output logic                                o_release_ready,

    input  wire logic                                i_student_input_valid,
    output logic                                o_student_input_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_student_input_sample,
    input  wire logic [INPUT_INDEX_WIDTH-1:0]        i_student_input_index,
    input  wire logic signed [15:0]                  i_student_input_q8_8,
    input  wire logic                                i_target_input_valid,
    output logic                                o_target_input_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_target_input_sample,
    input  wire logic [INPUT_INDEX_WIDTH-1:0]        i_target_input_index,
    input  wire logic signed [15:0]                  i_target_input_q8_8,
    input  wire logic                                i_student_trace_valid,
    output logic                                o_student_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_student_trace_sample,
    input  wire logic [TRACE_INDEX_WIDTH-1:0]        i_student_trace_index,
    input  wire logic signed [15:0]                  i_student_trace_q8_8,
    input  wire logic                                i_target_trace_valid,
    output logic                                o_target_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_target_trace_sample,
    input  wire logic [TRACE_INDEX_WIDTH-1:0]        i_target_trace_index,
    input  wire logic signed [15:0]                  i_target_trace_q8_8,
    input  wire logic                                i_student_surrogate_valid,
    output logic                                o_student_surrogate_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_student_surrogate_sample,
    input  wire logic [SURROGATE_INDEX_WIDTH-1:0]    i_student_surrogate_index,
    input  wire logic signed [15:0]                  i_student_surrogate_q8_8,
    input  wire logic                                i_target_surrogate_valid,
    output logic                                o_target_surrogate_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_target_surrogate_sample,
    input  wire logic [SURROGATE_INDEX_WIDTH-1:0]    i_target_surrogate_index,
    input  wire logic signed [15:0]                  i_target_surrogate_q8_8,
    input  wire logic                                i_modulation_valid,
    output logic                                o_modulation_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_modulation_row,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_modulation_col,
    input  wire logic signed [15:0]                  i_modulation_q8_8,
    output logic                                o_capture_complete,
    input  wire logic                                i_start,
    output logic                                o_start_ready,

    output logic                                o_window_done,
    output logic                                o_window_busy,
    output logic                                o_gradient_update_accepted,
    output logic                                o_gradient_zero_skipped,
    output logic                                o_gradient_valid,
    output logic                                o_gradient_ready,
    output logic signed [1:0]                   o_gradient_sign,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_gradient_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_gradient_parameter_addr,
    output logic [Q16_ADDR_WIDTH-1:0]           o_gradient_weight_q16_addr,
    output logic [Q64_ADDR_WIDTH-1:0]           o_gradient_weight_q64_addr,
    output logic [1:0]                          o_gradient_weight_q64_lane,
    output logic                                o_gradient_tile_last,

    output logic                                o_read_valid,
    input  wire logic                                i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_read_tag,
    input  wire logic                                i_read_response_valid,
    output logic                                o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]        i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]    i_read_response_residual,
    input  wire logic                                i_read_response_error,
    output logic                                o_result_valid,
    input  wire logic                                i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]         o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]          o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_result_tag,
    output logic signed [CODE_WIDTH-1:0]        o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_result_residual,
    output logic                                o_write_code_enable,
    output logic                                o_write_residual_enable,
    output logic                                o_write_code_fire,
    output logic                                o_write_residual_fire,
    output logic                                o_dirty,
    output logic                                o_saturated,
    output logic                                o_protocol_error
);
    localparam logic [PARAM_ADDR_WIDTH:0] PARAM_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    logic tile_cfg_ready;
    logic tile_active;
    logic tile_release_ready;
    logic tile_start_ready;
    logic tile_busy;
    logic tile_done;
    logic tile_protocol_error;
    logic tile_gradient_valid;
    logic tile_gradient_ready;
    logic signed [1:0] tile_gradient_sign;
    logic rmw_cfg_ready;
    logic rmw_cfg_error;
    logic rmw_active;
    logic rmw_release_ready;
    logic rmw_busy;
    logic gradient_tensor_error;
    logic gradient_sign_error;
    logic update_config_error;
    logic request_error;
    logic store_error;
    logic internal_error;
    logic completion_pending_q;
    logic cfg_legal;
    logic cfg_fire;
    logic release_fire;
    logic start_fire;
    logic [63:0] element_count_ext;
    logic [PARAM_ADDR_WIDTH:0] end_address_ext;
    logic [PARAM_ADDR_WIDTH-1:0] element_count;
    logic [TAG_WIDTH-1:0] gradient_tag;

    always_comb begin
        element_count_ext = 64'(i_cfg_total_input_channels) *
            64'(i_cfg_total_output_channels) * 64'(KERNEL_SIZE*KERNEL_SIZE);
        element_count = PARAM_ADDR_WIDTH'(element_count_ext);
        end_address_ext = {1'b0, i_cfg_parameter_base} +
            {1'b0, element_count};
        cfg_legal = i_cfg_batch_count != 0 &&
            i_cfg_batch_count <= BATCH_COUNT_WIDTH'(MAX_BATCH) &&
            i_cfg_input_channel_count != 0 &&
            i_cfg_input_channel_count <=
                CHANNEL_COUNT_WIDTH'(MAX_INPUT_CHANNELS) &&
            i_cfg_output_channel_count != 0 &&
            i_cfg_output_channel_count <=
                CHANNEL_COUNT_WIDTH'(MAX_OUTPUT_CHANNELS) &&
            i_cfg_total_input_channels != 0 &&
            int'(i_cfg_total_input_channels) <= 512 &&
            i_cfg_total_output_channels != 0 &&
            int'(i_cfg_total_output_channels) <= 512 &&
            int'(i_cfg_input_channel_base) +
                int'(i_cfg_input_channel_count) <=
                    int'(i_cfg_total_input_channels) &&
            int'(i_cfg_output_channel_base) +
                int'(i_cfg_output_channel_count) <=
                    int'(i_cfg_total_output_channels) &&
            element_count_ext != 0 &&
            element_count_ext < (64'd1 << PARAM_ADDR_WIDTH) &&
            end_address_ext <= PARAM_LIMIT &&
            i_cfg_vote_threshold != 0 &&
            {1'b0, i_cfg_vote_increment} <
                ({1'b0, i_cfg_vote_threshold} +
                 {1'b0, i_cfg_vote_threshold}) &&
            $signed(i_cfg_code_min) <= $signed(i_cfg_code_max);

        o_cfg_ready = tile_cfg_ready && rmw_cfg_ready;
        cfg_fire = i_cfg_valid && o_cfg_ready && cfg_legal;
        o_cfg_error = (i_cfg_valid && o_cfg_ready && !cfg_legal) ||
            rmw_cfg_error;
        o_active = tile_active || rmw_active;
        o_release_ready = tile_release_ready && rmw_release_ready &&
            !completion_pending_q;
        release_fire = i_release && o_release_ready;
        o_start_ready = tile_start_ready && !rmw_busy &&
            !completion_pending_q;
        start_fire = i_start && o_start_ready;
        o_window_busy = tile_busy || rmw_busy || completion_pending_q;
        o_protocol_error = tile_protocol_error || gradient_tensor_error ||
            gradient_sign_error || update_config_error || request_error ||
            store_error || internal_error ||
            (tile_active != rmw_active);
    end

    generate
        if (TAG_WIDTH >= PARAM_ADDR_WIDTH) begin : g_wide_tag
            assign gradient_tag =
                {{(TAG_WIDTH-PARAM_ADDR_WIDTH){1'b0}}, o_gradient_local_addr};
        end else begin : g_narrow_tag
            assign gradient_tag = o_gradient_local_addr[TAG_WIDTH-1:0];
        end
    endgenerate

    assign o_gradient_valid = tile_gradient_valid;
    assign o_gradient_ready = tile_gradient_ready;
    assign o_gradient_sign = tile_gradient_sign;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            completion_pending_q <= 1'b0;
            o_window_done <= 1'b0;
        end else begin
            o_window_done <= 1'b0;
            if (tile_done)
                completion_pending_q <= 1'b1;
            if (completion_pending_q && !rmw_busy) begin
                completion_pending_q <= 1'b0;
                o_window_done <= 1'b1;
            end
            if (release_fire)
                completion_pending_q <= 1'b0;
        end
    end

    unified_scnn_tp_conv_tile_sign_engine #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_INPUT_CHANNELS(MAX_INPUT_CHANNELS),
        .MAX_OUTPUT_CHANNELS(MAX_OUTPUT_CHANNELS),
        .SPATIAL(SPATIAL), .POOLED_SPATIAL(POOLED_SPATIAL),
        .KERNEL_SIZE(KERNEL_SIZE),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .INPUT_INDEX_WIDTH(INPUT_INDEX_WIDTH),
        .TRACE_INDEX_WIDTH(TRACE_INDEX_WIDTH),
        .SURROGATE_INDEX_WIDTH(SURROGATE_INDEX_WIDTH)
    ) u_tile_engine (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(tile_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_total_input_channels(i_cfg_total_input_channels),
        .i_cfg_total_output_channels(i_cfg_total_output_channels),
        .i_cfg_input_channel_count(i_cfg_input_channel_count),
        .i_cfg_output_channel_count(i_cfg_output_channel_count),
        .i_cfg_input_channel_base(i_cfg_input_channel_base),
        .i_cfg_output_channel_base(i_cfg_output_channel_base),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_weight_q16_base(i_cfg_weight_q16_base),
        .o_active(tile_active), .i_release(release_fire),
        .o_release_ready(tile_release_ready),
        .i_student_input_valid(i_student_input_valid),
        .o_student_input_ready(o_student_input_ready),
        .i_student_input_sample(i_student_input_sample),
        .i_student_input_index(i_student_input_index),
        .i_student_input_q8_8(i_student_input_q8_8),
        .i_target_input_valid(i_target_input_valid),
        .o_target_input_ready(o_target_input_ready),
        .i_target_input_sample(i_target_input_sample),
        .i_target_input_index(i_target_input_index),
        .i_target_input_q8_8(i_target_input_q8_8),
        .i_student_trace_valid(i_student_trace_valid),
        .o_student_trace_ready(o_student_trace_ready),
        .i_student_trace_sample(i_student_trace_sample),
        .i_student_trace_index(i_student_trace_index),
        .i_student_trace_q8_8(i_student_trace_q8_8),
        .i_target_trace_valid(i_target_trace_valid),
        .o_target_trace_ready(o_target_trace_ready),
        .i_target_trace_sample(i_target_trace_sample),
        .i_target_trace_index(i_target_trace_index),
        .i_target_trace_q8_8(i_target_trace_q8_8),
        .i_student_surrogate_valid(i_student_surrogate_valid),
        .o_student_surrogate_ready(o_student_surrogate_ready),
        .i_student_surrogate_sample(i_student_surrogate_sample),
        .i_student_surrogate_index(i_student_surrogate_index),
        .i_student_surrogate_q8_8(i_student_surrogate_q8_8),
        .i_target_surrogate_valid(i_target_surrogate_valid),
        .o_target_surrogate_ready(o_target_surrogate_ready),
        .i_target_surrogate_sample(i_target_surrogate_sample),
        .i_target_surrogate_index(i_target_surrogate_index),
        .i_target_surrogate_q8_8(i_target_surrogate_q8_8),
        .i_modulation_valid(i_modulation_valid),
        .o_modulation_ready(o_modulation_ready),
        .i_modulation_row(i_modulation_row),
        .i_modulation_col(i_modulation_col),
        .i_modulation_q8_8(i_modulation_q8_8),
        .o_capture_complete(o_capture_complete),
        .i_start(start_fire), .o_start_ready(tile_start_ready),
        .o_gradient_valid(tile_gradient_valid),
        .i_gradient_ready(tile_gradient_ready),
        .o_gradient_sign(tile_gradient_sign),
        .o_gradient_local_addr(o_gradient_local_addr),
        .o_gradient_parameter_addr(o_gradient_parameter_addr),
        .o_gradient_weight_q16_addr(o_gradient_weight_q16_addr),
        .o_gradient_weight_q64_addr(o_gradient_weight_q64_addr),
        .o_gradient_weight_q64_lane(o_gradient_weight_q64_lane),
        .o_gradient_tile_last(o_gradient_tile_last),
        .o_done(tile_done), .o_busy(tile_busy),
        .o_protocol_error(tile_protocol_error)
    );

    unified_fc_tp_gradient_rmw_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH), .VOTE_WIDTH(VOTE_WIDTH)
    ) u_shared_rmw (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(rmw_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_element_count(element_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min), .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(rmw_cfg_error), .i_release(release_fire),
        .o_release_ready(rmw_release_ready), .o_active(rmw_active),
        .o_active_profile_id(o_active_profile_id),
        .o_active_tensor_id(o_active_tensor_id),
        .i_gradient_valid(tile_gradient_valid),
        .o_gradient_ready(tile_gradient_ready),
        .i_gradient_tensor_id(o_active_tensor_id),
        .i_gradient_local_parameter_addr(o_gradient_local_addr),
        .i_gradient_tag(gradient_tag),
        .i_gradient_sign(tile_gradient_sign),
        .i_gradient_accumulator('0),
        .i_magnitude_mode(1'b0),
        .i_magnitude_multiplier('0),
        .o_gradient_update_accepted(o_gradient_update_accepted),
        .o_gradient_zero_skipped(o_gradient_zero_skipped),
        .o_gradient_tensor_error(gradient_tensor_error),
        .o_gradient_sign_error(gradient_sign_error),
        .o_gradient_scale_saturated(),
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
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty), .o_saturated(o_saturated),
        .o_update_config_error(update_config_error),
        .o_request_error(request_error), .o_store_error(store_error),
        .o_internal_error(internal_error), .o_busy(rmw_busy)
    );

endmodule

`default_nettype wire
