`timescale 1ns/1ps
`default_nettype none

// SCNN classifier adapter around the shared output-gradient and direct-RMW
// path. The SCNN observation boundary is Q8.8 and bank 12 is class-major;
// the shared learning core consumes uniformly scaled Q0.16 traces and Q6 logits.
module unified_scnn_tp_classifier_online_learning_path #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int MAX_GRADIENT_ELEMENTS = MAX_FEATURES * MAX_CLASSES,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int FEATURE_INDEX_WIDTH = (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_INDEX_WIDTH = (MAX_CLASSES <= 2) ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]           i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]            i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_parameter_base,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]        i_cfg_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]          i_cfg_class_count,
    input  wire logic [VOTE_WIDTH-1:0]                 i_cfg_vote_increment,
    input  wire logic [VOTE_WIDTH-1:0]                 i_cfg_vote_threshold,
    input  wire logic signed [CODE_WIDTH-1:0]          i_cfg_code_min,
    input  wire logic signed [CODE_WIDTH-1:0]          i_cfg_code_max,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  wire logic                                  i_release,
    output logic                                  o_release_ready,

    input  wire logic                                  i_feature_trace_valid,
    output logic                                  o_feature_trace_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]          i_feature_trace_sample_id,
    input  wire logic [FEATURE_INDEX_WIDTH-1:0]        i_feature_trace_index,
    input  wire logic signed [15:0]                    i_feature_trace_q8_8,
    output logic                                  o_feature_trace_error,
    input  wire logic                                  i_logit_row_valid,
    output logic                                  o_logit_row_ready,
    input  wire logic [BATCH_INDEX_WIDTH-1:0]          i_logit_sample_id,
    input  wire logic [CLASS_INDEX_WIDTH-1:0]          i_logit_label,
    input  wire logic                                  i_logit_valid,
    output logic                                  o_logit_ready,
    input  wire logic signed [15:0]                    i_logit_q8_8,
    output logic                                  o_logit_error,
    output logic                                  o_logit_row_done,
    input  wire logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,

    output logic                                  o_window_done,
    output logic                                  o_window_busy,
    output logic                                  o_context_error,
    output logic                                  o_gradient_overflow,
    output logic                                  o_read_valid,
    input  wire logic                                  i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_read_tag,
    input  wire logic                                  i_read_response_valid,
    output logic                                  o_read_response_ready,
    input  wire logic signed [CODE_WIDTH-1:0]          i_read_response_code,
    input  wire logic signed [RESIDUAL_WIDTH-1:0]      i_read_response_residual,
    input  wire logic                                  i_read_response_error,
    output logic                                  o_result_valid,
    input  wire logic                                  i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_result_tag,
    output logic signed [CODE_WIDTH-1:0]          o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]      o_result_residual,
    output logic                                  o_write_code_enable,
    output logic                                  o_write_residual_enable,
    output logic                                  o_write_code_fire,
    output logic                                  o_write_residual_fire,
    output logic                                  o_dirty,
    output logic                                  o_saturated,
    output logic                                  o_update_config_error,
    output logic                                  o_request_error,
    output logic                                  o_store_error,
    output logic                                  o_internal_error
);

    logic [15:0] feature_trace_q016;
    logic signed [15:0] logit_q6;
    logic shared_feature_trace_error;
    logic trace_negative;

    function automatic logic [15:0] q8_8_trace_to_q016(
        input logic signed [15:0] value
    );
        begin
            if ($signed(value) <= 0)
                return 16'd0;
            // Divide every real trace by 128: Q8.8 code * 2 is exact across
            // the full nonnegative range. Magnitude mode restores this factor
            // with multiplier 640 for the locked B4 fixed-SGD profile.
            return {value[14:0], 1'b0};
        end
    endfunction

    function automatic logic signed [15:0] q8_8_logit_to_q6(
        input logic signed [15:0] value
    );
        logic signed [16:0] magnitude;
        logic signed [16:0] rounded;
        begin
            if ($signed(value) >= 0) begin
                rounded = $signed({1'b0, value}) + 17'sd2;
                return 16'($signed(rounded) >>> 2);
            end
            magnitude = -$signed({value[15], value});
            rounded = magnitude + 17'sd2;
            return 16'(-$signed(rounded >>> 2));
        end
    endfunction

    assign feature_trace_q016 = q8_8_trace_to_q016(i_feature_trace_q8_8);
    assign logit_q6 = q8_8_logit_to_q6(i_logit_q8_8);
    assign trace_negative = $signed(i_feature_trace_q8_8) < 0;
    assign o_feature_trace_error = shared_feature_trace_error ||
        (i_feature_trace_valid && o_feature_trace_ready && trace_negative);

    unified_fc_tp_w2_online_learning_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_FEATURES),
        .MAX_OUTPUTS(MAX_CLASSES),
        .MAX_GRADIENT_ELEMENTS(MAX_GRADIENT_ELEMENTS),
        .OUTPUT_MAJOR_ADDRESS(1'b1),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_shared_classifier_path (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(i_cfg_valid), .o_cfg_ready(o_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_feature_count),
        .i_cfg_output_count(i_cfg_class_count),
        .i_cfg_output_major(1'b1),
        .i_cfg_hidden_stride('0),
        .i_cfg_output_stride('0),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min), .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(o_cfg_error), .o_active(o_active),
        .i_release(i_release), .o_release_ready(o_release_ready),
        .i_hidden_trace_valid(i_feature_trace_valid),
        .o_hidden_trace_ready(o_feature_trace_ready),
        .i_hidden_trace_sample_id(i_feature_trace_sample_id),
        .i_hidden_trace_hidden_index(i_feature_trace_index),
        .i_hidden_trace_q016(feature_trace_q016),
        .o_hidden_trace_error(shared_feature_trace_error),
        .i_logit_row_valid(i_logit_row_valid),
        .o_logit_row_ready(o_logit_row_ready),
        .i_logit_sample_id(i_logit_sample_id),
        .i_logit_label(i_logit_label), .i_logit_valid(i_logit_valid),
        .o_logit_ready(o_logit_ready), .i_logit_q6(logit_q6),
        .o_logit_error(o_logit_error),
        .o_logit_row_done(o_logit_row_done),
        .i_observation_done(i_observation_done),
        .o_observation_done_ready(o_observation_done_ready),
        .o_window_done(o_window_done), .o_window_busy(o_window_busy),
        .o_context_error(o_context_error),
        .o_gradient_overflow(o_gradient_overflow),
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
        .o_update_config_error(o_update_config_error),
        .o_request_error(o_request_error), .o_store_error(o_store_error),
        .o_internal_error(o_internal_error)
    );

endmodule

`default_nettype wire
