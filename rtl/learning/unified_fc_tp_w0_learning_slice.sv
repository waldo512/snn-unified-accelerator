`timescale 1ns/1ps
`default_nettype none

// FC W0 online-learning slice. A sparse feature/sample adapter reduces the
// precomputed TP hidden error and sends gradient signs to the shared RMW path.
module unified_fc_tp_w0_learning_slice #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int ERROR_WIDTH = 18,
    parameter int GRAD_ACC_WIDTH = 48,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_FEATURES = 2312,
    parameter int PARAMETER_HIDDEN_STRIDE = 0,
    parameter int SAMPLE_ID_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int FEATURE_INDEX_WIDTH = (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,

    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]            i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]             i_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]            i_cfg_parameter_base,
    input  logic [SAMPLE_ID_WIDTH:0]               i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH:0]            i_cfg_hidden_count,
    input  logic [FEATURE_INDEX_WIDTH:0]           i_cfg_feature_count,
    input  logic [VOTE_WIDTH-1:0]                  i_cfg_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                  i_cfg_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0]           i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]           i_cfg_code_max,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  logic                                  i_release,
    output logic                                  o_release_ready,

    input  logic                                  i_hidden_error_valid,
    output logic                                  o_hidden_error_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]             i_hidden_error_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]          i_hidden_error_hidden_index,
    input  logic signed [ERROR_WIDTH-1:0]          i_hidden_error_value,
    output logic                                  o_hidden_error_error,
    input  logic                                  i_event_valid,
    output logic                                  o_event_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]             i_event_sample_id,
    input  logic [FEATURE_INDEX_WIDTH-1:0]         i_event_feature_index,
    output logic                                  o_event_error,
    input  logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,

    input  logic                                  i_window_emit,
    output logic                                  o_window_ready,
    output logic                                  o_window_done,
    output logic                                  o_window_busy,
    output logic                                  o_context_error,

    output logic                                  o_read_valid,
    input  logic                                  i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]            o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]             o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                   o_read_tag,
    input  logic                                  i_read_response_valid,
    output logic                                  o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]           i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]       i_read_response_residual,
    input  logic                                  i_read_response_error,

    output logic                                  o_result_valid,
    input  logic                                  i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]            o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]             o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                   o_result_tag,
    output logic signed [CODE_WIDTH-1:0]           o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]       o_result_residual,
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

    localparam logic [PARAM_ADDR_WIDTH:0] ADDRESS_SPACE_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    logic gradient_cfg_ready;
    logic gradient_cfg_error;
    logic gradient_active;
    logic [TENSOR_ID_WIDTH-1:0] gradient_active_tensor_id;
    logic gradient_release_ready;
    logic gradient_emit_ready;
    logic gradient_sign_valid;
    logic gradient_sign_ready;
    logic [TENSOR_ID_WIDTH-1:0] gradient_sign_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] gradient_sign_local_addr;
    logic signed [1:0] gradient_sign;
    logic signed [GRAD_ACC_WIDTH-1:0] gradient_accumulator;
    logic gradient_emit_done;
    logic gradient_busy;

    logic update_cfg_ready;
    logic update_cfg_error;
    logic update_active;
    logic [PROFILE_ID_WIDTH-1:0] update_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] update_active_tensor_id;
    logic update_release_ready;
    logic update_gradient_ready;
    logic update_accepted;
    logic update_zero_skipped;
    logic update_tensor_error;
    logic update_sign_error;
    logic update_busy;

    logic shared_cfg_valid;
    logic shared_cfg_legal;
    logic shared_release;
    logic window_active_q;
    logic gradient_done_seen_q;
    logic cfg_reject_q;
    logic [PROFILE_ID_WIDTH-1:0] active_profile_id_q;
    logic [TAG_WIDTH-1:0] gradient_tag;
    logic [PARAM_ADDR_WIDTH:0] element_count_ext;
    logic [PARAM_ADDR_WIDTH-1:0] element_count;
    logic [PARAM_ADDR_WIDTH:0] cfg_end_exclusive_ext;
    logic [VOTE_WIDTH:0] cfg_twice_threshold_ext;
    logic [PARAM_ADDR_WIDTH:0] hidden_stride_ext;

    always_comb begin
        hidden_stride_ext = (PARAMETER_HIDDEN_STRIDE == 0) ?
            (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count) :
            (PARAM_ADDR_WIDTH+1)'(PARAMETER_HIDDEN_STRIDE);
        element_count_ext = '0;
        if ((i_cfg_hidden_count != '0) && (i_cfg_feature_count != '0))
            element_count_ext =
                ((PARAM_ADDR_WIDTH+1)'(i_cfg_feature_count - 1'b1) *
                 hidden_stride_ext) +
                (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count);
        element_count = element_count_ext[PARAM_ADDR_WIDTH-1:0];
        cfg_end_exclusive_ext = {1'b0, i_cfg_parameter_base} + element_count_ext;
        cfg_twice_threshold_ext = {1'b0, i_cfg_vote_threshold} +
            {1'b0, i_cfg_vote_threshold};
        shared_cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= (SAMPLE_ID_WIDTH+1)'(MAX_BATCH)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= (HIDDEN_INDEX_WIDTH+1)'(MAX_HIDDEN)) &&
            (i_cfg_feature_count != '0) &&
            (i_cfg_feature_count <= (FEATURE_INDEX_WIDTH+1)'(MAX_FEATURES)) &&
            (hidden_stride_ext >=
             (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count)) &&
            (element_count_ext <= ADDRESS_SPACE_LIMIT) &&
            (cfg_end_exclusive_ext <= ADDRESS_SPACE_LIMIT) &&
            (i_cfg_vote_threshold != '0) &&
            ({1'b0, i_cfg_vote_increment} < cfg_twice_threshold_ext) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));

        o_cfg_ready = gradient_cfg_ready && update_cfg_ready && !window_active_q;
        shared_cfg_valid = i_cfg_valid && o_cfg_ready && shared_cfg_legal;
        o_release_ready = gradient_release_ready && update_release_ready &&
            !window_active_q;
        shared_release = i_release && o_release_ready;
        o_active = gradient_active && update_active &&
            (gradient_active_tensor_id == update_active_tensor_id);
        o_context_error = (gradient_active && !update_active) ||
            (gradient_active && update_active &&
             (gradient_active_tensor_id != update_active_tensor_id)) ||
            (update_active && (update_active_profile_id != active_profile_id_q)) ||
            (gradient_sign_valid && (gradient_accumulator == '0)) ||
            (update_accepted && !window_active_q) || update_tensor_error ||
            update_sign_error || update_zero_skipped;
        o_cfg_error = cfg_reject_q || gradient_cfg_error || update_cfg_error;
        o_window_ready = o_active && gradient_emit_ready && !window_active_q;
        o_window_busy = window_active_q || gradient_busy || update_busy;
        gradient_sign_ready = update_gradient_ready;
    end

    generate
        if (TAG_WIDTH >= PARAM_ADDR_WIDTH) begin : gen_wide_tag
            assign gradient_tag = {
                {(TAG_WIDTH-PARAM_ADDR_WIDTH){1'b0}}, gradient_sign_local_addr};
        end else begin : gen_narrow_tag
            assign gradient_tag = gradient_sign_local_addr[TAG_WIDTH-1:0];
        end
    endgenerate

    unified_fc_tp_w0_sparse_sign_generator #(
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .ERROR_WIDTH(ERROR_WIDTH),
        .ACC_WIDTH(GRAD_ACC_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_FEATURES(MAX_FEATURES),
        .PARAMETER_HIDDEN_STRIDE(PARAMETER_HIDDEN_STRIDE),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH)
    ) gradient_generator_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(gradient_cfg_ready),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_feature_count(i_cfg_feature_count),
        .o_cfg_error(gradient_cfg_error),
        .o_active(gradient_active),
        .o_active_tensor_id(gradient_active_tensor_id),
        .i_release(shared_release),
        .o_release_ready(gradient_release_ready),
        .i_hidden_error_valid(i_hidden_error_valid && o_active && !window_active_q),
        .o_hidden_error_ready(o_hidden_error_ready),
        .i_hidden_error_sample_id(i_hidden_error_sample_id),
        .i_hidden_error_hidden_index(i_hidden_error_hidden_index),
        .i_hidden_error_value(i_hidden_error_value),
        .o_hidden_error_error(o_hidden_error_error),
        .i_event_valid(i_event_valid && o_active && !window_active_q),
        .o_event_ready(o_event_ready),
        .i_event_sample_id(i_event_sample_id),
        .i_event_feature_index(i_event_feature_index),
        .o_event_error(o_event_error),
        .i_observation_done(i_observation_done && o_active && !window_active_q),
        .o_observation_done_ready(o_observation_done_ready),
        .i_emit_start(i_window_emit && o_window_ready),
        .o_emit_ready(gradient_emit_ready),
        .o_sign_valid(gradient_sign_valid),
        .i_sign_ready(gradient_sign_ready),
        .o_sign_tensor_id(gradient_sign_tensor_id),
        .o_sign_local_addr(gradient_sign_local_addr),
        .o_gradient_sign(gradient_sign),
        .o_gradient_accumulator(gradient_accumulator),
        .o_emit_done(gradient_emit_done),
        .o_busy(gradient_busy)
    );

    unified_fc_tp_gradient_rmw_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) update_path_i (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(update_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_element_count(element_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(update_cfg_error),
        .i_release(shared_release),
        .o_release_ready(update_release_ready),
        .o_active(update_active),
        .o_active_profile_id(update_active_profile_id),
        .o_active_tensor_id(update_active_tensor_id),
        .i_gradient_valid(gradient_sign_valid),
        .o_gradient_ready(update_gradient_ready),
        .i_gradient_tensor_id(gradient_sign_tensor_id),
        .i_gradient_local_parameter_addr(gradient_sign_local_addr),
        .i_gradient_tag(gradient_tag),
        .i_gradient_sign(gradient_sign),
        .i_gradient_accumulator(gradient_accumulator),
        .i_magnitude_mode(1'b0),
        .i_magnitude_multiplier('0),
        .o_gradient_update_accepted(update_accepted),
        .o_gradient_zero_skipped(update_zero_skipped),
        .o_gradient_tensor_error(update_tensor_error),
        .o_gradient_sign_error(update_sign_error),
        .o_gradient_scale_saturated(),
        .o_read_valid(o_read_valid),
        .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr),
        .o_read_tag(o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(o_read_response_ready),
        .i_read_response_code(i_read_response_code),
        .i_read_response_residual(i_read_response_residual),
        .i_read_response_error(i_read_response_error),
        .o_result_valid(o_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_profile_id(o_result_profile_id),
        .o_result_tensor_id(o_result_tensor_id),
        .o_result_parameter_addr(o_result_parameter_addr),
        .o_result_tag(o_result_tag),
        .o_result_code(o_result_code),
        .o_result_residual(o_result_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty),
        .o_saturated(o_saturated),
        .o_update_config_error(o_update_config_error),
        .o_request_error(o_request_error),
        .o_store_error(o_store_error),
        .o_internal_error(o_internal_error),
        .o_busy(update_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            window_active_q <= 1'b0;
            gradient_done_seen_q <= 1'b0;
            cfg_reject_q <= 1'b0;
            active_profile_id_q <= '0;
            o_window_done <= 1'b0;
        end else begin
            o_window_done <= 1'b0;
            cfg_reject_q <= i_cfg_valid && o_cfg_ready && !shared_cfg_legal;
            if (shared_cfg_valid)
                active_profile_id_q <= i_cfg_profile_id;
            if (shared_release)
                active_profile_id_q <= '0;
            if (i_window_emit && o_window_ready) begin
                window_active_q <= 1'b1;
                gradient_done_seen_q <= 1'b0;
            end
            if (window_active_q && gradient_emit_done)
                gradient_done_seen_q <= 1'b1;
            if (window_active_q && (gradient_done_seen_q || gradient_emit_done) &&
                !gradient_busy && !update_busy && !gradient_sign_valid) begin
                window_active_q <= 1'b0;
                gradient_done_seen_q <= 1'b0;
                o_window_done <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
