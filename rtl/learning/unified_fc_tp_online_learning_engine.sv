`timescale 1ns/1ps
`default_nettype none

// FC training/inference integration boundary for TP online learning. One
// observation window feeds both W0 and W2 learning paths; their parameter RMW
// transactions share one backing-store port and a commit fence is emitted only
// after both tensor paths have drained.
module unified_fc_tp_online_learning_engine #(
    parameter bit ENABLE_W0_LEARNING = 1'b1,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_FEATURES = 2312,
    parameter int MAX_TARGET_DIMENSION = 16,
    parameter int MAX_OUTPUTS = 16,
    parameter int TARGET_TRACE_WIDTH = 20,
    parameter int W2_MAX_GRADIENT_ELEMENTS = 32768,
    parameter int W0_PARAMETER_HIDDEN_STRIDE = 0,
    parameter int W2_PARAMETER_OUTPUT_STRIDE = 0,
    parameter int SAMPLE_ID_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int FEATURE_INDEX_WIDTH = (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int TARGET_INDEX_WIDTH =
        (MAX_TARGET_DIMENSION <= 2) ? 1 : $clog2(MAX_TARGET_DIMENSION),
    parameter int TARGET_COUNT_WIDTH = $clog2(MAX_TARGET_DIMENSION + 1),
    parameter int OUTPUT_INDEX_WIDTH = (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_cfg_w0_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_w0_parameter_base,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w0_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w0_vote_threshold,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_cfg_w2_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_w2_parameter_base,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w2_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_w2_vote_threshold,
    input  logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]         i_cfg_hidden_count,
    input  logic [FEATURE_COUNT_WIDTH-1:0]        i_cfg_feature_count,
    input  logic [TARGET_COUNT_WIDTH-1:0]         i_cfg_target_dimension,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]         i_cfg_output_count,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w0_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w0_code_max,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w2_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_w2_code_max,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  logic                                  i_release,
    output logic                                  o_release_ready,

    input  logic                                  i_student_trace_valid,
    output logic                                  o_student_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_student_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_student_trace_hidden,
    input  logic [16:0]                           i_student_trace_q016,
    input  logic                                  i_target_hidden_trace_valid,
    output logic                                  o_target_hidden_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_target_hidden_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_target_hidden_trace_hidden,
    input  logic [16:0]                           i_target_hidden_trace_q016,
    input  logic                                  i_target_input_trace_valid,
    output logic                                  o_target_input_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_target_input_trace_sample,
    input  logic [TARGET_INDEX_WIDTH-1:0]         i_target_input_trace_index,
    input  logic [TARGET_TRACE_WIDTH-1:0]         i_target_input_trace_q016,
    input  logic                                  i_surrogate_valid,
    output logic                                  o_surrogate_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_surrogate_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_surrogate_hidden,
    input  logic [16:0]                           i_surrogate_q016,
    input  logic                                  i_event_valid,
    output logic                                  o_event_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_event_sample_id,
    input  logic [FEATURE_INDEX_WIDTH-1:0]        i_event_feature_index,
    input  logic                                  i_logit_row_valid,
    output logic                                  o_logit_row_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_logit_sample_id,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]         i_logit_label,
    input  logic                                  i_logit_valid,
    output logic                                  o_logit_ready,
    input  logic signed [15:0]                    i_logit_q6,
    output logic                                  o_logit_row_done,
    input  logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,

    output logic                                  o_window_commit_valid,
    input  logic                                  i_window_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_window_commit_profile_id,
    output logic [31:0]                           o_window_commit_sequence,
    output logic                                  o_window_busy,
    output logic                                  o_observation_error,
    output logic                                  o_context_error,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_parameter_error,
    output logic                                  o_protocol_error,

    output logic                                  o_read_valid,
    input  logic                                  i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                  o_read_tag,
    input  logic                                  i_read_response_valid,
    output logic                                  o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]          i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]      i_read_response_residual,
    input  logic                                  i_read_response_error,
    output logic                                  o_result_valid,
    input  logic                                  i_result_ready,
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
    output logic                                  o_saturated
);

    localparam logic [PARAM_ADDR_WIDTH:0] ADDRESS_SPACE_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_OBSERVE,
        STATE_DRAIN,
        STATE_COMMIT
    } state_t;

    state_t state_q;
    logic cfg_legal;
    logic cfg_fire;
    logic shared_cfg_valid;
    logic release_fire;
    logic observation_fire;
    logic cfg_reject_q;
    logic w0_done_seen_q;
    logic w2_done_seen_q;
    logic commit_valid_q;
    logic [PROFILE_ID_WIDTH-1:0] active_profile_id_q;
    logic [31:0] commit_sequence_q;
    logic [PARAM_ADDR_WIDTH:0] w0_element_count_ext;
    logic [PARAM_ADDR_WIDTH:0] w2_element_count_ext;
    logic [PARAM_ADDR_WIDTH:0] w0_end_ext;
    logic [PARAM_ADDR_WIDTH:0] w2_end_ext;
    logic [PARAM_ADDR_WIDTH:0] w0_hidden_stride_ext;
    logic [PARAM_ADDR_WIDTH:0] w2_output_stride_ext;
    logic [VOTE_WIDTH:0] w0_twice_threshold_ext;
    logic [VOTE_WIDTH:0] w2_twice_threshold_ext;

    logic w0_cfg_ready;
    logic w0_cfg_error;
    logic w0_active;
    logic w0_release_ready;
    logic w0_student_ready;
    logic w0_student_error;
    logic w0_target_hidden_ready;
    logic w0_target_hidden_error;
    logic w0_target_input_ready;
    logic w0_target_input_error;
    logic w0_surrogate_ready;
    logic w0_surrogate_error;
    logic w0_event_ready;
    logic w0_event_error;
    logic w0_observation_done_ready;
    logic w0_window_done;
    logic w0_window_busy;
    logic w0_context_error;
    logic w0_arithmetic_overflow;
    logic w0_read_valid;
    logic w0_read_ready;
    logic [PROFILE_ID_WIDTH-1:0] w0_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] w0_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] w0_read_parameter_addr;
    logic [TAG_WIDTH-1:0] w0_read_tag;
    logic w0_read_response_valid;
    logic w0_read_response_ready;
    logic signed [CODE_WIDTH-1:0] w0_read_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] w0_read_response_residual;
    logic w0_read_response_error;
    logic w0_result_valid;
    logic w0_result_ready;
    logic [PROFILE_ID_WIDTH-1:0] w0_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] w0_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] w0_result_parameter_addr;
    logic [TAG_WIDTH-1:0] w0_result_tag;
    logic signed [CODE_WIDTH-1:0] w0_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] w0_result_residual;
    logic w0_write_code_enable;
    logic w0_write_residual_enable;
    logic w0_write_code_fire;
    logic w0_write_residual_fire;
    logic w0_dirty;
    logic w0_saturated;
    logic w0_update_config_error;
    logic w0_request_error;
    logic w0_store_error;
    logic w0_internal_error;

    logic w2_cfg_ready;
    logic w2_cfg_error;
    logic w2_active;
    logic w2_release_ready;
    logic w2_hidden_ready;
    logic w2_hidden_error;
    logic w2_logit_row_ready;
    logic w2_logit_ready;
    logic w2_logit_error;
    logic w2_observation_done_ready;
    logic w2_window_done;
    logic w2_window_busy;
    logic w2_context_error;
    logic w2_gradient_overflow;
    logic w2_read_valid;
    logic w2_read_ready;
    logic [PROFILE_ID_WIDTH-1:0] w2_read_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] w2_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] w2_read_parameter_addr;
    logic [TAG_WIDTH-1:0] w2_read_tag;
    logic w2_read_response_valid;
    logic w2_read_response_ready;
    logic signed [CODE_WIDTH-1:0] w2_read_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] w2_read_response_residual;
    logic w2_read_response_error;
    logic w2_result_valid;
    logic w2_result_ready;
    logic [PROFILE_ID_WIDTH-1:0] w2_result_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] w2_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] w2_result_parameter_addr;
    logic [TAG_WIDTH-1:0] w2_result_tag;
    logic signed [CODE_WIDTH-1:0] w2_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] w2_result_residual;
    logic w2_write_code_enable;
    logic w2_write_residual_enable;
    logic w2_write_code_fire;
    logic w2_write_residual_fire;
    logic w2_dirty;
    logic w2_saturated;
    logic w2_update_config_error;
    logic w2_request_error;
    logic w2_store_error;
    logic w2_internal_error;
    logic rmw_busy;
    logic rmw_protocol_error;
    logic shared_student_fire;
    logic [15:0] w2_student_trace_q016;

    always_comb begin
        w0_hidden_stride_ext = (W0_PARAMETER_HIDDEN_STRIDE == 0) ?
            (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count) :
            (PARAM_ADDR_WIDTH+1)'(W0_PARAMETER_HIDDEN_STRIDE);
        w2_output_stride_ext = (W2_PARAMETER_OUTPUT_STRIDE == 0) ?
            (PARAM_ADDR_WIDTH+1)'(i_cfg_output_count) :
            (PARAM_ADDR_WIDTH+1)'(W2_PARAMETER_OUTPUT_STRIDE);
        w0_element_count_ext = '0;
        if ((i_cfg_hidden_count != '0) && (i_cfg_feature_count != '0))
            w0_element_count_ext =
                ((PARAM_ADDR_WIDTH+1)'(i_cfg_feature_count - 1'b1) *
                 w0_hidden_stride_ext) +
                (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count);
        w2_element_count_ext = '0;
        if ((i_cfg_hidden_count != '0) && (i_cfg_output_count != '0))
            w2_element_count_ext =
                ((PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count - 1'b1) *
                 w2_output_stride_ext) +
                (PARAM_ADDR_WIDTH+1)'(i_cfg_output_count);
        w0_end_ext = {1'b0, i_cfg_w0_parameter_base} + w0_element_count_ext;
        w2_end_ext = {1'b0, i_cfg_w2_parameter_base} + w2_element_count_ext;
        w0_twice_threshold_ext = {1'b0, i_cfg_w0_vote_threshold} +
            {1'b0, i_cfg_w0_vote_threshold};
        w2_twice_threshold_ext = {1'b0, i_cfg_w2_vote_threshold} +
            {1'b0, i_cfg_w2_vote_threshold};
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= BATCH_COUNT_WIDTH'(MAX_BATCH)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_COUNT_WIDTH'(MAX_HIDDEN)) &&
            (i_cfg_feature_count != '0) &&
            (i_cfg_feature_count <= FEATURE_COUNT_WIDTH'(MAX_FEATURES)) &&
            (i_cfg_target_dimension != '0) &&
            (i_cfg_target_dimension <=
             TARGET_COUNT_WIDTH'(MAX_TARGET_DIMENSION)) &&
            (i_cfg_output_count != '0) &&
            (i_cfg_output_count <= OUTPUT_COUNT_WIDTH'(MAX_OUTPUTS)) &&
            (!ENABLE_W0_LEARNING ||
             (w0_hidden_stride_ext >=
              (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count))) &&
            (w2_output_stride_ext >=
             (PARAM_ADDR_WIDTH+1)'(i_cfg_output_count)) &&
            (!ENABLE_W0_LEARNING ||
             (w0_element_count_ext <= ADDRESS_SPACE_LIMIT)) &&
            (w2_element_count_ext <=
             (PARAM_ADDR_WIDTH+1)'(W2_MAX_GRADIENT_ELEMENTS)) &&
            (!ENABLE_W0_LEARNING ||
             (w0_end_ext <= ADDRESS_SPACE_LIMIT)) &&
            (w2_end_ext <= ADDRESS_SPACE_LIMIT) &&
            (!ENABLE_W0_LEARNING ||
             ((w0_end_ext <= {1'b0, i_cfg_w2_parameter_base}) ||
              (w2_end_ext <= {1'b0, i_cfg_w0_parameter_base}))) &&
            (!ENABLE_W0_LEARNING ||
             (i_cfg_w0_tensor_id != i_cfg_w2_tensor_id)) &&
            (!ENABLE_W0_LEARNING ||
             ((i_cfg_w0_vote_threshold != '0) &&
              ({1'b0, i_cfg_w0_vote_increment} <
               w0_twice_threshold_ext))) &&
            (i_cfg_w2_vote_threshold != '0) &&
            ({1'b0, i_cfg_w2_vote_increment} < w2_twice_threshold_ext) &&
            (!ENABLE_W0_LEARNING ||
             ($signed(i_cfg_w0_code_min) <=
              $signed(i_cfg_w0_code_max))) &&
            ($signed(i_cfg_w2_code_min) <= $signed(i_cfg_w2_code_max));
        o_cfg_ready = (state_q == STATE_IDLE) && w0_cfg_ready && w2_cfg_ready;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        shared_cfg_valid = cfg_fire && cfg_legal;
        o_cfg_error = cfg_reject_q || w0_cfg_error || w2_cfg_error;
        o_active = (state_q != STATE_IDLE) && w0_active && w2_active;
        o_release_ready = (state_q == STATE_OBSERVE) && w0_release_ready &&
            w2_release_ready;
        release_fire = i_release && o_release_ready;

        o_student_trace_ready = (state_q == STATE_OBSERVE) &&
            w0_student_ready && w2_hidden_ready;
        shared_student_fire = i_student_trace_valid && o_student_trace_ready;
        w2_student_trace_q016 = i_student_trace_q016[16] ? 16'hffff :
            i_student_trace_q016[15:0];
        o_target_hidden_trace_ready = (state_q == STATE_OBSERVE) &&
            w0_target_hidden_ready;
        o_target_input_trace_ready = (state_q == STATE_OBSERVE) &&
            w0_target_input_ready;
        o_surrogate_ready = (state_q == STATE_OBSERVE) && w0_surrogate_ready;
        o_event_ready = (state_q == STATE_OBSERVE) && w0_event_ready;
        o_logit_row_ready = (state_q == STATE_OBSERVE) && w2_logit_row_ready;
        o_logit_ready = (state_q == STATE_OBSERVE) && w2_logit_ready;
        o_observation_done_ready = (state_q == STATE_OBSERVE) &&
            w0_observation_done_ready && w2_observation_done_ready;
        observation_fire = i_observation_done && o_observation_done_ready;

        o_window_commit_valid = commit_valid_q;
        o_window_commit_profile_id = active_profile_id_q;
        o_window_commit_sequence = commit_sequence_q;
        o_window_busy = (state_q == STATE_DRAIN) ||
            (state_q == STATE_COMMIT) || w0_window_busy || w2_window_busy ||
            rmw_busy;
        o_observation_error = w0_student_error || w2_hidden_error ||
            w0_target_hidden_error || w0_target_input_error ||
            w0_surrogate_error || w0_event_error || w2_logit_error;
        o_context_error = w0_context_error || w2_context_error;
        o_arithmetic_overflow = w0_arithmetic_overflow || w2_gradient_overflow;
        o_parameter_error = w0_update_config_error || w0_request_error ||
            w0_store_error || w0_internal_error || w2_update_config_error ||
            w2_request_error || w2_store_error || w2_internal_error;
        o_protocol_error = rmw_protocol_error;
    end

    generate
    if (ENABLE_W0_LEARNING) begin : g_w0_learning
    unified_fc_tp_w0_online_learning_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH),
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_FEATURES(MAX_FEATURES),
        .MAX_TARGET_DIMENSION(MAX_TARGET_DIMENSION),
        .TARGET_TRACE_WIDTH(TARGET_TRACE_WIDTH),
        .PARAMETER_HIDDEN_STRIDE(W0_PARAMETER_HIDDEN_STRIDE),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .TARGET_INDEX_WIDTH(TARGET_INDEX_WIDTH)
    ) u_w0 (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid), .o_cfg_ready(w0_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_w0_tensor_id),
        .i_cfg_parameter_base(i_cfg_w0_parameter_base),
        .i_cfg_batch_count((SAMPLE_ID_WIDTH+1)'(i_cfg_batch_count)),
        .i_cfg_hidden_count((HIDDEN_INDEX_WIDTH+1)'(i_cfg_hidden_count)),
        .i_cfg_feature_count((FEATURE_INDEX_WIDTH+1)'(i_cfg_feature_count)),
        .i_cfg_target_dimension((TARGET_INDEX_WIDTH+1)'(i_cfg_target_dimension)),
        .i_cfg_vote_increment(i_cfg_w0_vote_increment),
        .i_cfg_vote_threshold(i_cfg_w0_vote_threshold),
        .i_cfg_code_min(i_cfg_w0_code_min),
        .i_cfg_code_max(i_cfg_w0_code_max),
        .o_cfg_error(w0_cfg_error), .o_active(w0_active),
        .i_release(release_fire), .o_release_ready(w0_release_ready),
        .i_student_trace_valid(shared_student_fire),
        .o_student_trace_ready(w0_student_ready),
        .i_student_trace_sample(i_student_trace_sample),
        .i_student_trace_hidden(i_student_trace_hidden),
        .i_student_trace_q016(i_student_trace_q016),
        .o_student_trace_error(w0_student_error),
        .i_target_hidden_trace_valid(i_target_hidden_trace_valid &&
                                     (state_q == STATE_OBSERVE)),
        .o_target_hidden_trace_ready(w0_target_hidden_ready),
        .i_target_hidden_trace_sample(i_target_hidden_trace_sample),
        .i_target_hidden_trace_hidden(i_target_hidden_trace_hidden),
        .i_target_hidden_trace_q016(i_target_hidden_trace_q016),
        .o_target_hidden_trace_error(w0_target_hidden_error),
        .i_target_input_trace_valid(i_target_input_trace_valid &&
                                    (state_q == STATE_OBSERVE)),
        .o_target_input_trace_ready(w0_target_input_ready),
        .i_target_input_trace_sample(i_target_input_trace_sample),
        .i_target_input_trace_index(i_target_input_trace_index),
        .i_target_input_trace_q016(i_target_input_trace_q016),
        .o_target_input_trace_error(w0_target_input_error),
        .i_surrogate_valid(i_surrogate_valid &&
                           (state_q == STATE_OBSERVE)),
        .o_surrogate_ready(w0_surrogate_ready),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_hidden(i_surrogate_hidden),
        .i_surrogate_q016(i_surrogate_q016),
        .o_surrogate_error(w0_surrogate_error),
        .i_event_valid(i_event_valid && (state_q == STATE_OBSERVE)),
        .o_event_ready(w0_event_ready),
        .i_event_sample_id(i_event_sample_id),
        .i_event_feature_index(i_event_feature_index),
        .o_event_error(w0_event_error),
        .i_observation_done(observation_fire),
        .o_observation_done_ready(w0_observation_done_ready),
        .o_window_done(w0_window_done), .o_window_busy(w0_window_busy),
        .o_context_error(w0_context_error),
        .o_arithmetic_overflow(w0_arithmetic_overflow),
        .o_read_valid(w0_read_valid), .i_read_ready(w0_read_ready),
        .o_read_profile_id(w0_read_profile_id),
        .o_read_tensor_id(w0_read_tensor_id),
        .o_read_parameter_addr(w0_read_parameter_addr),
        .o_read_tag(w0_read_tag),
        .i_read_response_valid(w0_read_response_valid),
        .o_read_response_ready(w0_read_response_ready),
        .i_read_response_code(w0_read_response_code),
        .i_read_response_residual(w0_read_response_residual),
        .i_read_response_error(w0_read_response_error),
        .o_result_valid(w0_result_valid), .i_result_ready(w0_result_ready),
        .o_result_profile_id(w0_result_profile_id),
        .o_result_tensor_id(w0_result_tensor_id),
        .o_result_parameter_addr(w0_result_parameter_addr),
        .o_result_tag(w0_result_tag), .o_result_code(w0_result_code),
        .o_result_residual(w0_result_residual),
        .o_write_code_enable(w0_write_code_enable),
        .o_write_residual_enable(w0_write_residual_enable),
        .o_write_code_fire(w0_write_code_fire),
        .o_write_residual_fire(w0_write_residual_fire),
        .o_dirty(w0_dirty), .o_saturated(w0_saturated),
        .o_update_config_error(w0_update_config_error),
        .o_request_error(w0_request_error), .o_store_error(w0_store_error),
        .o_internal_error(w0_internal_error)
    );
    end else begin : g_no_w0_learning
        assign w0_cfg_ready = 1'b1;
        assign w0_cfg_error = 1'b0;
        assign w0_active = state_q != STATE_IDLE;
        assign w0_release_ready = 1'b1;
        assign w0_student_ready = 1'b1;
        assign w0_student_error = 1'b0;
        assign w0_target_hidden_ready = 1'b1;
        assign w0_target_hidden_error = 1'b0;
        assign w0_target_input_ready = 1'b1;
        assign w0_target_input_error = 1'b0;
        assign w0_surrogate_ready = 1'b1;
        assign w0_surrogate_error = 1'b0;
        assign w0_event_ready = 1'b1;
        assign w0_event_error = 1'b0;
        assign w0_observation_done_ready = 1'b1;
        assign w0_window_done = state_q == STATE_DRAIN;
        assign w0_window_busy = 1'b0;
        assign w0_context_error = 1'b0;
        assign w0_arithmetic_overflow = 1'b0;
        assign w0_read_valid = 1'b0;
        assign w0_read_profile_id = '0;
        assign w0_read_tensor_id = '0;
        assign w0_read_parameter_addr = '0;
        assign w0_read_tag = '0;
        assign w0_read_response_ready = 1'b0;
        assign w0_result_valid = 1'b0;
        assign w0_result_profile_id = '0;
        assign w0_result_tensor_id = '0;
        assign w0_result_parameter_addr = '0;
        assign w0_result_tag = '0;
        assign w0_result_code = '0;
        assign w0_result_residual = '0;
        assign w0_write_code_enable = 1'b0;
        assign w0_write_residual_enable = 1'b0;
        assign w0_write_code_fire = 1'b0;
        assign w0_write_residual_fire = 1'b0;
        assign w0_dirty = 1'b0;
        assign w0_saturated = 1'b0;
        assign w0_update_config_error = 1'b0;
        assign w0_request_error = 1'b0;
        assign w0_store_error = 1'b0;
        assign w0_internal_error = 1'b0;
    end
    endgenerate

    unified_fc_tp_w2_online_learning_path #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH), .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN), .MAX_OUTPUTS(MAX_OUTPUTS),
        .MAX_GRADIENT_ELEMENTS(W2_MAX_GRADIENT_ELEMENTS),
        .PARAMETER_OUTPUT_STRIDE(W2_PARAMETER_OUTPUT_STRIDE),
        .BATCH_INDEX_WIDTH(SAMPLE_ID_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH),
        .OUTPUT_INDEX_WIDTH(OUTPUT_INDEX_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_w2 (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid), .o_cfg_ready(w2_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_w2_tensor_id),
        .i_cfg_parameter_base(i_cfg_w2_parameter_base),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_output_count(i_cfg_output_count),
        .i_cfg_output_major(1'b0),
        .i_cfg_hidden_stride('0),
        .i_cfg_output_stride(PARAM_ADDR_WIDTH'(W2_PARAMETER_OUTPUT_STRIDE)),
        .i_cfg_vote_increment(i_cfg_w2_vote_increment),
        .i_cfg_vote_threshold(i_cfg_w2_vote_threshold),
        .i_cfg_code_min(i_cfg_w2_code_min),
        .i_cfg_code_max(i_cfg_w2_code_max),
        .o_cfg_error(w2_cfg_error), .o_active(w2_active),
        .i_release(release_fire), .o_release_ready(w2_release_ready),
        .i_hidden_trace_valid(shared_student_fire),
        .o_hidden_trace_ready(w2_hidden_ready),
        .i_hidden_trace_sample_id(i_student_trace_sample),
        .i_hidden_trace_hidden_index(i_student_trace_hidden),
        .i_hidden_trace_q016(w2_student_trace_q016),
        .o_hidden_trace_error(w2_hidden_error),
        .i_logit_row_valid(i_logit_row_valid &&
                           (state_q == STATE_OBSERVE)),
        .o_logit_row_ready(w2_logit_row_ready),
        .i_logit_sample_id(i_logit_sample_id), .i_logit_label(i_logit_label),
        .i_logit_valid(i_logit_valid && (state_q == STATE_OBSERVE)),
        .o_logit_ready(w2_logit_ready), .i_logit_q6(i_logit_q6),
        .o_logit_error(w2_logit_error), .o_logit_row_done(o_logit_row_done),
        .i_observation_done(observation_fire),
        .o_observation_done_ready(w2_observation_done_ready),
        .o_window_done(w2_window_done), .o_window_busy(w2_window_busy),
        .o_context_error(w2_context_error),
        .o_gradient_overflow(w2_gradient_overflow),
        .o_read_valid(w2_read_valid), .i_read_ready(w2_read_ready),
        .o_read_profile_id(w2_read_profile_id),
        .o_read_tensor_id(w2_read_tensor_id),
        .o_read_parameter_addr(w2_read_parameter_addr),
        .o_read_tag(w2_read_tag),
        .i_read_response_valid(w2_read_response_valid),
        .o_read_response_ready(w2_read_response_ready),
        .i_read_response_code(w2_read_response_code),
        .i_read_response_residual(w2_read_response_residual),
        .i_read_response_error(w2_read_response_error),
        .o_result_valid(w2_result_valid), .i_result_ready(w2_result_ready),
        .o_result_profile_id(w2_result_profile_id),
        .o_result_tensor_id(w2_result_tensor_id),
        .o_result_parameter_addr(w2_result_parameter_addr),
        .o_result_tag(w2_result_tag), .o_result_code(w2_result_code),
        .o_result_residual(w2_result_residual),
        .o_write_code_enable(w2_write_code_enable),
        .o_write_residual_enable(w2_write_residual_enable),
        .o_write_code_fire(w2_write_code_fire),
        .o_write_residual_fire(w2_write_residual_fire),
        .o_dirty(w2_dirty), .o_saturated(w2_saturated),
        .o_update_config_error(w2_update_config_error),
        .o_request_error(w2_request_error), .o_store_error(w2_store_error),
        .o_internal_error(w2_internal_error)
    );

    unified_tp_two_client_parameter_rmw_arbiter #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_rmw_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_c0_read_valid(w0_read_valid), .o_c0_read_ready(w0_read_ready),
        .i_c0_read_profile_id(w0_read_profile_id),
        .i_c0_read_tensor_id(w0_read_tensor_id),
        .i_c0_read_parameter_addr(w0_read_parameter_addr),
        .i_c0_read_tag(w0_read_tag),
        .o_c0_read_response_valid(w0_read_response_valid),
        .i_c0_read_response_ready(w0_read_response_ready),
        .o_c0_read_response_code(w0_read_response_code),
        .o_c0_read_response_residual(w0_read_response_residual),
        .o_c0_read_response_error(w0_read_response_error),
        .i_c0_result_valid(w0_result_valid), .o_c0_result_ready(w0_result_ready),
        .i_c0_result_profile_id(w0_result_profile_id),
        .i_c0_result_tensor_id(w0_result_tensor_id),
        .i_c0_result_parameter_addr(w0_result_parameter_addr),
        .i_c0_result_tag(w0_result_tag), .i_c0_result_code(w0_result_code),
        .i_c0_result_residual(w0_result_residual),
        .i_c0_write_code_enable(w0_write_code_enable),
        .i_c0_write_residual_enable(w0_write_residual_enable),
        .i_c0_write_code_fire(w0_write_code_fire),
        .i_c0_write_residual_fire(w0_write_residual_fire),
        .i_c0_dirty(w0_dirty), .i_c0_saturated(w0_saturated),
        .i_c1_read_valid(w2_read_valid), .o_c1_read_ready(w2_read_ready),
        .i_c1_read_profile_id(w2_read_profile_id),
        .i_c1_read_tensor_id(w2_read_tensor_id),
        .i_c1_read_parameter_addr(w2_read_parameter_addr),
        .i_c1_read_tag(w2_read_tag),
        .o_c1_read_response_valid(w2_read_response_valid),
        .i_c1_read_response_ready(w2_read_response_ready),
        .o_c1_read_response_code(w2_read_response_code),
        .o_c1_read_response_residual(w2_read_response_residual),
        .o_c1_read_response_error(w2_read_response_error),
        .i_c1_result_valid(w2_result_valid), .o_c1_result_ready(w2_result_ready),
        .i_c1_result_profile_id(w2_result_profile_id),
        .i_c1_result_tensor_id(w2_result_tensor_id),
        .i_c1_result_parameter_addr(w2_result_parameter_addr),
        .i_c1_result_tag(w2_result_tag), .i_c1_result_code(w2_result_code),
        .i_c1_result_residual(w2_result_residual),
        .i_c1_write_code_enable(w2_write_code_enable),
        .i_c1_write_residual_enable(w2_write_residual_enable),
        .i_c1_write_code_fire(w2_write_code_fire),
        .i_c1_write_residual_fire(w2_write_residual_fire),
        .i_c1_dirty(w2_dirty), .i_c1_saturated(w2_saturated),
        .o_read_valid(o_read_valid), .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr), .o_read_tag(o_read_tag),
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
        .o_busy(rmw_busy), .o_protocol_error(rmw_protocol_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            cfg_reject_q <= 1'b0;
            w0_done_seen_q <= 1'b0;
            w2_done_seen_q <= 1'b0;
            commit_valid_q <= 1'b0;
            active_profile_id_q <= '0;
            commit_sequence_q <= '0;
        end else begin
            cfg_reject_q <= cfg_fire && !cfg_legal;
            if (w0_window_done)
                w0_done_seen_q <= 1'b1;
            if (w2_window_done)
                w2_done_seen_q <= 1'b1;
            case (state_q)
                STATE_IDLE: begin
                    if (shared_cfg_valid) begin
                        active_profile_id_q <= i_cfg_profile_id;
                        w0_done_seen_q <= 1'b0;
                        w2_done_seen_q <= 1'b0;
                        state_q <= STATE_OBSERVE;
                    end
                end
                STATE_OBSERVE: begin
                    if (release_fire) begin
                        active_profile_id_q <= '0;
                        state_q <= STATE_IDLE;
                    end else if (observation_fire) begin
                        w0_done_seen_q <= 1'b0;
                        w2_done_seen_q <= 1'b0;
                        state_q <= STATE_DRAIN;
                    end
                end
                STATE_DRAIN: begin
                    if ((w0_done_seen_q || w0_window_done) &&
                        (w2_done_seen_q || w2_window_done) && !rmw_busy) begin
                        commit_valid_q <= 1'b1;
                        state_q <= STATE_COMMIT;
                    end
                end
                STATE_COMMIT: begin
                    if (commit_valid_q && i_window_commit_ready) begin
                        commit_valid_q <= 1'b0;
                        commit_sequence_q <= commit_sequence_q + 1'b1;
                        w0_done_seen_q <= 1'b0;
                        w2_done_seen_q <= 1'b0;
                        state_q <= STATE_OBSERVE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
