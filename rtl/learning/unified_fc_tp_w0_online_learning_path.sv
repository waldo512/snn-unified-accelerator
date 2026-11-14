`timescale 1ns/1ps
`default_nettype none

// Complete FC TP W0 online-learning path. Raw traces are transformed into
// hidden errors by the shared pairwise frontend and then combined with sparse
// input events before the shared parameter RMW service is launched.
module unified_fc_tp_w0_online_learning_path #(
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
    parameter int PARAMETER_HIDDEN_STRIDE = 0,
    parameter int MAX_TARGET_DIMENSION = 16,
    parameter int TARGET_TRACE_WIDTH = 20,
    parameter int SAMPLE_ID_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int FEATURE_INDEX_WIDTH = (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES),
    parameter int TARGET_INDEX_WIDTH =
        (MAX_TARGET_DIMENSION <= 2) ? 1 : $clog2(MAX_TARGET_DIMENSION)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_cfg_parameter_base,
    input  logic [SAMPLE_ID_WIDTH:0]              i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH:0]           i_cfg_hidden_count,
    input  logic [FEATURE_INDEX_WIDTH:0]          i_cfg_feature_count,
    input  logic [TARGET_INDEX_WIDTH:0]           i_cfg_target_dimension,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_vote_increment,
    input  logic [VOTE_WIDTH-1:0]                 i_cfg_vote_threshold,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]          i_cfg_code_max,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  logic                                  i_release,
    output logic                                  o_release_ready,

    input  logic                                  i_student_trace_valid,
    output logic                                  o_student_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_student_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_student_trace_hidden,
    input  logic [16:0]                           i_student_trace_q016,
    output logic                                  o_student_trace_error,
    input  logic                                  i_target_hidden_trace_valid,
    output logic                                  o_target_hidden_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_target_hidden_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_target_hidden_trace_hidden,
    input  logic [16:0]                           i_target_hidden_trace_q016,
    output logic                                  o_target_hidden_trace_error,
    input  logic                                  i_target_input_trace_valid,
    output logic                                  o_target_input_trace_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_target_input_trace_sample,
    input  logic [TARGET_INDEX_WIDTH-1:0]         i_target_input_trace_index,
    input  logic [TARGET_TRACE_WIDTH-1:0]         i_target_input_trace_q016,
    output logic                                  o_target_input_trace_error,
    input  logic                                  i_surrogate_valid,
    output logic                                  o_surrogate_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_surrogate_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_surrogate_hidden,
    input  logic [16:0]                           i_surrogate_q016,
    output logic                                  o_surrogate_error,
    input  logic                                  i_event_valid,
    output logic                                  o_event_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_event_sample_id,
    input  logic [FEATURE_INDEX_WIDTH-1:0]        i_event_feature_index,
    output logic                                  o_event_error,
    input  logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,

    output logic                                  o_window_done,
    output logic                                  o_window_busy,
    output logic                                  o_context_error,
    output logic                                  o_arithmetic_overflow,
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
    output logic                                  o_saturated,
    output logic                                  o_update_config_error,
    output logic                                  o_request_error,
    output logic                                  o_store_error,
    output logic                                  o_internal_error
);

    localparam logic [PARAM_ADDR_WIDTH:0] ADDRESS_SPACE_LIMIT =
        {1'b1, {PARAM_ADDR_WIDTH{1'b0}}};

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_OBSERVE,
        STATE_START_UPDATE,
        STATE_UPDATE
    } state_t;

    state_t state_q;
    logic frontend_cfg_ready;
    logic frontend_cfg_error;
    logic frontend_active;
    logic frontend_release_ready;
    logic frontend_hidden_error_valid;
    logic frontend_hidden_error_ready;
    logic [SAMPLE_ID_WIDTH-1:0] frontend_hidden_error_sample;
    logic [HIDDEN_INDEX_WIDTH-1:0] frontend_hidden_error_hidden;
    logic signed [17:0] frontend_hidden_error_q016;
    logic frontend_arithmetic_overflow;
    logic frontend_internal_error;
    logic frontend_window_done;
    logic frontend_busy;

    logic learning_cfg_ready;
    logic learning_cfg_error;
    logic learning_active;
    logic learning_release_ready;
    logic learning_hidden_error_error;
    logic learning_observation_done_ready;
    logic learning_window_ready;
    logic learning_window_done;
    logic learning_window_busy;
    logic learning_context_error;

    logic cfg_legal;
    logic cfg_fire;
    logic shared_cfg_valid;
    logic release_fire;
    logic frontend_done_seen_q;
    logic learning_observation_fire;
    logic learning_window_fire;
    logic cfg_reject_q;
    logic [PARAM_ADDR_WIDTH:0] element_count_ext;
    logic [PARAM_ADDR_WIDTH:0] end_exclusive_ext;
    logic [VOTE_WIDTH:0] twice_threshold_ext;
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
        end_exclusive_ext = {1'b0, i_cfg_parameter_base} + element_count_ext;
        twice_threshold_ext = {1'b0, i_cfg_vote_threshold} +
            {1'b0, i_cfg_vote_threshold};
        cfg_legal = (i_cfg_batch_count == (SAMPLE_ID_WIDTH+1)'(16)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= (HIDDEN_INDEX_WIDTH+1)'(MAX_HIDDEN)) &&
            (i_cfg_feature_count != '0) &&
            (i_cfg_feature_count <= (FEATURE_INDEX_WIDTH+1)'(MAX_FEATURES)) &&
            (hidden_stride_ext >=
             (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count)) &&
            (i_cfg_target_dimension != '0) &&
            (i_cfg_target_dimension <=
             (TARGET_INDEX_WIDTH+1)'(MAX_TARGET_DIMENSION)) &&
            (element_count_ext <= ADDRESS_SPACE_LIMIT) &&
            (end_exclusive_ext <= ADDRESS_SPACE_LIMIT) &&
            (i_cfg_vote_threshold != '0) &&
            ({1'b0, i_cfg_vote_increment} < twice_threshold_ext) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));
        o_cfg_ready = (state_q == STATE_IDLE) && frontend_cfg_ready &&
            learning_cfg_ready;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        shared_cfg_valid = cfg_fire && cfg_legal;
        o_cfg_error = cfg_reject_q || frontend_cfg_error || learning_cfg_error;
        o_active = (state_q != STATE_IDLE) && frontend_active && learning_active;
        o_release_ready = (state_q == STATE_OBSERVE) &&
            !frontend_done_seen_q && frontend_release_ready &&
            learning_release_ready;
        release_fire = i_release && o_release_ready;

        learning_observation_fire = (state_q == STATE_OBSERVE) &&
            (frontend_done_seen_q || frontend_window_done) &&
            learning_observation_done_ready;
        learning_window_fire = (state_q == STATE_START_UPDATE) &&
            learning_window_ready;
        o_window_done = learning_window_done;
        o_window_busy = frontend_busy || learning_window_busy ||
            (state_q == STATE_START_UPDATE);
        o_context_error = frontend_internal_error || learning_context_error ||
            learning_hidden_error_error;
        o_arithmetic_overflow = frontend_arithmetic_overflow;
    end

    unified_fc_tp_w0_hidden_error_frontend #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN),
        .MAX_TARGET_DIMENSION(MAX_TARGET_DIMENSION),
        .TARGET_TRACE_WIDTH(TARGET_TRACE_WIDTH),
        .BATCH_INDEX_WIDTH(SAMPLE_ID_WIDTH),
        .BATCH_COUNT_WIDTH(SAMPLE_ID_WIDTH + 1),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_INDEX_WIDTH + 1),
        .TARGET_INDEX_WIDTH(TARGET_INDEX_WIDTH),
        .TARGET_COUNT_WIDTH(TARGET_INDEX_WIDTH + 1)
    ) u_hidden_error_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(frontend_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_target_dimension(i_cfg_target_dimension),
        .o_cfg_error(frontend_cfg_error),
        .o_active(frontend_active),
        .i_release(release_fire),
        .o_release_ready(frontend_release_ready),
        .i_student_trace_valid(i_student_trace_valid && o_active),
        .o_student_trace_ready(o_student_trace_ready),
        .i_student_trace_sample(i_student_trace_sample),
        .i_student_trace_hidden(i_student_trace_hidden),
        .i_student_trace_q016(i_student_trace_q016),
        .o_student_trace_error(o_student_trace_error),
        .i_target_hidden_trace_valid(i_target_hidden_trace_valid && o_active),
        .o_target_hidden_trace_ready(o_target_hidden_trace_ready),
        .i_target_hidden_trace_sample(i_target_hidden_trace_sample),
        .i_target_hidden_trace_hidden(i_target_hidden_trace_hidden),
        .i_target_hidden_trace_q016(i_target_hidden_trace_q016),
        .o_target_hidden_trace_error(o_target_hidden_trace_error),
        .i_target_input_trace_valid(i_target_input_trace_valid && o_active),
        .o_target_input_trace_ready(o_target_input_trace_ready),
        .i_target_input_trace_sample(i_target_input_trace_sample),
        .i_target_input_trace_index(i_target_input_trace_index),
        .i_target_input_trace_q016(i_target_input_trace_q016),
        .o_target_input_trace_error(o_target_input_trace_error),
        .i_surrogate_valid(i_surrogate_valid && o_active),
        .o_surrogate_ready(o_surrogate_ready),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_hidden(i_surrogate_hidden),
        .i_surrogate_q016(i_surrogate_q016),
        .o_surrogate_error(o_surrogate_error),
        .i_observation_done(i_observation_done && o_active),
        .o_observation_done_ready(o_observation_done_ready),
        .o_hidden_error_valid(frontend_hidden_error_valid),
        .i_hidden_error_ready(frontend_hidden_error_ready),
        .o_hidden_error_sample(frontend_hidden_error_sample),
        .o_hidden_error_hidden(frontend_hidden_error_hidden),
        .o_hidden_error_q016(frontend_hidden_error_q016),
        .o_arithmetic_overflow(frontend_arithmetic_overflow),
        .o_internal_error(frontend_internal_error),
        .o_window_done(frontend_window_done),
        .o_busy(frontend_busy)
    );

    unified_fc_tp_w0_learning_slice #(
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
        .PARAMETER_HIDDEN_STRIDE(PARAMETER_HIDDEN_STRIDE),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH)
    ) u_learning_slice (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_valid),
        .o_cfg_ready(learning_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_feature_count(i_cfg_feature_count),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(learning_cfg_error),
        .o_active(learning_active),
        .i_release(release_fire),
        .o_release_ready(learning_release_ready),
        .i_hidden_error_valid(frontend_hidden_error_valid),
        .o_hidden_error_ready(frontend_hidden_error_ready),
        .i_hidden_error_sample_id(frontend_hidden_error_sample),
        .i_hidden_error_hidden_index(frontend_hidden_error_hidden),
        .i_hidden_error_value(frontend_hidden_error_q016),
        .o_hidden_error_error(learning_hidden_error_error),
        .i_event_valid(i_event_valid && o_active && (state_q == STATE_OBSERVE)),
        .o_event_ready(o_event_ready),
        .i_event_sample_id(i_event_sample_id),
        .i_event_feature_index(i_event_feature_index),
        .o_event_error(o_event_error),
        .i_observation_done(learning_observation_fire),
        .o_observation_done_ready(learning_observation_done_ready),
        .i_window_emit(learning_window_fire),
        .o_window_ready(learning_window_ready),
        .o_window_done(learning_window_done),
        .o_window_busy(learning_window_busy),
        .o_context_error(learning_context_error),
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
        .o_internal_error(o_internal_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            frontend_done_seen_q <= 1'b0;
            cfg_reject_q <= 1'b0;
        end else begin
            cfg_reject_q <= cfg_fire && !cfg_legal;
            case (state_q)
                STATE_IDLE: begin
                    if (shared_cfg_valid) begin
                        frontend_done_seen_q <= 1'b0;
                        state_q <= STATE_OBSERVE;
                    end
                end

                STATE_OBSERVE: begin
                    if (release_fire) begin
                        frontend_done_seen_q <= 1'b0;
                        state_q <= STATE_IDLE;
                    end else begin
                        if (frontend_window_done)
                            frontend_done_seen_q <= 1'b1;
                        if (learning_observation_fire) begin
                            frontend_done_seen_q <= 1'b0;
                            state_q <= STATE_START_UPDATE;
                        end
                    end
                end

                STATE_START_UPDATE: begin
                    if (learning_window_fire)
                        state_q <= STATE_UPDATE;
                end

                STATE_UPDATE: begin
                    if (learning_window_done)
                        state_q <= STATE_OBSERVE;
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
