`timescale 1ns/1ps
`default_nettype none

// Complete SRNN M2 hardware-native direct-update transaction. The topology
// frontend generates dense Q40 W_h1/W_out gradients; one shared two-tensor RMW
// backend converts their signs into profile-configured residual updates. The
// commit remains owned by the topology window and cannot occur before both
// tensor writeback completions return.
module unified_srnn_tp_direct_online_learning_engine #(
    parameter int BATCH_MAX = 64,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16,
    parameter int WINDOW_SEQUENCE_WIDTH = 32
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wh1_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wh1_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wh1_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wh1_threshold,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_base,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_wout_threshold,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_max,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_start_window,
    output logic                                      o_start_ready,
    input  logic                                      i_abort_window,

    input  logic                                      i_wh1_tp_contribution_valid,
    output logic                                      o_wh1_tp_contribution_ready,
    input  logic                                      i_wh1_tp_contribution_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wh1_tp_contribution_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_wh1_tp_contribution_hidden_id,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wh1_tp_contribution_column_id,
    input  logic signed [63:0]                        i_wh1_tp_error_q40,
    input  logic signed [31:0]                        i_wh1_tp_opposite_trace_q14,
    input  logic [24:0]                               i_wh1_tp_surrogate_uq24,
    input  logic                                      i_wh1_tp_contribution_last,

    input  logic                                      i_wh1_backbone_valid,
    output logic                                      o_wh1_backbone_ready,
    input  logic                                      i_wh1_backbone_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wh1_backbone_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_wh1_backbone_id,
    input  logic signed [31:0]                        i_wh1_backbone_q14,
    input  logic                                      i_wh1_backbone_last,

    input  logic                                      i_wout_contribution_valid,
    output logic                                      o_wout_contribution_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wout_contribution_addr,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wout_contribution_sample_id,
    input  logic signed [63:0]                        i_wout_error_q40,
    input  logic signed [31:0]                        i_wout_hidden_trace_q14,
    input  logic                                      i_wout_contribution_last,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_read_tag,
    input  logic                                      i_read_response_valid,
    output logic                                      o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]              i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_read_response_residual,
    input  logic                                      i_read_response_error,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_result_tag,
    output logic signed [CODE_WIDTH-1:0]              o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_result_residual,
    output logic                                      o_write_code_enable,
    output logic                                      o_write_residual_enable,
    output logic                                      o_write_code_fire,
    output logic                                      o_write_residual_fire,
    output logic                                      o_dirty,
    output logic                                      o_saturated,

    output logic                                      o_commit_valid,
    input  logic                                      i_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_commit_profile_id,
    output logic [WINDOW_SEQUENCE_WIDTH-1:0]          o_commit_window_sequence,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_window_aborted,
    output logic                                      o_error,
    output logic [PARAM_ADDR_WIDTH:0]                 o_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0]                 o_update_count
);

    logic [PARAM_ADDR_WIDTH-1:0] wh1_element_count;
    logic window_cfg_ready;
    logic window_cfg_error;
    logic backend_cfg_ready;
    logic backend_cfg_error;
    logic shared_cfg_fire;
    logic window_start_ready;
    logic backend_start_ready;
    logic shared_start_fire;
    logic shared_abort;
    logic gradient_valid;
    logic gradient_ready;
    logic [PROFILE_ID_WIDTH-1:0] gradient_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] gradient_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] gradient_local_addr;
    logic signed [63:0] gradient_q40;
    logic gradient_nonzero;
    logic gradient_tensor_last;
    logic gradient_window_last;
    logic tensor_complete_valid;
    logic tensor_complete_ready;
    logic [TENSOR_ID_WIDTH-1:0] tensor_complete_id;
    logic tensor_complete_error;
    logic window_busy;
    logic window_done;
    logic window_aborted;
    logic window_error;
    logic backend_busy;
    logic backend_done;
    logic backend_aborted;
    logic backend_protocol_error;
    logic backend_store_error;
    logic backend_internal_error;

    always_comb begin
        wh1_element_count = PARAM_ADDR_WIDTH'(
            i_cfg_hidden_count * i_cfg_backbone_count);
        o_cfg_ready = window_cfg_ready && backend_cfg_ready;
        shared_cfg_fire = i_cfg_valid && o_cfg_ready;
        o_start_ready = window_start_ready && backend_start_ready &&
            !i_cfg_valid;
        shared_start_fire = i_start_window && o_start_ready;
        shared_abort = i_abort_window || backend_protocol_error ||
            backend_store_error || backend_internal_error || window_aborted;
        o_cfg_error = window_cfg_error || backend_cfg_error;
        o_busy = window_busy || backend_busy || backend_done;
        o_done = window_done;
        o_window_aborted = window_aborted || backend_aborted;
        o_error = o_cfg_error || window_error || backend_protocol_error ||
            backend_store_error || backend_internal_error;
    end

    unified_srnn_tp_online_learning_window #(
        .BATCH_MAX(BATCH_MAX),
        .HIDDEN_MAX(HIDDEN_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .WINDOW_SEQUENCE_WIDTH(WINDOW_SEQUENCE_WIDTH)
    ) learning_window (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(window_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_wh1_tensor_id(i_cfg_wh1_tensor_id),
        .i_cfg_wout_tensor_id(i_cfg_wout_tensor_id),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_wout_element_count(i_cfg_wout_element_count),
        .o_cfg_error(window_cfg_error),
        .i_start_window(shared_start_fire),
        .o_start_ready(window_start_ready),
        .i_abort_window(shared_abort),
        .i_wh1_tp_contribution_valid(i_wh1_tp_contribution_valid),
        .o_wh1_tp_contribution_ready(o_wh1_tp_contribution_ready),
        .i_wh1_tp_contribution_path(i_wh1_tp_contribution_path),
        .i_wh1_tp_contribution_sample_id(i_wh1_tp_contribution_sample_id),
        .i_wh1_tp_contribution_hidden_id(i_wh1_tp_contribution_hidden_id),
        .i_wh1_tp_contribution_column_id(i_wh1_tp_contribution_column_id),
        .i_wh1_tp_error_q40(i_wh1_tp_error_q40),
        .i_wh1_tp_opposite_trace_q14(i_wh1_tp_opposite_trace_q14),
        .i_wh1_tp_surrogate_uq24(i_wh1_tp_surrogate_uq24),
        .i_wh1_tp_contribution_last(i_wh1_tp_contribution_last),
        .i_wh1_backbone_valid(i_wh1_backbone_valid),
        .o_wh1_backbone_ready(o_wh1_backbone_ready),
        .i_wh1_backbone_path(i_wh1_backbone_path),
        .i_wh1_backbone_sample_id(i_wh1_backbone_sample_id),
        .i_wh1_backbone_id(i_wh1_backbone_id),
        .i_wh1_backbone_q14(i_wh1_backbone_q14),
        .i_wh1_backbone_last(i_wh1_backbone_last),
        .i_wout_contribution_valid(i_wout_contribution_valid),
        .o_wout_contribution_ready(o_wout_contribution_ready),
        .i_wout_contribution_addr(i_wout_contribution_addr),
        .i_wout_contribution_sample_id(i_wout_contribution_sample_id),
        .i_wout_error_q40(i_wout_error_q40),
        .i_wout_hidden_trace_q14(i_wout_hidden_trace_q14),
        .i_wout_contribution_last(i_wout_contribution_last),
        .o_gradient_valid(gradient_valid),
        .i_gradient_ready(gradient_ready),
        .o_gradient_profile_id(gradient_profile_id),
        .o_gradient_tensor_id(gradient_tensor_id),
        .o_gradient_local_addr(gradient_local_addr),
        .o_gradient_q40(gradient_q40),
        .o_gradient_nonzero(gradient_nonzero),
        .o_gradient_tensor_last(gradient_tensor_last),
        .o_gradient_update_window_last(gradient_window_last),
        .i_tensor_complete_valid(tensor_complete_valid),
        .o_tensor_complete_ready(tensor_complete_ready),
        .i_tensor_complete_id(tensor_complete_id),
        .i_tensor_complete_error(tensor_complete_error),
        .o_commit_valid(o_commit_valid),
        .i_commit_ready(i_commit_ready),
        .o_commit_profile_id(o_commit_profile_id),
        .o_commit_window_sequence(o_commit_window_sequence),
        .o_busy(window_busy),
        .o_done(window_done),
        .o_window_aborted(window_aborted),
        .o_error(window_error)
    );

    unified_tp_two_tensor_direct_rmw_backend #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) direct_backend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(backend_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor0_id(i_cfg_wh1_tensor_id),
        .i_cfg_tensor0_base(i_cfg_wh1_base),
        .i_cfg_tensor0_count(wh1_element_count),
        .i_cfg_tensor0_increment(i_cfg_wh1_increment),
        .i_cfg_tensor0_threshold(i_cfg_wh1_threshold),
        .i_cfg_tensor1_id(i_cfg_wout_tensor_id),
        .i_cfg_tensor1_base(i_cfg_wout_base),
        .i_cfg_tensor1_count(i_cfg_wout_element_count),
        .i_cfg_tensor1_increment(i_cfg_wout_increment),
        .i_cfg_tensor1_threshold(i_cfg_wout_threshold),
        .i_cfg_code_min(i_cfg_code_min),
        .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(backend_cfg_error),
        .i_start(shared_start_fire),
        .o_start_ready(backend_start_ready),
        .i_abort(window_error || i_abort_window),
        .i_gradient_valid(gradient_valid),
        .o_gradient_ready(gradient_ready),
        .i_gradient_profile_id(gradient_profile_id),
        .i_gradient_tensor_id(gradient_tensor_id),
        .i_gradient_local_addr(gradient_local_addr),
        .i_gradient_q40(gradient_q40),
        .i_gradient_nonzero(gradient_nonzero),
        .i_gradient_tensor_last(gradient_tensor_last),
        .i_gradient_window_last(gradient_window_last),
        .o_tensor_complete_valid(tensor_complete_valid),
        .i_tensor_complete_ready(tensor_complete_ready),
        .o_tensor_complete_id(tensor_complete_id),
        .o_tensor_complete_error(tensor_complete_error),
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
        .o_busy(backend_busy),
        .o_done(backend_done),
        .o_aborted(backend_aborted),
        .o_protocol_error(backend_protocol_error),
        .o_store_error(backend_store_error),
        .o_internal_error(backend_internal_error),
        .o_zero_gradient_count(o_zero_gradient_count),
        .o_update_count(o_update_count)
    );

endmodule

`default_nettype wire
