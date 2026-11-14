`timescale 1ns/1ps
`default_nettype none

// SRNN M2 learning-window integration boundary. W_h1 gradients arrive from
// the TP hidden path; W_out gradients are reduced from CE error/trace tokens.
// Both tensors share one optimizer request stream and one atomic commit fence.
module unified_srnn_tp_wout_reduced_learning_window #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int WINDOW_SEQUENCE_WIDTH = 32
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wh1_tensor_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wh1_element_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_start_window,
    output logic                                      o_start_ready,

    input  logic                                      i_wh1_gradient_valid,
    output logic                                      o_wh1_gradient_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wh1_gradient_addr,
    input  logic signed [63:0]                        i_wh1_gradient_q40,
    input  logic                                      i_wh1_gradient_last,

    input  logic                                      i_wout_contribution_valid,
    output logic                                      o_wout_contribution_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wout_contribution_addr,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_wout_contribution_sample_id,
    input  logic signed [63:0]                        i_wout_error_q40,
    input  logic signed [31:0]                        i_wout_hidden_trace_q14,
    input  logic                                      i_wout_contribution_last,

    output logic                                      o_gradient_valid,
    input  logic                                      i_gradient_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_gradient_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_gradient_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_gradient_local_addr,
    output logic signed [63:0]                        o_gradient_q40,
    output logic                                      o_gradient_nonzero,
    output logic                                      o_gradient_tensor_last,
    output logic                                      o_gradient_update_window_last,

    input  logic                                      i_tensor_complete_valid,
    output logic                                      o_tensor_complete_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_tensor_complete_id,
    input  logic                                      i_tensor_complete_error,

    output logic                                      o_commit_valid,
    input  logic                                      i_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_commit_profile_id,
    output logic [WINDOW_SEQUENCE_WIDTH-1:0]          o_commit_window_sequence,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_window_aborted,
    output logic                                      o_error
);

    logic reducer_cfg_ready;
    logic reducer_cfg_error;
    logic reducer_start_ready;
    logic reducer_gradient_valid;
    logic reducer_gradient_ready;
    logic [PARAM_ADDR_WIDTH-1:0] reducer_gradient_addr;
    logic signed [63:0] reducer_gradient_q40;
    logic reducer_gradient_last;
    logic reducer_busy;
    logic reducer_done;
    logic reducer_protocol_error;
    logic reducer_overflow;

    logic adapter_cfg_ready;
    logic adapter_cfg_error;
    logic adapter_start_ready;
    logic adapter_busy;
    logic adapter_done;
    logic adapter_aborted;
    logic adapter_protocol_error;
    logic adapter_downstream_error;
    logic shared_cfg_fire;
    logic shared_start_fire;
    logic reducer_abort;

    always_comb begin
        o_cfg_ready = reducer_cfg_ready && adapter_cfg_ready;
        shared_cfg_fire = i_cfg_valid && o_cfg_ready;
        o_start_ready = reducer_start_ready && adapter_start_ready &&
            !i_cfg_valid;
        shared_start_fire = i_start_window && o_start_ready;
        reducer_abort = reducer_protocol_error || reducer_overflow;
        o_cfg_error = reducer_cfg_error || adapter_cfg_error;
        o_busy = reducer_busy || adapter_busy;
        o_done = adapter_done;
        o_window_aborted = adapter_aborted;
        o_error = o_cfg_error || reducer_abort || adapter_protocol_error ||
            adapter_downstream_error;
    end

    unified_srnn_tp_wout_gradient_reducer wout_reducer (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(reducer_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_element_count(i_cfg_wout_element_count),
        .o_cfg_error(reducer_cfg_error),
        .i_start(shared_start_fire),
        .o_start_ready(reducer_start_ready),
        .i_abort(reducer_abort),
        .i_contribution_valid(i_wout_contribution_valid),
        .o_contribution_ready(o_wout_contribution_ready),
        .i_contribution_local_addr(i_wout_contribution_addr),
        .i_contribution_sample_id(i_wout_contribution_sample_id),
        .i_error_q40(i_wout_error_q40),
        .i_hidden_trace_q14(i_wout_hidden_trace_q14),
        .i_contribution_tensor_last(i_wout_contribution_last),
        .o_gradient_valid(reducer_gradient_valid),
        .i_gradient_ready(reducer_gradient_ready),
        .o_gradient_local_addr(reducer_gradient_addr),
        .o_gradient_q40(reducer_gradient_q40),
        .o_gradient_tensor_last(reducer_gradient_last),
        .o_busy(reducer_busy),
        .o_done(reducer_done),
        .o_protocol_error(reducer_protocol_error),
        .o_overflow(reducer_overflow)
    );

    unified_srnn_tp_dual_tensor_gradient_adapter #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .WINDOW_SEQUENCE_WIDTH(WINDOW_SEQUENCE_WIDTH)
    ) gradient_adapter (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(adapter_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_wh1_tensor_id(i_cfg_wh1_tensor_id),
        .i_cfg_wout_tensor_id(i_cfg_wout_tensor_id),
        .i_cfg_wh1_element_count(i_cfg_wh1_element_count),
        .i_cfg_wout_element_count(i_cfg_wout_element_count),
        .o_cfg_error(adapter_cfg_error),
        .i_start_window(shared_start_fire),
        .o_start_ready(adapter_start_ready),
        .i_abort_window(reducer_abort),
        .i_wh1_gradient_valid(i_wh1_gradient_valid),
        .o_wh1_gradient_ready(o_wh1_gradient_ready),
        .i_wh1_gradient_addr(i_wh1_gradient_addr),
        .i_wh1_gradient_q40(i_wh1_gradient_q40),
        .i_wh1_gradient_last(i_wh1_gradient_last),
        .i_wout_gradient_valid(reducer_gradient_valid),
        .o_wout_gradient_ready(reducer_gradient_ready),
        .i_wout_gradient_addr(reducer_gradient_addr),
        .i_wout_gradient_q40(reducer_gradient_q40),
        .i_wout_gradient_last(reducer_gradient_last),
        .o_gradient_valid(o_gradient_valid),
        .i_gradient_ready(i_gradient_ready),
        .o_gradient_profile_id(o_gradient_profile_id),
        .o_gradient_tensor_id(o_gradient_tensor_id),
        .o_gradient_local_addr(o_gradient_local_addr),
        .o_gradient_q40(o_gradient_q40),
        .o_gradient_nonzero(o_gradient_nonzero),
        .o_gradient_tensor_last(o_gradient_tensor_last),
        .o_gradient_update_window_last(o_gradient_update_window_last),
        .i_tensor_complete_valid(i_tensor_complete_valid),
        .o_tensor_complete_ready(o_tensor_complete_ready),
        .i_tensor_complete_id(i_tensor_complete_id),
        .i_tensor_complete_error(i_tensor_complete_error),
        .o_commit_valid(o_commit_valid),
        .i_commit_ready(i_commit_ready),
        .o_commit_profile_id(o_commit_profile_id),
        .o_commit_window_sequence(o_commit_window_sequence),
        .o_busy(adapter_busy),
        .o_done(adapter_done),
        .o_window_aborted(adapter_aborted),
        .o_protocol_error(adapter_protocol_error),
        .o_downstream_error(adapter_downstream_error)
    );

    logic _unused_reducer_done;
    always_comb _unused_reducer_done = reducer_done;

endmodule

`default_nettype wire
