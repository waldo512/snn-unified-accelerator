`timescale 1ns/1ps
`default_nettype none

// SRNN M2 online-learning window up to the shared optimizer boundary. W_h1
// starts from pairwise TP observations; W_out starts from CE contributions.
// Both dense Q40 tensors share one ordered downstream stream and one commit.
module unified_srnn_tp_online_learning_window #(
    parameter int BATCH_MAX = 64,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int PARAM_ADDR_WIDTH = 24,
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
    input  logic [24:0]                              i_wh1_tp_surrogate_uq24,
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

    logic [PARAM_ADDR_WIDTH-1:0] wh1_element_count;
    logic shared_cfg_fire;
    logic shared_start_fire;
    logic abort_request;

    logic wh1_cfg_ready;
    logic wh1_cfg_error;
    logic wh1_start_ready;
    logic wh1_gradient_valid;
    logic wh1_gradient_ready;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_gradient_addr;
    logic signed [63:0] wh1_gradient_q40;
    logic wh1_gradient_last;
    logic wh1_busy;
    logic wh1_done;
    logic wh1_aborted;
    logic wh1_protocol_error;
    logic wh1_overflow;

    logic wout_cfg_ready;
    logic wout_cfg_error;
    logic wout_start_ready;
    logic wout_gradient_valid;
    logic wout_gradient_ready;
    logic [PARAM_ADDR_WIDTH-1:0] wout_gradient_addr;
    logic signed [63:0] wout_gradient_q40;
    logic wout_gradient_last;
    logic wout_busy;
    logic wout_done;
    logic wout_protocol_error;
    logic wout_overflow;

    logic adapter_cfg_ready;
    logic adapter_cfg_error;
    logic adapter_start_ready;
    logic adapter_busy;
    logic adapter_done;
    logic adapter_aborted;
    logic adapter_protocol_error;
    logic adapter_downstream_error;

    always_comb begin
        wh1_element_count = PARAM_ADDR_WIDTH'(
            i_cfg_hidden_count * i_cfg_backbone_count);
        o_cfg_ready = wh1_cfg_ready && wout_cfg_ready && adapter_cfg_ready;
        shared_cfg_fire = i_cfg_valid && o_cfg_ready;
        o_start_ready = wh1_start_ready && wout_start_ready &&
            adapter_start_ready && !i_cfg_valid;
        shared_start_fire = i_start_window && o_start_ready;

        abort_request = i_abort_window || wh1_protocol_error ||
            wh1_overflow || wout_protocol_error || wout_overflow ||
            adapter_protocol_error || adapter_downstream_error;
        o_cfg_error = wh1_cfg_error || wout_cfg_error || adapter_cfg_error;
        o_busy = wh1_busy || wout_busy || adapter_busy ||
            (adapter_busy && (wh1_done || wout_done));
        o_done = adapter_done;
        o_window_aborted = adapter_aborted || wh1_aborted;
        o_error = o_cfg_error || abort_request;
    end

    unified_srnn_tp_wh1_online_frontend #(
        .BATCH_MAX(BATCH_MAX),
        .HIDDEN_MAX(HIDDEN_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH)
    ) wh1_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(wh1_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .o_cfg_error(wh1_cfg_error),
        .i_start_window(shared_start_fire),
        .o_start_ready(wh1_start_ready),
        .i_abort_window(abort_request),
        .i_tp_contribution_valid(i_wh1_tp_contribution_valid),
        .o_tp_contribution_ready(o_wh1_tp_contribution_ready),
        .i_tp_contribution_path(i_wh1_tp_contribution_path),
        .i_tp_contribution_sample_id(i_wh1_tp_contribution_sample_id),
        .i_tp_contribution_hidden_id(i_wh1_tp_contribution_hidden_id),
        .i_tp_contribution_column_id(i_wh1_tp_contribution_column_id),
        .i_tp_error_q40(i_wh1_tp_error_q40),
        .i_tp_opposite_trace_q14(i_wh1_tp_opposite_trace_q14),
        .i_tp_surrogate_uq24(i_wh1_tp_surrogate_uq24),
        .i_tp_contribution_last(i_wh1_tp_contribution_last),
        .i_backbone_valid(i_wh1_backbone_valid),
        .o_backbone_ready(o_wh1_backbone_ready),
        .i_backbone_path(i_wh1_backbone_path),
        .i_backbone_sample_id(i_wh1_backbone_sample_id),
        .i_backbone_id(i_wh1_backbone_id),
        .i_backbone_q14(i_wh1_backbone_q14),
        .i_backbone_last(i_wh1_backbone_last),
        .o_gradient_valid(wh1_gradient_valid),
        .i_gradient_ready(wh1_gradient_ready),
        .o_gradient_local_addr(wh1_gradient_addr),
        .o_gradient_q40(wh1_gradient_q40),
        .o_gradient_tensor_last(wh1_gradient_last),
        .o_busy(wh1_busy),
        .o_done(wh1_done),
        .o_window_aborted(wh1_aborted),
        .o_protocol_error(wh1_protocol_error),
        .o_overflow(wh1_overflow)
    );

    unified_srnn_tp_wout_gradient_reducer #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH)
    ) wout_reducer (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(wout_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_element_count(i_cfg_wout_element_count),
        .o_cfg_error(wout_cfg_error),
        .i_start(shared_start_fire),
        .o_start_ready(wout_start_ready),
        .i_abort(abort_request),
        .i_contribution_valid(i_wout_contribution_valid),
        .o_contribution_ready(o_wout_contribution_ready),
        .i_contribution_local_addr(i_wout_contribution_addr),
        .i_contribution_sample_id(i_wout_contribution_sample_id),
        .i_error_q40(i_wout_error_q40),
        .i_hidden_trace_q14(i_wout_hidden_trace_q14),
        .i_contribution_tensor_last(i_wout_contribution_last),
        .o_gradient_valid(wout_gradient_valid),
        .i_gradient_ready(wout_gradient_ready),
        .o_gradient_local_addr(wout_gradient_addr),
        .o_gradient_q40(wout_gradient_q40),
        .o_gradient_tensor_last(wout_gradient_last),
        .o_busy(wout_busy),
        .o_done(wout_done),
        .o_protocol_error(wout_protocol_error),
        .o_overflow(wout_overflow)
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
        .i_cfg_wh1_element_count(wh1_element_count),
        .i_cfg_wout_element_count(i_cfg_wout_element_count),
        .o_cfg_error(adapter_cfg_error),
        .i_start_window(shared_start_fire),
        .o_start_ready(adapter_start_ready),
        .i_abort_window(abort_request),
        .i_wh1_gradient_valid(wh1_gradient_valid),
        .o_wh1_gradient_ready(wh1_gradient_ready),
        .i_wh1_gradient_addr(wh1_gradient_addr),
        .i_wh1_gradient_q40(wh1_gradient_q40),
        .i_wh1_gradient_last(wh1_gradient_last),
        .i_wout_gradient_valid(wout_gradient_valid),
        .o_wout_gradient_ready(wout_gradient_ready),
        .i_wout_gradient_addr(wout_gradient_addr),
        .i_wout_gradient_q40(wout_gradient_q40),
        .i_wout_gradient_last(wout_gradient_last),
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

endmodule

`default_nettype wire
