`timescale 1ns/1ps
`default_nettype none

// Wiring-only composition of final dispatch, physical drain, and error policy.
// Error state blocks new layer acceptance but is deliberately absent from the
// physical drain mask so fail-stop operation can still reach quiescence.
module unified_final_system_control_v1 (
    input  logic          i_clk,
    input  logic          i_rst_n,
    input  logic          i_clear_error,
    input  logic          i_abort,

    input  logic          i_layer_valid,
    output logic          o_layer_ready,
    input  logic [3:0]    i_layer_type,
    output logic          o_fc_valid,
    input  logic          i_fc_ready,
    input  logic          i_fc_done,
    output logic          o_recurrent_valid,
    input  logic          i_recurrent_ready,
    input  logic          i_recurrent_done,
    output logic          o_scnn_valid,
    input  logic          i_scnn_ready,
    input  logic          i_scnn_done,
    output logic          o_output_valid,
    input  logic          i_output_ready,
    input  logic          i_output_done,
    output logic          o_layer_done,
    output logic          o_layer_accept,
    output logic [1:0]    o_requested_owner,
    output logic [3:0]    o_active_layer_type,
    output logic          o_owner_active,
    output logic          o_dispatch_busy,

    input  logic          i_fc_busy,
    input  logic          i_extension_busy,
    input  logic          i_dynamic_output_busy,
    input  logic          i_scnn_finalize_busy,
    input  logic          i_scnn_li_busy,
    input  logic          i_scratchpad_busy,
    input  logic          i_prefetch_busy,
    input  logic          i_writeback_busy,
    input  logic          i_ddr_cmd_pending,
    input  logic          i_ddr_response_pending,
    input  logic          i_ddr_cdc_busy,
    input  logic [2:0]    i_ddr_outstanding,
    input  logic          i_w2_commit_pending,
    input  logic          i_w2_snapshot_busy,
    input  logic          i_extension_command_pending,
    input  logic          i_result_valid,
    input  logic          i_result_ready,

    input  logic          i_control_error,
    input  logic [7:0]    i_control_code,
    input  logic          i_datapath_error,
    input  logic [7:0]    i_datapath_code,
    input  logic          i_scnn_layer_error,
    input  logic [7:0]    i_scnn_layer_code,
    input  logic          i_scnn_output_error,
    input  logic [7:0]    i_scnn_output_code,
    input  logic          i_memory_error,
    input  logic [7:0]    i_memory_code,

    output logic          o_datapath_quiescent,
    output logic          o_layer_pending,
    output logic          o_drain_owner_active,
    output logic          o_owner_tracking_consistent,
    output logic          o_result_pending,
    output logic          o_abort_cleanup_pending,
    output logic [18:0]   o_physical_blocker_mask,
    output logic          o_dispatch_error,
    output logic [7:0]    o_dispatch_error_code,
    output logic          o_fail_stop,
    output logic          o_sticky_error,
    output logic [2:0]    o_error_source,
    output logic [7:0]    o_error_code
);
    logic layer_valid_allowed;
    logic dispatch_layer_ready;

    assign layer_valid_allowed = i_layer_valid && !o_fail_stop;
    assign o_layer_ready = dispatch_layer_ready && !o_fail_stop;
    assign o_owner_tracking_consistent =
        o_owner_active == o_drain_owner_active;

    unified_final_layer_dispatch_v1 u_layer_dispatch (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_abort(i_abort),
        .i_layer_valid(layer_valid_allowed),
        .o_layer_ready(dispatch_layer_ready),
        .i_layer_type(i_layer_type),
        .o_fc_valid(o_fc_valid),
        .i_fc_ready(i_fc_ready),
        .i_fc_done(i_fc_done),
        .o_recurrent_valid(o_recurrent_valid),
        .i_recurrent_ready(i_recurrent_ready),
        .i_recurrent_done(i_recurrent_done),
        .o_scnn_valid(o_scnn_valid),
        .i_scnn_ready(i_scnn_ready),
        .i_scnn_done(i_scnn_done),
        .o_output_valid(o_output_valid),
        .i_output_ready(i_output_ready),
        .i_output_done(i_output_done),
        .o_layer_done(o_layer_done),
        .o_layer_accept(o_layer_accept),
        .o_requested_owner(o_requested_owner),
        .o_active_layer_type(o_active_layer_type),
        .o_owner_active(o_owner_active),
        .o_busy(o_dispatch_busy),
        .o_sticky_error(o_dispatch_error),
        .o_error_code(o_dispatch_error_code)
    );

    unified_final_drain_status_v1 u_drain_status (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_fc_busy(i_fc_busy),
        .i_extension_busy(i_extension_busy),
        .i_dynamic_output_busy(i_dynamic_output_busy),
        .i_scnn_finalize_busy(i_scnn_finalize_busy),
        .i_scnn_li_busy(i_scnn_li_busy),
        .i_dispatch_busy(o_dispatch_busy),
        .i_layer_valid(layer_valid_allowed),
        .i_scratchpad_busy(i_scratchpad_busy),
        .i_prefetch_busy(i_prefetch_busy),
        .i_writeback_busy(i_writeback_busy),
        .i_ddr_cmd_pending(i_ddr_cmd_pending),
        .i_ddr_response_pending(i_ddr_response_pending),
        .i_ddr_cdc_busy(i_ddr_cdc_busy),
        .i_ddr_outstanding(i_ddr_outstanding),
        .i_w2_commit_pending(i_w2_commit_pending),
        .i_w2_snapshot_busy(i_w2_snapshot_busy),
        .i_extension_command_pending(i_extension_command_pending),
        .i_result_valid(i_result_valid),
        .i_result_ready(i_result_ready),
        .o_datapath_quiescent(o_datapath_quiescent),
        .o_layer_pending(o_layer_pending),
        .o_owner_active(o_drain_owner_active),
        .o_result_pending(o_result_pending),
        .o_abort_cleanup_pending(o_abort_cleanup_pending),
        .o_physical_blocker_mask(o_physical_blocker_mask)
    );

    unified_final_error_aggregator_v1 u_error_aggregator (
        .i_control_error(i_control_error),
        .i_control_code(i_control_code),
        .i_dispatch_error(o_dispatch_error),
        .i_dispatch_code(o_dispatch_error_code),
        .i_datapath_error(i_datapath_error),
        .i_datapath_code(i_datapath_code),
        .i_scnn_layer_error(i_scnn_layer_error),
        .i_scnn_layer_code(i_scnn_layer_code),
        .i_scnn_output_error(i_scnn_output_error),
        .i_scnn_output_code(i_scnn_output_code),
        .i_memory_error(i_memory_error),
        .i_memory_code(i_memory_code),
        .o_fail_stop(o_fail_stop),
        .o_sticky_error(o_sticky_error),
        .o_error_source(o_error_source),
        .o_error_code(o_error_code)
    );
endmodule

`default_nettype wire
