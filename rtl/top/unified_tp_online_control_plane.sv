`timescale 1ns/1ps
`default_nettype none

// Topology-neutral control plane for the unified online-learning accelerator.
// Parameter data remains in each topology's native memory organization; this
// module serializes ownership, run dispatch, abort and physical drain only.
module unified_tp_online_control_plane #(
    parameter logic [6:0] PROFILE_FC_H160 = 7'h01,
    parameter logic [6:0] PROFILE_FC_H480 = 7'h02,
    parameter logic [6:0] PROFILE_SRNN_M2 = 7'h10,
    parameter logic [6:0] PROFILE_SCNN_VGG9 = 7'h20
) (
    input  logic       i_clk,
    input  logic       i_rst_n,
    input  logic       i_clear_error,
    input  logic       i_abort,

    input  logic       i_profile_request_valid,
    input  logic [6:0] i_profile_request_code,
    input  logic       i_profile_learning_enable,
    output logic       o_profile_aligned,
    output logic       o_start_allowed,
    output logic       o_profile_switch_done,
    output logic       o_profile_switch_busy,
    output logic [6:0] o_active_profile_code,
    output logic [15:0] o_profile_version,
    output logic       o_active_learning_enable,

    input  logic       i_run_valid,
    output logic       o_run_ready,
    input  logic [6:0] i_run_profile_code,
    output logic       o_run_done,
    output logic       o_run_aborted,
    output logic       o_run_busy,

    output logic [2:0] o_client_acquire_valid,
    input  logic [2:0] i_client_acquire_ready,
    output logic [2:0] o_client_release_valid,
    input  logic [2:0] i_client_release_ready,
    output logic [2:0] o_client_start_valid,
    input  logic [2:0] i_client_start_ready,
    output logic [2:0] o_client_abort,
    input  logic [2:0] i_client_done,
    input  logic [2:0] i_client_aborted,
    input  logic [2:0] i_client_quiescent,
    input  logic [2:0] i_client_error,

    // Each native parameter service reports its physical state independently.
    // A profile switch and run retirement wait for every accepted transaction.
    input  logic [2:0] i_parameter_quiescent,
    input  logic [2:0] i_parameter_error,

    output logic [2:0] o_active_client,
    output logic [2:0] o_parameter_owner,
    output logic [2:0] o_topology_learning_enable,
    output logic       o_control_quiescent,
    output logic       o_system_quiescent,
    output logic       o_system_error,
    output logic [7:0] o_control_error_code,
    output logic       o_dispatch_protocol_error
);
    logic [1:0] active_owner;
    logic control_start_allowed;
    logic control_sticky_error;
    logic dispatch_quiescent;
    logic [2:0] topology_quiescent;
    logic [2:0] topology_error;
    logic exact_run_profile_aligned;
    logic learning_enable_q;
    logic learning_mode_aligned;
    logic [6:0] dispatch_active_profile_code;
    logic dispatch_active_learning_enable;
    logic dispatch_context_mismatch;
    logic owner_encoding_mismatch;
    logic parameter_plane_quiescent;

    assign topology_quiescent = i_client_quiescent &
        i_parameter_quiescent;
    assign topology_error = i_client_error | i_parameter_error;
    assign parameter_plane_quiescent = &i_parameter_quiescent;
    assign learning_mode_aligned =
        learning_enable_q == i_profile_learning_enable;
    assign exact_run_profile_aligned = o_profile_aligned &&
        o_active_profile_code == i_run_profile_code;
    assign o_active_learning_enable = learning_enable_q;
    assign o_start_allowed = control_start_allowed && learning_mode_aligned;
    assign dispatch_context_mismatch = o_run_busy &&
        (dispatch_active_profile_code != o_active_profile_code ||
         dispatch_active_learning_enable != learning_enable_q);
    assign owner_encoding_mismatch =
        (o_parameter_owner == 3'b001 && active_owner != 2'd1) ||
        (o_parameter_owner == 3'b010 && active_owner != 2'd2) ||
        (o_parameter_owner == 3'b100 && active_owner != 2'd3) ||
        (o_parameter_owner == 3'b000 && active_owner != 2'd0);
    assign o_system_error = control_sticky_error ||
        o_dispatch_protocol_error || dispatch_context_mismatch ||
        owner_encoding_mismatch || (|topology_error);
    assign o_control_quiescent = dispatch_quiescent &&
        !o_profile_switch_busy && !o_system_error;
    assign o_system_quiescent = o_control_quiescent &&
        parameter_plane_quiescent;

    // The requested learning mode is configuration state. Retain it from
    // profile admission through run completion and physical parameter drain.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            learning_enable_q <= 1'b0;
        else if (i_profile_request_valid && dispatch_quiescent &&
                 parameter_plane_quiescent && !(|topology_error))
            learning_enable_q <= i_profile_learning_enable;
    end

    unified_tp_train_infer_system_control #(
        .PROFILE_FC_H160(PROFILE_FC_H160),
        .PROFILE_FC_H480(PROFILE_FC_H480),
        .PROFILE_SRNN_M2(PROFILE_SRNN_M2),
        .PROFILE_SCNN_VGG9(PROFILE_SCNN_VGG9)
    ) u_system_control (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_requested_profile_valid(i_profile_request_valid),
        .i_requested_profile_code(i_profile_request_code),
        .i_inference_global_idle(dispatch_quiescent),
        .i_learning_enable(learning_enable_q),
        .i_backend_quiescent(parameter_plane_quiescent),
        .i_backend_error(|i_parameter_error),
        .o_fc_acquire_valid(o_client_acquire_valid[0]),
        .i_fc_acquire_ready(i_client_acquire_ready[0] &&
            i_parameter_quiescent[0] && !topology_error[0]),
        .o_srnn_acquire_valid(o_client_acquire_valid[1]),
        .i_srnn_acquire_ready(i_client_acquire_ready[1] &&
            i_parameter_quiescent[1] && !topology_error[1]),
        .o_scnn_acquire_valid(o_client_acquire_valid[2]),
        .i_scnn_acquire_ready(i_client_acquire_ready[2] &&
            i_parameter_quiescent[2] && !topology_error[2]),
        .o_fc_release_valid(o_client_release_valid[0]),
        .i_fc_release_ready(i_client_release_ready[0] &&
            topology_quiescent[0] && !topology_error[0]),
        .o_srnn_release_valid(o_client_release_valid[1]),
        .i_srnn_release_ready(i_client_release_ready[1] &&
            topology_quiescent[1] && !topology_error[1]),
        .o_scnn_release_valid(o_client_release_valid[2]),
        .i_scnn_release_ready(i_client_release_ready[2] &&
            topology_quiescent[2] && !topology_error[2]),
        .i_fc_quiescent(topology_quiescent[0]),
        .i_srnn_quiescent(topology_quiescent[1]),
        .i_scnn_quiescent(topology_quiescent[2]),
        .i_fc_error(topology_error[0]),
        .i_srnn_error(topology_error[1]),
        .i_scnn_error(topology_error[2]),
        .o_fc_parameter_owner(o_parameter_owner[0]),
        .o_srnn_parameter_owner(o_parameter_owner[1]),
        .o_scnn_parameter_owner(o_parameter_owner[2]),
        .o_fc_learning_enable(o_topology_learning_enable[0]),
        .o_srnn_learning_enable(o_topology_learning_enable[1]),
        .o_scnn_learning_enable(o_topology_learning_enable[2]),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(control_start_allowed),
        .o_profile_switch_done(o_profile_switch_done),
        .o_switch_busy(o_profile_switch_busy),
        .o_active_owner(active_owner),
        .o_active_profile_code(o_active_profile_code),
        .o_profile_version(o_profile_version),
        .o_sticky_error(control_sticky_error),
        .o_error_code(o_control_error_code)
    );

    unified_tp_online_run_dispatch #(
        .PROFILE_FC_H160(PROFILE_FC_H160),
        .PROFILE_FC_H480(PROFILE_FC_H480),
        .PROFILE_SRNN_M2(PROFILE_SRNN_M2),
        .PROFILE_SCNN_VGG9(PROFILE_SCNN_VGG9)
    ) u_dispatch (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_profile_aligned(exact_run_profile_aligned),
        .i_start_allowed(o_start_allowed),
        .i_parameter_owner(o_parameter_owner),
        .i_topology_learning_enable(o_topology_learning_enable),
        .i_run_valid(i_run_valid), .o_run_ready(o_run_ready),
        .i_profile_code(i_run_profile_code),
        .o_client_start_valid(o_client_start_valid),
        .i_client_start_ready(i_client_start_ready),
        .o_client_abort(o_client_abort),
        .i_client_done(i_client_done),
        .i_client_aborted(i_client_aborted),
        .i_client_quiescent(topology_quiescent),
        .i_client_error(topology_error),
        .o_active_client(o_active_client),
        .o_active_profile_code(dispatch_active_profile_code),
        .o_active_learning_enable(dispatch_active_learning_enable),
        .o_run_done(o_run_done), .o_run_aborted(o_run_aborted),
        .o_busy(o_run_busy), .o_quiescent(dispatch_quiescent),
        .o_protocol_error(o_dispatch_protocol_error)
    );
endmodule

`default_nettype wire
