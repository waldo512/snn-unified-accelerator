`timescale 1ns/1ps
`default_nettype none

// Common production control and parameter boundary for FC, SRNN and SCNN TP
// clients. Topology datapaths remain external clients; this module owns only
// profile selection, run serialization and the single shared parameter store.
module unified_tp_online_system_top #(
    parameter int DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter logic [6:0] PROFILE_FC_H160 = 7'h01,
    parameter logic [6:0] PROFILE_FC_H480 = 7'h02,
    parameter logic [6:0] PROFILE_SRNN_M2 = 7'h10,
    parameter logic [6:0] PROFILE_SCNN_VGG9 = 7'h20
) (
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_clear_error,
    input  logic i_abort,

    input  logic i_profile_request_valid,
    input  logic [6:0] i_profile_request_code,
    input  logic i_profile_learning_enable,
    output logic o_profile_aligned,
    output logic o_start_allowed,
    output logic o_profile_switch_done,
    output logic o_profile_switch_busy,
    output logic [6:0] o_active_profile_code,
    output logic [15:0] o_profile_version,
    output logic o_active_learning_enable,

    input  logic i_run_valid,
    output logic o_run_ready,
    input  logic [6:0] i_run_profile_code,
    output logic o_run_done,
    output logic o_run_aborted,
    output logic o_run_busy,

    output logic [2:0] o_client_start_valid,
    input  logic [2:0] i_client_start_ready,
    output logic [2:0] o_client_abort,
    input  logic [2:0] i_client_done,
    input  logic [2:0] i_client_aborted,
    input  logic [2:0] i_client_quiescent,
    input  logic [2:0] i_client_error,
    output logic [2:0] o_active_client,
    output logic [2:0] o_parameter_owner,
    output logic [2:0] o_topology_learning_enable,

    input  logic [2:0] i_load_valid,
    output logic [2:0] o_load_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0] i_load_addr,
    input  logic signed [3*CODE_WIDTH-1:0] i_load_code,
    input  logic signed [3*RESIDUAL_WIDTH-1:0] i_load_residual,

    input  logic [2:0] i_learning_read_valid,
    output logic [2:0] o_learning_read_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0] i_learning_read_addr,
    input  logic [3*TAG_WIDTH-1:0] i_learning_read_tag,
    output logic [2:0] o_learning_response_valid,
    input  logic [2:0] i_learning_response_ready,
    output logic [3*TAG_WIDTH-1:0] o_learning_response_tag,
    output logic signed [3*CODE_WIDTH-1:0]
        o_learning_response_code,
    output logic signed [3*RESIDUAL_WIDTH-1:0]
        o_learning_response_residual,
    output logic [2:0] o_learning_response_error,

    input  logic [2:0] i_learning_write_valid,
    output logic [2:0] o_learning_write_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0] i_learning_write_addr,
    input  logic signed [3*CODE_WIDTH-1:0] i_learning_write_code,
    input  logic signed [3*RESIDUAL_WIDTH-1:0]
        i_learning_write_residual,
    input  logic [2:0] i_learning_write_code_enable,
    input  logic [2:0] i_learning_write_residual_enable,

    input  logic [2:0] i_forward_read_valid,
    output logic [2:0] o_forward_read_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0] i_forward_read_addr,
    input  logic [3*TAG_WIDTH-1:0] i_forward_read_tag,
    output logic [2:0] o_forward_response_valid,
    input  logic [2:0] i_forward_response_ready,
    output logic [3*TAG_WIDTH-1:0] o_forward_response_tag,
    output logic signed [3*CODE_WIDTH-1:0] o_forward_response_code,
    output logic [2:0] o_forward_response_error,

    output logic o_backend_quiescent,
    output logic o_system_quiescent,
    output logic o_system_error,
    output logic [7:0] o_backend_error_code,
    output logic o_dispatch_protocol_error,
    output logic o_router_protocol_error,
    output logic o_store_protocol_error
);
    logic [2:0] acquire_valid;
    logic [2:0] release_valid;
    logic [1:0] active_owner;
    logic backend_start_allowed;
    logic backend_sticky_error;
    logic dispatch_quiescent;
    logic [2:0] dispatch_client_quiescent;
    logic exact_run_profile_aligned;
    logic learning_enable_q;
    logic learning_mode_aligned;
    logic [6:0] dispatch_active_profile_code;
    logic dispatch_active_learning_enable;
    logic dispatch_context_mismatch;
    logic owner_handshake_invalid;
    logic owner_encoding_mismatch;

    assign learning_mode_aligned =
        learning_enable_q == i_profile_learning_enable;
    assign exact_run_profile_aligned = o_profile_aligned &&
        o_active_profile_code == i_run_profile_code;

    // A run retires only after both the selected physical client and every
    // retained shared-store request/response have drained.
    assign dispatch_client_quiescent = i_client_quiescent &
        {3{o_backend_quiescent}};

    assign o_active_learning_enable = learning_enable_q;
    assign o_start_allowed = backend_start_allowed && learning_mode_aligned;
    assign dispatch_context_mismatch = o_run_busy &&
        (dispatch_active_profile_code != o_active_profile_code ||
         dispatch_active_learning_enable != learning_enable_q);
    assign owner_handshake_invalid =
        (acquire_valid[0] && acquire_valid[1]) ||
        (acquire_valid[0] && acquire_valid[2]) ||
        (acquire_valid[1] && acquire_valid[2]) ||
        (release_valid[0] && release_valid[1]) ||
        (release_valid[0] && release_valid[2]) ||
        (release_valid[1] && release_valid[2]);
    assign owner_encoding_mismatch =
        (o_parameter_owner == 3'b001 && active_owner != 2'd1) ||
        (o_parameter_owner == 3'b010 && active_owner != 2'd2) ||
        (o_parameter_owner == 3'b100 && active_owner != 2'd3) ||
        (o_parameter_owner == 3'b000 && active_owner != 2'd0);
    assign o_system_error = backend_sticky_error ||
        o_dispatch_protocol_error || o_router_protocol_error ||
        o_store_protocol_error || dispatch_context_mismatch ||
        owner_handshake_invalid || owner_encoding_mismatch ||
        (|i_client_error);
    assign o_system_quiescent = dispatch_quiescent &&
        o_backend_quiescent && !o_system_error;

    // Learning mode is configuration state. It can change between runs, but
    // never while an accepted topology command is active or draining.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            learning_enable_q <= 1'b0;
        else if (i_profile_request_valid && dispatch_quiescent &&
                 o_backend_quiescent && !(|i_client_error))
            learning_enable_q <= i_profile_learning_enable;
    end

    unified_tp_train_infer_shared_backend #(
        .DEPTH(DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .PROFILE_FC_H160(PROFILE_FC_H160),
        .PROFILE_FC_H480(PROFILE_FC_H480),
        .PROFILE_SRNN_M2(PROFILE_SRNN_M2),
        .PROFILE_SCNN_VGG9(PROFILE_SCNN_VGG9)
    ) u_backend (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_requested_profile_valid(i_profile_request_valid),
        .i_requested_profile_code(i_profile_request_code),
        .i_inference_global_idle(dispatch_quiescent),
        .i_learning_enable(learning_enable_q),
        .o_acquire_valid(acquire_valid),
        .i_acquire_ready(i_client_quiescent & ~i_client_error),
        .o_release_valid(release_valid),
        .i_release_ready(i_client_quiescent & ~i_client_error),
        .i_topology_quiescent(i_client_quiescent),
        .i_topology_error(i_client_error),
        .o_parameter_owner(o_parameter_owner),
        .o_topology_learning_enable(o_topology_learning_enable),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(backend_start_allowed),
        .o_profile_switch_done(o_profile_switch_done),
        .o_switch_busy(o_profile_switch_busy),
        .o_active_owner(active_owner),
        .o_active_profile_code(o_active_profile_code),
        .o_profile_version(o_profile_version),
        .o_backend_quiescent(o_backend_quiescent),
        .o_sticky_error(backend_sticky_error),
        .o_error_code(o_backend_error_code),
        .i_load_valid(i_load_valid), .o_load_ready(o_load_ready),
        .i_load_addr(i_load_addr), .i_load_code(i_load_code),
        .i_load_residual(i_load_residual),
        .i_learning_read_valid(i_learning_read_valid),
        .o_learning_read_ready(o_learning_read_ready),
        .i_learning_read_addr(i_learning_read_addr),
        .i_learning_read_tag(i_learning_read_tag),
        .o_learning_response_valid(o_learning_response_valid),
        .i_learning_response_ready(i_learning_response_ready),
        .o_learning_response_tag(o_learning_response_tag),
        .o_learning_response_code(o_learning_response_code),
        .o_learning_response_residual(o_learning_response_residual),
        .o_learning_response_error(o_learning_response_error),
        .i_learning_write_valid(i_learning_write_valid),
        .o_learning_write_ready(o_learning_write_ready),
        .i_learning_write_addr(i_learning_write_addr),
        .i_learning_write_code(i_learning_write_code),
        .i_learning_write_residual(i_learning_write_residual),
        .i_learning_write_code_enable(i_learning_write_code_enable),
        .i_learning_write_residual_enable(
            i_learning_write_residual_enable),
        .i_forward_read_valid(i_forward_read_valid),
        .o_forward_read_ready(o_forward_read_ready),
        .i_forward_read_addr(i_forward_read_addr),
        .i_forward_read_tag(i_forward_read_tag),
        .o_forward_response_valid(o_forward_response_valid),
        .i_forward_response_ready(i_forward_response_ready),
        .o_forward_response_tag(o_forward_response_tag),
        .o_forward_response_code(o_forward_response_code),
        .o_forward_response_error(o_forward_response_error),
        .o_router_protocol_error(o_router_protocol_error),
        .o_store_protocol_error(o_store_protocol_error)
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
        .i_client_quiescent(dispatch_client_quiescent),
        .i_client_error(i_client_error),
        .o_active_client(o_active_client),
        .o_active_profile_code(dispatch_active_profile_code),
        .o_active_learning_enable(dispatch_active_learning_enable),
        .o_run_done(o_run_done), .o_run_aborted(o_run_aborted),
        .o_busy(o_run_busy), .o_quiescent(dispatch_quiescent),
        .o_protocol_error(o_dispatch_protocol_error)
    );

endmodule

`default_nettype wire
