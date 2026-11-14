`timescale 1ns/1ps
`default_nettype none

// Production integration boundary for one profile owner, one three-topology
// router and one persistent forward/RMW parameter store. Topology-specific
// observation and gradient generation remain outside this shared substrate.
module unified_tp_train_infer_shared_backend #(
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
    input  logic i_requested_profile_valid,
    input  logic [6:0] i_requested_profile_code,
    input  logic i_inference_global_idle,
    input  logic i_learning_enable,

    output logic [2:0] o_acquire_valid,
    input  logic [2:0] i_acquire_ready,
    output logic [2:0] o_release_valid,
    input  logic [2:0] i_release_ready,
    input  logic [2:0] i_topology_quiescent,
    input  logic [2:0] i_topology_error,

    output logic [2:0] o_parameter_owner,
    output logic [2:0] o_topology_learning_enable,
    output logic o_profile_aligned,
    output logic o_start_allowed,
    output logic o_profile_switch_done,
    output logic o_switch_busy,
    output logic [1:0] o_active_owner,
    output logic [6:0] o_active_profile_code,
    output logic [15:0] o_profile_version,
    output logic o_backend_quiescent,
    output logic o_sticky_error,
    output logic [7:0] o_error_code,

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
    output logic signed [3*CODE_WIDTH-1:0] o_learning_response_code,
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

    output logic o_router_protocol_error,
    output logic o_store_protocol_error
);
    logic store_load_valid;
    logic store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_load_addr;
    logic signed [CODE_WIDTH-1:0] store_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] store_load_residual;
    logic store_learning_read_valid;
    logic store_learning_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_learning_read_addr;
    logic [TAG_WIDTH-1:0] store_learning_read_tag;
    logic store_learning_response_valid;
    logic store_learning_response_ready;
    logic [TAG_WIDTH-1:0] store_learning_response_tag;
    logic signed [CODE_WIDTH-1:0] store_learning_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] store_learning_response_residual;
    logic store_learning_response_error;
    logic store_learning_write_valid;
    logic store_learning_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_learning_write_addr;
    logic signed [CODE_WIDTH-1:0] store_learning_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] store_learning_write_residual;
    logic store_learning_write_code_enable;
    logic store_learning_write_residual_enable;
    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_read_addr;
    logic [TAG_WIDTH-1:0] store_forward_read_tag;
    logic store_forward_response_valid;
    logic store_forward_response_ready;
    logic [TAG_WIDTH-1:0] store_forward_response_tag;
    logic signed [CODE_WIDTH-1:0] store_forward_response_code;
    logic store_forward_response_error;
    logic [2:0] router_learning_read_ready;
    logic [2:0] router_learning_write_ready;
    logic [2:0] gated_learning_read_valid;
    logic [2:0] gated_learning_write_valid;

    assign gated_learning_read_valid = i_learning_read_valid &
        o_topology_learning_enable;
    assign gated_learning_write_valid = i_learning_write_valid &
        o_topology_learning_enable;
    assign o_learning_read_ready = router_learning_read_ready &
        o_topology_learning_enable;
    assign o_learning_write_ready = router_learning_write_ready &
        o_topology_learning_enable;

    unified_tp_train_infer_system_control #(
        .PROFILE_FC_H160(PROFILE_FC_H160),
        .PROFILE_FC_H480(PROFILE_FC_H480),
        .PROFILE_SRNN_M2(PROFILE_SRNN_M2),
        .PROFILE_SCNN_VGG9(PROFILE_SCNN_VGG9)
    ) u_system_control (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_requested_profile_valid(i_requested_profile_valid),
        .i_requested_profile_code(i_requested_profile_code),
        .i_inference_global_idle(i_inference_global_idle),
        .i_learning_enable(i_learning_enable),
        .i_backend_quiescent(o_backend_quiescent),
        .i_backend_error(o_router_protocol_error || o_store_protocol_error),
        .o_fc_acquire_valid(o_acquire_valid[0]),
        .i_fc_acquire_ready(i_acquire_ready[0]),
        .o_srnn_acquire_valid(o_acquire_valid[1]),
        .i_srnn_acquire_ready(i_acquire_ready[1]),
        .o_scnn_acquire_valid(o_acquire_valid[2]),
        .i_scnn_acquire_ready(i_acquire_ready[2]),
        .o_fc_release_valid(o_release_valid[0]),
        .i_fc_release_ready(i_release_ready[0]),
        .o_srnn_release_valid(o_release_valid[1]),
        .i_srnn_release_ready(i_release_ready[1]),
        .o_scnn_release_valid(o_release_valid[2]),
        .i_scnn_release_ready(i_release_ready[2]),
        .i_fc_quiescent(i_topology_quiescent[0]),
        .i_srnn_quiescent(i_topology_quiescent[1]),
        .i_scnn_quiescent(i_topology_quiescent[2]),
        .i_fc_error(i_topology_error[0]),
        .i_srnn_error(i_topology_error[1]),
        .i_scnn_error(i_topology_error[2]),
        .o_fc_parameter_owner(o_parameter_owner[0]),
        .o_srnn_parameter_owner(o_parameter_owner[1]),
        .o_scnn_parameter_owner(o_parameter_owner[2]),
        .o_fc_learning_enable(o_topology_learning_enable[0]),
        .o_srnn_learning_enable(o_topology_learning_enable[1]),
        .o_scnn_learning_enable(o_topology_learning_enable[2]),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(o_start_allowed),
        .o_profile_switch_done(o_profile_switch_done),
        .o_switch_busy(o_switch_busy),
        .o_active_owner(o_active_owner),
        .o_active_profile_code(o_active_profile_code),
        .o_profile_version(o_profile_version),
        .o_sticky_error(o_sticky_error),
        .o_error_code(o_error_code)
    );

    unified_tp_three_client_parameter_backend_router #(
        .CLIENTS(3), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_router (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_owner_active(|o_parameter_owner),
        .i_active_owner(o_active_owner),
        .i_load_valid(i_load_valid), .o_load_ready(o_load_ready),
        .i_load_addr(i_load_addr), .i_load_code(i_load_code),
        .i_load_residual(i_load_residual),
        .i_read_valid(gated_learning_read_valid),
        .o_read_ready(router_learning_read_ready),
        .i_read_addr(i_learning_read_addr),
        .i_read_tag(i_learning_read_tag),
        .o_response_valid(o_learning_response_valid),
        .i_response_ready(i_learning_response_ready),
        .o_response_tag(o_learning_response_tag),
        .o_response_code(o_learning_response_code),
        .o_response_residual(o_learning_response_residual),
        .o_response_error(o_learning_response_error),
        .i_write_valid(gated_learning_write_valid),
        .o_write_ready(router_learning_write_ready),
        .i_write_addr(i_learning_write_addr),
        .i_write_code(i_learning_write_code),
        .i_write_residual(i_learning_write_residual),
        .i_write_code_enable(i_learning_write_code_enable),
        .i_write_residual_enable(i_learning_write_residual_enable),
        .i_forward_read_valid(i_forward_read_valid),
        .o_forward_read_ready(o_forward_read_ready),
        .i_forward_read_addr(i_forward_read_addr),
        .i_forward_read_tag(i_forward_read_tag),
        .o_forward_response_valid(o_forward_response_valid),
        .i_forward_response_ready(i_forward_response_ready),
        .o_forward_response_tag(o_forward_response_tag),
        .o_forward_response_code(o_forward_response_code),
        .o_forward_response_error(o_forward_response_error),
        .o_store_load_valid(store_load_valid),
        .i_store_load_ready(store_load_ready),
        .o_store_load_addr(store_load_addr),
        .o_store_load_code(store_load_code),
        .o_store_load_residual(store_load_residual),
        .o_store_learning_read_valid(store_learning_read_valid),
        .i_store_learning_read_ready(store_learning_read_ready),
        .o_store_learning_read_addr(store_learning_read_addr),
        .o_store_learning_read_tag(store_learning_read_tag),
        .i_store_learning_response_valid(store_learning_response_valid),
        .o_store_learning_response_ready(store_learning_response_ready),
        .i_store_learning_response_tag(store_learning_response_tag),
        .i_store_learning_response_code(store_learning_response_code),
        .i_store_learning_response_residual(
            store_learning_response_residual),
        .i_store_learning_response_error(store_learning_response_error),
        .o_store_learning_write_valid(store_learning_write_valid),
        .i_store_learning_write_ready(store_learning_write_ready),
        .o_store_learning_write_addr(store_learning_write_addr),
        .o_store_learning_write_code(store_learning_write_code),
        .o_store_learning_write_residual(store_learning_write_residual),
        .o_store_learning_write_code_enable(
            store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            store_learning_write_residual_enable),
        .o_store_forward_read_valid(store_forward_read_valid),
        .i_store_forward_read_ready(store_forward_read_ready),
        .o_store_forward_read_addr(store_forward_read_addr),
        .o_store_forward_read_tag(store_forward_read_tag),
        .i_store_forward_response_valid(store_forward_response_valid),
        .o_store_forward_response_ready(store_forward_response_ready),
        .i_store_forward_response_tag(store_forward_response_tag),
        .i_store_forward_response_code(store_forward_response_code),
        .i_store_forward_response_error(store_forward_response_error),
        .i_store_protocol_error(o_store_protocol_error),
        .o_quiescent(o_backend_quiescent),
        .o_protocol_error(o_router_protocol_error)
    );

    unified_tp_shared_forward_rmw_store #(
        .DEPTH(DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_store (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_load_valid(store_load_valid),
        .o_load_ready(store_load_ready),
        .i_load_addr(store_load_addr),
        .i_load_code(store_load_code),
        .i_load_residual(store_load_residual),
        .i_learning_read_valid(store_learning_read_valid),
        .o_learning_read_ready(store_learning_read_ready),
        .i_learning_read_addr(store_learning_read_addr),
        .i_learning_read_tag(store_learning_read_tag),
        .o_learning_response_valid(store_learning_response_valid),
        .i_learning_response_ready(store_learning_response_ready),
        .o_learning_response_tag(store_learning_response_tag),
        .o_learning_response_code(store_learning_response_code),
        .o_learning_response_residual(store_learning_response_residual),
        .o_learning_response_error(store_learning_response_error),
        .i_learning_write_valid(store_learning_write_valid),
        .o_learning_write_ready(store_learning_write_ready),
        .i_learning_write_addr(store_learning_write_addr),
        .i_learning_write_code(store_learning_write_code),
        .i_learning_write_residual(store_learning_write_residual),
        .i_learning_write_code_enable(store_learning_write_code_enable),
        .i_learning_write_residual_enable(
            store_learning_write_residual_enable),
        .i_forward_read_valid(store_forward_read_valid),
        .o_forward_read_ready(store_forward_read_ready),
        .i_forward_read_addr(store_forward_read_addr),
        .i_forward_read_tag(store_forward_read_tag),
        .o_forward_response_valid(store_forward_response_valid),
        .i_forward_response_ready(store_forward_response_ready),
        .o_forward_response_tag(store_forward_response_tag),
        .o_forward_response_code(store_forward_response_code),
        .o_forward_response_error(store_forward_response_error),
        .o_protocol_error(o_store_protocol_error)
    );
endmodule

`default_nettype wire
