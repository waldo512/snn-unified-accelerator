`timescale 1ns/1ps
`default_nettype none

// Maps the production SCNN online client onto client 2 of the topology-neutral
// system. FC and SRNN are inactive here; they use the same vector boundary in
// the final composition rather than separate stores.
module unified_scnn_tp_shared_system_client_adapter #(
    parameter int DEPTH = 128,
    parameter int PARAM_ADDR_WIDTH = 8,
    parameter int TAG_WIDTH = 8,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_clear_error,
    input  logic i_abort,
    input  logic i_learning_enable,

    input  logic i_run_valid,
    output logic o_run_ready,
    output logic o_run_done,
    output logic o_run_aborted,

    output logic o_client_start_valid,
    input  logic i_client_start_ready,
    output logic o_client_abort,
    input  logic i_client_done,
    input  logic i_client_aborted,
    input  logic i_client_quiescent,
    input  logic i_client_error,
    output logic o_profile_aligned,
    output logic o_start_allowed,
    output logic o_parameter_owner,
    output logic o_client_learning_enable,

    input  logic i_load_valid,
    output logic o_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_load_addr,
    input  logic signed [CODE_WIDTH-1:0] i_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0] i_load_residual,

    input  logic i_learning_read_valid,
    output logic o_learning_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_learning_read_addr,
    input  logic [TAG_WIDTH-1:0] i_learning_read_tag,
    output logic o_learning_response_valid,
    input  logic i_learning_response_ready,
    output logic [TAG_WIDTH-1:0] o_learning_response_tag,
    output logic signed [CODE_WIDTH-1:0] o_learning_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]
        o_learning_response_residual,
    output logic o_learning_response_error,

    input  logic i_learning_write_valid,
    output logic o_learning_write_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_learning_write_addr,
    input  logic signed [CODE_WIDTH-1:0] i_learning_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0] i_learning_write_residual,
    input  logic i_learning_write_code_enable,
    input  logic i_learning_write_residual_enable,

    input  logic i_forward_read_valid,
    output logic o_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0] i_forward_read_addr,
    input  logic [TAG_WIDTH-1:0] i_forward_read_tag,
    output logic o_forward_response_valid,
    input  logic i_forward_response_ready,
    output logic [TAG_WIDTH-1:0] o_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0] o_forward_response_code,
    output logic o_forward_response_error,

    output logic o_backend_quiescent,
    output logic o_system_quiescent,
    output logic o_system_error
);
    logic [2:0] client_start_valid;
    logic [2:0] client_abort;
    logic [2:0] parameter_owner;
    logic [2:0] topology_learning_enable;
    logic [2:0] load_ready;
    logic [2:0] learning_read_ready;
    logic [2:0] learning_response_valid;
    logic [3*TAG_WIDTH-1:0] learning_response_tag;
    logic signed [3*CODE_WIDTH-1:0] learning_response_code;
    logic signed [3*RESIDUAL_WIDTH-1:0] learning_response_residual;
    logic [2:0] learning_response_error;
    logic [2:0] learning_write_ready;
    logic [2:0] forward_read_ready;
    logic [2:0] forward_response_valid;
    logic [3*TAG_WIDTH-1:0] forward_response_tag;
    logic signed [3*CODE_WIDTH-1:0] forward_response_code;
    logic [2:0] forward_response_error;
    logic system_start_allowed;

    assign o_client_start_valid = client_start_valid[2];
    assign o_client_abort = client_abort[2];
    assign o_parameter_owner = parameter_owner[2];
    assign o_client_learning_enable = topology_learning_enable[2];
    // The common dispatcher consumes system_start_allowed when it accepts the
    // run. The selected child then needs a launch permission that remains
    // asserted while that accepted command is waiting on child ready.
    assign o_start_allowed =
        (system_start_allowed || client_start_valid[2]) &&
        o_profile_aligned && o_parameter_owner && !o_system_error;
    assign o_load_ready = load_ready[2];
    assign o_learning_read_ready = learning_read_ready[2];
    assign o_learning_response_valid = learning_response_valid[2];
    assign o_learning_response_tag =
        learning_response_tag[2*TAG_WIDTH +: TAG_WIDTH];
    assign o_learning_response_code =
        learning_response_code[2*CODE_WIDTH +: CODE_WIDTH];
    assign o_learning_response_residual =
        learning_response_residual[2*RESIDUAL_WIDTH +: RESIDUAL_WIDTH];
    assign o_learning_response_error = learning_response_error[2];
    assign o_learning_write_ready = learning_write_ready[2];
    assign o_forward_read_ready = forward_read_ready[2];
    assign o_forward_response_valid = forward_response_valid[2];
    assign o_forward_response_tag =
        forward_response_tag[2*TAG_WIDTH +: TAG_WIDTH];
    assign o_forward_response_code =
        forward_response_code[2*CODE_WIDTH +: CODE_WIDTH];
    assign o_forward_response_error = forward_response_error[2];

    unified_tp_online_system_top #(
        .DEPTH(DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_system (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_profile_request_valid(1'b1),
        .i_profile_request_code(7'h20),
        .i_profile_learning_enable(i_learning_enable),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(system_start_allowed),
        .o_profile_switch_done(), .o_profile_switch_busy(),
        .o_active_profile_code(), .o_profile_version(),
        .o_active_learning_enable(),
        .i_run_valid(i_run_valid), .o_run_ready(o_run_ready),
        .i_run_profile_code(7'h20), .o_run_done(o_run_done),
        .o_run_aborted(o_run_aborted), .o_run_busy(),
        .o_client_start_valid(client_start_valid),
        .i_client_start_ready({i_client_start_ready, 2'b00}),
        .o_client_abort(client_abort),
        .i_client_done({i_client_done, 2'b00}),
        .i_client_aborted({i_client_aborted, 2'b00}),
        .i_client_quiescent({i_client_quiescent, 2'b11}),
        .i_client_error({i_client_error, 2'b00}),
        .o_active_client(), .o_parameter_owner(parameter_owner),
        .o_topology_learning_enable(topology_learning_enable),
        .i_load_valid({i_load_valid, 2'b00}),
        .o_load_ready(load_ready),
        .i_load_addr({i_load_addr, {2*PARAM_ADDR_WIDTH{1'b0}}}),
        .i_load_code({i_load_code, {2*CODE_WIDTH{1'b0}}}),
        .i_load_residual(
            {i_load_residual, {2*RESIDUAL_WIDTH{1'b0}}}),
        .i_learning_read_valid({i_learning_read_valid, 2'b00}),
        .o_learning_read_ready(learning_read_ready),
        .i_learning_read_addr(
            {i_learning_read_addr, {2*PARAM_ADDR_WIDTH{1'b0}}}),
        .i_learning_read_tag(
            {i_learning_read_tag, {2*TAG_WIDTH{1'b0}}}),
        .o_learning_response_valid(learning_response_valid),
        .i_learning_response_ready(
            {i_learning_response_ready, 2'b00}),
        .o_learning_response_tag(learning_response_tag),
        .o_learning_response_code(learning_response_code),
        .o_learning_response_residual(learning_response_residual),
        .o_learning_response_error(learning_response_error),
        .i_learning_write_valid({i_learning_write_valid, 2'b00}),
        .o_learning_write_ready(learning_write_ready),
        .i_learning_write_addr(
            {i_learning_write_addr, {2*PARAM_ADDR_WIDTH{1'b0}}}),
        .i_learning_write_code(
            {i_learning_write_code, {2*CODE_WIDTH{1'b0}}}),
        .i_learning_write_residual(
            {i_learning_write_residual, {2*RESIDUAL_WIDTH{1'b0}}}),
        .i_learning_write_code_enable(
            {i_learning_write_code_enable, 2'b00}),
        .i_learning_write_residual_enable(
            {i_learning_write_residual_enable, 2'b00}),
        .i_forward_read_valid({i_forward_read_valid, 2'b00}),
        .o_forward_read_ready(forward_read_ready),
        .i_forward_read_addr(
            {i_forward_read_addr, {2*PARAM_ADDR_WIDTH{1'b0}}}),
        .i_forward_read_tag(
            {i_forward_read_tag, {2*TAG_WIDTH{1'b0}}}),
        .o_forward_response_valid(forward_response_valid),
        .i_forward_response_ready({i_forward_response_ready, 2'b00}),
        .o_forward_response_tag(forward_response_tag),
        .o_forward_response_code(forward_response_code),
        .o_forward_response_error(forward_response_error),
        .o_backend_quiescent(o_backend_quiescent),
        .o_system_quiescent(o_system_quiescent),
        .o_system_error(o_system_error),
        .o_backend_error_code(), .o_dispatch_protocol_error(),
        .o_router_protocol_error(), .o_store_protocol_error()
    );
endmodule

`default_nettype wire
