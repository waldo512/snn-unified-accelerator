`timescale 1ns/1ps
`default_nettype none

// Shared scalar parameter service under the native runtime owner.  Profile,
// run and learning-mode control live in unified_tp_native_runtime_*; this
// block contains only the three-client router and persistent forward/RMW
// state, so a production composition has one ownership authority.
module unified_tp_native_shared_parameter_backend #(
    parameter int DEPTH = 120320,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_clear_error,
    input  logic [2:0] i_parameter_owner,
    input  logic [2:0] i_topology_learning_enable,

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

    output logic o_backend_quiescent,
    output logic o_sticky_error,
    output logic o_owner_encoding_error,
    output logic o_router_protocol_error,
    output logic o_store_protocol_error
);
    logic owner_active;
    logic [1:0] active_owner;
    logic [2:0] router_learning_read_ready;
    logic [2:0] router_learning_write_ready;
    logic [2:0] gated_learning_read_valid;
    logic [2:0] gated_learning_write_valid;

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
    logic signed [RESIDUAL_WIDTH-1:0]
        store_learning_response_residual;
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

    always_comb begin
        owner_active = 1'b1;
        active_owner = 2'd0;
        o_owner_encoding_error = 1'b0;
        unique case (i_parameter_owner)
            3'b000: begin
                owner_active = 1'b0;
            end
            3'b001: active_owner = 2'd1;
            3'b010: active_owner = 2'd2;
            3'b100: active_owner = 2'd3;
            default: begin
                owner_active = 1'b0;
                o_owner_encoding_error = 1'b1;
            end
        endcase
    end

    assign gated_learning_read_valid = i_learning_read_valid &
        i_topology_learning_enable;
    assign gated_learning_write_valid = i_learning_write_valid &
        i_topology_learning_enable;
    assign o_learning_read_ready = router_learning_read_ready &
        i_topology_learning_enable;
    assign o_learning_write_ready = router_learning_write_ready &
        i_topology_learning_enable;
    assign o_sticky_error = o_owner_encoding_error ||
        o_router_protocol_error || o_store_protocol_error;

    unified_tp_three_client_parameter_backend_router #(
        .CLIENTS(3), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_router (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_owner_active(owner_active), .i_active_owner(active_owner),
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
