`timescale 1ns/1ps
`default_nettype none

// Routes FC and SRNN learning clients to one code/residual parameter backend.
// Owner changes are legal only when o_quiescent is high.  Read-response owner
// metadata is retained across arbitrary backend and consumer backpressure.
module unified_tp_fc_srnn_parameter_backend_router #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_owner_active,
    input  logic [1:0]                                i_active_owner,

    input  logic                                      i_fc_load_valid,
    output logic                                      o_fc_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_fc_load_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_fc_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_fc_load_residual,
    input  logic                                      i_srnn_load_valid,
    output logic                                      o_srnn_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_srnn_load_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_srnn_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_srnn_load_residual,

    input  logic                                      i_fc_read_valid,
    output logic                                      o_fc_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_fc_read_addr,
    input  logic [TAG_WIDTH-1:0]                      i_fc_read_tag,
    output logic                                      o_fc_response_valid,
    input  logic                                      i_fc_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_fc_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_fc_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_fc_response_residual,
    output logic                                      o_fc_response_error,
    input  logic                                      i_fc_write_valid,
    output logic                                      o_fc_write_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_fc_write_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_fc_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_fc_write_residual,
    input  logic                                      i_fc_write_code_enable,
    input  logic                                      i_fc_write_residual_enable,

    input  logic                                      i_srnn_read_valid,
    output logic                                      o_srnn_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_srnn_read_addr,
    input  logic [TAG_WIDTH-1:0]                      i_srnn_read_tag,
    output logic                                      o_srnn_response_valid,
    input  logic                                      i_srnn_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_srnn_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_srnn_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_srnn_response_residual,
    output logic                                      o_srnn_response_error,
    input  logic                                      i_srnn_write_valid,
    output logic                                      o_srnn_write_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_srnn_write_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_srnn_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_srnn_write_residual,
    input  logic                                      i_srnn_write_code_enable,
    input  logic                                      i_srnn_write_residual_enable,

    input  logic                                      i_srnn_forward_read_valid,
    output logic                                      o_srnn_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_srnn_forward_read_addr,
    input  logic [TAG_WIDTH-1:0]                      i_srnn_forward_read_tag,
    output logic                                      o_srnn_forward_response_valid,
    input  logic                                      i_srnn_forward_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_srnn_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_srnn_forward_response_code,
    output logic                                      o_srnn_forward_response_error,

    output logic                                      o_store_load_valid,
    input  logic                                      i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0]              o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_store_load_residual,
    output logic                                      o_store_learning_read_valid,
    input  logic                                      i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_store_learning_read_tag,
    input  logic                                      i_store_learning_response_valid,
    output logic                                      o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0]              i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_store_learning_response_residual,
    input  logic                                      i_store_learning_response_error,
    output logic                                      o_store_learning_write_valid,
    input  logic                                      i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0]              o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_store_learning_write_residual,
    output logic                                      o_store_learning_write_code_enable,
    output logic                                      o_store_learning_write_residual_enable,
    output logic                                      o_store_forward_read_valid,
    input  logic                                      i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                      o_store_forward_read_tag,
    input  logic                                      i_store_forward_response_valid,
    output logic                                      o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]              i_store_forward_response_code,
    input  logic                                      i_store_forward_response_error,
    input  logic                                      i_store_protocol_error,

    output logic                                      o_quiescent,
    output logic                                      o_protocol_error
);

    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_SRNN = 2'd2;

    logic [1:0] learning_response_owner_q;
    logic learning_response_pending_q;
    logic forward_response_pending_q;
    logic [1:0] pending_owner_snapshot_q;
    logic learning_read_fire;
    logic learning_response_fire;
    logic forward_read_fire;
    logic forward_response_fire;
    logic inactive_request;
    logic owner_changed_while_pending;

    always_comb begin
        o_fc_load_ready = 1'b0;
        o_srnn_load_ready = 1'b0;
        o_store_load_valid = 1'b0;
        o_store_load_addr = '0;
        o_store_load_code = '0;
        o_store_load_residual = '0;

        o_fc_read_ready = 1'b0;
        o_srnn_read_ready = 1'b0;
        o_store_learning_read_valid = 1'b0;
        o_store_learning_read_addr = '0;
        o_store_learning_read_tag = '0;
        o_fc_write_ready = 1'b0;
        o_srnn_write_ready = 1'b0;
        o_store_learning_write_valid = 1'b0;
        o_store_learning_write_addr = '0;
        o_store_learning_write_code = '0;
        o_store_learning_write_residual = '0;
        o_store_learning_write_code_enable = 1'b0;
        o_store_learning_write_residual_enable = 1'b0;

        if (i_owner_active && (i_active_owner == OWNER_FC)) begin
            o_store_load_valid = i_fc_load_valid;
            o_fc_load_ready = i_store_load_ready;
            o_store_load_addr = i_fc_load_addr;
            o_store_load_code = i_fc_load_code;
            o_store_load_residual = i_fc_load_residual;
            o_store_learning_read_valid = i_fc_read_valid;
            o_fc_read_ready = i_store_learning_read_ready;
            o_store_learning_read_addr = i_fc_read_addr;
            o_store_learning_read_tag = i_fc_read_tag;
            o_store_learning_write_valid = i_fc_write_valid;
            o_fc_write_ready = i_store_learning_write_ready;
            o_store_learning_write_addr = i_fc_write_addr;
            o_store_learning_write_code = i_fc_write_code;
            o_store_learning_write_residual = i_fc_write_residual;
            o_store_learning_write_code_enable = i_fc_write_code_enable;
            o_store_learning_write_residual_enable =
                i_fc_write_residual_enable;
        end else if (i_owner_active && (i_active_owner == OWNER_SRNN)) begin
            o_store_load_valid = i_srnn_load_valid;
            o_srnn_load_ready = i_store_load_ready;
            o_store_load_addr = i_srnn_load_addr;
            o_store_load_code = i_srnn_load_code;
            o_store_load_residual = i_srnn_load_residual;
            o_store_learning_read_valid = i_srnn_read_valid;
            o_srnn_read_ready = i_store_learning_read_ready;
            o_store_learning_read_addr = i_srnn_read_addr;
            o_store_learning_read_tag = i_srnn_read_tag;
            o_store_learning_write_valid = i_srnn_write_valid;
            o_srnn_write_ready = i_store_learning_write_ready;
            o_store_learning_write_addr = i_srnn_write_addr;
            o_store_learning_write_code = i_srnn_write_code;
            o_store_learning_write_residual = i_srnn_write_residual;
            o_store_learning_write_code_enable = i_srnn_write_code_enable;
            o_store_learning_write_residual_enable =
                i_srnn_write_residual_enable;
        end

        o_fc_response_valid = i_store_learning_response_valid &&
            learning_response_pending_q &&
            (learning_response_owner_q == OWNER_FC);
        o_srnn_response_valid = i_store_learning_response_valid &&
            learning_response_pending_q &&
            (learning_response_owner_q == OWNER_SRNN);
        o_fc_response_tag = i_store_learning_response_tag;
        o_fc_response_code = i_store_learning_response_code;
        o_fc_response_residual = i_store_learning_response_residual;
        o_fc_response_error = i_store_learning_response_error;
        o_srnn_response_tag = i_store_learning_response_tag;
        o_srnn_response_code = i_store_learning_response_code;
        o_srnn_response_residual = i_store_learning_response_residual;
        o_srnn_response_error = i_store_learning_response_error;
        o_store_learning_response_ready =
            ((learning_response_owner_q == OWNER_FC) &&
                i_fc_response_ready) ||
            ((learning_response_owner_q == OWNER_SRNN) &&
                i_srnn_response_ready);

        o_store_forward_read_valid = i_owner_active &&
            (i_active_owner == OWNER_SRNN) && i_srnn_forward_read_valid;
        o_srnn_forward_read_ready = i_owner_active &&
            (i_active_owner == OWNER_SRNN) && i_store_forward_read_ready;
        o_store_forward_read_addr = i_srnn_forward_read_addr;
        o_store_forward_read_tag = i_srnn_forward_read_tag;
        o_srnn_forward_response_valid = i_store_forward_response_valid &&
            forward_response_pending_q;
        o_srnn_forward_response_tag = i_store_forward_response_tag;
        o_srnn_forward_response_code = i_store_forward_response_code;
        o_srnn_forward_response_error = i_store_forward_response_error;
        o_store_forward_response_ready =
            i_srnn_forward_response_ready && forward_response_pending_q;

        learning_read_fire = o_store_learning_read_valid &&
            i_store_learning_read_ready;
        learning_response_fire = i_store_learning_response_valid &&
            o_store_learning_response_ready;
        forward_read_fire = o_store_forward_read_valid &&
            i_store_forward_read_ready;
        forward_response_fire = i_store_forward_response_valid &&
            o_store_forward_response_ready;

        inactive_request =
            (!i_owner_active && (i_fc_load_valid || i_srnn_load_valid ||
                i_fc_read_valid || i_srnn_read_valid || i_fc_write_valid ||
                i_srnn_write_valid || i_srnn_forward_read_valid)) ||
            (i_owner_active && (i_active_owner == OWNER_FC) &&
                (i_srnn_load_valid || i_srnn_read_valid ||
                 i_srnn_write_valid || i_srnn_forward_read_valid)) ||
            (i_owner_active && (i_active_owner == OWNER_SRNN) &&
                (i_fc_load_valid || i_fc_read_valid || i_fc_write_valid));
        owner_changed_while_pending =
            (learning_response_pending_q || forward_response_pending_q) &&
            (!i_owner_active ||
             (i_active_owner != pending_owner_snapshot_q));
        o_quiescent = !learning_response_pending_q &&
            !forward_response_pending_q &&
            !i_store_learning_response_valid &&
            !i_store_forward_response_valid && !o_store_load_valid &&
            !o_store_learning_read_valid &&
            !o_store_learning_write_valid &&
            !o_store_forward_read_valid;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            learning_response_owner_q <= 2'd0;
            learning_response_pending_q <= 1'b0;
            forward_response_pending_q <= 1'b0;
            pending_owner_snapshot_q <= 2'd0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error)
                o_protocol_error <= 1'b0;
            if (inactive_request || owner_changed_while_pending ||
                i_store_protocol_error ||
                (i_store_learning_response_valid &&
                    !learning_response_pending_q) ||
                (i_store_forward_response_valid &&
                    !forward_response_pending_q))
                o_protocol_error <= 1'b1;

            if (learning_response_fire && !learning_read_fire) begin
                learning_response_pending_q <= 1'b0;
                learning_response_owner_q <= 2'd0;
            end
            if (learning_read_fire) begin
                learning_response_pending_q <= 1'b1;
                learning_response_owner_q <= i_active_owner;
                pending_owner_snapshot_q <= i_active_owner;
            end

            if (forward_response_fire && !forward_read_fire)
                forward_response_pending_q <= 1'b0;
            if (forward_read_fire) begin
                forward_response_pending_q <= 1'b1;
                pending_owner_snapshot_q <= i_active_owner;
            end
        end
    end

endmodule

`default_nettype wire
