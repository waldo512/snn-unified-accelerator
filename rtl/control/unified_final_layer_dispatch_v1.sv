`timescale 1ns/1ps
`default_nettype none

// Final layer-owner dispatch. This is separate from the frozen DIV1 dispatch
// so the final core can expose a stable compute owner without changing the
// active background regression closure.
module unified_final_layer_dispatch_v1 (
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
    output logic          o_busy,
    output logic          o_sticky_error,
    output logic [7:0]    o_error_code
);
    localparam logic [3:0] LAYER_SPARSE_FC = 4'd1;
    localparam logic [3:0] LAYER_CONV = 4'd2;
    localparam logic [3:0] LAYER_POOL = 4'd3;
    localparam logic [3:0] LAYER_OUTPUT = 4'd4;
    localparam logic [3:0] LAYER_RECURRENT_FC = 4'd5;

    localparam logic [2:0] DISPATCH_NONE = 3'd0;
    localparam logic [2:0] DISPATCH_FC = 3'd1;
    localparam logic [2:0] DISPATCH_RECURRENT = 3'd2;
    localparam logic [2:0] DISPATCH_SCNN = 3'd3;
    localparam logic [2:0] DISPATCH_OUTPUT = 3'd4;

    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_SRNN = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_INVALID_LAYER = 8'h01;
    localparam logic [7:0] ERROR_ABORT = 8'h02;

    logic [2:0] selected_dispatch;
    logic [2:0] active_dispatch_q;
    logic [3:0] active_layer_type_q;
    logic done_armed_q;
    logic selected_ready;
    logic active_done;
    logic [1:0] selected_compute_owner;
    logic [1:0] active_compute_owner;
    logic sticky_error_q;
    logic [7:0] error_code_q;

    always_comb begin
        case (i_layer_type)
            LAYER_SPARSE_FC: selected_dispatch = DISPATCH_FC;
            LAYER_RECURRENT_FC: selected_dispatch = DISPATCH_RECURRENT;
            LAYER_CONV, LAYER_POOL: selected_dispatch = DISPATCH_SCNN;
            LAYER_OUTPUT: selected_dispatch = DISPATCH_OUTPUT;
            default: selected_dispatch = DISPATCH_NONE;
        endcase

        case (selected_dispatch)
            DISPATCH_FC: selected_ready = i_fc_ready;
            DISPATCH_RECURRENT: selected_ready = i_recurrent_ready;
            DISPATCH_SCNN: selected_ready = i_scnn_ready;
            DISPATCH_OUTPUT: selected_ready = i_output_ready;
            default: selected_ready = 1'b0;
        endcase

        case (active_dispatch_q)
            DISPATCH_FC: active_done = i_fc_done;
            DISPATCH_RECURRENT: active_done = i_recurrent_done;
            DISPATCH_SCNN: active_done = i_scnn_done;
            DISPATCH_OUTPUT: active_done = i_output_done;
            default: active_done = 1'b0;
        endcase

        case (selected_dispatch)
            DISPATCH_FC: selected_compute_owner = OWNER_FC;
            DISPATCH_RECURRENT: selected_compute_owner = OWNER_SRNN;
            DISPATCH_SCNN, DISPATCH_OUTPUT:
                selected_compute_owner = OWNER_SCNN;
            default: selected_compute_owner = OWNER_IDLE;
        endcase
        case (active_dispatch_q)
            DISPATCH_FC: active_compute_owner = OWNER_FC;
            DISPATCH_RECURRENT: active_compute_owner = OWNER_SRNN;
            DISPATCH_SCNN, DISPATCH_OUTPUT:
                active_compute_owner = OWNER_SCNN;
            default: active_compute_owner = OWNER_IDLE;
        endcase

        o_layer_ready = !o_busy && !sticky_error_q &&
                        selected_dispatch != DISPATCH_NONE &&
                        selected_ready;
        o_fc_valid = i_layer_valid && o_layer_ready &&
                     selected_dispatch == DISPATCH_FC;
        o_recurrent_valid = i_layer_valid && o_layer_ready &&
                            selected_dispatch == DISPATCH_RECURRENT;
        o_scnn_valid = i_layer_valid && o_layer_ready &&
                       selected_dispatch == DISPATCH_SCNN;
        o_output_valid = i_layer_valid && o_layer_ready &&
                         selected_dispatch == DISPATCH_OUTPUT;
        o_layer_accept = i_layer_valid && o_layer_ready;
        o_layer_done = o_busy && done_armed_q && active_done;
        o_requested_owner = o_busy ? active_compute_owner :
                                    selected_compute_owner;
        o_active_layer_type = active_layer_type_q;
        o_owner_active = o_busy;
        o_sticky_error = sticky_error_q;
        o_error_code = sticky_error_q ? error_code_q : ERROR_NONE;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_dispatch_q <= DISPATCH_NONE;
            active_layer_type_q <= '0;
            done_armed_q <= 1'b0;
            o_busy <= 1'b0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
        end else begin
            if (i_clear_error && !o_busy) begin
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
            end
            if (i_layer_valid && !o_busy &&
                selected_dispatch == DISPATCH_NONE && !sticky_error_q) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_INVALID_LAYER;
            end
            if (o_layer_accept) begin
                active_dispatch_q <= selected_dispatch;
                active_layer_type_q <= i_layer_type;
                done_armed_q <= 1'b0;
                o_busy <= 1'b1;
            end
            if (o_busy && !active_done)
                done_armed_q <= 1'b1;
            if (o_layer_done) begin
                active_dispatch_q <= DISPATCH_NONE;
                active_layer_type_q <= '0;
                done_armed_q <= 1'b0;
                o_busy <= 1'b0;
            end
            if (i_abort) begin
                active_dispatch_q <= DISPATCH_NONE;
                active_layer_type_q <= '0;
                done_armed_q <= 1'b0;
                o_busy <= 1'b0;
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_ABORT;
            end
        end
    end
endmodule

`default_nettype wire
