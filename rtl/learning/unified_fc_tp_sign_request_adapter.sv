`timescale 1ns/1ps
`default_nettype none

// FC TP gradients arrive in tensor-local storage order. This adapter converts
// their sign to a descent direction and leaves numeric update policy in the
// shared active-tensor RMW service.
module unified_fc_tp_sign_request_adapter #(
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,
    input  wire logic                               i_active,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_active_tensor_id,

    input  wire logic                               i_gradient_valid,
    output logic                               o_gradient_ready,
    input  wire logic [TENSOR_ID_WIDTH-1:0]         i_gradient_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_gradient_local_parameter_addr,
    input  wire logic [TAG_WIDTH-1:0]                i_gradient_tag,
    input  wire logic signed [1:0]                   i_gradient_sign,

    output logic                               o_req_valid,
    input  wire logic                               i_req_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_req_local_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_req_tag,
    output logic signed [1:0]                   o_req_update_direction,

    output logic                               o_update_accepted,
    output logic                               o_zero_skipped,
    output logic                               o_tensor_error,
    output logic                               o_sign_error
);

    logic tensor_match;
    logic sign_legal;
    logic sign_nonzero;

    always_comb begin
        tensor_match = i_active &&
            (i_gradient_tensor_id == i_active_tensor_id);
        sign_legal = (i_gradient_sign == -2'sd1) ||
            (i_gradient_sign == 2'sd0) ||
            (i_gradient_sign == 2'sd1);
        sign_nonzero = i_gradient_sign != 2'sd0;

        o_req_valid = i_gradient_valid && tensor_match && sign_legal &&
            sign_nonzero;
        o_req_local_parameter_addr = i_gradient_local_parameter_addr;
        o_req_tag = i_gradient_tag;
        o_req_update_direction = -i_gradient_sign;

        // All gradient records remain ordered behind the single RMW service,
        // including records that are discarded without a parameter access.
        o_gradient_ready = i_req_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_update_accepted <= 1'b0;
            o_zero_skipped <= 1'b0;
            o_tensor_error <= 1'b0;
            o_sign_error <= 1'b0;
        end else begin
            o_update_accepted <= i_gradient_valid && i_req_ready &&
                i_active && (i_gradient_tensor_id == i_active_tensor_id) &&
                sign_legal && sign_nonzero;
            o_zero_skipped <= i_gradient_valid && i_req_ready &&
                i_active && (i_gradient_tensor_id == i_active_tensor_id) &&
                sign_legal && !sign_nonzero;
            o_tensor_error <= i_gradient_valid && i_req_ready &&
                (!i_active || (i_gradient_tensor_id != i_active_tensor_id));
            o_sign_error <= i_gradient_valid && i_req_ready && i_active &&
                (i_gradient_tensor_id == i_active_tensor_id) && !sign_legal;
        end
    end

endmodule

`default_nettype wire
