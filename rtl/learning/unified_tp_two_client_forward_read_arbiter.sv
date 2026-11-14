`timescale 1ns/1ps
`default_nettype none

// Arbitrates two forward-only clients onto one tagged parameter-store port.
// Response ownership is retained from request acceptance through response
// handshake, so clients may remain backpressured without misrouting data.
module unified_tp_two_client_forward_read_arbiter #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16
) (
    input  logic                              i_clk,
    input  logic                              i_rst_n,
    input  logic                              i_clear_error,

    input  logic                              i_c0_valid,
    output logic                              o_c0_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]       i_c0_addr,
    input  logic [TAG_WIDTH-1:0]              i_c0_tag,
    output logic                              o_c0_response_valid,
    input  logic                              i_c0_response_ready,
    output logic [TAG_WIDTH-1:0]              o_c0_response_tag,
    output logic signed [CODE_WIDTH-1:0]      o_c0_response_code,
    output logic                              o_c0_response_error,

    input  logic                              i_c1_valid,
    output logic                              o_c1_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]       i_c1_addr,
    input  logic [TAG_WIDTH-1:0]              i_c1_tag,
    output logic                              o_c1_response_valid,
    input  logic                              i_c1_response_ready,
    output logic [TAG_WIDTH-1:0]              o_c1_response_tag,
    output logic signed [CODE_WIDTH-1:0]      o_c1_response_code,
    output logic                              o_c1_response_error,

    output logic                              o_store_valid,
    input  logic                              i_store_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_store_addr,
    output logic [TAG_WIDTH-1:0]              o_store_tag,
    input  logic                              i_store_response_valid,
    output logic                              o_store_response_ready,
    input  logic [TAG_WIDTH-1:0]              i_store_response_tag,
    input  logic signed [CODE_WIDTH-1:0]      i_store_response_code,
    input  logic                              i_store_response_error,

    output logic                              o_busy,
    output logic                              o_protocol_error
);
    logic owner_q;
    logic last_grant_q;
    logic pending_q;
    logic selected_owner;
    logic selected_valid;
    logic request_fire;
    logic response_fire;

    always_comb begin
        selected_valid = i_c0_valid || i_c1_valid;
        if (i_c0_valid && i_c1_valid)
            selected_owner = ~last_grant_q;
        else
            selected_owner = i_c1_valid;

        o_store_valid = !pending_q && !o_protocol_error && selected_valid;
        o_store_addr = selected_owner ? i_c1_addr : i_c0_addr;
        o_store_tag = selected_owner ? i_c1_tag : i_c0_tag;
        o_c0_ready = o_store_valid && !selected_owner && i_store_ready;
        o_c1_ready = o_store_valid && selected_owner && i_store_ready;
        request_fire = o_store_valid && i_store_ready;

        o_c0_response_valid = pending_q && !owner_q &&
            i_store_response_valid;
        o_c1_response_valid = pending_q && owner_q &&
            i_store_response_valid;
        o_c0_response_tag = i_store_response_tag;
        o_c1_response_tag = i_store_response_tag;
        o_c0_response_code = i_store_response_code;
        o_c1_response_code = i_store_response_code;
        o_c0_response_error = i_store_response_error;
        o_c1_response_error = i_store_response_error;
        o_store_response_ready = pending_q &&
            (owner_q ? i_c1_response_ready : i_c0_response_ready);
        response_fire = i_store_response_valid && o_store_response_ready;
        o_busy = pending_q || i_store_response_valid;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_q <= 1'b0;
            last_grant_q <= 1'b1;
            pending_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error && !pending_q && !i_store_response_valid)
                o_protocol_error <= 1'b0;
            if (i_store_response_valid && !pending_q)
                o_protocol_error <= 1'b1;
            if (request_fire) begin
                owner_q <= selected_owner;
                last_grant_q <= selected_owner;
                pending_q <= 1'b1;
            end
            if (response_fire)
                pending_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
