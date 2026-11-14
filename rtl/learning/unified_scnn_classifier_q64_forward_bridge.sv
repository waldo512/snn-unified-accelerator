`timescale 1ns/1ps
`default_nettype none

// Adapts the OUTPUT frontend's paired Q64 weight reads to the scalar Q16
// forward port of the shared train/infer parameter store. One request is
// serviced at a time; the completed pair is held stable under backpressure.
module unified_scnn_classifier_q64_forward_bridge #(
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int PARAM_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16
) (
    input  logic                              i_clk,
    input  logic                              i_rst_n,
    input  logic                              i_clear_error,
    input  logic                              i_request_valid,
    output logic                              o_request_ready,
    input  logic [Q64_ADDR_WIDTH-1:0]         i_request_addr0,
    input  logic [Q64_ADDR_WIDTH-1:0]         i_request_addr1,
    output logic                              o_response_valid,
    input  logic                              i_response_ready,
    output logic [63:0]                       o_response_data0,
    output logic [63:0]                       o_response_data1,
    output logic                              o_forward_read_valid,
    input  logic                              i_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]       o_forward_read_addr,
    output logic [TAG_WIDTH-1:0]              o_forward_read_tag,
    input  logic                              i_forward_response_valid,
    output logic                              o_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]              i_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]      i_forward_response_code,
    input  logic                              i_forward_response_error,
    output logic                              o_busy,
    output logic                              o_protocol_error
);
    typedef enum logic [1:0] {IDLE, ISSUE, WAIT_RESPONSE, HOLD} state_t;
    state_t state_q;
    logic [Q64_ADDR_WIDTH-1:0] addr0_q, addr1_q;
    logic [2:0] word_q;
    logic [127:0] payload_q;
    logic request_fire, read_fire, response_fire;

    assign o_request_ready = state_q == IDLE && !o_protocol_error;
    assign request_fire = i_request_valid && o_request_ready;
    assign o_forward_read_valid = state_q == ISSUE;
    assign o_forward_read_addr = PARAM_ADDR_WIDTH'(
        ((word_q < 4 ? addr0_q : addr1_q) << 2) + (word_q & 3));
    assign o_forward_read_tag = TAG_WIDTH'(word_q);
    assign read_fire = o_forward_read_valid && i_forward_read_ready;
    assign o_forward_response_ready = state_q == WAIT_RESPONSE;
    assign response_fire = i_forward_response_valid && o_forward_response_ready;
    assign o_response_valid = state_q == HOLD;
    assign o_response_data0 = payload_q[63:0];
    assign o_response_data1 = payload_q[127:64];
    assign o_busy = state_q != IDLE;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            addr0_q <= '0;
            addr1_q <= '0;
            word_q <= '0;
            payload_q <= '0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error && state_q == IDLE)
                o_protocol_error <= 1'b0;
            if (i_forward_response_valid && state_q != WAIT_RESPONSE)
                o_protocol_error <= 1'b1;

            unique case (state_q)
                IDLE: if (request_fire) begin
                    addr0_q <= i_request_addr0;
                    addr1_q <= i_request_addr1;
                    word_q <= '0;
                    state_q <= ISSUE;
                end
                ISSUE: if (read_fire)
                    state_q <= WAIT_RESPONSE;
                WAIT_RESPONSE: if (response_fire) begin
                    if (i_forward_response_error ||
                        i_forward_response_tag != TAG_WIDTH'(word_q)) begin
                        o_protocol_error <= 1'b1;
                        state_q <= IDLE;
                    end else begin
                        payload_q[word_q*16 +: 16] <= i_forward_response_code;
                        if (word_q == 3'd7)
                            state_q <= HOLD;
                        else begin
                            word_q <= word_q + 1'b1;
                            state_q <= ISSUE;
                        end
                    end
                end
                HOLD: if (i_response_ready)
                    state_q <= IDLE;
                default: state_q <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
