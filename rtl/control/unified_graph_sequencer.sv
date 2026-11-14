`timescale 1ns/1ps
`default_nettype none

module unified_graph_sequencer #(
    parameter integer LAYER_WORDS = 12
) (
    input  wire logic                         clk,
    input  wire logic                         rst_n,
    input  wire logic                         i_start,
    input  wire logic                         i_clear_error,
    input  wire logic                         i_abort,
    output logic                              o_start_ready,
    input  wire logic [4:0]                   i_layer_count,
    input  wire logic [6:0]                   i_route_count,
    output logic [3:0]                        o_layer_read_index,
    input  wire logic [LAYER_WORDS*32-1:0]   i_layer_words,
    output logic [5:0]                        o_route_read_index,
    input  wire logic [63:0]                  i_route_words,

    output logic                              o_layer_valid,
    input  wire logic                         i_layer_ready,
    output logic [3:0]                        o_layer_index,
    output logic [LAYER_WORDS*32-1:0]        o_layer_words,
    output logic [3:0]                        o_inbound_valid,
    output logic [15:0]                       o_inbound_sources_flat,
    output logic [15:0]                       o_inbound_types_flat,
    input  wire logic                         i_layer_done,
    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_sequence_error
);
    localparam logic [2:0] STATE_IDLE = 3'd0;
    localparam logic [2:0] STATE_SCAN = 3'd1;
    localparam logic [2:0] STATE_ISSUE = 3'd2;
    localparam logic [2:0] STATE_WAIT = 3'd3;

    logic [2:0] state_q;
    logic [3:0] layer_q;
    logic [5:0] route_q;
    logic [3:0] inbound_valid_q;
    logic [15:0] inbound_sources_q;
    logic [15:0] inbound_types_q;
    logic route_enabled;
    logic [3:0] route_type;
    logic [3:0] route_source;
    logic [3:0] route_destination;
    logic [1:0] route_slot;

    always_comb begin
        route_enabled = i_route_words[0];
        route_type = i_route_words[4:1];
        route_source = i_route_words[8:5];
        route_destination = i_route_words[12:9];
        route_slot = i_route_words[14:13];
        o_start_ready = state_q == STATE_IDLE;
        o_layer_read_index = layer_q;
        o_route_read_index = route_q;
        o_layer_valid = state_q == STATE_ISSUE;
        o_layer_index = layer_q;
        o_layer_words = i_layer_words;
        o_inbound_valid = inbound_valid_q;
        o_inbound_sources_flat = inbound_sources_q;
        o_inbound_types_flat = inbound_types_q;
        o_busy = state_q != STATE_IDLE;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            layer_q <= 0;
            route_q <= 0;
            inbound_valid_q <= 0;
            inbound_sources_q <= 0;
            inbound_types_q <= 0;
            o_done <= 1'b0;
            o_sequence_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error)
                o_sequence_error <= 1'b0;
            if (i_abort) begin
                state_q <= STATE_IDLE;
                layer_q <= 0;
                route_q <= 0;
                inbound_valid_q <= 0;
                inbound_sources_q <= 0;
                inbound_types_q <= 0;
                o_sequence_error <= 1'b1;
            end else case (state_q)
                STATE_IDLE: begin
                    if (i_start) begin
                        if ((i_layer_count == 0) || (i_layer_count > 16) ||
                            (i_route_count > 64)) begin
                            o_sequence_error <= 1'b1;
                        end else begin
                            layer_q <= 0;
                            route_q <= 0;
                            inbound_valid_q <= 0;
                            inbound_sources_q <= 0;
                            inbound_types_q <= 0;
                            state_q <= (i_route_count == 0) ? STATE_ISSUE : STATE_SCAN;
                        end
                    end
                end
                STATE_SCAN: begin
                    if (route_enabled && (route_destination == layer_q)) begin
                        if (inbound_valid_q[route_slot]) begin
                            o_sequence_error <= 1'b1;
                            state_q <= STATE_IDLE;
                        end else begin
                            inbound_valid_q[route_slot] <= 1'b1;
                            inbound_sources_q[route_slot*4 +: 4] <= route_source;
                            inbound_types_q[route_slot*4 +: 4] <= route_type;
                        end
                    end
                    if (route_q + 1 >= i_route_count) begin
                        route_q <= 0;
                        if (!(route_enabled && (route_destination == layer_q) &&
                              inbound_valid_q[route_slot]))
                            state_q <= STATE_ISSUE;
                    end else begin
                        route_q <= route_q + 1'b1;
                    end
                end
                STATE_ISSUE: begin
                    if (i_layer_ready)
                        state_q <= STATE_WAIT;
                end
                STATE_WAIT: begin
                    if (i_layer_done) begin
                        if (layer_q + 1 >= i_layer_count) begin
                            state_q <= STATE_IDLE;
                            o_done <= 1'b1;
                        end else begin
                            layer_q <= layer_q + 1'b1;
                            route_q <= 0;
                            inbound_valid_q <= 0;
                            inbound_sources_q <= 0;
                            inbound_types_q <= 0;
                            state_q <= (i_route_count == 0) ? STATE_ISSUE : STATE_SCAN;
                        end
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
