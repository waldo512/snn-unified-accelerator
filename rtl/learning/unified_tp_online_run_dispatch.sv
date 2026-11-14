`timescale 1ns/1ps
`default_nettype none

// Topology-neutral run boundary for the unified online-learning system. The
// selected client remains locked from common-command acceptance through final
// drain; numeric datapaths, observations and parameter storage stay outside.
module unified_tp_online_run_dispatch #(
    parameter logic [6:0] PROFILE_FC_H160 = 7'h01,
    parameter logic [6:0] PROFILE_FC_H480 = 7'h02,
    parameter logic [6:0] PROFILE_SRNN_M2 = 7'h10,
    parameter logic [6:0] PROFILE_SCNN_VGG9 = 7'h20
) (
    input  logic         i_clk,
    input  logic         i_rst_n,
    input  logic         i_clear_error,
    input  logic         i_abort,

    input  logic         i_profile_aligned,
    input  logic         i_start_allowed,
    input  logic [2:0]   i_parameter_owner,
    input  logic [2:0]   i_topology_learning_enable,
    input  logic         i_run_valid,
    output logic         o_run_ready,
    input  logic [6:0]   i_profile_code,

    output logic [2:0]   o_client_start_valid,
    input  logic [2:0]   i_client_start_ready,
    output logic [2:0]   o_client_abort,
    input  logic [2:0]   i_client_done,
    input  logic [2:0]   i_client_aborted,
    input  logic [2:0]   i_client_quiescent,
    input  logic [2:0]   i_client_error,

    output logic [2:0]   o_active_client,
    output logic [6:0]   o_active_profile_code,
    output logic         o_active_learning_enable,
    output logic         o_run_done,
    output logic         o_run_aborted,
    output logic         o_busy,
    output logic         o_quiescent,
    output logic         o_protocol_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_DISPATCH,
        ST_ACTIVE,
        ST_DRAIN_DONE,
        ST_ABORT_WAIT,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [2:0] requested_client;
    logic requested_profile_legal;
    logic requested_owner_match;
    logic [2:0] active_client_q;
    logic [6:0] active_profile_q;
    logic active_learning_q;
    logic aborted_seen_q;
    logic selected_start_fire;
    logic selected_done;
    logic selected_aborted;
    logic selected_quiescent;
    logic selected_error;
    logic unexpected_completion;
    logic clients_all_quiescent;

    always_comb begin
        requested_client = 3'b000;
        requested_profile_legal = 1'b1;
        unique case (i_profile_code)
            PROFILE_FC_H160,
            PROFILE_FC_H480: requested_client = 3'b001;
            PROFILE_SRNN_M2: requested_client = 3'b010;
            PROFILE_SCNN_VGG9: requested_client = 3'b100;
            default: begin
                requested_client = 3'b000;
                requested_profile_legal = 1'b0;
            end
        endcase
    end

    assign requested_owner_match = i_parameter_owner == requested_client;
    assign clients_all_quiescent = &i_client_quiescent;
    assign o_run_ready = state_q == ST_IDLE && requested_profile_legal &&
        i_profile_aligned && i_start_allowed && requested_owner_match &&
        clients_all_quiescent && !(|i_client_error) && !o_protocol_error;
    assign o_client_start_valid = state_q == ST_DISPATCH ?
        active_client_q : 3'b000;
    assign selected_start_fire = |(o_client_start_valid &
        i_client_start_ready);
    assign o_client_abort = ((state_q == ST_DISPATCH ||
        state_q == ST_ACTIVE) && i_abort) ? active_client_q : 3'b000;
    assign selected_done = |(i_client_done & active_client_q);
    assign selected_aborted = |(i_client_aborted & active_client_q);
    assign selected_quiescent = &(i_client_quiescent |
        ~active_client_q);
    assign selected_error = |(i_client_error & active_client_q);
    assign unexpected_completion =
        |((i_client_done | i_client_aborted) & ~active_client_q);

    assign o_active_client = active_client_q;
    assign o_active_profile_code = active_profile_q;
    assign o_active_learning_enable = active_learning_q;
    assign o_busy = state_q != ST_IDLE && state_q != ST_ERROR;
    assign o_quiescent = state_q == ST_IDLE && clients_all_quiescent &&
        !(|i_client_error) && !o_protocol_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            active_client_q <= 3'b000;
            active_profile_q <= '0;
            active_learning_q <= 1'b0;
            aborted_seen_q <= 1'b0;
            o_run_done <= 1'b0;
            o_run_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_run_done <= 1'b0;
            o_run_aborted <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR &&
                clients_all_quiescent && !(|i_client_error)) begin
                state_q <= ST_IDLE;
                active_client_q <= 3'b000;
                active_profile_q <= '0;
                active_learning_q <= 1'b0;
                aborted_seen_q <= 1'b0;
                o_protocol_error <= 1'b0;
            end else begin
                if (unexpected_completion || selected_error ||
                    (state_q == ST_IDLE && i_run_valid &&
                     !requested_profile_legal)) begin
                    state_q <= ST_ERROR;
                    o_protocol_error <= 1'b1;
                end else begin
                    unique case (state_q)
                        ST_IDLE: begin
                            active_client_q <= 3'b000;
                            active_learning_q <= 1'b0;
                            aborted_seen_q <= 1'b0;
                            if (i_run_valid && o_run_ready) begin
                                active_client_q <= requested_client;
                                active_profile_q <= i_profile_code;
                                active_learning_q <=
                                    |(i_topology_learning_enable &
                                      requested_client);
                                state_q <= ST_DISPATCH;
                            end
                        end

                        ST_DISPATCH: begin
                            if (i_abort) begin
                                aborted_seen_q <= selected_aborted;
                                state_q <= ST_ABORT_WAIT;
                            end else if (selected_start_fire) begin
                                state_q <= ST_ACTIVE;
                            end
                        end

                        ST_ACTIVE: begin
                            if (selected_done)
                                state_q <= ST_DRAIN_DONE;
                            else if (i_abort || selected_aborted) begin
                                aborted_seen_q <= selected_aborted;
                                state_q <= ST_ABORT_WAIT;
                            end
                        end

                        ST_DRAIN_DONE: begin
                            if (selected_quiescent) begin
                                o_run_done <= 1'b1;
                                active_client_q <= 3'b000;
                                active_learning_q <= 1'b0;
                                state_q <= ST_IDLE;
                            end
                        end

                        ST_ABORT_WAIT: begin
                            if (selected_aborted)
                                aborted_seen_q <= 1'b1;
                            if ((aborted_seen_q || selected_aborted) &&
                                selected_quiescent) begin
                                o_run_aborted <= 1'b1;
                                active_client_q <= 3'b000;
                                active_learning_q <= 1'b0;
                                aborted_seen_q <= 1'b0;
                                state_q <= ST_IDLE;
                            end
                        end

                        default: begin
                            state_q <= ST_ERROR;
                            o_protocol_error <= 1'b1;
                        end
                    endcase
                end
            end
        end
    end
endmodule

`default_nettype wire
