`timescale 1ns/1ps
`default_nettype none

// Expands one topology-neutral SRNN run into a finite sequence of existing M2
// timestep transactions. Numeric forward, learning and parameter state remain
// inside the M2 coordinator and its native services.
module unified_srnn_m2_online_run_adapter #(
    parameter int MAX_TIMESTEPS = 128,
    parameter int VERSION_WIDTH = 16,
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int TIMESTEP_ID_WIDTH =
        (MAX_TIMESTEPS <= 2) ? 1 : $clog2(MAX_TIMESTEPS)
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,

    input  logic                            i_client_start_valid,
    output logic                            o_client_start_ready,
    input  logic                            i_client_abort,
    input  logic                            i_client_learning_enable,
    input  logic [TIMESTEP_COUNT_WIDTH-1:0] i_timestep_count,
    input  logic [VERSION_WIDTH-1:0]        i_sequence_version_base,

    output logic                            o_timestep_start,
    input  logic                            i_timestep_start_ready,
    output logic [TIMESTEP_ID_WIDTH-1:0]    o_timestep_id,
    output logic [VERSION_WIDTH-1:0]        o_sequence_version,
    output logic                            o_timestep_learning_enable,
    output logic                            o_abort_timestep,
    input  logic                            i_abort_ready,
    input  logic                            i_timestep_done,
    input  logic                            i_timestep_aborted,
    input  logic                            i_timestep_busy,
    input  logic                            i_timestep_error,

    output logic                            o_client_done,
    output logic                            o_client_aborted,
    output logic                            o_client_active,
    output logic                            o_client_quiescent,
    output logic                            o_client_error,
    output logic [TIMESTEP_COUNT_WIDTH-1:0] o_completed_timestep_count
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_START,
        ST_ACTIVE,
        ST_ABORT_REQUEST,
        ST_ABORT_DRAIN,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] timestep_count_q;
    logic [TIMESTEP_ID_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sequence_version_base_q;
    logic learning_enable_q;
    logic aborted_seen_q;
    logic start_legal;
    logic client_start_fire;
    logic timestep_start_fire;
    logic final_timestep;

    assign start_legal = i_timestep_count != '0 &&
        int'(i_timestep_count) <= MAX_TIMESTEPS &&
        i_sequence_version_base != '0;
    assign o_client_start_ready = state_q == ST_IDLE &&
        !i_timestep_busy && !o_client_error;
    assign client_start_fire = i_client_start_valid &&
        o_client_start_ready;
    assign o_timestep_start = state_q == ST_START && !i_client_abort;
    assign timestep_start_fire = o_timestep_start &&
        i_timestep_start_ready;
    assign o_timestep_id = timestep_q;
    // A sequence version identifies one temporal state lifetime, not one
    // timestep. Keep it stable so membrane, trace and accumulated logits carry
    // across every timestep in the admitted run. The caller changes the base
    // between independent batches to request a logical state clear.
    assign o_sequence_version = sequence_version_base_q;
    assign o_timestep_learning_enable = learning_enable_q;
    assign o_abort_timestep = state_q == ST_ABORT_REQUEST;
    assign final_timestep =
        TIMESTEP_COUNT_WIDTH'(timestep_q) + 1'b1 == timestep_count_q;
    assign o_client_quiescent = state_q == ST_IDLE &&
        !i_timestep_busy && !o_client_error;
    assign o_client_active = state_q != ST_IDLE;
    assign o_client_error = state_q == ST_ERROR || i_timestep_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            timestep_count_q <= '0;
            timestep_q <= '0;
            sequence_version_base_q <= '0;
            learning_enable_q <= 1'b0;
            aborted_seen_q <= 1'b0;
            o_client_done <= 1'b0;
            o_client_aborted <= 1'b0;
            o_completed_timestep_count <= '0;
        end else begin
            o_client_done <= 1'b0;
            o_client_aborted <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR &&
                !i_timestep_busy && !i_timestep_error) begin
                state_q <= ST_IDLE;
                aborted_seen_q <= 1'b0;
            end else if (i_timestep_error) begin
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        aborted_seen_q <= 1'b0;
                        if (client_start_fire) begin
                            if (!start_legal) begin
                                state_q <= ST_ERROR;
                            end else begin
                                timestep_count_q <= i_timestep_count;
                                timestep_q <= '0;
                                sequence_version_base_q <=
                                    i_sequence_version_base;
                                learning_enable_q <=
                                    i_client_learning_enable;
                                o_completed_timestep_count <= '0;
                                state_q <= ST_START;
                            end
                        end
                    end

                    ST_START: begin
                        if (i_client_abort) begin
                            o_client_aborted <= 1'b1;
                            state_q <= ST_IDLE;
                        end else if (timestep_start_fire) begin
                            state_q <= ST_ACTIVE;
                        end
                    end

                    ST_ACTIVE: begin
                        if (i_client_abort) begin
                            state_q <= ST_ABORT_REQUEST;
                        end else if (i_timestep_aborted) begin
                            state_q <= ST_ERROR;
                        end else if (i_timestep_done) begin
                            o_completed_timestep_count <=
                                o_completed_timestep_count + 1'b1;
                            if (final_timestep) begin
                                o_client_done <= 1'b1;
                                state_q <= ST_IDLE;
                            end else begin
                                timestep_q <= timestep_q + 1'b1;
                                state_q <= ST_START;
                            end
                        end
                    end

                    ST_ABORT_REQUEST: begin
                        if (o_abort_timestep && i_abort_ready)
                            state_q <= ST_ABORT_DRAIN;
                    end

                    ST_ABORT_DRAIN: begin
                        if (i_timestep_aborted)
                            aborted_seen_q <= 1'b1;
                        if ((aborted_seen_q || i_timestep_aborted) &&
                            !i_timestep_busy) begin
                            o_client_aborted <= 1'b1;
                            aborted_seen_q <= 1'b0;
                            state_q <= ST_IDLE;
                        end
                    end

                    default: state_q <= ST_ERROR;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
