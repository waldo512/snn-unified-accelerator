`timescale 1ns/1ps
`default_nettype none

// Production client-0 lifecycle for one FC batch. The surrounding final-core
// composition performs the actual sample starts and joins forward retirement
// with the existing learning bridge before asserting i_sample_retired.
module unified_fc_tp_online_runtime_client #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter logic [PROFILE_ID_WIDTH-1:0] PROFILE_FC_H160 = 8'h01,
    parameter logic [PROFILE_ID_WIDTH-1:0] PROFILE_FC_H480 = 8'h02,
    parameter int MAX_BATCH = 16,
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1)
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,

    input  logic                            i_system_parameter_owner,
    input  logic                            i_system_learning_enable,
    input  logic                            i_system_profile_aligned,
    input  logic [PROFILE_ID_WIDTH-1:0]     i_system_profile_id,
    input  logic [PROFILE_ID_WIDTH-1:0]     i_configured_profile_id,
    input  logic                            i_bridge_configured,
    input  logic                            i_bridge_release_ready,
    input  logic                            i_execution_quiescent,
    input  logic                            i_runtime_error,

    output logic                            o_client_acquire_ready,
    output logic                            o_client_release_ready,
    input  logic                            i_client_start_valid,
    output logic                            o_client_start_ready,
    input  logic                            i_client_abort,
    output logic                            o_client_done,
    output logic                            o_client_aborted,
    output logic                            o_client_quiescent,
    output logic                            o_client_error,
    output logic                            o_client_active,

    input  logic [BATCH_COUNT_WIDTH-1:0]    i_run_sample_count,
    output logic                            o_sample_start_allowed,
    input  logic                            i_sample_start_fire,
    input  logic                            i_sample_retired,
    input  logic                            i_batch_learning_done,
    output logic                            o_run_learning_enable,
    output logic                            o_abort_run,
    input  logic                            i_abort_complete,
    output logic [BATCH_COUNT_WIDTH-1:0]    o_started_sample_count,
    output logic [BATCH_COUNT_WIDTH-1:0]    o_retired_sample_count
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_ACTIVE,
        ST_WAIT_DRAIN,
        ST_ABORT,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [BATCH_COUNT_WIDTH-1:0] sample_count_q;
    logic [BATCH_COUNT_WIDTH-1:0] started_count_q;
    logic [BATCH_COUNT_WIDTH-1:0] retired_count_q;
    logic learning_enable_q;
    logic batch_done_seen_q;
    logic profile_supported;
    logic system_admit;
    logic start_legal;
    logic client_start_fire;
    logic sample_retire_legal;
    logic final_sample_retire;
    logic all_samples_retired_after;
    logic batch_done_after;
    logic learning_complete;

    assign profile_supported =
        i_configured_profile_id == PROFILE_FC_H160 ||
        i_configured_profile_id == PROFILE_FC_H480;
    assign system_admit = i_system_parameter_owner &&
        i_system_profile_aligned && profile_supported &&
        i_system_profile_id == i_configured_profile_id;
    assign start_legal = i_run_sample_count != '0 &&
        int'(i_run_sample_count) <= MAX_BATCH;
    assign o_client_start_ready = state_q == ST_IDLE && system_admit &&
        i_bridge_configured && i_execution_quiescent && !o_client_error;
    assign client_start_fire = i_client_start_valid &&
        o_client_start_ready;
    assign o_client_acquire_ready = state_q == ST_IDLE &&
        i_bridge_configured && i_bridge_release_ready &&
        i_execution_quiescent && !o_client_error;
    assign o_client_release_ready = o_client_acquire_ready;
    assign o_sample_start_allowed = state_q == ST_ACTIVE &&
        started_count_q < sample_count_q && !i_client_abort;
    assign sample_retire_legal = retired_count_q < started_count_q ||
        (i_sample_start_fire && o_sample_start_allowed);
    assign final_sample_retire = i_sample_retired && sample_retire_legal &&
        retired_count_q + 1'b1 == sample_count_q;
    assign all_samples_retired_after = final_sample_retire ||
        retired_count_q == sample_count_q;
    assign batch_done_after = batch_done_seen_q || i_batch_learning_done;
    assign learning_complete = learning_enable_q &&
        all_samples_retired_after && batch_done_after;
    assign o_run_learning_enable = learning_enable_q;
    assign o_abort_run = state_q == ST_ABORT;
    assign o_client_quiescent = state_q == ST_IDLE &&
        i_execution_quiescent && !o_client_error;
    assign o_client_error = state_q == ST_ERROR || i_runtime_error;
    assign o_client_active = state_q == ST_ACTIVE ||
        state_q == ST_WAIT_DRAIN || state_q == ST_ABORT;
    assign o_started_sample_count = started_count_q;
    assign o_retired_sample_count = retired_count_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            sample_count_q <= '0;
            started_count_q <= '0;
            retired_count_q <= '0;
            learning_enable_q <= 1'b0;
            batch_done_seen_q <= 1'b0;
            o_client_done <= 1'b0;
            o_client_aborted <= 1'b0;
        end else begin
            o_client_done <= 1'b0;
            o_client_aborted <= 1'b0;

            if (i_runtime_error)
                state_q <= ST_ERROR;
            else begin
                unique case (state_q)
                    ST_IDLE: begin
                        if (i_clear_error) begin
                            sample_count_q <= '0;
                            started_count_q <= '0;
                            retired_count_q <= '0;
                            batch_done_seen_q <= 1'b0;
                        end
                        if (i_client_start_valid &&
                            (!system_admit || !i_bridge_configured ||
                             !i_execution_quiescent)) begin
                            state_q <= ST_ERROR;
                        end else if (client_start_fire) begin
                            if (!start_legal) begin
                                state_q <= ST_ERROR;
                            end else begin
                                sample_count_q <= i_run_sample_count;
                                started_count_q <= '0;
                                retired_count_q <= '0;
                                learning_enable_q <=
                                    i_system_learning_enable;
                                batch_done_seen_q <= 1'b0;
                                state_q <= ST_ACTIVE;
                            end
                        end
                    end

                    ST_ACTIVE: begin
                        if (!system_admit || !i_bridge_configured ||
                            (i_sample_start_fire &&
                             !o_sample_start_allowed) ||
                            (i_sample_retired && !sample_retire_legal) ||
                            (i_batch_learning_done &&
                             (!learning_enable_q ||
                              !(final_sample_retire ||
                                retired_count_q == sample_count_q)))) begin
                            state_q <= ST_ERROR;
                        end else if (i_client_abort) begin
                            state_q <= ST_ABORT;
                        end else begin
                            if (i_sample_start_fire)
                                started_count_q <= started_count_q + 1'b1;
                            if (i_sample_retired)
                                retired_count_q <= retired_count_q + 1'b1;
                            if (i_batch_learning_done)
                                batch_done_seen_q <= 1'b1;

                            if ((!learning_enable_q &&
                                 final_sample_retire) || learning_complete)
                                state_q <= ST_WAIT_DRAIN;
                        end
                    end

                    ST_WAIT_DRAIN: begin
                        if (!system_admit || !i_bridge_configured)
                            state_q <= ST_ERROR;
                        else if (i_client_abort)
                            state_q <= ST_ABORT;
                        else if (i_execution_quiescent) begin
                            o_client_done <= 1'b1;
                            state_q <= ST_IDLE;
                        end
                    end

                    ST_ABORT: begin
                        if (i_abort_complete && i_execution_quiescent) begin
                            o_client_aborted <= 1'b1;
                            state_q <= ST_IDLE;
                        end
                    end

                    ST_ERROR: begin
                        if (i_clear_error && i_execution_quiescent &&
                            !i_runtime_error)
                            state_q <= ST_IDLE;
                    end

                    default: state_q <= ST_ERROR;
                endcase
            end
        end
    end

    initial begin
        if (MAX_BATCH < 1)
            $error("FC runtime client MAX_BATCH is invalid");
    end
endmodule

`default_nettype wire
