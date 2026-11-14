`timescale 1ns/1ps
`default_nettype none

// Binds the SCNN timestep scheduler to the serialized target branch and the
// selected-layer learner. Numeric datapaths and parameter storage stay in the
// existing children. Learner completion is reported only after its accepted
// transaction has completed and the learning path is physically quiescent.
module unified_scnn_tp_target_learner_command_adapter #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_TIMESTEPS = 128,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS)
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,
    input  logic                            i_abort,

    input  logic                            i_learner_command_valid,
    output logic                            o_learner_command_ready,
    input  logic [TIMESTEP_WIDTH-1:0]       i_learner_command_timestep,
    output logic                            o_learner_start_valid,
    input  logic                            i_learner_start_ready,
    output logic [TIMESTEP_WIDTH-1:0]       o_learner_timestep,
    input  logic                            i_learner_complete,
    input  logic                            i_learner_busy,
    input  logic                            i_learner_quiescent,
    input  logic                            i_learner_error,
    output logic                            o_learner_done,

    input  logic                            i_target_command_valid,
    output logic                            o_target_command_ready,
    input  logic [SAMPLE_WIDTH-1:0]         i_target_command_sample,
    input  logic [TIMESTEP_WIDTH-1:0]       i_target_command_timestep,
    output logic                            o_target_start_valid,
    input  logic                            i_target_start_ready,
    output logic [SAMPLE_WIDTH-1:0]         o_target_sample,
    output logic [TIMESTEP_WIDTH-1:0]       o_target_timestep,
    input  logic                            i_target_done,
    input  logic                            i_target_busy,
    input  logic                            i_target_error,
    output logic                            o_target_done,

    output logic                            o_learning_transaction_active,
    output logic                            o_busy,
    output logic                            o_quiescent,
    output logic                            o_aborted,
    output logic                            o_protocol_error
);
    logic learner_active_q;
    logic learner_complete_q;
    logic target_active_q;
    logic [TIMESTEP_WIDTH-1:0] learner_timestep_q;
    logic [SAMPLE_WIDTH-1:0] target_sample_q;
    logic [TIMESTEP_WIDTH-1:0] target_timestep_q;
    logic learner_start_fire;
    logic target_start_fire;
    logic learner_finish;
    logic target_metadata_error;
    logic child_idle;

    assign child_idle = !i_learner_busy && i_learner_quiescent &&
        !i_target_busy;
    assign o_learner_command_ready = !learner_active_q && !target_active_q &&
        child_idle && !o_protocol_error && i_learner_start_ready;
    assign o_learner_start_valid = i_learner_command_valid &&
        !learner_active_q && !target_active_q && child_idle &&
        !o_protocol_error;
    assign learner_start_fire = o_learner_start_valid &&
        i_learner_start_ready;
    assign o_learner_timestep = learner_active_q ? learner_timestep_q :
        i_learner_command_timestep;

    assign target_metadata_error = i_target_command_valid &&
        learner_active_q &&
        i_target_command_timestep != learner_timestep_q;
    assign o_target_command_ready = learner_active_q && !target_active_q &&
        !learner_complete_q && !i_learner_complete &&
        !o_protocol_error && !target_metadata_error && i_target_start_ready;
    assign o_target_start_valid = i_target_command_valid &&
        learner_active_q && !target_active_q && !learner_complete_q &&
        !i_learner_complete && !o_protocol_error && !target_metadata_error;
    assign target_start_fire = o_target_start_valid && i_target_start_ready;
    assign o_target_sample = target_active_q ? target_sample_q :
        i_target_command_sample;
    assign o_target_timestep = target_active_q ? target_timestep_q :
        i_target_command_timestep;

    assign learner_finish = learner_active_q &&
        (learner_complete_q || i_learner_complete) &&
        !target_active_q && child_idle;
    assign o_learning_transaction_active = learner_active_q;
    assign o_busy = learner_active_q || target_active_q || i_learner_busy ||
        i_target_busy || !i_learner_quiescent;
    assign o_quiescent = !o_busy && !o_protocol_error &&
        !i_learner_error && !i_target_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            learner_active_q <= 1'b0;
            learner_complete_q <= 1'b0;
            target_active_q <= 1'b0;
            learner_timestep_q <= '0;
            target_sample_q <= '0;
            target_timestep_q <= '0;
            o_learner_done <= 1'b0;
            o_target_done <= 1'b0;
            o_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_learner_done <= 1'b0;
            o_target_done <= 1'b0;
            o_aborted <= 1'b0;

            if (i_clear_error && child_idle) begin
                learner_active_q <= 1'b0;
                learner_complete_q <= 1'b0;
                target_active_q <= 1'b0;
                learner_timestep_q <= '0;
                target_sample_q <= '0;
                target_timestep_q <= '0;
                o_protocol_error <= 1'b0;
            end else if (i_abort) begin
                // Training aborts must be filtered upstream until the learner
                // reaches a transaction boundary. The shared learner may be
                // busy under an independently owned forward/version
                // transaction, so an abort is reusable whenever this adapter
                // itself owns no learner or target command.
                if (!learner_active_q && !target_active_q) begin
                    o_aborted <= 1'b1;
                end else begin
                    o_protocol_error <= 1'b1;
                end
            end else begin
                if (i_learner_error || i_target_error ||
                    target_metadata_error ||
                    (i_learner_complete && !learner_active_q) ||
                    (i_target_done && !target_active_q)) begin
                    o_protocol_error <= 1'b1;
                end

                if (learner_start_fire) begin
                    learner_active_q <= 1'b1;
                    learner_complete_q <= 1'b0;
                    learner_timestep_q <= i_learner_command_timestep;
                end

                if (target_start_fire) begin
                    target_active_q <= 1'b1;
                    target_sample_q <= i_target_command_sample;
                    target_timestep_q <= i_target_command_timestep;
                end

                if (i_target_done && target_active_q) begin
                    target_active_q <= 1'b0;
                    o_target_done <= 1'b1;
                end

                if (i_learner_complete && learner_active_q)
                    learner_complete_q <= 1'b1;

                if (learner_finish) begin
                    learner_active_q <= 1'b0;
                    learner_complete_q <= 1'b0;
                    o_learner_done <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (MAX_BATCH < 1 || MAX_TIMESTEPS < 1)
            $error("SCNN target/learner command dimensions are invalid");
    end
endmodule

`default_nettype wire
