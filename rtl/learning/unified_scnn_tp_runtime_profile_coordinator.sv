`timescale 1ns/1ps
`default_nettype none

// Binds the topology-neutral profile owner to the SCNN-local timestep
// scheduler. Numeric clients and parameter storage remain outside this block.
module unified_scnn_tp_runtime_profile_coordinator #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_TIMESTEPS = 128,
    parameter int BATCH_WIDTH = $clog2(MAX_BATCH+1),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS+1),
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS)
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,

    input  logic                                i_profile_aligned,
    input  logic                                i_start_allowed,
    input  logic                                i_scnn_parameter_owner,
    input  logic                                i_scnn_learning_enable,
    input  logic                                i_clients_quiescent,

    input  logic                                i_run_start_valid,
    output logic                                o_run_start_ready,
    input  logic [BATCH_WIDTH-1:0]              i_batch_count,
    input  logic [TIMESTEP_COUNT_WIDTH-1:0]     i_timestep_count,

    output logic                                o_learner_start_valid,
    input  logic                                i_learner_start_ready,
    input  logic                                i_learner_done,
    output logic                                o_target_start_valid,
    input  logic                                i_target_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_target_timestep,
    input  logic                                i_target_done,
    output logic                                o_student_start_valid,
    input  logic                                i_student_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_student_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_student_timestep,
    input  logic                                i_student_done,
    output logic                                o_classifier_start_valid,
    input  logic                                i_classifier_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_classifier_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_classifier_timestep,
    output logic                                o_classifier_capture_enable,
    input  logic                                i_classifier_done,
    output logic                                o_post_forward_start_valid,
    input  logic                                i_post_forward_start_ready,
    output logic [TIMESTEP_WIDTH-1:0]           o_post_forward_timestep,
    input  logic                                i_post_forward_done,

    output logic [SAMPLE_WIDTH-1:0]             o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_active_timestep,
    output logic                                o_run_done,
    output logic                                o_run_aborted,
    output logic                                o_busy,
    output logic                                o_quiescent,
    output logic                                o_error
);
    logic local_start_ready;
    logic local_start_valid;
    logic local_error;
    logic context_error_q;
    logic start_context_valid;

    assign start_context_valid = i_profile_aligned && i_start_allowed &&
        i_scnn_parameter_owner;
    assign local_start_valid = i_run_start_valid && start_context_valid &&
        i_clients_quiescent;
    assign o_run_start_ready = local_start_ready && start_context_valid &&
        i_clients_quiescent;
    assign o_error = local_error || context_error_q;
    assign o_quiescent = !o_busy && i_clients_quiescent && !o_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            context_error_q <= 1'b0;
        else if (i_clear_error)
            context_error_q <= 1'b0;
        // profile_aligned follows the newest request and may drop while a
        // different profile waits for this run to drain. Ownership itself
        // must remain locked until quiescent.
        else if ((o_busy || !i_clients_quiescent) &&
                 !i_scnn_parameter_owner)
            context_error_q <= 1'b1;
    end

    unified_scnn_tp_timestep_train_infer_coordinator #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .BATCH_WIDTH(BATCH_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_timestep_coordinator (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(i_abort),
        .i_start_valid(local_start_valid), .o_start_ready(local_start_ready),
        .i_batch_count(i_batch_count), .i_timestep_count(i_timestep_count),
        .i_learning_enable(i_scnn_learning_enable),
        .o_learner_start_valid(o_learner_start_valid),
        .i_learner_start_ready(i_learner_start_ready),
        .i_learner_done(i_learner_done),
        .o_target_start_valid(o_target_start_valid),
        .i_target_start_ready(i_target_start_ready),
        .o_target_sample(o_target_sample),
        .o_target_timestep(o_target_timestep), .i_target_done(i_target_done),
        .o_student_start_valid(o_student_start_valid),
        .i_student_start_ready(i_student_start_ready),
        .o_student_sample(o_student_sample),
        .o_student_timestep(o_student_timestep),
        .i_student_done(i_student_done),
        .o_classifier_start_valid(o_classifier_start_valid),
        .i_classifier_start_ready(i_classifier_start_ready),
        .o_classifier_sample(o_classifier_sample),
        .o_classifier_timestep(o_classifier_timestep),
        .o_classifier_capture_enable(o_classifier_capture_enable),
        .i_classifier_done(i_classifier_done),
        .o_post_forward_start_valid(o_post_forward_start_valid),
        .i_post_forward_start_ready(i_post_forward_start_ready),
        .o_post_forward_timestep(o_post_forward_timestep),
        .i_post_forward_done(i_post_forward_done),
        .o_active_sample(o_active_sample),
        .o_active_timestep(o_active_timestep),
        .o_busy(o_busy), .o_done(o_run_done), .o_aborted(o_run_aborted),
        .o_protocol_error(local_error)
    );
endmodule

`default_nettype wire
