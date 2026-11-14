`timescale 1ns/1ps
`default_nettype none

// Production SCNN control composition. It joins the runtime profile owner,
// timestep scheduler, target/learner command adapter and final-core command
// adapter without owning arithmetic, observations, state, or parameter memory.
module unified_scnn_tp_final_system_control #(
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
    input  logic                                i_run_start_valid,
    output logic                                o_run_start_ready,
    input  logic [BATCH_WIDTH-1:0]              i_batch_count,
    input  logic [TIMESTEP_COUNT_WIDTH-1:0]     i_timestep_count,

    output logic                                o_learner_start_valid,
    input  logic                                i_learner_start_ready,
    output logic [TIMESTEP_WIDTH-1:0]           o_learner_timestep,
    input  logic                                i_learner_complete,
    input  logic                                i_learner_busy,
    input  logic                                i_learner_quiescent,
    input  logic                                i_learner_error,

    output logic                                o_target_start_valid,
    input  logic                                i_target_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_target_timestep,
    input  logic                                i_target_done,
    input  logic                                i_target_busy,
    input  logic                                i_target_error,

    output logic                                o_core_start_valid,
    input  logic                                i_core_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_core_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_core_timestep,
    output logic                                o_classifier_capture_enable,
    input  logic                                i_selected_conv_done,
    input  logic                                i_core_sample_done,
    input  logic                                i_core_quiescent,
    input  logic                                i_core_error,
    output logic                                o_core_abort,

    output logic                                o_post_forward_start_valid,
    input  logic                                i_post_forward_start_ready,
    output logic [TIMESTEP_WIDTH-1:0]           o_post_forward_timestep,
    input  logic                                i_post_forward_done,
    input  logic                                i_post_forward_quiescent,
    input  logic                                i_post_forward_error,
    output logic                                o_post_forward_abort,

    output logic [SAMPLE_WIDTH-1:0]             o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_active_timestep,
    output logic                                o_run_done,
    output logic                                o_run_aborted,
    output logic                                o_abort_pending,
    output logic                                o_busy,
    output logic                                o_quiescent,
    output logic                                o_error
);
    logic scheduler_abort;
    logic child_abort;
    logic scheduler_busy;
    logic scheduler_quiescent;
    logic scheduler_error;
    logic scheduler_learner_start_valid;
    logic scheduler_learner_start_ready;
    logic scheduler_learner_done;
    logic scheduler_target_start_valid;
    logic scheduler_target_start_ready;
    logic [SAMPLE_WIDTH-1:0] scheduler_target_sample;
    logic [TIMESTEP_WIDTH-1:0] scheduler_target_timestep;
    logic scheduler_target_done;
    logic scheduler_student_start_valid;
    logic scheduler_student_start_ready;
    logic [SAMPLE_WIDTH-1:0] scheduler_student_sample;
    logic [TIMESTEP_WIDTH-1:0] scheduler_student_timestep;
    logic scheduler_student_done;
    logic scheduler_classifier_start_valid;
    logic scheduler_classifier_start_ready;
    logic [SAMPLE_WIDTH-1:0] scheduler_classifier_sample;
    logic [TIMESTEP_WIDTH-1:0] scheduler_classifier_timestep;
    logic scheduler_classifier_done;
    logic target_learner_busy;
    logic target_learner_quiescent;
    logic target_learner_error;
    logic learning_transaction_active;
    logic final_core_adapter_busy;
    logic final_core_adapter_error;
    logic clients_quiescent;
    /* verilator lint_off UNUSEDSIGNAL */
    logic deferred_abort_fire;
    logic target_learner_aborted;
    logic final_core_adapter_aborted;
    /* verilator lint_on UNUSEDSIGNAL */

    assign clients_quiescent = target_learner_quiescent &&
        !final_core_adapter_busy && !final_core_adapter_error &&
        i_core_quiescent && !i_core_error &&
        i_post_forward_quiescent && !i_post_forward_error;
    assign o_core_abort = child_abort;
    assign o_post_forward_abort = child_abort;
    assign o_busy = scheduler_busy || target_learner_busy ||
        final_core_adapter_busy || !i_core_quiescent ||
        !i_post_forward_quiescent;
    assign o_quiescent = scheduler_quiescent && clients_quiescent &&
        !o_error;
    assign o_error = scheduler_error || target_learner_error ||
        final_core_adapter_error || i_core_error || i_post_forward_error;

    unified_scnn_tp_learning_abort_guard u_abort_guard (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort_request(i_abort),
        .i_learning_transaction_active(learning_transaction_active),
        .i_learning_transaction_safe(scheduler_learner_done),
        .o_scheduler_abort(scheduler_abort), .o_child_abort(child_abort),
        .o_abort_pending(o_abort_pending),
        .o_deferred_abort_fire(deferred_abort_fire)
    );

    unified_scnn_tp_runtime_profile_coordinator #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .BATCH_WIDTH(BATCH_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_runtime (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(scheduler_abort),
        .i_profile_aligned(i_profile_aligned),
        .i_start_allowed(i_start_allowed),
        .i_scnn_parameter_owner(i_scnn_parameter_owner),
        .i_scnn_learning_enable(i_scnn_learning_enable),
        .i_clients_quiescent(clients_quiescent),
        .i_run_start_valid(i_run_start_valid),
        .o_run_start_ready(o_run_start_ready),
        .i_batch_count(i_batch_count), .i_timestep_count(i_timestep_count),
        .o_learner_start_valid(scheduler_learner_start_valid),
        .i_learner_start_ready(scheduler_learner_start_ready),
        .i_learner_done(scheduler_learner_done),
        .o_target_start_valid(scheduler_target_start_valid),
        .i_target_start_ready(scheduler_target_start_ready),
        .o_target_sample(scheduler_target_sample),
        .o_target_timestep(scheduler_target_timestep),
        .i_target_done(scheduler_target_done),
        .o_student_start_valid(scheduler_student_start_valid),
        .i_student_start_ready(scheduler_student_start_ready),
        .o_student_sample(scheduler_student_sample),
        .o_student_timestep(scheduler_student_timestep),
        .i_student_done(scheduler_student_done),
        .o_classifier_start_valid(scheduler_classifier_start_valid),
        .i_classifier_start_ready(scheduler_classifier_start_ready),
        .o_classifier_sample(scheduler_classifier_sample),
        .o_classifier_timestep(scheduler_classifier_timestep),
        .o_classifier_capture_enable(o_classifier_capture_enable),
        .i_classifier_done(scheduler_classifier_done),
        .o_post_forward_start_valid(o_post_forward_start_valid),
        .i_post_forward_start_ready(i_post_forward_start_ready),
        .o_post_forward_timestep(o_post_forward_timestep),
        .i_post_forward_done(i_post_forward_done),
        .o_active_sample(o_active_sample),
        .o_active_timestep(o_active_timestep),
        .o_run_done(o_run_done), .o_run_aborted(o_run_aborted),
        .o_busy(scheduler_busy), .o_quiescent(scheduler_quiescent),
        .o_error(scheduler_error)
    );

    unified_scnn_tp_target_learner_command_adapter #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_target_learner_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(child_abort),
        .i_learner_command_valid(scheduler_learner_start_valid),
        .o_learner_command_ready(scheduler_learner_start_ready),
        .i_learner_command_timestep(o_active_timestep),
        .o_learner_start_valid(o_learner_start_valid),
        .i_learner_start_ready(i_learner_start_ready),
        .o_learner_timestep(o_learner_timestep),
        .i_learner_complete(i_learner_complete),
        .i_learner_busy(i_learner_busy),
        .i_learner_quiescent(i_learner_quiescent),
        .i_learner_error(i_learner_error),
        .o_learner_done(scheduler_learner_done),
        .i_target_command_valid(scheduler_target_start_valid),
        .o_target_command_ready(scheduler_target_start_ready),
        .i_target_command_sample(scheduler_target_sample),
        .i_target_command_timestep(scheduler_target_timestep),
        .o_target_start_valid(o_target_start_valid),
        .i_target_start_ready(i_target_start_ready),
        .o_target_sample(o_target_sample),
        .o_target_timestep(o_target_timestep),
        .i_target_done(i_target_done), .i_target_busy(i_target_busy),
        .i_target_error(i_target_error),
        .o_target_done(scheduler_target_done),
        .o_learning_transaction_active(learning_transaction_active),
        .o_busy(target_learner_busy),
        .o_quiescent(target_learner_quiescent),
        .o_aborted(target_learner_aborted),
        .o_protocol_error(target_learner_error)
    );

    unified_scnn_tp_final_core_command_adapter #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH), .TIMESTEP_WIDTH(TIMESTEP_WIDTH)
    ) u_final_core_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(child_abort),
        .i_student_start_valid(scheduler_student_start_valid),
        .o_student_start_ready(scheduler_student_start_ready),
        .i_student_sample(scheduler_student_sample),
        .i_student_timestep(scheduler_student_timestep),
        .o_student_done(scheduler_student_done),
        .i_classifier_start_valid(scheduler_classifier_start_valid),
        .o_classifier_start_ready(scheduler_classifier_start_ready),
        .i_classifier_sample(scheduler_classifier_sample),
        .i_classifier_timestep(scheduler_classifier_timestep),
        .o_classifier_done(scheduler_classifier_done),
        .o_core_start_valid(o_core_start_valid),
        .i_core_start_ready(i_core_start_ready),
        .i_selected_conv_done(i_selected_conv_done),
        .i_core_sample_done(i_core_sample_done),
        .o_active_sample(o_core_sample),
        .o_active_timestep(o_core_timestep),
        .o_busy(final_core_adapter_busy),
        .o_aborted(final_core_adapter_aborted),
        .o_protocol_error(final_core_adapter_error)
    );
endmodule

`default_nettype wire
