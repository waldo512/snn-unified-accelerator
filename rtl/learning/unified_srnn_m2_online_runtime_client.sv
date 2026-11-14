`timescale 1ns/1ps
`default_nettype none

// Production client-1 wrapper for the SRNN M2 run lifecycle. It validates the
// common owner/profile context and expands one accepted run into existing M2
// timestep transactions. Numeric, observation and parameter state stay in the
// coordinator and its native Q2.14 services.
module unified_srnn_m2_online_runtime_client #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter logic [PROFILE_ID_WIDTH-1:0] PROFILE_ID = 8'h10,
    parameter int MAX_TIMESTEPS = 128,
    parameter int VERSION_WIDTH = 16,
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int TIMESTEP_ID_WIDTH =
        (MAX_TIMESTEPS <= 2) ? 1 : $clog2(MAX_TIMESTEPS)
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,

    input  logic                            i_system_parameter_owner,
    input  logic                            i_system_learning_enable,
    input  logic                            i_system_profile_aligned,
    input  logic [PROFILE_ID_WIDTH-1:0]     i_system_profile_id,
    input  logic                            i_coordinator_configured,

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

    output logic [TIMESTEP_COUNT_WIDTH-1:0]
                                                o_completed_timestep_count
);
    logic system_admit;
    logic adapter_start_valid;
    logic adapter_start_ready;
    logic adapter_done;
    logic adapter_aborted;
    logic adapter_active;
    logic adapter_quiescent;
    logic adapter_error;
    logic context_error_q;

    assign system_admit = i_system_parameter_owner &&
        i_system_profile_aligned && i_system_profile_id == PROFILE_ID;
    assign adapter_start_valid = i_client_start_valid && system_admit &&
        i_coordinator_configured && !context_error_q;
    assign o_client_start_ready = adapter_start_ready && system_admit &&
        i_coordinator_configured && !context_error_q;
    assign o_client_acquire_ready = i_coordinator_configured &&
        adapter_quiescent && !adapter_error && !context_error_q;
    assign o_client_release_ready = adapter_quiescent &&
        !adapter_error && !context_error_q;
    assign o_client_done = adapter_done;
    assign o_client_aborted = adapter_aborted;
    assign o_client_quiescent = adapter_quiescent && !context_error_q;
    assign o_client_error = adapter_error || context_error_q;
    assign o_client_active = adapter_active;

    // A dispatched start without its registered owner/profile context, or
    // loss of that context during a run, indicates integration failure.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            context_error_q <= 1'b0;
        else if (i_clear_error && adapter_quiescent && !adapter_error &&
                 !i_timestep_error)
            context_error_q <= 1'b0;
        else if ((i_client_start_valid &&
                  (!system_admit || !i_coordinator_configured)) ||
                 (adapter_active &&
                  (!system_admit || !i_coordinator_configured)))
            context_error_q <= 1'b1;
    end

    unified_srnn_m2_online_run_adapter #(
        .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .VERSION_WIDTH(VERSION_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .TIMESTEP_ID_WIDTH(TIMESTEP_ID_WIDTH)
    ) u_run_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_client_start_valid(adapter_start_valid),
        .o_client_start_ready(adapter_start_ready),
        .i_client_abort(i_client_abort),
        .i_client_learning_enable(i_system_learning_enable),
        .i_timestep_count(i_timestep_count),
        .i_sequence_version_base(i_sequence_version_base),
        .o_timestep_start(o_timestep_start),
        .i_timestep_start_ready(i_timestep_start_ready),
        .o_timestep_id(o_timestep_id),
        .o_sequence_version(o_sequence_version),
        .o_timestep_learning_enable(o_timestep_learning_enable),
        .o_abort_timestep(o_abort_timestep),
        .i_abort_ready(i_abort_ready),
        .i_timestep_done(i_timestep_done),
        .i_timestep_aborted(i_timestep_aborted),
        .i_timestep_busy(i_timestep_busy),
        .i_timestep_error(i_timestep_error),
        .o_client_done(adapter_done),
        .o_client_aborted(adapter_aborted),
        .o_client_active(adapter_active),
        .o_client_quiescent(adapter_quiescent),
        .o_client_error(adapter_error),
        .o_completed_timestep_count(o_completed_timestep_count)
    );
endmodule

`default_nettype wire
