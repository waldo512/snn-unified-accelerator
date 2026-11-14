`timescale 1ns/1ps
`default_nettype none

// Production composition boundary between topology lifecycle control and the
// single physical inference core. A topology run may issue one or more core
// commands before reporting its lifecycle completion.
module unified_tp_native_runtime_execution_fabric #(
    parameter int FC_MAX_BATCH = 16,
    parameter int FC_BATCH_COUNT_WIDTH = $clog2(FC_MAX_BATCH + 1),
    parameter int SRNN_MAX_TIMESTEPS = 128,
    parameter int VERSION_WIDTH = 16,
    parameter int SRNN_TIMESTEP_COUNT_WIDTH =
        $clog2(SRNN_MAX_TIMESTEPS + 1),
    parameter int SRNN_TIMESTEP_ID_WIDTH =
        SRNN_MAX_TIMESTEPS <= 2 ? 1 : $clog2(SRNN_MAX_TIMESTEPS)
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,

    input  logic                                i_profile_request_valid,
    input  logic [6:0]                          i_profile_request_code,
    input  logic                                i_profile_learning_enable,
    output logic                                o_profile_aligned,
    output logic                                o_start_allowed,
    output logic                                o_profile_switch_done,
    output logic                                o_profile_switch_busy,
    output logic [6:0]                          o_active_profile_code,
    output logic [15:0]                         o_profile_version,
    output logic                                o_active_learning_enable,

    input  logic                                i_run_valid,
    output logic                                o_run_ready,
    input  logic [6:0]                          i_run_profile_code,
    output logic                                o_run_done,
    output logic                                o_run_aborted,
    output logic                                o_run_busy,

    input  logic [2:0]                          i_parameter_quiescent,
    input  logic [2:0]                          i_parameter_error,
    output logic [2:0]                          o_client_acquire_valid,
    output logic [2:0]                          o_client_release_valid,
    output logic [2:0]                          o_active_client,
    output logic [2:0]                          o_parameter_owner,
    output logic [2:0]                          o_topology_learning_enable,
    output logic [2:0]                          o_client_active,
    output logic [2:0]                          o_client_quiescent,
    output logic [2:0]                          o_client_error,
    output logic                                o_control_quiescent,
    output logic                                o_system_quiescent,
    output logic                                o_system_error,
    output logic [7:0]                          o_control_error_code,
    output logic                                o_dispatch_protocol_error,

    input  logic [7:0]                          i_fc_configured_profile_id,
    input  logic                                i_fc_bridge_configured,
    input  logic                                i_fc_bridge_release_ready,
    input  logic                                i_fc_execution_quiescent,
    input  logic                                i_fc_runtime_error,
    input  logic [FC_BATCH_COUNT_WIDTH-1:0]     i_fc_run_sample_count,
    output logic                                o_fc_sample_start_allowed,
    input  logic                                i_fc_core_start_valid,
    output logic                                o_fc_core_start_ready,
    output logic                                o_fc_core_done,
    output logic                                o_fc_core_aborted,
    input  logic                                i_fc_sample_retired,
    input  logic                                i_fc_batch_learning_done,
    output logic                                o_fc_run_learning_enable,
    output logic                                o_fc_abort_run,
    input  logic                                i_fc_abort_complete,
    input  logic                                i_fc_core_abort,
    output logic [FC_BATCH_COUNT_WIDTH-1:0]     o_fc_started_sample_count,
    output logic [FC_BATCH_COUNT_WIDTH-1:0]     o_fc_retired_sample_count,

    input  logic                                i_srnn_coordinator_configured,
    input  logic [SRNN_TIMESTEP_COUNT_WIDTH-1:0]
                                                    i_srnn_timestep_count,
    input  logic [VERSION_WIDTH-1:0]            i_srnn_sequence_version_base,
    output logic                                o_srnn_timestep_start,
    input  logic                                i_srnn_timestep_start_ready,
    output logic [SRNN_TIMESTEP_ID_WIDTH-1:0]   o_srnn_timestep_id,
    output logic [VERSION_WIDTH-1:0]            o_srnn_sequence_version,
    output logic                                o_srnn_timestep_learning_enable,
    output logic                                o_srnn_abort_timestep,
    input  logic                                i_srnn_abort_ready,
    input  logic                                i_srnn_timestep_done,
    input  logic                                i_srnn_timestep_aborted,
    input  logic                                i_srnn_timestep_busy,
    input  logic                                i_srnn_timestep_error,
    input  logic                                i_srnn_core_start_valid,
    output logic                                o_srnn_core_start_ready,
    output logic                                o_srnn_core_done,
    output logic                                o_srnn_core_aborted,
    input  logic                                i_srnn_core_abort,
    output logic [SRNN_TIMESTEP_COUNT_WIDTH-1:0]
                                                    o_srnn_completed_timestep_count,

    input  logic [7:0]                          i_scnn_configured_profile_id,
    input  logic                                i_scnn_native_configured,
    output logic                                o_scnn_native_start_allowed,
    output logic                                o_scnn_native_run_start_valid,
    input  logic                                i_scnn_native_run_start_ready,
    output logic                                o_scnn_native_abort,
    input  logic                                i_scnn_native_run_done,
    input  logic                                i_scnn_native_run_aborted,
    input  logic                                i_scnn_native_busy,
    input  logic                                i_scnn_native_quiescent,
    input  logic                                i_scnn_native_error,
    input  logic                                i_scnn_core_start_valid,
    output logic                                o_scnn_core_start_ready,
    output logic                                o_scnn_core_done,
    output logic                                o_scnn_core_aborted,
    input  logic                                i_scnn_core_abort,
    output logic                                o_scnn_run_learning_enable,

    output logic                                o_core_start,
    input  logic                                i_core_start_ready,
    output logic [6:0]                          o_core_profile_code,
    output logic                                o_core_abort,
    input  logic                                i_core_sample_done,
    input  logic                                i_core_global_idle,
    input  logic                                i_core_error,
    output logic [2:0]                          o_core_active_source,
    output logic                                o_core_busy,
    output logic                                o_core_quiescent,
    output logic                                o_core_protocol_error
);
    logic runtime_start_allowed;
    logic runtime_run_ready;
    logic runtime_control_quiescent;
    logic runtime_system_quiescent;
    logic runtime_system_error;
    logic [7:0] runtime_control_error_code;
    logic runtime_dispatch_protocol_error;
    logic [2:0] runtime_parameter_quiescent;
    logic runtime_abort;
    logic physical_core_idle;
    logic fc_execution_quiescent;
    logic fc_runtime_error;
    logic fc_sample_start_fire;
    logic srnn_timestep_busy;
    logic srnn_timestep_error;
    logic scnn_native_busy;
    logic scnn_native_quiescent;
    logic scnn_native_error;
    logic selected_core_abort;

    assign physical_core_idle = !o_core_busy && i_core_global_idle;
    assign runtime_parameter_quiescent = i_parameter_quiescent &
        {3{physical_core_idle}};
    assign runtime_abort = i_abort || i_core_error ||
        o_core_protocol_error;

    assign o_start_allowed = runtime_start_allowed && o_core_quiescent;
    assign o_run_ready = runtime_run_ready && o_core_quiescent;
    assign o_control_quiescent = runtime_control_quiescent &&
        o_core_quiescent;
    assign o_system_quiescent = runtime_system_quiescent &&
        o_core_quiescent;
    assign o_system_error = runtime_system_error ||
        o_core_protocol_error;
    assign o_control_error_code = o_core_protocol_error &&
        !runtime_system_error ? 8'h40 : runtime_control_error_code;
    assign o_dispatch_protocol_error = runtime_dispatch_protocol_error ||
        o_core_protocol_error;

    assign fc_execution_quiescent = i_fc_execution_quiescent &&
        !(o_core_busy && o_core_active_source[0]);
    assign fc_runtime_error = i_fc_runtime_error ||
        (o_core_protocol_error && o_active_client[0]);
    assign fc_sample_start_fire = i_fc_core_start_valid &&
        o_fc_core_start_ready;

    assign srnn_timestep_busy = i_srnn_timestep_busy ||
        (o_core_busy && o_core_active_source[1]);
    assign srnn_timestep_error = i_srnn_timestep_error ||
        (o_core_protocol_error && o_active_client[1]);

    assign scnn_native_busy = i_scnn_native_busy ||
        (o_core_busy && o_core_active_source[2]);
    assign scnn_native_quiescent = i_scnn_native_quiescent &&
        !(o_core_busy && o_core_active_source[2]);
    assign scnn_native_error = i_scnn_native_error ||
        (o_core_protocol_error && o_active_client[2]);

    assign selected_core_abort =
        (i_fc_core_abort && o_core_active_source[0]) ||
        (i_srnn_core_abort && o_core_active_source[1]) ||
        (i_scnn_core_abort && o_core_active_source[2]);

    unified_tp_native_runtime_fabric #(
        .FC_MAX_BATCH(FC_MAX_BATCH),
        .FC_BATCH_COUNT_WIDTH(FC_BATCH_COUNT_WIDTH),
        .SRNN_MAX_TIMESTEPS(SRNN_MAX_TIMESTEPS),
        .VERSION_WIDTH(VERSION_WIDTH),
        .SRNN_TIMESTEP_COUNT_WIDTH(SRNN_TIMESTEP_COUNT_WIDTH),
        .SRNN_TIMESTEP_ID_WIDTH(SRNN_TIMESTEP_ID_WIDTH)
    ) u_runtime (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(runtime_abort),
        .i_profile_request_valid(i_profile_request_valid),
        .i_profile_request_code(i_profile_request_code),
        .i_profile_learning_enable(i_profile_learning_enable),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(runtime_start_allowed),
        .o_profile_switch_done(o_profile_switch_done),
        .o_profile_switch_busy(o_profile_switch_busy),
        .o_active_profile_code(o_active_profile_code),
        .o_profile_version(o_profile_version),
        .o_active_learning_enable(o_active_learning_enable),
        .i_run_valid(i_run_valid && o_core_quiescent),
        .o_run_ready(runtime_run_ready),
        .i_run_profile_code(i_run_profile_code),
        .o_run_done(o_run_done), .o_run_aborted(o_run_aborted),
        .o_run_busy(o_run_busy),
        .i_parameter_quiescent(runtime_parameter_quiescent),
        .i_parameter_error(i_parameter_error),
        .o_client_acquire_valid(o_client_acquire_valid),
        .o_client_release_valid(o_client_release_valid),
        .o_active_client(o_active_client),
        .o_parameter_owner(o_parameter_owner),
        .o_topology_learning_enable(o_topology_learning_enable),
        .o_client_active(o_client_active),
        .o_client_quiescent(o_client_quiescent),
        .o_client_error(o_client_error),
        .o_control_quiescent(runtime_control_quiescent),
        .o_system_quiescent(runtime_system_quiescent),
        .o_system_error(runtime_system_error),
        .o_control_error_code(runtime_control_error_code),
        .o_dispatch_protocol_error(runtime_dispatch_protocol_error),
        .i_fc_configured_profile_id(i_fc_configured_profile_id),
        .i_fc_bridge_configured(i_fc_bridge_configured),
        .i_fc_bridge_release_ready(i_fc_bridge_release_ready),
        .i_fc_execution_quiescent(fc_execution_quiescent),
        .i_fc_runtime_error(fc_runtime_error),
        .i_fc_run_sample_count(i_fc_run_sample_count),
        .o_fc_sample_start_allowed(o_fc_sample_start_allowed),
        .i_fc_sample_start_fire(fc_sample_start_fire),
        .i_fc_sample_retired(i_fc_sample_retired),
        .i_fc_batch_learning_done(i_fc_batch_learning_done),
        .o_fc_run_learning_enable(o_fc_run_learning_enable),
        .o_fc_abort_run(o_fc_abort_run),
        .i_fc_abort_complete(i_fc_abort_complete),
        .o_fc_started_sample_count(o_fc_started_sample_count),
        .o_fc_retired_sample_count(o_fc_retired_sample_count),
        .i_srnn_coordinator_configured(i_srnn_coordinator_configured),
        .i_srnn_timestep_count(i_srnn_timestep_count),
        .i_srnn_sequence_version_base(i_srnn_sequence_version_base),
        .o_srnn_timestep_start(o_srnn_timestep_start),
        .i_srnn_timestep_start_ready(i_srnn_timestep_start_ready),
        .o_srnn_timestep_id(o_srnn_timestep_id),
        .o_srnn_sequence_version(o_srnn_sequence_version),
        .o_srnn_timestep_learning_enable(
            o_srnn_timestep_learning_enable),
        .o_srnn_abort_timestep(o_srnn_abort_timestep),
        .i_srnn_abort_ready(i_srnn_abort_ready),
        .i_srnn_timestep_done(i_srnn_timestep_done),
        .i_srnn_timestep_aborted(i_srnn_timestep_aborted),
        .i_srnn_timestep_busy(srnn_timestep_busy),
        .i_srnn_timestep_error(srnn_timestep_error),
        .o_srnn_completed_timestep_count(
            o_srnn_completed_timestep_count),
        .i_scnn_configured_profile_id(i_scnn_configured_profile_id),
        .i_scnn_native_configured(i_scnn_native_configured),
        .o_scnn_native_start_allowed(o_scnn_native_start_allowed),
        .o_scnn_native_run_start_valid(o_scnn_native_run_start_valid),
        .i_scnn_native_run_start_ready(i_scnn_native_run_start_ready),
        .o_scnn_native_abort(o_scnn_native_abort),
        .i_scnn_native_run_done(i_scnn_native_run_done),
        .i_scnn_native_run_aborted(i_scnn_native_run_aborted),
        .i_scnn_native_busy(scnn_native_busy),
        .i_scnn_native_quiescent(scnn_native_quiescent),
        .i_scnn_native_error(scnn_native_error),
        .o_scnn_run_learning_enable(o_scnn_run_learning_enable)
    );

    unified_tp_shared_inference_run_arbiter u_core_arbiter (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        // Lifecycle abort is first interpreted by the selected topology.
        // Only its explicit core-abort request, a physical core failure or an
        // arbiter protocol failure may interrupt an in-flight core command.
        .i_abort(selected_core_abort || i_core_error ||
            o_core_protocol_error),
        .i_active_client(o_active_client),
        .i_active_profile_code(o_active_profile_code),
        .i_fc_start_valid(i_fc_core_start_valid),
        .o_fc_start_ready(o_fc_core_start_ready),
        .o_fc_done(o_fc_core_done),
        .o_fc_aborted(o_fc_core_aborted),
        .i_srnn_start_valid(i_srnn_core_start_valid),
        .o_srnn_start_ready(o_srnn_core_start_ready),
        .o_srnn_done(o_srnn_core_done),
        .o_srnn_aborted(o_srnn_core_aborted),
        .i_scnn_start_valid(i_scnn_core_start_valid),
        .o_scnn_start_ready(o_scnn_core_start_ready),
        .o_scnn_done(o_scnn_core_done),
        .o_scnn_aborted(o_scnn_core_aborted),
        .o_core_start(o_core_start),
        .i_core_start_ready(i_core_start_ready),
        .o_core_profile_code(o_core_profile_code),
        .o_core_abort(o_core_abort),
        .i_core_sample_done(i_core_sample_done),
        .i_core_global_idle(i_core_global_idle),
        .i_core_error(i_core_error),
        .o_active_source(o_core_active_source),
        .o_busy(o_core_busy),
        .o_quiescent(o_core_quiescent),
        .o_protocol_error(o_core_protocol_error)
    );
endmodule

`default_nettype wire
