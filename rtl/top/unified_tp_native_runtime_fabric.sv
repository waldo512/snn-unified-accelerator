`timescale 1ns/1ps
`default_nettype none

// One common profile/run plane with topology-native execution and parameter
// services. This module contains lifecycle state only; numeric datapaths and
// parameter memories remain authoritative in the connected FC/SRNN/SCNN cores.
module unified_tp_native_runtime_fabric #(
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
    input  logic                                i_fc_sample_start_fire,
    input  logic                                i_fc_sample_retired,
    input  logic                                i_fc_batch_learning_done,
    output logic                                o_fc_run_learning_enable,
    output logic                                o_fc_abort_run,
    input  logic                                i_fc_abort_complete,
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
    output logic                                o_scnn_run_learning_enable
);
    logic [2:0] client_acquire_ready;
    logic [2:0] client_release_ready;
    logic [2:0] client_start_valid;
    logic [2:0] client_start_ready;
    logic [2:0] client_abort;
    logic [2:0] client_done;
    logic [2:0] client_aborted;
    logic fc_client_active;
    logic srnn_client_active;
    logic scnn_client_active;

    assign o_client_active =
        {scnn_client_active, srnn_client_active, fc_client_active};

    unified_tp_online_control_plane u_control (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_profile_request_valid(i_profile_request_valid),
        .i_profile_request_code(i_profile_request_code),
        .i_profile_learning_enable(i_profile_learning_enable),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(o_start_allowed),
        .o_profile_switch_done(o_profile_switch_done),
        .o_profile_switch_busy(o_profile_switch_busy),
        .o_active_profile_code(o_active_profile_code),
        .o_profile_version(o_profile_version),
        .o_active_learning_enable(o_active_learning_enable),
        .i_run_valid(i_run_valid), .o_run_ready(o_run_ready),
        .i_run_profile_code(i_run_profile_code),
        .o_run_done(o_run_done), .o_run_aborted(o_run_aborted),
        .o_run_busy(o_run_busy),
        .o_client_acquire_valid(o_client_acquire_valid),
        .i_client_acquire_ready(client_acquire_ready),
        .o_client_release_valid(o_client_release_valid),
        .i_client_release_ready(client_release_ready),
        .o_client_start_valid(client_start_valid),
        .i_client_start_ready(client_start_ready),
        .o_client_abort(client_abort),
        .i_client_done(client_done),
        .i_client_aborted(client_aborted),
        .i_client_quiescent(o_client_quiescent),
        .i_client_error(o_client_error),
        .i_parameter_quiescent(i_parameter_quiescent),
        .i_parameter_error(i_parameter_error),
        .o_active_client(o_active_client),
        .o_parameter_owner(o_parameter_owner),
        .o_topology_learning_enable(o_topology_learning_enable),
        .o_control_quiescent(o_control_quiescent),
        .o_system_quiescent(o_system_quiescent),
        .o_system_error(o_system_error),
        .o_control_error_code(o_control_error_code),
        .o_dispatch_protocol_error(o_dispatch_protocol_error)
    );

    unified_fc_tp_online_runtime_client #(
        .MAX_BATCH(FC_MAX_BATCH),
        .BATCH_COUNT_WIDTH(FC_BATCH_COUNT_WIDTH)
    ) u_fc_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_system_parameter_owner(o_parameter_owner[0]),
        .i_system_learning_enable(o_topology_learning_enable[0]),
        .i_system_profile_aligned(o_profile_aligned),
        .i_system_profile_id({1'b0, o_active_profile_code}),
        .i_configured_profile_id(i_fc_configured_profile_id),
        .i_bridge_configured(i_fc_bridge_configured),
        .i_bridge_release_ready(i_fc_bridge_release_ready),
        .i_execution_quiescent(i_fc_execution_quiescent),
        .i_runtime_error(i_fc_runtime_error),
        .o_client_acquire_ready(client_acquire_ready[0]),
        .o_client_release_ready(client_release_ready[0]),
        .i_client_start_valid(client_start_valid[0]),
        .o_client_start_ready(client_start_ready[0]),
        .i_client_abort(client_abort[0]),
        .o_client_done(client_done[0]),
        .o_client_aborted(client_aborted[0]),
        .o_client_quiescent(o_client_quiescent[0]),
        .o_client_error(o_client_error[0]),
        .o_client_active(fc_client_active),
        .i_run_sample_count(i_fc_run_sample_count),
        .o_sample_start_allowed(o_fc_sample_start_allowed),
        .i_sample_start_fire(i_fc_sample_start_fire),
        .i_sample_retired(i_fc_sample_retired),
        .i_batch_learning_done(i_fc_batch_learning_done),
        .o_run_learning_enable(o_fc_run_learning_enable),
        .o_abort_run(o_fc_abort_run),
        .i_abort_complete(i_fc_abort_complete),
        .o_started_sample_count(o_fc_started_sample_count),
        .o_retired_sample_count(o_fc_retired_sample_count)
    );

    unified_srnn_m2_online_runtime_client #(
        .MAX_TIMESTEPS(SRNN_MAX_TIMESTEPS),
        .VERSION_WIDTH(VERSION_WIDTH),
        .TIMESTEP_COUNT_WIDTH(SRNN_TIMESTEP_COUNT_WIDTH),
        .TIMESTEP_ID_WIDTH(SRNN_TIMESTEP_ID_WIDTH)
    ) u_srnn_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_system_parameter_owner(o_parameter_owner[1]),
        .i_system_learning_enable(o_topology_learning_enable[1]),
        .i_system_profile_aligned(o_profile_aligned),
        .i_system_profile_id({1'b0, o_active_profile_code}),
        .i_coordinator_configured(i_srnn_coordinator_configured),
        .o_client_acquire_ready(client_acquire_ready[1]),
        .o_client_release_ready(client_release_ready[1]),
        .i_client_start_valid(client_start_valid[1]),
        .o_client_start_ready(client_start_ready[1]),
        .i_client_abort(client_abort[1]),
        .o_client_done(client_done[1]),
        .o_client_aborted(client_aborted[1]),
        .o_client_quiescent(o_client_quiescent[1]),
        .o_client_error(o_client_error[1]),
        .o_client_active(srnn_client_active),
        .i_timestep_count(i_srnn_timestep_count),
        .i_sequence_version_base(i_srnn_sequence_version_base),
        .o_timestep_start(o_srnn_timestep_start),
        .i_timestep_start_ready(i_srnn_timestep_start_ready),
        .o_timestep_id(o_srnn_timestep_id),
        .o_sequence_version(o_srnn_sequence_version),
        .o_timestep_learning_enable(o_srnn_timestep_learning_enable),
        .o_abort_timestep(o_srnn_abort_timestep),
        .i_abort_ready(i_srnn_abort_ready),
        .i_timestep_done(i_srnn_timestep_done),
        .i_timestep_aborted(i_srnn_timestep_aborted),
        .i_timestep_busy(i_srnn_timestep_busy),
        .i_timestep_error(i_srnn_timestep_error),
        .o_completed_timestep_count(o_srnn_completed_timestep_count)
    );

    unified_scnn_tp_common_runtime_client u_scnn_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_system_parameter_owner(o_parameter_owner[2]),
        .i_system_learning_enable(o_topology_learning_enable[2]),
        .i_system_profile_aligned(o_profile_aligned),
        .i_system_start_allowed(o_start_allowed),
        .i_system_profile_id({1'b0, o_active_profile_code}),
        .i_configured_profile_id(i_scnn_configured_profile_id),
        .o_client_acquire_ready(client_acquire_ready[2]),
        .o_client_release_ready(client_release_ready[2]),
        .i_client_start_valid(client_start_valid[2]),
        .o_client_start_ready(client_start_ready[2]),
        .i_client_abort(client_abort[2]),
        .o_client_done(client_done[2]),
        .o_client_aborted(client_aborted[2]),
        .o_client_quiescent(o_client_quiescent[2]),
        .o_client_error(o_client_error[2]),
        .o_client_active(scnn_client_active),
        .i_native_configured(i_scnn_native_configured),
        .o_native_start_allowed(o_scnn_native_start_allowed),
        .o_native_run_start_valid(o_scnn_native_run_start_valid),
        .i_native_run_start_ready(i_scnn_native_run_start_ready),
        .o_native_abort(o_scnn_native_abort),
        .i_native_run_done(i_scnn_native_run_done),
        .i_native_run_aborted(i_scnn_native_run_aborted),
        .i_native_busy(i_scnn_native_busy),
        .i_native_quiescent(i_scnn_native_quiescent),
        .i_native_error(i_scnn_native_error),
        .o_run_learning_enable(o_scnn_run_learning_enable)
    );
endmodule

`default_nettype wire
