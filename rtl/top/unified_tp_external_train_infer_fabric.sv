`timescale 1ns/1ps
`default_nettype none

// Production train/infer control fabric with one physical-core arbiter and an
// abstract parameter-store port. FC may retain its native banked weight path;
// SRNN/SCNN transactions share this owner-controlled external backend.
module unified_tp_external_train_infer_fabric #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
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
    output logic [FC_BATCH_COUNT_WIDTH-1:0]
                                                    o_fc_started_sample_count,
    output logic [FC_BATCH_COUNT_WIDTH-1:0]
                                                    o_fc_retired_sample_count,

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
    output logic                                o_core_protocol_error,

    input  logic [2:0]                          i_load_valid,
    output logic [2:0]                          o_load_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_load_addr,
    input  logic signed [3*CODE_WIDTH-1:0]      i_load_code,
    input  logic signed [3*RESIDUAL_WIDTH-1:0]  i_load_residual,
    input  logic [2:0]                          i_learning_read_valid,
    output logic [2:0]                          o_learning_read_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_learning_read_addr,
    input  logic [3*TAG_WIDTH-1:0]              i_learning_read_tag,
    output logic [2:0]                          o_learning_response_valid,
    input  logic [2:0]                          i_learning_response_ready,
    output logic [3*TAG_WIDTH-1:0]              o_learning_response_tag,
    output logic signed [3*CODE_WIDTH-1:0]
                                                    o_learning_response_code,
    output logic signed [3*RESIDUAL_WIDTH-1:0]
                                                    o_learning_response_residual,
    output logic [2:0]                          o_learning_response_error,
    input  logic [2:0]                          i_learning_write_valid,
    output logic [2:0]                          o_learning_write_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_learning_write_addr,
    input  logic signed [3*CODE_WIDTH-1:0]      i_learning_write_code,
    input  logic signed [3*RESIDUAL_WIDTH-1:0]
                                                    i_learning_write_residual,
    input  logic [2:0]                          i_learning_write_code_enable,
    input  logic [2:0]
                                                    i_learning_write_residual_enable,
    input  logic [2:0]                          i_forward_read_valid,
    output logic [2:0]                          o_forward_read_ready,
    input  logic [3*PARAM_ADDR_WIDTH-1:0]       i_forward_read_addr,
    input  logic [3*TAG_WIDTH-1:0]              i_forward_read_tag,
    output logic [2:0]                          o_forward_response_valid,
    input  logic [2:0]                          i_forward_response_ready,
    output logic [3*TAG_WIDTH-1:0]              o_forward_response_tag,
    output logic signed [3*CODE_WIDTH-1:0]      o_forward_response_code,
    output logic [2:0]                          o_forward_response_error,

    output logic                                o_store_load_valid,
    input  logic                                i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0]        o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_store_load_residual,
    output logic                                o_store_learning_read_valid,
    input  logic                                i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                o_store_learning_read_tag,
    input  logic                                i_store_learning_response_valid,
    output logic                                o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]
                                                    i_store_learning_response_residual,
    input  logic                                i_store_learning_response_error,
    output logic                                o_store_learning_write_valid,
    input  logic                                i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0]        o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]
                                                    o_store_learning_write_residual,
    output logic                                o_store_learning_write_code_enable,
    output logic
                                                    o_store_learning_write_residual_enable,
    output logic                                o_store_forward_read_valid,
    input  logic                                i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                o_store_forward_read_tag,
    input  logic                                i_store_forward_response_valid,
    output logic                                o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_store_forward_response_code,
    input  logic                                i_store_forward_response_error,
    input  logic                                i_store_quiescent,
    input  logic                                i_store_protocol_error,
    output logic                                o_backend_quiescent,
    output logic                                o_backend_error,
    output logic                                o_owner_encoding_error,
    output logic                                o_router_protocol_error
);
    logic [2:0] runtime_parameter_quiescent;
    logic [2:0] runtime_parameter_error;
    logic runtime_system_quiescent;
    logic runtime_system_error;

    assign runtime_parameter_quiescent = i_parameter_quiescent &
        {3{o_backend_quiescent}};
    assign runtime_parameter_error = i_parameter_error |
        {3{o_backend_error}};
    assign o_system_quiescent = runtime_system_quiescent &&
        o_backend_quiescent && !o_backend_error;
    assign o_system_error = runtime_system_error || o_backend_error;

    unified_tp_native_runtime_execution_fabric #(
        .FC_MAX_BATCH(FC_MAX_BATCH),
        .FC_BATCH_COUNT_WIDTH(FC_BATCH_COUNT_WIDTH),
        .SRNN_MAX_TIMESTEPS(SRNN_MAX_TIMESTEPS),
        .VERSION_WIDTH(VERSION_WIDTH),
        .SRNN_TIMESTEP_COUNT_WIDTH(SRNN_TIMESTEP_COUNT_WIDTH),
        .SRNN_TIMESTEP_ID_WIDTH(SRNN_TIMESTEP_ID_WIDTH)
    ) u_runtime (
        .i_parameter_quiescent(runtime_parameter_quiescent),
        .i_parameter_error(runtime_parameter_error),
        .o_system_quiescent(runtime_system_quiescent),
        .o_system_error(runtime_system_error),
        .*
    );

    unified_tp_external_parameter_backend #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
    ) u_backend (
        .i_parameter_owner(o_parameter_owner),
        .i_topology_learning_enable(o_topology_learning_enable),
        .*
    );

    initial begin
        if (PARAM_ADDR_WIDTH < 1 || TAG_WIDTH < 1 || CODE_WIDTH < 2 ||
            RESIDUAL_WIDTH < 2)
            $error("external train/infer fabric dimensions are invalid");
    end
endmodule

`default_nettype wire
