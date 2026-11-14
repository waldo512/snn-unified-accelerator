`timescale 1ns/1ps
`default_nettype none

// Client-2 control adapter for the production SCNN online-system binding.
// SCNN arithmetic, observation state and Q8.8 parameter storage remain in
// their native modules; this block only translates common run ownership.
module unified_scnn_tp_common_runtime_client #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter logic [PROFILE_ID_WIDTH-1:0] PROFILE_SCNN_VGG9 = 8'h20
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,

    input  logic                            i_system_parameter_owner,
    input  logic                            i_system_learning_enable,
    input  logic                            i_system_profile_aligned,
    input  logic                            i_system_start_allowed,
    input  logic [PROFILE_ID_WIDTH-1:0]     i_system_profile_id,
    input  logic [PROFILE_ID_WIDTH-1:0]     i_configured_profile_id,

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

    input  logic                            i_native_configured,
    output logic                            o_native_start_allowed,
    output logic                            o_native_run_start_valid,
    input  logic                            i_native_run_start_ready,
    output logic                            o_native_abort,
    input  logic                            i_native_run_done,
    input  logic                            i_native_run_aborted,
    input  logic                            i_native_busy,
    input  logic                            i_native_quiescent,
    input  logic                            i_native_error,
    output logic                            o_run_learning_enable
);
    logic active_q;
    logic learning_enable_q;
    logic wrapper_error_q;
    logic profile_supported;
    logic system_admit;
    logic native_start_fire;

    assign profile_supported =
        i_configured_profile_id == PROFILE_SCNN_VGG9;
    assign system_admit = i_system_parameter_owner &&
        i_system_profile_aligned && profile_supported &&
        i_system_profile_id == i_configured_profile_id;
    assign o_native_start_allowed = system_admit &&
        (i_system_start_allowed || i_client_start_valid) &&
        !o_client_error;
    assign o_native_run_start_valid = i_client_start_valid &&
        system_admit && i_native_configured && !o_client_error;
    assign o_client_start_ready = i_native_run_start_ready &&
        system_admit && i_native_configured && !active_q &&
        !o_client_error;
    assign native_start_fire = o_native_run_start_valid &&
        i_native_run_start_ready;
    assign o_native_abort = i_client_abort && active_q;
    assign o_client_done = i_native_run_done && active_q &&
        !o_client_error;
    assign o_client_aborted = i_native_run_aborted && active_q &&
        !o_client_error;
    assign o_client_active = active_q || i_native_busy;
    assign o_client_quiescent = !active_q && i_native_quiescent &&
        !o_client_error;
    assign o_client_error = wrapper_error_q || i_native_error;
    assign o_client_acquire_ready = i_native_configured &&
        i_native_quiescent && !active_q && !o_client_error;
    assign o_client_release_ready = o_client_acquire_ready;
    assign o_run_learning_enable = active_q ? learning_enable_q :
        i_system_learning_enable;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            learning_enable_q <= 1'b0;
            wrapper_error_q <= 1'b0;
        end else begin
            if (i_clear_error && !active_q && i_native_quiescent &&
                !i_native_error)
                wrapper_error_q <= 1'b0;

            if (native_start_fire) begin
                active_q <= 1'b1;
                learning_enable_q <= i_system_learning_enable;
            end

            if ((i_native_run_done || i_native_run_aborted) && active_q)
                active_q <= 1'b0;

            if (i_client_start_valid &&
                (!system_admit || !i_native_configured))
                wrapper_error_q <= 1'b1;
            if (active_q &&
                (!system_admit ||
                 i_system_learning_enable != learning_enable_q))
                wrapper_error_q <= 1'b1;
            if ((i_native_run_done || i_native_run_aborted) && !active_q)
                wrapper_error_q <= 1'b1;
            if (i_native_run_done && i_native_run_aborted)
                wrapper_error_q <= 1'b1;
        end
    end

    initial begin
        if (PROFILE_ID_WIDTH < 7)
            $error("SCNN common runtime profile width is invalid");
    end
endmodule

`default_nettype wire
