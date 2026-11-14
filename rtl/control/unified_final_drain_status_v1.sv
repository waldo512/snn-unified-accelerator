`timescale 1ns/1ps
`default_nettype none

// Collects only physical datapath/memory drain state. Configuration and
// activation state remain in the final control plane to avoid an idle loop.
module unified_final_drain_status_v1 (
    input  logic          i_clk,
    input  logic          i_rst_n,
    input  logic          i_abort,
    input  logic          i_fc_busy,
    input  logic          i_extension_busy,
    input  logic          i_dynamic_output_busy,
    input  logic          i_scnn_finalize_busy,
    input  logic          i_scnn_li_busy,
    input  logic          i_dispatch_busy,
    input  logic          i_layer_valid,
    input  logic          i_scratchpad_busy,
    input  logic          i_prefetch_busy,
    input  logic          i_writeback_busy,
    input  logic          i_ddr_cmd_pending,
    input  logic          i_ddr_response_pending,
    input  logic          i_ddr_cdc_busy,
    input  logic [2:0]    i_ddr_outstanding,
    input  logic          i_w2_commit_pending,
    input  logic          i_w2_snapshot_busy,
    input  logic          i_extension_command_pending,
    input  logic          i_result_valid,
    input  logic          i_result_ready,
    output logic          o_datapath_quiescent,
    output logic          o_layer_pending,
    output logic          o_owner_active,
    output logic          o_result_pending,
    output logic          o_abort_cleanup_pending,
    output logic [18:0]   o_physical_blocker_mask
);
    logic abort_cleanup_pending_q;
    logic abort_quiet_seen_q;

    always_comb begin
        o_result_pending = i_result_valid && !i_result_ready;
        o_layer_pending = i_layer_valid || i_dispatch_busy;
        o_owner_active = i_dispatch_busy;
        o_physical_blocker_mask = {
            o_result_pending,
            i_extension_command_pending,
            i_w2_snapshot_busy,
            i_w2_commit_pending,
            |i_ddr_outstanding,
            i_ddr_cdc_busy,
            i_ddr_response_pending,
            i_ddr_cmd_pending,
            i_writeback_busy,
            i_prefetch_busy,
            i_scratchpad_busy,
            i_scnn_li_busy,
            i_scnn_finalize_busy,
            i_dynamic_output_busy,
            i_extension_busy,
            i_fc_busy,
            i_dispatch_busy,
            i_layer_valid,
            1'b0
        };
        o_datapath_quiescent = !(|o_physical_blocker_mask);
        o_abort_cleanup_pending = abort_cleanup_pending_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            abort_cleanup_pending_q <= 1'b0;
            abort_quiet_seen_q <= 1'b0;
        end else begin
            if (i_abort) begin
                abort_cleanup_pending_q <= 1'b1;
                abort_quiet_seen_q <= 1'b0;
            end else if (abort_cleanup_pending_q && o_datapath_quiescent) begin
                if (abort_quiet_seen_q) begin
                    abort_cleanup_pending_q <= 1'b0;
                    abort_quiet_seen_q <= 1'b0;
                end else begin
                    abort_quiet_seen_q <= 1'b1;
                end
            end else if (abort_cleanup_pending_q) begin
                abort_quiet_seen_q <= 1'b0;
            end else begin
                abort_quiet_seen_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
