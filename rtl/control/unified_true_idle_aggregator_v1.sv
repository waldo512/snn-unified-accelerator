`default_nettype none

// Central definition of quiescence for profile and version activation.
// Every bit is a real drain condition; calibration state is intentionally not
// included because an idle FC/SRNN profile does not require calibrated DDR.
module unified_true_idle_aggregator_v1 (
    input  logic        i_result_pending,
    input  logic        i_fc_busy,
    input  logic        i_extension_busy,
    input  logic        i_dynamic_output_busy,
    input  logic        i_scnn_li_busy,
    input  logic        i_layer_pending,
    input  logic        i_scratchpad_busy,
    input  logic        i_prefetch_busy,
    input  logic        i_writeback_busy,
    input  logic        i_ddr_cmd_pending,
    input  logic        i_ddr_response_pending,
    input  logic        i_ddr_cdc_busy,
    input  logic [2:0]  i_ddr_outstanding,
    input  logic        i_w2_commit_pending,
    input  logic        i_w2_snapshot_busy,
    input  logic        i_profile_activation_pending,
    input  logic        i_model_activation_pending,
    input  logic        i_graph_activation_pending,
    input  logic        i_adapt_activation_pending,
    input  logic        i_config_copy_busy,
    input  logic        i_owner_active,
    input  logic        i_extension_command_pending,
    input  logic        i_sample_active,
    input  logic        i_abort_cleanup_pending,
    output logic [23:0] o_blocker_mask,
    output logic        o_result_drain_pending,
    output logic        o_compute_drain_pending,
    output logic        o_memory_drain_pending,
    output logic        o_version_drain_pending,
    output logic        o_global_idle,
    output logic        o_busy
);
    always_comb begin
        o_blocker_mask = '0;
        o_blocker_mask[0] = i_result_pending;
        o_blocker_mask[1] = i_fc_busy;
        o_blocker_mask[2] = i_extension_busy;
        o_blocker_mask[3] = i_dynamic_output_busy;
        o_blocker_mask[4] = i_scnn_li_busy;
        o_blocker_mask[5] = i_layer_pending;
        o_blocker_mask[6] = i_scratchpad_busy;
        o_blocker_mask[7] = i_prefetch_busy;
        o_blocker_mask[8] = i_writeback_busy;
        o_blocker_mask[9] = i_ddr_cmd_pending;
        o_blocker_mask[10] = i_ddr_response_pending;
        o_blocker_mask[11] = i_ddr_cdc_busy;
        o_blocker_mask[12] = i_ddr_outstanding != 3'd0;
        o_blocker_mask[13] = i_w2_commit_pending;
        o_blocker_mask[14] = i_w2_snapshot_busy;
        o_blocker_mask[15] = i_profile_activation_pending;
        o_blocker_mask[16] = i_model_activation_pending;
        o_blocker_mask[17] = i_graph_activation_pending;
        o_blocker_mask[18] = i_adapt_activation_pending;
        o_blocker_mask[19] = i_config_copy_busy;
        o_blocker_mask[20] = i_owner_active;
        o_blocker_mask[21] = i_extension_command_pending;
        o_blocker_mask[22] = i_sample_active;
        o_blocker_mask[23] = i_abort_cleanup_pending;

        o_result_drain_pending = |o_blocker_mask[4:0];
        o_compute_drain_pending = |o_blocker_mask[5:1] ||
                                  |o_blocker_mask[22:20];
        o_memory_drain_pending = |o_blocker_mask[12:6] ||
                                 o_blocker_mask[23];
        o_version_drain_pending = |o_blocker_mask[19:13];
        o_busy = |o_blocker_mask;
        o_global_idle = !o_busy;
    end
endmodule

`default_nettype wire
