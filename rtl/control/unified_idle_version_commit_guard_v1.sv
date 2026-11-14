`timescale 1ns/1ps
`default_nettype none

// Leaf-level contract between datapath drain, version activation and sample
// lifetime. i_datapath_quiescent deliberately excludes configuration/version
// activation so a queued commit never waits on o_global_idle.
module unified_idle_version_commit_guard_v1 (
    input  logic        i_clk,
    input  logic        i_rst_n,

    input  logic        i_datapath_quiescent,
    input  logic        i_result_valid,
    input  logic        i_result_ready,
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
    input  logic        i_owner_active,
    input  logic        i_extension_command_pending,
    input  logic        i_abort_cleanup_pending,

    input  logic        i_profile_activation_pending,
    input  logic        i_model_activation_pending,
    input  logic        i_graph_activation_pending,
    input  logic        i_adapt_activation_pending,
    input  logic        i_config_copy_busy,

    input  logic        i_commit_valid,
    output logic        o_commit_ready,
    input  logic [3:0]  i_commit_component_mask,
    input  logic [63:0] i_commit_versions,
    output logic        o_commit_pending,
    output logic        o_activation_valid,
    input  logic        i_activation_ready,
    output logic [3:0]  o_activation_component_mask,
    output logic [63:0] o_activation_versions,
    output logic        o_activation_fire,
    output logic        o_commit_reject_stale_pulse,
    output logic        o_commit_reject_mixed_pulse,

    input  logic [63:0] i_live_versions,
    input  logic        i_sample_start,
    output logic        o_sample_start_ready,
    input  logic        i_sample_compute_done,
    output logic        o_sample_active,
    output logic        o_sample_retire_pulse,
    output logic [63:0] o_locked_versions,

    input  logic        i_command_valid,
    input  logic [63:0] i_command_versions,
    output logic        o_command_accept,
    output logic        o_command_reject_stale,
    output logic        o_command_reject_mixed,

    input  logic        i_clear_error,
    input  logic        i_abort,
    output logic [25:0] o_blocker_mask,
    output logic        o_datapath_quiescent_for_commit,
    output logic        o_global_idle,
    output logic        o_busy,
    output logic        o_sticky_error,
    output logic [7:0]  o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_COMMIT_MIXED = 8'h11;
    localparam logic [7:0] ERROR_COMMIT_STALE = 8'h12;
    localparam logic [7:0] ERROR_PENDING_MIXED = 8'h13;
    localparam logic [7:0] ERROR_PENDING_STALE = 8'h14;
    localparam logic [7:0] ERROR_START_NOT_READY = 8'h20;
    localparam logic [7:0] ERROR_LIVE_VERSION_CHANGED = 8'h21;
    localparam logic [7:0] ERROR_COMMAND_STALE = 8'h30;
    localparam logic [7:0] ERROR_COMMAND_MIXED = 8'h31;

    logic        commit_pending_q;
    logic [3:0]  pending_component_mask_q;
    logic [63:0] pending_versions_q;
    logic        sample_active_q;
    logic [63:0] locked_versions_q;
    logic        compute_done_seen_q;
    logic        result_drained_seen_q;
    logic        sticky_error_q;
    logic [7:0]  error_code_q;

    logic commit_input_mixed;
    logic commit_input_stale;
    logic pending_mixed;
    logic pending_stale;
    logic sample_start_fire;
    logic result_fire;
    logic sample_retire_now;
    logic command_tuple_stale;
    logic command_tuple_mixed;

    function automatic logic has_unselected_change(
        input logic [63:0] candidate,
        input logic [63:0] reference,
        input logic [3:0]  component_mask
    );
        begin
            has_unselected_change =
                ((!component_mask[3]) &&
                 (candidate[63:48] != reference[63:48])) ||
                ((!component_mask[2]) &&
                 (candidate[47:32] != reference[47:32])) ||
                ((!component_mask[1]) &&
                 (candidate[31:16] != reference[31:16])) ||
                ((!component_mask[0]) &&
                 (candidate[15:0] != reference[15:0]));
        end
    endfunction

    function automatic logic has_selected_nonincrease(
        input logic [63:0] candidate,
        input logic [63:0] reference,
        input logic [3:0]  component_mask
    );
        begin
            has_selected_nonincrease =
                (component_mask[3] &&
                 (candidate[63:48] <= reference[63:48])) ||
                (component_mask[2] &&
                 (candidate[47:32] <= reference[47:32])) ||
                (component_mask[1] &&
                 (candidate[31:16] <= reference[31:16])) ||
                (component_mask[0] &&
                 (candidate[15:0] <= reference[15:0]));
        end
    endfunction

    function automatic logic tuple_is_fieldwise_stale(
        input logic [63:0] candidate,
        input logic [63:0] reference
    );
        logic all_not_newer;
        logic at_least_one_older;
        begin
            all_not_newer =
                (candidate[63:48] <= reference[63:48]) &&
                (candidate[47:32] <= reference[47:32]) &&
                (candidate[31:16] <= reference[31:16]) &&
                (candidate[15:0] <= reference[15:0]);
            at_least_one_older =
                (candidate[63:48] < reference[63:48]) ||
                (candidate[47:32] < reference[47:32]) ||
                (candidate[31:16] < reference[31:16]) ||
                (candidate[15:0] < reference[15:0]);
            tuple_is_fieldwise_stale = all_not_newer && at_least_one_older;
        end
    endfunction

    always_comb begin
        commit_input_mixed = (i_commit_component_mask == 4'b0000) ||
            has_unselected_change(i_commit_versions, i_live_versions,
                                  i_commit_component_mask);
        commit_input_stale = !commit_input_mixed &&
            has_selected_nonincrease(i_commit_versions, i_live_versions,
                                     i_commit_component_mask);
        pending_mixed = (pending_component_mask_q == 4'b0000) ||
            has_unselected_change(pending_versions_q, i_live_versions,
                                  pending_component_mask_q);
        pending_stale = !pending_mixed &&
            has_selected_nonincrease(pending_versions_q, i_live_versions,
                                     pending_component_mask_q);

        o_blocker_mask = '0;
        o_blocker_mask[0] = i_result_valid;
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
        o_blocker_mask[15] = i_owner_active;
        o_blocker_mask[16] = i_extension_command_pending;
        o_blocker_mask[17] = sample_active_q;
        o_blocker_mask[18] = i_abort_cleanup_pending;
        o_blocker_mask[19] = !i_datapath_quiescent;
        o_blocker_mask[20] = i_profile_activation_pending;
        o_blocker_mask[21] = i_model_activation_pending;
        o_blocker_mask[22] = i_graph_activation_pending;
        o_blocker_mask[23] = i_adapt_activation_pending;
        o_blocker_mask[24] = i_config_copy_busy;
        o_blocker_mask[25] = commit_pending_q;

        // Activation blockers are intentionally excluded from this signal.
        o_datapath_quiescent_for_commit = i_datapath_quiescent &&
                                          !(|o_blocker_mask[18:0]);
        o_busy = |o_blocker_mask;
        o_global_idle = !o_busy;

        o_commit_ready = !commit_pending_q;
        o_commit_pending = commit_pending_q;
        o_activation_component_mask = pending_component_mask_q;
        o_activation_versions = pending_versions_q;
        o_activation_valid = commit_pending_q &&
                             o_datapath_quiescent_for_commit &&
                             !pending_mixed && !pending_stale;
        o_activation_fire = o_activation_valid && i_activation_ready;

        o_sample_start_ready = o_global_idle && !sticky_error_q;
        sample_start_fire = i_sample_start && o_sample_start_ready;
        result_fire = i_result_valid && i_result_ready;
        sample_retire_now = sample_active_q &&
            (compute_done_seen_q || i_sample_compute_done) &&
            (result_drained_seen_q || result_fire);

        command_tuple_stale = tuple_is_fieldwise_stale(
            i_command_versions, locked_versions_q);
        command_tuple_mixed = (i_command_versions != locked_versions_q) &&
                              !command_tuple_stale;
        o_command_accept = i_command_valid && sample_active_q &&
                           !sticky_error_q &&
                           (i_command_versions == locked_versions_q);
        o_command_reject_stale = i_command_valid && sample_active_q &&
                                 command_tuple_stale;
        o_command_reject_mixed = i_command_valid &&
            ((!sample_active_q) || command_tuple_mixed);

        o_sample_active = sample_active_q;
        o_locked_versions = locked_versions_q;
        o_sticky_error = sticky_error_q;
        o_error_code = error_code_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            commit_pending_q <= 1'b0;
            pending_component_mask_q <= '0;
            pending_versions_q <= '0;
            sample_active_q <= 1'b0;
            locked_versions_q <= '0;
            compute_done_seen_q <= 1'b0;
            result_drained_seen_q <= 1'b0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            o_sample_retire_pulse <= 1'b0;
            o_commit_reject_stale_pulse <= 1'b0;
            o_commit_reject_mixed_pulse <= 1'b0;
        end else begin
            o_sample_retire_pulse <= 1'b0;
            o_commit_reject_stale_pulse <= 1'b0;
            o_commit_reject_mixed_pulse <= 1'b0;

            if (i_clear_error && o_global_idle) begin
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
            end

            if (i_commit_valid && o_commit_ready) begin
                if (commit_input_mixed) begin
                    o_commit_reject_mixed_pulse <= 1'b1;
                    if (!sticky_error_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_COMMIT_MIXED;
                    end
                end else if (commit_input_stale) begin
                    o_commit_reject_stale_pulse <= 1'b1;
                    if (!sticky_error_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_COMMIT_STALE;
                    end
                end else begin
                    commit_pending_q <= 1'b1;
                    pending_component_mask_q <= i_commit_component_mask;
                    pending_versions_q <= i_commit_versions;
                end
            end

            if (commit_pending_q && o_datapath_quiescent_for_commit) begin
                if (pending_mixed) begin
                    commit_pending_q <= 1'b0;
                    o_commit_reject_mixed_pulse <= 1'b1;
                    if (!sticky_error_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_PENDING_MIXED;
                    end
                end else if (pending_stale) begin
                    commit_pending_q <= 1'b0;
                    o_commit_reject_stale_pulse <= 1'b1;
                    if (!sticky_error_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_PENDING_STALE;
                    end
                end else if (o_activation_fire) begin
                    commit_pending_q <= 1'b0;
                end
            end

            if (i_abort) begin
                sample_active_q <= 1'b0;
                locked_versions_q <= '0;
                compute_done_seen_q <= 1'b0;
                result_drained_seen_q <= 1'b0;
            end else begin
                if (sample_start_fire) begin
                    sample_active_q <= 1'b1;
                    locked_versions_q <= i_live_versions;
                    compute_done_seen_q <= 1'b0;
                    result_drained_seen_q <= 1'b0;
                end else if (i_sample_start && !o_sample_start_ready) begin
                    if (!sticky_error_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_START_NOT_READY;
                    end
                end

                if (sample_active_q) begin
                    if (i_sample_compute_done)
                        compute_done_seen_q <= 1'b1;
                    if (result_fire)
                        result_drained_seen_q <= 1'b1;
                    if ((i_live_versions != locked_versions_q) &&
                        !sticky_error_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_LIVE_VERSION_CHANGED;
                    end
                    if (sample_retire_now) begin
                        sample_active_q <= 1'b0;
                        o_sample_retire_pulse <= 1'b1;
                    end
                end
            end

            if (o_command_reject_stale && !sticky_error_q) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_COMMAND_STALE;
            end else if (o_command_reject_mixed && !sticky_error_q) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_COMMAND_MIXED;
            end
        end
    end
endmodule

`default_nettype wire
