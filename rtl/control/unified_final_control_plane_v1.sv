`timescale 1ns/1ps
`default_nettype none

// Final-control candidate composed from the existing V2 graph control plane
// and the idle/version/commit guard. This wrapper binds control only; no FC,
// SRNN, SCNN, result arbiter or memory datapath is instantiated here.
module unified_final_control_plane_v1 #(
    parameter int PROFILE_COUNT = 4,
    parameter int MAX_LAYERS = 16,
    parameter int LAYER_WORDS = 12,
    parameter int MAX_ROUTES = 64
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,

    input  logic                         i_cfg_valid,
    output logic                         o_cfg_ready,
    input  logic [1:0]                   i_cfg_profile,
    input  logic [1:0]                   i_cfg_table,
    input  logic [8:0]                   i_cfg_index,
    input  logic [31:0]                  i_cfg_data,
    input  logic                         i_commit_valid,
    output logic                         o_commit_ready,
    input  logic [1:0]                   i_commit_profile,

    input  logic                         i_start,
    output logic                         o_start_ready,
    input  logic [6:0]                   i_profile_code,
    input  logic                         i_clear_error,
    input  logic                         i_abort,

    output logic                         o_layer_valid,
    input  logic                         i_layer_ready,
    output logic [3:0]                   o_layer_index,
    output logic [LAYER_WORDS*32-1:0]    o_layer_words,
    output logic [3:0]                   o_inbound_valid,
    output logic [15:0]                  o_inbound_sources_flat,
    output logic [15:0]                  o_inbound_types_flat,
    input  logic                         i_layer_done,

    input  logic                         i_result_valid,
    input  logic                         i_result_ready,
    input  logic                         i_versioned_command_valid,
    input  logic [63:0]                  i_versioned_command_versions,
    output logic                         o_versioned_command_accept,
    output logic                         o_versioned_command_reject_stale,
    output logic                         o_versioned_command_reject_mixed,

    input  logic [15:0]                  i_model_version,
    input  logic [15:0]                  i_adapt_version,
    input  logic                         i_model_activation_pending,
    input  logic                         i_adapt_activation_pending,

    input  logic                         i_datapath_quiescent,
    input  logic                         i_fc_busy,
    input  logic                         i_extension_busy,
    input  logic                         i_dynamic_output_busy,
    input  logic                         i_scnn_li_busy,
    input  logic                         i_layer_pending,
    input  logic                         i_scratchpad_busy,
    input  logic                         i_prefetch_busy,
    input  logic                         i_writeback_busy,
    input  logic                         i_ddr_cmd_pending,
    input  logic                         i_ddr_response_pending,
    input  logic                         i_ddr_cdc_busy,
    input  logic [2:0]                   i_ddr_outstanding,
    input  logic                         i_w2_commit_pending,
    input  logic                         i_w2_snapshot_busy,
    input  logic                         i_owner_active,
    input  logic                         i_extension_command_pending,
    input  logic                         i_abort_cleanup_pending,

    output logic [1:0]                   o_active_profile,
    output logic [6:0]                   o_active_profile_code,
    output logic [4:0]                   o_active_layer_count,
    output logic [6:0]                   o_active_route_count,
    output logic [63:0]                  o_live_versions,
    output logic [63:0]                  o_locked_versions,
    output logic                         o_profile_code_legal,
    output logic                         o_sequence_busy,
    output logic                         o_sample_active,
    output logic                         o_sample_compute_done,
    output logic                         o_sample_done,
    output logic                         o_commit_pending,
    output logic                         o_commit_pulse,
    output logic                         o_config_copy_busy,
    output logic [25:0]                  o_blocker_mask,
    output logic                         o_datapath_quiescent_for_commit,
    output logic                         o_control_idle,
    output logic                         o_global_idle,
    output logic                         o_busy,
    output logic                         o_version_observer_consistent,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_START_NOT_READY = 8'h40;
    localparam logic [7:0] ERROR_VERSION_OBSERVER = 8'h41;

    logic v2_start_ready;
    logic v2_sequence_busy;
    logic v2_sample_active;
    logic v2_sample_done;
    logic v2_commit_pending;
    logic v2_commit_pulse;
    logic v2_config_copy_busy;
    logic v2_control_idle;
    logic v2_sticky_error;
    logic [7:0] v2_error_code;
    logic [63:0] v2_active_versions_legacy;
    logic [63:0] v2_active_versions_reordered;

    logic [15:0] observed_profile_version_q;
    logic [15:0] observed_graph_version_q;
    logic [63:0] live_versions;
    logic [63:0] guard_locked_versions;
    logic guard_start_ready;
    logic guard_sample_active;
    logic guard_sample_retire;
    logic guard_global_idle;
    logic guard_busy;
    logic guard_sticky_error;
    logic [7:0] guard_error_code;
    logic guard_commit_ready_unused;
    logic guard_commit_pending_unused;
    logic guard_activation_valid_unused;
    logic [3:0] guard_activation_mask_unused;
    logic [63:0] guard_activation_versions_unused;
    logic guard_activation_fire_unused;
    logic guard_commit_stale_unused;
    logic guard_commit_mixed_unused;

    logic start_fire;
    logic observer_check_pending_q;
    logic local_error_q;
    logic [7:0] local_error_code_q;

    always_comb begin
        live_versions = {
            observed_profile_version_q,
            i_model_version,
            observed_graph_version_q,
            i_adapt_version
        };
        // Legacy V2 order is {profile, graph, model, adapt}.
        v2_active_versions_reordered = {
            v2_active_versions_legacy[63:48],
            v2_active_versions_legacy[31:16],
            v2_active_versions_legacy[47:32],
            v2_active_versions_legacy[15:0]
        };

        o_start_ready = v2_start_ready && guard_start_ready &&
                        !v2_sticky_error && !local_error_q;
        start_fire = i_start && o_start_ready;

        o_live_versions = live_versions;
        o_locked_versions = guard_locked_versions;
        o_sequence_busy = v2_sequence_busy;
        o_sample_active = guard_sample_active;
        o_sample_compute_done = v2_sample_done;
        o_sample_done = guard_sample_retire;
        o_commit_pending = v2_commit_pending;
        o_commit_pulse = v2_commit_pulse;
        o_config_copy_busy = v2_config_copy_busy;
        o_global_idle = guard_global_idle;
        o_busy = guard_busy;
        o_control_idle = guard_global_idle && v2_control_idle;
        o_version_observer_consistent = !guard_sample_active ||
            (v2_active_versions_reordered == guard_locked_versions);
        o_sticky_error = v2_sticky_error || guard_sticky_error ||
                         local_error_q;
        o_error_code = v2_sticky_error ? v2_error_code :
                       guard_sticky_error ? guard_error_code :
                       local_error_q ? local_error_code_q : ERROR_NONE;
    end

    // graph_config increments both versions exactly once in STATE_ACTIVATE,
    // in the same cycle as the public V2 commit pulse. The pulse therefore
    // provides an exact, synthesizable, read-only observation point.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            observed_profile_version_q <= 16'd0;
            observed_graph_version_q <= 16'd0;
            observer_check_pending_q <= 1'b0;
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else begin
            observer_check_pending_q <= start_fire;
            if (v2_commit_pulse) begin
                observed_profile_version_q <=
                    observed_profile_version_q + 1'b1;
                observed_graph_version_q <=
                    observed_graph_version_q + 1'b1;
            end
            if (i_clear_error && guard_global_idle) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end
            if (i_start && !o_start_ready && !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_START_NOT_READY;
            end
            if (observer_check_pending_q &&
                (v2_active_versions_reordered != guard_locked_versions) &&
                !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_VERSION_OBSERVER;
            end
        end
    end

    unified_snn_control_plane_v2 #(
        .PROFILE_COUNT(PROFILE_COUNT),
        .MAX_LAYERS(MAX_LAYERS),
        .LAYER_WORDS(LAYER_WORDS),
        .MAX_ROUTES(MAX_ROUTES)
    ) u_control_plane_v2 (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready),
        .i_cfg_profile(i_cfg_profile),
        .i_cfg_table(i_cfg_table),
        .i_cfg_index(i_cfg_index),
        .i_cfg_data(i_cfg_data),
        .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready),
        .i_commit_profile(i_commit_profile),
        .i_datapath_idle(o_datapath_quiescent_for_commit),
        .i_start(start_fire),
        .o_start_ready(v2_start_ready),
        .i_profile_code(i_profile_code),
        .i_model_version(i_model_version),
        .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error),
        .i_abort(i_abort),
        .o_layer_valid(o_layer_valid),
        .i_layer_ready(i_layer_ready),
        .o_layer_index(o_layer_index),
        .o_layer_words(o_layer_words),
        .o_inbound_valid(o_inbound_valid),
        .o_inbound_sources_flat(o_inbound_sources_flat),
        .o_inbound_types_flat(o_inbound_types_flat),
        .i_layer_done(i_layer_done),
        .o_active_profile(o_active_profile),
        .o_active_profile_code(o_active_profile_code),
        .o_active_layer_count(o_active_layer_count),
        .o_active_route_count(o_active_route_count),
        .o_active_versions(v2_active_versions_legacy),
        .o_profile_code_legal(o_profile_code_legal),
        .o_sequence_busy(v2_sequence_busy),
        .o_sample_active(v2_sample_active),
        .o_sample_done(v2_sample_done),
        .o_commit_pending(v2_commit_pending),
        .o_commit_pulse(v2_commit_pulse),
        .o_config_copy_busy(v2_config_copy_busy),
        .o_control_idle(v2_control_idle),
        .o_sticky_error(v2_sticky_error),
        .o_error_code(v2_error_code)
    );

    unified_idle_version_commit_guard_v1 u_idle_version_guard (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_datapath_quiescent(i_datapath_quiescent),
        .i_result_valid(i_result_valid),
        .i_result_ready(i_result_ready),
        .i_fc_busy(i_fc_busy),
        .i_extension_busy(i_extension_busy),
        .i_dynamic_output_busy(i_dynamic_output_busy),
        .i_scnn_li_busy(i_scnn_li_busy),
        .i_layer_pending(i_layer_pending || v2_sequence_busy),
        .i_scratchpad_busy(i_scratchpad_busy),
        .i_prefetch_busy(i_prefetch_busy),
        .i_writeback_busy(i_writeback_busy),
        .i_ddr_cmd_pending(i_ddr_cmd_pending),
        .i_ddr_response_pending(i_ddr_response_pending),
        .i_ddr_cdc_busy(i_ddr_cdc_busy),
        .i_ddr_outstanding(i_ddr_outstanding),
        .i_w2_commit_pending(i_w2_commit_pending),
        .i_w2_snapshot_busy(i_w2_snapshot_busy),
        .i_owner_active(i_owner_active),
        .i_extension_command_pending(i_extension_command_pending),
        .i_abort_cleanup_pending(i_abort_cleanup_pending),
        .i_profile_activation_pending(
            v2_commit_pending || v2_commit_pulse
        ),
        .i_model_activation_pending(i_model_activation_pending),
        .i_graph_activation_pending(
            v2_commit_pending || v2_commit_pulse
        ),
        .i_adapt_activation_pending(i_adapt_activation_pending),
        .i_config_copy_busy(v2_config_copy_busy),
        .i_commit_valid(1'b0),
        .o_commit_ready(guard_commit_ready_unused),
        .i_commit_component_mask(4'b0000),
        .i_commit_versions(64'd0),
        .o_commit_pending(guard_commit_pending_unused),
        .o_activation_valid(guard_activation_valid_unused),
        .i_activation_ready(1'b0),
        .o_activation_component_mask(guard_activation_mask_unused),
        .o_activation_versions(guard_activation_versions_unused),
        .o_activation_fire(guard_activation_fire_unused),
        .o_commit_reject_stale_pulse(guard_commit_stale_unused),
        .o_commit_reject_mixed_pulse(guard_commit_mixed_unused),
        .i_live_versions(live_versions),
        .i_sample_start(start_fire),
        .o_sample_start_ready(guard_start_ready),
        .i_sample_compute_done(v2_sample_done),
        .o_sample_active(guard_sample_active),
        .o_sample_retire_pulse(guard_sample_retire),
        .o_locked_versions(guard_locked_versions),
        .i_command_valid(i_versioned_command_valid),
        .i_command_versions(i_versioned_command_versions),
        .o_command_accept(o_versioned_command_accept),
        .o_command_reject_stale(o_versioned_command_reject_stale),
        .o_command_reject_mixed(o_versioned_command_reject_mixed),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .o_blocker_mask(o_blocker_mask),
        .o_datapath_quiescent_for_commit(
            o_datapath_quiescent_for_commit
        ),
        .o_global_idle(guard_global_idle),
        .o_busy(guard_busy),
        .o_sticky_error(guard_sticky_error),
        .o_error_code(guard_error_code)
    );

    // These tied-off guard commit outputs are retained as an integration
    // audit: graph commits are owned exclusively by V2 in this wrapper.
    logic _unused_guard_commit_outputs;
    always_comb begin
        _unused_guard_commit_outputs = guard_commit_ready_unused ||
            guard_commit_pending_unused || guard_activation_valid_unused ||
            (|guard_activation_mask_unused) ||
            (|guard_activation_versions_unused) ||
            guard_activation_fire_unused || guard_commit_stale_unused ||
            guard_commit_mixed_unused || v2_sample_active;
    end
endmodule

`default_nettype wire
