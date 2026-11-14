`timescale 1ns/1ps
`default_nettype none

// Wave-2 binding stage: the orchestration shell owns one real canonical H480
// adapter. SRNN and SCNN remain explicit extension boundaries until their
// shared-side-array frontends pass their independent gates.
module unified_snn_core_fc_bound_stage #(
    parameter FC_WEIGHT_DIR =
        "../../../data/weights/r19b_h480_complete_fc_input2312",
    parameter int TILES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,
    input  logic                                i_start,
    output logic                                o_start_ready,
    input  logic [6:0]                          i_profile_code,
    input  logic [15:0]                         i_model_version,
    input  logic [15:0]                         i_graph_version,
    input  logic [15:0]                         i_adapt_version,

    input  logic                                i_fc_clear_state,
    input  logic [15:0]                         i_fc_start_sample_id,
    input  logic                                i_event_valid,
    output logic                                o_event_ready,
    input  logic [7:0]                          i_event_record_valid,
    input  logic [15:0]                         i_event_sample_id,
    input  logic [8:0]                          i_event_timestep,
    input  logic [7:0]                          i_event_boundary_mask,
    input  logic                                i_event_sample_last,
    input  logic [8*12-1:0]                     i_event_addr_flat,
    input  logic [63:0]                         i_event_data_flat,

    output logic                                o_result_valid,
    input  logic                                i_result_ready,
    output logic [5:0]                          o_result_class,
    output logic signed [47:0]                  o_result_score,
    output logic [63:0]                         o_result_versions,
    output logic [1:0]                          o_result_owner,
    input  logic                                i_dynamic_result_valid,
    output logic                                o_dynamic_result_ready,
    input  logic [5:0]                          i_dynamic_result_class,
    input  logic signed [47:0]                  i_dynamic_result_score,
    input  logic [63:0]                         i_dynamic_result_versions,
    input  logic                                i_scnn_li_result_valid,
    output logic                                o_scnn_li_result_ready,
    input  logic [5:0]                          i_scnn_li_result_class,
    input  logic signed [47:0]                  i_scnn_li_result_score,
    input  logic [63:0]                         i_scnn_li_result_versions,
    output logic [10*24-1:0]                    o_fc_logits_flat,
    output logic [31:0]                         o_fc_sample_cycles,
    output logic [31:0]                         o_fc_accepted_windows,
    output logic [31:0]                         o_fc_accepted_events,
    output logic [31:0]                         o_fc_issue_groups,
    output logic [31:0]                         o_fc_boundary_count,
    output logic [31:0]                         o_fc_dropped_events,
    output logic [11:0]                         o_fc_locked_cluster_mask,
    output logic                                o_fc_active_bank,
    output logic [15:0]                         o_fc_active_version,
    input  logic [3:0]                          i_fc_debug_cluster,
    input  logic [6:0]                          i_fc_debug_word_addr,
    output logic [39:0]                         o_fc_debug_active_word,

    output logic                                o_recurrent_start,
    input  logic                                i_recurrent_start_ready,
    input  logic                                i_recurrent_busy,
    input  logic                                i_recurrent_done,
    input  logic                                i_recurrent_error,
    output logic                                o_scnn_start,
    input  logic                                i_scnn_start_ready,
    input  logic                                i_scnn_busy,
    input  logic                                i_scnn_done,
    input  logic                                i_scnn_error,

    input  logic                                i_scnn_mac_valid,
    output logic                                o_scnn_mac_ready,
    input  logic [TILES-1:0]                    i_scnn_tile_mask,
    input  logic [LANES*16-1:0]                 i_scnn_feature_flat,
    input  logic [TILES*LANES*16-1:0]           i_scnn_weight_flat,
    input  logic [LANES-1:0]                    i_scnn_lane_mask,
    input  logic                                i_scnn_first,
    input  logic                                i_scnn_last,
    input  logic [CHANNEL_WIDTH-1:0]            i_scnn_output_channel_base,
    output logic                                o_scnn_result_valid,
    input  logic                                i_scnn_result_ready,
    output logic [TILES-1:0]                    o_scnn_result_mask,
    output logic [TILES*CHANNEL_WIDTH-1:0]      o_scnn_result_channel_flat,
    output logic [TILES*ACC_WIDTH-1:0]          o_scnn_result_accumulator_flat,

    input  logic                                i_layer_pending,
    input  logic                                i_ddr_calib_complete,
    input  logic                                i_ddr_cmd_pending,
    input  logic                                i_ddr_response_pending,
    input  logic                                i_ddr_cdc_busy,
    input  logic [2:0]                          i_ddr_outstanding,
    input  logic                                i_prefetch_busy,
    input  logic                                i_scratchpad_busy,
    input  logic                                i_writeback_busy,
    input  logic                                i_extension_commit_pending,
    input  logic                                i_version_activation_pending,

    output logic                                o_global_idle,
    output logic                                o_busy,
    output logic                                o_sample_done,
    output logic [1:0]                          o_tile_owner,
    output logic [7:0]                          o_active_opcode,
    output logic [63:0]                         o_locked_versions,
    output logic [12:0]                         o_input_dim,
    output logic [9:0]                          o_hidden_or_channels,
    output logic [5:0]                          o_output_dim,
    output logic [4:0]                          o_active_h40_tiles,
    output logic [4:0]                          o_layer_count,
    output logic [7:0]                          o_fixed_timesteps,
    output logic [2:0]                          o_numeric_profile,
    output logic                                o_fc_w2_update_enable,
    output logic                                o_sticky_error,
    output logic [7:0]                          o_error_code
);
    localparam logic [7:0] PROFILE_FC_H160 = 8'h01;
    localparam logic [7:0] PROFILE_FC_H480 = 8'h02;

    logic [7:0] opcode;
    logic core_start_ready;
    logic core_sticky_error;
    logic [7:0] core_error_code;
    logic fc_start_request;
    logic fc_adapter_start_ready;
    logic fc_adapter_busy;
    logic fc_adapter_compute_done;
    logic fc_adapter_idle;
    logic fc_adapter_error;
    logic [7:0] fc_adapter_error_code;
    logic fc_adapter_result_valid;
    logic fc_adapter_result_ready;
    logic [5:0] fc_adapter_result_class;
    logic signed [31:0] fc_adapter_result_score;
    logic fc_window_ready;
    logic fc_commit_pending;
    logic fc_snapshot_busy;
    logic predecode_fc;
    logic [12:0] adapter_input_dim;
    logic [4:0] adapter_active_tiles;
    logic result_pending;
    logic [2:0] result_pending_mask;
    logic result_arbiter_error;
    logic [7:0] result_arbiter_error_code;
    logic fc_clear_state_latched_q;
    logic [15:0] fc_sample_id_latched_q;

    assign opcode = {1'b0, i_profile_code};
    assign predecode_fc = (opcode == PROFILE_FC_H160) ||
                          (opcode == PROFILE_FC_H480);
    assign adapter_input_dim = o_input_dim != 13'd0 ? o_input_dim :
                               (predecode_fc ? 13'd2312 : 13'd1);
    assign adapter_active_tiles = o_active_h40_tiles != 5'd0 ?
                                  o_active_h40_tiles :
                                  (opcode == PROFILE_FC_H160 ? 5'd4 : 5'd12);
    assign o_event_ready = fc_window_ready &&
                           ((o_active_opcode == PROFILE_FC_H160) ||
                            (o_active_opcode == PROFILE_FC_H480));
    assign o_start_ready = core_start_ready && !result_arbiter_error;
    assign o_sticky_error = core_sticky_error || result_arbiter_error;
    assign o_error_code = core_sticky_error ? core_error_code :
                          result_arbiter_error ?
                          (8'h60 | result_arbiter_error_code) : 8'h00;

    unified_fc_h480_bound_adapter #(
        .WEIGHT_DIR(FC_WEIGHT_DIR)
    ) u_fc_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_start(fc_start_request), .o_start_ready(fc_adapter_start_ready),
        .i_clear_state(fc_clear_state_latched_q),
        .i_start_sample_id(fc_sample_id_latched_q),
        .i_active_input_dim(adapter_input_dim),
        .i_active_h40_tiles(adapter_active_tiles),
        .i_window_valid(i_event_valid && predecode_fc),
        .o_window_ready(fc_window_ready),
        .i_record_valid(i_event_record_valid),
        .i_window_sample_id(i_event_sample_id),
        .i_window_timestep(i_event_timestep),
        .i_boundary_mask(i_event_boundary_mask),
        .i_window_sample_last(i_event_sample_last),
        .i_fi_flat(i_event_addr_flat), .i_x_flat(i_event_data_flat),
        .o_timestep_done(), .o_compute_done(fc_adapter_compute_done),
        .o_result_valid(fc_adapter_result_valid),
        .i_result_ready(fc_adapter_result_ready),
        .o_result_class(fc_adapter_result_class),
        .o_result_score(fc_adapter_result_score),
        .o_logits_flat(o_fc_logits_flat),
        .o_sample_cycles(o_fc_sample_cycles),
        .o_accepted_windows(o_fc_accepted_windows),
        .o_accepted_events(o_fc_accepted_events),
        .o_issue_groups(o_fc_issue_groups),
        .o_boundary_count(o_fc_boundary_count),
        .o_runtime_dropped_events(o_fc_dropped_events),
        .o_global_idle(fc_adapter_idle), .o_busy(fc_adapter_busy),
        .o_locked_cluster_mask(o_fc_locked_cluster_mask),
        .o_locked_input_dim(), .o_active_bank(o_fc_active_bank),
        .o_active_version(o_fc_active_version),
        .o_update_window_ready(), .o_commit_pending(fc_commit_pending),
        .o_snapshot_busy(fc_snapshot_busy),
        .i_debug_cluster(i_fc_debug_cluster),
        .i_debug_word_addr(i_fc_debug_word_addr),
        .o_debug_active_word(o_fc_debug_active_word),
        .o_sticky_error(fc_adapter_error),
        .o_error_code(fc_adapter_error_code)
    );

    unified_result_owner_arbiter_v1 #(
        .CLASS_WIDTH(6), .SCORE_WIDTH(48), .VERSION_WIDTH(64)
    ) u_result_owner (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error && !result_pending),
        .i_flush(i_abort),
        .i_expected_owner(o_tile_owner),
        .i_fc_valid(fc_adapter_result_valid),
        .o_fc_ready(fc_adapter_result_ready),
        .i_fc_class(fc_adapter_result_class),
        .i_fc_score({{16{fc_adapter_result_score[31]}},
                     fc_adapter_result_score}),
        .i_fc_versions(o_locked_versions),
        .i_dynamic_valid(i_dynamic_result_valid),
        .o_dynamic_ready(o_dynamic_result_ready),
        .i_dynamic_class(i_dynamic_result_class),
        .i_dynamic_score(i_dynamic_result_score),
        .i_dynamic_versions(i_dynamic_result_versions),
        .i_scnn_valid(i_scnn_li_result_valid),
        .o_scnn_ready(o_scnn_li_result_ready),
        .i_scnn_class(i_scnn_li_result_class),
        .i_scnn_score(i_scnn_li_result_score),
        .i_scnn_versions(i_scnn_li_result_versions),
        .o_result_valid(o_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_class(o_result_class),
        .o_result_score(o_result_score),
        .o_result_versions(o_result_versions),
        .o_result_owner(o_result_owner),
        .o_pending_mask(result_pending_mask),
        .o_pending(result_pending),
        .o_sticky_error(result_arbiter_error),
        .o_error_code(result_arbiter_error_code)
    );

    unified_snn_core #(
        .TILES(TILES), .LANES(LANES),
        .CHANNEL_WIDTH(CHANNEL_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_control_shell (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_start(i_start && !result_arbiter_error),
        .o_start_ready(core_start_ready),
        .i_opcode(opcode), .i_model_version(i_model_version),
        .i_graph_version(i_graph_version),
        .i_adapt_version(i_adapt_version),
        .o_fc_start(fc_start_request),
        .i_fc_start_ready(fc_adapter_start_ready),
        .i_fc_busy(fc_adapter_busy),
        .i_fc_done(fc_adapter_compute_done),
        .i_fc_error(fc_adapter_error),
        .o_recurrent_start(o_recurrent_start),
        .i_recurrent_start_ready(i_recurrent_start_ready),
        .i_recurrent_busy(i_recurrent_busy),
        .i_recurrent_done(i_recurrent_done),
        .i_recurrent_error(i_recurrent_error),
        .o_scnn_start(o_scnn_start),
        .i_scnn_start_ready(i_scnn_start_ready),
        .i_scnn_busy(i_scnn_busy), .i_scnn_done(i_scnn_done),
        .i_scnn_error(i_scnn_error),
        .i_scnn_mac_valid(i_scnn_mac_valid),
        .o_scnn_mac_ready(o_scnn_mac_ready),
        .i_scnn_tile_mask(i_scnn_tile_mask),
        .i_scnn_feature_flat(i_scnn_feature_flat),
        .i_scnn_weight_flat(i_scnn_weight_flat),
        .i_scnn_lane_mask(i_scnn_lane_mask),
        .i_scnn_first(i_scnn_first), .i_scnn_last(i_scnn_last),
        .i_scnn_output_channel_base(i_scnn_output_channel_base),
        .o_scnn_result_valid(o_scnn_result_valid),
        .i_scnn_result_ready(i_scnn_result_ready),
        .o_scnn_result_mask(o_scnn_result_mask),
        .o_scnn_result_channel_flat(o_scnn_result_channel_flat),
        .o_scnn_result_accumulator_flat(o_scnn_result_accumulator_flat),
        .i_output_busy(result_pending),
        .i_layer_pending(i_layer_pending),
        .i_ddr_calib_complete(i_ddr_calib_complete),
        .i_ddr_cmd_pending(i_ddr_cmd_pending),
        .i_ddr_response_pending(i_ddr_response_pending),
        .i_ddr_cdc_busy(i_ddr_cdc_busy),
        .i_ddr_outstanding(i_ddr_outstanding),
        .i_prefetch_busy(i_prefetch_busy),
        .i_scratchpad_busy(i_scratchpad_busy),
        .i_writeback_busy(i_writeback_busy),
        .i_w2_commit_pending(fc_commit_pending ||
                             i_extension_commit_pending),
        .i_w2_snapshot_busy(fc_snapshot_busy),
        .i_version_activation_pending(i_version_activation_pending),
        .o_global_idle(o_global_idle), .o_busy(o_busy),
        .o_sample_done(o_sample_done), .o_tile_owner(o_tile_owner),
        .o_active_opcode(o_active_opcode),
        .o_locked_versions(o_locked_versions),
        .o_input_dim(o_input_dim),
        .o_logical_hidden_or_channels(o_hidden_or_channels),
        .o_output_dim(o_output_dim),
        .o_active_h40_tiles(o_active_h40_tiles),
        .o_layer_count(o_layer_count),
        .o_fixed_timesteps(o_fixed_timesteps),
        .o_numeric_profile(o_numeric_profile),
        .o_fc_w2_update_enable(o_fc_w2_update_enable),
        .o_sticky_error(core_sticky_error), .o_error_code(core_error_code)
    );

    // The orchestration shell emits the engine start after accepting the
    // external request.  Capture all sample-scoped FC metadata at that first
    // handshake so a one-cycle TB/CPU command remains atomic until launch.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            fc_clear_state_latched_q <= 1'b0;
            fc_sample_id_latched_q <= '0;
        end else if (i_start && o_start_ready) begin
            fc_clear_state_latched_q <= i_fc_clear_state;
            fc_sample_id_latched_q <= i_fc_start_sample_id;
        end
    end

    logic unused_fc_status;
    assign unused_fc_status = &{1'b0, fc_adapter_idle,
                                fc_adapter_error_code, result_pending_mask};
endmodule

`default_nettype wire
