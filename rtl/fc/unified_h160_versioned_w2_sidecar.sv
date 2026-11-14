`timescale 1ns/1ps
`default_nettype none

// H160 VERSIONED_UPDATE_V1 closure. Normal FC INFER remains on the fixed-W2
// canonical path; this sidecar owns W2_ADAPT bank/version atomicity.
module unified_h160_versioned_w2_sidecar #(
    parameter WEIGHT_DIR = "../../../data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  wire logic         clk,
    input  wire logic         rst_n,
    input  wire logic [3:0]   i_active_run_mode,
    input  wire logic [1:0]   i_active_profile,
    input  wire logic         i_global_idle,
    input  wire logic         i_layer_boundary,

    input  wire logic         i_infer_start,
    output logic              o_infer_ready,
    input  wire logic [159:0] i_spikes,
    output logic              o_infer_done,
    output logic [10*24-1:0]  o_logits_flat,

    input  wire logic         i_update_valid,
    output logic              o_update_ready,
    input  wire logic [1:0]   i_update_mode,
    input  wire logic [15:0]  i_update_sample_id,
    input  wire logic [7:0]   i_update_layer_id,
    input  wire logic [7:0]   i_update_tensor_id,
    input  wire logic [15:0]  i_update_source_version,
    input  wire logic [15:0]  i_update_target_version,
    input  wire logic [31:0]  i_update_word_addr,
    input  wire logic [31:0]  i_update_lane_mask,
    input  wire logic signed [7:0] i_update_signed_delta,
    input  wire logic         i_update_commit_last,

    output logic [15:0]       o_weight_version,
    output logic              o_active_bank,
    output logic              o_version_commit_pulse,
    output logic              o_boundary_quiescent,
    output logic              o_snapshot_busy,
    output logic              o_update_error,
    output logic [4:0]        o_update_error_code,

    input  wire logic [1:0]   i_debug_cluster,
    input  wire logic [6:0]   i_debug_word_addr,
    output logic [39:0]       o_debug_active_word,
    output logic [39:0]       o_debug_bank0_word,
    output logic [39:0]       o_debug_bank1_word
);
    localparam logic [4:0] UPDATE_ERROR_SHAPE = 5'd7;
    localparam logic [4:0] UPDATE_ERROR_DELTA = 5'd8;
    localparam logic [4:0] UPDATE_ERROR_UNSUPPORTED = 5'd9;

    logic guard_update_valid;
    logic guard_update_ready;
    logic guarded_valid;
    logic guarded_ready;
    logic [31:0] guarded_word_addr;
    logic [31:0] guarded_lane_mask;
    logic signed [7:0] guarded_signed_delta;
    logic [15:0] guarded_target_version;
    logic guarded_commit_last;
    logic [15:0] guard_weight_version;
    logic guard_commit_pending;
    logic guard_commit_pulse;
    logic guard_quiescent;
    logic guard_error;
    logic [4:0] guard_error_code;

    logic local_error_q;
    logic [4:0] local_error_code_q;
    logic update_supported;
    logic update_shape_legal;
    logic update_delta_legal;
    logic update_local_legal;
    logic update_accept;

    logic [3:0] cmd_valid;
    logic [3:0] cmd_ready;
    logic cmd_target_bank;
    logic [6:0] cmd_local_word_addr;
    logic [9:0] cmd_deltas;
    logic [15:0] cmd_update_version;
    logic cmd_update_last;
    logic command_complete;
    logic adapter_error;

    logic plane_ready;
    logic plane_start;
    logic [3:0] snapshot_busy;
    logic [3:0] snapshot_done;
    logic [3:0] plane_error;
    logic barrier_inactive_bank;
    logic barrier_locked_bank;
    logic [15:0] barrier_locked_version;
    logic barrier_update_ready;
    logic barrier_commit_pending;
    logic barrier_commit_pulse;
    logic barrier_snapshot_start;
    logic barrier_snapshot_src_bank;
    logic barrier_snapshot_dst_bank;
    logic barrier_snapshot_busy;
    logic barrier_error;
    logic [3:0] barrier_update_done_mask;
    logic [3:0] barrier_snapshot_done_mask;
    logic [3:0] shard_update_done;
    logic [3:0] shard_target_bank;
    logic [4*16-1:0] shard_version_flat;

    always_comb begin
        update_supported = (i_update_mode == 2'd1) &&
            (i_update_tensor_id == 8'd2);
        update_shape_legal = (i_update_word_addr < 32'd160) &&
            (i_update_lane_mask[31:10] == 22'd0) &&
            (i_update_lane_mask[9:0] != 10'd0);
        update_delta_legal = (i_update_signed_delta == 8'sd1) ||
            (i_update_signed_delta == -8'sd1);
        update_local_legal = update_supported && update_shape_legal &&
            update_delta_legal;

        o_update_error = local_error_q || guard_error || adapter_error ||
            barrier_error || (|plane_error);
        if (local_error_q)
            o_update_error_code = local_error_code_q;
        else
            o_update_error_code = guard_error_code;

        o_update_ready = guard_update_ready && barrier_update_ready &&
            !o_update_error && !i_infer_start;
        update_accept = i_update_valid && o_update_ready;
        guard_update_valid = update_accept && update_local_legal;

        o_infer_ready = plane_ready && barrier_update_ready &&
            guard_quiescent && !guard_commit_pending && !i_update_valid &&
            !o_update_error;
        plane_start = i_infer_start && o_infer_ready;
        o_version_commit_pulse = barrier_commit_pulse;
        o_boundary_quiescent = guard_quiescent && barrier_update_ready;
        o_snapshot_busy = barrier_snapshot_busy;
        shard_update_done = {4{command_complete && guarded_commit_last}};
        shard_target_bank = {4{barrier_inactive_bank}};
        shard_version_flat = {4{guarded_target_version}};
    end

    newarch_r20b_versioned_update_guard u_guard (
        .clk(clk), .rst_n(rst_n),
        .i_active_run_mode(i_active_run_mode),
        .i_active_profile(i_active_profile),
        .i_global_idle(i_global_idle),
        .i_layer_boundary(i_layer_boundary),
        .i_update_valid(guard_update_valid),
        .o_update_ready(guard_update_ready),
        .i_update_mode(i_update_mode),
        .i_update_sample_id(i_update_sample_id),
        .i_update_layer_id(i_update_layer_id),
        .i_update_tensor_id(i_update_tensor_id),
        .i_update_source_version(i_update_source_version),
        .i_update_target_version(i_update_target_version),
        .i_update_word_addr(i_update_word_addr),
        .i_update_lane_mask(i_update_lane_mask),
        .i_update_signed_delta(i_update_signed_delta),
        .i_update_commit_last(i_update_commit_last),
        .o_guarded_valid(guarded_valid),
        .i_guarded_ready(guarded_ready),
        .o_guarded_mode(), .o_guarded_sample_id(),
        .o_guarded_layer_id(), .o_guarded_tensor_id(),
        .o_guarded_source_version(),
        .o_guarded_target_version(guarded_target_version),
        .o_guarded_word_addr(guarded_word_addr),
        .o_guarded_lane_mask(guarded_lane_mask),
        .o_guarded_signed_delta(guarded_signed_delta),
        .o_guarded_commit_last(guarded_commit_last),
        .o_weight_version(guard_weight_version),
        .o_version_commit_pending(guard_commit_pending),
        .o_version_commit_pulse(guard_commit_pulse),
        .o_boundary_quiescent(guard_quiescent),
        .o_update_error(guard_error),
        .o_update_error_code(guard_error_code)
    );

    unified_h160_w2_command_adapter u_adapter (
        .clk(clk), .rst_n(rst_n),
        .i_guarded_valid(guarded_valid),
        .o_guarded_ready(guarded_ready),
        .i_word_addr(guarded_word_addr),
        .i_lane_mask(guarded_lane_mask[9:0]),
        .i_signed_delta(guarded_signed_delta),
        .i_target_version(guarded_target_version),
        .i_commit_last(guarded_commit_last),
        .i_active_bank(o_active_bank),
        .o_cmd_valid(cmd_valid), .i_cmd_ready(cmd_ready),
        .o_command_complete(command_complete),
        .o_cmd_target_bank(cmd_target_bank),
        .o_cmd_local_word_addr(cmd_local_word_addr),
        .o_cmd_deltas(cmd_deltas),
        .o_cmd_update_version(cmd_update_version),
        .o_cmd_update_last(cmd_update_last),
        .o_protocol_error(adapter_error)
    );

    unified_h160_versioned_w2_plane #(
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_plane (
        .clk(clk), .rst_n(rst_n), .i_start(plane_start),
        .o_ready(plane_ready), .i_spikes(i_spikes),
        .i_active_bank(o_active_bank),
        .i_active_version(o_weight_version),
        .o_done(o_infer_done), .o_logits_flat(o_logits_flat),
        .i_cmd_valid(cmd_valid), .o_cmd_ready(cmd_ready),
        .i_cmd_target_bank(cmd_target_bank),
        .i_cmd_local_word_addr(cmd_local_word_addr),
        .i_cmd_deltas(cmd_deltas),
        .i_cmd_update_version(cmd_update_version),
        .i_cmd_update_last(cmd_update_last),
        .i_snapshot_start(barrier_snapshot_start),
        .o_snapshot_busy(snapshot_busy),
        .o_snapshot_done(snapshot_done),
        .o_protocol_error(plane_error),
        .i_debug_cluster(i_debug_cluster),
        .i_debug_word_addr(i_debug_word_addr),
        .o_debug_active_word(o_debug_active_word),
        .o_debug_bank0_word(o_debug_bank0_word),
        .o_debug_bank1_word(o_debug_bank1_word)
    );

    newarch_w2_version_barrier #(
        .CLUSTERS(4), .VERSION_BITS(16)
    ) u_version_barrier (
        .clk(clk), .rst_n(rst_n), .i_sample_start(plane_start),
        .i_sample_boundary(guard_commit_pulse),
        .i_inference_idle(plane_ready),
        .i_shard_update_done(shard_update_done),
        .i_shard_update_target_bank(shard_target_bank),
        .i_shard_update_version_flat(shard_version_flat),
        .i_shard_snapshot_done(snapshot_done),
        .i_shard_error(plane_error | {4{adapter_error}}),
        .o_active_bank(o_active_bank),
        .o_active_version(o_weight_version),
        .o_inactive_bank(barrier_inactive_bank),
        .o_locked_bank(barrier_locked_bank),
        .o_locked_version(barrier_locked_version),
        .o_update_window_ready(barrier_update_ready),
        .o_commit_pending(barrier_commit_pending),
        .o_commit_pulse(barrier_commit_pulse),
        .o_snapshot_start(barrier_snapshot_start),
        .o_snapshot_src_bank(barrier_snapshot_src_bank),
        .o_snapshot_dst_bank(barrier_snapshot_dst_bank),
        .o_snapshot_busy(barrier_snapshot_busy),
        .o_protocol_error(barrier_error),
        .o_update_done_mask(barrier_update_done_mask),
        .o_snapshot_done_mask(barrier_snapshot_done_mask)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_error_q <= 1'b0;
            local_error_code_q <= 5'd0;
        end else begin
            if (update_accept && !update_local_legal && !local_error_q) begin
                local_error_q <= 1'b1;
                if (!update_supported)
                    local_error_code_q <= UPDATE_ERROR_UNSUPPORTED;
                else if (!update_shape_legal)
                    local_error_code_q <= UPDATE_ERROR_SHAPE;
                else
                    local_error_code_q <= UPDATE_ERROR_DELTA;
            end
        end
    end

    wire unused_barrier_debug = &{1'b0, snapshot_busy,
        barrier_locked_bank, barrier_locked_version,
        barrier_commit_pending, barrier_snapshot_src_bank,
        barrier_snapshot_dst_bank, barrier_update_done_mask,
        barrier_snapshot_done_mask, guard_weight_version};
endmodule

`default_nettype wire
