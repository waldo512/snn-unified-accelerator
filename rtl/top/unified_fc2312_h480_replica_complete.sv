`timescale 1ns/1ps
`default_nettype none

// FC-first complete H480 wrapper. It freezes the R19B-derived 12xH40 online
// stream at INPUT_MAX=2312 without inserting SRNN/SCNN control into FC compute.
module unified_fc2312_h480_replica_complete #(
    parameter WEIGHT_DIR =
        "../../../data/weights/r19b_h480_replica_fc_input2312"
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_sample_start,
    output wire                         o_sample_ready,
    input  wire                         i_clear_state,
    input  wire [15:0]                  i_start_sample_id,
    input  wire [12:0]                  i_active_input_dim,
    input  wire [11:0]                  i_active_cluster_mask,
    input  wire                         i_window_valid,
    output wire                         o_window_ready,
    input  wire [7:0]                   i_record_valid,
    input  wire [15:0]                  i_window_sample_id,
    input  wire [8:0]                   i_window_timestep,
    input  wire [7:0]                   i_boundary_mask,
    input  wire                         i_window_sample_last,
    input  wire [8*12-1:0]              i_fi_flat,
    input  wire [63:0]                  i_x_flat,
    output wire                         o_timestep_done,
    output wire                         o_sample_done,
    output wire [8:0]                   o_completed_timestep,
    output wire [10*24-1:0]             o_logits_flat,
    output wire [3:0]                   o_pred,
    output wire [31:0]                  o_sample_cycles,
    output wire [31:0]                  o_accepted_windows,
    output wire [31:0]                  o_accepted_events,
    output wire [31:0]                  o_issue_groups,
    output wire [31:0]                  o_boundary_count,
    output wire [31:0]                  o_runtime_dropped_events,
    output wire                         o_global_idle,
    output wire                         o_active_bank,
    output wire [15:0]                  o_active_version,
    output wire                         o_update_window_ready,
    output wire                         o_commit_pending,
    output wire                         o_snapshot_busy,
    input  wire [3:0]                   i_debug_cluster,
    input  wire [6:0]                   i_debug_word_addr,
    output wire [39:0]                  o_debug_active_word,
    output wire                         o_protocol_error
);
    wire inactive_bank;
    wire commit_pulse;
    wire [4:0] record_count;
    wire [31:0] dispatched_records;
    wire cadence_blocked;
    wire [31:0] update_command_count;
    wire [15:0] commit_count;
    wire [39:0] debug_bank0_word;
    wire [39:0] debug_bank1_word;
    wire [31:0] ingress_stall_cycles;
    wire [31:0] output_stall_cycles;
    wire [31:0] inactive_update_command_count;

    unified_fc2312_h480_replica_online_stream_k8_b8 #(
        .IN_DIM(2312), .CLUSTERS(12), .H_PER_CLUSTER(40),
        .LIF_LANES(20), .ACC_BITS(18), .OUT_ACC_BITS(24),
        .W0_BITS(5), .W0_INPUT_SHIFT(6), .BANK_DEPTH(64),
        .ACCUM_SAMPLES(32), .W0_WEIGHT_DIR(WEIGHT_DIR),
        .W2_WEIGHT_DIR(WEIGHT_DIR), .W2_PACKED_WEIGHT_DIR(WEIGHT_DIR),
        .USE_PACKED_SDP_W2(1), .USE_SPARSE_PACKED_SDP_W2(0),
        .USE_SPARSE_FAST_JOIN(0), .USE_R18_DUAL_CONTEXT_W2(0),
        .USE_N8_FOLDED_OVERLAP(1), .USE_N9E2_START_CUT_THROUGH(0),
        .USE_ATOMIC_BOUNDARY_CLOSE(1),
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(0),
        .USE_LIF_ACC_SNAPSHOT(0), .ENABLE_RUNTIME_PROFILE(1),
        .USE_N9R19_DEPTH_SNAPSHOT_INGRESS(1),
        .USE_N9R19B_PREFIX_BALANCE4(1),
        .N9R17_RESERVATION_MODE(2),
        .N9R10_REPLICA_W0_WEIGHT_DIR(WEIGHT_DIR),
        .DEBUG_CLUSTER_BITS(4)
    ) u_fc (
        .clk(clk), .clk_fast(clk), .rst_n(rst_n),
        .i_learning_enable(1'b0),
        .i_sample_start(i_sample_start), .o_sample_ready(o_sample_ready),
        .i_clear_state(i_clear_state),
        .i_start_sample_id(i_start_sample_id),
        .i_runtime_active_input_dim(i_active_input_dim),
        .i_runtime_active_cluster_mask(i_active_cluster_mask),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid),
        .i_window_sample_id(i_window_sample_id),
        .i_window_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_window_sample_last(i_window_sample_last),
        .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
        .i_label_valid(1'b0), .o_label_ready(),
        .i_label_sample_id(16'd0), .i_label(4'd0),
        .o_timestep_done(o_timestep_done), .o_sample_done(o_sample_done),
        .o_completed_timestep(o_completed_timestep),
        .o_logits_flat(o_logits_flat), .o_pred(o_pred),
        .o_sample_cycles(o_sample_cycles),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups), .o_boundary_count(o_boundary_count),
        .o_ingress_stall_cycles(ingress_stall_cycles),
        .o_output_stall_cycles(output_stall_cycles),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_online_global_idle(o_global_idle),
        .o_runtime_inactive_update_command_count(
            inactive_update_command_count),
        .o_active_bank(o_active_bank), .o_active_version(o_active_version),
        .o_inactive_bank(inactive_bank),
        .o_update_window_ready(o_update_window_ready),
        .o_commit_pending(o_commit_pending), .o_commit_pulse(commit_pulse),
        .o_snapshot_busy(o_snapshot_busy), .o_record_count(record_count),
        .o_dispatched_records(dispatched_records),
        .o_cadence_blocked(cadence_blocked),
        .o_update_command_count(update_command_count),
        .o_commit_count(commit_count),
        .i_debug_cluster(i_debug_cluster),
        .i_debug_word_addr(i_debug_word_addr),
        .o_debug_active_word(o_debug_active_word),
        .o_debug_bank0_word(debug_bank0_word),
        .o_debug_bank1_word(debug_bank1_word),
        .o_protocol_error(o_protocol_error)
    );

    wire unused_status = &{1'b0, inactive_bank, commit_pulse, record_count,
        dispatched_records, cadence_blocked, update_command_count, commit_count,
        debug_bank0_word, debug_bank1_word, ingress_stall_cycles,
        output_stall_cycles, inactive_update_command_count};
endmodule

`default_nettype wire
