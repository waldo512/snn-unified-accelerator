`timescale 1ns/1ps
`default_nettype none

// Four canonical H40 dual-bank W2 engines plus the unchanged four-cluster
// registered reduction tree. This plane is a W2_ADAPT sidecar, not INFER muxing.
module unified_h160_versioned_w2_plane #(
    parameter WEIGHT_DIR = "../../../data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_start,
    output wire                         o_ready,
    input  wire [159:0]                 i_spikes,
    input  wire                         i_active_bank,
    input  wire [15:0]                  i_active_version,
    output reg                          o_done,
    output wire [10*24-1:0]             o_logits_flat,
    input  wire [3:0]                   i_cmd_valid,
    output wire [3:0]                   o_cmd_ready,
    input  wire                         i_cmd_target_bank,
    input  wire [6:0]                   i_cmd_local_word_addr,
    input  wire [9:0]                   i_cmd_deltas,
    input  wire [15:0]                  i_cmd_update_version,
    input  wire                         i_cmd_update_last,
    input  wire                         i_snapshot_start,
    output wire [3:0]                   o_snapshot_busy,
    output wire [3:0]                   o_snapshot_done,
    output wire [3:0]                   o_protocol_error,
    input  wire [1:0]                   i_debug_cluster,
    input  wire [6:0]                   i_debug_word_addr,
    output reg  [39:0]                  o_debug_active_word,
    output reg  [39:0]                  o_debug_bank0_word,
    output reg  [39:0]                  o_debug_bank1_word
);
    wire [3:0] cluster_ready;
    wire [3:0] cluster_done;
    wire [4*10*24-1:0] partials_flat;
    wire [4*40-1:0] debug_active_flat;
    wire [4*40-1:0] debug_bank0_flat;
    wire [4*40-1:0] debug_bank1_flat;
    reg [3:0] done_seen_q;
    reg infer_busy_q;
    reg reduction_launch_q;
    wire reduction_valid;
    wire [16:0] reduction_meta;
    wire [16:0] expected_meta = {i_active_bank, i_active_version};
    wire [3:0] done_seen_next = done_seen_q | cluster_done;
    wire launch = i_start && o_ready;

    assign o_ready = !infer_busy_q && (&cluster_ready) &&
        !(|o_snapshot_busy) && !(|o_protocol_error);

`define UNIFIED_H160_W2_CLUSTER(INDEX) \
    newarch_w2_sparse_fold4_lanebanked_cluster_h40 #( \
        .W2_PACKED_FILE({WEIGHT_DIR, "/w2_cluster_0", `"INDEX`", \
                         "_packed_dual_bank.hex"}) \
    ) u_cluster_``INDEX ( \
        .clk(clk), .rst_n(rst_n), .i_start(launch), \
        .i_spikes(i_spikes[INDEX*40 +: 40]), \
        .i_active_bank(i_active_bank), \
        .i_active_version(i_active_version), \
        .o_infer_ready(cluster_ready[INDEX]), .o_done(cluster_done[INDEX]), \
        .o_active_block_mask(), .o_active_block_count(), \
        .o_partial_flat(partials_flat[INDEX*10*24 +: 10*24]), \
        .i_cmd_valid(i_cmd_valid[INDEX]), \
        .o_cmd_ready(o_cmd_ready[INDEX]), \
        .i_cmd_target_bank(i_cmd_target_bank), \
        .i_cmd_local_word_addr(i_cmd_local_word_addr), \
        .i_cmd_deltas(i_cmd_deltas), \
        .i_cmd_update_version(i_cmd_update_version), \
        .i_cmd_update_last(i_cmd_update_last), \
        .i_snapshot_start(i_snapshot_start), \
        .i_snapshot_src_bank(i_active_bank), \
        .i_snapshot_dst_bank(~i_active_bank), \
        .o_snapshot_busy(o_snapshot_busy[INDEX]), \
        .o_snapshot_done(o_snapshot_done[INDEX]), \
        .o_snapshot_cycles(), .o_protocol_error(o_protocol_error[INDEX]), \
        .i_debug_word_addr(i_debug_word_addr), \
        .o_debug_active_word(debug_active_flat[INDEX*40 +: 40]), \
        .o_debug_bank0_word(debug_bank0_flat[INDEX*40 +: 40]), \
        .o_debug_bank1_word(debug_bank1_flat[INDEX*40 +: 40]) \
    );

    `UNIFIED_H160_W2_CLUSTER(0)
    `UNIFIED_H160_W2_CLUSTER(1)
    `UNIFIED_H160_W2_CLUSTER(2)
    `UNIFIED_H160_W2_CLUSTER(3)
`undef UNIFIED_H160_W2_CLUSTER

    newarch_p2c_w2_reduction_tree_pow2 #(
        .CLUSTERS(4), .O_DIM(10), .ACC_BITS(24), .META_BITS(17)
    ) u_reduction (
        .clk(clk), .rst_n(rst_n), .i_valid(reduction_launch_q),
        .i_partials_flat(partials_flat), .i_meta(expected_meta),
        .o_valid(reduction_valid), .o_reduced_flat(o_logits_flat),
        .o_meta(reduction_meta)
    );

    always_comb begin
        o_debug_active_word =
            debug_active_flat[i_debug_cluster*40 +: 40];
        o_debug_bank0_word = debug_bank0_flat[i_debug_cluster*40 +: 40];
        o_debug_bank1_word = debug_bank1_flat[i_debug_cluster*40 +: 40];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_seen_q <= 4'b0000;
            infer_busy_q <= 1'b0;
            reduction_launch_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            reduction_launch_q <= 1'b0;
            o_done <= 1'b0;
            if (launch) begin
                done_seen_q <= 4'b0000;
                infer_busy_q <= 1'b1;
            end else if (infer_busy_q && (&done_seen_next)) begin
                done_seen_q <= 4'b0000;
                reduction_launch_q <= 1'b1;
            end else if (infer_busy_q) begin
                done_seen_q <= done_seen_next;
            end
            if (reduction_valid) begin
                infer_busy_q <= 1'b0;
                if (reduction_meta == expected_meta)
                    o_done <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
