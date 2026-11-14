`timescale 1ns/1ps
`default_nettype none

// Complete FC fast path with INPUT_MAX=2312. The canonical control, reduction
// and argmax are reused unchanged; only the zero-padded W0 capacity is removed.
module unified_fc2312_h160_fixed_forward #(
    parameter WEIGHT_DIR = "../../../data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_sample_start,
    input  wire                         i_window_valid,
    output wire                         o_window_ready,
    input  wire [7:0]                   i_record_valid,
    input  wire [15:0]                  i_sample_id,
    input  wire [8:0]                   i_timestep,
    input  wire [7:0]                   i_boundary_mask,
    input  wire                         i_sample_last,
    input  wire [8*12-1:0]              i_fi_flat,
    input  wire [63:0]                  i_x_flat,
    output wire                         o_sample_done,
    output wire [3:0]                   o_pred,
    output wire [10*24-1:0]             o_logits_flat,
    output wire [4*40-1:0]              o_spikes_flat,
    output wire [4*40*18-1:0]           o_vmem_flat,
    output wire [4*40*16-1:0]           o_trace_flat,
    output wire [31:0]                  o_accepted_windows,
    output wire [31:0]                  o_accepted_events,
    output wire [31:0]                  o_issue_beats,
    output wire [31:0]                  o_process_beats,
    output wire [31:0]                  o_boundary_count,
    output wire [31:0]                  o_input_stall_cycles,
    output wire [31:0]                  o_output_stall_cycles,
    output wire [63:0]                  o_sample_cycles,
    output wire [63:0]                  o_boundary_wait_cycles,
    output wire [63:0]                  o_cluster_wait_cycles,
    output wire [63:0]                  o_reduction_wait_cycles,
    output wire                         o_protocol_error
);
    wire cluster_clear;
    wire cluster_step_start;
    wire [8:0] cluster_timestep;
    wire cluster_group_valid;
    wire [15:0] cluster_lane_valid;
    wire [16*12-1:0] cluster_fi_flat;
    wire [127:0] cluster_x_flat;
    wire cluster_groups_done;
    wire [3:0] cluster_step_ready;
    wire [3:0] cluster_group_ready;
    wire [3:0] cluster_done;
    wire [4*10*24-1:0] cluster_partials_flat;
    wire [3:0] cluster_protocol_error;

    newarch_p2c_matched_forward_control #(
        .K16_MODE(0), .CLUSTERS(4)
    ) u_control (
        .clk(clk), .rst_n(rst_n), .i_sample_start(i_sample_start),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid), .i_sample_id(i_sample_id),
        .i_timestep(i_timestep), .i_boundary_mask(i_boundary_mask),
        .i_sample_last(i_sample_last), .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat), .i_cluster_step_ready(cluster_step_ready),
        .i_cluster_group_ready(cluster_group_ready),
        .i_cluster_done(cluster_done),
        .i_cluster_partials_flat(cluster_partials_flat),
        .i_cluster_protocol_error(cluster_protocol_error),
        .o_cluster_clear(cluster_clear),
        .o_cluster_step_start(cluster_step_start),
        .o_cluster_timestep(cluster_timestep),
        .o_cluster_group_valid(cluster_group_valid),
        .o_cluster_lane_valid(cluster_lane_valid),
        .o_cluster_fi_flat(cluster_fi_flat),
        .o_cluster_x_flat(cluster_x_flat),
        .o_cluster_groups_done(cluster_groups_done),
        .o_sample_done(o_sample_done), .o_pred(o_pred),
        .o_logits_flat(o_logits_flat),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_beats(o_issue_beats), .o_process_beats(o_process_beats),
        .o_boundary_count(o_boundary_count),
        .o_input_stall_cycles(o_input_stall_cycles),
        .o_output_stall_cycles(o_output_stall_cycles),
        .o_sample_cycles(o_sample_cycles),
        .o_boundary_wait_cycles(o_boundary_wait_cycles),
        .o_cluster_wait_cycles(o_cluster_wait_cycles),
        .o_reduction_wait_cycles(o_reduction_wait_cycles),
        .o_protocol_error(o_protocol_error)
    );

    unified_fc2312_h160_k8_cluster_array #(
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_clusters (
        .clk(clk), .rst_n(rst_n), .i_clear(cluster_clear),
        .i_step_start(cluster_step_start), .i_timestep(cluster_timestep),
        .i_group_valid(cluster_group_valid),
        .i_lane_valid(cluster_lane_valid), .i_fi_flat(cluster_fi_flat),
        .i_x_flat(cluster_x_flat), .i_groups_done(cluster_groups_done),
        .o_step_ready(cluster_step_ready),
        .o_group_ready(cluster_group_ready), .o_done(cluster_done),
        .o_partials_flat(cluster_partials_flat),
        .o_spikes_flat(o_spikes_flat), .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_protocol_error(cluster_protocol_error)
    );
endmodule

`default_nettype wire
