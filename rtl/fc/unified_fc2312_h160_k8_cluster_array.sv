`timescale 1ns/1ps
`default_nettype none

// Trained H160 cluster array with the unused 4096-to-2312 W0 tail removed.
// Arithmetic, overlap and W2 files remain identical to the P3A FC baseline.
module unified_fc2312_h160_k8_cluster_array #(
    parameter integer T_BITS = 9,
    parameter integer FI_BITS = 12,
    parameter integer LIF_TRACE_FRAC_BITS = 10,
    parameter WEIGHT_DIR = "../../../data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          i_clear,
    input  wire                          i_step_start,
    input  wire [T_BITS-1:0]             i_timestep,
    input  wire                          i_group_valid,
    input  wire [15:0]                   i_lane_valid,
    input  wire [16*FI_BITS-1:0]         i_fi_flat,
    input  wire [127:0]                  i_x_flat,
    input  wire                          i_groups_done,
    output wire [3:0]                    o_step_ready,
    output wire [3:0]                    o_group_ready,
    output wire [3:0]                    o_done,
    output wire [4*10*24-1:0]            o_partials_flat,
    output wire [4*40-1:0]               o_spikes_flat,
    output wire [4*40*18-1:0]            o_vmem_flat,
    output wire [4*40*16-1:0]            o_trace_flat,
    output wire [3:0]                    o_protocol_error
);
    localparam integer INPUT_MAX = 2312;
    wire [7:0] bank_valid;
    wire [8*FI_BITS-1:0] bank_fi_flat;
    wire [63:0] bank_x_flat;

    genvar bank;
    generate
        for (bank = 0; bank < 8; bank = bank + 1) begin : gen_lane_map
            assign bank_valid[bank] = i_lane_valid[2*bank];
            assign bank_fi_flat[bank*FI_BITS +: FI_BITS] =
                i_fi_flat[(2*bank)*FI_BITS +: FI_BITS];
            assign bank_x_flat[bank*8 +: 8] = i_x_flat[(2*bank)*8 +: 8];
        end
    endgenerate

`define UNIFIED_FC2312_CLUSTER(INDEX) \
    newarch_hidden_cluster_k8_b8_folded_overlap #( \
        .IN_DIM(INPUT_MAX), .H_COUNT(40), .LIF_LANES(20), \
        .FI_BITS(FI_BITS), .ACC_BITS(18), .OUT_ACC_BITS(24), \
        .W0_BITS(5), .W0_INPUT_SHIFT(6), .T_BITS(T_BITS), \
        .LIF_TRACE_FRAC_BITS(LIF_TRACE_FRAC_BITS), \
        .USE_ATOMIC_BOUNDARY_CLOSE(1), \
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(0), \
        .USE_LIF_ACC_SNAPSHOT(0), .ENABLE_FIXED_W2(1), \
        .BANK_FILE_0({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_00.mem"}), \
        .BANK_FILE_1({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_01.mem"}), \
        .BANK_FILE_2({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_02.mem"}), \
        .BANK_FILE_3({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_03.mem"}), \
        .BANK_FILE_4({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_04.mem"}), \
        .BANK_FILE_5({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_05.mem"}), \
        .BANK_FILE_6({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_06.mem"}), \
        .BANK_FILE_7({WEIGHT_DIR, "/w0_cluster_0", `"INDEX`", "_bank_07.mem"}), \
        .W2_FILE({WEIGHT_DIR, "/w2_cluster_0", `"INDEX`", ".hex"}) \
    ) u_cluster_``INDEX ( \
        .clk(clk), .rst_n(rst_n), .i_clear_state(i_clear), \
        .i_step_start(i_step_start), .o_step_ready(o_step_ready[INDEX]), \
        .i_result_ready(1'b1), .o_result_start(), .i_timestep(i_timestep), \
        .i_group_valid(i_group_valid), .o_group_ready(o_group_ready[INDEX]), \
        .i_bank_valid(bank_valid), .i_fi_flat(bank_fi_flat), \
        .i_x_flat(bank_x_flat), .i_groups_done(i_groups_done), \
        .o_done(o_done[INDEX]), .o_completed_timestep(), \
        .o_completed_events(), \
        .o_completed_spikes(o_spikes_flat[INDEX*40 +: 40]), \
        .o_completed_spike_count(), \
        .o_partial_logits_flat(o_partials_flat[INDEX*10*24 +: 10*24]), \
        .o_vmem_flat(o_vmem_flat[INDEX*40*18 +: 40*18]), \
        .o_trace_flat(o_trace_flat[INDEX*40*16 +: 40*16]), \
        .o_w0_lif_overlap_cycles(), .o_w0_producer_active(), \
        .o_w0_consumer_active(), .o_protocol_error(o_protocol_error[INDEX]) \
    );

    `UNIFIED_FC2312_CLUSTER(0)
    `UNIFIED_FC2312_CLUSTER(1)
    `UNIFIED_FC2312_CLUSTER(2)
    `UNIFIED_FC2312_CLUSTER(3)
`undef UNIFIED_FC2312_CLUSTER
endmodule

`default_nettype wire
