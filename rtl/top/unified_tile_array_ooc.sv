`timescale 1ns/1ps
`default_nettype none

module unified_tile_array_ooc #(
    parameter integer TILE_COUNT = 1,
    parameter integer LANES = 8,
    parameter integer NEURONS_PER_TILE = 40
) (
    input  wire logic                                    clk,
    input  wire logic                                    rst_n,
    input  wire logic                                    i_clear_sample,
    input  wire logic                                    i_clear_accum,
    input  wire logic [TILE_COUNT-1:0]                   i_mac_valid,
    output logic [TILE_COUNT-1:0]                        o_mac_ready,
    input  wire logic [TILE_COUNT*LANES-1:0]             i_lane_valid_flat,
    input  wire logic [TILE_COUNT*LANES*6-1:0]           i_neuron_index_flat,
    input  wire logic [TILE_COUNT*LANES*8-1:0]           i_activation_flat,
    input  wire logic [TILE_COUNT*LANES*8-1:0]           i_weight_flat,
    input  wire logic                                    i_acc24_compat,
    input  wire logic                                    i_lif_start,
    input  wire logic signed [15:0]                      i_leak_q15,
    input  wire logic signed [31:0]                      i_threshold,
    input  wire logic [4:0]                              i_current_left_shift,
    input  wire logic                                    i_reset_to_zero,
    output logic [TILE_COUNT-1:0]                        o_lif_busy,
    output logic [TILE_COUNT-1:0]                        o_lif_done,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]       o_previous_spikes_flat,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]       o_current_spikes_flat,
    output logic [TILE_COUNT*32-1:0]                     o_debug_accum_flat,
    output logic [TILE_COUNT*32-1:0]                     o_debug_membrane_flat,
    output logic [TILE_COUNT-1:0]                        o_collision_error
);
    genvar tile;
    generate
        for (tile = 0; tile < TILE_COUNT; tile = tile + 1) begin : g_tile
            logic lif_spike_valid_unused;
            logic [5:0] lif_base_unused;
            logic [LANES-1:0] lif_spike_mask_unused;
            unified_h40_k8_tile #(
                .NEURONS(NEURONS_PER_TILE), .LANES(LANES)
            ) u_tile (
                .clk(clk), .rst_n(rst_n), .i_clear_sample(i_clear_sample),
                .i_clear_accum(i_clear_accum), .i_mac_valid(i_mac_valid[tile]),
                .o_mac_ready(o_mac_ready[tile]),
                .i_lane_valid(i_lane_valid_flat[tile*LANES +: LANES]),
                .i_neuron_index_flat(i_neuron_index_flat[tile*LANES*6 +: LANES*6]),
                .i_activation_flat(i_activation_flat[tile*LANES*8 +: LANES*8]),
                .i_weight_flat(i_weight_flat[tile*LANES*8 +: LANES*8]),
                .i_acc24_compat(i_acc24_compat), .i_lif_start(i_lif_start),
                .i_leak_q15(i_leak_q15), .i_threshold(i_threshold),
                .i_current_left_shift(i_current_left_shift),
                .i_reset_to_zero(i_reset_to_zero), .o_lif_busy(o_lif_busy[tile]),
                .o_lif_done(o_lif_done[tile]),
                .o_lif_spike_valid(lif_spike_valid_unused),
                .o_lif_base(lif_base_unused), .o_lif_spike_mask(lif_spike_mask_unused),
                .i_timestep_commit(1'b0),
                .o_previous_spikes(o_previous_spikes_flat[tile*NEURONS_PER_TILE +: NEURONS_PER_TILE]),
                .o_current_spikes(o_current_spikes_flat[tile*NEURONS_PER_TILE +: NEURONS_PER_TILE]),
                .i_debug_neuron(6'd0),
                .o_debug_accum(o_debug_accum_flat[tile*32 +: 32]),
                .o_debug_membrane(o_debug_membrane_flat[tile*32 +: 32]),
                .o_lane_collision_error(o_collision_error[tile])
            );
        end
    endgenerate
endmodule

`default_nettype wire
