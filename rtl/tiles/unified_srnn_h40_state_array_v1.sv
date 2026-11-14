`timescale 1ns/1ps
`default_nettype none

module unified_srnn_h40_state_array_v1 #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer TILE_NEURONS = 40,
    parameter integer LANES = 8
) (
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 i_abort,
    input  logic                                 i_clear_spikes,
    input  logic                                 i_clear_word_valid,
    input  logic [9:0]                           i_clear_word_base,
    input  logic                                 i_accumulate_valid,
    input  logic [9:0]                           i_accumulate_base,
    input  logic [LANES-1:0]                     i_accumulate_lane_mask,
    input  logic [LANES*32-1:0]                  i_accumulate_contribution_flat,
    input  logic [9:0]                           i_active_neurons,
    input  logic                                 i_lif_begin,
    input  logic                                 i_lif_word_valid,
    input  logic [9:0]                           i_lif_word_base,
    input  logic signed [15:0]                  i_leak_q15,
    input  logic signed [31:0]                  i_threshold,
    input  logic [4:0]                           i_current_left_shift,
    input  logic                                 i_reset_to_zero,
    input  logic                                 i_commit_timestep,
    output logic [HIDDEN_MAX-1:0]                o_previous_spikes,
    output logic [HIDDEN_MAX-1:0]                o_current_spikes,
    output logic [HIDDEN_MAX*32-1:0]             o_accumulator_flat,
    output logic [HIDDEN_MAX*32-1:0]             o_membrane_flat
);
    localparam integer TILES =
        (HIDDEN_MAX + TILE_NEURONS - 1) / TILE_NEURONS;
    localparam integer PHYSICAL_NEURONS = TILES * TILE_NEURONS;

    // The final physical H40 shard may contain padding neurons beyond the
    // logical HIDDEN_MAX. They stay locally masked and are intentionally not
    // exposed by this logical-width array interface.
    /* verilator lint_off UNUSEDSIGNAL */
    logic [PHYSICAL_NEURONS-1:0] previous_spikes_physical;
    logic [PHYSICAL_NEURONS-1:0] current_spikes_physical;
    logic [PHYSICAL_NEURONS*32-1:0] accumulator_physical;
    logic [PHYSICAL_NEURONS*32-1:0] membrane_physical;
    /* verilator lint_on UNUSEDSIGNAL */

    assign o_previous_spikes = previous_spikes_physical[HIDDEN_MAX-1:0];
    assign o_current_spikes = current_spikes_physical[HIDDEN_MAX-1:0];
    assign o_accumulator_flat = accumulator_physical[HIDDEN_MAX*32-1:0];
    assign o_membrane_flat = membrane_physical[HIDDEN_MAX*32-1:0];

    genvar tile;
    generate
        for (tile = 0; tile < TILES; tile++) begin : g_tile
            unified_srnn_h40_state_shard_v1 #(
                .TILE_BASE(tile*TILE_NEURONS),
                .TILE_NEURONS(TILE_NEURONS),
                .LANES(LANES)
            ) u_shard (
                .clk(clk),
                .rst_n(rst_n),
                .i_abort(i_abort),
                .i_clear_spikes(i_clear_spikes),
                .i_clear_word_valid(i_clear_word_valid),
                .i_clear_word_base(i_clear_word_base),
                .i_accumulate_valid(i_accumulate_valid),
                .i_accumulate_base(i_accumulate_base),
                .i_accumulate_lane_mask(i_accumulate_lane_mask),
                .i_accumulate_contribution_flat(
                    i_accumulate_contribution_flat),
                .i_active_neurons(i_active_neurons),
                .i_lif_begin(i_lif_begin),
                .i_lif_word_valid(i_lif_word_valid),
                .i_lif_word_base(i_lif_word_base),
                .i_leak_q15(i_leak_q15),
                .i_threshold(i_threshold),
                .i_current_left_shift(i_current_left_shift),
                .i_reset_to_zero(i_reset_to_zero),
                .i_commit_timestep(i_commit_timestep),
                .o_previous_spikes(previous_spikes_physical[
                    tile*TILE_NEURONS +: TILE_NEURONS]),
                .o_current_spikes(current_spikes_physical[
                    tile*TILE_NEURONS +: TILE_NEURONS]),
                .o_accumulator_flat(accumulator_physical[
                    tile*TILE_NEURONS*32 +: TILE_NEURONS*32]),
                .o_membrane_flat(membrane_physical[
                    tile*TILE_NEURONS*32 +: TILE_NEURONS*32])
            );
        end
    endgenerate
endmodule

`default_nettype wire
