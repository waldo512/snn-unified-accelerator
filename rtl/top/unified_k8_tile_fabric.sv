`timescale 1ns/1ps
`default_nettype none

// Shared sparse-FC/recurrent-FC fabric. One source row is broadcast across
// independent 40-neuron weight banks; the tiles execute concurrently.
module unified_k8_tile_fabric #(
    parameter integer TILE_COUNT = 12,
    parameter integer LANES = 8,
    parameter integer NEURONS_PER_TILE = 40
) (
    input  wire logic                                  clk,
    input  wire logic                                  rst_n,
    input  wire logic                                  i_clear_sample,
    input  wire logic                                  i_clear_accum,
    input  wire logic [3:0]                            i_active_tiles,

    input  wire logic                                  i_row_valid,
    output logic                                       o_row_ready,
    input  wire logic [31:0]                           i_weight_row_base,
    input  wire logic [9:0]                            i_destination_count,
    input  wire logic signed [7:0]                     i_activation,
    input  wire logic                                  i_acc24_compat,
    output logic                                       o_row_done,

    output logic [TILE_COUNT-1:0]                     o_weight_req_valid,
    input  wire logic [TILE_COUNT-1:0]                i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_weight_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]             i_weight_data_flat,

    input  wire logic                                  i_lif_start,
    input  wire logic signed [15:0]                   i_leak_q15,
    input  wire logic signed [31:0]                   i_threshold,
    input  wire logic [4:0]                            i_current_left_shift,
    input  wire logic                                  i_reset_to_zero,
    output logic                                       o_lif_busy,
    output logic                                       o_lif_done,
    input  wire logic                                  i_timestep_commit,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]    o_previous_spikes_flat,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]    o_current_spikes_flat,
    output logic                                       o_sticky_error
);
    logic [TILE_COUNT-1:0] tile_active;
    logic [TILE_COUNT-1:0] dispatcher_ready;
    logic [TILE_COUNT-1:0] dispatcher_done;
    logic [TILE_COUNT-1:0] tile_mac_ready;
    logic [TILE_COUNT-1:0] tile_lif_done;
    logic [TILE_COUNT-1:0] tile_collision;
    logic [TILE_COUNT-1:0] row_pending_q;
    logic [TILE_COUNT-1:0] lif_pending_q;
    logic row_active_q;
    logic lif_active_q;
    logic row_fire;
    logic lif_fire;
    integer active_i;
    genvar tile;

    always_comb begin
        tile_active = '0;
        for (active_i = 0; active_i < TILE_COUNT; active_i = active_i + 1)
            if ((active_i < i_active_tiles) &&
                (i_destination_count > active_i*NEURONS_PER_TILE))
                tile_active[active_i] = 1'b1;

        o_row_ready = !row_active_q &&
            ((dispatcher_ready | ~tile_active) == {TILE_COUNT{1'b1}});
        row_fire = i_row_valid && o_row_ready;
        lif_fire = i_lif_start && !lif_active_q && (i_active_tiles != 0) &&
            (i_active_tiles <= TILE_COUNT);
        o_lif_busy = lif_active_q;
    end

    generate
        for (tile = 0; tile < TILE_COUNT; tile = tile + 1) begin : g_tile
            localparam integer TILE_BASE = tile * NEURONS_PER_TILE;
            logic [9:0] remaining_destinations;
            logic [5:0] local_destination_count;
            logic dispatch_req_valid;
            logic [31:0] dispatch_weight_addr;
            logic [LANES-1:0] dispatch_lane_valid;
            logic [LANES*6-1:0] dispatch_neuron_index;
            logic [LANES*8-1:0] dispatch_activation;
            logic lif_busy_unused;
            logic lif_spike_valid_unused;
            logic [5:0] lif_base_unused;
            logic [LANES-1:0] lif_spike_mask_unused;
            logic signed [31:0] debug_accum_unused;
            logic signed [31:0] debug_membrane_unused;

            always_comb begin
                remaining_destinations = (i_destination_count > TILE_BASE) ?
                    i_destination_count - TILE_BASE : 10'd0;
                local_destination_count =
                    (remaining_destinations > NEURONS_PER_TILE) ?
                    NEURONS_PER_TILE : remaining_destinations[5:0];
                o_weight_req_valid[tile] = dispatch_req_valid;
                o_weight_addr_flat[tile*32 +: 32] = dispatch_weight_addr;
            end

            unified_k8_weight_row_dispatch #(
                .LANES(LANES), .MAX_DESTINATIONS(NEURONS_PER_TILE)
            ) u_dispatch (
                .clk(clk), .rst_n(rst_n),
                .i_cmd_valid(row_fire && tile_active[tile]),
                .o_cmd_ready(dispatcher_ready[tile]),
                .i_weight_row_base(i_weight_row_base + TILE_BASE),
                .i_destination_count(local_destination_count),
                .i_activation(i_activation),
                .o_req_valid(dispatch_req_valid),
                .i_req_ready(i_weight_req_ready[tile] && tile_mac_ready[tile]),
                .o_weight_addr(dispatch_weight_addr),
                .o_lane_valid(dispatch_lane_valid),
                .o_neuron_index_flat(dispatch_neuron_index),
                .o_activation_flat(dispatch_activation),
                .o_row_done(dispatcher_done[tile])
            );

            unified_h40_k8_tile #(
                .NEURONS(NEURONS_PER_TILE), .LANES(LANES)
            ) u_compute (
                .clk(clk), .rst_n(rst_n), .i_clear_sample(i_clear_sample),
                .i_clear_accum(i_clear_accum),
                .i_mac_valid(dispatch_req_valid && i_weight_req_ready[tile] &&
                             tile_mac_ready[tile]),
                .o_mac_ready(tile_mac_ready[tile]),
                .i_lane_valid(dispatch_lane_valid),
                .i_neuron_index_flat(dispatch_neuron_index),
                .i_activation_flat(dispatch_activation),
                .i_weight_flat(i_weight_data_flat[tile*64 +: 64]),
                .i_acc24_compat(i_acc24_compat),
                .i_lif_start(lif_fire && tile_active[tile]),
                .i_leak_q15(i_leak_q15), .i_threshold(i_threshold),
                .i_current_left_shift(i_current_left_shift),
                .i_reset_to_zero(i_reset_to_zero), .o_lif_busy(lif_busy_unused),
                .o_lif_done(tile_lif_done[tile]),
                .o_lif_spike_valid(lif_spike_valid_unused),
                .o_lif_base(lif_base_unused),
                .o_lif_spike_mask(lif_spike_mask_unused),
                .i_timestep_commit(i_timestep_commit && tile_active[tile]),
                .o_previous_spikes(
                    o_previous_spikes_flat[tile*NEURONS_PER_TILE +:
                                           NEURONS_PER_TILE]),
                .o_current_spikes(
                    o_current_spikes_flat[tile*NEURONS_PER_TILE +:
                                          NEURONS_PER_TILE]),
                .i_debug_neuron(6'd0), .o_debug_accum(debug_accum_unused),
                .o_debug_membrane(debug_membrane_unused),
                .o_lane_collision_error(tile_collision[tile])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_pending_q <= '0;
            lif_pending_q <= '0;
            row_active_q <= 1'b0;
            lif_active_q <= 1'b0;
            o_row_done <= 1'b0;
            o_lif_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_row_done <= 1'b0;
            o_lif_done <= 1'b0;
            if (i_clear_sample)
                o_sticky_error <= 1'b0;
            if ((row_fire &&
                 ((i_active_tiles == 0) || (i_active_tiles > TILE_COUNT) ||
                  (i_destination_count == 0) ||
                  (i_destination_count > i_active_tiles*NEURONS_PER_TILE))) ||
                (i_lif_start &&
                 ((i_active_tiles == 0) || (i_active_tiles > TILE_COUNT))) ||
                (|tile_collision))
                o_sticky_error <= 1'b1;

            if (row_fire) begin
                row_pending_q <= tile_active;
                row_active_q <= |tile_active;
                if (!(|tile_active))
                    o_row_done <= 1'b1;
            end else if (row_active_q) begin
                row_pending_q <= row_pending_q & ~dispatcher_done;
                if ((row_pending_q & ~dispatcher_done) == 0) begin
                    row_active_q <= 1'b0;
                    o_row_done <= 1'b1;
                end
            end

            if (lif_fire) begin
                lif_pending_q <= tile_active;
                lif_active_q <= |tile_active;
            end else if (lif_active_q) begin
                lif_pending_q <= lif_pending_q & ~tile_lif_done;
                if ((lif_pending_q & ~tile_lif_done) == 0) begin
                    lif_active_q <= 1'b0;
                    o_lif_done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
