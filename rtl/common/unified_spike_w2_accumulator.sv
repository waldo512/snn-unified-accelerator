`timescale 1ns/1ps
`default_nettype none

// Shared extension output primitive. Each H40 Tile scans its own current spike
// vector and accumulates row-major INT8 W2 values into per-Tile ACC32 partials.
// Canonical 10-class FC continues to bypass this block through its R18 reducer.
module unified_spike_w2_accumulator #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40,
    parameter integer OUTPUT_MAX = 32,
    parameter integer ACC_BITS = 32
) (
    input  wire logic                                            clk,
    input  wire logic                                            rst_n,
    input  wire logic                                            i_start,
    output logic                                                 o_ready,
    input  wire logic [3:0]                                      i_active_tiles,
    input  wire logic [5:0]                                      i_output_count,
    input  wire logic [5:0]                                      i_w2_row_stride,
    input  wire logic [31:0]                                     i_w2_base,
    input  wire logic [TILE_COUNT*NEURONS_PER_TILE-1:0]         i_spikes_flat,
    output logic [TILE_COUNT-1:0]                               o_w2_req_valid,
    input  wire logic [TILE_COUNT-1:0]                          i_w2_req_ready,
    output logic [TILE_COUNT*32-1:0]                            o_w2_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]                       i_w2_data_flat,
    output logic [TILE_COUNT*OUTPUT_MAX*ACC_BITS-1:0]           o_partials_flat,
    output logic                                                 o_done,
    output logic                                                 o_busy,
    output logic                                                 o_sticky_error
);
    logic busy_q;
    logic [TILE_COUNT-1:0] tile_busy_q;
    logic [TILE_COUNT-1:0][5:0] neuron_q;
    logic [TILE_COUNT-1:0][2:0] word_q;
    logic [3:0] active_tiles_q;
    logic [5:0] output_count_q;
    logic [5:0] row_stride_q;
    logic [31:0] w2_base_q;
    logic [TILE_COUNT*NEURONS_PER_TILE-1:0] spikes_q;
    logic signed [ACC_BITS-1:0] partial_q [0:TILE_COUNT-1][0:OUTPUT_MAX-1];
    integer comb_tile;
    integer comb_class;
    integer seq_tile;
    integer seq_class;
    integer lane;
    integer global_neuron;
    integer current_class;

    function automatic logic signed [7:0] lane_weight(
        input integer tile_index,
        input integer lane_index
    );
        lane_weight = $signed(i_w2_data_flat[tile_index*64 + lane_index*8 +: 8]);
    endfunction

    always_comb begin
        o_ready = !busy_q;
        o_busy = busy_q;
        o_w2_req_valid = '0;
        o_w2_addr_flat = '0;
        o_partials_flat = '0;
        for (comb_tile = 0; comb_tile < TILE_COUNT; comb_tile = comb_tile + 1) begin
            for (comb_class = 0; comb_class < OUTPUT_MAX; comb_class = comb_class + 1)
                o_partials_flat[(comb_tile*OUTPUT_MAX+comb_class)*ACC_BITS +: ACC_BITS] =
                    partial_q[comb_tile][comb_class];
            if (tile_busy_q[comb_tile] &&
                (neuron_q[comb_tile] < NEURONS_PER_TILE) &&
                spikes_q[comb_tile*NEURONS_PER_TILE + neuron_q[comb_tile]]) begin
                o_w2_req_valid[comb_tile] = 1'b1;
                o_w2_addr_flat[comb_tile*32 +: 32] = w2_base_q +
                    ((comb_tile*NEURONS_PER_TILE + neuron_q[comb_tile]) * row_stride_q) +
                    (word_q[comb_tile] * 8);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            tile_busy_q <= '0;
            neuron_q <= '0;
            word_q <= '0;
            active_tiles_q <= '0;
            output_count_q <= '0;
            row_stride_q <= '0;
            w2_base_q <= '0;
            spikes_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
            for (seq_tile = 0; seq_tile < TILE_COUNT; seq_tile = seq_tile + 1)
                for (seq_class = 0; seq_class < OUTPUT_MAX; seq_class = seq_class + 1)
                    partial_q[seq_tile][seq_class] <= '0;
        end else begin
            o_done <= 1'b0;
            if (!busy_q && i_start) begin
                if ((i_active_tiles == 0) || (i_active_tiles > TILE_COUNT) ||
                    (i_output_count == 0) || (i_output_count > OUTPUT_MAX) ||
                    (i_w2_row_stride < i_output_count)) begin
                    o_sticky_error <= 1'b1;
                end else begin
                    busy_q <= 1'b1;
                    active_tiles_q <= i_active_tiles;
                    output_count_q <= i_output_count;
                    row_stride_q <= i_w2_row_stride;
                    w2_base_q <= i_w2_base;
                    spikes_q <= i_spikes_flat;
                    for (seq_tile = 0; seq_tile < TILE_COUNT; seq_tile = seq_tile + 1) begin
                        tile_busy_q[seq_tile] <= seq_tile < i_active_tiles;
                        neuron_q[seq_tile] <= '0;
                        word_q[seq_tile] <= '0;
                        for (seq_class = 0; seq_class < OUTPUT_MAX; seq_class = seq_class + 1)
                            partial_q[seq_tile][seq_class] <= '0;
                    end
                end
            end else if (busy_q) begin
                for (seq_tile = 0; seq_tile < TILE_COUNT; seq_tile = seq_tile + 1) begin
                    if (tile_busy_q[seq_tile]) begin
                        if (neuron_q[seq_tile] >= NEURONS_PER_TILE) begin
                            tile_busy_q[seq_tile] <= 1'b0;
                        end else if (!spikes_q[seq_tile*NEURONS_PER_TILE + neuron_q[seq_tile]]) begin
                            neuron_q[seq_tile] <= neuron_q[seq_tile] + 1'b1;
                        end else if (o_w2_req_valid[seq_tile] && i_w2_req_ready[seq_tile]) begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                current_class = word_q[seq_tile]*8 + lane;
                                if (current_class < output_count_q)
                                    partial_q[seq_tile][current_class] <=
                                        partial_q[seq_tile][current_class] + lane_weight(seq_tile, lane);
                            end
                            if ((word_q[seq_tile] + 1)*8 >= output_count_q) begin
                                word_q[seq_tile] <= '0;
                                neuron_q[seq_tile] <= neuron_q[seq_tile] + 1'b1;
                            end else begin
                                word_q[seq_tile] <= word_q[seq_tile] + 1'b1;
                            end
                        end
                    end
                end
                if (tile_busy_q == '0) begin
                    busy_q <= 1'b0;
                    o_done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
