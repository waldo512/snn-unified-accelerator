`timescale 1ns/1ps
`default_nettype none

// Tile-local recurrent state for forty logical neurons. Global K8 response
// and LIF word addresses are accepted directly so the controller does not
// need a second state-address translation layer.
module unified_srnn_h40_state_shard_v1 #(
    parameter integer TILE_BASE = 0,
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

    output logic [TILE_NEURONS-1:0]              o_previous_spikes,
    output logic [TILE_NEURONS-1:0]              o_current_spikes,
    output logic [TILE_NEURONS*32-1:0]           o_accumulator_flat,
    output logic [TILE_NEURONS*32-1:0]           o_membrane_flat
);
    localparam integer WORD_COUNT = TILE_NEURONS / LANES;
    localparam integer WORD_INDEX_WIDTH =
        (WORD_COUNT <= 1) ? 1 : $clog2(WORD_COUNT);
    logic signed [31:0] accumulator_bank [0:LANES-1][0:WORD_COUNT-1];
    logic signed [31:0] membrane_bank [0:LANES-1][0:WORD_COUNT-1];
    logic [TILE_NEURONS-1:0] previous_spikes_q;
    logic [TILE_NEURONS-1:0] current_spikes_q;
    logic clear_hits;
    logic accumulate_hits;
    logic lif_hits;
    logic [10:0] clear_offset;
    logic [10:0] accumulate_offset;
    logic [10:0] lif_offset;
    logic [WORD_INDEX_WIDTH-1:0] clear_word;
    logic [WORD_INDEX_WIDTH-1:0] accumulate_word;
    logic [WORD_INDEX_WIDTH-1:0] lif_word;

    function automatic logic signed [31:0] saturating_add32(
        input logic signed [31:0] lhs,
        input logic signed [31:0] rhs
    );
        logic signed [32:0] sum_wide;
        begin
            sum_wide = {lhs[31], lhs} + {rhs[31], rhs};
            if (sum_wide > 33'sh07fffffff)
                saturating_add32 = 32'sh7fffffff;
            else if (sum_wide < -33'sh080000000)
                saturating_add32 = 32'sh80000000;
            else
                saturating_add32 = sum_wide[31:0];
        end
    endfunction

    function automatic logic signed [31:0] lif_pre_value(
        input logic signed [31:0] old_membrane,
        input logic signed [31:0] current,
        input logic signed [15:0] leak_q15,
        input logic [4:0] current_left_shift
    );
        logic signed [47:0] leak_product;
        logic signed [63:0] leak_scaled;
        logic signed [63:0] current_scaled;
        logic signed [63:0] sum_wide;
        begin
            leak_product = old_membrane * leak_q15;
            leak_scaled = {{16{leak_product[47]}}, leak_product};
            current_scaled = {{32{current[31]}}, current};
            current_scaled = current_scaled <<< current_left_shift;
            sum_wide = (leak_scaled >>> 15) + current_scaled;
            if (sum_wide > 64'sh000000007fffffff)
                lif_pre_value = 32'sh7fffffff;
            else if (sum_wide < -64'sh0000000080000000)
                lif_pre_value = 32'sh80000000;
            else
                lif_pre_value = sum_wide[31:0];
        end
    endfunction

    always_comb begin
        clear_offset = {1'b0, i_clear_word_base} - 11'(TILE_BASE);
        accumulate_offset = {1'b0, i_accumulate_base} - 11'(TILE_BASE);
        lif_offset = {1'b0, i_lif_word_base} - 11'(TILE_BASE);
        clear_hits = i_clear_word_valid && !clear_offset[10] &&
            clear_offset < 11'(TILE_NEURONS);
        accumulate_hits = i_accumulate_valid && !accumulate_offset[10] &&
            accumulate_offset < 11'(TILE_NEURONS);
        lif_hits = i_lif_word_valid && !lif_offset[10] &&
            lif_offset < 11'(TILE_NEURONS);
        clear_word = WORD_INDEX_WIDTH'(
            clear_offset / LANES);
        accumulate_word = WORD_INDEX_WIDTH'(
            accumulate_offset / LANES);
        lif_word = WORD_INDEX_WIDTH'(
            lif_offset / LANES);

        o_previous_spikes = previous_spikes_q;
        o_current_spikes = current_spikes_q;
        o_accumulator_flat = '0;
        o_membrane_flat = '0;
        for (int neuron = 0; neuron < TILE_NEURONS; neuron++) begin
            o_accumulator_flat[neuron*32 +: 32] =
                accumulator_bank[neuron % LANES][neuron / LANES];
            o_membrane_flat[neuron*32 +: 32] =
                membrane_bank[neuron % LANES][neuron / LANES];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
            for (int lane = 0; lane < LANES; lane++)
                for (int word = 0; word < WORD_COUNT; word++) begin
                    accumulator_bank[lane][word] <= 32'sd0;
                    membrane_bank[lane][word] <= 32'sd0;
                end
        end else if (i_abort) begin
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
        end else begin
            if (i_clear_spikes) begin
                previous_spikes_q <= '0;
                current_spikes_q <= '0;
            end

            if (clear_hits) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    accumulator_bank[lane][clear_word] <= 32'sd0;
                    membrane_bank[lane][clear_word] <= 32'sd0;
                    previous_spikes_q[clear_word*LANES + lane] <= 1'b0;
                    current_spikes_q[clear_word*LANES + lane] <= 1'b0;
                end
            end

            if (accumulate_hits) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    if (i_accumulate_lane_mask[lane] &&
                        (i_accumulate_base + 10'(lane)) < i_active_neurons)
                        accumulator_bank[lane][accumulate_word] <=
                            saturating_add32(
                                accumulator_bank[lane][accumulate_word],
                                $signed(i_accumulate_contribution_flat[
                                    lane*32 +: 32]));
                end
            end

            if (i_lif_begin)
                current_spikes_q <= '0;

            if (lif_hits) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    if ((i_lif_word_base + 10'(lane)) < i_active_neurons) begin
                        if (lif_pre_value(
                                membrane_bank[lane][lif_word],
                                accumulator_bank[lane][lif_word],
                                i_leak_q15, i_current_left_shift) >=
                            i_threshold) begin
                            current_spikes_q[lif_word*LANES + lane] <= 1'b1;
                            if (i_reset_to_zero)
                                membrane_bank[lane][lif_word] <= 32'sd0;
                            else
                                membrane_bank[lane][lif_word] <=
                                    lif_pre_value(
                                        membrane_bank[lane][lif_word],
                                        accumulator_bank[lane][lif_word],
                                        i_leak_q15, i_current_left_shift) -
                                    i_threshold;
                        end else begin
                            current_spikes_q[lif_word*LANES + lane] <= 1'b0;
                            membrane_bank[lane][lif_word] <=
                                lif_pre_value(
                                    membrane_bank[lane][lif_word],
                                    accumulator_bank[lane][lif_word],
                                    i_leak_q15, i_current_left_shift);
                        end
                        accumulator_bank[lane][lif_word] <= 32'sd0;
                    end
                end
            end

            if (i_commit_timestep) begin
                previous_spikes_q <= current_spikes_q;
                current_spikes_q <= '0;
            end
        end
    end

    initial begin
        if (TILE_NEURONS % LANES != 0)
            $error("SRNN state shard requires TILE_NEURONS divisible by LANES");
        if (TILE_BASE % LANES != 0)
            $error("SRNN state shard TILE_BASE must align to LANES");
    end
endmodule

`default_nettype wire
