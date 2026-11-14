`timescale 1ns/1ps
`default_nettype none

// Converts the existing contiguous 12xK8 row-response ABI into five
// tile-major words. The external memory organization remains unchanged;
// only the short row buffer is transposed before the fixed H40 tile array.
module unified_srnn_tile_major_weight_transposer_v1 #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer TILES = 12,
    parameter integer TILE_NEURONS = 40,
    parameter integer ENGINES = 12,
    parameter integer LANES = 8
) (
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 i_abort,
    input  logic                                 i_clear_error,

    input  logic                                 i_row_start_valid,
    output logic                                 o_row_start_ready,
    input  logic [9:0]                           i_destination_count,

    input  logic                                 i_wave_valid,
    output logic                                 o_wave_ready,
    input  logic [9:0]                           i_wave_destination_base,
    input  logic [ENGINES-1:0]                   i_wave_engine_mask,
    input  logic [LANES-1:0]                     i_wave_lane_mask,
    input  logic [ENGINES*LANES*8-1:0]           i_wave_weight_flat,

    output logic                                 o_word_valid,
    input  logic                                 i_word_ready,
    output logic [2:0]                           o_local_word,
    output logic [TILES-1:0]                     o_tile_mask,
    output logic [LANES-1:0]                     o_lane_mask,
    output logic [TILES*LANES*8-1:0]             o_tile_major_weight_flat,
    output logic                                 o_word_last,

    output logic                                 o_busy,
    output logic                                 o_idle,
    output logic                                 o_sticky_error
);
    localparam integer DESTINATIONS_PER_WAVE = ENGINES*LANES;
    localparam integer LOCAL_WORDS = TILE_NEURONS/LANES;
    localparam integer WAVE_COUNT =
        (HIDDEN_MAX + DESTINATIONS_PER_WAVE - 1) /
        DESTINATIONS_PER_WAVE;
    localparam logic [1:0] STATE_IDLE = 2'd0;
    localparam logic [1:0] STATE_GATHER = 2'd1;
    localparam logic [1:0] STATE_ISSUE = 2'd2;

    logic [1:0] state_q;
    logic [9:0] destination_count_q;
    logic [9:0] expected_wave_base_q;
    logic [2:0] local_word_q;
    logic signed [7:0] row_weight_q [0:HIDDEN_MAX-1];
    logic [ENGINES-1:0] expected_wave_engine_mask;
    logic wave_shape_legal;
    logic final_wave;

    always_comb begin
        expected_wave_engine_mask = '0;
        for (integer engine = 0; engine < ENGINES; engine++)
            if (integer'(expected_wave_base_q) + engine*LANES <
                integer'(destination_count_q))
                expected_wave_engine_mask[engine] = 1'b1;

        wave_shape_legal =
            i_wave_destination_base == expected_wave_base_q &&
            i_wave_engine_mask == expected_wave_engine_mask &&
            i_wave_lane_mask == {LANES{1'b1}};
        final_wave = integer'(expected_wave_base_q) +
            DESTINATIONS_PER_WAVE >= integer'(destination_count_q);

        o_row_start_ready = state_q == STATE_IDLE && !o_sticky_error;
        o_wave_ready = state_q == STATE_GATHER && !o_sticky_error;
        o_word_valid = state_q == STATE_ISSUE && !o_sticky_error;
        o_local_word = local_word_q;
        o_word_last = local_word_q == 3'(LOCAL_WORDS-1);
        o_lane_mask = {LANES{1'b1}};
        o_tile_mask = '0;
        o_tile_major_weight_flat = '0;
        for (integer tile = 0; tile < TILES; tile++) begin
            if (tile*TILE_NEURONS < integer'(destination_count_q))
                o_tile_mask[tile] = 1'b1;
            for (integer lane = 0; lane < LANES; lane++) begin
                if (tile*TILE_NEURONS + integer'(local_word_q)*LANES +
                    lane < integer'(destination_count_q))
                    o_tile_major_weight_flat[
                        (tile*LANES + lane)*8 +: 8
                    ] = row_weight_q[tile*TILE_NEURONS +
                        integer'(local_word_q)*LANES + lane];
            end
        end
        o_busy = state_q != STATE_IDLE;
        o_idle = state_q == STATE_IDLE;
    end

    // Each destination has one compile-time source position in one contiguous
    // response wave. Fixed write points avoid synthesizing a 96-write dynamic
    // decoder across the complete row buffer. The FSM remains the sole
    // validity authority, so these payload registers need no reset.
    for (genvar destination = 0; destination < HIDDEN_MAX; destination++) begin : g_destination_write
        localparam integer INPUT_ITEM =
            destination % DESTINATIONS_PER_WAVE;
        localparam integer WAVE_BASE = destination - INPUT_ITEM;
        localparam integer INPUT_ENGINE = INPUT_ITEM / LANES;

        always_ff @(posedge clk) begin
            if (!i_abort && state_q == STATE_GATHER &&
                i_wave_valid && o_wave_ready && wave_shape_legal &&
                expected_wave_base_q == 10'(WAVE_BASE) &&
                destination < integer'(destination_count_q) &&
                i_wave_engine_mask[INPUT_ENGINE]) begin
                row_weight_q[destination] <=
                    i_wave_weight_flat[INPUT_ITEM*8 +: 8];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            destination_count_q <= '0;
            expected_wave_base_q <= '0;
            local_word_q <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            destination_count_q <= '0;
            expected_wave_base_q <= '0;
            local_word_q <= '0;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (i_row_start_valid) begin
                        if (i_destination_count == 0 ||
                            i_destination_count > 10'(HIDDEN_MAX)) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            destination_count_q <= i_destination_count;
                            expected_wave_base_q <= '0;
                            state_q <= STATE_GATHER;
                        end
                    end
                    if (i_wave_valid)
                        o_sticky_error <= 1'b1;
                end
                STATE_GATHER: begin
                    if (i_wave_valid && o_wave_ready) begin
                        if (!wave_shape_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            if (final_wave) begin
                                local_word_q <= '0;
                                state_q <= STATE_ISSUE;
                            end else begin
                                expected_wave_base_q <=
                                    expected_wave_base_q +
                                    10'(DESTINATIONS_PER_WAVE);
                            end
                        end
                    end
                end
                STATE_ISSUE: begin
                    if (i_wave_valid)
                        o_sticky_error <= 1'b1;
                    if (o_word_valid && i_word_ready) begin
                        if (o_word_last) begin
                            state_q <= STATE_IDLE;
                            local_word_q <= '0;
                        end else begin
                            local_word_q <= local_word_q + 1'b1;
                        end
                    end
                end
                default: begin
                    state_q <= STATE_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (TILES*TILE_NEURONS != HIDDEN_MAX)
            $error("tile-major transposer requires exact HIDDEN_MAX coverage");
        if (TILE_NEURONS % LANES != 0 || LOCAL_WORDS != 5)
            $error("tile-major transposer requires H40/K8 geometry");
        if (WAVE_COUNT > 8)
            $error("wave counter assumptions require at most eight waves");
    end
endmodule

`default_nettype wire
