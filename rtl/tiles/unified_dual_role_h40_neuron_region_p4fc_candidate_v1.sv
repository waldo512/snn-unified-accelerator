`timescale 1ns/1ps
`default_nettype none

// P4-FC front end for the existing four-tile neuron service. It arbitrates one
// ten-lane group, merges the two FC accumulator banks in ten shared DSPs, and
// then presents the merged group to the unchanged normalize/LIF/trace region.
module unified_dual_role_h40_neuron_region_p4fc_candidate_v1 #(
    parameter int REGION_TILES = 4,
    parameter int META_WIDTH = 64,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_abort,

    input  wire logic [REGION_TILES-1:0]             i_tile_valid,
    output logic [REGION_TILES-1:0]                  o_tile_ready,
    input  wire logic [REGION_TILES*2-1:0]           i_tile_mode_flat,
    input  wire logic [REGION_TILES*10-1:0]          i_tile_lane_mask_flat,
    input  wire logic [REGION_TILES*480-1:0]         i_tile_raw_current_flat,
    input  wire logic [REGION_TILES-1:0]             i_tile_p4_fc,
    input  wire logic [REGION_TILES*480-1:0]         i_tile_raw_current_secondary_flat,
    input  wire logic [REGION_TILES*320-1:0]         i_tile_state_flat,
    input  wire logic [REGION_TILES*320-1:0]         i_tile_trace_old_flat,
    input  wire logic [REGION_TILES*16-1:0]          i_tile_leak_flat,
    input  wire logic [REGION_TILES*32-1:0]          i_tile_threshold_flat,
    input  wire logic [REGION_TILES-1:0]             i_tile_reset_to_zero,
    input  wire logic [REGION_TILES*16-1:0]          i_tile_trace_decay_flat,
    input  wire logic [REGION_TILES*META_WIDTH-1:0]  i_tile_meta_flat,

    output logic                                     o_record_valid,
    input  wire logic                                i_record_ready,
    output logic [$clog2(REGION_TILES)-1:0]          o_record_tile,
    output logic [1:0]                               o_record_mode,
    output logic [9:0]                               o_record_lane_mask,
    output logic [9:0]                               o_record_range_mask,
    output logic [9:0]                               o_record_spike,
    output logic [319:0]                             o_record_pre_state,
    output logic [319:0]                             o_record_next_state,
    output logic [319:0]                             o_record_trace_next,
    output logic [META_WIDTH-1:0]                    o_record_meta,
    output logic                                     o_record_mode_error,
    output logic                                     o_idle
);
    localparam int TILE_ID_WIDTH = $clog2(REGION_TILES);

    logic [TILE_ID_WIDTH-1:0] round_robin_q;
    logic selected_valid;
    logic [TILE_ID_WIDTH-1:0] selected_tile;
    logic [1:0] selected_mode;
    logic [9:0] selected_lane_mask;
    logic [479:0] selected_primary;
    logic selected_p4_fc;
    logic [479:0] selected_secondary;
    logic [319:0] selected_state;
    logic [319:0] selected_trace_old;
    logic signed [15:0] selected_leak;
    logic signed [31:0] selected_threshold;
    logic selected_reset_to_zero;
    logic [15:0] selected_trace_decay;
    logic [META_WIDTH-1:0] selected_meta;
    logic selected_fire;

    logic merge_input_ready;
    logic merge_output_valid;
    logic merge_output_ready;
    logic [9:0] merge_lane_mask;
    logic [479:0] merge_current;

    logic [TILE_ID_WIDTH-1:0] merge_tile_q;
    logic [1:0] merge_mode_q;
    logic [319:0] merge_state_q;
    logic [319:0] merge_trace_old_q;
    logic signed [15:0] merge_leak_q;
    logic signed [31:0] merge_threshold_q;
    logic merge_reset_to_zero_q;
    logic [15:0] merge_trace_decay_q;
    logic [META_WIDTH-1:0] merge_meta_q;

    logic [REGION_TILES-1:0] base_tile_valid;
    logic [REGION_TILES-1:0] base_tile_ready;
    logic [REGION_TILES*2-1:0] base_mode_flat;
    logic [REGION_TILES*10-1:0] base_lane_mask_flat;
    logic [REGION_TILES*480-1:0] base_raw_current_flat;
    logic [REGION_TILES*320-1:0] base_state_flat;
    logic [REGION_TILES*320-1:0] base_trace_old_flat;
    logic [REGION_TILES*16-1:0] base_leak_flat;
    logic [REGION_TILES*32-1:0] base_threshold_flat;
    logic [REGION_TILES-1:0] base_reset_to_zero;
    logic [REGION_TILES*16-1:0] base_trace_decay_flat;
    logic [REGION_TILES*META_WIDTH-1:0] base_meta_flat;
    logic base_idle;

    always_comb begin
        logic found;
        int unsigned candidate;

        selected_valid = 1'b0;
        selected_tile = round_robin_q;
        selected_mode = '0;
        selected_lane_mask = '0;
        selected_primary = '0;
        selected_p4_fc = 1'b0;
        selected_secondary = '0;
        selected_state = '0;
        selected_trace_old = '0;
        selected_leak = '0;
        selected_threshold = '0;
        selected_reset_to_zero = 1'b0;
        selected_trace_decay = '0;
        selected_meta = '0;
        o_tile_ready = '0;
        found = 1'b0;

        for (int offset = 0; offset < REGION_TILES; offset++) begin
            candidate = 32'(round_robin_q) + offset;
            if (candidate >= REGION_TILES)
                candidate = candidate - REGION_TILES;
            if (!found && i_tile_valid[candidate]) begin
                found = 1'b1;
                selected_valid = 1'b1;
                selected_tile = TILE_ID_WIDTH'(candidate);
            end
        end

        if (selected_valid) begin
            selected_mode = i_tile_mode_flat[selected_tile*2 +: 2];
            selected_lane_mask =
                i_tile_lane_mask_flat[selected_tile*10 +: 10];
            selected_primary =
                i_tile_raw_current_flat[selected_tile*480 +: 480];
            selected_p4_fc = i_tile_p4_fc[selected_tile];
            selected_secondary = i_tile_raw_current_secondary_flat[
                selected_tile*480 +: 480];
            selected_state =
                i_tile_state_flat[selected_tile*320 +: 320];
            selected_trace_old =
                i_tile_trace_old_flat[selected_tile*320 +: 320];
            selected_leak =
                $signed(i_tile_leak_flat[selected_tile*16 +: 16]);
            selected_threshold =
                $signed(i_tile_threshold_flat[selected_tile*32 +: 32]);
            selected_reset_to_zero = i_tile_reset_to_zero[selected_tile];
            selected_trace_decay =
                i_tile_trace_decay_flat[selected_tile*16 +: 16];
            selected_meta =
                i_tile_meta_flat[selected_tile*META_WIDTH +: META_WIDTH];
            o_tile_ready[selected_tile] = merge_input_ready && !i_abort;
        end
    end

    assign selected_fire = selected_valid && o_tile_ready[selected_tile];
    assign merge_output_ready = base_tile_ready[merge_tile_q];
    assign o_idle = !selected_valid && !merge_output_valid && base_idle;

    always_comb begin
        base_tile_valid = '0;
        base_mode_flat = '0;
        base_lane_mask_flat = '0;
        base_raw_current_flat = '0;
        base_state_flat = '0;
        base_trace_old_flat = '0;
        base_leak_flat = '0;
        base_threshold_flat = '0;
        base_reset_to_zero = '0;
        base_trace_decay_flat = '0;
        base_meta_flat = '0;

        if (merge_output_valid) begin
            base_tile_valid[merge_tile_q] = 1'b1;
            base_mode_flat[merge_tile_q*2 +: 2] = merge_mode_q;
            base_lane_mask_flat[merge_tile_q*10 +: 10] = merge_lane_mask;
            base_raw_current_flat[merge_tile_q*480 +: 480] = merge_current;
            base_state_flat[merge_tile_q*320 +: 320] = merge_state_q;
            base_trace_old_flat[merge_tile_q*320 +: 320] =
                merge_trace_old_q;
            base_leak_flat[merge_tile_q*16 +: 16] = merge_leak_q;
            base_threshold_flat[merge_tile_q*32 +: 32] =
                merge_threshold_q;
            base_reset_to_zero[merge_tile_q] = merge_reset_to_zero_q;
            base_trace_decay_flat[merge_tile_q*16 +: 16] =
                merge_trace_decay_q;
            base_meta_flat[merge_tile_q*META_WIDTH +: META_WIDTH] =
                merge_meta_q;
        end
    end

    unified_dual_role_p4fc_pair_merge10_candidate_v1 u_pair_merge (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(selected_valid), .o_ready(merge_input_ready),
        .i_p4_fc(selected_p4_fc), .i_lane_mask(selected_lane_mask),
        .i_primary_flat(selected_primary),
        .i_secondary_flat(selected_secondary),
        .o_valid(merge_output_valid), .i_ready(merge_output_ready),
        .o_lane_mask(merge_lane_mask), .o_merged_flat(merge_current)
    );

    unified_dual_role_h40_neuron_region_candidate_v1 #(
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .REGION_TILES(REGION_TILES), .META_WIDTH(META_WIDTH)
    ) u_base_region (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_tile_valid(base_tile_valid), .o_tile_ready(base_tile_ready),
        .i_tile_mode_flat(base_mode_flat),
        .i_tile_lane_mask_flat(base_lane_mask_flat),
        .i_tile_raw_current_flat(base_raw_current_flat),
        .i_tile_state_flat(base_state_flat),
        .i_tile_trace_old_flat(base_trace_old_flat),
        .i_tile_leak_flat(base_leak_flat),
        .i_tile_threshold_flat(base_threshold_flat),
        .i_tile_reset_to_zero(base_reset_to_zero),
        .i_tile_trace_decay_flat(base_trace_decay_flat),
        .i_tile_meta_flat(base_meta_flat),
        .o_record_valid(o_record_valid), .i_record_ready(i_record_ready),
        .o_record_tile(o_record_tile), .o_record_mode(o_record_mode),
        .o_record_lane_mask(o_record_lane_mask),
        .o_record_range_mask(o_record_range_mask),
        .o_record_spike(o_record_spike),
        .o_record_pre_state(o_record_pre_state),
        .o_record_next_state(o_record_next_state),
        .o_record_trace_next(o_record_trace_next),
        .o_record_meta(o_record_meta),
        .o_record_mode_error(o_record_mode_error), .o_idle(base_idle)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            round_robin_q <= '0;
            merge_tile_q <= '0;
            merge_mode_q <= '0;
            merge_state_q <= '0;
            merge_trace_old_q <= '0;
            merge_leak_q <= '0;
            merge_threshold_q <= '0;
            merge_reset_to_zero_q <= 1'b0;
            merge_trace_decay_q <= '0;
            merge_meta_q <= '0;
        end else if (selected_fire) begin
            merge_tile_q <= selected_tile;
            merge_mode_q <= selected_mode;
            merge_state_q <= selected_state;
            merge_trace_old_q <= selected_trace_old;
            merge_leak_q <= selected_leak;
            merge_threshold_q <= selected_threshold;
            merge_reset_to_zero_q <= selected_reset_to_zero;
            merge_trace_decay_q <= selected_trace_decay;
            merge_meta_q <= selected_meta;
            if (selected_tile == TILE_ID_WIDTH'(REGION_TILES - 1))
                round_robin_q <= '0;
            else
                round_robin_q <= selected_tile + 1'b1;
        end
    end

    initial begin
        if (REGION_TILES != 4 || META_WIDTH != 64)
            $error("P4-FC neuron region requires four tiles and 64-bit metadata");
    end
endmodule

`default_nettype wire
