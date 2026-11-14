`timescale 1ns/1ps
`default_nettype none

// Four neighboring H40 tiles share one fully pipelined ten-lane trace unit.
// Tile LIF result slots already retain unaccepted payloads, so this arbiter
// adds no duplicate input skid RAM. A two-stage sideband path stays lockstep
// with the trace arithmetic pipeline.
module unified_dual_role_h40_trace_region_candidate_v1 #(
    parameter int REGION_TILES = 4,
    parameter int META_WIDTH = 64
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,

    input  wire logic [REGION_TILES-1:0]           i_tile_valid,
    output logic [REGION_TILES-1:0]                o_tile_ready,
    input  wire logic [REGION_TILES*2-1:0]         i_tile_mode_flat,
    input  wire logic [REGION_TILES*10-1:0]        i_tile_lane_mask_flat,
    input  wire logic [REGION_TILES*10-1:0]        i_tile_range_mask_flat,
    input  wire logic [REGION_TILES*10-1:0]        i_tile_spike_flat,
    input  wire logic [REGION_TILES*320-1:0]       i_tile_pre_state_flat,
    input  wire logic [REGION_TILES*320-1:0]       i_tile_trace_old_flat,
    input  wire logic [REGION_TILES*16-1:0]        i_tile_trace_decay_flat,
    input  wire logic [REGION_TILES*META_WIDTH-1:0] i_tile_meta_flat,

    output logic                                    o_record_valid,
    input  wire logic                               i_record_ready,
    output logic [$clog2(REGION_TILES)-1:0]         o_record_tile,
    output logic [1:0]                              o_record_mode,
    output logic [9:0]                              o_record_lane_mask,
    output logic [9:0]                              o_record_range_mask,
    output logic [9:0]                              o_record_spike,
    output logic [319:0]                            o_record_pre_state,
    output logic [319:0]                            o_record_trace_next,
    output logic [META_WIDTH-1:0]                   o_record_meta,
    output logic                                    o_record_mode_error,
    output logic                                    o_idle
);
    localparam int TILE_ID_WIDTH = $clog2(REGION_TILES);

    logic [TILE_ID_WIDTH-1:0] round_robin_q;
    logic selected_valid;
    logic [TILE_ID_WIDTH-1:0] selected_tile;
    logic [1:0] selected_mode;
    logic [9:0] selected_lane_mask;
    logic [9:0] selected_range_mask;
    logic [9:0] selected_spike;
    logic [319:0] selected_pre_state;
    logic [319:0] selected_trace_old;
    logic [15:0] selected_trace_decay;
    logic [META_WIDTH-1:0] selected_meta;
    logic selected_fire;

    logic trace_input_ready;
    logic trace_result_valid;
    logic [319:0] trace_result;
    logic trace_mode_error;

    logic side_product_valid_q;
    logic side_result_valid_q;
    logic side_result_slot_ready;
    logic side_product_slot_ready;
    logic [TILE_ID_WIDTH-1:0] product_tile_q;
    logic [TILE_ID_WIDTH-1:0] result_tile_q;
    logic [1:0] product_mode_q;
    logic [1:0] result_mode_q;
    logic [9:0] product_lane_mask_q;
    logic [9:0] result_lane_mask_q;
    logic [9:0] product_range_mask_q;
    logic [9:0] result_range_mask_q;
    logic [9:0] product_spike_q;
    logic [9:0] result_spike_q;
    logic [319:0] product_pre_state_q;
    logic [319:0] result_pre_state_q;
    logic [META_WIDTH-1:0] product_meta_q;
    logic [META_WIDTH-1:0] result_meta_q;

    always_comb begin
        logic found;
        int unsigned candidate;

        selected_valid = 1'b0;
        selected_tile = round_robin_q;
        selected_mode = '0;
        selected_lane_mask = '0;
        selected_range_mask = '0;
        selected_spike = '0;
        selected_pre_state = '0;
        selected_trace_old = '0;
        selected_trace_decay = '0;
        selected_meta = '0;
        o_tile_ready = '0;
        found = 1'b0;

        for (int offset = 0; offset < REGION_TILES; offset++) begin
            candidate = 32'(round_robin_q);
            candidate = candidate + offset;
            if (candidate >= REGION_TILES)
                candidate = candidate - REGION_TILES;
            if (!found && i_tile_valid[candidate]) begin
                found = 1'b1;
                selected_valid = 1'b1;
                selected_tile = TILE_ID_WIDTH'(candidate);
            end
        end

        if (selected_valid) begin
            selected_mode =
                i_tile_mode_flat[selected_tile*2 +: 2];
            selected_lane_mask =
                i_tile_lane_mask_flat[selected_tile*10 +: 10];
            selected_range_mask =
                i_tile_range_mask_flat[selected_tile*10 +: 10];
            selected_spike =
                i_tile_spike_flat[selected_tile*10 +: 10];
            selected_pre_state =
                i_tile_pre_state_flat[selected_tile*320 +: 320];
            selected_trace_old =
                i_tile_trace_old_flat[selected_tile*320 +: 320];
            selected_trace_decay =
                i_tile_trace_decay_flat[selected_tile*16 +: 16];
            selected_meta =
                i_tile_meta_flat[selected_tile*META_WIDTH +: META_WIDTH];
            o_tile_ready[selected_tile] = trace_input_ready &&
                side_product_slot_ready && !i_abort;
        end
    end

    assign side_result_slot_ready = !side_result_valid_q || i_record_ready;
    assign side_product_slot_ready = !side_product_valid_q ||
        side_result_slot_ready;
    assign selected_fire = selected_valid &&
        o_tile_ready[selected_tile];

    assign o_record_valid = !i_abort && trace_result_valid &&
        side_result_valid_q;
    assign o_record_tile = result_tile_q;
    assign o_record_mode = result_mode_q;
    assign o_record_lane_mask = result_lane_mask_q;
    assign o_record_range_mask = result_range_mask_q;
    assign o_record_spike = result_spike_q;
    assign o_record_pre_state = result_pre_state_q;
    assign o_record_trace_next = trace_result;
    assign o_record_meta = result_meta_q;
    assign o_record_mode_error = trace_mode_error;
    assign o_idle = !selected_valid && !side_product_valid_q &&
        !side_result_valid_q && !trace_result_valid;

    unified_dual_role_trace10_arithmetic_candidate_v1 u_trace_arithmetic (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(selected_valid && side_product_slot_ready),
        .o_ready(trace_input_ready), .i_mode(selected_mode),
        .i_lane_mask(selected_lane_mask), .i_spike(selected_spike),
        .i_trace_old_flat(selected_trace_old),
        .i_trace_decay(selected_trace_decay),
        .o_valid(trace_result_valid), .i_ready(i_record_ready),
        .o_trace_next_flat(trace_result),
        .o_mode_error(trace_mode_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            round_robin_q <= '0;
            side_product_valid_q <= 1'b0;
            side_result_valid_q <= 1'b0;
            product_tile_q <= '0;
            result_tile_q <= '0;
            product_mode_q <= '0;
            result_mode_q <= '0;
            product_lane_mask_q <= '0;
            result_lane_mask_q <= '0;
            product_range_mask_q <= '0;
            result_range_mask_q <= '0;
            product_spike_q <= '0;
            result_spike_q <= '0;
            product_pre_state_q <= '0;
            result_pre_state_q <= '0;
            product_meta_q <= '0;
            result_meta_q <= '0;
        end else begin
            if (side_result_slot_ready) begin
                side_result_valid_q <= side_product_valid_q;
                if (side_product_valid_q) begin
                    result_tile_q <= product_tile_q;
                    result_mode_q <= product_mode_q;
                    result_lane_mask_q <= product_lane_mask_q;
                    result_range_mask_q <= product_range_mask_q;
                    result_spike_q <= product_spike_q;
                    result_pre_state_q <= product_pre_state_q;
                    result_meta_q <= product_meta_q;
                end
            end

            if (side_product_slot_ready) begin
                side_product_valid_q <= selected_fire;
                if (selected_fire) begin
                    product_tile_q <= selected_tile;
                    product_mode_q <= selected_mode;
                    product_lane_mask_q <= selected_lane_mask;
                    product_range_mask_q <= selected_range_mask;
                    product_spike_q <= selected_spike;
                    product_pre_state_q <= selected_pre_state;
                    product_meta_q <= selected_meta;
                end
            end

            if (selected_fire) begin
                if (selected_tile == TILE_ID_WIDTH'(REGION_TILES - 1))
                    round_robin_q <= '0;
                else
                    round_robin_q <= selected_tile + 1'b1;
            end
        end
    end

    initial begin
        if (REGION_TILES != 4 || META_WIDTH != 64)
            $error("trace region candidate requires four tiles and 64-bit metadata");
    end
endmodule

`default_nettype wire
