`timescale 1ns/1ps
`default_nettype none

// Transactional state boundary for SPARSE_FC graphs. A migration parameter
// allows one-layer FC graphs to use this shared path while the canonical path
// remains available as a differential reference.
module unified_multilayer_fc_state_controller #(
    parameter integer MAX_LAYERS = 16,
    parameter integer LAYER_WORDS = 12,
    parameter integer INPUT_MAX = 2312,
    parameter integer HIDDEN_MAX = 480,
    parameter integer OUTPUT_MAX = 32,
    parameter integer NEURONS_PER_TILE = 40,
    parameter bit SHARE_SINGLE_LAYER_FC = 1'b0
) (
    input  wire logic                               clk,
    input  wire logic                               rst_n,
    input  wire logic                               i_clear_error,
    input  wire logic                               i_sample_start,
    input  wire logic [4:0]                         i_layer_count,

    input  wire logic                               i_layer_valid,
    output logic                                    o_layer_ready,
    input  wire logic [3:0]                         i_layer_index,
    input  wire logic [LAYER_WORDS*32-1:0]         i_layer_words,
    input  wire logic [3:0]                         i_inbound_valid,
    input  wire logic [15:0]                        i_inbound_sources_flat,
    input  wire logic [15:0]                        i_inbound_types_flat,
    output logic                                    o_direct_fast_path_required,

    output logic                                    o_compute_valid,
    input  wire logic                               i_compute_ready,
    output logic                                    o_compute_uses_external_events,
    output logic [HIDDEN_MAX-1:0]                  o_compute_input_spikes,
    output logic [12:0]                             o_compute_input_dim,
    output logic [9:0]                              o_compute_state_dim,
    output logic [4:0]                              o_compute_active_tiles,
    output logic [3:0]                              o_compute_layer_index,
    output logic [31:0]                             o_compute_weight_base,
    input  wire logic                               i_compute_done,
    input  wire logic [HIDDEN_MAX-1:0]             i_compute_output_spikes,

    output logic                                    o_layer_done,
    output logic                                    o_busy,
    output logic                                    o_sticky_error
);
    localparam logic [3:0] LAYER_SPARSE_FC = 4'd1;
    localparam logic [3:0] EDGE_CURRENT_LAYER = 4'd0;
    localparam logic [1:0] STATE_IDLE = 2'd0;
    localparam logic [1:0] STATE_DISPATCH = 2'd1;
    localparam logic [1:0] STATE_WAIT = 2'd2;
    localparam logic [12:0] INPUT_MAX_LIMIT = 13'(INPUT_MAX);
    localparam logic [9:0] HIDDEN_MAX_LIMIT = 10'(HIDDEN_MAX);
    localparam logic [5:0] OUTPUT_MAX_LIMIT = 6'(OUTPUT_MAX);
    localparam logic [4:0] MAX_LAYERS_LIMIT = 5'(MAX_LAYERS);
    localparam logic [4:0] MAX_TILES_LIMIT =
        5'((HIDDEN_MAX + NEURONS_PER_TILE - 1) / NEURONS_PER_TILE);

    logic [1:0] state_q;
    logic [HIDDEN_MAX-1:0] layer_state_mem [0:MAX_LAYERS-1];
    logic [MAX_LAYERS-1:0] layer_state_valid_q;
    logic [9:0] layer_state_dim_q [0:MAX_LAYERS-1];
    logic [HIDDEN_MAX-1:0] input_spikes_q;
    logic input_uses_external_q;
    logic [12:0] input_dim_q;
    logic [9:0] state_dim_q;
    logic [4:0] active_tiles_q;
    logic [3:0] layer_index_q;
    logic [31:0] weight_base_q;
    logic [3:0] descriptor_type;
    logic [12:0] descriptor_input_dim;
    logic [9:0] descriptor_state_dim;
    logic [5:0] descriptor_output_dim;
    logic [4:0] descriptor_active_tiles;
    logic [3:0] source_layer;
    logic [3:0] source_type;
    logic one_layer_fc;
    logic [4:0] descriptor_required_tiles;
    logic descriptor_shape_valid;
    logic inbound_valid;
    integer reset_index;

    function automatic [4:0] tiles_for_neurons(input logic [9:0] neurons);
        logic [10:0] rounded_neurons;
        logic [10:0] tile_count;
        begin
            rounded_neurons = {1'b0, neurons} +
                11'(NEURONS_PER_TILE - 1);
            tile_count = rounded_neurons / 11'(NEURONS_PER_TILE);
            tiles_for_neurons = tile_count[4:0];
        end
    endfunction

    always_comb begin
        descriptor_type = i_layer_words[2*32 +: 4];
        descriptor_input_dim = i_layer_words[3*32 +: 13];
        descriptor_state_dim = i_layer_words[3*32 + 13 +: 10];
        descriptor_output_dim = i_layer_words[3*32 + 23 +: 6];
        descriptor_active_tiles = i_layer_words[4*32 +: 5];
        source_layer = i_inbound_sources_flat[3:0];
        source_type = i_inbound_types_flat[3:0];
        one_layer_fc = (i_layer_count == 5'd1) &&
            (descriptor_type == LAYER_SPARSE_FC);
        descriptor_required_tiles = tiles_for_neurons(descriptor_state_dim);
        descriptor_shape_valid =
            (descriptor_type == LAYER_SPARSE_FC) &&
            (descriptor_input_dim != 0) &&
            (descriptor_input_dim <= INPUT_MAX_LIMIT) &&
            (descriptor_state_dim != 0) &&
            (descriptor_state_dim <= HIDDEN_MAX_LIMIT) &&
            (descriptor_output_dim != 0) &&
            (descriptor_output_dim <= OUTPUT_MAX_LIMIT) &&
            (descriptor_active_tiles != 0) &&
            (descriptor_active_tiles <= MAX_TILES_LIMIT) &&
            (descriptor_active_tiles == descriptor_required_tiles);
        inbound_valid = (i_inbound_valid == 4'b0001) &&
            (source_type == EDGE_CURRENT_LAYER) &&
            (source_layer < i_layer_index) &&
            layer_state_valid_q[source_layer] &&
            (descriptor_input_dim ==
                {3'b000, layer_state_dim_q[source_layer]});

        o_direct_fast_path_required = i_layer_valid && one_layer_fc &&
            !SHARE_SINGLE_LAYER_FC;
        o_layer_ready = (state_q == STATE_IDLE) &&
            (!one_layer_fc || SHARE_SINGLE_LAYER_FC);
        o_compute_valid = state_q == STATE_DISPATCH;
        o_compute_uses_external_events = input_uses_external_q;
        o_compute_input_spikes = input_spikes_q;
        o_compute_input_dim = input_dim_q;
        o_compute_state_dim = state_dim_q;
        o_compute_active_tiles = active_tiles_q;
        o_compute_layer_index = layer_index_q;
        o_compute_weight_base = weight_base_q;
        o_busy = state_q != STATE_IDLE;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            layer_state_valid_q <= '0;
            input_spikes_q <= '0;
            input_uses_external_q <= 1'b0;
            input_dim_q <= '0;
            state_dim_q <= '0;
            active_tiles_q <= '0;
            layer_index_q <= '0;
            weight_base_q <= '0;
            o_layer_done <= 1'b0;
            o_sticky_error <= 1'b0;
            for (reset_index = 0; reset_index < MAX_LAYERS;
                 reset_index = reset_index + 1) begin
                layer_state_mem[reset_index] <= '0;
                layer_state_dim_q[reset_index] <= '0;
            end
        end else begin
            o_layer_done <= 1'b0;
            if (i_clear_error)
                o_sticky_error <= 1'b0;

            if (i_sample_start) begin
                if (state_q != STATE_IDLE) begin
                    o_sticky_error <= 1'b1;
                end else begin
                    layer_state_valid_q <= '0;
                end
            end

            if ((i_compute_done && (state_q != STATE_WAIT)) ||
                (((i_layer_count == 0) ||
                  (i_layer_count > MAX_LAYERS_LIMIT)) &&
                 (i_sample_start || i_layer_valid)))
                o_sticky_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (i_layer_valid &&
                        (!one_layer_fc || SHARE_SINGLE_LAYER_FC)) begin
                        if (!descriptor_shape_valid ||
                            ({1'b0, i_layer_index} >= i_layer_count) ||
                            ((i_layer_index == 0) ?
                             (i_inbound_valid != 0) : !inbound_valid)) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            layer_index_q <= i_layer_index;
                            input_dim_q <= descriptor_input_dim;
                            state_dim_q <= descriptor_state_dim;
                            active_tiles_q <= descriptor_active_tiles;
                            weight_base_q <= i_layer_words[7*32 +: 32];
                            input_uses_external_q <= i_layer_index == 0;
                            input_spikes_q <= (i_layer_index == 0) ? '0 :
                                layer_state_mem[source_layer];
                            state_q <= STATE_DISPATCH;
                        end
                    end
                end
                STATE_DISPATCH: begin
                    if (i_compute_ready)
                        state_q <= STATE_WAIT;
                end
                STATE_WAIT: begin
                    if (i_compute_done) begin
                        layer_state_mem[layer_index_q] <= i_compute_output_spikes;
                        layer_state_dim_q[layer_index_q] <= state_dim_q;
                        layer_state_valid_q[layer_index_q] <= 1'b1;
                        o_layer_done <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
