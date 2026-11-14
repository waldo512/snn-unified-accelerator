`timescale 1ns/1ps
`default_nettype none

// Direct descriptor-to-compute integration for a RECURRENT_FC layer. The
// graph-control wrapper will select this extension path; SPARSE_FC continues
// to use the independent canonical FC fast path.
module unified_recurrent_compute_output_top #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40
) (
    input  wire logic                                      clk,
    input  wire logic                                      rst_n,
    input  wire logic                                      i_clear_error,
    input  wire logic                                      i_layer_valid,
    output logic                                           o_layer_ready,
    input  wire logic [383:0]                              i_layer_words,
    output logic                                           o_layer_done,
    input  wire logic                                      i_event_valid,
    output logic                                           o_event_ready,
    input  wire logic [12:0]                               i_event_addr,
    input  wire logic signed [7:0]                         i_event_data,
    input  wire logic                                      i_timestep_last,
    input  wire logic                                      i_sample_last,
    output logic [TILE_COUNT-1:0]                          o_weight_req_valid,
    input  wire logic [TILE_COUNT-1:0]                     i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                       o_weight_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]                  i_weight_data_flat,
    input  wire logic [31:0]                               i_w2_base,
    input  wire logic [5:0]                                i_w2_row_stride,
    output logic [TILE_COUNT-1:0]                          o_w2_req_valid,
    input  wire logic [TILE_COUNT-1:0]                     i_w2_req_ready,
    output logic [TILE_COUNT*32-1:0]                       o_w2_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]                  i_w2_data_flat,
    input  wire logic signed [15:0]                        i_leak_q15,
    input  wire logic signed [31:0]                        i_threshold,
    input  wire logic [4:0]                                i_current_left_shift,
    input  wire logic                                      i_reset_to_zero,
    input  wire logic [15:0]                               i_version,
    output logic                                           o_result_valid,
    input  wire logic                                      i_result_ready,
    output logic [5:0]                                     o_result_class,
    output logic signed [31:0]                            o_result_score,
    output logic [32*32-1:0]                              o_result_logits_flat,
    output logic                                           o_busy,
    output logic                                           o_sticky_error
);
    logic [TILE_COUNT*NEURONS_PER_TILE-1:0] previous_spikes_unused;
    logic [TILE_COUNT*NEURONS_PER_TILE-1:0] current_spikes;
    logic timestep_valid;
    logic timestep_ready;
    logic timestep_sample_last;
    logic engine_busy;
    logic engine_error;
    logic bridge_busy;
    logic bridge_error;
    logic [5:0] descriptor_output_count;
    logic [4:0] descriptor_active_tiles;

    always_comb begin
        descriptor_output_count = i_layer_words[3*32 + 23 +: 6];
        descriptor_active_tiles = i_layer_words[4*32 +: 5];
        o_busy = engine_busy || bridge_busy;
        o_sticky_error = engine_error || bridge_error;
    end

    unified_fc_recurrent_layer_engine #(.TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE)) u_engine (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_layer_valid(i_layer_valid), .o_layer_ready(o_layer_ready), .i_layer_words(i_layer_words), .o_layer_done(o_layer_done),
        .i_event_valid(i_event_valid), .o_event_ready(o_event_ready), .i_event_addr(i_event_addr),
        .i_event_data(i_event_data), .i_timestep_last(i_timestep_last), .i_sample_last(i_sample_last),
        .o_weight_req_valid(o_weight_req_valid), .i_weight_req_ready(i_weight_req_ready),
        .o_weight_addr_flat(o_weight_addr_flat), .i_weight_data_flat(i_weight_data_flat),
        .i_leak_q15(i_leak_q15), .i_threshold(i_threshold), .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero), .o_previous_spikes_flat(previous_spikes_unused),
        .o_current_spikes_flat(current_spikes), .o_timestep_valid(timestep_valid),
        .i_timestep_ready(timestep_ready), .o_timestep_sample_last(timestep_sample_last),
        .o_busy(engine_busy), .o_sticky_error(engine_error)
    );
    unified_recurrent_output_bridge #(.TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE)) u_bridge (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_abort(1'b0), .i_timestep_valid(timestep_valid),
        .o_timestep_ready(timestep_ready), .i_timestep_sample_last(timestep_sample_last),
        .i_spikes_flat(current_spikes), .i_active_tiles(descriptor_active_tiles[3:0]),
        .i_output_count(descriptor_output_count), .i_w2_row_stride(i_w2_row_stride), .i_w2_base(i_w2_base),
        .i_version(i_version), .o_w2_req_valid(o_w2_req_valid), .i_w2_req_ready(i_w2_req_ready),
        .o_w2_addr_flat(o_w2_addr_flat), .i_w2_data_flat(i_w2_data_flat),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready), .o_result_class(o_result_class),
        .o_result_score(o_result_score), .o_result_logits_flat(o_result_logits_flat),
        .o_busy(bridge_busy), .o_sticky_error(bridge_error)
    );
endmodule

`default_nettype wire
