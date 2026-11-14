`timescale 1ns/1ps
`default_nettype none

// Integrated graph-control plus recurrent-extension prototype execution.
// SPARSE_FC must bypass this top and use the canonical FC fast path.
// OUTPUT reduction and CONV/POOL execution remain separate integration gates.
module unified_fc_recurrent_control_top #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40
) (
    input  wire logic                                  clk,
    input  wire logic                                  rst_n,
    input  wire logic                                  i_cfg_valid,
    output logic                                       o_cfg_ready,
    input  wire logic [1:0]                            i_cfg_profile,
    input  wire logic [1:0]                            i_cfg_table,
    input  wire logic [8:0]                            i_cfg_index,
    input  wire logic [31:0]                           i_cfg_data,
    input  wire logic                                  i_commit_valid,
    output logic                                       o_commit_ready,
    input  wire logic [1:0]                            i_commit_profile,
    input  wire logic                                  i_start,
    output logic                                       o_start_ready,
    input  wire logic [15:0]                           i_model_version,
    input  wire logic [15:0]                           i_adapt_version,
    input  wire logic                                  i_clear_error,

    input  wire logic                                  i_event_valid,
    output logic                                       o_event_ready,
    input  wire logic [12:0]                           i_event_addr,
    input  wire logic signed [7:0]                     i_event_data,
    input  wire logic                                  i_timestep_last,
    input  wire logic                                  i_sample_last,
    output logic [TILE_COUNT-1:0]                     o_weight_req_valid,
    input  wire logic [TILE_COUNT-1:0]                i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_weight_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]             i_weight_data_flat,
    input  wire logic signed [15:0]                   i_leak_q15,
    input  wire logic signed [31:0]                   i_threshold,
    input  wire logic [4:0]                            i_current_left_shift,
    input  wire logic                                  i_reset_to_zero,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]    o_previous_spikes_flat,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]    o_current_spikes_flat,
    output logic                                       o_busy,
    output logic                                       o_sample_done,
    output logic [63:0]                                o_active_versions,
    output logic                                       o_sticky_error,
    output logic [7:0]                                 o_error_code
);
    logic layer_valid;
    logic layer_ready;
    logic [383:0] layer_words;
    logic layer_done;
    logic [3:0] layer_index_unused;
    logic [3:0] inbound_valid_unused;
    logic [15:0] inbound_sources_unused;
    logic [15:0] inbound_types_unused;
    logic [4:0] active_layer_count_unused;
    logic control_busy;
    logic control_done;
    logic control_error;
    logic [7:0] control_error_code;
    logic engine_busy;
    logic engine_error;
    logic timestep_valid_unused;
    logic timestep_sample_last_unused;

    assign o_busy = control_busy || engine_busy;
    assign o_sample_done = control_done;
    assign o_sticky_error = control_error || engine_error;
    assign o_error_code = control_error ? control_error_code :
        engine_error ? 8'h30 : 8'h00;

    unified_snn_control_plane u_control (
        .clk(clk), .rst_n(rst_n), .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready), .i_cfg_profile(i_cfg_profile),
        .i_cfg_table(i_cfg_table), .i_cfg_index(i_cfg_index),
        .i_cfg_data(i_cfg_data), .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready), .i_commit_profile(i_commit_profile),
        .i_start(i_start), .o_start_ready(o_start_ready),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error), .i_abort(1'b0),
        .o_layer_valid(layer_valid),
        .i_layer_ready(layer_ready), .o_layer_index(layer_index_unused),
        .o_layer_words(layer_words), .o_inbound_valid(inbound_valid_unused),
        .o_inbound_sources_flat(inbound_sources_unused),
        .o_inbound_types_flat(inbound_types_unused), .i_layer_done(layer_done),
        .o_busy(control_busy), .o_sample_done(control_done),
        .o_active_layer_count(active_layer_count_unused),
        .o_active_versions(o_active_versions), .o_sticky_error(control_error),
        .o_error_code(control_error_code)
    );

    unified_fc_recurrent_layer_engine #(
        .TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE)
    ) u_engine (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_layer_valid(layer_valid), .o_layer_ready(layer_ready),
        .i_layer_words(layer_words), .o_layer_done(layer_done),
        .i_event_valid(i_event_valid), .o_event_ready(o_event_ready),
        .i_event_addr(i_event_addr), .i_event_data(i_event_data),
        .i_timestep_last(i_timestep_last), .i_sample_last(i_sample_last),
        .o_weight_req_valid(o_weight_req_valid),
        .i_weight_req_ready(i_weight_req_ready),
        .o_weight_addr_flat(o_weight_addr_flat),
        .i_weight_data_flat(i_weight_data_flat), .i_leak_q15(i_leak_q15),
        .i_threshold(i_threshold), .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero),
        .o_previous_spikes_flat(o_previous_spikes_flat),
        .o_current_spikes_flat(o_current_spikes_flat),
        .o_timestep_valid(timestep_valid_unused), .i_timestep_ready(1'b1),
        .o_timestep_sample_last(timestep_sample_last_unused),
        .o_busy(engine_busy), .o_sticky_error(engine_error)
    );
endmodule

`default_nettype wire
