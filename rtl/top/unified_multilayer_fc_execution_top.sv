`timescale 1ns/1ps
`default_nettype none

// Complete local integration of descriptor control, FC state storage and the
// shared K8 Tile adapter. The one-layer canonical output remains available as
// a migration reference and can be disabled in favor of the shared path.
module unified_multilayer_fc_execution_top #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40,
    parameter integer HIDDEN_MAX = 480,
    parameter bit SHARE_SINGLE_LAYER_FC = 1'b0
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

    output logic                                       o_direct_layer_valid,
    input  wire logic                                  i_direct_layer_ready,
    output logic [383:0]                              o_direct_layer_words,
    input  wire logic                                  i_direct_layer_done,

    input  wire logic                                  i_event_valid,
    output logic                                       o_event_ready,
    input  wire logic [12:0]                           i_event_addr,
    input  wire logic signed [7:0]                     i_event_data,
    input  wire logic                                  i_event_last,
    output logic [TILE_COUNT-1:0]                     o_weight_req_valid,
    input  wire logic [TILE_COUNT-1:0]                i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_weight_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]             i_weight_data_flat,
    input  wire logic signed [15:0]                   i_leak_q15,
    input  wire logic signed [31:0]                   i_threshold,
    input  wire logic [4:0]                            i_current_left_shift,
    input  wire logic                                  i_reset_to_zero,

    output logic                                       o_busy,
    output logic                                       o_sample_done,
    output logic [63:0]                               o_active_versions,
    output logic                                       o_sticky_error,
    output logic [7:0]                                o_error_code
);
    logic compute_valid;
    logic compute_ready;
    logic compute_uses_external_events;
    logic [HIDDEN_MAX-1:0] compute_input_spikes;
    logic [12:0] compute_input_dim;
    logic [9:0] compute_state_dim;
    logic [4:0] compute_active_tiles;
    logic [3:0] compute_layer_index_unused;
    logic [31:0] compute_weight_base;
    logic compute_done;
    logic [HIDDEN_MAX-1:0] compute_output_spikes;
    logic control_busy;
    logic control_sample_done;
    logic control_error;
    logic [7:0] control_error_code;
    logic adapter_busy;
    logic adapter_error;

    assign o_busy = control_busy || adapter_busy;
    assign o_sample_done = control_sample_done;
    assign o_sticky_error = control_error || adapter_error;
    assign o_error_code = control_error ? control_error_code :
        adapter_error ? 8'h32 : 8'h00;

    unified_multilayer_fc_control_top #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .SHARE_SINGLE_LAYER_FC(SHARE_SINGLE_LAYER_FC)
    ) u_control (
        .clk(clk), .rst_n(rst_n), .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready), .i_cfg_profile(i_cfg_profile),
        .i_cfg_table(i_cfg_table), .i_cfg_index(i_cfg_index),
        .i_cfg_data(i_cfg_data), .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready), .i_commit_profile(i_commit_profile),
        .i_start(i_start), .o_start_ready(o_start_ready),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error),
        .o_direct_layer_valid(o_direct_layer_valid),
        .i_direct_layer_ready(i_direct_layer_ready),
        .o_direct_layer_words(o_direct_layer_words),
        .i_direct_layer_done(i_direct_layer_done), .o_compute_valid(compute_valid),
        .i_compute_ready(compute_ready),
        .o_compute_uses_external_events(compute_uses_external_events),
        .o_compute_input_spikes(compute_input_spikes),
        .o_compute_input_dim(compute_input_dim),
        .o_compute_state_dim(compute_state_dim),
        .o_compute_active_tiles(compute_active_tiles),
        .o_compute_layer_index(compute_layer_index_unused),
        .o_compute_weight_base(compute_weight_base), .i_compute_done(compute_done),
        .i_compute_output_spikes(compute_output_spikes), .o_busy(control_busy),
        .o_sample_done(control_sample_done), .o_active_versions(o_active_versions),
        .o_sticky_error(control_error), .o_error_code(control_error_code)
    );

    unified_multilayer_fc_k8_adapter #(
        .TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE),
        .HIDDEN_MAX(HIDDEN_MAX)
    ) u_adapter (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_compute_valid(compute_valid), .o_compute_ready(compute_ready),
        .i_compute_uses_external_events(compute_uses_external_events),
        .i_compute_input_spikes(compute_input_spikes),
        .i_compute_input_dim(compute_input_dim),
        .i_compute_state_dim(compute_state_dim),
        .i_compute_active_tiles(compute_active_tiles),
        .i_compute_weight_base(compute_weight_base), .i_event_valid(i_event_valid),
        .o_event_ready(o_event_ready), .i_event_addr(i_event_addr),
        .i_event_data(i_event_data), .i_event_last(i_event_last),
        .o_compute_done(compute_done), .o_compute_output_spikes(compute_output_spikes),
        .o_weight_req_valid(o_weight_req_valid), .i_weight_req_ready(i_weight_req_ready),
        .o_weight_addr_flat(o_weight_addr_flat), .i_weight_data_flat(i_weight_data_flat),
        .i_leak_q15(i_leak_q15), .i_threshold(i_threshold),
        .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero), .o_busy(adapter_busy),
        .o_sticky_error(adapter_error)
    );
endmodule

`default_nettype wire
