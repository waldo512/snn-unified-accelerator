`timescale 1ns/1ps
`default_nettype none

// Descriptor/route integration for FC state transactions. During migration,
// one-layer FC can either use the external canonical reference or the shared
// compute path selected by SHARE_SINGLE_LAYER_FC.
module unified_multilayer_fc_control_top #(
    parameter integer HIDDEN_MAX = 480,
    parameter bit SHARE_SINGLE_LAYER_FC = 1'b0
) (
    input  wire logic                    clk,
    input  wire logic                    rst_n,
    input  wire logic                    i_cfg_valid,
    output logic                         o_cfg_ready,
    input  wire logic [1:0]              i_cfg_profile,
    input  wire logic [1:0]              i_cfg_table,
    input  wire logic [8:0]              i_cfg_index,
    input  wire logic [31:0]             i_cfg_data,
    input  wire logic                    i_commit_valid,
    output logic                         o_commit_ready,
    input  wire logic [1:0]              i_commit_profile,
    input  wire logic                    i_start,
    output logic                         o_start_ready,
    input  wire logic [15:0]             i_model_version,
    input  wire logic [15:0]             i_adapt_version,
    input  wire logic                    i_clear_error,

    output logic                         o_direct_layer_valid,
    input  wire logic                    i_direct_layer_ready,
    output logic [383:0]                 o_direct_layer_words,
    input  wire logic                    i_direct_layer_done,

    output logic                         o_compute_valid,
    input  wire logic                    i_compute_ready,
    output logic                         o_compute_uses_external_events,
    output logic [HIDDEN_MAX-1:0]       o_compute_input_spikes,
    output logic [12:0]                  o_compute_input_dim,
    output logic [9:0]                   o_compute_state_dim,
    output logic [4:0]                   o_compute_active_tiles,
    output logic [3:0]                   o_compute_layer_index,
    output logic [31:0]                  o_compute_weight_base,
    input  wire logic                    i_compute_done,
    input  wire logic [HIDDEN_MAX-1:0]  i_compute_output_spikes,

    output logic                         o_busy,
    output logic                         o_sample_done,
    output logic [63:0]                  o_active_versions,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    logic layer_valid;
    logic layer_ready;
    logic [3:0] layer_index;
    logic [383:0] layer_words;
    logic [3:0] inbound_valid;
    logic [15:0] inbound_sources;
    logic [15:0] inbound_types;
    logic layer_done;
    logic [4:0] active_layer_count;
    logic control_busy;
    logic control_done;
    logic control_error;
    logic [7:0] control_error_code;
    logic controller_layer_ready;
    logic controller_layer_done;
    logic controller_busy;
    logic controller_error;
    logic single_layer_fc;
    logic direct_single_layer_fc;

    assign single_layer_fc = (active_layer_count == 5'd1) &&
        (layer_words[2*32 +: 4] == 4'd1);
    assign direct_single_layer_fc = single_layer_fc &&
        !SHARE_SINGLE_LAYER_FC;
    assign o_direct_layer_valid = layer_valid && direct_single_layer_fc;
    assign o_direct_layer_words = layer_words;
    assign layer_ready = direct_single_layer_fc ? i_direct_layer_ready :
        controller_layer_ready;
    assign layer_done = direct_single_layer_fc ? i_direct_layer_done :
        controller_layer_done;
    assign o_busy = control_busy || controller_busy;
    assign o_sample_done = control_done;
    assign o_sticky_error = control_error || controller_error;
    assign o_error_code = control_error ? control_error_code :
        controller_error ? 8'h31 : 8'h00;

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
        .i_layer_ready(layer_ready), .o_layer_index(layer_index),
        .o_layer_words(layer_words), .o_inbound_valid(inbound_valid),
        .o_inbound_sources_flat(inbound_sources),
        .o_inbound_types_flat(inbound_types), .i_layer_done(layer_done),
        .o_busy(control_busy), .o_sample_done(control_done),
        .o_active_layer_count(active_layer_count),
        .o_active_versions(o_active_versions), .o_sticky_error(control_error),
        .o_error_code(control_error_code)
    );

    unified_multilayer_fc_state_controller #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .SHARE_SINGLE_LAYER_FC(SHARE_SINGLE_LAYER_FC)
    ) u_state_controller (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_sample_start(i_start && o_start_ready &&
            ((active_layer_count > 1) || SHARE_SINGLE_LAYER_FC)),
        .i_layer_count(active_layer_count),
        .i_layer_valid(layer_valid && !direct_single_layer_fc),
        .o_layer_ready(controller_layer_ready), .i_layer_index(layer_index),
        .i_layer_words(layer_words), .i_inbound_valid(inbound_valid),
        .i_inbound_sources_flat(inbound_sources), .i_inbound_types_flat(inbound_types),
        .o_direct_fast_path_required(), .o_compute_valid(o_compute_valid),
        .i_compute_ready(i_compute_ready),
        .o_compute_uses_external_events(o_compute_uses_external_events),
        .o_compute_input_spikes(o_compute_input_spikes),
        .o_compute_input_dim(o_compute_input_dim),
        .o_compute_state_dim(o_compute_state_dim),
        .o_compute_active_tiles(o_compute_active_tiles),
        .o_compute_layer_index(o_compute_layer_index),
        .o_compute_weight_base(o_compute_weight_base),
        .i_compute_done(i_compute_done),
        .i_compute_output_spikes(i_compute_output_spikes),
        .o_layer_done(controller_layer_done), .o_busy(controller_busy),
        .o_sticky_error(controller_error)
    );
endmodule

`default_nettype wire
