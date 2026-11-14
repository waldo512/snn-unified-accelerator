`timescale 1ns/1ps
`default_nettype none

module unified_snn_control_plane (
    input  wire logic         clk,
    input  wire logic         rst_n,
    input  wire logic         i_cfg_valid,
    output logic              o_cfg_ready,
    input  wire logic [1:0]   i_cfg_profile,
    input  wire logic [1:0]   i_cfg_table,
    input  wire logic [8:0]   i_cfg_index,
    input  wire logic [31:0]  i_cfg_data,
    input  wire logic         i_commit_valid,
    output logic              o_commit_ready,
    input  wire logic [1:0]   i_commit_profile,
    input  wire logic         i_start,
    output logic              o_start_ready,
    input  wire logic [15:0]  i_model_version,
    input  wire logic [15:0]  i_adapt_version,
    input  wire logic         i_clear_error,
    input  wire logic         i_abort,

    output logic              o_layer_valid,
    input  wire logic         i_layer_ready,
    output logic [3:0]        o_layer_index,
    output logic [383:0]      o_layer_words,
    output logic [3:0]        o_inbound_valid,
    output logic [15:0]       o_inbound_sources_flat,
    output logic [15:0]       o_inbound_types_flat,
    input  wire logic         i_layer_done,

    output logic              o_busy,
    output logic              o_sample_done,
    output logic [4:0]        o_active_layer_count,
    output logic [63:0]       o_active_versions,
    output logic              o_sticky_error,
    output logic [7:0]        o_error_code
);
    logic [3:0] layer_read_index;
    logic [5:0] route_read_index;
    logic [383:0] active_layer_words;
    logic [63:0] active_route_words;
    logic [1:0] active_profile;
    logic [4:0] active_layer_count;
    logic [6:0] active_route_count;
    logic [15:0] profile_version;
    logic [15:0] graph_version;
    logic commit_pending;
    logic commit_pulse;
    logic config_error;
    logic [7:0] config_error_code;
    logic sequencer_busy;
    logic sequencer_start_ready;
    logic sequencer_done;
    logic sample_done_q;
    logic sequence_error;
    logic sample_active;
    logic version_error;
    logic global_idle;

    assign global_idle = !sequencer_busy && !sample_active;
    assign o_start_ready = global_idle && !commit_pending &&
        !config_error && !sequence_error && !version_error &&
        (active_layer_count != 0) && sequencer_start_ready;
    assign o_busy = sequencer_busy || sample_active;
    assign o_sample_done = sample_done_q;
    assign o_active_layer_count = active_layer_count;
    assign o_sticky_error = config_error || sequence_error || version_error;
    assign o_error_code = config_error ? config_error_code :
        sequence_error ? 8'h20 : version_error ? 8'h21 : 8'h00;

    unified_graph_config u_config (
        .clk(clk), .rst_n(rst_n),
        .i_cfg_valid(i_cfg_valid), .o_cfg_ready(o_cfg_ready),
        .i_cfg_profile(i_cfg_profile), .i_cfg_table(i_cfg_table),
        .i_cfg_index(i_cfg_index), .i_cfg_data(i_cfg_data),
        .i_commit_valid(i_commit_valid), .o_commit_ready(o_commit_ready),
        .i_commit_profile(i_commit_profile), .i_global_idle(global_idle),
        .i_clear_error(i_clear_error),
        .i_active_layer_read_index(layer_read_index),
        .o_active_layer_words(active_layer_words),
        .i_active_route_read_index(route_read_index),
        .o_active_route_words(active_route_words),
        .o_active_profile(active_profile),
        .o_active_layer_count(active_layer_count),
        .o_active_route_count(active_route_count),
        .o_profile_version(profile_version), .o_graph_version(graph_version),
        .o_commit_pending(commit_pending), .o_commit_pulse(commit_pulse),
        .o_config_error(config_error), .o_error_code(config_error_code)
    );

    unified_graph_sequencer u_sequencer (
        .clk(clk), .rst_n(rst_n), .i_start(i_start && o_start_ready),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .o_start_ready(sequencer_start_ready),
        .i_layer_count(active_layer_count),
        .i_route_count(active_route_count), .o_layer_read_index(layer_read_index),
        .i_layer_words(active_layer_words), .o_route_read_index(route_read_index),
        .i_route_words(active_route_words), .o_layer_valid(o_layer_valid),
        .i_layer_ready(i_layer_ready), .o_layer_index(o_layer_index),
        .o_layer_words(o_layer_words), .o_inbound_valid(o_inbound_valid),
        .o_inbound_sources_flat(o_inbound_sources_flat),
        .o_inbound_types_flat(o_inbound_types_flat),
        .i_layer_done(i_layer_done), .o_busy(sequencer_busy),
        .o_done(sequencer_done), .o_sequence_error(sequence_error)
    );

    unified_sample_version_guard u_version_guard (
        .clk(clk), .rst_n(rst_n),
        .i_sample_start(i_start && o_start_ready),
        .i_sample_done(sequencer_done || i_abort),
        .i_profile_version(profile_version), .i_graph_version(graph_version),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error), .o_sample_active(sample_active),
        .o_locked_versions(o_active_versions), .o_version_error(version_error)
    );

    // The version guard observes sequencer_done on the following clock edge.
    // Publish sample_done on that same edge so done never overlaps o_busy.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            sample_done_q <= 1'b0;
        else
            sample_done_q <= sequencer_done;
    end
endmodule

`default_nettype wire
