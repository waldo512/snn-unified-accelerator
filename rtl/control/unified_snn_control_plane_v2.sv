`timescale 1ns/1ps
`default_nettype none

// Dynamic-graph control plane for the final binding. i_datapath_idle excludes
// configuration activation itself, allowing a queued commit to begin after all
// sample/result/memory drains without creating a global-idle dependency loop.
module unified_snn_control_plane_v2 #(
    parameter int PROFILE_COUNT = 4,
    parameter int MAX_LAYERS = 16,
    parameter int LAYER_WORDS = 12,
    parameter int MAX_ROUTES = 64
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_cfg_valid,
    output logic                         o_cfg_ready,
    input  logic [1:0]                   i_cfg_profile,
    input  logic [1:0]                   i_cfg_table,
    input  logic [8:0]                   i_cfg_index,
    input  logic [31:0]                  i_cfg_data,
    input  logic                         i_commit_valid,
    output logic                         o_commit_ready,
    input  logic [1:0]                   i_commit_profile,
    input  logic                         i_datapath_idle,
    input  logic                         i_start,
    output logic                         o_start_ready,
    input  logic [6:0]                   i_profile_code,
    input  logic [15:0]                  i_model_version,
    input  logic [15:0]                  i_adapt_version,
    input  logic                         i_clear_error,
    input  logic                         i_abort,

    output logic                         o_layer_valid,
    input  logic                         i_layer_ready,
    output logic [3:0]                   o_layer_index,
    output logic [LAYER_WORDS*32-1:0]    o_layer_words,
    output logic [3:0]                   o_inbound_valid,
    output logic [15:0]                  o_inbound_sources_flat,
    output logic [15:0]                  o_inbound_types_flat,
    input  logic                         i_layer_done,

    output logic [1:0]                   o_active_profile,
    output logic [6:0]                   o_active_profile_code,
    output logic [4:0]                   o_active_layer_count,
    output logic [6:0]                   o_active_route_count,
    output logic [63:0]                  o_active_versions,
    output logic                         o_profile_code_legal,
    output logic                         o_sequence_busy,
    output logic                         o_sample_active,
    output logic                         o_sample_done,
    output logic                         o_commit_pending,
    output logic                         o_commit_pulse,
    output logic                         o_config_copy_busy,
    output logic                         o_control_idle,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [6:0] PROFILE_FC_H160 = 7'h01;
    localparam logic [6:0] PROFILE_FC_H480 = 7'h02;
    localparam logic [6:0] PROFILE_SRNN_H480 = 7'h10;
    localparam logic [6:0] PROFILE_SCNN_VGG9 = 7'h20;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_START_NOT_READY = 8'h30;

    logic [1:0] selected_slot;
    logic profile_matches_active;
    logic datapath_and_sequence_idle;
    logic start_fire;
    logic [3:0] layer_read_index;
    logic [5:0] route_read_index;
    logic [LAYER_WORDS*32-1:0] active_layer_words;
    logic [63:0] active_route_words;
    logic [15:0] profile_version;
    logic [15:0] graph_version;
    logic config_error;
    logic [7:0] config_error_code;
    logic sequencer_start_ready;
    logic sequencer_done;
    logic sequence_error;
    logic version_error;
    logic sample_done_q;
    logic local_error_q;
    logic [7:0] local_error_code_q;

    always_comb begin
        o_profile_code_legal = 1'b1;
        selected_slot = 2'd0;
        case (i_profile_code)
            PROFILE_FC_H160: selected_slot = 2'd0;
            PROFILE_FC_H480: selected_slot = 2'd1;
            PROFILE_SRNN_H480: selected_slot = 2'd2;
            PROFILE_SCNN_VGG9: selected_slot = 2'd3;
            default: begin
                selected_slot = 2'd0;
                o_profile_code_legal = 1'b0;
            end
        endcase
        profile_matches_active = o_profile_code_legal &&
                                 (selected_slot == o_active_profile) &&
                                 (o_active_layer_count != 0);
        datapath_and_sequence_idle = i_datapath_idle &&
                                     !o_sequence_busy && !o_sample_active;
        o_start_ready = datapath_and_sequence_idle && !o_commit_pending &&
                        sequencer_start_ready && profile_matches_active &&
                        !config_error && !sequence_error && !version_error &&
                        !local_error_q;
        start_fire = i_start && o_start_ready;
        o_config_copy_busy = o_commit_pending;
        o_control_idle = datapath_and_sequence_idle && !o_commit_pending;
        o_sticky_error = config_error || sequence_error || version_error ||
                         local_error_q;
        o_error_code = config_error ? config_error_code :
                       sequence_error ? 8'h20 :
                       version_error ? 8'h21 :
                       local_error_q ? local_error_code_q : ERROR_NONE;
        o_sample_done = sample_done_q;
    end

    unified_graph_config #(
        .PROFILE_COUNT(PROFILE_COUNT), .MAX_LAYERS(MAX_LAYERS),
        .LAYER_WORDS(LAYER_WORDS), .MAX_ROUTES(MAX_ROUTES),
        .INPUT_MAX(2312), .HIDDEN_MAX(480), .OUTPUT_MAX(32),
        .NEURONS_PER_TILE(40)
    ) u_config (
        .clk(i_clk), .rst_n(i_rst_n),
        .i_cfg_valid(i_cfg_valid), .o_cfg_ready(o_cfg_ready),
        .i_cfg_profile(i_cfg_profile), .i_cfg_table(i_cfg_table),
        .i_cfg_index(i_cfg_index), .i_cfg_data(i_cfg_data),
        .i_commit_valid(i_commit_valid), .o_commit_ready(o_commit_ready),
        .i_commit_profile(i_commit_profile),
        .i_global_idle(datapath_and_sequence_idle),
        .i_clear_error(i_clear_error && i_datapath_idle),
        .i_active_layer_read_index(layer_read_index),
        .o_active_layer_words(active_layer_words),
        .i_active_route_read_index(route_read_index),
        .o_active_route_words(active_route_words),
        .o_active_profile(o_active_profile),
        .o_active_layer_count(o_active_layer_count),
        .o_active_route_count(o_active_route_count),
        .o_profile_version(profile_version),
        .o_graph_version(graph_version),
        .o_commit_pending(o_commit_pending),
        .o_commit_pulse(o_commit_pulse),
        .o_config_error(config_error), .o_error_code(config_error_code)
    );

    unified_graph_sequencer #(
        .LAYER_WORDS(LAYER_WORDS)
    ) u_sequencer (
        .clk(i_clk), .rst_n(i_rst_n), .i_start(start_fire),
        // The system-level control plane already admits clear only at the
        // recovery boundary. Do not re-gate the one-cycle clear pulse with a
        // changing datapath-idle view or the sequencer can retain abort error.
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .o_start_ready(sequencer_start_ready),
        .i_layer_count(o_active_layer_count),
        .i_route_count(o_active_route_count),
        .o_layer_read_index(layer_read_index),
        .i_layer_words(active_layer_words),
        .o_route_read_index(route_read_index),
        .i_route_words(active_route_words),
        .o_layer_valid(o_layer_valid), .i_layer_ready(i_layer_ready),
        .o_layer_index(o_layer_index), .o_layer_words(o_layer_words),
        .o_inbound_valid(o_inbound_valid),
        .o_inbound_sources_flat(o_inbound_sources_flat),
        .o_inbound_types_flat(o_inbound_types_flat),
        .i_layer_done(i_layer_done), .o_busy(o_sequence_busy),
        .o_done(sequencer_done), .o_sequence_error(sequence_error)
    );

    unified_sample_version_guard u_version_guard (
        .clk(i_clk), .rst_n(i_rst_n),
        .i_sample_start(start_fire),
        .i_sample_done(sequencer_done || i_abort),
        .i_profile_version(profile_version),
        .i_graph_version(graph_version),
        .i_model_version(i_model_version),
        .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error && i_datapath_idle),
        .o_sample_active(o_sample_active),
        .o_locked_versions(o_active_versions),
        .o_version_error(version_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            sample_done_q <= 1'b0;
            o_active_profile_code <= '0;
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else begin
            sample_done_q <= sequencer_done;
            if (i_clear_error && i_datapath_idle) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end
            if (start_fire)
                o_active_profile_code <= i_profile_code;
            else if (i_start && !o_start_ready) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_START_NOT_READY;
            end
        end
    end
endmodule

`default_nettype wire
