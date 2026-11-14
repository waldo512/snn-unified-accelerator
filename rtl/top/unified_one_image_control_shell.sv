`timescale 1ns/1ps
`default_nettype none

// Control-only integration boundary for the single runtime-configurable image.
// Datapath owners remain separate so SPARSE_FC can connect directly to the
// canonical FC implementation without adding mode logic to its critical path.
module unified_one_image_control_shell (
    input  logic          i_clk,
    input  logic          i_rst_n,
    input  logic          i_cfg_valid,
    output logic          o_cfg_ready,
    input  logic [1:0]    i_cfg_profile,
    input  logic [1:0]    i_cfg_table,
    input  logic [8:0]    i_cfg_index,
    input  logic [31:0]   i_cfg_data,
    input  logic          i_commit_valid,
    output logic          o_commit_ready,
    input  logic [1:0]    i_commit_profile,
    input  logic          i_start,
    output logic          o_start_ready,
    input  logic [15:0]   i_model_version,
    input  logic [15:0]   i_adapt_version,
    input  logic          i_clear_error,

    output logic          o_fc_layer_valid,
    input  logic          i_fc_layer_ready,
    input  logic          i_fc_layer_done,
    output logic          o_recurrent_layer_valid,
    input  logic          i_recurrent_layer_ready,
    input  logic          i_recurrent_layer_done,
    output logic          o_scnn_layer_valid,
    input  logic          i_scnn_layer_ready,
    input  logic          i_scnn_layer_done,
    output logic          o_output_layer_valid,
    input  logic          i_output_layer_ready,
    input  logic          i_output_layer_done,
    input  logic          i_fc_error,
    input  logic          i_recurrent_error,
    input  logic          i_scnn_error,
    input  logic          i_output_error,

    output logic [3:0]    o_layer_index,
    output logic [383:0]  o_layer_words,
    output logic [3:0]    o_inbound_valid,
    output logic [15:0]   o_inbound_sources_flat,
    output logic [15:0]   o_inbound_types_flat,
    output logic          o_busy,
    output logic          o_sample_done,
    output logic [4:0]    o_active_layer_count,
    output logic [63:0]   o_active_versions,
    output logic          o_sticky_error,
    output logic [7:0]    o_error_code
);
    logic control_layer_valid;
    logic control_layer_ready;
    logic control_layer_done;
    logic control_busy;
    logic control_error;
    logic [7:0] control_error_code;
    logic dispatch_busy;
    logic dispatch_error;
    logic dispatch_layer_ready;
    logic [3:0] layer_type;
    logic owner_abort;
    logic layer_pending_q;
    logic [3:0] layer_index_q;
    logic [383:0] layer_words_q;
    logic [3:0] inbound_valid_q;
    logic [15:0] inbound_sources_q;
    logic [15:0] inbound_types_q;
    logic [3:0] control_layer_index;
    logic [383:0] control_layer_words;
    logic [3:0] control_inbound_valid;
    logic [15:0] control_inbound_sources;
    logic [15:0] control_inbound_types;

    assign layer_type = layer_words_q[2*32 +: 4];
    assign owner_abort = i_fc_error || i_recurrent_error || i_scnn_error ||
                         i_output_error;
    assign o_layer_index = layer_index_q;
    assign o_layer_words = layer_words_q;
    assign o_inbound_valid = inbound_valid_q;
    assign o_inbound_sources_flat = inbound_sources_q;
    assign o_inbound_types_flat = inbound_types_q;
    assign o_busy = control_busy || dispatch_busy;
    assign o_sticky_error = control_error || dispatch_error;
    assign o_error_code = control_error ? control_error_code :
                          dispatch_error ? 8'h40 : 8'h00;

    unified_snn_control_plane u_control (
        .clk(i_clk), .rst_n(i_rst_n), .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready), .i_cfg_profile(i_cfg_profile),
        .i_cfg_table(i_cfg_table), .i_cfg_index(i_cfg_index),
        .i_cfg_data(i_cfg_data), .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready), .i_commit_profile(i_commit_profile),
        .i_start(i_start), .o_start_ready(o_start_ready),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error), .i_abort(owner_abort),
        .o_layer_valid(control_layer_valid),
        .i_layer_ready(control_layer_ready),
        .o_layer_index(control_layer_index),
        .o_layer_words(control_layer_words),
        .o_inbound_valid(control_inbound_valid),
        .o_inbound_sources_flat(control_inbound_sources),
        .o_inbound_types_flat(control_inbound_types),
        .i_layer_done(control_layer_done), .o_busy(control_busy),
        .o_sample_done(o_sample_done),
        .o_active_layer_count(o_active_layer_count),
        .o_active_versions(o_active_versions),
        .o_sticky_error(control_error), .o_error_code(control_error_code)
    );

    unified_layer_owner_dispatch u_dispatch (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(owner_abort), .i_layer_valid(layer_pending_q),
        .o_layer_ready(dispatch_layer_ready), .i_layer_type(layer_type),
        .o_fc_valid(o_fc_layer_valid), .i_fc_ready(i_fc_layer_ready),
        .i_fc_done(i_fc_layer_done),
        .o_recurrent_valid(o_recurrent_layer_valid),
        .i_recurrent_ready(i_recurrent_layer_ready),
        .i_recurrent_done(i_recurrent_layer_done),
        .o_scnn_valid(o_scnn_layer_valid),
        .i_scnn_ready(i_scnn_layer_ready), .i_scnn_done(i_scnn_layer_done),
        .o_output_valid(o_output_layer_valid),
        .i_output_ready(i_output_layer_ready),
        .i_output_done(i_output_layer_done),
        .o_layer_done(control_layer_done), .o_busy(dispatch_busy),
        .o_dispatch_error(dispatch_error)
    );

    assign control_layer_ready = !layer_pending_q && !dispatch_busy;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            layer_pending_q <= 1'b0;
            layer_index_q <= '0;
            layer_words_q <= '0;
            inbound_valid_q <= '0;
            inbound_sources_q <= '0;
            inbound_types_q <= '0;
        end else begin
            if (control_layer_valid && control_layer_ready) begin
                layer_pending_q <= 1'b1;
                layer_index_q <= control_layer_index;
                layer_words_q <= control_layer_words;
                inbound_valid_q <= control_inbound_valid;
                inbound_sources_q <= control_inbound_sources;
                inbound_types_q <= control_inbound_types;
            end
            if (layer_pending_q && dispatch_layer_ready)
                layer_pending_q <= 1'b0;
            if (owner_abort)
                layer_pending_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
