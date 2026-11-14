`timescale 1ns/1ps
`default_nettype none

// Admits graph-client Conv/classifier contexts onto one shared H40 cache.
// The classifier occupies one resident pair. Conv waves use the other pair
// and are forwarded to the selected-layer parameter plane for refill or an
// exact state-only residency check.
module unified_scnn_selected_layers_context_bridge_v1 #(
    parameter int PARAM_ADDR_WIDTH = 15,
    parameter int VERSION_WIDTH = 16,
    parameter int WORK_ADDR_WIDTH = 9,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int INPUT_CHANNEL_COUNT_WIDTH =
        $clog2(MAX_INPUT_CHANNELS + 1),
    parameter int CLASS_COUNT_WIDTH = 4
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_context_load_valid,
    output logic                                   o_context_load_ready,
    input  wire logic                              i_context_load_weights,
    input  wire logic [9:0]                        i_context_output_channel_base,
    input  wire logic [9:0]                        i_context_output_channel_count,
    input  wire logic [31:0]                       i_context_spatial_base,
    input  wire logic [5:0]                        i_context_spatial_count,
    input  wire logic                              i_context_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]         i_context_work_base_addr,

    input  wire logic                              i_context_release_valid,
    output logic                                   o_context_release_ready,
    input  wire logic [9:0]                        i_release_output_channel_base,
    input  wire logic [9:0]                        i_release_output_channel_count,
    input  wire logic [31:0]                       i_release_spatial_base,
    input  wire logic [5:0]                        i_release_spatial_count,
    input  wire logic                              i_release_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]         i_release_work_base_addr,

    input  wire logic [VERSION_WIDTH-1:0]           i_graph_version,
    input  wire logic                              i_classifier_cache_valid,
    input  wire logic                              i_classifier_pair,
    input  wire logic [VERSION_WIDTH-1:0]           i_classifier_version,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]       i_classifier_class_count,
    input  wire logic [WORK_ADDR_WIDTH-1:0]         i_classifier_work_base_addr,
    input  wire logic [INPUT_CHANNEL_COUNT_WIDTH-1:0]
                                                    i_conv_input_channel_count,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]        i_conv_parameter_base,

    output logic                                   o_conv_context_valid,
    input  wire logic                              i_conv_context_ready,
    output logic                                   o_conv_context_load_weights,
    output logic [9:0]                             o_conv_output_channel_base,
    output logic [4:0]                             o_conv_output_channel_count,
    output logic [INPUT_CHANNEL_COUNT_WIDTH-1:0]
                                                    o_conv_input_channel_count,
    output logic                                   o_conv_context_pair,
    output logic [PARAM_ADDR_WIDTH-1:0]             o_conv_parameter_base,
    output logic [VERSION_WIDTH-1:0]                o_conv_context_version,

    output logic                                   o_context_active,
    output logic                                   o_context_is_classifier,
    output logic                                   o_sticky_error
);
    logic active_q;
    logic active_classifier_q;
    logic [9:0] active_output_base_q;
    logic [9:0] active_output_count_q;
    logic [31:0] active_spatial_base_q;
    logic [5:0] active_spatial_count_q;
    logic active_pair_q;
    logic [WORK_ADDR_WIDTH-1:0] active_work_base_q;

    logic load_is_classifier;
    logic classifier_load_legal;
    logic conv_load_legal;
    logic load_legal;
    logic load_fire;
    logic release_matches;
    logic release_fire;

    assign load_is_classifier =
        i_context_active_pair == i_classifier_pair;
    assign classifier_load_legal = i_classifier_cache_valid &&
        i_graph_version == i_classifier_version &&
        i_context_output_channel_base == 10'd0 &&
        i_context_output_channel_count == 10'(i_classifier_class_count) &&
        i_context_spatial_base == 32'd0 &&
        i_context_spatial_count == 6'd1 &&
        i_context_work_base_addr == i_classifier_work_base_addr;
    assign conv_load_legal = i_classifier_cache_valid &&
        i_graph_version == i_classifier_version &&
        i_context_active_pair != i_classifier_pair &&
        i_conv_input_channel_count != 0 &&
        i_context_output_channel_count != 0 &&
        i_context_output_channel_count <= 10'd16 &&
        i_context_spatial_count != 0 &&
        i_context_spatial_count <= 6'd40;
    assign load_legal = load_is_classifier ?
        classifier_load_legal : conv_load_legal;

    assign o_conv_context_valid = i_context_load_valid && !active_q &&
        !load_is_classifier && conv_load_legal;
    assign o_conv_context_load_weights = i_context_load_weights;
    assign o_conv_output_channel_base = i_context_output_channel_base;
    assign o_conv_output_channel_count =
        i_context_output_channel_count[4:0];
    assign o_conv_input_channel_count = i_conv_input_channel_count;
    assign o_conv_context_pair = i_context_active_pair;
    assign o_conv_parameter_base = i_conv_parameter_base;
    assign o_conv_context_version = i_graph_version;

    always_comb begin
        o_context_load_ready = 1'b0;
        if (i_context_load_valid && !active_q && load_legal) begin
            if (load_is_classifier)
                o_context_load_ready = 1'b1;
            else
                o_context_load_ready = i_conv_context_ready;
        end
    end
    assign load_fire = i_context_load_valid && o_context_load_ready;

    assign release_matches = active_q &&
        i_release_output_channel_base == active_output_base_q &&
        i_release_output_channel_count == active_output_count_q &&
        i_release_spatial_base == active_spatial_base_q &&
        i_release_spatial_count == active_spatial_count_q &&
        i_release_active_pair == active_pair_q &&
        i_release_work_base_addr == active_work_base_q;
    assign o_context_release_ready = i_context_release_valid &&
        release_matches;
    assign release_fire = i_context_release_valid &&
        o_context_release_ready;

    assign o_context_active = active_q;
    assign o_context_is_classifier = active_classifier_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            active_classifier_q <= 1'b0;
            active_output_base_q <= '0;
            active_output_count_q <= '0;
            active_spatial_base_q <= '0;
            active_spatial_count_q <= '0;
            active_pair_q <= 1'b0;
            active_work_base_q <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            active_q <= 1'b0;
            active_classifier_q <= 1'b0;
        end else begin
            if (i_clear_error && !active_q && !i_context_load_valid &&
                    !i_context_release_valid)
                o_sticky_error <= 1'b0;

            if (i_context_load_valid && (active_q || !load_legal))
                o_sticky_error <= 1'b1;
            if (i_context_release_valid && !release_matches)
                o_sticky_error <= 1'b1;

            if (load_fire) begin
                active_q <= 1'b1;
                active_classifier_q <= load_is_classifier;
                active_output_base_q <= i_context_output_channel_base;
                active_output_count_q <= i_context_output_channel_count;
                active_spatial_base_q <= i_context_spatial_base;
                active_spatial_count_q <= i_context_spatial_count;
                active_pair_q <= i_context_active_pair;
                active_work_base_q <= i_context_work_base_addr;
            end
            if (release_fire) begin
                active_q <= 1'b0;
                active_classifier_q <= 1'b0;
            end
        end
    end

    initial begin
        if (PARAM_ADDR_WIDTH < 1 || WORK_ADDR_WIDTH < 1 ||
                MAX_INPUT_CHANNELS < 1 || MAX_INPUT_CHANNELS > 1024)
            $error("SCNN selected-layer context bridge geometry is invalid");
    end
endmodule

`default_nettype wire
