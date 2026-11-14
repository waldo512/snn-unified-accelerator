`timescale 1ns/1ps
`default_nettype none

// FC results are a direct handshake bypass. Dynamic reduction is extension-only.
module unified_fc_first_output #(
    parameter integer TILE_COUNT = 12,
    parameter integer OUTPUT_MAX = 32,
    parameter integer ACC_BITS = 32,
    parameter integer META_BITS = 16
) (
    input  wire logic                                      clk,
    input  wire logic                                      rst_n,
    input  wire logic                                      i_clear_error,
    input  wire logic                                      i_fc_fast_mode,
    input  wire logic                                      i_fc_valid,
    output logic                                           o_fc_ready,
    input  wire logic [5:0]                                i_fc_class,
    input  wire logic signed [ACC_BITS-1:0]                i_fc_score,
    input  wire logic [OUTPUT_MAX*ACC_BITS-1:0]            i_fc_logits_flat,
    input  wire logic [META_BITS-1:0]                      i_fc_meta,
    input  wire logic [15:0]                               i_fc_version,
    input  wire logic                                      i_dynamic_start,
    output logic                                           o_dynamic_ready,
    input  wire logic [3:0]                                i_active_tiles,
    input  wire logic [5:0]                                i_output_count,
    input  wire logic [TILE_COUNT*OUTPUT_MAX*ACC_BITS-1:0] i_partials_flat,
    input  wire logic [META_BITS-1:0]                      i_dynamic_meta,
    input  wire logic [15:0]                               i_dynamic_version,
    output logic                                           o_result_valid,
    input  wire logic                                      i_result_ready,
    output logic [5:0]                                     o_result_class,
    output logic signed [ACC_BITS-1:0]                     o_result_score,
    output logic [OUTPUT_MAX*ACC_BITS-1:0]                 o_result_logits_flat,
    output logic [META_BITS-1:0]                           o_result_meta,
    output logic [15:0]                                    o_result_version,
    output logic                                           o_extension_active,
    output logic                                           o_sticky_error
);
    logic dynamic_valid;
    logic dynamic_start_ready;
    logic dynamic_result_ready;
    logic [5:0] dynamic_class;
    logic signed [ACC_BITS-1:0] dynamic_score;
    logic [OUTPUT_MAX*ACC_BITS-1:0] dynamic_logits_flat;
    logic [META_BITS-1:0] dynamic_meta;
    logic [15:0] dynamic_version;
    logic dynamic_busy;
    logic dynamic_error;
    logic mode_error_q;
    logic dynamic_selected;

    unified_dynamic_output_engine #(
        .TILE_COUNT(TILE_COUNT), .OUTPUT_MAX(OUTPUT_MAX),
        .ACC_BITS(ACC_BITS), .META_BITS(META_BITS)
    ) u_dynamic (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_start(i_dynamic_start && !i_fc_fast_mode && !dynamic_busy),
        .o_ready(dynamic_start_ready), .i_active_tiles(i_active_tiles),
        .i_output_count(i_output_count), .i_partials_flat(i_partials_flat),
        .i_meta(i_dynamic_meta), .i_version(i_dynamic_version),
        .o_valid(dynamic_valid), .i_ready(dynamic_result_ready),
        .o_logits_flat(dynamic_logits_flat), .o_class(dynamic_class),
        .o_score(dynamic_score), .o_meta(dynamic_meta),
        .o_version(dynamic_version), .o_busy(dynamic_busy),
        .o_sticky_error(dynamic_error)
    );

    always_comb begin
        // Once accepted, a dynamic result owns the output until its handshake.
        // A mid-transaction mode change is an error but cannot hide the result.
        dynamic_selected = dynamic_busy || !i_fc_fast_mode;
        o_fc_ready = i_fc_fast_mode && !dynamic_busy && i_result_ready;
        dynamic_result_ready = dynamic_selected && i_result_ready;
        o_dynamic_ready = dynamic_start_ready && !i_fc_fast_mode && !dynamic_busy;
        o_extension_active = dynamic_busy;
        if (!dynamic_selected) begin
            o_result_valid = i_fc_valid;
            o_result_class = i_fc_class;
            o_result_score = i_fc_score;
            o_result_logits_flat = i_fc_logits_flat;
            o_result_meta = i_fc_meta;
            o_result_version = i_fc_version;
        end else begin
            o_result_valid = dynamic_valid;
            o_result_class = dynamic_class;
            o_result_score = dynamic_score;
            o_result_logits_flat = dynamic_logits_flat;
            o_result_meta = dynamic_meta;
            o_result_version = dynamic_version;
        end
        o_sticky_error = mode_error_q || dynamic_error;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            mode_error_q <= 1'b0;
        else if (i_clear_error)
            mode_error_q <= 1'b0;
        else if ((i_fc_fast_mode && i_dynamic_start) ||
                 (dynamic_busy && i_fc_fast_mode))
            mode_error_q <= 1'b1;
    end
endmodule

`default_nettype wire
