`timescale 1ns/1ps
`default_nettype none

// Control-plane boundary between descriptor execution and the tile-local
// quantized weight caches. Parameter refill completion and TP invalidation
// update residency here; a descriptor is released only after every selected
// tile has the required version and the array is idle.
module unified_array16_weight_cache_control_v1 #(
    parameter int TAG_WIDTH = 32
) (
    input  wire logic                     i_clk,
    input  wire logic                     i_rst_n,
    input  wire logic                     i_abort,
    input  wire logic                     i_clear_error,

    input  wire logic                     i_fill_complete_valid,
    output logic                          o_fill_complete_ready,
    input  wire logic [3:0]               i_fill_complete_tile,
    input  wire logic [1:0]               i_fill_complete_bank,
    input  wire logic [TAG_WIDTH-1:0]     i_fill_complete_tag,

    input  wire logic                     i_invalidate_valid,
    output logic                          o_invalidate_ready,
    input  wire logic [3:0]               i_invalidate_tile,
    input  wire logic [3:0]               i_invalidate_bank_mask,
    input  wire logic                     i_flush_all,

    input  wire logic                     i_request_valid,
    output logic                          o_request_ready,
    input  wire logic [1:0]               i_request_mode,
    input  wire logic [15:0]              i_request_tile_mask,
    input  wire logic                     i_request_scnn_active_bank,
    input  wire logic [TAG_WIDTH-1:0]     i_request_tag,

    output logic                          o_miss_valid,
    input  wire logic                     i_miss_ready,
    output logic [3:0]                    o_miss_tile,
    output logic [3:0]                    o_miss_bank_mask,
    output logic [TAG_WIDTH-1:0]          o_miss_tag,

    input  wire logic                     i_array_idle,
    output logic                          o_activate_valid,
    input  wire logic                     i_activate_ready,
    output logic [1:0]                    o_activate_mode,
    output logic [15:0]                   o_activate_tile_mask,
    output logic                          o_activate_scnn_active_bank,
    output logic [TAG_WIDTH-1:0]          o_activate_tag,

    output logic [63:0]                   o_bank_valid_flat,
    output logic                          o_busy,
    output logic                          o_waiting_for_idle,
    output logic                          o_sticky_error
);
    logic [3:0] query_tile;
    logic [3:0] query_bank_mask;
    logic [TAG_WIDTH-1:0] query_tag;
    logic query_hit;

    unified_array16_weight_cache_residency_v1 #(
        .TAG_WIDTH(TAG_WIDTH)
    ) u_residency (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_fill_complete_valid(i_fill_complete_valid),
        .o_fill_complete_ready(o_fill_complete_ready),
        .i_fill_complete_tile(i_fill_complete_tile),
        .i_fill_complete_bank(i_fill_complete_bank),
        .i_fill_complete_tag(i_fill_complete_tag),
        .i_invalidate_valid(i_invalidate_valid),
        .o_invalidate_ready(o_invalidate_ready),
        .i_invalidate_tile(i_invalidate_tile),
        .i_invalidate_bank_mask(i_invalidate_bank_mask),
        .i_flush_all(i_flush_all), .i_query_tile(query_tile),
        .i_query_bank_mask(query_bank_mask), .i_query_tag(query_tag),
        .o_query_hit(query_hit), .o_bank_valid_flat(o_bank_valid_flat)
    );

    unified_array16_weight_workset_activate_v1 #(
        .TAG_WIDTH(TAG_WIDTH)
    ) u_activate (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_request_valid(i_request_valid),
        .o_request_ready(o_request_ready),
        .i_request_mode(i_request_mode),
        .i_request_tile_mask(i_request_tile_mask),
        .i_request_scnn_active_bank(i_request_scnn_active_bank),
        .i_request_tag(i_request_tag), .o_query_tile(query_tile),
        .o_query_bank_mask(query_bank_mask), .o_query_tag(query_tag),
        .i_query_hit(query_hit), .o_miss_valid(o_miss_valid),
        .i_miss_ready(i_miss_ready), .o_miss_tile(o_miss_tile),
        .o_miss_bank_mask(o_miss_bank_mask), .o_miss_tag(o_miss_tag),
        .i_array_idle(i_array_idle),
        .o_activate_valid(o_activate_valid),
        .i_activate_ready(i_activate_ready),
        .o_activate_mode(o_activate_mode),
        .o_activate_tile_mask(o_activate_tile_mask),
        .o_activate_scnn_active_bank(o_activate_scnn_active_bank),
        .o_activate_tag(o_activate_tag), .o_busy(o_busy),
        .o_waiting_for_idle(o_waiting_for_idle),
        .o_sticky_error(o_sticky_error)
    );
endmodule

`default_nettype wire
