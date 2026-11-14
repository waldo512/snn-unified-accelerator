`timescale 1ns/1ps
`default_nettype none

// Binds topology-independent context ownership to the single tile workset
// transport. FC, SRNN and SCNN differ only in context geometry and policy.
module unified_shared_tile_context_transport_v1 #(
    parameter int ARRAY_TILES = 16,
    parameter int MEMORY_DEPTH = 1024,
    parameter int EXT_ADDR_WIDTH = 32,
    parameter int CONTEXT_WIDTH = 64,
    parameter int ACTIVE_UNITS_WIDTH = 10,
    parameter int ROWS_PER_UNIT_WIDTH = 7,
    parameter int TILE_INDEX_WIDTH = $clog2(ARRAY_TILES),
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int BTT_WIDTH = 23
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_request_valid,
    output logic                                   o_request_ready,
    input  wire logic [CONTEXT_WIDTH-1:0]          i_request_context,
    input  wire logic [ACTIVE_UNITS_WIDTH-1:0]     i_request_active_units,
    input  wire logic [ROWS_PER_UNIT_WIDTH-1:0]    i_request_rows_per_unit,
    input  wire logic [EXT_ADDR_WIDTH-1:0]         i_request_backing_base,
    input  wire logic                              i_request_restore,
    input  wire logic                              i_request_save_on_evict,
    output logic                                   o_request_complete,

    input  wire logic                              i_release_valid,
    output logic                                   o_release_ready,
    input  wire logic                              i_release_writeback,
    output logic                                   o_release_complete,

    input  wire logic                              i_array_idle,
    output logic                                   o_array_hold,

    output logic                                   o_dma_command_valid,
    input  wire logic                              i_dma_command_ready,
    output logic                                   o_dma_command_write,
    output logic [EXT_ADDR_WIDTH-1:0]              o_dma_command_base,
    output logic [31:0]                            o_dma_command_records,
    output logic [31:0]                            o_dma_command_beats,
    output logic [BTT_WIDTH-1:0]                   o_dma_command_bytes,
    output logic                                   o_dma_abort,
    input  wire logic                              i_dma_read_valid,
    output logic                                   o_dma_read_ready,
    input  wire logic [127:0]                      i_dma_read_data,
    input  wire logic                              i_dma_read_last,
    output logic                                   o_dma_write_valid,
    input  wire logic                              i_dma_write_ready,
    output logic [127:0]                           o_dma_write_data,
    output logic                                   o_dma_write_last,

    output logic [TILE_INDEX_WIDTH-1:0]            o_maintenance_tile,
    output logic                                   o_state_maintenance_write_valid,
    input  wire logic                              i_state_maintenance_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]           o_state_maintenance_write_addr,
    output logic [319:0]                           o_state_maintenance_write_data,
    output logic [39:0]                            o_state_maintenance_write_mask,
    output logic                                   o_state_maintenance_read_valid,
    input  wire logic                              i_state_maintenance_read_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]           o_state_maintenance_read_addr,
    input  wire logic                              i_state_maintenance_result_valid,
    output logic                                   o_state_maintenance_result_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]      i_state_maintenance_result_addr,
    input  wire logic [319:0]                      i_state_maintenance_read_data,
    output logic                                   o_trace_maintenance_write_valid,
    input  wire logic                              i_trace_maintenance_write_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]           o_trace_maintenance_write_addr,
    output logic [319:0]                           o_trace_maintenance_write_data,
    output logic [39:0]                            o_trace_maintenance_write_mask,
    output logic                                   o_trace_maintenance_read_valid,
    input  wire logic                              i_trace_maintenance_read_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]           o_trace_maintenance_read_addr,
    input  wire logic                              i_trace_maintenance_result_valid,
    output logic                                   o_trace_maintenance_result_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]      i_trace_maintenance_result_addr,
    input  wire logic [319:0]                      i_trace_maintenance_read_data,

    output logic                                   o_context_active,
    output logic [CONTEXT_WIDTH-1:0]               o_active_context,
    output logic [ACTIVE_UNITS_WIDTH-1:0]          o_active_units,
    output logic [ROWS_PER_UNIT_WIDTH-1:0]         o_active_rows_per_unit,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    logic transport_command_valid;
    logic transport_command_ready;
    logic [1:0] transport_command_op;
    logic [CONTEXT_WIDTH-1:0] transport_command_context;
    logic [ACTIVE_UNITS_WIDTH-1:0] transport_command_active_units;
    logic [ROWS_PER_UNIT_WIDTH-1:0] transport_command_rows_per_unit;
    logic [EXT_ADDR_WIDTH-1:0] transport_command_backing_base;
    logic transport_command_zero_initialize;
    logic transport_done;
    logic transport_busy;
    logic transport_sticky_error;
    logic transport_resident_valid;
    logic [CONTEXT_WIDTH-1:0] transport_resident_context;
    logic [ACTIVE_UNITS_WIDTH-1:0] transport_resident_active_units;
    logic [ROWS_PER_UNIT_WIDTH-1:0] transport_resident_rows_per_unit;

    unified_tile_context_owner_v1 #(
        .CONTEXT_WIDTH(CONTEXT_WIDTH),
        .EXT_ADDR_WIDTH(EXT_ADDR_WIDTH),
        .ACTIVE_UNITS_WIDTH(ACTIVE_UNITS_WIDTH),
        .ROWS_PER_UNIT_WIDTH(ROWS_PER_UNIT_WIDTH)
    ) u_context_owner (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_request_valid(i_request_valid),
        .o_request_ready(o_request_ready),
        .i_request_context(i_request_context),
        .i_request_active_units(i_request_active_units),
        .i_request_rows_per_unit(i_request_rows_per_unit),
        .i_request_backing_base(i_request_backing_base),
        .i_request_restore(i_request_restore),
        .i_request_save_on_evict(i_request_save_on_evict),
        .o_request_complete(o_request_complete),
        .i_release_valid(i_release_valid),
        .o_release_ready(o_release_ready),
        .i_release_writeback(i_release_writeback),
        .o_release_complete(o_release_complete),
        .o_transport_command_valid(transport_command_valid),
        .i_transport_command_ready(transport_command_ready),
        .o_transport_command_op(transport_command_op),
        .o_transport_command_context(transport_command_context),
        .o_transport_command_active_units(transport_command_active_units),
        .o_transport_command_rows_per_unit(transport_command_rows_per_unit),
        .o_transport_command_backing_base(transport_command_backing_base),
        .o_transport_command_zero_initialize(
            transport_command_zero_initialize),
        .i_transport_done(transport_done),
        .i_transport_busy(transport_busy),
        .i_transport_sticky_error(transport_sticky_error),
        .i_transport_resident_valid(transport_resident_valid),
        .i_transport_resident_context(transport_resident_context),
        .i_transport_resident_active_units(transport_resident_active_units),
        .i_transport_resident_rows_per_unit(
            transport_resident_rows_per_unit),
        .o_context_active(o_context_active),
        .o_active_context(o_active_context),
        .o_active_units(o_active_units),
        .o_active_rows_per_unit(o_active_rows_per_unit),
        .o_busy(o_busy), .o_sticky_error(o_sticky_error)
    );

    unified_scnn_layer_workset_transport_v1 #(
        .ARRAY_TILES(ARRAY_TILES),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .EXT_ADDR_WIDTH(EXT_ADDR_WIDTH),
        .CONTEXT_WIDTH(CONTEXT_WIDTH),
        .CHANNEL_COUNT_WIDTH(ACTIVE_UNITS_WIDTH),
        .ROWS_PER_CHANNEL_WIDTH(ROWS_PER_UNIT_WIDTH),
        .TILE_INDEX_WIDTH(TILE_INDEX_WIDTH),
        .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .BTT_WIDTH(BTT_WIDTH)
    ) u_workset_transport (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_command_valid(transport_command_valid),
        .o_command_ready(transport_command_ready),
        .i_command_op(transport_command_op),
        .i_command_context(transport_command_context),
        .i_command_channel_count(transport_command_active_units),
        .i_command_rows_per_channel(transport_command_rows_per_unit),
        .i_command_backing_base(transport_command_backing_base),
        .i_command_zero_initialize(transport_command_zero_initialize),
        .i_array_idle(i_array_idle), .o_array_hold(o_array_hold),
        .o_dma_command_valid(o_dma_command_valid),
        .i_dma_command_ready(i_dma_command_ready),
        .o_dma_command_write(o_dma_command_write),
        .o_dma_command_base(o_dma_command_base),
        .o_dma_command_records(o_dma_command_records),
        .o_dma_command_beats(o_dma_command_beats),
        .o_dma_command_bytes(o_dma_command_bytes),
        .o_dma_abort(o_dma_abort),
        .i_dma_read_valid(i_dma_read_valid),
        .o_dma_read_ready(o_dma_read_ready),
        .i_dma_read_data(i_dma_read_data),
        .i_dma_read_last(i_dma_read_last),
        .o_dma_write_valid(o_dma_write_valid),
        .i_dma_write_ready(i_dma_write_ready),
        .o_dma_write_data(o_dma_write_data),
        .o_dma_write_last(o_dma_write_last),
        .o_maintenance_tile(o_maintenance_tile),
        .o_state_maintenance_write_valid(o_state_maintenance_write_valid),
        .i_state_maintenance_write_ready(i_state_maintenance_write_ready),
        .o_state_maintenance_write_addr(o_state_maintenance_write_addr),
        .o_state_maintenance_write_data(o_state_maintenance_write_data),
        .o_state_maintenance_write_mask(o_state_maintenance_write_mask),
        .o_state_maintenance_read_valid(o_state_maintenance_read_valid),
        .i_state_maintenance_read_ready(i_state_maintenance_read_ready),
        .o_state_maintenance_read_addr(o_state_maintenance_read_addr),
        .i_state_maintenance_result_valid(i_state_maintenance_result_valid),
        .o_state_maintenance_result_ready(o_state_maintenance_result_ready),
        .i_state_maintenance_result_addr(i_state_maintenance_result_addr),
        .i_state_maintenance_read_data(i_state_maintenance_read_data),
        .o_trace_maintenance_write_valid(o_trace_maintenance_write_valid),
        .i_trace_maintenance_write_ready(i_trace_maintenance_write_ready),
        .o_trace_maintenance_write_addr(o_trace_maintenance_write_addr),
        .o_trace_maintenance_write_data(o_trace_maintenance_write_data),
        .o_trace_maintenance_write_mask(o_trace_maintenance_write_mask),
        .o_trace_maintenance_read_valid(o_trace_maintenance_read_valid),
        .i_trace_maintenance_read_ready(i_trace_maintenance_read_ready),
        .o_trace_maintenance_read_addr(o_trace_maintenance_read_addr),
        .i_trace_maintenance_result_valid(i_trace_maintenance_result_valid),
        .o_trace_maintenance_result_ready(o_trace_maintenance_result_ready),
        .i_trace_maintenance_result_addr(i_trace_maintenance_result_addr),
        .i_trace_maintenance_read_data(i_trace_maintenance_read_data),
        .o_resident_valid(transport_resident_valid),
        .o_resident_context(transport_resident_context),
        .o_resident_channel_count(transport_resident_active_units),
        .o_resident_rows_per_channel(transport_resident_rows_per_unit),
        .o_busy(transport_busy), .o_done(transport_done),
        .o_sticky_error(transport_sticky_error)
    );
endmodule

`default_nettype wire
