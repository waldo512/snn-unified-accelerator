`timescale 1ns/1ps
`default_nettype none

// Complete SCNN state/trace transport boundary. One controller owns the
// resident tile context while one shared gearbox couples its load/writeback
// phases to a 128-bit DataMover stream.
module unified_scnn_layer_workset_transport_v1 #(
    parameter int ARRAY_TILES = 16,
    parameter int MEMORY_DEPTH = 1024,
    parameter int EXT_ADDR_WIDTH = 32,
    parameter int CONTEXT_WIDTH = 32,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int ROWS_PER_CHANNEL_WIDTH = 7,
    parameter int TILE_INDEX_WIDTH = $clog2(ARRAY_TILES),
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int BTT_WIDTH = 23
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_command_valid,
    output logic                                   o_command_ready,
    input  wire logic [1:0]                        i_command_op,
    input  wire logic [CONTEXT_WIDTH-1:0]          i_command_context,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]    i_command_channel_count,
    input  wire logic [ROWS_PER_CHANNEL_WIDTH-1:0] i_command_rows_per_channel,
    input  wire logic [EXT_ADDR_WIDTH-1:0]         i_command_backing_base,
    input  wire logic                              i_command_zero_initialize,

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

    output logic                                   o_resident_valid,
    output logic [CONTEXT_WIDTH-1:0]               o_resident_context,
    output logic [CHANNEL_COUNT_WIDTH-1:0]         o_resident_channel_count,
    output logic [ROWS_PER_CHANNEL_WIDTH-1:0]      o_resident_rows_per_channel,
    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_sticky_error
);
    logic workset_dma_valid;
    logic workset_dma_ready;
    logic workset_dma_write;
    logic [EXT_ADDR_WIDTH-1:0] workset_dma_base;
    logic [31:0] workset_dma_records;
    logic controller_dma_abort;
    logic dma_transfer_done;
    logic load_record_valid;
    logic load_record_ready;
    logic [319:0] load_record_data;
    logic load_record_last;
    logic store_record_valid;
    logic store_record_ready;
    logic [319:0] store_record_data;
    logic store_record_last;
    logic controller_busy;
    logic controller_done;
    logic controller_error;
    logic bridge_busy;
    logic bridge_error;
    logic bridge_dma_abort;

    assign o_dma_abort = controller_dma_abort || bridge_dma_abort;
    assign o_busy = controller_busy || bridge_busy;
    assign o_done = controller_done;
    assign o_sticky_error = controller_error || bridge_error;

    unified_scnn_layer_state_workset_controller_v1 #(
        .ARRAY_TILES(ARRAY_TILES),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .EXT_ADDR_WIDTH(EXT_ADDR_WIDTH),
        .CONTEXT_WIDTH(CONTEXT_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .ROWS_PER_CHANNEL_WIDTH(ROWS_PER_CHANNEL_WIDTH),
        .TILE_INDEX_WIDTH(TILE_INDEX_WIDTH),
        .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH)
    ) u_controller (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_command_valid(i_command_valid),
        .o_command_ready(o_command_ready),
        .i_command_op(i_command_op),
        .i_command_context(i_command_context),
        .i_command_channel_count(i_command_channel_count),
        .i_command_rows_per_channel(i_command_rows_per_channel),
        .i_command_backing_base(i_command_backing_base),
        .i_command_zero_initialize(i_command_zero_initialize),
        .i_array_idle(i_array_idle), .o_array_hold(o_array_hold),
        .o_dma_command_valid(workset_dma_valid),
        .i_dma_command_ready(workset_dma_ready),
        .o_dma_command_write(workset_dma_write),
        .o_dma_command_base(workset_dma_base),
        .o_dma_command_records(workset_dma_records),
        .i_dma_transfer_done(dma_transfer_done),
        .o_dma_abort(controller_dma_abort),
        .i_load_stream_valid(load_record_valid),
        .o_load_stream_ready(load_record_ready),
        .i_load_stream_data(load_record_data),
        .i_load_stream_last(load_record_last),
        .o_store_stream_valid(store_record_valid),
        .i_store_stream_ready(store_record_ready),
        .o_store_stream_data(store_record_data),
        .o_store_stream_last(store_record_last),
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
        .o_resident_valid(o_resident_valid),
        .o_resident_context(o_resident_context),
        .o_resident_channel_count(o_resident_channel_count),
        .o_resident_rows_per_channel(o_resident_rows_per_channel),
        .o_busy(controller_busy), .o_done(controller_done),
        .o_sticky_error(controller_error)
    );

    unified_scnn_layer_workset_dma_bridge_v1 #(
        .EXT_ADDR_WIDTH(EXT_ADDR_WIDTH), .BTT_WIDTH(BTT_WIDTH)
    ) u_dma_bridge (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_workset_command_valid(workset_dma_valid),
        .o_workset_command_ready(workset_dma_ready),
        .i_workset_command_write(workset_dma_write),
        .i_workset_command_base(workset_dma_base),
        .i_workset_command_records(workset_dma_records),
        .o_dma_command_valid(o_dma_command_valid),
        .i_dma_command_ready(i_dma_command_ready),
        .o_dma_command_write(o_dma_command_write),
        .o_dma_command_base(o_dma_command_base),
        .o_dma_command_records(o_dma_command_records),
        .o_dma_command_beats(o_dma_command_beats),
        .o_dma_command_bytes(o_dma_command_bytes),
        .o_dma_abort(bridge_dma_abort),
        .i_dma_read_valid(i_dma_read_valid),
        .o_dma_read_ready(o_dma_read_ready),
        .i_dma_read_data(i_dma_read_data),
        .i_dma_read_last(i_dma_read_last),
        .o_dma_write_valid(o_dma_write_valid),
        .i_dma_write_ready(i_dma_write_ready),
        .o_dma_write_data(o_dma_write_data),
        .o_dma_write_last(o_dma_write_last),
        .o_load_record_valid(load_record_valid),
        .i_load_record_ready(load_record_ready),
        .o_load_record_data(load_record_data),
        .o_load_record_last(load_record_last),
        .i_store_record_valid(store_record_valid),
        .o_store_record_ready(store_record_ready),
        .i_store_record_data(store_record_data),
        .i_store_record_last(store_record_last),
        .o_busy(bridge_busy), .o_done(dma_transfer_done),
        .o_sticky_error(bridge_error)
    );
endmodule

`default_nettype wire
