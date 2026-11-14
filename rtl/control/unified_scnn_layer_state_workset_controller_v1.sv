`timescale 1ns/1ps
`default_nettype none

// Moves one SCNN layer's membrane/trace working set through the homogeneous
// tile array. Layers, samples, and student/target epochs reuse the same tile
// RAM; only one context may be resident at a time. A single DMA descriptor
// starts each external transfer, while 320-bit records stream in state-then-
// trace, tile-major, row-major order.
module unified_scnn_layer_state_workset_controller_v1 #(
    parameter int ARRAY_TILES = 16,
    parameter int MEMORY_DEPTH = 1024,
    parameter int EXT_ADDR_WIDTH = 32,
    parameter int CONTEXT_WIDTH = 32,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int ROWS_PER_CHANNEL_WIDTH = 7,
    parameter int TILE_INDEX_WIDTH = $clog2(ARRAY_TILES),
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH)
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
    input  wire logic                              i_dma_transfer_done,
    output logic                                   o_dma_abort,

    input  wire logic                              i_load_stream_valid,
    output logic                                   o_load_stream_ready,
    input  wire logic [319:0]                      i_load_stream_data,
    input  wire logic                              i_load_stream_last,
    output logic                                   o_store_stream_valid,
    input  wire logic                              i_store_stream_ready,
    output logic [319:0]                           o_store_stream_data,
    output logic                                   o_store_stream_last,

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
    localparam logic [1:0] COMMAND_ACQUIRE = 2'd0;
    localparam logic [1:0] COMMAND_RELEASE_WRITEBACK = 2'd1;
    localparam logic [1:0] COMMAND_RELEASE_DROP = 2'd2;
    localparam int HALF_DEPTH = MEMORY_DEPTH / 2;
    localparam int ROW_CALC_WIDTH = CHANNEL_COUNT_WIDTH +
        ROWS_PER_CHANNEL_WIDTH + 1;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_LOAD_DMA_COMMAND,
        STATE_LOAD_STREAM_WRITE,
        STATE_CLEAR_LOCAL_WRITE,
        STATE_EVICT_DMA_COMMAND,
        STATE_EVICT_LOCAL_READ,
        STATE_EVICT_STREAM_WRITE,
        STATE_EVICT_DMA_WAIT
    } state_t;

    state_t state_q;
    logic [CONTEXT_WIDTH-1:0] context_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] channel_count_q;
    logic [ROWS_PER_CHANNEL_WIDTH-1:0] rows_per_channel_q;
    logic [EXT_ADDR_WIDTH-1:0] backing_base_q;
    logic [31:0] total_records_q;
    logic [TILE_INDEX_WIDTH-1:0] tile_q;
    logic [MEMORY_ADDR_WIDTH-1:0] row_q;
    logic trace_record_q;

    logic [CHANNEL_COUNT_WIDTH:0] command_max_channels;
    logic [ROW_CALC_WIDTH-1:0] command_max_rows;
    logic [ROW_CALC_WIDTH-1:0] command_state_records;
    logic [ROW_CALC_WIDTH:0] command_total_records;
    logic command_dimensions_legal;
    logic command_context_legal;
    logic command_legal;
    logic command_fire;

    logic [CHANNEL_COUNT_WIDTH:0] tile_index_extended;
    logic [CHANNEL_COUNT_WIDTH:0] channels_on_tile;
    logic [ROW_CALC_WIDTH-1:0] rows_on_tile_wide;
    logic [TILE_INDEX_WIDTH:0] active_tiles;
    logic row_last;
    logic tile_last;
    logic record_last;
    logic [MEMORY_ADDR_WIDTH-1:0] local_addr;
    logic local_write_fire;
    logic local_read_fire;
    logic store_stream_fire;
    logic expected_result_addr_match;

    always_comb begin
        command_max_channels =
            ({1'b0, i_command_channel_count} +
             (CHANNEL_COUNT_WIDTH + 1)'(ARRAY_TILES - 1)) >>
            TILE_INDEX_WIDTH;
        command_max_rows = command_max_channels *
            i_command_rows_per_channel;
        // The exact record count is independent of round-robin distribution.
        command_state_records = i_command_channel_count *
            i_command_rows_per_channel;
        command_total_records = {command_state_records, 1'b0};
        command_dimensions_legal = i_command_channel_count != 0 &&
            i_command_rows_per_channel != 0 &&
            command_max_rows <= ROW_CALC_WIDTH'(HALF_DEPTH);

        command_context_legal = 1'b0;
        case (i_command_op)
            COMMAND_ACQUIRE:
                command_context_legal = !o_resident_valid &&
                    command_dimensions_legal;
            COMMAND_RELEASE_WRITEBACK,
            COMMAND_RELEASE_DROP:
                command_context_legal = o_resident_valid &&
                    i_command_context == o_resident_context;
            default:
                command_context_legal = 1'b0;
        endcase
        command_legal = command_context_legal;
    end

    always_comb begin
        tile_index_extended = (CHANNEL_COUNT_WIDTH + 1)'(tile_q);
        channels_on_tile = '0;
        if ({1'b0, channel_count_q} > tile_index_extended) begin
            channels_on_tile =
                (({1'b0, channel_count_q} -
                  (CHANNEL_COUNT_WIDTH + 1)'(1) - tile_index_extended) >>
                 TILE_INDEX_WIDTH) + (CHANNEL_COUNT_WIDTH + 1)'(1);
        end
        rows_on_tile_wide = channels_on_tile * rows_per_channel_q;
        if (channel_count_q < CHANNEL_COUNT_WIDTH'(ARRAY_TILES))
            active_tiles = (TILE_INDEX_WIDTH + 1)'(channel_count_q);
        else
            active_tiles = (TILE_INDEX_WIDTH + 1)'(ARRAY_TILES);
    end

    assign row_last = ROW_CALC_WIDTH'(row_q) + ROW_CALC_WIDTH'(1) >=
        rows_on_tile_wide;
    assign tile_last = {1'b0, tile_q} + (TILE_INDEX_WIDTH + 1)'(1) >=
        active_tiles;
    assign record_last = trace_record_q && tile_last && row_last;
    assign local_addr = trace_record_q ?
        MEMORY_ADDR_WIDTH'(HALF_DEPTH) + row_q : row_q;

    assign o_command_ready = !i_abort && !o_sticky_error &&
        state_q == STATE_IDLE && i_array_idle;
    assign command_fire = i_command_valid && o_command_ready;
    assign o_array_hold = state_q != STATE_IDLE || command_fire;
    assign o_busy = state_q != STATE_IDLE;

    assign o_dma_command_valid = !i_abort &&
        (state_q == STATE_LOAD_DMA_COMMAND ||
         state_q == STATE_EVICT_DMA_COMMAND);
    assign o_dma_command_write = state_q == STATE_EVICT_DMA_COMMAND;
    assign o_dma_command_base = backing_base_q;
    assign o_dma_command_records = total_records_q;
    assign o_dma_abort = i_abort && state_q != STATE_IDLE;

    assign o_maintenance_tile = tile_q;
    assign o_state_maintenance_write_valid = !i_abort &&
        !trace_record_q &&
        ((state_q == STATE_LOAD_STREAM_WRITE && i_load_stream_valid) ||
         state_q == STATE_CLEAR_LOCAL_WRITE);
    assign o_trace_maintenance_write_valid = !i_abort &&
        trace_record_q &&
        ((state_q == STATE_LOAD_STREAM_WRITE && i_load_stream_valid) ||
         state_q == STATE_CLEAR_LOCAL_WRITE);
    assign o_load_stream_ready = !i_abort &&
        state_q == STATE_LOAD_STREAM_WRITE &&
        (trace_record_q ? i_trace_maintenance_write_ready :
                          i_state_maintenance_write_ready);
    assign o_state_maintenance_write_addr = local_addr;
    assign o_trace_maintenance_write_addr = local_addr;
    assign o_state_maintenance_write_data =
        state_q == STATE_CLEAR_LOCAL_WRITE ? '0 : i_load_stream_data;
    assign o_trace_maintenance_write_data =
        state_q == STATE_CLEAR_LOCAL_WRITE ? '0 : i_load_stream_data;
    assign o_state_maintenance_write_mask = '1;
    assign o_trace_maintenance_write_mask = '1;
    assign local_write_fire =
        (o_state_maintenance_write_valid &&
         i_state_maintenance_write_ready) ||
        (o_trace_maintenance_write_valid &&
         i_trace_maintenance_write_ready);

    assign o_state_maintenance_read_valid = !i_abort &&
        state_q == STATE_EVICT_LOCAL_READ && !trace_record_q;
    assign o_trace_maintenance_read_valid = !i_abort &&
        state_q == STATE_EVICT_LOCAL_READ && trace_record_q;
    assign o_state_maintenance_read_addr = local_addr;
    assign o_trace_maintenance_read_addr = local_addr;
    assign local_read_fire =
        (o_state_maintenance_read_valid &&
         i_state_maintenance_read_ready) ||
        (o_trace_maintenance_read_valid &&
         i_trace_maintenance_read_ready);

    assign expected_result_addr_match = trace_record_q ?
        i_trace_maintenance_result_addr == local_addr :
        i_state_maintenance_result_addr == local_addr;
    assign o_store_stream_valid = !i_abort &&
        state_q == STATE_EVICT_STREAM_WRITE &&
        (trace_record_q ? i_trace_maintenance_result_valid :
                          i_state_maintenance_result_valid);
    assign o_store_stream_data = trace_record_q ?
        i_trace_maintenance_read_data : i_state_maintenance_read_data;
    assign o_store_stream_last = record_last;
    assign o_state_maintenance_result_ready = !i_abort &&
        state_q == STATE_EVICT_STREAM_WRITE && !trace_record_q &&
        i_store_stream_ready;
    assign o_trace_maintenance_result_ready = !i_abort &&
        state_q == STATE_EVICT_STREAM_WRITE && trace_record_q &&
        i_store_stream_ready;
    assign store_stream_fire = o_store_stream_valid && i_store_stream_ready;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            context_q <= '0;
            channel_count_q <= '0;
            rows_per_channel_q <= '0;
            backing_base_q <= '0;
            total_records_q <= '0;
            tile_q <= '0;
            row_q <= '0;
            trace_record_q <= 1'b0;
            o_resident_valid <= 1'b0;
            o_resident_context <= '0;
            o_resident_channel_count <= '0;
            o_resident_rows_per_channel <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            tile_q <= '0;
            row_q <= '0;
            trace_record_q <= 1'b0;
            o_resident_valid <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            // Array idleness is sampled when the command is accepted. During
            // maintenance the selected tile RAM is legitimately busy, so the
            // aggregate array-idle signal may fall. The physical integration
            // wrapper blocks new compute issues while o_array_hold is high.
            case (state_q)
                    STATE_IDLE: begin
                        if (command_fire) begin
                            if (!command_legal) begin
                                o_sticky_error <= 1'b1;
                            end else if (i_command_op ==
                                         COMMAND_RELEASE_DROP) begin
                                o_resident_valid <= 1'b0;
                                o_done <= 1'b1;
                            end else begin
                                tile_q <= '0;
                                row_q <= '0;
                                trace_record_q <= 1'b0;
                                if (i_command_op == COMMAND_ACQUIRE) begin
                                    context_q <= i_command_context;
                                    channel_count_q <=
                                        i_command_channel_count;
                                    rows_per_channel_q <=
                                        i_command_rows_per_channel;
                                    backing_base_q <=
                                        i_command_backing_base;
                                    total_records_q <=
                                        32'(command_total_records);
                                    if (i_command_zero_initialize)
                                        state_q <= STATE_CLEAR_LOCAL_WRITE;
                                    else
                                        state_q <= STATE_LOAD_DMA_COMMAND;
                                end else begin
                                    state_q <= STATE_EVICT_DMA_COMMAND;
                                end
                            end
                        end
                    end

                    STATE_LOAD_DMA_COMMAND: begin
                        if (o_dma_command_valid && i_dma_command_ready)
                            state_q <= STATE_LOAD_STREAM_WRITE;
                    end

                    STATE_EVICT_DMA_COMMAND: begin
                        if (o_dma_command_valid && i_dma_command_ready)
                            state_q <= STATE_EVICT_LOCAL_READ;
                    end

                    STATE_LOAD_STREAM_WRITE,
                    STATE_CLEAR_LOCAL_WRITE: begin
                        if (local_write_fire) begin
                            if (state_q == STATE_LOAD_STREAM_WRITE &&
                                i_load_stream_last != record_last)
                                o_sticky_error <= 1'b1;
                            if (record_last) begin
                                o_resident_valid <= 1'b1;
                                o_resident_context <= context_q;
                                o_resident_channel_count <= channel_count_q;
                                o_resident_rows_per_channel <=
                                    rows_per_channel_q;
                                state_q <= STATE_IDLE;
                                o_done <= 1'b1;
                            end else if (row_last) begin
                                row_q <= '0;
                                if (tile_last) begin
                                    tile_q <= '0;
                                    trace_record_q <= 1'b1;
                                end else begin
                                    tile_q <= tile_q +
                                        TILE_INDEX_WIDTH'(1);
                                end
                            end else begin
                                row_q <= row_q + MEMORY_ADDR_WIDTH'(1);
                            end
                        end
                    end

                    STATE_EVICT_LOCAL_READ: begin
                        if (local_read_fire)
                            state_q <= STATE_EVICT_STREAM_WRITE;
                    end

                    STATE_EVICT_STREAM_WRITE: begin
                        if (store_stream_fire) begin
                            if (!expected_result_addr_match)
                                o_sticky_error <= 1'b1;
                            if (record_last) begin
                                state_q <= STATE_EVICT_DMA_WAIT;
                            end else begin
                                if (row_last) begin
                                    row_q <= '0;
                                    if (tile_last) begin
                                        tile_q <= '0;
                                        trace_record_q <= 1'b1;
                                    end else begin
                                        tile_q <= tile_q +
                                            TILE_INDEX_WIDTH'(1);
                                    end
                                end else begin
                                    row_q <= row_q +
                                        MEMORY_ADDR_WIDTH'(1);
                                end
                                state_q <= STATE_EVICT_LOCAL_READ;
                            end
                        end
                    end

                    STATE_EVICT_DMA_WAIT: begin
                        if (i_dma_transfer_done) begin
                            o_resident_valid <= 1'b0;
                            state_q <= STATE_IDLE;
                            o_done <= 1'b1;
                        end
                    end

                    default: begin
                        state_q <= STATE_IDLE;
                        o_resident_valid <= 1'b0;
                        o_sticky_error <= 1'b1;
                    end
            endcase
        end
    end

    initial begin
        if (ARRAY_TILES != (1 << TILE_INDEX_WIDTH))
            $error("SCNN workset controller requires power-of-two tile count");
        if (MEMORY_DEPTH < 4 || (MEMORY_DEPTH & 1) != 0)
            $error("SCNN workset controller requires an even memory depth");
    end
endmodule

`default_nettype wire
