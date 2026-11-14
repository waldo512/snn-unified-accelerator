`timescale 1ns/1ps
`default_nettype none

// Physical integration boundary for the homogeneous 16-tile array. The
// context transport is the only maintenance owner of the tile state/trace RAM.
module unified_h40_array16_resident_context_v1 #(
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter int CONTEXT_WIDTH = 64,
    parameter int EXT_ADDR_WIDTH = 32,
    parameter int BTT_WIDTH = 23,
    parameter bit CUT_RECORD_READY = 1'b0
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_context_request_valid,
    output logic                                o_context_request_ready,
    input  wire logic [CONTEXT_WIDTH-1:0]       i_context_request_id,
    input  wire logic [9:0]                     i_context_active_units,
    input  wire logic [6:0]                     i_context_rows_per_unit,
    input  wire logic [EXT_ADDR_WIDTH-1:0]      i_context_backing_base,
    input  wire logic                           i_context_restore,
    input  wire logic                           i_context_save_on_evict,
    output logic                                o_context_request_complete,
    input  wire logic                           i_context_release_valid,
    output logic                                o_context_release_ready,
    input  wire logic                           i_context_release_writeback,
    output logic                                o_context_release_complete,
    output logic                                o_context_active,
    output logic [CONTEXT_WIDTH-1:0]            o_active_context,
    output logic [9:0]                          o_active_units,
    output logic [6:0]                          o_active_rows_per_unit,
    output logic                                o_context_busy,
    output logic                                o_context_sticky_error,
    output logic                                o_array_hold,

    output logic                                o_dma_command_valid,
    input  wire logic                           i_dma_command_ready,
    output logic                                o_dma_command_write,
    output logic [EXT_ADDR_WIDTH-1:0]           o_dma_command_base,
    output logic [31:0]                         o_dma_command_records,
    output logic [31:0]                         o_dma_command_beats,
    output logic [BTT_WIDTH-1:0]                o_dma_command_bytes,
    output logic                                o_dma_abort,
    input  wire logic                           i_dma_read_valid,
    output logic                                o_dma_read_ready,
    input  wire logic [127:0]                   i_dma_read_data,
    input  wire logic                           i_dma_read_last,
    output logic                                o_dma_write_valid,
    input  wire logic                           i_dma_write_ready,
    output logic [127:0]                        o_dma_write_data,
    output logic                                o_dma_write_last,

    input  wire logic [3:0]                     i_weight_fill_valid,
    output logic [3:0]                          o_weight_fill_ready,
    input  wire logic [7:0]                     i_weight_fill_tile_flat,
    input  wire logic [3:0]                     i_weight_fill_bank,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_fill_addr_flat,
    input  wire logic [799:0]                   i_weight_fill_data_flat,
    input  wire logic [15:0]                    i_issue_valid,
    output logic [15:0]                         o_issue_ready,
    input  wire logic [15:0]                    i_first,
    input  wire logic [15:0]                    i_last,
    input  wire logic [639:0]                   i_lane_mask_flat,
    input  wire logic [639:0]                   i_enable_a_flat,
    input  wire logic [639:0]                   i_enable_d_flat,
    input  wire logic [639:0]                   i_enable2_flat,
    input  wire logic [639:0]                   i_enable3_flat,
    input  wire logic [11519:0]                 i_factor_b_flat,
    input  wire logic [31:0]                    i_mode_flat,
    input  wire logic [15:0]                    i_scnn_active_bank,
    input  wire logic [16*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr0_flat,
    input  wire logic [16*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr1_flat,
    input  wire logic [16*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr2_flat,
    input  wire logic [16*WEIGHT_ADDR_WIDTH-1:0] i_weight_addr3_flat,
    input  wire logic [63:0]                    i_weight_subword0_flat,
    input  wire logic [63:0]                    i_weight_subword1_flat,
    input  wire logic [63:0]                    i_weight_subword2_flat,
    input  wire logic [63:0]                    i_weight_subword3_flat,
    input  wire logic [16*WORK_ADDR_WIDTH-1:0]  i_work_base_addr_flat,
    input  wire logic [255:0]                   i_leak_flat,
    input  wire logic [511:0]                   i_threshold_flat,
    input  wire logic [15:0]                    i_reset_to_zero,
    input  wire logic [255:0]                   i_trace_decay_flat,
    input  wire logic [991:0]                   i_transaction_meta_flat,

    output logic [3:0]                          o_observation_valid,
    input  wire logic [3:0]                     i_observation_ready,
    output logic [15:0]                         o_observation_tile_flat,
    output logic [7:0]                          o_observation_mode_flat,
    output logic [39:0]                         o_observation_lane_mask_flat,
    output logic [39:0]                         o_observation_range_mask_flat,
    output logic [39:0]                         o_observation_spike_flat,
    output logic [1279:0]                       o_observation_pre_state_flat,
    output logic [1279:0]                       o_observation_trace_next_flat,
    output logic [255:0]                        o_observation_meta_flat,
    output logic [15:0]                         o_done,
    output logic [15:0]                         o_tile_idle,
    output logic                                o_idle,
    output logic [15:0]                         o_tile_sticky_error
);
    logic [15:0] gated_issue_valid;
    logic [15:0] array_issue_ready;
    logic array_idle;
    logic [3:0] maintenance_tile;
    logic state_write_valid;
    logic state_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] state_write_addr;
    logic [319:0] state_write_data;
    logic [39:0] state_write_mask;
    logic state_read_valid;
    logic state_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] state_read_addr;
    logic state_result_valid;
    logic state_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] state_result_addr;
    logic [319:0] state_read_data;
    logic trace_write_valid;
    logic trace_write_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_write_addr;
    logic [319:0] trace_write_data;
    logic [39:0] trace_write_mask;
    logic trace_read_valid;
    logic trace_read_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_read_addr;
    logic trace_result_valid;
    logic trace_result_ready;
    logic [MEMORY_ADDR_WIDTH-1:0] trace_result_addr;
    logic [319:0] trace_read_data;

    assign gated_issue_valid = i_issue_valid &
        {16{o_context_active && !o_array_hold}};
    assign o_issue_ready = array_issue_ready &
        {16{o_context_active && !o_array_hold}};
    assign o_idle = array_idle && !o_context_busy;

    unified_shared_tile_context_transport_v1 #(
        .ARRAY_TILES(16), .MEMORY_DEPTH(MEMORY_DEPTH),
        .EXT_ADDR_WIDTH(EXT_ADDR_WIDTH), .CONTEXT_WIDTH(CONTEXT_WIDTH),
        .BTT_WIDTH(BTT_WIDTH)
    ) u_context_transport (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_request_valid(i_context_request_valid),
        .o_request_ready(o_context_request_ready),
        .i_request_context(i_context_request_id),
        .i_request_active_units(i_context_active_units),
        .i_request_rows_per_unit(i_context_rows_per_unit),
        .i_request_backing_base(i_context_backing_base),
        .i_request_restore(i_context_restore),
        .i_request_save_on_evict(i_context_save_on_evict),
        .o_request_complete(o_context_request_complete),
        .i_release_valid(i_context_release_valid),
        .o_release_ready(o_context_release_ready),
        .i_release_writeback(i_context_release_writeback),
        .o_release_complete(o_context_release_complete),
        .i_array_idle(array_idle), .o_array_hold(o_array_hold),
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
        .o_maintenance_tile(maintenance_tile),
        .o_state_maintenance_write_valid(state_write_valid),
        .i_state_maintenance_write_ready(state_write_ready),
        .o_state_maintenance_write_addr(state_write_addr),
        .o_state_maintenance_write_data(state_write_data),
        .o_state_maintenance_write_mask(state_write_mask),
        .o_state_maintenance_read_valid(state_read_valid),
        .i_state_maintenance_read_ready(state_read_ready),
        .o_state_maintenance_read_addr(state_read_addr),
        .i_state_maintenance_result_valid(state_result_valid),
        .o_state_maintenance_result_ready(state_result_ready),
        .i_state_maintenance_result_addr(state_result_addr),
        .i_state_maintenance_read_data(state_read_data),
        .o_trace_maintenance_write_valid(trace_write_valid),
        .i_trace_maintenance_write_ready(trace_write_ready),
        .o_trace_maintenance_write_addr(trace_write_addr),
        .o_trace_maintenance_write_data(trace_write_data),
        .o_trace_maintenance_write_mask(trace_write_mask),
        .o_trace_maintenance_read_valid(trace_read_valid),
        .i_trace_maintenance_read_ready(trace_read_ready),
        .o_trace_maintenance_read_addr(trace_read_addr),
        .i_trace_maintenance_result_valid(trace_result_valid),
        .o_trace_maintenance_result_ready(trace_result_ready),
        .i_trace_maintenance_result_addr(trace_result_addr),
        .i_trace_maintenance_read_data(trace_read_data),
        .o_context_active(o_context_active),
        .o_active_context(o_active_context),
        .o_active_units(o_active_units),
        .o_active_rows_per_unit(o_active_rows_per_unit),
        .o_busy(o_context_busy),
        .o_sticky_error(o_context_sticky_error)
    );

    unified_dual_role_h40_stationary_array16_candidate_v1 #(
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .CUT_RECORD_READY(CUT_RECORD_READY)
    ) u_array (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_weight_fill_valid(i_weight_fill_valid),
        .o_weight_fill_ready(o_weight_fill_ready),
        .i_weight_fill_tile_flat(i_weight_fill_tile_flat),
        .i_weight_fill_bank(i_weight_fill_bank),
        .i_weight_fill_addr_flat(i_weight_fill_addr_flat),
        .i_weight_fill_data_flat(i_weight_fill_data_flat),
        .i_issue_valid(gated_issue_valid),
        .o_issue_ready(array_issue_ready),
        .i_first(i_first), .i_last(i_last),
        .i_lane_mask_flat(i_lane_mask_flat),
        .i_enable_a_flat(i_enable_a_flat),
        .i_enable_d_flat(i_enable_d_flat),
        .i_enable2_flat(i_enable2_flat),
        .i_enable3_flat(i_enable3_flat),
        .i_factor_b_flat(i_factor_b_flat),
        .i_mode_flat(i_mode_flat),
        .i_scnn_active_bank(i_scnn_active_bank),
        .i_weight_addr0_flat(i_weight_addr0_flat),
        .i_weight_addr1_flat(i_weight_addr1_flat),
        .i_weight_addr2_flat(i_weight_addr2_flat),
        .i_weight_addr3_flat(i_weight_addr3_flat),
        .i_weight_subword0_flat(i_weight_subword0_flat),
        .i_weight_subword1_flat(i_weight_subword1_flat),
        .i_weight_subword2_flat(i_weight_subword2_flat),
        .i_weight_subword3_flat(i_weight_subword3_flat),
        .i_work_base_addr_flat(i_work_base_addr_flat),
        .i_leak_flat(i_leak_flat),
        .i_threshold_flat(i_threshold_flat),
        .i_reset_to_zero(i_reset_to_zero),
        .i_trace_decay_flat(i_trace_decay_flat),
        .i_transaction_meta_flat(i_transaction_meta_flat),
        .o_observation_valid(o_observation_valid),
        .i_observation_ready(i_observation_ready),
        .o_observation_tile_flat(o_observation_tile_flat),
        .o_observation_mode_flat(o_observation_mode_flat),
        .o_observation_lane_mask_flat(o_observation_lane_mask_flat),
        .o_observation_range_mask_flat(o_observation_range_mask_flat),
        .o_observation_spike_flat(o_observation_spike_flat),
        .o_observation_pre_state_flat(o_observation_pre_state_flat),
        .o_observation_trace_next_flat(o_observation_trace_next_flat),
        .o_observation_meta_flat(o_observation_meta_flat),
        .i_maintenance_tile(maintenance_tile),
        .i_state_maintenance_write_valid(state_write_valid),
        .o_state_maintenance_write_ready(state_write_ready),
        .i_state_maintenance_write_addr(state_write_addr),
        .i_state_maintenance_write_data(state_write_data),
        .i_state_maintenance_write_mask(state_write_mask),
        .i_state_maintenance_read_valid(state_read_valid),
        .o_state_maintenance_read_ready(state_read_ready),
        .i_state_maintenance_read_addr(state_read_addr),
        .o_state_maintenance_result_valid(state_result_valid),
        .i_state_maintenance_result_ready(state_result_ready),
        .o_state_maintenance_result_addr(state_result_addr),
        .o_state_maintenance_read_data(state_read_data),
        .i_trace_maintenance_write_valid(trace_write_valid),
        .o_trace_maintenance_write_ready(trace_write_ready),
        .i_trace_maintenance_write_addr(trace_write_addr),
        .i_trace_maintenance_write_data(trace_write_data),
        .i_trace_maintenance_write_mask(trace_write_mask),
        .i_trace_maintenance_read_valid(trace_read_valid),
        .o_trace_maintenance_read_ready(trace_read_ready),
        .i_trace_maintenance_read_addr(trace_read_addr),
        .o_trace_maintenance_result_valid(trace_result_valid),
        .i_trace_maintenance_result_ready(trace_result_ready),
        .o_trace_maintenance_result_addr(trace_result_addr),
        .o_trace_maintenance_read_data(trace_read_data),
        .o_done(o_done), .o_tile_idle(o_tile_idle),
        .o_idle(array_idle), .o_sticky_error(o_tile_sticky_error)
    );
endmodule

`default_nettype wire
