`timescale 1ns/1ps
`default_nettype none

// SCNN topology adapter around the common 16xH40 physical array. This module
// adds layer traversal and workset/feature handshakes only; it contains no
// SCNN-specific PE, accumulator, LIF, state or trace storage.
module unified_scnn_h40_array16_layer_path_v1 #(
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter int FEATURE_WORD_DEPTH = 8192,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH),
    parameter int PHYSICAL_TILE_COUNT = 16,
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter bit CUT_RECORD_READY = 1'b1
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,
    input  wire logic                           i_start_valid,
    output logic                                o_start_ready,
    input  wire logic [15:0]                    i_in_width,
    input  wire logic [15:0]                    i_in_height,
    input  wire logic [15:0]                    i_in_channels,
    input  wire logic [15:0]                    i_out_width,
    input  wire logic [15:0]                    i_out_height,
    input  wire logic [15:0]                    i_out_channels,
    input  wire logic [7:0]                     i_stride,
    input  wire logic [7:0]                     i_padding,
    input  wire logic [1:0]                     i_pool_mode,
    input  wire logic                           i_initial_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_work_base_addr,
    input  wire logic [15:0]                    i_leak,
    input  wire logic [31:0]                    i_threshold,
    input  wire logic                           i_reset_to_zero,
    input  wire logic [15:0]                    i_trace_decay,
    input  wire logic [19:0]                    i_layer_tag,

    output logic                                o_context_load_valid,
    input  wire logic                           i_context_load_ready,
    output logic                                o_context_load_weights,
    output logic [9:0]                          o_context_output_channel_base,
    output logic [9:0]                          o_context_output_channel_count,
    output logic [31:0]                         o_context_spatial_base,
    output logic [5:0]                          o_context_spatial_count,
    output logic                                o_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_context_work_base_addr,

    output logic                                o_feature_request_valid,
    input  wire logic                           i_feature_request_ready,
    output logic [3:0]                          o_feature_term_valid,
    output logic [39:0]                         o_feature_input_channel_flat,
    output logic [7:0]                          o_feature_kernel_row_flat,
    output logic [7:0]                          o_feature_kernel_col_flat,
    output logic [31:0]                         o_feature_spatial_base,
    output logic [5:0]                          o_feature_spatial_count,
    output logic [15:0]                         o_feature_output_row,
    output logic [15:0]                         o_feature_output_col,
    output logic [3:0]                          o_feature_value_bit,
    input  wire logic                           i_feature_response_valid,
    output logic                                o_feature_response_ready,
    input  wire logic [39:0]                    i_feature_mask0,
    input  wire logic [39:0]                    i_feature_mask1,
    input  wire logic [39:0]                    i_feature_mask2,
    input  wire logic [39:0]                    i_feature_mask3,
    input  wire logic                           i_feature_service_error,

    input  wire logic                           i_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_base_addr,
    input  wire logic [15:0]                    i_feature_write_row_stride_words,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_channel_stride_words,
    output logic                                o_feature_update_valid,
    input  wire logic                           i_feature_update_ready,
    output logic                                o_feature_update_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]       o_feature_update_addr,
    output logic [63:0]                         o_feature_update_data,
    output logic [63:0]                         o_feature_update_mask,
    input  wire logic                           i_feature_update_done,

    output logic                                o_context_release_valid,
    input  wire logic                           i_context_release_ready,
    output logic [9:0]                          o_release_output_channel_base,
    output logic [9:0]                          o_release_output_channel_count,
    output logic [31:0]                         o_release_spatial_base,
    output logic [5:0]                          o_release_spatial_count,
    output logic                                o_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_release_work_base_addr,

    input  wire logic [3:0]                     i_weight_fill_valid,
    output logic [3:0]                          o_weight_fill_ready,
    input  wire logic [7:0]                     i_weight_fill_tile_flat,
    input  wire logic [3:0]                     i_weight_fill_bank,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_weight_fill_addr_flat,
    input  wire logic [799:0]                   i_weight_fill_data_flat,

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

    input  wire logic [3:0]                     i_maintenance_tile,
    input  wire logic                           i_state_maintenance_write_valid,
    output logic                                o_state_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_write_addr,
    input  wire logic [319:0]                   i_state_maintenance_write_data,
    input  wire logic [39:0]                    i_state_maintenance_write_mask,
    input  wire logic                           i_state_maintenance_read_valid,
    output logic                                o_state_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_read_addr,
    output logic                                o_state_maintenance_result_valid,
    input  wire logic                           i_state_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_state_maintenance_result_addr,
    output logic [319:0]                        o_state_maintenance_read_data,
    input  wire logic                           i_trace_maintenance_write_valid,
    output logic                                o_trace_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_write_addr,
    input  wire logic [319:0]                   i_trace_maintenance_write_data,
    input  wire logic [39:0]                    i_trace_maintenance_write_mask,
    input  wire logic                           i_trace_maintenance_read_valid,
    output logic                                o_trace_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_read_addr,
    output logic                                o_trace_maintenance_result_valid,
    input  wire logic                           i_trace_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_trace_maintenance_result_addr,
    output logic [319:0]                        o_trace_maintenance_read_data,

    output logic                                o_busy,
    output logic                                o_done,
    output logic                                o_idle,
    output logic [15:0]                         o_tile_done,
    output logic [15:0]                         o_tile_idle,
    output logic [15:0]                         o_tile_sticky_error,
    output logic                                o_sticky_error
);
    logic command_valid;
    logic command_ready;
    logic [15:0] command_output_channels;
    logic command_first;
    logic command_last;
    logic [5:0] command_spatial_count;
    logic [9:0] command_output_channel_base;
    logic [39:0] command_feature_mask0;
    logic [39:0] command_feature_mask1;
    logic [39:0] command_feature_mask2;
    logic [39:0] command_feature_mask3;
    logic signed [17:0] command_factor_b;
    logic command_active_pair;
    logic [3:0] command_weight_subword0;
    logic [3:0] command_weight_subword1;
    logic [3:0] command_weight_subword2;
    logic [3:0] command_weight_subword3;
    logic [WEIGHT_ADDR_WIDTH-1:0] command_weight_addr0;
    logic [WEIGHT_ADDR_WIDTH-1:0] command_weight_addr1;
    logic [WEIGHT_ADDR_WIDTH-1:0] command_weight_addr2;
    logic [WEIGHT_ADDR_WIDTH-1:0] command_weight_addr3;
    logic [WORK_ADDR_WIDTH-1:0] command_work_base_addr;
    logic [15:0] command_leak;
    logic [31:0] command_threshold;
    logic command_reset_to_zero;
    logic [15:0] command_trace_decay;
    logic [51:0] command_transaction_tag;
    logic command_done;
    logic array_busy;
    logic array_idle;
    logic array_error;
    logic [3:0] array_observation_valid;
    logic [3:0] array_observation_ready;
    logic [15:0] array_observation_tile_flat;
    logic [7:0] array_observation_mode_flat;
    logic [39:0] array_observation_lane_mask_flat;
    logic [39:0] array_observation_range_mask_flat;
    logic [39:0] array_observation_spike_flat;
    logic [1279:0] array_observation_pre_state_flat;
    logic [1279:0] array_observation_trace_next_flat;
    logic [255:0] array_observation_meta_flat;

    unified_scnn_h40_array16_layer_client_v1 #(
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PHYSICAL_TILE_COUNT(PHYSICAL_TILE_COUNT)
    ) u_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(i_start_valid), .o_start_ready(o_start_ready),
        .i_in_width(i_in_width), .i_in_height(i_in_height),
        .i_in_channels(i_in_channels), .i_out_width(i_out_width),
        .i_out_height(i_out_height), .i_out_channels(i_out_channels),
        .i_stride(i_stride), .i_padding(i_padding),
        .i_pool_mode(i_pool_mode),
        .i_initial_active_pair(i_initial_active_pair),
        .i_work_base_addr(i_work_base_addr), .i_leak(i_leak),
        .i_threshold(i_threshold), .i_reset_to_zero(i_reset_to_zero),
        .i_trace_decay(i_trace_decay), .i_layer_tag(i_layer_tag),
        .o_context_load_valid(o_context_load_valid),
        .i_context_load_ready(i_context_load_ready),
        .o_context_load_weights(o_context_load_weights),
        .o_context_output_channel_base(o_context_output_channel_base),
        .o_context_output_channel_count(o_context_output_channel_count),
        .o_context_spatial_base(o_context_spatial_base),
        .o_context_spatial_count(o_context_spatial_count),
        .o_context_active_pair(o_context_active_pair),
        .o_context_work_base_addr(o_context_work_base_addr),
        .o_feature_request_valid(o_feature_request_valid),
        .i_feature_request_ready(i_feature_request_ready),
        .o_feature_term_valid(o_feature_term_valid),
        .o_feature_input_channel_flat(o_feature_input_channel_flat),
        .o_feature_kernel_row_flat(o_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(o_feature_kernel_col_flat),
        .o_feature_spatial_base(o_feature_spatial_base),
        .o_feature_spatial_count(o_feature_spatial_count),
        .o_feature_output_row(o_feature_output_row),
        .o_feature_output_col(o_feature_output_col),
        .o_feature_value_bit(o_feature_value_bit),
        .i_feature_response_valid(i_feature_response_valid),
        .o_feature_response_ready(o_feature_response_ready),
        .i_feature_mask0(i_feature_mask0),
        .i_feature_mask1(i_feature_mask1),
        .i_feature_mask2(i_feature_mask2),
        .i_feature_mask3(i_feature_mask3),
        .i_feature_service_error(i_feature_service_error),
        .i_feature_write_slot(i_feature_write_slot),
        .i_feature_write_base_addr(i_feature_write_base_addr),
        .i_feature_write_row_stride_words(
            i_feature_write_row_stride_words),
        .i_feature_write_channel_stride_words(
            i_feature_write_channel_stride_words),
        .o_feature_update_valid(o_feature_update_valid),
        .i_feature_update_ready(i_feature_update_ready),
        .o_feature_update_slot(o_feature_update_slot),
        .o_feature_update_addr(o_feature_update_addr),
        .o_feature_update_data(o_feature_update_data),
        .o_feature_update_mask(o_feature_update_mask),
        .i_feature_update_done(i_feature_update_done),
        .o_feature_commit_valid(),
        .i_feature_commit_ready(1'b1),
        .o_feature_commit_layer_tag(),
        .o_feature_commit_channel(),
        .o_feature_commit_spatial_base(),
        .o_feature_commit_spatial_count(),
        .o_feature_commit_spike(),
        .o_feature_commit_q8_8_flat(),
        .o_context_release_valid(o_context_release_valid),
        .i_context_release_ready(i_context_release_ready),
        .o_release_output_channel_base(o_release_output_channel_base),
        .o_release_output_channel_count(o_release_output_channel_count),
        .o_release_spatial_base(o_release_spatial_base),
        .o_release_spatial_count(o_release_spatial_count),
        .o_release_active_pair(o_release_active_pair),
        .o_release_work_base_addr(o_release_work_base_addr),
        .o_array_command_valid(command_valid),
        .i_array_command_ready(command_ready),
        .o_array_command_neuron_count(command_output_channels),
        .o_array_command_first(command_first),
        .o_array_command_last(command_last),
        .o_array_command_spatial_count(command_spatial_count),
        .o_array_command_output_channel_base(command_output_channel_base),
        .o_array_command_feature_mask0(command_feature_mask0),
        .o_array_command_feature_mask1(command_feature_mask1),
        .o_array_command_feature_mask2(command_feature_mask2),
        .o_array_command_feature_mask3(command_feature_mask3),
        .o_array_command_factor_b(command_factor_b),
        .o_array_command_active_pair(command_active_pair),
        .o_array_command_weight_subword0(command_weight_subword0),
        .o_array_command_weight_subword1(command_weight_subword1),
        .o_array_command_weight_subword2(command_weight_subword2),
        .o_array_command_weight_subword3(command_weight_subword3),
        .o_array_command_weight_addr0(command_weight_addr0),
        .o_array_command_weight_addr1(command_weight_addr1),
        .o_array_command_weight_addr2(command_weight_addr2),
        .o_array_command_weight_addr3(command_weight_addr3),
        .o_array_command_work_base_addr(command_work_base_addr),
        .o_array_command_leak(command_leak),
        .o_array_command_threshold(command_threshold),
        .o_array_command_reset_to_zero(command_reset_to_zero),
        .o_array_command_trace_decay(command_trace_decay),
        .o_array_command_transaction_tag(command_transaction_tag),
        .i_array_command_done(command_done),
        .i_array_busy(array_busy), .i_array_idle(array_idle),
        .i_array_error(array_error),
        .i_array_observation_valid(array_observation_valid),
        .o_array_observation_ready(array_observation_ready),
        .i_array_observation_tile_flat(array_observation_tile_flat),
        .i_array_observation_mode_flat(array_observation_mode_flat),
        .i_array_observation_lane_mask_flat(
            array_observation_lane_mask_flat),
        .i_array_observation_range_mask_flat(
            array_observation_range_mask_flat),
        .i_array_observation_spike_flat(array_observation_spike_flat),
        .i_array_observation_pre_state_flat(
            array_observation_pre_state_flat),
        .i_array_observation_trace_next_flat(
            array_observation_trace_next_flat),
        .i_array_observation_meta_flat(array_observation_meta_flat),
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
        .o_busy(o_busy), .o_done(o_done), .o_idle(o_idle),
        .o_sticky_error(o_sticky_error)
    );

    unified_binary_p4_h40_array16_path_v1 #(
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .CUT_RECORD_READY(CUT_RECORD_READY)
    ) u_array (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_command_valid(command_valid), .o_command_ready(command_ready),
        .i_mode(2'd2), .i_neuron_count(command_output_channels),
        .i_first(command_first), .i_last(command_last),
        .i_source_valid('0),
        .i_scnn_spatial_count(command_spatial_count),
        .i_scnn_output_channel_base(command_output_channel_base),
        .i_scnn_feature_mask0(command_feature_mask0),
        .i_scnn_feature_mask1(command_feature_mask1),
        .i_scnn_feature_mask2(command_feature_mask2),
        .i_scnn_feature_mask3(command_feature_mask3),
        .i_scnn_factor_b(command_factor_b),
        .i_scnn_active_pair(command_active_pair),
        .i_scnn_weight_subword0(command_weight_subword0),
        .i_scnn_weight_subword1(command_weight_subword1),
        .i_scnn_weight_subword2(command_weight_subword2),
        .i_scnn_weight_subword3(command_weight_subword3),
        .i_scnn_work_base_addr(command_work_base_addr),
        .i_weight_addr0(command_weight_addr0),
        .i_weight_addr1(command_weight_addr1),
        .i_weight_addr2(command_weight_addr2),
        .i_weight_addr3(command_weight_addr3),
        .i_leak(command_leak), .i_threshold(command_threshold),
        .i_reset_to_zero(command_reset_to_zero),
        .i_trace_decay(command_trace_decay),
        .i_transaction_tag(command_transaction_tag),
        .i_weight_fill_valid(i_weight_fill_valid),
        .o_weight_fill_ready(o_weight_fill_ready),
        .i_weight_fill_tile_flat(i_weight_fill_tile_flat),
        .i_weight_fill_bank(i_weight_fill_bank),
        .i_weight_fill_addr_flat(i_weight_fill_addr_flat),
        .i_weight_fill_data_flat(i_weight_fill_data_flat),
        .o_observation_valid(array_observation_valid),
        .i_observation_ready(array_observation_ready),
        .o_observation_tile_flat(array_observation_tile_flat),
        .o_observation_mode_flat(array_observation_mode_flat),
        .o_observation_lane_mask_flat(array_observation_lane_mask_flat),
        .o_observation_range_mask_flat(array_observation_range_mask_flat),
        .o_observation_spike_flat(array_observation_spike_flat),
        .o_observation_pre_state_flat(array_observation_pre_state_flat),
        .o_observation_trace_next_flat(
            array_observation_trace_next_flat),
        .o_observation_meta_flat(array_observation_meta_flat),
        .i_maintenance_tile(i_maintenance_tile),
        .i_state_maintenance_write_valid(i_state_maintenance_write_valid),
        .o_state_maintenance_write_ready(o_state_maintenance_write_ready),
        .i_state_maintenance_write_addr(i_state_maintenance_write_addr),
        .i_state_maintenance_write_data(i_state_maintenance_write_data),
        .i_state_maintenance_write_mask(i_state_maintenance_write_mask),
        .i_state_maintenance_read_valid(i_state_maintenance_read_valid),
        .o_state_maintenance_read_ready(o_state_maintenance_read_ready),
        .i_state_maintenance_read_addr(i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(o_state_maintenance_result_valid),
        .i_state_maintenance_result_ready(i_state_maintenance_result_ready),
        .o_state_maintenance_result_addr(o_state_maintenance_result_addr),
        .o_state_maintenance_read_data(o_state_maintenance_read_data),
        .i_trace_maintenance_write_valid(i_trace_maintenance_write_valid),
        .o_trace_maintenance_write_ready(o_trace_maintenance_write_ready),
        .i_trace_maintenance_write_addr(i_trace_maintenance_write_addr),
        .i_trace_maintenance_write_data(i_trace_maintenance_write_data),
        .i_trace_maintenance_write_mask(i_trace_maintenance_write_mask),
        .i_trace_maintenance_read_valid(i_trace_maintenance_read_valid),
        .o_trace_maintenance_read_ready(o_trace_maintenance_read_ready),
        .i_trace_maintenance_read_addr(i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(o_trace_maintenance_result_valid),
        .i_trace_maintenance_result_ready(i_trace_maintenance_result_ready),
        .o_trace_maintenance_result_addr(o_trace_maintenance_result_addr),
        .o_trace_maintenance_read_data(o_trace_maintenance_read_data),
        .o_busy(array_busy), .o_done(command_done), .o_idle(array_idle),
        .o_tile_done(o_tile_done), .o_tile_idle(o_tile_idle),
        .o_tile_sticky_error(o_tile_sticky_error),
        .o_sticky_error(array_error)
    );

endmodule

`default_nettype wire
