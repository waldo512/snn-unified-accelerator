`timescale 1ns/1ps
`default_nettype none

// State-free SRNN topology adapter for the shared H40 array. Persistent
// membrane/accumulator/spike state remains exclusively inside the array.
module unified_srnn_tile_major_frontend_v1 #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer INPUT_MAX = 2312,
    parameter integer TILES = 12,
    parameter integer TILE_NEURONS = 40,
    parameter integer ENGINES = 12,
    parameter integer LANES = 8,
    parameter integer VERSION_WIDTH = 16,
    parameter integer REQUEST_ID_WIDTH = 16
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_owner_enable,
    input  logic                                      i_clear_error,
    input  logic                                      i_state_reset,

    input  logic                                      i_start_valid,
    output logic                                      o_start_ready,
    input  logic [12:0]                               i_input_dim,
    input  logic [9:0]                                i_active_neurons,
    input  logic [31:0]                               i_w0_base,
    input  logic [31:0]                               i_recurrent_base,
    input  logic                                      i_recurrent_enable,
    input  logic signed [15:0]                       i_leak_q15,
    input  logic signed [31:0]                       i_threshold,
    input  logic [4:0]                                i_current_left_shift,
    input  logic                                      i_reset_to_zero,
    input  logic [VERSION_WIDTH-1:0]                  i_version,

    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [12:0]                               i_event_addr,
    input  logic signed [7:0]                        i_event_data,
    input  logic                                      i_timestep_last,
    input  logic                                      i_sample_last,

    output logic                                      o_weight_req_valid,
    input  logic                                      i_weight_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_weight_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_weight_req_id,
    output logic                                      o_weight_req_recurrent,
    output logic [31:0]                               o_weight_req_row_base,
    output logic [9:0]                                o_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_weight_req_lane_mask,
    input  logic                                      i_weight_rsp_valid,
    output logic                                      o_weight_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_weight_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_weight_rsp_id,
    input  logic [9:0]                                i_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_weight_rsp_flat,

    output logic                                      o_array_valid,
    input  logic                                      i_array_ready,
    output logic [TILES-1:0]                          o_array_engine_mask,
    output logic [LANES-1:0]                          o_array_lane_mask,
    output logic [LANES*8-1:0]                        o_array_int8_lhs_flat,
    output logic [TILES*LANES*8-1:0]                  o_array_int8_rhs_flat,
    output logic                                      o_array_first,
    output logic                                      o_array_last,
    output logic [2:0]                                o_array_local_word,
    input  logic                                      i_array_result_valid,
    output logic                                      o_array_result_ready,
    input  logic [1:0]                                i_array_result_owner,
    input  logic                                      i_array_result_mode_q8_8,
    input  logic [TILES-1:0]                          i_array_result_engine_mask,
    input  logic [LANES-1:0]                          i_array_result_lane_mask,

    output logic                                      o_state_clear_spikes,
    output logic                                      o_state_clear_word_valid,
    output logic [2:0]                                o_state_clear_local_word,
    output logic [9:0]                                o_state_active_neurons,
    output logic                                      o_state_lif_begin,
    output logic                                      o_state_lif_word_valid,
    output logic [2:0]                                o_state_lif_local_word,
    output logic signed [15:0]                       o_state_leak_q15,
    output logic signed [31:0]                       o_state_threshold,
    output logic [4:0]                                o_state_current_left_shift,
    output logic                                      o_state_reset_to_zero,
    output logic                                      o_state_commit_timestep,
    input  logic [HIDDEN_MAX-1:0]                     i_state_previous_spikes,
    input  logic [HIDDEN_MAX-1:0]                     i_state_current_spikes,

    output logic                                      o_timestep_valid,
    input  logic                                      i_timestep_ready,
    output logic [VERSION_WIDTH-1:0]                  o_timestep_version,
    output logic                                      o_timestep_sample_last,
    output logic [HIDDEN_MAX-1:0]                     o_timestep_spikes,
    output logic [HIDDEN_MAX-1:0]                     o_previous_spikes,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    logic row_req_valid;
    logic row_req_ready;
    logic [VERSION_WIDTH-1:0] row_req_version;
    logic [REQUEST_ID_WIDTH-1:0] row_req_id;
    logic row_req_recurrent;
    logic [31:0] row_req_base;
    logic [9:0] row_req_destination_count;
    logic signed [7:0] row_req_activation;
    logic row_done_valid;
    logic row_done_ready;
    logic [VERSION_WIDTH-1:0] row_done_version;
    logic [REQUEST_ID_WIDTH-1:0] row_done_id;
    logic controller_busy;
    logic controller_idle;
    logic controller_error;
    logic [7:0] controller_error_code;
    logic executor_busy;
    logic executor_idle;
    logic executor_error;
    logic [7:0] executor_error_code;

    assign o_busy = controller_busy || executor_busy;
    assign o_idle = controller_idle && executor_idle;
    assign o_sticky_error = controller_error || executor_error;
    assign o_error_code = controller_error ? controller_error_code :
        (executor_error ? (8'h80 | executor_error_code) : 8'h00);

    unified_srnn_tile_major_controller_v1 #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .INPUT_MAX(INPUT_MAX),
        .TILE_NEURONS(TILE_NEURONS),
        .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH)
    ) u_controller (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_owner_enable(i_owner_enable),
        .i_clear_error(i_clear_error),
        .i_state_reset(i_state_reset),
        .i_start_valid(i_start_valid),
        .o_start_ready(o_start_ready),
        .i_input_dim(i_input_dim),
        .i_active_neurons(i_active_neurons),
        .i_w0_base(i_w0_base),
        .i_recurrent_base(i_recurrent_base),
        .i_recurrent_enable(i_recurrent_enable),
        .i_leak_q15(i_leak_q15),
        .i_threshold(i_threshold),
        .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero),
        .i_version(i_version),
        .i_event_valid(i_event_valid),
        .o_event_ready(o_event_ready),
        .i_event_addr(i_event_addr),
        .i_event_data(i_event_data),
        .i_timestep_last(i_timestep_last),
        .i_sample_last(i_sample_last),
        .o_row_req_valid(row_req_valid),
        .i_row_req_ready(row_req_ready),
        .o_row_req_version(row_req_version),
        .o_row_req_id(row_req_id),
        .o_row_req_recurrent(row_req_recurrent),
        .o_row_req_base(row_req_base),
        .o_row_req_destination_count(row_req_destination_count),
        .o_row_req_activation(row_req_activation),
        .i_row_done_valid(row_done_valid),
        .o_row_done_ready(row_done_ready),
        .i_row_done_version(row_done_version),
        .i_row_done_id(row_done_id),
        .o_state_clear_spikes(o_state_clear_spikes),
        .o_state_clear_word_valid(o_state_clear_word_valid),
        .o_state_clear_local_word(o_state_clear_local_word),
        .o_state_active_neurons(o_state_active_neurons),
        .o_state_lif_begin(o_state_lif_begin),
        .o_state_lif_word_valid(o_state_lif_word_valid),
        .o_state_lif_local_word(o_state_lif_local_word),
        .o_state_leak_q15(o_state_leak_q15),
        .o_state_threshold(o_state_threshold),
        .o_state_current_left_shift(o_state_current_left_shift),
        .o_state_reset_to_zero(o_state_reset_to_zero),
        .o_state_commit_timestep(o_state_commit_timestep),
        .i_state_previous_spikes(i_state_previous_spikes),
        .i_state_current_spikes(i_state_current_spikes),
        .o_timestep_valid(o_timestep_valid),
        .i_timestep_ready(i_timestep_ready),
        .o_timestep_version(o_timestep_version),
        .o_timestep_sample_last(o_timestep_sample_last),
        .o_timestep_spikes(o_timestep_spikes),
        .o_previous_spikes(o_previous_spikes),
        .o_busy(controller_busy),
        .o_idle(controller_idle),
        .o_sticky_error(controller_error),
        .o_error_code(controller_error_code)
    );

    unified_srnn_tile_major_row_executor_v1 #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .TILES(TILES),
        .TILE_NEURONS(TILE_NEURONS),
        .ENGINES(ENGINES),
        .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH)
    ) u_executor (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_row_req_valid(row_req_valid),
        .o_row_req_ready(row_req_ready),
        .i_row_req_version(row_req_version),
        .i_row_req_id(row_req_id),
        .i_row_req_recurrent(row_req_recurrent),
        .i_row_req_base(row_req_base),
        .i_row_req_destination_count(row_req_destination_count),
        .i_row_req_activation(row_req_activation),
        .o_weight_req_valid(o_weight_req_valid),
        .i_weight_req_ready(i_weight_req_ready),
        .o_weight_req_version(o_weight_req_version),
        .o_weight_req_id(o_weight_req_id),
        .o_weight_req_recurrent(o_weight_req_recurrent),
        .o_weight_req_row_base(o_weight_req_row_base),
        .o_weight_req_destination_base(o_weight_req_destination_base),
        .o_weight_req_engine_mask(o_weight_req_engine_mask),
        .o_weight_req_lane_mask(o_weight_req_lane_mask),
        .i_weight_rsp_valid(i_weight_rsp_valid),
        .o_weight_rsp_ready(o_weight_rsp_ready),
        .i_weight_rsp_version(i_weight_rsp_version),
        .i_weight_rsp_id(i_weight_rsp_id),
        .i_weight_rsp_destination_base(i_weight_rsp_destination_base),
        .i_weight_rsp_engine_mask(i_weight_rsp_engine_mask),
        .i_weight_rsp_flat(i_weight_rsp_flat),
        .o_array_valid(o_array_valid),
        .i_array_ready(i_array_ready),
        .o_array_engine_mask(o_array_engine_mask),
        .o_array_lane_mask(o_array_lane_mask),
        .o_array_int8_lhs_flat(o_array_int8_lhs_flat),
        .o_array_int8_rhs_flat(o_array_int8_rhs_flat),
        .o_array_first(o_array_first),
        .o_array_last(o_array_last),
        .o_array_local_word(o_array_local_word),
        .i_array_result_valid(i_array_result_valid),
        .o_array_result_ready(o_array_result_ready),
        .i_array_result_owner(i_array_result_owner),
        .i_array_result_mode_q8_8(i_array_result_mode_q8_8),
        .i_array_result_engine_mask(i_array_result_engine_mask),
        .i_array_result_lane_mask(i_array_result_lane_mask),
        .o_row_done_valid(row_done_valid),
        .i_row_done_ready(row_done_ready),
        .o_row_done_version(row_done_version),
        .o_row_done_id(row_done_id),
        .o_busy(executor_busy),
        .o_idle(executor_idle),
        .o_sticky_error(executor_error),
        .o_error_code(executor_error_code)
    );
endmodule

`default_nettype wire
