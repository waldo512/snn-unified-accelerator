`timescale 1ns/1ps
`default_nettype none

// Complete homogeneous H40 tile. A topology walker supplies only addresses,
// contribution controls, and the source factor. The same local weight store,
// destination-stationary accumulator, state store, and LIF lanes serve FC,
// SRNN, and SCNN execution.
module unified_dual_role_h40_weighted_compute_tile_candidate_v1 #(
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int STATE_DEPTH = 1024,
    parameter int STATE_ADDR_WIDTH = $clog2(STATE_DEPTH)
) (
    input  wire logic                          i_clk,
    input  wire logic                          i_rst_n,
    input  wire logic                          i_abort,
    input  wire logic                          i_clear_error,

    input  wire logic                          i_weight_fill_valid,
    output logic                               o_weight_fill_ready,
    input  wire logic                          i_weight_fill_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_fill_addr,
    input  wire logic [199:0]                  i_weight_fill_data,

    input  wire logic                          i_issue_valid,
    output logic                               o_issue_ready,
    input  wire logic                          i_first,
    input  wire logic                          i_last,
    input  wire logic [39:0]                   i_lane_mask,
    input  wire logic [39:0]                   i_enable_a,
    input  wire logic [39:0]                   i_enable_d,
    input  wire logic [40*18-1:0]              i_factor_b_flat,
    input  wire logic [1:0]                    i_mode,
    input  wire logic                          i_scnn_active_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0]  i_weight_addr1,
    input  wire logic [3:0]                    i_weight_subword0,
    input  wire logic [3:0]                    i_weight_subword1,
    input  wire logic [STATE_ADDR_WIDTH-1:0]   i_state_base_addr,
    input  wire logic signed [15:0]            i_leak,
    input  wire logic signed [31:0]            i_threshold,
    input  wire logic                          i_reset_to_zero,

    output logic                               o_group_valid,
    input  wire logic                          i_group_ready,
    output logic [1:0]                         o_group_index,
    output logic [9:0]                         o_group_lane_mask,
    output logic [9:0]                         o_group_range_event_mask,
    output logic [9:0]                         o_group_spike,
    output logic [10*32-1:0]                   o_group_pre_state_flat,
    output logic [10*32-1:0]                   o_group_next_state_flat,
    output logic                               o_done,
    output logic                               o_idle,
    output logic                               o_sticky_error,

    input  wire logic                          i_state_mem_write_valid,
    output logic                               o_state_mem_write_ready,
    input  wire logic [STATE_ADDR_WIDTH-1:0]   i_state_mem_write_addr,
    input  wire logic [319:0]                  i_state_mem_write_data,
    input  wire logic                          i_state_mem_read_valid,
    output logic                               o_state_mem_read_ready,
    input  wire logic [STATE_ADDR_WIDTH-1:0]   i_state_mem_read_addr,
    output logic                               o_state_mem_read_result_valid,
    input  wire logic                          i_state_mem_read_result_ready,
    output logic [STATE_ADDR_WIDTH-1:0]         o_state_mem_read_result_addr,
    output logic [319:0]                       o_state_mem_read_data
);
    logic weight_read_ready;
    logic weight_result_valid;
    logic weight_result_ready;
    logic [40*25-1:0] weight_term0_flat;
    logic [40*25-1:0] weight_term1_flat;
    logic issue_fire;

    logic first_q;
    logic last_q;
    logic [39:0] lane_mask_q;
    logic [39:0] enable_a_q;
    logic [39:0] enable_d_q;
    logic [40*18-1:0] factor_b_flat_q;
    logic [1:0] mode_q;
    logic [STATE_ADDR_WIDTH-1:0] state_base_addr_q;
    logic signed [15:0] leak_q;
    logic signed [31:0] threshold_q;
    logic reset_to_zero_q;

    logic compute_contribution_ready;
    logic compute_idle;
    logic compute_sticky_error;

    assign o_issue_ready = weight_read_ready && !i_abort &&
        !compute_sticky_error;
    assign issue_fire = i_issue_valid && o_issue_ready;

    // Abort drains a pending synchronous RAM response without allowing it to
    // become a contribution. Validity, rather than payload reset, owns data.
    assign weight_result_ready = i_abort || compute_contribution_ready;
    assign o_idle = compute_idle && !weight_result_valid;
    assign o_sticky_error = compute_sticky_error;

    unified_dual_role_weight_store_candidate_v1 #(
        .BANK_DEPTH(WEIGHT_DEPTH),
        .ADDR_WIDTH(WEIGHT_ADDR_WIDTH)
    ) u_weight_store (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_fill_valid(i_weight_fill_valid),
        .o_fill_ready(o_weight_fill_ready),
        .i_fill_bank(i_weight_fill_bank),
        .i_fill_addr(i_weight_fill_addr),
        .i_fill_data(i_weight_fill_data),
        .i_read_valid(i_issue_valid && !i_abort && !compute_sticky_error),
        .o_read_ready(weight_read_ready), .i_mode(i_mode),
        .i_scnn_active_bank(i_scnn_active_bank),
        .i_read_addr0(i_weight_addr0), .i_read_addr1(i_weight_addr1),
        .i_read_subword0(i_weight_subword0),
        .i_read_subword1(i_weight_subword1),
        .o_read_result_valid(weight_result_valid),
        .i_read_result_ready(weight_result_ready),
        .o_term0_flat(weight_term0_flat),
        .o_term1_flat(weight_term1_flat)
    );

    unified_dual_role_h40_compute_tile_candidate_v1 #(
        .STATE_DEPTH(STATE_DEPTH),
        .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH)
    ) u_compute_tile (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_contribution_valid(weight_result_valid && !i_abort),
        .o_contribution_ready(compute_contribution_ready),
        .i_first(first_q), .i_last(last_q), .i_lane_mask(lane_mask_q),
        .i_enable_a(enable_a_q), .i_enable_d(enable_d_q),
        .i_term_a_flat(weight_term0_flat),
        .i_term_d_flat(weight_term1_flat),
        .i_factor_b_flat(factor_b_flat_q), .i_mode(mode_q),
        .i_state_base_addr(state_base_addr_q), .i_leak(leak_q),
        .i_threshold(threshold_q), .i_reset_to_zero(reset_to_zero_q),
        .o_group_valid(o_group_valid), .i_group_ready(i_group_ready),
        .o_group_index(o_group_index),
        .o_group_lane_mask(o_group_lane_mask),
        .o_group_range_event_mask(o_group_range_event_mask),
        .o_group_spike(o_group_spike),
        .o_group_pre_state_flat(o_group_pre_state_flat),
        .o_group_next_state_flat(o_group_next_state_flat),
        .o_done(o_done), .o_idle(compute_idle),
        .o_sticky_error(compute_sticky_error),
        .i_mem_write_valid(i_state_mem_write_valid),
        .o_mem_write_ready(o_state_mem_write_ready),
        .i_mem_write_addr(i_state_mem_write_addr),
        .i_mem_write_data(i_state_mem_write_data),
        .i_mem_read_valid(i_state_mem_read_valid),
        .o_mem_read_ready(o_state_mem_read_ready),
        .i_mem_read_addr(i_state_mem_read_addr),
        .o_mem_read_result_valid(o_state_mem_read_result_valid),
        .i_mem_read_result_ready(i_state_mem_read_result_ready),
        .o_mem_read_result_addr(o_state_mem_read_result_addr),
        .o_mem_read_data(o_state_mem_read_data)
    );

    always_ff @(posedge i_clk) begin
        if (issue_fire) begin
            first_q <= i_first;
            last_q <= i_last;
            lane_mask_q <= i_lane_mask;
            enable_a_q <= i_enable_a;
            enable_d_q <= i_enable_d;
            factor_b_flat_q <= i_factor_b_flat;
            mode_q <= i_mode;
            state_base_addr_q <= i_state_base_addr;
            leak_q <= i_leak;
            threshold_q <= i_threshold;
            reset_to_zero_q <= i_reset_to_zero;
        end
    end

    initial begin
        if (WEIGHT_DEPTH != 1156)
            $error("weighted tile requires the measured 2x200x1156 weight geometry");
    end
endmodule

`default_nettype wire
