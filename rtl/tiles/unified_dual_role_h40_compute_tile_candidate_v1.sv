`timescale 1ns/1ps
`default_nettype none

// Homogeneous H40 compute kernel. Topology walkers provide contribution
// operands; the tile accumulates forty destination-stationary currents and
// then hands the held DSP results to one shared 10-lane state/LIF backend.
module unified_dual_role_h40_compute_tile_candidate_v1 #(
    parameter int STATE_DEPTH = 1024,
    parameter int STATE_ADDR_WIDTH = $clog2(STATE_DEPTH)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_contribution_valid,
    output logic                              o_contribution_ready,
    input  wire logic                         i_first,
    input  wire logic                         i_last,
    input  wire logic [39:0]                  i_lane_mask,
    input  wire logic [39:0]                  i_enable_a,
    input  wire logic [39:0]                  i_enable_d,
    input  wire logic [40*25-1:0]             i_term_a_flat,
    input  wire logic [40*25-1:0]             i_term_d_flat,
    input  wire logic [40*18-1:0]             i_factor_b_flat,
    input  wire logic [1:0]                   i_mode,
    input  wire logic [STATE_ADDR_WIDTH-1:0]  i_state_base_addr,
    input  wire logic signed [15:0]           i_leak,
    input  wire logic signed [31:0]           i_threshold,
    input  wire logic                         i_reset_to_zero,

    output logic                              o_group_valid,
    input  wire logic                         i_group_ready,
    output logic [1:0]                        o_group_index,
    output logic [9:0]                        o_group_lane_mask,
    output logic [9:0]                        o_group_range_event_mask,
    output logic [9:0]                        o_group_spike,
    output logic [10*32-1:0]                  o_group_pre_state_flat,
    output logic [10*32-1:0]                  o_group_next_state_flat,
    output logic                              o_done,
    output logic                              o_idle,
    output logic                              o_sticky_error,

    input  wire logic                         i_mem_write_valid,
    output logic                              o_mem_write_ready,
    input  wire logic [STATE_ADDR_WIDTH-1:0]  i_mem_write_addr,
    input  wire logic [319:0]                 i_mem_write_data,
    input  wire logic                         i_mem_read_valid,
    output logic                              o_mem_read_ready,
    input  wire logic [STATE_ADDR_WIDTH-1:0]  i_mem_read_addr,
    output logic                              o_mem_read_result_valid,
    input  wire logic                         i_mem_read_result_ready,
    output logic [STATE_ADDR_WIDTH-1:0]        o_mem_read_result_addr,
    output logic [319:0]                      o_mem_read_data
);
    logic transaction_active_q;
    logic contribution_open_q;
    logic backend_started_q;
    logic protocol_error_q;
    logic [1:0] mode_q;
    logic [STATE_ADDR_WIDTH-1:0] state_base_addr_q;
    logic signed [15:0] leak_q;
    logic signed [31:0] threshold_q;
    logic reset_to_zero_q;

    logic contribution_sequence_legal;
    logic contribution_fire;
    logic accumulator_input_valid;
    logic accumulator_ready;
    logic accumulator_result_valid;
    logic accumulator_result_ready;
    logic [39:0] accumulator_result_lane_mask;
    logic [40*48-1:0] accumulator_result_flat;

    logic backend_start_valid;
    logic backend_start_ready;
    logic backend_done;
    logic backend_idle;
    logic backend_sticky_error;
    logic backend_mem_write_valid;
    logic backend_mem_write_ready;
    logic backend_mem_read_valid;
    logic backend_mem_read_ready;

    assign contribution_sequence_legal =
        (!transaction_active_q && i_first) ||
        (transaction_active_q && contribution_open_q && !i_first);
    assign o_contribution_ready = accumulator_ready &&
        !o_sticky_error && !i_mem_write_valid && !i_mem_read_valid;
    assign contribution_fire = i_contribution_valid &&
        o_contribution_ready;
    assign accumulator_input_valid = contribution_fire &&
        contribution_sequence_legal;

    assign backend_start_valid = accumulator_result_valid &&
        !backend_started_q;
    assign accumulator_result_ready = backend_done && backend_started_q;

    assign o_idle = !transaction_active_q && !contribution_open_q &&
        !backend_started_q && !accumulator_result_valid && backend_idle;
    assign o_sticky_error = protocol_error_q || backend_sticky_error;

    assign backend_mem_write_valid = i_mem_write_valid &&
        !transaction_active_q && !accumulator_result_valid &&
        !backend_started_q;
    assign backend_mem_read_valid = i_mem_read_valid &&
        !transaction_active_q && !accumulator_result_valid &&
        !backend_started_q;
    assign o_mem_write_ready = backend_mem_write_ready &&
        !transaction_active_q && !accumulator_result_valid &&
        !backend_started_q;
    assign o_mem_read_ready = backend_mem_read_ready &&
        !transaction_active_q && !accumulator_result_valid &&
        !backend_started_q;

    unified_dual_role_h40_tile_candidate_v3 u_accumulator (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(accumulator_input_valid), .o_ready(accumulator_ready),
        .i_first(i_first), .i_last(i_last), .i_lane_mask(i_lane_mask),
        .i_enable_a(i_enable_a), .i_enable_d(i_enable_d),
        .i_term_a_flat(i_term_a_flat), .i_term_d_flat(i_term_d_flat),
        .i_factor_b_flat(i_factor_b_flat),
        .o_result_valid(accumulator_result_valid),
        .i_result_ready(accumulator_result_ready),
        .o_result_lane_mask(accumulator_result_lane_mask),
        .o_result_flat(accumulator_result_flat)
    );

    unified_dual_role_h40_state_lif_backend_candidate_v1 #(
        .STATE_DEPTH(STATE_DEPTH),
        .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH)
    ) u_state_lif_backend (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(backend_start_valid),
        .o_start_ready(backend_start_ready), .i_mode(mode_q),
        .i_active_lane_mask(accumulator_result_lane_mask),
        .i_state_base_addr(state_base_addr_q),
        .i_raw_current_flat(accumulator_result_flat), .i_leak(leak_q),
        .i_threshold(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .o_group_valid(o_group_valid), .i_group_ready(i_group_ready),
        .o_group_index(o_group_index),
        .o_group_lane_mask(o_group_lane_mask),
        .o_group_range_event_mask(o_group_range_event_mask),
        .o_group_spike(o_group_spike),
        .o_group_pre_state_flat(o_group_pre_state_flat),
        .o_group_next_state_flat(o_group_next_state_flat),
        .o_done(backend_done), .o_idle(backend_idle),
        .o_sticky_error(backend_sticky_error),
        .i_mem_write_valid(backend_mem_write_valid),
        .o_mem_write_ready(backend_mem_write_ready),
        .i_mem_write_addr(i_mem_write_addr),
        .i_mem_write_data(i_mem_write_data),
        .i_mem_read_valid(backend_mem_read_valid),
        .o_mem_read_ready(backend_mem_read_ready),
        .i_mem_read_addr(i_mem_read_addr),
        .o_mem_read_result_valid(o_mem_read_result_valid),
        .i_mem_read_result_ready(i_mem_read_result_ready),
        .o_mem_read_result_addr(o_mem_read_result_addr),
        .o_mem_read_data(o_mem_read_data)
    );

    assign o_done = backend_done;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            transaction_active_q <= 1'b0;
            contribution_open_q <= 1'b0;
            backend_started_q <= 1'b0;
            protocol_error_q <= 1'b0;
            mode_q <= 2'd0;
            state_base_addr_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
        end else begin
            if (i_clear_error && o_idle)
                protocol_error_q <= 1'b0;

            if (i_abort) begin
                transaction_active_q <= 1'b0;
                contribution_open_q <= 1'b0;
                backend_started_q <= 1'b0;
            end else begin
                if (contribution_fire && !contribution_sequence_legal)
                    protocol_error_q <= 1'b1;

                if (accumulator_input_valid) begin
                    if (i_first) begin
                        transaction_active_q <= 1'b1;
                        mode_q <= i_mode;
                        state_base_addr_q <= i_state_base_addr;
                        leak_q <= i_leak;
                        threshold_q <= i_threshold;
                        reset_to_zero_q <= i_reset_to_zero;
                    end
                    contribution_open_q <= !i_last;
                end

                if (backend_start_valid && backend_start_ready)
                    backend_started_q <= 1'b1;

                if (backend_done) begin
                    transaction_active_q <= 1'b0;
                    contribution_open_q <= 1'b0;
                    backend_started_q <= 1'b0;
                end
            end

            if (backend_sticky_error)
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
