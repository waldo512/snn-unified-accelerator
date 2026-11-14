`timescale 1ns/1ps
`default_nettype none

// State/LIF backend for one homogeneous H40 tile. The upstream accumulator
// keeps i_raw_current_flat stable from start acceptance through o_done. Four
// groups reuse ten normalization and LIF lanes. FC/SRNN use one 10x32 state
// word per group; SCNN packs two 10x16 groups into each 20x16 state word.
module unified_dual_role_h40_state_lif_backend_candidate_v1 #(
    parameter int STATE_DEPTH = 1024,
    parameter int STATE_ADDR_WIDTH = $clog2(STATE_DEPTH)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_start_valid,
    output logic                              o_start_ready,
    input  wire logic [1:0]                   i_mode,
    input  wire logic [39:0]                  i_active_lane_mask,
    input  wire logic [STATE_ADDR_WIDTH-1:0]  i_state_base_addr,
    input  wire logic [40*48-1:0]             i_raw_current_flat,
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
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE,
        STATE_JOIN,
        STATE_WAIT_LIF
    } state_t;

    state_t state_q;
    logic [1:0] mode_q;
    logic [1:0] group_q;
    logic [39:0] active_lane_mask_q;
    logic [STATE_ADDR_WIDTH-1:0] state_base_addr_q;
    logic signed [15:0] leak_q;
    logic signed [31:0] threshold_q;
    logic reset_to_zero_q;
    logic [319:0] state_word_q;

    logic [10*48-1:0] selected_raw_current_flat;
    logic [10*32-1:0] selected_state_flat;
    logic [319:0] merged_state_word;
    logic [159:0] packed_scnn_next;
    logic [9:0] selected_lane_mask;
    logic [STATE_ADDR_WIDTH-1:0] group_state_addr;
    logic group_requires_read;
    logic group_requires_write;
    logic start_legal;
    logic [STATE_ADDR_WIDTH:0] start_last_addr;
    logic start_fire;
    logic issue_fire;
    logic join_fire;
    logic group_fire;

    logic normalizer_input_valid;
    logic normalizer_input_ready;
    logic normalizer_output_valid;
    logic normalizer_output_ready;
    logic [1:0] normalized_mode;
    logic [10*32-1:0] normalized_current_flat;
    logic [9:0] normalized_range_event_mask;
    logic normalizer_mode_error;

    logic lif_input_valid;
    logic lif_input_ready;
    logic lif_output_valid;
    logic lif_output_ready;
    logic [9:0] lif_spike;
    logic [10*32-1:0] lif_pre_state_flat;
    logic [10*32-1:0] lif_next_state_flat;

    logic state_write_valid;
    logic state_write_ready;
    logic [STATE_ADDR_WIDTH-1:0] state_write_addr;
    logic [319:0] state_write_data;
    logic state_read_valid;
    logic state_read_ready;
    logic [STATE_ADDR_WIDTH-1:0] state_read_addr;
    logic state_read_result_valid;
    logic state_read_result_ready;
    logic [STATE_ADDR_WIDTH-1:0] state_read_result_addr;
    logic [319:0] state_read_data;

    always_comb begin
        selected_raw_current_flat = '0;
        selected_state_flat = '0;
        packed_scnn_next = '0;
        merged_state_word = state_word_q;
        selected_lane_mask = active_lane_mask_q[group_q*10 +: 10];

        for (int lane = 0; lane < 10; lane++) begin
            selected_raw_current_flat[lane*48 +: 48] =
                i_raw_current_flat[(group_q*10 + lane)*48 +: 48];

            if (mode_q == MODE_SCNN) begin
                if (group_q[0]) begin
                    selected_state_flat[lane*32 +: 32] =
                        {{16{state_word_q[160 + lane*16 + 15]}},
                         state_word_q[160 + lane*16 +: 16]};
                end else begin
                    selected_state_flat[lane*32 +: 32] =
                        {{16{state_read_data[lane*16 + 15]}},
                         state_read_data[lane*16 +: 16]};
                end
                packed_scnn_next[lane*16 +: 16] =
                    selected_lane_mask[lane] ?
                    lif_next_state_flat[lane*32 +: 16] :
                    (group_q[0] ?
                        state_word_q[160 + lane*16 +: 16] :
                        state_word_q[lane*16 +: 16]);
            end else begin
                selected_state_flat[lane*32 +: 32] =
                    state_read_data[lane*32 +: 32];
                merged_state_word[lane*32 +: 32] =
                    selected_lane_mask[lane] ?
                    lif_next_state_flat[lane*32 +: 32] :
                    state_word_q[lane*32 +: 32];
            end
        end

        if (mode_q == MODE_SCNN) begin
            if (group_q[0])
                merged_state_word[319:160] = packed_scnn_next;
            else
                merged_state_word[159:0] = packed_scnn_next;
        end
    end

    always_comb begin
        if (mode_q == MODE_SCNN)
            group_state_addr = state_base_addr_q +
                STATE_ADDR_WIDTH'(group_q[1]);
        else
            group_state_addr = state_base_addr_q +
                STATE_ADDR_WIDTH'(group_q);

        group_requires_read = (mode_q != MODE_SCNN) || !group_q[0];
        group_requires_write = (mode_q != MODE_SCNN) || group_q[0];

        start_last_addr = {1'b0, i_state_base_addr};
        if (i_mode == MODE_SCNN)
            start_last_addr = start_last_addr +
                (STATE_ADDR_WIDTH+1)'(1);
        else
            start_last_addr = start_last_addr +
                (STATE_ADDR_WIDTH+1)'(3);
        start_legal = (i_mode == MODE_FC || i_mode == MODE_SRNN ||
            i_mode == MODE_SCNN) &&
            start_last_addr < (STATE_ADDR_WIDTH+1)'(STATE_DEPTH);
    end

    assign o_idle = state_q == STATE_IDLE &&
        !state_read_result_valid && !normalizer_output_valid &&
        !lif_output_valid;
    assign o_start_ready = o_idle && !o_sticky_error &&
        !i_mem_write_valid && !i_mem_read_valid;
    assign start_fire = i_start_valid && o_start_ready;

    assign o_mem_write_ready = o_idle && !i_start_valid &&
        !i_mem_read_valid && state_write_ready;
    assign o_mem_read_ready = o_idle && !i_start_valid &&
        !i_mem_write_valid && state_read_ready;
    assign o_mem_read_result_valid = state_q == STATE_IDLE &&
        state_read_result_valid;
    assign o_mem_read_result_addr = state_read_result_addr;
    assign o_mem_read_data = state_read_data;

    assign issue_fire = state_q == STATE_ISSUE &&
        normalizer_input_ready &&
        (!group_requires_read || state_read_ready);
    assign normalizer_input_valid = issue_fire;
    assign state_read_valid = (state_q == STATE_ISSUE) ?
        (issue_fire && group_requires_read) :
        (o_idle && !i_start_valid && i_mem_read_valid);
    assign state_read_addr = (state_q == STATE_ISSUE) ?
        group_state_addr : i_mem_read_addr;

    assign lif_input_valid = state_q == STATE_JOIN &&
        normalizer_output_valid &&
        (!group_requires_read || state_read_result_valid);
    assign normalizer_output_ready = state_q == STATE_JOIN &&
        lif_input_ready &&
        (!group_requires_read || state_read_result_valid);
    assign state_read_result_ready = state_q == STATE_JOIN ?
        (group_requires_read && lif_input_ready && normalizer_output_valid) :
        (state_q == STATE_IDLE && i_mem_read_result_ready);
    assign join_fire = lif_input_valid && lif_input_ready;

    assign o_group_valid = state_q == STATE_WAIT_LIF &&
        lif_output_valid && (!group_requires_write || state_write_ready);
    assign o_group_index = group_q;
    assign o_group_lane_mask = selected_lane_mask;
    assign o_group_range_event_mask =
        normalized_range_event_mask & selected_lane_mask;
    assign o_group_spike = lif_spike & selected_lane_mask;
    assign o_group_pre_state_flat = lif_pre_state_flat;
    assign o_group_next_state_flat = lif_next_state_flat;
    assign group_fire = o_group_valid && i_group_ready;
    assign lif_output_ready = state_q == STATE_WAIT_LIF && i_group_ready &&
        (!group_requires_write || state_write_ready);

    assign state_write_valid =
        (state_q == STATE_WAIT_LIF && lif_output_valid && i_group_ready &&
         group_requires_write) ||
        (o_idle && !i_start_valid && !i_mem_read_valid &&
         i_mem_write_valid);
    assign state_write_addr = state_q == STATE_WAIT_LIF ?
        group_state_addr : i_mem_write_addr;
    assign state_write_data = state_q == STATE_WAIT_LIF ?
        merged_state_word : i_mem_write_data;

    unified_dual_role_current_normalizer10_candidate_v1 u_normalizer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(normalizer_input_valid),
        .o_ready(normalizer_input_ready), .i_mode(mode_q),
        .i_raw_current_flat(selected_raw_current_flat),
        .o_valid(normalizer_output_valid),
        .i_ready(normalizer_output_ready), .o_mode(normalized_mode),
        .o_current_flat(normalized_current_flat),
        .o_range_event_mask(normalized_range_event_mask),
        .o_mode_error(normalizer_mode_error)
    );

    unified_dual_role_lif10_arithmetic_candidate_v2 u_lif (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(lif_input_valid), .o_ready(lif_input_ready),
        .i_mode(normalized_mode),
        .i_current_flat(normalized_current_flat),
        .i_state_flat(selected_state_flat), .i_leak(leak_q),
        .i_threshold(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .o_valid(lif_output_valid), .i_ready(lif_output_ready),
        .o_spike(lif_spike),
        .o_pre_state_flat(lif_pre_state_flat),
        .o_next_state_flat(lif_next_state_flat)
    );

    unified_dual_role_state_store_candidate_v3 #(
        .DEPTH(STATE_DEPTH), .ADDR_WIDTH(STATE_ADDR_WIDTH)
    ) u_state_store (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_write_valid(state_write_valid),
        .o_write_ready(state_write_ready),
        .i_write_addr(state_write_addr), .i_write_data(state_write_data),
        .i_read_valid(state_read_valid), .o_read_ready(state_read_ready),
        .i_read_addr(state_read_addr),
        .o_read_result_valid(state_read_result_valid),
        .i_read_result_ready(state_read_result_ready),
        .o_read_result_addr(state_read_result_addr),
        .o_read_data(state_read_data)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            state_q <= STATE_IDLE;
            mode_q <= MODE_FC;
            group_q <= '0;
            active_lane_mask_q <= '0;
            state_base_addr_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            state_word_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && o_idle)
                o_sticky_error <= 1'b0;

            if (start_fire) begin
                if (!start_legal) begin
                    o_sticky_error <= 1'b1;
                end else begin
                    mode_q <= i_mode;
                    group_q <= 2'd0;
                    active_lane_mask_q <= i_active_lane_mask;
                    state_base_addr_q <= i_state_base_addr;
                    leak_q <= i_leak;
                    threshold_q <= i_threshold;
                    reset_to_zero_q <= i_reset_to_zero;
                    state_q <= STATE_ISSUE;
                end
            end

            if (state_q == STATE_ISSUE && issue_fire)
                state_q <= STATE_JOIN;

            if (join_fire) begin
                if (group_requires_read)
                    state_word_q <= state_read_data;
                state_q <= STATE_WAIT_LIF;
                if (normalizer_mode_error)
                    o_sticky_error <= 1'b1;
            end

            if (group_fire) begin
                if (mode_q == MODE_SCNN && !group_q[0])
                    state_word_q <= merged_state_word;

                if (group_q == 2'd3) begin
                    state_q <= STATE_IDLE;
                    group_q <= '0;
                    o_done <= 1'b1;
                end else begin
                    group_q <= group_q + 1'b1;
                    state_q <= STATE_ISSUE;
                end
            end
        end
    end

    initial begin
        if (STATE_DEPTH < 4)
            $error("H40 state/LIF backend requires at least four state words");
    end
endmodule

`default_nettype wire
