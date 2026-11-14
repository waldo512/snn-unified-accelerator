`timescale 1ns/1ps
`default_nettype none

// Computes one pairwise metric row and immediately normalizes it with the
// shared softmax. A single metric primitive is reused for every item in the
// row, so the surrounding scheduler only has to stream concatenated vectors.
module unified_tp_pairwise_probability_row_q10 #(
    parameter int VALUE_WIDTH = 13,
    parameter int ACC_WIDTH = 40,
    parameter int MAX_VECTOR_LENGTH = 400,
    parameter int MAX_ITEMS = 16,
    parameter int ROW_TAG_WIDTH = 4,
    parameter int LENGTH_WIDTH = $clog2(MAX_VECTOR_LENGTH + 1),
    parameter int ITEM_INDEX_WIDTH = (MAX_ITEMS <= 2) ? 1 : $clog2(MAX_ITEMS),
    parameter int ITEM_COUNT_WIDTH = $clog2(MAX_ITEMS + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_start_valid,
    output logic                                  o_start_ready,
    input  logic                                  i_distance_mode,
    input  logic [LENGTH_WIDTH-1:0]               i_vector_length,
    input  logic [ITEM_COUNT_WIDTH-1:0]           i_row_length,
    input  logic [ROW_TAG_WIDTH-1:0]              i_row_tag,
    output logic                                  o_config_error,
    input  logic                                  i_pair_valid,
    output logic                                  o_pair_ready,
    input  logic [VALUE_WIDTH-1:0]                i_value_a_q10,
    input  logic [VALUE_WIDTH-1:0]                i_value_b_q10,
    output logic                                  o_probability_valid,
    input  logic                                  i_probability_ready,
    output logic [ROW_TAG_WIDTH-1:0]              o_probability_row_tag,
    output logic [ITEM_INDEX_WIDTH-1:0]           o_probability_index,
    output logic [16:0]                           o_probability_q016,
    output logic                                  o_probability_last,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_done,
    output logic                                  o_busy
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_START_SOFTMAX,
        STATE_START_METRIC,
        STATE_METRIC,
        STATE_OUTPUT
    } state_t;

    state_t state_q;
    logic distance_mode_q;
    logic [LENGTH_WIDTH-1:0] vector_length_q;
    logic [ITEM_COUNT_WIDTH-1:0] row_length_q;
    logic [ROW_TAG_WIDTH-1:0] row_tag_q;
    logic [ITEM_INDEX_WIDTH-1:0] metric_index_q;
    logic arithmetic_overflow_q;

    logic metric_start_ready;
    logic metric_config_error;
    logic metric_pair_ready;
    logic metric_valid;
    logic metric_ready;
    logic signed [15:0] metric_q6;
    logic metric_arithmetic_overflow;
    logic metric_done;
    logic metric_busy;

    logic softmax_start_ready;
    logic softmax_config_error;
    logic softmax_value_ready;
    logic softmax_probability_valid;
    logic [ITEM_INDEX_WIDTH-1:0] softmax_probability_index;
    logic [16:0] softmax_probability_q016;
    logic softmax_probability_last;
    logic softmax_done;
    logic softmax_busy;

    logic start_fire;
    logic metric_fire;
    logic probability_fire;
    logic metric_last;

    always_comb begin
        o_start_ready = state_q == STATE_IDLE;
        start_fire = i_start_valid && o_start_ready;
        o_pair_ready = (state_q == STATE_METRIC) && metric_pair_ready;
        metric_ready = (state_q == STATE_METRIC) && softmax_value_ready;
        metric_fire = metric_valid && metric_ready;

        o_probability_valid = (state_q == STATE_OUTPUT) &&
            softmax_probability_valid;
        probability_fire = o_probability_valid && i_probability_ready;
        o_probability_row_tag = row_tag_q;
        o_probability_index = softmax_probability_index;
        o_probability_q016 = softmax_probability_q016;
        o_probability_last = softmax_probability_last;
        o_arithmetic_overflow = arithmetic_overflow_q;
        o_busy = state_q != STATE_IDLE;
        metric_last = ({1'b0, metric_index_q} + 1'b1) >= row_length_q;
    end

    unified_tp_pairwise_metric_q10 #(
        .VALUE_WIDTH(VALUE_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MAX_VECTOR_LENGTH(MAX_VECTOR_LENGTH),
        .LENGTH_WIDTH(LENGTH_WIDTH)
    ) u_metric (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start_valid(state_q == STATE_START_METRIC),
        .o_start_ready(metric_start_ready),
        .i_distance_mode(distance_mode_q),
        .i_vector_length(vector_length_q),
        .o_config_error(metric_config_error),
        .i_pair_valid(i_pair_valid && (state_q == STATE_METRIC)),
        .o_pair_ready(metric_pair_ready),
        .i_value_a_q10(i_value_a_q10),
        .i_value_b_q10(i_value_b_q10),
        .o_metric_valid(metric_valid),
        .i_metric_ready(metric_ready),
        .o_metric_q6(metric_q6),
        .o_arithmetic_overflow(metric_arithmetic_overflow),
        .o_done(metric_done),
        .o_busy(metric_busy)
    );

    unified_tp_row_softmax_q6 #(
        .INPUT_WIDTH(16),
        .MAX_ITEMS(MAX_ITEMS),
        .ITEM_INDEX_WIDTH(ITEM_INDEX_WIDTH),
        .ITEM_COUNT_WIDTH(ITEM_COUNT_WIDTH)
    ) u_softmax (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start_valid(state_q == STATE_START_SOFTMAX),
        .o_start_ready(softmax_start_ready),
        .i_row_length(row_length_q),
        .o_config_error(softmax_config_error),
        .i_value_valid((state_q == STATE_METRIC) && metric_valid),
        .o_value_ready(softmax_value_ready),
        .i_value_q6(metric_q6),
        .o_probability_valid(softmax_probability_valid),
        .i_probability_ready(
            (state_q == STATE_OUTPUT) && i_probability_ready
        ),
        .o_probability_index(softmax_probability_index),
        .o_probability_q016(softmax_probability_q016),
        .o_probability_last(softmax_probability_last),
        .o_done(softmax_done),
        .o_busy(softmax_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            distance_mode_q <= 1'b0;
            vector_length_q <= '0;
            row_length_q <= '0;
            row_tag_q <= '0;
            metric_index_q <= '0;
            arithmetic_overflow_q <= 1'b0;
            o_config_error <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_config_error <= 1'b0;
            o_done <= 1'b0;
            if (metric_arithmetic_overflow)
                arithmetic_overflow_q <= 1'b1;
            if (metric_config_error || softmax_config_error)
                o_config_error <= 1'b1;
            if (metric_done && (state_q != STATE_START_METRIC) &&
                (state_q != STATE_OUTPUT))
                o_config_error <= 1'b1;
            if (softmax_done && (state_q != STATE_IDLE))
                o_config_error <= 1'b1;
            if ((state_q == STATE_METRIC) && !metric_busy)
                o_config_error <= 1'b1;
            if (((state_q == STATE_START_METRIC) ||
                 (state_q == STATE_METRIC) ||
                 (state_q == STATE_OUTPUT)) && !softmax_busy)
                o_config_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (start_fire) begin
                        if ((i_vector_length == '0) ||
                            (i_vector_length > LENGTH_WIDTH'(MAX_VECTOR_LENGTH)) ||
                            (i_row_length == '0) ||
                            (i_row_length > ITEM_COUNT_WIDTH'(MAX_ITEMS))) begin
                            o_config_error <= 1'b1;
                        end else begin
                            distance_mode_q <= i_distance_mode;
                            vector_length_q <= i_vector_length;
                            row_length_q <= i_row_length;
                            row_tag_q <= i_row_tag;
                            metric_index_q <= '0;
                            arithmetic_overflow_q <= 1'b0;
                            state_q <= STATE_START_SOFTMAX;
                        end
                    end
                end

                STATE_START_SOFTMAX: begin
                    if (softmax_start_ready)
                        state_q <= STATE_START_METRIC;
                end

                STATE_START_METRIC: begin
                    if (metric_start_ready)
                        state_q <= STATE_METRIC;
                end

                STATE_METRIC: begin
                    if (metric_fire) begin
                        if (metric_last) begin
                            state_q <= STATE_OUTPUT;
                        end else begin
                            metric_index_q <= metric_index_q + 1'b1;
                            state_q <= STATE_START_METRIC;
                        end
                    end
                end

                STATE_OUTPUT: begin
                    if (probability_fire && softmax_probability_last) begin
                        o_done <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
