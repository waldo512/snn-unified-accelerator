`timescale 1ns/1ps
`default_nettype none

// Shared Q10 pairwise metric primitive. Dot mode emits a positive Q6
// similarity; distance mode emits a negative Q6 Euclidean distance using a
// digit-by-digit integer square root over the accumulated squared difference.
module unified_tp_pairwise_metric_q10 #(
    parameter int VALUE_WIDTH = 13,
    parameter int ACC_WIDTH = 40,
    parameter int MAX_VECTOR_LENGTH = 400,
    parameter int LENGTH_WIDTH = $clog2(MAX_VECTOR_LENGTH + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_start_valid,
    output logic                                  o_start_ready,
    input  logic                                  i_distance_mode,
    input  logic [LENGTH_WIDTH-1:0]               i_vector_length,
    output logic                                  o_config_error,
    input  logic                                  i_pair_valid,
    output logic                                  o_pair_ready,
    input  logic [VALUE_WIDTH-1:0]                i_value_a_q10,
    input  logic [VALUE_WIDTH-1:0]                i_value_b_q10,
    output logic                                  o_metric_valid,
    input  logic                                  i_metric_ready,
    output logic signed [15:0]                    o_metric_q6,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_done,
    output logic                                  o_busy
);

    localparam int PRODUCT_WIDTH = VALUE_WIDTH * 2;
    localparam int DIFFERENCE_WIDTH = VALUE_WIDTH + 1;
    localparam int SQUARE_WIDTH = DIFFERENCE_WIDTH * 2;
    localparam int ROOT_WIDTH = (ACC_WIDTH + 1) / 2;
    localparam int SQRT_ITERATIONS = ROOT_WIDTH;
    localparam int ITERATION_WIDTH = (SQRT_ITERATIONS <= 2) ? 1 :
        $clog2(SQRT_ITERATIONS);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_ACCUMULATE,
        STATE_SQRT,
        STATE_OUTPUT
    } state_t;

    state_t state_q;
    logic distance_mode_q;
    logic [LENGTH_WIDTH-1:0] vector_length_q;
    logic [LENGTH_WIDTH-1:0] pair_count_q;
    logic [ACC_WIDTH-1:0] accumulator_q;
    logic signed [15:0] metric_q;
    logic [ACC_WIDTH-1:0] radicand_q;
    logic [ROOT_WIDTH+1:0] remainder_q;
    logic [ROOT_WIDTH-1:0] root_q;
    logic [ITERATION_WIDTH-1:0] sqrt_iteration_q;
    logic accumulation_overflow_q;

    logic start_fire;
    logic pair_fire;
    logic metric_fire;
    logic pair_last;
    logic [PRODUCT_WIDTH-1:0] dot_product;
    logic signed [DIFFERENCE_WIDTH-1:0] difference;
    logic signed [SQUARE_WIDTH-1:0] difference_square_signed;
    logic [SQUARE_WIDTH-1:0] difference_square;
    logic [ACC_WIDTH-1:0] contribution;
    logic [ACC_WIDTH:0] accumulator_extended;
    logic accumulator_overflow;
    logic [ACC_WIDTH-1:0] accumulator_next;
    logic [ACC_WIDTH:0] dot_rounded;
    logic [ACC_WIDTH-1:0] dot_q6;

    logic [ROOT_WIDTH+3:0] remainder_shifted;
    logic [ROOT_WIDTH+3:0] sqrt_trial;
    logic [ROOT_WIDTH+1:0] remainder_next;
    logic [ROOT_WIDTH-1:0] root_next;
    logic sqrt_last;
    logic [ROOT_WIDTH:0] nearest_root;
    logic [ROOT_WIDTH:0] distance_rounded;
    logic [ROOT_WIDTH:0] distance_q6;
    logic metric_clamp_overflow;
    logic signed [15:0] metric_next;

    always_comb begin
        o_start_ready = state_q == STATE_IDLE;
        start_fire = i_start_valid && o_start_ready;
        o_pair_ready = state_q == STATE_ACCUMULATE;
        pair_fire = i_pair_valid && o_pair_ready;
        o_metric_valid = state_q == STATE_OUTPUT;
        metric_fire = o_metric_valid && i_metric_ready;
        o_metric_q6 = metric_q;
        o_busy = state_q != STATE_IDLE;
        pair_last = (pair_count_q + 1'b1) >= vector_length_q;

        dot_product = i_value_a_q10 * i_value_b_q10;
        difference = $signed({1'b0, i_value_a_q10}) -
            $signed({1'b0, i_value_b_q10});
        difference_square_signed = difference * difference;
        difference_square = $unsigned(difference_square_signed);
        if (distance_mode_q)
            contribution = ACC_WIDTH'(difference_square);
        else
            contribution = ACC_WIDTH'(dot_product);
        accumulator_extended = {1'b0, accumulator_q} + {1'b0, contribution};
        accumulator_overflow = accumulator_extended[ACC_WIDTH];
        accumulator_next = accumulator_overflow ? {ACC_WIDTH{1'b1}} :
            accumulator_extended[ACC_WIDTH-1:0];

        dot_rounded = {1'b0, accumulator_next} + (ACC_WIDTH+1)'(8192);
        dot_q6 = ACC_WIDTH'(dot_rounded >> 14);

        remainder_shifted = {
            remainder_q, radicand_q[ACC_WIDTH-1 -: 2]
        };
        sqrt_trial = {{2{1'b0}}, root_q, 2'b01};
        root_next = root_q << 1;
        if (remainder_shifted >= sqrt_trial) begin
            remainder_next = (ROOT_WIDTH+2)'(remainder_shifted - sqrt_trial);
            root_next[0] = 1'b1;
        end else begin
            remainder_next = remainder_shifted[ROOT_WIDTH+1:0];
        end
        sqrt_last = sqrt_iteration_q == ITERATION_WIDTH'(SQRT_ITERATIONS - 1);
        nearest_root = {1'b0, root_next};
        if (remainder_next > (ROOT_WIDTH+2)'(root_next))
            nearest_root = {1'b0, root_next} + 1'b1;
        distance_rounded = nearest_root + (ROOT_WIDTH+1)'(8);
        distance_q6 = distance_rounded >> 4;

        metric_clamp_overflow = 1'b0;
        metric_next = '0;
        if (distance_mode_q) begin
            if (distance_q6 >= (ROOT_WIDTH+1)'(32768)) begin
                metric_next = -16'sd32768;
                metric_clamp_overflow = 1'b1;
            end else begin
                metric_next = -$signed(distance_q6[15:0]);
            end
        end else begin
            if (dot_q6 > ACC_WIDTH'(32767)) begin
                metric_next = 16'sd32767;
                metric_clamp_overflow = 1'b1;
            end else begin
                metric_next = $signed(dot_q6[15:0]);
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            distance_mode_q <= 1'b0;
            vector_length_q <= '0;
            pair_count_q <= '0;
            accumulator_q <= '0;
            metric_q <= '0;
            radicand_q <= '0;
            remainder_q <= '0;
            root_q <= '0;
            sqrt_iteration_q <= '0;
            accumulation_overflow_q <= 1'b0;
            o_config_error <= 1'b0;
            o_arithmetic_overflow <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_config_error <= 1'b0;
            o_arithmetic_overflow <= 1'b0;
            o_done <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (start_fire) begin
                        if ((i_vector_length == '0) ||
                            (i_vector_length > LENGTH_WIDTH'(MAX_VECTOR_LENGTH))) begin
                            o_config_error <= 1'b1;
                        end else begin
                            distance_mode_q <= i_distance_mode;
                            vector_length_q <= i_vector_length;
                            pair_count_q <= '0;
                            accumulator_q <= '0;
                            accumulation_overflow_q <= 1'b0;
                            state_q <= STATE_ACCUMULATE;
                        end
                    end
                end

                STATE_ACCUMULATE: begin
                    if (pair_fire) begin
                        accumulation_overflow_q <=
                            accumulation_overflow_q || accumulator_overflow;
                        if (pair_last) begin
                            if (distance_mode_q) begin
                                radicand_q <= accumulator_next;
                                remainder_q <= '0;
                                root_q <= '0;
                                sqrt_iteration_q <= '0;
                                state_q <= STATE_SQRT;
                            end else begin
                                metric_q <= metric_next;
                                o_arithmetic_overflow <= accumulation_overflow_q ||
                                    accumulator_overflow || metric_clamp_overflow;
                                state_q <= STATE_OUTPUT;
                            end
                        end else begin
                            accumulator_q <= accumulator_next;
                            pair_count_q <= pair_count_q + 1'b1;
                        end
                    end
                end

                STATE_SQRT: begin
                    remainder_q <= remainder_next;
                    root_q <= root_next;
                    radicand_q <= radicand_q << 2;
                    if (sqrt_last) begin
                        metric_q <= metric_next;
                        o_arithmetic_overflow <= accumulation_overflow_q ||
                            metric_clamp_overflow;
                        state_q <= STATE_OUTPUT;
                    end else begin
                        sqrt_iteration_q <= sqrt_iteration_q + 1'b1;
                    end
                end

                STATE_OUTPUT: begin
                    if (metric_fire) begin
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
