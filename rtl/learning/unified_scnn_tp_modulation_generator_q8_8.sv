`timescale 1ns/1ps
`default_nettype none

// Serial SCNN TP modulation generator for equation M = softmax(z) - softmax(y).
// Trace memories are written before i_start_valid and read synchronously during
// computation so the full Conv8 configuration can map to block memories.
module unified_scnn_tp_modulation_generator_q8_8 #(
    parameter int BATCH_SIZE = 2,
    parameter int VECTOR_SIZE = 2048,
    parameter int SPATIAL_SIZE = 4,
    parameter int PREV_CHANNELS = 512,
    parameter int PREV_SPATIAL_SIZE = 16,
    parameter bit TRACE_STREAM_SPATIAL_MAJOR = 1'b0,
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_WIDTH = 8,
    parameter int ACC_WIDTH = 48,
    parameter int SAMPLE_WIDTH = (BATCH_SIZE <= 2) ? 1 : $clog2(BATCH_SIZE),
    parameter int VECTOR_INDEX_WIDTH = (VECTOR_SIZE <= 2) ? 1 : $clog2(VECTOR_SIZE),
    parameter int PREV_VECTOR_SIZE = PREV_CHANNELS * PREV_SPATIAL_SIZE,
    parameter int PREV_INDEX_WIDTH = (PREV_VECTOR_SIZE <= 2) ?
        1 : $clog2(PREV_VECTOR_SIZE)
) (
    input  wire logic                                      i_clk,
    input  wire logic                                      i_rst_n,

    input  wire logic                                      i_student_trace_valid,
    output logic                                      o_student_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]                   i_student_trace_sample,
    input  wire logic [VECTOR_INDEX_WIDTH-1:0]             i_student_trace_index,
    input  wire logic signed [DATA_WIDTH-1:0]              i_student_trace_q8_8,

    input  wire logic                                      i_target_trace_valid,
    output logic                                      o_target_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]                   i_target_trace_sample,
    input  wire logic [VECTOR_INDEX_WIDTH-1:0]             i_target_trace_index,
    input  wire logic signed [DATA_WIDTH-1:0]              i_target_trace_q8_8,

    input  wire logic                                      i_prev_target_trace_valid,
    output logic                                      o_prev_target_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]                   i_prev_target_trace_sample,
    input  wire logic [PREV_INDEX_WIDTH-1:0]               i_prev_target_trace_index,
    input  wire logic signed [DATA_WIDTH-1:0]              i_prev_target_trace_q8_8,

    input  wire logic                                      i_start_valid,
    output logic                                      o_start_ready,
    output logic                                      o_capture_complete,
    output logic                                      o_protocol_error,

    output logic                                      o_modulation_valid,
    input  wire logic                                      i_modulation_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_modulation_row,
    output logic [SAMPLE_WIDTH-1:0]                   o_modulation_col,
    output logic signed [DATA_WIDTH-1:0]              o_modulation_q8_8,
    output logic                                      o_modulation_last,
    output logic                                      o_done,
    output logic                                      o_busy
);

    localparam int PRODUCT_WIDTH = 2 * DATA_WIDTH;
    localparam int DIFFERENCE_WIDTH = DATA_WIDTH + 1;
    localparam int DISTANCE_PRODUCT_WIDTH = 2 * DIFFERENCE_WIDTH;
    localparam int SQRT_RADICAND_WIDTH = 64;
    localparam int SQRT_ROOT_WIDTH = SQRT_RADICAND_WIDTH / 2;
    localparam int SQRT_ITER_WIDTH = $clog2(SQRT_ROOT_WIDTH);
    localparam int TRACE_COUNT_WIDTH = $clog2(BATCH_SIZE * VECTOR_SIZE + 1);
    localparam int PREV_COUNT_WIDTH = $clog2(BATCH_SIZE * PREV_VECTOR_SIZE + 1);
    localparam int PREV_CHANNEL_WIDTH = (PREV_CHANNELS <= 2) ?
        1 : $clog2(PREV_CHANNELS);
    localparam int SOFTMAX_SUM_WIDTH =
        DATA_WIDTH + $clog2(BATCH_SIZE + 1);
    localparam int DIV_WIDTH = 32;
    localparam int DIV_INDEX_WIDTH = $clog2(DIV_WIDTH);
    localparam logic [DATA_WIDTH-1:0] ONE_Q = DATA_WIDTH'(1 << FRAC_WIDTH);

    typedef enum logic [4:0] {
        STATE_CAPTURE,
        STATE_Z_READ,
        STATE_Z_ACCUMULATE,
        STATE_Y_READ,
        STATE_Y_ACCUMULATE,
        STATE_Y_SQRT,
        STATE_SOFTMAX_MAX_INIT,
        STATE_SOFTMAX_MAX_SCAN,
        STATE_SOFTMAX_EXP_Z_PREP,
        STATE_SOFTMAX_EXP_Z_DIV,
        STATE_SOFTMAX_EXP_Y_PREP,
        STATE_SOFTMAX_EXP_Y_DIV,
        STATE_SOFTMAX_PROB_Z_PREP,
        STATE_SOFTMAX_PROB_Z_DIV,
        STATE_SOFTMAX_PROB_Y_PREP,
        STATE_SOFTMAX_PROB_Y_DIV,
        STATE_OUTPUT
    } state_t;

    state_t state_q;

    // Flattened address spaces follow Vivado's synchronous simple/true
    // dual-port block-RAM templates. Sample boundaries are address arithmetic,
    // not physical array dimensions.
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] student_trace_mem
        [0:BATCH_SIZE*VECTOR_SIZE-1];
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] target_trace_mem
        [0:BATCH_SIZE*VECTOR_SIZE-1];
    // The Y-distance phase needs two simultaneous reads. Mirroring this trace
    // store gives each copy one write port and one synchronous read port,
    // matching true dual-port BRAM inference without a three-port register file.
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] prev_target_trace_mem_a
        [0:BATCH_SIZE*PREV_VECTOR_SIZE-1];
    (* ram_style = "block" *) logic signed [DATA_WIDTH-1:0] prev_target_trace_mem_b
        [0:BATCH_SIZE*PREV_VECTOR_SIZE-1];

    (* ram_style = "registers" *) logic signed [ACC_WIDTH-1:0] z_score_mem
        [0:BATCH_SIZE*BATCH_SIZE-1];
    (* ram_style = "registers" *) logic signed [ACC_WIDTH-1:0] y_score_mem
        [0:BATCH_SIZE*BATCH_SIZE-1];
    (* ram_style = "registers" *) logic signed [DATA_WIDTH-1:0] modulation_mem
        [0:BATCH_SIZE*BATCH_SIZE-1];

    logic [TRACE_COUNT_WIDTH-1:0] student_trace_count_q;
    logic [TRACE_COUNT_WIDTH-1:0] target_trace_count_q;
    logic [PREV_COUNT_WIDTH-1:0] prev_target_trace_count_q;
    logic [SAMPLE_WIDTH-1:0] student_expected_sample_q;
    logic [SAMPLE_WIDTH-1:0] target_expected_sample_q;
    logic [SAMPLE_WIDTH-1:0] prev_expected_sample_q;
    logic [VECTOR_INDEX_WIDTH-1:0] student_expected_index_q;
    logic [VECTOR_INDEX_WIDTH-1:0] target_expected_index_q;
    logic [PREV_INDEX_WIDTH-1:0] prev_expected_index_q;

    logic [SAMPLE_WIDTH-1:0] pair_row_q, pair_col_q;
    logic [VECTOR_INDEX_WIDTH-1:0] vector_index_q;
    logic [PREV_CHANNEL_WIDTH-1:0] prev_channel_q;
    logic [$clog2(PREV_SPATIAL_SIZE)-1:0] prev_spatial_q;
    logic signed [ACC_WIDTH-1:0] z_accumulator_q;
    logic [ACC_WIDTH-1:0] distance_accumulator_q;
    logic [ACC_WIDTH-1:0] spatial_root_sum_q;

    logic signed [DATA_WIDTH-1:0] student_read_q, target_read_q;
    logic signed [DATA_WIDTH-1:0] prev_a_read_q, prev_b_read_q;

    logic [SQRT_RADICAND_WIDTH-1:0] sqrt_radicand_q;
    logic [SQRT_ROOT_WIDTH+1:0] sqrt_remainder_q;
    logic [SQRT_ROOT_WIDTH-1:0] sqrt_root_q;
    logic [SQRT_ITER_WIDTH-1:0] sqrt_iteration_q;

    logic [SAMPLE_WIDTH-1:0] softmax_row_q, softmax_col_q;
    logic [SAMPLE_WIDTH-1:0] output_row_q, output_col_q;

    logic student_fire, target_fire, prev_target_fire, start_fire;
    logic modulation_fire;
    logic student_coordinates_legal, target_coordinates_legal;
    logic prev_coordinates_legal;
    logic signed [PRODUCT_WIDTH-1:0] z_product;
    logic signed [ACC_WIDTH-1:0] z_contribution;
    logic signed [ACC_WIDTH-1:0] z_accumulator_next;
    logic signed [DIFFERENCE_WIDTH-1:0] target_difference;
    logic signed [DISTANCE_PRODUCT_WIDTH-1:0] distance_product;
    logic [ACC_WIDTH-1:0] distance_contribution;
    logic [ACC_WIDTH-1:0] distance_accumulator_next;
    logic [SQRT_ROOT_WIDTH+3:0] sqrt_remainder_shifted;
    logic [SQRT_ROOT_WIDTH+3:0] sqrt_trial;
    logic [SQRT_ROOT_WIDTH+1:0] sqrt_remainder_next;
    logic [SQRT_ROOT_WIDTH-1:0] sqrt_root_next;
    logic sqrt_last;
    logic [ACC_WIDTH-1:0] spatial_root_sum_next;
    logic [ACC_WIDTH-1:0] y_magnitude;
    logic signed [ACC_WIDTH-1:0] y_score_next;

    logic signed [ACC_WIDTH-1:0] softmax_max_z_q, softmax_max_y_q;
    logic [DATA_WIDTH-1:0] softmax_exp_z_mem [0:BATCH_SIZE-1];
    logic [DATA_WIDTH-1:0] softmax_exp_y_mem [0:BATCH_SIZE-1];
    logic [SOFTMAX_SUM_WIDTH-1:0] softmax_sum_z_q, softmax_sum_y_q;
    logic [DATA_WIDTH-1:0] softmax_prob_z_q;

    logic [DIV_WIDTH-1:0] div_dividend_q, div_divisor_q;
    logic [DIV_WIDTH-1:0] div_quotient_q, div_remainder_q;
    logic [DIV_INDEX_WIDTH-1:0] div_bit_q;
    logic [DIV_WIDTH:0] div_remainder_shifted;
    logic [DIV_WIDTH-1:0] div_quotient_next, div_remainder_next;

    logic signed [ACC_WIDTH-1:0] softmax_delta_z, softmax_delta_y;
    logic [DATA_WIDTH-1:0] exp_magnitude_z, exp_magnitude_y;
    logic [2*DATA_WIDTH-1:0] exp_square_z, exp_square_y;
    logic [DIV_WIDTH-1:0] exp_denominator_z, exp_denominator_y;
    logic [DIV_WIDTH-1:0] probability_dividend_z;
    logic [DIV_WIDTH-1:0] probability_dividend_y;
    logic signed [DATA_WIDTH:0] modulation_without_y;
    logic signed [DATA_WIDTH:0] modulation_with_y;

    function automatic logic [VECTOR_INDEX_WIDTH-1:0] trace_stream_index(
        input logic [TRACE_COUNT_WIDTH-1:0] count
    );
        integer local_count;
        integer channel_count;
        begin
            local_count = integer'(count) % VECTOR_SIZE;
            channel_count = VECTOR_SIZE / SPATIAL_SIZE;
            if (TRACE_STREAM_SPATIAL_MAJOR)
                trace_stream_index = VECTOR_INDEX_WIDTH'(
                    (local_count % channel_count) * SPATIAL_SIZE +
                    local_count / channel_count);
            else
                trace_stream_index = VECTOR_INDEX_WIDTH'(local_count);
        end
    endfunction

    always_comb begin
        o_capture_complete =
            student_trace_count_q == TRACE_COUNT_WIDTH'(BATCH_SIZE * VECTOR_SIZE) &&
            target_trace_count_q == TRACE_COUNT_WIDTH'(BATCH_SIZE * VECTOR_SIZE) &&
            prev_target_trace_count_q ==
                PREV_COUNT_WIDTH'(BATCH_SIZE * PREV_VECTOR_SIZE);
        o_student_trace_ready = state_q == STATE_CAPTURE &&
            student_trace_count_q < TRACE_COUNT_WIDTH'(BATCH_SIZE * VECTOR_SIZE);
        o_target_trace_ready = state_q == STATE_CAPTURE &&
            target_trace_count_q < TRACE_COUNT_WIDTH'(BATCH_SIZE * VECTOR_SIZE);
        o_prev_target_trace_ready = state_q == STATE_CAPTURE &&
            prev_target_trace_count_q <
                PREV_COUNT_WIDTH'(BATCH_SIZE * PREV_VECTOR_SIZE);
        o_start_ready = state_q == STATE_CAPTURE && o_capture_complete &&
            !o_protocol_error;
        o_busy = state_q != STATE_CAPTURE;

        student_fire = i_student_trace_valid && o_student_trace_ready;
        target_fire = i_target_trace_valid && o_target_trace_ready;
        prev_target_fire = i_prev_target_trace_valid &&
            o_prev_target_trace_ready;
        start_fire = i_start_valid && o_start_ready;

        student_coordinates_legal =
            i_student_trace_sample == student_expected_sample_q &&
            i_student_trace_index == student_expected_index_q;
        target_coordinates_legal =
            i_target_trace_sample == target_expected_sample_q &&
            i_target_trace_index == target_expected_index_q;
        prev_coordinates_legal =
            i_prev_target_trace_sample == prev_expected_sample_q &&
            i_prev_target_trace_index == prev_expected_index_q;

        o_modulation_valid = state_q == STATE_OUTPUT;
        o_modulation_row = output_row_q;
        o_modulation_col = output_col_q;
        o_modulation_q8_8 = modulation_mem[
            int'(output_row_q) * BATCH_SIZE + int'(output_col_q)];
        o_modulation_last = o_modulation_valid &&
            output_row_q == SAMPLE_WIDTH'(BATCH_SIZE - 1) &&
            output_col_q == SAMPLE_WIDTH'(BATCH_SIZE - 1);
        modulation_fire = o_modulation_valid && i_modulation_ready;

        z_product = student_read_q * target_read_q;
        z_contribution = $signed({
            {(ACC_WIDTH-PRODUCT_WIDTH){z_product[PRODUCT_WIDTH-1]}},
            z_product
        }) >>> FRAC_WIDTH;
        z_accumulator_next = z_accumulator_q + z_contribution;

        target_difference = $signed(prev_a_read_q) - $signed(prev_b_read_q);
        distance_product = target_difference * target_difference;
        distance_contribution = ACC_WIDTH'(
            $unsigned(distance_product >>> FRAC_WIDTH));
        distance_accumulator_next = distance_accumulator_q +
            distance_contribution;

        sqrt_remainder_shifted = {
            sqrt_remainder_q,
            sqrt_radicand_q[SQRT_RADICAND_WIDTH-1 -: 2]
        };
        sqrt_trial = {{2{1'b0}}, sqrt_root_q, 2'b01};
        sqrt_root_next = sqrt_root_q << 1;
        if (sqrt_remainder_shifted >= sqrt_trial) begin
            sqrt_remainder_next = (SQRT_ROOT_WIDTH+2)'(
                sqrt_remainder_shifted - sqrt_trial);
            sqrt_root_next[0] = 1'b1;
        end else begin
            sqrt_remainder_next =
                sqrt_remainder_shifted[SQRT_ROOT_WIDTH+1:0];
        end
        sqrt_last = sqrt_iteration_q == SQRT_ITER_WIDTH'(SQRT_ROOT_WIDTH - 1);
        spatial_root_sum_next = spatial_root_sum_q +
            ACC_WIDTH'({1'b0, sqrt_root_next});
        y_magnitude = (spatial_root_sum_next +
            ACC_WIDTH'(PREV_SPATIAL_SIZE - 1)) / PREV_SPATIAL_SIZE;
        y_score_next = -$signed(y_magnitude);

        softmax_delta_z = z_score_mem[
            int'(softmax_row_q) * BATCH_SIZE + int'(softmax_col_q)] -
            softmax_max_z_q;
        softmax_delta_y = y_score_mem[
            int'(softmax_row_q) * BATCH_SIZE + int'(softmax_col_q)] -
            softmax_max_y_q;
        exp_magnitude_z = DATA_WIDTH'($unsigned(-softmax_delta_z));
        exp_magnitude_y = DATA_WIDTH'($unsigned(-softmax_delta_y));
        exp_square_z = exp_magnitude_z * exp_magnitude_z;
        exp_square_y = exp_magnitude_y * exp_magnitude_y;
        exp_denominator_z = DIV_WIDTH'(ONE_Q) +
            DIV_WIDTH'(exp_magnitude_z) +
            ((DIV_WIDTH'(exp_square_z) >> FRAC_WIDTH) >> 1);
        exp_denominator_y = DIV_WIDTH'(ONE_Q) +
            DIV_WIDTH'(exp_magnitude_y) +
            ((DIV_WIDTH'(exp_square_y) >> FRAC_WIDTH) >> 1);
        probability_dividend_z =
            DIV_WIDTH'(softmax_exp_z_mem[softmax_col_q]) << FRAC_WIDTH;
        probability_dividend_y =
            DIV_WIDTH'(softmax_exp_y_mem[softmax_col_q]) << FRAC_WIDTH;

        div_remainder_shifted = {
            div_remainder_q,
            div_dividend_q[div_bit_q]
        };
        div_quotient_next = div_quotient_q;
        if (div_remainder_shifted >= {1'b0, div_divisor_q}) begin
            div_remainder_next = DIV_WIDTH'(
                div_remainder_shifted - {1'b0, div_divisor_q});
            div_quotient_next[div_bit_q] = 1'b1;
        end else begin
            div_remainder_next = div_remainder_shifted[DIV_WIDTH-1:0];
            div_quotient_next[div_bit_q] = 1'b0;
        end

        modulation_without_y = $signed({1'b0, softmax_prob_z_q});
        modulation_with_y = $signed({1'b0, softmax_prob_z_q}) -
            $signed({1'b0, div_quotient_next[DATA_WIDTH-1:0]});
    end

    // Captured payload is valid only after the resettable counters reach their
    // configured totals, so these RAM contents do not need a reset value.
    always_ff @(posedge i_clk) begin
        if (student_fire && student_coordinates_legal)
            student_trace_mem[int'(i_student_trace_sample) * VECTOR_SIZE +
                int'(i_student_trace_index)] <= i_student_trace_q8_8;
        if (target_fire && target_coordinates_legal)
            target_trace_mem[int'(i_target_trace_sample) * VECTOR_SIZE +
                int'(i_target_trace_index)] <= i_target_trace_q8_8;
        if (prev_target_fire && prev_coordinates_legal)
            begin
                prev_target_trace_mem_a[
                    int'(i_prev_target_trace_sample) * PREV_VECTOR_SIZE +
                    int'(i_prev_target_trace_index)] <= i_prev_target_trace_q8_8;
                prev_target_trace_mem_b[
                    int'(i_prev_target_trace_sample) * PREV_VECTOR_SIZE +
                    int'(i_prev_target_trace_index)] <= i_prev_target_trace_q8_8;
            end

        if (state_q == STATE_Z_ACCUMULATE && vector_index_q ==
            VECTOR_INDEX_WIDTH'(VECTOR_SIZE - 1))
            z_score_mem[int'(pair_row_q) * BATCH_SIZE +
                int'(pair_col_q)] <= z_accumulator_next / SPATIAL_SIZE;

        if (state_q == STATE_Y_SQRT && sqrt_last && prev_spatial_q ==
            $clog2(PREV_SPATIAL_SIZE)'(PREV_SPATIAL_SIZE - 1))
            y_score_mem[int'(pair_row_q) * BATCH_SIZE +
                int'(pair_col_q)] <= y_score_next;

        if (state_q == STATE_SOFTMAX_EXP_Z_PREP) begin
            if (softmax_delta_z >= 0)
                softmax_exp_z_mem[softmax_col_q] <= ONE_Q;
            else if (softmax_delta_z <=
                -ACC_WIDTH'(8 * (1 << FRAC_WIDTH)))
                softmax_exp_z_mem[softmax_col_q] <= '0;
        end else if (state_q == STATE_SOFTMAX_EXP_Z_DIV &&
            div_bit_q == 0) begin
            softmax_exp_z_mem[softmax_col_q] <=
                DATA_WIDTH'(div_quotient_next);
        end

        if (state_q == STATE_SOFTMAX_EXP_Y_PREP) begin
            if (softmax_delta_y >= 0)
                softmax_exp_y_mem[softmax_col_q] <= ONE_Q;
            else if (softmax_delta_y <=
                -ACC_WIDTH'(8 * (1 << FRAC_WIDTH)))
                softmax_exp_y_mem[softmax_col_q] <= '0;
        end else if (state_q == STATE_SOFTMAX_EXP_Y_DIV &&
            div_bit_q == 0) begin
            softmax_exp_y_mem[softmax_col_q] <=
                DATA_WIDTH'(div_quotient_next);
        end

        if (state_q == STATE_SOFTMAX_PROB_Y_PREP &&
            softmax_sum_y_q == 0) begin
            modulation_mem[int'(softmax_row_q) * BATCH_SIZE +
                int'(softmax_col_q)] <=
                modulation_without_y[DATA_WIDTH-1:0];
        end else if (state_q == STATE_SOFTMAX_PROB_Y_DIV &&
            div_bit_q == 0) begin
            modulation_mem[int'(softmax_row_q) * BATCH_SIZE +
                int'(softmax_col_q)] <=
                modulation_with_y[DATA_WIDTH-1:0];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_CAPTURE;
            student_trace_count_q <= '0;
            target_trace_count_q <= '0;
            prev_target_trace_count_q <= '0;
            student_expected_sample_q <= '0;
            target_expected_sample_q <= '0;
            prev_expected_sample_q <= '0;
            student_expected_index_q <= '0;
            target_expected_index_q <= '0;
            prev_expected_index_q <= '0;
            pair_row_q <= '0;
            pair_col_q <= '0;
            vector_index_q <= '0;
            prev_channel_q <= '0;
            prev_spatial_q <= '0;
            z_accumulator_q <= '0;
            distance_accumulator_q <= '0;
            spatial_root_sum_q <= '0;
            student_read_q <= '0;
            target_read_q <= '0;
            prev_a_read_q <= '0;
            prev_b_read_q <= '0;
            sqrt_radicand_q <= '0;
            sqrt_remainder_q <= '0;
            sqrt_root_q <= '0;
            sqrt_iteration_q <= '0;
            softmax_row_q <= '0;
            softmax_col_q <= '0;
            softmax_max_z_q <= '0;
            softmax_max_y_q <= '0;
            softmax_sum_z_q <= '0;
            softmax_sum_y_q <= '0;
            softmax_prob_z_q <= '0;
            div_dividend_q <= '0;
            div_divisor_q <= '0;
            div_quotient_q <= '0;
            div_remainder_q <= '0;
            div_bit_q <= '0;
            output_row_q <= '0;
            output_col_q <= '0;
            o_protocol_error <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;

            if (student_fire) begin
                if (!student_coordinates_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    student_trace_count_q <= student_trace_count_q + 1'b1;
                    if (i_student_trace_index ==
                        VECTOR_INDEX_WIDTH'(VECTOR_SIZE - 1)) begin
                        student_expected_index_q <= '0;
                        if (i_student_trace_sample !=
                            SAMPLE_WIDTH'(BATCH_SIZE - 1))
                            student_expected_sample_q <=
                                student_expected_sample_q + 1'b1;
                    end else begin
                        student_expected_index_q <=
                            trace_stream_index(student_trace_count_q + 1'b1);
                    end
                end
            end

            if (target_fire) begin
                if (!target_coordinates_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    target_trace_count_q <= target_trace_count_q + 1'b1;
                    if (i_target_trace_index ==
                        VECTOR_INDEX_WIDTH'(VECTOR_SIZE - 1)) begin
                        target_expected_index_q <= '0;
                        if (i_target_trace_sample !=
                            SAMPLE_WIDTH'(BATCH_SIZE - 1))
                            target_expected_sample_q <=
                                target_expected_sample_q + 1'b1;
                    end else begin
                        target_expected_index_q <=
                            trace_stream_index(target_trace_count_q + 1'b1);
                    end
                end
            end

            if (prev_target_fire) begin
                if (!prev_coordinates_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    prev_target_trace_count_q <=
                        prev_target_trace_count_q + 1'b1;
                    if (i_prev_target_trace_index ==
                        PREV_INDEX_WIDTH'(PREV_VECTOR_SIZE - 1)) begin
                        prev_expected_index_q <= '0;
                        if (i_prev_target_trace_sample !=
                            SAMPLE_WIDTH'(BATCH_SIZE - 1))
                            prev_expected_sample_q <=
                                prev_expected_sample_q + 1'b1;
                    end else begin
                        prev_expected_index_q <= prev_expected_index_q + 1'b1;
                    end
                end
            end

            unique case (state_q)
                STATE_CAPTURE: begin
                    if (start_fire) begin
                        pair_row_q <= '0;
                        pair_col_q <= '0;
                        vector_index_q <= '0;
                        z_accumulator_q <= '0;
                        state_q <= STATE_Z_READ;
                    end
                end

                STATE_Z_READ: begin
                    student_read_q <= student_trace_mem[
                        int'(pair_row_q) * VECTOR_SIZE + int'(vector_index_q)];
                    target_read_q <= target_trace_mem[
                        int'(pair_col_q) * VECTOR_SIZE + int'(vector_index_q)];
                    state_q <= STATE_Z_ACCUMULATE;
                end

                STATE_Z_ACCUMULATE: begin
                    if (vector_index_q ==
                        VECTOR_INDEX_WIDTH'(VECTOR_SIZE - 1)) begin
                        vector_index_q <= '0;
                        z_accumulator_q <= '0;
                        if (pair_col_q == SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            pair_col_q <= '0;
                            if (pair_row_q == SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                                pair_row_q <= '0;
                                prev_channel_q <= '0;
                                prev_spatial_q <= '0;
                                distance_accumulator_q <= '0;
                                spatial_root_sum_q <= '0;
                                state_q <= STATE_Y_READ;
                            end else begin
                                pair_row_q <= pair_row_q + 1'b1;
                                state_q <= STATE_Z_READ;
                            end
                        end else begin
                            pair_col_q <= pair_col_q + 1'b1;
                            state_q <= STATE_Z_READ;
                        end
                    end else begin
                        z_accumulator_q <= z_accumulator_next;
                        vector_index_q <= vector_index_q + 1'b1;
                        state_q <= STATE_Z_READ;
                    end
                end

                STATE_Y_READ: begin
                    prev_a_read_q <= prev_target_trace_mem_a[
                        int'(pair_row_q) * PREV_VECTOR_SIZE +
                        int'(prev_channel_q) * PREV_SPATIAL_SIZE +
                        int'(prev_spatial_q)];
                    prev_b_read_q <= prev_target_trace_mem_b[
                        int'(pair_col_q) * PREV_VECTOR_SIZE +
                        int'(prev_channel_q) * PREV_SPATIAL_SIZE +
                        int'(prev_spatial_q)];
                    state_q <= STATE_Y_ACCUMULATE;
                end

                STATE_Y_ACCUMULATE: begin
                    if (prev_channel_q == PREV_CHANNEL_WIDTH'(
                        PREV_CHANNELS - 1)) begin
                        distance_accumulator_q <= '0;
                        prev_channel_q <= '0;
                        sqrt_radicand_q <=
                            SQRT_RADICAND_WIDTH'(distance_accumulator_next)
                            << FRAC_WIDTH;
                        sqrt_remainder_q <= '0;
                        sqrt_root_q <= '0;
                        sqrt_iteration_q <= '0;
                        state_q <= STATE_Y_SQRT;
                    end else begin
                        distance_accumulator_q <= distance_accumulator_next;
                        prev_channel_q <= prev_channel_q + 1'b1;
                        state_q <= STATE_Y_READ;
                    end
                end

                STATE_Y_SQRT: begin
                    sqrt_remainder_q <= sqrt_remainder_next;
                    sqrt_root_q <= sqrt_root_next;
                    sqrt_radicand_q <= sqrt_radicand_q << 2;
                    if (sqrt_last) begin
                        if (prev_spatial_q == $clog2(PREV_SPATIAL_SIZE)'(
                            PREV_SPATIAL_SIZE - 1)) begin
                            prev_spatial_q <= '0;
                            spatial_root_sum_q <= '0;
                            if (pair_col_q ==
                                SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                                pair_col_q <= '0;
                                if (pair_row_q ==
                                    SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                                    pair_row_q <= '0;
                                    softmax_row_q <= '0;
                                    softmax_col_q <= '0;
                                    state_q <= STATE_SOFTMAX_MAX_INIT;
                                end else begin
                                    pair_row_q <= pair_row_q + 1'b1;
                                    state_q <= STATE_Y_READ;
                                end
                            end else begin
                                pair_col_q <= pair_col_q + 1'b1;
                                state_q <= STATE_Y_READ;
                            end
                        end else begin
                            spatial_root_sum_q <= spatial_root_sum_next;
                            prev_spatial_q <= prev_spatial_q + 1'b1;
                            state_q <= STATE_Y_READ;
                        end
                    end else begin
                        sqrt_iteration_q <= sqrt_iteration_q + 1'b1;
                    end
                end

                STATE_SOFTMAX_MAX_INIT: begin
                    softmax_max_z_q <= z_score_mem[
                        int'(softmax_row_q) * BATCH_SIZE];
                    softmax_max_y_q <= y_score_mem[
                        int'(softmax_row_q) * BATCH_SIZE];
                    softmax_sum_z_q <= '0;
                    softmax_sum_y_q <= '0;
                    if (BATCH_SIZE == 1) begin
                        softmax_col_q <= '0;
                        state_q <= STATE_SOFTMAX_EXP_Z_PREP;
                    end else begin
                        softmax_col_q <= SAMPLE_WIDTH'(1);
                        state_q <= STATE_SOFTMAX_MAX_SCAN;
                    end
                end

                STATE_SOFTMAX_MAX_SCAN: begin
                    if (z_score_mem[int'(softmax_row_q) * BATCH_SIZE +
                        int'(softmax_col_q)] > softmax_max_z_q)
                        softmax_max_z_q <= z_score_mem[
                            int'(softmax_row_q) * BATCH_SIZE +
                            int'(softmax_col_q)];
                    if (y_score_mem[int'(softmax_row_q) * BATCH_SIZE +
                        int'(softmax_col_q)] > softmax_max_y_q)
                        softmax_max_y_q <= y_score_mem[
                            int'(softmax_row_q) * BATCH_SIZE +
                            int'(softmax_col_q)];
                    if (softmax_col_q == SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                        softmax_col_q <= '0;
                        state_q <= STATE_SOFTMAX_EXP_Z_PREP;
                    end else begin
                        softmax_col_q <= softmax_col_q + 1'b1;
                    end
                end

                STATE_SOFTMAX_EXP_Z_PREP: begin
                    if (softmax_delta_z >= 0) begin
                        softmax_sum_z_q <= softmax_sum_z_q +
                            SOFTMAX_SUM_WIDTH'(ONE_Q);
                        state_q <= STATE_SOFTMAX_EXP_Y_PREP;
                    end else if (softmax_delta_z <=
                        -ACC_WIDTH'(8 * (1 << FRAC_WIDTH))) begin
                        state_q <= STATE_SOFTMAX_EXP_Y_PREP;
                    end else begin
                        div_dividend_q <= DIV_WIDTH'(1 << (2 * FRAC_WIDTH));
                        div_divisor_q <= exp_denominator_z;
                        div_quotient_q <= '0;
                        div_remainder_q <= '0;
                        div_bit_q <= DIV_INDEX_WIDTH'(DIV_WIDTH - 1);
                        state_q <= STATE_SOFTMAX_EXP_Z_DIV;
                    end
                end

                STATE_SOFTMAX_EXP_Z_DIV: begin
                    div_quotient_q <= div_quotient_next;
                    div_remainder_q <= div_remainder_next;
                    if (div_bit_q == 0) begin
                        softmax_sum_z_q <= softmax_sum_z_q +
                            SOFTMAX_SUM_WIDTH'(div_quotient_next);
                        state_q <= STATE_SOFTMAX_EXP_Y_PREP;
                    end else begin
                        div_bit_q <= div_bit_q - 1'b1;
                    end
                end

                STATE_SOFTMAX_EXP_Y_PREP: begin
                    if (softmax_delta_y >= 0) begin
                        softmax_sum_y_q <= softmax_sum_y_q +
                            SOFTMAX_SUM_WIDTH'(ONE_Q);
                        if (softmax_col_q ==
                            SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            softmax_col_q <= '0;
                            state_q <= STATE_SOFTMAX_PROB_Z_PREP;
                        end else begin
                            softmax_col_q <= softmax_col_q + 1'b1;
                            state_q <= STATE_SOFTMAX_EXP_Z_PREP;
                        end
                    end else if (softmax_delta_y <=
                        -ACC_WIDTH'(8 * (1 << FRAC_WIDTH))) begin
                        if (softmax_col_q ==
                            SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            softmax_col_q <= '0;
                            state_q <= STATE_SOFTMAX_PROB_Z_PREP;
                        end else begin
                            softmax_col_q <= softmax_col_q + 1'b1;
                            state_q <= STATE_SOFTMAX_EXP_Z_PREP;
                        end
                    end else begin
                        div_dividend_q <= DIV_WIDTH'(1 << (2 * FRAC_WIDTH));
                        div_divisor_q <= exp_denominator_y;
                        div_quotient_q <= '0;
                        div_remainder_q <= '0;
                        div_bit_q <= DIV_INDEX_WIDTH'(DIV_WIDTH - 1);
                        state_q <= STATE_SOFTMAX_EXP_Y_DIV;
                    end
                end

                STATE_SOFTMAX_EXP_Y_DIV: begin
                    div_quotient_q <= div_quotient_next;
                    div_remainder_q <= div_remainder_next;
                    if (div_bit_q == 0) begin
                        softmax_sum_y_q <= softmax_sum_y_q +
                            SOFTMAX_SUM_WIDTH'(div_quotient_next);
                        if (softmax_col_q ==
                            SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            softmax_col_q <= '0;
                            state_q <= STATE_SOFTMAX_PROB_Z_PREP;
                        end else begin
                            softmax_col_q <= softmax_col_q + 1'b1;
                            state_q <= STATE_SOFTMAX_EXP_Z_PREP;
                        end
                    end else begin
                        div_bit_q <= div_bit_q - 1'b1;
                    end
                end

                STATE_SOFTMAX_PROB_Z_PREP: begin
                    if (softmax_sum_z_q == 0) begin
                        softmax_prob_z_q <= '0;
                        state_q <= STATE_SOFTMAX_PROB_Y_PREP;
                    end else begin
                        div_dividend_q <= probability_dividend_z;
                        div_divisor_q <= DIV_WIDTH'(softmax_sum_z_q);
                        div_quotient_q <= '0;
                        div_remainder_q <= '0;
                        div_bit_q <= DIV_INDEX_WIDTH'(DIV_WIDTH - 1);
                        state_q <= STATE_SOFTMAX_PROB_Z_DIV;
                    end
                end

                STATE_SOFTMAX_PROB_Z_DIV: begin
                    div_quotient_q <= div_quotient_next;
                    div_remainder_q <= div_remainder_next;
                    if (div_bit_q == 0) begin
                        softmax_prob_z_q <= DATA_WIDTH'(div_quotient_next);
                        state_q <= STATE_SOFTMAX_PROB_Y_PREP;
                    end else begin
                        div_bit_q <= div_bit_q - 1'b1;
                    end
                end

                STATE_SOFTMAX_PROB_Y_PREP: begin
                    if (softmax_sum_y_q == 0) begin
                        if (softmax_col_q ==
                            SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            if (softmax_row_q ==
                                SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                                output_row_q <= '0;
                                output_col_q <= '0;
                                state_q <= STATE_OUTPUT;
                            end else begin
                                softmax_row_q <= softmax_row_q + 1'b1;
                                softmax_col_q <= '0;
                                state_q <= STATE_SOFTMAX_MAX_INIT;
                            end
                        end else begin
                            softmax_col_q <= softmax_col_q + 1'b1;
                            state_q <= STATE_SOFTMAX_PROB_Z_PREP;
                        end
                    end else begin
                        div_dividend_q <= probability_dividend_y;
                        div_divisor_q <= DIV_WIDTH'(softmax_sum_y_q);
                        div_quotient_q <= '0;
                        div_remainder_q <= '0;
                        div_bit_q <= DIV_INDEX_WIDTH'(DIV_WIDTH - 1);
                        state_q <= STATE_SOFTMAX_PROB_Y_DIV;
                    end
                end

                STATE_SOFTMAX_PROB_Y_DIV: begin
                    div_quotient_q <= div_quotient_next;
                    div_remainder_q <= div_remainder_next;
                    if (div_bit_q == 0) begin
                        if (softmax_col_q ==
                            SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            if (softmax_row_q ==
                                SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                                output_row_q <= '0;
                                output_col_q <= '0;
                                state_q <= STATE_OUTPUT;
                            end else begin
                                softmax_row_q <= softmax_row_q + 1'b1;
                                softmax_col_q <= '0;
                                state_q <= STATE_SOFTMAX_MAX_INIT;
                            end
                        end else begin
                            softmax_col_q <= softmax_col_q + 1'b1;
                            state_q <= STATE_SOFTMAX_PROB_Z_PREP;
                        end
                    end else begin
                        div_bit_q <= div_bit_q - 1'b1;
                    end
                end

                STATE_OUTPUT: begin
                    if (modulation_fire) begin
                        if (o_modulation_last) begin
                            student_trace_count_q <= '0;
                            target_trace_count_q <= '0;
                            prev_target_trace_count_q <= '0;
                            student_expected_sample_q <= '0;
                            target_expected_sample_q <= '0;
                            prev_expected_sample_q <= '0;
                            student_expected_index_q <= '0;
                            target_expected_index_q <= '0;
                            prev_expected_index_q <= '0;
                            output_row_q <= '0;
                            output_col_q <= '0;
                            o_done <= 1'b1;
                            state_q <= STATE_CAPTURE;
                        end else if (output_col_q ==
                            SAMPLE_WIDTH'(BATCH_SIZE - 1)) begin
                            output_col_q <= '0;
                            output_row_q <= output_row_q + 1'b1;
                        end else begin
                            output_col_q <= output_col_q + 1'b1;
                        end
                    end
                end

                default: begin
                    o_protocol_error <= 1'b1;
                    state_q <= STATE_CAPTURE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
