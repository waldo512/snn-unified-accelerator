`timescale 1ns/1ps
`default_nettype none

// FC TP W0 hidden-error frontend. It stores one B16 observation window,
// replays the pairwise vectors through one shared metric/softmax row service,
// and feeds the resulting probability matrices to the hidden-error reducer.
module unified_fc_tp_w0_hidden_error_frontend #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_TARGET_DIMENSION = 16,
    parameter int TARGET_TRACE_WIDTH = 20,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int TARGET_INDEX_WIDTH =
        (MAX_TARGET_DIMENSION <= 2) ? 1 : $clog2(MAX_TARGET_DIMENSION),
    parameter int TARGET_COUNT_WIDTH = $clog2(MAX_TARGET_DIMENSION + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]         i_cfg_hidden_count,
    input  logic [TARGET_COUNT_WIDTH-1:0]         i_cfg_target_dimension,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  logic                                  i_release,
    output logic                                  o_release_ready,

    input  logic                                  i_student_trace_valid,
    output logic                                  o_student_trace_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_student_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_student_trace_hidden,
    input  logic [16:0]                           i_student_trace_q016,
    output logic                                  o_student_trace_error,

    input  logic                                  i_target_hidden_trace_valid,
    output logic                                  o_target_hidden_trace_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_target_hidden_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_target_hidden_trace_hidden,
    input  logic [16:0]                           i_target_hidden_trace_q016,
    output logic                                  o_target_hidden_trace_error,

    input  logic                                  i_target_input_trace_valid,
    output logic                                  o_target_input_trace_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_target_input_trace_sample,
    input  logic [TARGET_INDEX_WIDTH-1:0]         i_target_input_trace_index,
    input  logic [TARGET_TRACE_WIDTH-1:0]         i_target_input_trace_q016,
    output logic                                  o_target_input_trace_error,

    input  logic                                  i_surrogate_valid,
    output logic                                  o_surrogate_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]          i_surrogate_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_surrogate_hidden,
    input  logic [16:0]                           i_surrogate_q016,
    output logic                                  o_surrogate_error,

    input  logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,
    output logic                                  o_hidden_error_valid,
    input  logic                                  i_hidden_error_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]          o_hidden_error_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]         o_hidden_error_hidden,
    output logic signed [17:0]                    o_hidden_error_q016,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_internal_error,
    output logic                                  o_window_done,
    output logic                                  o_busy
);

    localparam int HIDDEN_DEPTH = MAX_BATCH * MAX_HIDDEN;
    localparam int TARGET_DEPTH = MAX_BATCH * MAX_TARGET_DIMENSION;
    localparam int HIDDEN_ADDR_WIDTH = (HIDDEN_DEPTH <= 2) ? 1 : $clog2(HIDDEN_DEPTH);
    localparam int TARGET_ADDR_WIDTH = (TARGET_DEPTH <= 2) ? 1 : $clog2(TARGET_DEPTH);
    localparam int METRIC_LENGTH_WIDTH = $clog2(MAX_HIDDEN + 1);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_LOAD,
        STATE_START_ROW,
        STATE_STREAM_ROW,
        STATE_WAIT_ROW,
        STATE_SEAL_REDUCER,
        STATE_START_EMIT,
        STATE_EMIT
    } state_t;

    state_t state_q;
    (* ram_style = "block" *) logic [16:0] student_trace_mem [0:HIDDEN_DEPTH-1];
    (* ram_style = "block" *) logic [16:0] target_hidden_trace_mem [0:HIDDEN_DEPTH-1];
    (* ram_style = "block" *) logic [TARGET_TRACE_WIDTH-1:0]
        target_input_trace_mem [0:TARGET_DEPTH-1];

    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_COUNT_WIDTH-1:0] hidden_count_q;
    logic [TARGET_COUNT_WIDTH-1:0] target_dimension_q;

    logic [BATCH_INDEX_WIDTH-1:0] student_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] student_hidden_q;
    logic [BATCH_INDEX_WIDTH-1:0] target_hidden_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] target_hidden_q;
    logic [BATCH_INDEX_WIDTH-1:0] target_input_sample_q;
    logic [TARGET_INDEX_WIDTH-1:0] target_input_index_q;
    logic [BATCH_INDEX_WIDTH-1:0] surrogate_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] surrogate_hidden_q;
    logic student_complete_q;
    logic target_hidden_complete_q;
    logic target_input_complete_q;
    logic surrogate_complete_q;

    logic row_distance_mode_q;
    logic [BATCH_INDEX_WIDTH-1:0] row_index_q;
    logic [BATCH_INDEX_WIDTH-1:0] column_index_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] vector_index_q;
    logic pair_pending_q;
    logic [12:0] pair_value_a_q;
    logic [12:0] pair_value_b_q;
    logic arithmetic_overflow_q;

    logic reducer_cfg_ready;
    logic reducer_cfg_error;
    logic reducer_active;
    logic reducer_release_ready;
    logic reducer_student_ready;
    logic reducer_student_error;
    logic reducer_target_ready;
    logic reducer_target_error;
    logic reducer_target_trace_ready;
    logic reducer_target_trace_error;
    logic reducer_surrogate_ready;
    logic reducer_surrogate_error;
    logic reducer_observation_done_ready;
    logic reducer_emit_ready;
    logic reducer_hidden_error_valid;
    logic reducer_arithmetic_overflow;
    logic reducer_emit_done;
    logic reducer_busy;

    logic row_start_ready;
    logic row_config_error;
    logic row_pair_ready;
    logic row_probability_valid;
    logic row_probability_ready;
    logic [BATCH_INDEX_WIDTH-1:0] row_probability_row;
    logic [BATCH_INDEX_WIDTH-1:0] row_probability_column;
    logic [16:0] row_probability_q016;
    logic row_probability_last;
    logic row_arithmetic_overflow;
    logic row_done;
    logic row_busy;

    logic cfg_legal;
    logic cfg_fire;
    logic release_fire;
    logic student_fire;
    logic target_hidden_fire;
    logic target_input_fire;
    logic surrogate_fire;
    logic observation_done_fire;
    logic pair_fire;
    logic all_sources_complete;
    logic student_matches;
    logic target_hidden_matches;
    logic target_input_matches;
    logic surrogate_matches;
    logic student_last;
    logic target_hidden_last;
    logic target_input_last;
    logic surrogate_last;
    logic row_start_fire;
    logic row_pair_last;
    logic row_last;
    logic hidden_error_fire;
    logic hidden_error_last;
    logic [METRIC_LENGTH_WIDTH-1:0] row_vector_length;
    logic [HIDDEN_ADDR_WIDTH-1:0] student_write_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] target_hidden_write_addr;
    logic [TARGET_ADDR_WIDTH-1:0] target_input_write_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] student_read_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] target_hidden_read_addr;
    logic [TARGET_ADDR_WIDTH-1:0] target_input_read_a_addr;
    logic [TARGET_ADDR_WIDTH-1:0] target_input_read_b_addr;
    logic [TARGET_TRACE_WIDTH:0] target_input_a_rounded;
    logic [TARGET_TRACE_WIDTH:0] target_input_b_rounded;
    logic [17:0] student_rounded;
    logic [17:0] target_hidden_rounded;
    logic pair_read_overflow;

    function automatic logic [HIDDEN_ADDR_WIDTH-1:0] hidden_address(
        input logic [BATCH_INDEX_WIDTH-1:0] sample_index,
        input logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index
    );
        hidden_address = HIDDEN_ADDR_WIDTH'((sample_index * MAX_HIDDEN) + hidden_index);
    endfunction

    function automatic logic [TARGET_ADDR_WIDTH-1:0] target_address(
        input logic [BATCH_INDEX_WIDTH-1:0] sample_index,
        input logic [TARGET_INDEX_WIDTH-1:0] target_index
    );
        target_address = TARGET_ADDR_WIDTH'(
            (sample_index * MAX_TARGET_DIMENSION) + target_index);
    endfunction

    function automatic logic [12:0] clip_q10_from_q16(
        input logic [TARGET_TRACE_WIDTH:0] rounded_value
    );
        if ((rounded_value >> 6) > (TARGET_TRACE_WIDTH+1)'(8191))
            clip_q10_from_q16 = 13'd8191;
        else
            clip_q10_from_q16 = 13'(rounded_value >> 6);
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_batch_count == BATCH_COUNT_WIDTH'(16)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_COUNT_WIDTH'(MAX_HIDDEN)) &&
            (i_cfg_target_dimension != '0) &&
            (i_cfg_target_dimension <= TARGET_COUNT_WIDTH'(MAX_TARGET_DIMENSION));
        o_cfg_ready = (state_q == STATE_IDLE) && reducer_cfg_ready;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_active = reducer_active && (state_q != STATE_IDLE);
        o_release_ready = (state_q == STATE_LOAD) && reducer_release_ready &&
            !student_complete_q && !target_hidden_complete_q &&
            !target_input_complete_q && !surrogate_complete_q &&
            (student_sample_q == '0) && (student_hidden_q == '0) &&
            (target_hidden_sample_q == '0) && (target_hidden_q == '0) &&
            (target_input_sample_q == '0) && (target_input_index_q == '0) &&
            (surrogate_sample_q == '0) && (surrogate_hidden_q == '0);
        release_fire = i_release && o_release_ready;

        o_student_trace_ready = (state_q == STATE_LOAD) && !student_complete_q;
        student_fire = i_student_trace_valid && o_student_trace_ready;
        student_matches = (i_student_trace_sample == student_sample_q) &&
            (i_student_trace_hidden == student_hidden_q);
        student_last = ({1'b0, student_sample_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, student_hidden_q} + 1'b1 >= {1'b0, hidden_count_q});

        o_target_hidden_trace_ready = (state_q == STATE_LOAD) &&
            !target_hidden_complete_q && reducer_target_trace_ready;
        target_hidden_fire = i_target_hidden_trace_valid &&
            o_target_hidden_trace_ready;
        target_hidden_matches =
            (i_target_hidden_trace_sample == target_hidden_sample_q) &&
            (i_target_hidden_trace_hidden == target_hidden_q);
        target_hidden_last =
            ({1'b0, target_hidden_sample_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, target_hidden_q} + 1'b1 >= {1'b0, hidden_count_q});

        o_target_input_trace_ready = (state_q == STATE_LOAD) &&
            !target_input_complete_q;
        target_input_fire = i_target_input_trace_valid &&
            o_target_input_trace_ready;
        target_input_matches =
            (i_target_input_trace_sample == target_input_sample_q) &&
            (i_target_input_trace_index == target_input_index_q);
        target_input_last =
            ({1'b0, target_input_sample_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, target_input_index_q} + 1'b1 >=
             {1'b0, target_dimension_q});

        o_surrogate_ready = (state_q == STATE_LOAD) && !surrogate_complete_q &&
            reducer_surrogate_ready;
        surrogate_fire = i_surrogate_valid && o_surrogate_ready;
        surrogate_matches = (i_surrogate_sample == surrogate_sample_q) &&
            (i_surrogate_hidden == surrogate_hidden_q);
        surrogate_last = ({1'b0, surrogate_sample_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, surrogate_hidden_q} + 1'b1 >= {1'b0, hidden_count_q});

        all_sources_complete = student_complete_q && target_hidden_complete_q &&
            target_input_complete_q && surrogate_complete_q;
        o_observation_done_ready = (state_q == STATE_LOAD) && all_sources_complete;
        observation_done_fire = i_observation_done && o_observation_done_ready;

        row_vector_length = row_distance_mode_q ?
            METRIC_LENGTH_WIDTH'(target_dimension_q) :
            METRIC_LENGTH_WIDTH'(hidden_count_q);
        row_start_fire = (state_q == STATE_START_ROW) && row_start_ready;
        pair_fire = pair_pending_q && row_pair_ready;
        row_pair_last =
            ({1'b0, column_index_q} + 1'b1 >= batch_count_q) &&
            (({1'b0, vector_index_q} + 1'b1) >= {1'b0, row_vector_length});
        row_last = ({1'b0, row_index_q} + 1'b1) >= batch_count_q;
        hidden_error_fire = reducer_hidden_error_valid && i_hidden_error_ready;
        hidden_error_last =
            (({1'b0, o_hidden_error_sample} + 1'b1) >= batch_count_q) &&
            (({1'b0, o_hidden_error_hidden} + 1'b1) >=
             {1'b0, hidden_count_q});

        student_write_addr = hidden_address(
            i_student_trace_sample, i_student_trace_hidden);
        target_hidden_write_addr = hidden_address(
            i_target_hidden_trace_sample, i_target_hidden_trace_hidden);
        target_input_write_addr = target_address(
            i_target_input_trace_sample, i_target_input_trace_index);
        student_read_addr = hidden_address(row_index_q, vector_index_q);
        target_hidden_read_addr = hidden_address(column_index_q, vector_index_q);
        target_input_read_a_addr = target_address(
            row_index_q, vector_index_q[TARGET_INDEX_WIDTH-1:0]);
        target_input_read_b_addr = target_address(
            column_index_q, vector_index_q[TARGET_INDEX_WIDTH-1:0]);

        student_rounded = {1'b0, student_trace_mem[student_read_addr]} + 18'd32;
        target_hidden_rounded =
            {1'b0, target_hidden_trace_mem[target_hidden_read_addr]} + 18'd32;
        target_input_a_rounded =
            {1'b0, target_input_trace_mem[target_input_read_a_addr]} +
            (TARGET_TRACE_WIDTH+1)'(32);
        target_input_b_rounded =
            {1'b0, target_input_trace_mem[target_input_read_b_addr]} +
            (TARGET_TRACE_WIDTH+1)'(32);
        if (row_distance_mode_q) begin
            pair_read_overflow = ((target_input_a_rounded >> 6) >
                                  (TARGET_TRACE_WIDTH+1)'(8191)) ||
                ((target_input_b_rounded >> 6) >
                 (TARGET_TRACE_WIDTH+1)'(8191));
        end else begin
            pair_read_overflow = ((student_rounded >> 6) > 18'd8191) ||
                ((target_hidden_rounded >> 6) > 18'd8191);
        end

        row_probability_ready = row_distance_mode_q ?
            reducer_target_ready : reducer_student_ready;
        o_hidden_error_valid = reducer_hidden_error_valid;
        o_arithmetic_overflow = arithmetic_overflow_q ||
            row_arithmetic_overflow || reducer_arithmetic_overflow;
        o_internal_error = row_config_error || reducer_cfg_error ||
            reducer_student_error || reducer_target_error ||
            reducer_target_trace_error || reducer_surrogate_error ||
            (row_probability_valid &&
             (row_probability_last !=
              (({1'b0, row_probability_column} + 1'b1) >= batch_count_q)));
        o_busy = ((state_q != STATE_IDLE) && (state_q != STATE_LOAD)) ||
            row_busy || reducer_busy;
        o_window_done = reducer_emit_done;
    end

    unified_tp_pairwise_probability_row_q10 #(
        .VALUE_WIDTH(13),
        .ACC_WIDTH(40),
        .MAX_VECTOR_LENGTH(MAX_HIDDEN),
        .MAX_ITEMS(MAX_BATCH),
        .ROW_TAG_WIDTH(BATCH_INDEX_WIDTH),
        .LENGTH_WIDTH(METRIC_LENGTH_WIDTH),
        .ITEM_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .ITEM_COUNT_WIDTH(BATCH_COUNT_WIDTH)
    ) u_pairwise_row (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start_valid(state_q == STATE_START_ROW),
        .o_start_ready(row_start_ready),
        .i_distance_mode(row_distance_mode_q),
        .i_vector_length(row_vector_length),
        .i_row_length(batch_count_q),
        .i_row_tag(row_index_q),
        .o_config_error(row_config_error),
        .i_pair_valid(pair_pending_q),
        .o_pair_ready(row_pair_ready),
        .i_value_a_q10(pair_value_a_q),
        .i_value_b_q10(pair_value_b_q),
        .o_probability_valid(row_probability_valid),
        .i_probability_ready(row_probability_ready),
        .o_probability_row_tag(row_probability_row),
        .o_probability_index(row_probability_column),
        .o_probability_q016(row_probability_q016),
        .o_probability_last(row_probability_last),
        .o_arithmetic_overflow(row_arithmetic_overflow),
        .o_done(row_done),
        .o_busy(row_busy)
    );

    unified_fc_tp_pairwise_hidden_error_reducer #(
        .MAX_BATCH(MAX_BATCH),
        .MAX_HIDDEN(MAX_HIDDEN),
        .BATCH_INDEX_WIDTH(BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .HIDDEN_COUNT_WIDTH(HIDDEN_COUNT_WIDTH)
    ) u_reducer (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(i_cfg_valid && o_cfg_ready && cfg_legal),
        .o_cfg_ready(reducer_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .o_cfg_error(reducer_cfg_error),
        .o_active(reducer_active),
        .i_release(i_release && o_release_ready),
        .o_release_ready(reducer_release_ready),
        .i_student_probability_valid(
            row_probability_valid && !row_distance_mode_q),
        .o_student_probability_ready(reducer_student_ready),
        .i_student_probability_row(row_probability_row),
        .i_student_probability_column(row_probability_column),
        .i_student_probability_q016(row_probability_q016),
        .o_student_probability_error(reducer_student_error),
        .i_target_probability_valid(
            row_probability_valid && row_distance_mode_q),
        .o_target_probability_ready(reducer_target_ready),
        .i_target_probability_row(row_probability_row),
        .i_target_probability_column(row_probability_column),
        .i_target_probability_q016(row_probability_q016),
        .o_target_probability_error(reducer_target_error),
        .i_target_trace_valid(
            i_target_hidden_trace_valid && o_target_hidden_trace_ready),
        .o_target_trace_ready(reducer_target_trace_ready),
        .i_target_trace_sample(i_target_hidden_trace_sample),
        .i_target_trace_hidden(i_target_hidden_trace_hidden),
        .i_target_trace_q016(i_target_hidden_trace_q016),
        .o_target_trace_error(reducer_target_trace_error),
        .i_surrogate_valid(i_surrogate_valid && o_surrogate_ready),
        .o_surrogate_ready(reducer_surrogate_ready),
        .i_surrogate_sample(i_surrogate_sample),
        .i_surrogate_hidden(i_surrogate_hidden),
        .i_surrogate_q016(i_surrogate_q016),
        .o_surrogate_error(reducer_surrogate_error),
        .i_observation_done(
            (state_q == STATE_SEAL_REDUCER) && reducer_observation_done_ready),
        .o_observation_done_ready(reducer_observation_done_ready),
        .i_emit_start((state_q == STATE_START_EMIT) && reducer_emit_ready),
        .o_emit_ready(reducer_emit_ready),
        .o_hidden_error_valid(reducer_hidden_error_valid),
        .i_hidden_error_ready(i_hidden_error_ready),
        .o_hidden_error_sample(o_hidden_error_sample),
        .o_hidden_error_hidden(o_hidden_error_hidden),
        .o_hidden_error_q016(o_hidden_error_q016),
        .o_arithmetic_overflow(reducer_arithmetic_overflow),
        .o_emit_done(reducer_emit_done),
        .o_busy(reducer_busy)
    );

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            target_dimension_q <= '0;
            student_sample_q <= '0;
            student_hidden_q <= '0;
            target_hidden_sample_q <= '0;
            target_hidden_q <= '0;
            target_input_sample_q <= '0;
            target_input_index_q <= '0;
            surrogate_sample_q <= '0;
            surrogate_hidden_q <= '0;
            student_complete_q <= 1'b0;
            target_hidden_complete_q <= 1'b0;
            target_input_complete_q <= 1'b0;
            surrogate_complete_q <= 1'b0;
            row_distance_mode_q <= 1'b0;
            row_index_q <= '0;
            column_index_q <= '0;
            vector_index_q <= '0;
            pair_pending_q <= 1'b0;
            pair_value_a_q <= '0;
            pair_value_b_q <= '0;
            arithmetic_overflow_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_student_trace_error <= 1'b0;
            o_target_hidden_trace_error <= 1'b0;
            o_target_input_trace_error <= 1'b0;
            o_surrogate_error <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_student_trace_error <= 1'b0;
            o_target_hidden_trace_error <= 1'b0;
            o_target_input_trace_error <= 1'b0;
            o_surrogate_error <= 1'b0;
            if (row_arithmetic_overflow || reducer_arithmetic_overflow)
                arithmetic_overflow_q <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (cfg_fire) begin
                        if (!cfg_legal) begin
                            o_cfg_error <= 1'b1;
                        end else begin
                            batch_count_q <= i_cfg_batch_count;
                            hidden_count_q <= i_cfg_hidden_count;
                            target_dimension_q <= i_cfg_target_dimension;
                            student_sample_q <= '0;
                            student_hidden_q <= '0;
                            target_hidden_sample_q <= '0;
                            target_hidden_q <= '0;
                            target_input_sample_q <= '0;
                            target_input_index_q <= '0;
                            surrogate_sample_q <= '0;
                            surrogate_hidden_q <= '0;
                            student_complete_q <= 1'b0;
                            target_hidden_complete_q <= 1'b0;
                            target_input_complete_q <= 1'b0;
                            surrogate_complete_q <= 1'b0;
                            arithmetic_overflow_q <= 1'b0;
                            state_q <= STATE_LOAD;
                        end
                    end
                end

                STATE_LOAD: begin
                    if (release_fire) begin
                        state_q <= STATE_IDLE;
                    end else begin
                        if (student_fire) begin
                            if (!student_matches) begin
                                o_student_trace_error <= 1'b1;
                            end else begin
                                student_trace_mem[student_write_addr] <=
                                    i_student_trace_q016;
                                if (student_last) begin
                                    student_complete_q <= 1'b1;
                                end else if (({1'b0, student_hidden_q} + 1'b1) >=
                                             {1'b0, hidden_count_q}) begin
                                    student_sample_q <= student_sample_q + 1'b1;
                                    student_hidden_q <= '0;
                                end else begin
                                    student_hidden_q <= student_hidden_q + 1'b1;
                                end
                            end
                        end
                        if (target_hidden_fire) begin
                            if (!target_hidden_matches) begin
                                o_target_hidden_trace_error <= 1'b1;
                            end else begin
                                target_hidden_trace_mem[target_hidden_write_addr] <=
                                    i_target_hidden_trace_q016;
                                if (target_hidden_last) begin
                                    target_hidden_complete_q <= 1'b1;
                                end else if (({1'b0, target_hidden_q} + 1'b1) >=
                                             {1'b0, hidden_count_q}) begin
                                    target_hidden_sample_q <=
                                        target_hidden_sample_q + 1'b1;
                                    target_hidden_q <= '0;
                                end else begin
                                    target_hidden_q <= target_hidden_q + 1'b1;
                                end
                            end
                        end
                        if (target_input_fire) begin
                            if (!target_input_matches) begin
                                o_target_input_trace_error <= 1'b1;
                            end else begin
                                target_input_trace_mem[target_input_write_addr] <=
                                    i_target_input_trace_q016;
                                if (target_input_last) begin
                                    target_input_complete_q <= 1'b1;
                                end else if (({1'b0, target_input_index_q} + 1'b1) >=
                                             {1'b0, target_dimension_q}) begin
                                    target_input_sample_q <=
                                        target_input_sample_q + 1'b1;
                                    target_input_index_q <= '0;
                                end else begin
                                    target_input_index_q <=
                                        target_input_index_q + 1'b1;
                                end
                            end
                        end
                        if (surrogate_fire) begin
                            if (!surrogate_matches) begin
                                o_surrogate_error <= 1'b1;
                            end else if (surrogate_last) begin
                                surrogate_complete_q <= 1'b1;
                            end else if (({1'b0, surrogate_hidden_q} + 1'b1) >=
                                         {1'b0, hidden_count_q}) begin
                                surrogate_sample_q <= surrogate_sample_q + 1'b1;
                                surrogate_hidden_q <= '0;
                            end else begin
                                surrogate_hidden_q <= surrogate_hidden_q + 1'b1;
                            end
                        end
                        if (observation_done_fire) begin
                            row_distance_mode_q <= 1'b0;
                            row_index_q <= '0;
                            column_index_q <= '0;
                            vector_index_q <= '0;
                            pair_pending_q <= 1'b0;
                            state_q <= STATE_START_ROW;
                        end
                    end
                end

                STATE_START_ROW: begin
                    if (row_start_fire) begin
                        column_index_q <= '0;
                        vector_index_q <= '0;
                        pair_pending_q <= 1'b0;
                        state_q <= STATE_STREAM_ROW;
                    end
                end

                STATE_STREAM_ROW: begin
                    if (!pair_pending_q) begin
                        if (row_distance_mode_q) begin
                            pair_value_a_q <= clip_q10_from_q16(target_input_a_rounded);
                            pair_value_b_q <= clip_q10_from_q16(target_input_b_rounded);
                        end else begin
                            pair_value_a_q <= clip_q10_from_q16(
                                (TARGET_TRACE_WIDTH+1)'(student_rounded));
                            pair_value_b_q <= clip_q10_from_q16(
                                (TARGET_TRACE_WIDTH+1)'(target_hidden_rounded));
                        end
                        if (pair_read_overflow)
                            arithmetic_overflow_q <= 1'b1;
                        pair_pending_q <= 1'b1;
                    end
                    if (pair_fire) begin
                        pair_pending_q <= 1'b0;
                        if (row_pair_last) begin
                            state_q <= STATE_WAIT_ROW;
                        end else if (({1'b0, vector_index_q} + 1'b1) >=
                                     {1'b0, row_vector_length}) begin
                            column_index_q <= column_index_q + 1'b1;
                            vector_index_q <= '0;
                        end else begin
                            vector_index_q <= vector_index_q + 1'b1;
                        end
                    end
                end

                STATE_WAIT_ROW: begin
                    if (row_done) begin
                        if (row_last) begin
                            if (row_distance_mode_q) begin
                                state_q <= STATE_SEAL_REDUCER;
                            end else begin
                                row_distance_mode_q <= 1'b1;
                                row_index_q <= '0;
                                state_q <= STATE_START_ROW;
                            end
                        end else begin
                            row_index_q <= row_index_q + 1'b1;
                            state_q <= STATE_START_ROW;
                        end
                    end
                end

                STATE_SEAL_REDUCER: begin
                    if (reducer_observation_done_ready)
                        state_q <= STATE_START_EMIT;
                end

                STATE_START_EMIT: begin
                    if (reducer_emit_ready)
                        state_q <= STATE_EMIT;
                end

                STATE_EMIT: begin
                    if (hidden_error_fire && hidden_error_last) begin
                        student_sample_q <= '0;
                        student_hidden_q <= '0;
                        target_hidden_sample_q <= '0;
                        target_hidden_q <= '0;
                        target_input_sample_q <= '0;
                        target_input_index_q <= '0;
                        surrogate_sample_q <= '0;
                        surrogate_hidden_q <= '0;
                        student_complete_q <= 1'b0;
                        target_hidden_complete_q <= 1'b0;
                        target_input_complete_q <= 1'b0;
                        surrogate_complete_q <= 1'b0;
                        pair_pending_q <= 1'b0;
                        state_q <= STATE_LOAD;
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
