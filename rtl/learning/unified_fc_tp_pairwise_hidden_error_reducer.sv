`timescale 1ns/1ps
`default_nettype none

// FC TP pairwise reducer for B16 online learning. It consumes the two Q0.16
// softmax matrices, target hidden traces and local surrogate values, then emits
// hidden-error tokens in sample-major order for the sparse W0 update path.
module unified_fc_tp_pairwise_hidden_error_reducer #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int BATCH_INDEX_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [BATCH_COUNT_WIDTH-1:0]           i_cfg_batch_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]          i_cfg_hidden_count,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    input  logic                                  i_release,
    output logic                                  o_release_ready,
    input  logic                                  i_student_probability_valid,
    output logic                                  o_student_probability_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]           i_student_probability_row,
    input  logic [BATCH_INDEX_WIDTH-1:0]           i_student_probability_column,
    input  logic [16:0]                           i_student_probability_q016,
    output logic                                  o_student_probability_error,
    input  logic                                  i_target_probability_valid,
    output logic                                  o_target_probability_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]           i_target_probability_row,
    input  logic [BATCH_INDEX_WIDTH-1:0]           i_target_probability_column,
    input  logic [16:0]                           i_target_probability_q016,
    output logic                                  o_target_probability_error,
    input  logic                                  i_target_trace_valid,
    output logic                                  o_target_trace_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]           i_target_trace_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]          i_target_trace_hidden,
    input  logic [16:0]                           i_target_trace_q016,
    output logic                                  o_target_trace_error,
    input  logic                                  i_surrogate_valid,
    output logic                                  o_surrogate_ready,
    input  logic [BATCH_INDEX_WIDTH-1:0]           i_surrogate_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]          i_surrogate_hidden,
    input  logic [16:0]                           i_surrogate_q016,
    output logic                                  o_surrogate_error,
    input  logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,
    input  logic                                  i_emit_start,
    output logic                                  o_emit_ready,
    output logic                                  o_hidden_error_valid,
    input  logic                                  i_hidden_error_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]           o_hidden_error_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]          o_hidden_error_hidden,
    output logic signed [17:0]                    o_hidden_error_q016,
    output logic                                  o_arithmetic_overflow,
    output logic                                  o_emit_done,
    output logic                                  o_busy
);

    localparam int MATRIX_DEPTH = MAX_BATCH * MAX_BATCH;
    localparam int HIDDEN_DEPTH = MAX_BATCH * MAX_HIDDEN;
    localparam int MATRIX_ADDR_WIDTH = (MATRIX_DEPTH <= 2) ? 1 : $clog2(MATRIX_DEPTH);
    localparam int HIDDEN_ADDR_WIDTH = (HIDDEN_DEPTH <= 2) ? 1 : $clog2(HIDDEN_DEPTH);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_OBSERVE,
        STATE_SEALED,
        STATE_PAIR_READ,
        STATE_PAIR_MAC,
        STATE_SURROGATE_READ,
        STATE_SURROGATE_CALC,
        STATE_OUTPUT
    } state_t;

    state_t state_q;
    logic [16:0] student_probability_mem [0:MATRIX_DEPTH-1];
    logic [16:0] target_probability_mem [0:MATRIX_DEPTH-1];
    logic [16:0] target_trace_mem [0:HIDDEN_DEPTH-1];
    logic [16:0] surrogate_mem [0:HIDDEN_DEPTH-1];
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_COUNT_WIDTH-1:0] hidden_count_q;

    logic [BATCH_INDEX_WIDTH-1:0] student_row_q;
    logic [BATCH_INDEX_WIDTH-1:0] student_column_q;
    logic [BATCH_INDEX_WIDTH-1:0] target_row_q;
    logic [BATCH_INDEX_WIDTH-1:0] target_column_q;
    logic [BATCH_INDEX_WIDTH-1:0] target_trace_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] target_trace_hidden_q;
    logic [BATCH_INDEX_WIDTH-1:0] surrogate_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] surrogate_hidden_q;
    logic student_complete_q;
    logic target_complete_q;
    logic target_trace_complete_q;
    logic surrogate_complete_q;

    logic [BATCH_INDEX_WIDTH-1:0] emit_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] emit_hidden_q;
    logic [BATCH_INDEX_WIDTH-1:0] pair_index_q;
    logic [16:0] student_probability_q;
    logic [16:0] target_probability_q;
    logic [16:0] target_trace_q;
    logic [16:0] surrogate_q;
    logic signed [47:0] pair_accumulator_q;
    logic signed [47:0] hidden_pre_q;
    logic signed [17:0] hidden_error_q;

    logic cfg_legal;
    logic cfg_fire;
    logic release_fire;
    logic student_fire;
    logic target_fire;
    logic target_trace_fire;
    logic surrogate_fire;
    logic observation_done_fire;
    logic emit_fire;
    logic hidden_error_fire;
    logic student_matches;
    logic target_matches;
    logic target_trace_matches;
    logic surrogate_matches;
    logic student_input_last;
    logic target_input_last;
    logic target_trace_input_last;
    logic surrogate_input_last;
    logic pair_last;
    logic emit_hidden_last;
    logic emit_sample_last;
    logic [MATRIX_ADDR_WIDTH-1:0] student_write_addr;
    logic [MATRIX_ADDR_WIDTH-1:0] target_write_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] target_trace_write_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] surrogate_write_addr;
    logic [MATRIX_ADDR_WIDTH-1:0] emit_matrix_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] emit_target_trace_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] emit_surrogate_addr;

    logic signed [17:0] probability_delta;
    logic signed [17:0] pair_error;
    logic signed [17:0] target_trace_extended;
    logic signed [35:0] pair_product;
    logic signed [47:0] pair_product_extended;
    logic signed [47:0] pair_accumulator_next;
    logic signed [17:0] surrogate_extended;
    logic signed [65:0] surrogate_product;
    logic signed [65:0] hidden_error_wide;
    logic hidden_error_overflow;
    logic signed [17:0] hidden_error_clamped;

    function automatic logic [MATRIX_ADDR_WIDTH-1:0] matrix_address(
        input logic [BATCH_INDEX_WIDTH-1:0] row_index,
        input logic [BATCH_INDEX_WIDTH-1:0] column_index
    );
        matrix_address = MATRIX_ADDR_WIDTH'((row_index * MAX_BATCH) + column_index);
    endfunction

    function automatic logic [HIDDEN_ADDR_WIDTH-1:0] hidden_address(
        input logic [BATCH_INDEX_WIDTH-1:0] sample_index,
        input logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index
    );
        hidden_address = HIDDEN_ADDR_WIDTH'((sample_index * MAX_HIDDEN) + hidden_index);
    endfunction

    function automatic logic signed [17:0] round_shift_4(
        input logic signed [17:0] value
    );
        logic [17:0] magnitude;
        logic [17:0] rounded;
        begin
            if (value < 0) begin
                magnitude = $unsigned(-value);
                rounded = (magnitude + 18'd8) >> 4;
                round_shift_4 = -$signed(rounded);
            end else begin
                rounded = ($unsigned(value) + 18'd8) >> 4;
                round_shift_4 = $signed(rounded);
            end
        end
    endfunction

    function automatic logic signed [47:0] round_shift_16_48(
        input logic signed [47:0] value
    );
        logic [47:0] magnitude;
        logic [47:0] rounded;
        begin
            if (value < 0) begin
                magnitude = $unsigned(-value);
                rounded = (magnitude + 48'd32768) >> 16;
                round_shift_16_48 = -$signed(rounded);
            end else begin
                rounded = ($unsigned(value) + 48'd32768) >> 16;
                round_shift_16_48 = $signed(rounded);
            end
        end
    endfunction

    function automatic logic signed [65:0] round_shift_16_66(
        input logic signed [65:0] value
    );
        logic [65:0] magnitude;
        logic [65:0] rounded;
        begin
            if (value < 0) begin
                magnitude = $unsigned(-value);
                rounded = (magnitude + 66'd32768) >> 16;
                round_shift_16_66 = -$signed(rounded);
            end else begin
                rounded = ($unsigned(value) + 66'd32768) >> 16;
                round_shift_16_66 = $signed(rounded);
            end
        end
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_batch_count == BATCH_COUNT_WIDTH'(16)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_COUNT_WIDTH'(MAX_HIDDEN));
        o_cfg_ready = state_q == STATE_IDLE;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_release_ready = (state_q == STATE_OBSERVE) && o_active &&
            !student_complete_q && !target_complete_q &&
            !target_trace_complete_q && !surrogate_complete_q &&
            (student_row_q == '0) && (student_column_q == '0) &&
            (target_row_q == '0) && (target_column_q == '0) &&
            (target_trace_sample_q == '0) && (target_trace_hidden_q == '0) &&
            (surrogate_sample_q == '0) && (surrogate_hidden_q == '0);
        release_fire = i_release && o_release_ready;

        o_student_probability_ready = (state_q == STATE_OBSERVE) && o_active &&
            !student_complete_q;
        student_fire = i_student_probability_valid && o_student_probability_ready;
        student_matches = (i_student_probability_row == student_row_q) &&
            (i_student_probability_column == student_column_q);
        student_input_last =
            ({1'b0, student_row_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, student_column_q} + 1'b1 >= batch_count_q);

        o_target_probability_ready = (state_q == STATE_OBSERVE) && o_active &&
            !target_complete_q;
        target_fire = i_target_probability_valid && o_target_probability_ready;
        target_matches = (i_target_probability_row == target_row_q) &&
            (i_target_probability_column == target_column_q);
        target_input_last =
            ({1'b0, target_row_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, target_column_q} + 1'b1 >= batch_count_q);

        o_target_trace_ready = (state_q == STATE_OBSERVE) && o_active &&
            !target_trace_complete_q;
        target_trace_fire = i_target_trace_valid && o_target_trace_ready;
        target_trace_matches = (i_target_trace_sample == target_trace_sample_q) &&
            (i_target_trace_hidden == target_trace_hidden_q);
        target_trace_input_last =
            ({1'b0, target_trace_sample_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, target_trace_hidden_q} + 1'b1 >= {1'b0, hidden_count_q});

        o_surrogate_ready = (state_q == STATE_OBSERVE) && o_active &&
            !surrogate_complete_q;
        surrogate_fire = i_surrogate_valid && o_surrogate_ready;
        surrogate_matches = (i_surrogate_sample == surrogate_sample_q) &&
            (i_surrogate_hidden == surrogate_hidden_q);
        surrogate_input_last =
            ({1'b0, surrogate_sample_q} + 1'b1 >= batch_count_q) &&
            ({1'b0, surrogate_hidden_q} + 1'b1 >= {1'b0, hidden_count_q});

        o_observation_done_ready = (state_q == STATE_OBSERVE) && o_active &&
            student_complete_q && target_complete_q &&
            target_trace_complete_q && surrogate_complete_q;
        observation_done_fire = i_observation_done && o_observation_done_ready;
        o_emit_ready = state_q == STATE_SEALED;
        emit_fire = i_emit_start && o_emit_ready;
        o_hidden_error_valid = state_q == STATE_OUTPUT;
        hidden_error_fire = o_hidden_error_valid && i_hidden_error_ready;
        o_hidden_error_sample = emit_sample_q;
        o_hidden_error_hidden = emit_hidden_q;
        o_hidden_error_q016 = hidden_error_q;
        o_busy = (state_q != STATE_IDLE) && (state_q != STATE_OBSERVE) &&
            (state_q != STATE_SEALED);

        pair_last = ({1'b0, pair_index_q} + 1'b1) >= batch_count_q;
        emit_hidden_last = ({1'b0, emit_hidden_q} + 1'b1) >=
            {1'b0, hidden_count_q};
        emit_sample_last = ({1'b0, emit_sample_q} + 1'b1) >= batch_count_q;
        student_write_addr = matrix_address(
            i_student_probability_row, i_student_probability_column);
        target_write_addr = matrix_address(
            i_target_probability_row, i_target_probability_column);
        target_trace_write_addr = hidden_address(
            i_target_trace_sample, i_target_trace_hidden);
        surrogate_write_addr = hidden_address(i_surrogate_sample, i_surrogate_hidden);
        emit_matrix_addr = matrix_address(emit_sample_q, pair_index_q);
        emit_target_trace_addr = hidden_address(pair_index_q, emit_hidden_q);
        emit_surrogate_addr = hidden_address(emit_sample_q, emit_hidden_q);

        probability_delta = $signed({1'b0, student_probability_q}) -
            $signed({1'b0, target_probability_q});
        pair_error = round_shift_4(probability_delta);
        target_trace_extended = $signed({1'b0, target_trace_q});
        pair_product = pair_error * target_trace_extended;
        pair_product_extended = {{12{pair_product[35]}}, pair_product};
        pair_accumulator_next = pair_accumulator_q + pair_product_extended;
        surrogate_extended = $signed({1'b0, surrogate_q});
        surrogate_product = hidden_pre_q * surrogate_extended;
        hidden_error_wide = round_shift_16_66(surrogate_product);
        hidden_error_overflow = 1'b0;
        if (hidden_error_wide > 66'sd131071) begin
            hidden_error_clamped = 18'sd131071;
            hidden_error_overflow = 1'b1;
        end else if (hidden_error_wide < -66'sd131072) begin
            hidden_error_clamped = -18'sd131072;
            hidden_error_overflow = 1'b1;
        end else begin
            hidden_error_clamped = hidden_error_wide[17:0];
        end
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            student_row_q <= '0;
            student_column_q <= '0;
            target_row_q <= '0;
            target_column_q <= '0;
            target_trace_sample_q <= '0;
            target_trace_hidden_q <= '0;
            surrogate_sample_q <= '0;
            surrogate_hidden_q <= '0;
            student_complete_q <= 1'b0;
            target_complete_q <= 1'b0;
            target_trace_complete_q <= 1'b0;
            surrogate_complete_q <= 1'b0;
            emit_sample_q <= '0;
            emit_hidden_q <= '0;
            pair_index_q <= '0;
            student_probability_q <= '0;
            target_probability_q <= '0;
            target_trace_q <= '0;
            surrogate_q <= '0;
            pair_accumulator_q <= '0;
            hidden_pre_q <= '0;
            hidden_error_q <= '0;
            o_active <= 1'b0;
            o_cfg_error <= 1'b0;
            o_student_probability_error <= 1'b0;
            o_target_probability_error <= 1'b0;
            o_target_trace_error <= 1'b0;
            o_surrogate_error <= 1'b0;
            o_arithmetic_overflow <= 1'b0;
            o_emit_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_student_probability_error <= 1'b0;
            o_target_probability_error <= 1'b0;
            o_target_trace_error <= 1'b0;
            o_surrogate_error <= 1'b0;
            o_arithmetic_overflow <= 1'b0;
            o_emit_done <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (cfg_fire) begin
                        if (!cfg_legal) begin
                            o_cfg_error <= 1'b1;
                        end else begin
                            batch_count_q <= i_cfg_batch_count;
                            hidden_count_q <= i_cfg_hidden_count;
                            student_row_q <= '0;
                            student_column_q <= '0;
                            target_row_q <= '0;
                            target_column_q <= '0;
                            target_trace_sample_q <= '0;
                            target_trace_hidden_q <= '0;
                            surrogate_sample_q <= '0;
                            surrogate_hidden_q <= '0;
                            student_complete_q <= 1'b0;
                            target_complete_q <= 1'b0;
                            target_trace_complete_q <= 1'b0;
                            surrogate_complete_q <= 1'b0;
                            o_active <= 1'b1;
                            state_q <= STATE_OBSERVE;
                        end
                    end
                end

                STATE_OBSERVE: begin
                    if (release_fire) begin
                        o_active <= 1'b0;
                        state_q <= STATE_IDLE;
                    end else begin
                        if (student_fire) begin
                            if (!student_matches) begin
                                o_student_probability_error <= 1'b1;
                            end else begin
                                student_probability_mem[student_write_addr] <=
                                    i_student_probability_q016;
                                if (student_input_last) begin
                                    student_complete_q <= 1'b1;
                                end else if (({1'b0, student_column_q} + 1'b1) >=
                                             batch_count_q) begin
                                    student_row_q <= student_row_q + 1'b1;
                                    student_column_q <= '0;
                                end else begin
                                    student_column_q <= student_column_q + 1'b1;
                                end
                            end
                        end
                        if (target_fire) begin
                            if (!target_matches) begin
                                o_target_probability_error <= 1'b1;
                            end else begin
                                target_probability_mem[target_write_addr] <=
                                    i_target_probability_q016;
                                if (target_input_last) begin
                                    target_complete_q <= 1'b1;
                                end else if (({1'b0, target_column_q} + 1'b1) >=
                                             batch_count_q) begin
                                    target_row_q <= target_row_q + 1'b1;
                                    target_column_q <= '0;
                                end else begin
                                    target_column_q <= target_column_q + 1'b1;
                                end
                            end
                        end
                        if (target_trace_fire) begin
                            if (!target_trace_matches) begin
                                o_target_trace_error <= 1'b1;
                            end else begin
                                target_trace_mem[target_trace_write_addr] <=
                                    i_target_trace_q016;
                                if (target_trace_input_last) begin
                                    target_trace_complete_q <= 1'b1;
                                end else if (({1'b0, target_trace_hidden_q} + 1'b1) >=
                                             {1'b0, hidden_count_q}) begin
                                    target_trace_sample_q <= target_trace_sample_q + 1'b1;
                                    target_trace_hidden_q <= '0;
                                end else begin
                                    target_trace_hidden_q <= target_trace_hidden_q + 1'b1;
                                end
                            end
                        end
                        if (surrogate_fire) begin
                            if (!surrogate_matches) begin
                                o_surrogate_error <= 1'b1;
                            end else begin
                                surrogate_mem[surrogate_write_addr] <= i_surrogate_q016;
                                if (surrogate_input_last) begin
                                    surrogate_complete_q <= 1'b1;
                                end else if (({1'b0, surrogate_hidden_q} + 1'b1) >=
                                             {1'b0, hidden_count_q}) begin
                                    surrogate_sample_q <= surrogate_sample_q + 1'b1;
                                    surrogate_hidden_q <= '0;
                                end else begin
                                    surrogate_hidden_q <= surrogate_hidden_q + 1'b1;
                                end
                            end
                        end
                        if (observation_done_fire)
                            state_q <= STATE_SEALED;
                    end
                end

                STATE_SEALED: begin
                    if (emit_fire) begin
                        emit_sample_q <= '0;
                        emit_hidden_q <= '0;
                        pair_index_q <= '0;
                        pair_accumulator_q <= '0;
                        state_q <= STATE_PAIR_READ;
                    end
                end

                STATE_PAIR_READ: begin
                    student_probability_q <= student_probability_mem[emit_matrix_addr];
                    target_probability_q <= target_probability_mem[emit_matrix_addr];
                    target_trace_q <= target_trace_mem[emit_target_trace_addr];
                    state_q <= STATE_PAIR_MAC;
                end

                STATE_PAIR_MAC: begin
                    if (pair_last) begin
                        hidden_pre_q <= round_shift_16_48(pair_accumulator_next);
                        state_q <= STATE_SURROGATE_READ;
                    end else begin
                        pair_accumulator_q <= pair_accumulator_next;
                        pair_index_q <= pair_index_q + 1'b1;
                        state_q <= STATE_PAIR_READ;
                    end
                end

                STATE_SURROGATE_READ: begin
                    surrogate_q <= surrogate_mem[emit_surrogate_addr];
                    state_q <= STATE_SURROGATE_CALC;
                end

                STATE_SURROGATE_CALC: begin
                    hidden_error_q <= hidden_error_clamped;
                    o_arithmetic_overflow <= hidden_error_overflow;
                    state_q <= STATE_OUTPUT;
                end

                STATE_OUTPUT: begin
                    if (hidden_error_fire) begin
                        if (emit_hidden_last) begin
                            if (emit_sample_last) begin
                                student_row_q <= '0;
                                student_column_q <= '0;
                                target_row_q <= '0;
                                target_column_q <= '0;
                                target_trace_sample_q <= '0;
                                target_trace_hidden_q <= '0;
                                surrogate_sample_q <= '0;
                                surrogate_hidden_q <= '0;
                                student_complete_q <= 1'b0;
                                target_complete_q <= 1'b0;
                                target_trace_complete_q <= 1'b0;
                                surrogate_complete_q <= 1'b0;
                                o_emit_done <= 1'b1;
                                state_q <= STATE_OBSERVE;
                            end else begin
                                emit_sample_q <= emit_sample_q + 1'b1;
                                emit_hidden_q <= '0;
                                pair_index_q <= '0;
                                pair_accumulator_q <= '0;
                                state_q <= STATE_PAIR_READ;
                            end
                        end else begin
                            emit_hidden_q <= emit_hidden_q + 1'b1;
                            pair_index_q <= '0;
                            pair_accumulator_q <= '0;
                            state_q <= STATE_PAIR_READ;
                        end
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
