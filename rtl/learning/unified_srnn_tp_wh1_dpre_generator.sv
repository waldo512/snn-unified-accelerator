`timescale 1ns/1ps
`default_nettype none

// Generates the two M2 W_h1 hidden-delta paths from pairwise TP error and the
// opposite trace. The topology adapter supplies the quantized LIF surrogate
// factor so this shared learning block does not replicate a divider.
module unified_srnn_tp_wh1_dpre_generator #(
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int ERROR_WIDTH = 64,
    parameter int TRACE_WIDTH = 32,
    parameter int DPRE_WIDTH = 64,
    parameter int ACC_WIDTH = 128
) (
    input  logic                                     i_clk,
    input  logic                                     i_rst_n,

    input  logic                                     i_cfg_valid,
    output logic                                     o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]               i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]            i_cfg_hidden_count,
    output logic                                     o_cfg_error,

    input  logic                                     i_start,
    output logic                                     o_start_ready,
    input  logic                                     i_abort,

    input  logic                                     i_contribution_valid,
    output logic                                     o_contribution_ready,
    input  logic                                     i_contribution_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]               i_contribution_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]            i_contribution_hidden_id,
    input  logic [SAMPLE_ID_WIDTH-1:0]               i_contribution_column_id,
    input  logic signed [ERROR_WIDTH-1:0]            i_error_q40,
    input  logic signed [TRACE_WIDTH-1:0]            i_opposite_trace_q14,
    input  logic [24:0]                              i_surrogate_uq24,
    input  logic                                     i_contribution_tensor_last,

    output logic                                     o_dpre_valid,
    input  logic                                     i_dpre_ready,
    output logic                                     o_dpre_path,
    output logic [SAMPLE_ID_WIDTH-1:0]               o_dpre_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]            o_dpre_hidden_id,
    output logic signed [DPRE_WIDTH-1:0]             o_dpre_q40,
    output logic                                     o_dpre_tensor_last,

    output logic                                     o_busy,
    output logic                                     o_done,
    output logic                                     o_protocol_error,
    output logic                                     o_overflow
);

    logic configured_q;
    logic active_q;
    logic finalize_sum_q;
    logic finalize_q;
    logic pending_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_count_q;
    logic expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] expected_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] expected_hidden_q;
    logic [SAMPLE_ID_WIDTH-1:0] expected_column_q;
    logic [24:0] surrogate_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [DPRE_WIDTH-1:0] finalize_incoming_q40_q;
    logic finalize_incoming_overflow_q;
    logic [24:0] finalize_surrogate_q;
    logic pending_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] pending_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] pending_hidden_q;
    logic signed [DPRE_WIDTH-1:0] pending_dpre_q;
    logic pending_last_q;

    logic cfg_legal;
    logic expected_last_column;
    logic expected_last_dpre;
    logic factor_matches;
    logic input_legal;
    logic input_fire;
    logic output_fire;
    logic signed [ERROR_WIDTH+TRACE_WIDTH-1:0] contribution_product;
    logic signed [ACC_WIDTH-1:0] contribution_extended;
    logic signed [ACC_WIDTH-1:0] accumulated_with_current;
    logic signed [ACC_WIDTH-1:0] incoming_rounded;
    logic incoming_overflow;
    logic signed [25:0] surrogate_factor_signed;
    logic signed [DPRE_WIDTH+26-1:0] surrogate_product;
    logic signed [ACC_WIDTH-1:0] surrogate_extended;
    logic signed [ACC_WIDTH-1:0] dpre_rounded;
    logic dpre_overflow;

    function automatic logic signed [ACC_WIDTH-1:0] round_shift_even(
        input logic signed [ACC_WIDTH-1:0] value,
        input integer shift
    );
        logic negative;
        logic [ACC_WIDTH-1:0] magnitude;
        logic [ACC_WIDTH-1:0] quotient;
        logic [ACC_WIDTH-1:0] mask;
        logic [ACC_WIDTH-1:0] remainder;
        logic [ACC_WIDTH-1:0] half;
        begin
            negative = value[ACC_WIDTH-1];
            magnitude = negative ? -value : value;
            quotient = magnitude >> shift;
            mask = ({ACC_WIDTH{1'b1}} >> (ACC_WIDTH-shift));
            remainder = magnitude & mask;
            half = {{(ACC_WIDTH-1){1'b0}}, 1'b1} << (shift-1);
            if ((remainder > half) ||
                ((remainder == half) && quotient[0]))
                quotient = quotient + 1'b1;
            round_shift_even = negative ? -$signed(quotient) :
                $signed(quotient);
        end
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_hidden_count != '0);
        o_cfg_ready = !active_q && !finalize_sum_q && !finalize_q &&
            !pending_q;
        o_start_ready = configured_q && !active_q && !finalize_sum_q &&
            !finalize_q && !pending_q && !i_cfg_valid;
        o_busy = active_q || finalize_sum_q || finalize_q || pending_q;

        expected_last_column =
            expected_column_q == (batch_count_q - 1'b1);
        expected_last_dpre = expected_last_column && expected_path_q &&
            (expected_sample_q == (batch_count_q - 1'b1)) &&
            (expected_hidden_q == (hidden_count_q - 1'b1));
        factor_matches = (expected_column_q == '0) ||
            (i_surrogate_uq24 == surrogate_q);
        input_legal = (i_contribution_path == expected_path_q) &&
            (i_contribution_sample_id == expected_sample_q) &&
            (i_contribution_hidden_id == expected_hidden_q) &&
            (i_contribution_column_id == expected_column_q) &&
            factor_matches &&
            (i_contribution_tensor_last == expected_last_dpre);
        o_contribution_ready = active_q && !finalize_sum_q && !finalize_q &&
            !pending_q &&
            (input_legal || i_contribution_valid);
        input_fire = i_contribution_valid && o_contribution_ready;

        contribution_product = $signed(i_error_q40) *
            $signed(i_opposite_trace_q14);
        contribution_extended = {{(ACC_WIDTH-(ERROR_WIDTH+TRACE_WIDTH)){
            contribution_product[ERROR_WIDTH+TRACE_WIDTH-1]}},
            contribution_product};
        accumulated_with_current = accumulator_q + contribution_extended;
        incoming_rounded = round_shift_even(accumulator_q, 14);
        incoming_overflow =
            incoming_rounded[ACC_WIDTH-1:DPRE_WIDTH] !=
            {(ACC_WIDTH-DPRE_WIDTH){incoming_rounded[DPRE_WIDTH-1]}};
        surrogate_factor_signed = $signed({1'b0, finalize_surrogate_q});
        surrogate_product = $signed(finalize_incoming_q40_q) *
            surrogate_factor_signed;
        surrogate_extended = {{(ACC_WIDTH-(DPRE_WIDTH+26)){
            surrogate_product[DPRE_WIDTH+26-1]}}, surrogate_product};
        dpre_rounded = round_shift_even(surrogate_extended, 24);
        dpre_overflow =
            dpre_rounded[ACC_WIDTH-1:DPRE_WIDTH] !=
            {(ACC_WIDTH-DPRE_WIDTH){dpre_rounded[DPRE_WIDTH-1]}};

        o_dpre_valid = pending_q;
        o_dpre_path = pending_path_q;
        o_dpre_sample_id = pending_sample_q;
        o_dpre_hidden_id = pending_hidden_q;
        o_dpre_q40 = pending_dpre_q;
        o_dpre_tensor_last = pending_last_q;
        output_fire = o_dpre_valid && i_dpre_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            active_q <= 1'b0;
            finalize_sum_q <= 1'b0;
            finalize_q <= 1'b0;
            pending_q <= 1'b0;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            expected_path_q <= 1'b0;
            expected_sample_q <= '0;
            expected_hidden_q <= '0;
            expected_column_q <= '0;
            surrogate_q <= '0;
            accumulator_q <= '0;
            finalize_incoming_q40_q <= '0;
            finalize_incoming_overflow_q <= 1'b0;
            finalize_surrogate_q <= '0;
            pending_path_q <= 1'b0;
            pending_sample_q <= '0;
            pending_hidden_q <= '0;
            pending_dpre_q <= '0;
            pending_last_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_protocol_error <= 1'b0;
            o_overflow <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    hidden_count_q <= i_cfg_hidden_count;
                end
            end

            if (i_start && o_start_ready) begin
                active_q <= 1'b1;
                finalize_sum_q <= 1'b0;
                finalize_q <= 1'b0;
                expected_path_q <= 1'b0;
                expected_sample_q <= '0;
                expected_hidden_q <= '0;
                expected_column_q <= '0;
                surrogate_q <= '0;
                accumulator_q <= '0;
                o_protocol_error <= 1'b0;
                o_overflow <= 1'b0;
            end

            if (input_fire) begin
                if (!input_legal) begin
                    active_q <= 1'b0;
                    o_protocol_error <= 1'b1;
                    o_done <= 1'b1;
                end else begin
                    if (expected_column_q == '0)
                        surrogate_q <= i_surrogate_uq24;
                    if (expected_last_column) begin
                        finalize_sum_q <= 1'b1;
                        pending_path_q <= expected_path_q;
                        pending_sample_q <= expected_sample_q;
                        pending_hidden_q <= expected_hidden_q;
                        pending_last_q <= expected_last_dpre;
                        finalize_surrogate_q <= i_surrogate_uq24;
                        expected_column_q <= '0;
                        accumulator_q <= accumulated_with_current;
                    end else begin
                        accumulator_q <= accumulated_with_current;
                        expected_column_q <= expected_column_q + 1'b1;
                    end
                end
            end

            // The final product is first accumulated into a register. Rounding
            // that registered sum in the following cycle removes the RAM-read
            // through multiply/add/round path measured after route.
            if (finalize_sum_q) begin
                finalize_sum_q <= 1'b0;
                finalize_q <= 1'b1;
                finalize_incoming_q40_q <=
                    incoming_rounded[DPRE_WIDTH-1:0];
                finalize_incoming_overflow_q <= incoming_overflow;
                accumulator_q <= '0;
            end

            // The rounded incoming value is registered before the surrogate
            // multiply, splitting the remaining round/multiply chains.
            if (finalize_q) begin
                finalize_q <= 1'b0;
                pending_q <= 1'b1;
                pending_dpre_q <= dpre_rounded[DPRE_WIDTH-1:0];
                if (finalize_incoming_overflow_q || dpre_overflow)
                    o_overflow <= 1'b1;
            end

            if (output_fire) begin
                pending_q <= 1'b0;
                if (pending_last_q) begin
                    active_q <= 1'b0;
                    o_done <= 1'b1;
                end else if (expected_hidden_q ==
                    (hidden_count_q - 1'b1)) begin
                    expected_hidden_q <= '0;
                    if (expected_sample_q == (batch_count_q - 1'b1)) begin
                        expected_sample_q <= '0;
                        expected_path_q <= 1'b1;
                    end else begin
                        expected_sample_q <= expected_sample_q + 1'b1;
                    end
                end else begin
                    expected_hidden_q <= expected_hidden_q + 1'b1;
                end
            end

            if (i_abort &&
                (active_q || finalize_sum_q || finalize_q || pending_q)) begin
                active_q <= 1'b0;
                finalize_sum_q <= 1'b0;
                finalize_q <= 1'b0;
                pending_q <= 1'b0;
                accumulator_q <= '0;
                o_done <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
