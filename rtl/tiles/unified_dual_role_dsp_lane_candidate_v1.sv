`timescale 1ns/1ps
`default_nettype none

// Candidate arithmetic lane for the homogeneous FC/SRNN/SCNN tile.
// The datapath is deliberately written as P <- (A + D) * B + P so Vivado
// can map the pre-adder, multiplier and accumulator into one DSP48E1.
//
// Binary-spike dual issue:
//   A = spike0 ? weight0 : 0, D = spike1 ? weight1 : 0, B = 1
// Dense Q-format MAC / LIF / learning:
//   A = operand0, D = 0, B = operand1
module unified_dual_role_dsp_lane_candidate_v1 (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,

    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic                    i_first,
    input  wire logic                    i_last,
    input  wire logic signed [24:0]      i_term_a,
    input  wire logic signed [24:0]      i_term_d,
    input  wire logic signed [17:0]      i_factor_b,

    output logic                         o_result_valid,
    input  wire logic                    i_result_ready,
    output logic signed [47:0]           o_result
);
    (* use_dsp = "yes" *) logic signed [47:0] accumulator_q;
    logic signed [24:0] preadd_value;
    logic signed [42:0] product_value;
    logic signed [47:0] product_extended;
    logic signed [47:0] accumulate_value;
    logic result_valid_q;
    logic input_fire;

    always_comb begin
        preadd_value = $signed(i_term_a) + $signed(i_term_d);
        product_value = $signed(preadd_value) * $signed(i_factor_b);
        product_extended = {{5{product_value[42]}}, product_value};
        accumulate_value = i_first ? product_extended :
            accumulator_q + product_extended;
    end

    assign o_ready = !i_abort && (!result_valid_q || i_result_ready);
    assign input_fire = i_valid && o_ready;
    assign o_result_valid = !i_abort && result_valid_q;
    assign o_result = accumulator_q;

    // accumulator_q is intentionally reset-free. i_first is the sole
    // arithmetic initialization authority; abort only invalidates ownership.
    always_ff @(posedge i_clk) begin
        if (input_fire)
            accumulator_q <= accumulate_value;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            result_valid_q <= 1'b0;
        else if (i_abort)
            result_valid_q <= 1'b0;
        else begin
            if (result_valid_q && i_result_ready)
                result_valid_q <= 1'b0;
            if (input_fire)
                result_valid_q <= i_last;
        end
    end

endmodule

`default_nettype wire
