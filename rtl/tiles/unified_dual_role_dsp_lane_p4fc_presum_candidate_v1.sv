`timescale 1ns/1ps
`default_nettype none

// Precision-adaptive front end for one shared DSP accumulator. FC/INT5 can
// consume four binary-spike contributions per cycle; wider SRNN/SCNN modes
// retain the original two-term DSP48E1 pre-adder path.
module unified_dual_role_dsp_lane_p4fc_presum_candidate_v1 (
    input  wire logic                    i_clk,
    input  wire logic                    i_fire,
    input  wire logic                    i_first,
    input  wire logic                    i_p4_fc,
    input  wire logic                    i_enable0,
    input  wire logic                    i_enable1,
    input  wire logic                    i_enable2,
    input  wire logic                    i_enable3,
    input  wire logic signed [24:0]      i_term0,
    input  wire logic signed [24:0]      i_term1,
    input  wire logic signed [4:0]       i_term2,
    input  wire logic signed [4:0]       i_term3,
    input  wire logic signed [17:0]      i_factor_b,
    output logic signed [47:0]           o_result
);
    logic signed [5:0] fc_term0;
    logic signed [5:0] fc_term1;
    logic signed [5:0] fc_term2;
    logic signed [5:0] fc_term3;
    logic signed [5:0] fc_pair01;
    logic signed [5:0] fc_pair23;
    logic signed [6:0] fc_sum;
    logic signed [24:0] dsp_term_a;
    logic signed [24:0] dsp_term_d;
    logic dsp_enable_a;
    logic dsp_enable_d;

    always_comb begin
        fc_term0 = i_enable0 ? {i_term0[4], i_term0[4:0]} : 6'sd0;
        fc_term1 = i_enable1 ? {i_term1[4], i_term1[4:0]} : 6'sd0;
        fc_term2 = i_enable2 ? {i_term2[4], i_term2} : 6'sd0;
        fc_term3 = i_enable3 ? {i_term3[4], i_term3} : 6'sd0;
        fc_pair01 = fc_term0 + fc_term1;
        fc_pair23 = fc_term2 + fc_term3;
        fc_sum = {fc_pair01[5], fc_pair01} +
                 {fc_pair23[5], fc_pair23};

        if (i_p4_fc) begin
            dsp_term_a = {{18{fc_sum[6]}}, fc_sum};
            dsp_term_d = 25'sd0;
            dsp_enable_a = 1'b1;
            dsp_enable_d = 1'b0;
        end else begin
            dsp_term_a = i_term0;
            dsp_term_d = i_term1;
            dsp_enable_a = i_enable0;
            dsp_enable_d = i_enable1;
        end
    end

    unified_dual_role_dsp_lane_candidate_v3 u_accumulator (
        .i_clk(i_clk),
        .i_fire(i_fire),
        .i_first(i_first),
        .i_enable_a(dsp_enable_a),
        .i_enable_d(dsp_enable_d),
        .i_term_a(dsp_term_a),
        .i_term_d(dsp_term_d),
        .i_factor_b(i_factor_b),
        .o_result(o_result)
    );
endmodule

`default_nettype wire
