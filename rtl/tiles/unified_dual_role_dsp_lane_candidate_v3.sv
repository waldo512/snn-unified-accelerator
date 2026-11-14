`timescale 1ns/1ps
`default_nettype none

// DSP48E1 lane with independent A/D contribution enables implemented by
// INMODE. This avoids an operand-width LUT mux for each binary spike bit.
module unified_dual_role_dsp_lane_candidate_v3 (
    input  wire logic                    i_clk,
    input  wire logic                    i_fire,
    input  wire logic                    i_first,
    input  wire logic                    i_enable_a,
    input  wire logic                    i_enable_d,
    input  wire logic signed [24:0]      i_term_a,
    input  wire logic signed [24:0]      i_term_d,
    input  wire logic signed [17:0]      i_factor_b,
    output logic signed [47:0]           o_result
);

`ifdef SYNTHESIS
    logic [47:0] dsp_result;
    logic [4:0] inmode;
    logic [6:0] opmode;

    // IS_INMODE_INVERTED[1] converts i_enable_a to the active-high
    // "force A to zero" control inside the primitive. IS_OPMODE_INVERTED[5]
    // similarly maps i_first onto multiply-only versus P-feedback accumulate.
    assign inmode = {2'b00, i_enable_d, i_enable_a, 1'b0};
    assign opmode = {1'b0, i_first, 5'b00101};
    assign o_result = $signed(dsp_result);

    DSP48E1 #(
        .A_INPUT("DIRECT"),
        .B_INPUT("DIRECT"),
        .USE_DPORT("TRUE"),
        .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48"),
        .AREG(0), .ACASCREG(0),
        .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0),
        .MREG(0), .PREG(1), .CREG(0),
        .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0),
        .IS_INMODE_INVERTED(5'b00010),
        .IS_OPMODE_INVERTED(7'b0100000)
    ) u_dsp48e1 (
        .CLK(i_clk),
        .A({{5{i_term_a[24]}}, i_term_a}),
        .B(i_factor_b),
        .C(48'b0),
        .D(i_term_d),
        .INMODE(inmode),
        .OPMODE(opmode),
        .ALUMODE(4'b0000),
        .CARRYINSEL(3'b000),
        .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0),
        .CEB1(1'b0), .CEB2(1'b0),
        .CEC(1'b0), .CED(1'b0), .CEAD(1'b0), .CEM(1'b0),
        .CEINMODE(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0),
        .CECARRYIN(1'b0), .CEP(i_fire),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0),
        .RSTM(1'b0), .RSTP(1'b0), .RSTINMODE(1'b0),
        .RSTCTRL(1'b0), .RSTALUMODE(1'b0), .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(dsp_result),
        .ACOUT(), .BCOUT(), .PCOUT(), .CARRYOUT(),
        .CARRYCASCOUT(), .MULTSIGNOUT(),
        .OVERFLOW(), .UNDERFLOW(),
        .PATTERNDETECT(), .PATTERNBDETECT()
    );
`else
    logic signed [47:0] accumulator_q;
    logic signed [24:0] selected_a;
    logic signed [24:0] selected_d;
    logic signed [24:0] preadd_value;
    logic signed [42:0] product_value;
    logic signed [47:0] product_extended;

    always_comb begin
        selected_a = i_enable_a ? i_term_a : 25'sd0;
        selected_d = i_enable_d ? i_term_d : 25'sd0;
        preadd_value = selected_a + selected_d;
        product_value = preadd_value * i_factor_b;
        product_extended = {{5{product_value[42]}}, product_value};
    end

    always_ff @(posedge i_clk) begin
        if (i_fire)
            accumulator_q <= i_first ? product_extended :
                accumulator_q + product_extended;
    end

    assign o_result = accumulator_q;
`endif

endmodule

`default_nettype wire
