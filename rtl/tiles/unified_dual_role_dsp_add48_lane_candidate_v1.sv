`timescale 1ns/1ps
`default_nettype none

// One DSP48E1 performs a 48-bit pass-through or addition. The A:B concatenated
// input carries the primary accumulator and C carries the optional secondary.
module unified_dual_role_dsp_add48_lane_candidate_v1 (
    input  wire logic                    i_clk,
    input  wire logic                    i_fire,
    input  wire logic                    i_add_secondary,
    input  wire logic signed [47:0]      i_primary,
    input  wire logic signed [47:0]      i_secondary,
    output logic signed [47:0]           o_result
);
`ifdef SYNTHESIS
    logic [47:0] dsp_result;
    logic [6:0] opmode;

    // 0000011: A:B. 0110011: A:B + C.
    assign opmode = i_add_secondary ? 7'b0110011 : 7'b0000011;
    assign o_result = $signed(dsp_result);

    DSP48E1 #(
        .A_INPUT("DIRECT"),
        .B_INPUT("DIRECT"),
        .USE_DPORT("FALSE"),
        .USE_MULT("NONE"),
        .USE_SIMD("ONE48"),
        .AREG(0), .ACASCREG(0),
        .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0),
        .MREG(0), .PREG(1), .CREG(0),
        .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0)
    ) u_dsp48e1 (
        .CLK(i_clk),
        .A(i_primary[47:18]),
        .B(i_primary[17:0]),
        .C(i_secondary),
        .D(25'b0),
        .INMODE(5'b00000),
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
    logic signed [47:0] result_q;

    always_ff @(posedge i_clk) begin
        if (i_fire)
            result_q <= i_primary +
                (i_add_secondary ? i_secondary : 48'sd0);
    end

    assign o_result = result_q;
`endif
endmodule

`default_nettype wire
