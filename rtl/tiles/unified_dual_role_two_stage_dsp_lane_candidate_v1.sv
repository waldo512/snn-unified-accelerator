`timescale 1ns/1ps
`default_nettype none

// A contribution accumulator cascaded into a LIF DSP. A/B/C on the LIF DSP
// remain state/leak/threshold; only its narrow operation controls change.
module unified_dual_role_two_stage_dsp_lane_candidate_v1 (
    input  wire logic                    i_clk,

    input  wire logic                    i_acc_fire,
    input  wire logic                    i_acc_clear,
    input  wire logic                    i_acc_enable_a,
    input  wire logic                    i_acc_enable_d,
    input  wire logic signed [24:0]      i_acc_term_a,
    input  wire logic signed [24:0]      i_acc_term_d,
    input  wire logic signed [17:0]      i_acc_factor_b,

    input  wire logic                    i_lif_fire,
    input  wire logic [1:0]              i_lif_op,
    input  wire logic signed [24:0]      i_state,
    input  wire logic signed [17:0]      i_leak,
    input  wire logic signed [24:0]      i_threshold,
    output logic                         o_spike,
    output logic signed [47:0]           o_state_result
);
    localparam logic [1:0] LIF_INTEGRATE = 2'd0;
    localparam logic [1:0] LIF_SUBTRACT = 2'd1;
    localparam logic [1:0] LIF_FINALIZE = 2'd2;

`ifdef SYNTHESIS
    logic [47:0] acc_result;
    logic [47:0] acc_pcout;
    logic [47:0] lif_result;
    logic [4:0] acc_inmode;
    logic [6:0] acc_opmode;
    logic [6:0] lif_opmode;
    logic [3:0] lif_alumode;

    assign acc_inmode = {2'b00, i_acc_enable_d, i_acc_enable_a, 1'b0};
    assign acc_opmode = {1'b0, i_acc_clear, 5'b00101};
    assign o_state_result = $signed(lif_result);

    always_comb begin
        lif_opmode = 7'b0000000;
        lif_alumode = 4'b0000;
        case (i_lif_op)
            LIF_INTEGRATE: begin
                lif_opmode = 7'b0010101; // PCIN + A*B
                lif_alumode = 4'b0000;
            end
            LIF_SUBTRACT: begin
                lif_opmode = 7'b0101100; // P - C
                lif_alumode = 4'b0011;
            end
            LIF_FINALIZE: begin
                lif_opmode = lif_result[47] ? 7'b0101100 : 7'b0000000;
                lif_alumode = 4'b0000; // negative: P+C; nonnegative: zero
            end
            default: begin
                lif_opmode = 7'b0000000;
                lif_alumode = 4'b0000;
            end
        endcase
    end

    DSP48E1 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48"),
        .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0), .MREG(0), .PREG(1), .CREG(0),
        .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0),
        .IS_INMODE_INVERTED(5'b00010),
        .IS_OPMODE_INVERTED(7'b0100000)
    ) u_acc_dsp48e1 (
        .CLK(i_clk),
        .A({{5{i_acc_term_a[24]}}, i_acc_term_a}),
        .B(i_acc_factor_b), .C(48'b0), .D(i_acc_term_d),
        .INMODE(acc_inmode), .OPMODE(acc_opmode), .ALUMODE(4'b0000),
        .CARRYINSEL(3'b000), .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0),
        .CEC(1'b0), .CED(1'b0), .CEAD(1'b0), .CEM(1'b0),
        .CEINMODE(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0),
        .CECARRYIN(1'b0), .CEP(i_acc_fire),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0),
        .RSTM(1'b0), .RSTP(1'b0), .RSTINMODE(1'b0),
        .RSTCTRL(1'b0), .RSTALUMODE(1'b0), .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(48'b0),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(acc_result), .PCOUT(acc_pcout),
        .ACOUT(), .BCOUT(), .CARRYOUT(), .CARRYCASCOUT(),
        .MULTSIGNOUT(), .OVERFLOW(), .UNDERFLOW(),
        .PATTERNDETECT(), .PATTERNBDETECT()
    );

    DSP48E1 #(
        .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
        .USE_DPORT("FALSE"), .USE_MULT("DYNAMIC"),
        .USE_SIMD("ONE48"),
        .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0),
        .DREG(0), .ADREG(0), .MREG(0), .PREG(1), .CREG(0),
        .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
        .CARRYINREG(0), .CARRYINSELREG(0)
    ) u_lif_dsp48e1 (
        .CLK(i_clk),
        .A({{5{i_state[24]}}, i_state}),
        .B(i_leak),
        .C({{23{i_threshold[24]}}, i_threshold}),
        .D(25'b0), .INMODE(5'b00000),
        .OPMODE(lif_opmode), .ALUMODE(lif_alumode),
        .CARRYINSEL(3'b000), .CARRYIN(1'b0),
        .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0),
        .CEC(1'b0), .CED(1'b0), .CEAD(1'b0), .CEM(1'b0),
        .CEINMODE(1'b0), .CECTRL(1'b0), .CEALUMODE(1'b0),
        .CECARRYIN(1'b0), .CEP(i_lif_fire),
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0),
        .RSTM(1'b0), .RSTP(1'b0), .RSTINMODE(1'b0),
        .RSTCTRL(1'b0), .RSTALUMODE(1'b0), .RSTALLCARRYIN(1'b0),
        .ACIN(30'b0), .BCIN(18'b0), .PCIN(acc_pcout),
        .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
        .P(lif_result),
        .ACOUT(), .BCOUT(), .PCOUT(), .CARRYOUT(), .CARRYCASCOUT(),
        .MULTSIGNOUT(), .OVERFLOW(), .UNDERFLOW(),
        .PATTERNDETECT(), .PATTERNBDETECT()
    );
`else
    logic signed [47:0] accumulator_q;
    logic signed [47:0] lif_q;
    logic signed [24:0] selected_a;
    logic signed [24:0] selected_d;
    logic signed [24:0] preadd_value;
    logic signed [42:0] product_value;
    logic signed [47:0] product_extended;
    logic spike_q;

    always_comb begin
        selected_a = i_acc_enable_a ? i_acc_term_a : 25'sd0;
        selected_d = i_acc_enable_d ? i_acc_term_d : 25'sd0;
        preadd_value = selected_a + selected_d;
        product_value = preadd_value * i_acc_factor_b;
        product_extended = {{5{product_value[42]}}, product_value};
    end

    always_ff @(posedge i_clk) begin
        if (i_acc_fire)
            accumulator_q <= i_acc_clear ? product_extended :
                accumulator_q + product_extended;
        if (i_lif_fire) begin
            case (i_lif_op)
                LIF_INTEGRATE:
                    lif_q <= accumulator_q + i_state * i_leak;
                LIF_SUBTRACT:
                    lif_q <= lif_q - {{23{i_threshold[24]}}, i_threshold};
                LIF_FINALIZE: begin
                    spike_q <= !lif_q[47];
                    lif_q <= lif_q[47] ?
                        lif_q + {{23{i_threshold[24]}}, i_threshold} :
                        48'sd0;
                end
                default: lif_q <= lif_q;
            endcase
        end
    end

    assign o_state_result = lif_q;
    assign o_spike = spike_q;
`endif

`ifdef SYNTHESIS
    always_ff @(posedge i_clk) begin
        if (i_lif_fire && i_lif_op == LIF_FINALIZE)
            o_spike <= !lif_result[47];
    end
`endif

endmodule

`default_nettype wire
