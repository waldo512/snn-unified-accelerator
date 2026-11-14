`timescale 1ns/1ps
`default_nettype none

// Adapts the production FC fixed-point observations to the TP learning
// engine. Hidden observations are forked exactly once to the student-trace
// and surrogate streams. The membrane input is the post-soft-reset state;
// the spike bit restores v_pre before evaluating the ArcTan derivative LUT.
module unified_fc_tp_student_numeric_adapter #(
    parameter int SAMPLE_ID_WIDTH = 4,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int VMEM_WIDTH = 18,
    parameter int VMEM_FRAC_BITS = 10,
    parameter int LOGIT_WIDTH = 24,
    parameter int LOGIT_SCALE_WIDTH = 24,
    parameter int LOGIT_SCALE_FRAC_BITS = 16
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,

    input  logic                                  i_hidden_valid,
    output logic                                  o_hidden_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]            i_hidden_sample,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]         i_hidden_index,
    input  logic                                  i_hidden_spike,
    input  logic [15:0]                           i_hidden_trace_q016,
    input  logic signed [VMEM_WIDTH-1:0]          i_hidden_vmem_post,
    input  logic signed [VMEM_WIDTH-1:0]          i_vth,

    output logic                                  o_student_trace_valid,
    input  logic                                  i_student_trace_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]            o_student_trace_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]         o_student_trace_hidden,
    output logic [16:0]                           o_student_trace_q016,

    output logic                                  o_surrogate_valid,
    input  logic                                  i_surrogate_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]            o_surrogate_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]         o_surrogate_hidden,
    output logic [16:0]                           o_surrogate_q016,

    input  logic                                  i_logit_valid,
    output logic                                  o_logit_ready,
    input  logic signed [LOGIT_WIDTH-1:0]         i_logit_raw,
    input  logic [LOGIT_SCALE_WIDTH-1:0]          i_logit_to_q6_scale,
    output logic                                  o_logit_valid,
    input  logic                                  i_logit_ready,
    output logic signed [15:0]                    o_logit_q6
);
    localparam int SURROGATE_INDEX_SHIFT = VMEM_FRAC_BITS - 4;
    localparam int PRODUCT_WIDTH = LOGIT_WIDTH + LOGIT_SCALE_WIDTH + 1;

    logic hidden_pending_q;
    logic trace_pending_q;
    logic surrogate_pending_q;
    logic [SAMPLE_ID_WIDTH-1:0] hidden_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index_q;
    logic [16:0] hidden_trace_q;
    logic [16:0] hidden_surrogate_q;
    logic logit_valid_q;
    logic signed [15:0] logit_q;

    logic hidden_fire;
    logic trace_fire;
    logic surrogate_fire;
    logic logit_input_fire;
    logic logit_output_fire;
    logic signed [VMEM_WIDTH:0] membrane_pre_ext;
    logic signed [VMEM_WIDTH:0] threshold_ext;
    logic signed [VMEM_WIDTH:0] surrogate_delta;
    logic [VMEM_WIDTH:0] surrogate_magnitude;
    logic [VMEM_WIDTH:0] surrogate_index_rounded;
    logic [5:0] surrogate_index;

    function automatic logic [16:0] surrogate_lut_q016(
        input logic [5:0] index
    );
        begin
            case (index)
                6'd0:  surrogate_lut_q016 = 17'd65536;
                6'd1:  surrogate_lut_q016 = 17'd63103;
                6'd2:  surrogate_lut_q016 = 17'd56780;
                6'd3:  surrogate_lut_q016 = 17'd48654;
                6'd4:  surrogate_lut_q016 = 17'd40533;
                6'd5:  surrogate_lut_q016 = 17'd33372;
                6'd6:  surrogate_lut_q016 = 17'd27445;
                6'd7:  surrogate_lut_q016 = 17'd22684;
                6'd8:  surrogate_lut_q016 = 17'd18901;
                6'd9:  surrogate_lut_q016 = 17'd15896;
                6'd10: surrogate_lut_q016 = 17'd13498;
                6'd11: surrogate_lut_q016 = 17'd11569;
                6'd12: surrogate_lut_q016 = 17'd10003;
                6'd13: surrogate_lut_q016 = 17'd8720;
                6'd14: surrogate_lut_q016 = 17'd7659;
                6'd15: surrogate_lut_q016 = 17'd6774;
                6'd16: surrogate_lut_q016 = 17'd6029;
                6'd17: surrogate_lut_q016 = 17'd5398;
                6'd18: surrogate_lut_q016 = 17'd4858;
                6'd19: surrogate_lut_q016 = 17'd4393;
                6'd20: surrogate_lut_q016 = 17'd3991;
                6'd21: surrogate_lut_q016 = 17'd3640;
                6'd22: surrogate_lut_q016 = 17'd3334;
                6'd23: surrogate_lut_q016 = 17'd3063;
                6'd24: surrogate_lut_q016 = 17'd2824;
                6'd25: surrogate_lut_q016 = 17'd2611;
                6'd26: surrogate_lut_q016 = 17'd2422;
                6'd27: surrogate_lut_q016 = 17'd2252;
                6'd28: surrogate_lut_q016 = 17'd2099;
                6'd29: surrogate_lut_q016 = 17'd1961;
                6'd30: surrogate_lut_q016 = 17'd1836;
                6'd31: surrogate_lut_q016 = 17'd1722;
                6'd32: surrogate_lut_q016 = 17'd1619;
                6'd33: surrogate_lut_q016 = 17'd1525;
                6'd34: surrogate_lut_q016 = 17'd1438;
                6'd35: surrogate_lut_q016 = 17'd1359;
                6'd36: surrogate_lut_q016 = 17'd1286;
                6'd37: surrogate_lut_q016 = 17'd1219;
                6'd38: surrogate_lut_q016 = 17'd1156;
                6'd39: surrogate_lut_q016 = 17'd1099;
                6'd40: surrogate_lut_q016 = 17'd1045;
                6'd41: surrogate_lut_q016 = 17'd996;
                6'd42: surrogate_lut_q016 = 17'd950;
                6'd43: surrogate_lut_q016 = 17'd907;
                6'd44: surrogate_lut_q016 = 17'd866;
                6'd45: surrogate_lut_q016 = 17'd829;
                6'd46: surrogate_lut_q016 = 17'd794;
                6'd47: surrogate_lut_q016 = 17'd761;
                6'd48: surrogate_lut_q016 = 17'd730;
                6'd49: surrogate_lut_q016 = 17'd700;
                6'd50: surrogate_lut_q016 = 17'd673;
                6'd51: surrogate_lut_q016 = 17'd647;
                6'd52: surrogate_lut_q016 = 17'd623;
                6'd53: surrogate_lut_q016 = 17'd600;
                6'd54: surrogate_lut_q016 = 17'd578;
                6'd55: surrogate_lut_q016 = 17'd557;
                6'd56: surrogate_lut_q016 = 17'd538;
                6'd57: surrogate_lut_q016 = 17'd519;
                6'd58: surrogate_lut_q016 = 17'd501;
                6'd59: surrogate_lut_q016 = 17'd485;
                6'd60: surrogate_lut_q016 = 17'd469;
                6'd61: surrogate_lut_q016 = 17'd454;
                6'd62: surrogate_lut_q016 = 17'd439;
                default: surrogate_lut_q016 = 17'd426;
            endcase
        end
    endfunction

    function automatic logic signed [15:0] scale_logit_q6(
        input logic signed [LOGIT_WIDTH-1:0] raw_value,
        input logic [LOGIT_SCALE_WIDTH-1:0] scale_value
    );
        logic signed [PRODUCT_WIDTH-1:0] product;
        logic [PRODUCT_WIDTH-1:0] magnitude;
        logic [PRODUCT_WIDTH-1:0] integer_part;
        logic [LOGIT_SCALE_FRAC_BITS-1:0] remainder;
        logic round_increment;
        logic [PRODUCT_WIDTH:0] rounded_magnitude;
        logic signed [PRODUCT_WIDTH+1:0] rounded_signed;
        begin
            product = raw_value * $signed({1'b0, scale_value});
            magnitude = product[PRODUCT_WIDTH-1] ?
                $unsigned(-product) : $unsigned(product);
            integer_part = magnitude >> LOGIT_SCALE_FRAC_BITS;
            remainder = magnitude[LOGIT_SCALE_FRAC_BITS-1:0];
            round_increment =
                (remainder > (LOGIT_SCALE_FRAC_BITS)'(
                    1 << (LOGIT_SCALE_FRAC_BITS-1))) ||
                ((remainder == (LOGIT_SCALE_FRAC_BITS)'(
                    1 << (LOGIT_SCALE_FRAC_BITS-1))) && integer_part[0]);
            rounded_magnitude = {1'b0, integer_part} + round_increment;
            rounded_signed = product[PRODUCT_WIDTH-1] ?
                -$signed({1'b0, rounded_magnitude}) :
                $signed({1'b0, rounded_magnitude});
            if (rounded_signed > 32767)
                scale_logit_q6 = 16'sh7fff;
            else if (rounded_signed < -32768)
                scale_logit_q6 = -16'sd32768;
            else
                scale_logit_q6 = rounded_signed[15:0];
        end
    endfunction

    always_comb begin
        o_hidden_ready = !hidden_pending_q;
        hidden_fire = i_hidden_valid && o_hidden_ready;

        o_student_trace_valid = hidden_pending_q && trace_pending_q;
        o_student_trace_sample = hidden_sample_q;
        o_student_trace_hidden = hidden_index_q;
        o_student_trace_q016 = hidden_trace_q;
        trace_fire = o_student_trace_valid && i_student_trace_ready;

        o_surrogate_valid = hidden_pending_q && surrogate_pending_q;
        o_surrogate_sample = hidden_sample_q;
        o_surrogate_hidden = hidden_index_q;
        o_surrogate_q016 = hidden_surrogate_q;
        surrogate_fire = o_surrogate_valid && i_surrogate_ready;

        threshold_ext = $signed({i_vth[VMEM_WIDTH-1], i_vth});
        membrane_pre_ext = $signed({i_hidden_vmem_post[VMEM_WIDTH-1],
                                    i_hidden_vmem_post});
        if (i_hidden_spike)
            membrane_pre_ext = membrane_pre_ext + threshold_ext;
        surrogate_delta = membrane_pre_ext - threshold_ext;
        surrogate_magnitude = surrogate_delta[VMEM_WIDTH] ?
            $unsigned(-surrogate_delta) : $unsigned(surrogate_delta);
        surrogate_index_rounded =
            (surrogate_magnitude + (1 << (SURROGATE_INDEX_SHIFT-1))) >>
            SURROGATE_INDEX_SHIFT;
        surrogate_index = (surrogate_index_rounded > 63) ?
            6'd63 : surrogate_index_rounded[5:0];

        o_logit_ready = !logit_valid_q;
        logit_input_fire = i_logit_valid && o_logit_ready;
        o_logit_valid = logit_valid_q;
        o_logit_q6 = logit_q;
        logit_output_fire = o_logit_valid && i_logit_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            hidden_pending_q <= 1'b0;
            trace_pending_q <= 1'b0;
            surrogate_pending_q <= 1'b0;
            hidden_sample_q <= '0;
            hidden_index_q <= '0;
            hidden_trace_q <= '0;
            hidden_surrogate_q <= '0;
            logit_valid_q <= 1'b0;
            logit_q <= '0;
        end else begin
            if (hidden_fire) begin
                hidden_pending_q <= 1'b1;
                trace_pending_q <= 1'b1;
                surrogate_pending_q <= 1'b1;
                hidden_sample_q <= i_hidden_sample;
                hidden_index_q <= i_hidden_index;
                hidden_trace_q <= {1'b0, i_hidden_trace_q016};
                hidden_surrogate_q <= surrogate_lut_q016(surrogate_index);
            end else if (hidden_pending_q) begin
                if (trace_fire)
                    trace_pending_q <= 1'b0;
                if (surrogate_fire)
                    surrogate_pending_q <= 1'b0;
                if ((!trace_pending_q || trace_fire) &&
                    (!surrogate_pending_q || surrogate_fire))
                    hidden_pending_q <= 1'b0;
            end

            if (logit_input_fire) begin
                logit_valid_q <= 1'b1;
                logit_q <= scale_logit_q6(i_logit_raw,
                                          i_logit_to_q6_scale);
            end else if (logit_output_fire) begin
                logit_valid_q <= 1'b0;
            end
        end
    end

    initial begin
        if (VMEM_FRAC_BITS <= 4 || LOGIT_SCALE_FRAC_BITS < 1)
            $error("FC TP numeric adapter fractional widths are invalid");
    end
endmodule

`default_nettype wire
