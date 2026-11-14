`default_nettype none

// One physical signed 8x8 multiplier serves either an INT8 contribution or
// one of the four exact Q8.8 decomposition phases.
module unified_shared_k8_numeric_v1_lane_soft_abort_v1 (
    input  logic               i_clk,
    input  logic               i_rst_n,
    input  logic               i_abort,
    input  logic               i_valid,
    output logic               o_ready,
    input  logic               i_mode_q8_8,
    input  logic               i_lane_enable,
    input  logic signed [15:0] i_q8_8_lhs,
    input  logic signed [15:0] i_q8_8_rhs,
    input  logic signed [7:0]  i_int8_lhs,
    input  logic signed [7:0]  i_int8_rhs,
    output logic               o_valid,
    input  logic               i_ready,
    output logic signed [31:0] o_q8_8_product,
    output logic signed [31:0] o_int8_contribution,
    output logic               o_busy
);
    logic busy_q;
    logic result_valid_q;
    logic mode_q8_8_q;
    logic lane_enable_q;
    logic [1:0] phase_q;
    logic signed [7:0] lhs_hi_q;
    logic signed [7:0] rhs_hi_q;
    logic signed [7:0] lhs_lo7_q;
    logic signed [7:0] rhs_lo7_q;
    logic signed [7:0] int8_lhs_q;
    logic signed [7:0] int8_rhs_q;
    logic signed [31:0] correction_q;
    logic signed [31:0] accumulator_q;
    logic signed [31:0] q8_8_product_q;
    logic signed [31:0] int8_contribution_q;

    logic signed [7:0] mul_lhs;
    logic signed [7:0] mul_rhs;
    logic signed [15:0] mul_product;
    logic signed [31:0] mul_extended;
    logic signed [31:0] shifted_term;
    logic signed [31:0] correction_in;
    logic signed [31:0] lhs_hi_extended;
    logic signed [31:0] rhs_hi_extended;
    logic signed [31:0] lhs_lo7_extended;
    logic signed [31:0] rhs_lo7_extended;

    assign o_ready = !busy_q && (!result_valid_q || i_ready);
    assign o_valid = result_valid_q;
    assign o_q8_8_product = q8_8_product_q;
    assign o_int8_contribution = int8_contribution_q;
    assign o_busy = busy_q || result_valid_q;

    always_comb begin
        lhs_hi_extended = {{24{i_q8_8_lhs[15]}}, i_q8_8_lhs[15:8]};
        rhs_hi_extended = {{24{i_q8_8_rhs[15]}}, i_q8_8_rhs[15:8]};
        lhs_lo7_extended = {25'd0, i_q8_8_lhs[6:0]};
        rhs_lo7_extended = {25'd0, i_q8_8_rhs[6:0]};
        correction_in = 32'sd0;
        if (i_lane_enable && i_q8_8_rhs[7]) begin
            correction_in = correction_in +
                (lhs_hi_extended <<< 15) + (lhs_lo7_extended <<< 7);
        end
        if (i_lane_enable && i_q8_8_lhs[7]) begin
            correction_in = correction_in +
                (rhs_hi_extended <<< 15) + (rhs_lo7_extended <<< 7);
        end
        if (i_lane_enable && i_q8_8_lhs[7] && i_q8_8_rhs[7])
            correction_in = correction_in + 32'sd16384;
    end

    always_comb begin
        mul_lhs = int8_lhs_q;
        mul_rhs = int8_rhs_q;
        if (mode_q8_8_q) begin
            unique case (phase_q)
                2'd0: begin
                    mul_lhs = lhs_hi_q;
                    mul_rhs = rhs_hi_q;
                end
                2'd1: begin
                    mul_lhs = lhs_hi_q;
                    mul_rhs = rhs_lo7_q;
                end
                2'd2: begin
                    mul_lhs = rhs_hi_q;
                    mul_rhs = lhs_lo7_q;
                end
                default: begin
                    mul_lhs = lhs_lo7_q;
                    mul_rhs = rhs_lo7_q;
                end
            endcase
        end
        mul_product = $signed(mul_lhs) * $signed(mul_rhs);
        mul_extended = {{16{mul_product[15]}}, mul_product};
        shifted_term = mul_extended;
        if (mode_q8_8_q) begin
            unique case (phase_q)
                2'd0: shifted_term = mul_extended <<< 16;
                2'd1, 2'd2: shifted_term = mul_extended <<< 8;
                default: shifted_term = mul_extended;
            endcase
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            mode_q8_8_q <= 1'b0;
            lane_enable_q <= 1'b0;
            phase_q <= 2'd0;
            lhs_hi_q <= 8'sd0;
            rhs_hi_q <= 8'sd0;
            lhs_lo7_q <= 8'sd0;
            rhs_lo7_q <= 8'sd0;
            int8_lhs_q <= 8'sd0;
            int8_rhs_q <= 8'sd0;
            correction_q <= 32'sd0;
            accumulator_q <= 32'sd0;
            q8_8_product_q <= 32'sd0;
            int8_contribution_q <= 32'sd0;
        end else if (i_abort) begin
            busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            mode_q8_8_q <= 1'b0;
            lane_enable_q <= 1'b0;
            phase_q <= 2'd0;
            lhs_hi_q <= 8'sd0;
            rhs_hi_q <= 8'sd0;
            lhs_lo7_q <= 8'sd0;
            rhs_lo7_q <= 8'sd0;
            int8_lhs_q <= 8'sd0;
            int8_rhs_q <= 8'sd0;
            correction_q <= 32'sd0;
            accumulator_q <= 32'sd0;
            q8_8_product_q <= 32'sd0;
            int8_contribution_q <= 32'sd0;
        end else begin
            if (result_valid_q && i_ready)
                result_valid_q <= 1'b0;

            if (i_valid && o_ready) begin
                mode_q8_8_q <= i_mode_q8_8;
                lane_enable_q <= i_lane_enable;
                lhs_hi_q <= i_lane_enable ? i_q8_8_lhs[15:8] : 8'sd0;
                rhs_hi_q <= i_lane_enable ? i_q8_8_rhs[15:8] : 8'sd0;
                lhs_lo7_q <= i_lane_enable ?
                    {1'b0, i_q8_8_lhs[6:0]} : 8'sd0;
                rhs_lo7_q <= i_lane_enable ?
                    {1'b0, i_q8_8_rhs[6:0]} : 8'sd0;
                int8_lhs_q <= i_lane_enable ? i_int8_lhs : 8'sd0;
                int8_rhs_q <= i_lane_enable ? i_int8_rhs : 8'sd0;
                correction_q <= correction_in;
                accumulator_q <= 32'sd0;
                phase_q <= 2'd0;
                busy_q <= 1'b1;
            end else if (busy_q) begin
                if (!mode_q8_8_q) begin
                    int8_contribution_q <= lane_enable_q ?
                        {{16{mul_product[15]}}, mul_product} : 32'sd0;
                    q8_8_product_q <= 32'sd0;
                    result_valid_q <= 1'b1;
                    busy_q <= 1'b0;
                end else if (phase_q == 2'd3) begin
                    q8_8_product_q <= lane_enable_q ?
                        accumulator_q + shifted_term + correction_q : 32'sd0;
                    int8_contribution_q <= 32'sd0;
                    result_valid_q <= 1'b1;
                    busy_q <= 1'b0;
                    phase_q <= 2'd0;
                end else begin
                    accumulator_q <= accumulator_q + shifted_term;
                    phase_q <= phase_q + 2'd1;
                end
            end
        end
    end
endmodule

`default_nettype wire
