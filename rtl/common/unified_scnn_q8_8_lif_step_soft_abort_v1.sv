`default_nettype none

module unified_scnn_q8_8_lif_step_soft_abort_v1 (
    input  logic               i_clk,
    input  logic               i_rst_n,
    input  logic               i_abort,
    input  logic               i_valid,
    output logic               o_ready,
    input  logic signed [15:0] i_current,
    input  logic signed [15:0] i_membrane_previous,
    input  logic signed [15:0] i_leak_q8_8,
    input  logic signed [15:0] i_threshold_q8_8,
    input  logic               i_reset_to_zero,
    output logic               o_valid,
    input  logic               i_ready,
    output logic        [15:0] o_spike_q8_8,
    output logic signed [15:0] o_membrane_pre,
    output logic signed [15:0] o_membrane_next
);

    logic multiplier_ready;
    logic multiplier_valid;
    logic multiplier_consume;
    logic signed [31:0] leak_product;
    logic signed [15:0] current_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic transaction_busy_q;
    logic result_valid_q;
    logic [15:0] spike_q;
    logic signed [15:0] membrane_pre_q;
    logic signed [15:0] membrane_next_q;

    logic signed [31:0] leaked_wide;
    logic signed [31:0] membrane_pre_wide;
    logic signed [16:0] membrane_pre_17;
    logic signed [16:0] threshold_17;
    logic fires;
    logic signed [16:0] membrane_after_reset;
    logic signed [15:0] membrane_next_comb;
    logic unused_upper;

    function automatic logic signed [15:0] saturate_membrane(
        input logic signed [16:0] value
    );
        begin
            if (value > 17'sd32767)
                saturate_membrane = 16'sh7fff;
            else if (value < -17'sd32768)
                saturate_membrane = 16'sh8000;
            else
                saturate_membrane = value[15:0];
        end
    endfunction

    // State is external and the interface has no address tag. Keep exactly one
    // transaction outstanding so the caller can write the returned state back
    // before another token reads that neuron.
    assign o_ready = multiplier_ready && !transaction_busy_q;
    assign multiplier_consume = !result_valid_q || i_ready;
    assign o_valid = result_valid_q;
    assign o_spike_q8_8 = spike_q;
    assign o_membrane_pre = membrane_pre_q;
    assign o_membrane_next = membrane_next_q;

    always_comb begin
        leaked_wide = leak_product >>> 8;
        membrane_pre_wide = leaked_wide +
                            {{16{current_q[15]}}, current_q};
        membrane_pre_17 = membrane_pre_wide[16:0];
        threshold_17 = {threshold_q[15], threshold_q};
        fires = membrane_pre_17 >= threshold_17;
        membrane_after_reset = membrane_pre_17 -
            (fires ? threshold_17 : 17'sd0);
        if (reset_to_zero_q && fires)
            membrane_next_comb = 16'sd0;
        else
            membrane_next_comb = saturate_membrane(membrane_after_reset);
    end

    assign unused_upper = &{1'b0, membrane_pre_wide[31:17]};

    unified_q8_8_mul_multipass_soft_abort_v1 u_leak_multiply (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_valid(i_valid && o_ready),
        .o_ready(multiplier_ready),
        .i_a(i_membrane_previous),
        .i_b(i_leak_q8_8),
        .o_valid(multiplier_valid),
        .i_ready(multiplier_consume),
        .o_product(leak_product)
    );

    // This result-valid state is part of the write-enable cone for the
    // versioned membrane BRAM. Reset it synchronously so assertion is timed
    // like the RAM address and enable signals.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            current_q <= 16'sd0;
            threshold_q <= 16'sd0;
            reset_to_zero_q <= 1'b0;
            transaction_busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            spike_q <= 16'd0;
            membrane_pre_q <= 16'sd0;
            membrane_next_q <= 16'sd0;
        end else if (i_abort) begin
            current_q <= 16'sd0;
            threshold_q <= 16'sd0;
            reset_to_zero_q <= 1'b0;
            transaction_busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            spike_q <= 16'd0;
            membrane_pre_q <= 16'sd0;
            membrane_next_q <= 16'sd0;
        end else begin
            if (result_valid_q && i_ready) begin
                result_valid_q <= 1'b0;
                transaction_busy_q <= 1'b0;
            end

            if (i_valid && o_ready) begin
                current_q <= i_current;
                threshold_q <= i_threshold_q8_8;
                reset_to_zero_q <= i_reset_to_zero;
                transaction_busy_q <= 1'b1;
            end

            if (multiplier_valid && multiplier_consume) begin
                spike_q <= fires ? 16'h0100 : 16'h0000;
                membrane_pre_q <= membrane_pre_17[15:0];
                membrane_next_q <= membrane_next_comb;
                result_valid_q <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
