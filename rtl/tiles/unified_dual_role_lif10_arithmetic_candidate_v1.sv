`timescale 1ns/1ps
`default_nettype none

// Ten physical neuron-update lanes serve all topology modes. The state/leak
// multiply is pipelined separately from mode-specific rounding, saturation,
// threshold and reset. Trace updates remain in the observation/learning path.
module unified_dual_role_lif10_arithmetic_candidate_v1 #(
    parameter int LANES = 10
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic [LANES*32-1:0]     i_current_flat,
    input  wire logic [LANES*32-1:0]     i_state_flat,
    input  wire logic signed [15:0]      i_leak,
    input  wire logic signed [31:0]      i_threshold,
    input  wire logic [4:0]              i_current_left_shift,
    input  wire logic                    i_reset_to_zero,
    output logic                         o_valid,
    input  wire logic                    i_ready,
    output logic [LANES-1:0]             o_spike,
    output logic [LANES*32-1:0]          o_pre_state_flat,
    output logic [LANES*32-1:0]          o_next_state_flat
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    logic product_valid_q;
    logic result_valid_q;
    logic [1:0] mode_q;
    logic [LANES*32-1:0] current_q;
    logic signed [31:0] threshold_q;
    logic [4:0] current_left_shift_q;
    logic reset_to_zero_q;
    logic signed [31:0] state_operand [0:LANES-1];
    (* use_dsp = "yes" *) logic signed [47:0] leak_product_q [0:LANES-1];
    logic [LANES-1:0] spike_next;
    logic [LANES*32-1:0] pre_state_next_flat;
    logic [LANES*32-1:0] next_state_next_flat;

    function automatic logic signed [31:0] saturate64_to32(
        input logic signed [63:0] value
    );
        if (value > 64'sh000000007fffffff)
            return 32'sh7fffffff;
        if (value < -64'sh0000000080000000)
            return 32'sh80000000;
        return value[31:0];
    endfunction

    function automatic logic signed [15:0] saturate17_to16(
        input logic signed [16:0] value
    );
        if (value > 17'sd32767)
            return 16'sh7fff;
        if (value < -17'sd32768)
            return 16'sh8000;
        return value[15:0];
    endfunction

    assign o_ready = !i_abort && !product_valid_q && !result_valid_q;
    assign o_valid = !i_abort && result_valid_q;

    always_comb begin
        spike_next = '0;
        pre_state_next_flat = '0;
        next_state_next_flat = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [31:0] current32;
            logic signed [17:0] fc_leaked;
            logic signed [17:0] fc_current;
            logic signed [17:0] fc_pre;
            logic signed [17:0] fc_threshold;
            logic signed [17:0] fc_next;
            logic signed [63:0] srnn_leaked;
            logic signed [63:0] srnn_current;
            logic signed [63:0] srnn_pre_wide;
            logic signed [31:0] srnn_pre;
            logic signed [16:0] scnn_pre17;
            logic signed [16:0] scnn_threshold17;
            logic signed [16:0] scnn_after_reset;
            logic signed [15:0] scnn_next;

            current32 = $signed(current_q[lane*32 +: 32]);
            fc_leaked = '0;
            fc_current = '0;
            fc_pre = '0;
            fc_threshold = '0;
            fc_next = '0;
            srnn_leaked = '0;
            srnn_current = '0;
            srnn_pre_wide = '0;
            srnn_pre = '0;
            scnn_pre17 = '0;
            scnn_threshold17 = '0;
            scnn_after_reset = '0;
            scnn_next = '0;
            if (mode_q == MODE_FC) begin
                fc_leaked = 18'(
                    (leak_product_q[lane] + 48'sd16384) >>> 15);
                fc_current = current32[17:0];
                fc_pre = fc_leaked + fc_current;
                fc_threshold = threshold_q[17:0];
                spike_next[lane] = fc_pre >= fc_threshold;
                fc_next = spike_next[lane] ?
                    (fc_pre - fc_threshold) : fc_pre;
                pre_state_next_flat[lane*32 +: 32] =
                    {{14{fc_pre[17]}}, fc_pre};
                next_state_next_flat[lane*32 +: 32] =
                    {{14{fc_next[17]}}, fc_next};
            end else if (mode_q == MODE_SRNN) begin
                srnn_leaked = {{16{leak_product_q[lane][47]}},
                                leak_product_q[lane]};
                srnn_leaked = srnn_leaked >>> 15;
                srnn_current = {{32{current32[31]}}, current32};
                srnn_current = srnn_current <<< current_left_shift_q;
                srnn_pre_wide = srnn_leaked + srnn_current;
                srnn_pre = saturate64_to32(srnn_pre_wide);
                spike_next[lane] = srnn_pre >= threshold_q;
                pre_state_next_flat[lane*32 +: 32] = srnn_pre;
                if (spike_next[lane]) begin
                    next_state_next_flat[lane*32 +: 32] =
                        reset_to_zero_q ? 32'sd0 : srnn_pre - threshold_q;
                end else begin
                    next_state_next_flat[lane*32 +: 32] = srnn_pre;
                end
            end else if (mode_q == MODE_SCNN) begin
                scnn_pre17 = 17'((leak_product_q[lane] >>> 8) +
                    {{32{current32[15]}}, current32[15:0]});
                scnn_threshold17 = {threshold_q[15], threshold_q[15:0]};
                spike_next[lane] = scnn_pre17 >= scnn_threshold17;
                scnn_after_reset = scnn_pre17 -
                    (spike_next[lane] ? scnn_threshold17 : 17'sd0);
                scnn_next = reset_to_zero_q && spike_next[lane] ?
                    16'sd0 : saturate17_to16(scnn_after_reset);
                pre_state_next_flat[lane*32 +: 32] =
                    {{16{scnn_pre17[15]}}, scnn_pre17[15:0]};
                next_state_next_flat[lane*32 +: 32] =
                    {{16{scnn_next[15]}}, scnn_next};
            end
        end
    end

    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            if (i_mode == MODE_FC) begin
                state_operand[lane] =
                    {{14{i_state_flat[lane*32 + 17]}},
                     i_state_flat[lane*32 +: 18]};
            end else if (i_mode == MODE_SCNN) begin
                state_operand[lane] =
                    {{16{i_state_flat[lane*32 + 15]}},
                     i_state_flat[lane*32 +: 16]};
            end else begin
                state_operand[lane] =
                    $signed(i_state_flat[lane*32 +: 32]);
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            product_valid_q <= 1'b0;
            result_valid_q <= 1'b0;
            mode_q <= MODE_FC;
            current_q <= '0;
            threshold_q <= '0;
            current_left_shift_q <= '0;
            reset_to_zero_q <= 1'b0;
            o_spike <= '0;
            o_pre_state_flat <= '0;
            o_next_state_flat <= '0;
            for (int lane = 0; lane < LANES; lane++)
                leak_product_q[lane] <= '0;
        end else begin
            if (result_valid_q && i_ready)
                result_valid_q <= 1'b0;

            if (i_valid && o_ready) begin
                product_valid_q <= 1'b1;
                mode_q <= i_mode;
                current_q <= i_current_flat;
                threshold_q <= i_threshold;
                current_left_shift_q <= i_current_left_shift;
                reset_to_zero_q <= i_reset_to_zero;
                for (int lane = 0; lane < LANES; lane++) begin
                    leak_product_q[lane] <=
                        state_operand[lane] * $signed(i_leak);
                end
            end else if (product_valid_q) begin
                product_valid_q <= 1'b0;
                result_valid_q <= 1'b1;
                o_spike <= spike_next;
                o_pre_state_flat <= pre_state_next_flat;
                o_next_state_flat <= next_state_next_flat;
            end
        end
    end

    initial begin
        if (LANES != 10)
            $error("unified LIF arithmetic candidate requires ten lanes");
    end
endmodule

`default_nettype wire
