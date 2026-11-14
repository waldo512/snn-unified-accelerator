`timescale 1ns/1ps
`default_nettype none

// Candidate for replacing the separate FC_COMPAT and SRNN state/LIF blocks.
// Forty neuron states remain resident while eight physical LIF lanes scan five
// words. FC mode preserves ACC18 wrap, rounded leak, soft reset and a UQ6.10
// trace semantics. SRNN mode preserves the existing ACC32 saturation,
// unrounded leak, current shift and selectable reset semantics.
//
// A mode change is legal only after the owner has drained traffic and cleared
// the tile state. This leaf intentionally keeps that policy outside the PE.
module unified_h40_fc_srnn_state_candidate_v1 #(
    parameter integer TILE_BASE = 0,
    parameter integer TILE_NEURONS = 40,
    parameter integer LANES = 8,
    parameter integer FC_ACC_BITS = 18,
    parameter integer TRACE_BITS = 16,
    parameter integer FC_TRACE_FRAC_BITS = 10
) (
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 i_abort,
    input  logic                                 i_clear_spikes,
    input  logic                                 i_mode_fc,
    input  logic                                 i_clear_all,
    input  logic [TILE_NEURONS*FC_ACC_BITS-1:0] i_fc_accumulator_flat,

    input  logic                                 i_clear_word_valid,
    input  logic [9:0]                           i_clear_word_base,

    input  logic                                 i_accumulate_valid,
    input  logic [9:0]                           i_accumulate_base,
    input  logic [LANES-1:0]                     i_accumulate_lane_mask,
    input  logic [LANES*32-1:0]                  i_accumulate_contribution_flat,
    input  logic [9:0]                           i_active_neurons,

    input  logic                                 i_lif_begin,
    input  logic                                 i_lif_word_valid,
    input  logic [9:0]                           i_lif_word_base,
    input  logic signed [15:0]                   i_leak_q15,
    input  logic signed [31:0]                   i_threshold,
    input  logic [4:0]                           i_current_left_shift,
    input  logic                                 i_reset_to_zero,
    input  logic [15:0]                          i_trace_decay_q16,
    input  logic                                 i_commit_timestep,

    output logic [TILE_NEURONS-1:0]              o_previous_spikes,
    output logic [TILE_NEURONS-1:0]              o_current_spikes,
    output logic [TILE_NEURONS*32-1:0]           o_accumulator_flat,
    output logic [TILE_NEURONS*32-1:0]           o_membrane_flat,
    output logic [TILE_NEURONS*TRACE_BITS-1:0]   o_trace_flat
);
    localparam integer WORD_COUNT = TILE_NEURONS / LANES;
    localparam integer WORD_INDEX_WIDTH =
        (WORD_COUNT <= 1) ? 1 : $clog2(WORD_COUNT);

    logic signed [31:0] accumulator_bank [0:LANES-1][0:WORD_COUNT-1];
    logic signed [31:0] membrane_bank [0:LANES-1][0:WORD_COUNT-1];
    logic [TRACE_BITS-1:0] trace_bank [0:LANES-1][0:WORD_COUNT-1];
    logic [TILE_NEURONS-1:0] previous_spikes_q;
    logic [TILE_NEURONS-1:0] current_spikes_q;

    logic clear_hits;
    logic accumulate_hits;
    logic lif_hits;
    logic [10:0] clear_offset;
    logic [10:0] accumulate_offset;
    logic [10:0] lif_offset;
    logic [WORD_INDEX_WIDTH-1:0] clear_word;
    logic [WORD_INDEX_WIDTH-1:0] accumulate_word;
    logic [WORD_INDEX_WIDTH-1:0] lif_word;

    logic signed [31:0] lif_membrane_operand [0:LANES-1];
    logic signed [47:0] lif_leak_product [0:LANES-1];
    logic signed [31:0] lif_next_membrane [0:LANES-1];
    logic [TRACE_BITS-1:0] lif_next_trace [0:LANES-1];
    logic [LANES-1:0] lif_next_spike;

    function automatic logic signed [31:0] saturating_add32(
        input logic signed [31:0] lhs,
        input logic signed [31:0] rhs
    );
        logic signed [32:0] sum_wide;
        begin
            sum_wide = {lhs[31], lhs} + {rhs[31], rhs};
            if (sum_wide > 33'sh07fffffff)
                saturating_add32 = 32'sh7fffffff;
            else if (sum_wide < -33'sh080000000)
                saturating_add32 = 32'sh80000000;
            else
                saturating_add32 = sum_wide[31:0];
        end
    endfunction

    function automatic logic signed [31:0] wrapping_add18(
        input logic signed [31:0] lhs,
        input logic signed [31:0] rhs
    );
        logic signed [FC_ACC_BITS-1:0] sum_narrow;
        begin
            sum_narrow = lhs[FC_ACC_BITS-1:0] + rhs[FC_ACC_BITS-1:0];
            wrapping_add18 = {{(32-FC_ACC_BITS){sum_narrow[FC_ACC_BITS-1]}},
                sum_narrow};
        end
    endfunction

    function automatic logic signed [31:0] saturate64_to32(
        input logic signed [63:0] value
    );
        begin
            if (value > 64'sh000000007fffffff)
                saturate64_to32 = 32'sh7fffffff;
            else if (value < -64'sh0000000080000000)
                saturate64_to32 = 32'sh80000000;
            else
                saturate64_to32 = value[31:0];
        end
    endfunction

    always_comb begin
        clear_offset = {1'b0, i_clear_word_base} - 11'(TILE_BASE);
        accumulate_offset = {1'b0, i_accumulate_base} - 11'(TILE_BASE);
        lif_offset = {1'b0, i_lif_word_base} - 11'(TILE_BASE);
        clear_hits = i_clear_word_valid && !clear_offset[10] &&
            clear_offset < 11'(TILE_NEURONS);
        accumulate_hits = i_accumulate_valid && !accumulate_offset[10] &&
            accumulate_offset < 11'(TILE_NEURONS);
        lif_hits = i_lif_word_valid && !lif_offset[10] &&
            lif_offset < 11'(TILE_NEURONS);
        clear_word = WORD_INDEX_WIDTH'(clear_offset / LANES);
        accumulate_word = WORD_INDEX_WIDTH'(accumulate_offset / LANES);
        lif_word = WORD_INDEX_WIDTH'(lif_offset / LANES);

        o_previous_spikes = previous_spikes_q;
        o_current_spikes = current_spikes_q;
        o_accumulator_flat = '0;
        o_membrane_flat = '0;
        o_trace_flat = '0;
        for (int neuron = 0; neuron < TILE_NEURONS; neuron++) begin
            o_accumulator_flat[neuron*32 +: 32] =
                accumulator_bank[neuron % LANES][neuron / LANES];
            o_membrane_flat[neuron*32 +: 32] =
                membrane_bank[neuron % LANES][neuron / LANES];
            o_trace_flat[neuron*TRACE_BITS +: TRACE_BITS] =
                trace_bank[neuron % LANES][neuron / LANES];
        end
    end

    // One membrane multiplier per physical lane is shared by both modes. The
    // FC operand is sign-extended from 18 bits before entering the common path.
    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            logic signed [47:0] fc_rounded_product;
            logic signed [FC_ACC_BITS-1:0] fc_leak;
            logic signed [FC_ACC_BITS-1:0] fc_current;
            logic signed [FC_ACC_BITS-1:0] fc_pre;
            logic signed [FC_ACC_BITS-1:0] fc_threshold;
            logic signed [FC_ACC_BITS-1:0] fc_next;
            logic signed [63:0] srnn_current_scaled;
            logic signed [63:0] srnn_pre_wide;
            logic signed [31:0] srnn_pre;
            logic [31:0] trace_product;
            logic [31:0] trace_shifted;
            logic [TRACE_BITS-1:0] trace_decayed;
            logic [TRACE_BITS:0] fc_trace_sum;

            if (i_mode_fc) begin
                lif_membrane_operand[lane] =
                    {{(32-FC_ACC_BITS){
                        membrane_bank[lane][lif_word][FC_ACC_BITS-1]}},
                     membrane_bank[lane][lif_word][FC_ACC_BITS-1:0]};
            end else begin
                lif_membrane_operand[lane] = membrane_bank[lane][lif_word];
            end
            lif_leak_product[lane] =
                lif_membrane_operand[lane] * i_leak_q15;

            fc_rounded_product = lif_leak_product[lane] + 48'sd16384;
            fc_leak = fc_rounded_product >>> 15;
            fc_current = i_fc_accumulator_flat[
                (lif_word*LANES + lane)*FC_ACC_BITS +: FC_ACC_BITS];
            fc_pre = fc_leak + fc_current;
            fc_threshold = i_threshold[FC_ACC_BITS-1:0];
            lif_next_spike[lane] = fc_pre >= fc_threshold;
            fc_next = lif_next_spike[lane] ?
                (fc_pre - fc_threshold) : fc_pre;

            srnn_current_scaled =
                {{32{accumulator_bank[lane][lif_word][31]}},
                 accumulator_bank[lane][lif_word]};
            srnn_current_scaled =
                srnn_current_scaled <<< i_current_left_shift;
            srnn_pre_wide =
                ($signed({{16{lif_leak_product[lane][47]}},
                           lif_leak_product[lane]}) >>> 15) +
                srnn_current_scaled;
            srnn_pre = saturate64_to32(srnn_pre_wide);

            trace_product = $unsigned(trace_bank[lane][lif_word]) *
                $unsigned(i_trace_decay_q16);
            trace_shifted = (trace_product + 32'd32768) >> 16;
            trace_decayed = trace_shifted[TRACE_BITS-1:0];
            fc_trace_sum = {1'b0, trace_decayed} +
                (lif_next_spike[lane] ?
                 ((TRACE_BITS+1)'(1) << FC_TRACE_FRAC_BITS) : '0);

            if (i_mode_fc) begin
                lif_next_membrane[lane] =
                    {{(32-FC_ACC_BITS){fc_next[FC_ACC_BITS-1]}}, fc_next};
                lif_next_trace[lane] = fc_trace_sum[TRACE_BITS] ?
                    {TRACE_BITS{1'b1}} : fc_trace_sum[TRACE_BITS-1:0];
            end else begin
                lif_next_spike[lane] = srnn_pre >= i_threshold;
                if (lif_next_spike[lane]) begin
                    lif_next_membrane[lane] = i_reset_to_zero ? 32'sd0 :
                        (srnn_pre - i_threshold);
                end else begin
                    lif_next_membrane[lane] = srnn_pre;
                end
                lif_next_trace[lane] = trace_bank[lane][lif_word];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
            for (int lane = 0; lane < LANES; lane++) begin
                for (int word = 0; word < WORD_COUNT; word++) begin
                    accumulator_bank[lane][word] <= 32'sd0;
                    membrane_bank[lane][word] <= 32'sd0;
                    trace_bank[lane][word] <= '0;
                end
            end
        end else if (i_clear_all) begin
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
            for (int lane = 0; lane < LANES; lane++) begin
                for (int word = 0; word < WORD_COUNT; word++) begin
                    accumulator_bank[lane][word] <= 32'sd0;
                    membrane_bank[lane][word] <= 32'sd0;
                    trace_bank[lane][word] <= '0;
                end
            end
        end else if (i_abort) begin
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
        end else begin
            if (i_clear_spikes) begin
                previous_spikes_q <= '0;
                current_spikes_q <= '0;
            end

            if (clear_hits) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    accumulator_bank[lane][clear_word] <= 32'sd0;
                    membrane_bank[lane][clear_word] <= 32'sd0;
                    trace_bank[lane][clear_word] <= '0;
                    previous_spikes_q[clear_word*LANES + lane] <= 1'b0;
                    current_spikes_q[clear_word*LANES + lane] <= 1'b0;
                end
            end

            if (accumulate_hits) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    if (i_accumulate_lane_mask[lane] &&
                        (i_accumulate_base + 10'(lane)) < i_active_neurons) begin
                        if (i_mode_fc) begin
                            accumulator_bank[lane][accumulate_word] <=
                                wrapping_add18(
                                    accumulator_bank[lane][accumulate_word],
                                    $signed(i_accumulate_contribution_flat[
                                        lane*32 +: 32]));
                        end else begin
                            accumulator_bank[lane][accumulate_word] <=
                                saturating_add32(
                                    accumulator_bank[lane][accumulate_word],
                                    $signed(i_accumulate_contribution_flat[
                                        lane*32 +: 32]));
                        end
                    end
                end
            end

            if (i_lif_begin)
                current_spikes_q <= '0;

            if (lif_hits) begin
                for (int lane = 0; lane < LANES; lane++) begin
                    if ((i_lif_word_base + 10'(lane)) < i_active_neurons) begin
                        current_spikes_q[lif_word*LANES + lane] <=
                            lif_next_spike[lane];
                        membrane_bank[lane][lif_word] <=
                            lif_next_membrane[lane];
                        trace_bank[lane][lif_word] <= lif_next_trace[lane];
                        accumulator_bank[lane][lif_word] <= 32'sd0;
                    end
                end
            end

            if (i_commit_timestep) begin
                previous_spikes_q <= current_spikes_q;
                current_spikes_q <= '0;
            end
        end
    end

    initial begin
        if (TILE_NEURONS % LANES != 0)
            $error("shared FC/SRNN state requires TILE_NEURONS divisible by LANES");
        if (TILE_BASE % LANES != 0)
            $error("shared FC/SRNN state TILE_BASE must align to LANES");
        if (FC_ACC_BITS >= 32)
            $error("shared FC/SRNN state requires FC_ACC_BITS below 32");
        if ((FC_TRACE_FRAC_BITS < 0) ||
            (FC_TRACE_FRAC_BITS >= TRACE_BITS))
            $error("FC trace fractional width is out of range");
    end
endmodule

`default_nettype wire
