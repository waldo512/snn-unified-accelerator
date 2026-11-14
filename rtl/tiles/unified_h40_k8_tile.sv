`timescale 1ns/1ps
`default_nettype none

module unified_h40_k8_tile #(
    parameter integer NEURONS = 40,
    parameter integer LANES = 8
) (
    input  wire logic                   clk,
    input  wire logic                   rst_n,
    input  wire logic                   i_clear_sample,
    input  wire logic                   i_clear_accum,

    input  wire logic                   i_mac_valid,
    output logic                        o_mac_ready,
    input  wire logic [LANES-1:0]       i_lane_valid,
    input  wire logic [LANES*6-1:0]     i_neuron_index_flat,
    input  wire logic [LANES*8-1:0]     i_activation_flat,
    input  wire logic [LANES*8-1:0]     i_weight_flat,
    input  wire logic                   i_acc24_compat,

    input  wire logic                   i_lif_start,
    input  wire logic signed [15:0]     i_leak_q15,
    input  wire logic signed [31:0]     i_threshold,
    input  wire logic [4:0]             i_current_left_shift,
    input  wire logic                   i_reset_to_zero,
    output logic                        o_lif_busy,
    output logic                        o_lif_done,
    output logic                        o_lif_spike_valid,
    output logic [5:0]                  o_lif_base,
    output logic [LANES-1:0]            o_lif_spike_mask,

    input  wire logic                   i_timestep_commit,
    output logic [NEURONS-1:0]          o_previous_spikes,
    output logic [NEURONS-1:0]          o_current_spikes,

    input  wire logic [5:0]             i_debug_neuron,
    output logic signed [31:0]          o_debug_accum,
    output logic signed [31:0]          o_debug_membrane,
    output logic                        o_lane_collision_error
);
    logic signed [31:0] accum [0:NEURONS-1];
    logic signed [31:0] membrane [0:NEURONS-1];
    logic [NEURONS-1:0] previous_spikes_q;
    logic [NEURONS-1:0] current_spikes_q;
    logic [5:0] lif_base_q;
    logic lane_collision;

    integer reset_i;
    integer comb_lane_i;
    integer comb_lane_j;
    integer seq_lane_i;
    integer neuron_i;
    logic [5:0] neuron_index [0:LANES-1];
    logic signed [7:0] activation [0:LANES-1];
    logic signed [7:0] weight [0:LANES-1];

    function automatic logic signed [31:0] add_product(
        input logic signed [31:0] current,
        input logic signed [7:0] lhs,
        input logic signed [7:0] rhs,
        input logic acc24_compat
    );
        logic signed [15:0] product;
        logic signed [32:0] sum_wide;
        logic signed [31:0] sum32;
        begin
            product = lhs * rhs;
            sum_wide = {current[31], current} +
                {{17{product[15]}}, product};
            if (sum_wide > 33'sh07fffffff)
                sum32 = 32'sh7fffffff;
            else if (sum_wide < -33'sh080000000)
                sum32 = 32'sh80000000;
            else
                sum32 = sum_wide[31:0];
            if (acc24_compat)
                add_product = {{8{sum32[23]}}, sum32[23:0]};
            else
                add_product = sum32;
        end
    endfunction

    function automatic logic signed [31:0] lif_pre_value(
        input logic signed [31:0] old_membrane,
        input logic signed [31:0] current,
        input logic signed [15:0] leak_q15,
        input logic [4:0] current_left_shift
    );
        logic signed [47:0] leak_product;
        logic signed [63:0] current_scaled;
        logic signed [63:0] sum_wide;
        begin
            leak_product = old_membrane * leak_q15;
            current_scaled = $signed(current);
            current_scaled = current_scaled <<< current_left_shift;
            sum_wide = (leak_product >>> 15) + current_scaled;
            if (sum_wide > 64'sh000000007fffffff)
                lif_pre_value = 32'sh7fffffff;
            else if (sum_wide < -64'sh0000000080000000)
                lif_pre_value = 32'sh80000000;
            else
                lif_pre_value = sum_wide[31:0];
        end
    endfunction

    always_comb begin
        for (comb_lane_i = 0; comb_lane_i < LANES;
             comb_lane_i = comb_lane_i + 1) begin
            neuron_index[comb_lane_i] =
                i_neuron_index_flat[comb_lane_i*6 +: 6];
            activation[comb_lane_i] =
                i_activation_flat[comb_lane_i*8 +: 8];
            weight[comb_lane_i] = i_weight_flat[comb_lane_i*8 +: 8];
        end

        lane_collision = 1'b0;
        for (comb_lane_i = 0; comb_lane_i < LANES;
             comb_lane_i = comb_lane_i + 1)
            for (comb_lane_j = comb_lane_i + 1; comb_lane_j < LANES;
                 comb_lane_j = comb_lane_j + 1)
                if (i_lane_valid[comb_lane_i] && i_lane_valid[comb_lane_j] &&
                    (neuron_index[comb_lane_i] ==
                     neuron_index[comb_lane_j]))
                    lane_collision = 1'b1;

        o_mac_ready = !o_lif_busy;
        o_previous_spikes = previous_spikes_q;
        o_current_spikes = current_spikes_q;
        o_debug_accum = (i_debug_neuron < NEURONS) ?
            accum[i_debug_neuron] : 32'sd0;
        o_debug_membrane = (i_debug_neuron < NEURONS) ?
            membrane[i_debug_neuron] : 32'sd0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (reset_i = 0; reset_i < NEURONS; reset_i = reset_i + 1) begin
                accum[reset_i] <= 32'sd0;
                membrane[reset_i] <= 32'sd0;
            end
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
            lif_base_q <= 6'd0;
            o_lif_busy <= 1'b0;
            o_lif_done <= 1'b0;
            o_lif_spike_valid <= 1'b0;
            o_lif_base <= 6'd0;
            o_lif_spike_mask <= '0;
            o_lane_collision_error <= 1'b0;
        end else begin
            o_lif_done <= 1'b0;
            o_lif_spike_valid <= 1'b0;

            if (i_clear_sample) begin
                for (reset_i = 0; reset_i < NEURONS; reset_i = reset_i + 1) begin
                    accum[reset_i] <= 32'sd0;
                    membrane[reset_i] <= 32'sd0;
                end
                previous_spikes_q <= '0;
                current_spikes_q <= '0;
                o_lif_busy <= 1'b0;
                o_lane_collision_error <= 1'b0;
            end else begin
                if (i_clear_accum)
                    for (reset_i = 0; reset_i < NEURONS; reset_i = reset_i + 1)
                        accum[reset_i] <= 32'sd0;

                if (i_mac_valid && o_mac_ready) begin
                    if (lane_collision) begin
                        o_lane_collision_error <= 1'b1;
                    end else begin
                        for (seq_lane_i = 0; seq_lane_i < LANES;
                             seq_lane_i = seq_lane_i + 1)
                            if (i_lane_valid[seq_lane_i] &&
                                (neuron_index[seq_lane_i] < NEURONS))
                                accum[neuron_index[seq_lane_i]] <= add_product(
                                    accum[neuron_index[seq_lane_i]],
                                    activation[seq_lane_i], weight[seq_lane_i],
                                    i_acc24_compat);
                    end
                end

                if (i_lif_start && !o_lif_busy) begin
                    lif_base_q <= 6'd0;
                    current_spikes_q <= '0;
                    o_lif_busy <= 1'b1;
                end else if (o_lif_busy) begin
                    o_lif_spike_valid <= 1'b1;
                    o_lif_base <= lif_base_q;
                    o_lif_spike_mask <= '0;
                    for (seq_lane_i = 0; seq_lane_i < LANES;
                         seq_lane_i = seq_lane_i + 1) begin
                        neuron_i = lif_base_q + seq_lane_i;
                        if (neuron_i < NEURONS) begin
                            if (lif_pre_value(membrane[neuron_i], accum[neuron_i],
                                              i_leak_q15,
                                              i_current_left_shift) >= i_threshold) begin
                                current_spikes_q[neuron_i] <= 1'b1;
                                o_lif_spike_mask[seq_lane_i] <= 1'b1;
                                if (i_reset_to_zero)
                                    membrane[neuron_i] <= 32'sd0;
                                else
                                    membrane[neuron_i] <=
                                        lif_pre_value(membrane[neuron_i],
                                                      accum[neuron_i], i_leak_q15,
                                                      i_current_left_shift) - i_threshold;
                            end else begin
                                current_spikes_q[neuron_i] <= 1'b0;
                                membrane[neuron_i] <=
                                    lif_pre_value(membrane[neuron_i], accum[neuron_i],
                                                  i_leak_q15,
                                                  i_current_left_shift);
                            end
                            accum[neuron_i] <= 32'sd0;
                        end
                    end
                    if ((lif_base_q + LANES) >= NEURONS) begin
                        o_lif_busy <= 1'b0;
                        o_lif_done <= 1'b1;
                    end else begin
                        lif_base_q <= lif_base_q + LANES;
                    end
                end

                if (i_timestep_commit) begin
                    previous_spikes_q <= current_spikes_q;
                    current_spikes_q <= '0;
                end
            end
        end
    end
endmodule

`default_nettype wire
