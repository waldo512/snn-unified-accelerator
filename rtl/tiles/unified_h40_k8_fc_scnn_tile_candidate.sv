`default_nettype none

// Candidate replacement for an H40 Tile. The same eight signed INT8
// multipliers feed either the unchanged FC accumulator/LIF path or a four-phase
// exact Q8.8 SCNN dot product. It remains a candidate until FC official
// regression and physical OOC gates pass.
module unified_h40_k8_fc_scnn_tile_candidate #(
    parameter int NEURONS = 40,
    parameter int LANES = 8
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     i_scnn_mode,
    input  logic                     i_clear_sample,
    input  logic                     i_clear_accum,

    input  logic                     i_mac_valid,
    output logic                     o_mac_ready,
    input  logic [LANES-1:0]         i_lane_valid,
    input  logic [LANES*6-1:0]       i_neuron_index_flat,
    input  logic [LANES*8-1:0]       i_activation_flat,
    input  logic [LANES*8-1:0]       i_weight_flat,
    input  logic                     i_acc24_compat,

    input  logic                     i_lif_start,
    input  logic signed [15:0]       i_leak_q15,
    input  logic signed [31:0]       i_threshold,
    input  logic [4:0]               i_current_left_shift,
    input  logic                     i_reset_to_zero,
    output logic                     o_lif_busy,
    output logic                     o_lif_done,
    output logic                     o_lif_spike_valid,
    output logic [5:0]               o_lif_base,
    output logic [LANES-1:0]         o_lif_spike_mask,

    input  logic                     i_timestep_commit,
    output logic [NEURONS-1:0]      o_previous_spikes,
    output logic [NEURONS-1:0]      o_current_spikes,
    input  logic [5:0]               i_debug_neuron,
    output logic signed [31:0]       o_debug_accum,
    output logic signed [31:0]       o_debug_membrane,
    output logic                     o_lane_collision_error,

    input  logic                     i_scnn_clear_accum,
    input  logic                     i_scnn_valid,
    output logic                     o_scnn_ready,
    input  logic [LANES*16-1:0]      i_scnn_feature_flat,
    input  logic [LANES*16-1:0]      i_scnn_weight_flat,
    input  logic [LANES-1:0]         i_scnn_lane_mask,
    input  logic [5:0]               i_scnn_destination,
    input  logic                     i_scnn_first,
    input  logic                     i_scnn_last,
    output logic                     o_scnn_valid,
    input  logic                     i_scnn_ready,
    output logic [5:0]               o_scnn_destination,
    output logic signed [47:0]       o_scnn_accumulator,
    output logic                     o_scnn_busy,
    output logic                     o_protocol_error
);
    logic signed [31:0] fc_accum [0:NEURONS-1];
    logic signed [31:0] membrane [0:NEURONS-1];
    logic [NEURONS-1:0] previous_spikes_q;
    logic [NEURONS-1:0] current_spikes_q;
    logic [5:0] lif_base_q;
    logic lane_collision;
    logic [5:0] neuron_index [0:LANES-1];

    logic [LANES*8-1:0] multiplier_lhs;
    logic [LANES*8-1:0] multiplier_rhs;
    logic [LANES*16-1:0] multiplier_product;
    logic [LANES*16-1:0] scnn_feature_q;
    logic [LANES*16-1:0] scnn_weight_q;
    logic [LANES-1:0] scnn_mask_q;
    logic [5:0] scnn_destination_q;
    logic scnn_first_q;
    logic scnn_last_q;
    logic [1:0] scnn_phase_q;
    logic signed [47:0] scnn_phase_sum_q;
    logic signed [47:0] scnn_correction_sum_q;
    logic signed [47:0] scnn_accum [0:NEURONS-1];
    logic signed [47:0] shifted_product_sum;
    logic signed [47:0] correction_sum_input;
    logic signed [31:0] correction_value [0:LANES-1];
    logic signed [47:0] scnn_chunk_total;
    logic signed [47:0] scnn_next_accum;
    logic scnn_result_valid_q;
    logic [5:0] scnn_result_destination_q;
    logic signed [47:0] scnn_result_q;

    function automatic logic signed [31:0] add_fc_product(
        input logic signed [31:0] current,
        input logic signed [15:0] product,
        input logic acc24_compat
    );
        logic signed [32:0] sum_wide;
        logic signed [31:0] sum32;
        begin
            sum_wide = {current[31], current} + {{17{product[15]}}, product};
            if (sum_wide > 33'sh07fffffff)
                sum32 = 32'sh7fffffff;
            else if (sum_wide < -33'sh080000000)
                sum32 = 32'sh80000000;
            else
                sum32 = sum_wide[31:0];
            if (acc24_compat)
                add_fc_product = {{8{sum32[23]}}, sum32[23:0]};
            else
                add_fc_product = sum32;
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

    function automatic logic signed [31:0] q8_correction(
        input logic signed [15:0] lhs,
        input logic signed [15:0] rhs
    );
        logic signed [31:0] lhs_hi;
        logic signed [31:0] rhs_hi;
        logic signed [31:0] lhs_lo7;
        logic signed [31:0] rhs_lo7;
        logic signed [31:0] value;
        begin
            lhs_hi = {{24{lhs[15]}}, lhs[15:8]};
            rhs_hi = {{24{rhs[15]}}, rhs[15:8]};
            lhs_lo7 = {25'd0, lhs[6:0]};
            rhs_lo7 = {25'd0, rhs[6:0]};
            value = 32'sd0;
            if (rhs[7]) value = value + (lhs_hi <<< 15) + (lhs_lo7 <<< 7);
            if (lhs[7]) value = value + (rhs_hi <<< 15) + (rhs_lo7 <<< 7);
            if (lhs[7] && rhs[7]) value = value + 32'sd16384;
            q8_correction = value;
        end
    endfunction

    unified_shared_k8_multiplier_bank #(.LANES(LANES)) u_multiplier_bank (
        .i_lhs_flat(multiplier_lhs), .i_rhs_flat(multiplier_rhs),
        .o_product_flat(multiplier_product)
    );

    always_comb begin
        multiplier_lhs = i_activation_flat;
        multiplier_rhs = i_weight_flat;
        for (int lane = 0; lane < LANES; lane = lane + 1) begin
            neuron_index[lane] = i_neuron_index_flat[lane*6 +: 6];
            if (i_scnn_mode) begin
                case (scnn_phase_q)
                    2'd0: begin
                        multiplier_lhs[lane*8 +: 8] = scnn_feature_q[lane*16+8 +: 8];
                        multiplier_rhs[lane*8 +: 8] = scnn_weight_q[lane*16+8 +: 8];
                    end
                    2'd1: begin
                        multiplier_lhs[lane*8 +: 8] = scnn_feature_q[lane*16+8 +: 8];
                        multiplier_rhs[lane*8 +: 8] = {1'b0, scnn_weight_q[lane*16 +: 7]};
                    end
                    2'd2: begin
                        multiplier_lhs[lane*8 +: 8] = scnn_weight_q[lane*16+8 +: 8];
                        multiplier_rhs[lane*8 +: 8] = {1'b0, scnn_feature_q[lane*16 +: 7]};
                    end
                    default: begin
                        multiplier_lhs[lane*8 +: 8] = {1'b0, scnn_feature_q[lane*16 +: 7]};
                        multiplier_rhs[lane*8 +: 8] = {1'b0, scnn_weight_q[lane*16 +: 7]};
                    end
                endcase
            end
        end

        lane_collision = 1'b0;
        for (int lhs_lane = 0; lhs_lane < LANES; lhs_lane = lhs_lane + 1)
            for (int rhs_lane = lhs_lane + 1; rhs_lane < LANES; rhs_lane = rhs_lane + 1)
                if (i_lane_valid[lhs_lane] && i_lane_valid[rhs_lane] &&
                    neuron_index[lhs_lane] == neuron_index[rhs_lane])
                    lane_collision = 1'b1;

        shifted_product_sum = '0;
        correction_sum_input = '0;
        for (int lane = 0; lane < LANES; lane = lane + 1) begin
            correction_value[lane] = q8_correction(
                $signed(i_scnn_feature_flat[lane*16 +: 16]),
                $signed(i_scnn_weight_flat[lane*16 +: 16]));
            if (scnn_mask_q[lane]) begin
                case (scnn_phase_q)
                    2'd0: shifted_product_sum = shifted_product_sum +
                        ({{32{multiplier_product[lane*16+15]}},
                          multiplier_product[lane*16 +: 16]} <<< 16);
                    2'd1, 2'd2: shifted_product_sum = shifted_product_sum +
                        ({{32{multiplier_product[lane*16+15]}},
                          multiplier_product[lane*16 +: 16]} <<< 8);
                    default: shifted_product_sum = shifted_product_sum +
                        {{32{multiplier_product[lane*16+15]}},
                          multiplier_product[lane*16 +: 16]};
                endcase
            end
            if (i_scnn_lane_mask[lane])
                correction_sum_input = correction_sum_input +
                    {{16{correction_value[lane][31]}},
                      correction_value[lane]};
        end
        scnn_chunk_total = scnn_phase_sum_q + shifted_product_sum +
                           scnn_correction_sum_q;
        scnn_next_accum = (scnn_first_q ? 48'sd0 :
            scnn_accum[scnn_destination_q]) + scnn_chunk_total;

        o_mac_ready = !i_scnn_mode && !o_lif_busy && !o_scnn_busy &&
                      !scnn_result_valid_q;
        o_scnn_ready = i_scnn_mode && !o_scnn_busy && !scnn_result_valid_q &&
                       i_scnn_destination < NEURONS;
        o_previous_spikes = previous_spikes_q;
        o_current_spikes = current_spikes_q;
        o_debug_accum = i_debug_neuron < NEURONS ? fc_accum[i_debug_neuron] : 32'sd0;
        o_debug_membrane = i_debug_neuron < NEURONS ? membrane[i_debug_neuron] : 32'sd0;
        o_scnn_valid = scnn_result_valid_q;
        o_scnn_destination = scnn_result_destination_q;
        o_scnn_accumulator = scnn_result_q;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
            lif_base_q <= '0;
            o_lif_busy <= 1'b0;
            o_lif_done <= 1'b0;
            o_lif_spike_valid <= 1'b0;
            o_lif_base <= '0;
            o_lif_spike_mask <= '0;
            o_lane_collision_error <= 1'b0;
            scnn_feature_q <= '0;
            scnn_weight_q <= '0;
            scnn_mask_q <= '0;
            scnn_destination_q <= '0;
            scnn_first_q <= 1'b0;
            scnn_last_q <= 1'b0;
            scnn_phase_q <= '0;
            scnn_phase_sum_q <= '0;
            scnn_correction_sum_q <= '0;
            o_scnn_busy <= 1'b0;
            scnn_result_valid_q <= 1'b0;
            scnn_result_destination_q <= '0;
            scnn_result_q <= '0;
            o_protocol_error <= 1'b0;
            for (int index = 0; index < NEURONS; index = index + 1) begin
                fc_accum[index] <= '0;
                membrane[index] <= '0;
                scnn_accum[index] <= '0;
            end
        end else begin
            o_lif_done <= 1'b0;
            o_lif_spike_valid <= 1'b0;
            if (scnn_result_valid_q && i_scnn_ready)
                scnn_result_valid_q <= 1'b0;

            if (i_clear_sample) begin
                previous_spikes_q <= '0;
                current_spikes_q <= '0;
                o_lif_busy <= 1'b0;
                o_scnn_busy <= 1'b0;
                scnn_result_valid_q <= 1'b0;
                o_lane_collision_error <= 1'b0;
                o_protocol_error <= 1'b0;
                for (int index = 0; index < NEURONS; index = index + 1) begin
                    fc_accum[index] <= '0;
                    membrane[index] <= '0;
                    scnn_accum[index] <= '0;
                end
            end else begin
                if (i_clear_accum && !i_scnn_mode)
                    for (int index = 0; index < NEURONS; index = index + 1)
                        fc_accum[index] <= '0;
                if (i_scnn_clear_accum && !o_scnn_busy && !scnn_result_valid_q)
                    for (int index = 0; index < NEURONS; index = index + 1)
                        scnn_accum[index] <= '0;
                else if (i_scnn_clear_accum)
                    o_protocol_error <= 1'b1;

                if (i_mac_valid && o_mac_ready) begin
                    if (lane_collision) begin
                        o_lane_collision_error <= 1'b1;
                    end else begin
                        for (int lane = 0; lane < LANES; lane = lane + 1)
                            if (i_lane_valid[lane] && neuron_index[lane] < NEURONS)
                                fc_accum[neuron_index[lane]] <= add_fc_product(
                                    fc_accum[neuron_index[lane]],
                                    $signed(multiplier_product[lane*16 +: 16]),
                                    i_acc24_compat);
                    end
                end

                if (i_lif_start && !i_scnn_mode && !o_lif_busy) begin
                    lif_base_q <= '0;
                    current_spikes_q <= '0;
                    o_lif_busy <= 1'b1;
                end else if (i_lif_start && i_scnn_mode) begin
                    o_protocol_error <= 1'b1;
                end else if (o_lif_busy) begin
                    o_lif_spike_valid <= 1'b1;
                    o_lif_base <= lif_base_q;
                    o_lif_spike_mask <= '0;
                    for (int lane = 0; lane < LANES; lane = lane + 1) begin
                        int neuron;
                        neuron = lif_base_q + lane;
                        if (neuron < NEURONS) begin
                            if (lif_pre_value(membrane[neuron], fc_accum[neuron],
                                              i_leak_q15, i_current_left_shift) >=
                                i_threshold) begin
                                current_spikes_q[neuron] <= 1'b1;
                                o_lif_spike_mask[lane] <= 1'b1;
                                if (i_reset_to_zero)
                                    membrane[neuron] <= '0;
                                else
                                    membrane[neuron] <= lif_pre_value(
                                        membrane[neuron], fc_accum[neuron],
                                        i_leak_q15, i_current_left_shift) - i_threshold;
                            end else begin
                                current_spikes_q[neuron] <= 1'b0;
                                membrane[neuron] <= lif_pre_value(
                                    membrane[neuron], fc_accum[neuron],
                                    i_leak_q15, i_current_left_shift);
                            end
                            fc_accum[neuron] <= '0;
                        end
                    end
                    if (lif_base_q + LANES >= NEURONS) begin
                        o_lif_busy <= 1'b0;
                        o_lif_done <= 1'b1;
                    end else begin
                        lif_base_q <= lif_base_q + LANES;
                    end
                end
                if (i_timestep_commit && !i_scnn_mode) begin
                    previous_spikes_q <= current_spikes_q;
                    current_spikes_q <= '0;
                end else if (i_timestep_commit && i_scnn_mode) begin
                    o_protocol_error <= 1'b1;
                end

                if (i_scnn_valid && o_scnn_ready) begin
                    scnn_feature_q <= i_scnn_feature_flat;
                    scnn_weight_q <= i_scnn_weight_flat;
                    scnn_mask_q <= i_scnn_lane_mask;
                    scnn_destination_q <= i_scnn_destination;
                    scnn_first_q <= i_scnn_first;
                    scnn_last_q <= i_scnn_last;
                    scnn_phase_q <= '0;
                    scnn_phase_sum_q <= '0;
                    scnn_correction_sum_q <= correction_sum_input;
                    o_scnn_busy <= 1'b1;
                end else if (i_scnn_valid && !i_scnn_mode) begin
                    o_protocol_error <= 1'b1;
                end

                if (o_scnn_busy) begin
                    if (scnn_phase_q == 2'd3) begin
                        o_scnn_busy <= 1'b0;
                        scnn_phase_q <= '0;
                        scnn_phase_sum_q <= '0;
                        if (scnn_last_q) begin
                            scnn_accum[scnn_destination_q] <= '0;
                            scnn_result_q <= scnn_next_accum;
                            scnn_result_destination_q <= scnn_destination_q;
                            scnn_result_valid_q <= 1'b1;
                        end else begin
                            scnn_accum[scnn_destination_q] <= scnn_next_accum;
                        end
                    end else begin
                        scnn_phase_sum_q <= scnn_phase_sum_q + shifted_product_sum;
                        scnn_phase_q <= scnn_phase_q + 1'b1;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
