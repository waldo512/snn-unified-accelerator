`timescale 1ns/1ps
`default_nettype none

// Replaceable tile-local boundary for the first physical-reuse optimization.
// FC keeps its two producer accumulator banks so W0 production can overlap a
// completed timestep. Their selected ACC18 vector is consumed directly by the
// same eight-lane state/LIF block that owns SRNN ACC32, membrane and spikes.
module unified_h40_fc_srnn_shared_state_engine_candidate_v1 #(
    parameter integer TILE_BASE = 0,
    parameter integer H_COUNT = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer W_BITS = 5,
    parameter integer FC_ACC_BITS = 18,
    parameter integer INPUT_SHIFT = 6,
    parameter integer T_BITS = 9,
    parameter integer LEAK_M_Q15 = 32113,
    parameter integer TRACE_DECAY_Q16 = 64225,
    parameter integer VTH = 1024
) (
    input  logic                                     clk,
    input  logic                                     rst_n,
    input  logic                                     i_abort,
    input  logic                                     i_mode_fc,

    input  logic                                     i_fc_clear_state,
    input  logic                                     i_fc_step_start,
    output logic                                     o_fc_step_ready,
    input  logic [T_BITS-1:0]                        i_fc_timestep,
    input  logic                                     i_fc_row_valid,
    output logic                                     o_fc_row_ready,
    input  logic [SOURCE_LANES-1:0]                  i_fc_lane_valid,
    input  logic [SOURCE_LANES*H_COUNT*W_BITS-1:0]   i_fc_weight_rows_flat,
    input  logic                                     i_fc_step_close,
    input  logic                                     i_fc_result_ready,
    output logic                                     o_fc_result_start,
    output logic                                     o_fc_done,
    output logic [T_BITS-1:0]                        o_fc_completed_timestep,
    output logic [31:0]                              o_fc_completed_events,
    output logic [H_COUNT-1:0]                       o_fc_spikes,
    output logic [15:0]                              o_fc_spike_count,
    output logic [H_COUNT*FC_ACC_BITS-1:0]           o_fc_vmem_flat,
    output logic [H_COUNT*16-1:0]                    o_fc_trace_flat,
    output logic                                     o_fc_producer_active,
    output logic                                     o_fc_consumer_active,
    output logic [31:0]                              o_fc_overlap_cycles,

    input  logic                                     i_srnn_clear_spikes,
    input  logic                                     i_srnn_clear_word_valid,
    input  logic [9:0]                               i_srnn_clear_word_base,
    input  logic                                     i_srnn_accumulate_valid,
    input  logic [9:0]                               i_srnn_accumulate_base,
    input  logic [SOURCE_LANES-1:0]                  i_srnn_accumulate_lane_mask,
    input  logic [SOURCE_LANES*32-1:0]               i_srnn_accumulate_contribution_flat,
    input  logic [9:0]                               i_srnn_active_neurons,
    input  logic                                     i_srnn_lif_begin,
    input  logic                                     i_srnn_lif_word_valid,
    input  logic [9:0]                               i_srnn_lif_word_base,
    input  logic signed [15:0]                       i_srnn_leak_q15,
    input  logic signed [31:0]                       i_srnn_threshold,
    input  logic [4:0]                               i_srnn_current_left_shift,
    input  logic                                     i_srnn_reset_to_zero,
    input  logic                                     i_srnn_commit_timestep,

    output logic [H_COUNT-1:0]                       o_srnn_previous_spikes,
    output logic [H_COUNT-1:0]                       o_srnn_current_spikes,
    output logic [H_COUNT*32-1:0]                    o_srnn_accumulator_flat,
    output logic [H_COUNT*32-1:0]                    o_srnn_membrane_flat,
    output logic                                     o_protocol_error
);
    localparam integer WORD_COUNT = H_COUNT / SOURCE_LANES;
    localparam integer WORD_INDEX_WIDTH =
        (WORD_COUNT <= 1) ? 1 : $clog2(WORD_COUNT);

    logic [H_COUNT*FC_ACC_BITS-1:0] fc_delta_flat;
    logic [H_COUNT*FC_ACC_BITS-1:0] fc_consume_accumulator_flat;
    logic [T_BITS-1:0] fc_consume_timestep;
    logic fc_accumulator_step_ready;
    logic fc_accumulator_consume_valid;
    logic fc_accumulator_consume_start;
    logic fc_accumulator_consume_done;
    logic fc_write_bank;
    logic fc_consume_bank;
    logic fc_accumulator_protocol_error;
    logic fc_row_accept;

    logic fc_lif_active_q;
    logic [WORD_INDEX_WIDTH-1:0] fc_lif_word_q;
    logic fc_done_q;
    logic [31:0] fc_write_event_count_q;
    logic [31:0] fc_event_count_by_bank_q [0:1];
    logic [T_BITS-1:0] fc_active_timestep_q;
    logic [31:0] fc_active_event_count_q;
    logic local_protocol_error_q;

    logic [H_COUNT-1:0] shared_previous_spikes;
    logic [H_COUNT-1:0] shared_current_spikes;
    logic [H_COUNT*32-1:0] shared_accumulator_flat;
    logic [H_COUNT*32-1:0] shared_membrane_flat;
    logic [H_COUNT*16-1:0] shared_trace_flat;

    function automatic logic [31:0] popcount_lanes(
        input logic [SOURCE_LANES-1:0] value
    );
        logic [31:0] count;
        begin
            count = 32'd0;
            for (int lane = 0; lane < SOURCE_LANES; lane++)
                count = count + value[lane];
            popcount_lanes = count;
        end
    endfunction

    function automatic logic [15:0] popcount_neurons(
        input logic [H_COUNT-1:0] value
    );
        logic [15:0] count;
        begin
            count = 16'd0;
            for (int neuron = 0; neuron < H_COUNT; neuron++)
                count = count + value[neuron];
            popcount_neurons = count;
        end
    endfunction

    unified_h40_fc_compat_w0_delta #(
        .H_COUNT(H_COUNT),
        .LANES(SOURCE_LANES),
        .W_BITS(W_BITS),
        .ACC_BITS(FC_ACC_BITS),
        .INPUT_SHIFT(INPUT_SHIFT)
    ) u_fc_w0_delta (
        .i_lane_valid(i_fc_lane_valid),
        .i_weight_rows_flat(i_fc_weight_rows_flat),
        .o_delta_flat(fc_delta_flat)
    );

    assign o_fc_step_ready = i_mode_fc && fc_accumulator_step_ready;
    assign o_fc_row_ready = i_mode_fc && o_fc_producer_active;
    assign fc_row_accept = i_fc_row_valid && o_fc_row_ready;
    assign fc_accumulator_consume_start = i_mode_fc &&
        fc_accumulator_consume_valid && !fc_lif_active_q &&
        i_fc_result_ready;
    assign fc_accumulator_consume_done = fc_lif_active_q &&
        fc_lif_word_q == WORD_INDEX_WIDTH'(WORD_COUNT-1);
    assign o_fc_result_start = fc_accumulator_consume_start;
    assign o_fc_done = fc_done_q;
    assign o_fc_completed_timestep = fc_active_timestep_q;
    assign o_fc_completed_events = fc_active_event_count_q;
    assign o_fc_spikes = shared_current_spikes;
    assign o_fc_spike_count = popcount_neurons(shared_current_spikes);
    assign o_fc_trace_flat = shared_trace_flat;

    always_comb begin
        o_fc_vmem_flat = '0;
        for (int neuron = 0; neuron < H_COUNT; neuron++) begin
            o_fc_vmem_flat[neuron*FC_ACC_BITS +: FC_ACC_BITS] =
                shared_membrane_flat[neuron*32 +: FC_ACC_BITS];
        end
    end

    newarch_n8d_dual_accumulator_bank #(
        .H_COUNT(H_COUNT),
        .ACC_BITS(FC_ACC_BITS),
        .T_BITS(T_BITS),
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(0)
    ) u_fc_accumulator_banks (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(i_abort || i_fc_clear_state),
        .i_step_start(i_fc_step_start && i_mode_fc),
        .o_step_ready(fc_accumulator_step_ready),
        .i_timestep(i_fc_timestep),
        .i_accumulate_valid(fc_row_accept && (i_fc_lane_valid != '0)),
        .i_delta_flat(fc_delta_flat),
        .i_step_close(i_fc_step_close && i_mode_fc),
        .o_consume_valid(fc_accumulator_consume_valid),
        .i_consume_start(fc_accumulator_consume_start),
        .i_consume_done(fc_accumulator_consume_done),
        .o_consume_timestep(fc_consume_timestep),
        .o_consume_acc_flat(fc_consume_accumulator_flat),
        .o_write_bank(fc_write_bank),
        .o_consume_bank(fc_consume_bank),
        .o_producer_active(o_fc_producer_active),
        .o_consumer_active(o_fc_consumer_active),
        .o_overlap_cycles(o_fc_overlap_cycles),
        .o_protocol_error(fc_accumulator_protocol_error)
    );

    unified_h40_fc_srnn_state_candidate_v1 #(
        .TILE_BASE(TILE_BASE),
        .TILE_NEURONS(H_COUNT),
        .LANES(SOURCE_LANES),
        .FC_ACC_BITS(FC_ACC_BITS),
        .TRACE_BITS(16)
    ) u_shared_state (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_clear_spikes(!i_mode_fc && i_srnn_clear_spikes),
        .i_mode_fc(i_mode_fc),
        .i_clear_all(i_mode_fc && i_fc_clear_state),
        .i_fc_accumulator_flat(fc_consume_accumulator_flat),
        .i_clear_word_valid(!i_mode_fc && i_srnn_clear_word_valid),
        .i_clear_word_base(i_srnn_clear_word_base),
        .i_accumulate_valid(!i_mode_fc && i_srnn_accumulate_valid),
        .i_accumulate_base(i_srnn_accumulate_base),
        .i_accumulate_lane_mask(i_srnn_accumulate_lane_mask),
        .i_accumulate_contribution_flat(
            i_srnn_accumulate_contribution_flat),
        .i_active_neurons(i_mode_fc ? 10'(TILE_BASE + H_COUNT) :
            i_srnn_active_neurons),
        .i_lif_begin(i_mode_fc ? fc_accumulator_consume_start :
            i_srnn_lif_begin),
        .i_lif_word_valid(i_mode_fc ? fc_lif_active_q :
            i_srnn_lif_word_valid),
        .i_lif_word_base(i_mode_fc ?
            10'(TILE_BASE + fc_lif_word_q*SOURCE_LANES) :
            i_srnn_lif_word_base),
        .i_leak_q15(i_mode_fc ? 16'(LEAK_M_Q15) : i_srnn_leak_q15),
        .i_threshold(i_mode_fc ? 32'(VTH) : i_srnn_threshold),
        .i_current_left_shift(i_mode_fc ? 5'd0 :
            i_srnn_current_left_shift),
        .i_reset_to_zero(i_mode_fc ? 1'b0 : i_srnn_reset_to_zero),
        .i_trace_decay_q16(i_mode_fc ? 16'(TRACE_DECAY_Q16) : 16'd0),
        .i_commit_timestep(!i_mode_fc && i_srnn_commit_timestep),
        .o_previous_spikes(shared_previous_spikes),
        .o_current_spikes(shared_current_spikes),
        .o_accumulator_flat(shared_accumulator_flat),
        .o_membrane_flat(shared_membrane_flat),
        .o_trace_flat(shared_trace_flat)
    );

    assign o_srnn_previous_spikes = shared_previous_spikes;
    assign o_srnn_current_spikes = shared_current_spikes;
    assign o_srnn_accumulator_flat = shared_accumulator_flat;
    assign o_srnn_membrane_flat = shared_membrane_flat;
    assign o_protocol_error = fc_accumulator_protocol_error ||
        local_protocol_error_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc_lif_active_q <= 1'b0;
            fc_lif_word_q <= '0;
            fc_done_q <= 1'b0;
            fc_write_event_count_q <= 32'd0;
            fc_event_count_by_bank_q[0] <= 32'd0;
            fc_event_count_by_bank_q[1] <= 32'd0;
            fc_active_timestep_q <= '0;
            fc_active_event_count_q <= 32'd0;
            local_protocol_error_q <= 1'b0;
        end else if (i_abort || i_fc_clear_state) begin
            fc_lif_active_q <= 1'b0;
            fc_lif_word_q <= '0;
            fc_done_q <= 1'b0;
            fc_write_event_count_q <= 32'd0;
            fc_event_count_by_bank_q[0] <= 32'd0;
            fc_event_count_by_bank_q[1] <= 32'd0;
            fc_active_timestep_q <= '0;
            fc_active_event_count_q <= 32'd0;
            local_protocol_error_q <= 1'b0;
        end else begin
            fc_done_q <= 1'b0;

            if (i_fc_step_start && o_fc_step_ready)
                fc_write_event_count_q <= 32'd0;

            if (fc_row_accept) begin
                fc_write_event_count_q <= fc_write_event_count_q +
                    popcount_lanes(i_fc_lane_valid);
            end else if (i_fc_row_valid && i_mode_fc) begin
                local_protocol_error_q <= 1'b1;
            end

            if (fc_row_accept && i_fc_lane_valid == '0)
                local_protocol_error_q <= 1'b1;

            if (i_fc_step_close && o_fc_producer_active && i_mode_fc) begin
                fc_event_count_by_bank_q[fc_write_bank] <=
                    fc_write_event_count_q +
                    (fc_row_accept ? popcount_lanes(i_fc_lane_valid) : 32'd0);
            end

            if (fc_accumulator_consume_start) begin
                fc_lif_active_q <= 1'b1;
                fc_lif_word_q <= '0;
                fc_active_timestep_q <= fc_consume_timestep;
                fc_active_event_count_q <=
                    fc_event_count_by_bank_q[fc_consume_bank];
            end else if (fc_lif_active_q) begin
                if (fc_lif_word_q == WORD_INDEX_WIDTH'(WORD_COUNT-1)) begin
                    fc_lif_active_q <= 1'b0;
                    fc_done_q <= 1'b1;
                end else begin
                    fc_lif_word_q <= fc_lif_word_q + 1'b1;
                end
            end
        end
    end

    initial begin
        if (H_COUNT != 40 || SOURCE_LANES != 8)
            $error("shared state engine candidate requires H40 with eight lanes");
        if (TILE_BASE % H_COUNT != 0)
            $error("shared state engine candidate TILE_BASE must align to H40");
    end
endmodule

`default_nettype wire
