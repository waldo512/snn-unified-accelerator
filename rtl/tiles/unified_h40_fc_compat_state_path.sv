`timescale 1ns/1ps
`default_nettype none

// Runtime-row FC_COMPAT state path for one H40 tile. Weight storage and source
// scheduling stay outside this module. The state path preserves the canonical
// INT5/shift6/ACC18 arithmetic, dual-timestep overlap, folded exact LIF, and
// trace observation needed by the later TP learning sideband.
module unified_h40_fc_compat_state_path #(
    parameter integer H_COUNT = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer LIF_LANES = 20,
    parameter integer W_BITS = 5,
    parameter integer ACC_BITS = 18,
    parameter integer INPUT_SHIFT = 6,
    parameter integer T_BITS = 9,
    parameter integer LEAK_M_Q15 = 32113,
    parameter integer TRACE_DECAY_Q16 = 64225,
    parameter integer TRACE_FRAC_BITS = 10,
    parameter integer VTH = 1024
) (
    input  logic                                     clk,
    input  logic                                     rst_n,
    input  logic                                     i_clear_state,

    input  logic                                     i_step_start,
    output logic                                     o_step_ready,
    input  logic [T_BITS-1:0]                        i_timestep,
    input  logic                                     i_row_valid,
    output logic                                     o_row_ready,
    input  logic [SOURCE_LANES-1:0]                  i_lane_valid,
    input  logic [SOURCE_LANES*H_COUNT*W_BITS-1:0]   i_weight_rows_flat,
    input  logic                                     i_step_close,

    input  logic                                     i_result_ready,
    output logic                                     o_result_start,
    output logic                                     o_done,
    output logic [T_BITS-1:0]                        o_completed_timestep,
    output logic [31:0]                              o_completed_events,
    output logic [H_COUNT-1:0]                       o_spikes,
    output logic [15:0]                              o_spike_count,
    output logic [H_COUNT*ACC_BITS-1:0]              o_vmem_flat,
    output logic [H_COUNT*16-1:0]                    o_trace_flat,
    output logic                                     o_producer_active,
    output logic                                     o_consumer_active,
    output logic [31:0]                              o_overlap_cycles,
    output logic                                     o_protocol_error
);
    logic [H_COUNT*ACC_BITS-1:0] delta_flat;
    logic [H_COUNT*ACC_BITS-1:0] accumulator_flat;
    logic [T_BITS-1:0] accumulator_timestep;
    logic accumulator_valid;
    logic accumulator_start;
    logic accumulator_done;
    logic write_bank;
    logic consume_bank;
    logic accumulator_protocol_error;
    logic lif_ready;
    logic lif_done;
    logic lif_protocol_error;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [15:0] lif_cycles;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [31:0] write_event_count;
    logic [31:0] event_count_by_bank [0:1];
    logic [T_BITS-1:0] active_timestep;
    logic [31:0] active_event_count;
    logic local_protocol_error;
    logic row_accept;

    function automatic logic [31:0] popcount_lanes(
        input logic [SOURCE_LANES-1:0] value
    );
        logic [31:0] count;
        integer lane_i;
        begin
            count = 32'd0;
            for (lane_i = 0; lane_i < SOURCE_LANES; lane_i = lane_i + 1)
                count = count + value[lane_i];
            popcount_lanes = count;
        end
    endfunction

    unified_h40_fc_compat_w0_delta #(
        .H_COUNT(H_COUNT),
        .LANES(SOURCE_LANES),
        .W_BITS(W_BITS),
        .ACC_BITS(ACC_BITS),
        .INPUT_SHIFT(INPUT_SHIFT)
    ) u_w0_delta (
        .i_lane_valid(i_lane_valid),
        .i_weight_rows_flat(i_weight_rows_flat),
        .o_delta_flat(delta_flat)
    );

    assign o_row_ready = o_producer_active;
    assign row_accept = i_row_valid && o_row_ready;
    assign accumulator_start = rst_n && !i_clear_state &&
        accumulator_valid && lif_ready && i_result_ready;
    assign accumulator_done = lif_done;
    assign o_result_start = accumulator_start;
    assign o_done = lif_done;
    assign o_completed_timestep = active_timestep;
    assign o_completed_events = active_event_count;
    assign o_protocol_error = local_protocol_error ||
        accumulator_protocol_error || lif_protocol_error;

    newarch_n8d_dual_accumulator_bank #(
        .H_COUNT(H_COUNT),
        .ACC_BITS(ACC_BITS),
        .T_BITS(T_BITS),
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(0)
    ) u_accumulator_banks (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(i_clear_state),
        .i_step_start(i_step_start),
        .o_step_ready(o_step_ready),
        .i_timestep(i_timestep),
        .i_accumulate_valid(row_accept && (i_lane_valid != '0)),
        .i_delta_flat(delta_flat),
        .i_step_close(i_step_close),
        .o_consume_valid(accumulator_valid),
        .i_consume_start(accumulator_start),
        .i_consume_done(accumulator_done),
        .o_consume_timestep(accumulator_timestep),
        .o_consume_acc_flat(accumulator_flat),
        .o_write_bank(write_bank),
        .o_consume_bank(consume_bank),
        .o_producer_active(o_producer_active),
        .o_consumer_active(o_consumer_active),
        .o_overlap_cycles(o_overlap_cycles),
        .o_protocol_error(accumulator_protocol_error)
    );

    newarch_lif_trace_cluster_folded #(
        .H_COUNT(H_COUNT),
        .LANES(LIF_LANES),
        .ACC_BITS(ACC_BITS),
        .STATE_FRAC_BITS(10),
        .TRACE_BITS(16),
        .TRACE_FRAC_BITS(TRACE_FRAC_BITS),
        .N8_CSD_MODE(0),
        .ENABLE_SURROGATE(1),
        .LEAK_M_Q15_PARAM(LEAK_M_Q15),
        .TRACE_DECAY_Q16_PARAM(TRACE_DECAY_Q16),
        .VTH_PARAM(VTH)
    ) u_lif_trace (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear_state(i_clear_state),
        .i_start(accumulator_start),
        .o_ready(lif_ready),
        .i_acc_flat(accumulator_flat),
        .o_done(lif_done),
        .o_cycles(lif_cycles),
        .o_spikes(o_spikes),
        .o_spike_count(o_spike_count),
        .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_protocol_error(lif_protocol_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_event_count <= 32'd0;
            event_count_by_bank[0] <= 32'd0;
            event_count_by_bank[1] <= 32'd0;
            active_timestep <= '0;
            active_event_count <= 32'd0;
            local_protocol_error <= 1'b0;
        end else if (i_clear_state) begin
            write_event_count <= 32'd0;
            event_count_by_bank[0] <= 32'd0;
            event_count_by_bank[1] <= 32'd0;
            active_timestep <= '0;
            active_event_count <= 32'd0;
            local_protocol_error <= 1'b0;
        end else begin
            if (i_step_start && o_step_ready)
                write_event_count <= 32'd0;

            if (row_accept)
                write_event_count <= write_event_count + popcount_lanes(i_lane_valid);
            else if (i_row_valid)
                local_protocol_error <= 1'b1;

            if (row_accept && i_lane_valid == '0)
                local_protocol_error <= 1'b1;

            if (i_step_close && o_producer_active) begin
                event_count_by_bank[write_bank] <= write_event_count +
                    (row_accept ? popcount_lanes(i_lane_valid) : 32'd0);
            end

            if (accumulator_start) begin
                active_timestep <= accumulator_timestep;
                active_event_count <= event_count_by_bank[consume_bank];
            end
        end
    end

    initial begin
        if (SOURCE_LANES != 8)
            $error("FC_COMPAT state path currently requires eight source lanes");
        if (H_COUNT % LIF_LANES != 0)
            $error("FC_COMPAT H_COUNT must be divisible by LIF_LANES");
    end
endmodule

`default_nettype wire
