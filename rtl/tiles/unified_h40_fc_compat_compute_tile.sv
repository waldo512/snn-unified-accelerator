`timescale 1ns/1ps
`default_nettype none

// Complete runtime-weight FC_COMPAT H40 compute tile. One output buffer keeps
// result backpressure outside the state/LIF and W2 pipelines.
module unified_h40_fc_compat_compute_tile #(
    parameter integer H_COUNT = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer LIF_LANES = 20,
    parameter integer W0_BITS = 5,
    parameter integer ACC_BITS = 18,
    parameter integer W0_INPUT_SHIFT = 6,
    parameter integer O_DIM = 10,
    parameter integer W2_BITS = 8,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer T_BITS = 9
) (
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         i_clear_state,
    input  logic                                         i_step_start,
    output logic                                         o_step_ready,
    input  logic [T_BITS-1:0]                            i_timestep,
    input  logic                                         i_row_valid,
    output logic                                         o_row_ready,
    input  logic [SOURCE_LANES-1:0]                      i_lane_valid,
    input  logic [SOURCE_LANES*H_COUNT*W0_BITS-1:0]      i_w0_weight_rows_flat,
    input  logic                                         i_step_close,
    input  logic [H_COUNT*O_DIM*W2_BITS-1:0]             i_w2_weight_flat,
    output logic                                         o_result_valid,
    input  logic                                         i_result_ready,
    output logic [T_BITS-1:0]                            o_result_timestep,
    output logic [31:0]                                  o_result_events,
    output logic [H_COUNT-1:0]                           o_result_spikes,
    output logic [O_DIM*OUT_ACC_BITS-1:0]                o_partial_logits_flat,
    output logic [H_COUNT*ACC_BITS-1:0]                  o_vmem_flat,
    output logic [H_COUNT*16-1:0]                        o_trace_flat,
    output logic                                         o_busy,
    output logic                                         o_protocol_error
);
    logic state_result_ready;
    logic state_result_start;
    logic state_done;
    logic [T_BITS-1:0] state_timestep;
    logic [31:0] state_events;
    logic [H_COUNT-1:0] state_spikes;
    logic [15:0] state_spike_count;
    logic state_producer_active;
    logic state_consumer_active;
    logic [31:0] state_overlap_cycles;
    logic state_protocol_error;
    logic w2_done;
    logic [O_DIM*OUT_ACC_BITS-1:0] w2_partial_flat;
    logic w2_inflight_q;
    logic [T_BITS-1:0] pending_timestep_q;
    logic [31:0] pending_events_q;
    logic [H_COUNT-1:0] pending_spikes_q;
    logic local_protocol_error_q;

    assign state_result_ready = !w2_inflight_q &&
        (!o_result_valid || i_result_ready);
    assign o_busy = state_producer_active || state_consumer_active ||
        w2_inflight_q || o_result_valid;
    assign o_protocol_error = state_protocol_error || local_protocol_error_q;

    unified_h40_fc_compat_state_path #(
        .H_COUNT(H_COUNT),
        .SOURCE_LANES(SOURCE_LANES),
        .LIF_LANES(LIF_LANES),
        .W_BITS(W0_BITS),
        .ACC_BITS(ACC_BITS),
        .INPUT_SHIFT(W0_INPUT_SHIFT),
        .T_BITS(T_BITS)
    ) u_state_path (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear_state(i_clear_state),
        .i_step_start(i_step_start),
        .o_step_ready(o_step_ready),
        .i_timestep(i_timestep),
        .i_row_valid(i_row_valid),
        .o_row_ready(o_row_ready),
        .i_lane_valid(i_lane_valid),
        .i_weight_rows_flat(i_w0_weight_rows_flat),
        .i_step_close(i_step_close),
        .i_result_ready(state_result_ready),
        .o_result_start(state_result_start),
        .o_done(state_done),
        .o_completed_timestep(state_timestep),
        .o_completed_events(state_events),
        .o_spikes(state_spikes),
        .o_spike_count(state_spike_count),
        .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_producer_active(state_producer_active),
        .o_consumer_active(state_consumer_active),
        .o_overlap_cycles(state_overlap_cycles),
        .o_protocol_error(state_protocol_error)
    );

    unified_h40_fc_compat_w2_partial #(
        .H_COUNT(H_COUNT),
        .O_DIM(O_DIM),
        .W_BITS(W2_BITS),
        .ACC_BITS(OUT_ACC_BITS)
    ) u_w2 (
        .clk(clk),
        .rst_n(rst_n),
        .i_start(state_done),
        .i_spikes(state_spikes),
        .i_weight_flat(i_w2_weight_flat),
        .o_done(w2_done),
        .o_partial_flat(w2_partial_flat)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w2_inflight_q <= 1'b0;
            pending_timestep_q <= '0;
            pending_events_q <= 32'd0;
            pending_spikes_q <= '0;
            o_result_valid <= 1'b0;
            o_result_timestep <= '0;
            o_result_events <= 32'd0;
            o_result_spikes <= '0;
            o_partial_logits_flat <= '0;
            local_protocol_error_q <= 1'b0;
        end else if (i_clear_state) begin
            w2_inflight_q <= 1'b0;
            pending_timestep_q <= '0;
            pending_events_q <= 32'd0;
            pending_spikes_q <= '0;
            o_result_valid <= 1'b0;
            o_result_timestep <= '0;
            o_result_events <= 32'd0;
            o_result_spikes <= '0;
            o_partial_logits_flat <= '0;
            local_protocol_error_q <= 1'b0;
        end else begin
            if (o_result_valid && i_result_ready)
                o_result_valid <= 1'b0;

            if (state_done) begin
                if (w2_inflight_q)
                    local_protocol_error_q <= 1'b1;
                w2_inflight_q <= 1'b1;
                pending_timestep_q <= state_timestep;
                pending_events_q <= state_events;
                pending_spikes_q <= state_spikes;
            end

            // A clear can invalidate a W2 request already inside the fixed
            // three-stage datapath. Ignore that stale completion because the
            // resettable in-flight bit is the transaction authority.
            if (w2_done && w2_inflight_q) begin
                if (o_result_valid && !i_result_ready)
                    local_protocol_error_q <= 1'b1;
                w2_inflight_q <= 1'b0;
                o_result_valid <= 1'b1;
                o_result_timestep <= pending_timestep_q;
                o_result_events <= pending_events_q;
                o_result_spikes <= pending_spikes_q;
                o_partial_logits_flat <= w2_partial_flat;
            end
        end
    end

    // These are retained as explicit observability points for later array-level
    // overlap and learning-sideband integration.
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_state_result_start;
    logic [15:0] unused_state_spike_count;
    logic [31:0] unused_state_overlap_cycles;
    always_comb begin
        unused_state_result_start = state_result_start;
        unused_state_spike_count = state_spike_count;
        unused_state_overlap_cycles = state_overlap_cycles;
    end
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
