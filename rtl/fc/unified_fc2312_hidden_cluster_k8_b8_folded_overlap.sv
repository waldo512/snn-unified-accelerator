`timescale 1ns/1ps

// Single-cluster N8D2B integration: real K8 W0 dual accumulators feed a
// folded exact LIF engine while W2 keeps the frozen H40 partial-logit contract.
module unified_fc2312_hidden_cluster_k8_b8_folded_overlap #(
    parameter integer IN_DIM = 4096,
    parameter integer PHYSICAL_BANK_DEPTH = 512,
    parameter integer H_COUNT = 40,
    parameter integer LIF_LANES = 20,
    parameter integer FI_BITS = 12,
    parameter integer ACC_BITS = 18,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer O_DIM = 10,
    parameter integer W0_BITS = 5,
    parameter integer W0_INPUT_SHIFT = 6,
    parameter integer W0_EVENT_X_IS_ONE = 1,
    parameter integer W0_EVENT_X_IS_SIGNED = 0,
    parameter integer USE_ATOMIC_BOUNDARY_CLOSE = 0,
    parameter integer USE_SAME_CYCLE_CONSUMER_RECYCLE = 0,
    parameter integer USE_LIF_ACC_SNAPSHOT = 0,
    parameter integer T_BITS = 9,
    parameter integer ENABLE_FIXED_W2 = 1,
    parameter integer LIF_LEAK_M_Q15 = 32113,
    parameter integer LIF_TRACE_DECAY_Q16 = 64225,
    parameter integer LIF_TRACE_FRAC_BITS = 10,
    parameter integer LIF_VTH = 1024,
    parameter BANK_FILE_0 = "",
    parameter BANK_FILE_1 = "",
    parameter BANK_FILE_2 = "",
    parameter BANK_FILE_3 = "",
    parameter BANK_FILE_4 = "",
    parameter BANK_FILE_5 = "",
    parameter BANK_FILE_6 = "",
    parameter BANK_FILE_7 = "",
    parameter W2_FILE = ""
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          i_clear_state,
    input  wire                          i_step_start,
    output wire                          o_step_ready,
    input  wire                          i_result_ready,
    output wire                          o_result_start,
    input  wire [T_BITS-1:0]             i_timestep,
    input  wire                          i_group_valid,
    output wire                          o_group_ready,
    input  wire [7:0]                    i_bank_valid,
    input  wire [8*FI_BITS-1:0]          i_fi_flat,
    input  wire [63:0]                   i_x_flat,
    input  wire                          i_groups_done,
    output wire                          o_done,
    output wire [T_BITS-1:0]             o_completed_timestep,
    output wire [31:0]                   o_completed_events,
    output wire [H_COUNT-1:0]            o_completed_spikes,
    output wire [15:0]                   o_completed_spike_count,
    output wire [O_DIM*OUT_ACC_BITS-1:0] o_partial_logits_flat,
    output wire [H_COUNT*ACC_BITS-1:0]   o_vmem_flat,
    output wire [H_COUNT*16-1:0]         o_trace_flat,
    output wire [31:0]                   o_w0_lif_overlap_cycles,
    output wire                          o_w0_producer_active,
    output wire                          o_w0_consumer_active,
    output wire                          o_protocol_error
);
    wire acc_valid;
    wire lif_ready;
    wire lif_done;
    wire lif_protocol_error;
    wire w0_protocol_error;
    wire [T_BITS-1:0] acc_timestep;
    wire [31:0] acc_events;
    wire [H_COUNT*ACC_BITS-1:0] acc_flat;
    wire [H_COUNT-1:0] lif_spikes;
    wire [15:0] lif_spike_count;
    wire [15:0] lif_cycles;
    wire producer_active;
    wire consumer_active;
    wire acc_start = rst_n && !i_clear_state && acc_valid && lif_ready &&
                     i_result_ready;
    wire acc_consume_done;
    wire [H_COUNT*ACC_BITS-1:0] lif_acc_flat;
    wire snapshot_protocol_error;
    wire w2_done;

    reg [T_BITS-1:0] active_timestep;
    reg [31:0] active_events;
    reg [T_BITS-1:0] timestep_pipe [0:2];
    reg [31:0] events_pipe [0:2];
    reg [H_COUNT-1:0] spikes_pipe [0:2];
    reg [2:0] metadata_valid;
    reg metadata_protocol_error;

    unified_fc2312_w0_cluster_k8_b8_dual_acc #(
        .IN_DIM(IN_DIM), .PHYSICAL_BANK_DEPTH(PHYSICAL_BANK_DEPTH),
        .H_COUNT(H_COUNT), .FI_BITS(FI_BITS),
        .W_BITS(W0_BITS), .ACC_BITS(ACC_BITS),
        .INPUT_SHIFT(W0_INPUT_SHIFT),
        .EVENT_X_IS_ONE(W0_EVENT_X_IS_ONE),
        .EVENT_X_IS_SIGNED(W0_EVENT_X_IS_SIGNED),
        .USE_ATOMIC_BOUNDARY_CLOSE(USE_ATOMIC_BOUNDARY_CLOSE),
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(USE_SAME_CYCLE_CONSUMER_RECYCLE),
        .T_BITS(T_BITS),
        .BANK_FILE_0(BANK_FILE_0), .BANK_FILE_1(BANK_FILE_1),
        .BANK_FILE_2(BANK_FILE_2), .BANK_FILE_3(BANK_FILE_3),
        .BANK_FILE_4(BANK_FILE_4), .BANK_FILE_5(BANK_FILE_5),
        .BANK_FILE_6(BANK_FILE_6), .BANK_FILE_7(BANK_FILE_7)
    ) u_w0 (
        .clk(clk), .rst_n(rst_n), .i_clear(i_clear_state),
        .i_step_start(i_step_start), .o_step_ready(o_step_ready),
        .i_timestep(i_timestep),
        .i_group_valid(i_group_valid), .o_group_ready(o_group_ready),
        .i_bank_valid(i_bank_valid), .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat), .i_groups_done(i_groups_done),
        .o_acc_valid(acc_valid), .i_acc_start(acc_start),
        .i_acc_done(acc_consume_done), .o_acc_timestep(acc_timestep),
        .o_acc_flat(acc_flat), .o_acc_accepted_events(acc_events),
        .o_producer_active(producer_active),
        .o_consumer_active(consumer_active),
        .o_overlap_cycles(o_w0_lif_overlap_cycles),
        .o_protocol_error(w0_protocol_error)
    );

    assign o_w0_producer_active = producer_active;
    assign o_w0_consumer_active = consumer_active;

    generate
        if (USE_LIF_ACC_SNAPSHOT != 0) begin : gen_lif_acc_snapshot
            wire snapshot_active;
            newarch_lif_accumulator_snapshot #(
                .H_COUNT(H_COUNT), .ACC_BITS(ACC_BITS)
            ) u_snapshot (
                .clk(clk), .rst_n(rst_n), .i_clear(i_clear_state),
                .i_capture(acc_start), .i_lif_done(lif_done),
                .i_acc_flat(acc_flat), .o_acc_flat(lif_acc_flat),
                .o_consume_release(acc_consume_done),
                .o_active(snapshot_active),
                .o_protocol_error(snapshot_protocol_error)
            );
        end else begin : gen_live_accumulator_read
            assign lif_acc_flat = acc_flat;
            assign acc_consume_done = lif_done;
            assign snapshot_protocol_error = 1'b0;
        end
    endgenerate

    newarch_lif_trace_cluster_folded #(
        .H_COUNT(H_COUNT), .LANES(LIF_LANES), .ACC_BITS(ACC_BITS),
        .STATE_FRAC_BITS(10), .TRACE_BITS(16),
        .TRACE_FRAC_BITS(LIF_TRACE_FRAC_BITS),
        .N8_CSD_MODE(0), .ENABLE_SURROGATE(1),
        .LEAK_M_Q15_PARAM(LIF_LEAK_M_Q15),
        .TRACE_DECAY_Q16_PARAM(LIF_TRACE_DECAY_Q16),
        .VTH_PARAM(LIF_VTH)
    ) u_hidden_state (
        .clk(clk), .rst_n(rst_n), .i_clear_state(i_clear_state),
        .i_start(acc_start), .o_ready(lif_ready), .i_acc_flat(lif_acc_flat),
        .o_done(lif_done), .o_cycles(lif_cycles), .o_spikes(lif_spikes),
        .o_spike_count(lif_spike_count), .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_protocol_error(lif_protocol_error)
    );

    generate
        if (ENABLE_FIXED_W2 != 0) begin : gen_fixed_w2
            newarch_w2_partial_cluster #(
                .H_COUNT(H_COUNT), .O_DIM(O_DIM), .W_BITS(8),
                .ACC_BITS(OUT_ACC_BITS), .W2_FILE(W2_FILE)
            ) u_w2 (
                .clk(clk), .rst_n(rst_n), .i_start(lif_done),
                .i_spikes(lif_spikes), .o_done(w2_done),
                .o_partial_flat(o_partial_logits_flat)
            );
            assign o_completed_timestep = timestep_pipe[2];
            assign o_completed_events = events_pipe[2];
            assign o_completed_spikes = spikes_pipe[2];
            assign o_completed_spike_count = 0;
        end else begin : gen_external_w2
            assign w2_done = lif_done;
            assign o_partial_logits_flat = 0;
            assign o_completed_timestep = active_timestep;
            assign o_completed_events = active_events;
            assign o_completed_spikes = lif_spikes;
            assign o_completed_spike_count = lif_spike_count;
        end
    endgenerate

    assign o_done = w2_done;
    assign o_result_start = acc_start;
    assign o_protocol_error = w0_protocol_error || lif_protocol_error ||
                              snapshot_protocol_error || metadata_protocol_error;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_timestep <= 0;
            active_events <= 0;
            timestep_pipe[0] <= 0;
            timestep_pipe[1] <= 0;
            timestep_pipe[2] <= 0;
            events_pipe[0] <= 0;
            events_pipe[1] <= 0;
            events_pipe[2] <= 0;
            spikes_pipe[0] <= 0;
            spikes_pipe[1] <= 0;
            spikes_pipe[2] <= 0;
            metadata_valid <= 0;
            metadata_protocol_error <= 1'b0;
        end else if (i_clear_state) begin
            active_timestep <= 0;
            active_events <= 0;
            timestep_pipe[0] <= 0;
            timestep_pipe[1] <= 0;
            timestep_pipe[2] <= 0;
            events_pipe[0] <= 0;
            events_pipe[1] <= 0;
            events_pipe[2] <= 0;
            spikes_pipe[0] <= 0;
            spikes_pipe[1] <= 0;
            spikes_pipe[2] <= 0;
            metadata_valid <= 0;
            metadata_protocol_error <= 1'b0;
        end else begin
            metadata_valid <= {metadata_valid[1:0], lif_done};
            timestep_pipe[2] <= timestep_pipe[1];
            timestep_pipe[1] <= timestep_pipe[0];
            events_pipe[2] <= events_pipe[1];
            events_pipe[1] <= events_pipe[0];
            spikes_pipe[2] <= spikes_pipe[1];
            spikes_pipe[1] <= spikes_pipe[0];
            if (acc_start) begin
                active_timestep <= acc_timestep;
                active_events <= acc_events;
            end
            if (lif_done) begin
                timestep_pipe[0] <= active_timestep;
                events_pipe[0] <= active_events;
                spikes_pipe[0] <= lif_spikes;
            end
            if (ENABLE_FIXED_W2 != 0 && w2_done != metadata_valid[2])
                metadata_protocol_error <= 1'b1;
        end
    end
endmodule
