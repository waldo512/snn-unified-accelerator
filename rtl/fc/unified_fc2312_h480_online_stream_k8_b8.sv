`timescale 1ns/1ps

// FC-first overlay of the canonical online stream. The only structural change
// is selecting the local 12xH40 forward stream; online W2 and version control
// remain canonical.

// N7C1E integration top. The K8 W0/LIF path supplies hidden spikes and traces;
// the existing 12-shard versioned W2 plane owns logits, updates and commits.
module unified_fc2312_h480_online_stream_k8_b8 #(
    parameter integer SAMPLE_BITS = 16,
    parameter integer T_BITS = 9,
    parameter integer FI_BITS = 12,
    parameter integer IN_DIM = 2312,
    parameter integer CLUSTERS = 12,
    parameter integer H_PER_CLUSTER = 40,
    parameter integer LIF_LANES = 20,
    parameter integer ACC_BITS = 18,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer W0_BITS = 8,
    parameter integer W0_INPUT_SHIFT = 3,
    parameter integer BANK_DEPTH = 64,
    parameter integer ACCUM_SAMPLES = 32,
    parameter W0_WEIGHT_DIR =
        "data/weights/clusters_h480_c12_h40_hash8_n7c1b",
    parameter W2_WEIGHT_DIR =
        "data/weights/clusters_h480_c12_h40_b4_trained_seed16340",
    parameter integer USE_PACKED_SDP_W2 = 0,
    parameter integer USE_SPARSE_PACKED_SDP_W2 = 0,
    parameter integer USE_SPARSE_FAST_JOIN = 0,
    parameter integer USE_R18_DUAL_CONTEXT_W2 = 0,
    parameter integer USE_N8_FOLDED_OVERLAP = 0,
    parameter integer USE_N9E2_START_CUT_THROUGH = 0,
    parameter integer USE_ATOMIC_BOUNDARY_CLOSE = 0,
    parameter integer USE_SAME_CYCLE_CONSUMER_RECYCLE = 0,
    parameter integer USE_LIF_ACC_SNAPSHOT = 0,
    parameter integer USE_N9F_REPLICA_INGRESS = 0,
    parameter integer USE_N9H_BUCKETED_REPLICA_INGRESS = 0,
    parameter integer USE_N9R10_BUCKETED192_REPLICA_INGRESS = 0,
    parameter integer USE_N9R17_PIPELINED_REPLICA_INGRESS = 0,
    parameter integer USE_N9R19_DEPTH_SNAPSHOT_INGRESS = 0,
    parameter integer USE_N9R19B_PREFIX_BALANCE4 = 0,
    parameter integer N9R17_RESERVATION_MODE = 1,
    parameter integer USE_R12D_MULTIPUMP_W0 = 0,
    parameter integer W2_TIMESTEP_QUEUE_DEPTH = 2,
    parameter integer USE_PIPELINED_GRAD_SCAN = 0,
    parameter integer USE_SPARSE_BITMAP_DRAIN = 0,
    parameter integer USE_PARALLEL_ERROR_DIVIDERS = 0,
    parameter integer ENABLE_RUNTIME_PROFILE = 0,
    parameter integer DEBUG_CLUSTER_BITS = (CLUSTERS > 16) ? 5 : 4,
    parameter W2_PACKED_WEIGHT_DIR =
        "data/weights/n7d5_h480_packed_w2_rows",
    parameter N9F_REPLICA_W0_WEIGHT_DIR =
        "data/weights/n9f3_h800_replica64_w0",
    parameter N9H_REPLICA_W0_WEIGHT_DIR =
        "data/weights/n9h_h800_replica96_w0",
    parameter N9R10_REPLICA_W0_WEIGHT_DIR =
        "data/weights/n9r10_h800_replica192_w0"
) (
    input  wire                         clk,
    input  wire                         clk_fast,
    input  wire                         rst_n,
    input  wire                         i_learning_enable,
    input  wire                         i_sample_start,
    output wire                         o_sample_ready,
    input  wire                         i_clear_state,
    input  wire [SAMPLE_BITS-1:0]       i_start_sample_id,
    input  wire [12:0]                  i_runtime_active_input_dim,
    input  wire [CLUSTERS-1:0]          i_runtime_active_cluster_mask,
    input  wire                         i_window_valid,
    output wire                         o_window_ready,
    input  wire [7:0]                   i_record_valid,
    input  wire [SAMPLE_BITS-1:0]       i_window_sample_id,
    input  wire [T_BITS-1:0]            i_window_timestep,
    input  wire [7:0]                   i_boundary_mask,
    input  wire                         i_window_sample_last,
    input  wire [8*FI_BITS-1:0]         i_fi_flat,
    input  wire [63:0]                  i_x_flat,

    input  wire                         i_label_valid,
    output wire                         o_label_ready,
    input  wire [15:0]                  i_label_sample_id,
    input  wire [3:0]                   i_label,

    output wire                         o_timestep_done,
    output reg                          o_sample_done,
    output wire [T_BITS-1:0]            o_completed_timestep,
    output wire [10*OUT_ACC_BITS-1:0]   o_logits_flat,
    output wire [3:0]                   o_pred,
    output wire [31:0]                  o_sample_cycles,
    output wire [31:0]                  o_accepted_windows,
    output wire [31:0]                  o_accepted_events,
    output wire [31:0]                  o_issue_groups,
    output wire [31:0]                  o_boundary_count,
    output wire [31:0]                  o_ingress_stall_cycles,
    output wire [31:0]                  o_output_stall_cycles,
    output wire [31:0]                  o_runtime_dropped_events,
    output wire                         o_online_global_idle,
    output reg  [31:0]                  o_runtime_inactive_update_command_count,

    output wire                         o_active_bank,
    output wire [15:0]                  o_active_version,
    output wire                         o_inactive_bank,
    output wire                         o_update_window_ready,
    output wire                         o_commit_pending,
    output wire                         o_commit_pulse,
    output wire                         o_snapshot_busy,
    output wire [4:0]                   o_record_count,
    output wire [31:0]                  o_dispatched_records,
    output wire                         o_cadence_blocked,
    output reg  [31:0]                  o_update_command_count,
    output reg  [15:0]                  o_commit_count,

    input  wire [DEBUG_CLUSTER_BITS-1:0] i_debug_cluster,
    input  wire [6:0]                   i_debug_word_addr,
    output wire [39:0]                  o_debug_active_word,
    output wire [39:0]                  o_debug_bank0_word,
    output wire [39:0]                  o_debug_bank1_word,
    output wire                         o_protocol_error
);
    localparam integer HIDDEN = CLUSTERS * H_PER_CLUSTER;
    localparam [5:0] ACCUM_LAST = ACCUM_SAMPLES[5:0] - 1'b1;
    localparam integer N8_W2_QUEUE_COUNT_BITS =
        $clog2(W2_TIMESTEP_QUEUE_DEPTH + 1);

    reg sample_running;
    reg top_protocol_error;
    reg [15:0] current_sample_id;
    reg [15:0] sample_weight_version;
    reg sample_target_bank;
    reg [5:0] cadence_record_count;
    reg cadence_blocked;
    reg saw_commit;

    wire sample_start_fire = i_sample_start && o_sample_ready;
    wire forward_timestep_done;
    wire forward_sample_done;
    wire [T_BITS-1:0] forward_timestep;
    wire [HIDDEN*ACC_BITS-1:0] forward_acc;
    wire [HIDDEN-1:0] forward_spikes;
    wire [HIDDEN*ACC_BITS-1:0] forward_vmem;
    wire [HIDDEN*16-1:0] forward_trace;
    wire [15:0] forward_spike_count;
    wire [10*OUT_ACC_BITS-1:0] unused_fixed_logits;
    wire [3:0] unused_fixed_pred;
    wire [3:0] scheduler_pending_records;
    wire forward_event_count_mismatch;
    wire forward_protocol_error;
    wire forward_result_start;

    wire online_infer_done;
    wire online_infer_ready;
    wire online_infer_idle;
    wire [T_BITS-1:0] online_infer_timestep;
    wire [10*OUT_ACC_BITS-1:0] online_step_delta;
    wire online_init_done;
    wire locked_bank;
    wire [15:0] locked_version;
    wire [CLUSTERS-1:0] shard_update_done;
    wire [CLUSTERS-1:0] shard_cmd_fire;
    wire [CLUSTERS*32-1:0] shard_update_cycles_flat;
    wire [CLUSTERS*16-1:0] shard_snapshot_cycles_flat;
    wire online_protocol_error;

    wire n8_w2_queue_input_ready;
    wire n8_w2_queue_valid;
    wire n8_w2_queue_output_ready;
    wire [HIDDEN-1:0] n8_w2_queue_spikes;
    wire [T_BITS-1:0] n8_w2_queue_timestep;
    wire [N8_W2_QUEUE_COUNT_BITS-1:0] n8_w2_queue_occupancy;
    wire n8_w2_queue_error;
    reg n8_w2_infer_busy;
    reg [3:0] r18_w2_inflight_count;
    reg r18_w2_inflight_error;
    reg [N8_W2_QUEUE_COUNT_BITS-1:0] n8_w2_result_reservations;
    reg n8_w2_credit_error;
    wire w2_dispatch_ready = (USE_R18_DUAL_CONTEXT_W2 != 0) ?
        online_infer_ready : !n8_w2_infer_busy;
    wire n8_w2_dispatch = (USE_N8_FOLDED_OVERLAP != 0) ?
        (n8_w2_queue_valid && w2_dispatch_ready) :
        forward_timestep_done;
    wire [HIDDEN-1:0] w2_dispatch_spikes =
        (USE_N8_FOLDED_OVERLAP != 0) ? n8_w2_queue_spikes : forward_spikes;
    wire [T_BITS-1:0] w2_dispatch_timestep =
        (USE_N8_FOLDED_OVERLAP != 0) ?
        n8_w2_queue_timestep : forward_timestep;
    assign n8_w2_queue_output_ready =
        (USE_N8_FOLDED_OVERLAP != 0) && w2_dispatch_ready;
    wire [N8_W2_QUEUE_COUNT_BITS:0] n8_w2_credit_used =
        {1'b0, n8_w2_queue_occupancy} +
        {1'b0, n8_w2_result_reservations};
    wire n8_w2_result_credit_ready =
        (n8_w2_credit_used < W2_TIMESTEP_QUEUE_DEPTH) || n8_w2_dispatch;

    reg [8:0] pending_count;
    reg forward_finished;
    reg inference_idle_state;
    reg signed [OUT_ACC_BITS-1:0] online_acc [0:9];
    wire [9:0] pending_after = {1'b0, pending_count} +
        (forward_timestep_done ? 10'd1 : 10'd0) -
        (online_infer_done ? 10'd1 : 10'd0);

    wire dispatcher_record_ready;
    wire dispatcher_capture_busy;
    wire dispatcher_dispatch_active;
    wire dispatcher_error;
    wire training_record_valid = o_sample_done && i_learning_enable;
    wire training_record_fire = training_record_valid && dispatcher_record_ready;
    wire dispatcher_meta_valid;
    wire dispatcher_meta_ready;
    wire [15:0] dispatcher_meta_sample_id;
    wire [15:0] dispatcher_meta_weight_version;
    wire dispatcher_meta_target_bank;
    wire [3:0] dispatcher_meta_label;
    wire [10*OUT_ACC_BITS-1:0] dispatcher_meta_logits;
    wire [CLUSTERS-1:0] dispatcher_trace_valid;
    wire [CLUSTERS-1:0] dispatcher_trace_ready;
    wire [CLUSTERS*6-1:0] dispatcher_trace_idx_flat;
    wire [CLUSTERS*16-1:0] dispatcher_trace_q016_flat;
    wire [CLUSTERS-1:0] dispatcher_trace_last;
    wire [4:0] dispatcher_record_count;
    wire dispatcher_label_ready;

    wire frontend_error_valid;
    wire frontend_error_ready;
    wire [15:0] frontend_sample_id;
    wire [15:0] frontend_weight_version;
    wire frontend_target_bank;
    wire [10*18-1:0] frontend_error_flat;
    wire frontend_protocol_error;

    assign o_sample_ready = online_init_done && !sample_running &&
                            !cadence_blocked &&
                            (!i_learning_enable || (o_record_count < 5'd15));
    assign o_online_global_idle = online_init_done && !sample_running &&
        inference_idle_state && (pending_count == 0) &&
        (n8_w2_queue_occupancy == 0) &&
        (n8_w2_result_reservations == 0) &&
        ((USE_R18_DUAL_CONTEXT_W2 != 0) ?
            ((r18_w2_inflight_count == 0) && online_infer_idle) :
            !n8_w2_infer_busy) &&
        (dispatcher_record_count == 0) && !dispatcher_capture_busy &&
        !dispatcher_dispatch_active && !frontend_error_valid &&
        !cadence_blocked && o_update_window_ready &&
        !o_commit_pending && !o_snapshot_busy;
    assign o_label_ready = i_learning_enable && dispatcher_label_ready;
    assign o_record_count = dispatcher_record_count;
    assign o_cadence_blocked = cadence_blocked;
    assign o_timestep_done = online_infer_done;
    assign o_completed_timestep = online_infer_timestep;
    assign o_protocol_error = top_protocol_error || forward_protocol_error ||
        forward_event_count_mismatch || online_protocol_error ||
        dispatcher_error || frontend_protocol_error || n8_w2_queue_error ||
        n8_w2_credit_error || r18_w2_inflight_error ||
        ((USE_N8_FOLDED_OVERLAP != 0) && forward_timestep_done &&
         !n8_w2_queue_input_ready) || pending_after[9];

    newarch_w2_timestep_elastic_queue #(
        .HIDDEN(HIDDEN), .META_BITS(T_BITS),
        .DEPTH(W2_TIMESTEP_QUEUE_DEPTH)
    ) u_n8_w2_timestep_queue (
        .clk(clk), .rst_n(rst_n), .i_clear(sample_start_fire),
        .i_valid(forward_timestep_done && (USE_N8_FOLDED_OVERLAP != 0)),
        .o_ready(n8_w2_queue_input_ready),
        .i_spikes(forward_spikes), .i_meta(forward_timestep),
        .o_valid(n8_w2_queue_valid), .i_ready(n8_w2_queue_output_ready),
        .o_spikes(n8_w2_queue_spikes), .o_meta(n8_w2_queue_timestep),
        .o_occupancy(n8_w2_queue_occupancy),
        .o_protocol_error(n8_w2_queue_error)
    );

    genvar output_class;
    generate
        for (output_class = 0; output_class < 10;
             output_class = output_class + 1) begin : gen_logits
            assign o_logits_flat[output_class*OUT_ACC_BITS +: OUT_ACC_BITS] =
                online_acc[output_class];
        end
    endgenerate

    argmax10 u_argmax (
        .in0(online_acc[0]), .in1(online_acc[1]), .in2(online_acc[2]),
        .in3(online_acc[3]), .in4(online_acc[4]), .in5(online_acc[5]),
        .in6(online_acc[6]), .in7(online_acc[7]), .in8(online_acc[8]),
        .in9(online_acc[9]), .idx(o_pred)
    );

`ifdef N9R12D_MULTIPUMP_W0
    newarch_h800_stream_r12d_multipump #(
        .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .FI_BITS(FI_BITS),
        .IN_DIM(IN_DIM), .CLUSTERS(CLUSTERS),
        .H_PER_CLUSTER(H_PER_CLUSTER), .LIF_LANES(LIF_LANES),
        .ACC_BITS(ACC_BITS), .OUT_ACC_BITS(OUT_ACC_BITS),
        .W0_BITS(W0_BITS), .W0_INPUT_SHIFT(W0_INPUT_SHIFT),
        .BANK_DEPTH(BANK_DEPTH), .W0_WEIGHT_DIR(W0_WEIGHT_DIR)
    ) u_forward (
        .clk(clk), .clk_fast(clk_fast), .rst_n(rst_n),
        .i_sample_start(sample_start_fire), .i_clear_state(i_clear_state),
        .i_timestep_result_ready((USE_N8_FOLDED_OVERLAP != 0) ?
            n8_w2_result_credit_ready : 1'b1),
        .i_start_sample_id(i_start_sample_id),
        .i_runtime_active_input_dim(i_runtime_active_input_dim),
        .i_runtime_active_cluster_mask(i_runtime_active_cluster_mask),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid),
        .i_window_sample_id(i_window_sample_id),
        .i_window_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_window_sample_last(i_window_sample_last),
        .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
        .o_timestep_done(forward_timestep_done),
        .o_sample_done(forward_sample_done),
        .o_completed_timestep(forward_timestep),
        .o_acc_flat(forward_acc), .o_spikes(forward_spikes),
        .o_vmem_flat(forward_vmem), .o_trace_flat(forward_trace),
        .o_spike_count(forward_spike_count),
        .o_logits_flat(unused_fixed_logits), .o_pred(unused_fixed_pred),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups), .o_boundary_count(o_boundary_count),
        .o_ingress_stall_cycles(o_ingress_stall_cycles),
        .o_output_stall_cycles(o_output_stall_cycles),
        .o_scheduler_pending_records(scheduler_pending_records),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_result_start(forward_result_start),
        .o_sample_cycles(o_sample_cycles),
        .o_event_count_mismatch(forward_event_count_mismatch),
        .o_protocol_error(forward_protocol_error)
    );
`else
    unified_fc2312_h480_stream_k8_b8 #(
        .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .FI_BITS(FI_BITS),
        .IN_DIM(IN_DIM), .CLUSTERS(CLUSTERS),
        .H_PER_CLUSTER(H_PER_CLUSTER), .LIF_LANES(LIF_LANES),
        .ACC_BITS(ACC_BITS), .OUT_ACC_BITS(OUT_ACC_BITS),
        .W0_BITS(W0_BITS), .W0_INPUT_SHIFT(W0_INPUT_SHIFT),
        .BANK_DEPTH(BANK_DEPTH), .ENABLE_FIXED_W2(0),
        .PIPELINE_CONTROL_OVERLAP(1), .REDUCTION_OUTPUT_REGISTER(0),
        .USE_N8_FOLDED_OVERLAP(USE_N8_FOLDED_OVERLAP),
        .USE_N9E2_START_CUT_THROUGH(USE_N9E2_START_CUT_THROUGH),
        .USE_ATOMIC_BOUNDARY_CLOSE(USE_ATOMIC_BOUNDARY_CLOSE),
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(USE_SAME_CYCLE_CONSUMER_RECYCLE),
        .USE_LIF_ACC_SNAPSHOT(USE_LIF_ACC_SNAPSHOT),
        .USE_N9F_REPLICA_INGRESS(USE_N9F_REPLICA_INGRESS),
        .USE_N9H_BUCKETED_REPLICA_INGRESS(USE_N9H_BUCKETED_REPLICA_INGRESS),
        .USE_N9R10_BUCKETED192_REPLICA_INGRESS(USE_N9R10_BUCKETED192_REPLICA_INGRESS),
        .USE_N9R17_PIPELINED_REPLICA_INGRESS(USE_N9R17_PIPELINED_REPLICA_INGRESS),
        .USE_N9R19_DEPTH_SNAPSHOT_INGRESS(USE_N9R19_DEPTH_SNAPSHOT_INGRESS),
        .USE_N9R19B_PREFIX_BALANCE4(USE_N9R19B_PREFIX_BALANCE4),
        .N9R17_RESERVATION_MODE(N9R17_RESERVATION_MODE),
        .ENABLE_RUNTIME_PROFILE(ENABLE_RUNTIME_PROFILE),
        .W0_WEIGHT_DIR(W0_WEIGHT_DIR),
        .N9F_REPLICA_W0_WEIGHT_DIR(N9F_REPLICA_W0_WEIGHT_DIR),
        .N9H_REPLICA_W0_WEIGHT_DIR(N9H_REPLICA_W0_WEIGHT_DIR),
        .N9R10_REPLICA_W0_WEIGHT_DIR(N9R10_REPLICA_W0_WEIGHT_DIR),
        .W2_WEIGHT_DIR(W2_WEIGHT_DIR)
    ) u_forward (
        .clk(clk), .rst_n(rst_n), .i_sample_start(sample_start_fire),
        .i_clear_state(i_clear_state),
        .i_timestep_result_ready((USE_N8_FOLDED_OVERLAP != 0) ?
            n8_w2_result_credit_ready : 1'b1),
        .i_start_sample_id(i_start_sample_id),
        .i_runtime_active_input_dim(i_runtime_active_input_dim),
        .i_runtime_active_cluster_mask(i_runtime_active_cluster_mask),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid),
        .i_window_sample_id(i_window_sample_id),
        .i_window_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_window_sample_last(i_window_sample_last),
        .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
        .o_timestep_done(forward_timestep_done),
        .o_sample_done(forward_sample_done),
        .o_completed_timestep(forward_timestep),
        .o_acc_flat(forward_acc), .o_spikes(forward_spikes),
        .o_vmem_flat(forward_vmem), .o_trace_flat(forward_trace),
        .o_spike_count(forward_spike_count),
        .o_logits_flat(unused_fixed_logits), .o_pred(unused_fixed_pred),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups), .o_boundary_count(o_boundary_count),
        .o_ingress_stall_cycles(o_ingress_stall_cycles),
        .o_output_stall_cycles(o_output_stall_cycles),
        .o_scheduler_pending_records(scheduler_pending_records),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_result_start(forward_result_start),
        .o_sample_cycles(o_sample_cycles),
        .o_event_count_mismatch(forward_event_count_mismatch),
        .o_protocol_error(forward_protocol_error)
    );
`endif

    newarch_h400_w2_online_plane #(
        .CLUSTERS(CLUSTERS), .H_PER_CLUSTER(H_PER_CLUSTER),
        .META_BITS(T_BITS), .USE_FOLD4_LANEBANKED_W2(1),
        .USE_COMBINED_D8_W2((USE_PACKED_SDP_W2 == 0) ? 1 : 0),
        .USE_TRAINED_H440_W2(0),
        .USE_H480_ZERO_TAIL_W2(1),
        .USE_BRAM_D8_W2((USE_PACKED_SDP_W2 == 0) ? 1 : 0),
        .USE_PACKED_SDP_BRAM_D8_W2(USE_PACKED_SDP_W2),
        .USE_H800_SPARSE_PACKED_SDP_W2(USE_SPARSE_PACKED_SDP_W2),
        .USE_H800_SPARSE_FAST_JOIN(USE_SPARSE_FAST_JOIN),
        .USE_R18_DUAL_CONTEXT_W2(USE_R18_DUAL_CONTEXT_W2),
        .USE_PIPELINED_GRAD_SCAN(USE_PIPELINED_GRAD_SCAN),
        .USE_SPARSE_BITMAP_DRAIN(USE_SPARSE_BITMAP_DRAIN),
        .DEBUG_CLUSTER_BITS(DEBUG_CLUSTER_BITS),
        .H480_WEIGHT_DIR(W2_WEIGHT_DIR),
        .H480_PACKED_WEIGHT_DIR(W2_PACKED_WEIGHT_DIR)
    ) u_online_w2 (
        .clk(clk), .rst_n(rst_n),
        .i_infer_start(n8_w2_dispatch),
        .o_infer_ready(online_infer_ready),
        .o_infer_idle(online_infer_idle),
        .i_spikes_flat(w2_dispatch_spikes),
        .i_infer_meta(w2_dispatch_timestep),
        .o_infer_done(online_infer_done),
        .o_logits_delta_flat(online_step_delta),
        .o_infer_meta(online_infer_timestep),
        .i_sample_start(sample_start_fire),
        .i_sample_boundary(forward_sample_done),
        .i_inference_idle(inference_idle_state),
        .i_error_valid(frontend_error_valid),
        .o_error_ready(frontend_error_ready),
        .i_sample_id(frontend_sample_id),
        .i_weight_version(frontend_weight_version),
        .i_target_bank(frontend_target_bank),
        .i_error_q016_flat(frontend_error_flat),
        .i_trace_valid(dispatcher_trace_valid),
        .o_trace_ready(dispatcher_trace_ready),
        .i_trace_local_idx_flat(dispatcher_trace_idx_flat),
        .i_trace_q016_flat(dispatcher_trace_q016_flat),
        .i_trace_last(dispatcher_trace_last),
        .o_init_done(online_init_done), .o_active_bank(o_active_bank),
        .o_active_version(o_active_version), .o_inactive_bank(o_inactive_bank),
        .o_locked_bank(locked_bank), .o_locked_version(locked_version),
        .o_update_window_ready(o_update_window_ready),
        .o_commit_pending(o_commit_pending), .o_commit_pulse(o_commit_pulse),
        .o_snapshot_busy(o_snapshot_busy),
        .o_shard_update_done(shard_update_done),
        .o_shard_cmd_fire(shard_cmd_fire),
        .o_shard_update_cycles_flat(shard_update_cycles_flat),
        .o_shard_snapshot_cycles_flat(shard_snapshot_cycles_flat),
        .o_protocol_error(online_protocol_error),
        .i_debug_cluster(i_debug_cluster),
        .i_debug_word_addr(i_debug_word_addr),
        .o_debug_active_word(o_debug_active_word),
        .o_debug_bank0_word(o_debug_bank0_word),
        .o_debug_bank1_word(o_debug_bank1_word)
    );

    newarch_h400_record_dispatcher #(
        .CLUSTERS(CLUSTERS), .H_PER_CLUSTER(H_PER_CLUSTER)
    ) u_dispatcher (
        .clk(clk), .rst_n(rst_n),
        .i_record_valid(training_record_valid),
        .o_record_ready(dispatcher_record_ready),
        .i_record_sample_id(current_sample_id),
        .i_record_weight_version(sample_weight_version),
        .i_record_target_bank(sample_target_bank),
        .i_record_logits_flat(o_logits_flat),
        .i_record_trace_flat(forward_trace),
        .i_label_valid(i_label_valid && i_learning_enable),
        .o_label_ready(dispatcher_label_ready),
        .i_label_sample_id(i_label_sample_id), .i_label(i_label),
        .o_meta_valid(dispatcher_meta_valid),
        .i_meta_ready(dispatcher_meta_ready),
        .o_meta_sample_id(dispatcher_meta_sample_id),
        .o_meta_weight_version(dispatcher_meta_weight_version),
        .o_meta_target_bank(dispatcher_meta_target_bank),
        .o_meta_label(dispatcher_meta_label),
        .o_meta_logits_flat(dispatcher_meta_logits),
        .o_trace_valid(dispatcher_trace_valid),
        .i_trace_ready(dispatcher_trace_ready),
        .o_trace_local_idx_flat(dispatcher_trace_idx_flat),
        .o_trace_q016_flat(dispatcher_trace_q016_flat),
        .o_trace_last(dispatcher_trace_last),
        .o_record_count(dispatcher_record_count),
        .o_capture_busy(dispatcher_capture_busy),
        .o_dispatch_active(dispatcher_dispatch_active),
        .o_dispatched_records(o_dispatched_records),
        .o_error(dispatcher_error)
    );

    newarch_w2_error_frontend #(
        .USE_PARALLEL_CLASS_DIVIDERS(USE_PARALLEL_ERROR_DIVIDERS)
    ) u_error_frontend (
        .clk(clk), .rst_n(rst_n),
        .i_valid(dispatcher_meta_valid), .o_ready(dispatcher_meta_ready),
        .i_sample_id(dispatcher_meta_sample_id),
        .i_weight_version(dispatcher_meta_weight_version),
        .i_target_bank(dispatcher_meta_target_bank),
        .i_label(dispatcher_meta_label),
        .i_logits_flat(dispatcher_meta_logits),
        .o_valid(frontend_error_valid), .i_ready(frontend_error_ready),
        .o_sample_id(frontend_sample_id),
        .o_weight_version(frontend_weight_version),
        .o_target_bank(frontend_target_bank),
        .o_error_q016_flat(frontend_error_flat),
        .o_protocol_error(frontend_protocol_error)
    );

    function [4:0] popcount_shards;
        input [CLUSTERS-1:0] value;
        integer bit_index;
        begin
            popcount_shards = 5'd0;
            for (bit_index = 0; bit_index < CLUSTERS; bit_index = bit_index + 1)
                popcount_shards = popcount_shards + value[bit_index];
        end
    endfunction

    integer cls;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            n8_w2_result_reservations <= 0;
            n8_w2_credit_error <= 1'b0;
        end else if (sample_start_fire) begin
            if ((USE_N8_FOLDED_OVERLAP != 0) &&
                (n8_w2_result_reservations != 0 ||
                 n8_w2_queue_occupancy != 0))
                n8_w2_credit_error <= 1'b1;
            else
                n8_w2_credit_error <= 1'b0;
            n8_w2_result_reservations <= 0;
        end else if (USE_N8_FOLDED_OVERLAP != 0) begin
            case ({forward_result_start, forward_timestep_done})
                2'b10: begin
                    if (!n8_w2_result_credit_ready)
                        n8_w2_credit_error <= 1'b1;
                    else
                        n8_w2_result_reservations <=
                            n8_w2_result_reservations + 1'b1;
                end
                2'b01: begin
                    if (n8_w2_result_reservations == 0)
                        n8_w2_credit_error <= 1'b1;
                    else
                        n8_w2_result_reservations <=
                            n8_w2_result_reservations - 1'b1;
                end
                default: begin end
            endcase
        end else begin
            n8_w2_result_reservations <= 0;
            n8_w2_credit_error <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            n8_w2_infer_busy <= 1'b0;
        end else if (sample_start_fire) begin
            n8_w2_infer_busy <= 1'b0;
        end else if (USE_R18_DUAL_CONTEXT_W2 != 0) begin
            n8_w2_infer_busy <= 1'b0;
        end else if (USE_N8_FOLDED_OVERLAP != 0) begin
            if (online_infer_done)
                n8_w2_infer_busy <= 1'b0;
            if (n8_w2_dispatch)
                n8_w2_infer_busy <= 1'b1;
        end else begin
            n8_w2_infer_busy <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r18_w2_inflight_count <= 4'd0;
            r18_w2_inflight_error <= 1'b0;
        end else if (sample_start_fire) begin
            if ((USE_R18_DUAL_CONTEXT_W2 != 0) &&
                (r18_w2_inflight_count != 0))
                r18_w2_inflight_error <= 1'b1;
            else
                r18_w2_inflight_error <= 1'b0;
            r18_w2_inflight_count <= 4'd0;
        end else if (USE_R18_DUAL_CONTEXT_W2 != 0) begin
            case ({n8_w2_dispatch, online_infer_done})
                2'b10: begin
                    if (r18_w2_inflight_count == 4'hf)
                        r18_w2_inflight_error <= 1'b1;
                    else
                        r18_w2_inflight_count <=
                            r18_w2_inflight_count + 1'b1;
                end
                2'b01: begin
                    if (r18_w2_inflight_count == 0)
                        r18_w2_inflight_error <= 1'b1;
                    else
                        r18_w2_inflight_count <=
                            r18_w2_inflight_count - 1'b1;
                end
                default: begin end
            endcase
        end else begin
            r18_w2_inflight_count <= 4'd0;
            r18_w2_inflight_error <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_count <= 0;
            forward_finished <= 1'b0;
            inference_idle_state <= 1'b1;
            o_sample_done <= 1'b0;
            for (cls = 0; cls < 10; cls = cls + 1)
                online_acc[cls] <= 0;
        end else begin
            o_sample_done <= 1'b0;
            if (sample_start_fire) begin
                pending_count <= 0;
                forward_finished <= 1'b0;
                inference_idle_state <= 1'b0;
                for (cls = 0; cls < 10; cls = cls + 1)
                    online_acc[cls] <= 0;
            end else begin
                pending_count <= pending_after[8:0];
                if (forward_sample_done) begin
                    forward_finished <= 1'b1;
                    if (USE_N8_FOLDED_OVERLAP == 0)
                        inference_idle_state <= 1'b1;
                end
                if (online_infer_done)
                    for (cls = 0; cls < 10; cls = cls + 1)
                        online_acc[cls] <= online_acc[cls] +
                            $signed(online_step_delta[
                                cls*OUT_ACC_BITS +: OUT_ACC_BITS]);
                if ((forward_finished || forward_sample_done) &&
                    (pending_after == 0) && online_infer_done) begin
                    o_sample_done <= 1'b1;
                    inference_idle_state <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_running <= 1'b0;
            top_protocol_error <= 1'b0;
            current_sample_id <= 0;
            sample_weight_version <= 0;
            sample_target_bank <= 1'b1;
            cadence_record_count <= 0;
            cadence_blocked <= 1'b0;
            saw_commit <= 1'b0;
            o_update_command_count <= 0;
            o_runtime_inactive_update_command_count <= 0;
            o_commit_count <= 0;
        end else begin
            if (|shard_cmd_fire)
                o_update_command_count <= o_update_command_count +
                    {27'd0, popcount_shards(shard_cmd_fire)};
            if ((ENABLE_RUNTIME_PROFILE != 0) &&
                |(shard_cmd_fire & ~i_runtime_active_cluster_mask))
                o_runtime_inactive_update_command_count <=
                    o_runtime_inactive_update_command_count +
                    {27'd0, popcount_shards(
                        shard_cmd_fire & ~i_runtime_active_cluster_mask)};
            if (o_commit_pulse)
                o_commit_count <= o_commit_count + 1'b1;
            if (i_sample_start && !o_sample_ready)
                top_protocol_error <= 1'b1;
            if (sample_start_fire) begin
                sample_running <= 1'b1;
                current_sample_id <= i_start_sample_id;
                sample_weight_version <= o_active_version;
                sample_target_bank <= o_inactive_bank;
            end
            if (o_sample_done)
                sample_running <= 1'b0;
            if (training_record_valid && !dispatcher_record_ready)
                top_protocol_error <= 1'b1;

            if (!i_learning_enable) begin
                cadence_record_count <= 0;
                cadence_blocked <= 1'b0;
                saw_commit <= 1'b0;
            end else begin
                if (training_record_fire) begin
                    if (cadence_record_count == ACCUM_LAST) begin
                        cadence_record_count <= 0;
                        cadence_blocked <= 1'b1;
                    end else begin
                        cadence_record_count <= cadence_record_count + 1'b1;
                    end
                end
                if (o_commit_pulse)
                    saw_commit <= 1'b1;
                if (cadence_blocked && saw_commit && !o_snapshot_busy &&
                    o_update_window_ready) begin
                    cadence_blocked <= 1'b0;
                    saw_commit <= 1'b0;
                end
            end
        end
    end

    wire unused_forward = |forward_acc || |forward_vmem ||
        |forward_spike_count || |unused_fixed_logits || |unused_fixed_pred ||
        |scheduler_pending_records;
    wire unused_online = locked_bank || |locked_version ||
        |shard_update_done || |shard_update_cycles_flat ||
        |shard_snapshot_cycles_flat || dispatcher_capture_busy ||
        dispatcher_dispatch_active;

    initial begin
        if (ACCUM_SAMPLES < 1 || ACCUM_SAMPLES > 64)
            $error("N7C1E ACCUM_SAMPLES must be 1..64");
        if ((CLUSTERS != 12) && (CLUSTERS != 16) && (CLUSTERS != 20))
            $error("N7H online top supports 12, 16 or 20 H40 clusters");
        if (H_PER_CLUSTER != 40)
            $error("N7D6 online top requires H_PER_CLUSTER=40");
        if (W2_TIMESTEP_QUEUE_DEPTH < 2 || W2_TIMESTEP_QUEUE_DEPTH > 8)
            $error("W2_TIMESTEP_QUEUE_DEPTH must be 2..8");
        if ((CLUSTERS >= 16) && (USE_PACKED_SDP_W2 == 0))
            $error("N7H H640/H800 requires packed-SDP W2 storage");
        if ((USE_SPARSE_PACKED_SDP_W2 != 0) &&
            ((CLUSTERS != 20) || (USE_PACKED_SDP_W2 == 0)))
            $error("N9D4 sparse W2 requires H800 packed-SDP storage");
        if ((USE_R18_DUAL_CONTEXT_W2 != 0) &&
            ((USE_N8_FOLDED_OVERLAP == 0) ||
             (USE_SPARSE_PACKED_SDP_W2 == 0) ||
             (CLUSTERS != 20) || (W2_TIMESTEP_QUEUE_DEPTH < 2)))
            $error("R18 W2 overlap requires folded H800 sparse W2 and queue depth >=2");
`ifdef N9R12D_MULTIPUMP_W0
        if (USE_R12D_MULTIPUMP_W0 == 0)
            $error("N9R12D compile mode requires USE_R12D_MULTIPUMP_W0=1");
`else
        if (USE_R12D_MULTIPUMP_W0 != 0)
            $error("USE_R12D_MULTIPUMP_W0 requires the N9R12D compile mode");
`endif
    end
endmodule


