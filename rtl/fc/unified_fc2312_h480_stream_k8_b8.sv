`timescale 1ns/1ps

// FC-first overlay of the canonical stream. Only the frozen 20-cluster
// folded-overlap fabric instance is replaced by the local 12-cluster H480
// derivative; scheduler and control behavior remain canonical.

// N7C1D complete fixed-W2 H480 forward path with direct eight-record ingress.
module unified_fc2312_h480_stream_k8_b8 #(
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
    parameter integer ENABLE_FIXED_W2 = 1,
    parameter integer PIPELINE_CONTROL_OVERLAP = 1,
    parameter integer REDUCTION_OUTPUT_REGISTER = 0,
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
    parameter integer ENABLE_RUNTIME_PROFILE = 0,
    parameter W0_WEIGHT_DIR =
        "data/weights/clusters_h480_c12_h40_hash8_n7c1b",
    parameter N9F_REPLICA_W0_WEIGHT_DIR =
        "data/weights/n9f3_h800_replica64_w0",
    parameter N9H_REPLICA_W0_WEIGHT_DIR =
        "data/weights/n9h_h800_replica96_w0",
    parameter N9R10_REPLICA_W0_WEIGHT_DIR =
        "data/weights/n9r10_h800_replica192_w0",
    parameter W2_WEIGHT_DIR =
        "data/weights/clusters_h480_c12_h40_b4_trained_seed16340"
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_sample_start,
    input  wire                         i_clear_state,
    input  wire                         i_timestep_result_ready,
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
    output reg                          o_timestep_done,
    output reg                          o_sample_done,
    output reg  [T_BITS-1:0]            o_completed_timestep,
    output wire [CLUSTERS*H_PER_CLUSTER*ACC_BITS-1:0] o_acc_flat,
    output wire [CLUSTERS*H_PER_CLUSTER-1:0]          o_spikes,
    output wire [CLUSTERS*H_PER_CLUSTER*ACC_BITS-1:0] o_vmem_flat,
    output wire [CLUSTERS*H_PER_CLUSTER*16-1:0]       o_trace_flat,
    output wire [15:0]                  o_spike_count,
    output wire [10*OUT_ACC_BITS-1:0]   o_logits_flat,
    output wire [3:0]                   o_pred,
    output wire [31:0]                  o_accepted_windows,
    output wire [31:0]                  o_accepted_events,
    output wire [31:0]                  o_issue_groups,
    output wire [31:0]                  o_boundary_count,
    output wire [31:0]                  o_ingress_stall_cycles,
    output wire [31:0]                  o_output_stall_cycles,
    output wire [3:0]                   o_scheduler_pending_records,
    output wire [31:0]                  o_runtime_dropped_events,
    output wire                         o_result_start,
    output reg  [31:0]                  o_sample_cycles,
    output wire                         o_event_count_mismatch,
    output wire                         o_protocol_error
);
    reg sample_running;
    reg [SAMPLE_BITS-1:0] active_sample_id;
    reg [T_BITS-1:0] expected_timestep;
    reg fabric_step_start;
    reg top_protocol_error;
    reg signed [OUT_ACC_BITS-1:0] out_acc [0:9];
    reg [12:0] sample_active_input_dim;
    integer output_class;

    wire scheduler_window_ready;
    wire ingress_window_ready;
    wire scheduler_ingress_window_valid;
    wire [7:0] scheduler_ingress_record_valid;
    wire [SAMPLE_BITS-1:0] scheduler_ingress_sample_id;
    wire [T_BITS-1:0] scheduler_ingress_timestep;
    wire [7:0] scheduler_ingress_boundary_mask;
    wire scheduler_ingress_sample_last;
    wire [8*FI_BITS-1:0] scheduler_ingress_fi_flat;
    wire [63:0] scheduler_ingress_x_flat;
    wire [8*8-1:0] scheduler_fill_bank_depth_flat;
    wire scheduler_group_valid;
    wire scheduler_group_ready;
    wire [3:0] scheduler_group_count;
    wire [7:0] scheduler_bank_valid;
    wire [SAMPLE_BITS-1:0] scheduler_group_sample_id;
    wire [T_BITS-1:0] scheduler_group_timestep;
    wire [8*FI_BITS-1:0] scheduler_fi_flat;
    wire [63:0] scheduler_x_flat;
    wire scheduler_boundary_valid;
    wire scheduler_boundary_ready;
    wire [SAMPLE_BITS-1:0] scheduler_boundary_sample_id;
    wire [T_BITS-1:0] scheduler_boundary_timestep;
    wire scheduler_boundary_sample_last;
    wire scheduler_idle;
    wire scheduler_protocol_error;
    wire runtime_filter_protocol_error;
    wire runtime_window_valid;
    wire runtime_window_ready;
    wire [7:0] runtime_record_valid;
    wire [SAMPLE_BITS-1:0] runtime_sample_id;
    wire [T_BITS-1:0] runtime_timestep;
    wire [7:0] runtime_boundary_mask;
    wire runtime_sample_last;
    wire [8*FI_BITS-1:0] runtime_fi_flat;
    wire [63:0] runtime_x_flat;
    wire [CLUSTERS-1:0] runtime_cluster_mask =
        (ENABLE_RUNTIME_PROFILE != 0) ? i_runtime_active_cluster_mask :
        {CLUSTERS{1'b1}};
    localparam N9F_EFFECTIVE_W0_WEIGHT_DIR =
        ((USE_N9R19_DEPTH_SNAPSHOT_INGRESS != 0) ||
         (USE_N9R17_PIPELINED_REPLICA_INGRESS != 0) ||
         (USE_N9R10_BUCKETED192_REPLICA_INGRESS != 0)) ? N9R10_REPLICA_W0_WEIGHT_DIR :
        ((USE_N9H_BUCKETED_REPLICA_INGRESS != 0) ? N9H_REPLICA_W0_WEIGHT_DIR :
        ((USE_N9F_REPLICA_INGRESS != 0) ? N9F_REPLICA_W0_WEIGHT_DIR : W0_WEIGHT_DIR));

    wire fabric_ready;
    wire fabric_step_done;
    wire [SAMPLE_BITS-1:0] fabric_completed_sample;
    wire [T_BITS-1:0] fabric_completed_timestep;
    wire fabric_completed_last;
    wire [10*OUT_ACC_BITS-1:0] fabric_logits;
    wire [31:0] fabric_issued_events;
    wire [31:0] fabric_issue_groups;
    wire [31:0] fabric_overlap_cycles;
    wire fabric_result_start;
    wire fabric_protocol_error;
    wire fabric_transaction_valid = sample_running &&
        (scheduler_group_valid || scheduler_boundary_valid);

    assign o_window_ready = runtime_window_ready && sample_running;
    assign scheduler_group_ready =
        sample_running && fabric_ready && !scheduler_boundary_valid;
    assign scheduler_boundary_ready =
        sample_running && fabric_ready && !scheduler_group_valid;
    assign o_event_count_mismatch =
        fabric_issued_events > o_accepted_events ||
        fabric_issue_groups > o_issue_groups;
    assign o_result_start = fabric_result_start;
    assign o_protocol_error = top_protocol_error || scheduler_protocol_error ||
        runtime_filter_protocol_error || fabric_protocol_error ||
        o_event_count_mismatch;

    generate
        if (ENABLE_RUNTIME_PROFILE != 0) begin : gen_runtime_ingress_filter
            newarch_runtime_window_filter_registered_k8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS),
                .FI_BITS(FI_BITS), .MAX_INPUT_DIM(IN_DIM[FI_BITS:0])
            ) u_runtime_filter (
                .clk(clk), .rst_n(rst_n), .i_clear(i_sample_start),
                .i_active_input_dim(sample_active_input_dim[FI_BITS:0]),
                .i_window_valid(i_window_valid && sample_running),
                .o_window_ready(runtime_window_ready),
                .i_record_valid(i_record_valid),
                .i_sample_id(i_window_sample_id),
                .i_timestep(i_window_timestep),
                .i_boundary_mask(i_boundary_mask),
                .i_sample_last(i_window_sample_last),
                .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
                .o_filtered_valid(runtime_window_valid),
                .i_filtered_ready(ingress_window_ready),
                .o_filtered_record_valid(runtime_record_valid),
                .o_filtered_sample_id(runtime_sample_id),
                .o_filtered_timestep(runtime_timestep),
                .o_filtered_boundary_mask(runtime_boundary_mask),
                .o_filtered_sample_last(runtime_sample_last),
                .o_filtered_fi_flat(runtime_fi_flat),
                .o_filtered_x_flat(runtime_x_flat),
                .o_accepted_windows(), .o_passed_events(),
                .o_dropped_events(o_runtime_dropped_events),
                .o_fully_dropped_windows(),
                .o_protocol_error(runtime_filter_protocol_error)
            );
        end else begin : gen_static_ingress
            assign runtime_window_valid = i_window_valid && sample_running;
            assign runtime_window_ready = ingress_window_ready;
            assign runtime_record_valid = i_record_valid;
            assign runtime_sample_id = i_window_sample_id;
            assign runtime_timestep = i_window_timestep;
            assign runtime_boundary_mask = i_boundary_mask;
            assign runtime_sample_last = i_window_sample_last;
            assign runtime_fi_flat = i_fi_flat;
            assign runtime_x_flat = i_x_flat;
            assign o_runtime_dropped_events = 0;
            assign runtime_filter_protocol_error = 1'b0;
        end
    endgenerate

    generate
        if (USE_N9R19_DEPTH_SNAPSHOT_INGRESS != 0) begin : gen_n9r19_replica_ingress
            newarch_n9r19_depth_snapshot_replica_ingress_adapter8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS),
                .RESERVATION_MODE(N9R17_RESERVATION_MODE),
                .USE_PREFIX_BALANCE4(USE_N9R19B_PREFIX_BALANCE4)
            ) u_n9r19_replica_ingress (
                .clk(clk), .rst_n(rst_n),
                .i_window_valid(runtime_window_valid),
                .o_window_ready(ingress_window_ready),
                .i_record_valid(runtime_record_valid),
                .i_sample_id(runtime_sample_id),
                .i_timestep(runtime_timestep),
                .i_boundary_mask(runtime_boundary_mask),
                .i_sample_last(runtime_sample_last),
                .i_fi_flat(runtime_fi_flat), .i_x_flat(runtime_x_flat),
                .i_scheduler_queue_depth_flat(scheduler_fill_bank_depth_flat),
                .o_scheduler_window_valid(scheduler_ingress_window_valid),
                .i_scheduler_window_ready(scheduler_window_ready),
                .o_record_valid(scheduler_ingress_record_valid),
                .o_sample_id(scheduler_ingress_sample_id),
                .o_timestep(scheduler_ingress_timestep),
                .o_boundary_mask(scheduler_ingress_boundary_mask),
                .o_sample_last(scheduler_ingress_sample_last),
                .o_fi_flat(scheduler_ingress_fi_flat),
                .o_x_flat(scheduler_ingress_x_flat)
            );
        end else if (USE_N9R17_PIPELINED_REPLICA_INGRESS != 0) begin : gen_n9r17_replica_ingress
            newarch_n9r17_two_stage_replica_ingress_adapter8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS),
                .RESERVATION_MODE(N9R17_RESERVATION_MODE)
            ) u_n9r17_replica_ingress (
                .clk(clk), .rst_n(rst_n),
                .i_window_valid(runtime_window_valid),
                .o_window_ready(ingress_window_ready),
                .i_record_valid(runtime_record_valid),
                .i_sample_id(runtime_sample_id),
                .i_timestep(runtime_timestep),
                .i_boundary_mask(runtime_boundary_mask),
                .i_sample_last(runtime_sample_last),
                .i_fi_flat(runtime_fi_flat), .i_x_flat(runtime_x_flat),
                .i_scheduler_queue_depth_flat(scheduler_fill_bank_depth_flat),
                .o_scheduler_window_valid(scheduler_ingress_window_valid),
                .i_scheduler_window_ready(scheduler_window_ready),
                .o_record_valid(scheduler_ingress_record_valid),
                .o_sample_id(scheduler_ingress_sample_id),
                .o_timestep(scheduler_ingress_timestep),
                .o_boundary_mask(scheduler_ingress_boundary_mask),
                .o_sample_last(scheduler_ingress_sample_last),
                .o_fi_flat(scheduler_ingress_fi_flat),
                .o_x_flat(scheduler_ingress_x_flat)
            );
        end else if (USE_N9R10_BUCKETED192_REPLICA_INGRESS != 0) begin : gen_n9r10_replica_ingress
            newarch_n9h_bucketed_replica_ingress_adapter8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .USE_REPLICA192(1)
            ) u_n9r10_replica_ingress (
                .clk(clk), .rst_n(rst_n),
                .i_window_valid(runtime_window_valid),
                .o_window_ready(ingress_window_ready),
                .i_record_valid(runtime_record_valid),
                .i_sample_id(runtime_sample_id),
                .i_timestep(runtime_timestep),
                .i_boundary_mask(runtime_boundary_mask),
                .i_sample_last(runtime_sample_last),
                .i_fi_flat(runtime_fi_flat), .i_x_flat(runtime_x_flat),
                .i_scheduler_queue_depth_flat(scheduler_fill_bank_depth_flat),
                .o_scheduler_window_valid(scheduler_ingress_window_valid),
                .i_scheduler_window_ready(scheduler_window_ready),
                .o_record_valid(scheduler_ingress_record_valid),
                .o_sample_id(scheduler_ingress_sample_id),
                .o_timestep(scheduler_ingress_timestep),
                .o_boundary_mask(scheduler_ingress_boundary_mask),
                .o_sample_last(scheduler_ingress_sample_last),
                .o_fi_flat(scheduler_ingress_fi_flat),
                .o_x_flat(scheduler_ingress_x_flat)
            );
        end else if (USE_N9H_BUCKETED_REPLICA_INGRESS != 0) begin : gen_n9h_replica_ingress
            newarch_n9h_bucketed_replica_ingress_adapter8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .USE_REPLICA192(0)
            ) u_n9h_replica_ingress (
                .clk(clk), .rst_n(rst_n),
                .i_window_valid(runtime_window_valid),
                .o_window_ready(ingress_window_ready),
                .i_record_valid(runtime_record_valid),
                .i_sample_id(runtime_sample_id),
                .i_timestep(runtime_timestep),
                .i_boundary_mask(runtime_boundary_mask),
                .i_sample_last(runtime_sample_last),
                .i_fi_flat(runtime_fi_flat), .i_x_flat(runtime_x_flat),
                .i_scheduler_queue_depth_flat(scheduler_fill_bank_depth_flat),
                .o_scheduler_window_valid(scheduler_ingress_window_valid),
                .i_scheduler_window_ready(scheduler_window_ready),
                .o_record_valid(scheduler_ingress_record_valid),
                .o_sample_id(scheduler_ingress_sample_id),
                .o_timestep(scheduler_ingress_timestep),
                .o_boundary_mask(scheduler_ingress_boundary_mask),
                .o_sample_last(scheduler_ingress_sample_last),
                .o_fi_flat(scheduler_ingress_fi_flat),
                .o_x_flat(scheduler_ingress_x_flat)
            );
        end else if (USE_N9F_REPLICA_INGRESS != 0) begin : gen_n9f_replica_ingress
            newarch_n9f5_replica_ingress_adapter8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS)
            ) u_n9f_replica_ingress (
                .clk(clk), .rst_n(rst_n),
                .i_window_valid(runtime_window_valid),
                .o_window_ready(ingress_window_ready),
                .i_record_valid(runtime_record_valid),
                .i_sample_id(runtime_sample_id),
                .i_timestep(runtime_timestep),
                .i_boundary_mask(runtime_boundary_mask),
                .i_sample_last(runtime_sample_last),
                .i_fi_flat(runtime_fi_flat), .i_x_flat(runtime_x_flat),
                .i_scheduler_queue_depth_flat(scheduler_fill_bank_depth_flat),
                .o_scheduler_window_valid(scheduler_ingress_window_valid),
                .i_scheduler_window_ready(scheduler_window_ready),
                .o_record_valid(scheduler_ingress_record_valid),
                .o_sample_id(scheduler_ingress_sample_id),
                .o_timestep(scheduler_ingress_timestep),
                .o_boundary_mask(scheduler_ingress_boundary_mask),
                .o_sample_last(scheduler_ingress_sample_last),
                .o_fi_flat(scheduler_ingress_fi_flat),
                .o_x_flat(scheduler_ingress_x_flat)
            );
        end else begin : gen_direct_scheduler_ingress
            assign ingress_window_ready = scheduler_window_ready;
            assign scheduler_ingress_window_valid = runtime_window_valid;
            assign scheduler_ingress_record_valid = runtime_record_valid;
            assign scheduler_ingress_sample_id = runtime_sample_id;
            assign scheduler_ingress_timestep = runtime_timestep;
            assign scheduler_ingress_boundary_mask = runtime_boundary_mask;
            assign scheduler_ingress_sample_last = runtime_sample_last;
            assign scheduler_ingress_fi_flat = runtime_fi_flat;
            assign scheduler_ingress_x_flat = runtime_x_flat;
        end
    endgenerate

    initial begin
        if (((USE_N9F_REPLICA_INGRESS != 0) +
             (USE_N9H_BUCKETED_REPLICA_INGRESS != 0) +
             (USE_N9R10_BUCKETED192_REPLICA_INGRESS != 0) +
             (USE_N9R17_PIPELINED_REPLICA_INGRESS != 0) +
             (USE_N9R19_DEPTH_SNAPSHOT_INGRESS != 0)) > 1)
            $error("N9F, N9H, N9R10, N9R17 and N9R19 replica ingress modes are mutually exclusive");
        if ((USE_N9R19B_PREFIX_BALANCE4 != 0) &&
            (USE_N9R19_DEPTH_SNAPSHOT_INGRESS == 0))
            $error("N9R19B prefix balance requires the N9R19 depth-snapshot ingress");
        if (((USE_N9F_REPLICA_INGRESS != 0) ||
             (USE_N9H_BUCKETED_REPLICA_INGRESS != 0) ||
             (USE_N9R10_BUCKETED192_REPLICA_INGRESS != 0) ||
             (USE_N9R17_PIPELINED_REPLICA_INGRESS != 0) ||
             (USE_N9R19_DEPTH_SNAPSHOT_INGRESS != 0)) &&
            ((FI_BITS != 12) || (IN_DIM != 4096) ||
             (CLUSTERS != 20) || (W0_BITS != 5)))
            $error("N9 replica ingress requires the H800 INT5 physical layout");
    end

    genvar cls;
    generate
        for (cls = 0; cls < 10; cls = cls + 1)
            assign o_logits_flat[cls*OUT_ACC_BITS +: OUT_ACC_BITS] = out_acc[cls];
    endgenerate
    argmax10 u_argmax (
        .in0(out_acc[0]), .in1(out_acc[1]), .in2(out_acc[2]),
        .in3(out_acc[3]), .in4(out_acc[4]), .in5(out_acc[5]),
        .in6(out_acc[6]), .in7(out_acc[7]), .in8(out_acc[8]),
        .in9(out_acc[9]), .idx(o_pred)
    );

    newarch_timestep_pingpong_bank_queues_k8 #(
        .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .FI_BITS(FI_BITS),
        .IN_DIM(IN_DIM), .BANK_DEPTH(BANK_DEPTH), .BANK_HASH_MODE(1)
    ) u_scheduler (
        .clk(clk), .rst_n(rst_n), .i_clear(i_sample_start),
        .i_window_valid(scheduler_ingress_window_valid),
        .o_window_ready(scheduler_window_ready),
        .i_record_valid(scheduler_ingress_record_valid),
        .i_sample_id(scheduler_ingress_sample_id),
        .i_timestep(scheduler_ingress_timestep),
        .i_boundary_mask(scheduler_ingress_boundary_mask),
        .i_sample_last(scheduler_ingress_sample_last),
        .i_fi_flat(scheduler_ingress_fi_flat),
        .i_x_flat(scheduler_ingress_x_flat),
        .o_group_valid(scheduler_group_valid),
        .i_group_ready(scheduler_group_ready),
        .o_group_count(scheduler_group_count),
        .o_bank_valid(scheduler_bank_valid),
        .o_group_sample_id(scheduler_group_sample_id),
        .o_group_timestep(scheduler_group_timestep),
        .o_fi_flat(scheduler_fi_flat), .o_x_flat(scheduler_x_flat),
        .o_boundary_valid(scheduler_boundary_valid),
        .i_boundary_ready(scheduler_boundary_ready),
        .o_boundary_sample_id(scheduler_boundary_sample_id),
        .o_boundary_timestep(scheduler_boundary_timestep),
        .o_boundary_sample_last(scheduler_boundary_sample_last),
        .o_fill_bank_depth_flat(scheduler_fill_bank_depth_flat),
        .o_pending_records(o_scheduler_pending_records), .o_idle(scheduler_idle),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events), .o_issue_groups(o_issue_groups),
        .o_boundary_count(o_boundary_count),
        .o_input_stall_cycles(o_ingress_stall_cycles),
        .o_output_stall_cycles(o_output_stall_cycles),
        .o_protocol_error(scheduler_protocol_error)
    );

    generate
        if (USE_N8_FOLDED_OVERLAP != 0) begin : gen_n8_overlap_fabric
            unified_fc_fast_h480_fabric #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .FI_BITS(FI_BITS),
                .IN_DIM(IN_DIM), .CLUSTERS(CLUSTERS),
                .H_PER_CLUSTER(H_PER_CLUSTER), .LIF_LANES(LIF_LANES),
                .ACC_BITS(ACC_BITS),
                .OUT_ACC_BITS(OUT_ACC_BITS), .W0_BITS(W0_BITS),
                .W0_INPUT_SHIFT(W0_INPUT_SHIFT),
                .ENABLE_RUNTIME_CLUSTER_MASK(ENABLE_RUNTIME_PROFILE),
                .USE_N9E2_START_CUT_THROUGH(USE_N9E2_START_CUT_THROUGH),
                .USE_ATOMIC_BOUNDARY_CLOSE(USE_ATOMIC_BOUNDARY_CLOSE),
                .USE_SAME_CYCLE_CONSUMER_RECYCLE(USE_SAME_CYCLE_CONSUMER_RECYCLE),
                .USE_LIF_ACC_SNAPSHOT(USE_LIF_ACC_SNAPSHOT),
                .W0_WEIGHT_DIR(N9F_EFFECTIVE_W0_WEIGHT_DIR)
            ) u_fabric (
                .clk(clk), .rst_n(rst_n), .i_clear_state(i_clear_state),
                .i_step_start(fabric_step_start),
                .i_active_cluster_mask(runtime_cluster_mask),
                .i_result_ready(i_timestep_result_ready),
                .i_transaction_valid(fabric_transaction_valid),
                .o_transaction_ready(fabric_ready),
                .i_is_boundary(scheduler_boundary_valid),
                .i_group_count(scheduler_boundary_valid ? 4'd0 : scheduler_group_count),
                .i_bank_valid(scheduler_boundary_valid ? 8'd0 : scheduler_bank_valid),
                .i_sample_id(scheduler_boundary_valid ?
                    scheduler_boundary_sample_id : scheduler_group_sample_id),
                .i_timestep(scheduler_boundary_valid ?
                    scheduler_boundary_timestep : scheduler_group_timestep),
                .i_sample_last(scheduler_boundary_valid && scheduler_boundary_sample_last),
                .i_fi_flat(scheduler_fi_flat), .i_x_flat(scheduler_x_flat),
                .o_step_done(fabric_step_done),
                .o_completed_sample_id(fabric_completed_sample),
                .o_completed_timestep(fabric_completed_timestep),
                .o_completed_sample_last(fabric_completed_last),
                .o_acc_flat(o_acc_flat), .o_spikes(o_spikes),
                .o_vmem_flat(o_vmem_flat), .o_trace_flat(o_trace_flat),
                .o_spike_count(o_spike_count),
                .o_reduced_logits_flat(fabric_logits),
                .o_issued_events(fabric_issued_events),
                .o_issue_groups(fabric_issue_groups),
                .o_overlap_cycles(fabric_overlap_cycles),
                .o_result_start(fabric_result_start),
                .o_protocol_error(fabric_protocol_error)
            );
        end else begin : gen_legacy_fabric
            newarch_h480_cluster_fabric_k8_b8 #(
                .SAMPLE_BITS(SAMPLE_BITS), .T_BITS(T_BITS), .FI_BITS(FI_BITS),
                .IN_DIM(IN_DIM), .CLUSTERS(CLUSTERS),
                .H_PER_CLUSTER(H_PER_CLUSTER),
                .ACC_BITS(ACC_BITS), .OUT_ACC_BITS(OUT_ACC_BITS),
                .W0_BITS(W0_BITS), .W0_INPUT_SHIFT(W0_INPUT_SHIFT),
                .ENABLE_FIXED_W2(ENABLE_FIXED_W2),
                .PIPELINE_CONTROL_OVERLAP(PIPELINE_CONTROL_OVERLAP),
                .REDUCTION_OUTPUT_REGISTER(REDUCTION_OUTPUT_REGISTER),
                .W0_WEIGHT_DIR(N9F_EFFECTIVE_W0_WEIGHT_DIR),
                .W2_WEIGHT_DIR(W2_WEIGHT_DIR)
            ) u_fabric (
                .clk(clk), .rst_n(rst_n), .i_clear_state(i_clear_state),
                .i_step_start(fabric_step_start),
                .i_transaction_valid(fabric_transaction_valid),
                .o_transaction_ready(fabric_ready),
                .i_is_boundary(scheduler_boundary_valid),
                .i_group_count(scheduler_boundary_valid ? 4'd0 : scheduler_group_count),
                .i_bank_valid(scheduler_boundary_valid ? 8'd0 : scheduler_bank_valid),
                .i_sample_id(scheduler_boundary_valid ?
                    scheduler_boundary_sample_id : scheduler_group_sample_id),
                .i_timestep(scheduler_boundary_valid ?
                    scheduler_boundary_timestep : scheduler_group_timestep),
                .i_sample_last(scheduler_boundary_valid && scheduler_boundary_sample_last),
                .i_fi_flat(scheduler_fi_flat), .i_x_flat(scheduler_x_flat),
                .o_step_done(fabric_step_done),
                .o_completed_sample_id(fabric_completed_sample),
                .o_completed_timestep(fabric_completed_timestep),
                .o_completed_sample_last(fabric_completed_last),
                .o_acc_flat(o_acc_flat), .o_spikes(o_spikes),
                .o_vmem_flat(o_vmem_flat), .o_trace_flat(o_trace_flat),
                .o_spike_count(o_spike_count),
                .o_reduced_logits_flat(fabric_logits),
                .o_issued_events(fabric_issued_events),
                .o_issue_groups(fabric_issue_groups),
                .o_protocol_error(fabric_protocol_error)
            );
            assign fabric_overlap_cycles = 0;
            assign fabric_result_start = 1'b0;
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_running <= 1'b0;
            sample_active_input_dim <= IN_DIM[12:0];
            active_sample_id <= 0;
            expected_timestep <= 0;
            fabric_step_start <= 1'b0;
            top_protocol_error <= 1'b0;
            o_timestep_done <= 1'b0;
            o_sample_done <= 1'b0;
            o_completed_timestep <= 0;
            o_sample_cycles <= 0;
            for (output_class = 0; output_class < 10;
                 output_class = output_class + 1)
                out_acc[output_class] <= 0;
        end else begin
            fabric_step_start <= 1'b0;
            o_timestep_done <= 1'b0;
            o_sample_done <= 1'b0;
            if (sample_running)
                o_sample_cycles <= o_sample_cycles + 1'b1;
            if (i_sample_start) begin
                if (sample_running || !scheduler_idle)
                    top_protocol_error <= 1'b1;
                else begin
                    sample_running <= 1'b1;
                    sample_active_input_dim <=
                        (ENABLE_RUNTIME_PROFILE != 0) ?
                        i_runtime_active_input_dim : IN_DIM[12:0];
                    active_sample_id <= i_start_sample_id;
                    expected_timestep <= 0;
                    fabric_step_start <= 1'b1;
                    top_protocol_error <= 1'b0;
                    o_sample_cycles <= 0;
                    for (output_class = 0; output_class < 10;
                         output_class = output_class + 1)
                        out_acc[output_class] <= 0;
                end
            end else if (fabric_step_done) begin
                if (fabric_completed_sample != active_sample_id ||
                    fabric_completed_timestep != expected_timestep)
                    top_protocol_error <= 1'b1;
                o_timestep_done <= 1'b1;
                o_completed_timestep <= fabric_completed_timestep;
                for (output_class = 0; output_class < 10;
                     output_class = output_class + 1)
                    out_acc[output_class] <= out_acc[output_class] +
                        $signed(fabric_logits[
                            output_class*OUT_ACC_BITS +: OUT_ACC_BITS]);
                if (fabric_completed_last) begin
                    if (fabric_issued_events != o_accepted_events ||
                        fabric_issue_groups != o_issue_groups)
                        top_protocol_error <= 1'b1;
                    o_sample_done <= 1'b1;
                    sample_running <= 1'b0;
                end else begin
                    expected_timestep <= expected_timestep + 1'b1;
                    if (PIPELINE_CONTROL_OVERLAP == 0)
                        fabric_step_start <= 1'b1;
                end
            end
        end
    end
endmodule


