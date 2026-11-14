`timescale 1ns/1ps

// Four-cluster FC-first derivative of the canonical R19B forward fabric.
// The N8D2B dual-accumulator/LIF overlap is preserved; W2 remains external.
module unified_fc_fast_h160_fabric #(
    parameter integer SAMPLE_BITS = 16,
    parameter integer T_BITS = 9,
    parameter integer FI_BITS = 12,
    parameter integer IN_DIM = 2312,
    parameter integer CLUSTERS = 4,
    parameter integer H_PER_CLUSTER = 40,
    parameter integer LIF_LANES = 20,
    parameter integer ACC_BITS = 18,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer W0_BITS = 5,
    parameter integer W0_INPUT_SHIFT = 6,
    parameter integer ENABLE_RUNTIME_CLUSTER_MASK = 0,
    parameter integer USE_N9E2_START_CUT_THROUGH = 0,
    parameter integer USE_ATOMIC_BOUNDARY_CLOSE = 0,
    parameter integer USE_SAME_CYCLE_CONSUMER_RECYCLE = 0,
    parameter integer USE_LIF_ACC_SNAPSHOT = 0,
    parameter W0_WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         i_clear_state,
    input  wire                         i_step_start,
    input  wire [CLUSTERS-1:0]          i_active_cluster_mask,
    input  wire                         i_result_ready,
    input  wire                         i_transaction_valid,
    output wire                         o_transaction_ready,
    input  wire                         i_is_boundary,
    input  wire [3:0]                   i_group_count,
    input  wire [7:0]                   i_bank_valid,
    input  wire [SAMPLE_BITS-1:0]       i_sample_id,
    input  wire [T_BITS-1:0]            i_timestep,
    input  wire                         i_sample_last,
    input  wire [8*FI_BITS-1:0]         i_fi_flat,
    input  wire [63:0]                  i_x_flat,
    output reg                          o_step_done,
    output reg  [SAMPLE_BITS-1:0]       o_completed_sample_id,
    output reg  [T_BITS-1:0]            o_completed_timestep,
    output reg                          o_completed_sample_last,
    output wire [CLUSTERS*H_PER_CLUSTER*ACC_BITS-1:0] o_acc_flat,
    output wire [CLUSTERS*H_PER_CLUSTER-1:0]          o_spikes,
    output wire [CLUSTERS*H_PER_CLUSTER*ACC_BITS-1:0] o_vmem_flat,
    output wire [CLUSTERS*H_PER_CLUSTER*16-1:0]       o_trace_flat,
    output wire [15:0]                  o_spike_count,
    output wire [10*OUT_ACC_BITS-1:0]   o_reduced_logits_flat,
    output reg  [31:0]                  o_issued_events,
    output reg  [31:0]                  o_issue_groups,
    output wire [31:0]                  o_overlap_cycles,
    output wire                         o_result_start,
    output wire                         o_protocol_error
);
    localparam integer META_DEPTH =
        ((USE_LIF_ACC_SNAPSHOT != 0) &&
         (USE_SAME_CYCLE_CONSUMER_RECYCLE != 0)) ? 4 : 2;
    localparam integer META_COUNT_BITS = $clog2(META_DEPTH + 1);

    reg sample_running;
    reg producer_active;
    reg final_boundary_seen;
    reg [SAMPLE_BITS-1:0] producer_sample;
    reg [T_BITS-1:0] producer_timestep;
    reg top_protocol_error;
    reg [CLUSTERS-1:0] done_seen;
    reg [CLUSTERS-1:0] sample_active_cluster_mask;

    reg [SAMPLE_BITS-1:0] meta_sample [0:META_DEPTH-1];
    reg [T_BITS-1:0] meta_timestep [0:META_DEPTH-1];
    reg meta_last [0:META_DEPTH-1];
    reg [META_COUNT_BITS-1:0] meta_count;

    wire [CLUSTERS-1:0] cluster_step_ready;
    wire [CLUSTERS-1:0] cluster_group_ready;
    wire [CLUSTERS-1:0] cluster_done;
    wire [CLUSTERS-1:0] cluster_result_start;
    wire [CLUSTERS-1:0] cluster_protocol_error;
    wire [CLUSTERS-1:0] cluster_group_valid;
    wire [CLUSTERS-1:0] cluster_groups_done;
    wire [CLUSTERS*8-1:0] cluster_bank_valid_flat;
    wire [CLUSTERS*8*FI_BITS-1:0] cluster_fi_flat;
    wire [CLUSTERS*64-1:0] cluster_x_flat;
    wire [CLUSTERS*T_BITS-1:0] cluster_completed_timestep_flat;
    wire [CLUSTERS*32-1:0] cluster_completed_events_flat;
    wire [CLUSTERS*16-1:0] cluster_spike_count_flat;
    wire [CLUSTERS*32-1:0] cluster_overlap_flat;
    wire [CLUSTERS-1:0] cluster_w0_producer_active;
    wire [CLUSTERS-1:0] cluster_w0_consumer_active;
    wire [CLUSTERS*10*OUT_ACC_BITS-1:0] cluster_partial_unused;
    wire [CLUSTERS*H_PER_CLUSTER-1:0] cluster_spikes_raw;
    wire [CLUSTERS*H_PER_CLUSTER*ACC_BITS-1:0] cluster_vmem_raw_flat;
    wire [CLUSTERS*H_PER_CLUSTER*16-1:0] cluster_trace_raw_flat;
    reg [CLUSTERS*H_PER_CLUSTER-1:0] completed_spikes_reg;
    reg [CLUSTERS*16-1:0] completed_spike_count_flat;

    wire multicast_ready;
    wire multicast_fire;
    wire multicast_boundary;
    wire [3:0] multicast_group_count;
    wire [SAMPLE_BITS-1:0] multicast_sample;
    wire [T_BITS-1:0] multicast_timestep;
    wire multicast_sample_last;
    wire multicast_idle;
    wire multicast_protocol_error;
    wire multicast_event_fire = multicast_fire && !multicast_boundary;
    wire multicast_boundary_fire = multicast_fire && multicast_boundary;
    wire [CLUSTERS-1:0] effective_step_ready =
        cluster_step_ready | ~sample_active_cluster_mask;
    wire [CLUSTERS-1:0] effective_group_ready =
        cluster_group_ready | ~sample_active_cluster_mask;
    wire [CLUSTERS-1:0] active_cluster_done =
        cluster_done & sample_active_cluster_mask;
    wire [CLUSTERS-1:0] active_cluster_result_start =
        cluster_result_start & sample_active_cluster_mask;
    wire [CLUSTERS-1:0] done_seen_next = done_seen | active_cluster_done;
    wire all_clusters_done =
        &(done_seen_next | ~sample_active_cluster_mask);
    wire start_producer = sample_running && !producer_active &&
        !final_boundary_seen && i_transaction_valid && multicast_idle &&
        (&effective_step_ready);
    wire producer_admission = producer_active ||
        ((USE_N9E2_START_CUT_THROUGH != 0) && start_producer);
    wire cluster_step_start = start_producer;
    wire step_complete = (|active_cluster_done) && all_clusters_done;
    wire meta_enqueue = multicast_boundary_fire;
    wire meta_dequeue = step_complete;
    wire result_start_mismatch = (|active_cluster_result_start) &&
        !(&(cluster_result_start | ~sample_active_cluster_mask));
    reg [15:0] spike_count_sum;
    integer count_cluster;
    integer check_cluster;
    integer meta_index;

    function automatic valid_prefix_mask;
        input [CLUSTERS-1:0] mask;
        reg [CLUSTERS-1:0] next_mask;
        begin
            next_mask = mask + {{CLUSTERS-1{1'b0}}, 1'b1};
            valid_prefix_mask = (mask != {CLUSTERS{1'b0}}) &&
                                ((mask & next_mask) == {CLUSTERS{1'b0}});
        end
    endfunction
    genvar output_cluster;

    assign o_transaction_ready = producer_admission && multicast_ready;
    assign o_acc_flat = 0;
    assign o_reduced_logits_flat = 0;
    assign o_spike_count = spike_count_sum;
    assign o_overlap_cycles = cluster_overlap_flat[0 +: 32];
    assign o_result_start =
        &(cluster_result_start | ~sample_active_cluster_mask);
    assign o_protocol_error = top_protocol_error || multicast_protocol_error ||
        (|(cluster_protocol_error & sample_active_cluster_mask)) ||
        result_start_mismatch;

    generate
        for (output_cluster = 0; output_cluster < CLUSTERS;
             output_cluster = output_cluster + 1) begin : gen_runtime_output_mask
            assign o_spikes[output_cluster*H_PER_CLUSTER +: H_PER_CLUSTER] =
                sample_active_cluster_mask[output_cluster] ?
                completed_spikes_reg[output_cluster*H_PER_CLUSTER +: H_PER_CLUSTER] :
                {H_PER_CLUSTER{1'b0}};
            assign o_vmem_flat[
                output_cluster*H_PER_CLUSTER*ACC_BITS +: H_PER_CLUSTER*ACC_BITS
            ] = sample_active_cluster_mask[output_cluster] ?
                cluster_vmem_raw_flat[
                    output_cluster*H_PER_CLUSTER*ACC_BITS +: H_PER_CLUSTER*ACC_BITS
                ] : {H_PER_CLUSTER*ACC_BITS{1'b0}};
            assign o_trace_flat[
                output_cluster*H_PER_CLUSTER*16 +: H_PER_CLUSTER*16
            ] = sample_active_cluster_mask[output_cluster] ?
                cluster_trace_raw_flat[
                    output_cluster*H_PER_CLUSTER*16 +: H_PER_CLUSTER*16
                ] : {H_PER_CLUSTER*16{1'b0}};
        end
    endgenerate

    always @* begin
        spike_count_sum = 0;
        for (count_cluster = 0; count_cluster < CLUSTERS;
             count_cluster = count_cluster + 1)
            if (sample_active_cluster_mask[count_cluster])
                spike_count_sum = spike_count_sum +
                    completed_spike_count_flat[count_cluster*16 +: 16];
    end

    newarch_event_multicast_k8_local_endpoint #(
        .CLUSTERS(CLUSTERS), .SAMPLE_BITS(SAMPLE_BITS),
        .T_BITS(T_BITS), .FI_BITS(FI_BITS)
    ) u_multicast (
        .clk(clk), .rst_n(rst_n),
        .i_valid(i_transaction_valid && producer_admission),
        .o_ready(multicast_ready), .i_is_boundary(i_is_boundary),
        .i_group_count(i_group_count), .i_bank_valid(i_bank_valid),
        .i_sample_id(i_sample_id), .i_timestep(i_timestep),
        .i_sample_last(i_sample_last), .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat), .i_cluster_ready(effective_group_ready),
        .o_cluster_group_valid(cluster_group_valid),
        .o_cluster_groups_done(cluster_groups_done),
        .o_bank_valid_flat(cluster_bank_valid_flat),
        .o_fi_flat(cluster_fi_flat), .o_x_flat(cluster_x_flat),
        .o_transaction_fire(multicast_fire),
        .o_transaction_is_boundary(multicast_boundary),
        .o_transaction_group_count(multicast_group_count),
        .o_transaction_sample_id(multicast_sample),
        .o_transaction_timestep(multicast_timestep),
        .o_transaction_sample_last(multicast_sample_last),
        .o_idle(multicast_idle), .o_protocol_error(multicast_protocol_error)
    );

`define N8D2B_CLUSTER(ID, STEM) \
    newarch_hidden_cluster_k8_b8_folded_overlap #( \
        .IN_DIM(IN_DIM), .H_COUNT(H_PER_CLUSTER), .LIF_LANES(LIF_LANES), \
        .FI_BITS(FI_BITS), .ACC_BITS(ACC_BITS), \
        .OUT_ACC_BITS(OUT_ACC_BITS), .W0_BITS(W0_BITS), \
        .W0_INPUT_SHIFT(W0_INPUT_SHIFT), .T_BITS(T_BITS), \
        .USE_ATOMIC_BOUNDARY_CLOSE(USE_ATOMIC_BOUNDARY_CLOSE), \
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(USE_SAME_CYCLE_CONSUMER_RECYCLE), \
        .USE_LIF_ACC_SNAPSHOT(USE_LIF_ACC_SNAPSHOT), \
        .ENABLE_FIXED_W2(0), \
        .BANK_FILE_0({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_00.mem"}), \
        .BANK_FILE_1({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_01.mem"}), \
        .BANK_FILE_2({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_02.mem"}), \
        .BANK_FILE_3({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_03.mem"}), \
        .BANK_FILE_4({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_04.mem"}), \
        .BANK_FILE_5({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_05.mem"}), \
        .BANK_FILE_6({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_06.mem"}), \
        .BANK_FILE_7({W0_WEIGHT_DIR, "/w0_cluster_", STEM, "_bank_07.mem"}) \
    ) u_cluster_``ID ( \
        .clk(clk), .rst_n(rst_n), .i_clear_state(i_clear_state), \
        .i_step_start(cluster_step_start && sample_active_cluster_mask[ID]), \
        .o_step_ready(cluster_step_ready[ID]), \
        .i_result_ready(i_result_ready), \
        .o_result_start(cluster_result_start[ID]), .i_timestep(i_timestep), \
        .i_group_valid(cluster_group_valid[ID] && sample_active_cluster_mask[ID]), \
        .o_group_ready(cluster_group_ready[ID]), \
        .i_bank_valid(cluster_bank_valid_flat[(ID)*8 +: 8]), \
        .i_fi_flat(cluster_fi_flat[(ID)*8*FI_BITS +: 8*FI_BITS]), \
        .i_x_flat(cluster_x_flat[(ID)*64 +: 64]), \
        .i_groups_done(cluster_groups_done[ID] && sample_active_cluster_mask[ID]), \
        .o_done(cluster_done[ID]), \
        .o_completed_timestep(cluster_completed_timestep_flat[(ID)*T_BITS +: T_BITS]), \
        .o_completed_events(cluster_completed_events_flat[(ID)*32 +: 32]), \
        .o_completed_spikes(cluster_spikes_raw[(ID)*H_PER_CLUSTER +: H_PER_CLUSTER]), \
        .o_completed_spike_count(cluster_spike_count_flat[(ID)*16 +: 16]), \
        .o_partial_logits_flat(cluster_partial_unused[(ID)*10*OUT_ACC_BITS +: 10*OUT_ACC_BITS]), \
        .o_vmem_flat(cluster_vmem_raw_flat[(ID)*H_PER_CLUSTER*ACC_BITS +: H_PER_CLUSTER*ACC_BITS]), \
        .o_trace_flat(cluster_trace_raw_flat[(ID)*H_PER_CLUSTER*16 +: H_PER_CLUSTER*16]), \
        .o_w0_lif_overlap_cycles(cluster_overlap_flat[(ID)*32 +: 32]), \
        .o_w0_producer_active(cluster_w0_producer_active[ID]), \
        .o_w0_consumer_active(cluster_w0_consumer_active[ID]), \
        .o_protocol_error(cluster_protocol_error[ID]) \
    );

    `N8D2B_CLUSTER(0, "00")
    `N8D2B_CLUSTER(1, "01")
    `N8D2B_CLUSTER(2, "02")
    `N8D2B_CLUSTER(3, "03")
`undef N8D2B_CLUSTER

    initial begin
        if (CLUSTERS != 4 || H_PER_CLUSTER != 40)
            $error("FC-first H160 fabric requires 4 x H40");
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_running <= 1'b0;
            producer_active <= 1'b0;
            final_boundary_seen <= 1'b0;
            producer_sample <= 0;
            producer_timestep <= 0;
            top_protocol_error <= 1'b0;
            done_seen <= 0;
            completed_spikes_reg <= 0;
            completed_spike_count_flat <= 0;
            sample_active_cluster_mask <= {CLUSTERS{1'b1}};
            meta_count <= 0;
            o_step_done <= 1'b0;
            o_completed_sample_id <= 0;
            o_completed_timestep <= 0;
            o_completed_sample_last <= 1'b0;
            o_issued_events <= 0;
            o_issue_groups <= 0;
            for (meta_index = 0; meta_index < META_DEPTH;
                 meta_index = meta_index + 1) begin
                meta_sample[meta_index] <= 0;
                meta_timestep[meta_index] <= 0;
                meta_last[meta_index] <= 1'b0;
            end
        end else begin
            o_step_done <= 1'b0;
            if (i_clear_state) begin
                sample_running <= 1'b0;
                producer_active <= 1'b0;
                final_boundary_seen <= 1'b0;
                done_seen <= 0;
                completed_spikes_reg <= 0;
                completed_spike_count_flat <= 0;
                sample_active_cluster_mask <= {CLUSTERS{1'b1}};
                meta_count <= 0;
                top_protocol_error <= 1'b0;
                o_issued_events <= 0;
                o_issue_groups <= 0;
            end

            if (i_step_start) begin
                if (sample_running || producer_active || meta_count != 0 ||
                    !multicast_idle)
                    top_protocol_error <= 1'b1;
                else if (ENABLE_RUNTIME_CLUSTER_MASK != 0 &&
                         !valid_prefix_mask(i_active_cluster_mask))
                    top_protocol_error <= 1'b1;
                else begin
                    sample_running <= 1'b1;
                    sample_active_cluster_mask <=
                        (ENABLE_RUNTIME_CLUSTER_MASK != 0) ?
                        i_active_cluster_mask : {CLUSTERS{1'b1}};
                    final_boundary_seen <= 1'b0;
                    top_protocol_error <= 1'b0;
                    o_issued_events <= 0;
                    o_issue_groups <= 0;
                end
            end

            if (start_producer) begin
                producer_active <= 1'b1;
                producer_sample <= i_sample_id;
                producer_timestep <= i_timestep;
            end

            if (multicast_event_fire) begin
                if (multicast_sample != producer_sample ||
                    multicast_timestep != producer_timestep)
                    top_protocol_error <= 1'b1;
                o_issued_events <= o_issued_events + multicast_group_count;
                o_issue_groups <= o_issue_groups + 1'b1;
            end

            if (multicast_boundary_fire) begin
                if (multicast_sample != producer_sample ||
                    multicast_timestep != producer_timestep)
                    top_protocol_error <= 1'b1;
                producer_active <= 1'b0;
                if (multicast_sample_last)
                    final_boundary_seen <= 1'b1;
            end

            if (|active_cluster_done) begin
                for (check_cluster = 0; check_cluster < CLUSTERS;
                     check_cluster = check_cluster + 1) begin
                    if (active_cluster_done[check_cluster]) begin
                        completed_spikes_reg[
                            check_cluster*H_PER_CLUSTER +: H_PER_CLUSTER
                        ] <= cluster_spikes_raw[
                            check_cluster*H_PER_CLUSTER +: H_PER_CLUSTER
                        ];
                        completed_spike_count_flat[
                            check_cluster*16 +: 16
                        ] <= cluster_spike_count_flat[
                            check_cluster*16 +: 16
                        ];
                    end
                end
                if (step_complete)
                    done_seen <= 0;
                else
                    done_seen <= done_seen_next;
            end

            if (meta_dequeue) begin
                if (meta_count == 0) begin
                    top_protocol_error <= 1'b1;
                end else begin
                    o_step_done <= 1'b1;
                    o_completed_sample_id <= meta_sample[0];
                    o_completed_timestep <= meta_timestep[0];
                    o_completed_sample_last <= meta_last[0];
                    for (check_cluster = 0; check_cluster < CLUSTERS;
                         check_cluster = check_cluster + 1) begin
                        if (sample_active_cluster_mask[check_cluster]) begin
                            if (cluster_completed_timestep_flat[check_cluster*T_BITS +: T_BITS] !=
                                meta_timestep[0])
                                top_protocol_error <= 1'b1;
                            if (cluster_completed_events_flat[check_cluster*32 +: 32] !=
                                cluster_completed_events_flat[0 +: 32])
                                top_protocol_error <= 1'b1;
                        end
                    end
                    if (meta_last[0]) begin
                        sample_running <= 1'b0;
                        final_boundary_seen <= 1'b0;
                    end
                end
            end

            case ({meta_enqueue, meta_dequeue})
                2'b10: begin
                    if (meta_count < META_DEPTH) begin
                        meta_sample[meta_count] <= multicast_sample;
                        meta_timestep[meta_count] <= multicast_timestep;
                        meta_last[meta_count] <= multicast_sample_last;
                        meta_count <= meta_count + 1'b1;
                    end else begin
                        top_protocol_error <= 1'b1;
                    end
                end
                2'b01: begin
                    if (meta_count != 0) begin
                        for (meta_index = 0; meta_index < META_DEPTH-1;
                             meta_index = meta_index + 1) begin
                            meta_sample[meta_index] <= meta_sample[meta_index+1];
                            meta_timestep[meta_index] <= meta_timestep[meta_index+1];
                            meta_last[meta_index] <= meta_last[meta_index+1];
                        end
                        meta_count <= meta_count - 1'b1;
                    end
                end
                2'b11: begin
                    if (meta_count != 0) begin
                        for (meta_index = 0; meta_index < META_DEPTH-1;
                             meta_index = meta_index + 1) begin
                            meta_sample[meta_index] <= meta_sample[meta_index+1];
                            meta_timestep[meta_index] <= meta_timestep[meta_index+1];
                            meta_last[meta_index] <= meta_last[meta_index+1];
                        end
                        meta_sample[meta_count-1'b1] <= multicast_sample;
                        meta_timestep[meta_count-1'b1] <= multicast_timestep;
                        meta_last[meta_count-1'b1] <= multicast_sample_last;
                    end else begin
                        top_protocol_error <= 1'b1;
                    end
                end
                default: begin end
            endcase
        end
    end
endmodule
