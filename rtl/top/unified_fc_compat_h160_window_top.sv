`timescale 1ns/1ps
`default_nettype none

// End-to-end FC_COMPAT inference path. Packed event windows are filtered by
// the locked runtime profile, conflict-scheduled into K8 bank groups, read from
// resident W0 storage and executed by the shared H40 tile array.
module unified_fc_compat_h160_window_top #(
    parameter integer SAMPLE_BITS = 16,
    parameter integer T_BITS = 9,
    parameter integer FI_BITS = 12,
    parameter integer TILES = 4,
    parameter integer H_PER_TILE = 40,
    parameter integer O_DIM = 10,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer SCHEDULER_BANK_DEPTH = 64,
    parameter WEIGHT_DIR = "data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         i_abort,
    input  logic                                         i_clear_state,
    input  logic                                         i_sample_start,
    output logic                                         o_sample_ready,
    input  logic [FI_BITS:0]                             i_active_input_dim,
    input  logic [TILES-1:0]                            i_active_tile_mask,
    input  logic                                         i_window_valid,
    output logic                                         o_window_ready,
    input  logic [7:0]                                   i_record_valid,
    input  logic [SAMPLE_BITS-1:0]                       i_sample_id,
    input  logic [T_BITS-1:0]                            i_timestep,
    input  logic [7:0]                                   i_boundary_mask,
    input  logic                                         i_sample_last,
    input  logic [8*FI_BITS-1:0]                         i_fi_flat,
    input  logic [63:0]                                  i_x_flat,
    output logic                                         o_timestep_done,
    output logic                                         o_sample_done,
    output logic [T_BITS-1:0]                            o_completed_timestep,
    output logic [O_DIM*OUT_ACC_BITS-1:0]                o_logits_flat,
    output logic [3:0]                                   o_pred,
    output logic [TILES*H_PER_TILE-1:0]                  o_spikes_flat,
    output logic [TILES*H_PER_TILE*18-1:0]               o_vmem_flat,
    output logic [TILES*H_PER_TILE*16-1:0]               o_trace_flat,
    output logic [31:0]                                  o_accepted_windows,
    output logic [31:0]                                  o_accepted_events,
    output logic [31:0]                                  o_issue_groups,
    output logic [31:0]                                  o_boundary_count,
    output logic [31:0]                                  o_runtime_dropped_events,
    output logic [63:0]                                  o_sample_cycles,
    output logic                                         o_global_idle,
    output logic                                         o_protocol_error
);
    localparam logic [1:0] STATE_WAIT_STEP = 2'd0;
    localparam logic [1:0] STATE_ACTIVE = 2'd1;
    localparam logic [1:0] STATE_WAIT_RESULT = 2'd2;
    localparam logic [1:0] STATE_FINISH = 2'd3;

    logic [1:0] state_q;
    logic sample_running_q;
    logic [FI_BITS:0] active_input_dim_q;
    logic [TILES-1:0] active_tile_mask_q;
    logic top_protocol_error_q;
    logic closing_sample_last_q;
    logic [SAMPLE_BITS-1:0] closing_sample_id_q;
    logic [T_BITS-1:0] closing_timestep_q;
    logic sample_start_accept;

    logic filter_ready;
    logic filtered_valid;
    logic scheduler_input_ready;
    logic [7:0] filtered_record_valid;
    logic [SAMPLE_BITS-1:0] filtered_sample_id;
    logic [T_BITS-1:0] filtered_timestep;
    logic [7:0] filtered_boundary_mask;
    logic filtered_sample_last;
    logic [8*FI_BITS-1:0] filtered_fi_flat;
    logic [63:0] filtered_x_flat;
    logic [31:0] filter_accepted_windows;
    logic [31:0] filter_passed_events;
    logic [31:0] filter_dropped_events;
    logic [31:0] filter_fully_dropped_windows;
    logic filter_protocol_error;

    logic group_valid;
    logic group_ready;
    logic [7:0] group_bank_valid;
    logic [SAMPLE_BITS-1:0] group_sample_id;
    logic [T_BITS-1:0] group_timestep;
    logic [8*FI_BITS-1:0] group_fi_flat;
    logic [63:0] group_x_flat;
    logic boundary_valid;
    logic boundary_ready;
    logic [SAMPLE_BITS-1:0] boundary_sample_id;
    logic [T_BITS-1:0] boundary_timestep;
    logic boundary_sample_last;
    logic scheduler_idle;
    logic scheduler_protocol_error;
    logic [31:0] scheduler_accepted_windows;
    logic [31:0] scheduler_accepted_events;
    logic [31:0] scheduler_issue_groups;
    logic [31:0] scheduler_boundary_count;
    logic [3:0] scheduler_group_count;
    logic [3:0] scheduler_pending_records;
    logic [63:0] scheduler_fill_bank_depth_flat;
    logic [31:0] scheduler_input_stall_cycles;
    logic [31:0] scheduler_output_stall_cycles;

    logic weight_request_ready;
    logic weight_row_valid;
    logic weight_row_ready;
    logic [7:0] weight_lane_valid;
    logic [TILES*8*H_PER_TILE*5-1:0] w0_rows_flat;
    logic [TILES*H_PER_TILE*O_DIM*8-1:0] w2_weight_flat;
    logic weight_idle;

    logic array_step_ready;
    logic array_result_valid;
    logic array_result_ready;
    logic [T_BITS-1:0] array_result_timestep;
    logic [31:0] array_result_events;
    logic [O_DIM*OUT_ACC_BITS-1:0] array_logits_flat;
    logic [TILES*O_DIM*OUT_ACC_BITS-1:0] array_partials_unused;
    logic [3:0] array_pred_unused;
    logic array_busy;
    logic array_protocol_error;
    logic start_step;
    logic close_step;
    logic group_fire;
    logic group_x_legal;

    function automatic logic prefix_mask_legal(
        input logic [TILES-1:0] mask
    );
        logic zero_seen;
        integer tile_i;
        begin
            prefix_mask_legal = mask[0];
            zero_seen = 1'b0;
            for (tile_i = 0; tile_i < TILES; tile_i = tile_i + 1) begin
                if (!mask[tile_i])
                    zero_seen = 1'b1;
                else if (zero_seen)
                    prefix_mask_legal = 1'b0;
            end
        end
    endfunction

    always_comb begin
        group_x_legal = 1'b1;
        for (integer bank_i = 0; bank_i < 8; bank_i = bank_i + 1)
            if (group_bank_valid[bank_i] &&
                group_x_flat[bank_i*8 +: 8] != 8'd1)
                group_x_legal = 1'b0;
    end

    assign o_sample_ready = !sample_running_q && scheduler_idle &&
        weight_idle && !array_busy;
    assign sample_start_accept = i_sample_start && o_sample_ready &&
        i_active_input_dim != 0 && i_active_input_dim <= 13'd2312 &&
        prefix_mask_legal(i_active_tile_mask);
    assign o_window_ready = sample_running_q && filter_ready;
    assign start_step = sample_running_q && state_q == STATE_WAIT_STEP &&
        (group_valid || boundary_valid) && array_step_ready;
    assign group_ready = state_q == STATE_ACTIVE && weight_request_ready;
    assign group_fire = group_valid && group_ready;
    assign close_step = state_q == STATE_ACTIVE && boundary_valid &&
        !group_valid && weight_idle;
    assign array_result_ready = state_q == STATE_WAIT_RESULT;
    assign boundary_ready = array_result_valid && array_result_ready;
    assign o_accepted_windows = filter_accepted_windows;
    assign o_accepted_events = filter_passed_events;
    assign o_issue_groups = scheduler_issue_groups;
    assign o_boundary_count = scheduler_boundary_count;
    assign o_runtime_dropped_events = filter_dropped_events;
    assign o_global_idle = o_sample_ready;
    assign o_protocol_error = top_protocol_error_q ||
        filter_protocol_error || scheduler_protocol_error ||
        array_protocol_error;

    newarch_runtime_window_filter_registered_k8 #(
        .SAMPLE_BITS(SAMPLE_BITS),
        .T_BITS(T_BITS),
        .FI_BITS(FI_BITS),
        .MAX_INPUT_DIM(13'd2312)
    ) u_filter (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(sample_start_accept || i_abort),
        .i_active_input_dim(active_input_dim_q),
        .i_window_valid(i_window_valid && sample_running_q),
        .o_window_ready(filter_ready),
        .i_record_valid(i_record_valid),
        .i_sample_id(i_sample_id),
        .i_timestep(i_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_sample_last(i_sample_last),
        .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat),
        .o_filtered_valid(filtered_valid),
        .i_filtered_ready(scheduler_input_ready),
        .o_filtered_record_valid(filtered_record_valid),
        .o_filtered_sample_id(filtered_sample_id),
        .o_filtered_timestep(filtered_timestep),
        .o_filtered_boundary_mask(filtered_boundary_mask),
        .o_filtered_sample_last(filtered_sample_last),
        .o_filtered_fi_flat(filtered_fi_flat),
        .o_filtered_x_flat(filtered_x_flat),
        .o_accepted_windows(filter_accepted_windows),
        .o_passed_events(filter_passed_events),
        .o_dropped_events(filter_dropped_events),
        .o_fully_dropped_windows(filter_fully_dropped_windows),
        .o_protocol_error(filter_protocol_error)
    );

    newarch_timestep_pingpong_bank_queues_k8 #(
        .SAMPLE_BITS(SAMPLE_BITS),
        .T_BITS(T_BITS),
        .FI_BITS(FI_BITS),
        .IN_DIM(2312),
        .BANK_DEPTH(SCHEDULER_BANK_DEPTH),
        .BANK_HASH_MODE(1)
    ) u_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(sample_start_accept || i_abort),
        .i_window_valid(filtered_valid && sample_running_q),
        .o_window_ready(scheduler_input_ready),
        .i_record_valid(filtered_record_valid),
        .i_sample_id(filtered_sample_id),
        .i_timestep(filtered_timestep),
        .i_boundary_mask(filtered_boundary_mask),
        .i_sample_last(filtered_sample_last),
        .i_fi_flat(filtered_fi_flat),
        .i_x_flat(filtered_x_flat),
        .o_group_valid(group_valid),
        .i_group_ready(group_ready),
        .o_group_count(scheduler_group_count),
        .o_bank_valid(group_bank_valid),
        .o_group_sample_id(group_sample_id),
        .o_group_timestep(group_timestep),
        .o_fi_flat(group_fi_flat),
        .o_x_flat(group_x_flat),
        .o_boundary_valid(boundary_valid),
        .i_boundary_ready(boundary_ready),
        .o_boundary_sample_id(boundary_sample_id),
        .o_boundary_timestep(boundary_timestep),
        .o_boundary_sample_last(boundary_sample_last),
        .o_pending_records(scheduler_pending_records),
        .o_idle(scheduler_idle),
        .o_fill_bank_depth_flat(scheduler_fill_bank_depth_flat),
        .o_accepted_windows(scheduler_accepted_windows),
        .o_accepted_events(scheduler_accepted_events),
        .o_issue_groups(scheduler_issue_groups),
        .o_boundary_count(scheduler_boundary_count),
        .o_input_stall_cycles(scheduler_input_stall_cycles),
        .o_output_stall_cycles(scheduler_output_stall_cycles),
        .o_protocol_error(scheduler_protocol_error)
    );

    unified_fc_compat_h160_resident_weight_service #(
        .FI_BITS(FI_BITS),
        .TILES(TILES),
        .H_PER_TILE(H_PER_TILE),
        .O_DIM(O_DIM),
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_weights (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(sample_start_accept || i_abort),
        .i_active_tile_mask(active_tile_mask_q),
        .i_request_valid(group_valid && state_q == STATE_ACTIVE),
        .o_request_ready(weight_request_ready),
        .i_bank_valid(group_bank_valid),
        .i_fi_flat(group_fi_flat),
        .o_row_valid(weight_row_valid),
        .i_row_ready(weight_row_ready),
        .o_lane_valid(weight_lane_valid),
        .o_w0_rows_flat(w0_rows_flat),
        .o_w2_weight_flat(w2_weight_flat),
        .o_idle(weight_idle)
    );

    unified_fc_compat_tile_array #(
        .TILES(TILES),
        .H_PER_TILE(H_PER_TILE),
        .SOURCE_LANES(8),
        .O_DIM(O_DIM),
        .T_BITS(T_BITS)
    ) u_tiles (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear_state(i_abort ||
                       (sample_start_accept && i_clear_state)),
        .i_active_tile_mask(active_tile_mask_q),
        .i_step_start(start_step),
        .o_step_ready(array_step_ready),
        .i_timestep(group_valid ? group_timestep : boundary_timestep),
        .i_row_valid(weight_row_valid),
        .o_row_ready(weight_row_ready),
        .i_lane_valid(weight_lane_valid),
        .i_w0_weight_rows_flat(w0_rows_flat),
        .i_step_close(close_step),
        .i_w2_weight_flat(w2_weight_flat),
        .o_result_valid(array_result_valid),
        .i_result_ready(array_result_ready),
        .o_result_timestep(array_result_timestep),
        .o_result_events(array_result_events),
        .o_result_spikes(o_spikes_flat),
        .o_tile_partials_flat(array_partials_unused),
        .o_logits_flat(array_logits_flat),
        .o_pred(array_pred_unused),
        .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_busy(array_busy),
        .o_protocol_error(array_protocol_error)
    );

    argmax10 u_sample_argmax (
        .in0(o_logits_flat[0*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in1(o_logits_flat[1*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in2(o_logits_flat[2*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in3(o_logits_flat[3*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in4(o_logits_flat[4*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in5(o_logits_flat[5*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in6(o_logits_flat[6*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in7(o_logits_flat[7*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in8(o_logits_flat[8*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in9(o_logits_flat[9*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .idx(o_pred)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_WAIT_STEP;
            sample_running_q <= 1'b0;
            active_input_dim_q <= 13'd2312;
            active_tile_mask_q <= {TILES{1'b1}};
            top_protocol_error_q <= 1'b0;
            closing_sample_last_q <= 1'b0;
            closing_sample_id_q <= '0;
            closing_timestep_q <= '0;
            o_timestep_done <= 1'b0;
            o_sample_done <= 1'b0;
            o_completed_timestep <= '0;
            o_logits_flat <= '0;
            o_sample_cycles <= 64'd0;
        end else if (i_abort) begin
            state_q <= STATE_WAIT_STEP;
            sample_running_q <= 1'b0;
            top_protocol_error_q <= 1'b0;
            closing_sample_last_q <= 1'b0;
            closing_sample_id_q <= '0;
            closing_timestep_q <= '0;
            o_timestep_done <= 1'b0;
            o_sample_done <= 1'b0;
            o_completed_timestep <= '0;
            o_logits_flat <= '0;
            o_sample_cycles <= 64'd0;
        end else begin
            o_timestep_done <= 1'b0;
            o_sample_done <= 1'b0;
            if (sample_running_q)
                o_sample_cycles <= o_sample_cycles + 1'b1;

            if (i_sample_start && !sample_start_accept)
                top_protocol_error_q <= 1'b1;

            if (sample_start_accept) begin
                state_q <= STATE_WAIT_STEP;
                sample_running_q <= 1'b1;
                active_input_dim_q <= i_active_input_dim;
                active_tile_mask_q <= i_active_tile_mask;
                top_protocol_error_q <= 1'b0;
                closing_sample_last_q <= 1'b0;
                closing_sample_id_q <= '0;
                closing_timestep_q <= '0;
                o_completed_timestep <= '0;
                o_logits_flat <= '0;
                o_sample_cycles <= 64'd0;
            end else begin
                case (state_q)
                    STATE_WAIT_STEP: begin
                        if (start_step)
                            state_q <= STATE_ACTIVE;
                    end
                    STATE_ACTIVE: begin
                        if (group_fire && !group_x_legal)
                            top_protocol_error_q <= 1'b1;
                        if (group_fire &&
                            (group_sample_id != boundary_sample_id) &&
                            boundary_valid)
                            top_protocol_error_q <= 1'b1;
                        if (close_step) begin
                            closing_sample_last_q <= boundary_sample_last;
                            closing_sample_id_q <= boundary_sample_id;
                            closing_timestep_q <= boundary_timestep;
                            state_q <= STATE_WAIT_RESULT;
                        end
                    end
                    STATE_WAIT_RESULT: begin
                        if (array_result_valid) begin
                            if (array_result_timestep != closing_timestep_q)
                                top_protocol_error_q <= 1'b1;
                            o_completed_timestep <= array_result_timestep;
                            o_timestep_done <= 1'b1;
                            for (integer class_i = 0; class_i < O_DIM;
                                 class_i = class_i + 1)
                                o_logits_flat[
                                    class_i*OUT_ACC_BITS +: OUT_ACC_BITS
                                ] <= o_logits_flat[
                                    class_i*OUT_ACC_BITS +: OUT_ACC_BITS
                                ] + array_logits_flat[
                                    class_i*OUT_ACC_BITS +: OUT_ACC_BITS
                                ];
                            state_q <= closing_sample_last_q ?
                                STATE_FINISH : STATE_WAIT_STEP;
                        end
                    end
                    STATE_FINISH: begin
                        o_sample_done <= 1'b1;
                        sample_running_q <= 1'b0;
                        state_q <= STATE_WAIT_STEP;
                    end
                    default: top_protocol_error_q <= 1'b1;
                endcase
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic [SAMPLE_BITS-1:0] unused_closing_sample_id;
    logic [31:0] unused_scheduler_accepted_windows;
    logic [31:0] unused_scheduler_accepted_events;
    logic [31:0] unused_filter_fully_dropped_windows;
    logic [3:0] unused_scheduler_group_count;
    logic [3:0] unused_scheduler_pending_records;
    logic [63:0] unused_scheduler_fill_bank_depth_flat;
    logic [31:0] unused_scheduler_input_stall_cycles;
    logic [31:0] unused_scheduler_output_stall_cycles;
    logic [31:0] unused_array_result_events;
    logic [TILES*O_DIM*OUT_ACC_BITS-1:0] unused_array_partials;
    logic [3:0] unused_array_pred;
    always_comb begin
        unused_closing_sample_id = closing_sample_id_q;
        unused_scheduler_accepted_windows = scheduler_accepted_windows;
        unused_scheduler_accepted_events = scheduler_accepted_events;
        unused_filter_fully_dropped_windows = filter_fully_dropped_windows;
        unused_scheduler_group_count = scheduler_group_count;
        unused_scheduler_pending_records = scheduler_pending_records;
        unused_scheduler_fill_bank_depth_flat = scheduler_fill_bank_depth_flat;
        unused_scheduler_input_stall_cycles = scheduler_input_stall_cycles;
        unused_scheduler_output_stall_cycles = scheduler_output_stall_cycles;
        unused_array_result_events = array_result_events;
        unused_array_partials = array_partials_unused;
        unused_array_pred = array_pred_unused;
    end
    /* verilator lint_on UNUSEDSIGNAL */

    initial begin
        if (TILES < 1 || TILES > 12 || H_PER_TILE != 40 || O_DIM != 10)
            $error("FC_COMPAT window top requires 1..12 H40 tiles and 10 outputs");
    end
endmodule

`default_nettype wire
