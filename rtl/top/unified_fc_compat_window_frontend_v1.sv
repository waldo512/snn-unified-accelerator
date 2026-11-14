`timescale 1ns/1ps
`default_nettype none

// State-free FC_COMPAT frontend. Window filtering, K8 conflict scheduling,
// resident weight reads and sample-level score accumulation remain here;
// neuron state and arithmetic live in the externally owned H40 array.
module unified_fc_compat_window_frontend_v1 #(
    parameter integer SAMPLE_BITS = 16,
    parameter integer T_BITS = 9,
    parameter integer FI_BITS = 12,
    parameter integer TILES = 4,
    parameter integer H_PER_TILE = 40,
    parameter integer O_DIM = 10,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer SCHEDULER_BANK_DEPTH = 64,
    parameter bit ENABLE_TP_NATIVE_WEIGHTS = 1'b0,
    parameter bit ENABLE_W0_LEARNING = 1'b1,
    parameter integer TP_TENSOR_ID_WIDTH = 8,
    parameter integer TP_PARAM_ADDR_WIDTH = 24,
    parameter integer TP_TAG_WIDTH = 16,
    parameter integer TP_CODE_WIDTH = 16,
    parameter integer TP_RESIDUAL_WIDTH = 16,
    parameter logic [TP_TENSOR_ID_WIDTH-1:0] TP_W0_TENSOR_ID = 8'h10,
    parameter logic [TP_TENSOR_ID_WIDTH-1:0] TP_W2_TENSOR_ID = 8'h11,
    parameter WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_k8_input2312",
    parameter W0_RESIDUAL_INIT_FILE = ""
) (
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         i_abort,
    input  logic                                         i_clear_error,

    input  logic [TP_PARAM_ADDR_WIDTH-1:0]              i_tp_w0_parameter_base,
    input  logic [TP_PARAM_ADDR_WIDTH-1:0]              i_tp_w2_parameter_base,
    input  logic                                         i_tp_learning_read_valid,
    output logic                                         o_tp_learning_read_ready,
    input  logic [TP_TENSOR_ID_WIDTH-1:0]               i_tp_learning_read_tensor_id,
    input  logic [TP_PARAM_ADDR_WIDTH-1:0]              i_tp_learning_read_parameter_addr,
    input  logic [TP_TAG_WIDTH-1:0]                     i_tp_learning_read_tag,
    output logic                                         o_tp_learning_response_valid,
    input  logic                                         i_tp_learning_response_ready,
    output logic [TP_TAG_WIDTH-1:0]                     o_tp_learning_response_tag,
    output logic signed [TP_CODE_WIDTH-1:0]             o_tp_learning_response_code,
    output logic signed [TP_RESIDUAL_WIDTH-1:0]         o_tp_learning_response_residual,
    output logic                                         o_tp_learning_response_error,
    input  logic                                         i_tp_learning_write_valid,
    output logic                                         o_tp_learning_write_ready,
    input  logic [TP_TENSOR_ID_WIDTH-1:0]               i_tp_learning_write_tensor_id,
    input  logic [TP_PARAM_ADDR_WIDTH-1:0]              i_tp_learning_write_parameter_addr,
    input  logic signed [TP_CODE_WIDTH-1:0]             i_tp_learning_write_code,
    input  logic signed [TP_RESIDUAL_WIDTH-1:0]         i_tp_learning_write_residual,
    input  logic                                         i_tp_learning_write_code_enable,
    input  logic                                         i_tp_learning_write_residual_enable,
    output logic                                         o_tp_learning_quiescent,

    output logic                                         o_owner_start_valid,
    input  logic                                         i_owner_start_ready,
    input  logic                                         i_owner_enable,
    output logic                                         o_owner_finish,
    input  logic                                         i_owner_finish_ready,

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

    output logic                                         o_array_clear_state,
    output logic [TILES-1:0]                            o_array_active_tile_mask,
    output logic                                         o_array_step_start,
    input  logic                                         i_array_step_ready,
    output logic [T_BITS-1:0]                            o_array_timestep,
    output logic                                         o_array_row_valid,
    input  logic                                         i_array_row_ready,
    output logic [7:0]                                   o_array_lane_valid,
    output logic [TILES*8*H_PER_TILE*5-1:0]             o_array_w0_weight_rows_flat,
    output logic                                         o_array_step_close,
    output logic [TILES*H_PER_TILE*O_DIM*8-1:0]         o_array_w2_weight_flat,
    input  logic                                         i_array_result_valid,
    output logic                                         o_array_result_ready,
    input  logic [T_BITS-1:0]                            i_array_result_timestep,
    input  logic [31:0]                                  i_array_result_events,
    input  logic [TILES*H_PER_TILE-1:0]                  i_array_result_spikes,
    input  logic [O_DIM*OUT_ACC_BITS-1:0]                i_array_result_logits_flat,
    input  logic [TILES*H_PER_TILE*18-1:0]               i_array_vmem_flat,
    input  logic [TILES*H_PER_TILE*16-1:0]               i_array_trace_flat,
    input  logic                                         i_array_protocol_error,

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
    logic array_commands_enabled_q;
    logic array_clear_pending_q;
    logic [FI_BITS:0] active_input_dim_q;
    logic [TILES-1:0] active_tile_mask_q;
    logic top_protocol_error_q;
    logic closing_sample_last_q;
    logic [SAMPLE_BITS-1:0] closing_sample_id_q;
    logic [T_BITS-1:0] closing_timestep_q;
    logic sample_start_accept;
    logic sample_config_legal;
    logic clear_idle_error;

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
    logic weight_protocol_error;
    logic tp_learning_access_legal;
    logic tp_learning_request_present;

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

    assign sample_config_legal = i_active_input_dim != 0 &&
        i_active_input_dim <= 13'd2312 &&
        prefix_mask_legal(i_active_tile_mask);
    assign clear_idle_error = i_clear_error && !sample_running_q;
    assign tp_learning_access_legal = !sample_running_q && scheduler_idle;
    assign tp_learning_request_present = ENABLE_TP_NATIVE_WEIGHTS &&
        (i_tp_learning_read_valid || i_tp_learning_write_valid);
    assign o_owner_start_valid = i_sample_start && sample_config_legal &&
        !sample_running_q && scheduler_idle && weight_idle &&
        !tp_learning_request_present;
    assign o_sample_ready = !sample_running_q && scheduler_idle &&
        weight_idle && i_owner_start_ready && !tp_learning_request_present;
    assign sample_start_accept = o_owner_start_valid &&
        i_owner_start_ready;
    assign o_owner_finish = sample_running_q && state_q == STATE_FINISH;

    assign o_window_ready = sample_running_q && filter_ready;
    assign start_step = sample_running_q && array_commands_enabled_q &&
        state_q == STATE_WAIT_STEP && (group_valid || boundary_valid) &&
        i_array_step_ready;
    assign group_ready = state_q == STATE_ACTIVE &&
        array_commands_enabled_q && weight_request_ready;
    assign group_fire = group_valid && group_ready;
    assign close_step = state_q == STATE_ACTIVE &&
        array_commands_enabled_q && boundary_valid && !group_valid &&
        weight_idle;
    assign weight_row_ready = array_commands_enabled_q &&
        i_array_row_ready;
    assign o_array_result_ready = state_q == STATE_WAIT_RESULT &&
        i_owner_enable;
    assign boundary_ready = i_array_result_valid &&
        o_array_result_ready;

    assign o_array_clear_state = i_owner_enable && array_clear_pending_q;
    assign o_array_active_tile_mask = active_tile_mask_q;
    assign o_array_step_start = start_step;
    assign o_array_timestep = group_valid ? group_timestep :
        boundary_timestep;
    assign o_array_row_valid = weight_row_valid &&
        array_commands_enabled_q;
    assign o_array_lane_valid = weight_lane_valid;
    assign o_array_w0_weight_rows_flat = w0_rows_flat;
    assign o_array_step_close = close_step;
    assign o_array_w2_weight_flat = w2_weight_flat;

    assign o_spikes_flat = i_array_result_spikes;
    assign o_vmem_flat = i_array_vmem_flat;
    assign o_trace_flat = i_array_trace_flat;
    assign o_accepted_windows = filter_accepted_windows;
    assign o_accepted_events = filter_passed_events;
    assign o_issue_groups = scheduler_issue_groups;
    assign o_boundary_count = scheduler_boundary_count;
    assign o_runtime_dropped_events = filter_dropped_events;
    assign o_global_idle = o_sample_ready;
    assign o_protocol_error = top_protocol_error_q ||
        filter_protocol_error || scheduler_protocol_error ||
        weight_protocol_error || i_array_protocol_error;

    newarch_runtime_window_filter_registered_k8 #(
        .SAMPLE_BITS(SAMPLE_BITS),
        .T_BITS(T_BITS),
        .FI_BITS(FI_BITS),
        .MAX_INPUT_DIM(13'd2312)
    ) u_filter (
        .clk(clk),
        .rst_n(rst_n),
        .i_clear(sample_start_accept || i_abort || clear_idle_error),
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
        .i_clear(sample_start_accept || i_abort || clear_idle_error),
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

    generate
        if (ENABLE_TP_NATIVE_WEIGHTS) begin : g_native_weights
            unified_fc_tp_native_resident_weight_service #(
                .ENABLE_W0_LEARNING(ENABLE_W0_LEARNING),
                .FI_BITS(FI_BITS), .TILES(TILES),
                .BANK_HASH_MODE(1),
                .H_PER_TILE(H_PER_TILE), .O_DIM(O_DIM),
                .TENSOR_ID_WIDTH(TP_TENSOR_ID_WIDTH),
                .PARAM_ADDR_WIDTH(TP_PARAM_ADDR_WIDTH),
                .TAG_WIDTH(TP_TAG_WIDTH), .CODE_WIDTH(TP_CODE_WIDTH),
                .RESIDUAL_WIDTH(TP_RESIDUAL_WIDTH),
                .W0_TENSOR_ID(TP_W0_TENSOR_ID),
                .W2_TENSOR_ID(TP_W2_TENSOR_ID),
                .WEIGHT_DIR(WEIGHT_DIR),
                .W0_RESIDUAL_INIT_FILE(W0_RESIDUAL_INIT_FILE)
            ) u_weights (
                .i_clk(clk), .i_rst_n(rst_n),
                .i_clear_pipeline(sample_start_accept || i_abort ||
                                  clear_idle_error),
                .i_clear_error(i_clear_error || i_abort),
                .i_active_tile_mask(active_tile_mask_q),
                .i_request_valid(group_valid && state_q == STATE_ACTIVE),
                .o_request_ready(weight_request_ready),
                .i_bank_valid(group_bank_valid), .i_fi_flat(group_fi_flat),
                .o_row_valid(weight_row_valid),
                .i_row_ready(weight_row_ready),
                .o_lane_valid(weight_lane_valid),
                .o_w0_rows_flat(w0_rows_flat),
                .o_w2_weight_flat(w2_weight_flat),
                .i_w0_parameter_base(i_tp_w0_parameter_base),
                .i_w2_parameter_base(i_tp_w2_parameter_base),
                .i_learning_read_valid(i_tp_learning_read_valid &&
                                       tp_learning_access_legal),
                .o_learning_read_ready(o_tp_learning_read_ready),
                .i_learning_read_tensor_id(i_tp_learning_read_tensor_id),
                .i_learning_read_parameter_addr(
                    i_tp_learning_read_parameter_addr),
                .i_learning_read_tag(i_tp_learning_read_tag),
                .o_learning_response_valid(o_tp_learning_response_valid),
                .i_learning_response_ready(i_tp_learning_response_ready),
                .o_learning_response_tag(o_tp_learning_response_tag),
                .o_learning_response_code(o_tp_learning_response_code),
                .o_learning_response_residual(
                    o_tp_learning_response_residual),
                .o_learning_response_error(o_tp_learning_response_error),
                .i_learning_write_valid(i_tp_learning_write_valid &&
                                        tp_learning_access_legal),
                .o_learning_write_ready(o_tp_learning_write_ready),
                .i_learning_write_tensor_id(i_tp_learning_write_tensor_id),
                .i_learning_write_parameter_addr(
                    i_tp_learning_write_parameter_addr),
                .i_learning_write_code(i_tp_learning_write_code),
                .i_learning_write_residual(i_tp_learning_write_residual),
                .i_learning_write_code_enable(
                    i_tp_learning_write_code_enable),
                .i_learning_write_residual_enable(
                    i_tp_learning_write_residual_enable),
                .o_idle(weight_idle),
                .o_learning_quiescent(o_tp_learning_quiescent),
                .o_protocol_error(weight_protocol_error)
            );
        end else begin : g_static_weights
            unified_fc_compat_h160_resident_weight_service #(
                .FI_BITS(FI_BITS), .TILES(TILES),
                .H_PER_TILE(H_PER_TILE), .O_DIM(O_DIM),
                .WEIGHT_DIR(WEIGHT_DIR)
            ) u_weights (
                .clk(clk), .rst_n(rst_n),
                .i_clear(sample_start_accept || i_abort || clear_idle_error),
                .i_active_tile_mask(active_tile_mask_q),
                .i_request_valid(group_valid && state_q == STATE_ACTIVE),
                .o_request_ready(weight_request_ready),
                .i_bank_valid(group_bank_valid), .i_fi_flat(group_fi_flat),
                .o_row_valid(weight_row_valid),
                .i_row_ready(weight_row_ready),
                .o_lane_valid(weight_lane_valid),
                .o_w0_rows_flat(w0_rows_flat),
                .o_w2_weight_flat(w2_weight_flat),
                .o_idle(weight_idle)
            );
            assign o_tp_learning_read_ready = 1'b0;
            assign o_tp_learning_response_valid = 1'b0;
            assign o_tp_learning_response_tag = '0;
            assign o_tp_learning_response_code = '0;
            assign o_tp_learning_response_residual = '0;
            assign o_tp_learning_response_error = 1'b0;
            assign o_tp_learning_write_ready = 1'b0;
            assign o_tp_learning_quiescent = 1'b1;
            assign weight_protocol_error = 1'b0;
        end
    endgenerate

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

    // Keep RAMB-facing command state off asynchronous reset control cones.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q <= STATE_WAIT_STEP;
            sample_running_q <= 1'b0;
            array_commands_enabled_q <= 1'b0;
            array_clear_pending_q <= 1'b0;
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
            array_commands_enabled_q <= 1'b0;
            array_clear_pending_q <= 1'b0;
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

            if (clear_idle_error) begin
                state_q <= STATE_WAIT_STEP;
                sample_running_q <= 1'b0;
                array_commands_enabled_q <= 1'b0;
                array_clear_pending_q <= 1'b0;
                top_protocol_error_q <= 1'b0;
                closing_sample_last_q <= 1'b0;
                closing_sample_id_q <= '0;
                closing_timestep_q <= '0;
                o_completed_timestep <= '0;
                o_logits_flat <= '0;
                o_sample_cycles <= 64'd0;
            end else begin
                if (sample_running_q)
                    o_sample_cycles <= o_sample_cycles + 1'b1;

                if (sample_running_q && i_owner_enable &&
                    !array_commands_enabled_q) begin
                    if (array_clear_pending_q)
                        array_clear_pending_q <= 1'b0;
                    else
                        array_commands_enabled_q <= 1'b1;
                end

                if (i_sample_start && !sample_start_accept)
                    top_protocol_error_q <= 1'b1;
                if (ENABLE_TP_NATIVE_WEIGHTS && sample_running_q &&
                    (i_tp_learning_read_valid ||
                     i_tp_learning_write_valid))
                    top_protocol_error_q <= 1'b1;

                if (sample_start_accept) begin
                    state_q <= STATE_WAIT_STEP;
                    sample_running_q <= 1'b1;
                    array_commands_enabled_q <= 1'b0;
                    array_clear_pending_q <= i_clear_state;
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
                        if (i_array_result_valid &&
                            o_array_result_ready) begin
                            if (i_array_result_timestep !=
                                closing_timestep_q)
                                top_protocol_error_q <= 1'b1;
                            o_completed_timestep <=
                                i_array_result_timestep;
                            o_timestep_done <= 1'b1;
                            for (integer class_i = 0; class_i < O_DIM;
                                 class_i = class_i + 1)
                                o_logits_flat[
                                    class_i*OUT_ACC_BITS +: OUT_ACC_BITS
                                ] <= o_logits_flat[
                                    class_i*OUT_ACC_BITS +: OUT_ACC_BITS
                                ] + i_array_result_logits_flat[
                                    class_i*OUT_ACC_BITS +: OUT_ACC_BITS
                                ];
                            state_q <= closing_sample_last_q ?
                                STATE_FINISH : STATE_WAIT_STEP;
                        end
                    end
                    STATE_FINISH: begin
                        if (o_owner_finish && i_owner_finish_ready) begin
                            o_sample_done <= 1'b1;
                            sample_running_q <= 1'b0;
                            array_commands_enabled_q <= 1'b0;
                            state_q <= STATE_WAIT_STEP;
                        end
                    end
                        default: top_protocol_error_q <= 1'b1;
                    endcase
                end
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
    always_comb begin
        unused_closing_sample_id = closing_sample_id_q;
        unused_scheduler_accepted_windows = scheduler_accepted_windows;
        unused_scheduler_accepted_events = scheduler_accepted_events;
        unused_filter_fully_dropped_windows = filter_fully_dropped_windows;
        unused_scheduler_group_count = scheduler_group_count;
        unused_scheduler_pending_records = scheduler_pending_records;
        unused_scheduler_fill_bank_depth_flat =
            scheduler_fill_bank_depth_flat;
        unused_scheduler_input_stall_cycles =
            scheduler_input_stall_cycles;
        unused_scheduler_output_stall_cycles =
            scheduler_output_stall_cycles;
        unused_array_result_events = i_array_result_events;
    end
    /* verilator lint_on UNUSEDSIGNAL */

    initial begin
        if (TILES < 1 || TILES > 12 || H_PER_TILE != 40 || O_DIM != 10)
            $error("FC_COMPAT frontend requires 1..12 H40 tiles and 10 outputs");
    end
endmodule

`default_nettype wire
