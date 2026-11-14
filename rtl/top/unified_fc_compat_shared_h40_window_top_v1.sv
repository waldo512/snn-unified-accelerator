`timescale 1ns/1ps
`default_nettype none

// Verification wrapper proving that the production FC frontend executes on
// the same owner-controlled H40 array used by the SRNN and SCNN frontends.
module unified_fc_compat_shared_h40_window_top_v1 #(
    parameter integer SAMPLE_BITS = 16,
    parameter integer T_BITS = 9,
    parameter integer FI_BITS = 12,
    parameter integer TILES = 4,
    parameter integer H_PER_TILE = 40,
    parameter integer O_DIM = 10,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer SCHEDULER_BANK_DEPTH = 64,
    parameter WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         i_abort,
    input  logic                                         i_clear_error,
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
    localparam logic [1:0] OWNER_FC = 2'd1;

    logic owner_start_valid;
    logic owner_start_ready;
    logic owner_enable;
    logic owner_finish;
    logic owner_finish_ready;
    logic owner_active;
    logic [1:0] active_owner;

    logic array_clear_state;
    logic [TILES-1:0] array_active_tile_mask;
    logic array_step_start;
    logic array_step_ready;
    logic [T_BITS-1:0] array_timestep;
    logic array_row_valid;
    logic array_row_ready;
    logic [7:0] array_lane_valid;
    logic [TILES*8*H_PER_TILE*5-1:0] array_w0_weight_rows_flat;
    logic array_step_close;
    logic [TILES*H_PER_TILE*O_DIM*8-1:0] array_w2_weight_flat;
    logic array_result_valid;
    logic array_result_ready;
    logic [1:0] array_result_owner;
    logic [T_BITS-1:0] array_result_timestep;
    logic [31:0] array_result_events;
    logic [TILES*H_PER_TILE-1:0] array_result_spikes;
    logic [O_DIM*OUT_ACC_BITS-1:0] array_result_logits_flat;
    logic [TILES*H_PER_TILE*18-1:0] array_vmem_flat;
    logic [TILES*H_PER_TILE*16-1:0] array_trace_flat;
    logic array_busy;
    logic array_idle;
    logic array_error;

    assign owner_enable = owner_active && active_owner == OWNER_FC;

    unified_fc_compat_window_frontend_v1 #(
        .SAMPLE_BITS(SAMPLE_BITS),
        .T_BITS(T_BITS),
        .FI_BITS(FI_BITS),
        .TILES(TILES),
        .H_PER_TILE(H_PER_TILE),
        .O_DIM(O_DIM),
        .OUT_ACC_BITS(OUT_ACC_BITS),
        .SCHEDULER_BANK_DEPTH(SCHEDULER_BANK_DEPTH),
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_frontend (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_tp_w0_parameter_base('0),
        .i_tp_w2_parameter_base('0),
        .i_tp_learning_read_valid(1'b0),
        .o_tp_learning_read_ready(),
        .i_tp_learning_read_tensor_id('0),
        .i_tp_learning_read_parameter_addr('0),
        .i_tp_learning_read_tag('0),
        .o_tp_learning_response_valid(),
        .i_tp_learning_response_ready(1'b0),
        .o_tp_learning_response_tag(),
        .o_tp_learning_response_code(),
        .o_tp_learning_response_residual(),
        .o_tp_learning_response_error(),
        .i_tp_learning_write_valid(1'b0),
        .o_tp_learning_write_ready(),
        .i_tp_learning_write_tensor_id('0),
        .i_tp_learning_write_parameter_addr('0),
        .i_tp_learning_write_code('0),
        .i_tp_learning_write_residual('0),
        .i_tp_learning_write_code_enable(1'b0),
        .i_tp_learning_write_residual_enable(1'b0),
        .o_tp_learning_quiescent(),
        .o_owner_start_valid(owner_start_valid),
        .i_owner_start_ready(owner_start_ready),
        .i_owner_enable(owner_enable),
        .o_owner_finish(owner_finish),
        .i_owner_finish_ready(owner_finish_ready),
        .i_clear_state(i_clear_state),
        .i_sample_start(i_sample_start),
        .o_sample_ready(o_sample_ready),
        .i_active_input_dim(i_active_input_dim),
        .i_active_tile_mask(i_active_tile_mask),
        .i_window_valid(i_window_valid),
        .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid),
        .i_sample_id(i_sample_id),
        .i_timestep(i_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_sample_last(i_sample_last),
        .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat),
        .o_array_clear_state(array_clear_state),
        .o_array_active_tile_mask(array_active_tile_mask),
        .o_array_step_start(array_step_start),
        .i_array_step_ready(array_step_ready),
        .o_array_timestep(array_timestep),
        .o_array_row_valid(array_row_valid),
        .i_array_row_ready(array_row_ready),
        .o_array_lane_valid(array_lane_valid),
        .o_array_w0_weight_rows_flat(array_w0_weight_rows_flat),
        .o_array_step_close(array_step_close),
        .o_array_w2_weight_flat(array_w2_weight_flat),
        .i_array_result_valid(array_result_valid),
        .o_array_result_ready(array_result_ready),
        .i_array_result_timestep(array_result_timestep),
        .i_array_result_events(array_result_events),
        .i_array_result_spikes(array_result_spikes),
        .i_array_result_logits_flat(array_result_logits_flat),
        .i_array_vmem_flat(array_vmem_flat),
        .i_array_trace_flat(array_trace_flat),
        .i_array_protocol_error(array_error),
        .o_timestep_done(o_timestep_done),
        .o_sample_done(o_sample_done),
        .o_completed_timestep(o_completed_timestep),
        .o_logits_flat(o_logits_flat),
        .o_pred(o_pred),
        .o_spikes_flat(o_spikes_flat),
        .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups),
        .o_boundary_count(o_boundary_count),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_sample_cycles(o_sample_cycles),
        .o_global_idle(o_global_idle),
        .o_protocol_error(o_protocol_error)
    );

    unified_h40_shared_inference_array_v1 #(
        .TILES(TILES),
        .H_PER_TILE(H_PER_TILE),
        .LANES(8),
        .O_DIM(O_DIM),
        .FC_OUT_ACC_BITS(OUT_ACC_BITS),
        .T_BITS(T_BITS)
    ) u_array (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_abort || i_clear_error),
        .i_owner_valid(owner_start_valid),
        .o_owner_ready(owner_start_ready),
        .i_owner(OWNER_FC),
        .i_owner_release_valid(owner_finish),
        .o_owner_release_ready(owner_finish_ready),
        .o_owner_active(owner_active),
        .o_active_owner(active_owner),
        .i_fc_clear_state(array_clear_state),
        .i_fc_active_tile_mask(array_active_tile_mask),
        .i_fc_step_start(array_step_start),
        .o_fc_step_ready(array_step_ready),
        .i_fc_timestep(array_timestep),
        .i_fc_row_valid(array_row_valid),
        .o_fc_row_ready(array_row_ready),
        .i_fc_lane_valid(array_lane_valid),
        .i_fc_w0_weight_rows_flat(array_w0_weight_rows_flat),
        .i_fc_step_close(array_step_close),
        .i_fc_w2_weight_flat(array_w2_weight_flat),
        .i_k8_valid(1'b0),
        .o_k8_ready(),
        .i_k8_engine_mask('0),
        .i_k8_lane_mask('0),
        .i_k8_q8_8_feature_flat('0),
        .i_k8_q8_8_weight_flat('0),
        .i_k8_int8_lhs_flat('0),
        .i_k8_int8_rhs_flat('0),
        .i_k8_first(1'b0),
        .i_k8_last(1'b0),
        .i_k8_local_word('0),
        .i_k8_output_channel_base('0),
        .i_srnn_clear_spikes(1'b0),
        .i_srnn_clear_word_valid(1'b0),
        .i_srnn_clear_local_word('0),
        .i_srnn_active_neurons('0),
        .i_srnn_lif_begin(1'b0),
        .i_srnn_lif_word_valid(1'b0),
        .i_srnn_lif_local_word('0),
        .i_srnn_leak_q15('0),
        .i_srnn_threshold('0),
        .i_srnn_current_left_shift('0),
        .i_srnn_reset_to_zero(1'b0),
        .i_srnn_commit_timestep(1'b0),
        .o_result_valid(array_result_valid),
        .i_result_ready(array_result_ready),
        .o_result_owner(array_result_owner),
        .o_result_mode_q8_8(),
        .o_result_engine_mask(),
        .o_result_lane_mask(),
        .o_result_channel_flat(),
        .o_result_k8_q8_8_accumulator_flat(),
        .o_result_k8_int8_contribution_flat(),
        .o_result_fc_timestep(array_result_timestep),
        .o_result_fc_events(array_result_events),
        .o_result_fc_spikes(array_result_spikes),
        .o_result_fc_tile_partials_flat(),
        .o_result_fc_logits_flat(array_result_logits_flat),
        .o_result_fc_pred(),
        .o_fc_vmem_flat(array_vmem_flat),
        .o_fc_trace_flat(array_trace_flat),
        .o_srnn_previous_spikes(),
        .o_srnn_current_spikes(),
        .o_srnn_accumulator_flat(),
        .o_srnn_membrane_flat(),
        .o_busy(array_busy),
        .o_idle(array_idle),
        .o_sticky_error(array_error)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic [1:0] unused_array_result_owner;
    logic unused_array_busy;
    logic unused_array_idle;
    always_comb begin
        unused_array_result_owner = array_result_owner;
        unused_array_busy = array_busy;
        unused_array_idle = array_idle;
    end
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
