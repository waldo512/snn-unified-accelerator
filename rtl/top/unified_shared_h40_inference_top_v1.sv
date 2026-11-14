`timescale 1ns/1ps
`default_nettype none

// Paper-oriented production inference top. FC, SRNN and SCNN retain their
// topology-specific ingress frontends while all arithmetic/state transactions
// converge on one owner-controlled H40 physical array.
module unified_shared_h40_inference_top_v1 #(
    parameter int ENGINES = 12,
    parameter int FC_TILES = ENGINES,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int VERSION_WIDTH = 16,
    parameter int REQUEST_ID_WIDTH = 16,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int FEATURE_DEPTH = 9216,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48,
    parameter bit INCLUDE_FC_COMPAT = 1'b1,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0,
    parameter bit ENABLE_FC_TP_NATIVE_WEIGHTS = 1'b0,
    parameter bit FC_LEARN_W0 = 1'b1,
    parameter int FC_TP_TENSOR_ID_WIDTH = 8,
    parameter int FC_TP_PARAM_ADDR_WIDTH = 24,
    parameter int FC_TP_TAG_WIDTH = 16,
    parameter int FC_TP_CODE_WIDTH = 16,
    parameter int FC_TP_RESIDUAL_WIDTH = 16,
    parameter FC_WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312",
    parameter FC_W0_RESIDUAL_INIT_FILE = ""
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,

    input  logic                                      i_fc_sample_start,
    output logic                                      o_fc_sample_ready,
    input  logic                                      i_fc_sample_clear_state,
    input  logic [12:0]                               i_fc_active_input_dim,
    input  logic [ENGINES-1:0]                        i_fc_requested_tile_mask,
    input  logic                                      i_fc_window_valid,
    output logic                                      o_fc_window_ready,
    input  logic [7:0]                                i_fc_record_valid,
    input  logic [15:0]                               i_fc_sample_id,
    input  logic [8:0]                                i_fc_window_timestep,
    input  logic [7:0]                                i_fc_boundary_mask,
    input  logic                                      i_fc_sample_last,
    input  logic [8*12-1:0]                           i_fc_fi_flat,
    input  logic [63:0]                               i_fc_x_flat,
    output logic                                      o_fc_timestep_done,
    output logic                                      o_fc_sample_done,
    output logic [8:0]                                o_fc_completed_timestep,
    output logic [10*24-1:0]                          o_fc_logits_flat,
    output logic [3:0]                                o_fc_pred,
    output logic [ENGINES*40-1:0]                     o_fc_spikes_flat,
    output logic [ENGINES*40*18-1:0]                  o_fc_vmem_flat,
    output logic [ENGINES*40*16-1:0]                  o_fc_trace_flat,
    output logic [31:0]                               o_fc_accepted_windows,
    output logic [31:0]                               o_fc_accepted_events,
    output logic [31:0]                               o_fc_issue_groups,
    output logic [31:0]                               o_fc_boundary_count,
    output logic [31:0]                               o_fc_dropped_events,
    output logic [63:0]                               o_fc_sample_cycles,
    output logic                                      o_fc_protocol_error,

    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_w0_parameter_base,
    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_w2_parameter_base,
    input  logic                                      i_fc_tp_learning_read_valid,
    output logic                                      o_fc_tp_learning_read_ready,
    input  logic [FC_TP_TENSOR_ID_WIDTH-1:0]          i_fc_tp_learning_read_tensor_id,
    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_learning_read_parameter_addr,
    input  logic [FC_TP_TAG_WIDTH-1:0]                i_fc_tp_learning_read_tag,
    output logic                                      o_fc_tp_learning_response_valid,
    input  logic                                      i_fc_tp_learning_response_ready,
    output logic [FC_TP_TAG_WIDTH-1:0]                o_fc_tp_learning_response_tag,
    output logic signed [FC_TP_CODE_WIDTH-1:0]        o_fc_tp_learning_response_code,
    output logic signed [FC_TP_RESIDUAL_WIDTH-1:0]    o_fc_tp_learning_response_residual,
    output logic                                      o_fc_tp_learning_response_error,
    input  logic                                      i_fc_tp_learning_write_valid,
    output logic                                      o_fc_tp_learning_write_ready,
    input  logic [FC_TP_TENSOR_ID_WIDTH-1:0]          i_fc_tp_learning_write_tensor_id,
    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_learning_write_parameter_addr,
    input  logic signed [FC_TP_CODE_WIDTH-1:0]        i_fc_tp_learning_write_code,
    input  logic signed [FC_TP_RESIDUAL_WIDTH-1:0]    i_fc_tp_learning_write_residual,
    input  logic                                      i_fc_tp_learning_write_code_enable,
    input  logic                                      i_fc_tp_learning_write_residual_enable,
    output logic                                      o_fc_tp_learning_quiescent,

    input  logic                                      i_srnn_start_valid,
    output logic                                      o_srnn_start_ready,
    input  logic                                      i_srnn_state_reset,
    input  logic [12:0]                               i_srnn_input_dim,
    input  logic [9:0]                                i_srnn_active_neurons,
    input  logic [31:0]                               i_srnn_w0_base,
    input  logic [31:0]                               i_srnn_recurrent_base,
    input  logic                                      i_srnn_recurrent_enable,
    input  logic signed [15:0]                       i_srnn_leak_q15,
    input  logic signed [31:0]                       i_srnn_threshold,
    input  logic [4:0]                                i_srnn_current_left_shift,
    input  logic                                      i_srnn_reset_to_zero,
    input  logic [VERSION_WIDTH-1:0]                  i_srnn_version,
    input  logic                                      i_srnn_event_valid,
    output logic                                      o_srnn_event_ready,
    input  logic [12:0]                               i_srnn_event_addr,
    input  logic signed [7:0]                        i_srnn_event_data,
    input  logic                                      i_srnn_timestep_last,
    input  logic                                      i_srnn_sample_last,
    output logic                                      o_srnn_weight_req_valid,
    input  logic                                      i_srnn_weight_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_srnn_weight_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_srnn_weight_req_id,
    output logic                                      o_srnn_weight_req_recurrent,
    output logic [31:0]                               o_srnn_weight_req_row_base,
    output logic [9:0]                                o_srnn_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_srnn_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_srnn_weight_req_lane_mask,
    input  logic                                      i_srnn_weight_rsp_valid,
    output logic                                      o_srnn_weight_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_srnn_weight_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_srnn_weight_rsp_id,
    input  logic [9:0]                                i_srnn_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_srnn_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_srnn_weight_rsp_flat,
    output logic                                      o_srnn_result_valid,
    input  logic                                      i_srnn_result_ready,
    output logic [VERSION_WIDTH-1:0]                  o_srnn_result_version,
    output logic                                      o_srnn_result_sample_last,
    output logic [HIDDEN_MAX-1:0]                     o_srnn_result_spikes,
    output logic [HIDDEN_MAX-1:0]                     o_srnn_previous_spikes,
    output logic                                      o_srnn_done,

    input  logic                                      i_scnn_start_valid,
    output logic                                      o_scnn_start_ready,
    input  logic [15:0]                               i_scnn_in_width,
    input  logic [15:0]                               i_scnn_in_height,
    input  logic [15:0]                               i_scnn_total_in_channels,
    input  logic [15:0]                               i_scnn_out_width,
    input  logic [15:0]                               i_scnn_out_height,
    input  logic [15:0]                               i_scnn_total_out_channels,
    input  logic [9:0]                                i_scnn_input_channel_base,
    input  logic [9:0]                                i_scnn_input_channel_count,
    input  logic [9:0]                                i_scnn_output_channel_base,
    input  logic [9:0]                                i_scnn_output_channel_count,
    input  logic [7:0]                                i_scnn_stride,
    input  logic [7:0]                                i_scnn_padding,
    input  logic                                      i_scnn_input_slot,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_scnn_feature_base,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_scnn_weight_q16_base,
    input  logic [OUTPUT_ADDR_WIDTH-1:0]              i_scnn_output_base,
    input  logic                                      i_scnn_unit_spike_features,
    input  logic                                      i_scnn_external_mode,
    output logic                                      o_scnn_feature_read_valid,
    input  logic                                      i_scnn_feature_read_ready,
    output logic                                      o_scnn_feature_read_slot,
    output logic [31:0]                               o_scnn_feature_read_bank_flat,
    output logic [LANES*FEATURE_ADDR_WIDTH-1:0]        o_scnn_feature_read_addr_flat,
    output logic [LANES-1:0]                          o_scnn_feature_read_lane_mask,
    input  logic                                      i_scnn_feature_valid,
    output logic                                      o_scnn_feature_ready,
    input  logic [LANES*16-1:0]                       i_scnn_feature_flat,
    input  logic [LANES-1:0]                          i_scnn_feature_lane_mask,
    output logic                                      o_scnn_weight_read_valid,
    input  logic                                      i_scnn_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_scnn_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_scnn_weight_read_addr1,
    input  logic                                      i_scnn_weight_result_valid,
    output logic                                      o_scnn_weight_result_ready,
    input  logic [63:0]                               i_scnn_weight_data0,
    input  logic [63:0]                               i_scnn_weight_data1,
    output logic                                      o_scnn_result_valid,
    input  logic                                      i_scnn_result_ready,
    output logic [ENGINES-1:0]                        o_scnn_result_mask,
    output logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      o_scnn_result_output_addr_flat,
    output logic [OUTPUT_ADDR_WIDTH-1:0]              o_scnn_result_spatial_addr,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_scnn_result_accumulator_flat,
    output logic                                      o_scnn_done,
    input  logic                                      i_scnn_external_request_valid,
    output logic                                      o_scnn_external_request_ready,
    input  logic [ENGINES-1:0]                        i_scnn_external_engine_mask,
    input  logic [LANES*16-1:0]                       i_scnn_external_feature_flat,
    input  logic [ENGINES*LANES*16-1:0]               i_scnn_external_weight_flat,
    input  logic [LANES-1:0]                          i_scnn_external_lane_mask,
    input  logic                                      i_scnn_external_first,
    input  logic                                      i_scnn_external_last,
    input  logic [9:0]                                i_scnn_external_output_channel_base,
    output logic                                      o_scnn_external_result_valid,
    input  logic                                      i_scnn_external_result_ready,
    output logic [ENGINES-1:0]                        o_scnn_external_result_mask,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_scnn_external_result_accumulator_flat,
    input  logic                                      i_scnn_external_done,
    input  logic                                      i_scnn_external_sticky_error,

    output logic [1:0]                                o_active_owner,
    output logic                                      o_shared_request_fire,
    output logic                                      o_shared_result_fire,
    output logic                                      o_scnn_zero_lane_normalized,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    logic i_fc_start_valid;
    logic o_fc_start_ready;
    logic i_fc_clear_state;
    logic [ENGINES-1:0] i_fc_active_tile_mask;
    logic i_fc_step_start;
    logic o_fc_step_ready;
    logic [8:0] i_fc_timestep;
    logic i_fc_row_valid;
    logic o_fc_row_ready;
    logic [LANES-1:0] i_fc_lane_valid;
    logic [ENGINES*LANES*40*5-1:0] i_fc_w0_weight_rows_flat;
    logic i_fc_step_close;
    logic [ENGINES*40*10*8-1:0] i_fc_w2_weight_flat;
    logic o_fc_result_valid;
    logic i_fc_result_ready;
    logic [8:0] o_fc_result_timestep;
    logic [31:0] o_fc_result_events;
    logic [ENGINES*40-1:0] o_fc_result_spikes;
    logic [10*24-1:0] o_fc_result_logits_flat;
    logic [3:0] o_fc_result_pred;
    logic i_fc_finish;
    logic o_fc_finish_ready;
    logic o_fc_done;
    logic shared_array_sticky_error;
    logic shared_owner_active;

    generate
        if (INCLUDE_FC_COMPAT) begin : g_fc_frontend
            logic [FC_TILES-1:0] fc_active_tile_mask;
            logic [FC_TILES*LANES*40*5-1:0] fc_w0_weight_rows_flat;
            logic [FC_TILES*40*10*8-1:0] fc_w2_weight_flat;
            logic [FC_TILES*40-1:0] fc_spikes_flat;

            assign i_fc_active_tile_mask = ENGINES'(fc_active_tile_mask);
            assign i_fc_w0_weight_rows_flat =
                (ENGINES*LANES*40*5)'(fc_w0_weight_rows_flat);
            assign i_fc_w2_weight_flat =
                (ENGINES*40*10*8)'(fc_w2_weight_flat);
            assign o_fc_spikes_flat = (ENGINES*40)'(fc_spikes_flat);

            unified_fc_compat_window_frontend_v1 #(
                .TILES(FC_TILES),
                .ENABLE_TP_NATIVE_WEIGHTS(ENABLE_FC_TP_NATIVE_WEIGHTS),
                .ENABLE_W0_LEARNING(FC_LEARN_W0),
                .TP_TENSOR_ID_WIDTH(FC_TP_TENSOR_ID_WIDTH),
                .TP_PARAM_ADDR_WIDTH(FC_TP_PARAM_ADDR_WIDTH),
                .TP_TAG_WIDTH(FC_TP_TAG_WIDTH),
                .TP_CODE_WIDTH(FC_TP_CODE_WIDTH),
                .TP_RESIDUAL_WIDTH(FC_TP_RESIDUAL_WIDTH),
                .WEIGHT_DIR(FC_WEIGHT_DIR),
                .W0_RESIDUAL_INIT_FILE(FC_W0_RESIDUAL_INIT_FILE)
            ) u_fc_frontend (
                .clk(i_clk),
                .rst_n(i_rst_n),
                .i_abort(i_abort),
                .i_clear_error(i_clear_error),
                .i_tp_w0_parameter_base(i_fc_tp_w0_parameter_base),
                .i_tp_w2_parameter_base(i_fc_tp_w2_parameter_base),
                .i_tp_learning_read_valid(i_fc_tp_learning_read_valid),
                .o_tp_learning_read_ready(o_fc_tp_learning_read_ready),
                .i_tp_learning_read_tensor_id(i_fc_tp_learning_read_tensor_id),
                .i_tp_learning_read_parameter_addr(
                    i_fc_tp_learning_read_parameter_addr),
                .i_tp_learning_read_tag(i_fc_tp_learning_read_tag),
                .o_tp_learning_response_valid(
                    o_fc_tp_learning_response_valid),
                .i_tp_learning_response_ready(
                    i_fc_tp_learning_response_ready),
                .o_tp_learning_response_tag(o_fc_tp_learning_response_tag),
                .o_tp_learning_response_code(o_fc_tp_learning_response_code),
                .o_tp_learning_response_residual(
                    o_fc_tp_learning_response_residual),
                .o_tp_learning_response_error(
                    o_fc_tp_learning_response_error),
                .i_tp_learning_write_valid(i_fc_tp_learning_write_valid),
                .o_tp_learning_write_ready(o_fc_tp_learning_write_ready),
                .i_tp_learning_write_tensor_id(
                    i_fc_tp_learning_write_tensor_id),
                .i_tp_learning_write_parameter_addr(
                    i_fc_tp_learning_write_parameter_addr),
                .i_tp_learning_write_code(i_fc_tp_learning_write_code),
                .i_tp_learning_write_residual(
                    i_fc_tp_learning_write_residual),
                .i_tp_learning_write_code_enable(
                    i_fc_tp_learning_write_code_enable),
                .i_tp_learning_write_residual_enable(
                    i_fc_tp_learning_write_residual_enable),
                .o_tp_learning_quiescent(o_fc_tp_learning_quiescent),
                .o_owner_start_valid(i_fc_start_valid),
                .i_owner_start_ready(o_fc_start_ready),
                .i_owner_enable(shared_owner_active &&
                    o_active_owner == 2'd1),
                .o_owner_finish(i_fc_finish),
                .i_owner_finish_ready(o_fc_finish_ready),
                .i_clear_state(i_fc_sample_clear_state),
                .i_sample_start(i_fc_sample_start),
                .o_sample_ready(o_fc_sample_ready),
                .i_active_input_dim(i_fc_active_input_dim),
                .i_active_tile_mask(i_fc_requested_tile_mask[FC_TILES-1:0]),
                .i_window_valid(i_fc_window_valid),
                .o_window_ready(o_fc_window_ready),
                .i_record_valid(i_fc_record_valid),
                .i_sample_id(i_fc_sample_id),
                .i_timestep(i_fc_window_timestep),
                .i_boundary_mask(i_fc_boundary_mask),
                .i_sample_last(i_fc_sample_last),
                .i_fi_flat(i_fc_fi_flat),
                .i_x_flat(i_fc_x_flat),
                .o_array_clear_state(i_fc_clear_state),
                .o_array_active_tile_mask(fc_active_tile_mask),
                .o_array_step_start(i_fc_step_start),
                .i_array_step_ready(o_fc_step_ready),
                .o_array_timestep(i_fc_timestep),
                .o_array_row_valid(i_fc_row_valid),
                .i_array_row_ready(o_fc_row_ready),
                .o_array_lane_valid(i_fc_lane_valid),
                .o_array_w0_weight_rows_flat(fc_w0_weight_rows_flat),
                .o_array_step_close(i_fc_step_close),
                .o_array_w2_weight_flat(fc_w2_weight_flat),
                .i_array_result_valid(o_fc_result_valid),
                .o_array_result_ready(i_fc_result_ready),
                .i_array_result_timestep(o_fc_result_timestep),
                .i_array_result_events(o_fc_result_events),
                .i_array_result_spikes(
                    o_fc_result_spikes[FC_TILES*40-1:0]),
                .i_array_result_logits_flat(o_fc_result_logits_flat),
                .i_array_vmem_flat(o_fc_vmem_flat[FC_TILES*40*18-1:0]),
                .i_array_trace_flat(
                    o_fc_trace_flat[FC_TILES*40*16-1:0]),
                .i_array_protocol_error(shared_array_sticky_error),
                .o_timestep_done(o_fc_timestep_done),
                .o_sample_done(o_fc_sample_done),
                .o_completed_timestep(o_fc_completed_timestep),
                .o_logits_flat(o_fc_logits_flat),
                .o_pred(o_fc_pred),
                .o_spikes_flat(fc_spikes_flat),
                .o_vmem_flat(),
                .o_trace_flat(),
                .o_accepted_windows(o_fc_accepted_windows),
                .o_accepted_events(o_fc_accepted_events),
                .o_issue_groups(o_fc_issue_groups),
                .o_boundary_count(o_fc_boundary_count),
                .o_runtime_dropped_events(o_fc_dropped_events),
                .o_sample_cycles(o_fc_sample_cycles),
                .o_global_idle(),
                .o_protocol_error(o_fc_protocol_error)
            );
        end else begin : g_no_fc_frontend
            assign i_fc_start_valid = 1'b0;
            assign i_fc_clear_state = 1'b0;
            assign i_fc_active_tile_mask = '0;
            assign i_fc_step_start = 1'b0;
            assign i_fc_timestep = '0;
            assign i_fc_row_valid = 1'b0;
            assign i_fc_lane_valid = '0;
            assign i_fc_w0_weight_rows_flat = '0;
            assign i_fc_step_close = 1'b0;
            assign i_fc_w2_weight_flat = '0;
            assign i_fc_result_ready = 1'b0;
            assign i_fc_finish = 1'b0;

            assign o_fc_sample_ready = 1'b0;
            assign o_fc_window_ready = 1'b0;
            assign o_fc_timestep_done = 1'b0;
            assign o_fc_sample_done = 1'b0;
            assign o_fc_completed_timestep = '0;
            assign o_fc_logits_flat = '0;
            assign o_fc_pred = '0;
            assign o_fc_spikes_flat = '0;
            assign o_fc_accepted_windows = '0;
            assign o_fc_accepted_events = '0;
            assign o_fc_issue_groups = '0;
            assign o_fc_boundary_count = '0;
            assign o_fc_dropped_events = '0;
            assign o_fc_sample_cycles = '0;
            assign o_fc_protocol_error = 1'b0;
            assign o_fc_tp_learning_read_ready = 1'b0;
            assign o_fc_tp_learning_response_valid = 1'b0;
            assign o_fc_tp_learning_response_tag = '0;
            assign o_fc_tp_learning_response_code = '0;
            assign o_fc_tp_learning_response_residual = '0;
            assign o_fc_tp_learning_response_error = 1'b0;
            assign o_fc_tp_learning_write_ready = 1'b0;
            assign o_fc_tp_learning_quiescent = 1'b1;
        end
    endgenerate

    unified_wave3_tile_major_shared_extension_leaf_v2 #(
        .ENGINES(ENGINES),
        .FC_TILES(FC_TILES),
        .LANES(LANES),
        .HIDDEN_MAX(HIDDEN_MAX),
        .INPUT_MAX(INPUT_MAX),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .INCLUDE_FC_COMPAT(INCLUDE_FC_COMPAT),
        .USE_SHARED_FC_SRNN_STATE(USE_SHARED_FC_SRNN_STATE)
    ) u_shared_leaf (
        .o_array_sticky_error(shared_array_sticky_error),
        .o_owner_active(shared_owner_active),
        .*
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic [3:0] unused_fc_result_pred;
    logic unused_fc_done;
    always_comb begin
        unused_fc_result_pred = o_fc_result_pred;
        unused_fc_done = o_fc_done;
    end
    /* verilator lint_on UNUSEDSIGNAL */

    initial begin
        if (FC_TILES < 1 || FC_TILES > ENGINES)
            $error("FC_TILES must be in the range 1..ENGINES");
    end
endmodule

`default_nettype wire
