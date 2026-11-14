`timescale 1ns/1ps
`default_nettype none

// Binds the existing V2.3 descriptor/scheduler/finalize layer to the formal
// FC/SRNN/SCNN shared-H40 inference top. This module owns no numeric array.
module unified_scnn_shared_h40_layer_top_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int MAX_WIDTH = 32,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH),
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int SPATIAL_DEPTH = 32768,
    parameter int SPATIAL_ADDR_WIDTH = $clog2(SPATIAL_DEPTH),
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int ACC_WIDTH = 48,
    parameter int BANK_ADDR_WIDTH = FEATURE_ADDR_WIDTH,
    parameter bit INCLUDE_FC_COMPAT = 1'b1,
    parameter bit ENABLE_TP_OBSERVER = 1'b0
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_layer_valid,
    output logic                                      o_layer_ready,
    input  logic [383:0]                              i_layer_words,
    input  logic                                      i_input_slot,
    input  logic                                      i_output_slot,
    input  logic [VERSION_WIDTH-1:0]                  i_sample_version,

    output logic                                      o_feature_read_valid,
    input  logic                                      i_feature_read_ready,
    output logic                                      o_feature_read_slot,
    output logic [31:0]                               o_feature_read_bank_flat,
    output logic [LANES*FEATURE_ADDR_WIDTH-1:0]        o_feature_read_addr_flat,
    output logic [LANES-1:0]                          o_feature_read_lane_mask,
    input  logic                                      i_feature_valid,
    output logic                                      o_feature_ready,
    input  logic [LANES*16-1:0]                       i_feature_flat,
    input  logic [LANES-1:0]                          i_feature_lane_mask,

    output logic                                      o_weight_read_valid,
    input  logic                                      i_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_weight_read_addr1,
    input  logic                                      i_weight_result_valid,
    output logic                                      o_weight_result_ready,
    input  logic [63:0]                               i_weight_data0,
    input  logic [63:0]                               i_weight_data1,

    output logic                                      o_feature_write_valid,
    input  logic                                      i_feature_write_ready,
    output logic                                      o_feature_write_slot,
    output logic [3:0]                                o_feature_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]                o_feature_write_addr,
    output logic signed [15:0]                        o_feature_write_data,
    input  logic                                      i_observation_enable,
    output logic                                      o_observation_valid,
    input  logic                                      i_observation_ready,
    output logic [15:0]                               o_observation_channel,
    output logic [SPATIAL_ADDR_WIDTH-1:0]             o_observation_spatial_addr,
    output logic [STATE_ADDR_WIDTH-1:0]               o_observation_output_addr,
    output logic [VERSION_WIDTH-1:0]                  o_observation_sample_version,
    output logic signed [15:0]                        o_observation_current_q8_8,
    output logic [15:0]                               o_observation_spike_q8_8,
    output logic signed [15:0]                        o_observation_membrane_pre_q8_8,
    output logic signed [15:0]                        o_observation_membrane_next_q8_8,
    output logic [15:0]                               o_observation_surrogate_q8_8,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_config_error,
    output logic                                      o_protocol_error,
    output logic                                      o_state_error
);
    logic conv_start_valid;
    logic conv_start_ready;
    logic [15:0] conv_in_width;
    logic [15:0] conv_in_height;
    logic [15:0] conv_total_in_channels;
    logic [15:0] conv_out_width;
    logic [15:0] conv_out_height;
    logic [15:0] conv_total_out_channels;
    logic [9:0] conv_input_channel_base;
    logic [9:0] conv_input_channel_count;
    logic [9:0] conv_output_channel_base;
    logic [9:0] conv_output_channel_count;
    logic [7:0] conv_stride;
    logic [7:0] conv_padding;
    logic conv_input_slot;
    logic [FEATURE_ADDR_WIDTH-1:0] conv_feature_base;
    logic [Q16_ADDR_WIDTH-1:0] conv_weight_q16_base;
    logic [STATE_ADDR_WIDTH-1:0] conv_output_base;
    logic conv_unit_spike_features;
    logic conv_idle;
    logic conv_sticky_error;
    logic [7:0] conv_error_code;

    logic raw_result_valid;
    logic raw_result_ready;
    logic [ENGINES-1:0] raw_result_mask;
    logic [ENGINES*STATE_ADDR_WIDTH-1:0] raw_result_output_addr_flat;
    logic [STATE_ADDR_WIDTH-1:0] raw_result_spatial_addr;
    logic [ENGINES*ACC_WIDTH-1:0] raw_result_accumulator_flat;
    logic raw_result_fire;
    logic raw_result_group_first;
    logic raw_result_group_final;
    logic [VERSION_WIDTH-1:0] locked_sample_version;
    logic layer_done;
    logic layer_busy;
    logic layer_idle;
    logic layer_sticky_error;
    logic [7:0] layer_error_code;

    logic shared_busy;
    logic shared_idle;
    logic shared_sticky_error;
    logic [7:0] shared_error_code;
    logic shared_scnn_done;
    logic conv_done_pending_q;
    logic conv_done_to_layer_q;
    logic layer_done_seen_q;
    logic done_q;

    assign o_busy = layer_busy || shared_busy || layer_done_seen_q;
    assign o_done = done_q;
    assign o_config_error = layer_sticky_error;
    assign o_protocol_error = shared_sticky_error;
    assign o_state_error = 1'b0;
    assign conv_idle = shared_idle;
    assign conv_sticky_error = shared_sticky_error;
    assign conv_error_code = shared_error_code;

    // The shared top raises SCNN done when the result stream has drained, then
    // releases its owner. The legacy layer finalizer consumes done at the
    // owner-idle boundary, so retain each group completion until that boundary.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            conv_done_pending_q <= 1'b0;
            conv_done_to_layer_q <= 1'b0;
        end else begin
            conv_done_to_layer_q <= 1'b0;
            if (shared_scnn_done)
                conv_done_pending_q <= 1'b1;
            if ((conv_done_pending_q || shared_scnn_done) && shared_idle) begin
                conv_done_pending_q <= 1'b0;
                conv_done_to_layer_q <= 1'b1;
            end
        end
    end

    // The scheduler's done pulse precedes the complete drain of both the
    // finalize path and the shared owner. Report completion only after both
    // sides are idle so a following layer cannot overlap residual state.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            layer_done_seen_q <= 1'b0;
            done_q <= 1'b0;
        end else begin
            done_q <= 1'b0;
            if (layer_done)
                layer_done_seen_q <= 1'b1;
            if ((layer_done_seen_q || layer_done) && layer_idle &&
                shared_idle) begin
                layer_done_seen_q <= 1'b0;
                done_q <= 1'b1;
            end
        end
    end

    unified_scnn_external_shared_layer_v1 #(
        .ENGINES(ENGINES), .LANES(LANES), .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH), .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .ENABLE_TP_OBSERVER(ENABLE_TP_OBSERVER)
    ) u_layer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_layer_valid(i_layer_valid), .o_layer_ready(o_layer_ready),
        .i_layer_words(i_layer_words), .i_input_slot(i_input_slot),
        .i_output_slot(i_output_slot), .i_sample_version(i_sample_version),
        .o_conv_start_valid(conv_start_valid),
        .i_conv_start_ready(conv_start_ready),
        .o_conv_in_width(conv_in_width),
        .o_conv_in_height(conv_in_height),
        .o_conv_total_in_channels(conv_total_in_channels),
        .o_conv_out_width(conv_out_width),
        .o_conv_out_height(conv_out_height),
        .o_conv_total_out_channels(conv_total_out_channels),
        .o_conv_input_channel_base(conv_input_channel_base),
        .o_conv_input_channel_count(conv_input_channel_count),
        .o_conv_output_channel_base(conv_output_channel_base),
        .o_conv_output_channel_count(conv_output_channel_count),
        .o_conv_stride(conv_stride), .o_conv_padding(conv_padding),
        .o_conv_input_slot(conv_input_slot),
        .o_conv_feature_base(conv_feature_base),
        .o_conv_weight_q16_base(conv_weight_q16_base),
        .o_conv_output_base(conv_output_base),
        .o_conv_unit_spike_features(conv_unit_spike_features),
        .i_conv_done(conv_done_to_layer_q),
        .i_conv_idle(conv_idle),
        .i_conv_sticky_error(conv_sticky_error),
        .i_conv_error_code(conv_error_code),
        .i_raw_result_valid(raw_result_valid),
        .o_raw_result_ready(raw_result_ready),
        .i_raw_result_mask(raw_result_mask),
        .i_raw_result_output_addr_flat(raw_result_output_addr_flat),
        .i_raw_result_spatial_addr(raw_result_spatial_addr),
        .i_raw_result_accumulator_flat(raw_result_accumulator_flat),
        .o_raw_result_fire(raw_result_fire),
        .o_raw_result_group_first(raw_result_group_first),
        .o_raw_result_group_final(raw_result_group_final),
        .o_feature_write_valid(o_feature_write_valid),
        .i_feature_write_ready(i_feature_write_ready),
        .o_feature_write_slot(o_feature_write_slot),
        .o_feature_write_bank(o_feature_write_bank),
        .o_feature_write_addr(o_feature_write_addr),
        .o_feature_write_data_q8_8(o_feature_write_data),
        .i_observation_enable(i_observation_enable),
        .o_observation_valid(o_observation_valid),
        .i_observation_ready(i_observation_ready),
        .o_observation_channel(o_observation_channel),
        .o_observation_spatial_addr(o_observation_spatial_addr),
        .o_observation_output_addr(o_observation_output_addr),
        .o_observation_sample_version(o_observation_sample_version),
        .o_observation_current_q8_8(o_observation_current_q8_8),
        .o_observation_spike_q8_8(o_observation_spike_q8_8),
        .o_observation_membrane_pre_q8_8(
            o_observation_membrane_pre_q8_8),
        .o_observation_membrane_next_q8_8(
            o_observation_membrane_next_q8_8),
        .o_observation_surrogate_q8_8(o_observation_surrogate_q8_8),
        .o_locked_sample_version(locked_sample_version),
        .o_layer_done(layer_done), .o_busy(layer_busy), .o_idle(layer_idle),
        .o_sticky_error(layer_sticky_error),
        .o_error_code(layer_error_code)
    );

    unified_shared_h40_inference_top_v1 #(
        .ENGINES(ENGINES), .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .INCLUDE_FC_COMPAT(INCLUDE_FC_COMPAT)
    ) u_shared_top (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_abort(1'b0),
        .i_clear_error(i_clear_error),

        .i_fc_sample_start(1'b0), .i_fc_sample_clear_state(1'b0),
        .i_fc_active_input_dim(13'd1),
        .i_fc_requested_tile_mask({{(ENGINES-1){1'b0}}, 1'b1}),
        .i_fc_window_valid(1'b0), .i_fc_record_valid('0),
        .i_fc_sample_id('0), .i_fc_window_timestep('0),
        .i_fc_boundary_mask('0), .i_fc_sample_last(1'b0),
        .i_fc_fi_flat('0), .i_fc_x_flat('0),
        .o_fc_sample_ready(), .o_fc_window_ready(),
        .o_fc_timestep_done(), .o_fc_sample_done(),
        .o_fc_completed_timestep(), .o_fc_logits_flat(), .o_fc_pred(),
        .o_fc_spikes_flat(), .o_fc_vmem_flat(), .o_fc_trace_flat(),
        .o_fc_accepted_windows(), .o_fc_accepted_events(),
        .o_fc_issue_groups(), .o_fc_boundary_count(),
        .o_fc_dropped_events(), .o_fc_sample_cycles(),
        .o_fc_protocol_error(),
        .i_fc_tp_w0_parameter_base('0),
        .i_fc_tp_w2_parameter_base('0),
        .i_fc_tp_learning_read_valid(1'b0),
        .o_fc_tp_learning_read_ready(),
        .i_fc_tp_learning_read_tensor_id('0),
        .i_fc_tp_learning_read_parameter_addr('0),
        .i_fc_tp_learning_read_tag('0),
        .o_fc_tp_learning_response_valid(),
        .i_fc_tp_learning_response_ready(1'b0),
        .o_fc_tp_learning_response_tag(),
        .o_fc_tp_learning_response_code(),
        .o_fc_tp_learning_response_residual(),
        .o_fc_tp_learning_response_error(),
        .i_fc_tp_learning_write_valid(1'b0),
        .o_fc_tp_learning_write_ready(),
        .i_fc_tp_learning_write_tensor_id('0),
        .i_fc_tp_learning_write_parameter_addr('0),
        .i_fc_tp_learning_write_code('0),
        .i_fc_tp_learning_write_residual('0),
        .i_fc_tp_learning_write_code_enable(1'b0),
        .i_fc_tp_learning_write_residual_enable(1'b0),
        .o_fc_tp_learning_quiescent(),

        .i_srnn_start_valid(1'b0), .i_srnn_state_reset(1'b0),
        .i_srnn_input_dim(13'd1), .i_srnn_active_neurons(10'd1),
        .i_srnn_w0_base('0), .i_srnn_recurrent_base('0),
        .i_srnn_recurrent_enable(1'b0), .i_srnn_leak_q15('0),
        .i_srnn_threshold(32'sd1), .i_srnn_current_left_shift('0),
        .i_srnn_reset_to_zero(1'b0), .i_srnn_version(16'd1),
        .i_srnn_event_valid(1'b0), .i_srnn_event_addr('0),
        .i_srnn_event_data('0), .i_srnn_timestep_last(1'b0),
        .i_srnn_sample_last(1'b0), .i_srnn_weight_req_ready(1'b0),
        .i_srnn_weight_rsp_valid(1'b0),
        .i_srnn_weight_rsp_version('0), .i_srnn_weight_rsp_id('0),
        .i_srnn_weight_rsp_destination_base('0),
        .i_srnn_weight_rsp_engine_mask('0),
        .i_srnn_weight_rsp_flat('0), .i_srnn_result_ready(1'b1),
        .o_srnn_start_ready(), .o_srnn_event_ready(),
        .o_srnn_weight_req_valid(), .o_srnn_weight_req_version(),
        .o_srnn_weight_req_id(), .o_srnn_weight_req_recurrent(),
        .o_srnn_weight_req_row_base(),
        .o_srnn_weight_req_destination_base(),
        .o_srnn_weight_req_engine_mask(), .o_srnn_weight_req_lane_mask(),
        .o_srnn_weight_rsp_ready(), .o_srnn_result_valid(),
        .o_srnn_result_version(), .o_srnn_result_sample_last(),
        .o_srnn_result_spikes(), .o_srnn_previous_spikes(), .o_srnn_done(),

        .i_scnn_start_valid(conv_start_valid),
        .o_scnn_start_ready(conv_start_ready),
        .i_scnn_in_width(conv_in_width),
        .i_scnn_in_height(conv_in_height),
        .i_scnn_total_in_channels(conv_total_in_channels),
        .i_scnn_out_width(conv_out_width),
        .i_scnn_out_height(conv_out_height),
        .i_scnn_total_out_channels(conv_total_out_channels),
        .i_scnn_input_channel_base(conv_input_channel_base),
        .i_scnn_input_channel_count(conv_input_channel_count),
        .i_scnn_output_channel_base(conv_output_channel_base),
        .i_scnn_output_channel_count(conv_output_channel_count),
        .i_scnn_stride(conv_stride), .i_scnn_padding(conv_padding),
        .i_scnn_input_slot(conv_input_slot),
        .i_scnn_feature_base(conv_feature_base),
        .i_scnn_weight_q16_base(conv_weight_q16_base),
        .i_scnn_output_base(conv_output_base),
        .i_scnn_unit_spike_features(conv_unit_spike_features),
        .i_scnn_external_mode(1'b0),
        .o_scnn_feature_read_valid(o_feature_read_valid),
        .i_scnn_feature_read_ready(i_feature_read_ready),
        .o_scnn_feature_read_slot(o_feature_read_slot),
        .o_scnn_feature_read_bank_flat(o_feature_read_bank_flat),
        .o_scnn_feature_read_addr_flat(o_feature_read_addr_flat),
        .o_scnn_feature_read_lane_mask(o_feature_read_lane_mask),
        .i_scnn_feature_valid(i_feature_valid),
        .o_scnn_feature_ready(o_feature_ready),
        .i_scnn_feature_flat(i_feature_flat),
        .i_scnn_feature_lane_mask(i_feature_lane_mask),
        .o_scnn_weight_read_valid(o_weight_read_valid),
        .i_scnn_weight_read_ready(i_weight_read_ready),
        .o_scnn_weight_read_addr0(o_weight_read_addr0),
        .o_scnn_weight_read_addr1(o_weight_read_addr1),
        .i_scnn_weight_result_valid(i_weight_result_valid),
        .o_scnn_weight_result_ready(o_weight_result_ready),
        .i_scnn_weight_data0(i_weight_data0),
        .i_scnn_weight_data1(i_weight_data1),
        .o_scnn_result_valid(raw_result_valid),
        .i_scnn_result_ready(raw_result_ready),
        .o_scnn_result_mask(raw_result_mask),
        .o_scnn_result_output_addr_flat(raw_result_output_addr_flat),
        .o_scnn_result_spatial_addr(raw_result_spatial_addr),
        .o_scnn_result_accumulator_flat(raw_result_accumulator_flat),
        .o_scnn_done(shared_scnn_done),
        .i_scnn_external_request_valid(1'b0),
        .i_scnn_external_engine_mask('0),
        .i_scnn_external_feature_flat('0),
        .i_scnn_external_weight_flat('0),
        .i_scnn_external_lane_mask('0),
        .i_scnn_external_first(1'b0),
        .i_scnn_external_last(1'b0),
        .i_scnn_external_output_channel_base('0),
        .i_scnn_external_result_ready(1'b1),
        .i_scnn_external_done(1'b0),
        .i_scnn_external_sticky_error(1'b0),
        .o_scnn_external_request_ready(),
        .o_scnn_external_result_valid(),
        .o_scnn_external_result_mask(),
        .o_scnn_external_result_accumulator_flat(),
        .o_active_owner(), .o_shared_request_fire(),
        .o_shared_result_fire(), .o_scnn_zero_lane_normalized(),
        .o_busy(shared_busy), .o_idle(shared_idle),
        .o_sticky_error(shared_sticky_error),
        .o_error_code(shared_error_code)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_observation;
    always_comb begin
        unused_observation = layer_idle ^ raw_result_fire ^
            raw_result_group_first ^ raw_result_group_final ^
            locked_sample_version[0] ^ layer_error_code[0];
    end
    /* verilator lint_on UNUSEDSIGNAL */
endmodule

`default_nettype wire
