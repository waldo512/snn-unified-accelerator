`default_nettype none

// Dataset-level SRNN closure for the Wave3 shared extension leaf. The only
// INT8 multiplier array is nested inside u_wave3. This wrapper adds the
// existing spike/W2 accumulator and dynamic output path without introducing
// another recurrent or generic compute fabric.
module unified_wave3_srnn_dataset_wrapper_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int OUTPUT_MAX = 32,
    parameter int VERSION_WIDTH = 16,
    parameter int REQUEST_ID_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_start_valid,
    output logic                                      o_start_ready,
    input  logic                                      i_state_reset,
    input  logic [12:0]                               i_input_dim,
    input  logic [9:0]                                i_active_neurons,
    input  logic [31:0]                               i_w0_base,
    input  logic [31:0]                               i_recurrent_base,
    input  logic                                      i_recurrent_enable,
    input  logic signed [15:0]                       i_leak_q15,
    input  logic signed [31:0]                       i_threshold,
    input  logic [4:0]                                i_current_left_shift,
    input  logic                                      i_reset_to_zero,
    input  logic [VERSION_WIDTH-1:0]                  i_version,

    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [12:0]                               i_event_addr,
    input  logic signed [7:0]                        i_event_data,
    input  logic                                      i_timestep_last,
    input  logic                                      i_sample_last,

    output logic                                      o_weight_req_valid,
    input  logic                                      i_weight_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_weight_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_weight_req_id,
    output logic                                      o_weight_req_recurrent,
    output logic [31:0]                               o_weight_req_row_base,
    output logic [9:0]                                o_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_weight_req_lane_mask,
    input  logic                                      i_weight_rsp_valid,
    output logic                                      o_weight_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_weight_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_weight_rsp_id,
    input  logic [9:0]                                i_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_weight_rsp_flat,

    output logic [ENGINES-1:0]                        o_w2_req_valid,
    input  logic [ENGINES-1:0]                        i_w2_req_ready,
    output logic [ENGINES*32-1:0]                     o_w2_addr_flat,
    input  logic [ENGINES*64-1:0]                     i_w2_data_flat,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [5:0]                                o_result_class,
    output logic signed [31:0]                       o_result_score,
    output logic [OUTPUT_MAX*32-1:0]                  o_result_logits_flat,
    output logic [VERSION_WIDTH-1:0]                  o_result_version,
    output logic                                      o_done,

    output logic                                      o_timestep_commit,
    output logic [VERSION_WIDTH-1:0]                  o_timestep_version,
    output logic                                      o_timestep_sample_last,
    output logic [HIDDEN_MAX-1:0]                     o_timestep_spikes,
    output logic [HIDDEN_MAX-1:0]                     o_previous_spikes,
    output logic                                      o_leaf_done,
    output logic [1:0]                                o_active_owner,
    output logic                                      o_shared_request_fire,
    output logic                                      o_shared_result_fire,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    logic leaf_start_valid;
    logic leaf_start_ready;
    logic leaf_result_valid;
    logic leaf_result_ready;
    logic [VERSION_WIDTH-1:0] leaf_result_version;
    logic leaf_result_sample_last;
    logic [HIDDEN_MAX-1:0] leaf_result_spikes;
    logic [HIDDEN_MAX-1:0] leaf_previous_spikes;
    logic leaf_busy;
    logic leaf_idle;
    logic leaf_sticky_error;
    logic [7:0] leaf_error_code;
    logic bridge_timestep_ready;
    logic bridge_result_valid;
    logic bridge_busy;
    logic bridge_sticky_error;
    logic [VERSION_WIDTH-1:0] locked_version_q;
    logic local_version_error_q;
    logic start_fire;

    assign o_start_ready = leaf_start_ready && !bridge_busy &&
        !bridge_result_valid && !local_version_error_q;
    assign leaf_start_valid = i_start_valid && !bridge_busy &&
        !bridge_result_valid && !local_version_error_q;
    assign start_fire = i_start_valid && o_start_ready;

    assign leaf_result_ready = bridge_timestep_ready;
    assign o_timestep_commit = leaf_result_valid && leaf_result_ready;
    assign o_timestep_version = leaf_result_version;
    assign o_timestep_sample_last = leaf_result_sample_last;
    assign o_timestep_spikes = leaf_result_spikes;
    assign o_previous_spikes = leaf_previous_spikes;

    assign o_result_valid = bridge_result_valid;
    assign o_result_version = locked_version_q;
    assign o_done = bridge_result_valid && i_result_ready;
    assign o_busy = leaf_busy || bridge_busy;
    assign o_idle = leaf_idle && !bridge_busy && !bridge_result_valid;
    assign o_sticky_error = local_version_error_q || leaf_sticky_error ||
        bridge_sticky_error;
    assign o_error_code = local_version_error_q ? 8'h90 :
        (leaf_sticky_error ? leaf_error_code :
         (bridge_sticky_error ? 8'h91 : 8'h00));

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            locked_version_q <= '0;
            local_version_error_q <= 1'b0;
        end else begin
            if (i_clear_error && o_idle)
                local_version_error_q <= 1'b0;
            if (start_fire)
                locked_version_q <= i_version;
            if (o_timestep_commit &&
                leaf_result_version != locked_version_q)
                local_version_error_q <= 1'b1;
        end
    end

    unified_wave3_shared_extension_leaf_v1 #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .HIDDEN_MAX(HIDDEN_MAX),
        .INPUT_MAX(INPUT_MAX),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH)
    ) u_wave3 (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_srnn_start_valid(leaf_start_valid),
        .o_srnn_start_ready(leaf_start_ready),
        .i_srnn_state_reset(i_state_reset),
        .i_srnn_input_dim(i_input_dim),
        .i_srnn_active_neurons(i_active_neurons),
        .i_srnn_w0_base(i_w0_base),
        .i_srnn_recurrent_base(i_recurrent_base),
        .i_srnn_recurrent_enable(i_recurrent_enable),
        .i_srnn_leak_q15(i_leak_q15),
        .i_srnn_threshold(i_threshold),
        .i_srnn_current_left_shift(i_current_left_shift),
        .i_srnn_reset_to_zero(i_reset_to_zero),
        .i_srnn_version(i_version),
        .i_srnn_event_valid(i_event_valid),
        .o_srnn_event_ready(o_event_ready),
        .i_srnn_event_addr(i_event_addr),
        .i_srnn_event_data(i_event_data),
        .i_srnn_timestep_last(i_timestep_last),
        .i_srnn_sample_last(i_sample_last),
        .o_srnn_weight_req_valid(o_weight_req_valid),
        .i_srnn_weight_req_ready(i_weight_req_ready),
        .o_srnn_weight_req_version(o_weight_req_version),
        .o_srnn_weight_req_id(o_weight_req_id),
        .o_srnn_weight_req_recurrent(o_weight_req_recurrent),
        .o_srnn_weight_req_row_base(o_weight_req_row_base),
        .o_srnn_weight_req_destination_base(
            o_weight_req_destination_base),
        .o_srnn_weight_req_engine_mask(o_weight_req_engine_mask),
        .o_srnn_weight_req_lane_mask(o_weight_req_lane_mask),
        .i_srnn_weight_rsp_valid(i_weight_rsp_valid),
        .o_srnn_weight_rsp_ready(o_weight_rsp_ready),
        .i_srnn_weight_rsp_version(i_weight_rsp_version),
        .i_srnn_weight_rsp_id(i_weight_rsp_id),
        .i_srnn_weight_rsp_destination_base(
            i_weight_rsp_destination_base),
        .i_srnn_weight_rsp_engine_mask(i_weight_rsp_engine_mask),
        .i_srnn_weight_rsp_flat(i_weight_rsp_flat),
        .o_srnn_result_valid(leaf_result_valid),
        .i_srnn_result_ready(leaf_result_ready),
        .o_srnn_result_version(leaf_result_version),
        .o_srnn_result_sample_last(leaf_result_sample_last),
        .o_srnn_result_spikes(leaf_result_spikes),
        .o_srnn_previous_spikes(leaf_previous_spikes),
        .o_srnn_done(o_leaf_done),

        .i_scnn_start_valid(1'b0),
        .o_scnn_start_ready(),
        .i_scnn_in_width('0),
        .i_scnn_in_height('0),
        .i_scnn_total_in_channels('0),
        .i_scnn_out_width('0),
        .i_scnn_out_height('0),
        .i_scnn_total_out_channels('0),
        .i_scnn_input_channel_base('0),
        .i_scnn_input_channel_count('0),
        .i_scnn_output_channel_base('0),
        .i_scnn_output_channel_count('0),
        .i_scnn_stride('0),
        .i_scnn_padding('0),
        .i_scnn_input_slot(1'b0),
        .i_scnn_feature_base('0),
        .i_scnn_weight_q16_base('0),
        .i_scnn_output_base('0),
        .i_scnn_unit_spike_features(1'b0),
        .i_scnn_external_mode(1'b0),
        .o_scnn_feature_read_valid(),
        .i_scnn_feature_read_ready(1'b1),
        .o_scnn_feature_read_slot(),
        .o_scnn_feature_read_bank_flat(),
        .o_scnn_feature_read_addr_flat(),
        .o_scnn_feature_read_lane_mask(),
        .i_scnn_feature_valid(1'b0),
        .o_scnn_feature_ready(),
        .i_scnn_feature_flat('0),
        .i_scnn_feature_lane_mask('0),
        .o_scnn_weight_read_valid(),
        .i_scnn_weight_read_ready(1'b1),
        .o_scnn_weight_read_addr0(),
        .o_scnn_weight_read_addr1(),
        .i_scnn_weight_result_valid(1'b0),
        .o_scnn_weight_result_ready(),
        .i_scnn_weight_data0('0),
        .i_scnn_weight_data1('0),
        .o_scnn_result_valid(),
        .i_scnn_result_ready(1'b1),
        .o_scnn_result_mask(),
        .o_scnn_result_output_addr_flat(),
        .o_scnn_result_spatial_addr(),
        .o_scnn_result_accumulator_flat(),
        .o_scnn_done(),
        .i_scnn_external_request_valid(1'b0),
        .o_scnn_external_request_ready(),
        .i_scnn_external_engine_mask('0),
        .i_scnn_external_feature_flat('0),
        .i_scnn_external_weight_flat('0),
        .i_scnn_external_lane_mask('0),
        .i_scnn_external_first(1'b0),
        .i_scnn_external_last(1'b0),
        .i_scnn_external_output_channel_base('0),
        .o_scnn_external_result_valid(),
        .i_scnn_external_result_ready(1'b0),
        .o_scnn_external_result_mask(),
        .o_scnn_external_result_accumulator_flat(),
        .i_scnn_external_done(1'b0),
        .i_scnn_external_sticky_error(1'b0),
        .o_active_owner(o_active_owner),
        .o_shared_request_fire(o_shared_request_fire),
        .o_shared_result_fire(o_shared_result_fire),
        .o_scnn_zero_lane_normalized(),
        .o_busy(leaf_busy),
        .o_idle(leaf_idle),
        .o_sticky_error(leaf_sticky_error),
        .o_error_code(leaf_error_code)
    );

    unified_recurrent_output_bridge #(
        .TILE_COUNT(ENGINES),
        .NEURONS_PER_TILE(40),
        .OUTPUT_MAX(OUTPUT_MAX),
        .ACC_BITS(32)
    ) u_output (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_abort(1'b0),
        .i_timestep_valid(leaf_result_valid),
        .o_timestep_ready(bridge_timestep_ready),
        .i_timestep_sample_last(leaf_result_sample_last),
        .i_spikes_flat(leaf_result_spikes),
        .i_active_tiles(4'd12),
        .i_output_count(6'd20),
        .i_w2_row_stride(6'd20),
        .i_w2_base(32'd0),
        .i_version(locked_version_q),
        .o_w2_req_valid(o_w2_req_valid),
        .i_w2_req_ready(i_w2_req_ready),
        .o_w2_addr_flat(o_w2_addr_flat),
        .i_w2_data_flat(i_w2_data_flat),
        .o_result_valid(bridge_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_class(o_result_class),
        .o_result_score(o_result_score),
        .o_result_logits_flat(o_result_logits_flat),
        .o_busy(bridge_busy),
        .o_sticky_error(bridge_sticky_error)
    );
endmodule

`default_nettype wire
