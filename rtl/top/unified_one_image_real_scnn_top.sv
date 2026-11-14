`default_nettype none

// One-image ownership boundary with the real V2.3 SCNN layer executor.
// Canonical FC, recurrent FC and OUTPUT owners remain external interfaces.
module unified_one_image_real_scnn_top #(
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
    parameter int BANK_ADDR_WIDTH = FEATURE_ADDR_WIDTH,
    parameter bit EXPLICIT_SAMPLE_BOUNDARY = 1'b0
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [1:0]                                i_cfg_profile,
    input  logic [1:0]                                i_cfg_table,
    input  logic [8:0]                                i_cfg_index,
    input  logic [31:0]                               i_cfg_data,
    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [1:0]                                i_commit_profile,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic                                      i_sample_boundary,
    input  logic [15:0]                               i_model_version,
    input  logic [15:0]                               i_adapt_version,
    input  logic                                      i_clear_error,

    output logic                                      o_fc_layer_valid,
    input  logic                                      i_fc_layer_ready,
    input  logic                                      i_fc_layer_done,
    input  logic                                      i_fc_error,
    output logic                                      o_recurrent_layer_valid,
    input  logic                                      i_recurrent_layer_ready,
    input  logic                                      i_recurrent_layer_done,
    input  logic                                      i_recurrent_error,
    output logic                                      o_output_layer_valid,
    input  logic                                      i_output_layer_ready,
    input  logic                                      i_output_layer_done,
    input  logic                                      i_output_error,

    output logic [3:0]                                o_layer_index,
    output logic [383:0]                              o_layer_words,
    output logic [3:0]                                o_inbound_valid,
    output logic [15:0]                               o_inbound_sources_flat,
    output logic [15:0]                               o_inbound_types_flat,

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

    output logic                                      o_busy,
    output logic                                      o_sample_done,
    output logic [4:0]                                o_active_layer_count,
    output logic [63:0]                               o_active_versions,
    output logic [VERSION_WIDTH-1:0]                  o_sample_epoch,
    output logic                                      o_scnn_busy,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    logic scnn_layer_valid;
    logic scnn_layer_ready;
    logic scnn_layer_done;
    logic scnn_config_error;
    logic scnn_protocol_error;
    logic scnn_state_error;
    logic scnn_error;
    logic [VERSION_WIDTH-1:0] sample_epoch_q;
    logic sample_start_fire;
    logic sample_epoch_advance;

    assign scnn_error = scnn_config_error || scnn_protocol_error ||
                        scnn_state_error;
    assign sample_start_fire = i_start && o_start_ready;
    assign sample_epoch_advance = sample_start_fire &&
        (!EXPLICIT_SAMPLE_BOUNDARY || i_sample_boundary);
    assign o_sample_epoch = sample_epoch_q;

    unified_one_image_control_shell u_control (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready), .i_cfg_profile(i_cfg_profile),
        .i_cfg_table(i_cfg_table), .i_cfg_index(i_cfg_index),
        .i_cfg_data(i_cfg_data), .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready), .i_commit_profile(i_commit_profile),
        .i_start(i_start), .o_start_ready(o_start_ready),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error), .o_fc_layer_valid(o_fc_layer_valid),
        .i_fc_layer_ready(i_fc_layer_ready),
        .i_fc_layer_done(i_fc_layer_done),
        .o_recurrent_layer_valid(o_recurrent_layer_valid),
        .i_recurrent_layer_ready(i_recurrent_layer_ready),
        .i_recurrent_layer_done(i_recurrent_layer_done),
        .o_scnn_layer_valid(scnn_layer_valid),
        .i_scnn_layer_ready(scnn_layer_ready),
        .i_scnn_layer_done(scnn_layer_done),
        .o_output_layer_valid(o_output_layer_valid),
        .i_output_layer_ready(i_output_layer_ready),
        .i_output_layer_done(i_output_layer_done),
        .i_fc_error(i_fc_error), .i_recurrent_error(i_recurrent_error),
        .i_scnn_error(scnn_error), .i_output_error(i_output_error),
        .o_layer_index(o_layer_index), .o_layer_words(o_layer_words),
        .o_inbound_valid(o_inbound_valid),
        .o_inbound_sources_flat(o_inbound_sources_flat),
        .o_inbound_types_flat(o_inbound_types_flat), .o_busy(o_busy),
        .o_sample_done(o_sample_done),
        .o_active_layer_count(o_active_layer_count),
        .o_active_versions(o_active_versions),
        .o_sticky_error(o_sticky_error), .o_error_code(o_error_code)
    );

`ifdef SHARED_H40_SCNN_GRAPH
    unified_scnn_shared_h40_layer_top_v1 #(
`else
    unified_scnn_real_v23_layer_execution_top #(
`endif
        .ENGINES(ENGINES), .LANES(LANES), .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
`ifdef SHARED_H40_SCNN_GRAPH
        , .INCLUDE_FC_COMPAT(1'b0)
`endif
    ) u_scnn (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_layer_valid(scnn_layer_valid), .o_layer_ready(scnn_layer_ready),
        .i_layer_words(o_layer_words), .i_input_slot(o_layer_index[0]),
        .i_output_slot(!o_layer_index[0]),
        .i_sample_version(sample_epoch_q),
        .o_feature_read_valid(o_feature_read_valid),
        .i_feature_read_ready(i_feature_read_ready),
        .o_feature_read_slot(o_feature_read_slot),
        .o_feature_read_bank_flat(o_feature_read_bank_flat),
        .o_feature_read_addr_flat(o_feature_read_addr_flat),
        .o_feature_read_lane_mask(o_feature_read_lane_mask),
        .i_feature_valid(i_feature_valid), .o_feature_ready(o_feature_ready),
        .i_feature_flat(i_feature_flat),
        .i_feature_lane_mask(i_feature_lane_mask),
        .o_weight_read_valid(o_weight_read_valid),
        .i_weight_read_ready(i_weight_read_ready),
        .o_weight_read_addr0(o_weight_read_addr0),
        .o_weight_read_addr1(o_weight_read_addr1),
        .i_weight_result_valid(i_weight_result_valid),
        .o_weight_result_ready(o_weight_result_ready),
        .i_weight_data0(i_weight_data0), .i_weight_data1(i_weight_data1),
        .o_feature_write_valid(o_feature_write_valid),
        .i_feature_write_ready(i_feature_write_ready),
        .o_feature_write_slot(o_feature_write_slot),
        .o_feature_write_bank(o_feature_write_bank),
        .o_feature_write_addr(o_feature_write_addr),
        .o_feature_write_data(o_feature_write_data),
`ifdef SHARED_H40_SCNN_GRAPH
        /* verilator lint_off PINCONNECTEMPTY */
        .i_observation_enable(1'b0), .o_observation_valid(),
        .i_observation_ready(1'b1), .o_observation_channel(),
        .o_observation_spatial_addr(), .o_observation_output_addr(),
        .o_observation_sample_version(), .o_observation_current_q8_8(),
        .o_observation_spike_q8_8(),
        .o_observation_membrane_pre_q8_8(),
        .o_observation_membrane_next_q8_8(),
        .o_observation_surrogate_q8_8(),
        /* verilator lint_on PINCONNECTEMPTY */
`endif
        .o_busy(o_scnn_busy),
        .o_done(scnn_layer_done), .o_config_error(scnn_config_error),
        .o_protocol_error(scnn_protocol_error),
        .o_state_error(scnn_state_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            sample_epoch_q <= '0;
        end else if (sample_epoch_advance) begin
            if (&sample_epoch_q)
                sample_epoch_q <= {{(VERSION_WIDTH-1){1'b0}}, 1'b1};
            else
                sample_epoch_q <= sample_epoch_q + 1'b1;
        end
    end
endmodule

`default_nettype wire
