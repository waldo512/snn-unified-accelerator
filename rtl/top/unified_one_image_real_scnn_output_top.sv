`default_nettype none

// Adds the real SCNN LI OUTPUT owner to the one-image CONV executor while
// preserving external canonical FC/recurrent OUTPUT ownership. A start runs
// one timestep; i_sample_boundary advances the shared sample epoch and resets
// the SCNN classifier only at a recording boundary.
module unified_one_image_real_scnn_output_top #(
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
    parameter int SCNN_OUTPUT_FEATURES = 2048,
    parameter int OUTPUT_MAX = 32
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

    output logic                                      o_scnn_result_valid,
    input  logic                                      i_scnn_result_ready,
    output logic [$clog2(OUTPUT_MAX)-1:0]             o_scnn_result_class,
    output logic signed [15:0]                        o_scnn_result_score_q8_8,
    output logic                                      o_busy,
    output logic                                      o_sample_done,
    output logic [4:0]                                o_active_layer_count,
    output logic [63:0]                               o_active_versions,
    output logic [VERSION_WIDTH-1:0]                  o_sample_epoch,
    output logic                                      o_scnn_busy,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    logic control_output_valid;
    logic base_output_ready;
    logic base_output_done;
    logic base_output_error;
    logic scnn_output_contract;
    logic scnn_output_active_q;
    logic scnn_output_layer_ready;
    logic scnn_output_done;
    logic scnn_output_busy;
    logic scnn_output_config_error;
    logic scnn_output_protocol_error;
    logic output_memory_owner;
    logic frame_boundary_q;

    logic conv_feature_read_valid;
    logic conv_feature_read_ready;
    logic conv_feature_read_slot;
    logic [31:0] conv_feature_read_bank_flat;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] conv_feature_read_addr_flat;
    logic [LANES-1:0] conv_feature_read_lane_mask;
    logic conv_feature_ready;
    logic conv_weight_read_valid;
    logic conv_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0] conv_weight_read_addr0;
    logic [Q64_ADDR_WIDTH-1:0] conv_weight_read_addr1;
    logic conv_weight_result_ready;

    logic out_feature_read_valid;
    logic out_feature_read_ready;
    logic out_feature_read_slot;
    logic [31:0] out_feature_read_bank_flat;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] out_feature_read_addr_flat;
    logic [LANES-1:0] out_feature_read_lane_mask;
    logic out_feature_ready;
    logic out_weight_read_valid;
    logic out_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0] out_weight_read_addr0;
    logic [Q64_ADDR_WIDTH-1:0] out_weight_read_addr1;
    logic out_weight_result_ready;
    logic base_scnn_busy;

    assign scnn_output_contract = o_layer_words[2*32 +: 4] == 4'd4 &&
        o_layer_words[1*32 +: 32] == 32'h000c0202 &&
        o_layer_words[3*32 +: 13] == 13'(SCNN_OUTPUT_FEATURES) &&
        o_layer_words[7*32 +: 32] != 0;
    assign base_output_ready = scnn_output_contract ?
                               scnn_output_layer_ready : i_output_layer_ready;
    assign base_output_done = scnn_output_active_q ? scnn_output_done :
                              i_output_layer_done;
    assign base_output_error = scnn_output_active_q ?
        (scnn_output_config_error || scnn_output_protocol_error) :
        i_output_error;
    assign o_output_layer_valid = control_output_valid &&
                                  !scnn_output_contract;
    assign output_memory_owner = scnn_output_active_q ||
        (control_output_valid && scnn_output_contract);
    assign o_scnn_busy = base_scnn_busy || scnn_output_busy;

    assign o_feature_read_valid = output_memory_owner ?
        out_feature_read_valid : conv_feature_read_valid;
    assign o_feature_read_slot = output_memory_owner ?
        out_feature_read_slot : conv_feature_read_slot;
    assign o_feature_read_bank_flat = output_memory_owner ?
        out_feature_read_bank_flat : conv_feature_read_bank_flat;
    assign o_feature_read_addr_flat = output_memory_owner ?
        out_feature_read_addr_flat : conv_feature_read_addr_flat;
    assign o_feature_read_lane_mask = output_memory_owner ?
        out_feature_read_lane_mask : conv_feature_read_lane_mask;
    assign conv_feature_read_ready = !output_memory_owner &&
                                     i_feature_read_ready;
    assign out_feature_read_ready = output_memory_owner &&
                                    i_feature_read_ready;
    assign o_feature_ready = output_memory_owner ? out_feature_ready :
                                                   conv_feature_ready;

    assign o_weight_read_valid = output_memory_owner ?
        out_weight_read_valid : conv_weight_read_valid;
    assign o_weight_read_addr0 = output_memory_owner ?
        out_weight_read_addr0 : conv_weight_read_addr0;
    assign o_weight_read_addr1 = output_memory_owner ?
        out_weight_read_addr1 : conv_weight_read_addr1;
    assign conv_weight_read_ready = !output_memory_owner &&
                                    i_weight_read_ready;
    assign out_weight_read_ready = output_memory_owner &&
                                   i_weight_read_ready;
    assign o_weight_result_ready = output_memory_owner ?
        out_weight_result_ready : conv_weight_result_ready;

    unified_one_image_real_scnn_top #(
        .ENGINES(ENGINES), .LANES(LANES), .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .EXPLICIT_SAMPLE_BOUNDARY(1'b1)
    ) u_base (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_cfg_valid(i_cfg_valid),
        .o_cfg_ready(o_cfg_ready), .i_cfg_profile(i_cfg_profile),
        .i_cfg_table(i_cfg_table), .i_cfg_index(i_cfg_index),
        .i_cfg_data(i_cfg_data), .i_commit_valid(i_commit_valid),
        .o_commit_ready(o_commit_ready), .i_commit_profile(i_commit_profile),
        .i_start(i_start), .o_start_ready(o_start_ready),
        .i_sample_boundary(i_sample_boundary),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_clear_error(i_clear_error), .o_fc_layer_valid(o_fc_layer_valid),
        .i_fc_layer_ready(i_fc_layer_ready),
        .i_fc_layer_done(i_fc_layer_done), .i_fc_error(i_fc_error),
        .o_recurrent_layer_valid(o_recurrent_layer_valid),
        .i_recurrent_layer_ready(i_recurrent_layer_ready),
        .i_recurrent_layer_done(i_recurrent_layer_done),
        .i_recurrent_error(i_recurrent_error),
        .o_output_layer_valid(control_output_valid),
        .i_output_layer_ready(base_output_ready),
        .i_output_layer_done(base_output_done),
        .i_output_error(base_output_error), .o_layer_index(o_layer_index),
        .o_layer_words(o_layer_words), .o_inbound_valid(o_inbound_valid),
        .o_inbound_sources_flat(o_inbound_sources_flat),
        .o_inbound_types_flat(o_inbound_types_flat),
        .o_feature_read_valid(conv_feature_read_valid),
        .i_feature_read_ready(conv_feature_read_ready),
        .o_feature_read_slot(conv_feature_read_slot),
        .o_feature_read_bank_flat(conv_feature_read_bank_flat),
        .o_feature_read_addr_flat(conv_feature_read_addr_flat),
        .o_feature_read_lane_mask(conv_feature_read_lane_mask),
        .i_feature_valid(i_feature_valid), .o_feature_ready(conv_feature_ready),
        .i_feature_flat(i_feature_flat),
        .i_feature_lane_mask(i_feature_lane_mask),
        .o_weight_read_valid(conv_weight_read_valid),
        .i_weight_read_ready(conv_weight_read_ready),
        .o_weight_read_addr0(conv_weight_read_addr0),
        .o_weight_read_addr1(conv_weight_read_addr1),
        .i_weight_result_valid(i_weight_result_valid),
        .o_weight_result_ready(conv_weight_result_ready),
        .i_weight_data0(i_weight_data0), .i_weight_data1(i_weight_data1),
        .o_feature_write_valid(o_feature_write_valid),
        .i_feature_write_ready(i_feature_write_ready),
        .o_feature_write_slot(o_feature_write_slot),
        .o_feature_write_bank(o_feature_write_bank),
        .o_feature_write_addr(o_feature_write_addr),
        .o_feature_write_data(o_feature_write_data), .o_busy(o_busy),
        .o_sample_done(o_sample_done),
        .o_active_layer_count(o_active_layer_count),
        .o_active_versions(o_active_versions), .o_sample_epoch(o_sample_epoch),
        .o_scnn_busy(base_scnn_busy), .o_sticky_error(o_sticky_error),
        .o_error_code(o_error_code)
    );

    unified_scnn_q8_8_output_layer_top #(
        .LANES(LANES), .INPUT_FEATURES(SCNN_OUTPUT_FEATURES),
        .OUTPUT_MAX(OUTPUT_MAX), .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH)
    ) u_scnn_output (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_layer_valid(control_output_valid && scnn_output_contract),
        .o_layer_ready(scnn_output_layer_ready),
        .i_layer_words(o_layer_words), .i_input_slot(o_layer_index[0]),
        .i_sample_reset(frame_boundary_q),
        .o_feature_read_valid(out_feature_read_valid),
        .i_feature_read_ready(out_feature_read_ready),
        .o_feature_read_slot(out_feature_read_slot),
        .o_feature_read_bank_flat(out_feature_read_bank_flat),
        .o_feature_read_addr_flat(out_feature_read_addr_flat),
        .o_feature_read_lane_mask(out_feature_read_lane_mask),
        .i_feature_valid(i_feature_valid), .o_feature_ready(out_feature_ready),
        .i_feature_flat(i_feature_flat),
        .i_feature_lane_mask(i_feature_lane_mask),
        .o_weight_read_valid(out_weight_read_valid),
        .i_weight_read_ready(out_weight_read_ready),
        .o_weight_read_addr0(out_weight_read_addr0),
        .o_weight_read_addr1(out_weight_read_addr1),
        .i_weight_result_valid(i_weight_result_valid),
        .o_weight_result_ready(out_weight_result_ready),
        .i_weight_data0(i_weight_data0), .i_weight_data1(i_weight_data1),
        .o_result_valid(o_scnn_result_valid),
        .i_result_ready(i_scnn_result_ready),
        .o_result_class(o_scnn_result_class),
        .o_result_score_q8_8(o_scnn_result_score_q8_8),
        .o_busy(scnn_output_busy), .o_done(scnn_output_done),
        .o_config_error(scnn_output_config_error),
        .o_protocol_error(scnn_output_protocol_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            scnn_output_active_q <= 1'b0;
            frame_boundary_q <= 1'b0;
        end else begin
            if (i_start && o_start_ready)
                frame_boundary_q <= i_sample_boundary;
            if (control_output_valid && scnn_output_contract &&
                scnn_output_layer_ready)
                scnn_output_active_q <= 1'b1;
            if (scnn_output_done || scnn_output_config_error ||
                scnn_output_protocol_error)
                scnn_output_active_q <= 1'b0;
            if (o_sample_done)
                frame_boundary_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
