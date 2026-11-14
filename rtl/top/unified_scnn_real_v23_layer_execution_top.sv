`default_nettype none

// V2.3 CONV-layer execution boundary using the real twelve-engine wave path.
// Canonical FC is external and remains the sole SPARSE_FC datapath.
module unified_scnn_real_v23_layer_execution_top #(
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
    parameter int BANK_ADDR_WIDTH = FEATURE_ADDR_WIDTH
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
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_config_error,
    output logic                                      o_protocol_error,
    output logic                                      o_state_error
);
    logic engine_start;
    logic engine_start_ready;
    logic engine_done;
    logic [15:0] in_width;
    logic [15:0] in_height;
    logic [15:0] in_channels;
    logic [15:0] out_width;
    logic [15:0] out_height;
    logic [15:0] out_channels;
    logic [7:0] stride;
    logic [7:0] padding;
    logic [1:0] pool_mode;
    logic input_slot;
    logic output_slot;
    logic [Q16_ADDR_WIDTH-1:0] weight_base;
    logic [STATE_ADDR_WIDTH-1:0] state_base;
    logic [VERSION_WIDTH-1:0] sample_version;
    logic signed [15:0] leak;
    logic signed [15:0] threshold;
    logic reset_to_zero;
    logic adapter_busy;
    logic adapter_done;
    logic descriptor_error;
    logic scheduler_busy;
    logic scheduler_error;
    logic wave_start;
    logic wave_start_ready;
    logic [9:0] output_channel_base;
    logic [9:0] output_channel_count;
    logic group_start;
    logic group_start_ready;
    logic [9:0] input_channel_base;
    logic [9:0] input_channel_count;
    logic first_input_group;
    logic final_input_group;
    logic group_done;
    logic wave_done;
    logic wave_busy;
    logic wave_error;

    assign o_busy = adapter_busy || scheduler_busy || wave_busy;
    assign o_done = adapter_done;
    assign o_config_error = descriptor_error || scheduler_error;
    assign o_protocol_error = wave_error;
    assign o_state_error = 1'b0;

    unified_scnn_v23_layer_adapter #(
        .LAYER_WORDS(12), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH), .VERSION_WIDTH(VERSION_WIDTH)
    ) u_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_layer_valid(i_layer_valid),
        .o_layer_ready(o_layer_ready), .i_layer_words(i_layer_words),
        .i_input_slot(i_input_slot), .i_output_slot(i_output_slot),
        .i_sample_version(i_sample_version), .o_engine_start(engine_start),
        .i_engine_start_ready(engine_start_ready),
        .i_engine_done(engine_done), .o_in_width(in_width),
        .o_in_height(in_height), .o_in_channels(in_channels),
        .o_out_width(out_width), .o_out_height(out_height),
        .o_out_channels(out_channels), .o_stride(stride),
        .o_padding(padding), .o_pool_mode(pool_mode),
        .o_input_slot(input_slot), .o_output_slot(output_slot),
        .o_weight_q16_base(weight_base), .o_state_base(state_base),
        .o_sample_version(sample_version), .o_leak_q8_8(leak),
        .o_threshold_q8_8(threshold), .o_reset_to_zero(reset_to_zero),
        .o_unit_spike_features(),
        .o_busy(adapter_busy), .o_layer_done(adapter_done),
        .o_descriptor_error(descriptor_error)
    );

    unified_scnn_real_layer_wave_scheduler #(
        .ENGINES(ENGINES), .LANES(LANES)
    ) u_scheduler (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_start(engine_start), .o_start_ready(engine_start_ready),
        .i_in_width(in_width), .i_in_height(in_height),
        .i_in_channels(in_channels), .i_out_width(out_width),
        .i_out_height(out_height), .i_out_channels(out_channels),
        .o_wave_start(wave_start),
        .i_wave_start_ready(wave_start_ready),
        .o_output_channel_base(output_channel_base),
        .o_output_channel_count(output_channel_count),
        .o_group_start(group_start),
        .i_group_start_ready(group_start_ready),
        .o_input_channel_base(input_channel_base),
        .o_input_channel_count(input_channel_count),
        .o_first_input_group(first_input_group),
        .o_final_input_group(final_input_group),
        .i_group_done(group_done), .i_wave_done(wave_done),
        .i_child_error(wave_error), .o_busy(scheduler_busy),
        .o_done(engine_done), .o_sticky_error(scheduler_error)
    );

    unified_scnn_real_wave_lif_pool_top #(
        .ENGINES(ENGINES), .LANES(LANES), .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) u_wave (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_owner_enable(scheduler_busy), .i_clear_error(i_clear_error),
        .i_wave_start(wave_start), .o_wave_start_ready(wave_start_ready),
        .i_in_width(in_width), .i_in_height(in_height),
        .i_total_in_channels(in_channels), .i_out_width(out_width),
        .i_out_height(out_height), .i_total_out_channels(out_channels),
        .i_output_channel_base(output_channel_base),
        .i_output_channel_count(output_channel_count), .i_stride(stride),
        .i_padding(padding), .i_pool_mode(pool_mode),
        .i_input_slot(input_slot), .i_output_slot(output_slot),
        .i_feature_base('0), .i_weight_q16_base(weight_base),
        .i_state_base(state_base), .i_sample_version(sample_version),
        .i_leak_q8_8(leak), .i_threshold_q8_8(threshold),
        .i_reset_to_zero(reset_to_zero), .i_group_start(group_start),
        .o_group_start_ready(group_start_ready),
        .i_input_channel_base(input_channel_base),
        .i_input_channel_count(input_channel_count),
        .i_first_input_group(first_input_group),
        .i_final_input_group(final_input_group), .o_group_done(group_done),
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
        .o_feature_write_data(o_feature_write_data), .o_busy(wave_busy),
        .o_wave_done(wave_done), .o_sticky_error(wave_error)
    );
endmodule

`default_nettype wire
