`default_nettype none

module unified_scnn_v23_execution_top #(
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int BANK_ADDR_WIDTH = 14,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_layer_valid,
    output logic                           o_layer_ready,
    input  logic [383:0]                   i_layer_words,
    input  logic                           i_input_slot,
    input  logic                           i_output_slot,
    input  logic [VERSION_WIDTH-1:0]       i_sample_version,

    output logic                           o_feature_read_valid,
    input  logic                           i_feature_read_ready,
    output logic                           o_feature_read_slot,
    output logic [31:0]                    o_feature_read_bank_flat,
    output logic [8*BANK_ADDR_WIDTH-1:0]   o_feature_read_addr_flat,
    output logic [7:0]                     o_feature_read_lane_mask,
    input  logic                           i_feature_valid,
    output logic                           o_feature_ready,
    input  logic [127:0]                   i_feature_flat,
    input  logic [7:0]                     i_feature_lane_mask,

    output logic                           o_weight_read_valid,
    input  logic                           i_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]      o_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]      o_weight_read_addr1,
    input  logic                           i_weight_result_valid,
    output logic                           o_weight_result_ready,
    input  logic [63:0]                    i_weight_data0,
    input  logic [63:0]                    i_weight_data1,

    output logic                           o_feature_write_valid,
    input  logic                           i_feature_write_ready,
    output logic                           o_feature_write_slot,
    output logic [3:0]                     o_feature_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]     o_feature_write_addr,
    output logic signed [15:0]             o_feature_write_data,
    output logic                           o_busy,
    output logic                           o_done,
    output logic                           o_config_error,
    output logic                           o_protocol_error,
    output logic                           o_state_error
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
    logic engine_busy;
    logic engine_config_error;
    logic engine_protocol_error;
    logic engine_state_error;

    assign o_busy = adapter_busy || engine_busy;
    assign o_done = adapter_done;
    assign o_config_error = descriptor_error || engine_config_error;
    assign o_protocol_error = engine_protocol_error;
    assign o_state_error = engine_state_error;

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

    unified_scnn_q8_8_tiled_layer_engine #(
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH)
    ) u_engine (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_start(engine_start),
        .o_start_ready(engine_start_ready), .i_in_width(in_width),
        .i_in_height(in_height), .i_in_channels(in_channels),
        .i_out_width(out_width), .i_out_height(out_height),
        .i_out_channels(out_channels), .i_stride(stride),
        .i_padding(padding), .i_pool_mode(pool_mode),
        .i_input_slot(input_slot), .i_output_slot(output_slot),
        .i_weight_q16_base(weight_base), .i_state_base(state_base),
        .i_sample_version(sample_version), .i_leak_q8_8(leak),
        .i_threshold_q8_8(threshold), .i_reset_to_zero(reset_to_zero),
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
        .o_busy(engine_busy), .o_done(engine_done),
        .o_config_error(engine_config_error),
        .o_protocol_error(engine_protocol_error),
        .o_state_error(engine_state_error)
    );
endmodule

`default_nettype wire
