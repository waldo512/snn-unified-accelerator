`default_nettype none

// Descriptor-to-command SCNN layer boundary. Convolution arithmetic is owned
// by the single external Wave3 engine; this wrapper only composes existing
// descriptor, wave-scheduling, and shared-finalization control blocks.
module unified_scnn_external_shared_layer_v1 #(
    parameter int LAYER_WORDS = 12,
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int MAX_WIDTH = 32,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter logic [FEATURE_ADDR_WIDTH-1:0] FEATURE_BASE = '0,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int SPATIAL_DEPTH = 32768,
    parameter int SPATIAL_ADDR_WIDTH = $clog2(SPATIAL_DEPTH),
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int ACC_WIDTH = 48,
    parameter int BANK_ADDR_WIDTH = FEATURE_ADDR_WIDTH,
    parameter bit ENABLE_TP_OBSERVER = 1'b0,
    parameter STATE_INIT_FILE = "",
    parameter SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,

    input  logic                                      i_layer_valid,
    output logic                                      o_layer_ready,
    input  logic [LAYER_WORDS*32-1:0]                 i_layer_words,
    input  logic                                      i_input_slot,
    input  logic                                      i_output_slot,
    input  logic [VERSION_WIDTH-1:0]                  i_sample_version,

    output logic                                      o_conv_start_valid,
    input  logic                                      i_conv_start_ready,
    output logic [15:0]                               o_conv_in_width,
    output logic [15:0]                               o_conv_in_height,
    output logic [15:0]                               o_conv_total_in_channels,
    output logic [15:0]                               o_conv_out_width,
    output logic [15:0]                               o_conv_out_height,
    output logic [15:0]                               o_conv_total_out_channels,
    output logic [9:0]                                o_conv_input_channel_base,
    output logic [9:0]                                o_conv_input_channel_count,
    output logic [9:0]                                o_conv_output_channel_base,
    output logic [9:0]                                o_conv_output_channel_count,
    output logic [7:0]                                o_conv_stride,
    output logic [7:0]                                o_conv_padding,
    output logic                                      o_conv_input_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]             o_conv_feature_base,
    output logic [Q16_ADDR_WIDTH-1:0]                 o_conv_weight_q16_base,
    output logic [STATE_ADDR_WIDTH-1:0]               o_conv_output_base,
    output logic                                      o_conv_unit_spike_features,
    input  logic                                      i_conv_done,
    input  logic                                      i_conv_idle,
    input  logic                                      i_conv_sticky_error,
    input  logic [7:0]                                i_conv_error_code,

    input  logic                                      i_raw_result_valid,
    output logic                                      o_raw_result_ready,
    input  logic [ENGINES-1:0]                        i_raw_result_mask,
    input  logic [ENGINES*STATE_ADDR_WIDTH-1:0]       i_raw_result_output_addr_flat,
    input  logic [STATE_ADDR_WIDTH-1:0]               i_raw_result_spatial_addr,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_raw_result_accumulator_flat,
    output logic                                      o_raw_result_fire,
    output logic                                      o_raw_result_group_first,
    output logic                                      o_raw_result_group_final,

    output logic                                      o_feature_write_valid,
    input  logic                                      i_feature_write_ready,
    output logic                                      o_feature_write_slot,
    output logic [3:0]                                o_feature_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]                o_feature_write_addr,
    output logic signed [15:0]                        o_feature_write_data_q8_8,

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

    output logic [VERSION_WIDTH-1:0]                  o_locked_sample_version,
    output logic                                      o_layer_done,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_DESCRIPTOR = 8'h01;
    localparam logic [7:0] ERROR_SCHEDULER = 8'h20;

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
    logic [Q16_ADDR_WIDTH-1:0] weight_q16_base;
    logic [STATE_ADDR_WIDTH-1:0] state_base;
    logic unit_spike_features;
    logic [VERSION_WIDTH-1:0] sample_version;
    logic signed [15:0] leak_q8_8;
    logic signed [15:0] threshold_q8_8;
    logic reset_to_zero;
    logic adapter_layer_ready;
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

    logic leaf_busy;
    logic leaf_idle;
    logic leaf_error;
    logic [7:0] leaf_error_code;
    logic local_error_q;
    logic [7:0] local_error_code_q;
    logic layer_accept_enable;

    assign layer_accept_enable = !local_error_q && !descriptor_error &&
                                 !scheduler_error && !leaf_error;
    assign o_layer_ready = adapter_layer_ready && layer_accept_enable;
    assign o_layer_done = adapter_done;
    assign o_idle = !adapter_busy && !scheduler_busy && leaf_idle;
    assign o_busy = !o_idle;
    assign o_sticky_error = local_error_q;
    assign o_error_code = local_error_q ? local_error_code_q : ERROR_NONE;
    assign o_conv_unit_spike_features = unit_spike_features;

    unified_scnn_v23_layer_adapter_soft_abort_v1 #(
        .LAYER_WORDS(LAYER_WORDS), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH), .VERSION_WIDTH(VERSION_WIDTH)
    ) u_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_layer_valid(i_layer_valid && layer_accept_enable),
        .o_layer_ready(adapter_layer_ready), .i_layer_words(i_layer_words),
        .i_input_slot(i_input_slot), .i_output_slot(i_output_slot),
        .i_sample_version(i_sample_version), .o_engine_start(engine_start),
        .i_engine_start_ready(engine_start_ready),
        .i_engine_done(engine_done), .o_in_width(in_width),
        .o_in_height(in_height), .o_in_channels(in_channels),
        .o_out_width(out_width), .o_out_height(out_height),
        .o_out_channels(out_channels), .o_stride(stride),
        .o_padding(padding), .o_pool_mode(pool_mode),
        .o_input_slot(input_slot), .o_output_slot(output_slot),
        .o_weight_q16_base(weight_q16_base), .o_state_base(state_base),
        .o_unit_spike_features(unit_spike_features),
        .o_sample_version(sample_version), .o_leak_q8_8(leak_q8_8),
        .o_threshold_q8_8(threshold_q8_8),
        .o_reset_to_zero(reset_to_zero), .o_busy(adapter_busy),
        .o_layer_done(adapter_done),
        .o_descriptor_error(descriptor_error)
    );

    unified_scnn_real_layer_wave_scheduler_soft_abort_v1 #(
        .ENGINES(ENGINES), .LANES(LANES)
    ) u_scheduler (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
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
        .i_child_error(leaf_error), .o_busy(scheduler_busy),
        .o_done(engine_done), .o_sticky_error(scheduler_error)
    );

    unified_scnn_external_shared_finalize_leaf_v1 #(
        .ENGINES(ENGINES), .LANES(LANES), .MAX_WIDTH(MAX_WIDTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH), .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .ENABLE_TP_OBSERVER(ENABLE_TP_OBSERVER),
        .STATE_INIT_FILE(STATE_INIT_FILE),
        .SURROGATE_ROM_FILE(SURROGATE_ROM_FILE)
    ) u_finalize_leaf (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_owner_enable(scheduler_busy), .i_clear_error(i_clear_error),
        .i_wave_start(wave_start), .o_wave_start_ready(wave_start_ready),
        .i_in_width(in_width), .i_in_height(in_height),
        .i_total_in_channels(in_channels), .i_out_width(out_width),
        .i_out_height(out_height), .i_total_out_channels(out_channels),
        .i_output_channel_base(output_channel_base),
        .i_output_channel_count(output_channel_count), .i_stride(stride),
        .i_padding(padding), .i_pool_mode(pool_mode),
        .i_input_slot(input_slot), .i_output_slot(output_slot),
        .i_feature_base(FEATURE_BASE),
        .i_weight_q16_base(weight_q16_base), .i_state_base(state_base),
        .i_sample_version(sample_version), .i_leak_q8_8(leak_q8_8),
        .i_threshold_q8_8(threshold_q8_8),
        .i_reset_to_zero(reset_to_zero), .i_group_start(group_start),
        .o_group_start_ready(group_start_ready),
        .i_input_channel_base(input_channel_base),
        .i_input_channel_count(input_channel_count),
        .i_first_input_group(first_input_group),
        .i_final_input_group(final_input_group),
        .o_group_done(group_done),
        .o_conv_start_valid(o_conv_start_valid),
        .i_conv_start_ready(i_conv_start_ready),
        .o_conv_in_width(o_conv_in_width),
        .o_conv_in_height(o_conv_in_height),
        .o_conv_total_in_channels(o_conv_total_in_channels),
        .o_conv_out_width(o_conv_out_width),
        .o_conv_out_height(o_conv_out_height),
        .o_conv_total_out_channels(o_conv_total_out_channels),
        .o_conv_input_channel_base(o_conv_input_channel_base),
        .o_conv_input_channel_count(o_conv_input_channel_count),
        .o_conv_output_channel_base(o_conv_output_channel_base),
        .o_conv_output_channel_count(o_conv_output_channel_count),
        .o_conv_stride(o_conv_stride), .o_conv_padding(o_conv_padding),
        .o_conv_input_slot(o_conv_input_slot),
        .o_conv_feature_base(o_conv_feature_base),
        .o_conv_weight_q16_base(o_conv_weight_q16_base),
        .o_conv_output_base(o_conv_output_base),
        .i_conv_done(i_conv_done), .i_conv_idle(i_conv_idle),
        .i_conv_sticky_error(i_conv_sticky_error),
        .i_conv_error_code(i_conv_error_code),
        .i_raw_result_valid(i_raw_result_valid),
        .o_raw_result_ready(o_raw_result_ready),
        .i_raw_result_mask(i_raw_result_mask),
        .i_raw_result_output_addr_flat(i_raw_result_output_addr_flat),
        .i_raw_result_spatial_addr(i_raw_result_spatial_addr),
        .i_raw_result_accumulator_flat(i_raw_result_accumulator_flat),
        .o_raw_result_fire(o_raw_result_fire),
        .o_raw_result_group_first(o_raw_result_group_first),
        .o_raw_result_group_final(o_raw_result_group_final),
        .o_write_valid(o_feature_write_valid),
        .i_write_ready(i_feature_write_ready),
        .o_write_slot(o_feature_write_slot),
        .o_write_bank(o_feature_write_bank),
        .o_write_addr(o_feature_write_addr),
        .o_write_data_q8_8(o_feature_write_data_q8_8),
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
        .o_locked_sample_version(o_locked_sample_version),
        .o_busy(leaf_busy), .o_idle(leaf_idle), .o_done(wave_done),
        .o_sticky_error(leaf_error), .o_error_code(leaf_error_code)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else if (!i_abort) begin
            if (i_clear_error && o_idle && !descriptor_error &&
                !scheduler_error && !leaf_error) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end
            if (!local_error_q) begin
                if (descriptor_error) begin
                    local_error_q <= 1'b1;
                    local_error_code_q <= ERROR_DESCRIPTOR;
                end else if (leaf_error) begin
                    local_error_q <= 1'b1;
                    local_error_code_q <= leaf_error_code;
                end else if (scheduler_error) begin
                    local_error_q <= 1'b1;
                    local_error_code_q <= ERROR_SCHEDULER;
                end
            end
        end
    end
endmodule

`default_nettype wire
