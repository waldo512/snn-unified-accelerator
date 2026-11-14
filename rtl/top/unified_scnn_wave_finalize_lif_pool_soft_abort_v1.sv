`default_nettype none

// Connects finalized channel-wave accumulators to one shared versioned LIF
// state engine and the spatial-major wave pool/writeback path.
module unified_scnn_wave_finalize_lif_pool_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int MAX_WIDTH = 32,
    parameter int SPATIAL_DEPTH = 32768,
    parameter int SPATIAL_ADDR_WIDTH = $clog2(SPATIAL_DEPTH),
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int ACC_WIDTH = 48,
    parameter int BANK_ADDR_WIDTH = 14,
    parameter bit ENABLE_TP_OBSERVER = 1'b0,
    parameter STATE_INIT_FILE = "",
    parameter SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic [15:0]                               i_width,
    input  logic [15:0]                               i_height,
    input  logic [15:0]                               i_output_channels,
    input  logic [15:0]                               i_channel_base,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [1:0]                                i_pool_mode,
    input  logic                                      i_output_slot,
    input  logic [VERSION_WIDTH-1:0]                  i_sample_version,
    input  logic signed [15:0]                        i_leak_q8_8,
    input  logic signed [15:0]                        i_threshold_q8_8,
    input  logic                                      i_reset_to_zero,

    input  logic                                      i_partial_valid,
    output logic                                      o_partial_ready,
    input  logic [ENGINES-1:0]                        i_partial_engine_mask,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]             i_partial_spatial_addr,
    input  logic [ENGINES*STATE_ADDR_WIDTH-1:0]       i_output_addr_flat,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_partial_flat,
    input  logic                                      i_accumulator_clear,
    input  logic                                      i_accumulator_finalize,

    output logic                                      o_write_valid,
    input  logic                                      i_write_ready,
    output logic                                      o_write_slot,
    output logic [3:0]                                o_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]                o_write_addr,
    output logic signed [15:0]                        o_write_data_q8_8,

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
    output logic                                      o_sticky_error
);
    logic partial_valid;
    logic partial_input_ready;
    logic partial_ready;
    logic [ENGINES-1:0] partial_mask;
    logic [SPATIAL_ADDR_WIDTH-1:0] partial_spatial;
    logic [ENGINES*STATE_ADDR_WIDTH-1:0] partial_addr_flat;
    logic [ENGINES*ACC_WIDTH-1:0] partial_accumulator_flat;
    logic partial_idle;
    logic partial_error;
    logic lif_valid;
    logic lif_ready;
    localparam int ENGINE_INDEX_WIDTH = $clog2(ENGINES);
    logic [ENGINE_INDEX_WIDTH-1:0] lif_engine;
    logic [SPATIAL_ADDR_WIDTH-1:0] lif_spatial;
    logic [15:0] lif_spike;
    logic lif_idle;
    logic lif_error;
    logic pool_busy;
    logic pool_config_error;
    logic pool_protocol_error;
    logic active_q;
    logic protocol_error_q;
    logic start_fire;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic signed [15:0] leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic unused_wave_done;
    logic [STATE_ADDR_WIDTH-1:0] lif_addr;
    logic [VERSION_WIDTH-1:0] lif_version;
    logic signed [15:0] lif_current;
    logic signed [15:0] lif_membrane_pre;
    logic signed [15:0] lif_membrane_next;
    logic pool_lif_valid;
    logic [ENGINE_INDEX_WIDTH-1:0] pool_lif_engine;
    logic [SPATIAL_ADDR_WIDTH-1:0] pool_lif_spatial;
    logic [15:0] pool_lif_spike;
    logic pool_lif_ready;
    logic observation_protocol_error;

    unified_scnn_wave_partial_sum_banks_soft_abort_v1 #(
        .ENGINES(ENGINES), .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(STATE_ADDR_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_partial_banks (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_valid(i_partial_valid && active_q && !o_sticky_error),
        .o_ready(partial_input_ready),
        .i_engine_mask(i_partial_engine_mask),
        .i_spatial_addr(i_partial_spatial_addr),
        .i_output_addr_flat(i_output_addr_flat),
        .i_partial_flat(i_partial_flat),
        .i_accumulator_clear(i_accumulator_clear),
        .i_accumulator_finalize(i_accumulator_finalize),
        .o_valid(partial_valid), .i_ready(partial_ready),
        .o_engine_mask(partial_mask), .o_spatial_addr(partial_spatial),
        .o_output_addr_flat(partial_addr_flat),
        .o_accumulator_flat(partial_accumulator_flat),
        .o_idle(partial_idle), .o_sticky_error(partial_error)
    );

    unified_scnn_wave_partial_lif_serializer_soft_abort_v1 #(
        .ENGINES(ENGINES), .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .STATE_INIT_FILE(STATE_INIT_FILE)
    ) u_lif_serializer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_valid(partial_valid), .o_ready(partial_ready),
        .i_engine_mask(partial_mask), .i_spatial_addr(partial_spatial),
        .i_output_addr_flat(partial_addr_flat),
        .i_accumulator_flat(partial_accumulator_flat),
        .i_sample_version(sample_version_q), .i_leak_q8_8(leak_q),
        .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q), .o_valid(lif_valid),
        .i_ready(lif_ready), .o_engine_index(lif_engine),
        .o_spatial_addr(lif_spatial), .o_output_addr(lif_addr),
        .o_sample_version(lif_version),
        .o_current_q8_8(lif_current), .o_spike_q8_8(lif_spike),
        .o_membrane_pre(lif_membrane_pre),
        .o_membrane_next(lif_membrane_next),
        .o_wave_done(unused_wave_done), .o_idle(lif_idle),
        .o_sticky_error(lif_error)
    );

    generate
        if (ENABLE_TP_OBSERVER) begin : g_tp_observer
            unified_scnn_tp_lif_observation_fork #(
                .ENGINE_INDEX_WIDTH(ENGINE_INDEX_WIDTH),
                .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
                .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
                .VERSION_WIDTH(VERSION_WIDTH),
                .SURROGATE_ROM_FILE(SURROGATE_ROM_FILE)
            ) u_lif_observation_fork (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_observation_enable(i_observation_enable),
                .i_valid(lif_valid), .o_ready(lif_ready),
                .i_engine_index(lif_engine), .i_spatial_addr(lif_spatial),
                .i_output_addr(lif_addr), .i_sample_version(lif_version),
                .i_current_q8_8(lif_current), .i_spike_q8_8(lif_spike),
                .i_membrane_pre_q8_8(lif_membrane_pre),
                .i_membrane_next_q8_8(lif_membrane_next),
                .i_threshold_q8_8(threshold_q),
                .i_channel_base(i_channel_base),
                .o_pool_valid(pool_lif_valid),
                .i_pool_ready(pool_lif_ready),
                .o_pool_engine_index(pool_lif_engine),
                .o_pool_spatial_addr(pool_lif_spatial),
                .o_pool_spike_q8_8(pool_lif_spike),
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
                .o_observation_surrogate_q8_8(
                    o_observation_surrogate_q8_8),
                .o_protocol_error(observation_protocol_error)
            );
        end else begin : g_no_tp_observer
            assign lif_ready = pool_lif_ready;
            assign pool_lif_valid = lif_valid;
            assign pool_lif_engine = lif_engine;
            assign pool_lif_spatial = lif_spatial;
            assign pool_lif_spike = lif_spike;
            assign o_observation_valid = 1'b0;
            assign o_observation_channel = '0;
            assign o_observation_spatial_addr = '0;
            assign o_observation_output_addr = '0;
            assign o_observation_sample_version = '0;
            assign o_observation_current_q8_8 = '0;
            assign o_observation_spike_q8_8 = '0;
            assign o_observation_membrane_pre_q8_8 = '0;
            assign o_observation_membrane_next_q8_8 = '0;
            assign o_observation_surrogate_q8_8 = '0;
            assign observation_protocol_error = 1'b0;
        end
    endgenerate

    unified_scnn_wave_pool_writeback_soft_abort_v1 #(
        .ENGINES(ENGINES), .MAX_WIDTH(MAX_WIDTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) u_pool (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start(i_start), .o_start_ready(o_start_ready), .i_width(i_width),
        .i_height(i_height), .i_output_channels(i_output_channels),
        .i_channel_base(i_channel_base), .i_engine_mask(i_engine_mask),
        .i_pool_mode(i_pool_mode), .i_output_slot(i_output_slot),
        .i_valid(pool_lif_valid), .o_ready(pool_lif_ready),
        .i_engine_index(pool_lif_engine),
        .i_spatial_addr(pool_lif_spatial),
        .i_value_q8_8(pool_lif_spike), .o_write_valid(o_write_valid),
        .i_write_ready(i_write_ready), .o_write_slot(o_write_slot),
        .o_write_bank(o_write_bank), .o_write_addr(o_write_addr),
        .o_write_data_q8_8(o_write_data_q8_8), .o_busy(pool_busy),
        .o_done(o_done), .o_config_error(pool_config_error),
        .o_protocol_error(pool_protocol_error)
    );

    assign start_fire = i_start && o_start_ready;
    assign o_partial_ready = active_q && partial_input_ready &&
                             !o_sticky_error;
    // Pool writeback can finish while its last TP observation is stalled.
    assign o_busy = active_q || pool_busy || !partial_idle || !lif_idle ||
                    o_observation_valid;
    assign o_sticky_error = partial_error || lif_error ||
                            pool_config_error || pool_protocol_error ||
                            protocol_error_q || observation_protocol_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            protocol_error_q <= 1'b0;
            sample_version_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
        end else if (i_abort) begin
            active_q <= 1'b0;
            sample_version_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy)
                protocol_error_q <= 1'b0;
            if (start_fire) begin
                active_q <= 1'b1;
                sample_version_q <= i_sample_version;
                leak_q <= i_leak_q8_8;
                threshold_q <= i_threshold_q8_8;
                reset_to_zero_q <= i_reset_to_zero;
            end
            if (active_q && pool_config_error)
                active_q <= 1'b0;
            if (o_done)
                active_q <= 1'b0;
            if (i_partial_valid && !active_q && !start_fire)
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
