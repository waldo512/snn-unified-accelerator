`default_nettype none

// Connects finalized channel-wave accumulators to one shared versioned LIF
// state engine and the spatial-major wave pool/writeback path.
module unified_scnn_wave_finalize_lif_pool #(
    parameter int ENGINES = 12,
    parameter int MAX_WIDTH = 32,
    parameter int SPATIAL_DEPTH = 32768,
    parameter int SPATIAL_ADDR_WIDTH = $clog2(SPATIAL_DEPTH),
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int ACC_WIDTH = 48,
    parameter int BANK_ADDR_WIDTH = 14
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
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
    logic [$clog2(ENGINES)-1:0] lif_engine;
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
    logic [STATE_ADDR_WIDTH-1:0] unused_lif_addr;
    logic [VERSION_WIDTH-1:0] unused_lif_version;
    logic signed [15:0] unused_lif_current;
    logic signed [15:0] unused_membrane_pre;
    logic signed [15:0] unused_membrane_next;

    unified_scnn_wave_partial_sum_banks #(
        .ENGINES(ENGINES), .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(STATE_ADDR_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_partial_banks (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
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

    unified_scnn_wave_partial_lif_serializer #(
        .ENGINES(ENGINES), .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_lif_serializer (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_valid(partial_valid), .o_ready(partial_ready),
        .i_engine_mask(partial_mask), .i_spatial_addr(partial_spatial),
        .i_output_addr_flat(partial_addr_flat),
        .i_accumulator_flat(partial_accumulator_flat),
        .i_sample_version(sample_version_q), .i_leak_q8_8(leak_q),
        .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q), .o_valid(lif_valid),
        .i_ready(lif_ready), .o_engine_index(lif_engine),
        .o_spatial_addr(lif_spatial), .o_output_addr(unused_lif_addr),
        .o_sample_version(unused_lif_version),
        .o_current_q8_8(unused_lif_current), .o_spike_q8_8(lif_spike),
        .o_membrane_pre(unused_membrane_pre),
        .o_membrane_next(unused_membrane_next),
        .o_wave_done(unused_wave_done), .o_idle(lif_idle),
        .o_sticky_error(lif_error)
    );

    unified_scnn_wave_pool_writeback #(
        .ENGINES(ENGINES), .MAX_WIDTH(MAX_WIDTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) u_pool (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_start(i_start), .o_start_ready(o_start_ready), .i_width(i_width),
        .i_height(i_height), .i_output_channels(i_output_channels),
        .i_channel_base(i_channel_base), .i_engine_mask(i_engine_mask),
        .i_pool_mode(i_pool_mode), .i_output_slot(i_output_slot),
        .i_valid(lif_valid), .o_ready(lif_ready),
        .i_engine_index(lif_engine), .i_spatial_addr(lif_spatial),
        .i_value_q8_8(lif_spike), .o_write_valid(o_write_valid),
        .i_write_ready(i_write_ready), .o_write_slot(o_write_slot),
        .o_write_bank(o_write_bank), .o_write_addr(o_write_addr),
        .o_write_data_q8_8(o_write_data_q8_8), .o_busy(pool_busy),
        .o_done(o_done), .o_config_error(pool_config_error),
        .o_protocol_error(pool_protocol_error)
    );

    assign start_fire = i_start && o_start_ready;
    assign o_partial_ready = active_q && partial_input_ready &&
                             !o_sticky_error;
    assign o_busy = active_q || pool_busy || !partial_idle || !lif_idle;
    assign o_sticky_error = partial_error || lif_error ||
                            pool_config_error || pool_protocol_error ||
                            protocol_error_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            protocol_error_q <= 1'b0;
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
