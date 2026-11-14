`timescale 1ns/1ps
`default_nettype none

// Captures one binary SCNN classifier input frame in physical-region arrival
// order, then replays it as strict channel-major feature waves. Only one bit is
// stored per feature; the analog state/trace and PE datapath are not copied.
module unified_scnn_h40_classifier_feature_wave_adapter_v1 #(
    parameter int MAX_CHANNELS = 512,
    parameter int SPATIAL = 4,
    parameter int CHANNELS_PER_WORKSET = 16,
    parameter int LANES = 8,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int CHANNEL_WIDTH = MAX_CHANNELS <= 2 ? 1 : $clog2(MAX_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(MAX_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_CHANNELS * SPATIAL <= 2 ? 1 :
        $clog2(MAX_CHANNELS * SPATIAL),
    parameter int CLASS_WIDTH = 4
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [19:0]                    i_cfg_layer_tag,
    input  wire logic [CHANNEL_WIDTH-1:0]       i_cfg_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_cfg_channel_count,
    input  wire logic [VERSION_WIDTH-1:0]       i_cfg_version,
    input  wire logic [FRAME_WIDTH-1:0]         i_cfg_frame_index,
    input  wire logic                           i_cfg_sample_reset,
    input  wire logic                           i_cfg_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]         i_cfg_sample_label,
    input  wire logic                           i_release,
    output logic                                o_release_ready,

    input  wire logic                           i_scalar_valid,
    output logic                                o_scalar_ready,
    input  wire logic [19:0]                    i_scalar_layer_tag,
    input  wire logic [9:0]                     i_scalar_channel,
    input  wire logic [31:0]                    i_scalar_spatial,
    input  wire logic                           i_scalar_spike,

    output logic                                o_feature_wave_valid,
    input  wire logic                           i_feature_wave_ready,
    output logic [LANES-1:0]                    o_feature_wave_mask,
    output logic [FEATURE_WIDTH-1:0]            o_feature_wave_base,
    output logic [LANES*16-1:0]                 o_feature_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]            o_feature_wave_version,
    output logic [FRAME_WIDTH-1:0]              o_feature_wave_frame_index,
    output logic                                o_feature_wave_first,
    output logic                                o_feature_wave_last,
    output logic                                o_feature_wave_sample_reset,
    output logic                                o_feature_wave_sample_last,
    output logic [CLASS_WIDTH-1:0]              o_sample_label,

    output logic                                o_capture_complete,
    output logic                                o_active,
    output logic                                o_idle,
    output logic                                o_protocol_error
);
    localparam int FRAME_FEATURES = MAX_CHANNELS * SPATIAL;
    localparam int FRAME_COUNT_WIDTH = $clog2(FRAME_FEATURES + 1);

    typedef enum logic [1:0] {IDLE, COLLECT, EMIT, COMPLETE} state_t;
    state_t state_q;

    logic [19:0] layer_tag_q;
    logic [CHANNEL_WIDTH-1:0] channel_base_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] channel_count_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [FRAME_WIDTH-1:0] frame_index_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic [CLASS_WIDTH-1:0] sample_label_q;
    logic [FRAME_FEATURES-1:0] feature_spike_q;
    logic [FRAME_FEATURES-1:0] feature_seen_q;
    logic [FRAME_COUNT_WIDTH-1:0] feature_count_q;
    logic [FRAME_COUNT_WIDTH-1:0] collected_count_q;
    logic [FEATURE_WIDTH-1:0] emit_offset_q;

    logic cfg_legal;
    logic cfg_fire;
    logic scalar_fire;
    logic scalar_legal;
    logic scalar_duplicate;
    logic [FEATURE_WIDTH-1:0] scalar_local_index;
    logic feature_fire;
    logic final_wave;
    integer unsigned scalar_local_integer;

    always_comb begin
        cfg_legal = i_cfg_channel_count != 0 &&
            integer'(i_cfg_channel_base) + integer'(i_cfg_channel_count) <=
                MAX_CHANNELS;
        o_cfg_ready = state_q == IDLE && !o_protocol_error;
        cfg_fire = i_cfg_valid && o_cfg_ready;

        scalar_local_integer =
            (integer'(i_scalar_channel) - integer'(channel_base_q)) *
                SPATIAL + integer'(i_scalar_spatial);
        scalar_local_index = FEATURE_WIDTH'(scalar_local_integer);
        scalar_legal = i_scalar_layer_tag == layer_tag_q &&
            integer'(i_scalar_channel) >= integer'(channel_base_q) &&
            integer'(i_scalar_channel) < integer'(channel_base_q) +
                integer'(channel_count_q) &&
            integer'(i_scalar_spatial) < SPATIAL &&
            scalar_local_integer < FRAME_FEATURES;
        scalar_duplicate = scalar_legal &&
            feature_seen_q[scalar_local_index];

        o_scalar_ready = state_q == COLLECT && !o_protocol_error;
        scalar_fire = i_scalar_valid && o_scalar_ready;

        o_feature_wave_valid = state_q == EMIT && !o_protocol_error;
        o_feature_wave_mask = '0;
        o_feature_wave_base = emit_offset_q;
        o_feature_wave_q8_8_flat = '0;
        for (integer lane = 0; lane < LANES; lane++) begin
            if (integer'(emit_offset_q) + lane <
                integer'(feature_count_q)) begin
                o_feature_wave_mask[lane] = 1'b1;
                o_feature_wave_q8_8_flat[lane*16 +: 16] =
                    feature_spike_q[integer'(emit_offset_q) + lane] ?
                        16'sd256 : 16'sd0;
            end
        end
        final_wave = integer'(emit_offset_q) + LANES >=
            integer'(feature_count_q);
        o_feature_wave_version = version_q;
        o_feature_wave_frame_index = frame_index_q;
        o_feature_wave_first = emit_offset_q == 0;
        o_feature_wave_last = final_wave;
        o_feature_wave_sample_reset = sample_reset_q &&
            o_feature_wave_first;
        o_feature_wave_sample_last = sample_last_q && final_wave;
        o_sample_label = sample_label_q;
        feature_fire = o_feature_wave_valid && i_feature_wave_ready;

        o_release_ready = state_q == COMPLETE;
        o_capture_complete = state_q == COMPLETE;
        o_active = state_q != IDLE;
        o_idle = state_q == IDLE || state_q == COMPLETE ||
            (state_q == COLLECT && collected_count_q == 0);
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            layer_tag_q <= '0;
            channel_base_q <= '0;
            channel_count_q <= '0;
            version_q <= '0;
            frame_index_q <= '0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            sample_label_q <= '0;
            feature_spike_q <= '0;
            feature_seen_q <= '0;
            feature_count_q <= '0;
            collected_count_q <= '0;
            emit_offset_q <= '0;
            o_protocol_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= IDLE;
            feature_seen_q <= '0;
            collected_count_q <= '0;
            emit_offset_q <= '0;
        end else begin
            if (i_clear_error && state_q == IDLE)
                o_protocol_error <= 1'b0;

            if (cfg_fire) begin
                if (!cfg_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    state_q <= COLLECT;
                    layer_tag_q <= i_cfg_layer_tag;
                    channel_base_q <= i_cfg_channel_base;
                    channel_count_q <= i_cfg_channel_count;
                    version_q <= i_cfg_version;
                    frame_index_q <= i_cfg_frame_index;
                    sample_reset_q <= i_cfg_sample_reset;
                    sample_last_q <= i_cfg_sample_last;
                    sample_label_q <= i_cfg_sample_label;
                    feature_spike_q <= '0;
                    feature_seen_q <= '0;
                    feature_count_q <= FRAME_COUNT_WIDTH'(
                        integer'(i_cfg_channel_count) * SPATIAL);
                    collected_count_q <= '0;
                    emit_offset_q <= '0;
                end
            end else if (scalar_fire) begin
                if (!scalar_legal || scalar_duplicate) begin
                    state_q <= IDLE;
                    o_protocol_error <= 1'b1;
                end else begin
                    feature_spike_q[scalar_local_index] <= i_scalar_spike;
                    feature_seen_q[scalar_local_index] <= 1'b1;
                    collected_count_q <= collected_count_q + 1'b1;
                    if (collected_count_q + 1'b1 == feature_count_q) begin
                        emit_offset_q <= '0;
                        state_q <= EMIT;
                    end
                end
            end else if (feature_fire) begin
                if (final_wave)
                    state_q <= COMPLETE;
                else
                    emit_offset_q <= emit_offset_q + FEATURE_WIDTH'(LANES);
            end else if (i_release && o_release_ready) begin
                state_q <= IDLE;
            end
        end
    end

    initial begin
        if (SPATIAL < 1 || CHANNELS_PER_WORKSET < 1 || LANES < 1 ||
            MAX_CHANNELS < CHANNELS_PER_WORKSET || FRAME_FEATURES > 65536)
            $error("SCNN classifier feature-wave adapter parameter mismatch");
    end
endmodule

`default_nettype wire
