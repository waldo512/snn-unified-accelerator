`default_nettype none

// Atomically forks selected Conv8 pooled-spike writes to the existing feature
// scratchpad and a temporal-trace state engine. The shared writeback order is
// output-wave, pooled-spatial, engine-lane; checking that order also recovers
// the channel/spatial metadata without a second feature-map buffer.
module unified_scnn_tp_target_conv8_trace_fork_t1 #(
    parameter int ENGINES = 12,
    parameter int MAX_BATCH = 10,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int VERSION_WIDTH = 16,
    parameter int PATH_CONTEXTS = 1,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int CHANNEL_WIDTH = MAX_OUTPUT_CHANNELS <= 2 ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS),
    parameter int TRACE_COUNT = PATH_CONTEXTS*MAX_BATCH*MAX_OUTPUT_CHANNELS*4,
    parameter int TRACE_ADDR_WIDTH = TRACE_COUNT <= 2 ? 1 :
        $clog2(TRACE_COUNT)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_state_reset,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [9:0]                                i_cfg_output_channels,
    input  logic [15:0]                               i_cfg_pooled_width,
    input  logic [15:0]                               i_cfg_pooled_height,
    input  logic                                      i_cfg_output_slot,
    input  logic                                      i_cfg_path,
    input  logic [SAMPLE_WIDTH-1:0]                   i_cfg_sample,
    input  logic                                      i_cfg_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_cfg_version,
    input  logic signed [15:0]                        i_cfg_trace_leak_q8_8,

    input  logic                                      i_write_valid,
    output logic                                      o_write_ready,
    input  logic                                      i_write_slot,
    input  logic [3:0]                                i_write_bank,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_write_addr,
    input  logic signed [15:0]                        i_write_spike_q8_8,

    output logic                                      o_feature_write_valid,
    input  logic                                      i_feature_write_ready,
    output logic                                      o_feature_write_slot,
    output logic [3:0]                                o_feature_write_bank,
    output logic [FEATURE_ADDR_WIDTH-1:0]             o_feature_write_addr,
    output logic signed [15:0]                        o_feature_write_data_q8_8,

    input  logic                                      i_layer_done,
    output logic                                      o_layer_done,
    output logic                                      o_observation_valid,
    input  logic                                      i_observation_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_observation_sample,
    output logic                                      o_observation_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_observation_version,
    output logic [CHANNEL_WIDTH-1:0]                  o_observation_channel,
    output logic [1:0]                                o_observation_spatial,
    output logic signed [15:0]                        o_observation_trace_q8_8,
    output logic                                      o_observation_last,
    output logic                                      o_busy,
    output logic                                      o_error
);
    logic active_q;
    logic layer_done_seen_q;
    logic output_slot_q;
    logic path_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic timestep_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic signed [15:0] trace_leak_q;
    logic [9:0] output_channels_q;
    logic [9:0] wave_base_q;
    logic [3:0] lane_q;
    logic [1:0] spatial_q;
    logic local_error_q;

    logic [9:0] current_channel;
    logic [9:0] current_wave_count;
    logic [3:0] expected_bank;
    logic [TRACE_ADDR_WIDTH-1:0] trace_input_addr;
    logic mapping_legal;
    logic selected_write_fire;
    logic trace_ready;
    logic trace_valid;
    logic [TRACE_ADDR_WIDTH-1:0] trace_output_addr;
    logic signed [15:0] trace_output_data;
    logic trace_idle;
    logic trace_reset_error;
    logic trace_address_error;
    logic [CHANNEL_WIDTH-1:0] observation_channel_q;
    logic [1:0] observation_spatial_q;
    logic observation_last_q;

    assign current_wave_count = output_channels_q-wave_base_q > ENGINES ?
        10'(ENGINES) : output_channels_q-wave_base_q;
    assign current_channel = wave_base_q + lane_q;
    always_comb begin
        case (spatial_q)
            2'd0: expected_bank = 4'd0;
            2'd1: expected_bank = 4'd1;
            2'd2: expected_bank = 4'd3;
            default: expected_bank = 4'd4;
        endcase
    end
    assign trace_input_addr = TRACE_ADDR_WIDTH'(
        int'(path_q)*MAX_BATCH*MAX_OUTPUT_CHANNELS*4 +
        int'(sample_q)*MAX_OUTPUT_CHANNELS*4 +
        int'(current_channel)*4 + int'(spatial_q));
    assign mapping_legal = active_q && current_wave_count != 0 &&
        lane_q < current_wave_count && current_channel < output_channels_q &&
        i_write_slot == output_slot_q && i_write_bank == expected_bank &&
        i_write_addr == FEATURE_ADDR_WIDTH'(current_channel);

    assign o_cfg_ready = !active_q && trace_idle && !o_error &&
        !i_state_reset;
    assign o_write_ready = i_feature_write_ready && !o_error &&
        (!active_q || (mapping_legal && trace_ready));
    assign o_feature_write_valid = i_write_valid && !o_error &&
        (!active_q || (mapping_legal && trace_ready));
    assign o_feature_write_slot = i_write_slot;
    assign o_feature_write_bank = i_write_bank;
    assign o_feature_write_addr = i_write_addr;
    assign o_feature_write_data_q8_8 = i_write_spike_q8_8;
    assign selected_write_fire = i_write_valid && o_write_ready && active_q;

    assign o_observation_valid = trace_valid;
    assign o_observation_sample = sample_q;
    assign o_observation_timestep = timestep_q;
    assign o_observation_version = version_q;
    assign o_observation_channel = observation_channel_q;
    assign o_observation_spatial = observation_spatial_q;
    assign o_observation_trace_q8_8 = trace_output_data;
    assign o_observation_last = observation_last_q;
    assign o_busy = active_q || !trace_idle;
    assign o_error = local_error_q || trace_reset_error ||
        trace_address_error;

    unified_scnn_q8_8_trace_state_engine #(
        .NUM_NEURONS(TRACE_COUNT), .ADDR_WIDTH(TRACE_ADDR_WIDTH)
    ) u_trace_state (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_sample_reset(i_state_reset),
        .i_valid(i_write_valid && i_feature_write_ready && active_q &&
            mapping_legal && !o_error),
        .o_ready(trace_ready), .i_addr(trace_input_addr),
        .i_spike_q8_8(i_write_spike_q8_8), .i_leak_q8_8(trace_leak_q),
        .o_valid(trace_valid), .i_ready(i_observation_ready),
        .o_addr(trace_output_addr), .o_trace_q8_8(trace_output_data),
        .o_idle(trace_idle),
        .o_reset_while_busy_error(trace_reset_error),
        .o_address_error(trace_address_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            layer_done_seen_q <= 1'b0;
            output_slot_q <= 1'b0;
            path_q <= 1'b0;
            sample_q <= '0;
            timestep_q <= 1'b0;
            version_q <= '0;
            trace_leak_q <= '0;
            output_channels_q <= '0;
            wave_base_q <= '0;
            lane_q <= '0;
            spatial_q <= '0;
            local_error_q <= 1'b0;
            observation_channel_q <= '0;
            observation_spatial_q <= '0;
            observation_last_q <= 1'b0;
            o_layer_done <= 1'b0;
        end else begin
            o_layer_done <= 1'b0;
            if (i_clear_error && !o_busy)
                local_error_q <= 1'b0;

            if (i_cfg_valid) begin
                if (!o_cfg_ready || i_cfg_output_channels == 0 ||
                    i_cfg_output_channels > MAX_OUTPUT_CHANNELS ||
                    i_cfg_pooled_width != 2 || i_cfg_pooled_height != 2 ||
                    i_cfg_version == 0 ||
                    (PATH_CONTEXTS == 1 && i_cfg_path)) begin
                    local_error_q <= 1'b1;
                end else begin
                    active_q <= 1'b1;
                    layer_done_seen_q <= 1'b0;
                    output_slot_q <= i_cfg_output_slot;
                    path_q <= i_cfg_path;
                    sample_q <= i_cfg_sample;
                    timestep_q <= i_cfg_timestep;
                    version_q <= i_cfg_version;
                    trace_leak_q <= i_cfg_trace_leak_q8_8;
                    output_channels_q <= i_cfg_output_channels;
                    wave_base_q <= '0;
                    lane_q <= '0;
                    spatial_q <= '0;
                end
            end

            if (i_write_valid && active_q && !mapping_legal)
                local_error_q <= 1'b1;

            if (selected_write_fire) begin
                observation_channel_q <= CHANNEL_WIDTH'(current_channel);
                observation_spatial_q <= spatial_q;
                observation_last_q <=
                    current_channel+1 == output_channels_q && spatial_q == 3;
                if (lane_q+1 == current_wave_count) begin
                    lane_q <= '0;
                    if (spatial_q == 3) begin
                        spatial_q <= '0;
                        wave_base_q <= wave_base_q + current_wave_count;
                    end else begin
                        spatial_q <= spatial_q + 1'b1;
                    end
                end else begin
                    lane_q <= lane_q + 1'b1;
                end
            end

            if (i_layer_done && active_q)
                layer_done_seen_q <= 1'b1;
            else if (i_layer_done && !active_q)
                o_layer_done <= 1'b1;

            if (active_q && (layer_done_seen_q || i_layer_done) &&
                trace_idle) begin
                if (wave_base_q != output_channels_q || lane_q != 0 ||
                    spatial_q != 0)
                    local_error_q <= 1'b1;
                active_q <= 1'b0;
                layer_done_seen_q <= 1'b0;
                o_layer_done <= 1'b1;
            end
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_trace_addr;
    always_comb unused_trace_addr = ^trace_output_addr;
    /* verilator lint_on UNUSEDSIGNAL */

    initial begin
        if (PATH_CONTEXTS < 1 || PATH_CONTEXTS > 2)
            $error("SCNN Conv8 trace fork supports one or two path contexts");
    end
endmodule

`default_nettype wire
