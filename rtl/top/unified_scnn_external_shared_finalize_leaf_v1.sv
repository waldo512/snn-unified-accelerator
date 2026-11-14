`default_nettype none

// Pure SCNN postprocess leaf. It schedules groups onto an external, unique
// Wave3 owner and accepts that owner's raw accumulator stream. This module
// owns no convolution arithmetic or shared numeric resources.
module unified_scnn_external_shared_finalize_leaf_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int MAX_WIDTH = 32,
    parameter int FEATURE_ADDR_WIDTH = 14,
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
    input  logic                                      i_owner_enable,
    input  logic                                      i_clear_error,

    input  logic                                      i_wave_start,
    output logic                                      o_wave_start_ready,
    input  logic [15:0]                               i_in_width,
    input  logic [15:0]                               i_in_height,
    input  logic [15:0]                               i_total_in_channels,
    input  logic [15:0]                               i_out_width,
    input  logic [15:0]                               i_out_height,
    input  logic [15:0]                               i_total_out_channels,
    input  logic [9:0]                                i_output_channel_base,
    input  logic [9:0]                                i_output_channel_count,
    input  logic [7:0]                                i_stride,
    input  logic [7:0]                                i_padding,
    input  logic [1:0]                                i_pool_mode,
    input  logic                                      i_input_slot,
    input  logic                                      i_output_slot,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_feature_base,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_weight_q16_base,
    input  logic [STATE_ADDR_WIDTH-1:0]               i_state_base,
    input  logic [VERSION_WIDTH-1:0]                  i_sample_version,
    input  logic signed [15:0]                        i_leak_q8_8,
    input  logic signed [15:0]                        i_threshold_q8_8,
    input  logic                                      i_reset_to_zero,

    input  logic                                      i_group_start,
    output logic                                      o_group_start_ready,
    input  logic [9:0]                                i_input_channel_base,
    input  logic [9:0]                                i_input_channel_count,
    input  logic                                      i_first_input_group,
    input  logic                                      i_final_input_group,
    output logic                                      o_group_done,

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

    output logic [VERSION_WIDTH-1:0]                  o_locked_sample_version,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_done,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_WAVE_CONFIG = 8'h01;
    localparam logic [7:0] ERROR_GROUP_ORDER = 8'h02;
    localparam logic [7:0] ERROR_RAW_BOUNDS = 8'h03;
    localparam logic [7:0] ERROR_DONE_ORDER = 8'h04;
    localparam logic [7:0] ERROR_RAW_ORDER = 8'h05;

    logic wave_active_q;
    logic group_active_q;
    logic conv_started_q;
    logic conv_done_seen_q;
    logic downstream_started_q;
    logic all_groups_issued_q;
    logic final_group_completed_q;
    logic writeback_done_seen_q;
    logic local_sticky_error_q;
    logic [7:0] local_error_code_q;
    logic [15:0] expected_input_base_q;

    logic [15:0] in_width_q;
    logic [15:0] in_height_q;
    logic [15:0] total_in_channels_q;
    logic [15:0] out_width_q;
    logic [15:0] out_height_q;
    logic [15:0] total_out_channels_q;
    logic [9:0] output_channel_base_q;
    logic [9:0] output_channel_count_q;
    logic [7:0] stride_q;
    logic [7:0] padding_q;
    logic [1:0] pool_mode_q;
    logic input_slot_q;
    logic output_slot_q;
    logic [FEATURE_ADDR_WIDTH-1:0] feature_base_q;
    logic [Q16_ADDR_WIDTH-1:0] weight_q16_base_q;
    logic [STATE_ADDR_WIDTH-1:0] state_base_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic signed [15:0] leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic [ENGINES-1:0] wave_engine_mask_q;
    logic [9:0] group_input_base_q;
    logic [9:0] group_input_count_q;
    logic group_first_q;
    logic group_final_q;

    logic wave_config_legal;
    logic group_config_legal;
    logic [ENGINES-1:0] requested_engine_mask;
    logic wave_cmd_fire;
    logic group_cmd_fire;
    logic conv_start_fire;
    logic downstream_start;
    logic downstream_start_ready;
    logic downstream_partial_ready;
    logic downstream_busy;
    logic downstream_done;
    logic downstream_error;
    logic raw_spatial_in_range;
    logic raw_payload_active;

    always_comb begin
        requested_engine_mask = '0;
        for (integer engine = 0; engine < ENGINES; engine++)
            if (engine < integer'(i_output_channel_count))
                requested_engine_mask[engine] = 1'b1;

        wave_config_legal = i_owner_enable && i_in_width != 0 &&
            i_in_height != 0 && i_total_in_channels != 0 &&
            i_total_in_channels <= 16'd1024 && i_out_width != 0 &&
            i_out_height != 0 && i_total_out_channels != 0 &&
            i_output_channel_count != 0 &&
            integer'(i_output_channel_count) <= ENGINES &&
            integer'(i_output_channel_base) +
                integer'(i_output_channel_count) <=
                integer'(i_total_out_channels) &&
            integer'(i_out_width) <= MAX_WIDTH &&
            integer'(i_out_width) * integer'(i_out_height) <=
                SPATIAL_DEPTH && i_stride != 0 && i_pool_mode <= 2'd2 &&
            i_sample_version != 0 &&
            integer'(i_state_base) +
                integer'(i_total_out_channels) *
                integer'(i_out_width) * integer'(i_out_height) <=
                NUM_NEURONS;

        group_config_legal = wave_active_q && !all_groups_issued_q &&
            i_input_channel_count != 0 &&
            integer'(i_input_channel_count) <= LANES &&
            {6'd0, i_input_channel_base} == expected_input_base_q &&
            integer'(i_input_channel_base) +
                integer'(i_input_channel_count) <=
                integer'(total_in_channels_q) &&
            (expected_input_base_q == 0 ? i_first_input_group :
                                         !i_first_input_group) &&
            (i_final_input_group ==
                (integer'(i_input_channel_base) +
                 integer'(i_input_channel_count) ==
                 integer'(total_in_channels_q)));
    end

    assign o_idle = !wave_active_q && !group_active_q && i_conv_idle &&
                    !downstream_busy;
    assign o_busy = !o_idle;
    assign o_sticky_error = local_sticky_error_q || i_conv_sticky_error ||
                            downstream_error;
    always_comb begin
        if (local_sticky_error_q)
            o_error_code = local_error_code_q;
        else if (i_conv_sticky_error)
            o_error_code = 8'h80 | i_conv_error_code;
        else if (downstream_error)
            o_error_code = 8'hc0;
        else
            o_error_code = ERROR_NONE;
    end

    assign o_wave_start_ready = i_owner_enable && o_idle &&
                                !o_sticky_error;
    assign wave_cmd_fire = i_wave_start && o_wave_start_ready;
    assign o_group_start_ready = wave_active_q && !group_active_q &&
                                 !all_groups_issued_q && !o_sticky_error;
    assign group_cmd_fire = i_group_start && o_group_start_ready;

    assign o_conv_start_valid = group_active_q && !conv_started_q &&
        !o_sticky_error && (downstream_started_q || downstream_start_ready);
    assign conv_start_fire = o_conv_start_valid && i_conv_start_ready;
    assign downstream_start = conv_start_fire && !downstream_started_q;

    assign o_conv_in_width = in_width_q;
    assign o_conv_in_height = in_height_q;
    assign o_conv_total_in_channels = total_in_channels_q;
    assign o_conv_out_width = out_width_q;
    assign o_conv_out_height = out_height_q;
    assign o_conv_total_out_channels = total_out_channels_q;
    assign o_conv_input_channel_base = group_input_base_q;
    assign o_conv_input_channel_count = group_input_count_q;
    assign o_conv_output_channel_base = output_channel_base_q;
    assign o_conv_output_channel_count = output_channel_count_q;
    assign o_conv_stride = stride_q;
    assign o_conv_padding = padding_q;
    assign o_conv_input_slot = input_slot_q;
    assign o_conv_feature_base = feature_base_q;
    assign o_conv_weight_q16_base = weight_q16_base_q;
    assign o_conv_output_base = state_base_q;

    assign raw_spatial_in_range = integer'(i_raw_result_spatial_addr) <
                                  SPATIAL_DEPTH;
    assign raw_payload_active = group_active_q && conv_started_q;
    assign o_raw_result_ready = raw_payload_active &&
        downstream_partial_ready && raw_spatial_in_range && !o_sticky_error;
    assign o_raw_result_fire = i_raw_result_valid && o_raw_result_ready;
    assign o_raw_result_group_first = group_first_q;
    assign o_raw_result_group_final = group_final_q;
    assign o_locked_sample_version = sample_version_q;

    unified_scnn_wave_finalize_lif_pool_soft_abort_v1 #(
        .ENGINES(ENGINES), .MAX_WIDTH(MAX_WIDTH),
        .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .ENABLE_TP_OBSERVER(ENABLE_TP_OBSERVER),
        .STATE_INIT_FILE(STATE_INIT_FILE),
        .SURROGATE_ROM_FILE(SURROGATE_ROM_FILE)
    ) u_finalize (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start(downstream_start), .o_start_ready(downstream_start_ready),
        .i_width(out_width_q), .i_height(out_height_q),
        .i_output_channels(total_out_channels_q),
        .i_channel_base(16'(output_channel_base_q)),
        .i_engine_mask(wave_engine_mask_q), .i_pool_mode(pool_mode_q),
        .i_output_slot(output_slot_q), .i_sample_version(sample_version_q),
        .i_leak_q8_8(leak_q), .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .i_partial_valid(i_raw_result_valid && raw_payload_active &&
                         raw_spatial_in_range && !o_sticky_error),
        .o_partial_ready(downstream_partial_ready),
        .i_partial_engine_mask(i_raw_result_mask),
        .i_partial_spatial_addr(
            i_raw_result_spatial_addr[SPATIAL_ADDR_WIDTH-1:0]),
        .i_output_addr_flat(i_raw_result_output_addr_flat),
        .i_partial_flat(i_raw_result_accumulator_flat),
        .i_accumulator_clear(group_first_q),
        .i_accumulator_finalize(group_final_q),
        .o_write_valid(o_write_valid), .i_write_ready(i_write_ready),
        .o_write_slot(o_write_slot), .o_write_bank(o_write_bank),
        .o_write_addr(o_write_addr),
        .o_write_data_q8_8(o_write_data_q8_8),
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
        .o_busy(downstream_busy), .o_done(downstream_done),
        .o_sticky_error(downstream_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            wave_active_q <= 1'b0;
            group_active_q <= 1'b0;
            conv_started_q <= 1'b0;
            conv_done_seen_q <= 1'b0;
            downstream_started_q <= 1'b0;
            all_groups_issued_q <= 1'b0;
            final_group_completed_q <= 1'b0;
            writeback_done_seen_q <= 1'b0;
            local_sticky_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
            expected_input_base_q <= '0;
            in_width_q <= '0;
            in_height_q <= '0;
            total_in_channels_q <= '0;
            out_width_q <= '0;
            out_height_q <= '0;
            total_out_channels_q <= '0;
            output_channel_base_q <= '0;
            output_channel_count_q <= '0;
            stride_q <= '0;
            padding_q <= '0;
            pool_mode_q <= '0;
            input_slot_q <= 1'b0;
            output_slot_q <= 1'b0;
            feature_base_q <= '0;
            weight_q16_base_q <= '0;
            state_base_q <= '0;
            sample_version_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            wave_engine_mask_q <= '0;
            group_input_base_q <= '0;
            group_input_count_q <= '0;
            group_first_q <= 1'b0;
            group_final_q <= 1'b0;
            o_group_done <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_group_done <= 1'b0;
            o_done <= 1'b0;
            if (i_abort) begin
                wave_active_q <= 1'b0;
                group_active_q <= 1'b0;
                conv_started_q <= 1'b0;
                conv_done_seen_q <= 1'b0;
                downstream_started_q <= 1'b0;
                all_groups_issued_q <= 1'b0;
                final_group_completed_q <= 1'b0;
                writeback_done_seen_q <= 1'b0;
                expected_input_base_q <= '0;
                in_width_q <= '0;
                in_height_q <= '0;
                total_in_channels_q <= '0;
                out_width_q <= '0;
                out_height_q <= '0;
                total_out_channels_q <= '0;
                output_channel_base_q <= '0;
                output_channel_count_q <= '0;
                stride_q <= '0;
                padding_q <= '0;
                pool_mode_q <= '0;
                input_slot_q <= 1'b0;
                output_slot_q <= 1'b0;
                feature_base_q <= '0;
                weight_q16_base_q <= '0;
                state_base_q <= '0;
                sample_version_q <= '0;
                leak_q <= '0;
                threshold_q <= '0;
                reset_to_zero_q <= 1'b0;
                wave_engine_mask_q <= '0;
                group_input_base_q <= '0;
                group_input_count_q <= '0;
                group_first_q <= 1'b0;
                group_final_q <= 1'b0;
            end else begin
            if (i_clear_error && o_idle) begin
                local_sticky_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (wave_cmd_fire) begin
                if (!wave_config_legal) begin
                    local_sticky_error_q <= 1'b1;
                    local_error_code_q <= ERROR_WAVE_CONFIG;
                end else begin
                    wave_active_q <= 1'b1;
                    downstream_started_q <= 1'b0;
                    all_groups_issued_q <= 1'b0;
                    final_group_completed_q <= 1'b0;
                    writeback_done_seen_q <= 1'b0;
                    expected_input_base_q <= '0;
                    in_width_q <= i_in_width;
                    in_height_q <= i_in_height;
                    total_in_channels_q <= i_total_in_channels;
                    out_width_q <= i_out_width;
                    out_height_q <= i_out_height;
                    total_out_channels_q <= i_total_out_channels;
                    output_channel_base_q <= i_output_channel_base;
                    output_channel_count_q <= i_output_channel_count;
                    stride_q <= i_stride;
                    padding_q <= i_padding;
                    pool_mode_q <= i_pool_mode;
                    input_slot_q <= i_input_slot;
                    output_slot_q <= i_output_slot;
                    feature_base_q <= i_feature_base;
                    weight_q16_base_q <= i_weight_q16_base;
                    state_base_q <= i_state_base;
                    sample_version_q <= i_sample_version;
                    leak_q <= i_leak_q8_8;
                    threshold_q <= i_threshold_q8_8;
                    reset_to_zero_q <= i_reset_to_zero;
                    wave_engine_mask_q <= requested_engine_mask;
                end
            end

            if (group_cmd_fire) begin
                if (!group_config_legal) begin
                    local_sticky_error_q <= 1'b1;
                    local_error_code_q <= ERROR_GROUP_ORDER;
                    if (!downstream_started_q)
                        wave_active_q <= 1'b0;
                end else begin
                    group_active_q <= 1'b1;
                    conv_started_q <= 1'b0;
                    conv_done_seen_q <= 1'b0;
                    group_input_base_q <= i_input_channel_base;
                    group_input_count_q <= i_input_channel_count;
                    group_first_q <= i_first_input_group;
                    group_final_q <= i_final_input_group;
                    expected_input_base_q <=
                        {6'd0, i_input_channel_base} +
                        16'(i_input_channel_count);
                    if (i_final_input_group)
                        all_groups_issued_q <= 1'b1;
                end
            end

            if (conv_start_fire) begin
                conv_started_q <= 1'b1;
                if (!downstream_started_q)
                    downstream_started_q <= 1'b1;
            end

            if (i_raw_result_valid && !raw_payload_active &&
                !local_sticky_error_q) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_RAW_ORDER;
            end else if (i_raw_result_valid && !raw_spatial_in_range &&
                !local_sticky_error_q) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_RAW_BOUNDS;
            end

            if (i_conv_done && conv_started_q)
                conv_done_seen_q <= 1'b1;
            if (group_active_q && conv_started_q && conv_done_seen_q &&
                i_conv_idle) begin
                group_active_q <= 1'b0;
                conv_started_q <= 1'b0;
                conv_done_seen_q <= 1'b0;
                o_group_done <= 1'b1;
                if (group_final_q)
                    final_group_completed_q <= 1'b1;
            end

            if (downstream_done)
                writeback_done_seen_q <= 1'b1;
            if (downstream_done && !all_groups_issued_q &&
                !local_sticky_error_q) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_DONE_ORDER;
            end

            if (wave_active_q && final_group_completed_q &&
                (writeback_done_seen_q || downstream_done) &&
                !group_active_q && i_conv_idle &&
                (!ENABLE_TP_OBSERVER || !downstream_busy)) begin
                wave_active_q <= 1'b0;
                downstream_started_q <= 1'b0;
                all_groups_issued_q <= 1'b0;
                final_group_completed_q <= 1'b0;
                writeback_done_seen_q <= 1'b0;
                expected_input_base_q <= '0;
                o_done <= 1'b1;
            end
            end
        end
    end
endmodule

`default_nettype wire
