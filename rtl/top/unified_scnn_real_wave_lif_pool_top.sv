`default_nettype none

// Executes one output-channel wave over one or more input-channel groups.
// The immutable FC path is external; this is an SCNN-only side datapath.
module unified_scnn_real_wave_lif_pool_top #(
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
    output logic                                      o_wave_done,
    output logic                                      o_sticky_error
);
    logic wave_active_q;
    logic group_active_q;
    logic all_groups_issued_q;
    logic protocol_error_q;
    logic [10:0] expected_input_base_q;
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
    logic input_slot_q;
    logic [FEATURE_ADDR_WIDTH-1:0] feature_base_q;
    logic [Q16_ADDR_WIDTH-1:0] weight_q16_base_q;
    logic [STATE_ADDR_WIDTH-1:0] state_base_q;
    logic group_first_q;
    logic group_final_q;
    logic [ENGINES-1:0] wave_engine_mask;
    logic wave_request_legal;
    logic group_request_legal;
    logic wave_start_fire;
    logic group_start_fire;

    logic conv_start_ready;
    logic conv_result_valid;
    logic conv_result_ready;
    logic [ENGINES-1:0] conv_result_mask;
    logic [ENGINES*STATE_ADDR_WIDTH-1:0] conv_result_addr_flat;
    logic [STATE_ADDR_WIDTH-1:0] conv_result_spatial;
    logic [ENGINES*ACC_WIDTH-1:0] conv_result_accumulator_flat;
    logic conv_busy;
    logic conv_done;
    logic conv_config_error;
    logic conv_protocol_error;
    logic downstream_start_ready;
    logic downstream_partial_ready;
    logic downstream_busy;
    logic downstream_done;
    logic downstream_error;

    always_comb begin
        wave_engine_mask = '0;
        for (integer engine = 0; engine < ENGINES; engine++)
            if (engine < integer'(i_output_channel_count))
                wave_engine_mask[engine] = 1'b1;
        wave_request_legal = i_owner_enable && i_in_width != 0 &&
            i_in_height != 0 && i_out_width != 0 && i_out_height != 0 &&
            i_total_in_channels != 0 && i_total_out_channels != 0 &&
            i_output_channel_count != 0 &&
            integer'(i_output_channel_count) <= ENGINES &&
            integer'(i_output_channel_base) +
                integer'(i_output_channel_count) <=
                integer'(i_total_out_channels) &&
            i_sample_version != 0;
        group_request_legal = wave_active_q && !all_groups_issued_q &&
            i_input_channel_count != 0 &&
            integer'(i_input_channel_count) <= LANES &&
            {1'b0, i_input_channel_base} == expected_input_base_q &&
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

    assign o_wave_start_ready = i_owner_enable && !wave_active_q &&
        !group_active_q && downstream_start_ready && !o_sticky_error;
    assign wave_start_fire = i_wave_start && o_wave_start_ready &&
                             wave_request_legal;
    assign o_group_start_ready = wave_active_q && !group_active_q &&
        !all_groups_issued_q && conv_start_ready && downstream_partial_ready &&
        !o_sticky_error;
    assign group_start_fire = i_group_start && o_group_start_ready &&
                              group_request_legal;
    assign o_group_done = conv_done;
    assign o_wave_done = downstream_done;
    assign o_busy = wave_active_q || group_active_q || conv_busy ||
                    downstream_busy;
    assign o_sticky_error = protocol_error_q || conv_config_error ||
                            conv_protocol_error || downstream_error;

    unified_scnn_q8_8_channel_wave_conv_engine #(
        .ENGINES(ENGINES), .LANES(LANES),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(STATE_ADDR_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_conv (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_owner_enable(i_owner_enable),
        .i_clear_error(i_clear_error), .i_start(group_start_fire),
        .o_start_ready(conv_start_ready), .i_in_width(in_width_q),
        .i_in_height(in_height_q),
        .i_total_in_channels(total_in_channels_q), .i_out_width(out_width_q),
        .i_out_height(out_height_q),
        .i_total_out_channels(total_out_channels_q),
        .i_input_channel_base(i_input_channel_base),
        .i_input_channel_count(i_input_channel_count),
        .i_output_channel_base(output_channel_base_q),
        .i_output_channel_count(output_channel_count_q),
        .i_stride(stride_q), .i_padding(padding_q),
        .i_input_slot(input_slot_q), .i_feature_base(feature_base_q),
        .i_weight_q16_base(weight_q16_base_q), .i_output_base(state_base_q),
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
        .o_result_valid(conv_result_valid),
        .i_result_ready(conv_result_ready), .o_result_mask(conv_result_mask),
        .o_result_output_addr_flat(conv_result_addr_flat),
        .o_result_spatial_addr(conv_result_spatial),
        .o_result_accumulator_flat(conv_result_accumulator_flat),
        .o_busy(conv_busy), .o_done(conv_done),
        .o_config_error(conv_config_error),
        .o_protocol_error(conv_protocol_error)
    );

    unified_scnn_wave_finalize_lif_pool #(
        .ENGINES(ENGINES), .MAX_WIDTH(MAX_WIDTH),
        .SPATIAL_DEPTH(SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SPATIAL_ADDR_WIDTH),
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) u_downstream (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_start(wave_start_fire), .o_start_ready(downstream_start_ready),
        .i_width(i_out_width), .i_height(i_out_height),
        .i_output_channels(i_total_out_channels),
        .i_channel_base(16'(i_output_channel_base)),
        .i_engine_mask(wave_engine_mask), .i_pool_mode(i_pool_mode),
        .i_output_slot(i_output_slot), .i_sample_version(i_sample_version),
        .i_leak_q8_8(i_leak_q8_8),
        .i_threshold_q8_8(i_threshold_q8_8),
        .i_reset_to_zero(i_reset_to_zero),
        .i_partial_valid(conv_result_valid),
        .o_partial_ready(downstream_partial_ready),
        .i_partial_engine_mask(conv_result_mask),
        .i_partial_spatial_addr(
            conv_result_spatial[SPATIAL_ADDR_WIDTH-1:0]),
        .i_output_addr_flat(conv_result_addr_flat),
        .i_partial_flat(conv_result_accumulator_flat),
        .i_accumulator_clear(group_first_q),
        .i_accumulator_finalize(group_final_q),
        .o_write_valid(o_feature_write_valid),
        .i_write_ready(i_feature_write_ready),
        .o_write_slot(o_feature_write_slot),
        .o_write_bank(o_feature_write_bank),
        .o_write_addr(o_feature_write_addr),
        .o_write_data_q8_8(o_feature_write_data),
        .o_busy(downstream_busy), .o_done(downstream_done),
        .o_sticky_error(downstream_error)
    );
    assign conv_result_ready = downstream_partial_ready;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            wave_active_q <= 1'b0;
            group_active_q <= 1'b0;
            all_groups_issued_q <= 1'b0;
            protocol_error_q <= 1'b0;
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
            input_slot_q <= 1'b0;
            feature_base_q <= '0;
            weight_q16_base_q <= '0;
            state_base_q <= '0;
            group_first_q <= 1'b0;
            group_final_q <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy)
                protocol_error_q <= 1'b0;
            if (i_wave_start && o_wave_start_ready && !wave_request_legal)
                protocol_error_q <= 1'b1;
            if (wave_start_fire) begin
                wave_active_q <= 1'b1;
                all_groups_issued_q <= 1'b0;
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
                input_slot_q <= i_input_slot;
                feature_base_q <= i_feature_base;
                weight_q16_base_q <= i_weight_q16_base;
                state_base_q <= i_state_base;
            end
            if (i_group_start && o_group_start_ready && !group_request_legal)
                protocol_error_q <= 1'b1;
            if (group_start_fire) begin
                group_active_q <= 1'b1;
                group_first_q <= i_first_input_group;
                group_final_q <= i_final_input_group;
                expected_input_base_q <= {1'b0, i_input_channel_base} +
                    11'(i_input_channel_count);
                if (i_final_input_group)
                    all_groups_issued_q <= 1'b1;
            end
            if (conv_done)
                group_active_q <= 1'b0;
            if (conv_result_valid &&
                integer'(conv_result_spatial) >= SPATIAL_DEPTH)
                protocol_error_q <= 1'b1;
            if (downstream_done) begin
                wave_active_q <= 1'b0;
                if (!all_groups_issued_q)
                    protocol_error_q <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
