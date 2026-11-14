`default_nettype none

module unified_scnn_q8_8_tiled_layer_engine #(
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int BANK_ADDR_WIDTH = 14,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_start,
    output logic                           o_start_ready,
    input  logic [15:0]                    i_in_width,
    input  logic [15:0]                    i_in_height,
    input  logic [15:0]                    i_in_channels,
    input  logic [15:0]                    i_out_width,
    input  logic [15:0]                    i_out_height,
    input  logic [15:0]                    i_out_channels,
    input  logic [7:0]                     i_stride,
    input  logic [7:0]                     i_padding,
    input  logic [1:0]                     i_pool_mode,
    input  logic                           i_input_slot,
    input  logic                           i_output_slot,
    input  logic [Q16_ADDR_WIDTH-1:0]      i_weight_q16_base,
    input  logic [STATE_ADDR_WIDTH-1:0]    i_state_base,
    input  logic [VERSION_WIDTH-1:0]       i_sample_version,
    input  logic signed [15:0]             i_leak_q8_8,
    input  logic signed [15:0]             i_threshold_q8_8,
    input  logic                           i_reset_to_zero,

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
    logic start_fire;
    logic generator_start_ready;
    logic generator_valid;
    logic generator_ready;
    logic [255:0] generator_input_addr_flat;
    logic [255:0] generator_weight_addr_flat;
    logic [31:0] generator_feature_bank_flat;
    logic [255:0] generator_feature_bank_addr_flat;
    logic generator_bank_conflict;
    logic [7:0] generator_lane_mask;
    logic [31:0] generator_output_addr;
    logic generator_first;
    logic generator_last;
    logic generator_busy;
    logic generator_done;
    logic generator_config_error;
    logic pool_start_ready;
    logic pool_ready;
    logic pool_busy;
    logic pool_done;
    logic pool_config_error;
    logic pool_protocol_error;
    logic reader_request_valid;
    logic reader_request_ready;
    logic [Q16_ADDR_WIDTH-1:0] reader_q16_base;
    logic [7:0] reader_request_lane_mask;
    logic reader_weight_valid;
    logic reader_weight_ready;
    logic [127:0] reader_weight_flat;
    logic [7:0] reader_lane_mask;
    logic beat_inflight_q;
    logic [7:0] beat_lane_mask_q;
    logic beat_first_q;
    logic beat_last_q;
    logic [STATE_ADDR_WIDTH-1:0] beat_output_addr_q;
    logic engine_ready;
    logic engine_input_valid;
    logic accumulator_commit;
    logic engine_output_valid;
    logic [STATE_ADDR_WIDTH-1:0] engine_output_addr;
    logic [VERSION_WIDTH-1:0] engine_output_version;
    logic signed [15:0] engine_current;
    logic [15:0] engine_spike;
    logic signed [15:0] engine_membrane_next;
    logic engine_idle;
    logic engine_protocol_error;
    logic engine_state_error;
    logic layer_busy_q;
    logic protocol_error_q;

    assign o_start_ready = !layer_busy_q && generator_start_ready &&
                           pool_start_ready;
    assign start_fire = i_start && o_start_ready;
    assign o_busy = layer_busy_q;
    assign o_config_error = generator_config_error || pool_config_error;
    assign o_protocol_error = protocol_error_q || pool_protocol_error ||
                              engine_protocol_error;
    assign o_state_error = engine_state_error;

    assign generator_ready = !beat_inflight_q &&
        i_feature_read_ready && reader_request_ready;
    assign o_feature_read_valid = generator_valid && !beat_inflight_q &&
                                  reader_request_ready;
    assign o_feature_read_slot = i_input_slot;
    assign o_feature_read_bank_flat = generator_feature_bank_flat;
    assign o_feature_read_lane_mask = generator_lane_mask;
    assign reader_request_valid = generator_valid && !beat_inflight_q &&
                                  i_feature_read_ready;
    assign reader_q16_base = Q16_ADDR_WIDTH'(
        generator_weight_addr_flat[31:0]);
    assign reader_request_lane_mask = generator_lane_mask;

    always_comb begin
        o_feature_read_addr_flat = '0;
        for (int lane = 0; lane < 8; lane = lane + 1)
            o_feature_read_addr_flat[lane*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH] =
                BANK_ADDR_WIDTH'(generator_feature_bank_addr_flat[
                    lane*32 +: 32]);
    end

    assign engine_input_valid = beat_inflight_q && i_feature_valid &&
                                reader_weight_valid;
    assign o_feature_ready = beat_inflight_q && reader_weight_valid &&
                             engine_ready;
    assign reader_weight_ready = beat_inflight_q && i_feature_valid &&
                                 engine_ready;

    unified_scnn_conv_k8_address_generator #(
        .LANES(8), .ADDRESS_WIDTH(32), .KERNEL_SIZE(3)
    ) u_generator (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_start(start_fire),
        .o_start_ready(generator_start_ready),
        .i_in_width(i_in_width), .i_in_height(i_in_height),
        .i_in_channels(i_in_channels), .i_out_width(i_out_width),
        .i_out_height(i_out_height), .i_out_channels(i_out_channels),
        .i_stride(i_stride), .i_padding(i_padding), .i_input_base(32'd0),
        .i_weight_base(32'(i_weight_q16_base)),
        .i_output_base(32'(i_state_base)),
        .o_valid(generator_valid), .i_ready(generator_ready),
        .i_result_commit(accumulator_commit),
        .o_input_addr_flat(generator_input_addr_flat),
        .o_weight_addr_flat(generator_weight_addr_flat),
        .o_feature_bank_flat(generator_feature_bank_flat),
        .o_feature_bank_addr_flat(generator_feature_bank_addr_flat),
        .o_feature_bank_conflict(generator_bank_conflict),
        .o_lane_mask(generator_lane_mask),
        .o_output_addr(generator_output_addr),
        .o_accumulate_first(generator_first),
        .o_accumulate_last(generator_last), .o_busy(generator_busy),
        .o_done(generator_done), .o_config_error(generator_config_error)
    );

    unified_q16_k8_packed_reader #(
        .Q64_ADDRESS_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDRESS_WIDTH(Q16_ADDR_WIDTH)
    ) u_weight_reader (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_request_valid(reader_request_valid),
        .o_request_ready(reader_request_ready), .i_q16_base(reader_q16_base),
        .i_lane_mask(reader_request_lane_mask),
        .o_read_valid(o_weight_read_valid),
        .i_read_ready(i_weight_read_ready),
        .o_read_word_addr0(o_weight_read_addr0),
        .o_read_word_addr1(o_weight_read_addr1),
        .i_read_result_valid(i_weight_result_valid),
        .o_read_result_ready(o_weight_result_ready),
        .i_read_data0(i_weight_data0), .i_read_data1(i_weight_data1),
        .o_weight_valid(reader_weight_valid),
        .i_weight_ready(reader_weight_ready),
        .o_weight_flat(reader_weight_flat), .o_lane_mask(reader_lane_mask)
    );

    unified_scnn_q8_8_conv_lif_engine #(
        .NUM_NEURONS(NUM_NEURONS), .STATE_ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_conv_lif (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(engine_input_valid), .o_ready(engine_ready),
        .i_feature_flat(i_feature_flat), .i_weight_flat(reader_weight_flat),
        .i_lane_mask(beat_lane_mask_q), .i_first(beat_first_q),
        .i_last(beat_last_q), .i_output_addr(beat_output_addr_q),
        .i_sample_version(i_sample_version), .i_leak_q8_8(i_leak_q8_8),
        .i_threshold_q8_8(i_threshold_q8_8),
        .i_reset_to_zero(i_reset_to_zero),
        .o_accumulator_commit(accumulator_commit),
        .o_valid(engine_output_valid), .i_ready(pool_ready),
        .o_output_addr(engine_output_addr),
        .o_sample_version(engine_output_version),
        .o_current_q8_8(engine_current), .o_spike_q8_8(engine_spike),
        .o_membrane_next(engine_membrane_next), .o_idle(engine_idle),
        .o_protocol_error(engine_protocol_error),
        .o_state_error(engine_state_error)
    );

    unified_scnn_pool_writeback #(
        .MAX_WIDTH(32), .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) u_pool_writeback (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_start(start_fire),
        .o_start_ready(pool_start_ready), .i_width(i_out_width),
        .i_height(i_out_height), .i_channels(i_out_channels),
        .i_pool_mode(i_pool_mode), .i_input_base(32'(i_state_base)),
        .i_output_slot(i_output_slot), .i_valid(engine_output_valid),
        .o_ready(pool_ready), .i_addr(32'(engine_output_addr)),
        .i_value_q8_8(engine_spike),
        .o_write_valid(o_feature_write_valid),
        .i_write_ready(i_feature_write_ready),
        .o_write_slot(o_feature_write_slot),
        .o_write_bank(o_feature_write_bank),
        .o_write_addr(o_feature_write_addr),
        .o_write_data_q8_8(o_feature_write_data),
        .o_busy(pool_busy), .o_done(pool_done),
        .o_config_error(pool_config_error),
        .o_protocol_error(pool_protocol_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            beat_inflight_q <= 1'b0;
            beat_lane_mask_q <= '0;
            beat_first_q <= 1'b0;
            beat_last_q <= 1'b0;
            beat_output_addr_q <= '0;
            layer_busy_q <= 1'b0;
            protocol_error_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (start_fire) begin
                layer_busy_q <= 1'b1;
                protocol_error_q <= 1'b0;
            end else if (i_start && !o_start_ready) begin
                protocol_error_q <= 1'b1;
            end
            if (generator_valid && generator_ready) begin
                beat_inflight_q <= 1'b1;
                beat_lane_mask_q <= generator_lane_mask;
                beat_first_q <= generator_first;
                beat_last_q <= generator_last;
                beat_output_addr_q <=
                    generator_output_addr[STATE_ADDR_WIDTH-1:0];
                if (generator_bank_conflict)
                    protocol_error_q <= 1'b1;
            end
            if (engine_input_valid && engine_ready) begin
                beat_inflight_q <= 1'b0;
                if (i_feature_lane_mask != beat_lane_mask_q ||
                    reader_lane_mask != beat_lane_mask_q)
                    protocol_error_q <= 1'b1;
            end
            if (pool_done) begin
                layer_busy_q <= 1'b0;
                o_done <= 1'b1;
                if (!generator_done && generator_busy)
                    protocol_error_q <= 1'b1;
            end
        end
    end

    logic unused_status;
    assign unused_status = &{1'b0, generator_done, generator_busy, pool_busy,
        generator_input_addr_flat,
        engine_output_version, engine_current, engine_membrane_next,
        engine_idle};

endmodule

`default_nettype wire
