`default_nettype none

// Channel-wave address and operand frontend for the single shared numeric array.
// Arithmetic is deliberately outside this module so FC, SRNN and SCNN can share
// one physical twelve-engine side array under the top-level owner.
module unified_scnn_external_side_frontend_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_owner_enable,
    input  logic                                      i_clear_error,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic [15:0]                               i_in_width,
    input  logic [15:0]                               i_in_height,
    input  logic [15:0]                               i_total_in_channels,
    input  logic [15:0]                               i_out_width,
    input  logic [15:0]                               i_out_height,
    input  logic [15:0]                               i_total_out_channels,
    input  logic [9:0]                                i_input_channel_base,
    input  logic [9:0]                                i_input_channel_count,
    input  logic [9:0]                                i_output_channel_base,
    input  logic [9:0]                                i_output_channel_count,
    input  logic [7:0]                                i_stride,
    input  logic [7:0]                                i_padding,
    input  logic                                      i_input_slot,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_feature_base,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_weight_q16_base,
    input  logic [OUTPUT_ADDR_WIDTH-1:0]              i_output_base,

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

    output logic                                      o_side_request_valid,
    input  logic                                      i_side_request_ready,
    output logic [ENGINES-1:0]                        o_side_engine_mask,
    output logic [LANES*16-1:0]                       o_side_feature_flat,
    output logic [ENGINES*LANES*16-1:0]               o_side_weight_flat,
    output logic [LANES-1:0]                          o_side_lane_mask,
    output logic                                      o_side_first,
    output logic                                      o_side_last,
    output logic [9:0]                                o_side_output_channel_base,
    input  logic                                      i_side_result_valid,
    output logic                                      o_side_result_ready,
    input  logic [ENGINES-1:0]                        i_side_result_mask,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_side_result_accumulator_flat,
    input  logic                                      i_side_busy,
    input  logic                                      i_side_idle,
    input  logic                                      i_side_sticky_error,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [ENGINES-1:0]                        o_result_mask,
    output logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      o_result_output_addr_flat,
    output logic [OUTPUT_ADDR_WIDTH-1:0]              o_result_spatial_addr,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_result_accumulator_flat,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_config_error,
    output logic                                      o_protocol_error
);
    localparam int GENERATOR_ADDR_WIDTH =
        FEATURE_ADDR_WIDTH > Q16_ADDR_WIDTH ?
            (FEATURE_ADDR_WIDTH > OUTPUT_ADDR_WIDTH ?
                FEATURE_ADDR_WIDTH : OUTPUT_ADDR_WIDTH) :
            (Q16_ADDR_WIDTH > OUTPUT_ADDR_WIDTH ?
                Q16_ADDR_WIDTH : OUTPUT_ADDR_WIDTH);

    typedef enum logic [1:0] {
        STATE_CAPTURE,
        STATE_REQUEST,
        STATE_WAIT_DATA,
        STATE_WAIT_RESULT
    } state_t;

    state_t state_q;
    logic generator_start_ready;
    logic generator_valid;
    logic generator_ready;
    logic [LANES*4-1:0] generator_feature_bank_flat;
    logic [LANES*GENERATOR_ADDR_WIDTH-1:0]
        generator_feature_addr_wide_flat;
    logic generator_bank_conflict;
    logic [LANES-1:0] generator_lane_mask;
    logic [ENGINES*LANES*GENERATOR_ADDR_WIDTH-1:0]
        generator_weight_addr_wide_flat;
    logic [ENGINES*GENERATOR_ADDR_WIDTH-1:0]
        generator_output_addr_wide_flat;
    logic [GENERATOR_ADDR_WIDTH-1:0] generator_spatial_addr_wide;
    logic [ENGINES-1:0] generator_engine_mask;
    logic generator_first;
    logic generator_last;
    logic generator_busy;
    logic generator_done;
    logic generator_config_error;

    logic [LANES*4-1:0] beat_feature_bank_flat_q;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] beat_feature_addr_flat_q;
    logic [LANES-1:0] beat_lane_mask_q;
    logic [ENGINES*LANES*Q16_ADDR_WIDTH-1:0] beat_weight_addr_flat_q;
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0] beat_output_addr_flat_q;
    logic [OUTPUT_ADDR_WIDTH-1:0] beat_spatial_addr_q;
    logic [ENGINES-1:0] beat_engine_mask_q;
    logic [9:0] beat_output_channel_base_q;
    logic beat_first_q;
    logic beat_last_q;
    logic beat_has_data_q;
    logic feature_request_pending_q;
    logic weight_request_pending_q;
    logic feature_request_accept;
    logic weight_request_accept;
    logic requests_complete;

    logic gather_request_ready;
    logic gather_weight_valid;
    logic gather_weight_ready;
    logic [ENGINES*LANES*16-1:0] gather_weight_flat;
    logic [ENGINES-1:0] gather_engine_mask;
    logic [LANES-1:0] gather_lane_mask;
    logic gather_busy;
    logic gather_error;

    logic local_protocol_error_q;
    logic side_request_fire;
    logic data_fire;
    logic result_payload_legal;
    logic result_fire;

    assign o_start_ready = i_owner_enable && state_q == STATE_CAPTURE &&
                           generator_start_ready && i_side_idle && !gather_busy &&
                           !o_protocol_error;
    assign generator_ready = state_q == STATE_CAPTURE && !o_protocol_error;

    assign o_feature_read_valid = state_q == STATE_REQUEST &&
                                  feature_request_pending_q && !o_protocol_error;
    assign o_feature_read_slot = i_input_slot;
    assign o_feature_read_bank_flat = beat_feature_bank_flat_q;
    assign o_feature_read_addr_flat = beat_feature_addr_flat_q;
    assign o_feature_read_lane_mask = beat_lane_mask_q;
    assign feature_request_accept = o_feature_read_valid && i_feature_read_ready;

    assign weight_request_accept = state_q == STATE_REQUEST &&
                                   weight_request_pending_q &&
                                   gather_request_ready && !o_protocol_error;
    assign requests_complete =
        (!feature_request_pending_q || feature_request_accept) &&
        (!weight_request_pending_q || weight_request_accept);

    assign o_side_request_valid = state_q == STATE_WAIT_DATA &&
                                  (!beat_has_data_q ||
                                   (i_feature_valid && gather_weight_valid)) &&
                                  i_owner_enable && !o_protocol_error;
    assign o_side_engine_mask = beat_engine_mask_q;
    assign o_side_feature_flat = beat_has_data_q ? i_feature_flat : '0;
    assign o_side_weight_flat = beat_has_data_q ? gather_weight_flat : '0;
    assign o_side_lane_mask = beat_lane_mask_q;
    assign o_side_first = beat_first_q;
    assign o_side_last = beat_last_q;
    assign o_side_output_channel_base = beat_output_channel_base_q;
    assign side_request_fire = o_side_request_valid && i_side_request_ready;

    assign o_feature_ready = state_q == STATE_WAIT_DATA && beat_has_data_q &&
                             gather_weight_valid && i_side_request_ready &&
                             i_owner_enable && !o_protocol_error;
    assign gather_weight_ready = state_q == STATE_WAIT_DATA && beat_has_data_q &&
                                 i_feature_valid && i_side_request_ready &&
                                 i_owner_enable && !o_protocol_error;
    assign data_fire = side_request_fire;

    assign result_payload_legal = i_side_result_mask == beat_engine_mask_q;
    assign o_result_valid = state_q == STATE_WAIT_RESULT &&
                            i_side_result_valid && result_payload_legal &&
                            !o_protocol_error;
    assign o_side_result_ready = state_q == STATE_WAIT_RESULT &&
                                 i_result_ready && result_payload_legal &&
                                 !o_protocol_error;
    assign result_fire = o_result_valid && i_result_ready;
    assign o_result_mask = i_side_result_mask;
    assign o_result_output_addr_flat = beat_output_addr_flat_q;
    assign o_result_spatial_addr = beat_spatial_addr_q;
    assign o_result_accumulator_flat = i_side_result_accumulator_flat;

    assign o_busy = generator_busy || state_q != STATE_CAPTURE ||
                    gather_busy || i_side_busy;
    assign o_done = generator_done;
    assign o_config_error = generator_config_error;
    assign o_protocol_error = local_protocol_error_q || gather_error ||
                              i_side_sticky_error;

    unified_scnn_conv_k8_channel_wave_address_generator #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .ADDRESS_WIDTH(GENERATOR_ADDR_WIDTH),
        .KERNEL_SIZE(3)
    ) u_generator (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(i_start && o_start_ready),
        .o_start_ready(generator_start_ready),
        .i_in_width(i_in_width),
        .i_in_height(i_in_height),
        .i_total_in_channels(i_total_in_channels),
        .i_out_width(i_out_width),
        .i_out_height(i_out_height),
        .i_total_out_channels(i_total_out_channels),
        .i_input_channel_base(i_input_channel_base),
        .i_input_channel_count(i_input_channel_count),
        .i_output_channel_base(i_output_channel_base),
        .i_output_channel_count(i_output_channel_count),
        .i_stride(i_stride),
        .i_padding(i_padding),
        .i_input_base(GENERATOR_ADDR_WIDTH'(i_feature_base)),
        .i_weight_q16_base(GENERATOR_ADDR_WIDTH'(i_weight_q16_base)),
        .i_output_base(GENERATOR_ADDR_WIDTH'(i_output_base)),
        .o_valid(generator_valid),
        .i_ready(generator_ready),
        .i_result_commit(result_fire),
        .o_feature_bank_flat(generator_feature_bank_flat),
        .o_feature_bank_addr_flat(generator_feature_addr_wide_flat),
        .o_feature_bank_conflict(generator_bank_conflict),
        .o_lane_mask(generator_lane_mask),
        .o_weight_q16_addr_flat(generator_weight_addr_wide_flat),
        .o_output_addr_flat(generator_output_addr_wide_flat),
        .o_spatial_addr(generator_spatial_addr_wide),
        .o_engine_mask(generator_engine_mask),
        .o_accumulate_first(generator_first),
        .o_accumulate_last(generator_last),
        .o_busy(generator_busy),
        .o_done(generator_done),
        .o_config_error(generator_config_error)
    );

    unified_scnn_q16_wave_weight_gather #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .Q64_ADDRESS_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDRESS_WIDTH(Q16_ADDR_WIDTH)
    ) u_weight_gather (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_request_valid(state_q == STATE_REQUEST &&
                         weight_request_pending_q && !o_protocol_error),
        .o_request_ready(gather_request_ready),
        .i_engine_mask(beat_engine_mask_q),
        .i_lane_mask(beat_lane_mask_q),
        .i_weight_q16_addr_flat(beat_weight_addr_flat_q),
        .o_read_valid(o_weight_read_valid),
        .i_read_ready(i_weight_read_ready),
        .o_read_word_addr0(o_weight_read_addr0),
        .o_read_word_addr1(o_weight_read_addr1),
        .i_read_result_valid(i_weight_result_valid),
        .o_read_result_ready(o_weight_result_ready),
        .i_read_data0(i_weight_data0),
        .i_read_data1(i_weight_data1),
        .o_weight_valid(gather_weight_valid),
        .i_weight_ready(gather_weight_ready),
        .o_weight_flat(gather_weight_flat),
        .o_engine_mask(gather_engine_mask),
        .o_lane_mask(gather_lane_mask),
        .o_busy(gather_busy),
        .o_sticky_error(gather_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_CAPTURE;
            beat_feature_bank_flat_q <= '0;
            beat_feature_addr_flat_q <= '0;
            beat_lane_mask_q <= '0;
            beat_weight_addr_flat_q <= '0;
            beat_output_addr_flat_q <= '0;
            beat_spatial_addr_q <= '0;
            beat_engine_mask_q <= '0;
            beat_output_channel_base_q <= '0;
            beat_first_q <= 1'b0;
            beat_last_q <= 1'b0;
            beat_has_data_q <= 1'b0;
            feature_request_pending_q <= 1'b0;
            weight_request_pending_q <= 1'b0;
            local_protocol_error_q <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy)
                local_protocol_error_q <= 1'b0;

            if (i_start && !o_start_ready)
                local_protocol_error_q <= 1'b1;
            if (state_q != STATE_CAPTURE && !i_owner_enable)
                local_protocol_error_q <= 1'b1;
            if (i_side_result_valid && state_q != STATE_WAIT_RESULT)
                local_protocol_error_q <= 1'b1;
            if (state_q == STATE_WAIT_RESULT && i_side_result_valid &&
                !result_payload_legal)
                local_protocol_error_q <= 1'b1;

            if (generator_valid && generator_ready) begin
                beat_feature_bank_flat_q <= generator_feature_bank_flat;
                for (integer lane = 0; lane < LANES; lane++) begin
                    beat_feature_addr_flat_q[
                        lane*FEATURE_ADDR_WIDTH +: FEATURE_ADDR_WIDTH
                    ] <= generator_feature_addr_wide_flat[
                        lane*GENERATOR_ADDR_WIDTH +: FEATURE_ADDR_WIDTH
                    ];
                end
                beat_lane_mask_q <= generator_lane_mask;
                for (integer item = 0; item < ENGINES*LANES; item++) begin
                    beat_weight_addr_flat_q[
                        item*Q16_ADDR_WIDTH +: Q16_ADDR_WIDTH
                    ] <= generator_weight_addr_wide_flat[
                        item*GENERATOR_ADDR_WIDTH +: Q16_ADDR_WIDTH
                    ];
                end
                for (integer engine = 0; engine < ENGINES; engine++) begin
                    beat_output_addr_flat_q[
                        engine*OUTPUT_ADDR_WIDTH +: OUTPUT_ADDR_WIDTH
                    ] <= generator_output_addr_wide_flat[
                        engine*GENERATOR_ADDR_WIDTH +: OUTPUT_ADDR_WIDTH
                    ];
                end
                beat_spatial_addr_q <=
                    generator_spatial_addr_wide[OUTPUT_ADDR_WIDTH-1:0];
                beat_engine_mask_q <= generator_engine_mask;
                beat_output_channel_base_q <= i_output_channel_base;
                beat_first_q <= generator_first;
                beat_last_q <= generator_last;
                beat_has_data_q <= |generator_lane_mask;
                feature_request_pending_q <= |generator_lane_mask;
                weight_request_pending_q <= |generator_lane_mask;
                state_q <= STATE_REQUEST;
                if (generator_bank_conflict)
                    local_protocol_error_q <= 1'b1;
            end

            if (state_q == STATE_REQUEST) begin
                if (feature_request_accept)
                    feature_request_pending_q <= 1'b0;
                if (weight_request_accept)
                    weight_request_pending_q <= 1'b0;
                if (requests_complete)
                    state_q <= STATE_WAIT_DATA;
            end

            if (data_fire) begin
                if (beat_has_data_q &&
                    (i_feature_lane_mask != beat_lane_mask_q ||
                     gather_lane_mask != beat_lane_mask_q ||
                     gather_engine_mask != beat_engine_mask_q))
                    local_protocol_error_q <= 1'b1;
                state_q <= beat_last_q ? STATE_WAIT_RESULT : STATE_CAPTURE;
            end

            if (state_q == STATE_WAIT_RESULT && result_fire)
                state_q <= STATE_CAPTURE;
        end
    end
endmodule

`default_nettype wire
