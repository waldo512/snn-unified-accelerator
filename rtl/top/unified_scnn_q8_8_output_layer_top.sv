`default_nettype none

// V2.2 SCNN OUTPUT owner for the fixed full-DIV1 final feature layout.
// Four 2x2 spatial positions occupy distinct feature banks, so one channel is
// consumed per beat without feature replication. LI state persists across
// timesteps and is reset only at a recording boundary.
module unified_scnn_q8_8_output_layer_top #(
    parameter int LANES = 8,
    parameter int INPUT_FEATURES = 2048,
    parameter int OUTPUT_MAX = 32,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH),
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_layer_valid,
    output logic                                      o_layer_ready,
    input  logic [383:0]                              i_layer_words,
    input  logic                                      i_input_slot,
    input  logic                                      i_sample_reset,

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

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [$clog2(OUTPUT_MAX)-1:0]             o_result_class,
    output logic signed [15:0]                        o_result_score_q8_8,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_config_error,
    output logic                                      o_protocol_error
);
    localparam int CLASS_WIDTH = $clog2(OUTPUT_MAX);
    localparam int CHANNELS = INPUT_FEATURES / 4;

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_RESET,
        STATE_FEATURE_ISSUE,
        STATE_FEATURE_WAIT,
        STATE_WEIGHT_ISSUE,
        STATE_WEIGHT_WAIT,
        STATE_LI_SEND,
        STATE_LI_SCORE,
        STATE_RESULT
    } state_t;

    state_t state_q;
    logic input_slot_q;
    logic [Q16_ADDR_WIDTH-1:0] weight_base_q;
    logic [CLASS_WIDTH-1:0] class_q;
    logic [15:0] output_count_q;
    logic [$clog2(CHANNELS)-1:0] channel_q;
    logic [LANES*16-1:0] feature_q;
    logic [LANES*16-1:0] weight_q;
    logic best_valid_q;
    logic [CLASS_WIDTH-1:0] best_class_q;
    logic signed [15:0] best_score_q;
    logic [CLASS_WIDTH-1:0] result_class_q;
    logic signed [15:0] result_score_q;

    logic gather_request_ready;
    logic [LANES*Q16_ADDR_WIDTH-1:0] gather_addr_flat;
    logic gather_weight_valid;
    logic [LANES*16-1:0] gather_weight_flat;
    logic gather_engine_mask;
    logic [LANES-1:0] gather_lane_mask;
    logic gather_busy;
    logic gather_error;
    logic li_ready;
    logic li_valid;
    logic [CLASS_WIDTH-1:0] li_class;
    logic signed [15:0] li_score;
    logic li_idle;
    logic li_protocol_error;
    logic li_class_error;
    logic li_reset_error;
    logic descriptor_valid;
    logic final_channel;
    logic final_class;
    logic signed [15:0] selected_best_score;
    logic [CLASS_WIDTH-1:0] selected_best_class;

    always_comb begin
        descriptor_valid = i_layer_words[0*32 +: 32] == 32'h4e4d5232 &&
            i_layer_words[1*32 +: 32] == 32'h000c0202 &&
            i_layer_words[2*32 +: 4] == 4'd4 &&
            i_layer_words[3*32 +: 13] == 13'(INPUT_FEATURES) &&
            i_layer_words[3*32 + 23 +: 6] != 0 &&
            i_layer_words[3*32 + 23 +: 6] <= 6'(OUTPUT_MAX) &&
            i_layer_words[7*32 +: 32] < (32'd1 << Q16_ADDR_WIDTH) &&
            i_layer_words[11*32 +: 32] == 32'h55534e31;
        o_layer_ready = state_q == STATE_IDLE && li_idle &&
                        gather_request_ready && !o_config_error &&
                        !o_protocol_error;
        o_feature_read_valid = state_q == STATE_FEATURE_ISSUE;
        o_feature_read_slot = input_slot_q;
        o_feature_read_bank_flat = '0;
        o_feature_read_addr_flat = '0;
        o_feature_read_lane_mask = 8'h0f;
        o_feature_read_bank_flat[0*4 +: 4] = 4'd0;
        o_feature_read_bank_flat[1*4 +: 4] = 4'd1;
        o_feature_read_bank_flat[2*4 +: 4] = 4'd3;
        o_feature_read_bank_flat[3*4 +: 4] = 4'd4;
        for (integer lane = 0; lane < 4; lane++)
            o_feature_read_addr_flat[
                lane*FEATURE_ADDR_WIDTH +: FEATURE_ADDR_WIDTH
            ] = FEATURE_ADDR_WIDTH'(channel_q);
        o_feature_ready = state_q == STATE_FEATURE_WAIT;

        gather_addr_flat = '0;
        for (integer lane = 0; lane < 4; lane++)
            gather_addr_flat[lane*Q16_ADDR_WIDTH +: Q16_ADDR_WIDTH] =
                weight_base_q + Q16_ADDR_WIDTH'(
                    integer'(class_q)*INPUT_FEATURES +
                    integer'(channel_q)*4 + lane);

        final_channel = integer'(channel_q) + 1 >= CHANNELS;
        final_class = integer'(class_q) + 1 >= integer'(output_count_q);
        selected_best_score = best_score_q;
        selected_best_class = best_class_q;
        if (!best_valid_q || $signed(li_score) > $signed(best_score_q)) begin
            selected_best_score = li_score;
            selected_best_class = class_q;
        end
        o_result_valid = state_q == STATE_RESULT;
        o_result_class = result_class_q;
        o_result_score_q8_8 = result_score_q;
        o_busy = state_q != STATE_IDLE;
    end

    unified_scnn_q16_wave_weight_gather #(
        .ENGINES(1), .LANES(LANES),
        .Q64_ADDRESS_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDRESS_WIDTH(Q16_ADDR_WIDTH)
    ) u_weight_gather (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_request_valid(state_q == STATE_WEIGHT_ISSUE),
        .o_request_ready(gather_request_ready), .i_engine_mask(1'b1),
        .i_lane_mask(8'h0f), .i_weight_q16_addr_flat(gather_addr_flat),
        .o_read_valid(o_weight_read_valid),
        .i_read_ready(i_weight_read_ready),
        .o_read_word_addr0(o_weight_read_addr0),
        .o_read_word_addr1(o_weight_read_addr1),
        .i_read_result_valid(i_weight_result_valid),
        .o_read_result_ready(o_weight_result_ready),
        .i_read_data0(i_weight_data0), .i_read_data1(i_weight_data1),
        .o_weight_valid(gather_weight_valid),
        .i_weight_ready(state_q == STATE_WEIGHT_WAIT),
        .o_weight_flat(gather_weight_flat),
        .o_engine_mask(gather_engine_mask),
        .o_lane_mask(gather_lane_mask), .o_busy(gather_busy),
        .o_sticky_error(gather_error)
    );

    unified_scnn_q8_8_li_class_engine #(
        .INPUT_SIZE(INPUT_FEATURES), .OUTPUT_CLASSES(OUTPUT_MAX),
        .CLASS_WIDTH(CLASS_WIDTH)
    ) u_li (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_sample_reset(state_q == STATE_RESET),
        .i_valid(state_q == STATE_LI_SEND), .o_ready(li_ready),
        .i_class(class_q), .i_feature_flat(feature_q),
        .i_weight_flat(weight_q), .i_lane_mask(8'h0f),
        .i_first(channel_q == 0), .i_last(final_channel),
        .o_valid(li_valid), .i_ready(state_q == STATE_LI_SCORE),
        .o_class(li_class), .o_score_q8_8(li_score), .o_idle(li_idle),
        .o_protocol_error(li_protocol_error), .o_class_error(li_class_error),
        .o_reset_while_busy_error(li_reset_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            input_slot_q <= 1'b0;
            weight_base_q <= '0;
            class_q <= '0;
            output_count_q <= '0;
            channel_q <= '0;
            feature_q <= '0;
            weight_q <= '0;
            best_valid_q <= 1'b0;
            best_class_q <= '0;
            best_score_q <= '0;
            result_class_q <= '0;
            result_score_q <= '0;
            o_done <= 1'b0;
            o_config_error <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE) begin
                o_config_error <= 1'b0;
                o_protocol_error <= 1'b0;
            end
            if (gather_error || li_protocol_error || li_class_error ||
                li_reset_error) begin
                o_protocol_error <= 1'b1;
                state_q <= STATE_IDLE;
            end

            case (state_q)
                STATE_IDLE: begin
                    if (i_layer_valid && o_layer_ready) begin
                        if (!descriptor_valid) begin
                            o_config_error <= 1'b1;
                        end else begin
                            input_slot_q <= i_input_slot;
                            weight_base_q <=
                                i_layer_words[7*32 +: Q16_ADDR_WIDTH];
                            output_count_q <=
                                {10'd0, i_layer_words[3*32 + 23 +: 6]};
                            class_q <= '0;
                            channel_q <= '0;
                            best_valid_q <= 1'b0;
                            state_q <= i_sample_reset ? STATE_RESET :
                                                       STATE_FEATURE_ISSUE;
                        end
                    end
                end
                STATE_RESET: state_q <= STATE_FEATURE_ISSUE;
                STATE_FEATURE_ISSUE: begin
                    if (i_feature_read_ready)
                        state_q <= STATE_FEATURE_WAIT;
                end
                STATE_FEATURE_WAIT: begin
                    if (i_feature_valid) begin
                        if (i_feature_lane_mask != 8'h0f) begin
                            o_protocol_error <= 1'b1;
                            state_q <= STATE_IDLE;
                        end else begin
                            feature_q <= i_feature_flat;
                            state_q <= STATE_WEIGHT_ISSUE;
                        end
                    end
                end
                STATE_WEIGHT_ISSUE: begin
                    if (gather_request_ready)
                        state_q <= STATE_WEIGHT_WAIT;
                end
                STATE_WEIGHT_WAIT: begin
                    if (gather_weight_valid) begin
                        weight_q <= gather_weight_flat;
                        state_q <= STATE_LI_SEND;
                    end
                end
                STATE_LI_SEND: begin
                    if (li_ready) begin
                        if (final_channel)
                            state_q <= STATE_LI_SCORE;
                        else begin
                            channel_q <= channel_q + 1'b1;
                            state_q <= STATE_FEATURE_ISSUE;
                        end
                    end
                end
                STATE_LI_SCORE: begin
                    if (li_valid) begin
                        best_valid_q <= 1'b1;
                        best_score_q <= selected_best_score;
                        best_class_q <= selected_best_class;
                        if (li_class != class_q) begin
                            o_protocol_error <= 1'b1;
                            state_q <= STATE_IDLE;
                        end else if (final_class) begin
                            result_class_q <= selected_best_class;
                            result_score_q <= selected_best_score;
                            state_q <= STATE_RESULT;
                        end else begin
                            class_q <= class_q + 1'b1;
                            channel_q <= '0;
                            state_q <= STATE_FEATURE_ISSUE;
                        end
                    end
                end
                STATE_RESULT: begin
                    if (i_result_ready) begin
                        o_done <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
