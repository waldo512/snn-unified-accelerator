`timescale 1ns/1ps
`default_nettype none

// Executes one dense INT8 SRNN row on the fixed H40 tile array. The external
// memory ABI remains contiguous ENGINESxK8 waves; the internal transposer
// converts the complete row into five tile-major words before computation.
module unified_srnn_tile_major_row_executor_v1 #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer TILES = 12,
    parameter integer TILE_NEURONS = 40,
    parameter integer ENGINES = 12,
    parameter integer LANES = 8,
    parameter integer VERSION_WIDTH = 16,
    parameter integer REQUEST_ID_WIDTH = 16
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,

    input  logic                                      i_row_req_valid,
    output logic                                      o_row_req_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_row_req_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_row_req_id,
    input  logic                                      i_row_req_recurrent,
    input  logic [31:0]                               i_row_req_base,
    input  logic [9:0]                                i_row_req_destination_count,
    input  logic signed [7:0]                        i_row_req_activation,

    output logic                                      o_weight_req_valid,
    input  logic                                      i_weight_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_weight_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_weight_req_id,
    output logic                                      o_weight_req_recurrent,
    output logic [31:0]                               o_weight_req_row_base,
    output logic [9:0]                                o_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_weight_req_lane_mask,

    input  logic                                      i_weight_rsp_valid,
    output logic                                      o_weight_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_weight_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_weight_rsp_id,
    input  logic [9:0]                                i_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_weight_rsp_flat,

    output logic                                      o_array_valid,
    input  logic                                      i_array_ready,
    output logic [TILES-1:0]                          o_array_engine_mask,
    output logic [LANES-1:0]                          o_array_lane_mask,
    output logic [LANES*8-1:0]                        o_array_int8_lhs_flat,
    output logic [TILES*LANES*8-1:0]                  o_array_int8_rhs_flat,
    output logic                                      o_array_first,
    output logic                                      o_array_last,
    output logic [2:0]                                o_array_local_word,

    input  logic                                      i_array_result_valid,
    output logic                                      o_array_result_ready,
    input  logic [1:0]                                i_array_result_owner,
    input  logic                                      i_array_result_mode_q8_8,
    input  logic [TILES-1:0]                          i_array_result_engine_mask,
    input  logic [LANES-1:0]                          i_array_result_lane_mask,

    output logic                                      o_row_done_valid,
    input  logic                                      i_row_done_ready,
    output logic [VERSION_WIDTH-1:0]                  o_row_done_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_row_done_id,

    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam integer DESTINATIONS_PER_WAVE = ENGINES*LANES;
    localparam logic [1:0] OWNER_RECURRENT_INT8 = 2'd2;

    localparam logic [3:0] STATE_IDLE = 4'd0;
    localparam logic [3:0] STATE_TRANSPOSE_START = 4'd1;
    localparam logic [3:0] STATE_WEIGHT_REQUEST = 4'd2;
    localparam logic [3:0] STATE_WEIGHT_RESPONSE = 4'd3;
    localparam logic [3:0] STATE_ARRAY_REQUEST = 4'd4;
    localparam logic [3:0] STATE_ARRAY_RESULT = 4'd5;
    localparam logic [3:0] STATE_ROW_DONE = 4'd6;
    localparam logic [3:0] STATE_ERROR = 4'd7;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_CONFIG = 8'h01;
    localparam logic [7:0] ERROR_WEIGHT_PROTOCOL = 8'h02;
    localparam logic [7:0] ERROR_ARRAY_PROTOCOL = 8'h03;
    localparam logic [7:0] ERROR_TRANSPOSE = 8'h04;

    logic [3:0] state_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [REQUEST_ID_WIDTH-1:0] request_id_q;
    logic recurrent_q;
    logic [31:0] row_base_q;
    logic [9:0] destination_count_q;
    logic signed [7:0] activation_q;
    logic [9:0] wave_base_q;
    logic word_last_q;
    logic [TILES-1:0] word_tile_mask_q;
    logic [LANES-1:0] word_lane_mask_q;

    logic [ENGINES-1:0] expected_engine_mask;
    logic final_wave;
    logic row_config_legal;
    logic weight_response_legal;
    logic array_result_legal;

    logic trans_row_start_ready;
    logic trans_wave_ready;
    logic trans_word_valid;
    logic trans_word_ready;
    logic [2:0] trans_local_word;
    logic [TILES-1:0] trans_tile_mask;
    logic [LANES-1:0] trans_lane_mask;
    logic [TILES*LANES*8-1:0] trans_weight_flat;
    logic trans_word_last;
    logic trans_busy;
    logic trans_idle;
    logic trans_sticky_error;

    function automatic logic [ENGINES-1:0] engine_mask_for(
        input logic [9:0] destination_base,
        input logic [9:0] destination_count
    );
        begin
            engine_mask_for = '0;
            for (integer engine = 0; engine < ENGINES; engine++)
                if (integer'(destination_base) + engine*LANES <
                    integer'(destination_count))
                    engine_mask_for[engine] = 1'b1;
        end
    endfunction

    always_comb begin
        expected_engine_mask = engine_mask_for(
            wave_base_q, destination_count_q);
        final_wave = integer'(wave_base_q) + DESTINATIONS_PER_WAVE >=
            integer'(destination_count_q);
        row_config_legal = i_row_req_destination_count != 0 &&
            i_row_req_destination_count <= 10'(HIDDEN_MAX);
        weight_response_legal =
            i_weight_rsp_version == version_q &&
            i_weight_rsp_id == request_id_q &&
            i_weight_rsp_destination_base == wave_base_q &&
            i_weight_rsp_engine_mask == expected_engine_mask;
        array_result_legal =
            i_array_result_owner == OWNER_RECURRENT_INT8 &&
            !i_array_result_mode_q8_8 &&
            i_array_result_engine_mask == word_tile_mask_q &&
            i_array_result_lane_mask == word_lane_mask_q;

        o_row_req_ready = state_q == STATE_IDLE && !o_sticky_error;

        o_weight_req_valid = state_q == STATE_WEIGHT_REQUEST &&
            !o_sticky_error;
        o_weight_req_version = version_q;
        o_weight_req_id = request_id_q;
        o_weight_req_recurrent = recurrent_q;
        o_weight_req_row_base = row_base_q;
        o_weight_req_destination_base = wave_base_q;
        o_weight_req_engine_mask = expected_engine_mask;
        o_weight_req_lane_mask = {LANES{1'b1}};
        o_weight_rsp_ready = state_q == STATE_WEIGHT_RESPONSE &&
            trans_wave_ready && !o_sticky_error;

        o_array_valid = state_q == STATE_ARRAY_REQUEST &&
            trans_word_valid && !o_sticky_error;
        o_array_engine_mask = trans_tile_mask;
        o_array_lane_mask = trans_lane_mask;
        o_array_int8_lhs_flat = '0;
        for (integer lane = 0; lane < LANES; lane++)
            o_array_int8_lhs_flat[lane*8 +: 8] = activation_q;
        o_array_int8_rhs_flat = trans_weight_flat;
        o_array_first = 1'b1;
        o_array_last = 1'b1;
        o_array_local_word = trans_local_word;
        trans_word_ready = o_array_valid && i_array_ready;

        o_array_result_ready = state_q == STATE_ARRAY_RESULT &&
            !o_sticky_error;

        o_row_done_valid = state_q == STATE_ROW_DONE && !o_sticky_error;
        o_row_done_version = version_q;
        o_row_done_id = request_id_q;
        o_busy = (state_q != STATE_IDLE && state_q != STATE_ERROR) ||
            trans_busy;
        o_idle = state_q == STATE_IDLE && trans_idle;
    end

    unified_srnn_tile_major_weight_transposer_v1 #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .TILES(TILES),
        .TILE_NEURONS(TILE_NEURONS),
        .ENGINES(ENGINES),
        .LANES(LANES)
    ) u_transposer (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_row_start_valid(state_q == STATE_TRANSPOSE_START),
        .o_row_start_ready(trans_row_start_ready),
        .i_destination_count(destination_count_q),
        .i_wave_valid(state_q == STATE_WEIGHT_RESPONSE &&
            i_weight_rsp_valid && weight_response_legal),
        .o_wave_ready(trans_wave_ready),
        .i_wave_destination_base(i_weight_rsp_destination_base),
        .i_wave_engine_mask(i_weight_rsp_engine_mask),
        .i_wave_lane_mask({LANES{1'b1}}),
        .i_wave_weight_flat(i_weight_rsp_flat),
        .o_word_valid(trans_word_valid),
        .i_word_ready(trans_word_ready),
        .o_local_word(trans_local_word),
        .o_tile_mask(trans_tile_mask),
        .o_lane_mask(trans_lane_mask),
        .o_tile_major_weight_flat(trans_weight_flat),
        .o_word_last(trans_word_last),
        .o_busy(trans_busy),
        .o_idle(trans_idle),
        .o_sticky_error(trans_sticky_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            version_q <= '0;
            request_id_q <= '0;
            recurrent_q <= 1'b0;
            row_base_q <= '0;
            destination_count_q <= '0;
            activation_q <= '0;
            wave_base_q <= '0;
            word_last_q <= 1'b0;
            word_tile_mask_q <= '0;
            word_lane_mask_q <= '0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            version_q <= '0;
            request_id_q <= '0;
            recurrent_q <= 1'b0;
            row_base_q <= '0;
            destination_count_q <= '0;
            activation_q <= '0;
            wave_base_q <= '0;
            word_last_q <= 1'b0;
            word_tile_mask_q <= '0;
            word_lane_mask_q <= '0;
        end else if (trans_sticky_error && state_q != STATE_ERROR) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_TRANSPOSE;
        end else if (i_weight_rsp_valid &&
                     state_q != STATE_WEIGHT_RESPONSE) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_WEIGHT_PROTOCOL;
        end else if (i_array_result_valid &&
                     state_q != STATE_ARRAY_RESULT) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_ARRAY_PROTOCOL;
        end else begin
            case (state_q)
                STATE_IDLE: begin
                    if (i_clear_error) begin
                        o_sticky_error <= 1'b0;
                        o_error_code <= ERROR_NONE;
                    end
                    if (i_row_req_valid) begin
                        if (!row_config_legal) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_CONFIG;
                        end else begin
                            version_q <= i_row_req_version;
                            request_id_q <= i_row_req_id;
                            recurrent_q <= i_row_req_recurrent;
                            row_base_q <= i_row_req_base;
                            destination_count_q <=
                                i_row_req_destination_count;
                            activation_q <= i_row_req_activation;
                            wave_base_q <= '0;
                            state_q <= STATE_TRANSPOSE_START;
                        end
                    end
                end
                STATE_TRANSPOSE_START: begin
                    if (trans_row_start_ready)
                        state_q <= STATE_WEIGHT_REQUEST;
                end
                STATE_WEIGHT_REQUEST: begin
                    if (o_weight_req_valid && i_weight_req_ready)
                        state_q <= STATE_WEIGHT_RESPONSE;
                end
                STATE_WEIGHT_RESPONSE: begin
                    if (i_weight_rsp_valid && o_weight_rsp_ready) begin
                        if (!weight_response_legal) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_WEIGHT_PROTOCOL;
                        end else if (final_wave) begin
                            state_q <= STATE_ARRAY_REQUEST;
                        end else begin
                            wave_base_q <= wave_base_q +
                                10'(DESTINATIONS_PER_WAVE);
                            state_q <= STATE_WEIGHT_REQUEST;
                        end
                    end
                end
                STATE_ARRAY_REQUEST: begin
                    if (o_array_valid && i_array_ready) begin
                        word_last_q <= trans_word_last;
                        word_tile_mask_q <= trans_tile_mask;
                        word_lane_mask_q <= trans_lane_mask;
                        state_q <= STATE_ARRAY_RESULT;
                    end
                end
                STATE_ARRAY_RESULT: begin
                    if (i_array_result_valid && o_array_result_ready) begin
                        if (!array_result_legal) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_ARRAY_PROTOCOL;
                        end else begin
                            state_q <= word_last_q ?
                                STATE_ROW_DONE : STATE_ARRAY_REQUEST;
                        end
                    end
                end
                STATE_ROW_DONE: begin
                    if (o_row_done_valid && i_row_done_ready)
                        state_q <= STATE_IDLE;
                end
                STATE_ERROR: begin
                    if (i_clear_error && !i_weight_rsp_valid &&
                        !i_array_result_valid) begin
                        state_q <= STATE_IDLE;
                        o_sticky_error <= 1'b0;
                        o_error_code <= ERROR_NONE;
                    end
                end
                default: begin
                    state_q <= STATE_ERROR;
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_CONFIG;
                end
            endcase
        end
    end

    initial begin
        if (TILES*TILE_NEURONS != HIDDEN_MAX)
            $error("row executor requires exact HIDDEN_MAX tile coverage");
        if (TILE_NEURONS != 40 || LANES != 8)
            $error("row executor requires H40/K8 geometry");
    end
endmodule

`default_nettype wire
