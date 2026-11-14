`timescale 1ns/1ps
`default_nettype none

// Recurrent-FC state and control beside a separately owned shared K8 array.
// The array executes one INT8 row transaction and returns ordered K8 ACC32
// contribution chunks. Recurrent state is physically partitioned into H40
// shards so it can migrate into the shared inference-tile array without
// changing this controller's eight-neuron-per-cycle protocol.
module unified_srnn_recurrent_sidecar_soft_abort_v1 #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer INPUT_MAX = 2312,
    parameter integer LANES = 8,
    parameter integer VERSION_WIDTH = 16,
    parameter integer REQUEST_ID_WIDTH = 16
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_owner_enable,
    input  logic                                      i_clear_error,
    input  logic                                      i_state_reset,

    input  logic                                      i_start_valid,
    output logic                                      o_start_ready,
    input  logic [12:0]                               i_input_dim,
    input  logic [9:0]                                i_active_neurons,
    input  logic [31:0]                               i_w0_base,
    input  logic [31:0]                               i_recurrent_base,
    input  logic                                      i_recurrent_enable,
    input  logic signed [15:0]                       i_leak_q15,
    input  logic signed [31:0]                       i_threshold,
    input  logic [4:0]                                i_current_left_shift,
    input  logic                                      i_reset_to_zero,
    input  logic [VERSION_WIDTH-1:0]                  i_version,

    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [12:0]                               i_event_addr,
    input  logic signed [7:0]                        i_event_data,
    input  logic                                      i_timestep_last,
    input  logic                                      i_sample_last,

    output logic                                      o_k8_req_valid,
    input  logic                                      i_k8_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_k8_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_k8_req_id,
    output logic                                      o_k8_req_recurrent,
    output logic [31:0]                               o_k8_req_row_base,
    output logic [9:0]                                o_k8_req_destination_count,
    output logic signed [7:0]                        o_k8_req_activation,
    output logic [2:0]                                o_k8_req_numeric_profile,

    input  logic                                      i_k8_rsp_valid,
    output logic                                      o_k8_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_k8_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_k8_rsp_id,
    input  logic [9:0]                                i_k8_rsp_destination_base,
    input  logic [LANES-1:0]                          i_k8_rsp_lane_mask,
    input  logic [LANES*32-1:0]                       i_k8_rsp_contribution_flat,
    input  logic                                      i_k8_rsp_last,

    output logic                                      o_timestep_valid,
    input  logic                                      i_timestep_ready,
    output logic [VERSION_WIDTH-1:0]                  o_timestep_version,
    output logic                                      o_timestep_sample_last,
    output logic [HIDDEN_MAX-1:0]                     o_timestep_spikes,
    output logic [HIDDEN_MAX-1:0]                     o_previous_spikes,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam integer WORD_COUNT = (HIDDEN_MAX + LANES - 1) / LANES;
    localparam integer WORD_INDEX_WIDTH =
        (WORD_COUNT <= 1) ? 1 : $clog2(WORD_COUNT);
    localparam integer HIDDEN_ADDRESS_WIDTH =
        (HIDDEN_MAX <= 1) ? 1 : $clog2(HIDDEN_MAX);

    localparam logic [3:0] STATE_IDLE = 4'd0;
    localparam logic [3:0] STATE_CLEAR = 4'd1;
    localparam logic [3:0] STATE_INPUT = 4'd2;
    localparam logic [3:0] STATE_REQUEST = 4'd3;
    localparam logic [3:0] STATE_RESPONSE = 4'd4;
    localparam logic [3:0] STATE_SCAN = 4'd5;
    localparam logic [3:0] STATE_LIF = 4'd6;
    localparam logic [3:0] STATE_RESULT = 4'd7;
    localparam logic [3:0] STATE_ERROR_DRAIN = 4'd8;
    localparam logic [3:0] STATE_ERROR = 4'd9;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER = 8'h01;
    localparam logic [7:0] ERROR_CONFIG = 8'h02;
    localparam logic [7:0] ERROR_EVENT = 8'h03;
    localparam logic [7:0] ERROR_UNEXPECTED_RESPONSE = 8'h04;
    localparam logic [7:0] ERROR_MIXED_COMPLETION = 8'h05;
    localparam logic [7:0] ERROR_RESPONSE_SHAPE = 8'h06;
    localparam logic [7:0] ERROR_RESET_BUSY = 8'h07;

    localparam logic [2:0] NUMERIC_SRNN_INT8_ACC32 = 3'd2;

    logic [3:0] state_q;
    logic clear_for_start_q;
    logic [WORD_INDEX_WIDTH-1:0] clear_word_q;
    logic [WORD_INDEX_WIDTH-1:0] lif_word_q;
    logic [9:0] scan_source_q;
    logic [9:0] expected_response_base_q;
    logic [12:0] input_dim_q;
    logic [9:0] active_neurons_q;
    logic [31:0] w0_base_q;
    logic [31:0] recurrent_base_q;
    logic recurrent_enable_q;
    logic signed [15:0] leak_q15_q;
    logic signed [31:0] threshold_q;
    logic [4:0] current_left_shift_q;
    logic reset_to_zero_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [REQUEST_ID_WIDTH-1:0] next_request_id_q;
    logic [REQUEST_ID_WIDTH-1:0] request_id_q;
    logic request_recurrent_q;
    logic [31:0] request_row_base_q;
    logic signed [7:0] request_activation_q;
    logic event_timestep_last_q;
    logic timestep_sample_last_q;
    logic [LANES-1:0] expected_lane_mask;
    logic expected_response_last;
    logic response_identity_valid;
    logic response_shape_valid;
    logic start_config_valid;
    logic state_clear_spikes;
    logic state_clear_word_valid;
    logic [9:0] state_clear_word_base;
    logic state_accumulate_valid;
    logic state_lif_begin;
    logic state_lif_word_valid;
    logic [9:0] state_lif_word_base;
    logic state_commit_timestep;
    logic [HIDDEN_MAX-1:0] state_previous_spikes;
    logic [HIDDEN_MAX-1:0] state_current_spikes;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [HIDDEN_MAX*32-1:0] state_accumulator_observe;
    logic [HIDDEN_MAX*32-1:0] state_membrane_observe;
    /* verilator lint_on UNUSEDSIGNAL */

    function automatic logic [LANES-1:0] lane_mask_for(
        input logic [9:0] destination_base,
        input logic [9:0] destination_count
    );
        integer lane_index;
        begin
            lane_mask_for = '0;
            for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1)
                if ((destination_base + 10'(lane_index)) < destination_count)
                    lane_mask_for[lane_index] = 1'b1;
        end
    endfunction

    unified_srnn_h40_state_array_v1 #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .TILE_NEURONS(40),
        .LANES(LANES)
    ) u_state_array (
        .clk(clk),
        .rst_n(rst_n),
        .i_abort(i_abort),
        .i_clear_spikes(state_clear_spikes),
        .i_clear_word_valid(state_clear_word_valid),
        .i_clear_word_base(state_clear_word_base),
        .i_accumulate_valid(state_accumulate_valid),
        .i_accumulate_base(expected_response_base_q),
        .i_accumulate_lane_mask(i_k8_rsp_lane_mask),
        .i_accumulate_contribution_flat(i_k8_rsp_contribution_flat),
        .i_active_neurons(active_neurons_q),
        .i_lif_begin(state_lif_begin),
        .i_lif_word_valid(state_lif_word_valid),
        .i_lif_word_base(state_lif_word_base),
        .i_leak_q15(leak_q15_q),
        .i_threshold(threshold_q),
        .i_current_left_shift(current_left_shift_q),
        .i_reset_to_zero(reset_to_zero_q),
        .i_commit_timestep(state_commit_timestep),
        .o_previous_spikes(state_previous_spikes),
        .o_current_spikes(state_current_spikes),
        .o_accumulator_flat(state_accumulator_observe),
        .o_membrane_flat(state_membrane_observe)
    );

    always_comb begin
        start_config_valid =
            (i_input_dim != 0) &&
            (i_input_dim <= 13'(INPUT_MAX)) &&
            (i_active_neurons != 0) &&
            (i_active_neurons <= 10'(HIDDEN_MAX));
        o_start_ready = !i_abort && (state_q == STATE_IDLE) && i_owner_enable &&
                        !o_sticky_error;
        o_event_ready = !i_abort && (state_q == STATE_INPUT) && i_owner_enable &&
                        !o_sticky_error;
        o_k8_req_valid = !i_abort && (state_q == STATE_REQUEST) &&
                         i_owner_enable &&
                         !o_sticky_error;
        o_k8_req_version = version_q;
        o_k8_req_id = request_id_q;
        o_k8_req_recurrent = request_recurrent_q;
        o_k8_req_row_base = request_row_base_q;
        o_k8_req_destination_count = active_neurons_q;
        o_k8_req_activation = request_activation_q;
        o_k8_req_numeric_profile = NUMERIC_SRNN_INT8_ACC32;

        o_k8_rsp_ready = !i_abort && (((state_q == STATE_RESPONSE) &&
                          i_owner_enable &&
                          !o_sticky_error) || (state_q == STATE_ERROR_DRAIN));
        expected_lane_mask = lane_mask_for(expected_response_base_q,
                                           active_neurons_q);
        expected_response_last =
            (expected_response_base_q + 10'(LANES)) >= active_neurons_q;
        response_identity_valid =
            (i_k8_rsp_version == version_q) &&
            (i_k8_rsp_id == request_id_q);
        response_shape_valid =
            (i_k8_rsp_destination_base == expected_response_base_q) &&
            (i_k8_rsp_lane_mask == expected_lane_mask) &&
            (i_k8_rsp_last == expected_response_last);

        state_clear_spikes =
            (i_start_valid && o_start_ready && start_config_valid) ||
            (i_state_reset &&
             ((state_q == STATE_IDLE) || (state_q == STATE_ERROR)));
        state_clear_word_valid = state_q == STATE_CLEAR;
        state_clear_word_base = 10'(clear_word_q) * 10'(LANES);
        state_accumulate_valid =
            i_k8_rsp_valid && o_k8_rsp_ready &&
            response_identity_valid && response_shape_valid;
        state_lif_begin =
            (state_q == STATE_SCAN) &&
            (!recurrent_enable_q || (scan_source_q >= active_neurons_q));
        state_lif_word_valid = state_q == STATE_LIF;
        state_lif_word_base = 10'(lif_word_q) * 10'(LANES);

        o_timestep_valid = !i_abort && state_q == STATE_RESULT;
        state_commit_timestep = o_timestep_valid && i_timestep_ready;
        o_timestep_version = version_q;
        o_timestep_sample_last = timestep_sample_last_q;
        o_timestep_spikes = state_current_spikes;
        o_previous_spikes = state_previous_spikes;
        o_busy = (state_q != STATE_IDLE) && (state_q != STATE_ERROR);
        o_idle = state_q == STATE_IDLE;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            clear_for_start_q <= 1'b0;
            clear_word_q <= '0;
            lif_word_q <= '0;
            scan_source_q <= 10'd0;
            expected_response_base_q <= 10'd0;
            input_dim_q <= 13'd0;
            active_neurons_q <= 10'd0;
            w0_base_q <= 32'd0;
            recurrent_base_q <= 32'd0;
            recurrent_enable_q <= 1'b0;
            leak_q15_q <= 16'sd0;
            threshold_q <= 32'sd0;
            current_left_shift_q <= 5'd0;
            reset_to_zero_q <= 1'b0;
            version_q <= '0;
            next_request_id_q <= '0;
            request_id_q <= '0;
            request_recurrent_q <= 1'b0;
            request_row_base_q <= 32'd0;
            request_activation_q <= 8'sd0;
            event_timestep_last_q <= 1'b0;
            timestep_sample_last_q <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            clear_for_start_q <= 1'b0;
            clear_word_q <= '0;
            lif_word_q <= '0;
            scan_source_q <= 10'd0;
            expected_response_base_q <= 10'd0;
            input_dim_q <= 13'd0;
            active_neurons_q <= 10'd0;
            w0_base_q <= 32'd0;
            recurrent_base_q <= 32'd0;
            recurrent_enable_q <= 1'b0;
            leak_q15_q <= 16'sd0;
            threshold_q <= 32'sd0;
            current_left_shift_q <= 5'd0;
            reset_to_zero_q <= 1'b0;
            version_q <= '0;
            request_id_q <= '0;
            request_recurrent_q <= 1'b0;
            request_row_base_q <= 32'd0;
            request_activation_q <= 8'sd0;
            event_timestep_last_q <= 1'b0;
            timestep_sample_last_q <= 1'b0;
        end else if (i_state_reset) begin
            if ((state_q == STATE_IDLE) || (state_q == STATE_ERROR)) begin
                state_q <= STATE_CLEAR;
                clear_for_start_q <= 1'b0;
                clear_word_q <= '0;
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end else begin
                state_q <= STATE_ERROR;
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_RESET_BUSY;
            end
        end else if ((state_q != STATE_IDLE) && (state_q != STATE_ERROR) &&
                     (state_q != STATE_ERROR_DRAIN) && !i_owner_enable) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_OWNER;
        end else if (i_k8_rsp_valid && (state_q != STATE_RESPONSE) &&
                     (state_q != STATE_ERROR_DRAIN)) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_UNEXPECTED_RESPONSE;
        end else begin
            case (state_q)
                STATE_IDLE: begin
                    if (i_clear_error) begin
                        o_sticky_error <= 1'b0;
                        o_error_code <= ERROR_NONE;
                    end
                    if (i_start_valid) begin
                        if (!i_owner_enable) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_OWNER;
                        end else if ((i_input_dim == 0) ||
                                     (i_input_dim > 13'(INPUT_MAX)) ||
                                     (i_active_neurons == 0) ||
                                     (i_active_neurons > 10'(HIDDEN_MAX))) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_CONFIG;
                        end else begin
                            input_dim_q <= i_input_dim;
                            active_neurons_q <= i_active_neurons;
                            w0_base_q <= i_w0_base;
                            recurrent_base_q <= i_recurrent_base;
                            recurrent_enable_q <= i_recurrent_enable;
                            leak_q15_q <= i_leak_q15;
                            threshold_q <= i_threshold;
                            current_left_shift_q <= i_current_left_shift;
                            reset_to_zero_q <= i_reset_to_zero;
                            version_q <= i_version;
                            clear_word_q <= '0;
                            clear_for_start_q <= 1'b1;
                            timestep_sample_last_q <= 1'b0;
                            state_q <= STATE_CLEAR;
                        end
                    end
                end
                STATE_CLEAR: begin
                    if (clear_word_q == WORD_INDEX_WIDTH'(WORD_COUNT-1)) begin
                        clear_word_q <= '0;
                        if (clear_for_start_q)
                            state_q <= STATE_INPUT;
                        else
                            state_q <= STATE_IDLE;
                    end else begin
                        clear_word_q <= clear_word_q + 1'b1;
                    end
                end
                STATE_INPUT: begin
                    if (i_event_valid) begin
                        if ((i_event_addr >= input_dim_q) ||
                            (i_sample_last && !i_timestep_last)) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_EVENT;
                        end else begin
                            request_id_q <= next_request_id_q;
                            next_request_id_q <= next_request_id_q + 1'b1;
                            request_recurrent_q <= 1'b0;
                            request_row_base_q <= w0_base_q +
                                i_event_addr * active_neurons_q;
                            request_activation_q <= i_event_data;
                            event_timestep_last_q <= i_timestep_last;
                            if (i_timestep_last)
                                timestep_sample_last_q <= i_sample_last;
                            state_q <= STATE_REQUEST;
                        end
                    end
                end
                STATE_REQUEST: begin
                    if (o_k8_req_valid && i_k8_req_ready) begin
                        expected_response_base_q <= 10'd0;
                        state_q <= STATE_RESPONSE;
                    end
                end
                STATE_RESPONSE: begin
                    if (i_k8_rsp_valid && o_k8_rsp_ready) begin
                        if (!response_identity_valid) begin
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_MIXED_COMPLETION;
                            if (i_k8_rsp_last)
                                state_q <= STATE_ERROR;
                            else
                                state_q <= STATE_ERROR_DRAIN;
                        end else if (!response_shape_valid) begin
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_RESPONSE_SHAPE;
                            if (i_k8_rsp_last)
                                state_q <= STATE_ERROR;
                            else
                                state_q <= STATE_ERROR_DRAIN;
                        end else begin
                            if (expected_response_last) begin
                                if (request_recurrent_q) begin
                                    scan_source_q <= scan_source_q + 1'b1;
                                    state_q <= STATE_SCAN;
                                end else if (event_timestep_last_q) begin
                                    scan_source_q <= 10'd0;
                                    state_q <= STATE_SCAN;
                                end else begin
                                    state_q <= STATE_INPUT;
                                end
                            end else begin
                                expected_response_base_q <=
                                    expected_response_base_q + 10'(LANES);
                            end
                        end
                    end
                end
                STATE_SCAN: begin
                    if (!recurrent_enable_q ||
                        (scan_source_q >= active_neurons_q)) begin
                        lif_word_q <= '0;
                        state_q <= STATE_LIF;
                    end else if (state_previous_spikes[
                            HIDDEN_ADDRESS_WIDTH'(scan_source_q)]) begin
                        request_id_q <= next_request_id_q;
                        next_request_id_q <= next_request_id_q + 1'b1;
                        request_recurrent_q <= 1'b1;
                        request_row_base_q <= recurrent_base_q +
                            scan_source_q * active_neurons_q;
                        request_activation_q <= 8'sd1;
                        event_timestep_last_q <= 1'b0;
                        state_q <= STATE_REQUEST;
                    end else begin
                        scan_source_q <= scan_source_q + 1'b1;
                    end
                end
                STATE_LIF: begin
                    if (((10'(lif_word_q) + 10'd1) * 10'(LANES)) >=
                        active_neurons_q) begin
                        state_q <= STATE_RESULT;
                    end else begin
                        lif_word_q <= lif_word_q + 1'b1;
                    end
                end
                STATE_RESULT: begin
                    if (o_timestep_valid && i_timestep_ready) begin
                        if (timestep_sample_last_q)
                            state_q <= STATE_IDLE;
                        else
                            state_q <= STATE_INPUT;
                    end
                end
                STATE_ERROR_DRAIN: begin
                    if (i_k8_rsp_valid && o_k8_rsp_ready && i_k8_rsp_last)
                        state_q <= STATE_ERROR;
                end
                STATE_ERROR: begin
                    if (i_clear_error) begin
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
endmodule

`default_nettype wire
