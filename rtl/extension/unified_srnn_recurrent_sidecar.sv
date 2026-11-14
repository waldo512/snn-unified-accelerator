`timescale 1ns/1ps
`default_nettype none

// Recurrent-FC state and control beside a separately owned shared K8 array.
// The array executes one INT8 row transaction and returns ordered K8 ACC32
// contribution chunks. This leaf owns no multiplier or H40 compute instance.
module unified_srnn_recurrent_sidecar #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer INPUT_MAX = 2312,
    parameter integer LANES = 8,
    parameter integer VERSION_WIDTH = 16,
    parameter integer REQUEST_ID_WIDTH = 16
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
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
    logic [HIDDEN_MAX-1:0] previous_spikes_q;
    logic [HIDDEN_MAX-1:0] current_spikes_q;
    logic [LANES-1:0] expected_lane_mask;
    logic expected_response_last;
    logic response_identity_valid;
    logic response_shape_valid;
    logic [WORD_INDEX_WIDTH-1:0] response_word_index;
    integer clear_lane;
    integer response_lane;
    integer lif_lane;

    // Eight banks permit one complete K8 contribution or LIF chunk per cycle.
    logic signed [31:0] accumulator_bank [0:LANES-1][0:WORD_COUNT-1];
    logic signed [31:0] membrane_bank [0:LANES-1][0:WORD_COUNT-1];

    function automatic logic signed [31:0] saturating_add32(
        input logic signed [31:0] lhs,
        input logic signed [31:0] rhs
    );
        logic signed [32:0] sum_wide;
        begin
            sum_wide = {lhs[31], lhs} + {rhs[31], rhs};
            if (sum_wide > 33'sh07fffffff)
                saturating_add32 = 32'sh7fffffff;
            else if (sum_wide < -33'sh080000000)
                saturating_add32 = 32'sh80000000;
            else
                saturating_add32 = sum_wide[31:0];
        end
    endfunction

    function automatic logic signed [31:0] lif_pre_value(
        input logic signed [31:0] old_membrane,
        input logic signed [31:0] current,
        input logic signed [15:0] leak_q15,
        input logic [4:0] current_left_shift
    );
        logic signed [47:0] leak_product;
        logic signed [63:0] leak_scaled;
        logic signed [63:0] current_scaled;
        logic signed [63:0] sum_wide;
        begin
            leak_product = old_membrane * leak_q15;
            leak_scaled = {{16{leak_product[47]}}, leak_product};
            current_scaled = {{32{current[31]}}, current};
            current_scaled = current_scaled <<< current_left_shift;
            sum_wide = (leak_scaled >>> 15) + current_scaled;
            if (sum_wide > 64'sh000000007fffffff)
                lif_pre_value = 32'sh7fffffff;
            else if (sum_wide < -64'sh0000000080000000)
                lif_pre_value = 32'sh80000000;
            else
                lif_pre_value = sum_wide[31:0];
        end
    endfunction

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

    always_comb begin
        o_start_ready = (state_q == STATE_IDLE) && i_owner_enable &&
                        !o_sticky_error;
        o_event_ready = (state_q == STATE_INPUT) && i_owner_enable &&
                        !o_sticky_error;
        o_k8_req_valid = (state_q == STATE_REQUEST) && i_owner_enable &&
                         !o_sticky_error;
        o_k8_req_version = version_q;
        o_k8_req_id = request_id_q;
        o_k8_req_recurrent = request_recurrent_q;
        o_k8_req_row_base = request_row_base_q;
        o_k8_req_destination_count = active_neurons_q;
        o_k8_req_activation = request_activation_q;
        o_k8_req_numeric_profile = NUMERIC_SRNN_INT8_ACC32;

        o_k8_rsp_ready = ((state_q == STATE_RESPONSE) && i_owner_enable &&
                          !o_sticky_error) || (state_q == STATE_ERROR_DRAIN);
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
        response_word_index = WORD_INDEX_WIDTH'(
            expected_response_base_q / LANES);

        o_timestep_valid = state_q == STATE_RESULT;
        o_timestep_version = version_q;
        o_timestep_sample_last = timestep_sample_last_q;
        o_timestep_spikes = current_spikes_q;
        o_previous_spikes = previous_spikes_q;
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
            previous_spikes_q <= '0;
            current_spikes_q <= '0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_state_reset) begin
            if ((state_q == STATE_IDLE) || (state_q == STATE_ERROR)) begin
                state_q <= STATE_CLEAR;
                clear_for_start_q <= 1'b0;
                clear_word_q <= '0;
                previous_spikes_q <= '0;
                current_spikes_q <= '0;
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
                            previous_spikes_q <= '0;
                            current_spikes_q <= '0;
                            timestep_sample_last_q <= 1'b0;
                            state_q <= STATE_CLEAR;
                        end
                    end
                end
                STATE_CLEAR: begin
                    for (clear_lane = 0; clear_lane < LANES;
                         clear_lane = clear_lane + 1) begin
                        accumulator_bank[clear_lane][clear_word_q] <= 32'sd0;
                        membrane_bank[clear_lane][clear_word_q] <= 32'sd0;
                    end
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
                            for (response_lane = 0; response_lane < LANES;
                                 response_lane = response_lane + 1)
                                if (expected_lane_mask[response_lane])
                                    accumulator_bank[response_lane][response_word_index] <=
                                        saturating_add32(
                                            accumulator_bank[response_lane][response_word_index],
                                            $signed(i_k8_rsp_contribution_flat[
                                                response_lane*32 +: 32]));
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
                        current_spikes_q <= '0;
                        state_q <= STATE_LIF;
                    end else if (previous_spikes_q[
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
                    for (lif_lane = 0; lif_lane < LANES;
                         lif_lane = lif_lane + 1) begin
                        if ((lif_word_q*LANES + lif_lane) < active_neurons_q) begin
                            if (lif_pre_value(
                                    membrane_bank[lif_lane][lif_word_q],
                                    accumulator_bank[lif_lane][lif_word_q],
                                    leak_q15_q, current_left_shift_q) >= threshold_q) begin
                                current_spikes_q[
                                    lif_word_q*LANES + lif_lane
                                ] <= 1'b1;
                                if (reset_to_zero_q)
                                    membrane_bank[lif_lane][lif_word_q] <= 32'sd0;
                                else
                                    membrane_bank[lif_lane][lif_word_q] <=
                                        lif_pre_value(
                                            membrane_bank[lif_lane][lif_word_q],
                                            accumulator_bank[lif_lane][lif_word_q],
                                            leak_q15_q, current_left_shift_q) - threshold_q;
                            end else begin
                                current_spikes_q[
                                    lif_word_q*LANES + lif_lane
                                ] <= 1'b0;
                                membrane_bank[lif_lane][lif_word_q] <=
                                    lif_pre_value(
                                        membrane_bank[lif_lane][lif_word_q],
                                        accumulator_bank[lif_lane][lif_word_q],
                                        leak_q15_q, current_left_shift_q);
                            end
                            accumulator_bank[lif_lane][lif_word_q] <= 32'sd0;
                        end
                    end
                    if (((10'(lif_word_q) + 10'd1) * 10'(LANES)) >=
                        active_neurons_q) begin
                        state_q <= STATE_RESULT;
                    end else begin
                        lif_word_q <= lif_word_q + 1'b1;
                    end
                end
                STATE_RESULT: begin
                    if (o_timestep_valid && i_timestep_ready) begin
                        previous_spikes_q <= current_spikes_q;
                        current_spikes_q <= '0;
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
