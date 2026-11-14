`timescale 1ns/1ps
`default_nettype none

// SRNN topology controller for a tile-major H40 array. It owns no recurrent
// accumulator, membrane or spike storage; all persistent state remains in the
// selected physical tile array.
module unified_srnn_tile_major_controller_v1 #(
    parameter integer HIDDEN_MAX = 480,
    parameter integer INPUT_MAX = 2312,
    parameter integer TILE_NEURONS = 40,
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

    output logic                                      o_row_req_valid,
    input  logic                                      i_row_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_row_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_row_req_id,
    output logic                                      o_row_req_recurrent,
    output logic [31:0]                               o_row_req_base,
    output logic [9:0]                                o_row_req_destination_count,
    output logic signed [7:0]                        o_row_req_activation,

    input  logic                                      i_row_done_valid,
    output logic                                      o_row_done_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_row_done_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_row_done_id,

    output logic                                      o_state_clear_spikes,
    output logic                                      o_state_clear_word_valid,
    output logic [2:0]                                o_state_clear_local_word,
    output logic [9:0]                                o_state_active_neurons,
    output logic                                      o_state_lif_begin,
    output logic                                      o_state_lif_word_valid,
    output logic [2:0]                                o_state_lif_local_word,
    output logic signed [15:0]                       o_state_leak_q15,
    output logic signed [31:0]                       o_state_threshold,
    output logic [4:0]                                o_state_current_left_shift,
    output logic                                      o_state_reset_to_zero,
    output logic                                      o_state_commit_timestep,
    input  logic [HIDDEN_MAX-1:0]                     i_state_previous_spikes,
    input  logic [HIDDEN_MAX-1:0]                     i_state_current_spikes,

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
    localparam integer LOCAL_WORDS = TILE_NEURONS/LANES;
    localparam integer HIDDEN_ADDRESS_WIDTH =
        (HIDDEN_MAX <= 1) ? 1 : $clog2(HIDDEN_MAX);
    localparam logic [3:0] STATE_IDLE = 4'd0;
    localparam logic [3:0] STATE_CLEAR = 4'd1;
    localparam logic [3:0] STATE_INPUT = 4'd2;
    localparam logic [3:0] STATE_REQUEST = 4'd3;
    localparam logic [3:0] STATE_WAIT_ROW = 4'd4;
    localparam logic [3:0] STATE_SCAN = 4'd5;
    localparam logic [3:0] STATE_LIF = 4'd6;
    localparam logic [3:0] STATE_RESULT = 4'd7;
    localparam logic [3:0] STATE_ERROR = 4'd8;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER = 8'h01;
    localparam logic [7:0] ERROR_CONFIG = 8'h02;
    localparam logic [7:0] ERROR_EVENT = 8'h03;
    localparam logic [7:0] ERROR_UNEXPECTED_DONE = 8'h04;
    localparam logic [7:0] ERROR_MIXED_COMPLETION = 8'h05;
    localparam logic [7:0] ERROR_RESET_BUSY = 8'h07;

    logic [3:0] state_q;
    logic clear_for_start_q;
    logic [2:0] clear_word_q;
    logic [2:0] lif_word_q;
    logic [9:0] scan_source_q;
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
    logic [31:0] request_base_q;
    logic signed [7:0] request_activation_q;
    logic event_timestep_last_q;
    logic timestep_sample_last_q;
    logic start_config_valid;
    logic row_done_identity_valid;

    always_comb begin
        start_config_valid = i_input_dim != 0 &&
            i_input_dim <= 13'(INPUT_MAX) && i_active_neurons != 0 &&
            i_active_neurons <= 10'(HIDDEN_MAX);
        row_done_identity_valid = i_row_done_version == version_q &&
            i_row_done_id == request_id_q;

        o_start_ready = !i_abort && state_q == STATE_IDLE &&
            i_owner_enable && !o_sticky_error;
        o_event_ready = !i_abort && state_q == STATE_INPUT &&
            i_owner_enable && !o_sticky_error;
        o_row_req_valid = !i_abort && state_q == STATE_REQUEST &&
            i_owner_enable && !o_sticky_error;
        o_row_req_version = version_q;
        o_row_req_id = request_id_q;
        o_row_req_recurrent = request_recurrent_q;
        o_row_req_base = request_base_q;
        o_row_req_destination_count = active_neurons_q;
        o_row_req_activation = request_activation_q;
        o_row_done_ready = !i_abort && state_q == STATE_WAIT_ROW &&
            i_owner_enable && !o_sticky_error;

        o_state_clear_spikes =
            (i_start_valid && o_start_ready && start_config_valid) ||
            (i_state_reset &&
             (state_q == STATE_IDLE || state_q == STATE_ERROR));
        o_state_clear_word_valid = state_q == STATE_CLEAR;
        o_state_clear_local_word = clear_word_q;
        o_state_active_neurons = active_neurons_q;
        o_state_lif_begin = state_q == STATE_SCAN &&
            (!recurrent_enable_q || scan_source_q >= active_neurons_q);
        o_state_lif_word_valid = state_q == STATE_LIF;
        o_state_lif_local_word = lif_word_q;
        o_state_leak_q15 = leak_q15_q;
        o_state_threshold = threshold_q;
        o_state_current_left_shift = current_left_shift_q;
        o_state_reset_to_zero = reset_to_zero_q;

        o_timestep_valid = !i_abort && state_q == STATE_RESULT;
        o_state_commit_timestep = o_timestep_valid && i_timestep_ready;
        o_timestep_version = version_q;
        o_timestep_sample_last = timestep_sample_last_q;
        o_timestep_spikes = i_state_current_spikes;
        o_previous_spikes = i_state_previous_spikes;
        o_busy = state_q != STATE_IDLE && state_q != STATE_ERROR;
        o_idle = state_q == STATE_IDLE;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            clear_for_start_q <= 1'b0;
            clear_word_q <= '0;
            lif_word_q <= '0;
            scan_source_q <= '0;
            input_dim_q <= '0;
            active_neurons_q <= '0;
            w0_base_q <= '0;
            recurrent_base_q <= '0;
            recurrent_enable_q <= 1'b0;
            leak_q15_q <= '0;
            threshold_q <= '0;
            current_left_shift_q <= '0;
            reset_to_zero_q <= 1'b0;
            version_q <= '0;
            next_request_id_q <= '0;
            request_id_q <= '0;
            request_recurrent_q <= 1'b0;
            request_base_q <= '0;
            request_activation_q <= '0;
            event_timestep_last_q <= 1'b0;
            timestep_sample_last_q <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            clear_for_start_q <= 1'b0;
            clear_word_q <= '0;
            lif_word_q <= '0;
            scan_source_q <= '0;
            input_dim_q <= '0;
            active_neurons_q <= '0;
            w0_base_q <= '0;
            recurrent_base_q <= '0;
            recurrent_enable_q <= 1'b0;
            leak_q15_q <= '0;
            threshold_q <= '0;
            current_left_shift_q <= '0;
            reset_to_zero_q <= 1'b0;
            version_q <= '0;
            request_id_q <= '0;
            request_recurrent_q <= 1'b0;
            request_base_q <= '0;
            request_activation_q <= '0;
            event_timestep_last_q <= 1'b0;
            timestep_sample_last_q <= 1'b0;
        end else if (i_state_reset) begin
            if (state_q == STATE_IDLE || state_q == STATE_ERROR) begin
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
        end else if (state_q != STATE_IDLE && state_q != STATE_ERROR &&
                     !i_owner_enable) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_OWNER;
        end else if (i_row_done_valid && state_q != STATE_WAIT_ROW) begin
            state_q <= STATE_ERROR;
            o_sticky_error <= 1'b1;
            o_error_code <= ERROR_UNEXPECTED_DONE;
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
                        end else if (!start_config_valid) begin
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
                    if (clear_word_q == 3'(LOCAL_WORDS-1)) begin
                        clear_word_q <= '0;
                        state_q <= clear_for_start_q ? STATE_INPUT : STATE_IDLE;
                    end else begin
                        clear_word_q <= clear_word_q + 1'b1;
                    end
                end
                STATE_INPUT: begin
                    if (i_event_valid) begin
                        if (i_event_addr >= input_dim_q ||
                            (i_sample_last && !i_timestep_last)) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_EVENT;
                        end else begin
                            request_id_q <= next_request_id_q;
                            next_request_id_q <= next_request_id_q + 1'b1;
                            request_recurrent_q <= 1'b0;
                            request_base_q <= w0_base_q +
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
                    if (o_row_req_valid && i_row_req_ready)
                        state_q <= STATE_WAIT_ROW;
                end
                STATE_WAIT_ROW: begin
                    if (i_row_done_valid && o_row_done_ready) begin
                        if (!row_done_identity_valid) begin
                            state_q <= STATE_ERROR;
                            o_sticky_error <= 1'b1;
                            o_error_code <= ERROR_MIXED_COMPLETION;
                        end else if (request_recurrent_q) begin
                            scan_source_q <= scan_source_q + 1'b1;
                            state_q <= STATE_SCAN;
                        end else if (event_timestep_last_q) begin
                            scan_source_q <= '0;
                            state_q <= STATE_SCAN;
                        end else begin
                            state_q <= STATE_INPUT;
                        end
                    end
                end
                STATE_SCAN: begin
                    if (!recurrent_enable_q ||
                        scan_source_q >= active_neurons_q) begin
                        lif_word_q <= '0;
                        state_q <= STATE_LIF;
                    end else if (i_state_previous_spikes[
                            HIDDEN_ADDRESS_WIDTH'(scan_source_q)]) begin
                        request_id_q <= next_request_id_q;
                        next_request_id_q <= next_request_id_q + 1'b1;
                        request_recurrent_q <= 1'b1;
                        request_base_q <= recurrent_base_q +
                            scan_source_q * active_neurons_q;
                        request_activation_q <= 8'sd1;
                        event_timestep_last_q <= 1'b0;
                        state_q <= STATE_REQUEST;
                    end else begin
                        scan_source_q <= scan_source_q + 1'b1;
                    end
                end
                STATE_LIF: begin
                    if (lif_word_q == 3'(LOCAL_WORDS-1))
                        state_q <= STATE_RESULT;
                    else
                        lif_word_q <= lif_word_q + 1'b1;
                end
                STATE_RESULT: begin
                    if (o_timestep_valid && i_timestep_ready)
                        state_q <= timestep_sample_last_q ?
                            STATE_IDLE : STATE_INPUT;
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

    initial begin
        if (TILE_NEURONS != 40 || LANES != 8 || LOCAL_WORDS != 5)
            $error("tile-major SRNN controller requires H40/K8 geometry");
    end
endmodule

`default_nettype wire
