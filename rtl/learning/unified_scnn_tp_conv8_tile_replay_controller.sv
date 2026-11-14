`timescale 1ns/1ps
`default_nettype none

// One captured timestep is scanned output-wave outer, input-group inner.
// A BxB modulation RAM is loaded once and reused for every tile. The learner's
// done input must mean RMW responses/writes have drained, not gradient-last.
module unified_scnn_tp_conv8_tile_replay_controller #(
    parameter int MAX_BATCH = 10,
    parameter int TILE_INPUTS = 8,
    parameter int TILE_OUTPUTS = 12,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH+1)
) (
    input wire logic i_clk, i_rst_n,
    input wire logic i_cfg_valid,
    output logic o_cfg_ready,
    input wire logic [BATCH_COUNT_WIDTH-1:0] i_cfg_batch_count,
    input wire logic [9:0] i_cfg_input_channels, i_cfg_output_channels,
    input wire logic i_modulation_valid,
    output logic o_modulation_ready,
    input wire logic [SAMPLE_WIDTH-1:0] i_modulation_row, i_modulation_col,
    input wire logic signed [15:0] i_modulation_q8_8,
    input wire logic i_store_capture_complete,
    input wire logic i_child_error,
    output logic o_tile_cfg_valid,
    input wire logic i_tile_cfg_ready,
    output logic [8:0] o_input_channel_base, o_output_channel_base,
    output logic [9:0] o_input_channel_count, o_output_channel_count,
    output logic o_replay_valid,
    input wire logic i_replay_ready, i_replay_done,
    output logic o_modulation_valid,
    input wire logic i_modulation_ready,
    output logic [SAMPLE_WIDTH-1:0] o_modulation_row, o_modulation_col,
    output logic signed [15:0] o_modulation_q8_8,
    input wire logic i_tile_capture_complete,
    output logic o_tile_start,
    input wire logic i_tile_start_ready, i_tile_done,
    output logic o_tile_release,
    input wire logic i_tile_release_ready,
    output logic o_window_done,
    input wire logic i_window_done_ready,
    output logic o_active, o_protocol_error,
    output logic [12:0] o_tiles_completed
);
    typedef enum logic [3:0] {IDLE, CAPTURE, CONFIGURE, REPLAY_REQUEST,
        MATRIX_READ, MATRIX_SEND, CAPTURE_WAIT, START_TILE, WAIT_TILE,
        RELEASE_TILE, WINDOW_DONE, FAULT} state_t;
    state_t state_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_q;
    logic [9:0] inputs_q, outputs_q;
    logic [8:0] input_base_q, output_base_q;
    logic [SAMPLE_WIDTH-1:0] capture_row_q, capture_col_q;
    logic [SAMPLE_WIDTH-1:0] replay_row_q, replay_col_q;
    logic matrix_complete_q, replay_done_q;
    logic cfg_legal, capture_legal;
    (* ram_style = "block" *) logic signed [15:0] matrix_mem [0:MAX_BATCH*MAX_BATCH-1];

    assign cfg_legal = i_cfg_batch_count != 0 &&
        int'(i_cfg_batch_count) <= MAX_BATCH &&
        i_cfg_input_channels != 0 && i_cfg_input_channels <= 10'd512 &&
        i_cfg_output_channels != 0 && i_cfg_output_channels <= 10'd512;
    assign capture_legal = i_modulation_row == capture_row_q &&
        i_modulation_col == capture_col_q;
    assign o_cfg_ready = i_rst_n && state_q == IDLE;
    assign o_active = state_q != IDLE;
    assign o_modulation_ready = state_q == CAPTURE && !matrix_complete_q &&
        !i_child_error;
    assign o_tile_cfg_valid = state_q == CONFIGURE && !i_child_error;
    assign o_input_channel_base = input_base_q;
    assign o_output_channel_base = output_base_q;
    assign o_input_channel_count = int'(inputs_q) - int'(input_base_q) > TILE_INPUTS ?
        10'(TILE_INPUTS) : inputs_q - {1'b0, input_base_q};
    assign o_output_channel_count = int'(outputs_q) - int'(output_base_q) > TILE_OUTPUTS ?
        10'(TILE_OUTPUTS) : outputs_q - {1'b0, output_base_q};
    assign o_replay_valid = state_q == REPLAY_REQUEST && !i_child_error;
    assign o_modulation_valid = state_q == MATRIX_SEND && !i_child_error;
    assign o_modulation_row = replay_row_q;
    assign o_modulation_col = replay_col_q;
    assign o_tile_start = state_q == START_TILE && !i_child_error;
    assign o_tile_release = state_q == RELEASE_TILE && !i_child_error;
    assign o_window_done = state_q == WINDOW_DONE && !i_child_error;

    always_ff @(posedge i_clk) begin
        if (i_modulation_valid && o_modulation_ready && capture_legal)
            matrix_mem[int'(capture_row_q)*MAX_BATCH+int'(capture_col_q)] <=
                i_modulation_q8_8;
        if (state_q == MATRIX_READ)
            o_modulation_q8_8 <= matrix_mem[
                int'(replay_row_q)*MAX_BATCH+int'(replay_col_q)];
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            batch_q <= '0;
            inputs_q <= '0;
            outputs_q <= '0;
            input_base_q <= '0;
            output_base_q <= '0;
            capture_row_q <= '0;
            capture_col_q <= '0;
            replay_row_q <= '0;
            replay_col_q <= '0;
            matrix_complete_q <= 1'b0;
            replay_done_q <= 1'b0;
            o_protocol_error <= 1'b0;
            o_tiles_completed <= '0;
        end else if (i_child_error && state_q != IDLE) begin
            o_protocol_error <= 1'b1;
            state_q <= FAULT;
        end else begin
            if (i_replay_done && state_q inside {MATRIX_READ, MATRIX_SEND, CAPTURE_WAIT})
                replay_done_q <= 1'b1;
            case (state_q)
                IDLE: if (i_cfg_valid) begin
                    o_protocol_error <= !cfg_legal;
                    if (cfg_legal) begin
                        batch_q <= i_cfg_batch_count;
                        inputs_q <= i_cfg_input_channels;
                        outputs_q <= i_cfg_output_channels;
                        input_base_q <= '0;
                        output_base_q <= '0;
                        capture_row_q <= '0;
                        capture_col_q <= '0;
                        matrix_complete_q <= 1'b0;
                        o_tiles_completed <= '0;
                        state_q <= CAPTURE;
                    end
                end
                CAPTURE: begin
                    if (i_modulation_valid && o_modulation_ready) begin
                        if (!capture_legal) begin
                            o_protocol_error <= 1'b1;
                            state_q <= FAULT;
                        end else if (int'(capture_col_q)+1 == int'(batch_q)) begin
                            capture_col_q <= '0;
                            if (int'(capture_row_q)+1 == int'(batch_q))
                                matrix_complete_q <= 1'b1;
                            else capture_row_q <= capture_row_q + 1'b1;
                        end else capture_col_q <= capture_col_q + 1'b1;
                    end
                    if (matrix_complete_q && i_store_capture_complete)
                        state_q <= CONFIGURE;
                end
                CONFIGURE: if (i_tile_cfg_ready) begin
                    replay_row_q <= '0;
                    replay_col_q <= '0;
                    replay_done_q <= 1'b0;
                    state_q <= REPLAY_REQUEST;
                end
                REPLAY_REQUEST: if (i_replay_ready) state_q <= MATRIX_READ;
                MATRIX_READ: state_q <= MATRIX_SEND;
                MATRIX_SEND: if (i_modulation_ready) begin
                    if (int'(replay_col_q)+1 == int'(batch_q)) begin
                        replay_col_q <= '0;
                        if (int'(replay_row_q)+1 == int'(batch_q))
                            state_q <= CAPTURE_WAIT;
                        else begin
                            replay_row_q <= replay_row_q + 1'b1;
                            state_q <= MATRIX_READ;
                        end
                    end else begin
                        replay_col_q <= replay_col_q + 1'b1;
                        state_q <= MATRIX_READ;
                    end
                end
                CAPTURE_WAIT: if (replay_done_q && i_tile_capture_complete)
                    state_q <= START_TILE;
                START_TILE: if (i_tile_start_ready) state_q <= WAIT_TILE;
                WAIT_TILE: if (i_tile_done) state_q <= RELEASE_TILE;
                RELEASE_TILE: if (i_tile_release_ready) begin
                    o_tiles_completed <= o_tiles_completed + 1'b1;
                    if (int'(input_base_q)+int'(o_input_channel_count) < int'(inputs_q)) begin
                        input_base_q <= 9'(int'(input_base_q)+int'(o_input_channel_count));
                        state_q <= CONFIGURE;
                    end else if (int'(output_base_q)+int'(o_output_channel_count) < int'(outputs_q)) begin
                        input_base_q <= '0;
                        output_base_q <= 9'(int'(output_base_q)+int'(o_output_channel_count));
                        state_q <= CONFIGURE;
                    end else state_q <= WINDOW_DONE;
                end
                WINDOW_DONE: if (i_window_done_ready) state_q <= IDLE;
                FAULT: state_q <= FAULT;
                default: begin
                    o_protocol_error <= 1'b1;
                    state_q <= FAULT;
                end
            endcase
        end
    end
    initial begin
        if (TILE_INPUTS < 1 || TILE_INPUTS > 512 || TILE_OUTPUTS < 1 || TILE_OUTPUTS > 512)
            $error("Invalid Conv8 tile dimensions");
    end
endmodule
`default_nettype wire
