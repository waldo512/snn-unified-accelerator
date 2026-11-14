`timescale 1ns/1ps
`default_nettype none

// Recurrent extension prototype over the shared Tile fabric. SPARSE_FC is
// intentionally rejected: authoritative FC dispatch uses the canonical fast path.
// Word 6 is the recurrent matrix base; word 7 is the input matrix base.
module unified_fc_recurrent_layer_engine #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40,
    parameter integer LAYER_WORDS = 12,
    parameter integer INPUT_MAX = 2312,
    parameter integer HIDDEN_MAX = 480,
    parameter integer OUTPUT_MAX = 32
) (
    input  wire logic                                  clk,
    input  wire logic                                  rst_n,
    input  wire logic                                  i_clear_error,
    input  wire logic                                  i_layer_valid,
    output logic                                       o_layer_ready,
    input  wire logic [LAYER_WORDS*32-1:0]            i_layer_words,
    output logic                                       o_layer_done,

    input  wire logic                                  i_event_valid,
    output logic                                       o_event_ready,
    input  wire logic [12:0]                           i_event_addr,
    input  wire logic signed [7:0]                     i_event_data,
    input  wire logic                                  i_timestep_last,
    input  wire logic                                  i_sample_last,

    output logic [TILE_COUNT-1:0]                     o_weight_req_valid,
    input  wire logic [TILE_COUNT-1:0]                i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_weight_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]             i_weight_data_flat,

    input  wire logic signed [15:0]                   i_leak_q15,
    input  wire logic signed [31:0]                   i_threshold,
    input  wire logic [4:0]                            i_current_left_shift,
    input  wire logic                                  i_reset_to_zero,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]    o_previous_spikes_flat,
    output logic [TILE_COUNT*NEURONS_PER_TILE-1:0]    o_current_spikes_flat,
    output logic                                       o_timestep_valid,
    input  wire logic                                  i_timestep_ready,
    output logic                                       o_timestep_sample_last,
    output logic                                       o_busy,
    output logic                                       o_sticky_error
);
    localparam logic [3:0] LAYER_RECURRENT_FC = 4'd5;

    localparam logic [3:0] STATE_IDLE = 4'd0;
    localparam logic [3:0] STATE_INPUT_CMD = 4'd1;
    localparam logic [3:0] STATE_INPUT_SEND = 4'd2;
    localparam logic [3:0] STATE_INPUT_DRAIN = 4'd3;
    localparam logic [3:0] STATE_SCAN_START = 4'd4;
    localparam logic [3:0] STATE_SCAN = 4'd5;
    localparam logic [3:0] STATE_SCAN_DRAIN = 4'd6;
    localparam logic [3:0] STATE_LIF_START = 4'd7;
    localparam logic [3:0] STATE_LIF_WAIT = 4'd8;
    localparam logic [3:0] STATE_COMMIT = 4'd9;
    localparam logic [3:0] STATE_ERROR = 4'd10;

    logic [3:0] state_q;
    logic [3:0] active_tiles_q;
    logic [12:0] input_dim_q;
    logic [9:0] state_dim_q;
    logic [31:0] recurrent_base_q;
    logic [31:0] w0_base_q;
    logic [31:0] input_row_base_q;
    logic signed [7:0] input_activation_q;
    logic timestep_last_q;
    logic sample_last_q;
    logic timestep_commit_issued_q;
    logic scanner_start;
    logic scanner_cmd_valid;
    logic scanner_cmd_ready;
    logic [31:0] scanner_row_base;
    logic signed [7:0] scanner_activation;
    logic scanner_done;
    logic fabric_row_valid;
    logic fabric_row_ready;
    logic fabric_row_done;
    logic fabric_lif_start;
    logic fabric_lif_done;
    logic fabric_lif_busy;
    logic fabric_clear_sample;
    logic fabric_timestep_commit;
    logic fabric_error;
    logic [3:0] descriptor_layer_type;
    logic [12:0] descriptor_input_dim;
    logic [9:0] descriptor_state_dim;
    logic [5:0] descriptor_output_dim;
    logic [4:0] descriptor_active_units;
    logic [3:0] descriptor_issue_k;
    logic descriptor_valid;

    always_comb begin
        descriptor_layer_type = i_layer_words[2*32 +: 4];
        descriptor_input_dim = i_layer_words[3*32 +: 13];
        descriptor_state_dim = i_layer_words[3*32 + 13 +: 10];
        descriptor_output_dim = i_layer_words[3*32 + 23 +: 6];
        descriptor_active_units = i_layer_words[4*32 +: 5];
        descriptor_issue_k = i_layer_words[4*32 + 14 +: 4];
        descriptor_valid =
            (descriptor_layer_type == LAYER_RECURRENT_FC) &&
            (descriptor_input_dim > 0) &&
            (descriptor_input_dim <= INPUT_MAX) &&
            (descriptor_active_units > 0) &&
            (descriptor_active_units <= TILE_COUNT) &&
            (descriptor_state_dim <= HIDDEN_MAX) &&
            (descriptor_state_dim == descriptor_active_units*NEURONS_PER_TILE) &&
            (descriptor_output_dim > 0) &&
            (descriptor_output_dim <= OUTPUT_MAX) &&
            (descriptor_issue_k == 8);

        o_layer_ready = state_q == STATE_IDLE;
        o_timestep_valid = state_q == STATE_COMMIT;
        o_timestep_sample_last = sample_last_q;
        o_event_ready = (state_q == STATE_INPUT_CMD) &&
            (!i_event_valid || (i_event_addr < input_dim_q));
        o_busy = state_q != STATE_IDLE;
        scanner_start = state_q == STATE_SCAN_START;
        fabric_lif_start = state_q == STATE_LIF_START;
        fabric_clear_sample = i_layer_valid && o_layer_ready && descriptor_valid;
        // Previous-timestep state belongs only to RECURRENT_FC. Sparse FC must
        // not toggle or overwrite the recurrent sidecar state.
        fabric_timestep_commit = (state_q == STATE_COMMIT) && !timestep_commit_issued_q;

        fabric_row_valid = 1'b0;
        scanner_cmd_ready = 1'b0;
        if (state_q == STATE_INPUT_SEND)
            fabric_row_valid = 1'b1;
        else if (state_q == STATE_SCAN) begin
            fabric_row_valid = scanner_cmd_valid;
            scanner_cmd_ready = fabric_row_ready;
        end
    end

    unified_previous_spike_scanner #(
        .NEURONS(TILE_COUNT*NEURONS_PER_TILE)
    ) u_recurrent_scanner (
        .clk(clk), .rst_n(rst_n), .i_start(scanner_start),
        .i_previous_spikes(o_previous_spikes_flat),
        .i_recurrent_base(recurrent_base_q), .i_row_stride(state_dim_q),
        .o_cmd_valid(scanner_cmd_valid), .i_cmd_ready(scanner_cmd_ready),
        .o_weight_row_base(scanner_row_base),
        .o_activation(scanner_activation), .o_busy(), .o_done(scanner_done)
    );

    unified_k8_tile_fabric #(
        .TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE)
    ) u_fabric (
        .clk(clk), .rst_n(rst_n), .i_clear_sample(fabric_clear_sample),
        .i_clear_accum(1'b0), .i_active_tiles(active_tiles_q),
        .i_row_valid(fabric_row_valid), .o_row_ready(fabric_row_ready),
        .i_weight_row_base((state_q == STATE_SCAN) ?
                           scanner_row_base : input_row_base_q),
        .i_destination_count(state_dim_q),
        .i_activation((state_q == STATE_SCAN) ?
                      scanner_activation : input_activation_q),
        .i_acc24_compat(1'b0), .o_row_done(fabric_row_done),
        .o_weight_req_valid(o_weight_req_valid),
        .i_weight_req_ready(i_weight_req_ready),
        .o_weight_addr_flat(o_weight_addr_flat),
        .i_weight_data_flat(i_weight_data_flat),
        .i_lif_start(fabric_lif_start), .i_leak_q15(i_leak_q15),
        .i_threshold(i_threshold), .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero), .o_lif_busy(fabric_lif_busy),
        .o_lif_done(fabric_lif_done),
        .i_timestep_commit(fabric_timestep_commit),
        .o_previous_spikes_flat(o_previous_spikes_flat),
        .o_current_spikes_flat(o_current_spikes_flat),
        .o_sticky_error(fabric_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            active_tiles_q <= 4'd0;
            input_dim_q <= 13'd0;
            state_dim_q <= 10'd0;
            recurrent_base_q <= 32'd0;
            w0_base_q <= 32'd0;
            input_row_base_q <= 32'd0;
            input_activation_q <= 8'sd0;
            timestep_last_q <= 1'b0;
            sample_last_q <= 1'b0;
            timestep_commit_issued_q <= 1'b0;
            o_layer_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_layer_done <= 1'b0;
            if (i_clear_error)
                o_sticky_error <= 1'b0;
            if (fabric_error)
                o_sticky_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (i_layer_valid) begin
                        if (!descriptor_valid) begin
                            o_sticky_error <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else begin
                            active_tiles_q <= descriptor_active_units[3:0];
                            input_dim_q <= descriptor_input_dim;
                            state_dim_q <= descriptor_state_dim;
                            recurrent_base_q <= i_layer_words[6*32 +: 32];
                            w0_base_q <= i_layer_words[7*32 +: 32];
                            state_q <= STATE_INPUT_CMD;
                        end
                    end
                end
                STATE_INPUT_CMD: begin
                    if (i_event_valid) begin
                        if (i_event_addr >= input_dim_q) begin
                            o_sticky_error <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else begin
                            input_row_base_q <= w0_base_q +
                                i_event_addr*state_dim_q;
                            input_activation_q <= i_event_data;
                            timestep_last_q <= i_timestep_last;
                            sample_last_q <= i_sample_last;
                            state_q <= STATE_INPUT_SEND;
                        end
                    end
                end
                STATE_INPUT_SEND: begin
                    if (fabric_row_valid && fabric_row_ready)
                        state_q <= STATE_INPUT_DRAIN;
                end
                STATE_INPUT_DRAIN: begin
                    if (fabric_row_done) begin
                        if (!timestep_last_q)
                            state_q <= STATE_INPUT_CMD;
                        else
                            state_q <= STATE_SCAN_START;
                    end
                end
                STATE_SCAN_START: state_q <= STATE_SCAN;
                STATE_SCAN: begin
                    if (scanner_done)
                        state_q <= STATE_SCAN_DRAIN;
                end
                STATE_SCAN_DRAIN: begin
                    if (fabric_row_ready)
                        state_q <= STATE_LIF_START;
                end
                STATE_LIF_START: state_q <= STATE_LIF_WAIT;
                STATE_LIF_WAIT: begin
                    if (fabric_lif_done) begin
                        timestep_commit_issued_q <= 1'b0;
                        state_q <= STATE_COMMIT;
                    end
                end
                STATE_COMMIT: begin
                    timestep_commit_issued_q <= 1'b1;
                    if (i_timestep_ready) begin
                        if (sample_last_q) begin
                            o_layer_done <= 1'b1;
                            timestep_commit_issued_q <= 1'b0;
                            state_q <= STATE_IDLE;
                        end else begin
                            timestep_commit_issued_q <= 1'b0;
                            state_q <= STATE_INPUT_CMD;
                        end
                    end
                end
                STATE_ERROR: begin
                    o_layer_done <= 1'b1;
                    state_q <= STATE_IDLE;
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
