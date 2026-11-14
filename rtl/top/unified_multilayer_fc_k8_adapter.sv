`timescale 1ns/1ps
`default_nettype none

// Generic INT8 FC adapter. It translates either sparse external input events or
// a committed predecessor spike vector into shared K8 Tile row commands. The
// one-layer control path can select it for integration tests, but canonical FC
// replacement additionally requires the distinct FC_COMPAT numeric mode.
module unified_multilayer_fc_k8_adapter #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40,
    parameter integer HIDDEN_MAX = 480
) (
    input  wire logic                                  clk,
    input  wire logic                                  rst_n,
    input  wire logic                                  i_clear_error,
    input  wire logic                                  i_compute_valid,
    output logic                                       o_compute_ready,
    input  wire logic                                  i_compute_uses_external_events,
    input  wire logic [HIDDEN_MAX-1:0]                 i_compute_input_spikes,
    input  wire logic [12:0]                           i_compute_input_dim,
    input  wire logic [9:0]                            i_compute_state_dim,
    input  wire logic [4:0]                            i_compute_active_tiles,
    input  wire logic [31:0]                           i_compute_weight_base,
    input  wire logic                                  i_event_valid,
    output logic                                       o_event_ready,
    input  wire logic [12:0]                           i_event_addr,
    input  wire logic signed [7:0]                     i_event_data,
    input  wire logic                                  i_event_last,
    output logic                                       o_compute_done,
    output logic [HIDDEN_MAX-1:0]                     o_compute_output_spikes,

    output logic [TILE_COUNT-1:0]                     o_weight_req_valid,
    input  wire logic [TILE_COUNT-1:0]                i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_weight_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]             i_weight_data_flat,
    input  wire logic signed [15:0]                   i_leak_q15,
    input  wire logic signed [31:0]                   i_threshold,
    input  wire logic [4:0]                            i_current_left_shift,
    input  wire logic                                  i_reset_to_zero,
    output logic                                       o_busy,
    output logic                                       o_sticky_error
);
    localparam logic [3:0] STATE_IDLE = 4'd0;
    localparam logic [3:0] STATE_EXTERNAL_WAIT = 4'd1;
    localparam logic [3:0] STATE_EXTERNAL_SEND = 4'd2;
    localparam logic [3:0] STATE_EXTERNAL_DRAIN = 4'd3;
    localparam logic [3:0] STATE_SCAN_START = 4'd4;
    localparam logic [3:0] STATE_SCAN = 4'd5;
    localparam logic [3:0] STATE_SCAN_DRAIN = 4'd6;
    localparam logic [3:0] STATE_LIF_START = 4'd7;
    localparam logic [3:0] STATE_LIF_WAIT = 4'd8;

    logic [3:0] state_q;
    logic [9:0] state_dim_q;
    logic [4:0] active_tiles_q;
    logic [31:0] weight_base_q;
    logic [12:0] input_dim_q;
    logic external_last_q;
    logic [31:0] external_row_base_q;
    logic signed [7:0] external_activation_q;
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
    logic fabric_clear_sample;
    logic fabric_error;
    logic [TILE_COUNT*NEURONS_PER_TILE-1:0] current_spikes_flat;
    logic [TILE_COUNT*NEURONS_PER_TILE-1:0] previous_spikes_unused;

    always_comb begin
        o_compute_ready = state_q == STATE_IDLE;
        o_event_ready = (state_q == STATE_EXTERNAL_WAIT) &&
            (!i_event_valid || (i_event_addr < input_dim_q));
        o_busy = state_q != STATE_IDLE;
        scanner_start = state_q == STATE_SCAN_START;
        fabric_clear_sample = i_compute_valid && o_compute_ready;
        fabric_lif_start = state_q == STATE_LIF_START;
        fabric_row_valid = 1'b0;
        scanner_cmd_ready = 1'b0;
        if (state_q == STATE_EXTERNAL_SEND)
            fabric_row_valid = 1'b1;
        else if (state_q == STATE_SCAN) begin
            fabric_row_valid = scanner_cmd_valid;
            scanner_cmd_ready = fabric_row_ready;
        end
        o_compute_output_spikes = current_spikes_flat;
    end

    unified_previous_spike_scanner #(.NEURONS(HIDDEN_MAX)) u_input_scanner (
        .clk(clk), .rst_n(rst_n), .i_start(scanner_start),
        .i_previous_spikes(i_compute_input_spikes),
        .i_recurrent_base(weight_base_q), .i_row_stride(state_dim_q),
        .o_cmd_valid(scanner_cmd_valid), .i_cmd_ready(scanner_cmd_ready),
        .o_weight_row_base(scanner_row_base), .o_activation(scanner_activation),
        .o_busy(), .o_done(scanner_done)
    );

    unified_k8_tile_fabric #(
        .TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE)
    ) u_fabric (
        .clk(clk), .rst_n(rst_n), .i_clear_sample(fabric_clear_sample),
        .i_clear_accum(1'b0), .i_active_tiles(active_tiles_q[3:0]),
        .i_row_valid(fabric_row_valid), .o_row_ready(fabric_row_ready),
        .i_weight_row_base((state_q == STATE_SCAN) ? scanner_row_base : external_row_base_q),
        .i_destination_count(state_dim_q),
        .i_activation((state_q == STATE_SCAN) ? scanner_activation : external_activation_q),
        .i_acc24_compat(1'b0), .o_row_done(fabric_row_done),
        .o_weight_req_valid(o_weight_req_valid), .i_weight_req_ready(i_weight_req_ready),
        .o_weight_addr_flat(o_weight_addr_flat), .i_weight_data_flat(i_weight_data_flat),
        .i_lif_start(fabric_lif_start), .i_leak_q15(i_leak_q15),
        .i_threshold(i_threshold), .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero), .o_lif_busy(), .o_lif_done(fabric_lif_done),
        .i_timestep_commit(1'b0), .o_previous_spikes_flat(previous_spikes_unused),
        .o_current_spikes_flat(current_spikes_flat), .o_sticky_error(fabric_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            state_dim_q <= '0;
            active_tiles_q <= '0;
            weight_base_q <= '0;
            input_dim_q <= '0;
            external_last_q <= 1'b0;
            external_row_base_q <= '0;
            external_activation_q <= '0;
            o_compute_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_compute_done <= 1'b0;
            if (i_clear_error)
                o_sticky_error <= 1'b0;
            if (fabric_error ||
                (i_event_valid && (state_q == STATE_EXTERNAL_WAIT) &&
                 (i_event_addr >= input_dim_q)))
                o_sticky_error <= 1'b1;
            case (state_q)
                STATE_IDLE: begin
                    if (i_compute_valid && o_compute_ready) begin
                        if ((i_compute_state_dim == 0) ||
                            (i_compute_state_dim > HIDDEN_MAX) ||
                            (i_compute_active_tiles == 0) ||
                            (i_compute_active_tiles > TILE_COUNT) ||
                            (i_compute_state_dim > i_compute_active_tiles * NEURONS_PER_TILE)) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            state_dim_q <= i_compute_state_dim;
                            active_tiles_q <= i_compute_active_tiles;
                            weight_base_q <= i_compute_weight_base;
                            input_dim_q <= i_compute_input_dim;
                            state_q <= i_compute_uses_external_events ?
                                STATE_EXTERNAL_WAIT : STATE_SCAN_START;
                        end
                    end
                end
                STATE_EXTERNAL_WAIT: begin
                    if (i_event_valid && o_event_ready) begin
                        external_row_base_q <= weight_base_q + i_event_addr * state_dim_q;
                        external_activation_q <= i_event_data;
                        external_last_q <= i_event_last;
                        state_q <= STATE_EXTERNAL_SEND;
                    end
                end
                STATE_EXTERNAL_SEND: if (fabric_row_ready) state_q <= STATE_EXTERNAL_DRAIN;
                STATE_EXTERNAL_DRAIN: if (fabric_row_done)
                    state_q <= external_last_q ? STATE_LIF_START : STATE_EXTERNAL_WAIT;
                STATE_SCAN_START: state_q <= STATE_SCAN;
                STATE_SCAN: if (scanner_done) state_q <= STATE_SCAN_DRAIN;
                STATE_SCAN_DRAIN: if (fabric_row_ready) state_q <= STATE_LIF_START;
                STATE_LIF_START: state_q <= STATE_LIF_WAIT;
                STATE_LIF_WAIT: if (fabric_lif_done) begin
                    o_compute_done <= 1'b1;
                    state_q <= STATE_IDLE;
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
