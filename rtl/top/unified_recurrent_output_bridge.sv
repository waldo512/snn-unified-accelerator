`timescale 1ns/1ps
`default_nettype none

// Backpressured output bridge for recurrent and convolutional extension modes.
// It receives one committed Tile-spike vector per timestep, shares the W2
// reader, and calls the shared dynamic argmax only after the final timestep.
module unified_recurrent_output_bridge #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40,
    parameter integer OUTPUT_MAX = 32,
    parameter integer ACC_BITS = 32
) (
    input  wire logic                                            clk,
    input  wire logic                                            rst_n,
    input  wire logic                                            i_clear_error,
    input  wire logic                                            i_abort,
    input  wire logic                                            i_timestep_valid,
    output logic                                                 o_timestep_ready,
    input  wire logic                                            i_timestep_sample_last,
    input  wire logic [TILE_COUNT*NEURONS_PER_TILE-1:0]         i_spikes_flat,
    input  wire logic [3:0]                                      i_active_tiles,
    input  wire logic [5:0]                                      i_output_count,
    input  wire logic [5:0]                                      i_w2_row_stride,
    input  wire logic [31:0]                                     i_w2_base,
    input  wire logic [15:0]                                     i_version,
    output logic [TILE_COUNT-1:0]                               o_w2_req_valid,
    input  wire logic [TILE_COUNT-1:0]                          i_w2_req_ready,
    output logic [TILE_COUNT*32-1:0]                            o_w2_addr_flat,
    input  wire logic [TILE_COUNT*64-1:0]                       i_w2_data_flat,
    output logic                                                 o_result_valid,
    input  wire logic                                            i_result_ready,
    output logic [5:0]                                           o_result_class,
    output logic signed [ACC_BITS-1:0]                          o_result_score,
    output logic [OUTPUT_MAX*ACC_BITS-1:0]                      o_result_logits_flat,
    output logic                                                 o_busy,
    output logic                                                 o_sticky_error
);
    localparam logic [2:0] STATE_ACCEPT = 3'd0;
    localparam logic [2:0] STATE_ACCUM = 3'd1;
    localparam logic [2:0] STATE_OUTPUT_START = 3'd2;
    localparam logic [2:0] STATE_OUTPUT_WAIT = 3'd3;
    localparam logic [2:0] STATE_ABORT_DRAIN = 3'd4;
    logic [2:0] state_q;
    logic accumulator_start;
    logic accumulator_ready;
    logic accumulator_done;
    logic accumulator_busy;
    logic accumulator_error;
    logic pending_sample_last_q;
    logic [TILE_COUNT*OUTPUT_MAX*ACC_BITS-1:0] timestep_partials;
    logic [TILE_COUNT*OUTPUT_MAX*ACC_BITS-1:0] output_partials_q;
    logic dynamic_ready;
    logic dynamic_valid;
    logic dynamic_busy;
    logic dynamic_error;
    logic dynamic_result_ready;
    integer seq_tile;
    integer seq_class;

    always_comb begin
        o_timestep_ready = !i_abort && state_q == STATE_ACCEPT &&
            accumulator_ready;
        accumulator_start = i_timestep_valid && o_timestep_ready;
        o_busy = state_q != STATE_ACCEPT || accumulator_busy || dynamic_busy;
        o_sticky_error = accumulator_error || dynamic_error;
    end

    unified_spike_w2_accumulator #(
        .TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE),
        .OUTPUT_MAX(OUTPUT_MAX), .ACC_BITS(ACC_BITS)
    ) u_accumulator (
        .clk(clk), .rst_n(rst_n), .i_start(accumulator_start), .o_ready(accumulator_ready),
        .i_active_tiles(i_active_tiles), .i_output_count(i_output_count),
        .i_w2_row_stride(i_w2_row_stride), .i_w2_base(i_w2_base),
        .i_spikes_flat(i_spikes_flat), .o_w2_req_valid(o_w2_req_valid),
        .i_w2_req_ready(i_w2_req_ready), .o_w2_addr_flat(o_w2_addr_flat),
        .i_w2_data_flat(i_w2_data_flat), .o_partials_flat(timestep_partials),
        .o_done(accumulator_done), .o_busy(accumulator_busy), .o_sticky_error(accumulator_error)
    );

    unified_dynamic_output_engine #(
        .TILE_COUNT(TILE_COUNT), .OUTPUT_MAX(OUTPUT_MAX), .ACC_BITS(ACC_BITS)
    ) u_dynamic_output (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_start(!i_abort && state_q == STATE_OUTPUT_START && dynamic_ready),
        .o_ready(dynamic_ready),
        .i_active_tiles(i_active_tiles), .i_output_count(i_output_count),
        .i_partials_flat(output_partials_q), .i_meta(16'd0), .i_version(i_version),
        .o_valid(dynamic_valid), .i_ready(dynamic_result_ready),
        .o_logits_flat(o_result_logits_flat), .o_class(o_result_class),
        .o_score(o_result_score), .o_meta(), .o_version(), .o_busy(dynamic_busy),
        .o_sticky_error(dynamic_error)
    );
    assign dynamic_result_ready = i_abort || state_q == STATE_ABORT_DRAIN ?
        1'b1 : i_result_ready;
    assign o_result_valid = !i_abort && state_q != STATE_ABORT_DRAIN &&
        dynamic_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_ACCEPT;
            pending_sample_last_q <= 1'b0;
            output_partials_q <= '0;
        end else if (i_abort) begin
            pending_sample_last_q <= 1'b0;
            output_partials_q <= '0;
            state_q <= (accumulator_busy || dynamic_busy) ?
                STATE_ABORT_DRAIN : STATE_ACCEPT;
        end else begin
            if (i_clear_error)
                output_partials_q <= '0;
            case (state_q)
                STATE_ACCEPT: if (accumulator_start) begin
                    pending_sample_last_q <= i_timestep_sample_last;
                    state_q <= STATE_ACCUM;
                end
                STATE_ACCUM: if (accumulator_done) begin
                    for (seq_tile = 0; seq_tile < TILE_COUNT; seq_tile = seq_tile + 1)
                        for (seq_class = 0; seq_class < OUTPUT_MAX; seq_class = seq_class + 1)
                            output_partials_q[(seq_tile*OUTPUT_MAX+seq_class)*ACC_BITS +: ACC_BITS] <=
                                output_partials_q[(seq_tile*OUTPUT_MAX+seq_class)*ACC_BITS +: ACC_BITS] +
                                timestep_partials[(seq_tile*OUTPUT_MAX+seq_class)*ACC_BITS +: ACC_BITS];
                    state_q <= pending_sample_last_q ? STATE_OUTPUT_START : STATE_ACCEPT;
                end
                STATE_OUTPUT_START: if (dynamic_ready) state_q <= STATE_OUTPUT_WAIT;
                STATE_OUTPUT_WAIT: if (dynamic_valid && i_result_ready) begin
                    output_partials_q <= '0;
                    state_q <= STATE_ACCEPT;
                end
                STATE_ABORT_DRAIN: begin
                    output_partials_q <= '0;
                    if (!accumulator_busy && !dynamic_busy)
                        state_q <= STATE_ACCEPT;
                end
                default: state_q <= STATE_ACCEPT;
            endcase
        end
    end
endmodule

`default_nettype wire
