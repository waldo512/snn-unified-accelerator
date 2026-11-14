`timescale 1ns/1ps
`default_nettype none

// Shared non-FC output path. The canonical 10-class FC path bypasses this block.
module unified_dynamic_output_engine #(
    parameter integer TILE_COUNT = 12,
    parameter integer OUTPUT_MAX = 32,
    parameter integer ACC_BITS = 32,
    parameter integer META_BITS = 16
) (
    input  wire logic                                      clk,
    input  wire logic                                      rst_n,
    input  wire logic                                      i_clear_error,
    input  wire logic                                      i_start,
    output logic                                           o_ready,
    input  wire logic [3:0]                                i_active_tiles,
    input  wire logic [5:0]                                i_output_count,
    input  wire logic [TILE_COUNT*OUTPUT_MAX*ACC_BITS-1:0] i_partials_flat,
    input  wire logic [META_BITS-1:0]                      i_meta,
    input  wire logic [15:0]                               i_version,
    output logic                                           o_valid,
    input  wire logic                                      i_ready,
    output logic [OUTPUT_MAX*ACC_BITS-1:0]                 o_logits_flat,
    output logic [5:0]                                     o_class,
    output logic signed [ACC_BITS-1:0]                     o_score,
    output logic [META_BITS-1:0]                           o_meta,
    output logic [15:0]                                    o_version,
    output logic                                           o_busy,
    output logic                                           o_sticky_error
);
    localparam logic [2:0] STATE_IDLE = 3'd0;
    localparam logic [2:0] STATE_ACCUM = 3'd1;
    localparam logic [2:0] STATE_ARGMAX_INIT = 3'd2;
    localparam logic [2:0] STATE_ARGMAX = 3'd3;
    localparam logic [2:0] STATE_RESULT = 3'd4;

    logic [2:0] state_q;
    logic [3:0] tile_q;
    logic [5:0] output_q;
    logic [3:0] active_tiles_q;
    logic [5:0] output_count_q;
    logic signed [ACC_BITS-1:0] logits_q [0:OUTPUT_MAX-1];
    logic signed [ACC_BITS-1:0] best_score_q;
    logic [5:0] best_class_q;
    logic [META_BITS-1:0] meta_q;
    logic [15:0] version_q;
    logic [TILE_COUNT*OUTPUT_MAX*ACC_BITS-1:0] partials_q;
    logic capture_partials;
    integer comb_output_index;
    integer seq_output_index;

    function automatic logic signed [ACC_BITS-1:0] partial(
        input integer tile_index,
        input integer class_index
    );
        partial = $signed(partials_q[
            (tile_index*OUTPUT_MAX+class_index)*ACC_BITS +: ACC_BITS]);
    endfunction

    assign capture_partials = state_q == STATE_IDLE && i_start &&
        (i_active_tiles != 0) && (i_active_tiles <= TILE_COUNT) &&
        (i_output_count != 0) && (i_output_count <= OUTPUT_MAX);

    // Payload validity is owned by state_q. Keeping this wide data register out
    // of the asynchronous-reset process avoids reset/set inference on payload.
    always_ff @(posedge clk) begin
        if (capture_partials)
            partials_q <= i_partials_flat;
    end

    always_comb begin
        o_ready = state_q == STATE_IDLE;
        o_valid = state_q == STATE_RESULT;
        o_busy = state_q != STATE_IDLE;
        o_class = best_class_q;
        o_score = best_score_q;
        o_meta = meta_q;
        o_version = version_q;
        o_logits_flat = '0;
        for (comb_output_index = 0; comb_output_index < OUTPUT_MAX;
             comb_output_index = comb_output_index + 1)
            o_logits_flat[comb_output_index*ACC_BITS +: ACC_BITS] =
                logits_q[comb_output_index];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            tile_q <= '0;
            output_q <= '0;
            active_tiles_q <= '0;
            output_count_q <= '0;
            best_score_q <= '0;
            best_class_q <= '0;
            meta_q <= '0;
            version_q <= '0;
            o_sticky_error <= 1'b0;
            for (seq_output_index = 0; seq_output_index < OUTPUT_MAX;
                 seq_output_index = seq_output_index + 1)
                logits_q[seq_output_index] <= '0;
        end else begin
            if (i_clear_error)
                o_sticky_error <= 1'b0;
            case (state_q)
                STATE_IDLE: begin
                    if (i_start) begin
                        if ((i_active_tiles == 0) || (i_active_tiles > TILE_COUNT) ||
                            (i_output_count == 0) || (i_output_count > OUTPUT_MAX)) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            active_tiles_q <= i_active_tiles;
                            output_count_q <= i_output_count;
                            meta_q <= i_meta;
                            version_q <= i_version;
                            tile_q <= 0;
                            for (seq_output_index = 0; seq_output_index < OUTPUT_MAX;
                                 seq_output_index = seq_output_index + 1)
                                logits_q[seq_output_index] <= '0;
                            state_q <= STATE_ACCUM;
                        end
                    end
                end
                STATE_ACCUM: begin
                    for (seq_output_index = 0; seq_output_index < OUTPUT_MAX;
                         seq_output_index = seq_output_index + 1)
                        if (seq_output_index < output_count_q)
                            logits_q[seq_output_index] <= logits_q[seq_output_index] +
                                partial(tile_q, seq_output_index);
                    if (tile_q + 1 >= active_tiles_q)
                        state_q <= STATE_ARGMAX_INIT;
                    else
                        tile_q <= tile_q + 1'b1;
                end
                STATE_ARGMAX_INIT: begin
                    best_score_q <= logits_q[0];
                    best_class_q <= 0;
                    output_q <= 1;
                    if (output_count_q == 1)
                        state_q <= STATE_RESULT;
                    else
                        state_q <= STATE_ARGMAX;
                end
                STATE_ARGMAX: begin
                    if ($signed(logits_q[output_q]) > best_score_q) begin
                        best_score_q <= logits_q[output_q];
                        best_class_q <= output_q;
                    end
                    if (output_q + 1 >= output_count_q)
                        state_q <= STATE_RESULT;
                    else
                        output_q <= output_q + 1'b1;
                end
                STATE_RESULT: if (i_ready) state_q <= STATE_IDLE;
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
