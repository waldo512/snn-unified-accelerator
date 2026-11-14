`timescale 1ns/1ps
`default_nettype none

// Serially checks one shared residency-table query port before publishing a
// new execution context. This keeps version comparison off the tile issue
// paths while making stale or partially refilled weights unlaunchable.
module unified_array16_weight_workset_activate_v1 #(
    parameter int TAG_WIDTH = 32
) (
    input  wire logic                     i_clk,
    input  wire logic                     i_rst_n,
    input  wire logic                     i_abort,
    input  wire logic                     i_clear_error,

    input  wire logic                     i_request_valid,
    output logic                          o_request_ready,
    input  wire logic [1:0]               i_request_mode,
    input  wire logic [15:0]              i_request_tile_mask,
    input  wire logic                     i_request_scnn_active_bank,
    input  wire logic [TAG_WIDTH-1:0]     i_request_tag,

    output logic [3:0]                    o_query_tile,
    output logic [3:0]                    o_query_bank_mask,
    output logic [TAG_WIDTH-1:0]          o_query_tag,
    input  wire logic                     i_query_hit,

    output logic                          o_miss_valid,
    input  wire logic                     i_miss_ready,
    output logic [3:0]                    o_miss_tile,
    output logic [3:0]                    o_miss_bank_mask,
    output logic [TAG_WIDTH-1:0]          o_miss_tag,

    input  wire logic                     i_array_idle,
    output logic                          o_activate_valid,
    input  wire logic                     i_activate_ready,
    output logic [1:0]                    o_activate_mode,
    output logic [15:0]                   o_activate_tile_mask,
    output logic                          o_activate_scnn_active_bank,
    output logic [TAG_WIDTH-1:0]          o_activate_tag,

    output logic                          o_busy,
    output logic                          o_waiting_for_idle,
    output logic                          o_sticky_error
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_SCAN,
        STATE_WAIT_IDLE,
        STATE_MISS
    } state_t;

    state_t state_q;
    logic [1:0] mode_q;
    logic [15:0] tile_mask_q;
    logic [15:0] remaining_mask_q;
    logic scnn_active_bank_q;
    logic [TAG_WIDTH-1:0] tag_q;
    logic [3:0] required_bank_mask_q;
    logic [3:0] selected_tile;
    logic [15:0] selected_onehot;
    logic selected_valid;
    logic request_fire;
    logic request_legal;
    logic scan_last;

    always_comb begin
        selected_tile = '0;
        selected_onehot = '0;
        selected_valid = 1'b0;
        for (int tile = 0; tile < 16; tile++) begin
            if (!selected_valid && remaining_mask_q[tile]) begin
                selected_tile = 4'(tile);
                selected_onehot[tile] = 1'b1;
                selected_valid = 1'b1;
            end
        end
    end

    assign request_legal = i_request_tile_mask != 0 &&
        (i_request_mode == MODE_FC || i_request_mode == MODE_SRNN ||
         i_request_mode == MODE_SCNN);
    assign o_request_ready = !i_abort && state_q == STATE_IDLE &&
        !o_sticky_error;
    assign request_fire = i_request_valid && o_request_ready;
    assign scan_last = selected_valid &&
        (remaining_mask_q & ~selected_onehot) == 0;

    assign o_query_tile = selected_tile;
    assign o_query_bank_mask = required_bank_mask_q;
    assign o_query_tag = tag_q;

    assign o_miss_valid = !i_abort && state_q == STATE_MISS;
    assign o_miss_tile = selected_tile;
    assign o_miss_bank_mask = required_bank_mask_q;
    assign o_miss_tag = tag_q;

    assign o_activate_valid = !i_abort && state_q == STATE_WAIT_IDLE &&
        i_array_idle;
    assign o_activate_mode = mode_q;
    assign o_activate_tile_mask = tile_mask_q;
    assign o_activate_scnn_active_bank = scnn_active_bank_q;
    assign o_activate_tag = tag_q;
    assign o_busy = state_q != STATE_IDLE;
    assign o_waiting_for_idle = state_q == STATE_WAIT_IDLE && !i_array_idle;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            mode_q <= MODE_FC;
            tile_mask_q <= '0;
            remaining_mask_q <= '0;
            scnn_active_bank_q <= 1'b0;
            tag_q <= '0;
            required_bank_mask_q <= '0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            remaining_mask_q <= '0;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (request_fire) begin
                        if (!request_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            mode_q <= i_request_mode;
                            tile_mask_q <= i_request_tile_mask;
                            remaining_mask_q <= i_request_tile_mask;
                            scnn_active_bank_q <=
                                i_request_scnn_active_bank;
                            tag_q <= i_request_tag;
                            if (i_request_mode == MODE_SCNN)
                                required_bank_mask_q <=
                                    i_request_scnn_active_bank ?
                                    4'b1100 : 4'b0011;
                            else
                                required_bank_mask_q <= 4'b1111;
                            state_q <= STATE_SCAN;
                        end
                    end
                end

                STATE_SCAN: begin
                    if (!selected_valid) begin
                        o_sticky_error <= 1'b1;
                        state_q <= STATE_IDLE;
                    end else if (!i_query_hit) begin
                        state_q <= STATE_MISS;
                    end else if (scan_last) begin
                        remaining_mask_q <= '0;
                        state_q <= STATE_WAIT_IDLE;
                    end else begin
                        remaining_mask_q <=
                            remaining_mask_q & ~selected_onehot;
                    end
                end

                STATE_WAIT_IDLE: begin
                    if (o_activate_valid && i_activate_ready)
                        state_q <= STATE_IDLE;
                end

                STATE_MISS: begin
                    if (o_miss_valid && i_miss_ready)
                        state_q <= STATE_IDLE;
                end

                default: begin
                    state_q <= STATE_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
