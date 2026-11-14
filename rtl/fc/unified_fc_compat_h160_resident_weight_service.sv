`timescale 1ns/1ps
`default_nettype none

// Resident trained weights for one to twelve H40 FC_COMPAT tiles. The request/output
// handshake is a one-entry elastic stage, so all eight independent W0 banks
// can sustain one scheduler group per cycle without exposing ROM latency to
// the tile array.
module unified_fc_compat_h160_resident_weight_service #(
    parameter integer FI_BITS = 12,
    parameter integer TILES = 4,
    parameter integer BANKS = 8,
    parameter integer H_PER_TILE = 40,
    parameter integer W0_BITS = 5,
    parameter integer W0_BANK_DEPTH = 289,
    parameter integer O_DIM = 10,
    parameter integer W2_BITS = 8,
    parameter WEIGHT_DIR = "data/weights/p3a_nmnist_h160_k8_input2312"
) (
    input  logic                                             clk,
    input  logic                                             rst_n,
    input  logic                                             i_clear,
    input  logic [TILES-1:0]                                i_active_tile_mask,
    input  logic                                             i_request_valid,
    output logic                                             o_request_ready,
    input  logic [BANKS-1:0]                                i_bank_valid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [BANKS*FI_BITS-1:0]                        i_fi_flat,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic                                             o_row_valid,
    input  logic                                             i_row_ready,
    output logic [BANKS-1:0]                                o_lane_valid,
    output logic [TILES*BANKS*H_PER_TILE*W0_BITS-1:0]       o_w0_rows_flat,
    output logic [TILES*H_PER_TILE*O_DIM*W2_BITS-1:0]        o_w2_weight_flat,
    output logic                                             o_idle
);
    localparam integer TILE_W0_BITS = BANKS * H_PER_TILE * W0_BITS;
    localparam integer TILE_W2_BITS = H_PER_TILE * O_DIM * W2_BITS;
    logic request_fire;

    assign o_request_ready = !o_row_valid || i_row_ready;
    assign request_fire = i_request_valid && o_request_ready;
    assign o_idle = !o_row_valid;

`define FC_RESIDENT_TILE(INDEX, ID) \
    if (TILES > INDEX) begin : gen_tile_``INDEX \
        unified_fc_compat_resident_tile_weights #( \
            .FI_BITS(FI_BITS), .BANKS(BANKS), \
            .H_PER_TILE(H_PER_TILE), .W0_BITS(W0_BITS), \
            .W0_BANK_DEPTH(W0_BANK_DEPTH), .O_DIM(O_DIM), \
            .W2_BITS(W2_BITS), .WEIGHT_DIR(WEIGHT_DIR), \
            .CLUSTER_ID(ID) \
        ) u_weights ( \
            .clk(clk), \
            .i_enable(request_fire && i_active_tile_mask[INDEX]), \
            .i_bank_valid(i_bank_valid), \
            .i_fi_flat(i_fi_flat), \
            .o_w0_rows_flat(o_w0_rows_flat[ \
                INDEX*TILE_W0_BITS +: TILE_W0_BITS]), \
            .o_w2_weight_flat(o_w2_weight_flat[ \
                INDEX*TILE_W2_BITS +: TILE_W2_BITS]) \
        ); \
    end

    generate
        `FC_RESIDENT_TILE(0, "00")
        `FC_RESIDENT_TILE(1, "01")
        `FC_RESIDENT_TILE(2, "02")
        `FC_RESIDENT_TILE(3, "03")
        `FC_RESIDENT_TILE(4, "04")
        `FC_RESIDENT_TILE(5, "05")
        `FC_RESIDENT_TILE(6, "06")
        `FC_RESIDENT_TILE(7, "07")
        `FC_RESIDENT_TILE(8, "08")
        `FC_RESIDENT_TILE(9, "09")
        `FC_RESIDENT_TILE(10, "10")
        `FC_RESIDENT_TILE(11, "11")
    endgenerate
`undef FC_RESIDENT_TILE

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_row_valid <= 1'b0;
            o_lane_valid <= '0;
        end else if (i_clear) begin
            o_row_valid <= 1'b0;
            o_lane_valid <= '0;
        end else if (o_request_ready) begin
            o_row_valid <= i_request_valid;
            if (request_fire)
                o_lane_valid <= i_bank_valid;
        end
    end

    initial begin
        if (TILES < 1 || TILES > 12 || BANKS != 8 || H_PER_TILE != 40)
            $error("FC_COMPAT resident weights require 1..12 H40 tiles and K8");
    end
endmodule

`default_nettype wire
