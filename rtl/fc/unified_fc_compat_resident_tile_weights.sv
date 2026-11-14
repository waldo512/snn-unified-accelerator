`timescale 1ns/1ps
`default_nettype none

// Resident W0/W2 storage for one H40 FC_COMPAT tile. Cluster numbering is a
// packaging concern; the compute array sees only one row per K8 bank and the
// tile's output-weight matrix.
module unified_fc_compat_resident_tile_weights #(
    parameter integer FI_BITS = 12,
    parameter integer BANKS = 8,
    parameter integer H_PER_TILE = 40,
    parameter integer W0_BITS = 5,
    parameter integer W0_BANK_DEPTH = 289,
    parameter integer O_DIM = 10,
    parameter integer W2_BITS = 8,
    parameter WEIGHT_DIR = "data/weights/p3a_nmnist_h160_k8_input2312",
    parameter CLUSTER_ID = "00"
) (
    input  logic                                      clk,
    input  logic                                      i_enable,
    input  logic [BANKS-1:0]                         i_bank_valid,
    input  logic [BANKS*FI_BITS-1:0]                 i_fi_flat,
    output logic [BANKS*H_PER_TILE*W0_BITS-1:0]      o_w0_rows_flat,
    output logic [H_PER_TILE*O_DIM*W2_BITS-1:0]      o_w2_weight_flat
);
    localparam integer ROW_BITS = H_PER_TILE * W0_BITS;
    localparam integer ADDR_BITS = $clog2(W0_BANK_DEPTH);

    logic [ROW_BITS-1:0] bank_row [0:BANKS-1];
    logic [O_DIM*W2_BITS-1:0] w2_mem [0:H_PER_TILE-1];

`define FC_TILE_W0_ROM(INST, BANK, BANK_ID) \
    unified_fc_compat_w0_bank_rom #( \
        .WORD_BITS(ROW_BITS), \
        .DEPTH(W0_BANK_DEPTH), \
        .INIT_FILE({WEIGHT_DIR, "/w0_cluster_", CLUSTER_ID, \
                    "_bank_", BANK_ID, ".mem"}) \
    ) INST ( \
        .clk(clk), \
        .i_enable(i_enable && i_bank_valid[BANK]), \
        .i_addr(i_fi_flat[BANK*FI_BITS+3 +: ADDR_BITS]), \
        .o_data(bank_row[BANK]) \
    );

    `FC_TILE_W0_ROM(u_bank0, 0, "00")
    `FC_TILE_W0_ROM(u_bank1, 1, "01")
    `FC_TILE_W0_ROM(u_bank2, 2, "02")
    `FC_TILE_W0_ROM(u_bank3, 3, "03")
    `FC_TILE_W0_ROM(u_bank4, 4, "04")
    `FC_TILE_W0_ROM(u_bank5, 5, "05")
    `FC_TILE_W0_ROM(u_bank6, 6, "06")
    `FC_TILE_W0_ROM(u_bank7, 7, "07")
`undef FC_TILE_W0_ROM

    initial begin
        $readmemh({WEIGHT_DIR, "/w2_cluster_", CLUSTER_ID, ".hex"},
                  w2_mem);
    end

    generate
        for (genvar bank_i = 0; bank_i < BANKS; bank_i = bank_i + 1) begin : gen_w0_output
            assign o_w0_rows_flat[bank_i*ROW_BITS +: ROW_BITS] =
                bank_row[bank_i];
        end
        for (genvar hidden_i = 0; hidden_i < H_PER_TILE;
             hidden_i = hidden_i + 1) begin : gen_w2_output
            assign o_w2_weight_flat[
                hidden_i*O_DIM*W2_BITS +: O_DIM*W2_BITS
            ] = w2_mem[hidden_i];
        end
    endgenerate

    initial begin
        if (BANKS != 8 || H_PER_TILE != 40)
            $error("FC_COMPAT resident tile weights require K8 and H40");
    end
endmodule

`default_nettype wire
