`timescale 1ns/1ps
`default_nettype none

// One tag per physical tile bank describes the currently resident execution
// workset. Refill completion installs a tag; parameter commit invalidates only
// affected banks. Soft transaction abort is intentionally absent because
// completed cache contents remain reusable and partial refills never install.
module unified_array16_weight_cache_residency_v1 #(
    parameter int TAG_WIDTH = 32
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_fill_complete_valid,
    output logic                         o_fill_complete_ready,
    input  wire logic [3:0]              i_fill_complete_tile,
    input  wire logic [1:0]              i_fill_complete_bank,
    input  wire logic [TAG_WIDTH-1:0]    i_fill_complete_tag,

    input  wire logic                    i_invalidate_valid,
    output logic                         o_invalidate_ready,
    input  wire logic [3:0]              i_invalidate_tile,
    input  wire logic [3:0]              i_invalidate_bank_mask,
    input  wire logic                    i_flush_all,

    input  wire logic [3:0]              i_query_tile,
    input  wire logic [3:0]              i_query_bank_mask,
    input  wire logic [TAG_WIDTH-1:0]    i_query_tag,
    output logic                         o_query_hit,
    output logic [63:0]                  o_bank_valid_flat
);
    logic [3:0] bank_valid_q [0:15];
    logic [TAG_WIDTH-1:0] bank_tag_q [0:15][0:3];

    assign o_invalidate_ready = 1'b1;
    assign o_fill_complete_ready = !i_flush_all && !i_invalidate_valid;

    always_comb begin
        o_query_hit = i_query_bank_mask != 0;
        o_bank_valid_flat = '0;
        for (int tile = 0; tile < 16; tile++)
            o_bank_valid_flat[tile*4 +: 4] = bank_valid_q[tile];
        for (int bank = 0; bank < 4; bank++) begin
            if (i_query_bank_mask[bank] &&
                (!bank_valid_q[i_query_tile][bank] ||
                 bank_tag_q[i_query_tile][bank] != i_query_tag))
                o_query_hit = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            for (int tile = 0; tile < 16; tile++)
                bank_valid_q[tile] <= '0;
        end else begin
            if (i_fill_complete_valid && o_fill_complete_ready) begin
                bank_valid_q[i_fill_complete_tile][
                    i_fill_complete_bank] <= 1'b1;
                bank_tag_q[i_fill_complete_tile][
                    i_fill_complete_bank] <= i_fill_complete_tag;
            end
            if (i_invalidate_valid && o_invalidate_ready)
                bank_valid_q[i_invalidate_tile] <=
                    bank_valid_q[i_invalidate_tile] &
                    ~i_invalidate_bank_mask;
            if (i_flush_all) begin
                for (int tile = 0; tile < 16; tile++)
                    bank_valid_q[tile] <= '0;
            end
        end
    end
endmodule

`default_nettype wire
