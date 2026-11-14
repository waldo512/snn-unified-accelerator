`timescale 1ns/1ps
`default_nettype none

module unified_k8_weight_memory_ooc #(
    parameter integer BANKS = 12,
    parameter integer DEPTH = 4096,
    parameter integer ADDRESS_WIDTH = 12
) (
    input  wire logic                              clk,
    input  wire logic                              i_write_enable,
    input  wire logic [3:0]                        i_write_bank,
    input  wire logic [ADDRESS_WIDTH-1:0]          i_write_addr,
    input  wire logic [63:0]                       i_write_data,
    input  wire logic [BANKS-1:0]                  i_read_enable,
    input  wire logic [BANKS*ADDRESS_WIDTH-1:0]    i_read_addr_flat,
    output logic [BANKS*64-1:0]                    o_read_data_flat
);
    genvar bank;
    generate
        for (bank = 0; bank < BANKS; bank = bank + 1) begin : g_bank
            (* ram_style = "block" *) logic [63:0] memory [0:DEPTH-1];
            always_ff @(posedge clk) begin
                if (i_write_enable && (i_write_bank == bank))
                    memory[i_write_addr] <= i_write_data;
                if (i_read_enable[bank])
                    o_read_data_flat[bank*64 +: 64] <=
                        memory[i_read_addr_flat[bank*ADDRESS_WIDTH +: ADDRESS_WIDTH]];
            end
        end
    endgenerate
endmodule

`default_nettype wire
