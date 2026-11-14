`timescale 1ns/1ps
`default_nettype none

module unified_k8_weight_memory_2312_ooc (
    input  wire logic                 clk,
    input  wire logic                 i_write_enable,
    input  wire logic [3:0]           i_write_bank,
    input  wire logic [11:0]          i_write_addr,
    input  wire logic [63:0]          i_write_data,
    input  wire logic [11:0]          i_read_enable,
    input  wire logic [12*12-1:0]     i_read_addr_flat,
    output logic [12*64-1:0]          o_read_data_flat
);
    unified_k8_weight_memory_ooc #(
        .BANKS(12), .DEPTH(2312), .ADDRESS_WIDTH(12)
    ) u_memory (
        .clk(clk), .i_write_enable(i_write_enable),
        .i_write_bank(i_write_bank), .i_write_addr(i_write_addr),
        .i_write_data(i_write_data), .i_read_enable(i_read_enable),
        .i_read_addr_flat(i_read_addr_flat),
        .o_read_data_flat(o_read_data_flat)
    );
endmodule

`default_nettype wire
