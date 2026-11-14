`timescale 1ns/1ps
`default_nettype none

// One synchronous W0 bank. The FPGA implementation uses an explicit block
// ROM while Verilator uses the same initialization file and one-cycle read.
module unified_fc_compat_w0_bank_rom #(
    parameter integer WORD_BITS = 200,
    parameter integer DEPTH = 289,
    parameter INIT_FILE = ""
) (
    input  logic                         clk,
    input  logic                         i_enable,
    input  logic [$clog2(DEPTH)-1:0]     i_addr,
    output logic [WORD_BITS-1:0]         o_data
);
    localparam integer ADDR_BITS = $clog2(DEPTH);

`ifdef SYNTHESIS
    xpm_memory_sprom #(
        .MEMORY_SIZE(DEPTH * WORD_BITS),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_INIT_FILE(INIT_FILE),
        .USE_MEM_INIT(1),
        .MEMORY_OPTIMIZATION("true"),
        .READ_DATA_WIDTH_A(WORD_BITS),
        .ADDR_WIDTH_A(ADDR_BITS),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC")
    ) u_rom (
        .sleep(1'b0),
        .clka(clk),
        .rsta(1'b0),
        .ena(i_enable),
        .regcea(1'b1),
        .addra(i_addr),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .douta(o_data),
        .sbiterra(),
        .dbiterra()
    );
`else
    (* rom_style = "block" *) logic [WORD_BITS-1:0] mem [0:DEPTH-1];

    initial begin
        o_data = '0;
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    always_ff @(posedge clk) begin
        if (i_enable)
            o_data <= mem[i_addr];
    end
`endif

    initial begin
        if ((1 << ADDR_BITS) < DEPTH)
            $error("FC_COMPAT W0 ROM address width is too small");
    end
endmodule

`default_nettype wire
