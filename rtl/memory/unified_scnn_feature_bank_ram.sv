`default_nettype none

module unified_scnn_feature_bank_ram #(
    parameter int BANK_DEPTH = 9216,
    parameter int ADDR_WIDTH = $clog2(BANK_DEPTH)
) (
    input  logic                  i_clk,
    input  logic                  i_write_enable,
    input  logic [ADDR_WIDTH-1:0] i_write_addr,
    input  logic [15:0]           i_write_data,
    input  logic                  i_read_enable,
    input  logic [ADDR_WIDTH-1:0] i_read_addr,
    output logic [15:0]           o_read_data
);

`ifdef UNIFIED_USE_XPM_MEMORY
    logic [15:0] xpm_read_data;

    assign o_read_data = xpm_read_data;

    xpm_memory_sdpram #(
        .MEMORY_SIZE(BANK_DEPTH * 16),
        .MEMORY_PRIMITIVE("block"),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM(""),
        .USE_MEM_INIT(0),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep"),
        .AUTO_SLEEP_TIME(0),
        .MESSAGE_CONTROL(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .MEMORY_OPTIMIZATION("true"),
        .CASCADE_HEIGHT(0),
        .SIM_ASSERT_CHK(0),
        .WRITE_PROTECT(1),
        .IGNORE_INIT_SYNTH(0),
        .WRITE_DATA_WIDTH_A(16),
        .BYTE_WRITE_WIDTH_A(16),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .RST_MODE_A("SYNC"),
        .READ_DATA_WIDTH_B(16),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_feature_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .ena(i_write_enable),
        .wea(i_write_enable),
        .addra(i_write_addr),
        .dina(i_write_data),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .clkb(i_clk),
        .rstb(1'b0),
        .enb(i_read_enable),
        .regceb(1'b1),
        .addrb(i_read_addr),
        .doutb(xpm_read_data),
        .sbiterrb(),
        .dbiterrb()
    );
`else
    (* ram_style = "block" *) logic [15:0] memory [0:BANK_DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (i_write_enable)
            memory[i_write_addr] <= i_write_data;
        if (i_read_enable)
            o_read_data <= memory[i_read_addr];
    end
`endif

endmodule

`default_nettype wire
