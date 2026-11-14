`timescale 1ns/1ps
`default_nettype none

// One mode-neutral tile state working set. SCNN stores eight Q8.8 states per
// address. FC/SRNN store the low and high 16-bit halves of eight 32-bit states
// at consecutive addresses. Sequence-zero substitution and address traversal
// remain scheduler responsibilities, so the RAM itself never requires reset.
module unified_dual_role_state_store_candidate_v2 #(
    parameter int LANES = 8,
    parameter int SLICE_WIDTH = 16,
    parameter int DEPTH = 5040,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,

    input  wire logic                         i_write_valid,
    output logic                              o_write_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_write_addr,
    input  wire logic [LANES*SLICE_WIDTH-1:0] i_write_data,

    input  wire logic                         i_read_valid,
    output logic                              o_read_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_read_addr,
    output logic                              o_read_result_valid,
    input  wire logic                         i_read_result_ready,
    output logic [ADDR_WIDTH-1:0]             o_read_result_addr,
    output logic [LANES*SLICE_WIDTH-1:0]      o_read_data
);
    localparam int WORD_WIDTH = LANES * SLICE_WIDTH;

    logic read_result_valid_q;
    logic [ADDR_WIDTH-1:0] read_result_addr_q;
    logic [WORD_WIDTH-1:0] memory_read_data;
    logic read_fire;
    logic write_fire;

    assign o_write_ready = !i_abort;
    assign write_fire = i_write_valid && o_write_ready;
    assign o_read_ready = !i_abort &&
        (!read_result_valid_q || i_read_result_ready);
    assign read_fire = i_read_valid && o_read_ready;
    assign o_read_result_valid = !i_abort && read_result_valid_q;
    assign o_read_result_addr = read_result_addr_q;
    assign o_read_data = memory_read_data;

`ifdef UNIFIED_USE_XPM_MEMORY
    xpm_memory_sdpram #(
        .MEMORY_SIZE(DEPTH * WORD_WIDTH),
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
        .WRITE_DATA_WIDTH_A(WORD_WIDTH),
        .BYTE_WRITE_WIDTH_A(WORD_WIDTH),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .RST_MODE_A("SYNC"),
        .READ_DATA_WIDTH_B(WORD_WIDTH),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_state_memory (
        .sleep(1'b0),
        .clka(i_clk), .ena(write_fire), .wea(write_fire),
        .addra(i_write_addr), .dina(i_write_data),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .clkb(i_clk), .rstb(1'b0), .enb(read_fire), .regceb(1'b1),
        .addrb(i_read_addr), .doutb(memory_read_data),
        .sbiterrb(), .dbiterrb()
    );
`else
    logic [WORD_WIDTH-1:0] memory [0:DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (write_fire)
            memory[i_write_addr] <= i_write_data;
        if (read_fire)
            memory_read_data <= memory[i_read_addr];
    end
`endif

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            read_result_valid_q <= 1'b0;
            read_result_addr_q <= '0;
        end else begin
            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire) begin
                read_result_valid_q <= 1'b1;
                read_result_addr_q <= i_read_addr;
            end
        end
    end

    initial begin
        if (LANES != 8 || SLICE_WIDTH != 16 || DEPTH != 5040)
            $error("state-store candidate requires the fixed 128x5040 geometry");
    end
endmodule

`default_nettype wire
