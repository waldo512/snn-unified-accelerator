`timescale 1ns/1ps
`default_nettype none

// Two ping-pong binary feature slots. Each slot is one true-dual-port RAM:
// an active slot supplies two independent H40 words per cycle, while the
// inactive slot may be filled concurrently through port A.
module unified_dual_role_binary_feature_tdp_candidate_v1 #(
    parameter int LANES = 40,
    parameter int WORDS_PER_SLOT = 1792,
    parameter int ADDR_WIDTH = $clog2(WORDS_PER_SLOT)
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_write_valid,
    output logic                         o_write_ready,
    input  wire logic                    i_write_slot,
    input  wire logic [ADDR_WIDTH-1:0]   i_write_addr,
    input  wire logic [LANES-1:0]        i_write_data,

    input  wire logic                    i_read_valid,
    output logic                         o_read_ready,
    input  wire logic                    i_read_slot,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr0,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr1,
    output logic                         o_read_result_valid,
    input  wire logic                    i_read_result_ready,
    output logic [LANES-1:0]             o_read_data0,
    output logic [LANES-1:0]             o_read_data1
);
    logic read_result_valid_q;
    logic response_slot_q;
    logic read_fire;
    logic write_fire;
    logic same_slot_request;
    logic [LANES-1:0] memory_data_a [0:1];
    logic [LANES-1:0] memory_data_b [0:1];

    assign same_slot_request = i_read_valid && i_write_valid &&
                               i_read_slot == i_write_slot;
    assign o_read_ready = !read_result_valid_q || i_read_result_ready;
    assign o_write_ready = !same_slot_request;
    assign read_fire = i_read_valid && o_read_ready;
    assign write_fire = i_write_valid && o_write_ready;
    assign o_read_result_valid = read_result_valid_q;
    assign o_read_data0 = memory_data_a[response_slot_q];
    assign o_read_data1 = memory_data_b[response_slot_q];

    generate
        for (genvar slot = 0; slot < 2; slot++) begin : g_slot
            logic port_a_write;
            logic port_a_read;
            logic port_b_read;

            assign port_a_write = write_fire && i_write_slot == slot;
            assign port_a_read = read_fire && i_read_slot == slot;
            assign port_b_read = read_fire && i_read_slot == slot;

`ifdef UNIFIED_USE_XPM_MEMORY
            logic [ADDR_WIDTH-1:0] port_a_addr;

            assign port_a_addr = port_a_write ? i_write_addr : i_read_addr0;

            xpm_memory_tdpram #(
                .MEMORY_SIZE(WORDS_PER_SLOT * LANES),
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
                .WRITE_DATA_WIDTH_A(LANES),
                .READ_DATA_WIDTH_A(LANES),
                .BYTE_WRITE_WIDTH_A(LANES),
                .ADDR_WIDTH_A(ADDR_WIDTH),
                .READ_RESET_VALUE_A("0"),
                .READ_LATENCY_A(1),
                .WRITE_MODE_A("read_first"),
                .RST_MODE_A("SYNC"),
                .WRITE_DATA_WIDTH_B(LANES),
                .READ_DATA_WIDTH_B(LANES),
                .BYTE_WRITE_WIDTH_B(LANES),
                .ADDR_WIDTH_B(ADDR_WIDTH),
                .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1),
                .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_feature_memory (
                .sleep(1'b0),
                .clka(i_clk), .rsta(1'b0),
                .ena(port_a_write || port_a_read), .regcea(1'b1),
                .wea(port_a_write), .addra(port_a_addr),
                .dina(i_write_data),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .douta(memory_data_a[slot]), .sbiterra(), .dbiterra(),
                .clkb(i_clk), .rstb(1'b0),
                .enb(port_b_read), .regceb(1'b1), .web(1'b0),
                .addrb(i_read_addr1), .dinb('0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .doutb(memory_data_b[slot]), .sbiterrb(), .dbiterrb()
            );
`else
            logic [LANES-1:0] memory [0:WORDS_PER_SLOT-1];

            always_ff @(posedge i_clk) begin
                if (port_a_write)
                    memory[i_write_addr] <= i_write_data;
                if (port_a_read)
                    memory_data_a[slot] <= memory[i_read_addr0];
                if (port_b_read)
                    memory_data_b[slot] <= memory[i_read_addr1];
            end
`endif
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            read_result_valid_q <= 1'b0;
            response_slot_q <= 1'b0;
        end else begin
            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire) begin
                read_result_valid_q <= 1'b1;
                response_slot_q <= i_read_slot;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("binary feature candidate requires 40-bit H40 words");
    end

endmodule

`default_nettype wire
