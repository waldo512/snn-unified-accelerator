`timescale 1ns/1ps
`default_nettype none

// Two 200x1156 true-dual-port banks form one tile-local, topology-readdressed
// weight store. FC packs two 20xINT5 half-rows per word; SRNN uses one
// 20xINT8 row per bank; SCNN packs twelve INT16 kernel weights per word and
// uses the inactive bank for prefetch.
module unified_dual_role_weight_store_candidate_v1 #(
    parameter int LANES = 40,
    parameter int BANK_WIDTH = 200,
    parameter int BANK_DEPTH = 1156,
    parameter int ADDR_WIDTH = $clog2(BANK_DEPTH)
) (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_fill_valid,
    output logic                         o_fill_ready,
    input  wire logic                    i_fill_bank,
    input  wire logic [ADDR_WIDTH-1:0]   i_fill_addr,
    input  wire logic [BANK_WIDTH-1:0]   i_fill_data,

    input  wire logic                    i_read_valid,
    output logic                         o_read_ready,
    input  wire logic [1:0]              i_mode,
    input  wire logic                    i_scnn_active_bank,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr0,
    input  wire logic [ADDR_WIDTH-1:0]   i_read_addr1,
    input  wire logic [3:0]              i_read_subword0,
    input  wire logic [3:0]              i_read_subword1,
    output logic                         o_read_result_valid,
    input  wire logic                    i_read_result_ready,
    output logic [LANES*25-1:0]          o_term0_flat,
    output logic [LANES*25-1:0]          o_term1_flat
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    logic [BANK_WIDTH-1:0] bank_read_data [0:1][0:1];
    logic [1:0] mode_q;
    logic scnn_active_bank_q;
    logic [3:0] read_subword0_q;
    logic [3:0] read_subword1_q;
    logic read_result_valid_q;
    logic read_fire;
    logic fill_fire;
    logic [1:0] read_bank_enable;
    logic fill_conflict;

    assign read_bank_enable = i_mode == MODE_SCNN ?
        (i_scnn_active_bank ? 2'b10 : 2'b01) : 2'b11;
    assign fill_conflict = i_read_valid &&
                           read_bank_enable[i_fill_bank];
    assign o_read_ready = !read_result_valid_q || i_read_result_ready;
    assign o_fill_ready = !fill_conflict;
    assign read_fire = i_read_valid && o_read_ready;
    assign fill_fire = i_fill_valid && o_fill_ready;
    assign o_read_result_valid = read_result_valid_q;

    generate
        for (genvar bank = 0; bank < 2; bank++) begin : g_bank
            logic port_a_write;
            logic port_a_read;
            logic port_b_read;

            assign port_a_write = fill_fire && i_fill_bank == bank;
            assign port_a_read = read_fire && read_bank_enable[bank];
            assign port_b_read = read_fire && read_bank_enable[bank];

`ifdef UNIFIED_USE_XPM_MEMORY
            logic [ADDR_WIDTH-1:0] port_a_addr;

            assign port_a_addr = port_a_write ? i_fill_addr : i_read_addr0;

            xpm_memory_tdpram #(
                .MEMORY_SIZE(BANK_DEPTH * BANK_WIDTH),
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
                .WRITE_DATA_WIDTH_A(BANK_WIDTH),
                .READ_DATA_WIDTH_A(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_A(BANK_WIDTH),
                .ADDR_WIDTH_A(ADDR_WIDTH),
                .READ_RESET_VALUE_A("0"),
                .READ_LATENCY_A(1),
                .WRITE_MODE_A("read_first"),
                .RST_MODE_A("SYNC"),
                .WRITE_DATA_WIDTH_B(BANK_WIDTH),
                .READ_DATA_WIDTH_B(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_B(BANK_WIDTH),
                .ADDR_WIDTH_B(ADDR_WIDTH),
                .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1),
                .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_weight_memory (
                .sleep(1'b0),
                .clka(i_clk), .rsta(1'b0),
                .ena(port_a_write || port_a_read), .regcea(1'b1),
                .wea(port_a_write), .addra(port_a_addr),
                .dina(i_fill_data),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .douta(bank_read_data[bank][0]), .sbiterra(), .dbiterra(),
                .clkb(i_clk), .rstb(1'b0),
                .enb(port_b_read), .regceb(1'b1), .web(1'b0),
                .addrb(i_read_addr1), .dinb('0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .doutb(bank_read_data[bank][1]), .sbiterrb(), .dbiterrb()
            );
`else
            logic [BANK_WIDTH-1:0] memory [0:BANK_DEPTH-1];

            always_ff @(posedge i_clk) begin
                if (port_a_write)
                    memory[i_fill_addr] <= i_fill_data;
                if (port_a_read)
                    bank_read_data[bank][0] <= memory[i_read_addr0];
                if (port_b_read)
                    bank_read_data[bank][1] <= memory[i_read_addr1];
            end
`endif
        end
    endgenerate

    always_comb begin
        o_term0_flat = '0;
        o_term1_flat = '0;
        for (integer lane = 0; lane < LANES; lane++) begin
            if (mode_q == MODE_FC) begin
                if (lane < 20) begin
                    o_term0_flat[lane*25 +: 25] =
                        {{20{bank_read_data[0][0][
                            read_subword0_q[0]*100 + lane*5 + 4]}},
                         bank_read_data[0][0][
                            read_subword0_q[0]*100 + lane*5 +: 5]};
                    o_term1_flat[lane*25 +: 25] =
                        {{20{bank_read_data[0][1][
                            read_subword1_q[0]*100 + lane*5 + 4]}},
                         bank_read_data[0][1][
                            read_subword1_q[0]*100 + lane*5 +: 5]};
                end else begin
                    o_term0_flat[lane*25 +: 25] =
                        {{20{bank_read_data[1][0][
                            read_subword0_q[0]*100 + (lane-20)*5 + 4]}},
                         bank_read_data[1][0][
                            read_subword0_q[0]*100 + (lane-20)*5 +: 5]};
                    o_term1_flat[lane*25 +: 25] =
                        {{20{bank_read_data[1][1][
                            read_subword1_q[0]*100 + (lane-20)*5 + 4]}},
                         bank_read_data[1][1][
                            read_subword1_q[0]*100 + (lane-20)*5 +: 5]};
                end
            end else if (mode_q == MODE_SRNN) begin
                if (lane < 20) begin
                    o_term0_flat[lane*25 +: 25] =
                        {{17{bank_read_data[0][0][lane*8 + 7]}},
                         bank_read_data[0][0][lane*8 +: 8]};
                    o_term1_flat[lane*25 +: 25] =
                        {{17{bank_read_data[0][1][lane*8 + 7]}},
                         bank_read_data[0][1][lane*8 +: 8]};
                end else begin
                    o_term0_flat[lane*25 +: 25] =
                        {{17{bank_read_data[1][0][(lane-20)*8 + 7]}},
                         bank_read_data[1][0][(lane-20)*8 +: 8]};
                    o_term1_flat[lane*25 +: 25] =
                        {{17{bank_read_data[1][1][(lane-20)*8 + 7]}},
                         bank_read_data[1][1][(lane-20)*8 +: 8]};
                end
            end else if (mode_q == MODE_SCNN) begin
                o_term0_flat[lane*25 +: 25] =
                    {{9{bank_read_data[scnn_active_bank_q][0][
                        read_subword0_q*16 + 15]}},
                     bank_read_data[scnn_active_bank_q][0][
                        read_subword0_q*16 +: 16]};
                o_term1_flat[lane*25 +: 25] =
                    {{9{bank_read_data[scnn_active_bank_q][1][
                        read_subword1_q*16 + 15]}},
                     bank_read_data[scnn_active_bank_q][1][
                        read_subword1_q*16 +: 16]};
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            mode_q <= MODE_FC;
            scnn_active_bank_q <= 1'b0;
            read_subword0_q <= '0;
            read_subword1_q <= '0;
            read_result_valid_q <= 1'b0;
        end else begin
            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire) begin
                mode_q <= i_mode;
                scnn_active_bank_q <= i_scnn_active_bank;
                read_subword0_q <= i_read_subword0;
                read_subword1_q <= i_read_subword1;
                read_result_valid_q <= 1'b1;
            end
        end
    end

    initial begin
        if (LANES != 40 || BANK_WIDTH != 200 || BANK_DEPTH != 1156)
            $error("weight-store candidate requires the fixed 2x200x1156 geometry");
    end

endmodule

`default_nettype wire
