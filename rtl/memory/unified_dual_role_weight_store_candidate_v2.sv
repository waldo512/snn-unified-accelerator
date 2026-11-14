`timescale 1ns/1ps
`default_nettype none

// Geometry-aware version of the topology-readdressed tile weight store.
// Splitting 1156 rows into 1024+132 avoids paying for a second 1024-row BRAM
// bank for the short FC tail while preserving the v1 external protocol.
module unified_dual_role_weight_store_candidate_v2 #(
    parameter int LANES = 40,
    parameter int BANK_WIDTH = 200,
    parameter int MAIN_DEPTH = 1024,
    parameter int TAIL_DEPTH = 132,
    parameter int BANK_DEPTH = MAIN_DEPTH + TAIL_DEPTH,
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
    localparam int MAIN_ADDR_WIDTH = $clog2(MAIN_DEPTH);
    localparam int TAIL_ADDR_WIDTH = $clog2(TAIL_DEPTH);

    logic [BANK_WIDTH-1:0] main_read_data [0:1][0:1];
    logic [BANK_WIDTH-1:0] tail_read_data [0:1][0:1];
    logic [BANK_WIDTH-1:0] bank_read_data [0:1][0:1];
    logic [1:0] mode_q;
    logic scnn_active_bank_q;
    logic [3:0] read_subword0_q;
    logic [3:0] read_subword1_q;
    logic [1:0] read_port_tail_q;
    logic read_result_valid_q;
    logic read_fire;
    logic fill_fire;
    logic [1:0] read_bank_enable;
    logic fill_conflict;
    logic fill_targets_tail;
    logic [1:0] read_targets_tail;

    assign read_bank_enable = i_mode == MODE_SCNN ?
        (i_scnn_active_bank ? 2'b10 : 2'b01) : 2'b11;
    assign fill_conflict = i_read_valid && read_bank_enable[i_fill_bank];
    assign o_read_ready = !read_result_valid_q || i_read_result_ready;
    assign o_fill_ready = !fill_conflict;
    assign read_fire = i_read_valid && o_read_ready;
    assign fill_fire = i_fill_valid && o_fill_ready;
    assign o_read_result_valid = read_result_valid_q;
    assign fill_targets_tail = i_fill_addr >= ADDR_WIDTH'(MAIN_DEPTH);
    assign read_targets_tail[0] =
        i_read_addr0 >= ADDR_WIDTH'(MAIN_DEPTH);
    assign read_targets_tail[1] =
        i_read_addr1 >= ADDR_WIDTH'(MAIN_DEPTH);

    generate
        for (genvar bank = 0; bank < 2; bank++) begin : g_bank
            logic main_port_a_write;
            logic main_port_a_read;
            logic main_port_b_read;
            logic tail_port_a_write;
            logic tail_port_a_read;
            logic tail_port_b_read;

            assign main_port_a_write = fill_fire && i_fill_bank == bank &&
                !fill_targets_tail;
            assign main_port_a_read = read_fire && read_bank_enable[bank] &&
                !read_targets_tail[0];
            assign main_port_b_read = read_fire && read_bank_enable[bank] &&
                !read_targets_tail[1];
            assign tail_port_a_write = fill_fire && i_fill_bank == bank &&
                fill_targets_tail;
            assign tail_port_a_read = read_fire && read_bank_enable[bank] &&
                read_targets_tail[0];
            assign tail_port_b_read = read_fire && read_bank_enable[bank] &&
                read_targets_tail[1];

`ifdef UNIFIED_USE_XPM_MEMORY
            logic [MAIN_ADDR_WIDTH-1:0] main_port_a_addr;
            logic [TAIL_ADDR_WIDTH-1:0] tail_port_a_addr;

            assign main_port_a_addr = main_port_a_write ?
                i_fill_addr[MAIN_ADDR_WIDTH-1:0] :
                i_read_addr0[MAIN_ADDR_WIDTH-1:0];
            assign tail_port_a_addr = tail_port_a_write ?
                i_fill_addr[TAIL_ADDR_WIDTH-1:0] :
                i_read_addr0[TAIL_ADDR_WIDTH-1:0];

            xpm_memory_tdpram #(
                .MEMORY_SIZE(MAIN_DEPTH * BANK_WIDTH),
                .MEMORY_PRIMITIVE("block"), .CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"), .MEMORY_INIT_FILE("none"),
                .MEMORY_INIT_PARAM(""), .USE_MEM_INIT(0),
                .USE_MEM_INIT_MMI(0), .WAKEUP_TIME("disable_sleep"),
                .AUTO_SLEEP_TIME(0), .MESSAGE_CONTROL(0),
                .USE_EMBEDDED_CONSTRAINT(0), .MEMORY_OPTIMIZATION("true"),
                .CASCADE_HEIGHT(0), .SIM_ASSERT_CHK(0), .WRITE_PROTECT(1),
                .IGNORE_INIT_SYNTH(0),
                .WRITE_DATA_WIDTH_A(BANK_WIDTH),
                .READ_DATA_WIDTH_A(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_A(BANK_WIDTH),
                .ADDR_WIDTH_A(MAIN_ADDR_WIDTH), .READ_RESET_VALUE_A("0"),
                .READ_LATENCY_A(1), .WRITE_MODE_A("read_first"),
                .RST_MODE_A("SYNC"), .WRITE_DATA_WIDTH_B(BANK_WIDTH),
                .READ_DATA_WIDTH_B(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_B(BANK_WIDTH),
                .ADDR_WIDTH_B(MAIN_ADDR_WIDTH), .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1), .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_main_memory (
                .sleep(1'b0), .clka(i_clk), .rsta(1'b0),
                .ena(main_port_a_write || main_port_a_read), .regcea(1'b1),
                .wea(main_port_a_write), .addra(main_port_a_addr),
                .dina(i_fill_data), .injectsbiterra(1'b0),
                .injectdbiterra(1'b0), .douta(main_read_data[bank][0]),
                .sbiterra(), .dbiterra(), .clkb(i_clk), .rstb(1'b0),
                .enb(main_port_b_read), .regceb(1'b1), .web(1'b0),
                .addrb(i_read_addr1[MAIN_ADDR_WIDTH-1:0]), .dinb('0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .doutb(main_read_data[bank][1]), .sbiterrb(), .dbiterrb()
            );

            xpm_memory_tdpram #(
                .MEMORY_SIZE(TAIL_DEPTH * BANK_WIDTH),
                .MEMORY_PRIMITIVE("block"), .CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"), .MEMORY_INIT_FILE("none"),
                .MEMORY_INIT_PARAM(""), .USE_MEM_INIT(0),
                .USE_MEM_INIT_MMI(0), .WAKEUP_TIME("disable_sleep"),
                .AUTO_SLEEP_TIME(0), .MESSAGE_CONTROL(0),
                .USE_EMBEDDED_CONSTRAINT(0), .MEMORY_OPTIMIZATION("true"),
                .CASCADE_HEIGHT(0), .SIM_ASSERT_CHK(0), .WRITE_PROTECT(1),
                .IGNORE_INIT_SYNTH(0),
                .WRITE_DATA_WIDTH_A(BANK_WIDTH),
                .READ_DATA_WIDTH_A(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_A(BANK_WIDTH),
                .ADDR_WIDTH_A(TAIL_ADDR_WIDTH), .READ_RESET_VALUE_A("0"),
                .READ_LATENCY_A(1), .WRITE_MODE_A("read_first"),
                .RST_MODE_A("SYNC"), .WRITE_DATA_WIDTH_B(BANK_WIDTH),
                .READ_DATA_WIDTH_B(BANK_WIDTH),
                .BYTE_WRITE_WIDTH_B(BANK_WIDTH),
                .ADDR_WIDTH_B(TAIL_ADDR_WIDTH), .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1), .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_tail_memory (
                .sleep(1'b0), .clka(i_clk), .rsta(1'b0),
                .ena(tail_port_a_write || tail_port_a_read), .regcea(1'b1),
                .wea(tail_port_a_write), .addra(tail_port_a_addr),
                .dina(i_fill_data), .injectsbiterra(1'b0),
                .injectdbiterra(1'b0), .douta(tail_read_data[bank][0]),
                .sbiterra(), .dbiterra(), .clkb(i_clk), .rstb(1'b0),
                .enb(tail_port_b_read), .regceb(1'b1), .web(1'b0),
                .addrb(i_read_addr1[TAIL_ADDR_WIDTH-1:0]), .dinb('0),
                .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
                .doutb(tail_read_data[bank][1]), .sbiterrb(), .dbiterrb()
            );
`else
            logic [BANK_WIDTH-1:0] main_memory [0:MAIN_DEPTH-1];
            logic [BANK_WIDTH-1:0] tail_memory [0:TAIL_DEPTH-1];

            always_ff @(posedge i_clk) begin
                if (main_port_a_write)
                    main_memory[i_fill_addr[MAIN_ADDR_WIDTH-1:0]] <= i_fill_data;
                if (main_port_a_read)
                    main_read_data[bank][0] <=
                        main_memory[i_read_addr0[MAIN_ADDR_WIDTH-1:0]];
                if (main_port_b_read)
                    main_read_data[bank][1] <=
                        main_memory[i_read_addr1[MAIN_ADDR_WIDTH-1:0]];
                if (tail_port_a_write)
                    tail_memory[i_fill_addr[TAIL_ADDR_WIDTH-1:0]] <= i_fill_data;
                if (tail_port_a_read)
                    tail_read_data[bank][0] <=
                        tail_memory[i_read_addr0[TAIL_ADDR_WIDTH-1:0]];
                if (tail_port_b_read)
                    tail_read_data[bank][1] <=
                        tail_memory[i_read_addr1[TAIL_ADDR_WIDTH-1:0]];
            end
`endif

            always_comb begin
                bank_read_data[bank][0] = read_port_tail_q[0] ?
                    tail_read_data[bank][0] : main_read_data[bank][0];
                bank_read_data[bank][1] = read_port_tail_q[1] ?
                    tail_read_data[bank][1] : main_read_data[bank][1];
            end
        end
    endgenerate

    always_comb begin
        o_term0_flat = '0;
        o_term1_flat = '0;
        for (integer lane = 0; lane < LANES; lane++) begin
            if (mode_q == MODE_FC) begin
                if (lane < 20) begin
                    o_term0_flat[lane*25 +: 25] =
                        {{20{bank_read_data[0][0][read_subword0_q[0]*100 + lane*5 + 4]}},
                         bank_read_data[0][0][read_subword0_q[0]*100 + lane*5 +: 5]};
                    o_term1_flat[lane*25 +: 25] =
                        {{20{bank_read_data[0][1][read_subword1_q[0]*100 + lane*5 + 4]}},
                         bank_read_data[0][1][read_subword1_q[0]*100 + lane*5 +: 5]};
                end else begin
                    o_term0_flat[lane*25 +: 25] =
                        {{20{bank_read_data[1][0][read_subword0_q[0]*100 + (lane-20)*5 + 4]}},
                         bank_read_data[1][0][read_subword0_q[0]*100 + (lane-20)*5 +: 5]};
                    o_term1_flat[lane*25 +: 25] =
                        {{20{bank_read_data[1][1][read_subword1_q[0]*100 + (lane-20)*5 + 4]}},
                         bank_read_data[1][1][read_subword1_q[0]*100 + (lane-20)*5 +: 5]};
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
                    {{9{bank_read_data[scnn_active_bank_q][0][read_subword0_q*16 + 15]}},
                     bank_read_data[scnn_active_bank_q][0][read_subword0_q*16 +: 16]};
                o_term1_flat[lane*25 +: 25] =
                    {{9{bank_read_data[scnn_active_bank_q][1][read_subword1_q*16 + 15]}},
                     bank_read_data[scnn_active_bank_q][1][read_subword1_q*16 +: 16]};
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            mode_q <= MODE_FC;
            scnn_active_bank_q <= 1'b0;
            read_subword0_q <= '0;
            read_subword1_q <= '0;
            read_port_tail_q <= '0;
            read_result_valid_q <= 1'b0;
        end else begin
            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire) begin
                mode_q <= i_mode;
                scnn_active_bank_q <= i_scnn_active_bank;
                read_subword0_q <= i_read_subword0;
                read_subword1_q <= i_read_subword1;
                read_port_tail_q <= read_targets_tail;
                read_result_valid_q <= 1'b1;
            end
        end
    end

    initial begin
        if (LANES != 40 || BANK_WIDTH != 200 || MAIN_DEPTH != 1024 ||
            TAIL_DEPTH != 132)
            $error("weight-store v2 requires fixed 2x200x(1024+132) geometry");
    end
endmodule

`default_nettype wire
