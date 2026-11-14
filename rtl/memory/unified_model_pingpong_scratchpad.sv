`default_nettype none

module unified_model_pingpong_scratchpad #(
    parameter int WORD_WIDTH = 64,
    parameter int BANK_WORDS = 256,
    parameter int TAG_WIDTH = 16
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,

    input  logic                         i_prefetch_valid,
    output logic                         o_prefetch_ready,
    input  logic [TAG_WIDTH-1:0]         i_prefetch_tag,
    input  logic [$clog2(BANK_WORDS+1)-1:0] i_prefetch_word_count,
    output logic                         o_prefetch_bank,

    input  logic                         i_fill_valid,
    output logic                         o_fill_ready,
    input  logic [$clog2(BANK_WORDS)-1:0] i_fill_word_addr,
    input  logic [WORD_WIDTH-1:0]        i_fill_data,
    input  logic                         i_fill_last,
    input  logic                         i_prefetch_abort,
    input  logic                         i_cancel,

    input  logic                         i_compute_acquire_valid,
    output logic                         o_compute_acquire_ready,
    input  logic [TAG_WIDTH-1:0]         i_compute_acquire_tag,
    output logic                         o_compute_bank,
    output logic [TAG_WIDTH-1:0]         o_compute_tag,
    input  logic                         i_compute_release,

    input  logic                         i_read_valid,
    output logic                         o_read_ready,
    input  logic [$clog2(BANK_WORDS)-1:0] i_read_word_addr0,
    input  logic [$clog2(BANK_WORDS)-1:0] i_read_word_addr1,
    output logic                         o_read_valid,
    input  logic                         i_read_ready,
    output logic [WORD_WIDTH-1:0]        o_read_data0,
    output logic [WORD_WIDTH-1:0]        o_read_data1,

    output logic [1:0]                   o_bank_ready,
    output logic                         o_prefetch_active,
    output logic                         o_compute_active,
    output logic                         o_idle,
    output logic                         o_bounds_error,
    output logic                         o_protocol_error
);

    localparam int COUNT_WIDTH = $clog2(BANK_WORDS + 1);

`ifndef UNIFIED_USE_XPM_MEMORY
    (* ram_style = "block" *) logic [WORD_WIDTH-1:0]
        bank0_mem [0:BANK_WORDS-1];
    (* ram_style = "block" *) logic [WORD_WIDTH-1:0]
        bank1_mem [0:BANK_WORDS-1];
`endif

    logic [1:0] bank_ready_q;
    logic [TAG_WIDTH-1:0] bank_tag_q [0:1];
    logic [COUNT_WIDTH-1:0] bank_words_q [0:1];
    logic prefetch_active_q;
    logic prefetch_bank_q;
    logic [COUNT_WIDTH-1:0] prefetch_words_q;
    logic [COUNT_WIDTH-1:0] received_words_q;
    logic prefetch_fault_q;
    logic compute_active_q;
    logic compute_bank_q;
    logic [TAG_WIDTH-1:0] compute_tag_q;
    logic [COUNT_WIDTH-1:0] compute_words_q;
    logic read_valid_q;
`ifdef UNIFIED_USE_XPM_MEMORY
    logic read_invalid_q;
    logic [WORD_WIDTH-1:0] xpm_read_data [0:1][0:1];
    logic [$clog2(BANK_WORDS)-1:0] xpm_read_addr [0:1][0:1];
    logic [1:0] xpm_fill_write_enable;
    logic [1:0] xpm_compute_read_enable;
`else
    logic [WORD_WIDTH-1:0] read_data0_q;
    logic [WORD_WIDTH-1:0] read_data1_q;
`endif

    logic free_bank0;
    logic free_bank1;
    logic selected_free_bank;
    logic matching_bank0;
    logic matching_bank1;
    logic selected_compute_bank;
    logic prefetch_accept;
    logic fill_accept;
    logic fill_address_valid;
    logic fill_expected_last;
    logic fill_fault;
    logic compute_accept;
    logic read_accept;
    logic read_addresses_valid;

    assign free_bank0 = !bank_ready_q[0] &&
                        !(compute_active_q && compute_bank_q == 1'b0);
    assign free_bank1 = !bank_ready_q[1] &&
                        !(compute_active_q && compute_bank_q == 1'b1);
    assign selected_free_bank = free_bank0 ? 1'b0 : 1'b1;
    assign o_prefetch_ready = !prefetch_active_q && (free_bank0 || free_bank1);
    assign prefetch_accept = i_prefetch_valid && o_prefetch_ready;
    assign o_prefetch_bank = prefetch_active_q ? prefetch_bank_q : selected_free_bank;

    assign o_fill_ready = prefetch_active_q;
    assign fill_accept = i_fill_valid && o_fill_ready;
    assign fill_address_valid =
        COUNT_WIDTH'(i_fill_word_addr) == received_words_q &&
        COUNT_WIDTH'(i_fill_word_addr) < prefetch_words_q;
    assign fill_expected_last = received_words_q + 1'b1 == prefetch_words_q;
    assign fill_fault = !fill_address_valid || i_fill_last != fill_expected_last;

    assign matching_bank0 = bank_ready_q[0] &&
                            bank_tag_q[0] == i_compute_acquire_tag;
    assign matching_bank1 = bank_ready_q[1] &&
                            bank_tag_q[1] == i_compute_acquire_tag;
    assign selected_compute_bank = matching_bank0 ? 1'b0 : 1'b1;
    assign o_compute_acquire_ready = !compute_active_q &&
                                     (matching_bank0 || matching_bank1);
    assign compute_accept = i_compute_acquire_valid && o_compute_acquire_ready;

    assign read_addresses_valid =
        COUNT_WIDTH'(i_read_word_addr0) < compute_words_q &&
        COUNT_WIDTH'(i_read_word_addr1) < compute_words_q;
    // Release is handled by the registered protocol state below. Keeping it
    // out of the BRAM read-enable cone prevents an asynchronously reset abort
    // tree from reaching every scratchpad EN pin.
    assign o_read_ready = compute_active_q &&
                          (!read_valid_q || i_read_ready);
    assign read_accept = i_read_valid && o_read_ready;
    assign o_read_valid = read_valid_q;
`ifdef UNIFIED_USE_XPM_MEMORY
    assign o_read_data0 = read_invalid_q ? '0 :
        xpm_read_data[compute_bank_q][0];
    assign o_read_data1 = read_invalid_q ? '0 :
        xpm_read_data[compute_bank_q][1];
    assign xpm_fill_write_enable[0] = fill_accept &&
        fill_address_valid && prefetch_bank_q == 1'b0;
    assign xpm_fill_write_enable[1] = fill_accept &&
        fill_address_valid && prefetch_bank_q == 1'b1;
    assign xpm_compute_read_enable[0] = read_accept &&
        read_addresses_valid && compute_bank_q == 1'b0;
    assign xpm_compute_read_enable[1] = read_accept &&
        read_addresses_valid && compute_bank_q == 1'b1;
`else
    assign o_read_data0 = read_data0_q;
    assign o_read_data1 = read_data1_q;
`endif

    assign o_bank_ready = bank_ready_q;
    assign o_prefetch_active = prefetch_active_q;
    assign o_compute_active = compute_active_q;
    assign o_compute_bank = compute_bank_q;
    assign o_compute_tag = compute_tag_q;
    assign o_idle = !prefetch_active_q && !compute_active_q && !read_valid_q;

`ifdef UNIFIED_USE_XPM_MEMORY
    generate
        for (genvar bank = 0; bank < 2; bank++) begin : g_xpm_bank
            for (genvar read_port = 0; read_port < 2;
                 read_port++) begin : g_xpm_read_copy
                for (genvar address_bit = 0;
                     address_bit < $clog2(BANK_WORDS);
                     address_bit++) begin : g_local_address_buffer
                    // A 8640x64 XPM instance expands to 32 RAMB36 primitives.
                    // Keep one physical buffer per ping-pong bank/read port so
                    // an upstream address bit does not span both BRAM regions.
                    (* dont_touch = "true" *) LUT1 #(
                        .INIT(2'b10)
                    ) u_address_buffer (
                        .I0(read_port == 0 ?
                            i_read_word_addr0[address_bit] :
                            i_read_word_addr1[address_bit]),
                        .O(xpm_read_addr[bank][read_port][address_bit])
                    );
                end

                xpm_memory_sdpram #(
                    .MEMORY_SIZE(BANK_WORDS * WORD_WIDTH),
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
                    .ADDR_WIDTH_A($clog2(BANK_WORDS)),
                    .RST_MODE_A("SYNC"),
                    .READ_DATA_WIDTH_B(WORD_WIDTH),
                    .ADDR_WIDTH_B($clog2(BANK_WORDS)),
                    .READ_RESET_VALUE_B("0"),
                    .READ_LATENCY_B(1),
                    .WRITE_MODE_B("read_first"),
                    .RST_MODE_B("SYNC")
                ) u_memory (
                    .sleep(1'b0),
                    .clka(i_clk),
                    .ena(xpm_fill_write_enable[bank]),
                    .wea(xpm_fill_write_enable[bank]),
                    .addra(i_fill_word_addr),
                    .dina(i_fill_data),
                    .injectsbiterra(1'b0),
                    .injectdbiterra(1'b0),
                    .clkb(i_clk),
                    .rstb(1'b0),
                    .enb(xpm_compute_read_enable[bank]),
                    .regceb(1'b1),
                    .addrb(xpm_read_addr[bank][read_port]),
                    .doutb(xpm_read_data[bank][read_port]),
                    .sbiterrb(),
                    .dbiterrb()
                );
            end
        end
    endgenerate
`endif

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            bank_ready_q <= 2'b00;
            bank_tag_q[0] <= '0;
            bank_tag_q[1] <= '0;
            bank_words_q[0] <= '0;
            bank_words_q[1] <= '0;
            prefetch_active_q <= 1'b0;
            prefetch_bank_q <= 1'b0;
            prefetch_words_q <= '0;
            received_words_q <= '0;
            prefetch_fault_q <= 1'b0;
            compute_active_q <= 1'b0;
            compute_bank_q <= 1'b0;
            compute_tag_q <= '0;
            compute_words_q <= '0;
            read_valid_q <= 1'b0;
`ifdef UNIFIED_USE_XPM_MEMORY
            read_invalid_q <= 1'b0;
`else
            read_data0_q <= '0;
            read_data1_q <= '0;
`endif
            o_bounds_error <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (prefetch_accept) begin
                if (i_prefetch_word_count == 0 ||
                    i_prefetch_word_count > COUNT_WIDTH'(BANK_WORDS)) begin
                    o_bounds_error <= 1'b1;
                end else begin
                    prefetch_active_q <= 1'b1;
                    prefetch_bank_q <= selected_free_bank;
                    prefetch_words_q <= i_prefetch_word_count;
                    received_words_q <= '0;
                    prefetch_fault_q <= 1'b0;
                    bank_tag_q[selected_free_bank] <= i_prefetch_tag;
                end
            end

            if (fill_accept) begin
                if (fill_fault) begin
                    o_protocol_error <= 1'b1;
                    prefetch_fault_q <= 1'b1;
                end
`ifndef UNIFIED_USE_XPM_MEMORY
                if (fill_address_valid) begin
                    if (prefetch_bank_q == 1'b0)
                        bank0_mem[i_fill_word_addr] <= i_fill_data;
                    else
                        bank1_mem[i_fill_word_addr] <= i_fill_data;
                end
`endif
                if (fill_expected_last) begin
                    prefetch_active_q <= 1'b0;
                    if (!(prefetch_fault_q || fill_fault)) begin
                        bank_ready_q[prefetch_bank_q] <= 1'b1;
                        bank_words_q[prefetch_bank_q] <= prefetch_words_q;
                    end
                end else begin
                    received_words_q <= received_words_q + 1'b1;
                end
            end

            if (i_prefetch_abort) begin
                prefetch_active_q <= 1'b0;
                received_words_q <= '0;
                prefetch_fault_q <= 1'b1;
            end

            if (i_cancel) begin
                bank_ready_q <= 2'b00;
                prefetch_active_q <= 1'b0;
                received_words_q <= '0;
                prefetch_fault_q <= 1'b0;
                compute_active_q <= 1'b0;
                read_valid_q <= 1'b0;
`ifdef UNIFIED_USE_XPM_MEMORY
                read_invalid_q <= 1'b0;
`endif
            end

            if (compute_accept) begin
                compute_active_q <= 1'b1;
                compute_bank_q <= selected_compute_bank;
                compute_tag_q <= bank_tag_q[selected_compute_bank];
                compute_words_q <= bank_words_q[selected_compute_bank];
                bank_ready_q[selected_compute_bank] <= 1'b0;
            end
            if (read_valid_q && i_read_ready)
                read_valid_q <= 1'b0;
            if (read_accept) begin
                read_valid_q <= 1'b1;
`ifdef UNIFIED_USE_XPM_MEMORY
                read_invalid_q <= !read_addresses_valid;
                if (!read_addresses_valid)
                    o_bounds_error <= 1'b1;
`else
                if (read_addresses_valid) begin
                    if (compute_bank_q == 1'b0) begin
                        read_data0_q <= bank0_mem[i_read_word_addr0];
                        read_data1_q <= bank0_mem[i_read_word_addr1];
                    end else begin
                        read_data0_q <= bank1_mem[i_read_word_addr0];
                        read_data1_q <= bank1_mem[i_read_word_addr1];
                    end
                end else begin
                    read_data0_q <= '0;
                    read_data1_q <= '0;
                    o_bounds_error <= 1'b1;
                end
`endif
            end
            if (i_compute_release && compute_active_q) begin
                if (read_valid_q || read_accept)
                    o_protocol_error <= 1'b1;
                compute_active_q <= 1'b0;
                read_valid_q <= 1'b0;
`ifdef UNIFIED_USE_XPM_MEMORY
                read_invalid_q <= 1'b0;
`endif
            end
        end
    end

endmodule

`default_nettype wire
