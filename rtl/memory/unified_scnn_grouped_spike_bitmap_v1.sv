`default_nettype none

// Stores one K8 channel-activity mask per spatial coordinate. The two slots
// follow the feature-memory ping-pong ownership, so a layer may read one map
// while the preceding writeback fills the other. Payload RAM is reset-free.
module unified_scnn_grouped_spike_bitmap_v1 #(
    parameter int LANES = 8,
    parameter int MAX_CHANNELS = 512,
    parameter int MAX_SPATIAL = 1024,
    parameter int MAX_PACKED_WORDS = 8192,
    parameter int CHANNEL_ADDR_WIDTH = $clog2(MAX_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(MAX_CHANNELS + 1),
    parameter int GROUP_ADDR_WIDTH = $clog2((MAX_CHANNELS + LANES - 1) / LANES),
    parameter int SPATIAL_ADDR_WIDTH = $clog2(MAX_SPATIAL),
    parameter int SPATIAL_COUNT_WIDTH = $clog2(MAX_SPATIAL + 1),
    parameter int WORD_ADDR_WIDTH = $clog2(MAX_PACKED_WORDS)
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_clear_error,

    input  logic                           i_write_valid,
    output logic                           o_write_ready,
    input  logic                           i_write_slot,
    input  logic [CHANNEL_ADDR_WIDTH-1:0]  i_write_channel,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]  i_write_spatial,
    input  logic [CHANNEL_COUNT_WIDTH-1:0] i_write_channel_count,
    input  logic [SPATIAL_COUNT_WIDTH-1:0] i_write_spatial_count,
    input  logic                           i_write_spike,

    input  logic                           i_read_valid,
    output logic                           o_read_ready,
    input  logic                           i_read_slot,
    input  logic [GROUP_ADDR_WIDTH-1:0]    i_read_channel_group,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]  i_read_spatial,
    input  logic [CHANNEL_COUNT_WIDTH-1:0] i_read_channel_count,
    input  logic [SPATIAL_COUNT_WIDTH-1:0] i_read_spatial_count,

    output logic                           o_read_result_valid,
    input  logic                           i_read_result_ready,
    output logic [LANES-1:0]               o_read_lane_mask,

    output logic                           o_busy,
    output logic                           o_sticky_error,
    output logic [7:0]                     o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_WRITE_BOUNDS = 8'h01;
    localparam logic [7:0] ERROR_READ_BOUNDS = 8'h02;
    localparam logic [7:0] ERROR_SLOT_COLLISION = 8'h03;

    logic write_fire;
    logic read_fire;
    logic write_legal;
    logic read_legal;
    logic slot_collision;
    logic [WORD_ADDR_WIDTH-1:0] write_word_addr;
    logic [WORD_ADDR_WIDTH-1:0] read_word_addr;
    logic [$clog2(LANES)-1:0] write_lane;
    logic [LANES-1:0] read_tail_mask;
    logic [LANES-1:0] response_tail_mask_q;
    logic response_slot_q;
    logic read_result_valid_q;
    logic [LANES-1:0] memory_read_data [0:1];
    logic [1:0] memory_write_enable;
    logic [1:0] memory_read_enable;
    logic [LANES-1:0] memory_write_mask [0:1];

    integer write_group_integer;
    integer write_word_integer;
    integer read_word_integer;
    integer read_group_count_integer;

    initial begin
        if (LANES != 8)
            $error("unified_scnn_grouped_spike_bitmap_v1 requires LANES=8");
        if (MAX_PACKED_WORDS < MAX_SPATIAL)
            $error("MAX_PACKED_WORDS must cover one spatial plane");
    end

    always_comb begin
        o_write_ready = !o_sticky_error;
        o_read_ready = !o_sticky_error &&
                       (!read_result_valid_q || i_read_result_ready);
        write_fire = i_write_valid && o_write_ready;
        read_fire = i_read_valid && o_read_ready;
        slot_collision = write_fire && read_fire &&
                         i_write_slot == i_read_slot;

        write_group_integer = integer'(i_write_channel) / LANES;
        write_word_integer = write_group_integer *
                             integer'(i_write_spatial_count) +
                             integer'(i_write_spatial);
        read_word_integer = integer'(i_read_channel_group) *
                            integer'(i_read_spatial_count) +
                            integer'(i_read_spatial);
        read_group_count_integer =
            (integer'(i_read_channel_count) + LANES - 1) / LANES;
        write_word_addr = WORD_ADDR_WIDTH'(write_word_integer);
        read_word_addr = WORD_ADDR_WIDTH'(read_word_integer);
        write_lane = $clog2(LANES)'(integer'(i_write_channel) % LANES);

        write_legal = i_write_channel_count != 0 &&
                      i_write_channel_count <=
                          CHANNEL_COUNT_WIDTH'(MAX_CHANNELS) &&
                      i_write_spatial_count != 0 &&
                      i_write_spatial_count <=
                          SPATIAL_COUNT_WIDTH'(MAX_SPATIAL) &&
                      CHANNEL_COUNT_WIDTH'(i_write_channel) <
                          i_write_channel_count &&
                      SPATIAL_COUNT_WIDTH'(i_write_spatial) <
                          i_write_spatial_count &&
                      write_word_integer < MAX_PACKED_WORDS;
        read_legal = i_read_channel_count != 0 &&
                     i_read_channel_count <=
                         CHANNEL_COUNT_WIDTH'(MAX_CHANNELS) &&
                     i_read_spatial_count != 0 &&
                     i_read_spatial_count <=
                         SPATIAL_COUNT_WIDTH'(MAX_SPATIAL) &&
                     integer'(i_read_channel_group) <
                         read_group_count_integer &&
                     SPATIAL_COUNT_WIDTH'(i_read_spatial) <
                         i_read_spatial_count &&
                     read_word_integer < MAX_PACKED_WORDS;

        read_tail_mask = '0;
        for (integer lane = 0; lane < LANES; lane++)
            read_tail_mask[lane] =
                integer'(i_read_channel_group) * LANES + lane <
                integer'(i_read_channel_count);

        memory_write_enable = '0;
        memory_read_enable = '0;
        memory_write_mask[0] = '0;
        memory_write_mask[1] = '0;
        if (write_fire && write_legal && !slot_collision) begin
            memory_write_enable[i_write_slot] = 1'b1;
            memory_write_mask[i_write_slot][write_lane] = 1'b1;
        end
        if (read_fire && read_legal && !slot_collision)
            memory_read_enable[i_read_slot] = 1'b1;
    end

    assign o_read_result_valid = read_result_valid_q;
    assign o_read_lane_mask = memory_read_data[response_slot_q] &
                              response_tail_mask_q;
    assign o_busy = read_result_valid_q;

`ifdef UNIFIED_USE_XPM_MEMORY
    generate
        for (genvar slot = 0; slot < 2; slot++) begin : g_xpm_slot
            xpm_memory_sdpram #(
                .MEMORY_SIZE(MAX_PACKED_WORDS * LANES),
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
                .BYTE_WRITE_WIDTH_A(1),
                .ADDR_WIDTH_A(WORD_ADDR_WIDTH),
                .RST_MODE_A("SYNC"),
                .READ_DATA_WIDTH_B(LANES),
                .ADDR_WIDTH_B(WORD_ADDR_WIDTH),
                .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1),
                .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_bitmap_memory (
                .sleep(1'b0),
                .clka(i_clk),
                .ena(memory_write_enable[slot]),
                .wea(memory_write_mask[slot]),
                .addra(write_word_addr),
                .dina({LANES{i_write_spike}}),
                .injectsbiterra(1'b0),
                .injectdbiterra(1'b0),
                .clkb(i_clk),
                .rstb(1'b0),
                .enb(memory_read_enable[slot]),
                .regceb(1'b1),
                .addrb(read_word_addr),
                .doutb(memory_read_data[slot]),
                .sbiterrb(),
                .dbiterrb()
            );
        end
    endgenerate
`else
    (* ram_style = "block" *) logic [LANES-1:0]
        memory [0:1][0:MAX_PACKED_WORDS-1];

    always_ff @(posedge i_clk) begin
        for (integer slot = 0; slot < 2; slot++) begin
            if (memory_write_enable[slot])
                for (integer lane = 0; lane < LANES; lane++)
                    if (memory_write_mask[slot][lane])
                        memory[slot][write_word_addr][lane] <= i_write_spike;
            if (memory_read_enable[slot])
                memory_read_data[slot] <= memory[slot][read_word_addr];
        end
    end
`endif

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            response_tail_mask_q <= '0;
            response_slot_q <= 1'b0;
            read_result_valid_q <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else begin
            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire && read_legal && !slot_collision) begin
                response_tail_mask_q <= read_tail_mask;
                response_slot_q <= i_read_slot;
                read_result_valid_q <= 1'b1;
            end

            if (i_clear_error && !read_result_valid_q) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end
            if (slot_collision) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_SLOT_COLLISION;
            end else if (write_fire && !write_legal) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_WRITE_BOUNDS;
            end else if (read_fire && !read_legal) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_READ_BOUNDS;
            end
        end
    end
endmodule

`default_nettype wire
