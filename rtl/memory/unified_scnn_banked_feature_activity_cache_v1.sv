`default_nettype none

// Read-through activity cache for one SCNN input feature map. Each of the nine
// feature banks stores {known, active}; payload is cleared by a bounded scan at
// layer admission rather than by reset so block-memory inference is preserved.
module unified_scnn_banked_feature_activity_cache_v1 #(
    parameter int LANES = 8,
    parameter int BANKS = 9,
    parameter int BANK_WIDTH = 4,
    parameter int DEPTH = 9216,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_abort,
    input  logic                         i_clear_error,

    input  logic                         i_map_clear_start,
    output logic                         o_map_clear_ready,
    output logic                         o_map_clear_busy,
    output logic                         o_map_clear_done,

    input  logic                         i_lookup_valid,
    output logic                         o_lookup_ready,
    input  logic [LANES*BANK_WIDTH-1:0]  i_lookup_bank_flat,
    input  logic [LANES*ADDR_WIDTH-1:0]  i_lookup_addr_flat,
    input  logic [LANES-1:0]             i_lookup_lane_mask,
    output logic                         o_lookup_result_valid,
    input  logic                         i_lookup_result_ready,
    output logic [LANES-1:0]             o_lookup_known_mask,
    output logic [LANES-1:0]             o_lookup_active_mask,

    input  logic                         i_fill_valid,
    output logic                         o_fill_ready,
    input  logic [LANES*BANK_WIDTH-1:0]  i_fill_bank_flat,
    input  logic [LANES*ADDR_WIDTH-1:0]  i_fill_addr_flat,
    input  logic [LANES-1:0]             i_fill_lane_mask,
    input  logic [LANES-1:0]             i_fill_active_mask,

    output logic                         o_busy,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_LOOKUP = 8'h01;
    localparam logic [7:0] ERROR_FILL = 8'h02;
    localparam logic [7:0] ERROR_CLEAR = 8'h03;

    logic map_clear_busy_q;
    logic [ADDR_WIDTH-1:0] map_clear_addr_q;
    logic lookup_result_valid_q;
    logic lookup_read_pending_q;
    logic [LANES-1:0] lookup_lane_mask_q;
    logic [LANES*BANK_WIDTH-1:0] lookup_bank_flat_q;
    logic [LANES-1:0] lookup_known_mask_q;
    logic [LANES-1:0] lookup_active_mask_q;
    logic [1:0] bank_read_data [0:BANKS-1];
    logic [BANKS-1:0] bank_read_enable;
    logic [ADDR_WIDTH-1:0] bank_read_addr [0:BANKS-1];
    logic [BANKS-1:0] bank_write_enable;
    logic [ADDR_WIDTH-1:0] bank_write_addr [0:BANKS-1];
    logic [1:0] bank_write_data [0:BANKS-1];

    logic lookup_fire;
    logic fill_fire;
    logic lookup_legal;
    logic fill_legal;

    initial begin
        if (LANES != 8)
            $error("unified_scnn_banked_feature_activity_cache_v1 requires LANES=8");
        if (BANKS != 9)
            $error("unified_scnn_banked_feature_activity_cache_v1 requires BANKS=9");
    end

    assign o_map_clear_ready = !map_clear_busy_q && !lookup_result_valid_q &&
                               !lookup_read_pending_q &&
                               !i_lookup_valid && !i_fill_valid &&
                               !o_sticky_error;
    assign o_map_clear_busy = map_clear_busy_q;
    assign o_lookup_ready = !map_clear_busy_q && !lookup_read_pending_q &&
                            (!lookup_result_valid_q ||
                             i_lookup_result_ready) &&
                            !o_sticky_error;
    assign o_fill_ready = !map_clear_busy_q && !i_lookup_valid &&
                          !o_sticky_error;
    assign lookup_fire = i_lookup_valid && o_lookup_ready;
    assign fill_fire = i_fill_valid && o_fill_ready;

    always_comb begin
        lookup_legal = i_lookup_lane_mask != '0;
        fill_legal = i_fill_lane_mask != '0;
        for (integer lane = 0; lane < LANES; lane++) begin
            if (i_lookup_lane_mask[lane]) begin
                if (integer'(i_lookup_bank_flat[lane*BANK_WIDTH +: BANK_WIDTH]) >=
                    BANKS ||
                    integer'(i_lookup_addr_flat[lane*ADDR_WIDTH +: ADDR_WIDTH]) >=
                    DEPTH)
                    lookup_legal = 1'b0;
                for (integer prior = 0; prior < lane; prior++)
                    if (i_lookup_lane_mask[prior] &&
                        i_lookup_bank_flat[lane*BANK_WIDTH +: BANK_WIDTH] ==
                        i_lookup_bank_flat[prior*BANK_WIDTH +: BANK_WIDTH])
                        lookup_legal = 1'b0;
            end
            if (i_fill_lane_mask[lane]) begin
                if (integer'(i_fill_bank_flat[lane*BANK_WIDTH +: BANK_WIDTH]) >=
                    BANKS ||
                    integer'(i_fill_addr_flat[lane*ADDR_WIDTH +: ADDR_WIDTH]) >=
                    DEPTH)
                    fill_legal = 1'b0;
                for (integer prior = 0; prior < lane; prior++)
                    if (i_fill_lane_mask[prior] &&
                        i_fill_bank_flat[lane*BANK_WIDTH +: BANK_WIDTH] ==
                        i_fill_bank_flat[prior*BANK_WIDTH +: BANK_WIDTH])
                        fill_legal = 1'b0;
            end
        end

        bank_read_enable = '0;
        bank_write_enable = '0;
        for (integer bank = 0; bank < BANKS; bank++) begin
            bank_read_addr[bank] = '0;
            bank_write_addr[bank] = map_clear_addr_q;
            bank_write_data[bank] = 2'b00;
            if (map_clear_busy_q) begin
                bank_write_enable[bank] = 1'b1;
            end else begin
                for (integer lane = 0; lane < LANES; lane++) begin
                    if (lookup_fire && lookup_legal &&
                        i_lookup_lane_mask[lane] &&
                        integer'(i_lookup_bank_flat[
                            lane*BANK_WIDTH +: BANK_WIDTH]) == bank) begin
                        bank_read_enable[bank] = 1'b1;
                        bank_read_addr[bank] = i_lookup_addr_flat[
                            lane*ADDR_WIDTH +: ADDR_WIDTH];
                    end
                    if (fill_fire && fill_legal &&
                        i_fill_lane_mask[lane] &&
                        integer'(i_fill_bank_flat[
                            lane*BANK_WIDTH +: BANK_WIDTH]) == bank) begin
                        bank_write_enable[bank] = 1'b1;
                        bank_write_addr[bank] = i_fill_addr_flat[
                            lane*ADDR_WIDTH +: ADDR_WIDTH];
                        bank_write_data[bank] =
                            {1'b1, i_fill_active_mask[lane]};
                    end
                end
            end
        end

        o_lookup_known_mask = lookup_known_mask_q;
        o_lookup_active_mask = lookup_active_mask_q;
    end

    assign o_lookup_result_valid = lookup_result_valid_q;
    assign o_busy = map_clear_busy_q || lookup_read_pending_q ||
                    lookup_result_valid_q;

    generate
        for (genvar bank = 0; bank < BANKS; bank++) begin : g_bank
`ifdef UNIFIED_USE_XPM_MEMORY
            xpm_memory_sdpram #(
                .MEMORY_SIZE(DEPTH * 2),
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
                .WRITE_DATA_WIDTH_A(2),
                .BYTE_WRITE_WIDTH_A(2),
                .ADDR_WIDTH_A(ADDR_WIDTH),
                .RST_MODE_A("SYNC"),
                .READ_DATA_WIDTH_B(2),
                .ADDR_WIDTH_B(ADDR_WIDTH),
                .READ_RESET_VALUE_B("0"),
                .READ_LATENCY_B(1),
                .WRITE_MODE_B("read_first"),
                .RST_MODE_B("SYNC")
            ) u_activity_memory (
                .sleep(1'b0),
                .clka(i_clk), .ena(bank_write_enable[bank]),
                .wea(bank_write_enable[bank]),
                .addra(bank_write_addr[bank]),
                .dina(bank_write_data[bank]),
                .injectsbiterra(1'b0), .injectdbiterra(1'b0),
                .clkb(i_clk), .rstb(1'b0),
                .enb(bank_read_enable[bank]), .regceb(1'b1),
                .addrb(bank_read_addr[bank]),
                .doutb(bank_read_data[bank]),
                .sbiterrb(), .dbiterrb()
            );
`else
            (* ram_style = "block" *) logic [1:0] memory [0:DEPTH-1];
            always_ff @(posedge i_clk) begin
                if (bank_write_enable[bank])
                    memory[bank_write_addr[bank]] <= bank_write_data[bank];
                if (bank_read_enable[bank])
                    bank_read_data[bank] <= memory[bank_read_addr[bank]];
            end
`endif
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            map_clear_busy_q <= 1'b0;
            map_clear_addr_q <= '0;
            o_map_clear_done <= 1'b0;
            lookup_result_valid_q <= 1'b0;
            lookup_read_pending_q <= 1'b0;
            lookup_lane_mask_q <= '0;
            lookup_bank_flat_q <= '0;
            lookup_known_mask_q <= '0;
            lookup_active_mask_q <= '0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            map_clear_busy_q <= 1'b0;
            map_clear_addr_q <= '0;
            o_map_clear_done <= 1'b0;
            lookup_result_valid_q <= 1'b0;
            lookup_read_pending_q <= 1'b0;
            lookup_lane_mask_q <= '0;
            lookup_bank_flat_q <= '0;
            lookup_known_mask_q <= '0;
            lookup_active_mask_q <= '0;
        end else begin
            o_map_clear_done <= 1'b0;
            lookup_read_pending_q <= 1'b0;
            if (lookup_result_valid_q && i_lookup_result_ready)
                lookup_result_valid_q <= 1'b0;
            if (lookup_read_pending_q) begin
                lookup_result_valid_q <= 1'b1;
                lookup_known_mask_q <= '0;
                lookup_active_mask_q <= '0;
                for (integer lane = 0; lane < LANES; lane++) begin
                    if (lookup_lane_mask_q[lane]) begin
                        lookup_known_mask_q[lane] <= bank_read_data[
                            lookup_bank_flat_q[
                                lane*BANK_WIDTH +: BANK_WIDTH]][1];
                        lookup_active_mask_q[lane] <= bank_read_data[
                            lookup_bank_flat_q[
                                lane*BANK_WIDTH +: BANK_WIDTH]][0];
                    end
                end
            end

            if (i_map_clear_start && o_map_clear_ready) begin
                map_clear_busy_q <= 1'b1;
                map_clear_addr_q <= '0;
            end else if (i_map_clear_start) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_CLEAR;
            end

            if (map_clear_busy_q) begin
                if (integer'(map_clear_addr_q) + 1 >= DEPTH) begin
                    map_clear_busy_q <= 1'b0;
                    map_clear_addr_q <= '0;
                    o_map_clear_done <= 1'b1;
                end else begin
                    map_clear_addr_q <= map_clear_addr_q + 1'b1;
                end
            end

            if (lookup_fire) begin
                if (lookup_legal) begin
                    lookup_lane_mask_q <= i_lookup_lane_mask;
                    lookup_bank_flat_q <= i_lookup_bank_flat;
                    lookup_read_pending_q <= 1'b1;
                end else begin
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_LOOKUP;
                end
            end
            if (fill_fire && !fill_legal) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_FILL;
            end

            if (i_clear_error && !map_clear_busy_q &&
                !lookup_read_pending_q &&
                !lookup_result_valid_q && !i_lookup_valid && !i_fill_valid) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end
        end
    end
endmodule

`default_nettype wire
