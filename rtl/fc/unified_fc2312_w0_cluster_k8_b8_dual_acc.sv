`timescale 1ns/1ps

// Real K8 W0 producer with two accumulator banks. The eight ROM lanes and
// arithmetic contract match newarch_w0_cluster_k8_b8; only accumulator
// ownership changes so W0(t+1) may overlap folded LIF(t).
module unified_fc2312_w0_cluster_k8_b8_dual_acc #(
    parameter integer IN_DIM = 4096,
    parameter integer PHYSICAL_BANK_DEPTH = 512,
    parameter integer H_COUNT = 40,
    parameter integer FI_BITS = 12,
    parameter integer W_BITS = 5,
    parameter integer ACC_BITS = 18,
    parameter integer INPUT_SHIFT = 6,
    parameter integer EVENT_X_IS_ONE = 1,
    parameter integer EVENT_X_IS_SIGNED = 0,
    parameter integer USE_ATOMIC_BOUNDARY_CLOSE = 0,
    parameter integer USE_SAME_CYCLE_CONSUMER_RECYCLE = 0,
    parameter integer T_BITS = 9,
    parameter BANK_FILE_0 = "",
    parameter BANK_FILE_1 = "",
    parameter BANK_FILE_2 = "",
    parameter BANK_FILE_3 = "",
    parameter BANK_FILE_4 = "",
    parameter BANK_FILE_5 = "",
    parameter BANK_FILE_6 = "",
    parameter BANK_FILE_7 = ""
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          i_clear,
    input  wire                          i_step_start,
    output wire                          o_step_ready,
    input  wire [T_BITS-1:0]             i_timestep,
    input  wire                          i_group_valid,
    output wire                          o_group_ready,
    input  wire [7:0]                    i_bank_valid,
    input  wire [8*FI_BITS-1:0]          i_fi_flat,
    input  wire [63:0]                   i_x_flat,
    input  wire                          i_groups_done,
    output wire                          o_acc_valid,
    input  wire                          i_acc_start,
    input  wire                          i_acc_done,
    output wire [T_BITS-1:0]             o_acc_timestep,
    output wire [H_COUNT*ACC_BITS-1:0]   o_acc_flat,
    output wire [31:0]                   o_acc_accepted_events,
    output wire                          o_producer_active,
    output wire                          o_consumer_active,
    output wire [31:0]                   o_overlap_cycles,
    output wire                          o_protocol_error
);
    localparam integer FEATURE_BANKS = 8;
    localparam integer BANK_BITS = 3;
    // Logical INPUT_MAX remains 2312. Replica routing uses otherwise-unused
    // addresses in the same physical 512-deep BRAM geometry.
    localparam integer BANK_DEPTH = PHYSICAL_BANK_DEPTH;
    localparam integer BANK_ADDR_BITS = $clog2(BANK_DEPTH);
    localparam integer WORD_BITS = H_COUNT * W_BITS;

`ifdef SYNTHESIS
    wire [WORD_BITS-1:0] bank_word0, bank_word1, bank_word2, bank_word3;
    wire [WORD_BITS-1:0] bank_word4, bank_word5, bank_word6, bank_word7;
`else
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank0 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank1 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank2 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank3 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank4 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank5 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank6 [0:BANK_DEPTH-1];
    (* rom_style = "block" *) reg [WORD_BITS-1:0] w0_bank7 [0:BANK_DEPTH-1];
    reg [WORD_BITS-1:0] bank_word0, bank_word1, bank_word2, bank_word3;
    reg [WORD_BITS-1:0] bank_word4, bank_word5, bank_word6, bank_word7;
`endif

    reg [7:0] bank_valid_q;
    reg [63:0] bank_x_q;
    reg running;
    reg done_pending;
    reg [31:0] accepted_events;
    reg [31:0] accepted_events_by_bank [0:1];
    reg local_protocol_error;
    wire bank_step_ready;
    wire write_bank;
    wire consume_bank;
    wire bank_protocol_error;
    wire step_accept = i_step_start && o_step_ready;
    wire accept_group = i_group_valid && o_group_ready;
    wire any_writeback = |bank_valid_q;
    // The scheduler's boundary follows the last accepted group. In atomic
    // mode that boundary may close the writer on the same edge as the final
    // registered ROM writeback. The accumulator bank applies the delta and
    // transitions WRITE -> READY with nonblocking assignments on that edge.
    wire atomic_boundary_close = (USE_ATOMIC_BOUNDARY_CLOSE != 0) &&
        running && i_groups_done && !accept_group;
    wire drained_pending_close = running && done_pending &&
        !any_writeback && !accept_group;
    wire step_close = atomic_boundary_close || drained_pending_close;
    wire [H_COUNT*ACC_BITS-1:0] delta_flat;
    integer check_bank;

    function [2:0] feature_bank;
        input [FI_BITS-1:0] fi;
        begin
            feature_bank = fi[2:0] ^ {fi[6], fi[8], fi[7]};
        end
    endfunction

    function [3:0] popcount8;
        input [7:0] value;
        integer index;
        begin
            popcount8 = 0;
            for (index = 0; index < 8; index = index + 1)
                popcount8 = popcount8 + value[index];
        end
    endfunction

    function signed [ACC_BITS-1:0] scaled_contribution;
        input valid;
        input [W_BITS-1:0] packed_weight;
        input [7:0] event_x;
        reg signed [W_BITS-1:0] signed_weight;
        reg signed [15:0] product;
        reg signed [ACC_BITS-1:0] acc_width_product;
        begin
            if (!valid) begin
                scaled_contribution = 0;
            end else begin
                signed_weight = $signed(packed_weight);
                if (EVENT_X_IS_ONE != 0)
                    product = {{(16-W_BITS){signed_weight[W_BITS-1]}}, signed_weight};
                else if (EVENT_X_IS_SIGNED != 0)
                    product = $signed(event_x) * signed_weight;
                else
                    product = $signed({1'b0, event_x}) * signed_weight;
                acc_width_product = {{(ACC_BITS-16){product[15]}}, product};
                scaled_contribution = acc_width_product <<< INPUT_SHIFT;
            end
        end
    endfunction

    wire [BANK_ADDR_BITS-1:0] addr0 = i_fi_flat[0*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr1 = i_fi_flat[1*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr2 = i_fi_flat[2*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr3 = i_fi_flat[3*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr4 = i_fi_flat[4*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr5 = i_fi_flat[5*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr6 = i_fi_flat[6*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];
    wire [BANK_ADDR_BITS-1:0] addr7 = i_fi_flat[7*FI_BITS+BANK_BITS +: BANK_ADDR_BITS];

    assign o_step_ready = bank_step_ready && !running;
    assign o_group_ready = running && !done_pending;
    assign o_acc_accepted_events = accepted_events_by_bank[consume_bank];
    assign o_protocol_error = local_protocol_error || bank_protocol_error;

    genvar hidden;
    generate
        for (hidden = 0; hidden < H_COUNT; hidden = hidden + 1) begin : gen_delta
            wire signed [ACC_BITS-1:0] delta_sum;
            assign delta_sum =
                scaled_contribution(bank_valid_q[0], bank_word0[hidden*W_BITS +: W_BITS], bank_x_q[0*8 +: 8]) +
                scaled_contribution(bank_valid_q[1], bank_word1[hidden*W_BITS +: W_BITS], bank_x_q[1*8 +: 8]) +
                scaled_contribution(bank_valid_q[2], bank_word2[hidden*W_BITS +: W_BITS], bank_x_q[2*8 +: 8]) +
                scaled_contribution(bank_valid_q[3], bank_word3[hidden*W_BITS +: W_BITS], bank_x_q[3*8 +: 8]) +
                scaled_contribution(bank_valid_q[4], bank_word4[hidden*W_BITS +: W_BITS], bank_x_q[4*8 +: 8]) +
                scaled_contribution(bank_valid_q[5], bank_word5[hidden*W_BITS +: W_BITS], bank_x_q[5*8 +: 8]) +
                scaled_contribution(bank_valid_q[6], bank_word6[hidden*W_BITS +: W_BITS], bank_x_q[6*8 +: 8]) +
                scaled_contribution(bank_valid_q[7], bank_word7[hidden*W_BITS +: W_BITS], bank_x_q[7*8 +: 8]);
            assign delta_flat[hidden*ACC_BITS +: ACC_BITS] = delta_sum;
        end
    endgenerate

`ifdef SYNTHESIS
`define N8D2B_W0_ROM(ID, FILE_NAME, ENABLE, ADDRESS, DATA) \
    xpm_memory_sprom #( \
        .MEMORY_SIZE(BANK_DEPTH * WORD_BITS), \
        .MEMORY_PRIMITIVE("block"), .MEMORY_INIT_FILE(FILE_NAME), \
        .USE_MEM_INIT(1), .MEMORY_OPTIMIZATION("true"), \
        .READ_DATA_WIDTH_A(WORD_BITS), .ADDR_WIDTH_A(BANK_ADDR_BITS), \
        .READ_LATENCY_A(1), .READ_RESET_VALUE_A("0"), .RST_MODE_A("SYNC") \
    ) ID ( \
        .sleep(1'b0), .clka(clk), .rsta(1'b0), .ena(ENABLE), \
        .regcea(1'b1), .addra(ADDRESS), .injectsbiterra(1'b0), \
        .injectdbiterra(1'b0), .douta(DATA), .sbiterra(), .dbiterra() \
    );
    `N8D2B_W0_ROM(u_bank0, BANK_FILE_0, accept_group && i_bank_valid[0], addr0, bank_word0)
    `N8D2B_W0_ROM(u_bank1, BANK_FILE_1, accept_group && i_bank_valid[1], addr1, bank_word1)
    `N8D2B_W0_ROM(u_bank2, BANK_FILE_2, accept_group && i_bank_valid[2], addr2, bank_word2)
    `N8D2B_W0_ROM(u_bank3, BANK_FILE_3, accept_group && i_bank_valid[3], addr3, bank_word3)
    `N8D2B_W0_ROM(u_bank4, BANK_FILE_4, accept_group && i_bank_valid[4], addr4, bank_word4)
    `N8D2B_W0_ROM(u_bank5, BANK_FILE_5, accept_group && i_bank_valid[5], addr5, bank_word5)
    `N8D2B_W0_ROM(u_bank6, BANK_FILE_6, accept_group && i_bank_valid[6], addr6, bank_word6)
    `N8D2B_W0_ROM(u_bank7, BANK_FILE_7, accept_group && i_bank_valid[7], addr7, bank_word7)
`undef N8D2B_W0_ROM
`else
    initial begin
        if (BANK_FILE_0 != "") $readmemh(BANK_FILE_0, w0_bank0);
        if (BANK_FILE_1 != "") $readmemh(BANK_FILE_1, w0_bank1);
        if (BANK_FILE_2 != "") $readmemh(BANK_FILE_2, w0_bank2);
        if (BANK_FILE_3 != "") $readmemh(BANK_FILE_3, w0_bank3);
        if (BANK_FILE_4 != "") $readmemh(BANK_FILE_4, w0_bank4);
        if (BANK_FILE_5 != "") $readmemh(BANK_FILE_5, w0_bank5);
        if (BANK_FILE_6 != "") $readmemh(BANK_FILE_6, w0_bank6);
        if (BANK_FILE_7 != "") $readmemh(BANK_FILE_7, w0_bank7);
    end
    always @(posedge clk) begin
        if (accept_group && i_bank_valid[0]) bank_word0 <= w0_bank0[addr0];
        if (accept_group && i_bank_valid[1]) bank_word1 <= w0_bank1[addr1];
        if (accept_group && i_bank_valid[2]) bank_word2 <= w0_bank2[addr2];
        if (accept_group && i_bank_valid[3]) bank_word3 <= w0_bank3[addr3];
        if (accept_group && i_bank_valid[4]) bank_word4 <= w0_bank4[addr4];
        if (accept_group && i_bank_valid[5]) bank_word5 <= w0_bank5[addr5];
        if (accept_group && i_bank_valid[6]) bank_word6 <= w0_bank6[addr6];
        if (accept_group && i_bank_valid[7]) bank_word7 <= w0_bank7[addr7];
    end
`endif

    newarch_n8d_dual_accumulator_bank #(
        .H_COUNT(H_COUNT), .ACC_BITS(ACC_BITS), .T_BITS(T_BITS),
        .USE_SAME_CYCLE_CONSUMER_RECYCLE(USE_SAME_CYCLE_CONSUMER_RECYCLE)
    ) u_accumulator_banks (
        .clk(clk), .rst_n(rst_n), .i_clear(i_clear),
        .i_step_start(step_accept), .o_step_ready(bank_step_ready),
        .i_timestep(i_timestep),
        .i_accumulate_valid(any_writeback), .i_delta_flat(delta_flat),
        .i_step_close(step_close), .o_consume_valid(o_acc_valid),
        .i_consume_start(i_acc_start), .i_consume_done(i_acc_done),
        .o_consume_timestep(o_acc_timestep),
        .o_consume_acc_flat(o_acc_flat),
        .o_write_bank(write_bank), .o_consume_bank(consume_bank),
        .o_producer_active(o_producer_active),
        .o_consumer_active(o_consumer_active),
        .o_overlap_cycles(o_overlap_cycles),
        .o_protocol_error(bank_protocol_error)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bank_valid_q <= 0;
            bank_x_q <= 0;
            running <= 1'b0;
            done_pending <= 1'b0;
            accepted_events <= 0;
            accepted_events_by_bank[0] <= 0;
            accepted_events_by_bank[1] <= 0;
            local_protocol_error <= 1'b0;
        end else if (i_clear) begin
            bank_valid_q <= 0;
            bank_x_q <= 0;
            running <= 1'b0;
            done_pending <= 1'b0;
            accepted_events <= 0;
            accepted_events_by_bank[0] <= 0;
            accepted_events_by_bank[1] <= 0;
            local_protocol_error <= 1'b0;
        end else begin
            bank_valid_q <= 0;
            if (i_step_start && !o_step_ready)
                local_protocol_error <= 1'b1;
            if (step_accept) begin
                running <= 1'b1;
                done_pending <= 1'b0;
                accepted_events <= 0;
            end else if (running) begin
                if (i_groups_done && i_group_valid)
                    local_protocol_error <= 1'b1;
                if (accept_group) begin
                    if (i_bank_valid == 0)
                        local_protocol_error <= 1'b1;
                    for (check_bank = 0; check_bank < 8; check_bank = check_bank + 1)
                        if (i_bank_valid[check_bank] &&
                            feature_bank(i_fi_flat[check_bank*FI_BITS +: FI_BITS]) != check_bank)
                            local_protocol_error <= 1'b1;
                    bank_valid_q <= i_bank_valid;
                    bank_x_q <= i_x_flat;
                    accepted_events <= accepted_events + popcount8(i_bank_valid);
                end
                if (i_groups_done)
                    done_pending <= 1'b1;
                if (step_close) begin
                    running <= 1'b0;
                    done_pending <= 1'b0;
                    accepted_events_by_bank[write_bank] <= accepted_events;
                end
            end
        end
    end

    initial begin
        if (IN_DIM % 8 != 0)
            $error("N8D2B K8 W0 requires IN_DIM divisible by eight");
        if (PHYSICAL_BANK_DEPTH < (IN_DIM / 8))
            $error("FC2312 physical W0 depth is smaller than logical rows");
        if (PHYSICAL_BANK_DEPTH != 512)
            $error("FC2312 replica W0 contract requires 512 physical rows");
        if (FI_BITS < 9)
            $error("N8D2B K8 W0 requires FI_BITS >= 9");
        if (EVENT_X_IS_ONE != 0 && EVENT_X_IS_SIGNED != 0)
            $error("EVENT_X_IS_ONE and EVENT_X_IS_SIGNED are mutually exclusive");
    end
endmodule
