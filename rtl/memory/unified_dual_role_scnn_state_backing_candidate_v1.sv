`timescale 1ns/1ps
`default_nettype none

// One tile's packed SCNN membrane-state backing store. Every legal timestep
// visits and overwrites all words in ascending order. Timestep zero of a new
// sequence ignores old RAM contents, so abort recovery does not need a version
// tag per neuron: an abort simply requires a new sequence restart at t=0.
module unified_dual_role_scnn_state_backing_candidate_v1 #(
    parameter int LANES = 40,
    parameter int STATE_WIDTH = 16,
    parameter int WORDS_PER_TILE = 1008,
    parameter int TIMESTEP_WIDTH = 9,
    parameter int ADDR_WIDTH = $clog2(WORDS_PER_TILE)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,

    input  wire logic                         i_step_start,
    output logic                              o_step_start_ready,
    input  wire logic                         i_new_sequence,
    input  wire logic [TIMESTEP_WIDTH-1:0]    i_timestep,
    output logic                              o_step_active,
    output logic                              o_zero_fill_active,
    output logic                              o_step_done,

    input  wire logic                         i_read_valid,
    output logic                              o_read_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_read_addr,
    output logic                              o_read_result_valid,
    input  wire logic                         i_read_result_ready,
    output logic [ADDR_WIDTH-1:0]             o_read_result_addr,
    output logic [LANES*STATE_WIDTH-1:0]      o_read_state,

    input  wire logic                         i_write_valid,
    output logic                              o_write_ready,
    input  wire logic [ADDR_WIDTH-1:0]        i_write_addr,
    input  wire logic [LANES*STATE_WIDTH-1:0] i_write_state,
    input  wire logic                         i_write_last,

    output logic                              o_restart_required,
    output logic                              o_protocol_error
);
    localparam int WORD_WIDTH = LANES * STATE_WIDTH;
    localparam int COUNT_WIDTH = $clog2(WORDS_PER_TILE + 1);

    logic step_active_q;
    logic zero_fill_q;
    logic restart_required_q;
    logic [TIMESTEP_WIDTH-1:0] expected_timestep_q;
    logic [COUNT_WIDTH-1:0] write_count_q;
    logic read_result_valid_q;
    logic [ADDR_WIDTH-1:0] read_addr_q;
    logic read_zero_fill_q;
    logic [WORD_WIDTH-1:0] memory_read_data;
    logic step_start_legal;
    logic read_fire;
    logic write_fire;
    logic write_legal;

    assign step_start_legal = i_new_sequence ? i_timestep == '0 :
        !restart_required_q && i_timestep == expected_timestep_q;
    assign o_step_start_ready = !step_active_q && !read_result_valid_q &&
                                step_start_legal;
    assign o_step_active = step_active_q;
    assign o_zero_fill_active = step_active_q && zero_fill_q;
    assign o_restart_required = restart_required_q;

    assign o_read_ready = step_active_q &&
                          (!read_result_valid_q || i_read_result_ready) &&
                          integer'(i_read_addr) < WORDS_PER_TILE;
    assign read_fire = i_read_valid && o_read_ready;
    assign o_read_result_valid = read_result_valid_q;
    assign o_read_result_addr = read_addr_q;
    assign o_read_state = read_zero_fill_q ? '0 : memory_read_data;

    assign o_write_ready = step_active_q;
    assign write_fire = i_write_valid && o_write_ready;
    assign write_legal = integer'(i_write_addr) < WORDS_PER_TILE &&
        COUNT_WIDTH'(i_write_addr) == write_count_q &&
        i_write_last == (write_count_q == COUNT_WIDTH'(WORDS_PER_TILE - 1));

`ifdef UNIFIED_USE_XPM_MEMORY
    xpm_memory_sdpram #(
        .MEMORY_SIZE(WORDS_PER_TILE * WORD_WIDTH),
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
        .clka(i_clk), .ena(write_fire && write_legal),
        .wea(write_fire && write_legal), .addra(i_write_addr),
        .dina(i_write_state),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .clkb(i_clk), .rstb(1'b0), .enb(read_fire), .regceb(1'b1),
        .addrb(i_read_addr), .doutb(memory_read_data),
        .sbiterrb(), .dbiterrb()
    );
`else
    logic [WORD_WIDTH-1:0] memory [0:WORDS_PER_TILE-1];

    always_ff @(posedge i_clk) begin
        if (write_fire && write_legal)
            memory[i_write_addr] <= i_write_state;
        if (read_fire)
            memory_read_data <= memory[i_read_addr];
    end
`endif

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            step_active_q <= 1'b0;
            zero_fill_q <= 1'b0;
            restart_required_q <= 1'b1;
            expected_timestep_q <= '0;
            write_count_q <= '0;
            read_result_valid_q <= 1'b0;
            read_addr_q <= '0;
            read_zero_fill_q <= 1'b0;
            o_step_done <= 1'b0;
            o_protocol_error <= 1'b0;
        end else if (i_abort) begin
            step_active_q <= 1'b0;
            zero_fill_q <= 1'b0;
            restart_required_q <= 1'b1;
            expected_timestep_q <= '0;
            write_count_q <= '0;
            read_result_valid_q <= 1'b0;
            read_addr_q <= '0;
            read_zero_fill_q <= 1'b0;
            o_step_done <= 1'b0;
        end else begin
            o_step_done <= 1'b0;

            if (i_step_start && !o_step_start_ready)
                o_protocol_error <= 1'b1;
            if (i_step_start && o_step_start_ready) begin
                step_active_q <= 1'b1;
                zero_fill_q <= i_new_sequence;
                write_count_q <= '0;
            end

            if (read_result_valid_q && i_read_result_ready)
                read_result_valid_q <= 1'b0;
            if (read_fire) begin
                read_result_valid_q <= 1'b1;
                read_addr_q <= i_read_addr;
                read_zero_fill_q <= zero_fill_q;
            end else if (i_read_valid && step_active_q &&
                         integer'(i_read_addr) >= WORDS_PER_TILE) begin
                o_protocol_error <= 1'b1;
            end

            if (write_fire) begin
                if (!write_legal) begin
                    o_protocol_error <= 1'b1;
                    step_active_q <= 1'b0;
                    restart_required_q <= 1'b1;
                    write_count_q <= '0;
                end else if (i_write_last) begin
                    step_active_q <= 1'b0;
                    zero_fill_q <= 1'b0;
                    restart_required_q <= 1'b0;
                    expected_timestep_q <= i_timestep + 1'b1;
                    write_count_q <= '0;
                    o_step_done <= 1'b1;
                end else begin
                    write_count_q <= write_count_q + 1'b1;
                end
            end
        end
    end

    initial begin
        if (LANES != 40 || STATE_WIDTH != 16)
            $error("SCNN state candidate requires 40 lanes of Q8.8 state");
    end

endmodule

`default_nettype wire
