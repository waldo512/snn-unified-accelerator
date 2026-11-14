`default_nettype none

// Twelve independent banks retain one output-channel wave across scheduler
// input groups. Each bank needs one synchronous read/write port only.
module unified_scnn_wave_partial_sum_banks_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int SPATIAL_DEPTH = 32768,
    parameter int SPATIAL_ADDR_WIDTH = $clog2(SPATIAL_DEPTH),
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_valid,
    output logic                                      o_ready,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]             i_spatial_addr,
    input  logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      i_output_addr_flat,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_partial_flat,
    input  logic                                      i_accumulator_clear,
    input  logic                                      i_accumulator_finalize,
    output logic                                      o_valid,
    input  logic                                      i_ready,
    output logic [ENGINES-1:0]                        o_engine_mask,
    output logic [SPATIAL_ADDR_WIDTH-1:0]             o_spatial_addr,
    output logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      o_output_addr_flat,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_accumulator_flat,
    output logic                                      o_idle,
    output logic                                      o_sticky_error
);
    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_READ,
        STATE_ACCUMULATE,
        STATE_OUTPUT
    } state_t;

    localparam int RAM_BLOCK_MAX_DEPTH = 16384;
    localparam int RAM_BLOCK_DEPTH =
        SPATIAL_DEPTH < RAM_BLOCK_MAX_DEPTH ?
            SPATIAL_DEPTH : RAM_BLOCK_MAX_DEPTH;
    localparam int RAM_BLOCKS =
        (SPATIAL_DEPTH + RAM_BLOCK_DEPTH - 1) / RAM_BLOCK_DEPTH;
    localparam int RAM_BLOCK_ADDR_WIDTH =
        RAM_BLOCK_DEPTH <= 1 ? 1 : $clog2(RAM_BLOCK_DEPTH);
    localparam int RAM_BLOCK_INDEX_WIDTH =
        RAM_BLOCKS <= 1 ? 1 : $clog2(RAM_BLOCKS);

    logic signed [ACC_WIDTH-1:0]
        ram_read_data [0:ENGINES-1][0:RAM_BLOCKS-1];
    logic signed [ACC_WIDTH-1:0] selected_read_data [0:ENGINES-1];
    logic [ENGINES-1:0] engine_mask_q;
    logic [SPATIAL_ADDR_WIDTH-1:0] spatial_addr_q;
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0] output_addr_flat_q;
    logic [ENGINES*ACC_WIDTH-1:0] partial_flat_q;
    logic clear_q;
    logic finalize_q;
    state_t state_q;
    logic request_legal;
    logic clear_write_fire;
    logic accumulate_write_fire;
    logic [RAM_BLOCK_INDEX_WIDTH-1:0] latched_block_index;
    logic [RAM_BLOCK_ADDR_WIDTH-1:0] latched_block_addr;

    assign request_legal = i_engine_mask != '0 &&
        {1'b0, i_spatial_addr} <
            (SPATIAL_ADDR_WIDTH+1)'(SPATIAL_DEPTH);
    assign o_ready = state_q == STATE_IDLE && !o_sticky_error;
    assign o_valid = state_q == STATE_OUTPUT;
    assign o_engine_mask = engine_mask_q;
    assign o_spatial_addr = spatial_addr_q;
    assign o_output_addr_flat = output_addr_flat_q;
    assign o_idle = state_q == STATE_IDLE;
    assign clear_write_fire =
        state_q == STATE_ACCUMULATE && clear_q && !finalize_q;
    assign accumulate_write_fire =
        state_q == STATE_ACCUMULATE && !clear_q && !finalize_q;
    assign latched_block_index = RAM_BLOCK_INDEX_WIDTH'(
        spatial_addr_q >> RAM_BLOCK_ADDR_WIDTH);
    assign latched_block_addr =
        spatial_addr_q[RAM_BLOCK_ADDR_WIDTH-1:0];

    always_comb begin
        for (integer engine = 0; engine < ENGINES; engine++)
            selected_read_data[engine] =
                ram_read_data[engine][latched_block_index];
    end

    generate
        for (genvar engine = 0; engine < ENGINES; engine++) begin : g_engine
            for (genvar block = 0; block < RAM_BLOCKS; block++) begin : g_block
                logic block_read_enable;
                logic block_write_enable;
                logic [RAM_BLOCK_ADDR_WIDTH-1:0] block_addr;
                logic signed [ACC_WIDTH-1:0] block_write_data;

                assign block_read_enable = state_q == STATE_READ &&
                    engine_mask_q[engine] &&
                    latched_block_index == RAM_BLOCK_INDEX_WIDTH'(block);
                assign block_write_enable =
                    (clear_write_fire && engine_mask_q[engine] &&
                     latched_block_index == RAM_BLOCK_INDEX_WIDTH'(block)) ||
                    (accumulate_write_fire && engine_mask_q[engine] &&
                     latched_block_index == RAM_BLOCK_INDEX_WIDTH'(block));
                assign block_addr = latched_block_addr;
                assign block_write_data = clear_write_fire ?
                    $signed(partial_flat_q[
                        engine*ACC_WIDTH +: ACC_WIDTH]) :
                    selected_read_data[engine] +
                    $signed(partial_flat_q[engine*ACC_WIDTH +: ACC_WIDTH]);

                unified_scnn_wave_partial_sum_ram_soft_abort_v1 #(
                    .DEPTH(RAM_BLOCK_DEPTH),
                    .ADDR_WIDTH(RAM_BLOCK_ADDR_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_ram (
                    .i_clk(i_clk),
                    .i_read_enable(block_read_enable),
                    .i_write_enable(block_write_enable),
                    .i_addr(block_addr),
                    .i_write_data(block_write_data),
                    .o_read_data(ram_read_data[engine][block])
                );
            end
        end
    endgenerate

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            o_accumulator_flat <= '0;
            clear_q <= 1'b0;
            finalize_q <= 1'b0;
            state_q <= STATE_IDLE;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            o_accumulator_flat <= '0;
            clear_q <= 1'b0;
            finalize_q <= 1'b0;
            state_q <= STATE_IDLE;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (i_valid && !request_legal) begin
                        o_sticky_error <= 1'b1;
                    end else if (i_valid && o_ready) begin
                        clear_q <= i_accumulator_clear;
                        finalize_q <= i_accumulator_finalize;
                        if (i_accumulator_finalize)
                            o_accumulator_flat <= '0;
                        if (i_accumulator_clear &&
                            i_accumulator_finalize) begin
                                o_accumulator_flat <= i_partial_flat;
                                state_q <= STATE_OUTPUT;
                        end else if (i_accumulator_clear) begin
                            state_q <= STATE_ACCUMULATE;
                        end else begin
                            state_q <= STATE_READ;
                        end
                    end
                end
                STATE_READ: begin
                    state_q <= STATE_ACCUMULATE;
                end
                STATE_ACCUMULATE: begin
                    if (!clear_q) begin
                        for (integer engine = 0;
                             engine < ENGINES; engine++) begin
                            if (engine_mask_q[engine] && finalize_q)
                                o_accumulator_flat[
                                    engine*ACC_WIDTH +: ACC_WIDTH] <=
                                    selected_read_data[engine] +
                                    $signed(partial_flat_q[
                                        engine*ACC_WIDTH +: ACC_WIDTH]);
                        end
                    end
                    if (finalize_q)
                        state_q <= STATE_OUTPUT;
                    else
                        state_q <= STATE_IDLE;
                end
                STATE_OUTPUT: begin
                    if (i_ready) begin
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

    // Payload fields are reset-free; state_q is the sole validity authority.
    // Every BRAM port therefore sees only a locally registered command, never
    // a wide top-level mux or asynchronous abort/reset cone.
    always_ff @(posedge i_clk) begin
        if (state_q == STATE_IDLE && i_valid && o_ready && request_legal) begin
            engine_mask_q <= i_engine_mask;
            spatial_addr_q <= i_spatial_addr;
            output_addr_flat_q <= i_output_addr_flat;
            partial_flat_q <= i_partial_flat;
        end
    end
endmodule

// Keep every inferred array below the Vivado 2022.2 single-variable limit.
// Simultaneous read and write is not requested by the parent protocol.
/* verilator lint_off DECLFILENAME */
module unified_scnn_wave_partial_sum_ram_soft_abort_v1 #(
    parameter int DEPTH = 16384,
    parameter int ADDR_WIDTH = $clog2(DEPTH),
    parameter int ACC_WIDTH = 48
) (
    input  logic                        i_clk,
    input  logic                        i_read_enable,
    input  logic                        i_write_enable,
    input  logic [ADDR_WIDTH-1:0]       i_addr,
    input  logic signed [ACC_WIDTH-1:0] i_write_data,
    output logic signed [ACC_WIDTH-1:0] o_read_data
);
`ifdef UNIFIED_USE_XPM_MEMORY
    logic [ACC_WIDTH-1:0] xpm_read_data;

    assign o_read_data = $signed(xpm_read_data);

    xpm_memory_spram #(
        .MEMORY_SIZE(DEPTH * ACC_WIDTH),
        .MEMORY_PRIMITIVE("block"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM(""),
        .USE_MEM_INIT(0),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep"),
        .AUTO_SLEEP_TIME(0),
        .MESSAGE_CONTROL(0),
        .MEMORY_OPTIMIZATION("true"),
        .CASCADE_HEIGHT(0),
        .SIM_ASSERT_CHK(0),
        .WRITE_PROTECT(1),
        .IGNORE_INIT_SYNTH(0),
        .WRITE_DATA_WIDTH_A(ACC_WIDTH),
        .READ_DATA_WIDTH_A(ACC_WIDTH),
        .BYTE_WRITE_WIDTH_A(ACC_WIDTH),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .READ_RESET_VALUE_A("0"),
        .READ_LATENCY_A(1),
        .WRITE_MODE_A("read_first"),
        .RST_MODE_A("SYNC")
    ) u_partial_sum_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .rsta(1'b0),
        .ena(i_read_enable || i_write_enable),
        .regcea(1'b1),
        .wea(i_write_enable),
        .addra(i_addr),
        .dina(i_write_data),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .douta(xpm_read_data),
        .sbiterra(),
        .dbiterra()
    );
`else
    (* ram_style = "block" *) logic signed [ACC_WIDTH-1:0]
        memory [0:DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (i_write_enable)
            memory[i_addr] <= i_write_data;
        if (i_read_enable)
            o_read_data <= memory[i_addr];
    end
`endif
endmodule
/* verilator lint_on DECLFILENAME */

`default_nettype wire
