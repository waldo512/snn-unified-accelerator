`default_nettype none

module unified_scnn_q8_8_versioned_state_engine_soft_abort_v1 #(
    parameter int NUM_NEURONS = 409600,
    parameter int ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter STATE_INIT_FILE = ""
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_abort,
    input  logic                           i_valid,
    output logic                           o_ready,
    input  logic [ADDR_WIDTH-1:0]          i_addr,
    input  logic [VERSION_WIDTH-1:0]       i_sample_version,
    input  logic signed [15:0]             i_current,
    input  logic signed [15:0]             i_leak_q8_8,
    input  logic signed [15:0]             i_threshold_q8_8,
    input  logic                           i_reset_to_zero,
    output logic                           o_valid,
    input  logic                           i_ready,
    output logic [ADDR_WIDTH-1:0]          o_addr,
    output logic [VERSION_WIDTH-1:0]       o_sample_version,
    output logic [15:0]                    o_spike_q8_8,
    output logic signed [15:0]             o_membrane_pre,
    output logic signed [15:0]             o_membrane_next,
    output logic                           o_idle,
    output logic                           o_address_error,
    output logic                           o_zero_version_error
);

    localparam int MEMORY_WIDTH = VERSION_WIDTH + 16;
    localparam int RAM_BLOCK_MAX_DEPTH = 16384;
    localparam int RAM_BLOCK_DEPTH =
        NUM_NEURONS < RAM_BLOCK_MAX_DEPTH ?
            NUM_NEURONS : RAM_BLOCK_MAX_DEPTH;
    localparam int RAM_BLOCKS =
        (NUM_NEURONS + RAM_BLOCK_DEPTH - 1) / RAM_BLOCK_DEPTH;
    localparam int RAM_BLOCK_ADDR_WIDTH =
        RAM_BLOCK_DEPTH <= 1 ? 1 : $clog2(RAM_BLOCK_DEPTH);
    localparam int RAM_BLOCK_INDEX_WIDTH =
        RAM_BLOCKS <= 1 ? 1 : $clog2(RAM_BLOCKS);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE_STEP,
        STATE_WAIT_RESULT
    } state_t;

    state_t state_q;
    logic [MEMORY_WIDTH-1:0] ram_read_data [0:RAM_BLOCKS-1];
    logic [MEMORY_WIDTH-1:0] state_read_data;
    logic [ADDR_WIDTH-1:0] addr_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic signed [15:0] current_q;
    logic signed [15:0] leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic signed [15:0] previous_membrane;
    logic step_ready;
    logic step_valid;
    logic step_accept;
    logic result_accept;
    logic address_in_range;
    logic request_fire;
    logic [RAM_BLOCK_INDEX_WIDTH-1:0] input_block_index;
    logic [RAM_BLOCK_INDEX_WIDTH-1:0] latched_block_index;
    logic [RAM_BLOCK_ADDR_WIDTH-1:0] input_block_addr;
    logic [RAM_BLOCK_ADDR_WIDTH-1:0] latched_block_addr;

    // Keep the capacity comparison one bit wider than the address. Casting a
    // power-of-two NUM_NEURONS to ADDR_WIDTH bits would truncate it to zero.
    assign address_in_range = {1'b0, i_addr} <
                              (ADDR_WIDTH+1)'(NUM_NEURONS);
    assign o_ready = state_q == STATE_IDLE && address_in_range &&
                     i_sample_version != 0;
    assign previous_membrane =
        state_read_data[MEMORY_WIDTH-1:16] == version_q ?
        $signed(state_read_data[15:0]) : 16'sd0;
    assign step_accept = state_q == STATE_ISSUE_STEP && step_ready;
    assign result_accept = state_q == STATE_WAIT_RESULT && step_valid && i_ready;
    assign request_fire = state_q == STATE_IDLE && i_valid && o_ready;
    assign o_valid = state_q == STATE_WAIT_RESULT && step_valid;
    assign o_addr = addr_q;
    assign o_sample_version = version_q;
    assign o_idle = state_q == STATE_IDLE;
    assign input_block_index = RAM_BLOCK_INDEX_WIDTH'(
        i_addr >> RAM_BLOCK_ADDR_WIDTH);
    assign latched_block_index = RAM_BLOCK_INDEX_WIDTH'(
        addr_q >> RAM_BLOCK_ADDR_WIDTH);
    assign input_block_addr = i_addr[RAM_BLOCK_ADDR_WIDTH-1:0];
    assign latched_block_addr = addr_q[RAM_BLOCK_ADDR_WIDTH-1:0];
    assign state_read_data = ram_read_data[latched_block_index];

    generate
        for (genvar block = 0; block < RAM_BLOCKS; block++) begin : g_block
            unified_scnn_versioned_state_ram_soft_abort_v1 #(
                .DEPTH(RAM_BLOCK_DEPTH),
                .ADDR_WIDTH(RAM_BLOCK_ADDR_WIDTH),
                .DATA_WIDTH(MEMORY_WIDTH),
                .INIT_FILE(STATE_INIT_FILE)
            ) u_ram (
                .i_clk(i_clk),
                .i_read_enable(request_fire &&
                    input_block_index == RAM_BLOCK_INDEX_WIDTH'(block)),
                .i_write_enable(result_accept &&
                    latched_block_index == RAM_BLOCK_INDEX_WIDTH'(block)),
                .i_addr(result_accept ? latched_block_addr : input_block_addr),
                .i_write_data({version_q, o_membrane_next}),
                .o_read_data(ram_read_data[block])
            );
        end
    endgenerate

    unified_scnn_q8_8_lif_step_soft_abort_v1 u_lif_step (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(state_q == STATE_ISSUE_STEP), .o_ready(step_ready),
        .i_current(current_q), .i_membrane_previous(previous_membrane),
        .i_leak_q8_8(leak_q), .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .o_valid(step_valid), .i_ready(state_q == STATE_WAIT_RESULT && i_ready),
        .o_spike_q8_8(o_spike_q8_8),
        .o_membrane_pre(o_membrane_pre),
        .o_membrane_next(o_membrane_next)
    );

    // Request state directly selects BRAM addresses and write enables. RAM
    // contents stay reset-free and are invalidated by version tags.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            addr_q <= '0;
            version_q <= '0;
            current_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            o_address_error <= 1'b0;
            o_zero_version_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            addr_q <= '0;
            version_q <= '0;
            current_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
        end else begin
            if (i_valid && !address_in_range)
                o_address_error <= 1'b1;
            if (i_valid && i_sample_version == 0)
                o_zero_version_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (i_valid && o_ready) begin
                        addr_q <= i_addr;
                        version_q <= i_sample_version;
                        current_q <= i_current;
                        leak_q <= i_leak_q8_8;
                        threshold_q <= i_threshold_q8_8;
                        reset_to_zero_q <= i_reset_to_zero;
                        state_q <= STATE_ISSUE_STEP;
                    end
                end
                STATE_ISSUE_STEP: begin
                    if (step_accept)
                        state_q <= STATE_WAIT_RESULT;
                end
                STATE_WAIT_RESULT: begin
                    if (result_accept) begin
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

// One synchronous 1RW block. Runtime reset is implemented by version tags;
// only FPGA configuration initializes the physical BRAM contents.
/* verilator lint_off DECLFILENAME */
module unified_scnn_versioned_state_ram_soft_abort_v1 #(
    parameter int DEPTH = 16384,
    parameter int ADDR_WIDTH = $clog2(DEPTH),
    parameter int DATA_WIDTH = 32,
    parameter INIT_FILE = ""
) (
    input  logic                       i_clk,
    input  logic                       i_read_enable,
    input  logic                       i_write_enable,
    input  logic [ADDR_WIDTH-1:0]      i_addr,
    input  logic [DATA_WIDTH-1:0]      i_write_data,
    output logic [DATA_WIDTH-1:0]      o_read_data
);
`ifdef UNIFIED_USE_XPM_MEMORY
    localparam XPM_INIT_FILE =
        (INIT_FILE == "" || INIT_FILE == "NONE") ? "none" : INIT_FILE;
    localparam integer XPM_USE_MEM_INIT =
        (INIT_FILE == "" || INIT_FILE == "NONE") ? 0 : 1;

    logic [DATA_WIDTH-1:0] xpm_read_data;

    assign o_read_data = xpm_read_data;

    xpm_memory_spram #(
        .MEMORY_SIZE(DEPTH * DATA_WIDTH),
        .MEMORY_PRIMITIVE("block"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(XPM_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .USE_MEM_INIT(XPM_USE_MEM_INIT),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep"),
        .AUTO_SLEEP_TIME(0),
        .MESSAGE_CONTROL(0),
        .MEMORY_OPTIMIZATION("true"),
        .CASCADE_HEIGHT(0),
        .SIM_ASSERT_CHK(0),
        .WRITE_PROTECT(1),
        .IGNORE_INIT_SYNTH(0),
        .WRITE_DATA_WIDTH_A(DATA_WIDTH),
        .READ_DATA_WIDTH_A(DATA_WIDTH),
        .BYTE_WRITE_WIDTH_A(DATA_WIDTH),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .READ_RESET_VALUE_A("0"),
        .READ_LATENCY_A(1),
        .WRITE_MODE_A("read_first"),
        .RST_MODE_A("SYNC")
    ) u_state_memory (
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
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];
    integer init_index;

    generate
        if (INIT_FILE == "NONE") begin : g_default_zero_init
            // 7-series block RAM configuration defaults to all-zero INIT.
        end else if (INIT_FILE != "") begin : g_file_init
            initial $readmemb(INIT_FILE, memory);
        end else begin : g_simulation_zero_init
            initial begin
                for (init_index = 0; init_index < DEPTH;
                     init_index = init_index + 1)
                    memory[init_index] = '0;
            end
        end
    endgenerate

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
