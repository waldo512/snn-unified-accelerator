`default_nettype none

module unified_scnn_q8_8_versioned_trace_state_engine #(
    parameter int NUM_NEURONS = 480,
    parameter int ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter STATE_INIT_FILE = "NONE"
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_abort,
    input  logic                           i_valid,
    output logic                           o_ready,
    input  logic [ADDR_WIDTH-1:0]          i_addr,
    input  logic [VERSION_WIDTH-1:0]       i_state_epoch,
    input  logic signed [15:0]             i_spike_q8_8,
    input  logic signed [15:0]             i_leak_q8_8,
    output logic                           o_valid,
    input  logic                           i_ready,
    output logic [ADDR_WIDTH-1:0]          o_addr,
    output logic [VERSION_WIDTH-1:0]       o_state_epoch,
    output logic signed [15:0]             o_trace_q8_8,
    output logic                           o_idle,
    output logic                           o_address_error,
    output logic                           o_zero_epoch_error
);
    localparam int MEMORY_WIDTH = VERSION_WIDTH + 16;
    localparam int RAM_BLOCK_MAX_DEPTH = 16384;
    localparam int RAM_BLOCK_DEPTH = NUM_NEURONS < RAM_BLOCK_MAX_DEPTH ?
        NUM_NEURONS : RAM_BLOCK_MAX_DEPTH;
    localparam int RAM_BLOCKS =
        (NUM_NEURONS + RAM_BLOCK_DEPTH - 1) / RAM_BLOCK_DEPTH;
    localparam int RAM_BLOCK_ADDR_WIDTH = RAM_BLOCK_DEPTH <= 1 ? 1 :
        $clog2(RAM_BLOCK_DEPTH);
    localparam int RAM_BLOCK_INDEX_WIDTH = RAM_BLOCKS <= 1 ? 1 :
        $clog2(RAM_BLOCKS);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE_MULTIPLY,
        STATE_WAIT_MULTIPLY,
        STATE_WAIT_OUTPUT
    } state_t;

    state_t state_q;
    logic [MEMORY_WIDTH-1:0] ram_read_data [0:RAM_BLOCKS-1];
    logic [MEMORY_WIDTH-1:0] state_read_data;
    logic [ADDR_WIDTH-1:0] addr_q;
    logic [VERSION_WIDTH-1:0] epoch_q;
    logic signed [15:0] spike_q;
    logic signed [15:0] leak_q;
    logic signed [15:0] previous_trace;
    logic signed [31:0] leak_product;
    logic signed [31:0] trace_sum_wide;
    logic signed [15:0] result_trace_q;
    logic multiplier_ready;
    logic multiplier_valid;
    logic multiplier_accept;
    logic multiplier_result_accept;
    logic request_fire;
    logic result_accept;
    logic address_in_range;
    logic [RAM_BLOCK_INDEX_WIDTH-1:0] input_block_index;
    logic [RAM_BLOCK_INDEX_WIDTH-1:0] latched_block_index;
    logic [RAM_BLOCK_ADDR_WIDTH-1:0] input_block_addr;
    logic [RAM_BLOCK_ADDR_WIDTH-1:0] latched_block_addr;

    assign address_in_range = {1'b0, i_addr} <
        (ADDR_WIDTH+1)'(NUM_NEURONS);
    assign o_ready = state_q == STATE_IDLE && address_in_range &&
        i_state_epoch != 0;
    assign request_fire = state_q == STATE_IDLE && i_valid && o_ready;
    assign previous_trace =
        state_read_data[MEMORY_WIDTH-1:16] == epoch_q ?
            $signed(state_read_data[15:0]) : 16'sd0;
    assign multiplier_accept = state_q == STATE_ISSUE_MULTIPLY &&
        multiplier_ready;
    assign multiplier_result_accept = state_q == STATE_WAIT_MULTIPLY &&
        multiplier_valid;
    assign trace_sum_wide = (leak_product >>> 8) + spike_q;
    assign o_valid = state_q == STATE_WAIT_OUTPUT;
    assign result_accept = o_valid && i_ready;
    assign o_addr = addr_q;
    assign o_state_epoch = epoch_q;
    assign o_trace_q8_8 = result_trace_q;
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
                .i_write_data({epoch_q, result_trace_q}),
                .o_read_data(ram_read_data[block])
            );
        end
    endgenerate

    unified_q8_8_mul_multipass u_trace_leak_multiply (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(state_q == STATE_ISSUE_MULTIPLY),
        .o_ready(multiplier_ready),
        .i_a(previous_trace), .i_b(leak_q),
        .o_valid(multiplier_valid),
        .i_ready(state_q == STATE_WAIT_MULTIPLY),
        .o_product(leak_product)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            addr_q <= '0;
            epoch_q <= '0;
            spike_q <= '0;
            leak_q <= '0;
            result_trace_q <= '0;
            o_address_error <= 1'b0;
            o_zero_epoch_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            addr_q <= '0;
            epoch_q <= '0;
            spike_q <= '0;
            leak_q <= '0;
            result_trace_q <= '0;
        end else begin
            if (i_valid && !address_in_range)
                o_address_error <= 1'b1;
            if (i_valid && i_state_epoch == 0)
                o_zero_epoch_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (request_fire) begin
                        addr_q <= i_addr;
                        epoch_q <= i_state_epoch;
                        spike_q <= i_spike_q8_8;
                        leak_q <= i_leak_q8_8;
                        state_q <= STATE_ISSUE_MULTIPLY;
                    end
                end
                STATE_ISSUE_MULTIPLY: begin
                    if (multiplier_accept)
                        state_q <= STATE_WAIT_MULTIPLY;
                end
                STATE_WAIT_MULTIPLY: begin
                    if (multiplier_result_accept) begin
                        result_trace_q <= trace_sum_wide[15:0];
                        state_q <= STATE_WAIT_OUTPUT;
                    end
                end
                STATE_WAIT_OUTPUT: begin
                    if (result_accept)
                        state_q <= STATE_IDLE;
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
