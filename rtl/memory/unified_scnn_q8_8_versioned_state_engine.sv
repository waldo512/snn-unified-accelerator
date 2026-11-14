`default_nettype none

module unified_scnn_q8_8_versioned_state_engine #(
    parameter int NUM_NEURONS = 409600,
    parameter int ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
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

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE_STEP,
        STATE_WAIT_RESULT
    } state_t;

    (* ram_style = "block" *) logic [MEMORY_WIDTH-1:0]
        state_memory [0:NUM_NEURONS-1];
    state_t state_q;
    logic [MEMORY_WIDTH-1:0] state_read_q;
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
    integer init_index;

    // FPGA block-RAM initialization only. Runtime sample changes use version
    // tags and never scan-clear this memory.
    initial begin
        for (init_index = 0; init_index < NUM_NEURONS;
             init_index = init_index + 1)
            state_memory[init_index] = '0;
    end

    // Keep the capacity comparison one bit wider than the address. Casting a
    // power-of-two NUM_NEURONS to ADDR_WIDTH bits would truncate it to zero.
    assign address_in_range = {1'b0, i_addr} <
                              (ADDR_WIDTH+1)'(NUM_NEURONS);
    assign o_ready = state_q == STATE_IDLE && address_in_range &&
                     i_sample_version != 0;
    assign previous_membrane =
        state_read_q[MEMORY_WIDTH-1:16] == version_q ?
        $signed(state_read_q[15:0]) : 16'sd0;
    assign step_accept = state_q == STATE_ISSUE_STEP && step_ready;
    assign result_accept = state_q == STATE_WAIT_RESULT && step_valid && i_ready;
    assign o_valid = state_q == STATE_WAIT_RESULT && step_valid;
    assign o_addr = addr_q;
    assign o_sample_version = version_q;
    assign o_idle = state_q == STATE_IDLE;

    unified_scnn_q8_8_lif_step u_lif_step (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(state_q == STATE_ISSUE_STEP), .o_ready(step_ready),
        .i_current(current_q), .i_membrane_previous(previous_membrane),
        .i_leak_q8_8(leak_q), .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .o_valid(step_valid), .i_ready(state_q == STATE_WAIT_RESULT && i_ready),
        .o_spike_q8_8(o_spike_q8_8),
        .o_membrane_pre(o_membrane_pre),
        .o_membrane_next(o_membrane_next)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            state_read_q <= '0;
            addr_q <= '0;
            version_q <= '0;
            current_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            o_address_error <= 1'b0;
            o_zero_version_error <= 1'b0;
        end else begin
            if (i_valid && !address_in_range)
                o_address_error <= 1'b1;
            if (i_valid && i_sample_version == 0)
                o_zero_version_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (i_valid && o_ready) begin
                        state_read_q <= state_memory[i_addr];
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
                        state_memory[addr_q] <= {version_q, o_membrane_next};
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
