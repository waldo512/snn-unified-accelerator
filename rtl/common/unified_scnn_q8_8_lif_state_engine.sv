`default_nettype none

module unified_scnn_q8_8_lif_state_engine #(
    parameter int NUM_NEURONS = 480,
    parameter int ADDR_WIDTH = 17
) (
    input  logic                  i_clk,
    input  logic                  i_rst_n,
    input  logic                  i_sample_reset,
    input  logic                  i_valid,
    output logic                  o_ready,
    input  logic [ADDR_WIDTH-1:0] i_addr,
    input  logic signed [15:0]    i_current,
    input  logic signed [15:0]    i_leak_q8_8,
    input  logic signed [15:0]    i_threshold_q8_8,
    input  logic                  i_reset_to_zero,
    output logic                  o_valid,
    input  logic                  i_ready,
    output logic [ADDR_WIDTH-1:0] o_addr,
    output logic [15:0]           o_spike_q8_8,
    output logic signed [15:0]    o_membrane_pre,
    output logic signed [15:0]    o_membrane_next,
    output logic                  o_idle,
    output logic                  o_reset_while_busy_error,
    output logic                  o_address_error
);

    logic signed [15:0] state_memory [0:NUM_NEURONS-1];
    logic               state_valid [0:NUM_NEURONS-1];
    logic [ADDR_WIDTH-1:0] transaction_addr_q;
    logic signed [15:0] selected_state;
    logic address_in_range;
    logic step_ready;
    logic step_valid;
    logic step_accept;
    logic result_accept;
    integer state_index;

    assign address_in_range = {1'b0, i_addr} < (ADDR_WIDTH+1)'(NUM_NEURONS);
    assign selected_state = (address_in_range && state_valid[i_addr]) ?
                            state_memory[i_addr] : 16'sd0;
    assign o_ready = step_ready && !i_sample_reset && address_in_range;
    assign step_accept = i_valid && o_ready;
    assign result_accept = step_valid && i_ready;
    assign o_valid = step_valid;
    assign o_addr = transaction_addr_q;
    assign o_idle = step_ready && !step_valid;

    unified_scnn_q8_8_lif_step u_lif_step (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(step_accept), .o_ready(step_ready),
        .i_current(i_current), .i_membrane_previous(selected_state),
        .i_leak_q8_8(i_leak_q8_8),
        .i_threshold_q8_8(i_threshold_q8_8),
        .i_reset_to_zero(i_reset_to_zero),
        .o_valid(step_valid), .i_ready(i_ready),
        .o_spike_q8_8(o_spike_q8_8),
        .o_membrane_pre(o_membrane_pre),
        .o_membrane_next(o_membrane_next)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            transaction_addr_q <= '0;
            o_reset_while_busy_error <= 1'b0;
            o_address_error <= 1'b0;
            for (state_index = 0; state_index < NUM_NEURONS;
                 state_index = state_index + 1) begin
                state_valid[state_index] <= 1'b0;
            end
        end else begin
            if (i_valid && !address_in_range)
                o_address_error <= 1'b1;

            if (step_accept)
                transaction_addr_q <= i_addr;

            if (result_accept) begin
                state_memory[transaction_addr_q] <= o_membrane_next;
                state_valid[transaction_addr_q] <= 1'b1;
            end

            if (i_sample_reset) begin
                if (!o_idle) begin
                    o_reset_while_busy_error <= 1'b1;
                end else begin
                    for (state_index = 0; state_index < NUM_NEURONS;
                         state_index = state_index + 1) begin
                        state_valid[state_index] <= 1'b0;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
