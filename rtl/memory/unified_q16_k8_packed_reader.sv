`default_nettype none

module unified_q16_k8_packed_reader #(
    parameter int Q64_ADDRESS_WIDTH = 14,
    parameter int Q16_ADDRESS_WIDTH = Q64_ADDRESS_WIDTH + 2
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,

    input  logic                           i_request_valid,
    output logic                           o_request_ready,
    input  logic [Q16_ADDRESS_WIDTH-1:0]   i_q16_base,
    input  logic [7:0]                     i_lane_mask,

    output logic                           o_read_valid,
    input  logic                           i_read_ready,
    output logic [Q64_ADDRESS_WIDTH-1:0]   o_read_word_addr0,
    output logic [Q64_ADDRESS_WIDTH-1:0]   o_read_word_addr1,
    input  logic                           i_read_result_valid,
    output logic                           o_read_result_ready,
    input  logic [63:0]                    i_read_data0,
    input  logic [63:0]                    i_read_data1,

    output logic                           o_weight_valid,
    input  logic                           i_weight_ready,
    output logic [127:0]                   o_weight_flat,
    output logic [7:0]                     o_lane_mask
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_ISSUE_FIRST,
        STATE_WAIT_FIRST,
        STATE_ISSUE_THIRD,
        STATE_WAIT_THIRD,
        STATE_OUTPUT
    } state_t;

    state_t state_q;
    logic [Q64_ADDRESS_WIDTH-1:0] base_word_q;
    logic [1:0] q16_offset_q;
    logic [7:0] lane_mask_q;
    logic [63:0] first_word0_q;
    logic [63:0] first_word1_q;
    logic [127:0] weight_q;
    logic [191:0] unaligned_window;

    assign o_request_ready = state_q == STATE_IDLE;
    assign o_read_valid = state_q == STATE_ISSUE_FIRST ||
                          state_q == STATE_ISSUE_THIRD;
    assign o_read_word_addr0 = state_q == STATE_ISSUE_THIRD ?
                               base_word_q + Q64_ADDRESS_WIDTH'(2) : base_word_q;
    assign o_read_word_addr1 = state_q == STATE_ISSUE_THIRD ?
                               base_word_q + Q64_ADDRESS_WIDTH'(2) :
                               base_word_q + Q64_ADDRESS_WIDTH'(1);
    assign o_read_result_ready = state_q == STATE_WAIT_FIRST ||
                                 state_q == STATE_WAIT_THIRD;
    assign o_weight_valid = state_q == STATE_OUTPUT;
    assign o_weight_flat = weight_q;
    assign o_lane_mask = lane_mask_q;
    assign unaligned_window = {i_read_data0, first_word1_q, first_word0_q};

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            base_word_q <= '0;
            q16_offset_q <= '0;
            lane_mask_q <= '0;
            first_word0_q <= '0;
            first_word1_q <= '0;
            weight_q <= '0;
        end else begin
            case (state_q)
                STATE_IDLE: begin
                    if (i_request_valid) begin
                        base_word_q <= i_q16_base[Q16_ADDRESS_WIDTH-1:2];
                        q16_offset_q <= i_q16_base[1:0];
                        lane_mask_q <= i_lane_mask;
                        state_q <= STATE_ISSUE_FIRST;
                    end
                end
                STATE_ISSUE_FIRST: begin
                    if (i_read_ready)
                        state_q <= STATE_WAIT_FIRST;
                end
                STATE_WAIT_FIRST: begin
                    if (i_read_result_valid) begin
                        if (q16_offset_q == 0) begin
                            weight_q <= {i_read_data1, i_read_data0};
                            state_q <= STATE_OUTPUT;
                        end else begin
                            first_word0_q <= i_read_data0;
                            first_word1_q <= i_read_data1;
                            state_q <= STATE_ISSUE_THIRD;
                        end
                    end
                end
                STATE_ISSUE_THIRD: begin
                    if (i_read_ready)
                        state_q <= STATE_WAIT_THIRD;
                end
                STATE_WAIT_THIRD: begin
                    if (i_read_result_valid) begin
                        weight_q <= 128'(unaligned_window >>
                                             (q16_offset_q * 16));
                        state_q <= STATE_OUTPUT;
                    end
                end
                STATE_OUTPUT: begin
                    if (i_weight_ready)
                        state_q <= STATE_IDLE;
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
