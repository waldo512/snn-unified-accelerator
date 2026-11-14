`default_nettype none

// Gathers arbitrary Q16 weights for one output-channel wave over a dual-Q64
// cache port. The first functional version prioritizes exact addressing.
module unified_scnn_q16_wave_weight_gather_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int Q64_ADDRESS_WIDTH = 22,
    parameter int Q16_ADDRESS_WIDTH = Q64_ADDRESS_WIDTH + 2
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_request_valid,
    output logic                                      o_request_ready,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [LANES-1:0]                          i_lane_mask,
    input  logic [ENGINES*LANES*Q16_ADDRESS_WIDTH-1:0] i_weight_q16_addr_flat,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic [Q64_ADDRESS_WIDTH-1:0]              o_read_word_addr0,
    output logic [Q64_ADDRESS_WIDTH-1:0]              o_read_word_addr1,
    input  logic                                      i_read_result_valid,
    output logic                                      o_read_result_ready,
    input  logic [63:0]                               i_read_data0,
    input  logic [63:0]                               i_read_data1,

    output logic                                      o_weight_valid,
    input  logic                                      i_weight_ready,
    output logic [ENGINES*LANES*16-1:0]               o_weight_flat,
    output logic [ENGINES-1:0]                        o_engine_mask,
    output logic [LANES-1:0]                          o_lane_mask,
    output logic                                      o_busy,
    output logic                                      o_sticky_error
);
    localparam int ITEMS = ENGINES * LANES;
    localparam int INDEX_WIDTH = $clog2(ITEMS + 1);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_SCAN,
        STATE_ISSUE,
        STATE_WAIT,
        STATE_OUTPUT
    } state_t;

    state_t state_q;
    logic [ENGINES-1:0] engine_mask_q;
    logic [LANES-1:0] lane_mask_q;
    logic [ENGINES*LANES*Q16_ADDRESS_WIDTH-1:0] address_flat_q;
    logic [ENGINES*LANES*16-1:0] weight_flat_q;
    logic [INDEX_WIDTH-1:0] scan_index_q;
    logic pair0_valid;
    logic pair1_valid;
    logic [INDEX_WIDTH-1:0] pair0_index;
    logic [INDEX_WIDTH-1:0] pair1_index;
    logic [INDEX_WIDTH-1:0] next_scan_index;
    logic pair_last;
    logic pair0_valid_q;
    logic pair1_valid_q;
    logic [INDEX_WIDTH-1:0] pair0_index_q;
    logic [INDEX_WIDTH-1:0] pair1_index_q;
    logic [1:0] pair0_offset_q;
    logic [1:0] pair1_offset_q;
    logic [INDEX_WIDTH-1:0] next_scan_index_q;
    logic pair_last_q;
    logic request_has_data;
    integer flat_index;
    integer flat_engine;
    integer flat_lane;
    logic found_first;
    logic found_second;

    always_comb begin
        pair0_valid = 1'b0;
        pair1_valid = 1'b0;
        pair0_index = '0;
        pair1_index = '0;
        next_scan_index = scan_index_q;
        pair_last = 1'b1;
        found_first = 1'b0;
        found_second = 1'b0;
        flat_index = 0;
        flat_engine = 0;
        flat_lane = 0;
        for (integer item = 0; item < ITEMS; item++) begin
            if (item >= scan_index_q) begin
                flat_engine = item / LANES;
                flat_lane = item % LANES;
                if (engine_mask_q[flat_engine] && lane_mask_q[flat_lane]) begin
                    if (!found_first) begin
                        pair0_valid = 1'b1;
                        pair0_index = INDEX_WIDTH'(item);
                        found_first = 1'b1;
                        next_scan_index = INDEX_WIDTH'(item + 1);
                    end else if (!found_second) begin
                        pair1_valid = 1'b1;
                        pair1_index = INDEX_WIDTH'(item);
                        found_second = 1'b1;
                        next_scan_index = INDEX_WIDTH'(item + 1);
                    end else begin
                        pair_last = 1'b0;
                    end
                end
            end
        end
    end

    assign request_has_data = (i_engine_mask != '0) && (i_lane_mask != '0);
    assign o_request_ready = !i_abort && state_q == STATE_IDLE &&
                             !o_sticky_error;
    assign o_read_valid = !i_abort && state_q == STATE_ISSUE;
    assign o_read_result_ready = !i_abort && state_q == STATE_WAIT;
    assign o_weight_valid = !i_abort && state_q == STATE_OUTPUT;
    assign o_weight_flat = weight_flat_q;
    assign o_engine_mask = engine_mask_q;
    assign o_lane_mask = lane_mask_q;
    assign o_busy = state_q != STATE_IDLE;
    assign o_read_word_addr0 = address_flat_q[
        pair0_index_q*Q16_ADDRESS_WIDTH + 2 +: Q64_ADDRESS_WIDTH
    ];
    assign o_read_word_addr1 = pair1_valid_q ? address_flat_q[
        pair1_index_q*Q16_ADDRESS_WIDTH + 2 +: Q64_ADDRESS_WIDTH
    ] : o_read_word_addr0;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            engine_mask_q <= '0;
            lane_mask_q <= '0;
            address_flat_q <= '0;
            weight_flat_q <= '0;
            scan_index_q <= '0;
            pair0_valid_q <= 1'b0;
            pair1_valid_q <= 1'b0;
            pair0_index_q <= '0;
            pair1_index_q <= '0;
            pair0_offset_q <= '0;
            pair1_offset_q <= '0;
            next_scan_index_q <= '0;
            pair_last_q <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            engine_mask_q <= '0;
            lane_mask_q <= '0;
            address_flat_q <= '0;
            weight_flat_q <= '0;
            scan_index_q <= '0;
            pair0_valid_q <= 1'b0;
            pair1_valid_q <= 1'b0;
            pair0_index_q <= '0;
            pair1_index_q <= '0;
            pair0_offset_q <= '0;
            pair1_offset_q <= '0;
            next_scan_index_q <= '0;
            pair_last_q <= 1'b0;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;
            case (state_q)
                STATE_IDLE: begin
                    if (i_request_valid) begin
                        if (!request_has_data) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            engine_mask_q <= i_engine_mask;
                            lane_mask_q <= i_lane_mask;
                            address_flat_q <= i_weight_q16_addr_flat;
                            weight_flat_q <= '0;
                            scan_index_q <= '0;
                            state_q <= STATE_SCAN;
                        end
                    end
                end
                STATE_SCAN: begin
                    if (!pair0_valid) begin
                        state_q <= STATE_OUTPUT;
                    end else begin
                        pair0_valid_q <= pair0_valid;
                        pair1_valid_q <= pair1_valid;
                        pair0_index_q <= pair0_index;
                        pair1_index_q <= pair1_index;
                        pair0_offset_q <= address_flat_q[
                            pair0_index*Q16_ADDRESS_WIDTH +: 2
                        ];
                        pair1_offset_q <= address_flat_q[
                            pair1_index*Q16_ADDRESS_WIDTH +: 2
                        ];
                        next_scan_index_q <= next_scan_index;
                        pair_last_q <= pair_last;
                        state_q <= STATE_ISSUE;
                    end
                end
                STATE_ISSUE: begin
                    if (i_read_ready)
                        state_q <= STATE_WAIT;
                end
                STATE_WAIT: begin
                    if (i_read_result_valid) begin
                        if (pair0_valid_q)
                            weight_flat_q[pair0_index_q*16 +: 16] <=
                                16'(i_read_data0 >> (pair0_offset_q * 16));
                        if (pair1_valid_q)
                            weight_flat_q[pair1_index_q*16 +: 16] <=
                                16'(i_read_data1 >> (pair1_offset_q * 16));
                        scan_index_q <= next_scan_index_q;
                        state_q <= pair_last_q ? STATE_OUTPUT : STATE_SCAN;
                    end
                end
                STATE_OUTPUT: begin
                    if (i_weight_ready) begin
                        engine_mask_q <= '0;
                        lane_mask_q <= '0;
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
