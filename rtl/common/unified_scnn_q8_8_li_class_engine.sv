`default_nettype none

module unified_scnn_q8_8_li_class_engine #(
    parameter int INPUT_SIZE = 2048,
    parameter int OUTPUT_CLASSES = 11,
    parameter int CLASS_WIDTH = 5
) (
    input  logic                    i_clk,
    input  logic                    i_rst_n,
    input  logic                    i_sample_reset,
    input  logic                    i_valid,
    output logic                    o_ready,
    input  logic [CLASS_WIDTH-1:0]  i_class,
    input  logic [127:0]            i_feature_flat,
    input  logic [127:0]            i_weight_flat,
    input  logic [7:0]              i_lane_mask,
    input  logic                    i_first,
    input  logic                    i_last,
    output logic                    o_valid,
    input  logic                    i_ready,
    output logic [CLASS_WIDTH-1:0]  o_class,
    output logic signed [15:0]      o_score_q8_8,
    output logic                    o_idle,
    output logic                    o_protocol_error,
    output logic                    o_class_error,
    output logic                    o_reset_while_busy_error
);

    localparam int STATE_WIDTH = 32 + $clog2(INPUT_SIZE + 1);
    localparam int CLASS_CAPACITY = 1 << CLASS_WIDTH;
    localparam logic signed [STATE_WIDTH-1:0] OUTPUT_MAX_STATE =
        STATE_WIDTH'(32767);
    localparam logic signed [STATE_WIDTH-1:0] OUTPUT_MIN_STATE =
        -STATE_WIDTH'(32768);

    logic signed [STATE_WIDTH-1:0] state_memory [0:OUTPUT_CLASSES-1];
    logic                          state_valid [0:OUTPUT_CLASSES-1];
    logic [CLASS_WIDTH-1:0] class_q;
    logic class_active_q;
    logic class_result_pending_q;
    logic accumulator_ready;
    logic accumulator_valid;
    logic accumulator_consume;
    logic signed [47:0] dot_sum;
    logic signed [STATE_WIDTH-1:0] previous_state;
    logic signed [STATE_WIDTH-1:0] dot_current;
    logic signed [STATE_WIDTH:0] state_next_wide;
    logic signed [STATE_WIDTH-1:0] state_next;
    logic result_valid_q;
    logic [CLASS_WIDTH-1:0] result_class_q;
    logic signed [15:0] result_score_q;
    logic class_in_range;
    integer state_index;

    function automatic logic signed [15:0] saturate_score(
        input logic signed [STATE_WIDTH-1:0] value
    );
        begin
            if (value > OUTPUT_MAX_STATE)
                saturate_score = 16'sh7fff;
            else if (value < OUTPUT_MIN_STATE)
                saturate_score = 16'sh8000;
            else
                saturate_score = value[15:0];
        end
    endfunction

    assign class_in_range = OUTPUT_CLASSES >= CLASS_CAPACITY ? 1'b1 :
                            i_class < CLASS_WIDTH'(OUTPUT_CLASSES);
    assign o_ready = accumulator_ready && !result_valid_q &&
                     !class_result_pending_q &&
                     !i_sample_reset && class_in_range;
    assign accumulator_consume = !result_valid_q || i_ready;
    assign previous_state = state_valid[class_q] ? state_memory[class_q] : '0;
    assign dot_current = STATE_WIDTH'($signed(dot_sum >>> 8));
    assign state_next_wide = $signed(previous_state) + $signed(dot_current);
    assign state_next = state_next_wide[STATE_WIDTH-1:0];
    assign o_valid = result_valid_q;
    assign o_class = result_class_q;
    assign o_score_q8_8 = result_score_q;
    assign o_idle = !class_active_q && !class_result_pending_q &&
                    !result_valid_q && accumulator_ready;

    unified_q8_8_k8_accumulator #(
        .LANES(8), .ACC_WIDTH(48)
    ) u_class_dot (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(i_valid && o_ready), .o_ready(accumulator_ready),
        .i_a_flat(i_feature_flat), .i_b_flat(i_weight_flat),
        .i_lane_mask(i_lane_mask), .i_first(i_first), .i_last(i_last),
        .o_valid(accumulator_valid), .i_ready(accumulator_consume),
        .o_accumulator(dot_sum)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            class_q <= '0;
            class_active_q <= 1'b0;
            class_result_pending_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_class_q <= '0;
            result_score_q <= 16'sd0;
            o_protocol_error <= 1'b0;
            o_class_error <= 1'b0;
            o_reset_while_busy_error <= 1'b0;
            for (state_index = 0; state_index < OUTPUT_CLASSES;
                 state_index = state_index + 1) begin
                state_valid[state_index] <= 1'b0;
            end
        end else begin
            if (i_valid && !class_in_range)
                o_class_error <= 1'b1;

            if (i_valid && o_ready) begin
                if (i_last)
                    class_result_pending_q <= 1'b1;
                if (i_first) begin
                    if (class_active_q)
                        o_protocol_error <= 1'b1;
                    class_q <= i_class;
                    class_active_q <= !i_last;
                end else begin
                    if (!class_active_q || i_class != class_q)
                        o_protocol_error <= 1'b1;
                    if (i_last)
                        class_active_q <= 1'b0;
                end
            end

            if (accumulator_valid && accumulator_consume) begin
                state_memory[class_q] <= state_next;
                state_valid[class_q] <= 1'b1;
                result_class_q <= class_q;
                result_score_q <= saturate_score(state_next);
                result_valid_q <= 1'b1;
                class_active_q <= 1'b0;
            end

            if (result_valid_q && i_ready) begin
                result_valid_q <= 1'b0;
                class_result_pending_q <= 1'b0;
            end

            if (i_sample_reset) begin
                if (!o_idle) begin
                    o_reset_while_busy_error <= 1'b1;
                end else begin
                    for (state_index = 0; state_index < OUTPUT_CLASSES;
                         state_index = state_index + 1) begin
                        state_valid[state_index] <= 1'b0;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
