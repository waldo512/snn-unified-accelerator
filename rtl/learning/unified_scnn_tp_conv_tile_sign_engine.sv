`timescale 1ns/1ps
`default_nettype none

// One reusable convolution-learning working set. It captures a bounded
// student/target output wave and input group, forms the two TP postfactors,
// and emits one gradient sign per OIHW kernel coordinate. The shared direct
// residual updater consumes the sign, so no full gradient tensor is stored.
module unified_scnn_tp_conv_tile_sign_engine #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int SAMPLE_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int INPUT_INDEX_WIDTH =
        (MAX_INPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_INPUT_CHANNELS*SPATIAL*SPATIAL),
    parameter int TRACE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL),
    parameter int SURROGATE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL)
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]        i_cfg_batch_count,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_total_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_total_output_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_input_channel_count,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_output_channel_count,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_cfg_input_channel_base,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_cfg_output_channel_base,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_cfg_parameter_base,
    input  wire logic [Q16_ADDR_WIDTH-1:0]           i_cfg_weight_q16_base,
    output logic                                o_active,
    input  wire logic                                i_release,
    output logic                                o_release_ready,

    input  wire logic                                i_student_input_valid,
    output logic                                o_student_input_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_student_input_sample,
    input  wire logic [INPUT_INDEX_WIDTH-1:0]        i_student_input_index,
    input  wire logic signed [15:0]                  i_student_input_q8_8,
    input  wire logic                                i_target_input_valid,
    output logic                                o_target_input_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_target_input_sample,
    input  wire logic [INPUT_INDEX_WIDTH-1:0]        i_target_input_index,
    input  wire logic signed [15:0]                  i_target_input_q8_8,
    input  wire logic                                i_student_trace_valid,
    output logic                                o_student_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_student_trace_sample,
    input  wire logic [TRACE_INDEX_WIDTH-1:0]        i_student_trace_index,
    input  wire logic signed [15:0]                  i_student_trace_q8_8,
    input  wire logic                                i_target_trace_valid,
    output logic                                o_target_trace_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_target_trace_sample,
    input  wire logic [TRACE_INDEX_WIDTH-1:0]        i_target_trace_index,
    input  wire logic signed [15:0]                  i_target_trace_q8_8,
    input  wire logic                                i_student_surrogate_valid,
    output logic                                o_student_surrogate_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_student_surrogate_sample,
    input  wire logic [SURROGATE_INDEX_WIDTH-1:0]    i_student_surrogate_index,
    input  wire logic signed [15:0]                  i_student_surrogate_q8_8,
    input  wire logic                                i_target_surrogate_valid,
    output logic                                o_target_surrogate_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_target_surrogate_sample,
    input  wire logic [SURROGATE_INDEX_WIDTH-1:0]    i_target_surrogate_index,
    input  wire logic signed [15:0]                  i_target_surrogate_q8_8,
    input  wire logic                                i_modulation_valid,
    output logic                                o_modulation_ready,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_modulation_row,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_modulation_col,
    input  wire logic signed [15:0]                  i_modulation_q8_8,
    output logic                                o_capture_complete,
    input  wire logic                                i_start,
    output logic                                o_start_ready,

    output logic                                o_gradient_valid,
    input  wire logic                                i_gradient_ready,
    output logic signed [1:0]                   o_gradient_sign,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_gradient_local_addr,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_gradient_parameter_addr,
    output logic [Q16_ADDR_WIDTH-1:0]           o_gradient_weight_q16_addr,
    output logic [Q64_ADDR_WIDTH-1:0]           o_gradient_weight_q64_addr,
    output logic [1:0]                          o_gradient_weight_q64_lane,
    output logic                                o_gradient_tile_last,
    output logic                                o_done,
    output logic                                o_busy,
    output logic                                o_protocol_error
);
    localparam int INPUT_PLANE = SPATIAL * SPATIAL;
    localparam int TRACE_PLANE = POOLED_SPATIAL * POOLED_SPATIAL;
    localparam int MAX_INPUT_VALUES = MAX_INPUT_CHANNELS * INPUT_PLANE;
    localparam int MAX_TRACE_VALUES = MAX_OUTPUT_CHANNELS * TRACE_PLANE;
    localparam int MAX_LOCAL_VALUES = MAX_OUTPUT_CHANNELS * INPUT_PLANE;
    localparam int FACTOR_ADDR_WIDTH =
        (MAX_BATCH*MAX_LOCAL_VALUES <= 2) ? 1 :
        $clog2(MAX_BATCH*MAX_LOCAL_VALUES);
    localparam int INPUT_MEMORY_ADDR_WIDTH =
        (MAX_BATCH*MAX_INPUT_VALUES <= 2) ? 1 :
        $clog2(MAX_BATCH*MAX_INPUT_VALUES);
    localparam int GRADIENT_POSITION_WIDTH =
        (INPUT_PLANE <= 2) ? 1 : $clog2(INPUT_PLANE);

    typedef enum logic [3:0] {
        ST_INACTIVE, ST_CAPTURE, ST_FACTOR_S, ST_FACTOR_T,
        ST_POSTFACTOR_READ, ST_POSTFACTOR_MUL, ST_POSTFACTOR_WRITE,
        ST_GRADIENT, ST_GRADIENT_SIGN, ST_EMIT, ST_DONE
    } state_t;
    state_t state_q;

    logic signed [15:0] student_input_mem [0:MAX_BATCH*MAX_INPUT_VALUES-1];
    logic signed [15:0] target_input_mem [0:MAX_BATCH*MAX_INPUT_VALUES-1];
    logic signed [15:0] student_trace_mem [0:MAX_BATCH*MAX_TRACE_VALUES-1];
    logic signed [15:0] target_trace_mem [0:MAX_BATCH*MAX_TRACE_VALUES-1];
    logic signed [15:0] student_surrogate_mem [0:MAX_BATCH*MAX_LOCAL_VALUES-1];
    logic signed [15:0] target_surrogate_mem [0:MAX_BATCH*MAX_LOCAL_VALUES-1];
    logic signed [15:0] modulation_mem [0:MAX_BATCH*MAX_BATCH-1];
    logic signed [15:0] student_postfactor_mem [0:MAX_BATCH*MAX_LOCAL_VALUES-1];
    logic signed [15:0] target_postfactor_mem [0:MAX_BATCH*MAX_LOCAL_VALUES-1];

    integer batch_count_q;
    integer total_input_channels_q;
    integer total_output_channels_q;
    integer input_count_q;
    integer output_count_q;
    integer input_base_q;
    integer output_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [Q16_ADDR_WIDTH-1:0] weight_q16_base_q;
    integer student_input_count_q;
    integer target_input_count_q;
    integer student_trace_count_q;
    integer target_trace_count_q;
    integer student_surrogate_count_q;
    integer target_surrogate_count_q;
    integer modulation_count_q;
    integer factor_sample_q;
    integer factor_local_q;
    integer factor_pair_q;
    logic signed [63:0] factor_accum_q;
    logic signed [63:0] factor_final_sum_q;
    logic factor_target_q;
    logic [FACTOR_ADDR_WIDTH-1:0] factor_sample_base_q;
    logic [FACTOR_ADDR_WIDTH-1:0] factor_mem_addr_q;
    logic signed [15:0] postfactor_surrogate_q;
    logic signed [79:0] surrogate_product_q;
    integer grad_output_q;
    integer grad_input_q;
    integer grad_row_q;
    integer grad_col_q;
    logic [SAMPLE_WIDTH-1:0] gradient_issue_sample_q;
    logic [GRADIENT_POSITION_WIDTH-1:0] gradient_issue_position_q;
    logic gradient_issue_done_q;
    logic [FACTOR_ADDR_WIDTH-1:0] gradient_postfactor_addr_q;
    logic [INPUT_MEMORY_ADDR_WIDTH-1:0] gradient_input_addr_q;
    logic gradient_address_valid_q;
    logic gradient_address_in_range_q;
    logic gradient_address_last_q;
    logic signed [31:0] gradient_product_s_q;
    logic signed [31:0] gradient_product_t_q;
    logic gradient_product_valid_q;
    logic gradient_product_last_q;
    logic signed [63:0] gradient_term_pipe_q;
    logic gradient_term_valid_q;
    logic gradient_term_last_q;
    logic signed [63:0] gradient_accum_q;
    logic signed [1:0] gradient_sign_q;

    logic input_coordinates_legal;
    logic target_input_coordinates_legal;
    logic student_trace_coordinates_legal;
    logic target_trace_coordinates_legal;
    logic student_surrogate_coordinates_legal;
    logic target_surrogate_coordinates_legal;
    logic modulation_coordinates_legal;
    integer factor_oc;
    integer factor_position;
    integer factor_pool;
    logic signed [31:0] factor_product;
    logic signed [63:0] factor_term;
    logic signed [63:0] factor_sum;
    logic signed [63:0] postfactor_value;
    integer gradient_sample;
    integer gradient_position;
    integer gradient_out_row;
    integer gradient_out_col;
    integer gradient_in_row;
    integer gradient_in_col;
    integer gradient_input_index;
    integer gradient_local_index;
    integer gradient_issue_postfactor_addr;
    integer gradient_issue_input_addr;
    logic gradient_issue_in_range;
    logic signed [31:0] gradient_product_s;
    logic signed [31:0] gradient_product_t;
    logic signed [63:0] gradient_term;
    logic signed [63:0] gradient_sum;
    logic signed [63:0] gradient_batch_threshold;
    logic mapper_valid;
    logic mapper_error;
    logic [5:0] mapper_output_wave;
    logic [3:0] mapper_output_lane;
    logic [5:0] mapper_input_group;
    logic [2:0] mapper_input_lane;

    function automatic logic signed [15:0] sat16(
        input logic signed [63:0] value
    );
        if (value > 64'sd32767)
            return 16'sd32767;
        if (value < -64'sd32768)
            return -16'sd32768;
        return 16'($signed(value));
    endfunction

    always_comb begin
        input_coordinates_legal =
            int'(i_student_input_sample) < batch_count_q &&
            int'(i_student_input_index) < input_count_q * INPUT_PLANE;
        target_input_coordinates_legal =
            int'(i_target_input_sample) < batch_count_q &&
            int'(i_target_input_index) < input_count_q * INPUT_PLANE;
        student_trace_coordinates_legal =
            int'(i_student_trace_sample) < batch_count_q &&
            int'(i_student_trace_index) < output_count_q * TRACE_PLANE;
        target_trace_coordinates_legal =
            int'(i_target_trace_sample) < batch_count_q &&
            int'(i_target_trace_index) < output_count_q * TRACE_PLANE;
        student_surrogate_coordinates_legal =
            int'(i_student_surrogate_sample) < batch_count_q &&
            int'(i_student_surrogate_index) < output_count_q * INPUT_PLANE;
        target_surrogate_coordinates_legal =
            int'(i_target_surrogate_sample) < batch_count_q &&
            int'(i_target_surrogate_index) < output_count_q * INPUT_PLANE;
        modulation_coordinates_legal =
            int'(i_modulation_row) < batch_count_q &&
            int'(i_modulation_col) < batch_count_q;

        o_cfg_ready = state_q == ST_INACTIVE;
        o_active = state_q != ST_INACTIVE;
        o_release_ready = state_q == ST_CAPTURE &&
            student_input_count_q == 0 && target_input_count_q == 0 &&
            student_trace_count_q == 0 && target_trace_count_q == 0 &&
            student_surrogate_count_q == 0 && target_surrogate_count_q == 0 &&
            modulation_count_q == 0;
        o_student_input_ready = state_q == ST_CAPTURE &&
            student_input_count_q < batch_count_q * input_count_q * INPUT_PLANE;
        o_target_input_ready = state_q == ST_CAPTURE &&
            target_input_count_q < batch_count_q * input_count_q * INPUT_PLANE;
        o_student_trace_ready = state_q == ST_CAPTURE &&
            student_trace_count_q < batch_count_q * output_count_q * TRACE_PLANE;
        o_target_trace_ready = state_q == ST_CAPTURE &&
            target_trace_count_q < batch_count_q * output_count_q * TRACE_PLANE;
        o_student_surrogate_ready = state_q == ST_CAPTURE &&
            student_surrogate_count_q < batch_count_q * output_count_q * INPUT_PLANE;
        o_target_surrogate_ready = state_q == ST_CAPTURE &&
            target_surrogate_count_q < batch_count_q * output_count_q * INPUT_PLANE;
        o_modulation_ready = state_q == ST_CAPTURE &&
            modulation_count_q < batch_count_q * batch_count_q;
        o_capture_complete = state_q == ST_CAPTURE &&
            student_input_count_q == batch_count_q * input_count_q * INPUT_PLANE &&
            target_input_count_q == batch_count_q * input_count_q * INPUT_PLANE &&
            student_trace_count_q == batch_count_q * output_count_q * TRACE_PLANE &&
            target_trace_count_q == batch_count_q * output_count_q * TRACE_PLANE &&
            student_surrogate_count_q == batch_count_q * output_count_q * INPUT_PLANE &&
            target_surrogate_count_q == batch_count_q * output_count_q * INPUT_PLANE &&
            modulation_count_q == batch_count_q * batch_count_q;
        o_start_ready = o_capture_complete;
        o_gradient_valid = state_q == ST_EMIT;
        o_gradient_sign = gradient_sign_q;
        o_gradient_tile_last = state_q == ST_EMIT &&
            grad_output_q == output_count_q-1 && grad_input_q == input_count_q-1 &&
            grad_row_q == KERNEL_SIZE-1 && grad_col_q == KERNEL_SIZE-1;
        o_done = state_q == ST_DONE;
        o_busy = state_q inside {
            ST_FACTOR_S, ST_FACTOR_T, ST_POSTFACTOR_READ,
            ST_POSTFACTOR_MUL, ST_POSTFACTOR_WRITE, ST_GRADIENT,
            ST_GRADIENT_SIGN, ST_EMIT
        };

        factor_oc = factor_local_q / INPUT_PLANE;
        factor_position = factor_local_q % INPUT_PLANE;
        factor_pool = factor_oc * TRACE_PLANE +
            (factor_position / SPATIAL / 2) * POOLED_SPATIAL +
            (factor_position % SPATIAL) / 2;
        factor_product = '0;
        if (state_q == ST_FACTOR_S)
            factor_product = modulation_mem[factor_sample_q*MAX_BATCH+factor_pair_q] *
                target_trace_mem[factor_pair_q*MAX_TRACE_VALUES+factor_pool];
        else if (state_q == ST_FACTOR_T)
            factor_product = modulation_mem[factor_pair_q*MAX_BATCH+factor_sample_q] *
                student_trace_mem[factor_pair_q*MAX_TRACE_VALUES+factor_pool];
        factor_term = 64'($signed(factor_product) >>> 8);
        factor_sum = factor_accum_q + factor_term;
        postfactor_value = 64'($signed(surrogate_product_q) >>> 8);
        postfactor_value = (postfactor_value / 64'sd4) / 64'sd4;

        gradient_sample = int'(gradient_issue_sample_q);
        gradient_position = int'(gradient_issue_position_q);
        gradient_out_row = gradient_position / SPATIAL;
        gradient_out_col = gradient_position % SPATIAL;
        gradient_in_row = gradient_out_row + grad_row_q - 1;
        gradient_in_col = gradient_out_col + grad_col_q - 1;
        gradient_input_index = grad_input_q * INPUT_PLANE +
            gradient_in_row * SPATIAL + gradient_in_col;
        gradient_local_index = grad_output_q * INPUT_PLANE + gradient_position;
        gradient_issue_postfactor_addr =
            gradient_sample * MAX_LOCAL_VALUES + gradient_local_index;
        gradient_issue_input_addr =
            gradient_sample * MAX_INPUT_VALUES + gradient_input_index;
        gradient_issue_in_range = gradient_in_row >= 0 &&
            gradient_in_row < SPATIAL && gradient_in_col >= 0 &&
            gradient_in_col < SPATIAL;
        gradient_product_s = '0;
        gradient_product_t = '0;
        if (state_q == ST_GRADIENT && gradient_address_valid_q &&
            gradient_address_in_range_q) begin
            gradient_product_s =
                student_postfactor_mem[gradient_postfactor_addr_q] *
                student_input_mem[gradient_input_addr_q];
            gradient_product_t =
                target_postfactor_mem[gradient_postfactor_addr_q] *
                target_input_mem[gradient_input_addr_q];
        end
        gradient_term = 64'($signed(gradient_product_s_q) >>> 8) +
            64'($signed(gradient_product_t_q) >>> 8);
        gradient_sum = gradient_accum_q + gradient_term_pipe_q;
        // Only the sign of the integer batch average is consumed. Comparing
        // against +/-batch_count preserves truncation-toward-zero semantics
        // without synthesizing a variable 64-bit divider.
        gradient_batch_threshold = 64'($signed(batch_count_q));
    end

    unified_scnn_conv_weight_address_mapper #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH), .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .MAX_INPUT_CHANNELS(512), .MAX_OUTPUT_CHANNELS(512),
        .KERNEL_SIZE(KERNEL_SIZE),
        .CHANNEL_INDEX_WIDTH(CHANNEL_INDEX_WIDTH),
        .CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH)
    ) u_address_mapper (
        .i_valid(state_q == ST_EMIT),
        .i_input_channels(CHANNEL_COUNT_WIDTH'(total_input_channels_q)),
        .i_output_channels(CHANNEL_COUNT_WIDTH'(total_output_channels_q)),
        .i_output_channel(CHANNEL_INDEX_WIDTH'(output_base_q + grad_output_q)),
        .i_input_channel(CHANNEL_INDEX_WIDTH'(input_base_q + grad_input_q)),
        .i_kernel_row(2'(grad_row_q)), .i_kernel_col(2'(grad_col_q)),
        .i_parameter_base(parameter_base_q),
        .i_weight_q16_base(weight_q16_base_q),
        .o_valid(mapper_valid), .o_error(mapper_error),
        .o_local_parameter_addr(o_gradient_local_addr),
        .o_parameter_addr(o_gradient_parameter_addr),
        .o_weight_q16_addr(o_gradient_weight_q16_addr),
        .o_weight_q64_addr(o_gradient_weight_q64_addr),
        .o_weight_q64_lane(o_gradient_weight_q64_lane),
        .o_output_wave(mapper_output_wave),
        .o_output_lane(mapper_output_lane),
        .o_input_group(mapper_input_group),
        .o_input_lane(mapper_input_lane)
    );

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= ST_INACTIVE;
            batch_count_q <= 0;
            total_input_channels_q <= 0;
            total_output_channels_q <= 0;
            input_count_q <= 0;
            output_count_q <= 0;
            input_base_q <= 0;
            output_base_q <= 0;
            parameter_base_q <= '0;
            weight_q16_base_q <= '0;
            student_input_count_q <= 0;
            target_input_count_q <= 0;
            student_trace_count_q <= 0;
            target_trace_count_q <= 0;
            student_surrogate_count_q <= 0;
            target_surrogate_count_q <= 0;
            modulation_count_q <= 0;
            factor_sample_q <= 0;
            factor_local_q <= 0;
            factor_pair_q <= 0;
            factor_accum_q <= '0;
            factor_final_sum_q <= '0;
            factor_target_q <= 1'b0;
            factor_sample_base_q <= '0;
            factor_mem_addr_q <= '0;
            postfactor_surrogate_q <= '0;
            surrogate_product_q <= '0;
            grad_output_q <= 0;
            grad_input_q <= 0;
            grad_row_q <= 0;
            grad_col_q <= 0;
            gradient_issue_sample_q <= '0;
            gradient_issue_position_q <= '0;
            gradient_issue_done_q <= 1'b0;
            gradient_postfactor_addr_q <= '0;
            gradient_input_addr_q <= '0;
            gradient_address_valid_q <= 1'b0;
            gradient_address_in_range_q <= 1'b0;
            gradient_address_last_q <= 1'b0;
            gradient_product_s_q <= '0;
            gradient_product_t_q <= '0;
            gradient_product_valid_q <= 1'b0;
            gradient_product_last_q <= 1'b0;
            gradient_term_pipe_q <= '0;
            gradient_term_valid_q <= 1'b0;
            gradient_term_last_q <= 1'b0;
            gradient_accum_q <= '0;
            gradient_sign_q <= '0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_cfg_valid && o_cfg_ready) begin
                if (i_cfg_batch_count == 0 ||
                    int'(i_cfg_batch_count) > MAX_BATCH ||
                    i_cfg_input_channel_count == 0 ||
                    int'(i_cfg_input_channel_count) > MAX_INPUT_CHANNELS ||
                    i_cfg_output_channel_count == 0 ||
                    int'(i_cfg_output_channel_count) > MAX_OUTPUT_CHANNELS ||
                    i_cfg_total_input_channels == 0 ||
                    int'(i_cfg_total_input_channels) > 512 ||
                    i_cfg_total_output_channels == 0 ||
                    int'(i_cfg_total_output_channels) > 512 ||
                    int'(i_cfg_input_channel_base) +
                        int'(i_cfg_input_channel_count) >
                        int'(i_cfg_total_input_channels) ||
                    int'(i_cfg_output_channel_base) +
                        int'(i_cfg_output_channel_count) >
                        int'(i_cfg_total_output_channels)) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    batch_count_q <= int'(i_cfg_batch_count);
                    total_input_channels_q <= int'(i_cfg_total_input_channels);
                    total_output_channels_q <= int'(i_cfg_total_output_channels);
                    input_count_q <= int'(i_cfg_input_channel_count);
                    output_count_q <= int'(i_cfg_output_channel_count);
                    input_base_q <= int'(i_cfg_input_channel_base);
                    output_base_q <= int'(i_cfg_output_channel_base);
                    parameter_base_q <= i_cfg_parameter_base;
                    weight_q16_base_q <= i_cfg_weight_q16_base;
                    state_q <= ST_CAPTURE;
                end
            end

            if (i_release && o_release_ready) begin
                state_q <= ST_INACTIVE;
            end else if (state_q == ST_CAPTURE) begin
                if (i_student_input_valid && o_student_input_ready) begin
                    if (!input_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        student_input_mem[int'(i_student_input_sample)*MAX_INPUT_VALUES+
                            int'(i_student_input_index)] <= i_student_input_q8_8;
                        student_input_count_q <= student_input_count_q + 1;
                    end
                end
                if (i_target_input_valid && o_target_input_ready) begin
                    if (!target_input_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        target_input_mem[int'(i_target_input_sample)*MAX_INPUT_VALUES+
                            int'(i_target_input_index)] <= i_target_input_q8_8;
                        target_input_count_q <= target_input_count_q + 1;
                    end
                end
                if (i_student_trace_valid && o_student_trace_ready) begin
                    if (!student_trace_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        student_trace_mem[int'(i_student_trace_sample)*MAX_TRACE_VALUES+
                            int'(i_student_trace_index)] <= i_student_trace_q8_8;
                        student_trace_count_q <= student_trace_count_q + 1;
                    end
                end
                if (i_target_trace_valid && o_target_trace_ready) begin
                    if (!target_trace_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        target_trace_mem[int'(i_target_trace_sample)*MAX_TRACE_VALUES+
                            int'(i_target_trace_index)] <= i_target_trace_q8_8;
                        target_trace_count_q <= target_trace_count_q + 1;
                    end
                end
                if (i_student_surrogate_valid && o_student_surrogate_ready) begin
                    if (!student_surrogate_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        student_surrogate_mem[int'(i_student_surrogate_sample)*MAX_LOCAL_VALUES+
                            int'(i_student_surrogate_index)] <= i_student_surrogate_q8_8;
                        student_surrogate_count_q <= student_surrogate_count_q + 1;
                    end
                end
                if (i_target_surrogate_valid && o_target_surrogate_ready) begin
                    if (!target_surrogate_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        target_surrogate_mem[int'(i_target_surrogate_sample)*MAX_LOCAL_VALUES+
                            int'(i_target_surrogate_index)] <= i_target_surrogate_q8_8;
                        target_surrogate_count_q <= target_surrogate_count_q + 1;
                    end
                end
                if (i_modulation_valid && o_modulation_ready) begin
                    if (!modulation_coordinates_legal)
                        o_protocol_error <= 1'b1;
                    else begin
                        modulation_mem[int'(i_modulation_row)*MAX_BATCH+
                            int'(i_modulation_col)] <= i_modulation_q8_8;
                        modulation_count_q <= modulation_count_q + 1;
                    end
                end
                if (i_start) begin
                    if (!o_start_ready) begin
                        o_protocol_error <= 1'b1;
                    end else begin
                        factor_sample_q <= 0;
                        factor_local_q <= 0;
                        factor_pair_q <= 0;
                        factor_accum_q <= '0;
                        factor_final_sum_q <= '0;
                        factor_target_q <= 1'b0;
                        factor_sample_base_q <= '0;
                        factor_mem_addr_q <= '0;
                        postfactor_surrogate_q <= '0;
                        surrogate_product_q <= '0;
                        state_q <= ST_FACTOR_S;
                    end
                end
            end else if (state_q == ST_FACTOR_S || state_q == ST_FACTOR_T) begin
                if (factor_pair_q == batch_count_q-1) begin
                    factor_final_sum_q <= factor_sum;
                    factor_target_q <= state_q == ST_FACTOR_T;
                    factor_pair_q <= 0;
                    factor_accum_q <= '0;
                    state_q <= ST_POSTFACTOR_READ;
                end else begin
                    factor_pair_q <= factor_pair_q + 1;
                    factor_accum_q <= factor_sum;
                end
            end else if (state_q == ST_POSTFACTOR_READ) begin
                if (factor_target_q)
                    postfactor_surrogate_q <= target_surrogate_mem[factor_mem_addr_q];
                else
                    postfactor_surrogate_q <= student_surrogate_mem[factor_mem_addr_q];
                state_q <= ST_POSTFACTOR_MUL;
            end else if (state_q == ST_POSTFACTOR_MUL) begin
                surrogate_product_q <= factor_final_sum_q * postfactor_surrogate_q;
                state_q <= ST_POSTFACTOR_WRITE;
            end else if (state_q == ST_POSTFACTOR_WRITE) begin
                if (factor_target_q)
                    target_postfactor_mem[factor_mem_addr_q] <= sat16(postfactor_value);
                else
                    student_postfactor_mem[factor_mem_addr_q] <= sat16(postfactor_value);
                factor_final_sum_q <= '0;
                if (factor_local_q == output_count_q*INPUT_PLANE-1) begin
                    factor_local_q <= 0;
                    if (factor_sample_q == batch_count_q-1) begin
                        factor_sample_q <= 0;
                        if (!factor_target_q) begin
                            factor_target_q <= 1'b1;
                            factor_sample_base_q <= '0;
                            factor_mem_addr_q <= '0;
                            state_q <= ST_FACTOR_T;
                        end else begin
                            grad_output_q <= 0;
                            grad_input_q <= 0;
                            grad_row_q <= 0;
                            grad_col_q <= 0;
                            gradient_issue_sample_q <= '0;
                            gradient_issue_position_q <= '0;
                            gradient_issue_done_q <= 1'b0;
                            gradient_address_valid_q <= 1'b0;
                            gradient_product_valid_q <= 1'b0;
                            gradient_term_valid_q <= 1'b0;
                            gradient_accum_q <= '0;
                            state_q <= ST_GRADIENT;
                        end
                    end else begin
                        factor_sample_q <= factor_sample_q + 1;
                        factor_sample_base_q <=
                            factor_sample_base_q + FACTOR_ADDR_WIDTH'(MAX_LOCAL_VALUES);
                        factor_mem_addr_q <=
                            factor_sample_base_q + FACTOR_ADDR_WIDTH'(MAX_LOCAL_VALUES);
                        state_q <= factor_target_q ? ST_FACTOR_T : ST_FACTOR_S;
                    end
                end else begin
                    factor_local_q <= factor_local_q + 1;
                    factor_mem_addr_q <= factor_mem_addr_q + 1'b1;
                    state_q <= factor_target_q ? ST_FACTOR_T : ST_FACTOR_S;
                end
            end else if (state_q == ST_GRADIENT) begin
                // Four elastic arithmetic stages sustain one batch/spatial
                // contribution per cycle after fill: registered addresses,
                // parallel DSP products, a registered term, then the sole
                // dependent accumulator. The final sign comparison is kept
                // out of the accumulator cycle.
                gradient_term_valid_q <= gradient_product_valid_q;
                gradient_term_last_q <= gradient_product_last_q;
                if (gradient_product_valid_q)
                    gradient_term_pipe_q <= gradient_term;

                gradient_product_valid_q <= gradient_address_valid_q;
                gradient_product_last_q <= gradient_address_last_q;
                if (gradient_address_valid_q) begin
                    gradient_product_s_q <= gradient_product_s;
                    gradient_product_t_q <= gradient_product_t;
                end

                gradient_address_valid_q <= !gradient_issue_done_q;
                if (!gradient_issue_done_q) begin
                    gradient_postfactor_addr_q <=
                        FACTOR_ADDR_WIDTH'(gradient_issue_postfactor_addr);
                    gradient_input_addr_q <=
                        INPUT_MEMORY_ADDR_WIDTH'(gradient_issue_input_addr);
                    gradient_address_in_range_q <= gradient_issue_in_range;
                    gradient_address_last_q <=
                        int'(gradient_issue_sample_q) == batch_count_q-1 &&
                        int'(gradient_issue_position_q) == INPUT_PLANE-1;
                    if (int'(gradient_issue_position_q) == INPUT_PLANE-1) begin
                        gradient_issue_position_q <= '0;
                        if (int'(gradient_issue_sample_q) == batch_count_q-1)
                            gradient_issue_done_q <= 1'b1;
                        else
                            gradient_issue_sample_q <=
                                gradient_issue_sample_q + 1'b1;
                    end else begin
                        gradient_issue_position_q <=
                            gradient_issue_position_q + 1'b1;
                    end
                end

                if (gradient_term_valid_q) begin
                    gradient_accum_q <= gradient_sum;
                    if (gradient_term_last_q) begin
                        gradient_address_valid_q <= 1'b0;
                        gradient_product_valid_q <= 1'b0;
                        gradient_term_valid_q <= 1'b0;
                        state_q <= ST_GRADIENT_SIGN;
                    end
                end
            end else if (state_q == ST_GRADIENT_SIGN) begin
                if (gradient_accum_q >= gradient_batch_threshold)
                    gradient_sign_q <= 2'sd1;
                else if (gradient_accum_q <= -gradient_batch_threshold)
                    gradient_sign_q <= -2'sd1;
                else
                    gradient_sign_q <= 2'sd0;
                state_q <= ST_EMIT;
            end else if (state_q == ST_EMIT) begin
                if (mapper_error || !mapper_valid)
                    o_protocol_error <= 1'b1;
                if (i_gradient_ready) begin
                    gradient_issue_sample_q <= '0;
                    gradient_issue_position_q <= '0;
                    gradient_issue_done_q <= 1'b0;
                    gradient_address_valid_q <= 1'b0;
                    gradient_product_valid_q <= 1'b0;
                    gradient_term_valid_q <= 1'b0;
                    gradient_accum_q <= '0;
                    if (grad_col_q < KERNEL_SIZE-1) begin
                        grad_col_q <= grad_col_q + 1;
                        state_q <= ST_GRADIENT;
                    end else if (grad_row_q < KERNEL_SIZE-1) begin
                        grad_col_q <= 0;
                        grad_row_q <= grad_row_q + 1;
                        state_q <= ST_GRADIENT;
                    end else if (grad_input_q < input_count_q-1) begin
                        grad_col_q <= 0;
                        grad_row_q <= 0;
                        grad_input_q <= grad_input_q + 1;
                        state_q <= ST_GRADIENT;
                    end else if (grad_output_q < output_count_q-1) begin
                        grad_col_q <= 0;
                        grad_row_q <= 0;
                        grad_input_q <= 0;
                        grad_output_q <= grad_output_q + 1;
                        state_q <= ST_GRADIENT;
                    end else begin
                        state_q <= ST_DONE;
                    end
                end
            end else if (state_q == ST_DONE) begin
                student_input_count_q <= 0;
                target_input_count_q <= 0;
                student_trace_count_q <= 0;
                target_trace_count_q <= 0;
                student_surrogate_count_q <= 0;
                target_surrogate_count_q <= 0;
                modulation_count_q <= 0;
                state_q <= ST_CAPTURE;
            end
        end
    end

    initial begin
        if (SPATIAL != 4 || POOLED_SPATIAL != 2 || KERNEL_SIZE != 3)
            $error("Conv tile sign engine currently implements the Conv8 4x4/AAVG2/3x3 contract");
    end
endmodule

`default_nettype wire
