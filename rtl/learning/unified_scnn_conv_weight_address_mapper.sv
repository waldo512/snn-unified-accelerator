`timescale 1ns/1ps
`default_nettype none

// Map one logical convolution-kernel coordinate to both the scalar learning
// store and the packed-Q64 inference image. The order matches PyTorch OIHW and
// the existing VGG-9 DDR stream: output, input, kernel row, kernel column.
module unified_scnn_conv_weight_address_mapper #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int KERNEL_SIZE = 3,
    parameter int OUTPUT_ENGINES = 12,
    parameter int INPUT_LANES = 8,
    parameter int CHANNEL_INDEX_WIDTH =
        (MAX_INPUT_CHANNELS <= 2 && MAX_OUTPUT_CHANNELS <= 2) ? 1 :
        $clog2((MAX_INPUT_CHANNELS > MAX_OUTPUT_CHANNELS) ?
               MAX_INPUT_CHANNELS : MAX_OUTPUT_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH =
        $clog2(((MAX_INPUT_CHANNELS > MAX_OUTPUT_CHANNELS) ?
                MAX_INPUT_CHANNELS : MAX_OUTPUT_CHANNELS) + 1),
    parameter int KERNEL_INDEX_WIDTH =
        (KERNEL_SIZE <= 2) ? 1 : $clog2(KERNEL_SIZE),
    parameter int OUTPUT_WAVE_WIDTH =
        (((MAX_OUTPUT_CHANNELS + OUTPUT_ENGINES - 1) / OUTPUT_ENGINES) <= 2) ?
        1 : $clog2((MAX_OUTPUT_CHANNELS + OUTPUT_ENGINES - 1) / OUTPUT_ENGINES),
    parameter int OUTPUT_LANE_WIDTH =
        (OUTPUT_ENGINES <= 2) ? 1 : $clog2(OUTPUT_ENGINES),
    parameter int INPUT_GROUP_WIDTH =
        (((MAX_INPUT_CHANNELS + INPUT_LANES - 1) / INPUT_LANES) <= 2) ?
        1 : $clog2((MAX_INPUT_CHANNELS + INPUT_LANES - 1) / INPUT_LANES),
    parameter int INPUT_LANE_WIDTH =
        (INPUT_LANES <= 2) ? 1 : $clog2(INPUT_LANES)
) (
    input  wire logic                                 i_valid,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]       i_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]       i_output_channels,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]       i_output_channel,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]       i_input_channel,
    input  wire logic [KERNEL_INDEX_WIDTH-1:0]        i_kernel_row,
    input  wire logic [KERNEL_INDEX_WIDTH-1:0]        i_kernel_col,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]          i_parameter_base,
    input  wire logic [Q16_ADDR_WIDTH-1:0]            i_weight_q16_base,

    output logic                                 o_valid,
    output logic                                 o_error,
    output logic [PARAM_ADDR_WIDTH-1:0]          o_local_parameter_addr,
    output logic [PARAM_ADDR_WIDTH-1:0]          o_parameter_addr,
    output logic [Q16_ADDR_WIDTH-1:0]            o_weight_q16_addr,
    output logic [Q64_ADDR_WIDTH-1:0]            o_weight_q64_addr,
    output logic [1:0]                           o_weight_q64_lane,
    output logic [OUTPUT_WAVE_WIDTH-1:0]         o_output_wave,
    output logic [OUTPUT_LANE_WIDTH-1:0]         o_output_lane,
    output logic [INPUT_GROUP_WIDTH-1:0]         o_input_group,
    output logic [INPUT_LANE_WIDTH-1:0]          o_input_lane
);
    localparam logic [63:0] PARAM_LIMIT = 64'd1 << PARAM_ADDR_WIDTH;
    localparam logic [63:0] Q16_LIMIT = 64'd1 << Q16_ADDR_WIDTH;

    logic [63:0] local_addr;
    logic [63:0] parameter_addr;
    logic [63:0] weight_q16_addr;
    logic coordinates_legal;

    always_comb begin
        local_addr =
            (((64'(i_output_channel) * 64'(i_input_channels)) +
              64'(i_input_channel)) * 64'(KERNEL_SIZE * KERNEL_SIZE)) +
            (64'(i_kernel_row) * 64'(KERNEL_SIZE)) +
            64'(i_kernel_col);
        parameter_addr = 64'(i_parameter_base) + local_addr;
        weight_q16_addr = 64'(i_weight_q16_base) + local_addr;

        coordinates_legal =
            i_input_channels != 0 &&
            int'($unsigned(i_input_channels)) <= MAX_INPUT_CHANNELS &&
            i_output_channels != 0 &&
            int'($unsigned(i_output_channels)) <= MAX_OUTPUT_CHANNELS &&
            int'($unsigned(i_input_channel)) <
                int'($unsigned(i_input_channels)) &&
            int'($unsigned(i_output_channel)) <
                int'($unsigned(i_output_channels)) &&
            int'($unsigned(i_kernel_row)) < KERNEL_SIZE &&
            int'($unsigned(i_kernel_col)) < KERNEL_SIZE &&
            parameter_addr < PARAM_LIMIT && weight_q16_addr < Q16_LIMIT;

        o_error = i_valid && !coordinates_legal;
        o_valid = i_valid && coordinates_legal;
        o_local_parameter_addr = PARAM_ADDR_WIDTH'(local_addr);
        o_parameter_addr = PARAM_ADDR_WIDTH'(parameter_addr);
        o_weight_q16_addr = Q16_ADDR_WIDTH'(weight_q16_addr);
        o_weight_q64_addr = Q64_ADDR_WIDTH'(weight_q16_addr >> 2);
        o_weight_q64_lane = weight_q16_addr[1:0];
        o_output_wave = OUTPUT_WAVE_WIDTH'(
            i_output_channel / CHANNEL_INDEX_WIDTH'(OUTPUT_ENGINES));
        o_output_lane = OUTPUT_LANE_WIDTH'(
            i_output_channel % CHANNEL_INDEX_WIDTH'(OUTPUT_ENGINES));
        o_input_group = INPUT_GROUP_WIDTH'(
            i_input_channel / CHANNEL_INDEX_WIDTH'(INPUT_LANES));
        o_input_lane = INPUT_LANE_WIDTH'(
            i_input_channel % CHANNEL_INDEX_WIDTH'(INPUT_LANES));
    end

    initial begin
        if (MAX_INPUT_CHANNELS < 1 || MAX_OUTPUT_CHANNELS < 1 ||
            KERNEL_SIZE < 1 || OUTPUT_ENGINES < 1 || INPUT_LANES < 1)
            $error("SCNN convolution address dimensions must be positive");
        if (PARAM_ADDR_WIDTH > 63 || Q16_ADDR_WIDTH > 63)
            $error("SCNN convolution address widths exceed mapper arithmetic");
    end
endmodule

`default_nettype wire
