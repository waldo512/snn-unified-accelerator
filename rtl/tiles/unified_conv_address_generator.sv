`timescale 1ns/1ps
`default_nettype none

module unified_conv_address_generator #(
    parameter integer LANES = 8,
    parameter integer ADDRESS_WIDTH = 32
) (
    input  wire logic                            clk,
    input  wire logic                            rst_n,
    input  wire logic                            i_start,
    output logic                                 o_start_ready,
    input  wire logic [15:0]                     i_in_width,
    input  wire logic [15:0]                     i_in_height,
    input  wire logic [15:0]                     i_in_channels,
    input  wire logic [15:0]                     i_out_width,
    input  wire logic [15:0]                     i_out_height,
    input  wire logic [15:0]                     i_out_channels,
    input  wire logic [7:0]                      i_kernel_width,
    input  wire logic [7:0]                      i_kernel_height,
    input  wire logic [7:0]                      i_stride_x,
    input  wire logic [7:0]                      i_stride_y,
    input  wire logic [7:0]                      i_pad_x,
    input  wire logic [7:0]                      i_pad_y,
    input  wire logic signed [ADDRESS_WIDTH-1:0] i_input_origin_addr,
    input  wire logic [ADDRESS_WIDTH-1:0]        i_input_x_step,
    input  wire logic [ADDRESS_WIDTH-1:0]        i_input_y_step,
    input  wire logic signed [ADDRESS_WIDTH-1:0] i_input_row_skip,
    input  wire logic [ADDRESS_WIDTH-1:0]        i_weight_base_addr,
    input  wire logic [ADDRESS_WIDTH-1:0]        i_output_base_addr,

    output logic                                 o_valid,
    input  wire logic                            i_ready,
    output logic                                 o_input_in_bounds,
    output logic [ADDRESS_WIDTH-1:0]             o_input_addr,
    output logic [LANES*ADDRESS_WIDTH-1:0]       o_weight_addr_flat,
    output logic [ADDRESS_WIDTH-1:0]             o_output_addr,
    output logic [LANES-1:0]                     o_lane_mask,
    output logic                                 o_accumulate_first,
    output logic                                 o_accumulate_last,
    output logic                                 o_busy,
    output logic                                 o_done,
    output logic                                 o_config_error
);
    logic [15:0] out_x_q;
    logic [15:0] out_y_q;
    logic [15:0] out_channel_base_q;
    logic [15:0] in_channel_q;
    logic [7:0] kernel_x_q;
    logic [7:0] kernel_y_q;
    logic signed [16:0] input_x_q;
    logic signed [16:0] input_y_q;
    logic signed [16:0] pixel_origin_x_q;
    logic signed [16:0] pixel_origin_y_q;
    logic signed [ADDRESS_WIDTH-1:0] row_origin_addr_q;
    logic signed [ADDRESS_WIDTH-1:0] pixel_origin_addr_q;
    logic signed [ADDRESS_WIDTH-1:0] input_addr_q;
    logic [ADDRESS_WIDTH-1:0] weight_addr_q;
    logic [ADDRESS_WIDTH-1:0] output_pixel_base_q;
    logic config_valid;
    logic final_beat;
    integer lane_i;

    always_comb begin
        config_valid =
            (i_in_width != 0) && (i_in_height != 0) &&
            (i_in_channels != 0) && (i_out_width != 0) &&
            (i_out_height != 0) && (i_out_channels != 0) &&
            (i_kernel_width != 0) && (i_kernel_height != 0) &&
            (i_stride_x != 0) && (i_stride_y != 0) &&
            (i_kernel_width <= i_in_width);
        o_start_ready = !o_busy;
        o_valid = o_busy;
        o_input_in_bounds =
            (input_x_q >= 0) && (input_y_q >= 0) &&
            (input_x_q < $signed({1'b0, i_in_width})) &&
            (input_y_q < $signed({1'b0, i_in_height}));
        o_input_addr = input_addr_q;
        o_output_addr = output_pixel_base_q + out_channel_base_q;
        o_lane_mask = '0;
        o_weight_addr_flat = '0;
        for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
            if (out_channel_base_q + lane_i < i_out_channels) begin
                o_lane_mask[lane_i] = 1'b1;
                // One 64-bit working-set word supplies all K8 lane weights.
                o_weight_addr_flat[lane_i*ADDRESS_WIDTH +: ADDRESS_WIDTH] = weight_addr_q;
            end
        end
        o_accumulate_first =
            (in_channel_q == 0) && (kernel_y_q == 0) && (kernel_x_q == 0);
        o_accumulate_last =
            (in_channel_q + 1 >= i_in_channels) &&
            (kernel_y_q + 1 >= i_kernel_height) &&
            (kernel_x_q + 1 >= i_kernel_width);
        final_beat =
            o_accumulate_last &&
            (out_channel_base_q + LANES >= i_out_channels) &&
            (out_x_q + 1 >= i_out_width) &&
            (out_y_q + 1 >= i_out_height);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_x_q <= 0;
            out_y_q <= 0;
            out_channel_base_q <= 0;
            in_channel_q <= 0;
            kernel_x_q <= 0;
            kernel_y_q <= 0;
            input_x_q <= 0;
            input_y_q <= 0;
            pixel_origin_x_q <= 0;
            pixel_origin_y_q <= 0;
            row_origin_addr_q <= 0;
            pixel_origin_addr_q <= 0;
            input_addr_q <= 0;
            weight_addr_q <= 0;
            output_pixel_base_q <= 0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_config_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_start && o_start_ready) begin
                out_x_q <= 0;
                out_y_q <= 0;
                out_channel_base_q <= 0;
                in_channel_q <= 0;
                kernel_x_q <= 0;
                kernel_y_q <= 0;
                input_x_q <= -$signed({1'b0, i_pad_x});
                input_y_q <= -$signed({1'b0, i_pad_y});
                pixel_origin_x_q <= -$signed({1'b0, i_pad_x});
                pixel_origin_y_q <= -$signed({1'b0, i_pad_y});
                row_origin_addr_q <= i_input_origin_addr;
                pixel_origin_addr_q <= i_input_origin_addr;
                input_addr_q <= i_input_origin_addr;
                weight_addr_q <= i_weight_base_addr;
                output_pixel_base_q <= i_output_base_addr;
                o_busy <= config_valid;
                o_config_error <= !config_valid;
            end else if (o_valid && i_ready) begin
                if (final_beat) begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end else begin
                    weight_addr_q <= weight_addr_q + 1'b1;
                    if (in_channel_q + 1 < i_in_channels) begin
                        in_channel_q <= in_channel_q + 1'b1;
                        input_addr_q <= input_addr_q + 1'b1;
                    end else begin
                        in_channel_q <= 0;
                        if (kernel_x_q + 1 < i_kernel_width) begin
                            kernel_x_q <= kernel_x_q + 1'b1;
                            input_x_q <= input_x_q + 1'b1;
                            input_addr_q <= input_addr_q + 1'b1;
                        end else begin
                            kernel_x_q <= 0;
                            if (kernel_y_q + 1 < i_kernel_height) begin
                                kernel_y_q <= kernel_y_q + 1'b1;
                                input_y_q <= input_y_q + 1'b1;
                                input_addr_q <= input_addr_q + i_input_row_skip + 1'b1;
                                input_x_q <= pixel_origin_x_q;
                            end else begin
                                kernel_y_q <= 0;
                                if (out_channel_base_q + LANES < i_out_channels) begin
                                    out_channel_base_q <= out_channel_base_q + LANES;
                                    input_x_q <= pixel_origin_x_q;
                                    input_y_q <= pixel_origin_y_q;
                                    input_addr_q <= pixel_origin_addr_q;
                                end else begin
                                    out_channel_base_q <= 0;
                                    weight_addr_q <= i_weight_base_addr;
                                    output_pixel_base_q <= output_pixel_base_q + i_out_channels;
                                    if (out_x_q + 1 < i_out_width) begin
                                        out_x_q <= out_x_q + 1'b1;
                                        pixel_origin_x_q <= pixel_origin_x_q +
                                            $signed({1'b0, i_stride_x});
                                        input_x_q <= pixel_origin_x_q +
                                            $signed({1'b0, i_stride_x});
                                        input_y_q <= pixel_origin_y_q;
                                        pixel_origin_addr_q <= pixel_origin_addr_q + i_input_x_step;
                                        input_addr_q <= pixel_origin_addr_q + i_input_x_step;
                                    end else begin
                                        out_x_q <= 0;
                                        out_y_q <= out_y_q + 1'b1;
                                        pixel_origin_x_q <= -$signed({1'b0, i_pad_x});
                                        pixel_origin_y_q <= pixel_origin_y_q +
                                            $signed({1'b0, i_stride_y});
                                        input_x_q <= -$signed({1'b0, i_pad_x});
                                        input_y_q <= pixel_origin_y_q +
                                            $signed({1'b0, i_stride_y});
                                        row_origin_addr_q <= row_origin_addr_q + i_input_y_step;
                                        pixel_origin_addr_q <= row_origin_addr_q + i_input_y_step;
                                        input_addr_q <= row_origin_addr_q + i_input_y_step;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
