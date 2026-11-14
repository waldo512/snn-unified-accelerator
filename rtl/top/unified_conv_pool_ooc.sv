`timescale 1ns/1ps
`default_nettype none

module unified_conv_pool_ooc (
    input  wire logic                 clk,
    input  wire logic                 rst_n,
    input  wire logic                 i_start,
    input  wire logic [15:0]          i_dimension,
    input  wire logic                 i_ready,
    output logic                      o_conv_valid,
    output logic [31:0]               o_input_addr,
    output logic [255:0]              o_weight_addr_flat,
    output logic [7:0]                o_lane_mask,
    input  wire logic                 i_pool_valid,
    input  wire logic signed [31:0]   i_pool_value,
    input  wire logic                 i_pool_first,
    input  wire logic                 i_pool_last,
    output logic                      o_pool_valid,
    output logic signed [31:0]        o_pool_value
);
    unified_conv_address_generator u_conv (
        .clk(clk), .rst_n(rst_n), .i_start(i_start), .o_start_ready(),
        .i_in_width(i_dimension), .i_in_height(i_dimension), .i_in_channels(16'd64),
        .i_out_width(i_dimension), .i_out_height(i_dimension), .i_out_channels(16'd128),
        .i_kernel_width(8'd3), .i_kernel_height(8'd3),
        .i_stride_x(8'd1), .i_stride_y(8'd1), .i_pad_x(8'd1), .i_pad_y(8'd1),
        .i_input_origin_addr(-$signed(({16'd0, i_dimension} << 6) + 32'd64)),
        .i_input_x_step(32'd64), .i_input_y_step({16'd0, i_dimension} << 6),
        .i_input_row_skip(({16'd0, i_dimension} - 32'd3) << 6),
        .i_weight_base_addr(32'd0), .i_output_base_addr(32'd0),
        .o_valid(o_conv_valid), .i_ready(i_ready), .o_input_in_bounds(),
        .o_input_addr(o_input_addr), .o_weight_addr_flat(o_weight_addr_flat),
        .o_output_addr(), .o_lane_mask(o_lane_mask), .o_accumulate_first(),
        .o_accumulate_last(), .o_busy(), .o_done(), .o_config_error()
    );
    unified_pool2d_unit u_pool (
        .clk(clk), .rst_n(rst_n), .i_valid(i_pool_valid), .o_ready(),
        .i_value(i_pool_value), .i_window_first(i_pool_first),
        .i_window_last(i_pool_last), .i_average_mode(1'b0),
        .i_average_shift(5'd2), .o_valid(o_pool_valid), .i_ready(i_ready),
        .o_value(o_pool_value), .o_protocol_error()
    );
endmodule

`default_nettype wire
