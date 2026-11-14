`default_nettype none

module unified_scnn_conv_k8_address_generator #(
    parameter int LANES = 8,
    parameter int ADDRESS_WIDTH = 32,
    parameter int KERNEL_SIZE = 3
) (
    input  logic                              i_clk,
    input  logic                              i_rst_n,
    input  logic                              i_start,
    output logic                              o_start_ready,
    input  logic [15:0]                       i_in_width,
    input  logic [15:0]                       i_in_height,
    input  logic [15:0]                       i_in_channels,
    input  logic [15:0]                       i_out_width,
    input  logic [15:0]                       i_out_height,
    input  logic [15:0]                       i_out_channels,
    input  logic [7:0]                        i_stride,
    input  logic [7:0]                        i_padding,
    input  logic [ADDRESS_WIDTH-1:0]          i_input_base,
    input  logic [ADDRESS_WIDTH-1:0]          i_weight_base,
    input  logic [ADDRESS_WIDTH-1:0]          i_output_base,
    output logic                              o_valid,
    input  logic                              i_ready,
    input  logic                              i_result_commit,
    output logic [LANES*ADDRESS_WIDTH-1:0]    o_input_addr_flat,
    output logic [LANES*ADDRESS_WIDTH-1:0]    o_weight_addr_flat,
    output logic [LANES*4-1:0]                o_feature_bank_flat,
    output logic [LANES*ADDRESS_WIDTH-1:0]    o_feature_bank_addr_flat,
    output logic                              o_feature_bank_conflict,
    output logic [LANES-1:0]                  o_lane_mask,
    output logic [ADDRESS_WIDTH-1:0]          o_output_addr,
    output logic                              o_accumulate_first,
    output logic                              o_accumulate_last,
    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_config_error
);

    logic [15:0] output_channel_q;
    logic [15:0] output_row_q;
    logic [15:0] output_col_q;
    logic [15:0] mac_base_q;
    logic wait_result_q;
    logic config_valid;
    logic final_beat;
    integer lane_index;
    integer mac_index;
    integer input_channel;
    integer kernel_row;
    integer kernel_col;
    integer input_row;
    integer input_col;
    integer input_linear;
    integer weight_linear;
    integer output_linear;
    integer mac_count;
    integer bank_rows;
    integer bank_cols;
    integer feature_bank_local;
    integer compare_lane;

    always_comb begin
        mac_index = 0;
        input_channel = 0;
        kernel_row = 0;
        kernel_col = 0;
        input_row = 0;
        input_col = 0;
        input_linear = 0;
        weight_linear = 0;
        output_linear = 0;
        mac_count = i_in_channels * KERNEL_SIZE * KERNEL_SIZE;
        bank_rows = (integer'(i_in_height) + 2) / 3;
        bank_cols = (integer'(i_in_width) + 2) / 3;
        feature_bank_local = 0;
        config_valid = i_in_width != 0 && i_in_height != 0 &&
            i_in_channels != 0 && i_out_width != 0 && i_out_height != 0 &&
            i_out_channels != 0 && i_stride != 0;
        o_start_ready = !o_busy;
        o_valid = o_busy && !wait_result_q;
        o_input_addr_flat = '0;
        o_weight_addr_flat = '0;
        o_feature_bank_flat = '0;
        o_feature_bank_addr_flat = '0;
        o_feature_bank_conflict = 1'b0;
        o_lane_mask = '0;
        output_linear = output_channel_q * i_out_height * i_out_width +
                        output_row_q * i_out_width + integer'(output_col_q);
        o_output_addr = i_output_base + ADDRESS_WIDTH'(output_linear);
        for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
            mac_index = integer'(mac_base_q) + lane_index;
            if (mac_index < mac_count) begin
                input_channel = mac_index / (KERNEL_SIZE * KERNEL_SIZE);
                kernel_row = (mac_index / KERNEL_SIZE) % KERNEL_SIZE;
                kernel_col = mac_index % KERNEL_SIZE;
                input_row = output_row_q * i_stride + kernel_row -
                            integer'(i_padding);
                input_col = output_col_q * i_stride + kernel_col -
                            integer'(i_padding);
                weight_linear = output_channel_q * mac_count + mac_index;
                o_weight_addr_flat[lane_index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
                    i_weight_base + ADDRESS_WIDTH'(weight_linear);
                if (input_row >= 0 && input_row < i_in_height &&
                    input_col >= 0 && input_col < i_in_width) begin
                    input_linear = input_channel * i_in_height * i_in_width +
                                   input_row * i_in_width + input_col;
                    o_input_addr_flat[lane_index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
                        i_input_base + ADDRESS_WIDTH'(input_linear);
                    feature_bank_local = input_channel * bank_rows * bank_cols +
                        (input_row / 3) * bank_cols + (input_col / 3);
                    o_feature_bank_flat[lane_index*4 +: 4] =
                        4'((input_row % 3) * 3 + (input_col % 3));
                    o_feature_bank_addr_flat[
                        lane_index*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
                        ADDRESS_WIDTH'(feature_bank_local);
                    o_lane_mask[lane_index] = 1'b1;
                end
            end
        end
        for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
            for (compare_lane = lane_index + 1; compare_lane < LANES;
                 compare_lane = compare_lane + 1) begin
                if (o_lane_mask[lane_index] && o_lane_mask[compare_lane] &&
                    o_feature_bank_flat[lane_index*4 +: 4] ==
                    o_feature_bank_flat[compare_lane*4 +: 4])
                    o_feature_bank_conflict = 1'b1;
            end
        end
        o_accumulate_first = mac_base_q == 0;
        o_accumulate_last = integer'(mac_base_q) + LANES >= mac_count;
        final_beat = o_accumulate_last &&
            output_col_q + 1 >= i_out_width &&
            output_row_q + 1 >= i_out_height &&
            output_channel_q + 1 >= i_out_channels;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            output_channel_q <= '0;
            output_row_q <= '0;
            output_col_q <= '0;
            mac_base_q <= '0;
            wait_result_q <= 1'b0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_config_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_start && o_start_ready) begin
                output_channel_q <= '0;
                output_row_q <= '0;
                output_col_q <= '0;
                mac_base_q <= '0;
                wait_result_q <= 1'b0;
                o_busy <= config_valid;
                o_config_error <= !config_valid;
            end else if (wait_result_q && i_result_commit) begin
                wait_result_q <= 1'b0;
                mac_base_q <= '0;
                if (final_beat) begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end else if (output_col_q + 1 < i_out_width) begin
                    output_col_q <= output_col_q + 1'b1;
                end else begin
                    output_col_q <= '0;
                    if (output_row_q + 1 < i_out_height) begin
                        output_row_q <= output_row_q + 1'b1;
                    end else begin
                        output_row_q <= '0;
                        output_channel_q <= output_channel_q + 1'b1;
                    end
                end
            end else if (o_valid && i_ready) begin
                if (o_accumulate_last) begin
                    wait_result_q <= 1'b1;
                end else begin
                    mac_base_q <= mac_base_q + 16'(LANES);
                end
            end
        end
    end

endmodule

`default_nettype wire
