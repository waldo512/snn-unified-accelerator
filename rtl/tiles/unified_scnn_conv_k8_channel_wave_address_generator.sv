`default_nettype none

// Generates one spatial/output-channel wave for the twelve-engine SCNN side
// array. Input channels are a scheduler-selected group of at most eight.
module unified_scnn_conv_k8_channel_wave_address_generator #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int ADDRESS_WIDTH = 32,
    parameter int KERNEL_SIZE = 3
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic [15:0]                               i_in_width,
    input  logic [15:0]                               i_in_height,
    input  logic [15:0]                               i_total_in_channels,
    input  logic [15:0]                               i_out_width,
    input  logic [15:0]                               i_out_height,
    input  logic [15:0]                               i_total_out_channels,
    input  logic [9:0]                                i_input_channel_base,
    input  logic [9:0]                                i_input_channel_count,
    input  logic [9:0]                                i_output_channel_base,
    input  logic [9:0]                                i_output_channel_count,
    input  logic [7:0]                                i_stride,
    input  logic [7:0]                                i_padding,
    input  logic [ADDRESS_WIDTH-1:0]                  i_input_base,
    input  logic [ADDRESS_WIDTH-1:0]                  i_weight_q16_base,
    input  logic [ADDRESS_WIDTH-1:0]                  i_output_base,
    output logic                                      o_valid,
    input  logic                                      i_ready,
    input  logic                                      i_result_commit,
    output logic [LANES*4-1:0]                        o_feature_bank_flat,
    output logic [LANES*ADDRESS_WIDTH-1:0]            o_feature_bank_addr_flat,
    output logic                                      o_feature_bank_conflict,
    output logic [LANES-1:0]                          o_lane_mask,
    output logic [ENGINES*LANES*ADDRESS_WIDTH-1:0]    o_weight_q16_addr_flat,
    output logic [ENGINES*ADDRESS_WIDTH-1:0]          o_output_addr_flat,
    output logic [ADDRESS_WIDTH-1:0]                  o_spatial_addr,
    output logic [ENGINES-1:0]                        o_engine_mask,
    output logic                                      o_accumulate_first,
    output logic                                      o_accumulate_last,
    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_config_error
);
    logic [15:0] output_row_q;
    logic [15:0] output_col_q;
    logic [15:0] mac_base_q;
    logic wait_result_q;
    logic config_valid;
    logic final_spatial;
    integer mac_count;
    integer bank_rows;
    integer bank_cols;
    integer local_mac;
    integer global_input_channel;
    integer kernel_row;
    integer kernel_col;
    integer input_row;
    integer input_col;
    integer feature_bank_addr;
    integer global_output_channel;
    integer weight_linear;
    integer output_linear;

    always_comb begin
        mac_count = integer'(i_input_channel_count) * KERNEL_SIZE * KERNEL_SIZE;
        bank_rows = (integer'(i_in_height) + 2) / 3;
        bank_cols = (integer'(i_in_width) + 2) / 3;
        local_mac = 0;
        global_input_channel = 0;
        kernel_row = 0;
        kernel_col = 0;
        input_row = 0;
        input_col = 0;
        feature_bank_addr = 0;
        global_output_channel = 0;
        weight_linear = 0;
        output_linear = 0;
        config_valid = i_in_width != 0 && i_in_height != 0 &&
            i_total_in_channels != 0 && i_out_width != 0 &&
            i_out_height != 0 && i_total_out_channels != 0 &&
            i_input_channel_count != 0 &&
            i_input_channel_count <= 10'(LANES) &&
            16'(i_input_channel_base) + 16'(i_input_channel_count) <=
                i_total_in_channels &&
            i_output_channel_count != 0 &&
            i_output_channel_count <= 10'(ENGINES) &&
            16'(i_output_channel_base) + 16'(i_output_channel_count) <=
                i_total_out_channels &&
            i_stride != 0;
        o_start_ready = !o_busy;
        o_valid = o_busy && !wait_result_q;
        o_feature_bank_flat = '0;
        o_feature_bank_addr_flat = '0;
        o_feature_bank_conflict = 1'b0;
        o_lane_mask = '0;
        o_weight_q16_addr_flat = '0;
        o_output_addr_flat = '0;
        o_spatial_addr = ADDRESS_WIDTH'(
            integer'(output_row_q) * i_out_width + integer'(output_col_q));
        o_engine_mask = '0;
        o_accumulate_first = 1'b0;
        o_accumulate_last = 1'b0;
        final_spatial = 1'b0;

        for (integer engine = 0; engine < ENGINES; engine++) begin
            if (engine < i_output_channel_count) begin
                o_engine_mask[engine] = 1'b1;
                global_output_channel = integer'(i_output_channel_base) + engine;
                output_linear = global_output_channel * i_out_height * i_out_width +
                    integer'(output_row_q) * i_out_width + integer'(output_col_q);
                o_output_addr_flat[engine*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
                    i_output_base + ADDRESS_WIDTH'(output_linear);
                for (integer lane = 0; lane < LANES; lane++) begin
                    local_mac = integer'(mac_base_q) + lane;
                    if (local_mac < mac_count) begin
                        global_input_channel = integer'(i_input_channel_base) +
                            local_mac / (KERNEL_SIZE * KERNEL_SIZE);
                        kernel_row = (local_mac / KERNEL_SIZE) % KERNEL_SIZE;
                        kernel_col = local_mac % KERNEL_SIZE;
                        weight_linear = global_output_channel *
                            i_total_in_channels * KERNEL_SIZE * KERNEL_SIZE +
                            global_input_channel * KERNEL_SIZE * KERNEL_SIZE +
                            kernel_row * KERNEL_SIZE + kernel_col;
                        o_weight_q16_addr_flat[
                            (engine*LANES+lane)*ADDRESS_WIDTH +: ADDRESS_WIDTH
                        ] = i_weight_q16_base + ADDRESS_WIDTH'(weight_linear);
                    end
                end
            end
        end

        for (integer lane = 0; lane < LANES; lane++) begin
            local_mac = integer'(mac_base_q) + lane;
            if (local_mac < mac_count) begin
                global_input_channel = integer'(i_input_channel_base) +
                    local_mac / (KERNEL_SIZE * KERNEL_SIZE);
                kernel_row = (local_mac / KERNEL_SIZE) % KERNEL_SIZE;
                kernel_col = local_mac % KERNEL_SIZE;
                input_row = integer'(output_row_q) * i_stride + kernel_row -
                            integer'(i_padding);
                input_col = integer'(output_col_q) * i_stride + kernel_col -
                            integer'(i_padding);
                if (input_row >= 0 && input_row < i_in_height &&
                    input_col >= 0 && input_col < i_in_width) begin
                    feature_bank_addr = global_input_channel * bank_rows * bank_cols +
                        (input_row / 3) * bank_cols + (input_col / 3);
                    o_feature_bank_flat[lane*4 +: 4] =
                        4'((input_row % 3) * 3 + (input_col % 3));
                    o_feature_bank_addr_flat[
                        lane*ADDRESS_WIDTH +: ADDRESS_WIDTH
                    ] = i_input_base + ADDRESS_WIDTH'(feature_bank_addr);
                    o_lane_mask[lane] = 1'b1;
                end
            end
        end
        for (integer lane = 0; lane < LANES; lane++)
            for (integer prior = 0; prior < lane; prior++)
                if (o_lane_mask[lane] && o_lane_mask[prior] &&
                    o_feature_bank_flat[lane*4 +: 4] ==
                    o_feature_bank_flat[prior*4 +: 4])
                    o_feature_bank_conflict = 1'b1;

        o_accumulate_first = mac_base_q == 0;
        o_accumulate_last = integer'(mac_base_q) + LANES >= mac_count;
        final_spatial = output_col_q + 1 >= i_out_width &&
                        output_row_q + 1 >= i_out_height;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
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
                output_row_q <= '0;
                output_col_q <= '0;
                mac_base_q <= '0;
                wait_result_q <= 1'b0;
                o_busy <= config_valid;
                o_config_error <= !config_valid;
            end else if (i_start && !o_start_ready) begin
                o_config_error <= 1'b1;
            end else if (wait_result_q && i_result_commit) begin
                wait_result_q <= 1'b0;
                mac_base_q <= '0;
                if (final_spatial) begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end else if (output_col_q + 1 < i_out_width) begin
                    output_col_q <= output_col_q + 1'b1;
                end else begin
                    output_col_q <= '0;
                    output_row_q <= output_row_q + 1'b1;
                end
            end else if (o_valid && i_ready) begin
                if (o_accumulate_last)
                    wait_result_q <= 1'b1;
                else
                    mac_base_q <= mac_base_q + 16'(LANES);
            end
        end
    end
endmodule

`default_nettype wire
