`default_nettype none

// Generates one spatial/output-channel wave for the twelve-engine SCNN side
// array. Runtime emission uses registered bases and counter updates only; the
// variable geometry arithmetic is isolated in bounded preparation states.
module unified_scnn_conv_k8_channel_wave_address_generator_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int ADDRESS_WIDTH = 32,
    parameter int KERNEL_SIZE = 3
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
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
    input  logic                                      i_collapse_current_spatial,
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
    localparam int KERNEL_WORDS = 9;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_PREP_GEOMETRY,
        STATE_PREP_PRODUCTS,
        STATE_PREP_BASES,
        STATE_PREP_SPATIAL,
        STATE_RUN,
        STATE_WAIT_RESULT
    } state_t;

    state_t state_q;

    logic [15:0] in_width_q;
    logic [15:0] in_height_q;
    logic [15:0] total_in_channels_q;
    logic [15:0] out_width_q;
    logic [15:0] out_height_q;
    logic [9:0] input_channel_base_q;
    logic [9:0] input_channel_count_q;
    logic [9:0] output_channel_base_q;
    logic [9:0] output_channel_count_q;
    logic [7:0] stride_q;
    logic [7:0] padding_q;
    logic [ADDRESS_WIDTH-1:0] input_base_q;
    logic [ADDRESS_WIDTH-1:0] weight_q16_base_q;
    logic [ADDRESS_WIDTH-1:0] output_base_q;

    logic [15:0] output_row_q;
    logic [15:0] output_col_q;
    logic [ADDRESS_WIDTH-1:0] spatial_index_q;
    logic signed [17:0] input_origin_row_q;
    logic signed [17:0] input_origin_col_q;

    logic [15:0] bank_rows_q;
    logic [15:0] bank_cols_q;
    logic [ADDRESS_WIDTH-1:0] bank_plane_q;
    logic [ADDRESS_WIDTH-1:0] output_plane_q;
    logic [ADDRESS_WIDTH-1:0] weight_output_stride_q;
    logic [ADDRESS_WIDTH-1:0] weight_output_offset_q;
    logic [ADDRESS_WIDTH-1:0] output_channel_offset_q;
    logic [15:0] mac_count_q;
    logic collapse_current_spatial;

    logic [ADDRESS_WIDTH-1:0] initial_feature_channel_base_q;
    logic [ADDRESS_WIDTH-1:0] feature_channel_base_q;
    logic [ADDRESS_WIDTH-1:0] next_feature_channel_base_q;
    logic [ADDRESS_WIDTH-1:0] weight_engine_base_q [0:ENGINES-1];
    logic [ADDRESS_WIDTH-1:0] output_engine_base_q [0:ENGINES-1];
    logic [ENGINES-1:0] engine_mask_q;

    logic [ADDRESS_WIDTH-1:0] spatial_feature_offset_q [0:KERNEL_WORDS-1];
    logic [3:0] spatial_feature_bank_q [0:KERNEL_WORDS-1];
    logic [KERNEL_WORDS-1:0] spatial_tap_valid_q;

    logic [15:0] mac_base_q;
    logic [3:0] kernel_base_q;
    logic config_valid;
    logic final_spatial;

    always_comb begin
        config_valid = KERNEL_SIZE == 3 && LANES <= KERNEL_WORDS &&
            i_in_width != 0 && i_in_height != 0 &&
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

        o_start_ready = !i_abort && state_q == STATE_IDLE;
        o_valid = !i_abort && state_q == STATE_RUN;
        o_busy = state_q != STATE_IDLE;
        o_feature_bank_flat = '0;
        o_feature_bank_addr_flat = '0;
        o_feature_bank_conflict = 1'b0;
        o_lane_mask = '0;
        o_weight_q16_addr_flat = '0;
        o_output_addr_flat = '0;
        o_spatial_addr = spatial_index_q;
        o_engine_mask = engine_mask_q;
        collapse_current_spatial = i_collapse_current_spatial &&
                                   mac_base_q == 0;
        o_accumulate_first = mac_base_q == 0;
        o_accumulate_last = collapse_current_spatial ||
            16'(mac_base_q + 16'(LANES)) >= mac_count_q;
        final_spatial = output_col_q + 1 >= out_width_q &&
                        output_row_q + 1 >= out_height_q;

        for (integer engine = 0; engine < ENGINES; engine++) begin
            if (engine_mask_q[engine]) begin
                o_output_addr_flat[engine*ADDRESS_WIDTH +: ADDRESS_WIDTH] =
                    output_engine_base_q[engine] + spatial_index_q;
                for (integer lane = 0; lane < LANES; lane++) begin
                    if (!collapse_current_spatial &&
                        16'(mac_base_q + 16'(lane)) < mac_count_q)
                        o_weight_q16_addr_flat[
                            (engine*LANES+lane)*ADDRESS_WIDTH +: ADDRESS_WIDTH
                        ] = weight_engine_base_q[engine] +
                            ADDRESS_WIDTH'(mac_base_q) + ADDRESS_WIDTH'(lane);
                end
            end
        end

        for (integer lane = 0; lane < LANES; lane++) begin
            logic [4:0] kernel_sum;
            logic [3:0] kernel_index;
            logic channel_carry;
            kernel_sum = 5'(kernel_base_q) + 5'(lane);
            channel_carry = kernel_sum >= KERNEL_WORDS;
            kernel_index = channel_carry ?
                4'(kernel_sum - KERNEL_WORDS) : 4'(kernel_sum);
            if (!collapse_current_spatial &&
                16'(mac_base_q + 16'(lane)) < mac_count_q &&
                spatial_tap_valid_q[kernel_index]) begin
                o_feature_bank_flat[lane*4 +: 4] =
                    spatial_feature_bank_q[kernel_index];
                o_feature_bank_addr_flat[
                    lane*ADDRESS_WIDTH +: ADDRESS_WIDTH
                ] = (channel_carry ? next_feature_channel_base_q :
                                      feature_channel_base_q) +
                    spatial_feature_offset_q[kernel_index];
                o_lane_mask[lane] = 1'b1;
            end
        end

        for (integer lane = 0; lane < LANES; lane++)
            for (integer prior = 0; prior < lane; prior++)
                if (o_lane_mask[lane] && o_lane_mask[prior] &&
                    o_feature_bank_flat[lane*4 +: 4] ==
                    o_feature_bank_flat[prior*4 +: 4])
                    o_feature_bank_conflict = 1'b1;
    end

    // This product is consumed only after the PREP sequence completes. Keeping
    // the data register reset-free lets Vivado absorb it into the DSP output
    // register; state_q remains the validity authority across reset and abort.
    always_ff @(posedge i_clk) begin
        if (state_q == STATE_PREP_BASES)
            initial_feature_channel_base_q <= input_base_q +
                ADDRESS_WIDTH'(input_channel_base_q) * bank_plane_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            in_width_q <= '0;
            in_height_q <= '0;
            total_in_channels_q <= '0;
            out_width_q <= '0;
            out_height_q <= '0;
            input_channel_base_q <= '0;
            input_channel_count_q <= '0;
            output_channel_base_q <= '0;
            output_channel_count_q <= '0;
            stride_q <= '0;
            padding_q <= '0;
            input_base_q <= '0;
            weight_q16_base_q <= '0;
            output_base_q <= '0;
            output_row_q <= '0;
            output_col_q <= '0;
            spatial_index_q <= '0;
            input_origin_row_q <= '0;
            input_origin_col_q <= '0;
            bank_rows_q <= '0;
            bank_cols_q <= '0;
            bank_plane_q <= '0;
            output_plane_q <= '0;
            weight_output_stride_q <= '0;
            weight_output_offset_q <= '0;
            output_channel_offset_q <= '0;
            mac_count_q <= '0;
            feature_channel_base_q <= '0;
            next_feature_channel_base_q <= '0;
            engine_mask_q <= '0;
            spatial_tap_valid_q <= '0;
            mac_base_q <= '0;
            kernel_base_q <= '0;
            o_done <= 1'b0;
            o_config_error <= 1'b0;
            for (integer engine = 0; engine < ENGINES; engine++) begin
                weight_engine_base_q[engine] <= '0;
                output_engine_base_q[engine] <= '0;
            end
            for (integer tap = 0; tap < KERNEL_WORDS; tap++) begin
                spatial_feature_offset_q[tap] <= '0;
                spatial_feature_bank_q[tap] <= '0;
            end
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            output_row_q <= '0;
            output_col_q <= '0;
            spatial_index_q <= '0;
            input_origin_row_q <= '0;
            input_origin_col_q <= '0;
            feature_channel_base_q <= '0;
            next_feature_channel_base_q <= '0;
            spatial_tap_valid_q <= '0;
            mac_base_q <= '0;
            kernel_base_q <= '0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            case (state_q)
                STATE_IDLE: begin
                    if (i_start) begin
                        o_config_error <= !config_valid;
                        if (config_valid) begin
                            in_width_q <= i_in_width;
                            in_height_q <= i_in_height;
                            total_in_channels_q <= i_total_in_channels;
                            out_width_q <= i_out_width;
                            out_height_q <= i_out_height;
                            input_channel_base_q <= i_input_channel_base;
                            input_channel_count_q <= i_input_channel_count;
                            output_channel_base_q <= i_output_channel_base;
                            output_channel_count_q <= i_output_channel_count;
                            stride_q <= i_stride;
                            padding_q <= i_padding;
                            input_base_q <= i_input_base;
                            weight_q16_base_q <= i_weight_q16_base;
                            output_base_q <= i_output_base;
                            output_row_q <= '0;
                            output_col_q <= '0;
                            spatial_index_q <= '0;
                            input_origin_row_q <= -18'($unsigned(i_padding));
                            input_origin_col_q <= -18'($unsigned(i_padding));
                            mac_base_q <= '0;
                            kernel_base_q <= '0;
                            state_q <= STATE_PREP_GEOMETRY;
                        end
                    end
                end

                STATE_PREP_GEOMETRY: begin
                    bank_rows_q <= (in_height_q + 16'd2) / 3;
                    bank_cols_q <= (in_width_q + 16'd2) / 3;
                    output_plane_q <= ADDRESS_WIDTH'(out_width_q) *
                                      ADDRESS_WIDTH'(out_height_q);
                    weight_output_stride_q <=
                        (ADDRESS_WIDTH'(total_in_channels_q) << 3) +
                         ADDRESS_WIDTH'(total_in_channels_q);
                    mac_count_q <= (16'(input_channel_count_q) << 3) +
                                   16'(input_channel_count_q);
                    state_q <= STATE_PREP_PRODUCTS;
                end

                STATE_PREP_PRODUCTS: begin
                    bank_plane_q <= ADDRESS_WIDTH'(bank_rows_q) *
                                    ADDRESS_WIDTH'(bank_cols_q);
                    weight_output_offset_q <=
                        ADDRESS_WIDTH'(output_channel_base_q) *
                        weight_output_stride_q;
                    output_channel_offset_q <=
                        ADDRESS_WIDTH'(output_channel_base_q) * output_plane_q;
                    state_q <= STATE_PREP_BASES;
                end

                STATE_PREP_BASES: begin
                    feature_channel_base_q <= input_base_q +
                        ADDRESS_WIDTH'(input_channel_base_q) * bank_plane_q;
                    next_feature_channel_base_q <= input_base_q +
                        (ADDRESS_WIDTH'(input_channel_base_q) + 1'b1) *
                        bank_plane_q;
                    for (integer engine = 0; engine < ENGINES; engine++) begin
                        engine_mask_q[engine] <= engine < output_channel_count_q;
                        weight_engine_base_q[engine] <= weight_q16_base_q +
                            weight_output_offset_q +
                            ((ADDRESS_WIDTH'(input_channel_base_q) << 3) +
                             ADDRESS_WIDTH'(input_channel_base_q)) +
                            ADDRESS_WIDTH'(engine) * weight_output_stride_q;
                        output_engine_base_q[engine] <= output_base_q +
                            output_channel_offset_q +
                            ADDRESS_WIDTH'(engine) * output_plane_q;
                    end
                    state_q <= STATE_PREP_SPATIAL;
                end

                STATE_PREP_SPATIAL: begin
                    for (integer row = 0; row < 3; row++) begin
                        for (integer col = 0; col < 3; col++) begin
                            automatic integer tap = row * 3 + col;
                            automatic integer input_row =
                                $signed(input_origin_row_q) + row;
                            automatic integer input_col =
                                $signed(input_origin_col_q) + col;
                            spatial_tap_valid_q[tap] <=
                                input_row >= 0 && input_row < in_height_q &&
                                input_col >= 0 && input_col < in_width_q;
                            if (input_row >= 0 && input_col >= 0) begin
                                spatial_feature_offset_q[tap] <=
                                    ADDRESS_WIDTH'((input_row / 3) * bank_cols_q +
                                                   input_col / 3);
                                spatial_feature_bank_q[tap] <=
                                    4'((input_row % 3) * 3 + input_col % 3);
                            end else begin
                                spatial_feature_offset_q[tap] <= '0;
                                spatial_feature_bank_q[tap] <= '0;
                            end
                        end
                    end
                    state_q <= STATE_RUN;
                end

                STATE_RUN: begin
                    if (i_start)
                        o_config_error <= 1'b1;
                    if (o_valid && i_ready) begin
                        if (o_accumulate_last) begin
                            state_q <= STATE_WAIT_RESULT;
                        end else begin
                            mac_base_q <= mac_base_q + 16'(LANES);
                            if (integer'(kernel_base_q) + LANES >= KERNEL_WORDS) begin
                                kernel_base_q <= 4'(
                                    integer'(kernel_base_q) + LANES - KERNEL_WORDS);
                                feature_channel_base_q <=
                                    next_feature_channel_base_q;
                                next_feature_channel_base_q <=
                                    next_feature_channel_base_q + bank_plane_q;
                            end else begin
                                kernel_base_q <= kernel_base_q + 4'(LANES);
                            end
                        end
                    end
                end

                STATE_WAIT_RESULT: begin
                    if (i_start)
                        o_config_error <= 1'b1;
                    if (i_result_commit) begin
                        mac_base_q <= '0;
                        kernel_base_q <= '0;
                        feature_channel_base_q <= initial_feature_channel_base_q;
                        next_feature_channel_base_q <=
                            initial_feature_channel_base_q + bank_plane_q;
                        if (final_spatial) begin
                            state_q <= STATE_IDLE;
                            o_done <= 1'b1;
                        end else begin
                            spatial_index_q <= spatial_index_q + 1'b1;
                            if (output_col_q + 1 < out_width_q) begin
                                output_col_q <= output_col_q + 1'b1;
                                input_origin_col_q <= input_origin_col_q +
                                                      18'($unsigned(stride_q));
                            end else begin
                                output_col_q <= '0;
                                output_row_q <= output_row_q + 1'b1;
                                input_origin_col_q <=
                                    -18'($unsigned(padding_q));
                                input_origin_row_q <= input_origin_row_q +
                                                      18'($unsigned(stride_q));
                            end
                            state_q <= STATE_PREP_SPATIAL;
                        end
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
