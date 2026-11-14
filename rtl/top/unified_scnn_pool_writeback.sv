`default_nettype none

module unified_scnn_pool_writeback #(
    parameter int MAX_WIDTH = 32,
    parameter int BANK_ADDR_WIDTH = 14
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_start,
    output logic                         o_start_ready,
    input  logic [15:0]                  i_width,
    input  logic [15:0]                  i_height,
    input  logic [15:0]                  i_channels,
    input  logic [1:0]                   i_pool_mode,
    input  logic [31:0]                  i_input_base,
    input  logic                         i_output_slot,

    input  logic                         i_valid,
    output logic                         o_ready,
    input  logic [31:0]                  i_addr,
    input  logic signed [15:0]           i_value_q8_8,

    output logic                         o_write_valid,
    input  logic                         i_write_ready,
    output logic                         o_write_slot,
    output logic [3:0]                   o_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]   o_write_addr,
    output logic signed [15:0]           o_write_data_q8_8,

    output logic                         o_busy,
    output logic                         o_done,
    output logic                         o_config_error,
    output logic                         o_protocol_error
);
    localparam int COL_INDEX_WIDTH = $clog2(MAX_WIDTH);
    localparam logic [1:0] POOL_IDENTITY = 2'd0;
    localparam logic [1:0] POOL_MAX2 = 2'd1;
    localparam logic [1:0] POOL_AAVG2 = 2'd2;

    logic [15:0] width_q;
    logic [15:0] height_q;
    logic [15:0] channels_q;
    logic [1:0] pool_mode_q;
    logic output_slot_q;
    logic [31:0] expected_addr_q;
    logic [15:0] channel_q;
    logic [15:0] row_q;
    logic [15:0] col_q;
    logic signed [15:0] line_buffer_q [0:MAX_WIDTH-1];
    logic signed [15:0] bottom_left_q;
    logic config_valid;
    logic produces_output;
    logic final_input;
    logic [15:0] output_width;
    logic [15:0] output_height;
    logic [15:0] output_row;
    logic [15:0] output_col;
    logic [COL_INDEX_WIDTH-1:0] col_index;
    logic [COL_INDEX_WIDTH-1:0] prior_col_index;
    logic [15:0] bank_rows;
    logic [15:0] bank_cols;
    logic signed [15:0] max_value;
    logic signed [17:0] average_sum;

    always_comb begin
        config_valid = i_width != 0 && i_width <= 16'(MAX_WIDTH) &&
            i_height != 0 && i_channels != 0 &&
            i_pool_mode <= POOL_AAVG2 &&
            (i_pool_mode == POOL_IDENTITY ||
             (!i_width[0] && !i_height[0]));
        o_start_ready = !o_busy;
        produces_output = pool_mode_q == POOL_IDENTITY ||
            (row_q[0] && col_q[0]);
        o_ready = o_busy && (!produces_output || i_write_ready);
        o_write_valid = o_busy && i_valid && produces_output;
        o_write_slot = output_slot_q;

        output_width = pool_mode_q == POOL_IDENTITY ? width_q :
                       (width_q >> 1);
        output_height = pool_mode_q == POOL_IDENTITY ? height_q :
                        (height_q >> 1);
        output_row = pool_mode_q == POOL_IDENTITY ? row_q : (row_q >> 1);
        output_col = pool_mode_q == POOL_IDENTITY ? col_q : (col_q >> 1);
        col_index = col_q[COL_INDEX_WIDTH-1:0];
        prior_col_index = col_index == 0 ? '0 : col_index - 1'b1;
        bank_rows = 16'((integer'(output_height) + 2) / 3);
        bank_cols = 16'((integer'(output_width) + 2) / 3);
        o_write_bank = 4'((integer'(output_row) % 3) * 3 +
                          integer'(output_col) % 3);
        o_write_addr = BANK_ADDR_WIDTH'(
            integer'(channel_q) * integer'(bank_rows) *
            integer'(bank_cols) + (integer'(output_row) / 3) *
            integer'(bank_cols) + integer'(output_col) / 3);

        max_value = i_value_q8_8;
        if (line_buffer_q[prior_col_index] > max_value)
            max_value = line_buffer_q[prior_col_index];
        if (line_buffer_q[col_index] > max_value)
            max_value = line_buffer_q[col_index];
        if (bottom_left_q > max_value)
            max_value = bottom_left_q;
        average_sum = {{2{line_buffer_q[prior_col_index][15]}},
                       line_buffer_q[prior_col_index]} +
                      {{2{line_buffer_q[col_index][15]}}, line_buffer_q[col_index]} +
                      {{2{bottom_left_q[15]}}, bottom_left_q} +
                      {{2{i_value_q8_8[15]}}, i_value_q8_8};
        if (pool_mode_q == POOL_IDENTITY)
            o_write_data_q8_8 = i_value_q8_8;
        else if (pool_mode_q == POOL_MAX2)
            o_write_data_q8_8 = max_value;
        else
            o_write_data_q8_8 = 16'(average_sum >>> 2);

        final_input = channel_q + 1 >= channels_q &&
            row_q + 1 >= height_q && col_q + 1 >= width_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            width_q <= '0;
            height_q <= '0;
            channels_q <= '0;
            pool_mode_q <= POOL_IDENTITY;
            output_slot_q <= 1'b0;
            expected_addr_q <= '0;
            channel_q <= '0;
            row_q <= '0;
            col_q <= '0;
            bottom_left_q <= '0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_config_error <= 1'b0;
            o_protocol_error <= 1'b0;
            for (int index = 0; index < MAX_WIDTH; index = index + 1)
                line_buffer_q[index] <= '0;
        end else begin
            o_done <= 1'b0;
            if (i_start && o_start_ready) begin
                width_q <= i_width;
                height_q <= i_height;
                channels_q <= i_channels;
                pool_mode_q <= i_pool_mode;
                output_slot_q <= i_output_slot;
                expected_addr_q <= i_input_base;
                channel_q <= '0;
                row_q <= '0;
                col_q <= '0;
                bottom_left_q <= '0;
                o_busy <= config_valid;
                o_config_error <= !config_valid;
                o_protocol_error <= 1'b0;
            end else if (i_start && !o_start_ready) begin
                o_protocol_error <= 1'b1;
            end

            if (i_valid && o_ready) begin
                if (i_addr != expected_addr_q)
                    o_protocol_error <= 1'b1;
                expected_addr_q <= expected_addr_q + 1'b1;
                if (pool_mode_q != POOL_IDENTITY) begin
                    if (!row_q[0])
                        line_buffer_q[col_index] <= i_value_q8_8;
                    else if (!col_q[0])
                        bottom_left_q <= i_value_q8_8;
                end

                if (final_input) begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end else if (col_q + 1 < width_q) begin
                    col_q <= col_q + 1'b1;
                end else begin
                    col_q <= '0;
                    if (row_q + 1 < height_q) begin
                        row_q <= row_q + 1'b1;
                    end else begin
                        row_q <= '0;
                        channel_q <= channel_q + 1'b1;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
