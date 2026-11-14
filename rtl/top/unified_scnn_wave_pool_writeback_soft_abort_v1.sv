`default_nettype none

// Pools the spatial-major, engine-minor stream produced by the SCNN channel
// wave. One short line buffer per physical engine avoids a full feature-map
// reorder between shared LIF and pooling.
module unified_scnn_wave_pool_writeback_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int MAX_WIDTH = 32,
    parameter int SPATIAL_ADDR_WIDTH = 15,
    parameter int BANK_ADDR_WIDTH = 14
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic [15:0]                               i_width,
    input  logic [15:0]                               i_height,
    input  logic [15:0]                               i_output_channels,
    input  logic [15:0]                               i_channel_base,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [1:0]                                i_pool_mode,
    input  logic                                      i_output_slot,

    input  logic                                      i_valid,
    output logic                                      o_ready,
    input  logic [$clog2(ENGINES)-1:0]                i_engine_index,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]             i_spatial_addr,
    input  logic signed [15:0]                        i_value_q8_8,

    output logic                                      o_write_valid,
    input  logic                                      i_write_ready,
    output logic                                      o_write_slot,
    output logic [3:0]                                o_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]                o_write_addr,
    output logic signed [15:0]                        o_write_data_q8_8,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_config_error,
    output logic                                      o_protocol_error
);
    localparam int ENGINE_INDEX_WIDTH = $clog2(ENGINES);
    localparam int COL_INDEX_WIDTH = $clog2(MAX_WIDTH);
    localparam logic [1:0] POOL_IDENTITY = 2'd0;
    localparam logic [1:0] POOL_MAX2 = 2'd1;
    localparam logic [1:0] POOL_AAVG2 = 2'd2;

    logic [15:0] width_q;
    logic [15:0] height_q;
    logic [15:0] channel_base_q;
    logic [ENGINES-1:0] engine_mask_q;
    logic [1:0] pool_mode_q;
    logic output_slot_q;
    logic [ENGINE_INDEX_WIDTH-1:0] expected_engine_q;
    logic [SPATIAL_ADDR_WIDTH-1:0] expected_spatial_q;
    logic [15:0] row_q;
    logic [15:0] col_q;
    logic signed [15:0] line_buffer_q [0:ENGINES-1][0:MAX_WIDTH-1];
    logic signed [15:0] bottom_left_q [0:ENGINES-1];

    logic config_valid;
    logic request_first_found;
    logic [ENGINE_INDEX_WIDTH-1:0] request_first_engine;
    logic active_first_found;
    logic [ENGINE_INDEX_WIDTH-1:0] active_first_engine;
    logic next_found;
    logic [ENGINE_INDEX_WIDTH-1:0] next_engine;
    logic input_matches;
    logic produces_output;
    logic input_fire;
    logic final_spatial;
    logic final_input;
    logic [15:0] output_width;
    logic [15:0] output_height;
    logic [15:0] output_row;
    logic [15:0] output_col;
    logic [15:0] global_channel;
    logic [15:0] bank_rows;
    logic [15:0] bank_cols;
    logic [COL_INDEX_WIDTH-1:0] col_index;
    logic [COL_INDEX_WIDTH-1:0] prior_col_index;
    logic signed [15:0] max_value;
    logic signed [17:0] average_sum;
    logic write_valid_q;
    logic write_last_q;
    logic write_fire;
    logic write_capture;
    logic write_slot_q;
    logic [3:0] write_bank_q;
    logic [BANK_ADDR_WIDTH-1:0] write_addr_q;
    logic signed [15:0] write_data_q;
    logic [3:0] next_write_bank;
    logic [BANK_ADDR_WIDTH-1:0] next_write_addr;
    logic signed [15:0] next_write_data;

    assign o_write_valid = write_valid_q;
    assign o_write_slot = write_slot_q;
    assign o_write_bank = write_bank_q;
    assign o_write_addr = write_addr_q;
    assign o_write_data_q8_8 = write_data_q;

    always_comb begin
        config_valid = i_width != 0 && i_width <= 16'(MAX_WIDTH) &&
            i_height != 0 && i_output_channels != 0 &&
            i_engine_mask != '0 && i_pool_mode <= POOL_AAVG2 &&
            (i_pool_mode == POOL_IDENTITY ||
             (!i_width[0] && !i_height[0]));
        request_first_found = 1'b0;
        request_first_engine = '0;
        active_first_found = 1'b0;
        active_first_engine = '0;
        next_found = 1'b0;
        next_engine = '0;
        for (integer engine = 0; engine < ENGINES; engine++) begin
            if (i_engine_mask[engine] &&
                integer'(i_channel_base) + engine >=
                integer'(i_output_channels))
                config_valid = 1'b0;
            if (i_engine_mask[engine] && !request_first_found) begin
                request_first_found = 1'b1;
                request_first_engine = ENGINE_INDEX_WIDTH'(engine);
            end
            if (engine_mask_q[engine] && !active_first_found) begin
                active_first_found = 1'b1;
                active_first_engine = ENGINE_INDEX_WIDTH'(engine);
            end
            if (engine > integer'(expected_engine_q) &&
                engine_mask_q[engine] && !next_found) begin
                next_found = 1'b1;
                next_engine = ENGINE_INDEX_WIDTH'(engine);
            end
        end

        o_start_ready = !o_busy && !o_config_error && !o_protocol_error;
        input_matches = i_engine_index == expected_engine_q &&
                        i_spatial_addr == expected_spatial_q;
        produces_output = pool_mode_q == POOL_IDENTITY ||
                          (row_q[0] && col_q[0]);
        write_fire = write_valid_q && i_write_ready;
        o_ready = o_busy &&
                  (!produces_output || !write_valid_q || i_write_ready);
        input_fire = i_valid && o_ready;
        write_capture = input_fire && input_matches && produces_output;

        output_width = pool_mode_q == POOL_IDENTITY ? width_q : width_q >> 1;
        output_height = pool_mode_q == POOL_IDENTITY ? height_q : height_q >> 1;
        output_row = pool_mode_q == POOL_IDENTITY ? row_q : row_q >> 1;
        output_col = pool_mode_q == POOL_IDENTITY ? col_q : col_q >> 1;
        global_channel = channel_base_q + 16'(i_engine_index);
        bank_rows = 16'((integer'(output_height) + 2) / 3);
        bank_cols = 16'((integer'(output_width) + 2) / 3);
        next_write_bank = 4'((integer'(output_row) % 3) * 3 +
                             integer'(output_col) % 3);
        next_write_addr = BANK_ADDR_WIDTH'(
            integer'(global_channel) * integer'(bank_rows) *
            integer'(bank_cols) + (integer'(output_row) / 3) *
            integer'(bank_cols) + integer'(output_col) / 3);

        col_index = col_q[COL_INDEX_WIDTH-1:0];
        prior_col_index = col_index == 0 ? '0 : col_index - 1'b1;
        max_value = i_value_q8_8;
        if (line_buffer_q[i_engine_index][prior_col_index] > max_value)
            max_value = line_buffer_q[i_engine_index][prior_col_index];
        if (line_buffer_q[i_engine_index][col_index] > max_value)
            max_value = line_buffer_q[i_engine_index][col_index];
        if (bottom_left_q[i_engine_index] > max_value)
            max_value = bottom_left_q[i_engine_index];
        average_sum =
            {{2{line_buffer_q[i_engine_index][prior_col_index][15]}},
              line_buffer_q[i_engine_index][prior_col_index]} +
            {{2{line_buffer_q[i_engine_index][col_index][15]}},
              line_buffer_q[i_engine_index][col_index]} +
            {{2{bottom_left_q[i_engine_index][15]}},
              bottom_left_q[i_engine_index]} +
            {{2{i_value_q8_8[15]}}, i_value_q8_8};
        if (pool_mode_q == POOL_IDENTITY)
            next_write_data = i_value_q8_8;
        else if (pool_mode_q == POOL_MAX2)
            next_write_data = max_value;
        else
            next_write_data = 16'(average_sum >>> 2);

        final_spatial = row_q + 1 >= height_q && col_q + 1 >= width_q;
        final_input = input_matches && !next_found && final_spatial;
    end

    // Payload state is intentionally reset-free. write_valid_q is the sole
    // authority, so reset/abort can discard an in-flight write without adding
    // reset control to the address and data registers on the timing boundary.
    always_ff @(posedge i_clk) begin
        if (write_capture) begin
            write_last_q <= final_input;
            write_slot_q <= output_slot_q;
            write_bank_q <= next_write_bank;
            write_addr_q <= next_write_addr;
            write_data_q <= next_write_data;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            width_q <= '0;
            height_q <= '0;
            channel_base_q <= '0;
            engine_mask_q <= '0;
            pool_mode_q <= POOL_IDENTITY;
            output_slot_q <= 1'b0;
            expected_engine_q <= '0;
            expected_spatial_q <= '0;
            row_q <= '0;
            col_q <= '0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_config_error <= 1'b0;
            o_protocol_error <= 1'b0;
            write_valid_q <= 1'b0;
            for (integer engine = 0; engine < ENGINES; engine++) begin
                bottom_left_q[engine] <= '0;
                for (integer col = 0; col < MAX_WIDTH; col++)
                    line_buffer_q[engine][col] <= '0;
            end
        end else if (i_abort) begin
            width_q <= '0;
            height_q <= '0;
            channel_base_q <= '0;
            engine_mask_q <= '0;
            pool_mode_q <= POOL_IDENTITY;
            output_slot_q <= 1'b0;
            expected_engine_q <= '0;
            expected_spatial_q <= '0;
            row_q <= '0;
            col_q <= '0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            write_valid_q <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (write_fire)
                write_valid_q <= 1'b0;
            if (i_clear_error && !o_busy) begin
                o_config_error <= 1'b0;
                o_protocol_error <= 1'b0;
            end

            if (i_start && o_start_ready) begin
                width_q <= i_width;
                height_q <= i_height;
                channel_base_q <= i_channel_base;
                engine_mask_q <= i_engine_mask;
                pool_mode_q <= i_pool_mode;
                output_slot_q <= i_output_slot;
                expected_engine_q <= request_first_engine;
                expected_spatial_q <= '0;
                row_q <= '0;
                col_q <= '0;
                o_busy <= config_valid;
                o_config_error <= !config_valid;
                o_protocol_error <= 1'b0;
                write_valid_q <= 1'b0;
            end else if (i_start) begin
                o_protocol_error <= 1'b1;
            end

            if (input_fire) begin
                if (!input_matches) begin
                    o_protocol_error <= 1'b1;
                    o_busy <= 1'b0;
                end else begin
                    if (produces_output)
                        write_valid_q <= 1'b1;
                    if (pool_mode_q != POOL_IDENTITY) begin
                        if (!row_q[0])
                            line_buffer_q[i_engine_index][col_index] <=
                                i_value_q8_8;
                        else if (!col_q[0])
                            bottom_left_q[i_engine_index] <= i_value_q8_8;
                    end

                    if (final_input) begin
                        if (!produces_output) begin
                            o_busy <= 1'b0;
                            o_done <= 1'b1;
                        end
                    end else if (next_found) begin
                        expected_engine_q <= next_engine;
                    end else begin
                        expected_engine_q <= active_first_engine;
                        expected_spatial_q <= expected_spatial_q + 1'b1;
                        if (col_q + 1 < width_q) begin
                            col_q <= col_q + 1'b1;
                        end else begin
                            col_q <= '0;
                            row_q <= row_q + 1'b1;
                        end
                    end
                end
            end

            if (write_fire && write_last_q) begin
                o_busy <= 1'b0;
                o_done <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
