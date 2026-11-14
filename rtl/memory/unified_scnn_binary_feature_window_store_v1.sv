`timescale 1ns/1ps
`default_nettype none

// Two-slot bit-packed post-LIF feature store. One 64-bit word contains
// consecutive spatial positions from one channel row. A request gathers up
// to four 3x3/stride-1 terms into four 40-position masks serially through one
// BRAM read port; this avoids four feature-map replicas while the P4 array is
// processing the preceding command.
module unified_scnn_binary_feature_window_store_v1 #(
    parameter int WORD_DEPTH = 8192,
    parameter int ADDR_WIDTH = $clog2(WORD_DEPTH)
) (
    input  wire logic                       i_clk,
    input  wire logic                       i_rst_n,
    input  wire logic                       i_abort,
    input  wire logic                       i_clear_error,

    input  wire logic                       i_fill_valid,
    output logic                            o_fill_ready,
    input  wire logic                       i_fill_slot,
    input  wire logic [ADDR_WIDTH-1:0]      i_fill_addr,
    input  wire logic [63:0]                i_fill_data,

    input  wire logic                       i_update_valid,
    output logic                            o_update_ready,
    input  wire logic                       i_update_slot,
    input  wire logic [ADDR_WIDTH-1:0]      i_update_addr,
    input  wire logic [63:0]                i_update_data,
    input  wire logic [63:0]                i_update_mask,
    output logic                            o_update_done,

    input  wire logic                       i_request_valid,
    output logic                            o_request_ready,
    input  wire logic                       i_read_slot,
    input  wire logic [ADDR_WIDTH-1:0]      i_feature_base_addr,
    input  wire logic [15:0]                i_row_stride_words,
    input  wire logic [ADDR_WIDTH-1:0]      i_channel_stride_words,
    input  wire logic [15:0]                i_in_width,
    input  wire logic [15:0]                i_in_height,
    input  wire logic [15:0]                i_in_channels,
    input  wire logic [7:0]                 i_stride,
    input  wire logic [7:0]                 i_padding,
    input  wire logic [3:0]                 i_term_valid,
    input  wire logic [39:0]                i_input_channel_flat,
    input  wire logic [7:0]                 i_kernel_row_flat,
    input  wire logic [7:0]                 i_kernel_col_flat,
    input  wire logic [15:0]                i_output_row,
    input  wire logic [15:0]                i_output_col,
    input  wire logic [5:0]                 i_spatial_count,

    output logic                            o_response_valid,
    input  wire logic                       i_response_ready,
    output logic [39:0]                     o_feature_mask0,
    output logic [39:0]                     o_feature_mask1,
    output logic [39:0]                     o_feature_mask2,
    output logic [39:0]                     o_feature_mask3,

    output logic                            o_busy,
    output logic                            o_sticky_error
);
    typedef enum logic [3:0] {
        IDLE,
        PREPARE_TERM,
        READ_WORD0,
        CAPTURE_WORD0,
        READ_WORD1,
        CAPTURE_WORD1,
        STORE_TERM,
        RESPONSE,
        UPDATE_READ,
        UPDATE_CAPTURE,
        UPDATE_WRITE
    } state_t;

    state_t state_q;
    logic read_slot_q;
    logic [ADDR_WIDTH-1:0] feature_base_addr_q;
    logic [15:0] row_stride_words_q;
    logic [ADDR_WIDTH-1:0] channel_stride_words_q;
    logic [15:0] in_width_q;
    logic [15:0] in_height_q;
    logic [7:0] padding_q;
    logic [3:0] term_valid_q;
    logic [9:0] input_channel_q [0:3];
    logic [1:0] kernel_row_q [0:3];
    logic [1:0] kernel_col_q [0:3];
    logic [15:0] output_row_q;
    logic [15:0] output_col_q;
    logic [5:0] spatial_count_q;
    logic [1:0] term_q;
    logic [39:0] response_mask_q [0:3];

    logic [ADDR_WIDTH-1:0] read_addr0_q;
    logic [ADDR_WIDTH-1:0] read_addr1_q;
    logic [15:0] word_index0_q;
    logic [63:0] first_word_q;
    logic [63:0] second_word_q;
    logic need_second_q;
    logic update_slot_q;
    logic [ADDR_WIDTH-1:0] update_addr_q;
    logic [63:0] update_data_q;
    logic [63:0] update_mask_q;
    logic [63:0] update_word_q;

    (* ram_style = "block" *) logic [63:0]
        memory [0:1][0:WORD_DEPTH-1];
    logic [63:0] memory_read_data [0:1];
    logic memory_read_enable;
    logic [ADDR_WIDTH-1:0] memory_read_addr;
    logic fill_fire;
    logic update_fire;
    logic request_fire;
    logic response_fire;
    logic request_legal;
    logic current_term_has_data;
    logic current_term_legal;
    logic current_term_need_second;
    logic [ADDR_WIDTH-1:0] current_addr0;
    logic [ADDR_WIDTH-1:0] current_addr1;
    logic [15:0] current_word_index0;
    logic [39:0] assembled_mask;

    integer unsigned required_row_words;
    integer unsigned required_channel_words;
    integer unsigned request_last_word;
    integer signed current_input_row;
    integer signed current_start_col;
    integer signed current_col;
    integer unsigned current_first_word;
    integer unsigned current_last_word;
    integer unsigned current_address0_integer;
    integer unsigned current_address1_integer;

    always_comb begin
        required_row_words = (integer'(i_in_width) + 63) / 64;
        required_channel_words = integer'(i_row_stride_words) *
            integer'(i_in_height);
        request_last_word = integer'(i_feature_base_addr) +
            integer'(i_channel_stride_words) * integer'(i_in_channels);
        request_legal = i_in_width != 0 && i_in_height != 0 &&
            i_in_channels != 0 && i_stride == 8'd1 &&
            i_spatial_count != 0 && i_spatial_count <= 6'd40 &&
            integer'(i_row_stride_words) >= required_row_words &&
            integer'(i_channel_stride_words) >= required_channel_words &&
            request_last_word <= WORD_DEPTH;
        for (integer term = 0; term < 4; term++) begin
            if (i_term_valid[term] &&
                (16'(i_input_channel_flat[term*10 +: 10]) >=
                     i_in_channels ||
                 i_kernel_row_flat[term*2 +: 2] >= 3 ||
                 i_kernel_col_flat[term*2 +: 2] >= 3))
                request_legal = 1'b0;
        end

        current_input_row = integer'(output_row_q) +
            integer'(kernel_row_q[term_q]) - integer'(padding_q);
        current_start_col = integer'(output_col_q) +
            integer'(kernel_col_q[term_q]) - integer'(padding_q);
        current_first_word = 0;
        current_last_word = 0;
        current_term_has_data = 1'b0;
        for (integer lane = 0; lane < 40; lane++) begin
            current_col = current_start_col + lane;
            if (term_valid_q[term_q] &&
                lane < integer'(spatial_count_q) &&
                current_input_row >= 0 &&
                current_input_row < integer'(in_height_q) &&
                current_col >= 0 && current_col < integer'(in_width_q)) begin
                if (!current_term_has_data)
                    current_first_word = current_col / 64;
                current_last_word = current_col / 64;
                current_term_has_data = 1'b1;
            end
        end
        current_address0_integer = 0;
        current_address1_integer = 0;
        if (current_term_has_data) begin
            current_address0_integer = integer'(feature_base_addr_q) +
                integer'(input_channel_q[term_q]) *
                    integer'(channel_stride_words_q) +
                current_input_row * integer'(row_stride_words_q) +
                current_first_word;
            current_address1_integer = current_address0_integer + 1;
        end
        current_term_need_second = current_term_has_data &&
            current_last_word != current_first_word;
        current_term_legal = !current_term_has_data ||
            (current_address0_integer < WORD_DEPTH &&
             (!current_term_need_second ||
              current_address1_integer < WORD_DEPTH));
        current_addr0 = ADDR_WIDTH'(current_address0_integer);
        current_addr1 = ADDR_WIDTH'(current_address1_integer);
        current_word_index0 = 16'(current_first_word);

        assembled_mask = '0;
        for (integer lane = 0; lane < 40; lane++) begin
            current_col = integer'(output_col_q) + lane +
                integer'(kernel_col_q[term_q]) - integer'(padding_q);
            if (term_valid_q[term_q] &&
                lane < integer'(spatial_count_q) &&
                current_input_row >= 0 &&
                current_input_row < integer'(in_height_q) &&
                current_col >= 0 && current_col < integer'(in_width_q)) begin
                if ((current_col / 64) == integer'(word_index0_q))
                    assembled_mask[lane] = first_word_q[current_col % 64];
                else
                    assembled_mask[lane] = second_word_q[current_col % 64];
            end
        end

        o_update_ready = state_q == IDLE && !o_sticky_error;
        o_fill_ready = state_q == IDLE && !o_sticky_error &&
            !i_update_valid && !i_request_valid &&
            integer'(i_fill_addr) < WORD_DEPTH;
        o_request_ready = state_q == IDLE && !o_sticky_error &&
            !i_update_valid;
        o_response_valid = state_q == RESPONSE;
        o_feature_mask0 = response_mask_q[0];
        o_feature_mask1 = response_mask_q[1];
        o_feature_mask2 = response_mask_q[2];
        o_feature_mask3 = response_mask_q[3];
        o_busy = state_q != IDLE;
        fill_fire = i_fill_valid && o_fill_ready;
        update_fire = i_update_valid && o_update_ready;
        request_fire = i_request_valid && o_request_ready;
        response_fire = o_response_valid && i_response_ready;

        memory_read_enable = (state_q == READ_WORD0 ||
            state_q == READ_WORD1 || state_q == UPDATE_READ) && !i_abort;
        if (state_q == UPDATE_READ)
            memory_read_addr = update_addr_q;
        else
            memory_read_addr = state_q == READ_WORD1 ?
                read_addr1_q : read_addr0_q;
    end

    always_ff @(posedge i_clk) begin
        for (integer slot = 0; slot < 2; slot++) begin
            if (fill_fire && i_fill_slot == slot[0])
                memory[slot][i_fill_addr] <= i_fill_data;
            if (state_q == UPDATE_WRITE && !i_abort &&
                update_slot_q == slot[0])
                memory[slot][update_addr_q] <=
                    (update_word_q & ~update_mask_q) |
                    (update_data_q & update_mask_q);
            if (memory_read_enable &&
                ((state_q == UPDATE_READ && update_slot_q == slot[0]) ||
                 (state_q != UPDATE_READ && read_slot_q == slot[0])))
                memory_read_data[slot] <= memory[slot][memory_read_addr];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            read_slot_q <= 1'b0;
            feature_base_addr_q <= '0;
            row_stride_words_q <= '0;
            channel_stride_words_q <= '0;
            in_width_q <= '0;
            in_height_q <= '0;
            padding_q <= '0;
            term_valid_q <= '0;
            for (integer term = 0; term < 4; term++) begin
                input_channel_q[term] <= '0;
                kernel_row_q[term] <= '0;
                kernel_col_q[term] <= '0;
                response_mask_q[term] <= '0;
            end
            output_row_q <= '0;
            output_col_q <= '0;
            spatial_count_q <= '0;
            term_q <= '0;
            read_addr0_q <= '0;
            read_addr1_q <= '0;
            word_index0_q <= '0;
            first_word_q <= '0;
            second_word_q <= '0;
            need_second_q <= 1'b0;
            update_slot_q <= 1'b0;
            update_addr_q <= '0;
            update_data_q <= '0;
            update_mask_q <= '0;
            update_word_q <= '0;
            o_update_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_update_done <= 1'b0;
            if (i_abort) begin
                state_q <= IDLE;
            end else begin
                if (i_clear_error && state_q == IDLE)
                    o_sticky_error <= 1'b0;
                if (i_fill_valid && !o_fill_ready)
                    o_sticky_error <= 1'b1;

                case (state_q)
                    IDLE: begin
                        if (update_fire) begin
                            if (integer'(i_update_addr) >= WORD_DEPTH ||
                                i_update_mask == 0) begin
                                o_sticky_error <= 1'b1;
                            end else begin
                                update_slot_q <= i_update_slot;
                                update_addr_q <= i_update_addr;
                                update_data_q <= i_update_data;
                                update_mask_q <= i_update_mask;
                                state_q <= UPDATE_READ;
                            end
                        end else if (request_fire) begin
                            if (!request_legal) begin
                                o_sticky_error <= 1'b1;
                            end else begin
                                read_slot_q <= i_read_slot;
                                feature_base_addr_q <= i_feature_base_addr;
                                row_stride_words_q <= i_row_stride_words;
                                channel_stride_words_q <=
                                    i_channel_stride_words;
                                in_width_q <= i_in_width;
                                in_height_q <= i_in_height;
                                padding_q <= i_padding;
                                term_valid_q <= i_term_valid;
                                for (integer term = 0; term < 4; term++) begin
                                    input_channel_q[term] <=
                                        i_input_channel_flat[
                                            term*10 +: 10];
                                    kernel_row_q[term] <=
                                        i_kernel_row_flat[term*2 +: 2];
                                    kernel_col_q[term] <=
                                        i_kernel_col_flat[term*2 +: 2];
                                    response_mask_q[term] <= '0;
                                end
                                output_row_q <= i_output_row;
                                output_col_q <= i_output_col;
                                spatial_count_q <= i_spatial_count;
                                term_q <= '0;
                                state_q <= PREPARE_TERM;
                            end
                        end
                    end
                    PREPARE_TERM: begin
                        if (!current_term_legal) begin
                            state_q <= IDLE;
                            o_sticky_error <= 1'b1;
                        end else if (!current_term_has_data) begin
                            response_mask_q[term_q] <= '0;
                            if (term_q == 2'd3)
                                state_q <= RESPONSE;
                            else
                                term_q <= term_q + 1'b1;
                        end else begin
                            read_addr0_q <= current_addr0;
                            read_addr1_q <= current_addr1;
                            word_index0_q <= current_word_index0;
                            need_second_q <= current_term_need_second;
                            state_q <= READ_WORD0;
                        end
                    end
                    READ_WORD0: state_q <= CAPTURE_WORD0;
                    CAPTURE_WORD0: begin
                        first_word_q <= memory_read_data[read_slot_q];
                        if (need_second_q)
                            state_q <= READ_WORD1;
                        else begin
                            second_word_q <= '0;
                            state_q <= STORE_TERM;
                        end
                    end
                    READ_WORD1: state_q <= CAPTURE_WORD1;
                    CAPTURE_WORD1: begin
                        second_word_q <= memory_read_data[read_slot_q];
                        state_q <= STORE_TERM;
                    end
                    STORE_TERM: begin
                        response_mask_q[term_q] <= assembled_mask;
                        if (term_q == 2'd3)
                            state_q <= RESPONSE;
                        else begin
                            term_q <= term_q + 1'b1;
                            state_q <= PREPARE_TERM;
                        end
                    end
                    RESPONSE: begin
                        if (response_fire)
                            state_q <= IDLE;
                    end
                    UPDATE_READ: state_q <= UPDATE_CAPTURE;
                    UPDATE_CAPTURE: begin
                        update_word_q <= memory_read_data[update_slot_q];
                        state_q <= UPDATE_WRITE;
                    end
                    UPDATE_WRITE: begin
                        o_update_done <= 1'b1;
                        state_q <= IDLE;
                    end
                    default: begin
                        state_q <= IDLE;
                        o_sticky_error <= 1'b1;
                    end
                endcase
            end
        end
    end

    initial begin
        if (WORD_DEPTH < 2 || ADDR_WIDTH != $clog2(WORD_DEPTH))
            $error("binary feature window store parameter mismatch");
    end
endmodule

`default_nettype wire
