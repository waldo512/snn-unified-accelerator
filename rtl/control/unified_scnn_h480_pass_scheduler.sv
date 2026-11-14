`default_nettype none

module unified_scnn_h480_pass_scheduler (
    input  logic         i_clk,
    input  logic         i_rst_n,
    input  logic         i_sequence_start,
    output logic         o_sequence_start_ready,
    input  logic         i_clear_error,
    input  logic         i_abort,

    input  logic         i_tile_valid,
    output logic         o_tile_ready,
    input  logic [3:0]   i_layer_index,
    input  logic [9:0]   i_output_channel_base,
    input  logic [9:0]   i_output_channel_count,
    input  logic [9:0]   i_input_channel_base,
    input  logic [9:0]   i_input_channel_count,
    input  logic [15:0]  i_tile_tag,

    output logic         o_compute_start,
    input  logic         i_compute_done,
    output logic [3:0]   o_layer_index,
    output logic [9:0]   o_output_channel_base,
    output logic [9:0]   o_output_channel_count,
    output logic [9:0]   o_input_channel_base,
    output logic [9:0]   o_input_channel_count,
    output logic [15:0]  o_tile_tag,
    output logic [3:0]   o_active_h40_tiles,
    output logic         o_input_slot,
    output logic         o_output_slot,
    output logic         o_accumulator_clear,
    output logic         o_accumulator_finalize,
    output logic         o_lif_pool_commit,
    output logic         o_output_pass_first,
    output logic         o_output_pass_last,
    output logic         o_layer_first_tile,
    output logic         o_layer_last_tile,
    output logic         o_classifier_tile,

    output logic         o_tile_busy,
    output logic         o_sequence_busy,
    output logic         o_tile_done,
    output logic         o_sequence_done,
    output logic         o_sticky_error
);
    logic [3:0] expected_layer_q;
    logic [9:0] expected_output_base_q;
    logic [9:0] expected_input_base_q;
    logic done_armed_q;
    logic tile_valid_armed_q;
    logic descriptor_valid;
    logic tile_accept;
    logic [9:0] expected_layer_inputs;
    logic [9:0] expected_layer_outputs;
    logic [9:0] expected_output_count;
    logic [9:0] expected_input_count;
    logic [9:0] active_tiles_wide;
    logic final_input_group;
    logic final_output_pass;

    function automatic logic [9:0] layer_inputs(input logic [3:0] layer);
        case (layer)
            4'd0: layer_inputs = 10'd2;
            4'd1: layer_inputs = 10'd64;
            4'd2: layer_inputs = 10'd128;
            4'd3, 4'd4: layer_inputs = 10'd256;
            default: layer_inputs = 10'd512;
        endcase
    endfunction

    function automatic logic [9:0] layer_outputs(input logic [3:0] layer);
        case (layer)
            4'd0: layer_outputs = 10'd64;
            4'd1: layer_outputs = 10'd128;
            4'd2, 4'd3: layer_outputs = 10'd256;
            4'd4, 4'd5, 4'd6, 4'd7: layer_outputs = 10'd512;
            default: layer_outputs = 10'd11;
        endcase
    endfunction

    always_comb begin
        expected_layer_inputs = layer_inputs(expected_layer_q);
        expected_layer_outputs = layer_outputs(expected_layer_q);
        expected_output_count = expected_layer_outputs - expected_output_base_q;
        if (expected_output_count > 10'd480)
            expected_output_count = 10'd480;
        expected_input_count = expected_layer_inputs - expected_input_base_q;
        if (expected_input_count > 10'd8)
            expected_input_count = 10'd8;
        if (expected_layer_q == 4'd0) begin
            expected_output_count = 10'd64;
            expected_input_count = 10'd2;
        end else if (expected_layer_q == 4'd8) begin
            expected_output_count = 10'd11;
            expected_input_count = 10'd512;
        end

        descriptor_valid = i_layer_index == expected_layer_q &&
            i_output_channel_base == expected_output_base_q &&
            i_output_channel_count == expected_output_count &&
            i_input_channel_base == expected_input_base_q &&
            i_input_channel_count == expected_input_count;
        o_sequence_start_ready = !o_sequence_busy && !o_tile_busy;
        o_tile_ready = o_sequence_busy && !o_tile_busy && tile_valid_armed_q;
        tile_accept = i_tile_valid && o_tile_ready;

        final_input_group = o_input_channel_base + o_input_channel_count >=
                            layer_inputs(o_layer_index);
        final_output_pass = o_output_channel_base + o_output_channel_count >=
                            layer_outputs(o_layer_index);
        active_tiles_wide = (o_output_channel_count + 10'd39) / 10'd40;
        o_active_h40_tiles = active_tiles_wide[3:0];
        o_input_slot = o_layer_index[0];
        o_output_slot = !o_layer_index[0];
        o_accumulator_clear = o_input_channel_base == 0;
        o_accumulator_finalize = final_input_group;
        o_classifier_tile = o_layer_index == 4'd8;
        o_lif_pool_commit = final_input_group && !o_classifier_tile;
        o_output_pass_first = o_output_channel_base == 0;
        o_output_pass_last = final_output_pass;
        o_layer_first_tile = o_output_channel_base == 0 &&
                             o_input_channel_base == 0;
        o_layer_last_tile = final_input_group && final_output_pass;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            expected_layer_q <= '0;
            expected_output_base_q <= '0;
            expected_input_base_q <= '0;
            done_armed_q <= 1'b0;
            tile_valid_armed_q <= 1'b1;
            o_layer_index <= '0;
            o_output_channel_base <= '0;
            o_output_channel_count <= '0;
            o_input_channel_base <= '0;
            o_input_channel_count <= '0;
            o_tile_tag <= '0;
            o_compute_start <= 1'b0;
            o_tile_busy <= 1'b0;
            o_sequence_busy <= 1'b0;
            o_tile_done <= 1'b0;
            o_sequence_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_compute_start <= 1'b0;
            o_tile_done <= 1'b0;
            o_sequence_done <= 1'b0;
            if (i_clear_error)
                o_sticky_error <= 1'b0;
            if (!i_tile_valid)
                tile_valid_armed_q <= 1'b1;

            if (i_abort && (o_sequence_busy || o_tile_busy)) begin
                o_tile_busy <= 1'b0;
                o_sequence_busy <= 1'b0;
                done_armed_q <= 1'b0;
                tile_valid_armed_q <= 1'b0;
                o_sticky_error <= 1'b1;
            end else begin
                if (i_sequence_start && o_sequence_start_ready) begin
                    expected_layer_q <= '0;
                    expected_output_base_q <= '0;
                    expected_input_base_q <= '0;
                    o_sequence_busy <= 1'b1;
                    tile_valid_armed_q <= !i_tile_valid;
                end else if (i_sequence_start && !o_sequence_start_ready) begin
                    o_sticky_error <= 1'b1;
                end

                if (tile_accept) begin
                    tile_valid_armed_q <= 1'b0;
                    if (!descriptor_valid) begin
                        o_sequence_busy <= 1'b0;
                        o_sticky_error <= 1'b1;
                    end else begin
                        o_layer_index <= i_layer_index;
                        o_output_channel_base <= i_output_channel_base;
                        o_output_channel_count <= i_output_channel_count;
                        o_input_channel_base <= i_input_channel_base;
                        o_input_channel_count <= i_input_channel_count;
                        o_tile_tag <= i_tile_tag;
                        o_compute_start <= 1'b1;
                        o_tile_busy <= 1'b1;
                        done_armed_q <= !i_compute_done;
                    end
                end

                if (o_tile_busy && !i_compute_done)
                    done_armed_q <= 1'b1;
                if (o_tile_busy && done_armed_q && i_compute_done) begin
                    o_tile_busy <= 1'b0;
                    done_armed_q <= 1'b0;
                    o_tile_done <= 1'b1;
                    if (o_layer_index == 4'd8) begin
                        o_sequence_busy <= 1'b0;
                        o_sequence_done <= 1'b1;
                    end else if (o_input_channel_base + o_input_channel_count <
                                 layer_inputs(o_layer_index)) begin
                        expected_input_base_q <= o_input_channel_base +
                                                 o_input_channel_count;
                    end else if (o_output_channel_base + o_output_channel_count <
                                 layer_outputs(o_layer_index)) begin
                        expected_input_base_q <= '0;
                        expected_output_base_q <= o_output_channel_base +
                                                  o_output_channel_count;
                    end else begin
                        expected_layer_q <= o_layer_index + 1'b1;
                        expected_input_base_q <= '0;
                        expected_output_base_q <= '0;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
