`default_nettype none

module unified_scnn_vgg9_wave_group_sequencer_v2 #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int BANK_WORDS = 8640,
    parameter int TAG_WIDTH = 16
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_start,
    output logic                            o_busy,
    output logic                            o_done,
    output logic                            o_tile_valid,
    input  logic                            i_tile_ready,
    output logic [3:0]                      o_layer_index,
    output logic [9:0]                      o_output_channel_base,
    output logic [9:0]                      o_output_channel_count,
    output logic [9:0]                      o_input_channel_base,
    output logic [9:0]                      o_input_channel_count,
    output logic [ADDRESS_WIDTH-1:0]        o_source_q64_word,
    output logic [$clog2(BANK_WORDS+1)-1:0] o_total_q64_words,
    output logic [$clog2(BANK_WORDS+1)-1:0] o_segment_q64_words,
    output logic [ADDRESS_WIDTH-1:0]        o_source_stride_q64_words,
    output logic [TAG_WIDTH-1:0]            o_tile_tag,
    output logic                            o_last_tile
);

    localparam int COUNT_WIDTH = $clog2(BANK_WORDS + 1);

    logic [3:0] layer_q;
    logic [9:0] output_base_q;
    logic [9:0] input_base_q;
    logic [5:0] output_wave_q;
    logic [5:0] input_group_q;
    logic [9:0] layer_input_channels;
    logic [9:0] layer_output_channels;
    logic [ADDRESS_WIDTH-1:0] layer_start_q64;
    logic [9:0] output_count;
    logic [9:0] input_count;
    logic [63:0] source_offset_q16;
    logic [COUNT_WIDTH-1:0] total_q64;
    logic [ADDRESS_WIDTH-1:0] stride_q64;
    logic tile_accept;

    always_comb begin
        layer_input_channels = 10'd0;
        layer_output_channels = 10'd0;
        layer_start_q64 = '0;
        case (layer_q)
            4'd0: begin layer_input_channels = 10'd2;   layer_output_channels = 10'd64;  layer_start_q64 = ADDRESS_WIDTH'(0);       end
            4'd1: begin layer_input_channels = 10'd64;  layer_output_channels = 10'd128; layer_start_q64 = ADDRESS_WIDTH'(288);     end
            4'd2: begin layer_input_channels = 10'd128; layer_output_channels = 10'd256; layer_start_q64 = ADDRESS_WIDTH'(18720);   end
            4'd3: begin layer_input_channels = 10'd256; layer_output_channels = 10'd256; layer_start_q64 = ADDRESS_WIDTH'(92448);   end
            4'd4: begin layer_input_channels = 10'd256; layer_output_channels = 10'd512; layer_start_q64 = ADDRESS_WIDTH'(239904);  end
            4'd5: begin layer_input_channels = 10'd512; layer_output_channels = 10'd512; layer_start_q64 = ADDRESS_WIDTH'(534816);  end
            4'd6: begin layer_input_channels = 10'd512; layer_output_channels = 10'd512; layer_start_q64 = ADDRESS_WIDTH'(1124640); end
            4'd7: begin layer_input_channels = 10'd512; layer_output_channels = 10'd512; layer_start_q64 = ADDRESS_WIDTH'(1714464); end
            default: begin
                layer_input_channels = 10'd512;
                layer_output_channels = 10'd11;
                layer_start_q64 = ADDRESS_WIDTH'(2304288);
            end
        endcase

        output_count = layer_output_channels - output_base_q > 10'd12 ?
                       10'd12 : layer_output_channels - output_base_q;
        input_count = layer_input_channels - input_base_q > 10'd8 ?
                      10'd8 : layer_input_channels - input_base_q;
        source_offset_q16 = 64'(output_base_q) * layer_input_channels * 9 +
                            64'(input_base_q) * 9;
        stride_q64 = ADDRESS_WIDTH'((64'(layer_input_channels) * 9) >> 2);
        total_q64 = COUNT_WIDTH'(18 * output_count);

        o_layer_index = layer_q;
        o_output_channel_base = output_base_q;
        o_output_channel_count = output_count;
        o_input_channel_base = input_base_q;
        o_input_channel_count = input_count;
        o_source_q64_word = layer_start_q64 + ADDRESS_WIDTH'(source_offset_q16 >> 2);
        o_total_q64_words = total_q64;
        o_segment_q64_words = COUNT_WIDTH'(18);
        o_source_stride_q64_words = stride_q64;

        // Layer 0 rows contain 18 Q16 values. Twelve-row wave boundaries are
        // Q64 aligned, so each descriptor is one contiguous block.
        if (layer_q == 0) begin
            o_source_q64_word = ADDRESS_WIDTH'((64'(output_base_q) * 18) >> 2);
            o_total_q64_words = COUNT_WIDTH'((64'(output_count) * 18) >> 2);
            o_segment_q64_words = COUNT_WIDTH'((64'(output_count) * 18) >> 2);
            o_source_stride_q64_words = ADDRESS_WIDTH'((64'(output_count) * 18) >> 2);
            o_input_channel_count = 10'd2;
        end else if (layer_q == 8) begin
            o_source_q64_word = ADDRESS_WIDTH'(2304288);
            o_total_q64_words = COUNT_WIDTH'(5632);
            o_segment_q64_words = COUNT_WIDTH'(5632);
            o_source_stride_q64_words = ADDRESS_WIDTH'(5632);
            o_input_channel_base = 10'd0;
            o_input_channel_count = 10'd512;
            o_output_channel_base = 10'd0;
            o_output_channel_count = 10'd11;
        end

        // The default 16-bit tag is collision-free for nine layers, up to 43
        // output waves and up to 64 input groups.
        o_tile_tag = TAG_WIDTH'({layer_q, output_wave_q, input_group_q});
        o_last_tile = o_busy && layer_q == 8;
    end

    assign o_tile_valid = o_busy;
    assign tile_accept = o_tile_valid && i_tile_ready;

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            layer_q <= '0;
            output_base_q <= '0;
            input_base_q <= '0;
            output_wave_q <= '0;
            input_group_q <= '0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_start && !o_busy) begin
                layer_q <= '0;
                output_base_q <= '0;
                input_base_q <= '0;
                output_wave_q <= '0;
                input_group_q <= '0;
                o_busy <= 1'b1;
            end else if (tile_accept) begin
                if (layer_q == 8) begin
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end else if (input_base_q + input_count < layer_input_channels) begin
                    input_base_q <= input_base_q + input_count;
                    input_group_q <= input_group_q + 1'b1;
                end else if (output_base_q + output_count < layer_output_channels) begin
                    output_base_q <= output_base_q + output_count;
                    output_wave_q <= output_wave_q + 1'b1;
                    input_base_q <= '0;
                    input_group_q <= '0;
                end else begin
                    layer_q <= layer_q + 1'b1;
                    output_base_q <= '0;
                    input_base_q <= '0;
                    output_wave_q <= '0;
                    input_group_q <= '0;
                end
            end
        end
    end

endmodule

`default_nettype wire
