`default_nettype none

module unified_scnn_v23_layer_adapter #(
    parameter int LAYER_WORDS = 12,
    parameter int Q16_ADDR_WIDTH = 23,
    parameter int STATE_ADDR_WIDTH = 19,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_layer_valid,
    output logic                           o_layer_ready,
    input  logic [LAYER_WORDS*32-1:0]      i_layer_words,
    input  logic                           i_input_slot,
    input  logic                           i_output_slot,
    input  logic [VERSION_WIDTH-1:0]       i_sample_version,
    output logic                           o_engine_start,
    input  logic                           i_engine_start_ready,
    input  logic                           i_engine_done,
    output logic [15:0]                    o_in_width,
    output logic [15:0]                    o_in_height,
    output logic [15:0]                    o_in_channels,
    output logic [15:0]                    o_out_width,
    output logic [15:0]                    o_out_height,
    output logic [15:0]                    o_out_channels,
    output logic [7:0]                     o_stride,
    output logic [7:0]                     o_padding,
    output logic [1:0]                     o_pool_mode,
    output logic                           o_input_slot,
    output logic                           o_output_slot,
    output logic [Q16_ADDR_WIDTH-1:0]      o_weight_q16_base,
    output logic [STATE_ADDR_WIDTH-1:0]    o_state_base,
    output logic                           o_unit_spike_features,
    output logic [VERSION_WIDTH-1:0]       o_sample_version,
    output logic signed [15:0]             o_leak_q8_8,
    output logic signed [15:0]             o_threshold_q8_8,
    output logic                           o_reset_to_zero,
    output logic                           o_busy,
    output logic                           o_layer_done,
    output logic                           o_descriptor_error
);
    localparam logic [31:0] LAYER_MAGIC = 32'h4e4d5232;
    localparam logic [31:0] ABI_V23 = 32'h000c0302;
    localparam logic [3:0] LAYER_CONV = 4'd2;

    logic [31:0] word0;
    logic [31:0] word1;
    logic [31:0] word2;
    logic [31:0] word4;
    logic [31:0] word5;
    logic [31:0] word6;
    logic [31:0] word7;
    logic [31:0] word8;
    logic [31:0] word9;
    logic descriptor_valid;

    always_comb begin
        word0 = i_layer_words[0*32 +: 32];
        word1 = i_layer_words[1*32 +: 32];
        word2 = i_layer_words[2*32 +: 32];
        word4 = i_layer_words[4*32 +: 32];
        word5 = i_layer_words[5*32 +: 32];
        word6 = i_layer_words[6*32 +: 32];
        word7 = i_layer_words[7*32 +: 32];
        word8 = i_layer_words[8*32 +: 32];
        word9 = i_layer_words[9*32 +: 32];
        descriptor_valid = word0 == LAYER_MAGIC && word1 == ABI_V23 &&
            word2[3:0] == LAYER_CONV && word4[4:0] > 0 &&
            word4[4:0] <= 12 && word4[17:14] == 4'd8 &&
            word5[5:0] > 0 && word5[5:0] <= 32 &&
            word5[11:6] > 0 && word5[11:6] <= 32 &&
            word5[21:12] > 0 && word5[21:12] <= 512 &&
            word5[23:22] > 0 && word5[23:22] <= 2 &&
            word5[25:24] <= 1 &&
            word6[5:0] > 0 && word6[5:0] <= 32 &&
            word6[11:6] > 0 && word6[11:6] <= 32 &&
            word6[21:12] > 0 && word6[21:12] <= 512 &&
            word6[23:22] <= 2 &&
            word6[5:0] == word5[5:0] &&
            word6[11:6] == word5[11:6] &&
            i_sample_version != 0;
        o_layer_ready = !o_busy && i_engine_start_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_engine_start <= 1'b0;
            o_in_width <= '0;
            o_in_height <= '0;
            o_in_channels <= '0;
            o_out_width <= '0;
            o_out_height <= '0;
            o_out_channels <= '0;
            o_stride <= '0;
            o_padding <= '0;
            o_pool_mode <= '0;
            o_input_slot <= 1'b0;
            o_output_slot <= 1'b0;
            o_weight_q16_base <= '0;
            o_state_base <= '0;
            o_unit_spike_features <= 1'b0;
            o_sample_version <= '0;
            o_leak_q8_8 <= '0;
            o_threshold_q8_8 <= '0;
            o_reset_to_zero <= 1'b0;
            o_busy <= 1'b0;
            o_layer_done <= 1'b0;
            o_descriptor_error <= 1'b0;
        end else begin
            o_engine_start <= 1'b0;
            o_layer_done <= 1'b0;
            if (i_layer_valid && o_layer_ready) begin
                if (!descriptor_valid) begin
                    o_descriptor_error <= 1'b1;
                end else begin
                    o_engine_start <= 1'b1;
                    o_in_width <= {10'd0, word5[5:0]};
                    o_in_height <= {10'd0, word5[11:6]};
                    o_in_channels <= {6'd0, word5[21:12]};
                    o_out_width <= {10'd0, word6[5:0]};
                    o_out_height <= {10'd0, word6[11:6]};
                    o_out_channels <= {6'd0, word6[21:12]};
                    o_stride <= {6'd0, word5[23:22]};
                    o_padding <= {6'd0, word5[25:24]};
                    o_pool_mode <= word6[23:22];
                    o_input_slot <= i_input_slot;
                    o_output_slot <= i_output_slot;
                    o_weight_q16_base <= word7[Q16_ADDR_WIDTH-1:0];
                    o_state_base <= word8[STATE_ADDR_WIDTH-1:0];
                    o_unit_spike_features <= word5[26];
                    o_sample_version <= i_sample_version;
                    o_leak_q8_8 <= word9[15:0];
                    o_threshold_q8_8 <= word9[31:16];
                    o_reset_to_zero <= 1'b0;
                    o_busy <= 1'b1;
                end
            end
            if (i_engine_done && o_busy) begin
                o_busy <= 1'b0;
                o_layer_done <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
