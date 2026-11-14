`timescale 1ns/1ps
`default_nettype none

// Converts one RECURRENT_FC graph descriptor into a registered Wave3 command.
// Weight/numeric metadata is sampled before engine start and remains stable
// until the layer completes.
module unified_srnn_graph_layer_adapter_v1 #(
    parameter int VERSION_WIDTH = 16
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,
    input  logic                         i_abort,
    input  logic                         i_layer_valid,
    output logic                         o_layer_ready,
    input  logic [383:0]                 i_layer_words,
    input  logic                         i_sample_state_reset,
    input  logic [VERSION_WIDTH-1:0]     i_sample_version,

    output logic                         o_engine_start_valid,
    input  logic                         i_engine_start_ready,
    input  logic                         i_engine_done,
    input  logic                         i_engine_error,
    output logic                         o_state_reset,
    output logic [12:0]                  o_input_dim,
    output logic [9:0]                   o_active_neurons,
    output logic [3:0]                   o_active_tiles,
    output logic [5:0]                   o_output_count,
    output logic [31:0]                  o_w2_base,
    output logic [5:0]                   o_w2_row_stride,
    output logic [31:0]                  o_recurrent_base,
    output logic                         o_recurrent_enable,
    output logic [31:0]                  o_w0_base,
    output logic signed [15:0]           o_leak_q15,
    output logic signed [31:0]           o_threshold,
    output logic [4:0]                   o_current_left_shift,
    output logic                         o_reset_to_zero,
    output logic [VERSION_WIDTH-1:0]     o_version,
    output logic                         o_layer_done,
    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [31:0] LAYER_MAGIC = 32'h4e4d5232;
    localparam logic [31:0] ABI_V22 = 32'h000c0202;
    localparam logic [3:0] LAYER_RECURRENT_FC = 4'd5;
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_DESCRIPTOR = 8'h01;
    localparam logic [7:0] ERROR_START_WHILE_BUSY = 8'h02;
    localparam logic [7:0] ERROR_ENGINE_DONE_IDLE = 8'h03;
    localparam logic [7:0] ERROR_ENGINE = 8'h04;
    localparam logic [7:0] ERROR_ABORT = 8'h05;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_START,
        STATE_RUN
    } state_t;

    state_t state_q;
    logic local_error_q;
    logic [7:0] local_error_code_q;
    logic [31:0] word0;
    logic [31:0] word1;
    logic [31:0] word2;
    logic [31:0] word3;
    logic [31:0] word4;
    logic [31:0] word5;
    logic [31:0] word6;
    logic [31:0] word7;
    logic [31:0] word8;
    logic [31:0] word9;
    logic [12:0] descriptor_input_dim;
    logic [9:0] descriptor_state_dim;
    logic [5:0] descriptor_output_dim;
    logic [4:0] descriptor_active_tiles;
    logic [3:0] descriptor_issue_k;
    logic descriptor_legal;
    logic layer_fire;

    function automatic logic state_dim_fits_h40_tiles(
        input logic [9:0] state_dim,
        input logic [4:0] tile_count
    );
        integer capacity;
        begin
            state_dim_fits_h40_tiles = 1'b0;
            if (tile_count != 0) begin
                capacity = integer'(tile_count) * 40;
                state_dim_fits_h40_tiles =
                    integer'(state_dim) > capacity - 40 &&
                    integer'(state_dim) <= capacity;
            end
        end
    endfunction

    always_comb begin
        word0 = i_layer_words[0*32 +: 32];
        word1 = i_layer_words[1*32 +: 32];
        word2 = i_layer_words[2*32 +: 32];
        word3 = i_layer_words[3*32 +: 32];
        word4 = i_layer_words[4*32 +: 32];
        word5 = i_layer_words[5*32 +: 32];
        word6 = i_layer_words[6*32 +: 32];
        word7 = i_layer_words[7*32 +: 32];
        word8 = i_layer_words[8*32 +: 32];
        word9 = i_layer_words[9*32 +: 32];
        descriptor_input_dim = word3[12:0];
        descriptor_state_dim = word3[22:13];
        descriptor_output_dim = word3[28:23];
        descriptor_active_tiles = word4[4:0];
        descriptor_issue_k = word4[17:14];
        descriptor_legal = word0 == LAYER_MAGIC && word1 == ABI_V22 &&
            word2[3:0] == LAYER_RECURRENT_FC &&
            descriptor_input_dim != 0 && descriptor_input_dim <= 13'd2312 &&
            descriptor_active_tiles != 0 &&
            descriptor_active_tiles <= 5'd12 &&
            state_dim_fits_h40_tiles(
                descriptor_state_dim, descriptor_active_tiles) &&
            descriptor_output_dim != 0 &&
            descriptor_output_dim <= 6'd32 &&
            descriptor_issue_k == 4'd8 &&
            (word2[4] || word6 != 0) && i_sample_version != 0;
        o_layer_ready = state_q == STATE_IDLE && !local_error_q;
        layer_fire = i_layer_valid && o_layer_ready;
        o_engine_start_valid = state_q == STATE_START;
        o_busy = state_q != STATE_IDLE;
        o_idle = state_q == STATE_IDLE;
        o_sticky_error = local_error_q;
        o_error_code = local_error_q ? local_error_code_q : ERROR_NONE;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
            o_state_reset <= 1'b0;
            o_input_dim <= '0;
            o_active_neurons <= '0;
            o_active_tiles <= '0;
            o_output_count <= '0;
            o_w2_base <= '0;
            o_w2_row_stride <= '0;
            o_recurrent_base <= '0;
            o_recurrent_enable <= 1'b0;
            o_w0_base <= '0;
            o_leak_q15 <= '0;
            o_threshold <= '0;
            o_current_left_shift <= '0;
            o_reset_to_zero <= 1'b0;
            o_version <= '0;
            o_layer_done <= 1'b0;
        end else begin
            o_layer_done <= 1'b0;
            // The sidecar treats state reset as a transaction pulse. Holding
            // it while the descriptor runs would reset again during CLEAR and
            // correctly trip its reset-while-busy protocol error.
            o_state_reset <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (layer_fire) begin
                if (!descriptor_legal) begin
                    local_error_q <= 1'b1;
                    local_error_code_q <= ERROR_DESCRIPTOR;
                end else begin
                    o_state_reset <= i_sample_state_reset;
                    o_input_dim <= descriptor_input_dim;
                    o_active_neurons <= descriptor_state_dim;
                    o_active_tiles <= descriptor_active_tiles[3:0];
                    o_output_count <= descriptor_output_dim;
                    o_w2_base <= word5;
                    o_w2_row_stride <= descriptor_output_dim;
                    o_recurrent_base <= word6;
                    // ABI V2.2 reserved bit 4 is a backward-compatible
                    // recurrence-disable bit. Existing descriptors keep it 0.
                    o_recurrent_enable <= !word2[4];
                    o_w0_base <= word7;
                    o_leak_q15 <= word8[15:0];
                    o_current_left_shift <= word8[20:16];
                    o_reset_to_zero <= word8[21];
                    o_threshold <= word9;
                    o_version <= i_sample_version;
                    state_q <= STATE_START;
                end
            end else if (i_layer_valid && state_q != STATE_IDLE &&
                         !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_START_WHILE_BUSY;
            end

            if (state_q == STATE_START && i_engine_start_ready)
                state_q <= STATE_RUN;
            if (i_engine_done) begin
                if (state_q == STATE_RUN) begin
                    state_q <= STATE_IDLE;
                    o_layer_done <= 1'b1;
                end else if (!local_error_q) begin
                    local_error_q <= 1'b1;
                    local_error_code_q <= ERROR_ENGINE_DONE_IDLE;
                end
            end
            if (i_engine_error && !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_ENGINE;
            end
            if (i_abort) begin
                state_q <= STATE_IDLE;
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_ABORT;
            end
        end
    end
endmodule

`default_nettype wire
