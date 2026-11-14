`timescale 1ns/1ps
`default_nettype none

// Direct runtime-profile boundary for the verified 700-480-20 recurrent path.
// Profile completion is the final result handshake, not the earlier recurrent
// layer compute completion, so a new profile cannot overwrite output state.
module unified_srnn_h480_profile_execution_top #(
    parameter integer TILE_COUNT = 12,
    parameter integer NEURONS_PER_TILE = 40
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic [7:0]                                i_profile_opcode,
    input  logic                                      i_clear_error,
    input  logic                                      i_profile_start,
    output logic                                      o_profile_start_ready,
    input  logic [383:0]                              i_layer_words,
    output logic                                      o_layer_done,
    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [12:0]                               i_event_addr,
    input  logic signed [7:0]                         i_event_data,
    input  logic                                      i_timestep_last,
    input  logic                                      i_sample_last,
    output logic [TILE_COUNT-1:0]                     o_weight_req_valid,
    input  logic [TILE_COUNT-1:0]                     i_weight_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_weight_addr_flat,
    input  logic [TILE_COUNT*64-1:0]                  i_weight_data_flat,
    input  logic [31:0]                               i_w2_base,
    input  logic [5:0]                                i_w2_row_stride,
    output logic [TILE_COUNT-1:0]                     o_w2_req_valid,
    input  logic [TILE_COUNT-1:0]                     i_w2_req_ready,
    output logic [TILE_COUNT*32-1:0]                  o_w2_addr_flat,
    input  logic [TILE_COUNT*64-1:0]                  i_w2_data_flat,
    input  logic signed [15:0]                        i_leak_q15,
    input  logic signed [31:0]                        i_threshold,
    input  logic [4:0]                                i_current_left_shift,
    input  logic                                      i_reset_to_zero,
    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [5:0]                                o_result_class,
    output logic signed [31:0]                        o_result_score,
    output logic [32*32-1:0]                          o_result_logits_flat,
    output logic                                      o_busy,
    output logic [15:0]                               o_profile_version,
    output logic                                      o_sticky_error
);
    localparam logic [7:0] PROFILE_SRNN_H480 = 8'h10;

    logic decoder_ready;
    logic decoder_start_pulse;
    logic decoder_busy;
    logic [7:0] decoder_opcode;
    logic decoder_fc;
    logic decoder_recurrent;
    logic decoder_conv;
    logic [12:0] decoder_input_dim;
    logic [9:0] decoder_hidden;
    logic [5:0] decoder_output_dim;
    logic [4:0] decoder_tiles;
    logic [4:0] decoder_layers;
    logic [7:0] decoder_timesteps;
    logic [2:0] decoder_numeric;
    logic decoder_update;
    logic decoder_error;
    logic [7:0] decoder_error_code_unused;
    logic profile_supported;
    logic layer_start;
    logic inner_layer_ready;
    logic inner_busy;
    logic inner_error;
    logic unsupported_error_q;
    logic contract_error;
    logic result_fire;

    assign profile_supported = i_profile_opcode == PROFILE_SRNN_H480;
    assign o_profile_start_ready = decoder_ready && !decoder_busy &&
                                   inner_layer_ready && profile_supported;
    assign layer_start = i_profile_start && o_profile_start_ready;
    assign result_fire = o_result_valid && i_result_ready;
    assign contract_error = (decoder_start_pulse && !decoder_busy) ||
        (decoder_busy && (decoder_opcode != PROFILE_SRNN_H480 || decoder_fc ||
         !decoder_recurrent || decoder_conv || decoder_input_dim != 13'd700 ||
         decoder_hidden != 10'd480 || decoder_output_dim != 6'd20 ||
         decoder_tiles != 5'd12 || decoder_layers != 5'd1 ||
         decoder_timesteps != 8'd0 || decoder_numeric != 3'd2 ||
         decoder_update));
    assign o_busy = decoder_busy || inner_busy;
    assign o_sticky_error = decoder_error || inner_error ||
                            unsupported_error_q || contract_error;

    unified_runtime_profile_decoder u_profile (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_start(i_profile_start && profile_supported),
        .i_opcode(i_profile_opcode), .i_profile_done(result_fire),
        .i_clear_error(i_clear_error), .o_start_ready(decoder_ready),
        .o_start_pulse(decoder_start_pulse), .o_busy(decoder_busy),
        .o_active_opcode(decoder_opcode),
        .o_profile_version(o_profile_version), .o_fc_fast_path(decoder_fc),
        .o_recurrent_extension(decoder_recurrent),
        .o_conv_extension(decoder_conv), .o_input_dim(decoder_input_dim),
        .o_logical_hidden_or_channels(decoder_hidden),
        .o_output_dim(decoder_output_dim), .o_active_h40_tiles(decoder_tiles),
        .o_layer_count(decoder_layers), .o_fixed_timesteps(decoder_timesteps),
        .o_numeric_profile(decoder_numeric),
        .o_fc_w2_update_enable(decoder_update),
        .o_config_error(decoder_error),
        .o_error_code(decoder_error_code_unused)
    );

    unified_recurrent_compute_output_top #(
        .TILE_COUNT(TILE_COUNT), .NEURONS_PER_TILE(NEURONS_PER_TILE)
    ) u_recurrent (
        .clk(clk), .rst_n(rst_n), .i_clear_error(i_clear_error),
        .i_layer_valid(layer_start), .o_layer_ready(inner_layer_ready),
        .i_layer_words(i_layer_words), .o_layer_done(o_layer_done),
        .i_event_valid(i_event_valid), .o_event_ready(o_event_ready),
        .i_event_addr(i_event_addr), .i_event_data(i_event_data),
        .i_timestep_last(i_timestep_last), .i_sample_last(i_sample_last),
        .o_weight_req_valid(o_weight_req_valid),
        .i_weight_req_ready(i_weight_req_ready),
        .o_weight_addr_flat(o_weight_addr_flat),
        .i_weight_data_flat(i_weight_data_flat), .i_w2_base(i_w2_base),
        .i_w2_row_stride(i_w2_row_stride), .o_w2_req_valid(o_w2_req_valid),
        .i_w2_req_ready(i_w2_req_ready), .o_w2_addr_flat(o_w2_addr_flat),
        .i_w2_data_flat(i_w2_data_flat), .i_leak_q15(i_leak_q15),
        .i_threshold(i_threshold), .i_current_left_shift(i_current_left_shift),
        .i_reset_to_zero(i_reset_to_zero), .i_version(o_profile_version),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_class(o_result_class), .o_result_score(o_result_score),
        .o_result_logits_flat(o_result_logits_flat), .o_busy(inner_busy),
        .o_sticky_error(inner_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            unsupported_error_q <= 1'b0;
        else begin
            if (i_clear_error)
                unsupported_error_q <= 1'b0;
            if (i_profile_start && !profile_supported)
                unsupported_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
