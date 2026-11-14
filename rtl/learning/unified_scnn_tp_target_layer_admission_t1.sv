`default_nettype none

// Inserts the existing selected-layer input prescan between the serialized
// target-branch sequencer and the shared SCNN layer executor. Only the selected
// Conv8 command is delayed; all other Conv2..Conv7 commands pass through.
module unified_scnn_tp_target_layer_admission_t1 #(
    parameter int LAYER_WORDS = 12,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int FEATURE_DEPTH = 9216,
    parameter int MAX_WIDTH = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_prescan_enable,

    input  logic                                      i_branch_layer_valid,
    output logic                                      o_branch_layer_ready,
    input  logic [3:0]                                i_branch_layer_index,
    input  logic [LAYER_WORDS*32-1:0]                 i_branch_layer_words,
    input  logic                                      i_branch_input_slot,
    input  logic                                      i_branch_output_slot,
    input  logic [VERSION_WIDTH-1:0]                  i_branch_state_version,

    output logic                                      o_shared_layer_valid,
    input  logic                                      i_shared_layer_ready,
    output logic [3:0]                                o_shared_layer_index,
    output logic [LAYER_WORDS*32-1:0]                 o_shared_layer_words,
    output logic                                      o_shared_input_slot,
    output logic                                      o_shared_output_slot,
    output logic [VERSION_WIDTH-1:0]                  o_shared_state_version,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic                                      o_read_slot,
    output logic [31:0]                               o_read_banks,
    output logic [8*FEATURE_ADDR_WIDTH-1:0]           o_read_addrs,
    output logic [7:0]                                o_read_mask,
    input  logic                                      i_response_valid,
    output logic                                      o_response_ready,
    input  logic [127:0]                              i_response_data,
    input  logic [7:0]                                i_response_mask,

    output logic                                      o_observation_valid,
    input  logic                                      i_observation_ready,
    output logic [15:0]                               o_observation_channel,
    output logic [15:0]                               o_observation_spatial,
    output logic [3:0]                                o_observation_layer,
    output logic [VERSION_WIDTH-1:0]                  o_observation_version,
    output logic signed [15:0]                        o_observation_data_q8_8,
    output logic                                      o_observation_last,
    output logic                                      o_read_owner,
    output logic                                      o_busy,
    output logic                                      o_error
);
    assign o_shared_layer_index = i_branch_layer_index;
    assign o_shared_layer_words = i_branch_layer_words;
    assign o_shared_input_slot = i_branch_input_slot;
    assign o_shared_output_slot = i_branch_output_slot;
    assign o_shared_state_version = i_branch_state_version;

    unified_scnn_tp_input_prescan #(
        .ADDR_WIDTH(FEATURE_ADDR_WIDTH), .BANK_DEPTH(FEATURE_DEPTH),
        .MAX_WIDTH(MAX_WIDTH), .MAX_CHANNELS(MAX_CHANNELS),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_prescan (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_enable(i_prescan_enable),
        .i_layer_valid(i_branch_layer_valid),
        .o_layer_ready(o_branch_layer_ready),
        .o_layer_valid(o_shared_layer_valid),
        .i_layer_ready(i_shared_layer_ready),
        .i_width({10'd0, i_branch_layer_words[5*32 +: 6]}),
        .i_height({10'd0, i_branch_layer_words[5*32+6 +: 6]}),
        .i_channels({6'd0, i_branch_layer_words[5*32+12 +: 10]}),
        .i_slot(i_branch_input_slot), .i_base('0),
        .i_layer(i_branch_layer_index), .i_version(i_branch_state_version),
        .o_read_valid(o_read_valid), .i_read_ready(i_read_ready),
        .o_read_slot(o_read_slot), .o_read_banks(o_read_banks),
        .o_read_addrs(o_read_addrs), .o_read_mask(o_read_mask),
        .i_response_valid(i_response_valid),
        .o_response_ready(o_response_ready),
        .i_response_data(i_response_data),
        .i_response_mask(i_response_mask),
        .o_observation_valid(o_observation_valid),
        .i_observation_ready(i_observation_ready),
        .o_channel(o_observation_channel),
        .o_spatial(o_observation_spatial),
        .o_layer(o_observation_layer),
        .o_version(o_observation_version),
        .o_data_q8_8(o_observation_data_q8_8),
        .o_last(o_observation_last), .o_read_owner(o_read_owner),
        .o_busy(o_busy), .o_error(o_error)
    );
endmodule

`default_nettype wire
