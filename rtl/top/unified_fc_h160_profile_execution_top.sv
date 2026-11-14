`timescale 1ns/1ps
`default_nettype none

// G2 FC-first profile boundary. Opcode 0x01 starts the canonical H160 FC
// datapath on the same clock edge; no graph-sequencer or generic row-command
// cycle is inserted into the FC sample interval.
module unified_fc_h160_profile_execution_top (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic [7:0]                   i_profile_opcode,
    input  logic                         i_clear_error,
    input  logic                         i_sample_start,
    output logic                         o_start_ready,
    input  logic                         i_window_valid,
    output logic                         o_window_ready,
    input  logic [7:0]                   i_record_valid,
    input  logic [15:0]                  i_sample_id,
    input  logic [8:0]                   i_timestep,
    input  logic [7:0]                   i_boundary_mask,
    input  logic                         i_sample_last,
    input  logic [8*12-1:0]              i_fi_flat,
    input  logic [63:0]                  i_x_flat,
    output logic                         o_sample_done,
    output logic [3:0]                   o_pred,
    output logic [10*24-1:0]             o_logits_flat,
    output logic [4*40-1:0]              o_spikes_flat,
    output logic [4*40*18-1:0]           o_vmem_flat,
    output logic [4*40*16-1:0]           o_trace_flat,
    output logic [31:0]                  o_accepted_windows,
    output logic [31:0]                  o_accepted_events,
    output logic [31:0]                  o_issue_beats,
    output logic [31:0]                  o_process_beats,
    output logic [31:0]                  o_boundary_count,
    output logic [31:0]                  o_input_stall_cycles,
    output logic [31:0]                  o_output_stall_cycles,
    output logic [63:0]                  o_sample_cycles,
    output logic [63:0]                  o_boundary_wait_cycles,
    output logic [63:0]                  o_cluster_wait_cycles,
    output logic [63:0]                  o_reduction_wait_cycles,
    output logic                         o_protocol_error,
    output logic                         o_fc_active,
    output logic                         o_extension_active,
    output logic [15:0]                  o_profile_version,
    output logic                         o_profile_error
);
    localparam logic [7:0] PROFILE_FC_H160 = 8'h01;

    logic decoder_start_ready;
    logic decoder_busy;
    logic decoder_error;
    logic [7:0] decoder_error_code_unused;
    logic decoder_start_pulse;
    logic [7:0] decoder_active_opcode;
    logic decoder_fc_fast;
    logic decoder_recurrent;
    logic decoder_conv;
    logic [12:0] decoder_input_dim;
    logic [9:0] decoder_hidden;
    logic [5:0] decoder_output_dim;
    logic [4:0] decoder_tiles;
    logic [4:0] decoder_layers;
    logic [7:0] decoder_timesteps;
    logic [2:0] decoder_numeric;
    logic decoder_update_enable;
    logic decoder_contract_error;
    logic fc_sample_start;
    logic fc_mode_error;
    logic unsupported_error_q;
    logic profile_supported;

    assign profile_supported = i_profile_opcode == PROFILE_FC_H160;
    assign o_start_ready = decoder_start_ready && !decoder_busy &&
                           profile_supported;
    assign fc_sample_start = i_sample_start && o_start_ready;
    assign decoder_contract_error = (decoder_start_pulse && !decoder_busy) ||
        (decoder_busy && (decoder_active_opcode != PROFILE_FC_H160 ||
         !decoder_fc_fast ||
         decoder_recurrent || decoder_conv || decoder_input_dim != 13'd2312 ||
         decoder_hidden != 10'd160 || decoder_output_dim != 6'd10 ||
         decoder_tiles != 5'd4 || decoder_layers != 5'd1 ||
         decoder_timesteps != 8'd0 || decoder_numeric != 3'd1 ||
         !decoder_update_enable));
    assign o_profile_error = decoder_error || unsupported_error_q ||
                             fc_mode_error || decoder_contract_error;

    unified_runtime_profile_decoder u_profile (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_start(i_sample_start && profile_supported),
        .i_opcode(i_profile_opcode), .i_profile_done(o_sample_done),
        .i_clear_error(i_clear_error), .o_start_ready(decoder_start_ready),
        .o_start_pulse(decoder_start_pulse), .o_busy(decoder_busy),
        .o_active_opcode(decoder_active_opcode),
        .o_profile_version(o_profile_version),
        .o_fc_fast_path(decoder_fc_fast),
        .o_recurrent_extension(decoder_recurrent),
        .o_conv_extension(decoder_conv), .o_input_dim(decoder_input_dim),
        .o_logical_hidden_or_channels(decoder_hidden),
        .o_output_dim(decoder_output_dim), .o_active_h40_tiles(decoder_tiles),
        .o_layer_count(decoder_layers), .o_fixed_timesteps(decoder_timesteps),
        .o_numeric_profile(decoder_numeric),
        .o_fc_w2_update_enable(decoder_update_enable),
        .o_config_error(decoder_error), .o_error_code(decoder_error_code_unused)
    );

    unified_fc_h160_execution_top u_fc (
        .clk(clk), .rst_n(rst_n), .i_mode_q(2'd0),
        .i_clear_error(i_clear_error), .i_sample_start(fc_sample_start),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid), .i_sample_id(i_sample_id),
        .i_timestep(i_timestep), .i_boundary_mask(i_boundary_mask),
        .i_sample_last(i_sample_last), .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat), .o_sample_done(o_sample_done), .o_pred(o_pred),
        .o_logits_flat(o_logits_flat), .o_spikes_flat(o_spikes_flat),
        .o_vmem_flat(o_vmem_flat), .o_trace_flat(o_trace_flat),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events), .o_issue_beats(o_issue_beats),
        .o_process_beats(o_process_beats), .o_boundary_count(o_boundary_count),
        .o_input_stall_cycles(o_input_stall_cycles),
        .o_output_stall_cycles(o_output_stall_cycles),
        .o_sample_cycles(o_sample_cycles),
        .o_boundary_wait_cycles(o_boundary_wait_cycles),
        .o_cluster_wait_cycles(o_cluster_wait_cycles),
        .o_reduction_wait_cycles(o_reduction_wait_cycles),
        .o_protocol_error(o_protocol_error), .o_fc_active(o_fc_active),
        .o_extension_active(o_extension_active), .o_mode_error(fc_mode_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            unsupported_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                unsupported_error_q <= 1'b0;
            if (i_sample_start && !profile_supported)
                unsupported_error_q <= 1'b1;
        end
    end

endmodule

`default_nettype wire
