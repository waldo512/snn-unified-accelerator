`timescale 1ns/1ps
`default_nettype none

module unified_recurrent_tile_ooc (
    input  wire logic             clk,
    input  wire logic             rst_n,
    input  wire logic             i_start,
    input  wire logic [39:0]      i_previous_spikes,
    input  wire logic [31:0]      i_recurrent_base,
    output logic                  o_weight_req_valid,
    input  wire logic             i_weight_req_ready,
    output logic [31:0]           o_weight_addr,
    input  wire logic [63:0]      i_weight_data,
    output logic                  o_done,
    output logic [39:0]           o_current_spikes
);
    logic scanner_cmd_valid;
    logic scanner_cmd_ready;
    logic [31:0] scanner_row_base;
    logic signed [7:0] scanner_activation;
    logic scanner_done;
    logic [7:0] lane_valid;
    logic [47:0] neuron_index_flat;
    logic [63:0] activation_flat;
    logic row_done;
    logic mac_ready;
    logic scan_complete_q;
    logic row_inflight_q;
    logic lif_start_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scan_complete_q <= 1'b0;
            row_inflight_q <= 1'b0;
            lif_start_q <= 1'b0;
        end else begin
            lif_start_q <= 1'b0;
            if (i_start) begin
                scan_complete_q <= 1'b0;
                row_inflight_q <= 1'b0;
            end
            if (row_done)
                row_inflight_q <= 1'b0;
            if (scanner_cmd_valid && scanner_cmd_ready)
                row_inflight_q <= 1'b1;
            if (scanner_done)
                scan_complete_q <= 1'b1;
            if (scan_complete_q && !row_inflight_q) begin
                scan_complete_q <= 1'b0;
                lif_start_q <= 1'b1;
            end
        end
    end

    unified_previous_spike_scanner #(.NEURONS(40)) u_scanner (
        .clk(clk), .rst_n(rst_n), .i_start(i_start),
        .i_previous_spikes(i_previous_spikes), .i_recurrent_base(i_recurrent_base),
        .i_row_stride(10'd40), .o_cmd_valid(scanner_cmd_valid),
        .i_cmd_ready(scanner_cmd_ready), .o_weight_row_base(scanner_row_base),
        .o_activation(scanner_activation), .o_busy(), .o_done(scanner_done)
    );

    unified_k8_weight_row_dispatch u_dispatch (
        .clk(clk), .rst_n(rst_n), .i_cmd_valid(scanner_cmd_valid),
        .o_cmd_ready(scanner_cmd_ready), .i_weight_row_base(scanner_row_base),
        .i_destination_count(6'd40), .i_activation(scanner_activation),
        .o_req_valid(o_weight_req_valid), .i_req_ready(i_weight_req_ready && mac_ready),
        .o_weight_addr(o_weight_addr), .o_lane_valid(lane_valid),
        .o_neuron_index_flat(neuron_index_flat),
        .o_activation_flat(activation_flat), .o_row_done(row_done)
    );

    unified_h40_k8_tile u_tile (
        .clk(clk), .rst_n(rst_n), .i_clear_sample(i_start), .i_clear_accum(1'b0),
        .i_mac_valid(o_weight_req_valid && i_weight_req_ready), .o_mac_ready(mac_ready),
        .i_lane_valid(lane_valid), .i_neuron_index_flat(neuron_index_flat),
        .i_activation_flat(activation_flat), .i_weight_flat(i_weight_data),
        .i_acc24_compat(1'b0), .i_lif_start(lif_start_q),
        .i_leak_q15(16'sd27853), .i_threshold(32'sd32),
        .i_current_left_shift(5'd0), .i_reset_to_zero(1'b0),
        .o_lif_busy(), .o_lif_done(o_done), .o_lif_spike_valid(),
        .o_lif_base(), .o_lif_spike_mask(), .i_timestep_commit(1'b0),
        .o_previous_spikes(), .o_current_spikes(o_current_spikes),
        .i_debug_neuron(6'd0), .o_debug_accum(), .o_debug_membrane(),
        .o_lane_collision_error()
    );
endmodule

`default_nettype wire
