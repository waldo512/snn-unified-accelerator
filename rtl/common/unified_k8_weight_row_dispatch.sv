`timescale 1ns/1ps
`default_nettype none

// Shared row dispatcher for sparse-FC input events and recurrent spike rows.
// The caller supplies a source row and activation; the same request stream is
// consumed by the working-set weight memory for either operation type.
module unified_k8_weight_row_dispatch #(
    parameter integer LANES = 8,
    parameter integer MAX_DESTINATIONS = 40
) (
    input  wire logic                 clk,
    input  wire logic                 rst_n,
    input  wire logic                 i_cmd_valid,
    output logic                      o_cmd_ready,
    input  wire logic [31:0]          i_weight_row_base,
    input  wire logic [5:0]           i_destination_count,
    input  wire logic signed [7:0]    i_activation,

    output logic                      o_req_valid,
    input  wire logic                 i_req_ready,
    output logic [31:0]               o_weight_addr,
    output logic [LANES-1:0]          o_lane_valid,
    output logic [LANES*6-1:0]        o_neuron_index_flat,
    output logic [LANES*8-1:0]        o_activation_flat,
    output logic                      o_row_done
);
    logic active_q;
    logic [31:0] row_base_q;
    logic [5:0] destination_count_q;
    logic signed [7:0] activation_q;
    logic [5:0] destination_base_q;
    integer lane_i;

    always_comb begin
        o_cmd_ready = !active_q;
        o_req_valid = active_q;
        o_weight_addr = row_base_q + destination_base_q;
        o_lane_valid = '0;
        o_neuron_index_flat = '0;
        o_activation_flat = '0;
        for (lane_i = 0; lane_i < LANES; lane_i = lane_i + 1) begin
            if (active_q &&
                ((destination_base_q + lane_i) < destination_count_q)) begin
                o_lane_valid[lane_i] = 1'b1;
                o_neuron_index_flat[lane_i*6 +: 6] =
                    destination_base_q + lane_i;
                o_activation_flat[lane_i*8 +: 8] = activation_q;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q <= 1'b0;
            row_base_q <= 32'd0;
            destination_count_q <= 6'd0;
            activation_q <= 8'sd0;
            destination_base_q <= 6'd0;
            o_row_done <= 1'b0;
        end else begin
            o_row_done <= 1'b0;
            if (i_cmd_valid && o_cmd_ready) begin
                if ((i_destination_count > 0) &&
                    (i_destination_count <= MAX_DESTINATIONS)) begin
                    active_q <= 1'b1;
                    row_base_q <= i_weight_row_base;
                    destination_count_q <= i_destination_count;
                    activation_q <= i_activation;
                    destination_base_q <= 6'd0;
                end else begin
                    o_row_done <= 1'b1;
                end
            end
            if (o_req_valid && i_req_ready) begin
                if ((destination_base_q + LANES) >= destination_count_q) begin
                    active_q <= 1'b0;
                    o_row_done <= 1'b1;
                end else begin
                    destination_base_q <= destination_base_q + LANES;
                end
            end
        end
    end
endmodule

`default_nettype wire

