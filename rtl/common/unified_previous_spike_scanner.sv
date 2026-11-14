`timescale 1ns/1ps
`default_nettype none

// Converts a previous-timestep spike bitmap into weight-row commands. Zero
// spikes consume no recurrent memory bandwidth.
module unified_previous_spike_scanner #(
    parameter integer NEURONS = 40
) (
    input  wire logic                 clk,
    input  wire logic                 rst_n,
    input  wire logic                 i_start,
    input  wire logic [NEURONS-1:0]   i_previous_spikes,
    input  wire logic [31:0]          i_recurrent_base,
    input  wire logic [9:0]           i_row_stride,
    output logic                      o_cmd_valid,
    input  wire logic                 i_cmd_ready,
    output logic [31:0]               o_weight_row_base,
    output logic signed [7:0]         o_activation,
    output logic                      o_busy,
    output logic                      o_done
);
    logic [NEURONS-1:0] spikes_q;
    logic [9:0] source_q;

    always_comb begin
        o_cmd_valid = o_busy && spikes_q[source_q];
        o_weight_row_base = i_recurrent_base + source_q * i_row_stride;
        o_activation = 8'sd1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spikes_q <= '0;
            source_q <= 10'd0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_start && !o_busy) begin
                spikes_q <= i_previous_spikes;
                source_q <= 10'd0;
                o_busy <= 1'b1;
            end else if (o_busy) begin
                if (!spikes_q[source_q] || (o_cmd_valid && i_cmd_ready)) begin
                    if (source_q == NEURONS-1) begin
                        o_busy <= 1'b0;
                        o_done <= 1'b1;
                    end else begin
                        source_q <= source_q + 1'b1;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
