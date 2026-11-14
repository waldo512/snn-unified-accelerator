`timescale 1ns/1ps
`default_nettype none

// A regional ten-lane stage merges the two FC accumulator banks immediately
// before normalization. P2 SRNN/SCNN traffic passes through the same DSPs.
module unified_dual_role_p4fc_pair_merge10_candidate_v1 (
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,
    input  wire logic                    i_abort,
    input  wire logic                    i_valid,
    output logic                         o_ready,
    input  wire logic                    i_p4_fc,
    input  wire logic [9:0]              i_lane_mask,
    input  wire logic [10*48-1:0]        i_primary_flat,
    input  wire logic [10*48-1:0]        i_secondary_flat,
    output logic                         o_valid,
    input  wire logic                    i_ready,
    output logic [9:0]                   o_lane_mask,
    output logic [10*48-1:0]             o_merged_flat
);
    logic valid_q;
    logic [9:0] lane_mask_q;
    logic input_fire;

    assign o_ready = !i_abort && (!valid_q || i_ready);
    assign input_fire = i_valid && o_ready;
    assign o_valid = !i_abort && valid_q;
    assign o_lane_mask = lane_mask_q;

    generate
        for (genvar lane = 0; lane < 10; lane++) begin : g_lane
            unified_dual_role_dsp_add48_lane_candidate_v1 u_merge (
                .i_clk(i_clk),
                .i_fire(input_fire && i_lane_mask[lane]),
                .i_add_secondary(i_p4_fc),
                .i_primary($signed(i_primary_flat[lane*48 +: 48])),
                .i_secondary($signed(i_secondary_flat[lane*48 +: 48])),
                .o_result(o_merged_flat[lane*48 +: 48])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            valid_q <= 1'b0;
            lane_mask_q <= '0;
        end else if (i_abort) begin
            valid_q <= 1'b0;
            lane_mask_q <= '0;
        end else begin
            if (valid_q && i_ready)
                valid_q <= 1'b0;
            if (input_fire) begin
                valid_q <= 1'b1;
                lane_mask_q <= i_lane_mask;
            end
        end
    end
endmodule

`default_nettype wire
