`timescale 1ns/1ps
`default_nettype none

// Registered signed reduction for 1..12 H40 tile partials. Up to eight tiles
// use a three-stage tree; nine to twelve tiles use a four-stage tree. Missing
// leaves are constant zero, so a tail tile does not need a special reducer.
module unified_fc_compat_w2_reduction_tree #(
    parameter integer TILES = 4,
    parameter integer O_DIM = 10,
    parameter integer ACC_BITS = 24,
    parameter integer META_BITS = 9
) (
    input  logic                                  clk,
    input  logic                                  rst_n,
    input  logic                                  i_valid,
    input  logic [TILES*O_DIM*ACC_BITS-1:0]       i_partials_flat,
    input  logic [META_BITS-1:0]                  i_meta,
    output logic                                  o_valid,
    output logic [O_DIM*ACC_BITS-1:0]             o_reduced_flat,
    output logic [META_BITS-1:0]                  o_meta
);
    localparam integer PAD_TILES = (TILES <= 8) ? 8 : 16;
    localparam integer PAD_BITS = PAD_TILES * O_DIM * ACC_BITS;

    logic [PAD_BITS-1:0] padded_partials;
    logic stage1_valid_q;
    logic stage2_valid_q;
    logic stage3_valid_q;
    logic stage4_valid_q;
    logic [META_BITS-1:0] stage1_meta_q;
    logic [META_BITS-1:0] stage2_meta_q;
    logic [META_BITS-1:0] stage3_meta_q;
    logic [META_BITS-1:0] stage4_meta_q;
    logic signed [ACC_BITS-1:0] stage1_sum_q [0:(PAD_TILES/2)*O_DIM-1];
    logic signed [ACC_BITS-1:0] stage2_sum_q [0:(PAD_TILES/4)*O_DIM-1];
    logic signed [ACC_BITS-1:0] stage3_sum_q [0:(PAD_TILES/8)*O_DIM-1];
    logic signed [ACC_BITS-1:0] stage4_sum_q [0:O_DIM-1];

    assign padded_partials = {{(PAD_TILES-TILES)*O_DIM*ACC_BITS{1'b0}},
                              i_partials_flat};

    generate
        if (PAD_TILES == 8) begin : gen_eight_leaf_output
            assign o_valid = stage3_valid_q;
            assign o_meta = stage3_meta_q;
            for (genvar output_i = 0; output_i < O_DIM;
                 output_i = output_i + 1) begin : gen_output
                assign o_reduced_flat[output_i*ACC_BITS +: ACC_BITS] =
                    stage3_sum_q[output_i];
            end
        end else begin : gen_sixteen_leaf_output
            assign o_valid = stage4_valid_q;
            assign o_meta = stage4_meta_q;
            for (genvar output_i = 0; output_i < O_DIM;
                 output_i = output_i + 1) begin : gen_output
                assign o_reduced_flat[output_i*ACC_BITS +: ACC_BITS] =
                    stage4_sum_q[output_i];
            end
        end
    endgenerate

    integer sum_i;
    integer output_i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid_q <= 1'b0;
            stage2_valid_q <= 1'b0;
            stage3_valid_q <= 1'b0;
            stage4_valid_q <= 1'b0;
            stage1_meta_q <= '0;
            stage2_meta_q <= '0;
            stage3_meta_q <= '0;
            stage4_meta_q <= '0;
            for (sum_i = 0; sum_i < (PAD_TILES/2)*O_DIM;
                 sum_i = sum_i + 1)
                stage1_sum_q[sum_i] <= '0;
            for (sum_i = 0; sum_i < (PAD_TILES/4)*O_DIM;
                 sum_i = sum_i + 1)
                stage2_sum_q[sum_i] <= '0;
            for (sum_i = 0; sum_i < (PAD_TILES/8)*O_DIM;
                 sum_i = sum_i + 1)
                stage3_sum_q[sum_i] <= '0;
            for (sum_i = 0; sum_i < O_DIM; sum_i = sum_i + 1)
                stage4_sum_q[sum_i] <= '0;
        end else begin
            stage1_valid_q <= i_valid;
            stage2_valid_q <= stage1_valid_q;
            stage3_valid_q <= stage2_valid_q;
            stage4_valid_q <= (PAD_TILES == 16) && stage3_valid_q;

            if (i_valid) begin
                stage1_meta_q <= i_meta;
                for (sum_i = 0; sum_i < PAD_TILES/2;
                     sum_i = sum_i + 1)
                    for (output_i = 0; output_i < O_DIM;
                         output_i = output_i + 1)
                        stage1_sum_q[sum_i*O_DIM+output_i] <=
                            $signed(padded_partials[
                                ((2*sum_i)*O_DIM+output_i)*ACC_BITS +:
                                ACC_BITS
                            ]) +
                            $signed(padded_partials[
                                ((2*sum_i+1)*O_DIM+output_i)*ACC_BITS +:
                                ACC_BITS
                            ]);
            end
            if (stage1_valid_q) begin
                stage2_meta_q <= stage1_meta_q;
                for (sum_i = 0; sum_i < PAD_TILES/4;
                     sum_i = sum_i + 1)
                    for (output_i = 0; output_i < O_DIM;
                         output_i = output_i + 1)
                        stage2_sum_q[sum_i*O_DIM+output_i] <=
                            stage1_sum_q[(2*sum_i)*O_DIM+output_i] +
                            stage1_sum_q[(2*sum_i+1)*O_DIM+output_i];
            end
            if (stage2_valid_q) begin
                stage3_meta_q <= stage2_meta_q;
                for (sum_i = 0; sum_i < PAD_TILES/8;
                     sum_i = sum_i + 1)
                    for (output_i = 0; output_i < O_DIM;
                         output_i = output_i + 1)
                        stage3_sum_q[sum_i*O_DIM+output_i] <=
                            stage2_sum_q[(2*sum_i)*O_DIM+output_i] +
                            stage2_sum_q[(2*sum_i+1)*O_DIM+output_i];
            end
            if ((PAD_TILES == 16) && stage3_valid_q) begin
                stage4_meta_q <= stage3_meta_q;
                for (output_i = 0; output_i < O_DIM;
                     output_i = output_i + 1)
                    stage4_sum_q[output_i] <=
                        stage3_sum_q[output_i] +
                        stage3_sum_q[O_DIM+output_i];
            end
        end
    end

    initial begin
        if (TILES < 1 || TILES > 12)
            $error("FC_COMPAT reduction supports 1..12 H40 tiles");
    end
endmodule

`default_nettype wire
