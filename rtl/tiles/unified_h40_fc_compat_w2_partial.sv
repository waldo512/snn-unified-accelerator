`timescale 1ns/1ps
`default_nettype none

// Runtime-weight form of the canonical H40 W2 partial-logit tree. The three
// registered reduction levels preserve the existing modular ACC24 result and
// accept one spike vector per cycle.
module unified_h40_fc_compat_w2_partial #(
    parameter integer H_COUNT = 40,
    parameter integer O_DIM = 10,
    parameter integer W_BITS = 8,
    parameter integer ACC_BITS = 24
) (
    input  logic                                    clk,
    input  logic                                    rst_n,
    input  logic                                    i_start,
    input  logic [H_COUNT-1:0]                      i_spikes,
    input  logic [H_COUNT*O_DIM*W_BITS-1:0]         i_weight_flat,
    output logic                                    o_done,
    output logic [O_DIM*ACC_BITS-1:0]               o_partial_flat
);
    localparam integer LOCAL_GROUP_SIZE = 5;
    localparam integer GROUP_COUNT = H_COUNT / LOCAL_GROUP_SIZE;
    localparam integer SUPERGROUP_COUNT = 2;
    localparam integer GROUPS_PER_SUPERGROUP =
        GROUP_COUNT / SUPERGROUP_COUNT;

    logic signed [ACC_BITS-1:0] stage1_next [0:O_DIM*GROUP_COUNT-1];
    logic signed [ACC_BITS-1:0] stage1_q [0:O_DIM*GROUP_COUNT-1];
    logic signed [ACC_BITS-1:0] stage2_next [0:O_DIM*SUPERGROUP_COUNT-1];
    logic signed [ACC_BITS-1:0] stage2_q [0:O_DIM*SUPERGROUP_COUNT-1];
    logic signed [ACC_BITS-1:0] partial_q [0:O_DIM-1];
    logic stage1_valid_q;
    logic stage2_valid_q;
    integer stage1_class_i;
    integer stage1_group_i;
    integer stage1_member_i;
    integer stage2_class_i;
    integer stage2_group_i;
    integer stage2_supergroup_i;
    integer register_i;

    always_comb begin
        for (stage1_class_i = 0; stage1_class_i < O_DIM;
             stage1_class_i = stage1_class_i + 1)
            for (stage1_group_i = 0; stage1_group_i < GROUP_COUNT;
                 stage1_group_i = stage1_group_i + 1) begin
                stage1_next[stage1_class_i*GROUP_COUNT + stage1_group_i] = '0;
                for (stage1_member_i = 0;
                     stage1_member_i < LOCAL_GROUP_SIZE;
                     stage1_member_i = stage1_member_i + 1)
                    if (i_spikes[
                        stage1_group_i*LOCAL_GROUP_SIZE + stage1_member_i
                    ])
                        stage1_next[
                            stage1_class_i*GROUP_COUNT + stage1_group_i
                        ] = stage1_next[
                            stage1_class_i*GROUP_COUNT + stage1_group_i
                        ] +
                            {{(ACC_BITS-W_BITS){
                                i_weight_flat[
                                    (stage1_group_i*LOCAL_GROUP_SIZE +
                                     stage1_member_i)*O_DIM*W_BITS +
                                    stage1_class_i*W_BITS + W_BITS-1
                                ]
                            }},
                            i_weight_flat[
                                (stage1_group_i*LOCAL_GROUP_SIZE +
                                 stage1_member_i)*O_DIM*W_BITS +
                                stage1_class_i*W_BITS +: W_BITS
                            ]};
            end
    end

    always_comb begin
        for (stage2_class_i = 0; stage2_class_i < O_DIM;
             stage2_class_i = stage2_class_i + 1)
            for (stage2_supergroup_i = 0;
                 stage2_supergroup_i < SUPERGROUP_COUNT;
                 stage2_supergroup_i = stage2_supergroup_i + 1) begin
                stage2_next[
                    stage2_class_i*SUPERGROUP_COUNT + stage2_supergroup_i
                ] = '0;
                for (stage2_group_i = 0;
                     stage2_group_i < GROUPS_PER_SUPERGROUP;
                     stage2_group_i = stage2_group_i + 1)
                    stage2_next[
                        stage2_class_i*SUPERGROUP_COUNT + stage2_supergroup_i
                    ] = stage2_next[
                        stage2_class_i*SUPERGROUP_COUNT + stage2_supergroup_i
                    ] +
                        stage1_q[
                            stage2_class_i*GROUP_COUNT +
                            stage2_supergroup_i*GROUPS_PER_SUPERGROUP +
                            stage2_group_i
                        ];
            end
    end

    genvar output_class;
    generate
        for (output_class = 0; output_class < O_DIM;
             output_class = output_class + 1) begin : gen_partial_output
            assign o_partial_flat[output_class*ACC_BITS +: ACC_BITS] =
                partial_q[output_class];
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid_q <= 1'b0;
            stage2_valid_q <= 1'b0;
            o_done <= 1'b0;
            for (register_i = 0; register_i < O_DIM*GROUP_COUNT;
                 register_i = register_i + 1)
                stage1_q[register_i] <= '0;
            for (register_i = 0; register_i < O_DIM*SUPERGROUP_COUNT;
                 register_i = register_i + 1)
                stage2_q[register_i] <= '0;
            for (register_i = 0; register_i < O_DIM;
                 register_i = register_i + 1)
                partial_q[register_i] <= '0;
        end else begin
            stage1_valid_q <= i_start;
            stage2_valid_q <= stage1_valid_q;
            o_done <= stage2_valid_q;
            if (i_start)
                for (register_i = 0; register_i < O_DIM*GROUP_COUNT;
                     register_i = register_i + 1)
                    stage1_q[register_i] <= stage1_next[register_i];
            if (stage1_valid_q)
                for (register_i = 0;
                     register_i < O_DIM*SUPERGROUP_COUNT;
                     register_i = register_i + 1)
                    stage2_q[register_i] <= stage2_next[register_i];
            if (stage2_valid_q)
                for (register_i = 0; register_i < O_DIM;
                     register_i = register_i + 1)
                    partial_q[register_i] <=
                        stage2_q[register_i*SUPERGROUP_COUNT] +
                        stage2_q[register_i*SUPERGROUP_COUNT + 1];
        end
    end

    initial begin
        if (H_COUNT != 40 || GROUP_COUNT != 8 ||
            GROUPS_PER_SUPERGROUP != 4)
            $error("FC_COMPAT runtime W2 requires H_COUNT=40");
        if (ACC_BITS < W_BITS)
            $error("FC_COMPAT W2 ACC_BITS must be at least W_BITS");
    end
endmodule

`default_nettype wire
