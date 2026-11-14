`timescale 1ns/1ps
`default_nettype none

// SCNN P2 path through the proposed shared storage and DSP tile. Two binary
// feature words select two kernel weights inside each DSP48E1 lane.
module unified_dual_role_scnn_p2_pipeline_candidate_v1 #(
    parameter int LANES = 40,
    parameter int FEATURE_WORDS_PER_SLOT = 1792,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORDS_PER_SLOT),
    parameter int WEIGHT_BANK_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_BANK_DEPTH)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,

    input  wire logic                         i_feature_write_valid,
    output logic                              o_feature_write_ready,
    input  wire logic                         i_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0] i_feature_write_addr,
    input  wire logic [LANES-1:0]             i_feature_write_data,

    input  wire logic                         i_weight_fill_valid,
    output logic                              o_weight_fill_ready,
    input  wire logic                         i_weight_fill_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_weight_fill_addr,
    input  wire logic [199:0]                 i_weight_fill_data,

    input  wire logic                         i_issue_valid,
    output logic                              o_issue_ready,
    input  wire logic                         i_first,
    input  wire logic                         i_last,
    input  wire logic [LANES-1:0]             i_lane_mask,
    input  wire logic                         i_feature_read_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0] i_feature_read_addr0,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0] i_feature_read_addr1,
    input  wire logic                         i_weight_active_bank,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_weight_read_addr0,
    input  wire logic [WEIGHT_ADDR_WIDTH-1:0] i_weight_read_addr1,
    input  wire logic [3:0]                   i_weight_subword0,
    input  wire logic [3:0]                   i_weight_subword1,

    output logic                              o_result_valid,
    input  wire logic                         i_result_ready,
    output logic [LANES-1:0]                  o_result_lane_mask,
    output logic [LANES*48-1:0]               o_result_flat
);
    logic feature_request_ready;
    logic weight_request_ready;
    logic feature_response_valid;
    logic weight_response_valid;
    logic feature_response_ready;
    logic weight_response_ready;
    logic [LANES-1:0] feature_data0;
    logic [LANES-1:0] feature_data1;
    logic [LANES*25-1:0] weight_term0;
    logic [LANES*25-1:0] weight_term1;
    logic [LANES*18-1:0] unit_factor_flat;
    logic issue_fire;
    logic tile_input_valid;
    logic tile_input_ready;
    logic tile_input_fire;
    logic metadata_valid_q;
    logic first_q;
    logic last_q;
    logic [LANES-1:0] lane_mask_q;

    assign o_issue_ready = !i_abort && feature_request_ready &&
                           weight_request_ready;
    assign issue_fire = i_issue_valid && o_issue_ready;
    assign tile_input_valid = !i_abort && metadata_valid_q &&
                              feature_response_valid &&
                              weight_response_valid;
    assign tile_input_fire = tile_input_valid && tile_input_ready;

    // A cleared metadata token marks any still-returning RAM response stale.
    assign feature_response_ready = i_abort || !metadata_valid_q ||
                                    (tile_input_ready && weight_response_valid);
    assign weight_response_ready = i_abort || !metadata_valid_q ||
                                   (tile_input_ready && feature_response_valid);

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_unit_factor
            assign unit_factor_flat[lane*18 +: 18] = 18'sd1;
        end
    endgenerate

    unified_dual_role_binary_feature_tdp_candidate_v1 #(
        .LANES(LANES),
        .WORDS_PER_SLOT(FEATURE_WORDS_PER_SLOT),
        .ADDR_WIDTH(FEATURE_ADDR_WIDTH)
    ) u_feature_store (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_write_valid(i_feature_write_valid),
        .o_write_ready(o_feature_write_ready),
        .i_write_slot(i_feature_write_slot),
        .i_write_addr(i_feature_write_addr),
        .i_write_data(i_feature_write_data),
        .i_read_valid(issue_fire),
        .o_read_ready(feature_request_ready),
        .i_read_slot(i_feature_read_slot),
        .i_read_addr0(i_feature_read_addr0),
        .i_read_addr1(i_feature_read_addr1),
        .o_read_result_valid(feature_response_valid),
        .i_read_result_ready(feature_response_ready),
        .o_read_data0(feature_data0),
        .o_read_data1(feature_data1)
    );

    unified_dual_role_weight_store_candidate_v1 #(
        .LANES(LANES),
        .BANK_WIDTH(200),
        .BANK_DEPTH(WEIGHT_BANK_DEPTH),
        .ADDR_WIDTH(WEIGHT_ADDR_WIDTH)
    ) u_weight_store (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_fill_valid(i_weight_fill_valid),
        .o_fill_ready(o_weight_fill_ready),
        .i_fill_bank(i_weight_fill_bank),
        .i_fill_addr(i_weight_fill_addr),
        .i_fill_data(i_weight_fill_data),
        .i_read_valid(issue_fire),
        .o_read_ready(weight_request_ready),
        .i_mode(2'd2),
        .i_scnn_active_bank(i_weight_active_bank),
        .i_read_addr0(i_weight_read_addr0),
        .i_read_addr1(i_weight_read_addr1),
        .i_read_subword0(i_weight_subword0),
        .i_read_subword1(i_weight_subword1),
        .o_read_result_valid(weight_response_valid),
        .i_read_result_ready(weight_response_ready),
        .o_term0_flat(weight_term0),
        .o_term1_flat(weight_term1)
    );

    unified_dual_role_h40_tile_candidate_v3 #(
        .LANES(LANES)
    ) u_h40_tile (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_valid(tile_input_valid),
        .o_ready(tile_input_ready),
        .i_first(first_q),
        .i_last(last_q),
        .i_lane_mask(lane_mask_q),
        .i_enable_a(feature_data0),
        .i_enable_d(feature_data1),
        .i_term_a_flat(weight_term0),
        .i_term_d_flat(weight_term1),
        .i_factor_b_flat(unit_factor_flat),
        .o_result_valid(o_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_lane_mask(o_result_lane_mask),
        .o_result_flat(o_result_flat)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            metadata_valid_q <= 1'b0;
            first_q <= 1'b0;
            last_q <= 1'b0;
            lane_mask_q <= '0;
        end else if (i_abort) begin
            metadata_valid_q <= 1'b0;
            first_q <= 1'b0;
            last_q <= 1'b0;
            lane_mask_q <= '0;
        end else begin
            if (tile_input_fire)
                metadata_valid_q <= 1'b0;
            if (issue_fire) begin
                metadata_valid_q <= 1'b1;
                first_q <= i_first;
                last_q <= i_last;
                lane_mask_q <= i_lane_mask;
            end
        end
    end

    initial begin
        if (LANES != 40)
            $error("SCNN P2 candidate requires exactly 40 lanes");
    end

endmodule

`default_nettype wire
