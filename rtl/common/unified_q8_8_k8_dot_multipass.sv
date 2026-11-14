module unified_q8_8_k8_dot_multipass #(
    parameter int LANES = 8
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_valid,
    output logic                         o_ready,
    input  logic [LANES*16-1:0]          i_a_flat,
    input  logic [LANES*16-1:0]          i_b_flat,
    input  logic [LANES-1:0]             i_lane_mask,
    output logic                         o_valid,
    input  logic                         i_ready,
    output logic signed [35:0]           o_sum
);

    logic [LANES-1:0] lane_input_ready;
    logic [LANES-1:0] lane_output_valid;
    logic [LANES*32-1:0] lane_products_flat;
    logic launch;
    logic retire;
    logic signed [35:0] sum_comb;

    assign o_ready = &lane_input_ready;
    assign launch = i_valid && o_ready;
    assign o_valid = &lane_output_valid;
    assign retire = i_ready && o_valid;
    assign o_sum = sum_comb;

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin : g_lane
            logic signed [15:0] lane_a;
            logic signed [15:0] lane_b;
            logic signed [31:0] lane_product;

            assign lane_a = i_lane_mask[lane] ?
                $signed(i_a_flat[lane*16 +: 16]) : 16'sd0;
            assign lane_b = i_lane_mask[lane] ?
                $signed(i_b_flat[lane*16 +: 16]) : 16'sd0;
            assign lane_products_flat[lane*32 +: 32] = lane_product;

            unified_q8_8_mul_multipass u_mul (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_valid(launch),
                .o_ready(lane_input_ready[lane]),
                .i_a(lane_a),
                .i_b(lane_b),
                .o_valid(lane_output_valid[lane]),
                .i_ready(retire),
                .o_product(lane_product)
            );
        end
    endgenerate

    integer sum_lane;
    always_comb begin
        sum_comb = 36'sd0;
        for (sum_lane = 0; sum_lane < LANES; sum_lane++) begin
            sum_comb = sum_comb +
                $signed({{4{lane_products_flat[sum_lane*32 + 31]}},
                         lane_products_flat[sum_lane*32 +: 32]});
        end
    end

endmodule
