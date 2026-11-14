module unified_q8_8_k8_accumulator #(
    parameter int LANES = 8,
    parameter int ACC_WIDTH = 48
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_valid,
    output logic                         o_ready,
    input  logic [LANES*16-1:0]          i_a_flat,
    input  logic [LANES*16-1:0]          i_b_flat,
    input  logic [LANES-1:0]             i_lane_mask,
    input  logic                         i_first,
    input  logic                         i_last,
    output logic                         o_valid,
    input  logic                         i_ready,
    output logic signed [ACC_WIDTH-1:0]  o_accumulator
);

    logic dot_input_ready;
    logic dot_output_valid;
    logic dot_output_ready;
    logic signed [35:0] dot_sum;
    logic chunk_first_q;
    logic chunk_last_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [ACC_WIDTH-1:0] result_q;
    logic result_valid_q;
    logic signed [ACC_WIDTH-1:0] dot_sum_extended;
    logic signed [ACC_WIDTH-1:0] next_accumulator;

    assign o_ready = dot_input_ready;
    assign o_valid = result_valid_q;
    assign o_accumulator = result_q;
    assign dot_sum_extended = {{(ACC_WIDTH-36){dot_sum[35]}}, dot_sum};
    assign next_accumulator =
        (chunk_first_q ? {ACC_WIDTH{1'b0}} : accumulator_q) + dot_sum_extended;
    assign dot_output_ready = !chunk_last_q || !result_valid_q || i_ready;

    unified_q8_8_k8_dot_multipass #(
        .LANES(LANES)
    ) u_dot (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_valid(i_valid),
        .o_ready(dot_input_ready),
        .i_a_flat(i_a_flat),
        .i_b_flat(i_b_flat),
        .i_lane_mask(i_lane_mask),
        .o_valid(dot_output_valid),
        .i_ready(dot_output_ready),
        .o_sum(dot_sum)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            chunk_first_q <= 1'b0;
            chunk_last_q <= 1'b0;
            accumulator_q <= {ACC_WIDTH{1'b0}};
            result_q <= {ACC_WIDTH{1'b0}};
            result_valid_q <= 1'b0;
        end else begin
            if (result_valid_q && i_ready) begin
                result_valid_q <= 1'b0;
            end

            if (i_valid && dot_input_ready) begin
                chunk_first_q <= i_first;
                chunk_last_q <= i_last;
            end

            if (dot_output_valid && dot_output_ready) begin
                if (chunk_last_q) begin
                    result_q <= next_accumulator;
                    result_valid_q <= 1'b1;
                    accumulator_q <= {ACC_WIDTH{1'b0}};
                end else begin
                    accumulator_q <= next_accumulator;
                end
            end
        end
    end

endmodule
