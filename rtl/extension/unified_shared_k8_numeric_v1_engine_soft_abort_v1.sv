`default_nettype none

module unified_shared_k8_numeric_v1_engine_soft_abort_v1 #(
    parameter int LANES = 8,
    parameter int ACC_WIDTH = 48
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_abort,
    input  logic                         i_valid,
    output logic                         o_ready,
    input  logic                         i_mode_q8_8,
    input  logic [LANES-1:0]             i_lane_mask,
    input  logic [LANES*16-1:0]          i_q8_8_feature_flat,
    input  logic [LANES*16-1:0]          i_q8_8_weight_flat,
    input  logic [LANES*8-1:0]           i_int8_lhs_flat,
    input  logic [LANES*8-1:0]           i_int8_rhs_flat,
    input  logic                         i_first,
    input  logic                         i_last,
    output logic                         o_valid,
    input  logic                         i_ready,
    output logic                         o_result_mode_q8_8,
    output logic signed [ACC_WIDTH-1:0]  o_q8_8_accumulator,
    output logic [LANES*32-1:0]          o_int8_contribution_flat,
    output logic                         o_busy
);
    logic [LANES-1:0] lane_ready;
    logic [LANES-1:0] lane_valid;
    logic [LANES-1:0] lane_busy;
    logic [LANES*32-1:0] lane_q8_8_product_flat;
    logic [LANES*32-1:0] lane_int8_contribution_flat;
    logic metadata_pending_q;
    logic mode_q8_8_q;
    logic first_q;
    logic last_q;
    logic [1:0] result_count_q;
    logic result_read_ptr_q;
    logic result_write_ptr_q;
    logic [1:0] result_mode_q8_8_q;
    logic signed [ACC_WIDTH-1:0] q8_8_accumulator_q;
    logic signed [ACC_WIDTH-1:0] q8_8_result_q [0:1];
    logic [LANES*32-1:0] int8_result_flat_q [0:1];
    logic launch;
    logic all_lane_valid;
    logic consume_lane_result;
    logic result_push;
    logic result_pop;
    logic result_fifo_full;
    logic signed [ACC_WIDTH-1:0] q8_8_lane_sum;
    logic signed [ACC_WIDTH-1:0] q8_8_next_accumulator;

    assign all_lane_valid = &lane_valid;
    assign result_fifo_full = result_count_q == 2'd2;
    assign consume_lane_result = metadata_pending_q && all_lane_valid &&
        (!(mode_q8_8_q && !last_q) ? !result_fifo_full : 1'b1);
    assign result_push = consume_lane_result &&
        (!mode_q8_8_q || last_q);
    assign result_pop = (result_count_q != 2'd0) && i_ready;
    assign o_ready = !metadata_pending_q && (&lane_ready) &&
        !result_fifo_full;
    assign launch = i_valid && o_ready;
    assign o_valid = result_count_q != 2'd0;
    assign o_result_mode_q8_8 = result_mode_q8_8_q[result_read_ptr_q];
    assign o_q8_8_accumulator = q8_8_result_q[result_read_ptr_q];
    assign o_int8_contribution_flat =
        int8_result_flat_q[result_read_ptr_q];
    assign o_busy = metadata_pending_q || (result_count_q != 2'd0) ||
        (|lane_busy);
    assign q8_8_next_accumulator =
        (first_q ? {ACC_WIDTH{1'b0}} : q8_8_accumulator_q) +
        q8_8_lane_sum;

    always_comb begin
        q8_8_lane_sum = {ACC_WIDTH{1'b0}};
        for (int lane_index = 0; lane_index < LANES; lane_index++) begin
            q8_8_lane_sum = q8_8_lane_sum +
                {{(ACC_WIDTH-32){lane_q8_8_product_flat[
                    lane_index*32 + 31]}},
                  lane_q8_8_product_flat[lane_index*32 +: 32]};
        end
    end

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin : g_lane
            unified_shared_k8_numeric_v1_lane_soft_abort_v1 u_lane (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_abort(i_abort),
                .i_valid(launch),
                .o_ready(lane_ready[lane]),
                .i_mode_q8_8(i_mode_q8_8),
                .i_lane_enable(i_lane_mask[lane]),
                .i_q8_8_lhs($signed(i_q8_8_feature_flat[
                    lane*16 +: 16])),
                .i_q8_8_rhs($signed(i_q8_8_weight_flat[
                    lane*16 +: 16])),
                .i_int8_lhs($signed(i_int8_lhs_flat[lane*8 +: 8])),
                .i_int8_rhs($signed(i_int8_rhs_flat[lane*8 +: 8])),
                .o_valid(lane_valid[lane]),
                .i_ready(consume_lane_result),
                .o_q8_8_product(lane_q8_8_product_flat[
                    lane*32 +: 32]),
                .o_int8_contribution(lane_int8_contribution_flat[
                    lane*32 +: 32]),
                .o_busy(lane_busy[lane])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            metadata_pending_q <= 1'b0;
            mode_q8_8_q <= 1'b0;
            first_q <= 1'b0;
            last_q <= 1'b0;
            result_count_q <= 2'd0;
            result_read_ptr_q <= 1'b0;
            result_write_ptr_q <= 1'b0;
            q8_8_accumulator_q <= {ACC_WIDTH{1'b0}};
        end else if (i_abort) begin
            metadata_pending_q <= 1'b0;
            mode_q8_8_q <= 1'b0;
            first_q <= 1'b0;
            last_q <= 1'b0;
            result_count_q <= 2'd0;
            result_read_ptr_q <= 1'b0;
            result_write_ptr_q <= 1'b0;
            q8_8_accumulator_q <= {ACC_WIDTH{1'b0}};
        end else begin
            unique case ({result_push, result_pop})
                2'b10: begin
                    result_count_q <= result_count_q + 2'd1;
                    result_write_ptr_q <= ~result_write_ptr_q;
                end
                2'b01: begin
                    result_count_q <= result_count_q - 2'd1;
                    result_read_ptr_q <= ~result_read_ptr_q;
                end
                2'b11: begin
                    result_read_ptr_q <= ~result_read_ptr_q;
                    result_write_ptr_q <= ~result_write_ptr_q;
                end
                default: begin
                end
            endcase

            if (launch) begin
                metadata_pending_q <= 1'b1;
                mode_q8_8_q <= i_mode_q8_8;
                first_q <= i_first;
                last_q <= i_last;
            end

            if (consume_lane_result) begin
                metadata_pending_q <= 1'b0;
                if (mode_q8_8_q) begin
                    if (last_q) begin
                        q8_8_accumulator_q <= {ACC_WIDTH{1'b0}};
                    end else begin
                        q8_8_accumulator_q <= q8_8_next_accumulator;
                    end
                end else begin
                end
            end
        end
    end

    // Payload storage is reset-free. result_count_q is the sole validity
    // authority, so downstream backpressure cannot reach lane retirement or
    // input clock enables through a ready chain.
    always_ff @(posedge i_clk) begin
        if (result_push) begin
            result_mode_q8_8_q[result_write_ptr_q] <= mode_q8_8_q;
            if (mode_q8_8_q) begin
                q8_8_result_q[result_write_ptr_q] <= q8_8_next_accumulator;
                int8_result_flat_q[result_write_ptr_q] <= '0;
            end else begin
                q8_8_result_q[result_write_ptr_q] <= {ACC_WIDTH{1'b0}};
                int8_result_flat_q[result_write_ptr_q] <=
                    lane_int8_contribution_flat;
            end
        end
    end
endmodule

`default_nettype wire
