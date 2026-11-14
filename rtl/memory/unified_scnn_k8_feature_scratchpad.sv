`default_nettype none

module unified_scnn_k8_feature_scratchpad #(
    parameter int BANK_DEPTH = 9216,
    parameter int ADDR_WIDTH = $clog2(BANK_DEPTH)
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_write_valid,
    input  logic                         i_write_slot,
    input  logic [3:0]                   i_write_bank,
    input  logic [ADDR_WIDTH-1:0]        i_write_addr,
    input  logic [15:0]                  i_write_data,
    output logic                         o_write_ready,

    input  logic                         i_read_valid,
    output logic                         o_read_ready,
    input  logic                         i_read_slot,
    input  logic [31:0]                  i_read_bank_flat,
    input  logic [8*ADDR_WIDTH-1:0]      i_read_addr_flat,
    input  logic [7:0]                   i_read_lane_mask,
    output logic                         o_read_valid,
    input  logic                         i_read_result_ready,
    output logic [127:0]                 o_read_data_flat,
    output logic [7:0]                   o_read_lane_mask,

    output logic                         o_bank_conflict_error,
    output logic                         o_bounds_error
);

    logic write_enable [0:1][0:8];
    logic read_enable [0:1][0:8];
    logic [ADDR_WIDTH-1:0] read_address [0:1][0:8];
    logic [15:0] read_data [0:1][0:8];
    logic result_valid_q;
    logic result_slot_q;
    logic [31:0] result_bank_flat_q;
    logic [7:0] result_lane_mask_q;
    logic read_contract_error;
    integer slot_index;
    integer bank_index;

    assign o_write_ready = i_write_bank < 9 &&
        {1'b0, i_write_addr} < (ADDR_WIDTH+1)'(BANK_DEPTH);
    assign o_read_ready = (!result_valid_q || i_read_result_ready) &&
                          !read_contract_error;
    assign o_read_valid = result_valid_q;
    assign o_read_lane_mask = result_lane_mask_q;

    always_comb begin
        read_contract_error = 1'b0;
        o_read_data_flat = '0;
        for (slot_index = 0; slot_index < 2; slot_index = slot_index + 1) begin
            for (bank_index = 0; bank_index < 9; bank_index = bank_index + 1) begin
                write_enable[slot_index][bank_index] = i_write_valid &&
                    o_write_ready && i_write_slot == slot_index[0] &&
                    i_write_bank == 4'(bank_index);
                read_enable[slot_index][bank_index] = 1'b0;
                read_address[slot_index][bank_index] = '0;
            end
        end
        for (int lane = 0; lane < 8; lane = lane + 1) begin
            if (i_read_lane_mask[lane]) begin
                if (i_read_bank_flat[lane*4 +: 4] >= 9 ||
                    {1'b0, i_read_addr_flat[
                        lane*ADDR_WIDTH +: ADDR_WIDTH]} >=
                        (ADDR_WIDTH+1)'(BANK_DEPTH)) begin
                    read_contract_error = 1'b1;
                end else begin
                    read_enable[i_read_slot][
                        i_read_bank_flat[lane*4 +: 4]] =
                        i_read_valid && o_read_ready;
                    read_address[i_read_slot][
                        i_read_bank_flat[lane*4 +: 4]] =
                        i_read_addr_flat[
                            lane*ADDR_WIDTH +: ADDR_WIDTH];
                end
                for (int prior = 0; prior < lane; prior = prior + 1) begin
                    if (i_read_lane_mask[prior] &&
                        i_read_bank_flat[prior*4 +: 4] ==
                        i_read_bank_flat[lane*4 +: 4])
                        read_contract_error = 1'b1;
                end
            end
            if (result_lane_mask_q[lane] &&
                result_bank_flat_q[lane*4 +: 4] < 9)
                o_read_data_flat[lane*16 +: 16] =
                    read_data[result_slot_q][
                        result_bank_flat_q[lane*4 +: 4]];
        end
    end

    generate
        for (genvar slot = 0; slot < 2; slot = slot + 1) begin : g_slot
            for (genvar bank = 0; bank < 9; bank = bank + 1) begin : g_bank
                unified_scnn_feature_bank_ram #(
                    .BANK_DEPTH(BANK_DEPTH), .ADDR_WIDTH(ADDR_WIDTH)
                ) u_bank (
                    .i_clk(i_clk),
                    .i_write_enable(write_enable[slot][bank]),
                    .i_write_addr(i_write_addr), .i_write_data(i_write_data),
                    .i_read_enable(read_enable[slot][bank]),
                    .i_read_addr(read_address[slot][bank]),
                    .o_read_data(read_data[slot][bank])
                );
            end
        end
    endgenerate

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            result_valid_q <= 1'b0;
            result_slot_q <= 1'b0;
            result_bank_flat_q <= '0;
            result_lane_mask_q <= '0;
            o_bank_conflict_error <= 1'b0;
            o_bounds_error <= 1'b0;
        end else begin
            if (result_valid_q && i_read_result_ready)
                result_valid_q <= 1'b0;
            if (i_read_valid && read_contract_error) begin
                for (int lane = 0; lane < 8; lane = lane + 1) begin
                    if (i_read_lane_mask[lane] &&
                        (i_read_bank_flat[lane*4 +: 4] >= 9 ||
                         {1'b0, i_read_addr_flat[
                             lane*ADDR_WIDTH +: ADDR_WIDTH]} >=
                             (ADDR_WIDTH+1)'(BANK_DEPTH)))
                        o_bounds_error <= 1'b1;
                    for (int prior = 0; prior < lane; prior = prior + 1) begin
                        if (i_read_lane_mask[lane] &&
                            i_read_lane_mask[prior] &&
                            i_read_bank_flat[lane*4 +: 4] ==
                            i_read_bank_flat[prior*4 +: 4])
                            o_bank_conflict_error <= 1'b1;
                    end
                end
            end
            if (i_write_valid && !o_write_ready)
                o_bounds_error <= 1'b1;
            if (i_read_valid && o_read_ready) begin
                result_valid_q <= 1'b1;
                result_slot_q <= i_read_slot;
                result_bank_flat_q <= i_read_bank_flat;
                result_lane_mask_q <= i_read_lane_mask;
            end
        end
    end

endmodule

`default_nettype wire
