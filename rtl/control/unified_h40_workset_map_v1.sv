`timescale 1ns/1ps
`default_nettype none

// Maps one logical topology unit and its local transaction onto the shared
// 16-tile state/trace RAM. Units are striped round-robin across tiles and are
// stored contiguously within each tile.
module unified_h40_workset_map_v1 #(
    parameter int ARRAY_TILES = 16,
    parameter int MEMORY_ROWS_PER_HALF = 512,
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_ROWS_PER_HALF)
) (
    input  wire logic [1:0]  i_mode,
    input  wire logic        i_geometry_legal,
    input  wire logic [9:0]  i_workset_units,
    input  wire logic [6:0]  i_rows_per_unit,
    input  wire logic [6:0]  i_transactions_per_unit,
    input  wire logic [39:0] i_tail_lane_mask,
    input  wire logic [9:0]  i_unit_index,
    input  wire logic [6:0]  i_transaction_index,

    output logic [3:0]       o_tile,
    output logic [WORK_ADDR_WIDTH-1:0] o_work_base_addr,
    output logic [39:0]      o_lane_mask,
    output logic             o_legal
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    integer unsigned unit_slot_calc;
    integer unsigned base_calc;
    integer unsigned final_addr_calc;
    logic mode_legal;
    logic shape_legal;
    logic index_legal;

    always_comb begin
        unit_slot_calc = integer'(i_unit_index) / ARRAY_TILES;
        base_calc = unit_slot_calc * integer'(i_rows_per_unit);
        if (i_mode == MODE_SCNN)
            base_calc = base_calc + integer'(i_transaction_index) * 2;

        final_addr_calc = base_calc +
            ((i_mode == MODE_SCNN) ? 1 : 3);
        mode_legal = i_mode == MODE_FC || i_mode == MODE_SRNN ||
            i_mode == MODE_SCNN;
        shape_legal = i_mode == MODE_SCNN ?
            (integer'(i_rows_per_unit) ==
             integer'(i_transactions_per_unit) * 2) :
            (i_rows_per_unit == 7'd4 &&
             i_transactions_per_unit == 7'd1);
        index_legal = i_unit_index < i_workset_units &&
            i_transaction_index < i_transactions_per_unit;
        if (i_mode != MODE_SCNN)
            index_legal = index_legal && i_transaction_index == 0;

        o_tile = i_unit_index[3:0];
        o_work_base_addr = WORK_ADDR_WIDTH'(base_calc);
        o_lane_mask = '1;
        if ((i_mode == MODE_SCNN &&
             i_transaction_index + 1'b1 == i_transactions_per_unit) ||
            (i_mode != MODE_SCNN &&
             i_unit_index + 1'b1 == i_workset_units))
            o_lane_mask = i_tail_lane_mask;

        o_legal = i_geometry_legal && mode_legal && shape_legal &&
            index_legal && final_addr_calc < MEMORY_ROWS_PER_HALF;
    end

    initial begin
        if (ARRAY_TILES != 16 || MEMORY_ROWS_PER_HALF != 512)
            $error("H40 workset mapper requires the selected 16xH40 array");
    end
endmodule

`default_nettype wire
