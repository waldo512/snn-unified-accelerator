`timescale 1ns/1ps
`default_nettype none

// Converts a topology descriptor into the common H40 state/trace workset
// geometry. FC/SRNN units are forty-neuron blocks. SCNN units are output
// channels whose spatial state is processed in forty-position transactions.
module unified_h40_topology_geometry_v1 #(
    parameter int ARRAY_TILES = 16,
    parameter int MEMORY_ROWS_PER_HALF = 512
) (
    input  wire logic [1:0]  i_mode,
    input  wire logic [15:0] i_primary_extent,
    input  wire logic [15:0] i_secondary_extent,

    output logic [9:0]       o_workset_units,
    output logic [6:0]       o_rows_per_unit,
    output logic [6:0]       o_transactions_per_unit,
    output logic [39:0]      o_tail_lane_mask,
    output logic [15:0]      o_active_tile_mask,
    output logic [31:0]      o_state_records,
    output logic [15:0]      o_rows_on_busiest_tile,
    output logic             o_legal
);
    localparam logic [1:0] MODE_FC = 2'd0;
    localparam logic [1:0] MODE_SRNN = 2'd1;
    localparam logic [1:0] MODE_SCNN = 2'd2;

    integer unsigned units_calc;
    integer unsigned rows_calc;
    integer unsigned transactions_calc;
    integer unsigned tail_calc;
    integer unsigned units_on_busiest_tile_calc;
    integer unsigned rows_on_busiest_tile_calc;
    integer unsigned state_records_calc;

    always_comb begin
        units_calc = 0;
        rows_calc = 0;
        transactions_calc = 0;
        tail_calc = 0;

        if (i_mode == MODE_FC || i_mode == MODE_SRNN) begin
            if (i_primary_extent != 0) begin
                units_calc = (integer'(i_primary_extent) + 39) / 40;
                rows_calc = 4;
                transactions_calc = 1;
                tail_calc = integer'(i_primary_extent) -
                    (units_calc - 1) * 40;
            end
        end else if (i_mode == MODE_SCNN) begin
            if (i_primary_extent != 0 && i_secondary_extent != 0) begin
                units_calc = integer'(i_primary_extent);
                transactions_calc =
                    (integer'(i_secondary_extent) + 39) / 40;
                // One SCNN H40 transaction packs its forty INT16 states in
                // two 320-bit rows. Keep every transaction row-aligned.
                rows_calc = transactions_calc * 2;
                tail_calc = integer'(i_secondary_extent) -
                    (transactions_calc - 1) * 40;
            end
        end

        units_on_busiest_tile_calc = units_calc == 0 ? 0 :
            (units_calc + ARRAY_TILES - 1) / ARRAY_TILES;
        rows_on_busiest_tile_calc =
            units_on_busiest_tile_calc * rows_calc;
        state_records_calc = units_calc * rows_calc;

        o_workset_units = 10'(units_calc);
        o_rows_per_unit = 7'(rows_calc);
        o_transactions_per_unit = 7'(transactions_calc);
        o_state_records = 32'(state_records_calc);
        o_rows_on_busiest_tile = 16'(rows_on_busiest_tile_calc);
        o_active_tile_mask = '0;
        o_tail_lane_mask = '0;

        for (int tile = 0; tile < ARRAY_TILES; tile++) begin
            if (tile < units_calc)
                o_active_tile_mask[tile] = 1'b1;
        end
        for (int lane = 0; lane < 40; lane++) begin
            if (lane < tail_calc)
                o_tail_lane_mask[lane] = 1'b1;
        end

        o_legal = units_calc != 0 && rows_calc != 0 &&
            transactions_calc != 0 && tail_calc >= 1 && tail_calc <= 40 &&
            units_calc <= 1023 && rows_calc <= 127 &&
            rows_on_busiest_tile_calc <= MEMORY_ROWS_PER_HALF;
    end

    initial begin
        if (ARRAY_TILES < 4 || ARRAY_TILES > 16 ||
            (ARRAY_TILES % 4) != 0 || MEMORY_ROWS_PER_HALF != 512)
            $error("H40 topology geometry requires 4, 8, 12 or 16 H40 tiles");
    end
endmodule

`default_nettype wire
