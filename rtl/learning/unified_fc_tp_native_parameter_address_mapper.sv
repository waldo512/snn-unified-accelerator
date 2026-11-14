`timescale 1ns/1ps
`default_nettype none

// Translate the topology-neutral FC TP parameter address into the native
// H40/K8 resident-memory coordinates. All divisors are elaboration-time
// constants; no runtime-configurable divider is required.
// HIDDEN_COUNT is the physical W0 stride, independent of the active tile mask.
module unified_fc_tp_native_parameter_address_mapper #(
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int FEATURE_COUNT = 2312,
    parameter int HIDDEN_COUNT = 400,
    parameter int HIDDEN_PER_TILE = 40,
    parameter int FEATURE_BANKS = 8,
    parameter int BANK_HASH_MODE = 1,
    parameter int OUTPUT_COUNT = 10,
    parameter logic [TENSOR_ID_WIDTH-1:0] W0_TENSOR_ID = 8'h10,
    parameter logic [TENSOR_ID_WIDTH-1:0] W2_TENSOR_ID = 8'h11,
    parameter int FEATURE_INDEX_WIDTH =
        (FEATURE_COUNT <= 2) ? 1 : $clog2(FEATURE_COUNT),
    parameter int HIDDEN_INDEX_WIDTH =
        (HIDDEN_COUNT <= 2) ? 1 : $clog2(HIDDEN_COUNT),
    parameter int TILE_INDEX_WIDTH =
        ((HIDDEN_COUNT / HIDDEN_PER_TILE) <= 2) ? 1 :
        $clog2(HIDDEN_COUNT / HIDDEN_PER_TILE),
    parameter int TILE_LANE_WIDTH =
        (HIDDEN_PER_TILE <= 2) ? 1 : $clog2(HIDDEN_PER_TILE),
    parameter int FEATURE_BANK_WIDTH =
        (FEATURE_BANKS <= 2) ? 1 : $clog2(FEATURE_BANKS),
    parameter int FEATURE_ROW_WIDTH =
        (((FEATURE_COUNT + FEATURE_BANKS - 1) / FEATURE_BANKS) <= 2) ? 1 :
        $clog2((FEATURE_COUNT + FEATURE_BANKS - 1) / FEATURE_BANKS),
    parameter int OUTPUT_INDEX_WIDTH =
        (OUTPUT_COUNT <= 2) ? 1 : $clog2(OUTPUT_COUNT)
) (
    input  logic                                  i_valid,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_local_parameter_addr,

    output logic                                  o_valid,
    output logic                                  o_error,
    output logic                                  o_is_w0,
    output logic                                  o_is_w2,
    output logic [FEATURE_INDEX_WIDTH-1:0]        o_feature_index,
    output logic [HIDDEN_INDEX_WIDTH-1:0]         o_hidden_index,
    output logic [TILE_INDEX_WIDTH-1:0]           o_tile_index,
    output logic [TILE_LANE_WIDTH-1:0]            o_tile_lane,
    output logic [FEATURE_BANK_WIDTH-1:0]         o_feature_bank,
    output logic [FEATURE_ROW_WIDTH-1:0]          o_feature_bank_row,
    output logic [OUTPUT_INDEX_WIDTH-1:0]         o_output_index
);
    localparam int TILE_COUNT = HIDDEN_COUNT / HIDDEN_PER_TILE;
    localparam int FEATURE_ROWS =
        (FEATURE_COUNT + FEATURE_BANKS - 1) / FEATURE_BANKS;
    localparam logic [PARAM_ADDR_WIDTH:0] W0_ELEMENTS =
        (PARAM_ADDR_WIDTH + 1)'(FEATURE_COUNT * HIDDEN_COUNT);
    localparam logic [PARAM_ADDR_WIDTH:0] W2_ELEMENTS =
        (PARAM_ADDR_WIDTH + 1)'(HIDDEN_COUNT * OUTPUT_COUNT);

    logic [PARAM_ADDR_WIDTH:0] address_ext;
    logic [PARAM_ADDR_WIDTH-1:0] feature_value;
    logic [PARAM_ADDR_WIDTH-1:0] hidden_value;

    always_comb begin
        address_ext = {1'b0, i_local_parameter_addr};
        feature_value = '0;
        hidden_value = '0;

        o_valid = i_valid;
        o_error = 1'b0;
        o_is_w0 = 1'b0;
        o_is_w2 = 1'b0;
        o_feature_index = '0;
        o_hidden_index = '0;
        o_tile_index = '0;
        o_tile_lane = '0;
        o_feature_bank = '0;
        o_feature_bank_row = '0;
        o_output_index = '0;

        if (i_valid && i_tensor_id == W0_TENSOR_ID) begin
            o_is_w0 = 1'b1;
            if (address_ext >= W0_ELEMENTS) begin
                o_error = 1'b1;
            end else begin
                feature_value = i_local_parameter_addr /
                    PARAM_ADDR_WIDTH'(HIDDEN_COUNT);
                hidden_value = i_local_parameter_addr %
                    PARAM_ADDR_WIDTH'(HIDDEN_COUNT);
                o_feature_index = FEATURE_INDEX_WIDTH'(feature_value);
                o_hidden_index = HIDDEN_INDEX_WIDTH'(hidden_value);
                o_tile_index = TILE_INDEX_WIDTH'(
                    hidden_value / PARAM_ADDR_WIDTH'(HIDDEN_PER_TILE));
                o_tile_lane = TILE_LANE_WIDTH'(
                    hidden_value % PARAM_ADDR_WIDTH'(HIDDEN_PER_TILE));
                o_feature_bank = FEATURE_BANK_WIDTH'(
                    feature_value % PARAM_ADDR_WIDTH'(FEATURE_BANKS));
                // Match canonical K8 mode 1: fi[2:0] ^ {fi[6],fi[8],fi[7]}.
                // Shifts also zero-fill absent high bits in reduced configurations.
                if (BANK_HASH_MODE == 1)
                    o_feature_bank = o_feature_bank ^ FEATURE_BANK_WIDTH'({
                        1'(feature_value >> 6), 1'(feature_value >> 8),
                        1'(feature_value >> 7)});
                o_feature_bank_row = FEATURE_ROW_WIDTH'(
                    feature_value / PARAM_ADDR_WIDTH'(FEATURE_BANKS));
            end
        end else if (i_valid && i_tensor_id == W2_TENSOR_ID) begin
            o_is_w2 = 1'b1;
            if (address_ext >= W2_ELEMENTS) begin
                o_error = 1'b1;
            end else begin
                hidden_value = i_local_parameter_addr /
                    PARAM_ADDR_WIDTH'(OUTPUT_COUNT);
                o_hidden_index = HIDDEN_INDEX_WIDTH'(hidden_value);
                o_tile_index = TILE_INDEX_WIDTH'(
                    hidden_value / PARAM_ADDR_WIDTH'(HIDDEN_PER_TILE));
                o_tile_lane = TILE_LANE_WIDTH'(
                    hidden_value % PARAM_ADDR_WIDTH'(HIDDEN_PER_TILE));
                o_output_index = OUTPUT_INDEX_WIDTH'(
                    i_local_parameter_addr %
                    PARAM_ADDR_WIDTH'(OUTPUT_COUNT));
            end
        end else if (i_valid) begin
            o_error = 1'b1;
        end
    end

    initial begin
        if (FEATURE_COUNT < 1 || HIDDEN_COUNT < 1 ||
            HIDDEN_PER_TILE < 1 || FEATURE_BANKS < 1 || OUTPUT_COUNT < 1)
            $error("FC native parameter dimensions must be positive");
        if ((HIDDEN_COUNT % HIDDEN_PER_TILE) != 0)
            $error("FC hidden count must contain complete physical tiles");
        if (TILE_COUNT < 1 || FEATURE_ROWS < 1)
            $error("FC native parameter coordinate widths are invalid");
        if (W0_TENSOR_ID == W2_TENSOR_ID)
            $error("FC W0 and W2 tensor IDs must be distinct");
        if (BANK_HASH_MODE < 0 || BANK_HASH_MODE > 1 ||
            (BANK_HASH_MODE == 1 && FEATURE_BANKS != 8))
            $error("FC native bank hash requires mode 0 or K8 mode 1");
    end
endmodule

`default_nettype wire
