`timescale 1ns/1ps
`default_nettype none

// Bridge the FC TP engine's absolute parameter transactions to native H40
// tile stores. A read is tagged until its selected tile responds. A write is
// accepted when the tile accepts it, but quiescence remains low until the
// native whole-row update has physically completed.
module unified_fc_tp_native_parameter_router #(
    parameter int TILES = 4,
    parameter int FEATURE_COUNT = 2312,
    parameter int HIDDEN_PER_TILE = 40,
    parameter int FEATURE_BANKS = 8,
    parameter int BANK_HASH_MODE = 1,
    parameter int OUTPUT_COUNT = 10,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter logic [TENSOR_ID_WIDTH-1:0] W0_TENSOR_ID = 8'h10,
    parameter logic [TENSOR_ID_WIDTH-1:0] W2_TENSOR_ID = 8'h11,
    parameter int HIDDEN_COUNT = TILES * HIDDEN_PER_TILE,
    parameter int TILE_INDEX_WIDTH = (TILES <= 2) ? 1 : $clog2(TILES),
    parameter int TILE_LANE_WIDTH =
        (HIDDEN_PER_TILE <= 2) ? 1 : $clog2(HIDDEN_PER_TILE),
    parameter int FEATURE_ROW_WIDTH =
        (((FEATURE_COUNT + FEATURE_BANKS - 1) / FEATURE_BANKS) <= 2) ? 1 :
        $clog2((FEATURE_COUNT + FEATURE_BANKS - 1) / FEATURE_BANKS),
    parameter int OUTPUT_INDEX_WIDTH =
        (OUTPUT_COUNT <= 2) ? 1 : $clog2(OUTPUT_COUNT)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_w0_parameter_base,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_w2_parameter_base,

    input  logic                                      i_read_valid,
    output logic                                      o_read_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_read_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_read_parameter_addr,
    input  logic [TAG_WIDTH-1:0]                      i_read_tag,
    output logic                                      o_response_valid,
    input  logic                                      i_response_ready,
    output logic [TAG_WIDTH-1:0]                      o_response_tag,
    output logic signed [CODE_WIDTH-1:0]              o_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_response_residual,
    output logic                                      o_response_error,

    input  logic                                      i_write_valid,
    output logic                                      o_write_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_write_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_write_parameter_addr,
    input  logic signed [CODE_WIDTH-1:0]              i_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_write_residual,
    input  logic                                      i_write_code_enable,
    input  logic                                      i_write_residual_enable,

    output logic [TILES-1:0]                          o_tile_read_valid,
    input  logic [TILES-1:0]                          i_tile_read_ready,
    output logic                                      o_tile_read_is_w0,
    output logic [2:0]                                o_tile_read_feature_bank,
    output logic [FEATURE_ROW_WIDTH-1:0]              o_tile_read_feature_row,
    output logic [TILE_LANE_WIDTH-1:0]                o_tile_read_tile_lane,
    output logic [OUTPUT_INDEX_WIDTH-1:0]             o_tile_read_output,
    input  logic [TILES-1:0]                          i_tile_response_valid,
    output logic [TILES-1:0]                          o_tile_response_ready,
    input  logic signed [TILES*CODE_WIDTH-1:0]        i_tile_response_code,
    input  logic signed [TILES*RESIDUAL_WIDTH-1:0]
                                                       i_tile_response_residual,

    output logic [TILES-1:0]                          o_tile_write_valid,
    input  logic [TILES-1:0]                          i_tile_write_ready,
    output logic                                      o_tile_write_is_w0,
    output logic [2:0]                                o_tile_write_feature_bank,
    output logic [FEATURE_ROW_WIDTH-1:0]              o_tile_write_feature_row,
    output logic [TILE_LANE_WIDTH-1:0]                o_tile_write_tile_lane,
    output logic [OUTPUT_INDEX_WIDTH-1:0]             o_tile_write_output,
    output logic signed [CODE_WIDTH-1:0]              o_tile_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_tile_write_residual,
    output logic                                      o_tile_write_code_enable,
    output logic                                      o_tile_write_residual_enable,
    input  logic [TILES-1:0]                          i_tile_write_done,
    input  logic [TILES-1:0]                          i_tile_error,

    output logic                                      o_quiescent,
    output logic                                      o_protocol_error
);
    logic [PARAM_ADDR_WIDTH-1:0] read_local_addr;
    logic [PARAM_ADDR_WIDTH-1:0] write_local_addr;
    logic read_base_error;
    logic write_base_error;
    logic read_map_error;
    logic write_map_error;
    logic read_map_w0;
    logic write_map_w0;
    logic [TILE_INDEX_WIDTH-1:0] read_tile;
    logic [TILE_INDEX_WIDTH-1:0] write_tile;
    logic [TILE_LANE_WIDTH-1:0] read_lane;
    logic [TILE_LANE_WIDTH-1:0] write_lane;
    logic [2:0] read_bank;
    logic [2:0] write_bank;
    logic [FEATURE_ROW_WIDTH-1:0] read_row;
    logic [FEATURE_ROW_WIDTH-1:0] write_row;
    logic [OUTPUT_INDEX_WIDTH-1:0] read_output;
    logic [OUTPUT_INDEX_WIDTH-1:0] write_output;
    logic read_pending_q;
    logic read_error_q;
    logic [TILE_INDEX_WIDTH-1:0] read_tile_q;
    logic [TAG_WIDTH-1:0] read_tag_q;
    logic write_pending_q;
    logic [TILE_INDEX_WIDTH-1:0] write_tile_q;
    logic protocol_error_q;
    logic read_fire;
    logic write_fire;
    logic response_fire;

    always_comb begin
        read_local_addr = i_read_parameter_addr;
        write_local_addr = i_write_parameter_addr;
        read_base_error = 1'b0;
        write_base_error = 1'b0;
        if (i_read_tensor_id == W0_TENSOR_ID) begin
            read_base_error = i_read_parameter_addr < i_w0_parameter_base;
            read_local_addr = i_read_parameter_addr - i_w0_parameter_base;
        end else if (i_read_tensor_id == W2_TENSOR_ID) begin
            read_base_error = i_read_parameter_addr < i_w2_parameter_base;
            read_local_addr = i_read_parameter_addr - i_w2_parameter_base;
        end
        if (i_write_tensor_id == W0_TENSOR_ID) begin
            write_base_error = i_write_parameter_addr < i_w0_parameter_base;
            write_local_addr = i_write_parameter_addr - i_w0_parameter_base;
        end else if (i_write_tensor_id == W2_TENSOR_ID) begin
            write_base_error = i_write_parameter_addr < i_w2_parameter_base;
            write_local_addr = i_write_parameter_addr - i_w2_parameter_base;
        end
    end

    always_comb begin
        o_tile_read_valid = '0;
        o_read_ready = !read_pending_q && !write_pending_q &&
            !i_write_valid &&
            ((read_map_error || read_base_error) ||
             i_tile_read_ready[read_tile]);
        read_fire = i_read_valid && o_read_ready;
        if (i_read_valid && !i_write_valid &&
            !read_pending_q && !write_pending_q &&
            !read_map_error && !read_base_error)
            o_tile_read_valid[read_tile] = 1'b1;
    end

    always_comb begin
        o_tile_response_ready = '0;
        o_response_valid = read_pending_q &&
            (read_error_q || i_tile_response_valid[read_tile_q]);
        o_response_tag = read_tag_q;
        o_response_error = read_error_q;
        o_response_code = '0;
        o_response_residual = '0;
        if (read_pending_q && !read_error_q) begin
            o_response_code = $signed(i_tile_response_code[
                read_tile_q*CODE_WIDTH +: CODE_WIDTH]);
            o_response_residual = $signed(i_tile_response_residual[
                read_tile_q*RESIDUAL_WIDTH +: RESIDUAL_WIDTH]);
            o_tile_response_ready[read_tile_q] = i_response_ready;
        end
        response_fire = o_response_valid && i_response_ready;
    end

    always_comb begin
        o_tile_write_valid = '0;
        o_write_ready = !read_pending_q && !write_pending_q &&
            !i_read_valid &&
            ((write_map_error || write_base_error) ||
             i_tile_write_ready[write_tile]);
        write_fire = i_write_valid && o_write_ready;
        if (i_write_valid && !i_read_valid &&
            !read_pending_q && !write_pending_q &&
            !write_map_error && !write_base_error)
            o_tile_write_valid[write_tile] = 1'b1;
    end

    assign o_tile_read_is_w0 = read_map_w0;
    assign o_tile_read_feature_bank = read_bank;
    assign o_tile_read_feature_row = read_row;
    assign o_tile_read_tile_lane = read_lane;
    assign o_tile_read_output = read_output;
    assign o_tile_write_is_w0 = write_map_w0;
    assign o_tile_write_feature_bank = write_bank;
    assign o_tile_write_feature_row = write_row;
    assign o_tile_write_tile_lane = write_lane;
    assign o_tile_write_output = write_output;
    assign o_tile_write_code = i_write_code;
    assign o_tile_write_residual = i_write_residual;
    assign o_tile_write_code_enable = i_write_code_enable;
    assign o_tile_write_residual_enable = i_write_residual_enable;
    assign o_quiescent = !read_pending_q && !write_pending_q;
    assign o_protocol_error = protocol_error_q || (|i_tile_error);

    // The router needs native coordinates, not the mapper's diagnostic
    // feature/hidden mirrors.
    /* verilator lint_off PINCONNECTEMPTY */
    unified_fc_tp_native_parameter_address_mapper #(
        .FEATURE_COUNT(FEATURE_COUNT), .HIDDEN_COUNT(HIDDEN_COUNT),
        .HIDDEN_PER_TILE(HIDDEN_PER_TILE),
        .FEATURE_BANKS(FEATURE_BANKS), .OUTPUT_COUNT(OUTPUT_COUNT),
        .BANK_HASH_MODE(BANK_HASH_MODE),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .W0_TENSOR_ID(W0_TENSOR_ID), .W2_TENSOR_ID(W2_TENSOR_ID)
    ) u_read_mapper (
        .i_valid(i_read_valid), .i_tensor_id(i_read_tensor_id),
        .i_local_parameter_addr(read_local_addr), .o_valid(),
        .o_error(read_map_error), .o_is_w0(read_map_w0), .o_is_w2(),
        .o_feature_index(), .o_hidden_index(), .o_tile_index(read_tile),
        .o_tile_lane(read_lane), .o_feature_bank(read_bank),
        .o_feature_bank_row(read_row), .o_output_index(read_output)
    );

    unified_fc_tp_native_parameter_address_mapper #(
        .FEATURE_COUNT(FEATURE_COUNT), .HIDDEN_COUNT(HIDDEN_COUNT),
        .HIDDEN_PER_TILE(HIDDEN_PER_TILE),
        .FEATURE_BANKS(FEATURE_BANKS), .OUTPUT_COUNT(OUTPUT_COUNT),
        .BANK_HASH_MODE(BANK_HASH_MODE),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .W0_TENSOR_ID(W0_TENSOR_ID), .W2_TENSOR_ID(W2_TENSOR_ID)
    ) u_write_mapper (
        .i_valid(i_write_valid), .i_tensor_id(i_write_tensor_id),
        .i_local_parameter_addr(write_local_addr), .o_valid(),
        .o_error(write_map_error), .o_is_w0(write_map_w0), .o_is_w2(),
        .o_feature_index(), .o_hidden_index(), .o_tile_index(write_tile),
        .o_tile_lane(write_lane), .o_feature_bank(write_bank),
        .o_feature_bank_row(write_row), .o_output_index(write_output)
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // Pending request state directly controls RAMB address and enable inputs.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            read_pending_q <= 1'b0;
            read_error_q <= 1'b0;
            read_tile_q <= '0;
            read_tag_q <= '0;
            write_pending_q <= 1'b0;
            write_tile_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                protocol_error_q <= 1'b0;
            if (i_read_valid && i_write_valid)
                protocol_error_q <= 1'b1;

            if (read_fire) begin
                read_pending_q <= 1'b1;
                read_error_q <= read_map_error || read_base_error;
                read_tile_q <= read_tile;
                read_tag_q <= i_read_tag;
                if (read_map_error || read_base_error)
                    protocol_error_q <= 1'b1;
            end else if (response_fire) begin
                read_pending_q <= 1'b0;
                read_error_q <= 1'b0;
            end

            if (write_fire) begin
                if (write_map_error || write_base_error) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    write_pending_q <= 1'b1;
                    write_tile_q <= write_tile;
                end
            end
            if (write_pending_q && i_tile_write_done[write_tile_q])
                write_pending_q <= 1'b0;
        end
    end

    initial begin
        if (TILES < 1 || HIDDEN_COUNT != TILES * HIDDEN_PER_TILE ||
            FEATURE_BANKS != 8)
            $error("FC TP native parameter router dimensions are invalid");
    end
endmodule

`default_nettype wire
