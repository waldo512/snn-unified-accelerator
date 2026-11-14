`timescale 1ns/1ps
`default_nettype none

// Keeps the packed FC W0 working-set cache coherent with the authoritative
// scalar parameter store. Consecutive writes to one feature/tile row are
// coalesced; a row transition stalls the writer until that row is refreshed.
module unified_fc_h40_w0_coherent_write_bridge_v1 #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int MAX_FEATURES = 2312,
    parameter int HIDDEN_STRIDE = 480,
    parameter int HIDDEN_PER_TILE = 40,
    parameter logic [TENSOR_ID_WIDTH-1:0] W0_TENSOR_ID = 8'h10,
    parameter logic [TENSOR_ID_WIDTH-1:0] W2_TENSOR_ID = 8'h11,
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(HIDDEN_STRIDE + 1),
    parameter int FEATURE_INDEX_WIDTH =
        (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES),
    parameter int HIDDEN_INDEX_WIDTH =
        (HIDDEN_STRIDE <= 2) ? 1 : $clog2(HIDDEN_STRIDE),
    parameter int TILE_INDEX_WIDTH =
        (((HIDDEN_STRIDE + HIDDEN_PER_TILE - 1) / HIDDEN_PER_TILE) <= 2) ?
        1 : $clog2((HIDDEN_STRIDE + HIDDEN_PER_TILE - 1) /
            HIDDEN_PER_TILE)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic [PARAM_ADDR_WIDTH-1:0]           i_w0_parameter_base,
    input  logic [FEATURE_COUNT_WIDTH-1:0]        i_feature_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]         i_hidden_count,

    input  logic                                  i_write_valid,
    output logic                                  o_write_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_write_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_write_parameter_addr,
    input  logic signed [CODE_WIDTH-1:0]          i_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]      i_write_residual,
    input  logic                                  i_write_code_enable,
    input  logic                                  i_write_residual_enable,

    output logic                                  o_backend_write_valid,
    input  logic                                  i_backend_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_backend_write_parameter_addr,
    output logic signed [CODE_WIDTH-1:0]          o_backend_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]      o_backend_write_residual,
    output logic                                  o_backend_write_code_enable,
    output logic                                  o_backend_write_residual_enable,

    input  logic                                  i_flush_valid,
    output logic                                  o_flush_ready,
    output logic                                  o_flush_done,

    output logic                                  o_refresh_valid,
    input  logic                                  i_refresh_ready,
    output logic [FEATURE_INDEX_WIDTH-1:0]        o_refresh_feature,
    output logic [3:0]                            o_refresh_tile,
    input  logic                                  i_refresh_done,
    input  logic                                  i_refresh_error,

    output logic                                  o_quiescent,
    output logic                                  o_sticky_error
);
    typedef enum logic [1:0] {
        STATE_PASS,
        STATE_ISSUE_REFRESH,
        STATE_WAIT_REFRESH,
        STATE_ERROR
    } state_t;

    state_t state_q;
    logic dirty_q;
    logic flush_q;
    logic [FEATURE_INDEX_WIDTH-1:0] dirty_feature_q;
    logic [3:0] dirty_tile_q;

    logic [PARAM_ADDR_WIDTH-1:0] write_local_addr;
    logic write_base_error;
    logic map_error;
    logic map_is_w0;
    logic [FEATURE_INDEX_WIDTH-1:0] map_feature;
    logic [HIDDEN_INDEX_WIDTH-1:0] map_hidden;
    logic [TILE_INDEX_WIDTH-1:0] map_tile;
    logic write_mapping_error;
    logic write_affects_cache;
    logic write_same_row;
    logic row_transition;
    logic pass_write;
    logic write_fire;
    logic flush_fire;
    logic refresh_fire;
    logic [PARAM_ADDR_WIDTH:0] write_local_addr_ext;
    logic [PARAM_ADDR_WIDTH:0] w0_elements_ext;
    logic [PARAM_ADDR_WIDTH-1:0] map_feature_ext;
    logic [PARAM_ADDR_WIDTH-1:0] map_hidden_ext;

    assign write_base_error = i_write_tensor_id == W0_TENSOR_ID &&
        i_write_parameter_addr < i_w0_parameter_base;
    assign write_local_addr = i_write_parameter_addr - i_w0_parameter_base;
    assign write_local_addr_ext = {1'b0, write_local_addr};
    assign w0_elements_ext = (PARAM_ADDR_WIDTH + 1)'(MAX_FEATURES) *
        (PARAM_ADDR_WIDTH + 1)'(HIDDEN_STRIDE);
    assign map_is_w0 = i_write_valid &&
        i_write_tensor_id == W0_TENSOR_ID && !write_base_error;
    assign map_error = map_is_w0 && write_local_addr_ext >= w0_elements_ext;
    assign map_feature_ext = write_local_addr /
        PARAM_ADDR_WIDTH'(HIDDEN_STRIDE);
    assign map_hidden_ext = write_local_addr %
        PARAM_ADDR_WIDTH'(HIDDEN_STRIDE);
    assign map_feature = FEATURE_INDEX_WIDTH'(map_feature_ext);
    assign map_hidden = HIDDEN_INDEX_WIDTH'(map_hidden_ext);
    assign map_tile = TILE_INDEX_WIDTH'(
        map_hidden_ext / PARAM_ADDR_WIDTH'(HIDDEN_PER_TILE));
    assign write_mapping_error = i_write_tensor_id == W0_TENSOR_ID &&
        (write_base_error || map_error ||
         FEATURE_COUNT_WIDTH'(map_feature) >= i_feature_count ||
         HIDDEN_COUNT_WIDTH'(map_hidden) >= i_hidden_count);
    assign write_affects_cache = i_write_tensor_id == W0_TENSOR_ID &&
        i_write_code_enable && !write_mapping_error && map_is_w0;
    assign write_same_row = dirty_q && write_affects_cache &&
        map_feature == dirty_feature_q && 4'(map_tile) == dirty_tile_q;
    assign row_transition = state_q == STATE_PASS && dirty_q &&
        i_write_valid && write_affects_cache && !write_same_row;

    assign o_flush_ready = state_q == STATE_PASS && !i_write_valid &&
        !o_sticky_error;
    assign flush_fire = i_flush_valid && o_flush_ready;
    assign pass_write = state_q == STATE_PASS && !row_transition &&
        !i_flush_valid && !o_sticky_error;
    assign o_backend_write_valid = i_write_valid && pass_write;
    assign o_write_ready = i_backend_write_ready && pass_write;
    assign write_fire = i_write_valid && o_write_ready;

    assign o_backend_write_parameter_addr = i_write_parameter_addr;
    assign o_backend_write_code = i_write_code;
    assign o_backend_write_residual = i_write_residual;
    assign o_backend_write_code_enable = i_write_code_enable;
    assign o_backend_write_residual_enable = i_write_residual_enable;

    assign o_refresh_valid = state_q == STATE_ISSUE_REFRESH;
    assign refresh_fire = o_refresh_valid && i_refresh_ready;
    assign o_refresh_feature = dirty_feature_q;
    assign o_refresh_tile = dirty_tile_q;
    assign o_quiescent = state_q == STATE_PASS && !dirty_q &&
        !i_write_valid && !i_flush_valid && !o_sticky_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_PASS;
            dirty_q <= 1'b0;
            flush_q <= 1'b0;
            dirty_feature_q <= '0;
            dirty_tile_q <= '0;
            o_flush_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_PASS;
            dirty_q <= 1'b0;
            flush_q <= 1'b0;
            o_flush_done <= 1'b0;
        end else begin
            o_flush_done <= 1'b0;
            if (i_clear_error && state_q == STATE_ERROR) begin
                state_q <= STATE_PASS;
                dirty_q <= 1'b0;
                flush_q <= 1'b0;
                o_sticky_error <= 1'b0;
            end

            unique case (state_q)
                STATE_PASS: begin
                    if (flush_fire) begin
                        if (dirty_q) begin
                            flush_q <= 1'b1;
                            state_q <= STATE_ISSUE_REFRESH;
                        end else begin
                            o_flush_done <= 1'b1;
                        end
                    end else if (row_transition) begin
                        flush_q <= 1'b0;
                        state_q <= STATE_ISSUE_REFRESH;
                    end else if (write_fire) begin
                        if (write_mapping_error &&
                                i_write_code_enable) begin
                            o_sticky_error <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else if (write_affects_cache) begin
                            if (!dirty_q) begin
                                dirty_q <= 1'b1;
                                dirty_feature_q <= map_feature;
                                dirty_tile_q <= 4'(map_tile);
                            end
                        end
                    end
                end

                STATE_ISSUE_REFRESH: begin
                    if (refresh_fire)
                        state_q <= STATE_WAIT_REFRESH;
                end

                STATE_WAIT_REFRESH: begin
                    if (i_refresh_error) begin
                        o_sticky_error <= 1'b1;
                        state_q <= STATE_ERROR;
                    end else if (i_refresh_done) begin
                        dirty_q <= 1'b0;
                        if (flush_q) begin
                            flush_q <= 1'b0;
                            o_flush_done <= 1'b1;
                        end
                        state_q <= STATE_PASS;
                    end
                end

                default: begin
                    o_sticky_error <= 1'b1;
                    state_q <= STATE_ERROR;
                end
            endcase
        end
    end

    initial begin
        if (MAX_FEATURES < 1 || HIDDEN_STRIDE < 1 ||
                HIDDEN_PER_TILE != 40 ||
                ((HIDDEN_STRIDE + HIDDEN_PER_TILE - 1) /
                    HIDDEN_PER_TILE) > 16 ||
                W0_TENSOR_ID == W2_TENSOR_ID)
            $error("FC H40 coherent write bridge parameter mismatch");
    end
endmodule

`default_nettype wire
