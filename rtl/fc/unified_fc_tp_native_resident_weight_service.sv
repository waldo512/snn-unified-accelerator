`timescale 1ns/1ps
`default_nettype none

// Forward-compatible FC resident weight service backed by writable native
// W0/W2 tile stores. Inference and TP learning share one code image; the
// residual image is colocated and is visible only through the learning port.
module unified_fc_tp_native_resident_weight_service #(
    parameter bit ENABLE_W0_LEARNING = 1'b1,
    parameter int FI_BITS = 12,
    parameter int TILES = 4,
    parameter int BANKS = 8,
    parameter int BANK_HASH_MODE = 1,
    parameter int H_PER_TILE = 40,
    parameter int W0_BITS = 5,
    parameter int W0_BANK_DEPTH = 289,
    parameter int O_DIM = 10,
    parameter int W2_BITS = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter logic [TENSOR_ID_WIDTH-1:0] W0_TENSOR_ID = 8'h10,
    parameter logic [TENSOR_ID_WIDTH-1:0] W2_TENSOR_ID = 8'h11,
    parameter WEIGHT_DIR = "data/weights/p3a_nmnist_h160_k8_input2312",
    parameter W0_RESIDUAL_INIT_FILE = "",
    parameter int TILE_W0_BITS = BANKS * H_PER_TILE * W0_BITS,
    parameter int TILE_W2_BITS = H_PER_TILE * O_DIM * W2_BITS,
    parameter int W0_ROW_ADDR_WIDTH =
        (W0_BANK_DEPTH <= 2) ? 1 : $clog2(W0_BANK_DEPTH),
    parameter int TILE_LANE_WIDTH =
        (H_PER_TILE <= 2) ? 1 : $clog2(H_PER_TILE),
    parameter int OUTPUT_INDEX_WIDTH =
        (O_DIM <= 2) ? 1 : $clog2(O_DIM)
) (
    input  logic                                          i_clk,
    input  logic                                          i_rst_n,
    input  logic                                          i_clear_pipeline,
    input  logic                                          i_clear_error,

    input  logic [TILES-1:0]                             i_active_tile_mask,
    input  logic                                          i_request_valid,
    output logic                                          o_request_ready,
    input  logic [BANKS-1:0]                             i_bank_valid,
    input  logic [BANKS*FI_BITS-1:0]                     i_fi_flat,
    output logic                                          o_row_valid,
    input  logic                                          i_row_ready,
    output logic [BANKS-1:0]                             o_lane_valid,
    output logic [TILES*TILE_W0_BITS-1:0]                o_w0_rows_flat,
    output logic [TILES*TILE_W2_BITS-1:0]                o_w2_weight_flat,

    input  logic [PARAM_ADDR_WIDTH-1:0]                  i_w0_parameter_base,
    input  logic [PARAM_ADDR_WIDTH-1:0]                  i_w2_parameter_base,
    input  logic                                          i_learning_read_valid,
    output logic                                          o_learning_read_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                   i_learning_read_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]                  i_learning_read_parameter_addr,
    input  logic [TAG_WIDTH-1:0]                         i_learning_read_tag,
    output logic                                          o_learning_response_valid,
    input  logic                                          i_learning_response_ready,
    output logic [TAG_WIDTH-1:0]                         o_learning_response_tag,
    output logic signed [CODE_WIDTH-1:0]                 o_learning_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]             o_learning_response_residual,
    output logic                                          o_learning_response_error,
    input  logic                                          i_learning_write_valid,
    output logic                                          o_learning_write_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                   i_learning_write_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]                  i_learning_write_parameter_addr,
    input  logic signed [CODE_WIDTH-1:0]                 i_learning_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]             i_learning_write_residual,
    input  logic                                          i_learning_write_code_enable,
    input  logic                                          i_learning_write_residual_enable,

    output logic                                          o_idle,
    output logic                                          o_learning_quiescent,
    output logic                                          o_protocol_error
);
    logic request_fire;
    logic router_read_valid;
    logic router_read_ready;
    logic router_write_valid;
    logic router_write_ready;
    logic router_quiescent;
    logic router_error;
    logic service_error_q;
    logic [TILES-1:0] tile_read_valid;
    logic [TILES-1:0] tile_read_ready;
    logic tile_read_is_w0;
    logic [2:0] tile_read_feature_bank;
    logic [W0_ROW_ADDR_WIDTH-1:0] tile_read_feature_row;
    logic [TILE_LANE_WIDTH-1:0] tile_read_tile_lane;
    logic [OUTPUT_INDEX_WIDTH-1:0] tile_read_output;
    logic [TILES-1:0] tile_response_valid;
    logic [TILES-1:0] tile_response_ready;
    logic signed [TILES*CODE_WIDTH-1:0] tile_response_code;
    logic signed [TILES*RESIDUAL_WIDTH-1:0] tile_response_residual;
    logic [TILES-1:0] tile_write_valid;
    logic [TILES-1:0] tile_write_ready;
    logic tile_write_is_w0;
    logic [2:0] tile_write_feature_bank;
    logic [W0_ROW_ADDR_WIDTH-1:0] tile_write_feature_row;
    logic [TILE_LANE_WIDTH-1:0] tile_write_tile_lane;
    logic [OUTPUT_INDEX_WIDTH-1:0] tile_write_output;
    logic signed [CODE_WIDTH-1:0] tile_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] tile_write_residual;
    logic tile_write_code_enable;
    logic tile_write_residual_enable;
    logic [TILES-1:0] tile_write_done;
    logic [TILES-1:0] tile_error;

    assign o_request_ready = (!o_row_valid || i_row_ready) &&
        router_quiescent && !i_learning_read_valid &&
        !i_learning_write_valid;
    assign request_fire = i_request_valid && o_request_ready;
    assign router_read_valid = i_learning_read_valid && !o_row_valid &&
        !i_request_valid;
    assign router_write_valid = i_learning_write_valid && !o_row_valid &&
        !i_request_valid;
    assign o_learning_read_ready = router_read_ready && !o_row_valid &&
        !i_request_valid;
    assign o_learning_write_ready = router_write_ready && !o_row_valid &&
        !i_request_valid;
    assign o_idle = !o_row_valid && router_quiescent;
    assign o_learning_quiescent = router_quiescent;
    assign o_protocol_error = service_error_q || router_error;

    unified_fc_tp_native_parameter_router #(
        .TILES(TILES), .FEATURE_COUNT(BANKS * W0_BANK_DEPTH),
        .HIDDEN_PER_TILE(H_PER_TILE), .FEATURE_BANKS(BANKS),
        .BANK_HASH_MODE(BANK_HASH_MODE),
        .OUTPUT_COUNT(O_DIM), .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .W0_TENSOR_ID(W0_TENSOR_ID), .W2_TENSOR_ID(W2_TENSOR_ID)
    ) u_router (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_w0_parameter_base(i_w0_parameter_base),
        .i_w2_parameter_base(i_w2_parameter_base),
        .i_read_valid(router_read_valid), .o_read_ready(router_read_ready),
        .i_read_tensor_id(i_learning_read_tensor_id),
        .i_read_parameter_addr(i_learning_read_parameter_addr),
        .i_read_tag(i_learning_read_tag),
        .o_response_valid(o_learning_response_valid),
        .i_response_ready(i_learning_response_ready),
        .o_response_tag(o_learning_response_tag),
        .o_response_code(o_learning_response_code),
        .o_response_residual(o_learning_response_residual),
        .o_response_error(o_learning_response_error),
        .i_write_valid(router_write_valid),
        .o_write_ready(router_write_ready),
        .i_write_tensor_id(i_learning_write_tensor_id),
        .i_write_parameter_addr(i_learning_write_parameter_addr),
        .i_write_code(i_learning_write_code),
        .i_write_residual(i_learning_write_residual),
        .i_write_code_enable(i_learning_write_code_enable),
        .i_write_residual_enable(i_learning_write_residual_enable),
        .o_tile_read_valid(tile_read_valid),
        .i_tile_read_ready(tile_read_ready),
        .o_tile_read_is_w0(tile_read_is_w0),
        .o_tile_read_feature_bank(tile_read_feature_bank),
        .o_tile_read_feature_row(tile_read_feature_row),
        .o_tile_read_tile_lane(tile_read_tile_lane),
        .o_tile_read_output(tile_read_output),
        .i_tile_response_valid(tile_response_valid),
        .o_tile_response_ready(tile_response_ready),
        .i_tile_response_code(tile_response_code),
        .i_tile_response_residual(tile_response_residual),
        .o_tile_write_valid(tile_write_valid),
        .i_tile_write_ready(tile_write_ready),
        .o_tile_write_is_w0(tile_write_is_w0),
        .o_tile_write_feature_bank(tile_write_feature_bank),
        .o_tile_write_feature_row(tile_write_feature_row),
        .o_tile_write_tile_lane(tile_write_tile_lane),
        .o_tile_write_output(tile_write_output),
        .o_tile_write_code(tile_write_code),
        .o_tile_write_residual(tile_write_residual),
        .o_tile_write_code_enable(tile_write_code_enable),
        .o_tile_write_residual_enable(tile_write_residual_enable),
        .i_tile_write_done(tile_write_done), .i_tile_error(tile_error),
        .o_quiescent(router_quiescent), .o_protocol_error(router_error)
    );

`define FC_TP_NATIVE_TILE(INDEX, ID) \
    if (TILES > INDEX) begin : g_tile_``INDEX \
        unified_fc_tp_native_tile_parameter_store #( \
            .ENABLE_W0_LEARNING(ENABLE_W0_LEARNING), \
            .FI_BITS(FI_BITS), .FEATURE_BANKS(BANKS), \
            .HIDDEN_PER_TILE(H_PER_TILE), .W0_BITS(W0_BITS), \
            .W0_BANK_DEPTH(W0_BANK_DEPTH), .OUTPUT_COUNT(O_DIM), \
            .W2_BITS(W2_BITS), .CODE_WIDTH(CODE_WIDTH), \
            .RESIDUAL_WIDTH(RESIDUAL_WIDTH), \
            .W0_BANK_FILE_0({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_00.mem"}), \
            .W0_BANK_FILE_1({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_01.mem"}), \
            .W0_BANK_FILE_2({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_02.mem"}), \
            .W0_BANK_FILE_3({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_03.mem"}), \
            .W0_BANK_FILE_4({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_04.mem"}), \
            .W0_BANK_FILE_5({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_05.mem"}), \
            .W0_BANK_FILE_6({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_06.mem"}), \
            .W0_BANK_FILE_7({WEIGHT_DIR, "/w0_cluster_", ID, \
                "_bank_07.mem"}), \
            .W0_RESIDUAL_INIT_FILE(W0_RESIDUAL_INIT_FILE), \
            .W2_FILE({WEIGHT_DIR, "/w2_cluster_", ID, ".hex"}) \
        ) u_store ( \
            .i_clk(i_clk), .i_rst_n(i_rst_n), \
            .i_clear_error(i_clear_error), \
            .i_forward_enable(request_fire && i_active_tile_mask[INDEX]), \
            .i_forward_bank_valid(i_bank_valid), \
            .i_forward_fi_flat(i_fi_flat), \
            .o_forward_w0_rows_flat(o_w0_rows_flat[ \
                INDEX*TILE_W0_BITS +: TILE_W0_BITS]), \
            .o_forward_w2_weight_flat(o_w2_weight_flat[ \
                INDEX*TILE_W2_BITS +: TILE_W2_BITS]), \
            .i_learning_read_valid(tile_read_valid[INDEX]), \
            .o_learning_read_ready(tile_read_ready[INDEX]), \
            .i_learning_read_is_w0(tile_read_is_w0), \
            .i_learning_read_feature_bank(tile_read_feature_bank), \
            .i_learning_read_feature_row(tile_read_feature_row), \
            .i_learning_read_tile_lane(tile_read_tile_lane), \
            .i_learning_read_output(tile_read_output), \
            .o_learning_response_valid(tile_response_valid[INDEX]), \
            .i_learning_response_ready(tile_response_ready[INDEX]), \
            .o_learning_response_code(tile_response_code[ \
                INDEX*CODE_WIDTH +: CODE_WIDTH]), \
            .o_learning_response_residual(tile_response_residual[ \
                INDEX*RESIDUAL_WIDTH +: RESIDUAL_WIDTH]), \
            .i_learning_write_valid(tile_write_valid[INDEX]), \
            .o_learning_write_ready(tile_write_ready[INDEX]), \
            .i_learning_write_is_w0(tile_write_is_w0), \
            .i_learning_write_feature_bank(tile_write_feature_bank), \
            .i_learning_write_feature_row(tile_write_feature_row), \
            .i_learning_write_tile_lane(tile_write_tile_lane), \
            .i_learning_write_output(tile_write_output), \
            .i_learning_write_code(tile_write_code), \
            .i_learning_write_residual(tile_write_residual), \
            .i_learning_write_code_enable(tile_write_code_enable), \
            .i_learning_write_residual_enable( \
                tile_write_residual_enable), \
            .o_learning_write_done(tile_write_done[INDEX]), \
            .o_protocol_error(tile_error[INDEX]) \
        ); \
    end

    generate
        `FC_TP_NATIVE_TILE(0, "00")
        `FC_TP_NATIVE_TILE(1, "01")
        `FC_TP_NATIVE_TILE(2, "02")
        `FC_TP_NATIVE_TILE(3, "03")
        `FC_TP_NATIVE_TILE(4, "04")
        `FC_TP_NATIVE_TILE(5, "05")
        `FC_TP_NATIVE_TILE(6, "06")
        `FC_TP_NATIVE_TILE(7, "07")
        `FC_TP_NATIVE_TILE(8, "08")
        `FC_TP_NATIVE_TILE(9, "09")
        `FC_TP_NATIVE_TILE(10, "10")
        `FC_TP_NATIVE_TILE(11, "11")
    endgenerate
`undef FC_TP_NATIVE_TILE

    // Keep row-valid control synchronous before it reaches resident RAMB ports.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            o_row_valid <= 1'b0;
            o_lane_valid <= '0;
            service_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                service_error_q <= 1'b0;
            if (i_request_valid &&
                (i_learning_read_valid || i_learning_write_valid))
                service_error_q <= 1'b1;
            if (i_clear_pipeline) begin
                o_row_valid <= 1'b0;
                o_lane_valid <= '0;
            end else begin
                if (o_row_valid && i_row_ready) begin
                    o_row_valid <= request_fire;
                    if (!request_fire)
                        o_lane_valid <= '0;
                end else if (request_fire) begin
                    o_row_valid <= 1'b1;
                end
                if (request_fire)
                    o_lane_valid <= i_bank_valid;
            end
        end
    end

    initial begin
        if (TILES < 1 || TILES > 12 || BANKS != 8 || H_PER_TILE < 1 ||
            W0_BANK_DEPTH < 1 || O_DIM < 1)
            $error("FC TP native resident service dimensions are invalid");
    end
endmodule

`default_nettype wire
