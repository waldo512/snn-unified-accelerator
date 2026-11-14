`timescale 1ns/1ps
`default_nettype none

// Native parameter store for one H40 FC tile. Eight independently readable
// W0 banks preserve K8 inference bandwidth; TP reads and scalar updates are
// serialized onto each bank's second port. W2 and both residual tensors live
// beside the same forward-visible code state.
module unified_fc_tp_native_tile_parameter_store #(
    parameter bit ENABLE_W0_LEARNING = 1'b1,
    parameter int FI_BITS = 12,
    parameter int FEATURE_BANKS = 8,
    parameter int HIDDEN_PER_TILE = 40,
    parameter int W0_BITS = 5,
    parameter int W0_BANK_DEPTH = 289,
    parameter int OUTPUT_COUNT = 10,
    parameter int W2_BITS = 8,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter W0_BANK_FILE_0 = "",
    parameter W0_BANK_FILE_1 = "",
    parameter W0_BANK_FILE_2 = "",
    parameter W0_BANK_FILE_3 = "",
    parameter W0_BANK_FILE_4 = "",
    parameter W0_BANK_FILE_5 = "",
    parameter W0_BANK_FILE_6 = "",
    parameter W0_BANK_FILE_7 = "",
    parameter W0_RESIDUAL_INIT_FILE = "",
    parameter W2_FILE = "",
    parameter int W0_ROW_ADDR_WIDTH =
        (W0_BANK_DEPTH <= 2) ? 1 : $clog2(W0_BANK_DEPTH),
    parameter int TILE_LANE_WIDTH =
        (HIDDEN_PER_TILE <= 2) ? 1 : $clog2(HIDDEN_PER_TILE),
    parameter int OUTPUT_INDEX_WIDTH =
        (OUTPUT_COUNT <= 2) ? 1 : $clog2(OUTPUT_COUNT)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_forward_enable,
    input  logic [FEATURE_BANKS-1:0]                  i_forward_bank_valid,
    input  logic [FEATURE_BANKS*FI_BITS-1:0]          i_forward_fi_flat,
    output logic [FEATURE_BANKS*HIDDEN_PER_TILE*W0_BITS-1:0]
                                                       o_forward_w0_rows_flat,
    output logic [HIDDEN_PER_TILE*OUTPUT_COUNT*W2_BITS-1:0]
                                                       o_forward_w2_weight_flat,

    input  logic                                      i_learning_read_valid,
    output logic                                      o_learning_read_ready,
    input  logic                                      i_learning_read_is_w0,
    input  logic [2:0]                                i_learning_read_feature_bank,
    input  logic [W0_ROW_ADDR_WIDTH-1:0]              i_learning_read_feature_row,
    input  logic [TILE_LANE_WIDTH-1:0]                i_learning_read_tile_lane,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]             i_learning_read_output,
    output logic                                      o_learning_response_valid,
    input  logic                                      i_learning_response_ready,
    output logic signed [CODE_WIDTH-1:0]              o_learning_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_learning_response_residual,

    input  logic                                      i_learning_write_valid,
    output logic                                      o_learning_write_ready,
    input  logic                                      i_learning_write_is_w0,
    input  logic [2:0]                                i_learning_write_feature_bank,
    input  logic [W0_ROW_ADDR_WIDTH-1:0]              i_learning_write_feature_row,
    input  logic [TILE_LANE_WIDTH-1:0]                i_learning_write_tile_lane,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]             i_learning_write_output,
    input  logic signed [CODE_WIDTH-1:0]              i_learning_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_learning_write_residual,
    input  logic                                      i_learning_write_code_enable,
    input  logic                                      i_learning_write_residual_enable,
    output logic                                      o_learning_write_done,
    output logic                                      o_protocol_error
);
    localparam int W0_ROW_BITS = HIDDEN_PER_TILE * W0_BITS;
    localparam int W2_ROW_BITS = OUTPUT_COUNT * W2_BITS;
    localparam int W2_SCALAR_DEPTH = HIDDEN_PER_TILE * OUTPUT_COUNT;
    localparam int W2_SCALAR_ADDR_WIDTH =
        (W2_SCALAR_DEPTH <= 2) ? 1 : $clog2(W2_SCALAR_DEPTH);

    logic [FEATURE_BANKS*W0_ROW_BITS-1:0] w0_rows_flat;
    logic [FEATURE_BANKS-1:0] w0_read_ready;
    logic [FEATURE_BANKS-1:0] w0_response_valid;
    logic [FEATURE_BANKS*CODE_WIDTH-1:0] w0_response_code;
    logic [FEATURE_BANKS*RESIDUAL_WIDTH-1:0] w0_response_residual;
    logic [FEATURE_BANKS-1:0] w0_write_ready;
    logic [FEATURE_BANKS-1:0] w0_write_done;
    logic [FEATURE_BANKS-1:0] w0_error;

    logic [W2_ROW_BITS-1:0] w2_code_mem [0:HIDDEN_PER_TILE-1];
    (* ram_style = "block" *)
    logic signed [RESIDUAL_WIDTH-1:0]
        w2_residual_mem [0:W2_SCALAR_DEPTH-1];
    logic w2_response_valid_q;
    logic signed [CODE_WIDTH-1:0] w2_response_code_q;
    logic signed [RESIDUAL_WIDTH-1:0] w2_response_residual_q;
    logic w2_write_done_q;
    logic protocol_error_q;
    logic read_is_w0;
    logic write_is_w0;
    logic w2_read_fire;
    logic w2_write_fire;
    logic w2_write_code_fits;
    logic [W2_SCALAR_ADDR_WIDTH-1:0] w2_read_index;
    logic [W2_SCALAR_ADDR_WIDTH-1:0] w2_write_index;

`define FC_TP_W0_BANK(INDEX, INIT) \
    unified_fc_tp_w0_native_bank_ram #( \
        .HIDDEN_PER_TILE(HIDDEN_PER_TILE), .W0_BITS(W0_BITS), \
        .DEPTH(W0_BANK_DEPTH), .CODE_WIDTH(CODE_WIDTH), \
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH), .INIT_FILE(INIT), \
        .RESIDUAL_INIT_FILE(W0_RESIDUAL_INIT_FILE) \
    ) u_w0_bank_``INDEX ( \
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error), \
        .i_forward_enable(i_forward_enable && \
            i_forward_bank_valid[INDEX]), \
        .i_forward_addr(W0_ROW_ADDR_WIDTH'( \
            i_forward_fi_flat[INDEX*FI_BITS +: FI_BITS] / \
            FI_BITS'(FEATURE_BANKS))), \
        .o_forward_row(w0_rows_flat[INDEX*W0_ROW_BITS +: W0_ROW_BITS]), \
        .i_learning_read_valid(i_learning_read_valid && read_is_w0 && \
            i_learning_read_feature_bank == 3'(INDEX)), \
        .o_learning_read_ready(w0_read_ready[INDEX]), \
        .i_learning_read_addr(i_learning_read_feature_row), \
        .i_learning_read_lane(i_learning_read_tile_lane), \
        .o_learning_response_valid(w0_response_valid[INDEX]), \
        .i_learning_response_ready(i_learning_response_ready), \
        .o_learning_response_code(w0_response_code[INDEX*CODE_WIDTH +: \
            CODE_WIDTH]), \
        .o_learning_response_residual(w0_response_residual[ \
            INDEX*RESIDUAL_WIDTH +: RESIDUAL_WIDTH]), \
        .i_learning_write_valid(i_learning_write_valid && write_is_w0 && \
            i_learning_write_feature_bank == 3'(INDEX)), \
        .o_learning_write_ready(w0_write_ready[INDEX]), \
        .i_learning_write_addr(i_learning_write_feature_row), \
        .i_learning_write_lane(i_learning_write_tile_lane), \
        .i_learning_write_code(i_learning_write_code), \
        .i_learning_write_residual(i_learning_write_residual), \
        .i_learning_write_code_enable(i_learning_write_code_enable), \
        .i_learning_write_residual_enable( \
            i_learning_write_residual_enable), \
        .o_learning_write_done(w0_write_done[INDEX]), \
        .o_protocol_error(w0_error[INDEX]) \
    );

    generate
        if (ENABLE_W0_LEARNING) begin : g_w0_learning
            `FC_TP_W0_BANK(0, W0_BANK_FILE_0)
            `FC_TP_W0_BANK(1, W0_BANK_FILE_1)
            `FC_TP_W0_BANK(2, W0_BANK_FILE_2)
            `FC_TP_W0_BANK(3, W0_BANK_FILE_3)
            `FC_TP_W0_BANK(4, W0_BANK_FILE_4)
            `FC_TP_W0_BANK(5, W0_BANK_FILE_5)
            `FC_TP_W0_BANK(6, W0_BANK_FILE_6)
            `FC_TP_W0_BANK(7, W0_BANK_FILE_7)
        end else begin : g_w0_inference_only
`define FC_TP_W0_ROM(INDEX, INIT) \
            unified_fc_compat_w0_bank_rom #( \
                .WORD_BITS(W0_ROW_BITS), .DEPTH(W0_BANK_DEPTH), \
                .INIT_FILE(INIT) \
            ) u_w0_bank_``INDEX ( \
                .clk(i_clk), \
                .i_enable(i_forward_enable && \
                    i_forward_bank_valid[INDEX]), \
                .i_addr(W0_ROW_ADDR_WIDTH'( \
                    i_forward_fi_flat[INDEX*FI_BITS +: FI_BITS] / \
                    FI_BITS'(FEATURE_BANKS))), \
                .o_data(w0_rows_flat[INDEX*W0_ROW_BITS +: W0_ROW_BITS]) \
            );
            `FC_TP_W0_ROM(0, W0_BANK_FILE_0)
            `FC_TP_W0_ROM(1, W0_BANK_FILE_1)
            `FC_TP_W0_ROM(2, W0_BANK_FILE_2)
            `FC_TP_W0_ROM(3, W0_BANK_FILE_3)
            `FC_TP_W0_ROM(4, W0_BANK_FILE_4)
            `FC_TP_W0_ROM(5, W0_BANK_FILE_5)
            `FC_TP_W0_ROM(6, W0_BANK_FILE_6)
            `FC_TP_W0_ROM(7, W0_BANK_FILE_7)
`undef FC_TP_W0_ROM
            assign w0_read_ready = '0;
            assign w0_response_valid = '0;
            assign w0_response_code = '0;
            assign w0_response_residual = '0;
            assign w0_write_ready = '0;
            assign w0_write_done = '0;
            assign w0_error = '0;
        end
    endgenerate
`undef FC_TP_W0_BANK

    always_comb begin
        read_is_w0 = i_learning_read_is_w0;
        write_is_w0 = i_learning_write_is_w0;
        o_forward_w0_rows_flat = w0_rows_flat;
        for (int hidden = 0; hidden < HIDDEN_PER_TILE; hidden++)
            o_forward_w2_weight_flat[
                hidden*W2_ROW_BITS +: W2_ROW_BITS] = w2_code_mem[hidden];

        o_learning_read_ready = read_is_w0 ?
            w0_read_ready[i_learning_read_feature_bank] :
            !w2_response_valid_q && !i_learning_write_valid;
        o_learning_write_ready = write_is_w0 ?
            w0_write_ready[i_learning_write_feature_bank] :
            !w2_response_valid_q && !i_learning_read_valid;
        w2_read_fire = i_learning_read_valid &&
            o_learning_read_ready && !read_is_w0;
        w2_write_fire = i_learning_write_valid &&
            o_learning_write_ready && !write_is_w0;
        w2_write_code_fits = i_learning_write_code ==
            CODE_WIDTH'($signed(i_learning_write_code[W2_BITS-1:0]));
        w2_read_index = W2_SCALAR_ADDR_WIDTH'(
            $unsigned(i_learning_read_tile_lane) * OUTPUT_COUNT +
            $unsigned(i_learning_read_output));
        w2_write_index = W2_SCALAR_ADDR_WIDTH'(
            $unsigned(i_learning_write_tile_lane) * OUTPUT_COUNT +
            $unsigned(i_learning_write_output));

        o_learning_response_valid = (|w0_response_valid) ||
            w2_response_valid_q;
        o_learning_response_code = w2_response_code_q;
        o_learning_response_residual = w2_response_residual_q;
        for (int bank = 0; bank < FEATURE_BANKS; bank++) begin
            if (w0_response_valid[bank]) begin
                o_learning_response_code = $signed(w0_response_code[
                    bank*CODE_WIDTH +: CODE_WIDTH]);
                o_learning_response_residual = $signed(
                    w0_response_residual[
                        bank*RESIDUAL_WIDTH +: RESIDUAL_WIDTH]);
            end
        end
        o_learning_write_done = (|w0_write_done) || w2_write_done_q;
        o_protocol_error = protocol_error_q || (|w0_error);
    end

    initial begin
        if (W2_FILE != "")
            $readmemh(W2_FILE, w2_code_mem);
        for (int parameter_index = 0;
             parameter_index < W2_SCALAR_DEPTH; parameter_index++)
            w2_residual_mem[parameter_index] = '0;
    end

    // The data memories and their response payload registers are intentionally
    // reset-free so Vivado can use supported RAM/register templates. Validity
    // remains the sole authority after reset.
    always_ff @(posedge i_clk) begin
        if (w2_read_fire) begin
            w2_response_code_q <= CODE_WIDTH'($signed(w2_code_mem[
                i_learning_read_tile_lane][
                i_learning_read_output*W2_BITS +: W2_BITS]));
            w2_response_residual_q <= w2_residual_mem[w2_read_index];
        end

        if (w2_write_fire &&
            (!i_learning_write_code_enable || w2_write_code_fits)) begin
            if (i_learning_write_code_enable)
                w2_code_mem[i_learning_write_tile_lane][
                    i_learning_write_output*W2_BITS +: W2_BITS] <=
                    i_learning_write_code[W2_BITS-1:0];
            if (i_learning_write_residual_enable)
                w2_residual_mem[w2_write_index] <=
                    i_learning_write_residual;
        end
    end

    // Response validity is synchronous to avoid an asynchronous RAMB control cone.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            w2_response_valid_q <= 1'b0;
            w2_write_done_q <= 1'b0;
            protocol_error_q <= 1'b0;
        end else begin
            w2_write_done_q <= 1'b0;
            if (i_clear_error)
                protocol_error_q <= 1'b0;
            if (w2_response_valid_q && i_learning_response_ready)
                w2_response_valid_q <= 1'b0;
            if (i_learning_read_valid && i_learning_write_valid)
                protocol_error_q <= 1'b1;

            if (w2_read_fire) begin
                w2_response_valid_q <= 1'b1;
            end

            if (w2_write_fire) begin
                w2_write_done_q <= 1'b1;
                if (i_learning_write_code_enable && !w2_write_code_fits) begin
                    protocol_error_q <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (FEATURE_BANKS != 8 || HIDDEN_PER_TILE < 1 ||
            W0_BITS < 2 || W2_BITS < 2 || OUTPUT_COUNT < 1 ||
            CODE_WIDTH < W0_BITS || CODE_WIDTH < W2_BITS)
            $error("FC TP native tile parameter dimensions are invalid");
    end
endmodule

`default_nettype wire
