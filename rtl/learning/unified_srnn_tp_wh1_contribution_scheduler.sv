`timescale 1ns/1ps
`default_nettype none

// Stores one SRNN W_h1 observation window in reset-free block RAMs and replays
// the two-path outer-product contributions in optimizer address order.
module unified_srnn_tp_wh1_contribution_scheduler #(
    parameter int BATCH_MAX = 64,
    parameter int HIDDEN_MAX = 256,
    parameter int BACKBONE_MAX = 450,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int PARAM_ADDR_WIDTH = 24
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_dpre_valid,
    output logic                                      o_dpre_ready,
    input  logic                                      i_dpre_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_dpre_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_dpre_hidden_id,
    input  logic signed [63:0]                        i_dpre_q40,
    input  logic                                      i_dpre_last,

    input  logic                                      i_backbone_valid,
    output logic                                      o_backbone_ready,
    input  logic                                      i_backbone_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_backbone_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_backbone_id,
    input  logic signed [31:0]                        i_backbone_q14,
    input  logic                                      i_backbone_last,

    // Opens a new observation window without reloading the persistent tensor
    // configuration. RAM contents are overwritten by the incoming window.
    input  logic                                      i_window_start,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic                                      i_abort,

    output logic                                      o_contribution_valid,
    input  logic                                      i_contribution_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_contribution_local_addr,
    output logic                                      o_contribution_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_contribution_sample_id,
    output logic signed [63:0]                        o_dpre_q40,
    output logic signed [31:0]                        o_backbone_q14,
    output logic                                      o_contribution_tensor_last,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_protocol_error
);

    localparam int DPRE_DEPTH = 2 * BATCH_MAX * HIDDEN_MAX;
    localparam int BACKBONE_DEPTH = 2 * BATCH_MAX * BACKBONE_MAX;
    localparam int DPRE_ADDR_WIDTH = $clog2(DPRE_DEPTH);
    localparam int BACKBONE_ADDR_WIDTH = $clog2(BACKBONE_DEPTH);

    (* ram_style = "block" *) logic signed [63:0]
        dpre_mem [0:DPRE_DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0]
        backbone_mem [0:BACKBONE_DEPTH-1];

    logic configured_q;
    logic active_q;
    logic output_valid_q;
    logic dpre_complete_q;
    logic backbone_complete_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_count_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_count_q;
    logic [DPRE_ADDR_WIDTH-1:0] batch_hidden_span_q;
    logic [BACKBONE_ADDR_WIDTH-1:0] batch_backbone_span_q;
    logic [PARAM_ADDR_WIDTH-1:0] element_count_q;

    logic dpre_expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] dpre_expected_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] dpre_expected_hidden_q;
    logic [DPRE_ADDR_WIDTH-1:0] dpre_write_addr_q;
    logic backbone_expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] backbone_expected_sample_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_expected_id_q;
    logic [BACKBONE_ADDR_WIDTH-1:0] backbone_write_addr_q;

    logic emit_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] emit_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] emit_hidden_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] emit_backbone_q;
    logic [PARAM_ADDR_WIDTH-1:0] emit_local_addr_q;
    logic [DPRE_ADDR_WIDTH-1:0] emit_dpre_addr_q;
    logic [BACKBONE_ADDR_WIDTH-1:0] emit_backbone_addr_q;

    logic [PARAM_ADDR_WIDTH-1:0] output_local_addr_q;
    logic output_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] output_sample_q;
    logic signed [63:0] output_dpre_q;
    logic signed [31:0] output_backbone_q;
    logic output_last_q;

    logic cfg_legal;
    logic dpre_expected_last;
    logic backbone_expected_last;
    logic dpre_input_legal;
    logic backbone_input_legal;
    logic dpre_fire;
    logic backbone_fire;
    logic read_issue;
    logic output_fire;
    logic emit_last;

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= SAMPLE_ID_WIDTH'(BATCH_MAX)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= HIDDEN_INDEX_WIDTH'(HIDDEN_MAX)) &&
            (i_cfg_backbone_count != '0) &&
            (i_cfg_backbone_count <= BACKBONE_INDEX_WIDTH'(BACKBONE_MAX));
        o_cfg_ready = !active_q && !output_valid_q;

        dpre_expected_last = dpre_expected_path_q &&
            (dpre_expected_sample_q == (batch_count_q - 1'b1)) &&
            (dpre_expected_hidden_q == (hidden_count_q - 1'b1));
        dpre_input_legal = (i_dpre_path == dpre_expected_path_q) &&
            (i_dpre_sample_id == dpre_expected_sample_q) &&
            (i_dpre_hidden_id == dpre_expected_hidden_q) &&
            (i_dpre_last == dpre_expected_last);
        o_dpre_ready = configured_q && !dpre_complete_q && !active_q &&
            !output_valid_q;
        dpre_fire = i_dpre_valid && o_dpre_ready;

        backbone_expected_last = backbone_expected_path_q &&
            (backbone_expected_sample_q == (batch_count_q - 1'b1)) &&
            (backbone_expected_id_q == (backbone_count_q - 1'b1));
        backbone_input_legal =
            (i_backbone_path == backbone_expected_path_q) &&
            (i_backbone_sample_id == backbone_expected_sample_q) &&
            (i_backbone_id == backbone_expected_id_q) &&
            (i_backbone_last == backbone_expected_last);
        o_backbone_ready = configured_q && !backbone_complete_q &&
            !active_q && !output_valid_q;
        backbone_fire = i_backbone_valid && o_backbone_ready;

        o_start_ready = configured_q && dpre_complete_q &&
            backbone_complete_q && !o_protocol_error && !active_q &&
            !output_valid_q && !i_cfg_valid;
        o_busy = active_q || output_valid_q;

        emit_last = emit_path_q &&
            (emit_sample_q == (batch_count_q - 1'b1)) &&
            (emit_local_addr_q == (element_count_q - 1'b1));
        read_issue = active_q &&
            (!output_valid_q || i_contribution_ready);
        o_contribution_valid = output_valid_q;
        o_contribution_local_addr = output_local_addr_q;
        o_contribution_path = output_path_q;
        o_contribution_sample_id = output_sample_q;
        o_dpre_q40 = output_dpre_q;
        o_backbone_q14 = output_backbone_q;
        o_contribution_tensor_last = output_last_q;
        output_fire = o_contribution_valid && i_contribution_ready;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            active_q <= 1'b0;
            output_valid_q <= 1'b0;
            dpre_complete_q <= 1'b0;
            backbone_complete_q <= 1'b0;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            backbone_count_q <= '0;
            batch_hidden_span_q <= '0;
            batch_backbone_span_q <= '0;
            element_count_q <= '0;
            dpre_expected_path_q <= 1'b0;
            dpre_expected_sample_q <= '0;
            dpre_expected_hidden_q <= '0;
            dpre_write_addr_q <= '0;
            backbone_expected_path_q <= 1'b0;
            backbone_expected_sample_q <= '0;
            backbone_expected_id_q <= '0;
            backbone_write_addr_q <= '0;
            emit_path_q <= 1'b0;
            emit_sample_q <= '0;
            emit_hidden_q <= '0;
            emit_backbone_q <= '0;
            emit_local_addr_q <= '0;
            emit_dpre_addr_q <= '0;
            emit_backbone_addr_q <= '0;
            output_local_addr_q <= '0;
            output_path_q <= 1'b0;
            output_sample_q <= '0;
            output_dpre_q <= '0;
            output_backbone_q <= '0;
            output_last_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    hidden_count_q <= i_cfg_hidden_count;
                    backbone_count_q <= i_cfg_backbone_count;
                    batch_hidden_span_q <=
                        DPRE_ADDR_WIDTH'(
                            i_cfg_batch_count * i_cfg_hidden_count);
                    batch_backbone_span_q <=
                        BACKBONE_ADDR_WIDTH'(
                            i_cfg_batch_count * i_cfg_backbone_count);
                    element_count_q <=
                        PARAM_ADDR_WIDTH'(
                            i_cfg_hidden_count * i_cfg_backbone_count);
                    dpre_complete_q <= 1'b0;
                    backbone_complete_q <= 1'b0;
                    dpre_expected_path_q <= 1'b0;
                    dpre_expected_sample_q <= '0;
                    dpre_expected_hidden_q <= '0;
                    dpre_write_addr_q <= '0;
                    backbone_expected_path_q <= 1'b0;
                    backbone_expected_sample_q <= '0;
                    backbone_expected_id_q <= '0;
                    backbone_write_addr_q <= '0;
                    o_protocol_error <= 1'b0;
                end
            end

            if (i_window_start && configured_q && !active_q &&
                !output_valid_q) begin
                dpre_complete_q <= 1'b0;
                backbone_complete_q <= 1'b0;
                dpre_expected_path_q <= 1'b0;
                dpre_expected_sample_q <= '0;
                dpre_expected_hidden_q <= '0;
                dpre_write_addr_q <= '0;
                backbone_expected_path_q <= 1'b0;
                backbone_expected_sample_q <= '0;
                backbone_expected_id_q <= '0;
                backbone_write_addr_q <= '0;
                o_protocol_error <= 1'b0;
            end

            if (dpre_fire) begin
                if (!dpre_input_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    dpre_mem[dpre_write_addr_q] <= i_dpre_q40;
                    dpre_write_addr_q <= dpre_write_addr_q + 1'b1;
                    if (dpre_expected_last) begin
                        dpre_complete_q <= 1'b1;
                    end else if (dpre_expected_hidden_q ==
                        (hidden_count_q - 1'b1)) begin
                        dpre_expected_hidden_q <= '0;
                        if (dpre_expected_sample_q ==
                            (batch_count_q - 1'b1)) begin
                            dpre_expected_sample_q <= '0;
                            dpre_expected_path_q <= 1'b1;
                        end else begin
                            dpre_expected_sample_q <=
                                dpre_expected_sample_q + 1'b1;
                        end
                    end else begin
                        dpre_expected_hidden_q <=
                            dpre_expected_hidden_q + 1'b1;
                    end
                end
            end

            if (backbone_fire) begin
                if (!backbone_input_legal) begin
                    o_protocol_error <= 1'b1;
                end else begin
                    backbone_mem[backbone_write_addr_q] <= i_backbone_q14;
                    backbone_write_addr_q <= backbone_write_addr_q + 1'b1;
                    if (backbone_expected_last) begin
                        backbone_complete_q <= 1'b1;
                    end else if (backbone_expected_id_q ==
                        (backbone_count_q - 1'b1)) begin
                        backbone_expected_id_q <= '0;
                        if (backbone_expected_sample_q ==
                            (batch_count_q - 1'b1)) begin
                            backbone_expected_sample_q <= '0;
                            backbone_expected_path_q <= 1'b1;
                        end else begin
                            backbone_expected_sample_q <=
                                backbone_expected_sample_q + 1'b1;
                        end
                    end else begin
                        backbone_expected_id_q <=
                            backbone_expected_id_q + 1'b1;
                    end
                end
            end

            if (i_start && o_start_ready) begin
                active_q <= 1'b1;
                emit_path_q <= 1'b0;
                emit_sample_q <= '0;
                emit_hidden_q <= '0;
                emit_backbone_q <= '0;
                emit_local_addr_q <= '0;
                emit_dpre_addr_q <= '0;
                emit_backbone_addr_q <= '0;
            end

            if (read_issue) begin
                output_valid_q <= 1'b1;
                output_local_addr_q <= emit_local_addr_q;
                output_path_q <= emit_path_q;
                output_sample_q <= emit_sample_q;
                output_dpre_q <= dpre_mem[emit_dpre_addr_q];
                output_backbone_q <= backbone_mem[emit_backbone_addr_q];
                output_last_q <= emit_last;

                if (emit_last) begin
                    active_q <= 1'b0;
                end else if (emit_sample_q !=
                    (batch_count_q - 1'b1)) begin
                    emit_sample_q <= emit_sample_q + 1'b1;
                    emit_dpre_addr_q <=
                        emit_dpre_addr_q +
                        DPRE_ADDR_WIDTH'(hidden_count_q);
                    emit_backbone_addr_q <=
                        emit_backbone_addr_q +
                        BACKBONE_ADDR_WIDTH'(backbone_count_q);
                end else if (!emit_path_q) begin
                    emit_path_q <= 1'b1;
                    emit_sample_q <= '0;
                    emit_dpre_addr_q <= batch_hidden_span_q +
                        DPRE_ADDR_WIDTH'(emit_hidden_q);
                    emit_backbone_addr_q <=
                        batch_backbone_span_q +
                        BACKBONE_ADDR_WIDTH'(emit_backbone_q);
                end else begin
                    emit_path_q <= 1'b0;
                    emit_sample_q <= '0;
                    emit_local_addr_q <= emit_local_addr_q + 1'b1;
                    if (emit_backbone_q == (backbone_count_q - 1'b1)) begin
                        emit_backbone_q <= '0;
                        emit_hidden_q <= emit_hidden_q + 1'b1;
                        emit_dpre_addr_q <=
                            DPRE_ADDR_WIDTH'(emit_hidden_q) + 1'b1;
                        emit_backbone_addr_q <= '0;
                    end else begin
                        emit_backbone_q <= emit_backbone_q + 1'b1;
                        emit_dpre_addr_q <= DPRE_ADDR_WIDTH'(emit_hidden_q);
                        emit_backbone_addr_q <=
                            BACKBONE_ADDR_WIDTH'(emit_backbone_q) + 1'b1;
                    end
                end
            end else if (output_fire) begin
                output_valid_q <= 1'b0;
            end

            if (output_fire && output_last_q)
                o_done <= 1'b1;

            if (i_abort && (active_q || output_valid_q)) begin
                active_q <= 1'b0;
                output_valid_q <= 1'b0;
                o_done <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
