`timescale 1ns/1ps
`default_nettype none

// Serializes the two trainable SRNN M2 tensors into the shared optimizer
// request protocol. Gradients remain Q40 and dense: an optimizer such as Adam
// must see zero-gradient addresses as well as nonzero addresses. The adapter
// commits an update window only after both tensor writeback completions arrive.
module unified_srnn_tp_dual_tensor_gradient_adapter #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int GRADIENT_WIDTH = 64,
    parameter int WINDOW_SEQUENCE_WIDTH = 32
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wh1_tensor_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wh1_element_count,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_wout_element_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_start_window,
    output logic                                      o_start_ready,
    input  logic                                      i_abort_window,

    input  logic                                      i_wh1_gradient_valid,
    output logic                                      o_wh1_gradient_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wh1_gradient_addr,
    input  logic signed [GRADIENT_WIDTH-1:0]          i_wh1_gradient_q40,
    input  logic                                      i_wh1_gradient_last,

    input  logic                                      i_wout_gradient_valid,
    output logic                                      o_wout_gradient_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_wout_gradient_addr,
    input  logic signed [GRADIENT_WIDTH-1:0]          i_wout_gradient_q40,
    input  logic                                      i_wout_gradient_last,

    output logic                                      o_gradient_valid,
    input  logic                                      i_gradient_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_gradient_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_gradient_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_gradient_local_addr,
    output logic signed [GRADIENT_WIDTH-1:0]          o_gradient_q40,
    output logic                                      o_gradient_nonzero,
    output logic                                      o_gradient_tensor_last,
    output logic                                      o_gradient_update_window_last,

    input  logic                                      i_tensor_complete_valid,
    output logic                                      o_tensor_complete_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_tensor_complete_id,
    input  logic                                      i_tensor_complete_error,

    output logic                                      o_commit_valid,
    input  logic                                      i_commit_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_commit_profile_id,
    output logic [WINDOW_SEQUENCE_WIDTH-1:0]          o_commit_window_sequence,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_window_aborted,
    output logic                                      o_protocol_error,
    output logic                                      o_downstream_error
);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_WH1,
        STATE_WOUT,
        STATE_WAIT_COMPLETION,
        STATE_COMMIT,
        STATE_ABORT
    } state_t;

    state_t state_q;
    logic configured_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] wh1_tensor_id_q;
    logic [TENSOR_ID_WIDTH-1:0] wout_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_element_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] wout_element_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] expected_addr_q;
    logic [WINDOW_SEQUENCE_WIDTH-1:0] window_sequence_q;
    logic wh1_issued_q;
    logic wout_issued_q;
    logic wh1_complete_q;
    logic wout_complete_q;

    logic cfg_legal;
    logic selected_valid;
    logic selected_ready;
    logic selected_last;
    logic selected_expected_last;
    logic selected_legal;
    logic selected_fire;
    logic gradient_fire;
    logic wh1_last_fire;
    logic wout_last_fire;
    logic completion_fire;
    logic completion_is_wh1;
    logic completion_is_wout;
    logic completion_wh1_available;
    logic completion_wout_available;
    logic completion_legal;

    always_comb begin
        cfg_legal = (i_cfg_wh1_element_count != '0) &&
            (i_cfg_wout_element_count != '0) &&
            (i_cfg_wh1_tensor_id != i_cfg_wout_tensor_id);

        o_cfg_ready = state_q == STATE_IDLE;
        o_start_ready = (state_q == STATE_IDLE) && configured_q &&
            !i_cfg_valid;
        o_busy = state_q != STATE_IDLE;

        selected_valid = 1'b0;
        selected_ready = 1'b0;
        selected_last = 1'b0;
        selected_expected_last = 1'b0;
        selected_legal = 1'b0;

        o_wh1_gradient_ready = 1'b0;
        o_wout_gradient_ready = 1'b0;
        o_gradient_valid = 1'b0;
        o_gradient_profile_id = profile_id_q;
        o_gradient_tensor_id = '0;
        o_gradient_local_addr = '0;
        o_gradient_q40 = '0;
        o_gradient_nonzero = 1'b0;
        o_gradient_tensor_last = 1'b0;
        o_gradient_update_window_last = 1'b0;

        if (state_q == STATE_WH1) begin
            selected_valid = i_wh1_gradient_valid;
            selected_last = i_wh1_gradient_last;
            selected_expected_last =
                expected_addr_q == (wh1_element_count_q - 1'b1);
            selected_legal = (i_wh1_gradient_addr == expected_addr_q) &&
                (selected_last == selected_expected_last);
            selected_ready = selected_legal ? i_gradient_ready :
                i_wh1_gradient_valid;
            o_wh1_gradient_ready = selected_ready;
            o_gradient_valid = selected_valid && selected_legal;
            o_gradient_tensor_id = wh1_tensor_id_q;
            o_gradient_local_addr = i_wh1_gradient_addr;
            o_gradient_q40 = i_wh1_gradient_q40;
            o_gradient_nonzero = i_wh1_gradient_q40 != '0;
            o_gradient_tensor_last = selected_expected_last;
        end else if (state_q == STATE_WOUT) begin
            selected_valid = i_wout_gradient_valid;
            selected_last = i_wout_gradient_last;
            selected_expected_last =
                expected_addr_q == (wout_element_count_q - 1'b1);
            selected_legal = (i_wout_gradient_addr == expected_addr_q) &&
                (selected_last == selected_expected_last);
            selected_ready = selected_legal ? i_gradient_ready :
                i_wout_gradient_valid;
            o_wout_gradient_ready = selected_ready;
            o_gradient_valid = selected_valid && selected_legal;
            o_gradient_tensor_id = wout_tensor_id_q;
            o_gradient_local_addr = i_wout_gradient_addr;
            o_gradient_q40 = i_wout_gradient_q40;
            o_gradient_nonzero = i_wout_gradient_q40 != '0;
            o_gradient_tensor_last = selected_expected_last;
            o_gradient_update_window_last = selected_expected_last;
        end

        selected_fire = selected_valid && selected_ready;
        gradient_fire = o_gradient_valid && i_gradient_ready;
        wh1_last_fire = (state_q == STATE_WH1) && gradient_fire &&
            selected_expected_last;
        wout_last_fire = (state_q == STATE_WOUT) && gradient_fire &&
            selected_expected_last;

        o_tensor_complete_ready = (state_q != STATE_IDLE) &&
            (state_q != STATE_COMMIT) && (state_q != STATE_ABORT);
        completion_fire = i_tensor_complete_valid &&
            o_tensor_complete_ready;
        completion_is_wh1 = i_tensor_complete_id == wh1_tensor_id_q;
        completion_is_wout = i_tensor_complete_id == wout_tensor_id_q;
        completion_wh1_available = wh1_issued_q || wh1_last_fire;
        completion_wout_available = wout_issued_q || wout_last_fire;
        completion_legal =
            (completion_is_wh1 && completion_wh1_available &&
                !wh1_complete_q) ||
            (completion_is_wout && completion_wout_available &&
                !wout_complete_q);

        o_commit_valid = state_q == STATE_COMMIT;
        o_commit_profile_id = profile_id_q;
        o_commit_window_sequence = window_sequence_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            configured_q <= 1'b0;
            profile_id_q <= '0;
            wh1_tensor_id_q <= '0;
            wout_tensor_id_q <= '0;
            wh1_element_count_q <= '0;
            wout_element_count_q <= '0;
            expected_addr_q <= '0;
            window_sequence_q <= '0;
            wh1_issued_q <= 1'b0;
            wout_issued_q <= 1'b0;
            wh1_complete_q <= 1'b0;
            wout_complete_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_window_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
            o_downstream_error <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_window_aborted <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    profile_id_q <= i_cfg_profile_id;
                    wh1_tensor_id_q <= i_cfg_wh1_tensor_id;
                    wout_tensor_id_q <= i_cfg_wout_tensor_id;
                    wh1_element_count_q <= i_cfg_wh1_element_count;
                    wout_element_count_q <= i_cfg_wout_element_count;
                end
            end

            case (state_q)
                STATE_IDLE: begin
                    if (i_start_window && o_start_ready) begin
                        expected_addr_q <= '0;
                        wh1_issued_q <= 1'b0;
                        wout_issued_q <= 1'b0;
                        wh1_complete_q <= 1'b0;
                        wout_complete_q <= 1'b0;
                        o_protocol_error <= 1'b0;
                        o_downstream_error <= 1'b0;
                        state_q <= STATE_WH1;
                    end
                end
                STATE_WH1: begin
                    if (selected_fire) begin
                        if (!selected_legal) begin
                            o_protocol_error <= 1'b1;
                            state_q <= STATE_ABORT;
                        end else if (selected_expected_last) begin
                            expected_addr_q <= '0;
                            wh1_issued_q <= 1'b1;
                            state_q <= STATE_WOUT;
                        end else begin
                            expected_addr_q <= expected_addr_q + 1'b1;
                        end
                    end
                end
                STATE_WOUT: begin
                    if (selected_fire) begin
                        if (!selected_legal) begin
                            o_protocol_error <= 1'b1;
                            state_q <= STATE_ABORT;
                        end else if (selected_expected_last) begin
                            expected_addr_q <= '0;
                            wout_issued_q <= 1'b1;
                            state_q <= STATE_WAIT_COMPLETION;
                        end else begin
                            expected_addr_q <= expected_addr_q + 1'b1;
                        end
                    end
                end
                STATE_WAIT_COMPLETION: begin
                    if (wh1_complete_q && wout_complete_q)
                        state_q <= STATE_COMMIT;
                end
                STATE_COMMIT: begin
                    if (o_commit_valid && i_commit_ready) begin
                        window_sequence_q <= window_sequence_q + 1'b1;
                        o_done <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                end
                STATE_ABORT: begin
                    o_done <= 1'b1;
                    o_window_aborted <= 1'b1;
                    state_q <= STATE_IDLE;
                end
                default: state_q <= STATE_IDLE;
            endcase

            if (completion_fire) begin
                if (i_tensor_complete_error) begin
                    o_downstream_error <= 1'b1;
                    state_q <= STATE_ABORT;
                end else if (!completion_legal) begin
                    o_protocol_error <= 1'b1;
                    state_q <= STATE_ABORT;
                end else if (completion_is_wh1) begin
                    wh1_complete_q <= 1'b1;
                end else begin
                    wout_complete_q <= 1'b1;
                end
            end

            if (i_abort_window && (state_q != STATE_IDLE) &&
                (state_q != STATE_ABORT)) begin
                o_downstream_error <= 1'b1;
                state_q <= STATE_ABORT;
            end
        end
    end

endmodule

`default_nettype wire
