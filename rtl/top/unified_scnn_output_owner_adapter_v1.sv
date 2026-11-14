`timescale 1ns/1ps
`default_nettype none

// Control-only bridge between final-layer dispatch, the Wave3 SCNN owner,
// and the existing SCNN OUTPUT request frontend. The layer descriptor is
// snapshotted before owner acquisition and remains stable until retirement.
module unified_scnn_output_owner_adapter_v1 #(
    parameter int VERSION_WIDTH = 16
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,
    input  logic                         i_abort,

    input  logic                         i_layer_valid,
    output logic                         o_layer_ready,
    input  logic [383:0]                 i_layer_words,
    input  logic                         i_input_slot,
    input  logic                         i_sample_reset,
    input  logic                         i_sample_last,
    input  logic [VERSION_WIDTH-1:0]     i_version,

    output logic                         o_wave3_start_valid,
    input  logic                         i_wave3_start_ready,
    output logic                         o_wave3_external_mode,
    output logic                         o_wave3_external_done,
    output logic                         o_wave3_clear_error,
    input  logic                         i_wave3_sticky_error,
    input  logic [7:0]                   i_wave3_error_code,

    output logic                         o_frontend_owner_enable,
    output logic                         o_frontend_clear_error,
    output logic                         o_frontend_layer_valid,
    input  logic                         i_frontend_layer_ready,
    output logic [383:0]                 o_frontend_layer_words,
    output logic                         o_frontend_input_slot,
    output logic                         o_frontend_sample_reset,
    output logic                         o_frontend_sample_last,
    output logic [VERSION_WIDTH-1:0]     o_frontend_version,
    input  logic                         i_frontend_busy,
    input  logic                         i_frontend_done,
    input  logic                         i_frontend_sticky_error,
    input  logic [7:0]                   i_frontend_error_code,
    input  logic                         i_public_result_retire,

    output logic                         o_frontend_enqueued,
    output logic                         o_result_enqueued,
    output logic                         o_layer_done,
    output logic                         o_owner_held,
    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_START_WHILE_BUSY = 8'h01;
    localparam logic [7:0] ERROR_DONE_ORDER = 8'h02;
    localparam logic [7:0] ERROR_ABORT = 8'h03;
    localparam logic [7:0] ERROR_RETIRE_ORDER = 8'h04;
    localparam logic [7:0] ERROR_FRONTEND_BASE = 8'h80;
    localparam logic [7:0] ERROR_WAVE3_BASE = 8'h40;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_OWNER_REQUEST,
        STATE_FRONTEND_START,
        STATE_RUN,
        STATE_WAIT_RETIRE,
        STATE_SUCCESS_RELEASE,
        STATE_CANCEL_RELEASE,
        STATE_CANCEL_DRAIN
    } state_t;

    state_t state_q;
    logic [383:0] layer_words_q;
    logic input_slot_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic local_sticky_error_q;
    logic [7:0] local_error_code_q;
    logic layer_fire;
    logic owner_fire;
    logic frontend_fire;
    logic frontend_enqueue_fire;
    logic retire_legal;
    logic owner_acquired;
    logic result_enqueued_q;

    assign o_layer_ready = state_q == STATE_IDLE && !i_frontend_busy &&
        !result_enqueued_q &&
        !local_sticky_error_q && !i_frontend_sticky_error &&
        !i_wave3_sticky_error && !i_abort;
    assign layer_fire = i_layer_valid && o_layer_ready;

    assign o_wave3_start_valid = state_q == STATE_OWNER_REQUEST &&
        !i_abort && !local_sticky_error_q && !i_wave3_sticky_error;
    assign owner_fire = o_wave3_start_valid && i_wave3_start_ready;
    assign o_wave3_external_mode = 1'b1;
    assign o_wave3_external_done = state_q == STATE_SUCCESS_RELEASE ||
        state_q == STATE_CANCEL_RELEASE;
    assign o_wave3_clear_error = i_clear_error;

    assign owner_acquired = state_q == STATE_FRONTEND_START ||
        state_q == STATE_RUN || state_q == STATE_WAIT_RETIRE ||
        state_q == STATE_SUCCESS_RELEASE ||
        state_q == STATE_CANCEL_RELEASE ||
        state_q == STATE_CANCEL_DRAIN;
    assign o_owner_held = owner_acquired;
    assign o_frontend_owner_enable =
        (state_q == STATE_FRONTEND_START || state_q == STATE_RUN ||
         state_q == STATE_WAIT_RETIRE ||
         state_q == STATE_SUCCESS_RELEASE) && !i_abort;
    assign o_frontend_clear_error = i_clear_error;
    assign o_frontend_layer_valid = state_q == STATE_FRONTEND_START &&
        !i_abort && !local_sticky_error_q &&
        !i_frontend_sticky_error && !i_wave3_sticky_error;
    assign frontend_fire = o_frontend_layer_valid &&
        i_frontend_layer_ready;
    assign frontend_enqueue_fire = state_q == STATE_RUN &&
        i_frontend_done;
    assign retire_legal = result_enqueued_q || frontend_enqueue_fire;
    assign o_result_enqueued = result_enqueued_q;

    assign o_frontend_layer_words = layer_words_q;
    assign o_frontend_input_slot = input_slot_q;
    assign o_frontend_sample_reset = sample_reset_q;
    assign o_frontend_sample_last = sample_last_q;
    assign o_frontend_version = version_q;

    // Frontend done means that its result entered the shared result arbiter.
    // Final layer retirement is separately qualified at the public boundary.
    assign o_layer_done = state_q == STATE_SUCCESS_RELEASE;
    assign o_busy = state_q != STATE_IDLE || i_frontend_busy ||
        result_enqueued_q;
    assign o_idle = !o_busy;
    assign o_sticky_error = local_sticky_error_q ||
        i_frontend_sticky_error || i_wave3_sticky_error;
    always_comb begin
        if (local_sticky_error_q)
            o_error_code = local_error_code_q;
        else if (i_frontend_sticky_error)
            o_error_code = ERROR_FRONTEND_BASE | i_frontend_error_code;
        else if (i_wave3_sticky_error)
            o_error_code = ERROR_WAVE3_BASE | i_wave3_error_code;
        else
            o_error_code = ERROR_NONE;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            layer_words_q <= '0;
            input_slot_q <= 1'b0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            version_q <= '0;
            local_sticky_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE &&
                !i_frontend_busy) begin
                local_sticky_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (state_q == STATE_SUCCESS_RELEASE ||
                state_q == STATE_CANCEL_RELEASE) begin
                state_q <= STATE_IDLE;
            end else if (i_abort && state_q != STATE_IDLE) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_ABORT;
                if (!owner_acquired)
                    state_q <= STATE_IDLE;
                else if (i_frontend_busy)
                    state_q <= STATE_CANCEL_DRAIN;
                else
                    state_q <= STATE_CANCEL_RELEASE;
            end else if (i_frontend_sticky_error ||
                         i_wave3_sticky_error) begin
                local_sticky_error_q <= 1'b1;
                if (i_frontend_sticky_error)
                    local_error_code_q <= ERROR_FRONTEND_BASE |
                                          i_frontend_error_code;
                else
                    local_error_code_q <= ERROR_WAVE3_BASE |
                                          i_wave3_error_code;
                if (!owner_acquired)
                    state_q <= STATE_IDLE;
                else if (i_frontend_busy)
                    state_q <= STATE_CANCEL_DRAIN;
                else
                    state_q <= STATE_CANCEL_RELEASE;
            end else if (i_public_result_retire && !retire_legal) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_RETIRE_ORDER;
                state_q <= owner_acquired ? STATE_CANCEL_RELEASE :
                                            STATE_IDLE;
            end else if (i_frontend_done && state_q != STATE_RUN) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_DONE_ORDER;
                state_q <= owner_acquired && i_frontend_busy ?
                    STATE_CANCEL_DRAIN :
                    (owner_acquired ? STATE_CANCEL_RELEASE : STATE_IDLE);
            end else begin
                unique case (state_q)
                    STATE_IDLE: begin
                        if (layer_fire) begin
                            layer_words_q <= i_layer_words;
                            input_slot_q <= i_input_slot;
                            sample_reset_q <= i_sample_reset;
                            sample_last_q <= i_sample_last;
                            version_q <= i_version;
                            state_q <= STATE_OWNER_REQUEST;
                        end
                    end
                    STATE_OWNER_REQUEST: begin
                        if (i_layer_valid && !local_sticky_error_q) begin
                            local_sticky_error_q <= 1'b1;
                            local_error_code_q <= ERROR_START_WHILE_BUSY;
                            state_q <= STATE_IDLE;
                        end else if (owner_fire) begin
                            state_q <= STATE_FRONTEND_START;
                        end
                    end
                    STATE_FRONTEND_START: begin
                        if (i_layer_valid && !local_sticky_error_q) begin
                            local_sticky_error_q <= 1'b1;
                            local_error_code_q <= ERROR_START_WHILE_BUSY;
                            state_q <= STATE_CANCEL_RELEASE;
                        end else if (frontend_fire) begin
                            state_q <= STATE_RUN;
                        end
                    end
                    STATE_RUN: begin
                        if (i_layer_valid && !local_sticky_error_q) begin
                            local_sticky_error_q <= 1'b1;
                            local_error_code_q <= ERROR_START_WHILE_BUSY;
                            state_q <= STATE_CANCEL_RELEASE;
                        end else if (frontend_enqueue_fire)
                            state_q <= i_public_result_retire ?
                                STATE_SUCCESS_RELEASE : STATE_WAIT_RETIRE;
                    end
                    STATE_WAIT_RETIRE: begin
                        if (i_public_result_retire)
                            state_q <= STATE_SUCCESS_RELEASE;
                    end
                    // i_frontend_busy is the frontend contract's physical
                    // drain witness. It must cover any outstanding LI or
                    // result response that is being consumed and dropped.
                    STATE_CANCEL_DRAIN: begin
                        if (!i_frontend_busy)
                            state_q <= STATE_CANCEL_RELEASE;
                    end
                    default: state_q <= STATE_IDLE;
                endcase
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            result_enqueued_q <= 1'b0;
            o_frontend_enqueued <= 1'b0;
        end else begin
            o_frontend_enqueued <= 1'b0;
            // Cancel means the shared arbiter has flushed this queued result.
            // Do not wait for a public retire that can no longer occur.
            if (i_abort) begin
                result_enqueued_q <= 1'b0;
            end else if (frontend_enqueue_fire) begin
                o_frontend_enqueued <= 1'b1;
                result_enqueued_q <= !i_public_result_retire;
            end else if (i_public_result_retire && result_enqueued_q) begin
                result_enqueued_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
