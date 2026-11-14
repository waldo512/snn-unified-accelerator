`timescale 1ns/1ps
`default_nettype none

// End-to-end SRNN M2 W_h1 frontend: pairwise TP terms produce dpre values,
// reset-free block RAMs retain dpre/backbone observations, and one shared
// reducer emits the dense Q40 gradient stream in parameter address order.
module unified_srnn_tp_wh1_online_frontend #(
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

    input  logic                                      i_start_window,
    output logic                                      o_start_ready,
    input  logic                                      i_abort_window,

    input  logic                                      i_tp_contribution_valid,
    output logic                                      o_tp_contribution_ready,
    input  logic                                      i_tp_contribution_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_tp_contribution_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_tp_contribution_hidden_id,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_tp_contribution_column_id,
    input  logic signed [63:0]                        i_tp_error_q40,
    input  logic signed [31:0]                        i_tp_opposite_trace_q14,
    input  logic [24:0]                              i_tp_surrogate_uq24,
    input  logic                                      i_tp_contribution_last,

    input  logic                                      i_backbone_valid,
    output logic                                      o_backbone_ready,
    input  logic                                      i_backbone_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_backbone_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_backbone_id,
    input  logic signed [31:0]                        i_backbone_q14,
    input  logic                                      i_backbone_last,

    output logic                                      o_gradient_valid,
    input  logic                                      i_gradient_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_gradient_local_addr,
    output logic signed [63:0]                        o_gradient_q40,
    output logic                                      o_gradient_tensor_last,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_window_aborted,
    output logic                                      o_protocol_error,
    output logic                                      o_overflow
);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_OBSERVE,
        STATE_REDUCE,
        STATE_ABORT
    } state_t;

    state_t state_q;
    logic generation_complete_q;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_element_count;
    logic shared_cfg_fire;
    logic observation_start_fire;
    logic reduction_start_fire;
    logic abort_request;

    logic generator_cfg_ready;
    logic generator_cfg_error;
    logic generator_start_ready;
    logic generator_dpre_valid;
    logic generator_dpre_ready;
    logic generator_dpre_path;
    logic [SAMPLE_ID_WIDTH-1:0] generator_dpre_sample;
    logic [HIDDEN_INDEX_WIDTH-1:0] generator_dpre_hidden;
    logic signed [63:0] generator_dpre_q40;
    logic generator_dpre_last;
    logic generator_busy;
    logic generator_done;
    logic generator_protocol_error;
    logic generator_overflow;
    logic generator_input_ready;

    logic scheduler_cfg_ready;
    logic scheduler_cfg_error;
    logic scheduler_dpre_ready;
    logic scheduler_backbone_ready;
    logic scheduler_start_ready;
    logic scheduler_contribution_valid;
    logic scheduler_contribution_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scheduler_contribution_addr;
    logic scheduler_contribution_path;
    logic [SAMPLE_ID_WIDTH-1:0] scheduler_contribution_sample;
    logic signed [63:0] scheduler_dpre_q40;
    logic signed [31:0] scheduler_backbone_q14;
    logic scheduler_contribution_last;
    logic scheduler_busy;
    logic scheduler_done;
    logic scheduler_protocol_error;

    logic reducer_cfg_ready;
    logic reducer_cfg_error;
    logic reducer_start_ready;
    logic reducer_busy;
    logic reducer_done;
    logic reducer_protocol_error;
    logic reducer_overflow;
    logic reducer_input_ready;

    always_comb begin
        wh1_element_count = PARAM_ADDR_WIDTH'(
            i_cfg_hidden_count * i_cfg_backbone_count);
        o_cfg_ready = (state_q == STATE_IDLE) && generator_cfg_ready &&
            scheduler_cfg_ready && reducer_cfg_ready;
        shared_cfg_fire = i_cfg_valid && o_cfg_ready;
        o_cfg_error = generator_cfg_error || scheduler_cfg_error ||
            reducer_cfg_error;

        o_start_ready = (state_q == STATE_IDLE) && generator_start_ready &&
            !i_cfg_valid;
        observation_start_fire = i_start_window && o_start_ready;
        reduction_start_fire = (state_q == STATE_OBSERVE) &&
            generation_complete_q && scheduler_start_ready &&
            reducer_start_ready;

        abort_request = i_abort_window || generator_protocol_error ||
            generator_overflow || scheduler_protocol_error ||
            reducer_protocol_error || reducer_overflow;
        o_protocol_error = generator_protocol_error ||
            scheduler_protocol_error || reducer_protocol_error;
        o_overflow = generator_overflow || reducer_overflow;
        o_busy = (state_q != STATE_IDLE) || generator_busy ||
            scheduler_busy || reducer_busy ||
            (scheduler_done && reducer_busy);

        o_tp_contribution_ready = (state_q == STATE_OBSERVE) &&
            generator_input_ready;
        generator_dpre_ready = (state_q == STATE_OBSERVE) &&
            scheduler_dpre_ready;
        o_backbone_ready = (state_q == STATE_OBSERVE) ?
            scheduler_backbone_ready : 1'b0;

        scheduler_contribution_ready = (state_q == STATE_REDUCE) &&
            reducer_input_ready;
    end

    // The generator's input ready is driven directly below; its output ready
    // is independently controlled by the scheduler RAM write port.
    unified_srnn_tp_wh1_dpre_generator #(
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH)
    ) dpre_generator (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(generator_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .o_cfg_error(generator_cfg_error),
        .i_start(observation_start_fire),
        .o_start_ready(generator_start_ready),
        .i_abort(abort_request),
        .i_contribution_valid(i_tp_contribution_valid &&
            (state_q == STATE_OBSERVE)),
        .o_contribution_ready(generator_input_ready),
        .i_contribution_path(i_tp_contribution_path),
        .i_contribution_sample_id(i_tp_contribution_sample_id),
        .i_contribution_hidden_id(i_tp_contribution_hidden_id),
        .i_contribution_column_id(i_tp_contribution_column_id),
        .i_error_q40(i_tp_error_q40),
        .i_opposite_trace_q14(i_tp_opposite_trace_q14),
        .i_surrogate_uq24(i_tp_surrogate_uq24),
        .i_contribution_tensor_last(i_tp_contribution_last),
        .o_dpre_valid(generator_dpre_valid),
        .i_dpre_ready(generator_dpre_ready),
        .o_dpre_path(generator_dpre_path),
        .o_dpre_sample_id(generator_dpre_sample),
        .o_dpre_hidden_id(generator_dpre_hidden),
        .o_dpre_q40(generator_dpre_q40),
        .o_dpre_tensor_last(generator_dpre_last),
        .o_busy(generator_busy),
        .o_done(generator_done),
        .o_protocol_error(generator_protocol_error),
        .o_overflow(generator_overflow)
    );

    unified_srnn_tp_wh1_contribution_scheduler #(
        .BATCH_MAX(BATCH_MAX),
        .HIDDEN_MAX(HIDDEN_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH),
        .HIDDEN_INDEX_WIDTH(HIDDEN_INDEX_WIDTH),
        .BACKBONE_INDEX_WIDTH(BACKBONE_INDEX_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH)
    ) contribution_scheduler (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(scheduler_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_hidden_count(i_cfg_hidden_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .o_cfg_error(scheduler_cfg_error),
        .i_dpre_valid(generator_dpre_valid &&
            (state_q == STATE_OBSERVE)),
        .o_dpre_ready(scheduler_dpre_ready),
        .i_dpre_path(generator_dpre_path),
        .i_dpre_sample_id(generator_dpre_sample),
        .i_dpre_hidden_id(generator_dpre_hidden),
        .i_dpre_q40(generator_dpre_q40),
        .i_dpre_last(generator_dpre_last),
        .i_backbone_valid(i_backbone_valid &&
            (state_q == STATE_OBSERVE)),
        .o_backbone_ready(scheduler_backbone_ready),
        .i_backbone_path(i_backbone_path),
        .i_backbone_sample_id(i_backbone_sample_id),
        .i_backbone_id(i_backbone_id),
        .i_backbone_q14(i_backbone_q14),
        .i_backbone_last(i_backbone_last),
        .i_window_start(observation_start_fire),
        .i_start(reduction_start_fire),
        .o_start_ready(scheduler_start_ready),
        .i_abort(abort_request),
        .o_contribution_valid(scheduler_contribution_valid),
        .i_contribution_ready(scheduler_contribution_ready),
        .o_contribution_local_addr(scheduler_contribution_addr),
        .o_contribution_path(scheduler_contribution_path),
        .o_contribution_sample_id(scheduler_contribution_sample),
        .o_dpre_q40(scheduler_dpre_q40),
        .o_backbone_q14(scheduler_backbone_q14),
        .o_contribution_tensor_last(scheduler_contribution_last),
        .o_busy(scheduler_busy),
        .o_done(scheduler_done),
        .o_protocol_error(scheduler_protocol_error)
    );

    unified_srnn_tp_wh1_gradient_reducer #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .SAMPLE_ID_WIDTH(SAMPLE_ID_WIDTH)
    ) gradient_reducer (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(shared_cfg_fire),
        .o_cfg_ready(reducer_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_element_count(wh1_element_count),
        .o_cfg_error(reducer_cfg_error),
        .i_start(reduction_start_fire),
        .o_start_ready(reducer_start_ready),
        .i_abort(abort_request),
        .i_contribution_valid(scheduler_contribution_valid &&
            (state_q == STATE_REDUCE)),
        .o_contribution_ready(reducer_input_ready),
        .i_contribution_local_addr(scheduler_contribution_addr),
        .i_contribution_path(scheduler_contribution_path),
        .i_contribution_sample_id(scheduler_contribution_sample),
        .i_dpre_q40(scheduler_dpre_q40),
        .i_backbone_q14(scheduler_backbone_q14),
        .i_contribution_tensor_last(scheduler_contribution_last),
        .o_gradient_valid(o_gradient_valid),
        .i_gradient_ready(i_gradient_ready),
        .o_gradient_local_addr(o_gradient_local_addr),
        .o_gradient_q40(o_gradient_q40),
        .o_gradient_tensor_last(o_gradient_tensor_last),
        .o_busy(reducer_busy),
        .o_done(reducer_done),
        .o_protocol_error(reducer_protocol_error),
        .o_overflow(reducer_overflow)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            generation_complete_q <= 1'b0;
            o_done <= 1'b0;
            o_window_aborted <= 1'b0;
        end else begin
            o_done <= 1'b0;
            o_window_aborted <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (observation_start_fire) begin
                        generation_complete_q <= 1'b0;
                        state_q <= STATE_OBSERVE;
                    end
                end
                STATE_OBSERVE: begin
                    if (generator_done)
                        generation_complete_q <= 1'b1;
                    if (reduction_start_fire)
                        state_q <= STATE_REDUCE;
                end
                STATE_REDUCE: begin
                    if (reducer_done) begin
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

            if (abort_request && (state_q != STATE_IDLE) &&
                (state_q != STATE_ABORT))
                state_q <= STATE_ABORT;
        end
    end

endmodule

`default_nettype wire
