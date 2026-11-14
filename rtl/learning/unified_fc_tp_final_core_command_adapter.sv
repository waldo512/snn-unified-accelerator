`timescale 1ns/1ps
`default_nettype none

// Connects the FC learning bridge's event-window stream to one shared final
// core command. The first stable window request launches the physical sample;
// windows stream only after that launch is accepted. Runtime retirement stays
// downstream of the bridge's per-sample learning completion.
module unified_fc_tp_final_core_command_adapter #(
    parameter int LANES = 8,
    parameter int SAMPLE_ID_WIDTH = 16,
    parameter int FEATURE_INDEX_WIDTH = 12
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_clear_error,

    input  logic                                  i_runtime_sample_start_allowed,
    input  logic                                  i_runtime_abort,
    output logic                                  o_runtime_abort_complete,
    output logic                                  o_runtime_sample_retired,
    output logic                                  o_runtime_batch_learning_done,
    output logic                                  o_execution_quiescent,
    output logic                                  o_runtime_error,

    input  logic                                  i_infer_window_valid,
    output logic                                  o_infer_window_ready,
    input  logic [LANES-1:0]                     i_infer_record_valid,
    input  logic [SAMPLE_ID_WIDTH-1:0]           i_infer_window_sample_id,
    input  logic [8:0]                            i_infer_window_timestep,
    input  logic [LANES-1:0]                     i_infer_boundary_mask,
    input  logic                                  i_infer_window_sample_last,
    input  logic [LANES*FEATURE_INDEX_WIDTH-1:0] i_infer_addr_flat,
    input  logic [LANES*8-1:0]                   i_infer_data_flat,

    output logic                                  o_core_event_window_valid,
    input  logic                                  i_core_event_window_ready,
    output logic [LANES-1:0]                     o_core_event_record_valid,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_core_event_window_sample_id,
    output logic [8:0]                            o_core_event_window_timestep,
    output logic [LANES-1:0]                     o_core_event_boundary_mask,
    output logic                                  o_core_event_window_sample_last,
    output logic [LANES*FEATURE_INDEX_WIDTH-1:0] o_core_event_addr_flat,
    output logic [LANES*8-1:0]                   o_core_event_data_flat,

    output logic                                  o_core_start_valid,
    input  logic                                  i_core_start_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]           o_core_sample_id,
    output logic                                  o_core_sample_state_reset,
    output logic                                  o_core_fc_clear_state,
    output logic                                  o_core_sample_last,
    input  logic                                  i_core_sample_done,
    input  logic                                  i_core_aborted,
    output logic                                  o_bridge_sample_compute_done,
    input  logic                                  i_bridge_sample_learning_done,
    input  logic                                  i_bridge_batch_learning_done,
    input  logic                                  i_bridge_window_busy,
    input  logic                                  i_bridge_parameter_quiescent,
    input  logic                                  i_bridge_error,

    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    typedef enum logic [2:0] {
        ST_WAIT_WINDOW,
        ST_START,
        ST_STREAM,
        ST_WAIT_CORE,
        ST_WAIT_RETIRE,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [SAMPLE_ID_WIDTH-1:0] sample_id_q;
    logic [8:0] timestep_q;
    logic timestep_closed_q;
    logic abort_pending_q;
    logic window_fire;
    logic core_start_fire;
    logic sample_context_mismatch;
    logic [9:0] expected_timestep;

    assign o_core_event_window_valid = i_infer_window_valid &&
        state_q == ST_STREAM && !o_protocol_error;
    assign o_infer_window_ready = state_q == ST_STREAM &&
        !o_protocol_error && i_core_event_window_ready;
    assign window_fire = o_core_event_window_valid &&
        i_core_event_window_ready;
    assign expected_timestep = timestep_closed_q ?
        {1'b0, timestep_q} + 1'b1 : {1'b0, timestep_q};
    assign sample_context_mismatch =
        i_infer_window_sample_id != sample_id_q ||
        expected_timestep[9] ||
        i_infer_window_timestep != expected_timestep[8:0];

    assign o_core_event_record_valid = i_infer_record_valid;
    assign o_core_event_window_sample_id = i_infer_window_sample_id;
    assign o_core_event_window_timestep = i_infer_window_timestep;
    assign o_core_event_boundary_mask = i_infer_boundary_mask;
    assign o_core_event_window_sample_last = i_infer_window_sample_last;
    assign o_core_event_addr_flat = i_infer_addr_flat;
    assign o_core_event_data_flat = i_infer_data_flat;

    assign o_core_start_valid = state_q == ST_START &&
        !i_runtime_abort && !o_protocol_error;
    assign core_start_fire = o_core_start_valid && i_core_start_ready;
    assign o_core_sample_id = sample_id_q;
    assign o_core_sample_state_reset = 1'b1;
    assign o_core_fc_clear_state = 1'b1;
    assign o_core_sample_last = 1'b1;
    assign o_bridge_sample_compute_done = state_q == ST_WAIT_CORE &&
        i_core_sample_done && !i_core_aborted;
    assign o_runtime_sample_retired = state_q == ST_WAIT_RETIRE &&
        i_bridge_sample_learning_done && !abort_pending_q &&
        !i_runtime_abort;
    assign o_runtime_batch_learning_done =
        o_runtime_sample_retired && i_bridge_batch_learning_done;
    assign o_runtime_abort_complete = i_runtime_abort &&
        ((state_q == ST_WAIT_WINDOW) || (state_q == ST_START) ||
         (state_q == ST_WAIT_RETIRE &&
          i_bridge_sample_learning_done));
    assign o_execution_quiescent = state_q == ST_WAIT_WINDOW &&
        !i_bridge_window_busy && i_bridge_parameter_quiescent &&
        !o_protocol_error;
    assign o_runtime_error = o_protocol_error || i_bridge_error;
    assign o_busy = state_q != ST_WAIT_WINDOW;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_WAIT_WINDOW;
            sample_id_q <= '0;
            timestep_q <= '0;
            timestep_closed_q <= 1'b0;
            abort_pending_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error && state_q == ST_ERROR &&
                !i_bridge_window_busy && i_bridge_parameter_quiescent &&
                !i_bridge_error) begin
                state_q <= ST_WAIT_WINDOW;
                timestep_closed_q <= 1'b0;
                abort_pending_q <= 1'b0;
                o_protocol_error <= 1'b0;
            end else if (i_bridge_error) begin
                o_protocol_error <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_WAIT_WINDOW: begin
                        abort_pending_q <= 1'b0;
                        timestep_closed_q <= 1'b0;
                        if (i_infer_window_valid &&
                            i_runtime_sample_start_allowed &&
                            !i_runtime_abort) begin
                            sample_id_q <= i_infer_window_sample_id;
                            timestep_q <= i_infer_window_timestep;
                            state_q <= ST_START;
                        end
                    end

                    ST_START: begin
                        if (i_runtime_abort) begin
                            state_q <= ST_WAIT_WINDOW;
                        end else if (!i_infer_window_valid ||
                                     i_infer_window_sample_id != sample_id_q ||
                                     i_infer_window_timestep != timestep_q) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (core_start_fire) begin
                            state_q <= ST_STREAM;
                        end
                    end

                    ST_STREAM: begin
                        if (i_runtime_abort)
                            abort_pending_q <= 1'b1;
                        if (window_fire) begin
                            if (sample_context_mismatch ||
                                (i_infer_window_sample_last &&
                                 !(|i_infer_boundary_mask))) begin
                                o_protocol_error <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                if (timestep_closed_q)
                                    timestep_q <= i_infer_window_timestep;
                                timestep_closed_q <=
                                    |i_infer_boundary_mask;
                                if (i_infer_window_sample_last)
                                    state_q <= ST_WAIT_CORE;
                            end
                        end
                    end

                    ST_WAIT_CORE: begin
                        if (i_runtime_abort)
                            abort_pending_q <= 1'b1;
                        if (i_core_aborted) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (i_core_sample_done) begin
                            state_q <= ST_WAIT_RETIRE;
                        end
                    end

                    ST_WAIT_RETIRE: begin
                        if (i_runtime_abort)
                            abort_pending_q <= 1'b1;
                        if (i_core_sample_done || i_core_aborted) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (i_bridge_batch_learning_done &&
                                     !i_bridge_sample_learning_done) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (i_bridge_sample_learning_done) begin
                            abort_pending_q <= 1'b0;
                            timestep_closed_q <= 1'b0;
                            state_q <= ST_WAIT_WINDOW;
                        end
                    end

                    default: begin
                        o_protocol_error <= 1'b1;
                        state_q <= ST_ERROR;
                    end
                endcase
            end
        end
    end

    initial begin
        if (LANES < 1 || SAMPLE_ID_WIDTH < 1 || FEATURE_INDEX_WIDTH < 1)
            $error("FC final-core command adapter widths are invalid");
    end
endmodule

`default_nettype wire
