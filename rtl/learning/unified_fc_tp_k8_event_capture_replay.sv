`timescale 1ns/1ps
`default_nettype none

// Mirrors accepted K8 inference windows into a per-sample packed buffer, then
// serializes their valid feature addresses for the scalar TP observation port.
// The learning-disabled path is a combinational pass-through and stores no data.
module unified_fc_tp_k8_event_capture_replay #(
    parameter int LANES = 8,
    parameter int FEATURE_INDEX_WIDTH = 12,
    parameter int SAMPLE_ID_WIDTH = 16,
    parameter int TIMESTEP_WIDTH = 9,
    parameter int EVENT_DATA_WIDTH = 8,
    parameter int MAX_WINDOWS_PER_SAMPLE = 512,
    parameter int WINDOW_ADDR_WIDTH =
        (MAX_WINDOWS_PER_SAMPLE <= 2) ? 1 : $clog2(MAX_WINDOWS_PER_SAMPLE),
    parameter int WINDOW_COUNT_WIDTH = $clog2(MAX_WINDOWS_PER_SAMPLE + 1),
    parameter int LANE_INDEX_WIDTH = (LANES <= 2) ? 1 : $clog2(LANES)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_learning_enable,

    input  logic                                      i_window_valid,
    output logic                                      o_window_ready,
    input  logic [LANES-1:0]                         i_record_valid,
    input  logic [SAMPLE_ID_WIDTH-1:0]               i_window_sample_id,
    input  logic [TIMESTEP_WIDTH-1:0]                i_window_timestep,
    input  logic [LANES-1:0]                         i_boundary_mask,
    input  logic                                      i_window_sample_last,
    input  logic [LANES*FEATURE_INDEX_WIDTH-1:0]     i_addr_flat,
    input  logic [LANES*EVENT_DATA_WIDTH-1:0]        i_data_flat,

    output logic                                      o_infer_window_valid,
    input  logic                                      i_infer_window_ready,
    output logic [LANES-1:0]                         o_infer_record_valid,
    output logic [SAMPLE_ID_WIDTH-1:0]               o_infer_window_sample_id,
    output logic [TIMESTEP_WIDTH-1:0]                o_infer_window_timestep,
    output logic [LANES-1:0]                         o_infer_boundary_mask,
    output logic                                      o_infer_window_sample_last,
    output logic [LANES*FEATURE_INDEX_WIDTH-1:0]     o_infer_addr_flat,
    output logic [LANES*EVENT_DATA_WIDTH-1:0]        o_infer_data_flat,

    input  logic                                      i_drain_start,
    output logic                                      o_drain_start_ready,
    output logic                                      o_event_valid,
    input  logic                                      i_event_ready,
    output logic [FEATURE_INDEX_WIDTH-1:0]            o_event_feature_index,
    output logic                                      o_drain_done,
    input  logic                                      i_drain_done_ready,

    output logic                                      o_capture_idle,
    output logic                                      o_busy,
    output logic [WINDOW_COUNT_WIDTH-1:0]             o_captured_window_count,
    output logic                                      o_protocol_error
);
    localparam int WINDOW_PAYLOAD_WIDTH =
        LANES + LANES * FEATURE_INDEX_WIDTH;

    typedef enum logic [1:0] {
        STATE_CAPTURE,
        STATE_LOAD,
        STATE_EMIT,
        STATE_DONE
    } state_t;

    (* ram_style = "block" *) logic [WINDOW_PAYLOAD_WIDTH-1:0]
        window_mem [0:MAX_WINDOWS_PER_SAMPLE-1];

    state_t state_q;
    logic [WINDOW_COUNT_WIDTH-1:0] window_count_q;
    logic [WINDOW_ADDR_WIDTH-1:0] replay_window_q;
    logic [WINDOW_PAYLOAD_WIDTH-1:0] replay_payload_q;
    logic sample_id_valid_q;
    logic [SAMPLE_ID_WIDTH-1:0] sample_id_q;
    logic protocol_error_q;

    logic window_fire;
    logic store_window;
    logic drain_start_fire;
    logic event_fire;
    logic selected_valid;
    logic [LANE_INDEX_WIDTH-1:0] selected_lane;
    logic [LANES-1:0] replay_mask;
    logic [LANES-1:0] replay_mask_after_fire;
    logic [LANES*FEATURE_INDEX_WIDTH-1:0] replay_addr_flat;

    always_comb begin
        o_infer_window_valid = i_window_valid && state_q == STATE_CAPTURE;
        o_infer_record_valid = i_record_valid;
        o_infer_window_sample_id = i_window_sample_id;
        o_infer_window_timestep = i_window_timestep;
        o_infer_boundary_mask = i_boundary_mask;
        o_infer_window_sample_last = i_window_sample_last;
        o_infer_addr_flat = i_addr_flat;
        o_infer_data_flat = i_data_flat;

        o_window_ready = (state_q == STATE_CAPTURE) &&
            i_infer_window_ready &&
            (!i_learning_enable || i_record_valid == '0 ||
             window_count_q < WINDOW_COUNT_WIDTH'(MAX_WINDOWS_PER_SAMPLE));
        window_fire = i_window_valid && o_window_ready;
        store_window = window_fire && i_learning_enable &&
            i_record_valid != '0;

        o_drain_start_ready = (state_q == STATE_CAPTURE) &&
            i_learning_enable && !i_window_valid;
        drain_start_fire = i_drain_start && o_drain_start_ready;

        replay_mask = replay_payload_q[0 +: LANES];
        replay_addr_flat = replay_payload_q[LANES +:
            LANES*FEATURE_INDEX_WIDTH];
        selected_valid = 1'b0;
        selected_lane = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            if (!selected_valid && replay_mask[lane]) begin
                selected_valid = 1'b1;
                selected_lane = LANE_INDEX_WIDTH'(lane);
            end
        end

        o_event_valid = (state_q == STATE_EMIT) && selected_valid;
        o_event_feature_index = replay_addr_flat[
            selected_lane*FEATURE_INDEX_WIDTH +: FEATURE_INDEX_WIDTH];
        event_fire = o_event_valid && i_event_ready;
        replay_mask_after_fire = replay_mask;
        if (event_fire)
            replay_mask_after_fire[selected_lane] = 1'b0;

        o_drain_done = state_q == STATE_DONE;
        o_capture_idle = (state_q == STATE_CAPTURE) &&
            window_count_q == 0;
        o_busy = !o_capture_idle;
        o_captured_window_count = window_count_q;
        o_protocol_error = protocol_error_q;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_CAPTURE;
            window_count_q <= '0;
            replay_window_q <= '0;
            replay_payload_q <= '0;
            sample_id_valid_q <= 1'b0;
            sample_id_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            if (i_clear_error)
                protocol_error_q <= 1'b0;

            if (!i_learning_enable &&
                (state_q != STATE_CAPTURE || window_count_q != 0))
                protocol_error_q <= 1'b1;

            if (store_window) begin
                window_mem[window_count_q[WINDOW_ADDR_WIDTH-1:0]] <=
                    {i_addr_flat, i_record_valid};
                window_count_q <= window_count_q + 1'b1;
                if (!sample_id_valid_q) begin
                    sample_id_valid_q <= 1'b1;
                    sample_id_q <= i_window_sample_id;
                end else if (i_window_sample_id != sample_id_q) begin
                    protocol_error_q <= 1'b1;
                end
            end

            unique case (state_q)
                STATE_CAPTURE: begin
                    if (drain_start_fire) begin
                        replay_window_q <= '0;
                        if (window_count_q == 0)
                            state_q <= STATE_DONE;
                        else
                            state_q <= STATE_LOAD;
                    end
                end

                STATE_LOAD: begin
                    replay_payload_q <= window_mem[replay_window_q];
                    state_q <= STATE_EMIT;
                end

                STATE_EMIT: begin
                    if (!selected_valid) begin
                        if ((WINDOW_COUNT_WIDTH'(replay_window_q) + 1'b1) >=
                            window_count_q) begin
                            state_q <= STATE_DONE;
                        end else begin
                            replay_window_q <= replay_window_q + 1'b1;
                            state_q <= STATE_LOAD;
                        end
                    end else if (event_fire) begin
                        replay_payload_q[0 +: LANES] <= replay_mask_after_fire;
                        if (replay_mask_after_fire == '0) begin
                            if ((WINDOW_COUNT_WIDTH'(replay_window_q) + 1'b1) >=
                                window_count_q) begin
                                state_q <= STATE_DONE;
                            end else begin
                                replay_window_q <= replay_window_q + 1'b1;
                                state_q <= STATE_LOAD;
                            end
                        end
                    end
                end

                STATE_DONE: begin
                    if (i_drain_done_ready) begin
                        state_q <= STATE_CAPTURE;
                        window_count_q <= '0;
                        replay_window_q <= '0;
                        replay_payload_q <= '0;
                        sample_id_valid_q <= 1'b0;
                        sample_id_q <= '0;
                    end
                end

                default: state_q <= STATE_CAPTURE;
            endcase
        end
    end

endmodule

`default_nettype wire
