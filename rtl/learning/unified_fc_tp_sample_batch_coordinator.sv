`timescale 1ns/1ps
`default_nettype none

// Sequences one FC sample from completed inference through event drain and
// observation snapshot. The final sample seals the batch and holds inference
// until the TP parameter commit is accepted, making updates forward-visible.
module unified_fc_tp_sample_batch_coordinator #(
    parameter int MAX_BATCH = 16,
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_clear_error,

    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [BATCH_COUNT_WIDTH-1:0]          i_cfg_batch_count,
    input  logic                                  i_release,
    output logic                                  o_release_ready,
    output logic                                  o_active,

    input  logic                                  i_learning_enable,
    input  logic                                  i_sample_compute_done,

    output logic                                  o_event_drain_start,
    input  logic                                  i_event_drain_start_ready,
    input  logic                                  i_event_drain_done,
    output logic                                  o_event_drain_done_ready,

    output logic                                  o_snapshot_start,
    input  logic                                  i_snapshot_start_ready,
    input  logic                                  i_snapshot_done,

    output logic                                  o_batch_seal,
    input  logic                                  i_batch_seal_ready,
    input  logic                                  i_window_commit_valid,
    input  logic                                  i_window_commit_accept_ready,
    output logic                                  o_window_commit_ready,

    output logic                                  o_sample_learning_done,
    output logic                                  o_batch_learning_done,
    output logic                                  o_inference_hold,
    output logic [BATCH_COUNT_WIDTH-1:0]          o_captured_sample_count,
    output logic                                  o_protocol_error
);
    typedef enum logic [3:0] {
        STATE_WAIT_SAMPLE,
        STATE_START_DRAIN,
        STATE_WAIT_DRAIN,
        STATE_START_SNAPSHOT,
        STATE_WAIT_SNAPSHOT,
        STATE_ACK_DRAIN,
        STATE_START_SEAL,
        STATE_WAIT_COMMIT
    } state_t;

    state_t state_q;
    logic configured_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [BATCH_COUNT_WIDTH-1:0] captured_samples_q;
    logic final_sample_q;
    logic protocol_error_q;
    logic cfg_legal;
    logic cfg_fire;
    logic release_fire;
    logic drain_start_fire;
    logic snapshot_start_fire;
    logic batch_seal_fire;
    logic commit_fire;

    generate
        if (MAX_BATCH == ((1 << BATCH_COUNT_WIDTH) - 1)) begin : g_full_range
            assign cfg_legal = i_cfg_batch_count != 0;
        end else begin : g_bounded
            assign cfg_legal = (i_cfg_batch_count != 0) &&
                (i_cfg_batch_count <= BATCH_COUNT_WIDTH'(MAX_BATCH));
        end
    endgenerate

    always_comb begin
        o_cfg_ready = !configured_q && state_q == STATE_WAIT_SAMPLE;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_release_ready = configured_q && state_q == STATE_WAIT_SAMPLE &&
            captured_samples_q == 0;
        release_fire = i_release && o_release_ready;

        o_event_drain_start = state_q == STATE_START_DRAIN;
        drain_start_fire = o_event_drain_start && i_event_drain_start_ready;
        o_event_drain_done_ready = state_q == STATE_ACK_DRAIN;
        o_snapshot_start = state_q == STATE_START_SNAPSHOT;
        snapshot_start_fire = o_snapshot_start && i_snapshot_start_ready;
        o_batch_seal = state_q == STATE_START_SEAL;
        batch_seal_fire = o_batch_seal && i_batch_seal_ready;
        o_window_commit_ready = (state_q == STATE_WAIT_COMMIT) &&
            i_window_commit_accept_ready;
        commit_fire = i_window_commit_valid && o_window_commit_ready;

        o_active = configured_q;
        o_inference_hold = configured_q && i_learning_enable &&
            state_q != STATE_WAIT_SAMPLE;
        o_captured_sample_count = captured_samples_q;
        o_protocol_error = protocol_error_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_WAIT_SAMPLE;
            configured_q <= 1'b0;
            batch_count_q <= '0;
            captured_samples_q <= '0;
            final_sample_q <= 1'b0;
            protocol_error_q <= 1'b0;
            o_sample_learning_done <= 1'b0;
            o_batch_learning_done <= 1'b0;
        end else begin
            o_sample_learning_done <= 1'b0;
            o_batch_learning_done <= 1'b0;

            if (i_clear_error)
                protocol_error_q <= 1'b0;
            if ((cfg_fire && !cfg_legal) ||
                (i_sample_compute_done &&
                 state_q != STATE_WAIT_SAMPLE) ||
                (!i_learning_enable &&
                 state_q != STATE_WAIT_SAMPLE))
                protocol_error_q <= 1'b1;

            if (cfg_fire && cfg_legal) begin
                configured_q <= 1'b1;
                batch_count_q <= i_cfg_batch_count;
                captured_samples_q <= '0;
            end
            if (release_fire) begin
                configured_q <= 1'b0;
                batch_count_q <= '0;
                captured_samples_q <= '0;
            end

            unique case (state_q)
                STATE_WAIT_SAMPLE: begin
                    if (i_sample_compute_done) begin
                        if (!configured_q || !i_learning_enable) begin
                            o_sample_learning_done <= 1'b1;
                        end else begin
                            final_sample_q <=
                                ({1'b0, captured_samples_q} + 1'b1) >=
                                {1'b0, batch_count_q};
                            state_q <= STATE_START_DRAIN;
                        end
                    end
                end

                STATE_START_DRAIN: begin
                    if (drain_start_fire)
                        state_q <= STATE_WAIT_DRAIN;
                end

                STATE_WAIT_DRAIN: begin
                    if (i_event_drain_done)
                        state_q <= STATE_START_SNAPSHOT;
                end

                STATE_START_SNAPSHOT: begin
                    if (snapshot_start_fire)
                        state_q <= STATE_WAIT_SNAPSHOT;
                end

                STATE_WAIT_SNAPSHOT: begin
                    if (i_snapshot_done)
                        state_q <= STATE_ACK_DRAIN;
                end

                STATE_ACK_DRAIN: begin
                    captured_samples_q <= captured_samples_q + 1'b1;
                    if (final_sample_q) begin
                        state_q <= STATE_START_SEAL;
                    end else begin
                        o_sample_learning_done <= 1'b1;
                        state_q <= STATE_WAIT_SAMPLE;
                    end
                end

                STATE_START_SEAL: begin
                    if (batch_seal_fire)
                        state_q <= STATE_WAIT_COMMIT;
                end

                STATE_WAIT_COMMIT: begin
                    if (commit_fire) begin
                        captured_samples_q <= '0;
                        final_sample_q <= 1'b0;
                        o_sample_learning_done <= 1'b1;
                        o_batch_learning_done <= 1'b1;
                        state_q <= STATE_WAIT_SAMPLE;
                    end
                end

                default: state_q <= STATE_WAIT_SAMPLE;
            endcase
        end
    end

    initial begin
        if (MAX_BATCH < 1)
            $error("FC TP sample/batch coordinator MAX_BATCH is invalid");
    end
endmodule

`default_nettype wire
