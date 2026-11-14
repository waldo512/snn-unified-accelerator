`timescale 1ns/1ps
`default_nettype none

// Orders one SRNN runtime transaction as:
//   B recurrent backbone commands -> complete BxT capture -> M2 replay.
// The bridge contains no numeric or parameter state. It only prevents the M2
// timestep client from starting before the live sample-major tape is complete.
module unified_srnn_m2_live_runtime_capture_bridge #(
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int BACKBONE_MAX = 450,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [$clog2(BATCH_MAX+1)-1:0]            i_cfg_batch_count,
    input  logic [$clog2(TIMESTEP_MAX+1)-1:0]         i_cfg_timestep_count,
    input  logic [$clog2(BACKBONE_MAX+1)-1:0]         i_cfg_backbone_count,
    input  logic [VERSION_WIDTH-1:0]                  i_cfg_expected_version,
    input  logic                                      i_cfg_check_version,
    output logic                                      o_runtime_configured,

    input  logic                                      i_runtime_client_active,
    input  logic                                      i_runtime_timestep_start,
    output logic                                      o_runtime_timestep_start_ready,
    output logic                                      o_coordinator_timestep_start,
    input  logic                                      i_coordinator_timestep_start_ready,

    input  logic                                      i_backbone_sample_start_valid,
    output logic                                      o_backbone_sample_start_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_backbone_sample_id,
    output logic                                      o_core_start_valid,
    input  logic                                      i_core_start_ready,
    input  logic                                      i_core_done,
    input  logic                                      i_core_aborted,
    output logic                                      o_core_abort,

    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_commit_version,
    input  logic                                      i_commit_sample_last,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_commit_spikes,

    input  logic                                      i_replay_start,
    output logic                                      o_replay_start_ready,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_replay_timestep_id,
    output logic                                      o_replay_valid,
    input  logic                                      i_replay_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_replay_sample_id,
    output logic [SOURCE_VECTOR_MAX-1:0]              o_replay_spikes,
    output logic                                      o_replay_done,

    output logic [$clog2(BATCH_MAX+1)-1:0]            o_started_sample_count,
    output logic [$clog2(BATCH_MAX+1)-1:0]            o_completed_sample_count,
    output logic                                      o_capture_complete,
    output logic                                      o_capture_busy,
    output logic                                      o_quiescent,
    output logic                                      o_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_CAPTURE_START,
        ST_CAPTURE,
        ST_REPLAY_READY,
        ST_ABORT_DRAIN,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [$clog2(BATCH_MAX+1)-1:0] batch_count_q;
    logic [$clog2(BATCH_MAX+1)-1:0] started_count_q;
    logic [$clog2(BATCH_MAX+1)-1:0] completed_count_q;
    logic [$clog2(BATCH_MAX+1)-1:0] sample_last_count_q;
    logic core_inflight_q;
    logic capture_done_seen_q;
    logic local_error_q;

    logic service_cfg_ready;
    logic service_cfg_error;
    logic service_capture_start;
    logic service_capture_start_ready;
    logic service_commit_ready;
    logic service_capture_done;
    logic service_busy;
    logic service_configured;
    logic service_protocol_error;
    logic service_version_error;
    logic service_tail_error;
    logic service_abort;
    logic cfg_fire;
    logic sample_start_eligible;
    logic sample_start_fire;
    logic core_done_fire;
    logic core_aborted_fire;
    logic commit_fire;
    logic capture_counts_complete;

    assign o_cfg_ready = state_q == ST_IDLE && !i_runtime_client_active &&
        service_cfg_ready && !local_error_q;
    assign cfg_fire = i_cfg_valid && o_cfg_ready;
    // Configuration is runtime context, not an idle-only ready pulse. The
    // common SRNN client requires it to remain asserted for the whole run.
    assign o_runtime_configured = service_configured &&
        batch_count_q != '0 && !o_error;

    assign o_backbone_sample_id =
        $clog2(BATCH_MAX)'(started_count_q);
    assign sample_start_eligible = state_q == ST_CAPTURE &&
        !core_inflight_q && started_count_q < batch_count_q && !o_error;
    assign o_core_start_valid = sample_start_eligible &&
        i_backbone_sample_start_valid;
    assign o_backbone_sample_start_ready = sample_start_eligible &&
        i_core_start_ready;
    assign sample_start_fire = o_core_start_valid && i_core_start_ready;
    assign core_done_fire = i_core_done && core_inflight_q;
    assign core_aborted_fire = i_core_aborted && core_inflight_q;

    assign service_capture_start = state_q == ST_CAPTURE_START &&
        service_capture_start_ready && !i_abort;
    assign service_abort = i_abort || state_q == ST_ABORT_DRAIN ||
        state_q == ST_ERROR;
    assign o_commit_ready = state_q == ST_CAPTURE && service_commit_ready;
    assign commit_fire = i_commit_valid && o_commit_ready;

    assign o_coordinator_timestep_start =
        state_q == ST_REPLAY_READY && i_runtime_timestep_start && !o_error;
    assign o_runtime_timestep_start_ready =
        state_q == ST_REPLAY_READY && i_coordinator_timestep_start_ready &&
        !o_error;

    assign o_core_abort = core_inflight_q && service_abort;
    assign capture_counts_complete =
        (capture_done_seen_q || service_capture_done) &&
        ((completed_count_q == batch_count_q) ||
         (core_done_fire && completed_count_q + 1'b1 == batch_count_q));
    assign o_started_sample_count = started_count_q;
    assign o_completed_sample_count = completed_count_q;
    assign o_capture_complete = state_q == ST_REPLAY_READY;
    assign o_capture_busy = state_q == ST_CAPTURE_START ||
        state_q == ST_CAPTURE || state_q == ST_ABORT_DRAIN ||
        core_inflight_q;
    assign o_quiescent = state_q == ST_IDLE && !service_busy &&
        !core_inflight_q && !o_error;
    assign o_error = state_q == ST_ERROR || local_error_q || service_cfg_error ||
        service_protocol_error || service_version_error || service_tail_error;

    unified_srnn_m2_live_actual_replay_service #(
        .BATCH_MAX(BATCH_MAX),
        .TIMESTEP_MAX(TIMESTEP_MAX),
        .BACKBONE_MAX(BACKBONE_MAX),
        .SOURCE_VECTOR_MAX(SOURCE_VECTOR_MAX),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) replay_service (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(cfg_fire), .o_cfg_ready(service_cfg_ready),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_timestep_count(i_cfg_timestep_count),
        .i_cfg_backbone_count(i_cfg_backbone_count),
        .i_cfg_expected_version(i_cfg_expected_version),
        .i_cfg_check_version(i_cfg_check_version),
        .o_cfg_error(service_cfg_error),
        .i_capture_start(service_capture_start),
        .o_capture_start_ready(service_capture_start_ready),
        .i_abort(service_abort),
        .i_commit_valid(i_commit_valid && state_q == ST_CAPTURE),
        .o_commit_ready(service_commit_ready),
        .i_commit_version(i_commit_version),
        .i_commit_sample_last(i_commit_sample_last),
        .i_commit_spikes(i_commit_spikes),
        .o_capture_done(service_capture_done),
        .i_replay_start(i_replay_start),
        .o_replay_start_ready(o_replay_start_ready),
        .i_replay_timestep_id(i_replay_timestep_id),
        .o_replay_valid(o_replay_valid),
        .i_replay_ready(i_replay_ready),
        .o_replay_sample_id(o_replay_sample_id),
        .o_replay_spikes(o_replay_spikes),
        .o_replay_done(o_replay_done),
        .o_busy(service_busy), .o_configured(service_configured),
        .o_protocol_error(service_protocol_error),
        .o_version_error(service_version_error),
        .o_tail_error(service_tail_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            batch_count_q <= '0;
            started_count_q <= '0;
            completed_count_q <= '0;
            sample_last_count_q <= '0;
            core_inflight_q <= 1'b0;
            capture_done_seen_q <= 1'b0;
            local_error_q <= 1'b0;
        end else begin
            if (cfg_fire)
                batch_count_q <= i_cfg_batch_count;

            if (i_clear_error && state_q == ST_ERROR &&
                !i_runtime_client_active && !core_inflight_q &&
                !service_busy) begin
                state_q <= ST_IDLE;
                local_error_q <= 1'b0;
            end else if (service_cfg_error || service_protocol_error ||
                         service_version_error || service_tail_error) begin
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        started_count_q <= '0;
                        completed_count_q <= '0;
                        sample_last_count_q <= '0;
                        core_inflight_q <= 1'b0;
                        capture_done_seen_q <= 1'b0;
                        if (i_runtime_client_active) begin
                            if (!service_configured || batch_count_q == '0) begin
                                local_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                state_q <= ST_CAPTURE_START;
                            end
                        end
                    end

                    ST_CAPTURE_START: begin
                        if (i_abort || !i_runtime_client_active) begin
                            state_q <= ST_ABORT_DRAIN;
                        end else if (service_capture_start) begin
                            state_q <= ST_CAPTURE;
                        end
                    end

                    ST_CAPTURE: begin
                        if (sample_start_fire) begin
                            started_count_q <= started_count_q + 1'b1;
                            core_inflight_q <= 1'b1;
                        end
                        if (commit_fire && i_commit_sample_last)
                            sample_last_count_q <= sample_last_count_q + 1'b1;
                        if (i_commit_valid && !o_commit_ready) begin
                            local_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                        if (service_capture_done)
                            capture_done_seen_q <= 1'b1;
                        if (core_done_fire) begin
                            core_inflight_q <= 1'b0;
                            completed_count_q <= completed_count_q + 1'b1;
                            if (sample_last_count_q +
                                $clog2(BATCH_MAX+1)'(
                                    commit_fire && i_commit_sample_last)
                                != completed_count_q + 1'b1) begin
                                local_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end
                        end else if (i_core_done) begin
                            local_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                        if (core_aborted_fire || i_abort ||
                            !i_runtime_client_active) begin
                            core_inflight_q <= core_inflight_q &&
                                !core_aborted_fire;
                            state_q <= ST_ABORT_DRAIN;
                        end else if (i_core_aborted) begin
                            local_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (capture_counts_complete) begin
                            state_q <= ST_REPLAY_READY;
                        end
                    end

                    ST_REPLAY_READY: begin
                        if (i_abort)
                            state_q <= ST_ABORT_DRAIN;
                        else if (!i_runtime_client_active)
                            state_q <= ST_IDLE;
                    end

                    ST_ABORT_DRAIN: begin
                        if (core_done_fire || core_aborted_fire)
                            core_inflight_q <= 1'b0;
                        if (!i_runtime_client_active && !core_inflight_q &&
                            !service_busy)
                            state_q <= ST_IDLE;
                    end

                    default: begin
                        if (i_clear_error && !i_runtime_client_active &&
                            !core_inflight_q && !service_busy) begin
                            state_q <= ST_IDLE;
                            local_error_q <= 1'b0;
                        end
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
