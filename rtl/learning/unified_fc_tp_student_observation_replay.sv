`timescale 1ns/1ps
`default_nettype none

// Captures production FC student observations between samples and replays one
// batch into the TP learning frontend. The source snapshot must remain stable
// from i_snapshot_start acceptance through o_snapshot_done.
module unified_fc_tp_student_observation_replay #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 480,
    parameter int MAX_OUTPUTS = 10,
    parameter int MAX_EVENTS = 8192,
    parameter int FEATURE_INDEX_WIDTH = 12,
    parameter int TRACE_WIDTH = 16,
    parameter int VMEM_WIDTH = 18,
    parameter int LOGIT_WIDTH = 24,
    parameter int SAMPLE_INDEX_WIDTH =
        (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int HIDDEN_INDEX_WIDTH =
        (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int HIDDEN_COUNT_WIDTH = $clog2(MAX_HIDDEN + 1),
    parameter int OUTPUT_INDEX_WIDTH =
        (MAX_OUTPUTS <= 2) ? 1 : $clog2(MAX_OUTPUTS),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(MAX_OUTPUTS + 1),
    parameter int EVENT_COUNT_WIDTH = $clog2(MAX_EVENTS + 1)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [BATCH_COUNT_WIDTH-1:0]              i_cfg_batch_count,
    input  logic [HIDDEN_COUNT_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]             i_cfg_output_count,
    input  logic                                      i_release,
    output logic                                      o_release_ready,

    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [FEATURE_INDEX_WIDTH-1:0]            i_event_feature_index,

    input  logic                                      i_snapshot_start,
    output logic                                      o_snapshot_start_ready,
    input  logic [OUTPUT_INDEX_WIDTH-1:0]             i_snapshot_label,
    input  logic [MAX_HIDDEN-1:0]                     i_snapshot_spikes_flat,
    input  logic [MAX_HIDDEN*TRACE_WIDTH-1:0]         i_snapshot_trace_flat,
    input  logic [MAX_HIDDEN*VMEM_WIDTH-1:0]          i_snapshot_vmem_flat,
    input  logic [MAX_OUTPUTS*LOGIT_WIDTH-1:0]         i_snapshot_logits_flat,
    output logic                                      o_snapshot_busy,
    output logic                                      o_snapshot_done,

    input  logic                                      i_batch_seal,
    output logic                                      o_batch_seal_ready,

    output logic                                      o_hidden_valid,
    input  logic                                      i_hidden_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_hidden_sample,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_hidden_index,
    output logic                                      o_hidden_spike,
    output logic [TRACE_WIDTH-1:0]                    o_hidden_trace,
    output logic signed [VMEM_WIDTH-1:0]              o_hidden_vmem,

    output logic                                      o_replay_event_valid,
    input  logic                                      i_replay_event_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_replay_event_sample,
    output logic [FEATURE_INDEX_WIDTH-1:0]            o_replay_event_feature_index,

    output logic                                      o_logit_row_valid,
    input  logic                                      i_logit_row_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]             o_logit_sample,
    output logic [OUTPUT_INDEX_WIDTH-1:0]             o_logit_label,
    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic signed [LOGIT_WIDTH-1:0]             o_logit,

    output logic                                      o_replay_done,
    input  logic                                      i_replay_done_ready,
    output logic                                      o_active,
    output logic                                      o_protocol_error
);
    localparam int HIDDEN_DEPTH = MAX_BATCH * MAX_HIDDEN;
    localparam int LOGIT_DEPTH = MAX_BATCH * MAX_OUTPUTS;
    localparam int HIDDEN_ADDR_WIDTH =
        (HIDDEN_DEPTH <= 2) ? 1 : $clog2(HIDDEN_DEPTH);
    localparam int LOGIT_ADDR_WIDTH =
        (LOGIT_DEPTH <= 2) ? 1 : $clog2(LOGIT_DEPTH);
    localparam int EVENT_ADDR_WIDTH =
        (MAX_EVENTS <= 2) ? 1 : $clog2(MAX_EVENTS);

    typedef enum logic [2:0] {
        STATE_CAPTURE,
        STATE_REPLAY_HIDDEN,
        STATE_REPLAY_EVENT,
        STATE_REPLAY_LOGIT_ROW,
        STATE_REPLAY_LOGIT,
        STATE_DONE
    } state_t;

    (* ram_style = "block" *) logic [TRACE_WIDTH-1:0]
        trace_mem [0:HIDDEN_DEPTH-1];
    (* ram_style = "block" *) logic
        spike_mem [0:HIDDEN_DEPTH-1];
    (* ram_style = "block" *) logic [VMEM_WIDTH-1:0]
        vmem_mem [0:HIDDEN_DEPTH-1];
    (* ram_style = "block" *) logic [LOGIT_WIDTH-1:0]
        logit_mem [0:LOGIT_DEPTH-1];
    (* ram_style = "block" *) logic [FEATURE_INDEX_WIDTH-1:0]
        event_feature_mem [0:MAX_EVENTS-1];
    (* ram_style = "block" *) logic [SAMPLE_INDEX_WIDTH-1:0]
        event_sample_mem [0:MAX_EVENTS-1];
    logic [OUTPUT_INDEX_WIDTH-1:0] label_mem [0:MAX_BATCH-1];

    state_t state_q;
    logic configured_q;
    logic capture_busy_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [HIDDEN_COUNT_WIDTH-1:0] hidden_count_q;
    logic [OUTPUT_COUNT_WIDTH-1:0] output_count_q;
    logic [BATCH_COUNT_WIDTH-1:0] captured_samples_q;
    logic [EVENT_COUNT_WIDTH-1:0] captured_events_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] capture_hidden_q;
    logic [OUTPUT_INDEX_WIDTH-1:0] capture_output_q;
    logic capture_hidden_done_q;
    logic capture_output_done_q;

    logic [SAMPLE_INDEX_WIDTH-1:0] replay_hidden_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] replay_hidden_index_q;
    logic hidden_last_issued_q;
    logic hidden_valid_q;
    logic [SAMPLE_INDEX_WIDTH-1:0] hidden_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index_q;
    logic hidden_spike_q;
    logic [TRACE_WIDTH-1:0] hidden_trace_q;
    logic signed [VMEM_WIDTH-1:0] hidden_vmem_q;

    logic [EVENT_COUNT_WIDTH-1:0] replay_event_q;
    logic event_last_issued_q;
    logic event_valid_q;
    logic [SAMPLE_INDEX_WIDTH-1:0] event_sample_q;
    logic [FEATURE_INDEX_WIDTH-1:0] event_feature_q;

    logic [SAMPLE_INDEX_WIDTH-1:0] replay_logit_sample_q;
    logic [OUTPUT_INDEX_WIDTH-1:0] replay_logit_index_q;
    logic logit_last_issued_q;
    logic logit_valid_q;
    logic signed [LOGIT_WIDTH-1:0] logit_q;

    logic cfg_legal;
    logic batch_count_legal;
    logic hidden_count_legal;
    logic output_count_legal;
    logic cfg_fire;
    logic release_fire;
    logic event_fire;
    logic snapshot_fire;
    logic seal_fire;
    logic hidden_fire;
    logic replay_event_fire;
    logic logit_row_fire;
    logic logit_fire;
    logic done_fire;
    logic [HIDDEN_ADDR_WIDTH-1:0] capture_hidden_addr;
    logic [LOGIT_ADDR_WIDTH-1:0] capture_logit_addr;
    logic [HIDDEN_ADDR_WIDTH-1:0] replay_hidden_addr;
    logic [LOGIT_ADDR_WIDTH-1:0] replay_logit_addr;

    generate
        if (MAX_BATCH == ((1 << BATCH_COUNT_WIDTH) - 1)) begin : g_batch_full_range
            assign batch_count_legal = i_cfg_batch_count != 0;
        end else begin : g_batch_bounded
            assign batch_count_legal = (i_cfg_batch_count != 0) &&
                (i_cfg_batch_count <= BATCH_COUNT_WIDTH'(MAX_BATCH));
        end
        if (MAX_HIDDEN == ((1 << HIDDEN_COUNT_WIDTH) - 1)) begin : g_hidden_full_range
            assign hidden_count_legal = i_cfg_hidden_count != 0;
        end else begin : g_hidden_bounded
            assign hidden_count_legal = (i_cfg_hidden_count != 0) &&
                (i_cfg_hidden_count <= HIDDEN_COUNT_WIDTH'(MAX_HIDDEN));
        end
        if (MAX_OUTPUTS == ((1 << OUTPUT_COUNT_WIDTH) - 1)) begin : g_output_full_range
            assign output_count_legal = i_cfg_output_count != 0;
        end else begin : g_output_bounded
            assign output_count_legal = (i_cfg_output_count != 0) &&
                (i_cfg_output_count <= OUTPUT_COUNT_WIDTH'(MAX_OUTPUTS));
        end
    endgenerate

    always_comb begin
        cfg_legal = batch_count_legal && hidden_count_legal &&
            output_count_legal;
        o_cfg_ready = !configured_q && (state_q == STATE_CAPTURE);
        cfg_fire = i_cfg_valid && o_cfg_ready;
        o_release_ready = configured_q && !capture_busy_q &&
            (state_q == STATE_CAPTURE) && (captured_samples_q == 0) &&
            (captured_events_q == 0);
        release_fire = i_release && o_release_ready;

        o_event_ready = configured_q && !capture_busy_q &&
            (state_q == STATE_CAPTURE) &&
            (captured_samples_q < batch_count_q) &&
            (captured_events_q < EVENT_COUNT_WIDTH'(MAX_EVENTS));
        event_fire = i_event_valid && o_event_ready;
        o_snapshot_start_ready = configured_q && !capture_busy_q &&
            (state_q == STATE_CAPTURE) &&
            (captured_samples_q < batch_count_q);
        snapshot_fire = i_snapshot_start && o_snapshot_start_ready;
        o_snapshot_busy = capture_busy_q;
        o_batch_seal_ready = configured_q && !capture_busy_q &&
            (state_q == STATE_CAPTURE) &&
            (captured_samples_q == batch_count_q);
        seal_fire = i_batch_seal && o_batch_seal_ready;

        capture_hidden_addr = HIDDEN_ADDR_WIDTH'(
            captured_samples_q * MAX_HIDDEN + capture_hidden_q);
        capture_logit_addr = LOGIT_ADDR_WIDTH'(
            captured_samples_q * MAX_OUTPUTS + capture_output_q);
        replay_hidden_addr = HIDDEN_ADDR_WIDTH'(
            replay_hidden_sample_q * MAX_HIDDEN + replay_hidden_index_q);
        replay_logit_addr = LOGIT_ADDR_WIDTH'(
            replay_logit_sample_q * MAX_OUTPUTS + replay_logit_index_q);

        o_hidden_valid = hidden_valid_q;
        o_hidden_sample = hidden_sample_q;
        o_hidden_index = hidden_index_q;
        o_hidden_spike = hidden_spike_q;
        o_hidden_trace = hidden_trace_q;
        o_hidden_vmem = hidden_vmem_q;
        hidden_fire = o_hidden_valid && i_hidden_ready;

        o_replay_event_valid = event_valid_q;
        o_replay_event_sample = event_sample_q;
        o_replay_event_feature_index = event_feature_q;
        replay_event_fire = o_replay_event_valid && i_replay_event_ready;

        o_logit_row_valid = state_q == STATE_REPLAY_LOGIT_ROW;
        o_logit_sample = replay_logit_sample_q;
        o_logit_label = label_mem[replay_logit_sample_q];
        logit_row_fire = o_logit_row_valid && i_logit_row_ready;
        o_logit_valid = logit_valid_q;
        o_logit = logit_q;
        logit_fire = o_logit_valid && i_logit_ready;

        o_replay_done = state_q == STATE_DONE;
        done_fire = o_replay_done && i_replay_done_ready;
        o_active = configured_q;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_CAPTURE;
            configured_q <= 1'b0;
            capture_busy_q <= 1'b0;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            output_count_q <= '0;
            captured_samples_q <= '0;
            captured_events_q <= '0;
            capture_hidden_q <= '0;
            capture_output_q <= '0;
            capture_hidden_done_q <= 1'b0;
            capture_output_done_q <= 1'b0;
            replay_hidden_sample_q <= '0;
            replay_hidden_index_q <= '0;
            hidden_last_issued_q <= 1'b0;
            hidden_valid_q <= 1'b0;
            hidden_sample_q <= '0;
            hidden_index_q <= '0;
            hidden_spike_q <= 1'b0;
            hidden_trace_q <= '0;
            hidden_vmem_q <= '0;
            replay_event_q <= '0;
            event_last_issued_q <= 1'b0;
            event_valid_q <= 1'b0;
            event_sample_q <= '0;
            event_feature_q <= '0;
            replay_logit_sample_q <= '0;
            replay_logit_index_q <= '0;
            logit_last_issued_q <= 1'b0;
            logit_valid_q <= 1'b0;
            logit_q <= '0;
            o_snapshot_done <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_snapshot_done <= 1'b0;
            if (i_clear_error)
                o_protocol_error <= 1'b0;
            if ((cfg_fire && !cfg_legal) ||
                (i_batch_seal && configured_q &&
                 (state_q == STATE_CAPTURE) && !o_batch_seal_ready) ||
                (i_event_valid && configured_q &&
                 (captured_events_q == EVENT_COUNT_WIDTH'(MAX_EVENTS))))
                o_protocol_error <= 1'b1;

            if (cfg_fire && cfg_legal) begin
                configured_q <= 1'b1;
                batch_count_q <= i_cfg_batch_count;
                hidden_count_q <= i_cfg_hidden_count;
                output_count_q <= i_cfg_output_count;
                captured_samples_q <= '0;
                captured_events_q <= '0;
            end
            if (release_fire) begin
                configured_q <= 1'b0;
                batch_count_q <= '0;
                hidden_count_q <= '0;
                output_count_q <= '0;
            end
            if (event_fire) begin
                event_feature_mem[captured_events_q[EVENT_ADDR_WIDTH-1:0]] <=
                    i_event_feature_index;
                event_sample_mem[captured_events_q[EVENT_ADDR_WIDTH-1:0]] <=
                    SAMPLE_INDEX_WIDTH'(captured_samples_q);
                captured_events_q <= captured_events_q + 1'b1;
            end

            if (snapshot_fire) begin
                label_mem[SAMPLE_INDEX_WIDTH'(captured_samples_q)] <=
                    i_snapshot_label;
                capture_busy_q <= 1'b1;
                capture_hidden_q <= '0;
                capture_output_q <= '0;
                capture_hidden_done_q <= 1'b0;
                capture_output_done_q <= 1'b0;
            end else if (capture_busy_q) begin
                if (!capture_hidden_done_q) begin
                    spike_mem[capture_hidden_addr] <=
                        i_snapshot_spikes_flat[capture_hidden_q];
                    trace_mem[capture_hidden_addr] <= i_snapshot_trace_flat[
                        capture_hidden_q*TRACE_WIDTH +: TRACE_WIDTH];
                    vmem_mem[capture_hidden_addr] <= i_snapshot_vmem_flat[
                        capture_hidden_q*VMEM_WIDTH +: VMEM_WIDTH];
                    if (({1'b0, capture_hidden_q} + 1'b1) >=
                        {1'b0, hidden_count_q})
                        capture_hidden_done_q <= 1'b1;
                    else
                        capture_hidden_q <= capture_hidden_q + 1'b1;
                end
                if (!capture_output_done_q) begin
                    logit_mem[capture_logit_addr] <= i_snapshot_logits_flat[
                        capture_output_q*LOGIT_WIDTH +: LOGIT_WIDTH];
                    if (({1'b0, capture_output_q} + 1'b1) >=
                        {1'b0, output_count_q})
                    capture_output_done_q <= 1'b1;
                    else
                        capture_output_q <= capture_output_q + 1'b1;
                end
                if ((capture_hidden_done_q ||
                     (({1'b0, capture_hidden_q} + 1'b1) >=
                      {1'b0, hidden_count_q})) &&
                    (capture_output_done_q ||
                     (({1'b0, capture_output_q} + 1'b1) >=
                      {1'b0, output_count_q}))) begin
                    capture_busy_q <= 1'b0;
                    captured_samples_q <= captured_samples_q + 1'b1;
                    o_snapshot_done <= 1'b1;
                end
            end

            if (seal_fire) begin
                state_q <= STATE_REPLAY_HIDDEN;
                replay_hidden_sample_q <= '0;
                replay_hidden_index_q <= '0;
                hidden_last_issued_q <= 1'b0;
                hidden_valid_q <= 1'b0;
            end

            case (state_q)
                STATE_REPLAY_HIDDEN: begin
                    if (!hidden_valid_q && !hidden_last_issued_q) begin
                        hidden_sample_q <= replay_hidden_sample_q;
                        hidden_index_q <= replay_hidden_index_q;
                        hidden_spike_q <= spike_mem[replay_hidden_addr];
                        hidden_trace_q <= trace_mem[replay_hidden_addr];
                        hidden_vmem_q <= $signed(vmem_mem[replay_hidden_addr]);
                        hidden_valid_q <= 1'b1;
                        if (({1'b0, replay_hidden_sample_q} + 1'b1 >=
                             batch_count_q) &&
                            ({1'b0, replay_hidden_index_q} + 1'b1 >=
                             {1'b0, hidden_count_q})) begin
                            hidden_last_issued_q <= 1'b1;
                        end else if (({1'b0, replay_hidden_index_q} + 1'b1) >=
                                     {1'b0, hidden_count_q}) begin
                            replay_hidden_sample_q <=
                                replay_hidden_sample_q + 1'b1;
                            replay_hidden_index_q <= '0;
                        end else begin
                            replay_hidden_index_q <=
                                replay_hidden_index_q + 1'b1;
                        end
                    end
                    if (hidden_fire) begin
                        hidden_valid_q <= 1'b0;
                        if (hidden_last_issued_q) begin
                            state_q <= captured_events_q == 0 ?
                                STATE_REPLAY_LOGIT_ROW : STATE_REPLAY_EVENT;
                            replay_event_q <= '0;
                            event_last_issued_q <= 1'b0;
                        end
                    end
                end
                STATE_REPLAY_EVENT: begin
                    if (!event_valid_q && !event_last_issued_q) begin
                        event_sample_q <= event_sample_mem[
                            replay_event_q[EVENT_ADDR_WIDTH-1:0]];
                        event_feature_q <= event_feature_mem[
                            replay_event_q[EVENT_ADDR_WIDTH-1:0]];
                        event_valid_q <= 1'b1;
                        if ((replay_event_q + 1'b1) >= captured_events_q)
                            event_last_issued_q <= 1'b1;
                        else
                            replay_event_q <= replay_event_q + 1'b1;
                    end
                    if (replay_event_fire) begin
                        event_valid_q <= 1'b0;
                        if (event_last_issued_q) begin
                            state_q <= STATE_REPLAY_LOGIT_ROW;
                            replay_logit_sample_q <= '0;
                        end
                    end
                end
                STATE_REPLAY_LOGIT_ROW: begin
                    if (logit_row_fire) begin
                        state_q <= STATE_REPLAY_LOGIT;
                        replay_logit_index_q <= '0;
                        logit_last_issued_q <= 1'b0;
                        logit_valid_q <= 1'b0;
                    end
                end
                STATE_REPLAY_LOGIT: begin
                    if (!logit_valid_q && !logit_last_issued_q) begin
                        logit_q <= $signed(logit_mem[replay_logit_addr]);
                        logit_valid_q <= 1'b1;
                        if (({1'b0, replay_logit_index_q} + 1'b1) >=
                            {1'b0, output_count_q})
                            logit_last_issued_q <= 1'b1;
                        else
                            replay_logit_index_q <=
                                replay_logit_index_q + 1'b1;
                    end
                    if (logit_fire) begin
                        logit_valid_q <= 1'b0;
                        if (logit_last_issued_q) begin
                            if (({1'b0, replay_logit_sample_q} + 1'b1) >=
                                batch_count_q) begin
                                state_q <= STATE_DONE;
                            end else begin
                                replay_logit_sample_q <=
                                    replay_logit_sample_q + 1'b1;
                                state_q <= STATE_REPLAY_LOGIT_ROW;
                            end
                        end
                    end
                end
                STATE_DONE: begin
                    if (done_fire) begin
                        state_q <= STATE_CAPTURE;
                        captured_samples_q <= '0;
                        captured_events_q <= '0;
                        hidden_last_issued_q <= 1'b0;
                        event_last_issued_q <= 1'b0;
                        logit_last_issued_q <= 1'b0;
                    end
                end
                default: begin
                end
            endcase
        end
    end

    initial begin
        if (MAX_BATCH < 1 || MAX_HIDDEN < 1 || MAX_OUTPUTS < 1 ||
            MAX_EVENTS < 1)
            $error("FC TP student observation replay dimensions are invalid");
    end
endmodule

`default_nettype wire
