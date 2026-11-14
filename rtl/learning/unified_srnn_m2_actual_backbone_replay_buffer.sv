`timescale 1ns/1ps

// Reorders live sample-major SRNN commits into one-timestep, batch-major replay.
// The payload RAM is reset-free; capture metadata is the sole validity authority.
module unified_srnn_m2_actual_backbone_replay_buffer #(
    parameter int unsigned BATCH_MAX         = 64,
    parameter int unsigned TIMESTEP_MAX      = 128,
    parameter int unsigned BACKBONE_MAX      = 450,
    parameter int unsigned SOURCE_VECTOR_MAX = 480,
    parameter int unsigned WORD_WIDTH        = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic i_cfg_valid,
    output logic o_cfg_ready,
    input  logic [$clog2(BATCH_MAX+1)-1:0]    i_cfg_batch_count,
    input  logic [$clog2(TIMESTEP_MAX+1)-1:0] i_cfg_timestep_count,
    input  logic [$clog2(BACKBONE_MAX+1)-1:0] i_cfg_backbone_count,

    input  logic i_capture_start,
    input  logic i_capture_valid,
    output logic o_capture_ready,
    input  logic [$clog2(BATCH_MAX)-1:0]       i_capture_sample_id,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]    i_capture_timestep_id,
    input  logic [SOURCE_VECTOR_MAX-1:0]       i_capture_spikes,
    output logic o_capture_done,

    input  logic i_replay_start,
    input  logic [$clog2(TIMESTEP_MAX)-1:0] i_replay_timestep_id,
    output logic o_replay_valid,
    input  logic i_replay_ready,
    output logic [$clog2(BATCH_MAX)-1:0] o_replay_sample_id,
    output logic [SOURCE_VECTOR_MAX-1:0] o_replay_spikes,
    output logic o_replay_done,

    input  logic i_abort,
    input  logic i_clear_error,
    output logic o_busy,
    output logic o_configured,
    output logic o_protocol_error,
    output logic o_tail_error
);
    localparam int unsigned BATCH_W = $clog2(BATCH_MAX);
    localparam int unsigned TIMESTEP_W = $clog2(TIMESTEP_MAX);
    localparam int unsigned WORDS_PER_VECTOR =
        (SOURCE_VECTOR_MAX + WORD_WIDTH - 1) / WORD_WIDTH;
    localparam int unsigned PADDED_VECTOR_WIDTH = WORDS_PER_VECTOR * WORD_WIDTH;
    localparam int unsigned VECTOR_DEPTH = BATCH_MAX * TIMESTEP_MAX;
    localparam int unsigned ADDRESS_W =
        (VECTOR_DEPTH <= 1) ? 1 : $clog2(VECTOR_DEPTH);

    logic [$clog2(BATCH_MAX+1)-1:0] batch_count_q;
    logic [$clog2(TIMESTEP_MAX+1)-1:0] timestep_count_q;
    logic [$clog2(BACKBONE_MAX+1)-1:0] backbone_count_q;

    logic capture_open_q;
    logic capture_complete_q;
    logic [BATCH_W-1:0] capture_expected_sample_q;
    logic [TIMESTEP_W-1:0] capture_expected_timestep_q;

    logic replay_active_q;
    logic replay_read_pending_q;
    logic replay_output_valid_q;
    logic [TIMESTEP_W-1:0] replay_timestep_q;
    logic [BATCH_W-1:0] replay_sample_q;
    logic [WORD_WIDTH-1:0] replay_word_data_q [0:WORDS_PER_VECTOR-1];
    logic [PADDED_VECTOR_WIDTH-1:0] replay_vector_q;
    logic [PADDED_VECTOR_WIDTH-1:0] capture_vector;
    logic [ADDRESS_W-1:0] capture_address;
    logic [ADDRESS_W-1:0] replay_address;
    logic capture_fire;
    logic replay_read_request;

    logic protocol_error_q;
    logic tail_error_q;

    function automatic logic [ADDRESS_W-1:0] vector_address(
        input logic [BATCH_W-1:0] sample_id,
        input logic [TIMESTEP_W-1:0] timestep_id
    );
        begin
            vector_address = ADDRESS_W'(
                ((int'(sample_id) * int'(timestep_count_q))
                + int'(timestep_id)));
        end
    endfunction

    function automatic logic tail_is_nonzero(
        input logic [SOURCE_VECTOR_MAX-1:0] spikes
    );
        logic [SOURCE_VECTOR_MAX-1:0] shifted;
        begin
            shifted = spikes >> int'(backbone_count_q);
            tail_is_nonzero = |shifted;
        end
    endfunction

    assign capture_vector = {
        {(PADDED_VECTOR_WIDTH-SOURCE_VECTOR_MAX){1'b0}}, i_capture_spikes};
    assign capture_address = vector_address(
        i_capture_sample_id, i_capture_timestep_id);
    assign replay_address = vector_address(replay_sample_q, replay_timestep_q);
    assign capture_fire = i_capture_valid && o_capture_ready;
    assign replay_read_request = replay_active_q && !replay_read_pending_q
        && !replay_output_valid_q;

    assign o_cfg_ready = !capture_open_q && !replay_active_q
        && !replay_read_pending_q && !replay_output_valid_q;
    assign o_capture_ready = capture_open_q
        && !protocol_error_q && !tail_error_q;
    assign o_replay_valid = replay_output_valid_q;
    assign o_replay_sample_id = replay_sample_q;
    assign o_replay_spikes = replay_vector_q[SOURCE_VECTOR_MAX-1:0];
    assign o_busy = capture_open_q || replay_active_q
        || replay_read_pending_q || replay_output_valid_q;
    assign o_configured = (batch_count_q != '0)
        && (timestep_count_q != '0) && (backbone_count_q != '0);
    assign o_protocol_error = protocol_error_q;
    assign o_tail_error = tail_error_q;

    // Replay counters and validity directly control the inferred BRAM ports.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            batch_count_q <= '0;
            timestep_count_q <= '0;
            backbone_count_q <= '0;
            capture_open_q <= 1'b0;
            capture_complete_q <= 1'b0;
            capture_expected_sample_q <= '0;
            capture_expected_timestep_q <= '0;
            replay_active_q <= 1'b0;
            replay_read_pending_q <= 1'b0;
            replay_output_valid_q <= 1'b0;
            replay_timestep_q <= '0;
            replay_sample_q <= '0;
            replay_vector_q <= '0;
            protocol_error_q <= 1'b0;
            tail_error_q <= 1'b0;
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;
        end else begin
            o_capture_done <= 1'b0;
            o_replay_done <= 1'b0;

            if (i_clear_error) begin
                protocol_error_q <= 1'b0;
                tail_error_q <= 1'b0;
            end

            if (i_abort) begin
                if (capture_open_q) begin
                    capture_complete_q <= 1'b0;
                end
                capture_open_q <= 1'b0;
                replay_active_q <= 1'b0;
                replay_read_pending_q <= 1'b0;
                replay_output_valid_q <= 1'b0;
            end else begin
                if (i_cfg_valid && o_cfg_ready) begin
                    if ((i_cfg_batch_count == '0)
                        || (int'(i_cfg_batch_count) > BATCH_MAX)
                        || (i_cfg_timestep_count == '0)
                        || (int'(i_cfg_timestep_count) > TIMESTEP_MAX)
                        || (i_cfg_backbone_count == '0)
                        || (int'(i_cfg_backbone_count) > BACKBONE_MAX)
                        || (int'(i_cfg_backbone_count) > SOURCE_VECTOR_MAX)) begin
                        protocol_error_q <= 1'b1;
                        batch_count_q <= '0;
                        timestep_count_q <= '0;
                        backbone_count_q <= '0;
                    end else begin
                        batch_count_q <= i_cfg_batch_count;
                        timestep_count_q <= i_cfg_timestep_count;
                        backbone_count_q <= i_cfg_backbone_count;
                        capture_complete_q <= 1'b0;
                        protocol_error_q <= 1'b0;
                        tail_error_q <= 1'b0;
                    end
                end

                if (i_capture_start && o_cfg_ready) begin
                    if (!o_configured) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        capture_open_q <= 1'b1;
                        capture_complete_q <= 1'b0;
                        capture_expected_sample_q <= '0;
                        capture_expected_timestep_q <= '0;
                        protocol_error_q <= 1'b0;
                        tail_error_q <= 1'b0;
                    end
                end

                if (capture_fire) begin
                    if ((i_capture_sample_id != capture_expected_sample_q)
                        || (i_capture_timestep_id != capture_expected_timestep_q)) begin
                        protocol_error_q <= 1'b1;
                        capture_open_q <= 1'b0;
                        capture_complete_q <= 1'b0;
                    end else if (tail_is_nonzero(i_capture_spikes)) begin
                        tail_error_q <= 1'b1;
                        capture_open_q <= 1'b0;
                        capture_complete_q <= 1'b0;
                    end else begin
                        if ((capture_expected_sample_q
                                == BATCH_W'(batch_count_q-1'b1))
                            && (capture_expected_timestep_q
                                == TIMESTEP_W'(timestep_count_q-1'b1))) begin
                            capture_open_q <= 1'b0;
                            capture_complete_q <= 1'b1;
                            o_capture_done <= 1'b1;
                        end else if (capture_expected_timestep_q
                            == TIMESTEP_W'(timestep_count_q-1'b1)) begin
                            capture_expected_sample_q <= capture_expected_sample_q + 1'b1;
                            capture_expected_timestep_q <= '0;
                        end else begin
                            capture_expected_timestep_q <= capture_expected_timestep_q + 1'b1;
                        end
                    end
                end

                if (i_replay_start && o_cfg_ready) begin
                    if (!capture_complete_q
                        || (int'(i_replay_timestep_id) >= int'(timestep_count_q))) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        replay_active_q <= 1'b1;
                        replay_read_pending_q <= 1'b0;
                        replay_output_valid_q <= 1'b0;
                        replay_timestep_q <= i_replay_timestep_id;
                        replay_sample_q <= '0;
                        replay_vector_q <= '0;
                    end
                end

                if (replay_read_request) begin
                    replay_read_pending_q <= 1'b1;
                end else if (replay_read_pending_q) begin
                    for (int unsigned word = 0; word < WORDS_PER_VECTOR; word++) begin
                        replay_vector_q[word*WORD_WIDTH +: WORD_WIDTH]
                            <= replay_word_data_q[word];
                    end
                    replay_read_pending_q <= 1'b0;
                    replay_output_valid_q <= 1'b1;
                end

                if (replay_output_valid_q && i_replay_ready) begin
                    replay_output_valid_q <= 1'b0;
                    if (replay_sample_q == BATCH_W'(batch_count_q-1'b1)) begin
                        replay_active_q <= 1'b0;
                        o_replay_done <= 1'b1;
                    end else begin
                        replay_sample_q <= replay_sample_q + 1'b1;
                        replay_vector_q <= '0;
                    end
                end
            end
        end
    end

    // A live recurrent commit is not backpressurable. Banking by vector word
    // keeps the original storage capacity while accepting one complete vector
    // per cycle instead of serializing its words across the capture interface.
    for (genvar word = 0; word < WORDS_PER_VECTOR; word++) begin : gen_replay_word_bank
        (* ram_style = "block" *) logic [WORD_WIDTH-1:0]
            replay_mem [0:VECTOR_DEPTH-1];

        always_ff @(posedge clk) begin
            if (capture_fire && !protocol_error_q && !tail_error_q
                    && (i_capture_sample_id == capture_expected_sample_q)
                    && (i_capture_timestep_id == capture_expected_timestep_q)
                    && !tail_is_nonzero(i_capture_spikes)) begin
                replay_mem[capture_address]
                    <= capture_vector[word*WORD_WIDTH +: WORD_WIDTH];
            end else if (replay_read_request) begin
                replay_word_data_q[word] <= replay_mem[replay_address];
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (SOURCE_VECTOR_MAX < BACKBONE_MAX) begin
            $fatal(1, "SOURCE_VECTOR_MAX must cover BACKBONE_MAX");
        end
        if ((WORD_WIDTH == 0) || (WORD_WIDTH > SOURCE_VECTOR_MAX)) begin
            $fatal(1, "WORD_WIDTH must be within SOURCE_VECTOR_MAX");
        end
    end
`endif
endmodule
