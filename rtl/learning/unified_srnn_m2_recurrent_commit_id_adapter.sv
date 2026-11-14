`timescale 1ns/1ps
`default_nettype none

// Tags the live SRNN recurrent commit stream with sample/timestep IDs before it
// enters the M2 replay buffer. The single elastic stage supports simultaneous
// downstream retirement and upstream replacement so consecutive commit pulses
// do not create an artificial bubble.
module unified_srnn_m2_recurrent_commit_id_adapter #(
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [$clog2(BATCH_MAX+1)-1:0]            i_cfg_batch_count,
    input  logic [$clog2(TIMESTEP_MAX+1)-1:0]         i_cfg_timestep_count,
    input  logic [VERSION_WIDTH-1:0]                  i_cfg_expected_version,
    input  logic                                      i_cfg_check_version,
    output logic                                      o_cfg_error,

    input  logic                                      i_window_start,
    output logic                                      o_window_start_ready,
    input  logic                                      i_abort_window,

    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_commit_version,
    input  logic                                      i_commit_sample_last,
    input  logic [SOURCE_VECTOR_MAX-1:0]              i_commit_spikes,

    output logic                                      o_capture_valid,
    input  logic                                      i_capture_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_capture_sample_id,
    output logic [$clog2(TIMESTEP_MAX)-1:0]           o_capture_timestep_id,
    output logic [SOURCE_VECTOR_MAX-1:0]              o_capture_spikes,

    output logic                                      o_window_done,
    output logic                                      o_busy,
    output logic                                      o_configured,
    output logic                                      o_protocol_error,
    output logic                                      o_version_error
);

    localparam int BATCH_W = $clog2(BATCH_MAX);
    localparam int TIMESTEP_W = $clog2(TIMESTEP_MAX);

    logic configured_q;
    logic window_open_q;
    logic output_valid_q;
    logic [$clog2(BATCH_MAX+1)-1:0] batch_count_q;
    logic [$clog2(TIMESTEP_MAX+1)-1:0] timestep_count_q;
    logic [VERSION_WIDTH-1:0] expected_version_q;
    logic check_version_q;
    logic [BATCH_W-1:0] expected_sample_q;
    logic [TIMESTEP_W-1:0] expected_timestep_q;
    logic [BATCH_W-1:0] output_sample_q;
    logic [TIMESTEP_W-1:0] output_timestep_q;
    logic [SOURCE_VECTOR_MAX-1:0] output_spikes_q;
    logic protocol_error_q;
    logic version_error_q;

    logic cfg_legal;
    logic expected_sample_last;
    logic commit_legal;
    logic commit_version_legal;
    logic commit_fire;
    logic capture_fire;
    logic output_is_final;

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (int'(i_cfg_batch_count) <= BATCH_MAX) &&
            (i_cfg_timestep_count != '0) &&
            (int'(i_cfg_timestep_count) <= TIMESTEP_MAX);

        o_cfg_ready = !window_open_q && !output_valid_q;
        o_window_start_ready = configured_q && !window_open_q &&
            !output_valid_q && !protocol_error_q && !version_error_q &&
            !i_cfg_valid;

        o_commit_ready = window_open_q &&
            (!output_valid_q || i_capture_ready) &&
            !protocol_error_q && !version_error_q;
        commit_fire = i_commit_valid && o_commit_ready;
        expected_sample_last =
            expected_timestep_q == TIMESTEP_W'(timestep_count_q - 1'b1);
        commit_version_legal = !check_version_q ||
            (i_commit_version == expected_version_q);
        commit_legal = (i_commit_sample_last == expected_sample_last) &&
            commit_version_legal;

        o_capture_valid = output_valid_q;
        o_capture_sample_id = output_sample_q;
        o_capture_timestep_id = output_timestep_q;
        o_capture_spikes = output_spikes_q;
        capture_fire = o_capture_valid && i_capture_ready;
        output_is_final =
            (output_sample_q == BATCH_W'(batch_count_q - 1'b1)) &&
            (output_timestep_q == TIMESTEP_W'(timestep_count_q - 1'b1));

        o_busy = window_open_q || output_valid_q;
        o_configured = configured_q;
        o_protocol_error = protocol_error_q;
        o_version_error = version_error_q;
    end

    // Captured commit metadata participates in replay-memory write control.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            window_open_q <= 1'b0;
            output_valid_q <= 1'b0;
            batch_count_q <= '0;
            timestep_count_q <= '0;
            expected_version_q <= '0;
            check_version_q <= 1'b0;
            expected_sample_q <= '0;
            expected_timestep_q <= '0;
            output_sample_q <= '0;
            output_timestep_q <= '0;
            output_spikes_q <= '0;
            protocol_error_q <= 1'b0;
            version_error_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_window_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_window_done <= 1'b0;

            if (i_clear_error) begin
                protocol_error_q <= 1'b0;
                version_error_q <= 1'b0;
            end

            if (i_abort_window) begin
                window_open_q <= 1'b0;
                output_valid_q <= 1'b0;
                expected_sample_q <= '0;
                expected_timestep_q <= '0;
            end else begin
                if (i_cfg_valid && o_cfg_ready) begin
                    if (!cfg_legal) begin
                        configured_q <= 1'b0;
                        o_cfg_error <= 1'b1;
                    end else begin
                        configured_q <= 1'b1;
                        batch_count_q <= i_cfg_batch_count;
                        timestep_count_q <= i_cfg_timestep_count;
                        expected_version_q <= i_cfg_expected_version;
                        check_version_q <= i_cfg_check_version;
                        protocol_error_q <= 1'b0;
                        version_error_q <= 1'b0;
                    end
                end

                if (i_window_start && o_window_start_ready) begin
                    window_open_q <= 1'b1;
                    expected_sample_q <= '0;
                    expected_timestep_q <= '0;
                    protocol_error_q <= 1'b0;
                    version_error_q <= 1'b0;
                end

                if (capture_fire) begin
                    output_valid_q <= 1'b0;
                    if (output_is_final) begin
                        o_window_done <= 1'b1;
                    end
                end

                if (i_commit_valid && !o_commit_ready) begin
                    protocol_error_q <= 1'b1;
                    window_open_q <= 1'b0;
                    output_valid_q <= 1'b0;
                end else if (commit_fire) begin
                    if (!commit_version_legal) begin
                        version_error_q <= 1'b1;
                        window_open_q <= 1'b0;
                    end else if (!commit_legal) begin
                        protocol_error_q <= 1'b1;
                        window_open_q <= 1'b0;
                    end else begin
                        output_valid_q <= 1'b1;
                        output_sample_q <= expected_sample_q;
                        output_timestep_q <= expected_timestep_q;
                        output_spikes_q <= i_commit_spikes;
                        if ((expected_sample_q ==
                                BATCH_W'(batch_count_q - 1'b1)) &&
                            (expected_timestep_q ==
                                TIMESTEP_W'(timestep_count_q - 1'b1))) begin
                            window_open_q <= 1'b0;
                        end else if (expected_timestep_q ==
                            TIMESTEP_W'(timestep_count_q - 1'b1)) begin
                            expected_sample_q <= expected_sample_q + 1'b1;
                            expected_timestep_q <= '0;
                        end else begin
                            expected_timestep_q <= expected_timestep_q + 1'b1;
                        end
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
