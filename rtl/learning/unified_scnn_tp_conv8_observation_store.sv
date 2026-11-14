`timescale 1ns/1ps
`default_nettype none

// Bounded full-layer observation retention for selected-layer Conv8 TP.
// The six payload arrays are reset-free block-RAM candidates. Capture is
// sample-major; replay emits one configured input-group/output-wave working
// set for the production-coordinate adapter and bounded gradient engine.
module unified_scnn_tp_conv8_observation_store #(
    parameter bit ALLOW_UNORDERED_SURROGATE_CAPTURE = 1'b0,
    parameter int SURROGATE_WAVE_CHANNELS = 0,
    parameter int SURROGATE_SPATIAL_BLOCK_SIZE = 0,
    parameter int TRACE_WAVE_CHANNELS = 0,
    parameter int MAX_BATCH = 10,
    parameter int MAX_INPUT_CHANNELS = 512,
    parameter int MAX_OUTPUT_CHANNELS = 512,
    parameter int MAX_TIMESTEPS = 20,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int VERSION_WIDTH = 16,
    parameter int SAMPLE_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_WIDTH = (MAX_TIMESTEPS <= 2) ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]        i_cfg_batch_count,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_cfg_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_cfg_sample_version,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_input_channels,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_output_channels,
    output logic                                o_active,
    output logic                                o_capture_complete,
    input  wire logic                                i_release,
    output logic                                o_release_ready,
    output logic                                o_protocol_error,

    input  wire logic                                i_input_valid,
    output logic                                o_input_ready,
    input  wire logic                                i_input_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_input_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_input_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_input_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_input_channel,
    input  wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_input_spatial,
    input  wire logic signed [15:0]                  i_input_q8_8,
    input  wire logic                                i_trace_valid,
    output logic                                o_trace_ready,
    input  wire logic                                i_trace_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_trace_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_trace_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_trace_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_trace_channel,
    input  wire logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0] i_trace_spatial,
    input  wire logic signed [15:0]                  i_trace_q8_8,
    input  wire logic                                i_surrogate_valid,
    output logic                                o_surrogate_ready,
    input  wire logic                                i_surrogate_target,
    input  wire logic [SAMPLE_WIDTH-1:0]             i_surrogate_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_surrogate_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_surrogate_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_surrogate_channel,
    input  wire logic [$clog2(SPATIAL*SPATIAL)-1:0] i_surrogate_spatial,
    input  wire logic signed [15:0]                  i_surrogate_q8_8,

    input  wire logic                                i_replay_valid,
    output logic                                o_replay_ready,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_replay_input_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_replay_input_channel_count,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_replay_output_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_replay_output_channel_count,
    output logic                                o_replay_busy,
    output logic                                o_replay_done,

    output logic                                o_replay_input_valid,
    input  wire logic                                i_replay_input_ready,
    output logic                                o_replay_input_target,
    output logic [SAMPLE_WIDTH-1:0]             o_replay_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_replay_input_timestep,
    output logic [VERSION_WIDTH-1:0]            o_replay_input_sample_version,
    output logic [CHANNEL_INDEX_WIDTH-1:0]      o_replay_input_channel,
    output logic [$clog2(SPATIAL*SPATIAL)-1:0] o_replay_input_spatial,
    output logic signed [15:0]                  o_replay_input_q8_8,
    output logic                                o_replay_trace_valid,
    input  wire logic                                i_replay_trace_ready,
    output logic                                o_replay_trace_target,
    output logic [SAMPLE_WIDTH-1:0]             o_replay_trace_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_replay_trace_timestep,
    output logic [VERSION_WIDTH-1:0]            o_replay_trace_sample_version,
    output logic [CHANNEL_INDEX_WIDTH-1:0]      o_replay_trace_channel,
    output logic [$clog2(POOLED_SPATIAL*POOLED_SPATIAL)-1:0] o_replay_trace_spatial,
    output logic signed [15:0]                  o_replay_trace_q8_8,
    output logic                                o_replay_surrogate_valid,
    input  wire logic                                i_replay_surrogate_ready,
    output logic                                o_replay_surrogate_target,
    output logic [SAMPLE_WIDTH-1:0]             o_replay_surrogate_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_replay_surrogate_timestep,
    output logic [VERSION_WIDTH-1:0]            o_replay_surrogate_sample_version,
    output logic [CHANNEL_INDEX_WIDTH-1:0]      o_replay_surrogate_channel,
    output logic [$clog2(SPATIAL*SPATIAL)-1:0] o_replay_surrogate_spatial,
    output logic signed [15:0]                  o_replay_surrogate_q8_8
);
    localparam int INPUT_PLANE = SPATIAL * SPATIAL;
    localparam int TRACE_PLANE = POOLED_SPATIAL * POOLED_SPATIAL;
    localparam int INPUT_DEPTH = MAX_BATCH * MAX_INPUT_CHANNELS * INPUT_PLANE;
    localparam int TRACE_DEPTH = MAX_BATCH * MAX_OUTPUT_CHANNELS * TRACE_PLANE;
    localparam int SURROGATE_DEPTH = MAX_BATCH * MAX_OUTPUT_CHANNELS * INPUT_PLANE;
    localparam int INPUT_CAPTURE_COUNT_WIDTH = $clog2(INPUT_DEPTH + 1);
    localparam int TRACE_CAPTURE_COUNT_WIDTH = $clog2(TRACE_DEPTH + 1);
    localparam int SURROGATE_CAPTURE_COUNT_WIDTH = $clog2(SURROGATE_DEPTH + 1);
    typedef enum logic [1:0] {RP_IDLE, RP_ISSUE, RP_HOLD, RP_DONE} replay_state_t;
    replay_state_t replay_state_q;
    logic [2:0] replay_phase_q;
    logic [SAMPLE_WIDTH-1:0] replay_sample_q;
    logic [CHANNEL_INDEX_WIDTH-1:0] replay_channel_q;
    logic [$clog2(INPUT_PLANE)-1:0] replay_spatial_q;
    logic [CHANNEL_INDEX_WIDTH-1:0] replay_input_base_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] replay_input_count_q;
    logic [CHANNEL_INDEX_WIDTH-1:0] replay_output_base_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] replay_output_count_q;
    logic signed [15:0] replay_data;
    logic signed [15:0] student_input_read_q, target_input_read_q;
    logic signed [15:0] student_trace_read_q, target_trace_read_q;
    logic signed [15:0] student_surrogate_read_q, target_surrogate_read_q;

    logic active_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] input_channels_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] output_channels_q;
    logic [INPUT_CAPTURE_COUNT_WIDTH-1:0] student_input_count_q;
    logic [INPUT_CAPTURE_COUNT_WIDTH-1:0] target_input_count_q;
    logic [TRACE_CAPTURE_COUNT_WIDTH-1:0] student_trace_count_q;
    logic [TRACE_CAPTURE_COUNT_WIDTH-1:0] target_trace_count_q;
    logic [SURROGATE_CAPTURE_COUNT_WIDTH-1:0] student_surrogate_count_q;
    logic [SURROGATE_CAPTURE_COUNT_WIDTH-1:0] target_surrogate_count_q;
    logic [SURROGATE_CAPTURE_COUNT_WIDTH-1:0] surrogate_expected_count;
    logic [SAMPLE_WIDTH-1:0] input_sample_q [0:1];
    logic [CHANNEL_INDEX_WIDTH-1:0] input_channel_q [0:1];
    logic [$clog2(INPUT_PLANE)-1:0] input_spatial_q [0:1];
    logic input_done_q [0:1];
    logic trace_done [0:1];
    logic surrogate_done [0:1];

    (* ram_style = "block" *) logic signed [15:0] student_input_mem [0:INPUT_DEPTH-1];
    (* ram_style = "block" *) logic signed [15:0] target_input_mem [0:INPUT_DEPTH-1];
    (* ram_style = "block" *) logic signed [15:0] student_trace_mem [0:TRACE_DEPTH-1];
    (* ram_style = "block" *) logic signed [15:0] target_trace_mem [0:TRACE_DEPTH-1];
    (* ram_style = "block" *) logic signed [15:0] student_surrogate_mem [0:SURROGATE_DEPTH-1];
    (* ram_style = "block" *) logic signed [15:0] target_surrogate_mem [0:SURROGATE_DEPTH-1];

    integer input_address;
    integer trace_address;
    integer surrogate_address;
    integer replay_plane;
    integer replay_address;
    logic cfg_legal;
    logic input_legal, trace_legal, surrogate_legal;
    logic input_order_legal;
    logic trace_order_legal;
    logic surrogate_order_legal;
    logic replay_cfg_legal;
    logic replay_sink_ready;
    logic replay_last_spatial;
    logic replay_last_channel;
    logic replay_last_sample;

    always_comb begin
        input_address = int'(i_input_sample) * MAX_INPUT_CHANNELS * INPUT_PLANE +
            int'(i_input_channel) * INPUT_PLANE + int'(i_input_spatial);
        trace_address = int'(i_trace_sample) * MAX_OUTPUT_CHANNELS * TRACE_PLANE +
            int'(i_trace_channel) * TRACE_PLANE + int'(i_trace_spatial);
        surrogate_address = int'(i_surrogate_sample) * MAX_OUTPUT_CHANNELS * INPUT_PLANE +
            int'(i_surrogate_channel) * INPUT_PLANE + int'(i_surrogate_spatial);
        surrogate_expected_count = SURROGATE_CAPTURE_COUNT_WIDTH'(
            int'(batch_count_q) * int'(output_channels_q) * INPUT_PLANE);

        cfg_legal = i_cfg_batch_count != 0 && int'(i_cfg_batch_count) <= MAX_BATCH &&
            int'(i_cfg_timestep) < MAX_TIMESTEPS &&
            i_cfg_input_channels != 0 &&
            int'(i_cfg_input_channels) <= MAX_INPUT_CHANNELS &&
            i_cfg_output_channels != 0 &&
            int'(i_cfg_output_channels) <= MAX_OUTPUT_CHANNELS;
        input_order_legal = !input_done_q[i_input_target] &&
            i_input_sample == input_sample_q[i_input_target] &&
            i_input_channel == input_channel_q[i_input_target] &&
            i_input_spatial == input_spatial_q[i_input_target];
        input_legal =
            i_input_timestep == timestep_q &&
            i_input_sample_version == sample_version_q &&
            input_order_legal;
        trace_legal =
            i_trace_timestep == timestep_q &&
            i_trace_sample_version == sample_version_q &&
            trace_order_legal;
        surrogate_legal =
            i_surrogate_timestep == timestep_q &&
            i_surrogate_sample_version == sample_version_q &&
            surrogate_order_legal;

        replay_plane = (replay_phase_q inside {3'd0, 3'd1, 3'd4, 3'd5}) ?
            INPUT_PLANE : TRACE_PLANE;
        if (replay_phase_q inside {3'd0, 3'd1})
            replay_address = int'(replay_sample_q) * MAX_INPUT_CHANNELS * INPUT_PLANE +
                int'(replay_channel_q) * INPUT_PLANE + int'(replay_spatial_q);
        else if (replay_phase_q inside {3'd2, 3'd3})
            replay_address = int'(replay_sample_q) * MAX_OUTPUT_CHANNELS * TRACE_PLANE +
                int'(replay_channel_q) * TRACE_PLANE + int'(replay_spatial_q);
        else
            replay_address = int'(replay_sample_q) * MAX_OUTPUT_CHANNELS * INPUT_PLANE +
                int'(replay_channel_q) * INPUT_PLANE + int'(replay_spatial_q);

        replay_last_spatial = int'(replay_spatial_q) + 1 == replay_plane;
        replay_last_channel = replay_phase_q inside {3'd0, 3'd1} ?
            int'(replay_channel_q) + 1 ==
                int'(replay_input_base_q) + int'(replay_input_count_q) :
            int'(replay_channel_q) + 1 ==
                int'(replay_output_base_q) + int'(replay_output_count_q);
        replay_last_sample = int'(replay_sample_q) + 1 == int'(batch_count_q);

        replay_cfg_legal = i_replay_input_channel_count != 0 &&
            i_replay_output_channel_count != 0 &&
            int'(i_replay_input_channel_base) + int'(i_replay_input_channel_count) <=
                int'(input_channels_q) &&
            int'(i_replay_output_channel_base) + int'(i_replay_output_channel_count) <=
                int'(output_channels_q);
        replay_sink_ready = replay_phase_q inside {3'd0, 3'd1} ?
            i_replay_input_ready :
            (replay_phase_q inside {3'd2, 3'd3} ? i_replay_trace_ready :
                i_replay_surrogate_ready);
    end

    assign o_cfg_ready = i_rst_n && !active_q;
    assign o_active = active_q;
    assign o_input_ready = active_q && replay_state_q == RP_IDLE &&
        !o_capture_complete && !input_done_q[i_input_target];
    assign o_trace_ready = active_q && replay_state_q == RP_IDLE &&
        !o_capture_complete && !trace_done[i_trace_target];
    assign o_surrogate_ready = active_q && replay_state_q == RP_IDLE &&
        !o_capture_complete && !surrogate_done[i_surrogate_target];
    assign o_capture_complete = active_q &&
        input_done_q[0] && input_done_q[1] &&
        trace_done[0] && trace_done[1] &&
        surrogate_done[0] && surrogate_done[1];
    assign o_replay_ready = o_capture_complete && replay_state_q == RP_IDLE;
    assign o_replay_busy = replay_state_q inside {RP_ISSUE, RP_HOLD};
    assign o_replay_done = replay_state_q == RP_DONE;
    assign o_release_ready = active_q && replay_state_q == RP_IDLE;

    assign o_replay_input_valid = replay_state_q == RP_HOLD &&
        replay_phase_q inside {3'd0, 3'd1};
    assign o_replay_trace_valid = replay_state_q == RP_HOLD &&
        replay_phase_q inside {3'd2, 3'd3};
    assign o_replay_surrogate_valid = replay_state_q == RP_HOLD &&
        replay_phase_q inside {3'd4, 3'd5};
    assign o_replay_input_target = replay_phase_q == 3'd1;
    assign o_replay_trace_target = replay_phase_q == 3'd3;
    assign o_replay_surrogate_target = replay_phase_q == 3'd5;
    assign o_replay_input_sample = replay_sample_q;
    assign o_replay_trace_sample = replay_sample_q;
    assign o_replay_surrogate_sample = replay_sample_q;
    assign o_replay_input_timestep = timestep_q;
    assign o_replay_trace_timestep = timestep_q;
    assign o_replay_surrogate_timestep = timestep_q;
    assign o_replay_input_sample_version = sample_version_q;
    assign o_replay_trace_sample_version = sample_version_q;
    assign o_replay_surrogate_sample_version = sample_version_q;
    assign o_replay_input_channel = replay_channel_q;
    assign o_replay_trace_channel = replay_channel_q;
    assign o_replay_surrogate_channel = replay_channel_q;
    assign o_replay_input_spatial = $clog2(INPUT_PLANE)'(replay_spatial_q);
    assign o_replay_trace_spatial = $clog2(TRACE_PLANE)'(replay_spatial_q);
    assign o_replay_surrogate_spatial = $clog2(INPUT_PLANE)'(replay_spatial_q);
    always_comb begin
        unique case (replay_phase_q)
            3'd0: replay_data = student_input_read_q;
            3'd1: replay_data = target_input_read_q;
            3'd2: replay_data = student_trace_read_q;
            3'd3: replay_data = target_trace_read_q;
            3'd4: replay_data = student_surrogate_read_q;
            default: replay_data = target_surrogate_read_q;
        endcase
    end
    assign o_replay_input_q8_8 = replay_data;
    assign o_replay_trace_q8_8 = replay_data;
    assign o_replay_surrogate_q8_8 = replay_data;

    // Capture coordinates are explicit state. Keeping division/modulo out of
    // the RAM write-enable cone avoids a variable-divider control path.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            for (int branch = 0; branch < 2; branch++) begin
                input_sample_q[branch] <= '0;
                input_channel_q[branch] <= '0;
                input_spatial_q[branch] <= '0;
                input_done_q[branch] <= 1'b0;
            end
        end else if (i_cfg_valid && o_cfg_ready) begin
            for (int branch = 0; branch < 2; branch++) begin
                input_sample_q[branch] <= '0;
                input_channel_q[branch] <= '0;
                input_spatial_q[branch] <= '0;
                input_done_q[branch] <= 1'b0;
            end
        end else if (i_input_valid && o_input_ready && input_legal) begin
            for (int branch = 0; branch < 2; branch++) begin
                if (i_input_target == 1'(branch)) begin
                    if (int'(input_spatial_q[branch]) + 1 == INPUT_PLANE) begin
                        input_spatial_q[branch] <= '0;
                        if (int'(input_channel_q[branch]) + 1 ==
                            int'(input_channels_q)) begin
                            input_channel_q[branch] <= '0;
                            if (int'(input_sample_q[branch]) + 1 ==
                                int'(batch_count_q)) begin
                                input_sample_q[branch] <= '0;
                                input_done_q[branch] <= 1'b1;
                            end else
                                input_sample_q[branch] <=
                                    input_sample_q[branch] + 1'b1;
                        end else
                            input_channel_q[branch] <=
                                input_channel_q[branch] + 1'b1;
                    end else
                        input_spatial_q[branch] <=
                            input_spatial_q[branch] + 1'b1;
                end
            end
        end
    end

    // Production emits wave -> spatial -> channel; replay remains channel-major.
    // Write by coordinate into the existing store, without a second transpose RAM.
    generate
        if (TRACE_WAVE_CHANNELS > 0) begin : g_trace_wave_capture_order
            logic [CHANNEL_COUNT_WIDTH-1:0] base_q [0:1];
            logic [CHANNEL_COUNT_WIDTH-1:0] lane_q [0:1];
            logic [$clog2(TRACE_PLANE)-1:0] spatial_q [0:1];
            logic [SAMPLE_WIDTH-1:0] sample_q [0:1];
            logic done_q [0:1];
            assign trace_done[0] = done_q[0];
            assign trace_done[1] = done_q[1];
            assign trace_order_legal = !done_q[i_trace_target] &&
                i_trace_sample == sample_q[i_trace_target] &&
                int'(i_trace_channel) == int'(base_q[i_trace_target]) +
                    int'(lane_q[i_trace_target]) &&
                i_trace_spatial == spatial_q[i_trace_target];
            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        base_q[branch] <= '0;
                        lane_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        sample_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_cfg_valid && o_cfg_ready) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        base_q[branch] <= '0;
                        lane_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        sample_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_trace_valid && o_trace_ready && trace_legal) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        if (i_trace_target == 1'(branch)) begin
                            if (int'(lane_q[branch]) + 1 == TRACE_WAVE_CHANNELS ||
                                int'(base_q[branch]) + int'(lane_q[branch]) + 1 ==
                                    int'(output_channels_q)) begin
                                lane_q[branch] <= '0;
                                if (int'(spatial_q[branch]) + 1 == TRACE_PLANE) begin
                                    spatial_q[branch] <= '0;
                                    if (int'(base_q[branch]) + int'(lane_q[branch]) + 1 ==
                                        int'(output_channels_q)) begin
                                        base_q[branch] <= '0;
                                        if (int'(sample_q[branch]) + 1 ==
                                            int'(batch_count_q)) begin
                                            sample_q[branch] <= '0;
                                            done_q[branch] <= 1'b1;
                                        end else
                                            sample_q[branch] <=
                                                sample_q[branch] + 1'b1;
                                    end else
                                        base_q[branch] <= base_q[branch] +
                                            CHANNEL_COUNT_WIDTH'(TRACE_WAVE_CHANNELS);
                                end else
                                    spatial_q[branch] <= spatial_q[branch] + 1'b1;
                            end else
                                lane_q[branch] <= lane_q[branch] + 1'b1;
                        end
                    end
                end
            end
        end else begin : g_trace_channel_capture_order
            logic [SAMPLE_WIDTH-1:0] sample_q [0:1];
            logic [CHANNEL_INDEX_WIDTH-1:0] channel_q [0:1];
            logic [$clog2(TRACE_PLANE)-1:0] spatial_q [0:1];
            logic done_q [0:1];
            assign trace_done[0] = done_q[0];
            assign trace_done[1] = done_q[1];
            assign trace_order_legal = !done_q[i_trace_target] &&
                i_trace_sample == sample_q[i_trace_target] &&
                i_trace_channel == channel_q[i_trace_target] &&
                i_trace_spatial == spatial_q[i_trace_target];
            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        sample_q[branch] <= '0;
                        channel_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_cfg_valid && o_cfg_ready) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        sample_q[branch] <= '0;
                        channel_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_trace_valid && o_trace_ready && trace_legal) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        if (i_trace_target == 1'(branch)) begin
                            if (int'(spatial_q[branch]) + 1 == TRACE_PLANE) begin
                                spatial_q[branch] <= '0;
                                if (int'(channel_q[branch]) + 1 ==
                                    int'(output_channels_q)) begin
                                    channel_q[branch] <= '0;
                                    if (int'(sample_q[branch]) + 1 ==
                                        int'(batch_count_q)) begin
                                        sample_q[branch] <= '0;
                                        done_q[branch] <= 1'b1;
                                    end else
                                        sample_q[branch] <= sample_q[branch] + 1'b1;
                                end else
                                    channel_q[branch] <= channel_q[branch] + 1'b1;
                            end else
                                spatial_q[branch] <= spatial_q[branch] + 1'b1;
                        end
                    end
                end
            end
        end

        if (ALLOW_UNORDERED_SURROGATE_CAPTURE) begin : g_unordered_capture
            // This mode is only valid behind a producer that guarantees each
            // explicit (branch, sample, channel, spatial) coordinate once.
            // Addressed writes absorb tile completion order without a reorder RAM.
            assign surrogate_done[0] =
                student_surrogate_count_q == surrogate_expected_count;
            assign surrogate_done[1] =
                target_surrogate_count_q == surrogate_expected_count;
            assign surrogate_order_legal =
                !surrogate_done[i_surrogate_target] &&
                int'(i_surrogate_sample) < int'(batch_count_q) &&
                int'(i_surrogate_channel) < int'(output_channels_q) &&
                int'(i_surrogate_spatial) < INPUT_PLANE;
        end else if (SURROGATE_SPATIAL_BLOCK_SIZE > 0) begin : g_block_capture_order
            logic [SAMPLE_WIDTH-1:0] sample_q [0:1];
            logic [CHANNEL_INDEX_WIDTH-1:0] channel_q [0:1];
            logic [$clog2(INPUT_PLANE)-1:0] block_base_q [0:1];
            logic [$clog2(INPUT_PLANE)-1:0] lane_q [0:1];
            logic done_q [0:1];
            assign surrogate_done[0] = done_q[0];
            assign surrogate_done[1] = done_q[1];
            assign surrogate_order_legal = !done_q[i_surrogate_target] &&
                i_surrogate_sample == sample_q[i_surrogate_target] &&
                i_surrogate_channel == channel_q[i_surrogate_target] &&
                int'(i_surrogate_spatial) ==
                    int'(block_base_q[i_surrogate_target]) +
                    int'(lane_q[i_surrogate_target]);
            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        sample_q[branch] <= '0;
                        channel_q[branch] <= '0;
                        block_base_q[branch] <= '0;
                        lane_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_cfg_valid && o_cfg_ready) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        sample_q[branch] <= '0;
                        channel_q[branch] <= '0;
                        block_base_q[branch] <= '0;
                        lane_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_surrogate_valid && o_surrogate_ready &&
                             surrogate_legal) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        if (i_surrogate_target == 1'(branch)) begin
                            if (int'(lane_q[branch]) + 1 ==
                                SURROGATE_SPATIAL_BLOCK_SIZE) begin
                                lane_q[branch] <= '0;
                                if (int'(channel_q[branch]) + 1 ==
                                    int'(output_channels_q)) begin
                                    channel_q[branch] <= '0;
                                    if (int'(block_base_q[branch]) +
                                        SURROGATE_SPATIAL_BLOCK_SIZE ==
                                        INPUT_PLANE) begin
                                        block_base_q[branch] <= '0;
                                        if (int'(sample_q[branch]) + 1 ==
                                            int'(batch_count_q)) begin
                                            sample_q[branch] <= '0;
                                            done_q[branch] <= 1'b1;
                                        end else
                                            sample_q[branch] <=
                                                sample_q[branch] + 1'b1;
                                    end else
                                        block_base_q[branch] <=
                                            block_base_q[branch] +
                                            $clog2(INPUT_PLANE)'(
                                                SURROGATE_SPATIAL_BLOCK_SIZE);
                                end else
                                    channel_q[branch] <=
                                        channel_q[branch] + 1'b1;
                            end else
                                lane_q[branch] <= lane_q[branch] + 1'b1;
                        end
                    end
                end
            end
        end else if (SURROGATE_WAVE_CHANNELS > 0) begin : g_wave_capture_order
            logic [CHANNEL_COUNT_WIDTH-1:0] base_q [0:1];
            logic [CHANNEL_COUNT_WIDTH-1:0] lane_q [0:1];
            logic [$clog2(INPUT_PLANE)-1:0] spatial_q [0:1];
            logic [SAMPLE_WIDTH-1:0] sample_q [0:1];
            logic done_q [0:1];
            assign surrogate_done[0] = done_q[0];
            assign surrogate_done[1] = done_q[1];
            assign surrogate_order_legal = !done_q[i_surrogate_target] &&
                i_surrogate_sample == sample_q[i_surrogate_target] &&
                int'(i_surrogate_channel) == int'(base_q[i_surrogate_target]) +
                    int'(lane_q[i_surrogate_target]) &&
                i_surrogate_spatial == spatial_q[i_surrogate_target];
            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        base_q[branch] <= '0;
                        lane_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        sample_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_cfg_valid && o_cfg_ready) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        base_q[branch] <= '0;
                        lane_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        sample_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_surrogate_valid && o_surrogate_ready &&
                             surrogate_legal) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        if (i_surrogate_target == 1'(branch)) begin
                            if (int'(lane_q[branch]) + 1 == SURROGATE_WAVE_CHANNELS ||
                                int'(base_q[branch]) + int'(lane_q[branch]) + 1 ==
                                    int'(output_channels_q)) begin
                                lane_q[branch] <= '0;
                                if (int'(spatial_q[branch]) + 1 == INPUT_PLANE) begin
                                    spatial_q[branch] <= '0;
                                    if (int'(base_q[branch]) + int'(lane_q[branch]) + 1 ==
                                        int'(output_channels_q)) begin
                                        base_q[branch] <= '0;
                                        if (int'(sample_q[branch]) + 1 ==
                                            int'(batch_count_q)) begin
                                            sample_q[branch] <= '0;
                                            done_q[branch] <= 1'b1;
                                        end else
                                            sample_q[branch] <=
                                                sample_q[branch] + 1'b1;
                                    end else
                                        base_q[branch] <= base_q[branch] +
                                            CHANNEL_COUNT_WIDTH'(SURROGATE_WAVE_CHANNELS);
                                end else
                                    spatial_q[branch] <= spatial_q[branch] + 1'b1;
                            end else
                                lane_q[branch] <= lane_q[branch] + 1'b1;
                        end
                    end
                end
            end
        end else begin : g_channel_capture_order
            logic [SAMPLE_WIDTH-1:0] sample_q [0:1];
            logic [CHANNEL_INDEX_WIDTH-1:0] channel_q [0:1];
            logic [$clog2(INPUT_PLANE)-1:0] spatial_q [0:1];
            logic done_q [0:1];
            assign surrogate_done[0] = done_q[0];
            assign surrogate_done[1] = done_q[1];
            assign surrogate_order_legal = !done_q[i_surrogate_target] &&
                i_surrogate_sample == sample_q[i_surrogate_target] &&
                i_surrogate_channel == channel_q[i_surrogate_target] &&
                i_surrogate_spatial == spatial_q[i_surrogate_target];
            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        sample_q[branch] <= '0;
                        channel_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_cfg_valid && o_cfg_ready) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        sample_q[branch] <= '0;
                        channel_q[branch] <= '0;
                        spatial_q[branch] <= '0;
                        done_q[branch] <= 1'b0;
                    end
                end else if (i_surrogate_valid && o_surrogate_ready &&
                             surrogate_legal) begin
                    for (int branch = 0; branch < 2; branch++) begin
                        if (i_surrogate_target == 1'(branch)) begin
                            if (int'(spatial_q[branch]) + 1 == INPUT_PLANE) begin
                                spatial_q[branch] <= '0;
                                if (int'(channel_q[branch]) + 1 ==
                                    int'(output_channels_q)) begin
                                    channel_q[branch] <= '0;
                                    if (int'(sample_q[branch]) + 1 ==
                                        int'(batch_count_q)) begin
                                        sample_q[branch] <= '0;
                                        done_q[branch] <= 1'b1;
                                    end else
                                        sample_q[branch] <= sample_q[branch] + 1'b1;
                                end else
                                    channel_q[branch] <= channel_q[branch] + 1'b1;
                            end else
                                spatial_q[branch] <= spatial_q[branch] + 1'b1;
                        end
                    end
                end
            end
        end
    endgenerate

    // Payload storage is intentionally reset-free. Counters and metadata are
    // the only validity authority, preserving block-memory inference.
    always_ff @(posedge i_clk) begin
        if (i_input_valid && o_input_ready && input_legal) begin
            if (i_input_target)
                target_input_mem[input_address] <= i_input_q8_8;
            else
                student_input_mem[input_address] <= i_input_q8_8;
        end
        if (i_trace_valid && o_trace_ready && trace_legal) begin
            if (i_trace_target)
                target_trace_mem[trace_address] <= i_trace_q8_8;
            else
                student_trace_mem[trace_address] <= i_trace_q8_8;
        end
        if (i_surrogate_valid && o_surrogate_ready && surrogate_legal) begin
            if (i_surrogate_target)
                target_surrogate_mem[surrogate_address] <= i_surrogate_q8_8;
            else
                student_surrogate_mem[surrogate_address] <= i_surrogate_q8_8;
        end
        if (replay_state_q == RP_ISSUE)
            unique case (replay_phase_q)
                3'd0: student_input_read_q <= student_input_mem[replay_address];
                3'd1: target_input_read_q <= target_input_mem[replay_address];
                3'd2: student_trace_read_q <= student_trace_mem[replay_address];
                3'd3: target_trace_read_q <= target_trace_mem[replay_address];
                3'd4: student_surrogate_read_q <= student_surrogate_mem[replay_address];
                default: target_surrogate_read_q <= target_surrogate_mem[replay_address];
            endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            batch_count_q <= '0;
            timestep_q <= '0;
            sample_version_q <= '0;
            input_channels_q <= '0;
            output_channels_q <= '0;
            student_input_count_q <= '0;
            target_input_count_q <= '0;
            student_trace_count_q <= '0;
            target_trace_count_q <= '0;
            student_surrogate_count_q <= '0;
            target_surrogate_count_q <= '0;
            replay_state_q <= RP_IDLE;
            replay_phase_q <= '0;
            replay_sample_q <= '0;
            replay_channel_q <= '0;
            replay_spatial_q <= '0;
            replay_input_base_q <= '0;
            replay_input_count_q <= '0;
            replay_output_base_q <= '0;
            replay_output_count_q <= '0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_cfg_valid && o_cfg_ready) begin
                active_q <= cfg_legal;
                batch_count_q <= i_cfg_batch_count;
                timestep_q <= i_cfg_timestep;
                sample_version_q <= i_cfg_sample_version;
                input_channels_q <= i_cfg_input_channels;
                output_channels_q <= i_cfg_output_channels;
                student_input_count_q <= '0;
                target_input_count_q <= '0;
                student_trace_count_q <= '0;
                target_trace_count_q <= '0;
                student_surrogate_count_q <= '0;
                target_surrogate_count_q <= '0;
                replay_state_q <= RP_IDLE;
                o_protocol_error <= !cfg_legal;
            end else if (i_release && o_release_ready) begin
                active_q <= 1'b0;
            end else begin
                if (i_input_valid && o_input_ready) begin
                    if (!input_legal)
                        o_protocol_error <= 1'b1;
                    else if (i_input_target)
                        target_input_count_q <= target_input_count_q + 1'b1;
                    else
                        student_input_count_q <= student_input_count_q + 1'b1;
                end
                if (i_trace_valid && o_trace_ready) begin
                    if (!trace_legal)
                        o_protocol_error <= 1'b1;
                    else if (i_trace_target)
                        target_trace_count_q <= target_trace_count_q + 1'b1;
                    else
                        student_trace_count_q <= student_trace_count_q + 1'b1;
                end
                if (i_surrogate_valid && o_surrogate_ready) begin
                    if (!surrogate_legal)
                        o_protocol_error <= 1'b1;
                    else if (i_surrogate_target)
                        target_surrogate_count_q <= target_surrogate_count_q + 1'b1;
                    else
                        student_surrogate_count_q <= student_surrogate_count_q + 1'b1;
                end

                case (replay_state_q)
                    RP_IDLE: if (i_replay_valid && o_replay_ready) begin
                        if (!replay_cfg_legal) begin
                            o_protocol_error <= 1'b1;
                        end else begin
                            replay_input_base_q <= i_replay_input_channel_base;
                            replay_input_count_q <= i_replay_input_channel_count;
                            replay_output_base_q <= i_replay_output_channel_base;
                            replay_output_count_q <= i_replay_output_channel_count;
                            replay_phase_q <= 3'd0;
                            replay_sample_q <= '0;
                            replay_channel_q <= i_replay_input_channel_base;
                            replay_spatial_q <= '0;
                            replay_state_q <= RP_ISSUE;
                        end
                    end
                    RP_ISSUE: replay_state_q <= RP_HOLD;
                    RP_HOLD: if (replay_sink_ready) begin
                        replay_state_q <= RP_ISSUE;
                        if (replay_last_spatial) begin
                            replay_spatial_q <= '0;
                            if (replay_last_channel) begin
                                if (replay_last_sample) begin
                                    replay_sample_q <= '0;
                                    if (replay_phase_q == 3'd5) begin
                                        replay_state_q <= RP_DONE;
                                        replay_channel_q <= replay_output_base_q;
                                    end else begin
                                        replay_phase_q <= replay_phase_q + 1'b1;
                                        replay_channel_q <= replay_phase_q == 3'd0 ?
                                            replay_input_base_q : replay_output_base_q;
                                    end
                                end else begin
                                    replay_sample_q <= replay_sample_q + 1'b1;
                                    replay_channel_q <= replay_phase_q inside {3'd0, 3'd1} ?
                                        replay_input_base_q : replay_output_base_q;
                                end
                            end else
                                replay_channel_q <= replay_channel_q + 1'b1;
                        end else
                            replay_spatial_q <= replay_spatial_q + 1'b1;
                    end
                    RP_DONE: replay_state_q <= RP_IDLE;
                    default: begin
                        replay_state_q <= RP_IDLE;
                        o_protocol_error <= 1'b1;
                    end
                endcase
            end
        end
    end

    initial begin
        if (SURROGATE_WAVE_CHANNELS < 0 ||
            SURROGATE_WAVE_CHANNELS > (1 << CHANNEL_INDEX_WIDTH))
            $error("Conv8 surrogate capture wave width is invalid");
        if (SURROGATE_SPATIAL_BLOCK_SIZE < 0 ||
            SURROGATE_SPATIAL_BLOCK_SIZE > INPUT_PLANE ||
            (SURROGATE_SPATIAL_BLOCK_SIZE > 0 &&
             (INPUT_PLANE % SURROGATE_SPATIAL_BLOCK_SIZE) != 0))
            $error("Conv8 surrogate spatial block size is invalid");
        if (SURROGATE_SPATIAL_BLOCK_SIZE > 0 &&
            SURROGATE_WAVE_CHANNELS > 0)
            $error("Conv8 surrogate capture order is ambiguous");
        if (TRACE_WAVE_CHANNELS < 0 ||
            TRACE_WAVE_CHANNELS > (1 << CHANNEL_INDEX_WIDTH))
            $error("Conv8 trace capture wave width is invalid");
        if (SPATIAL != 4 || POOLED_SPATIAL != 2)
            $error("Conv8 observation store requires the 4x4/AAVG2 contract");
    end
endmodule

`default_nettype wire
