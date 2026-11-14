`timescale 1ns/1ps
`default_nettype none

// Selects one Conv8 output-wave/input-group working set from production
// sample-major observations. Matching beats are translated from global
// channel/spatial coordinates to the dense local indices consumed by the
// bounded TP tile engine. Nonmatching channels pass without backpressure.
module unified_scnn_tp_conv_tile_observation_adapter #(
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int MAX_INPUT_CHANNELS = 8,
    parameter int MAX_OUTPUT_CHANNELS = 12,
    parameter int SPATIAL = 4,
    parameter int POOLED_SPATIAL = 2,
    parameter int VERSION_WIDTH = 16,
    parameter int SAMPLE_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_WIDTH = (MAX_TIMESTEPS <= 2) ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int CHANNEL_INDEX_WIDTH = 9,
    parameter int CHANNEL_COUNT_WIDTH = 10,
    parameter int INPUT_INDEX_WIDTH =
        (MAX_INPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_INPUT_CHANNELS*SPATIAL*SPATIAL),
    parameter int TRACE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*POOLED_SPATIAL*POOLED_SPATIAL),
    parameter int SURROGATE_INDEX_WIDTH =
        (MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL <= 2) ? 1 :
        $clog2(MAX_OUTPUT_CHANNELS*SPATIAL*SPATIAL)
) (
    input  wire logic                                i_clk,
    input  wire logic                                i_rst_n,
    input  wire logic                                i_cfg_valid,
    output logic                                o_cfg_ready,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]        i_cfg_batch_count,
    input  wire logic [TIMESTEP_WIDTH-1:0]           i_cfg_timestep,
    input  wire logic [VERSION_WIDTH-1:0]            i_cfg_sample_version,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_cfg_input_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_input_channel_count,
    input  wire logic [CHANNEL_INDEX_WIDTH-1:0]      i_cfg_output_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0]      i_cfg_output_channel_count,
    output logic                                o_active,
    input  wire logic                                i_release,
    output logic                                o_release_ready,
    output logic                                o_capture_complete,
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

    output logic                                o_student_input_valid,
    input  wire logic                                i_student_input_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_student_input_sample,
    output logic [INPUT_INDEX_WIDTH-1:0]        o_student_input_index,
    output logic signed [15:0]                  o_student_input_q8_8,
    output logic                                o_target_input_valid,
    input  wire logic                                i_target_input_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_input_sample,
    output logic [INPUT_INDEX_WIDTH-1:0]        o_target_input_index,
    output logic signed [15:0]                  o_target_input_q8_8,
    output logic                                o_student_trace_valid,
    input  wire logic                                i_student_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_student_trace_sample,
    output logic [TRACE_INDEX_WIDTH-1:0]        o_student_trace_index,
    output logic signed [15:0]                  o_student_trace_q8_8,
    output logic                                o_target_trace_valid,
    input  wire logic                                i_target_trace_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_trace_sample,
    output logic [TRACE_INDEX_WIDTH-1:0]        o_target_trace_index,
    output logic signed [15:0]                  o_target_trace_q8_8,
    output logic                                o_student_surrogate_valid,
    input  wire logic                                i_student_surrogate_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_student_surrogate_sample,
    output logic [SURROGATE_INDEX_WIDTH-1:0]    o_student_surrogate_index,
    output logic signed [15:0]                  o_student_surrogate_q8_8,
    output logic                                o_target_surrogate_valid,
    input  wire logic                                i_target_surrogate_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_surrogate_sample,
    output logic [SURROGATE_INDEX_WIDTH-1:0]    o_target_surrogate_index,
    output logic signed [15:0]                  o_target_surrogate_q8_8
);
    localparam int INPUT_PLANE = SPATIAL * SPATIAL;
    localparam int TRACE_PLANE = POOLED_SPATIAL * POOLED_SPATIAL;
    localparam int MAX_INPUT_BEATS = MAX_BATCH * MAX_INPUT_CHANNELS * INPUT_PLANE;
    localparam int MAX_TRACE_BEATS = MAX_BATCH * MAX_OUTPUT_CHANNELS * TRACE_PLANE;
    localparam int MAX_SURROGATE_BEATS = MAX_BATCH * MAX_OUTPUT_CHANNELS * INPUT_PLANE;
    localparam int INPUT_COUNT_WIDTH = $clog2(MAX_INPUT_BEATS + 1);
    localparam int TRACE_COUNT_WIDTH = $clog2(MAX_TRACE_BEATS + 1);
    localparam int SURROGATE_COUNT_WIDTH = $clog2(MAX_SURROGATE_BEATS + 1);

    logic active_q;
    logic [BATCH_COUNT_WIDTH-1:0] batch_count_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic [CHANNEL_INDEX_WIDTH-1:0] input_base_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] input_count_q;
    logic [CHANNEL_INDEX_WIDTH-1:0] output_base_q;
    logic [CHANNEL_COUNT_WIDTH-1:0] output_count_q;
    logic [INPUT_COUNT_WIDTH-1:0] student_input_count_q;
    logic [INPUT_COUNT_WIDTH-1:0] target_input_count_q;
    logic [TRACE_COUNT_WIDTH-1:0] student_trace_count_q;
    logic [TRACE_COUNT_WIDTH-1:0] target_trace_count_q;
    logic [SURROGATE_COUNT_WIDTH-1:0] student_surrogate_count_q;
    logic [SURROGATE_COUNT_WIDTH-1:0] target_surrogate_count_q;
    logic [SAMPLE_WIDTH-1:0] input_expected_sample_q [0:1];
    logic [CHANNEL_COUNT_WIDTH-1:0] input_expected_channel_q [0:1];
    logic [$clog2(INPUT_PLANE)-1:0] input_expected_spatial_q [0:1];
    logic [SAMPLE_WIDTH-1:0] trace_expected_sample_q [0:1];
    logic [CHANNEL_COUNT_WIDTH-1:0] trace_expected_channel_q [0:1];
    logic [$clog2(TRACE_PLANE)-1:0] trace_expected_spatial_q [0:1];
    logic [SAMPLE_WIDTH-1:0] surrogate_expected_sample_q [0:1];
    logic [CHANNEL_COUNT_WIDTH-1:0] surrogate_expected_channel_q [0:1];
    logic [$clog2(INPUT_PLANE)-1:0] surrogate_expected_spatial_q [0:1];

    integer input_beats;
    integer trace_beats;
    integer surrogate_beats;
    integer input_local_index;
    integer trace_local_index;
    integer surrogate_local_index;
    logic input_selected, trace_selected, surrogate_selected;
    logic input_metadata_legal, trace_metadata_legal, surrogate_metadata_legal;
    logic input_order_legal, trace_order_legal, surrogate_order_legal;
    logic input_forward, trace_forward, surrogate_forward;
    logic cfg_legal;

    always_comb begin
        input_beats = int'(batch_count_q) * int'(input_count_q) * INPUT_PLANE;
        trace_beats = int'(batch_count_q) * int'(output_count_q) * TRACE_PLANE;
        surrogate_beats = int'(batch_count_q) * int'(output_count_q) * INPUT_PLANE;
        input_local_index =
            (int'(i_input_channel) - int'(input_base_q)) * INPUT_PLANE +
            int'(i_input_spatial);
        trace_local_index =
            (int'(i_trace_channel) - int'(output_base_q)) * TRACE_PLANE +
            int'(i_trace_spatial);
        surrogate_local_index =
            (int'(i_surrogate_channel) - int'(output_base_q)) * INPUT_PLANE +
            int'(i_surrogate_spatial);

        input_selected = active_q &&
            int'(i_input_channel) >= int'(input_base_q) &&
            int'(i_input_channel) < int'(input_base_q) + int'(input_count_q);
        trace_selected = active_q &&
            int'(i_trace_channel) >= int'(output_base_q) &&
            int'(i_trace_channel) < int'(output_base_q) + int'(output_count_q);
        surrogate_selected = active_q &&
            int'(i_surrogate_channel) >= int'(output_base_q) &&
            int'(i_surrogate_channel) < int'(output_base_q) + int'(output_count_q);

        input_metadata_legal = int'(i_input_sample) < int'(batch_count_q) &&
            i_input_timestep == timestep_q &&
            i_input_sample_version == sample_version_q &&
            int'(i_input_spatial) < INPUT_PLANE;
        trace_metadata_legal = int'(i_trace_sample) < int'(batch_count_q) &&
            i_trace_timestep == timestep_q &&
            i_trace_sample_version == sample_version_q &&
            int'(i_trace_spatial) < TRACE_PLANE;
        surrogate_metadata_legal = int'(i_surrogate_sample) < int'(batch_count_q) &&
            i_surrogate_timestep == timestep_q &&
            i_surrogate_sample_version == sample_version_q &&
            int'(i_surrogate_spatial) < INPUT_PLANE;

        input_order_legal =
            int'(input_expected_sample_q[i_input_target]) < int'(batch_count_q) &&
            i_input_sample == input_expected_sample_q[i_input_target] &&
            int'(i_input_channel) == int'(input_base_q) +
                int'(input_expected_channel_q[i_input_target]) &&
            i_input_spatial == input_expected_spatial_q[i_input_target];
        trace_order_legal =
            int'(trace_expected_sample_q[i_trace_target]) < int'(batch_count_q) &&
            i_trace_sample == trace_expected_sample_q[i_trace_target] &&
            int'(i_trace_channel) == int'(output_base_q) +
                int'(trace_expected_channel_q[i_trace_target]) &&
            i_trace_spatial == trace_expected_spatial_q[i_trace_target];
        surrogate_order_legal =
            int'(surrogate_expected_sample_q[i_surrogate_target]) <
                int'(batch_count_q) &&
            i_surrogate_sample == surrogate_expected_sample_q[i_surrogate_target] &&
            int'(i_surrogate_channel) == int'(output_base_q) +
                int'(surrogate_expected_channel_q[i_surrogate_target]) &&
            i_surrogate_spatial == surrogate_expected_spatial_q[i_surrogate_target];

        input_forward = input_selected && input_metadata_legal && input_order_legal;
        trace_forward = trace_selected && trace_metadata_legal && trace_order_legal;
        surrogate_forward = surrogate_selected && surrogate_metadata_legal &&
            surrogate_order_legal;

        o_student_input_valid = i_input_valid && input_forward && !i_input_target;
        o_target_input_valid = i_input_valid && input_forward && i_input_target;
        o_student_trace_valid = i_trace_valid && trace_forward && !i_trace_target;
        o_target_trace_valid = i_trace_valid && trace_forward && i_trace_target;
        o_student_surrogate_valid = i_surrogate_valid && surrogate_forward &&
            !i_surrogate_target;
        o_target_surrogate_valid = i_surrogate_valid && surrogate_forward &&
            i_surrogate_target;

        o_input_ready = !active_q || !input_selected || !input_forward ||
            (i_input_target ? i_target_input_ready : i_student_input_ready);
        o_trace_ready = !active_q || !trace_selected || !trace_forward ||
            (i_trace_target ? i_target_trace_ready : i_student_trace_ready);
        o_surrogate_ready = !active_q || !surrogate_selected || !surrogate_forward ||
            (i_surrogate_target ? i_target_surrogate_ready :
                i_student_surrogate_ready);

        o_student_input_sample = i_input_sample;
        o_target_input_sample = i_input_sample;
        o_student_input_index = INPUT_INDEX_WIDTH'(input_local_index);
        o_target_input_index = INPUT_INDEX_WIDTH'(input_local_index);
        o_student_input_q8_8 = i_input_q8_8;
        o_target_input_q8_8 = i_input_q8_8;
        o_student_trace_sample = i_trace_sample;
        o_target_trace_sample = i_trace_sample;
        o_student_trace_index = TRACE_INDEX_WIDTH'(trace_local_index);
        o_target_trace_index = TRACE_INDEX_WIDTH'(trace_local_index);
        o_student_trace_q8_8 = i_trace_q8_8;
        o_target_trace_q8_8 = i_trace_q8_8;
        o_student_surrogate_sample = i_surrogate_sample;
        o_target_surrogate_sample = i_surrogate_sample;
        o_student_surrogate_index = SURROGATE_INDEX_WIDTH'(surrogate_local_index);
        o_target_surrogate_index = SURROGATE_INDEX_WIDTH'(surrogate_local_index);
        o_student_surrogate_q8_8 = i_surrogate_q8_8;
        o_target_surrogate_q8_8 = i_surrogate_q8_8;

        cfg_legal = i_cfg_batch_count != 0 &&
            int'(i_cfg_batch_count) <= MAX_BATCH &&
            i_cfg_input_channel_count != 0 &&
            int'(i_cfg_input_channel_count) <= MAX_INPUT_CHANNELS &&
            i_cfg_output_channel_count != 0 &&
            int'(i_cfg_output_channel_count) <= MAX_OUTPUT_CHANNELS &&
            int'(i_cfg_input_channel_base) + int'(i_cfg_input_channel_count) <= 512 &&
            int'(i_cfg_output_channel_base) + int'(i_cfg_output_channel_count) <= 512;
    end

    assign o_cfg_ready = i_rst_n && !active_q;
    assign o_active = active_q;
    assign o_capture_complete = active_q &&
        int'(student_input_count_q) == input_beats &&
        int'(target_input_count_q) == input_beats &&
        int'(student_trace_count_q) == trace_beats &&
        int'(target_trace_count_q) == trace_beats &&
        int'(student_surrogate_count_q) == surrogate_beats &&
        int'(target_surrogate_count_q) == surrogate_beats;
    assign o_release_ready = o_capture_complete;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_q <= 1'b0;
            batch_count_q <= '0;
            timestep_q <= '0;
            sample_version_q <= '0;
            input_base_q <= '0;
            input_count_q <= '0;
            output_base_q <= '0;
            output_count_q <= '0;
            student_input_count_q <= '0;
            target_input_count_q <= '0;
            student_trace_count_q <= '0;
            target_trace_count_q <= '0;
            student_surrogate_count_q <= '0;
            target_surrogate_count_q <= '0;
            for (int branch = 0; branch < 2; branch++) begin
                input_expected_sample_q[branch] <= '0;
                input_expected_channel_q[branch] <= '0;
                input_expected_spatial_q[branch] <= '0;
                trace_expected_sample_q[branch] <= '0;
                trace_expected_channel_q[branch] <= '0;
                trace_expected_spatial_q[branch] <= '0;
                surrogate_expected_sample_q[branch] <= '0;
                surrogate_expected_channel_q[branch] <= '0;
                surrogate_expected_spatial_q[branch] <= '0;
            end
            o_protocol_error <= 1'b0;
        end else begin
            if (i_cfg_valid && o_cfg_ready) begin
                if (cfg_legal) begin
                    active_q <= 1'b1;
                    batch_count_q <= i_cfg_batch_count;
                    timestep_q <= i_cfg_timestep;
                    sample_version_q <= i_cfg_sample_version;
                    input_base_q <= i_cfg_input_channel_base;
                    input_count_q <= i_cfg_input_channel_count;
                    output_base_q <= i_cfg_output_channel_base;
                    output_count_q <= i_cfg_output_channel_count;
                end else begin
                    active_q <= 1'b0;
                end
                student_input_count_q <= '0;
                target_input_count_q <= '0;
                student_trace_count_q <= '0;
                target_trace_count_q <= '0;
                student_surrogate_count_q <= '0;
                target_surrogate_count_q <= '0;
                for (int branch = 0; branch < 2; branch++) begin
                    input_expected_sample_q[branch] <= '0;
                    input_expected_channel_q[branch] <= '0;
                    input_expected_spatial_q[branch] <= '0;
                    trace_expected_sample_q[branch] <= '0;
                    trace_expected_channel_q[branch] <= '0;
                    trace_expected_spatial_q[branch] <= '0;
                    surrogate_expected_sample_q[branch] <= '0;
                    surrogate_expected_channel_q[branch] <= '0;
                    surrogate_expected_spatial_q[branch] <= '0;
                end
                o_protocol_error <= !cfg_legal;
            end else if (i_release && o_release_ready) begin
                active_q <= 1'b0;
            end else begin
                if (i_input_valid && o_input_ready && input_selected) begin
                    if (!input_forward)
                        o_protocol_error <= 1'b1;
                    else begin
                        if (i_input_target)
                            target_input_count_q <= target_input_count_q + 1'b1;
                        else
                            student_input_count_q <= student_input_count_q + 1'b1;
                        for (int branch = 0; branch < 2; branch++) begin
                            if (i_input_target == 1'(branch)) begin
                                if (int'(input_expected_spatial_q[branch]) + 1 ==
                                    INPUT_PLANE) begin
                                    input_expected_spatial_q[branch] <= '0;
                                    if (int'(input_expected_channel_q[branch]) + 1 ==
                                        int'(input_count_q)) begin
                                        input_expected_channel_q[branch] <= '0;
                                        input_expected_sample_q[branch] <=
                                            input_expected_sample_q[branch] + 1'b1;
                                    end else
                                        input_expected_channel_q[branch] <=
                                            input_expected_channel_q[branch] + 1'b1;
                                end else
                                    input_expected_spatial_q[branch] <=
                                        input_expected_spatial_q[branch] + 1'b1;
                            end
                        end
                    end
                end
                if (i_trace_valid && o_trace_ready && trace_selected) begin
                    if (!trace_forward)
                        o_protocol_error <= 1'b1;
                    else begin
                        if (i_trace_target)
                            target_trace_count_q <= target_trace_count_q + 1'b1;
                        else
                            student_trace_count_q <= student_trace_count_q + 1'b1;
                        for (int branch = 0; branch < 2; branch++) begin
                            if (i_trace_target == 1'(branch)) begin
                                if (int'(trace_expected_spatial_q[branch]) + 1 ==
                                    TRACE_PLANE) begin
                                    trace_expected_spatial_q[branch] <= '0;
                                    if (int'(trace_expected_channel_q[branch]) + 1 ==
                                        int'(output_count_q)) begin
                                        trace_expected_channel_q[branch] <= '0;
                                        trace_expected_sample_q[branch] <=
                                            trace_expected_sample_q[branch] + 1'b1;
                                    end else
                                        trace_expected_channel_q[branch] <=
                                            trace_expected_channel_q[branch] + 1'b1;
                                end else
                                    trace_expected_spatial_q[branch] <=
                                        trace_expected_spatial_q[branch] + 1'b1;
                            end
                        end
                    end
                end
                if (i_surrogate_valid && o_surrogate_ready && surrogate_selected) begin
                    if (!surrogate_forward)
                        o_protocol_error <= 1'b1;
                    else begin
                        if (i_surrogate_target)
                            target_surrogate_count_q <= target_surrogate_count_q + 1'b1;
                        else
                            student_surrogate_count_q <= student_surrogate_count_q + 1'b1;
                        for (int branch = 0; branch < 2; branch++) begin
                            if (i_surrogate_target == 1'(branch)) begin
                                if (int'(surrogate_expected_spatial_q[branch]) + 1 ==
                                    INPUT_PLANE) begin
                                    surrogate_expected_spatial_q[branch] <= '0;
                                    if (int'(surrogate_expected_channel_q[branch]) + 1 ==
                                        int'(output_count_q)) begin
                                        surrogate_expected_channel_q[branch] <= '0;
                                        surrogate_expected_sample_q[branch] <=
                                            surrogate_expected_sample_q[branch] + 1'b1;
                                    end else
                                        surrogate_expected_channel_q[branch] <=
                                            surrogate_expected_channel_q[branch] + 1'b1;
                                end else
                                    surrogate_expected_spatial_q[branch] <=
                                        surrogate_expected_spatial_q[branch] + 1'b1;
                            end
                        end
                    end
                end
            end
        end
    end

    initial begin
        if (SPATIAL != 4 || POOLED_SPATIAL != 2)
            $error("Conv8 observation adapter requires the 4x4/AAVG2 contract");
    end
endmodule

`default_nettype wire
