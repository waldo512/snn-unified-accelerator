`timescale 1ns/1ps
`default_nettype none

// Shared scalar microsequencer for the trainable SHD M2 head.  One streamed
// Q2.14 parameter working set serves both actual and target paths; recurrent
// membrane/trace and actual-path logits are retained by sequence-version tags.
module unified_srnn_m2_numeric_forward_observation_consumer #(
    parameter int BATCH_MAX = 64,
    parameter int TIMESTEP_MAX = 128,
    parameter int BACKBONE_MAX = 450,
    parameter int HIDDEN_MAX = 256,
    parameter int CLASS_MAX = 20,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int SAMPLE_ID_WIDTH = 8,
    parameter int TIMESTEP_ID_WIDTH = $clog2(TIMESTEP_MAX),
    parameter int BACKBONE_INDEX_WIDTH = 10,
    parameter int HIDDEN_INDEX_WIDTH = 9,
    parameter int CLASS_INDEX_WIDTH = 5,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int VERSION_WIDTH = 16,
    parameter logic signed [31:0] LEAK_Q14 = 32'sd13926,
    parameter logic signed [31:0] THRESHOLD_Q14 = 32'sd8192
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wh1_tensor_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_wout_tensor_id,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_cfg_batch_count,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_cfg_backbone_count,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]             i_cfg_hidden_count,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_cfg_class_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_parameter_valid,
    output logic                                      o_parameter_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_parameter_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_parameter_local_addr,
    input  logic signed [15:0]                        i_parameter_code_q14,
    input  logic                                      i_parameter_last,
    output logic                                      o_parameter_scan_done,

    input  logic                                      i_timestep_start,
    output logic                                      o_timestep_start_ready,
    input  logic [TIMESTEP_ID_WIDTH-1:0]              i_timestep_id,
    input  logic [VERSION_WIDTH-1:0]                  i_sequence_version,
    input  logic                                      i_include_target_path,
    input  logic                                      i_abort_timestep,

    input  logic                                      i_observation_valid,
    output logic                                      o_observation_ready,
    input  logic                                      i_observation_path,
    input  logic [SAMPLE_ID_WIDTH-1:0]                i_observation_sample_id,
    input  logic [BACKBONE_INDEX_WIDTH-1:0]           i_observation_backbone_id,
    input  logic signed [31:0]                        i_observation_q14,
    input  logic                                      i_observation_last,
    input  logic                                      i_observation_done,

    output logic                                      o_hidden_valid,
    input  logic                                      i_hidden_ready,
    output logic                                      o_hidden_path,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_hidden_sample_id,
    output logic [HIDDEN_INDEX_WIDTH-1:0]             o_hidden_id,
    output logic signed [31:0]                        o_hidden_membrane_pre_q14,
    output logic signed [31:0]                        o_hidden_membrane_next_q14,
    output logic                                      o_hidden_spike,
    output logic signed [31:0]                        o_hidden_trace_q14,
    output logic signed [31:0]                        o_hidden_output_feature_q14,
    output logic                                      o_hidden_last,

    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic [SAMPLE_ID_WIDTH-1:0]                o_logit_sample_id,
    output logic [CLASS_INDEX_WIDTH-1:0]              o_logit_class_id,
    output logic signed [63:0]                        o_logit_q28,
    output logic                                      o_logit_last,

    output logic [TIMESTEP_ID_WIDTH-1:0]              o_timestep_id,
    output logic                                      o_timestep_done,
    output logic                                      o_busy,
    output logic                                      o_protocol_error,
    output logic                                      o_overflow
);

    localparam int WH1_CAPACITY = HIDDEN_MAX * BACKBONE_MAX;
    localparam int WOUT_CAPACITY = CLASS_MAX * HIDDEN_MAX;
    localparam int HIDDEN_STATE_CAPACITY = 2 * BATCH_MAX * HIDDEN_MAX;
    localparam int FEATURE_CAPACITY = BATCH_MAX * HIDDEN_MAX;
    localparam int LOGIT_CAPACITY = BATCH_MAX * CLASS_MAX;
    localparam int HIDDEN_STATE_WIDTH = VERSION_WIDTH + 64;
    localparam int FEATURE_STATE_WIDTH = VERSION_WIDTH + 32;
    localparam int LOGIT_STATE_WIDTH = VERSION_WIDTH + 64;
    localparam int WH1_ADDR_WIDTH = $clog2(WH1_CAPACITY);
    localparam int WOUT_ADDR_WIDTH = $clog2(WOUT_CAPACITY);
    localparam int HIDDEN_STATE_ADDR_WIDTH = $clog2(HIDDEN_STATE_CAPACITY);
    localparam int FEATURE_ADDR_WIDTH = $clog2(FEATURE_CAPACITY);
    localparam int LOGIT_ADDR_WIDTH = $clog2(LOGIT_CAPACITY);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_OBSERVATION_WAIT,
        STATE_WH1_READ,
        STATE_WH1_ACCUMULATE,
        STATE_HIDDEN_STATE_READ,
        STATE_HIDDEN_EMIT,
        STATE_OBSERVATION_DONE_WAIT,
        STATE_WOUT_READ,
        STATE_WOUT_ACCUMULATE,
        STATE_LOGIT_STATE_READ,
        STATE_LOGIT_EMIT,
        STATE_ERROR
    } state_t;

    (* ram_style = "block" *) logic [HIDDEN_STATE_WIDTH-1:0]
        hidden_state_memory [0:HIDDEN_STATE_CAPACITY-1];
    (* ram_style = "block" *) logic [FEATURE_STATE_WIDTH-1:0]
        output_feature_memory [0:FEATURE_CAPACITY-1];
    (* ram_style = "block" *) logic [LOGIT_STATE_WIDTH-1:0]
        output_logit_memory [0:LOGIT_CAPACITY-1];
    (* ram_style = "distributed" *) logic signed [63:0]
        vector_accumulator_memory [0:HIDDEN_MAX-1];
    (* ram_style = "distributed" *) logic
        actual_spike_memory [0:FEATURE_CAPACITY-1];

    state_t state_q;
    logic configured_q;
    logic parameter_set_valid_q;
    logic parameter_scan_active_q;
    logic parameter_phase_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_expected_local_q;
    logic [TENSOR_ID_WIDTH-1:0] wh1_tensor_id_q;
    logic [TENSOR_ID_WIDTH-1:0] wout_tensor_id_q;
    logic [SAMPLE_ID_WIDTH-1:0] batch_count_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] backbone_count_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_count_q;
    logic [CLASS_INDEX_WIDTH-1:0] class_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] wh1_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] wout_count_q;

    logic [TIMESTEP_ID_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sequence_version_q;
    logic include_target_q;
    logic expected_path_q;
    logic [SAMPLE_ID_WIDTH-1:0] expected_sample_q;
    logic [BACKBONE_INDEX_WIDTH-1:0] expected_backbone_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_q;
    logic [CLASS_INDEX_WIDTH-1:0] class_q;
    logic signed [31:0] observation_q14_q;
    logic observation_done_q;

    logic signed [15:0] wh1_weight_read_data;
    logic signed [15:0] wout_weight_read_data;
    logic signed [15:0] weight_read_value;
    logic signed [63:0] accumulator_read_q;
    logic [HIDDEN_STATE_WIDTH-1:0] hidden_state_read_q;
    logic [FEATURE_STATE_WIDTH-1:0] feature_state_read_q;
    logic [LOGIT_STATE_WIDTH-1:0] logit_state_read_q;
    logic actual_spike_read_q;
    logic signed [63:0] wout_accumulator_q;

    logic cfg_legal;
    logic parameter_fire;
    logic parameter_metadata_legal;
    logic wh1_write_enable;
    logic wout_write_enable;
    logic timestep_start_fire;
    logic observation_fire;
    logic hidden_fire;
    logic logit_fire;
    logic observation_metadata_legal;
    logic expected_observation_last;
    logic final_backbone;
    logic final_hidden;
    logic final_class;
    logic final_sample;

    logic [WH1_ADDR_WIDTH-1:0] wh1_read_address;
    logic [WOUT_ADDR_WIDTH-1:0] wout_read_address;
    logic [HIDDEN_STATE_ADDR_WIDTH-1:0] hidden_state_address;
    logic [FEATURE_ADDR_WIDTH-1:0] feature_address;
    logic [LOGIT_ADDR_WIDTH-1:0] logit_address;

    logic signed [63:0] wh1_product_wide;
    logic signed [63:0] wh1_total_q28;
    logic signed [63:0] current_q14_wide;
    logic signed [63:0] previous_membrane_wide;
    logic signed [63:0] previous_trace_wide;
    logic signed [63:0] previous_feature_wide;
    logic signed [63:0] membrane_leak_wide;
    logic signed [63:0] trace_leak_wide;
    logic signed [63:0] membrane_pre_wide;
    logic signed [63:0] membrane_next_wide;
    logic signed [63:0] trace_next_wide;
    logic signed [63:0] feature_next_wide;
    logic signed [63:0] threshold_wide;
    logic signed [31:0] membrane_pre_clamped;
    logic signed [31:0] membrane_next_clamped;
    logic signed [31:0] trace_next_clamped;
    logic signed [31:0] feature_next_clamped;
    logic hidden_spike_calculated;
    logic hidden_overflow_calculated;

    logic signed [63:0] wout_product_wide;
    logic signed [63:0] wout_total_q28;
    logic signed [63:0] previous_logit;
    logic signed [127:0] logit_sum_wide;
    logic signed [63:0] logit_next;
    logic logit_overflow_calculated;

    unified_srnn_m2_parameter_ram_q14 #(
        .DEPTH(WH1_CAPACITY),
        .ADDR_WIDTH(WH1_ADDR_WIDTH)
    ) u_wh1_weight_memory (
        .i_clk(i_clk),
        .i_write_enable(wh1_write_enable),
        .i_write_addr(i_parameter_local_addr[WH1_ADDR_WIDTH-1:0]),
        .i_write_data(i_parameter_code_q14),
        .i_read_enable(state_q == STATE_WH1_READ),
        .i_read_addr(wh1_read_address),
        .o_read_data(wh1_weight_read_data)
    );

    unified_srnn_m2_parameter_ram_q14 #(
        .DEPTH(WOUT_CAPACITY),
        .ADDR_WIDTH(WOUT_ADDR_WIDTH)
    ) u_wout_weight_memory (
        .i_clk(i_clk),
        .i_write_enable(wout_write_enable),
        .i_write_addr(i_parameter_local_addr[WOUT_ADDR_WIDTH-1:0]),
        .i_write_data(i_parameter_code_q14),
        .i_read_enable(state_q == STATE_WOUT_READ),
        .i_read_addr(wout_read_address),
        .o_read_data(wout_weight_read_data)
    );

    integer init_index;
    initial begin
        for (init_index = 0; init_index < HIDDEN_STATE_CAPACITY;
             init_index = init_index + 1)
            hidden_state_memory[init_index] = '0;
        for (init_index = 0; init_index < FEATURE_CAPACITY;
             init_index = init_index + 1)
            output_feature_memory[init_index] = '0;
        for (init_index = 0; init_index < LOGIT_CAPACITY;
             init_index = init_index + 1)
            output_logit_memory[init_index] = '0;
    end

    function automatic logic signed [63:0] round_shift_even_128(
        input logic signed [127:0] value,
        input integer shift
    );
        logic negative;
        logic [127:0] magnitude;
        logic [127:0] quotient;
        logic [127:0] remainder;
        logic [127:0] half;
        logic [127:0] mask;
        begin
            negative = value[127];
            magnitude = negative ? $unsigned(-value) : $unsigned(value);
            mask = (128'd1 << shift) - 1'b1;
            half = 128'd1 << (shift - 1);
            quotient = magnitude >> shift;
            remainder = magnitude & mask;
            if ((remainder > half) ||
                ((remainder == half) && quotient[0]))
                quotient = quotient + 1'b1;
            round_shift_even_128 = negative ?
                -$signed(quotient[63:0]) : $signed(quotient[63:0]);
        end
    endfunction

    function automatic logic signed [31:0] clamp_s32(
        input logic signed [63:0] value
    );
        begin
            if (value > 64'sd2147483647)
                clamp_s32 = 32'sh7fffffff;
            else if (value < -64'sd2147483648)
                clamp_s32 = 32'sh80000000;
            else
                clamp_s32 = value[31:0];
        end
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_wh1_tensor_id != i_cfg_wout_tensor_id) &&
            (i_cfg_batch_count != '0) &&
            (int'(i_cfg_batch_count) <= BATCH_MAX) &&
            (i_cfg_backbone_count != '0) &&
            (int'(i_cfg_backbone_count) <= BACKBONE_MAX) &&
            (i_cfg_hidden_count != '0) &&
            (int'(i_cfg_hidden_count) <= HIDDEN_MAX) &&
            (i_cfg_class_count != '0) &&
            (int'(i_cfg_class_count) <= CLASS_MAX);

        o_cfg_ready = state_q == STATE_IDLE && !parameter_scan_active_q;
        o_parameter_ready = state_q == STATE_IDLE && configured_q &&
            !i_timestep_start;
        parameter_fire = i_parameter_valid && o_parameter_ready;
        parameter_metadata_legal =
            ((!parameter_scan_active_q || !parameter_phase_q) &&
                (i_parameter_tensor_id == wh1_tensor_id_q) &&
                (i_parameter_local_addr ==
                    (parameter_scan_active_q ?
                        parameter_expected_local_q : '0)) &&
                !i_parameter_last) ||
            (parameter_scan_active_q && parameter_phase_q &&
                (i_parameter_tensor_id == wout_tensor_id_q) &&
                (i_parameter_local_addr == parameter_expected_local_q) &&
                (i_parameter_last ==
                    (parameter_expected_local_q ==
                        (wout_count_q - 1'b1))));
        wh1_write_enable = parameter_fire && parameter_metadata_legal &&
            (!parameter_scan_active_q || !parameter_phase_q);
        wout_write_enable = parameter_fire && parameter_metadata_legal &&
            parameter_scan_active_q && parameter_phase_q;
        o_timestep_start_ready = state_q == STATE_IDLE && configured_q &&
            parameter_set_valid_q && !parameter_scan_active_q &&
            !i_parameter_valid && (i_sequence_version != '0);
        timestep_start_fire = i_timestep_start && o_timestep_start_ready;

        o_observation_ready = state_q == STATE_OBSERVATION_WAIT &&
            !o_protocol_error;
        observation_fire = i_observation_valid && o_observation_ready;
        final_backbone = expected_backbone_q ==
            (backbone_count_q - 1'b1);
        final_hidden = hidden_q == (hidden_count_q - 1'b1);
        final_class = class_q == (class_count_q - 1'b1);
        final_sample = expected_sample_q == (batch_count_q - 1'b1);
        expected_observation_last = (expected_path_q || !include_target_q) &&
            final_sample && final_backbone;
        observation_metadata_legal =
            (i_observation_path == expected_path_q) &&
            (i_observation_sample_id == expected_sample_q) &&
            (i_observation_backbone_id == expected_backbone_q) &&
            (i_observation_last == expected_observation_last);

        wh1_read_address = WH1_ADDR_WIDTH'(
            int'(hidden_q) * int'(backbone_count_q) +
            int'(expected_backbone_q));
        feature_address = FEATURE_ADDR_WIDTH'(
            int'(expected_sample_q) * HIDDEN_MAX + int'(hidden_q));
        hidden_state_address = HIDDEN_STATE_ADDR_WIDTH'(
            (expected_path_q ? BATCH_MAX * HIDDEN_MAX : 0) +
            int'(feature_address));
        wout_read_address = WOUT_ADDR_WIDTH'(
            int'(class_q) * int'(hidden_count_q) + int'(hidden_q));
        logit_address = LOGIT_ADDR_WIDTH'(
            int'(expected_sample_q) * CLASS_MAX + int'(class_q));

        weight_read_value = state_q == STATE_WOUT_ACCUMULATE ?
            wout_weight_read_data : wh1_weight_read_data;
        wh1_product_wide = $signed(observation_q14_q) *
            $signed(weight_read_value);
        wh1_total_q28 = (expected_backbone_q == '0) ?
            wh1_product_wide : accumulator_read_q + wh1_product_wide;

        current_q14_wide = round_shift_even_128(
            {{64{accumulator_read_q[63]}}, accumulator_read_q}, 14);
        previous_membrane_wide =
            hidden_state_read_q[HIDDEN_STATE_WIDTH-1:64] ==
                sequence_version_q ?
            {{32{hidden_state_read_q[63]}}, hidden_state_read_q[63:32]} :
            64'sd0;
        previous_trace_wide =
            hidden_state_read_q[HIDDEN_STATE_WIDTH-1:64] ==
                sequence_version_q ?
            {{32{hidden_state_read_q[31]}}, hidden_state_read_q[31:0]} :
            64'sd0;
        previous_feature_wide =
            feature_state_read_q[FEATURE_STATE_WIDTH-1:32] ==
                sequence_version_q ?
            {{32{feature_state_read_q[31]}}, feature_state_read_q[31:0]} :
            64'sd0;
        membrane_leak_wide = round_shift_even_128(
            $signed(previous_membrane_wide) * $signed(LEAK_Q14), 14);
        trace_leak_wide = round_shift_even_128(
            $signed(previous_trace_wide) * $signed(LEAK_Q14), 14);
        membrane_pre_wide = membrane_leak_wide + current_q14_wide;
        threshold_wide = $signed(
            {{32{THRESHOLD_Q14[31]}}, THRESHOLD_Q14});
        hidden_spike_calculated = membrane_pre_wide >= threshold_wide;
        membrane_next_wide = membrane_pre_wide -
            (hidden_spike_calculated ? threshold_wide : 64'sd0);
        trace_next_wide = trace_leak_wide +
            (hidden_spike_calculated ? 64'sd16384 : 64'sd0);
        feature_next_wide = previous_feature_wide +
            ((!expected_path_q && hidden_spike_calculated) ?
                64'sd16384 : 64'sd0);
        membrane_pre_clamped = clamp_s32(membrane_pre_wide);
        membrane_next_clamped = clamp_s32(membrane_next_wide);
        trace_next_clamped = clamp_s32(trace_next_wide);
        feature_next_clamped = clamp_s32(feature_next_wide);
        hidden_overflow_calculated =
            ($signed(membrane_pre_wide) > 64'sd2147483647) ||
            ($signed(membrane_pre_wide) < -64'sd2147483648) ||
            ($signed(membrane_next_wide) > 64'sd2147483647) ||
            ($signed(membrane_next_wide) < -64'sd2147483648) ||
            ($signed(trace_next_wide) > 64'sd2147483647) ||
            ($signed(trace_next_wide) < -64'sd2147483648) ||
            ($signed(feature_next_wide) > 64'sd2147483647) ||
            ($signed(feature_next_wide) < -64'sd2147483648);

        o_hidden_valid = state_q == STATE_HIDDEN_EMIT;
        o_hidden_path = expected_path_q;
        o_hidden_sample_id = expected_sample_q;
        o_hidden_id = hidden_q;
        o_hidden_membrane_pre_q14 = membrane_pre_clamped;
        o_hidden_membrane_next_q14 = membrane_next_clamped;
        o_hidden_spike = hidden_spike_calculated;
        o_hidden_trace_q14 = trace_next_clamped;
        o_hidden_output_feature_q14 = expected_path_q ? 32'sd0 :
            feature_next_clamped;
        o_hidden_last = o_hidden_valid &&
            (expected_path_q || !include_target_q) && final_sample &&
            final_hidden;
        hidden_fire = o_hidden_valid && i_hidden_ready;

        wout_product_wide = (actual_spike_read_q ? 64'sd16384 :
            64'sd0) * $signed(weight_read_value);
        wout_total_q28 = wout_accumulator_q + wout_product_wide;
        previous_logit =
            logit_state_read_q[LOGIT_STATE_WIDTH-1:64] ==
                sequence_version_q ?
            $signed(logit_state_read_q[63:0]) : 64'sd0;
        logit_sum_wide =
            {{64{previous_logit[63]}}, previous_logit} +
            {{64{wout_accumulator_q[63]}}, wout_accumulator_q};
        logit_next = logit_sum_wide[63:0];
        logit_overflow_calculated =
            logit_sum_wide[127:64] != {64{logit_sum_wide[63]}};

        o_logit_valid = state_q == STATE_LOGIT_EMIT;
        o_logit_sample_id = expected_sample_q;
        o_logit_class_id = class_q;
        o_logit_q28 = logit_next;
        o_logit_last = o_logit_valid && final_sample && final_class;
        logit_fire = o_logit_valid && i_logit_ready;

        o_timestep_id = timestep_q;
        o_busy = state_q != STATE_IDLE;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            configured_q <= 1'b0;
            parameter_set_valid_q <= 1'b0;
            parameter_scan_active_q <= 1'b0;
            parameter_phase_q <= 1'b0;
            parameter_expected_local_q <= '0;
            wh1_tensor_id_q <= '0;
            wout_tensor_id_q <= '0;
            batch_count_q <= '0;
            backbone_count_q <= '0;
            hidden_count_q <= '0;
            class_count_q <= '0;
            wh1_count_q <= '0;
            wout_count_q <= '0;
            timestep_q <= '0;
            sequence_version_q <= '0;
            include_target_q <= 1'b0;
            expected_path_q <= 1'b0;
            expected_sample_q <= '0;
            expected_backbone_q <= '0;
            hidden_q <= '0;
            class_q <= '0;
            observation_q14_q <= '0;
            observation_done_q <= 1'b0;
            accumulator_read_q <= '0;
            hidden_state_read_q <= '0;
            feature_state_read_q <= '0;
            logit_state_read_q <= '0;
            actual_spike_read_q <= 1'b0;
            wout_accumulator_q <= '0;
            o_cfg_error <= 1'b0;
            o_parameter_scan_done <= 1'b0;
            o_timestep_done <= 1'b0;
            o_protocol_error <= 1'b0;
            o_overflow <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_parameter_scan_done <= 1'b0;
            o_timestep_done <= 1'b0;

            if (i_clear_error) begin
                o_protocol_error <= 1'b0;
                o_overflow <= 1'b0;
                if (state_q == STATE_ERROR)
                    state_q <= STATE_IDLE;
            end

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    parameter_set_valid_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    parameter_set_valid_q <= 1'b0;
                    parameter_scan_active_q <= 1'b0;
                    parameter_phase_q <= 1'b0;
                    parameter_expected_local_q <= '0;
                    wh1_tensor_id_q <= i_cfg_wh1_tensor_id;
                    wout_tensor_id_q <= i_cfg_wout_tensor_id;
                    batch_count_q <= i_cfg_batch_count;
                    backbone_count_q <= i_cfg_backbone_count;
                    hidden_count_q <= i_cfg_hidden_count;
                    class_count_q <= i_cfg_class_count;
                    wh1_count_q <= PARAM_ADDR_WIDTH'(
                        int'(i_cfg_hidden_count) *
                        int'(i_cfg_backbone_count));
                    wout_count_q <= PARAM_ADDR_WIDTH'(
                        int'(i_cfg_class_count) *
                        int'(i_cfg_hidden_count));
                    o_protocol_error <= 1'b0;
                    o_overflow <= 1'b0;
                end
            end

            if (parameter_fire) begin
                if (!parameter_scan_active_q) begin
                    parameter_scan_active_q <= 1'b1;
                    parameter_phase_q <= 1'b0;
                    parameter_expected_local_q <= '0;
                    parameter_set_valid_q <= 1'b0;
                end

                if (!parameter_metadata_legal) begin
                    o_protocol_error <= 1'b1;
                    parameter_scan_active_q <= 1'b0;
                    parameter_set_valid_q <= 1'b0;
                    state_q <= STATE_ERROR;
                end else if (!parameter_scan_active_q || !parameter_phase_q) begin
                    if (i_parameter_local_addr == (wh1_count_q - 1'b1)) begin
                        parameter_phase_q <= 1'b1;
                        parameter_expected_local_q <= '0;
                    end else begin
                        parameter_expected_local_q <=
                            i_parameter_local_addr + 1'b1;
                    end
                end else begin
                    if (i_parameter_local_addr == (wout_count_q - 1'b1)) begin
                        parameter_scan_active_q <= 1'b0;
                        parameter_phase_q <= 1'b0;
                        parameter_expected_local_q <= '0;
                        parameter_set_valid_q <= 1'b1;
                        o_parameter_scan_done <= 1'b1;
                    end else begin
                        parameter_expected_local_q <=
                            i_parameter_local_addr + 1'b1;
                    end
                end
            end

            if (i_abort_timestep && (state_q != STATE_IDLE)) begin
                state_q <= STATE_IDLE;
                observation_done_q <= 1'b0;
                expected_path_q <= 1'b0;
                expected_sample_q <= '0;
                expected_backbone_q <= '0;
                hidden_q <= '0;
                class_q <= '0;
                wout_accumulator_q <= '0;
            end else begin
                if (i_observation_done && (state_q != STATE_IDLE) &&
                    (state_q != STATE_ERROR))
                    observation_done_q <= 1'b1;

            if (timestep_start_fire) begin
                timestep_q <= i_timestep_id;
                sequence_version_q <= i_sequence_version;
                include_target_q <= i_include_target_path;
                expected_path_q <= 1'b0;
                expected_sample_q <= '0;
                expected_backbone_q <= '0;
                hidden_q <= '0;
                class_q <= '0;
                observation_done_q <= 1'b0;
                wout_accumulator_q <= '0;
                o_overflow <= 1'b0;
                state_q <= STATE_OBSERVATION_WAIT;
            end else if (i_timestep_start && (state_q == STATE_IDLE) &&
                (i_sequence_version == '0)) begin
                o_protocol_error <= 1'b1;
                state_q <= STATE_ERROR;
            end

                unique case (state_q)
                STATE_IDLE: begin
                end

                STATE_OBSERVATION_WAIT: begin
                    if (observation_fire) begin
                        if (!observation_metadata_legal) begin
                            o_protocol_error <= 1'b1;
                            state_q <= STATE_ERROR;
                        end else begin
                            observation_q14_q <= i_observation_q14;
                            hidden_q <= '0;
                            state_q <= STATE_WH1_READ;
                        end
                    end
                end

                STATE_WH1_READ: begin
                    accumulator_read_q <=
                        vector_accumulator_memory[int'(hidden_q)];
                    state_q <= STATE_WH1_ACCUMULATE;
                end

                STATE_WH1_ACCUMULATE: begin
                    vector_accumulator_memory[int'(hidden_q)] <=
                        wh1_total_q28;
                    if (final_hidden) begin
                        hidden_q <= '0;
                        if (final_backbone)
                            state_q <= STATE_HIDDEN_STATE_READ;
                        else begin
                            expected_backbone_q <= expected_backbone_q + 1'b1;
                            state_q <= STATE_OBSERVATION_WAIT;
                        end
                    end else begin
                        hidden_q <= hidden_q + 1'b1;
                        state_q <= STATE_WH1_READ;
                    end
                end

                STATE_HIDDEN_STATE_READ: begin
                    accumulator_read_q <=
                        vector_accumulator_memory[int'(hidden_q)];
                    hidden_state_read_q <=
                        hidden_state_memory[hidden_state_address];
                    feature_state_read_q <=
                        output_feature_memory[feature_address];
                    state_q <= STATE_HIDDEN_EMIT;
                end

                STATE_HIDDEN_EMIT: begin
                    if (hidden_fire) begin
                        hidden_state_memory[hidden_state_address] <= {
                            sequence_version_q,
                            membrane_next_clamped,
                            trace_next_clamped
                        };
                        if (!expected_path_q) begin
                            output_feature_memory[feature_address] <= {
                                sequence_version_q,
                                feature_next_clamped
                            };
                            actual_spike_memory[feature_address] <=
                                hidden_spike_calculated;
                        end
                        if (hidden_overflow_calculated)
                            o_overflow <= 1'b1;

                        if (final_hidden) begin
                            hidden_q <= '0;
                            expected_backbone_q <= '0;
                            if ((expected_path_q || !include_target_q) &&
                                final_sample) begin
                                state_q <= STATE_OBSERVATION_DONE_WAIT;
                            end else begin
                                if (final_sample) begin
                                    expected_path_q <= 1'b1;
                                    expected_sample_q <= '0;
                                end else begin
                                    expected_sample_q <=
                                        expected_sample_q + 1'b1;
                                end
                                state_q <= STATE_OBSERVATION_WAIT;
                            end
                        end else begin
                            hidden_q <= hidden_q + 1'b1;
                            state_q <= STATE_HIDDEN_STATE_READ;
                        end
                    end
                end

                STATE_OBSERVATION_DONE_WAIT: begin
                    if (observation_done_q || i_observation_done) begin
                        expected_path_q <= 1'b0;
                        expected_sample_q <= '0;
                        hidden_q <= '0;
                        class_q <= '0;
                        wout_accumulator_q <= '0;
                        state_q <= STATE_WOUT_READ;
                    end
                end

                STATE_WOUT_READ: begin
                    actual_spike_read_q <=
                        actual_spike_memory[feature_address];
                    state_q <= STATE_WOUT_ACCUMULATE;
                end

                STATE_WOUT_ACCUMULATE: begin
                    wout_accumulator_q <= wout_total_q28;
                    if (final_hidden) begin
                        state_q <= STATE_LOGIT_STATE_READ;
                    end else begin
                        hidden_q <= hidden_q + 1'b1;
                        state_q <= STATE_WOUT_READ;
                    end
                end

                STATE_LOGIT_STATE_READ: begin
                    logit_state_read_q <= output_logit_memory[logit_address];
                    state_q <= STATE_LOGIT_EMIT;
                end

                STATE_LOGIT_EMIT: begin
                    if (logit_fire) begin
                        output_logit_memory[logit_address] <= {
                            sequence_version_q,
                            logit_next
                        };
                        if (logit_overflow_calculated)
                            o_overflow <= 1'b1;
                        hidden_q <= '0;
                        wout_accumulator_q <= '0;
                        if (final_class) begin
                            class_q <= '0;
                            if (final_sample) begin
                                expected_sample_q <= '0;
                                o_timestep_done <= 1'b1;
                                state_q <= STATE_IDLE;
                            end else begin
                                expected_sample_q <= expected_sample_q + 1'b1;
                                state_q <= STATE_WOUT_READ;
                            end
                        end else begin
                            class_q <= class_q + 1'b1;
                            state_q <= STATE_WOUT_READ;
                        end
                    end
                end

                STATE_ERROR: begin
                end

                default: begin
                    o_protocol_error <= 1'b1;
                    state_q <= STATE_ERROR;
                end
                endcase
            end
        end
    end

endmodule

`default_nettype wire
