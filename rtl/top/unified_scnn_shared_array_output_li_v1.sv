`default_nettype none

// SCNN OUTPUT/LI backend for raw Q8.8 partials produced by the one shared
// numeric array. This block performs no feature/weight multiplication. It
// only retains cross-input-group sums, cross-frame LI state, and frame argmax.
module unified_scnn_shared_array_output_li_v1 #(
    parameter int ENGINES = 12,
    parameter int OUTPUT_MAX = 32,
    parameter int ACC_WIDTH = 48,
    parameter int LI_STATE_WIDTH = 44,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int CLASS_WIDTH = OUTPUT_MAX <= 1 ? 1 : $clog2(OUTPUT_MAX),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(OUTPUT_MAX + 1)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_sample_reset,

    input  logic                                      i_raw_valid,
    output logic                                      o_raw_ready,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [CLASS_WIDTH-1:0]                    i_output_base,
    input  logic [OUTPUT_COUNT_WIDTH-1:0]             i_output_count,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_raw_partial_flat,
    input  logic                                      i_first_input_group,
    input  logic                                      i_final_input_group,
    input  logic                                      i_frame_last,
    input  logic                                      i_sample_last,
    input  logic [VERSION_WIDTH-1:0]                  i_version,

    output logic                                      o_logit_wave_valid,
    input  logic                                      i_logit_wave_ready,
    output logic [ENGINES-1:0]                        o_logit_wave_mask,
    output logic [CLASS_WIDTH-1:0]                    o_logit_wave_base,
    output logic [OUTPUT_COUNT_WIDTH-1:0]             o_logit_wave_count,
    output logic [ENGINES*16-1:0]                     o_logit_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]                  o_logit_wave_version,
    output logic [FRAME_WIDTH-1:0]                    o_logit_wave_frame_index,
    output logic                                      o_logit_wave_frame_last,
    output logic                                      o_logit_wave_sample_last,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [CLASS_WIDTH-1:0]                    o_result_class,
    output logic signed [15:0]                        o_result_score_q8_8,
    output logic [VERSION_WIDTH-1:0]                  o_result_version,
    output logic [FRAME_WIDTH-1:0]                    o_result_frame_index,
    output logic                                      o_result_sample_last,

    output logic [VERSION_WIDTH-1:0]                  o_locked_version,
    output logic [FRAME_WIDTH-1:0]                    o_frames_seen,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_RESET_WHILE_BUSY = 8'h01;
    localparam logic [7:0] ERROR_CONFIG = 8'h02;
    localparam logic [7:0] ERROR_GROUP_ORDER = 8'h03;
    localparam logic [7:0] ERROR_BOUNDARY_ORDER = 8'h04;
    localparam logic [7:0] ERROR_MIXED_VERSION = 8'h05;

    localparam logic signed [LI_STATE_WIDTH-1:0] SCORE_MAX =
        LI_STATE_WIDTH'(32767);
    localparam logic signed [LI_STATE_WIDTH-1:0] SCORE_MIN =
        -LI_STATE_WIDTH'(32768);
    localparam int ARGMAX_LEAVES = 1 << $clog2(ENGINES + 1);
    localparam int ARGMAX_LEVELS = $clog2(ARGMAX_LEAVES);

    logic sample_active_q;
    logic group_active_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic [OUTPUT_COUNT_WIDTH-1:0] output_count_q;
    logic [CLASS_WIDTH-1:0] expected_output_base_q;
    logic [CLASS_WIDTH-1:0] group_output_base_q;
    logic [ENGINES-1:0] group_engine_mask_q;
    logic [FRAME_WIDTH-1:0] frame_count_q;

    logic signed [ACC_WIDTH-1:0] group_sum_q [0:ENGINES-1];
    logic signed [LI_STATE_WIDTH-1:0] li_state_q [0:OUTPUT_MAX-1];

    logic frame_best_valid_q;
    logic [CLASS_WIDTH-1:0] frame_best_class_q;
    logic signed [15:0] frame_best_score_q;

    logic result_valid_q;
    logic [CLASS_WIDTH-1:0] result_class_q;
    logic signed [15:0] result_score_q;
    logic [VERSION_WIDTH-1:0] result_version_q;
    logic [FRAME_WIDTH-1:0] result_frame_index_q;
    logic result_sample_last_q;
    logic sticky_error_q;
    logic [7:0] error_code_q;

    logic [OUTPUT_COUNT_WIDTH-1:0] effective_output_count;
    logic [CLASS_WIDTH-1:0] effective_expected_base;
    logic [ENGINES-1:0] required_engine_mask;
    logic [$clog2(ENGINES + 1)-1:0] wave_span;
    logic last_output_wave;
    logic [7:0] raw_error_code;
    logic raw_fire;

    logic signed [ACC_WIDTH-1:0] raw_partial [0:ENGINES-1];
    logic signed [ACC_WIDTH-1:0] completed_group_sum [0:ENGINES-1];
    logic signed [LI_STATE_WIDTH-1:0] li_delta [0:ENGINES-1];
    logic signed [LI_STATE_WIDTH-1:0] li_next [0:ENGINES-1];
    logic signed [15:0] li_next_score [0:ENGINES-1];
    logic candidate_best_valid;
    logic [CLASS_WIDTH-1:0] candidate_best_class;
    logic signed [15:0] candidate_best_score;
    logic argmax_valid [0:ARGMAX_LEVELS][0:ARGMAX_LEAVES-1];
    logic [CLASS_WIDTH-1:0]
        argmax_class [0:ARGMAX_LEVELS][0:ARGMAX_LEAVES-1];
    logic signed [15:0]
        argmax_score [0:ARGMAX_LEVELS][0:ARGMAX_LEAVES-1];

    integer state_index;
    integer engine_index;

    function automatic logic [ENGINES-1:0] expected_mask(
        input logic [CLASS_WIDTH-1:0] base,
        input logic [OUTPUT_COUNT_WIDTH-1:0] count
    );
        logic [ENGINES-1:0] mask;
        begin
            mask = '0;
            for (integer engine = 0; engine < ENGINES; engine++) begin
                if (integer'(base) + engine < integer'(count))
                    mask[engine] = 1'b1;
            end
            return mask;
        end
    endfunction

    function automatic logic signed [15:0] saturate_score(
        input logic signed [LI_STATE_WIDTH-1:0] value
    );
        begin
            if ($signed(value) > $signed(SCORE_MAX))
                return 16'sh7fff;
            if ($signed(value) < $signed(SCORE_MIN))
                return 16'sh8000;
            return value[15:0];
        end
    endfunction

    always_comb begin
        effective_output_count = sample_active_q ? output_count_q :
                                                   i_output_count;
        effective_expected_base = sample_active_q ? expected_output_base_q :
                                                    '0;
        required_engine_mask = expected_mask(i_output_base,
                                             effective_output_count);
        wave_span = '0;
        for (integer engine = 0; engine < ENGINES; engine++) begin
            if (required_engine_mask[engine])
                wave_span = wave_span + 1'b1;
        end
        last_output_wave = integer'(i_output_base) + integer'(wave_span) ==
                           integer'(effective_output_count);

        raw_error_code = ERROR_NONE;
        if (!sample_active_q &&
            (i_version == 0 || i_output_count == 0 ||
             integer'(i_output_count) > OUTPUT_MAX)) begin
            raw_error_code = ERROR_CONFIG;
        end else if (sample_active_q && i_version != sample_version_q) begin
            raw_error_code = ERROR_MIXED_VERSION;
        end else if (sample_active_q && i_output_count != output_count_q) begin
            raw_error_code = ERROR_CONFIG;
        end else if (integer'(i_output_base) >=
                     integer'(effective_output_count) ||
                     required_engine_mask == '0 ||
                     i_engine_mask != required_engine_mask) begin
            raw_error_code = ERROR_CONFIG;
        end else if (group_active_q &&
                     (i_first_input_group ||
                      i_output_base != group_output_base_q ||
                      i_engine_mask != group_engine_mask_q)) begin
            raw_error_code = ERROR_GROUP_ORDER;
        end else if (!group_active_q &&
                     (!i_first_input_group ||
                      i_output_base != effective_expected_base)) begin
            raw_error_code = ERROR_GROUP_ORDER;
        end else if ((!i_final_input_group &&
                      (i_frame_last || i_sample_last)) ||
                     (i_sample_last && !i_frame_last) ||
                     (i_final_input_group &&
                      (i_frame_last != last_output_wave))) begin
            raw_error_code = ERROR_BOUNDARY_ORDER;
        end

        for (integer engine = 0; engine < ENGINES; engine++) begin
            raw_partial[engine] = $signed(
                i_raw_partial_flat[engine*ACC_WIDTH +: ACC_WIDTH]);
            completed_group_sum[engine] =
                (group_active_q ? group_sum_q[engine] : '0) +
                raw_partial[engine];
            li_delta[engine] = LI_STATE_WIDTH'(
                $signed(completed_group_sum[engine]) >>> 8);
            li_next[engine] = '0;
            li_next_score[engine] = '0;
            if (i_engine_mask[engine] &&
                integer'(i_output_base) + engine < OUTPUT_MAX) begin
                li_next[engine] = $signed(
                    li_state_q[integer'(i_output_base) + engine]) +
                    $signed(li_delta[engine]);
                li_next_score[engine] = saturate_score(li_next[engine]);
            end
        end

        for (integer level = 0; level <= ARGMAX_LEVELS; level++) begin
            for (integer node = 0; node < ARGMAX_LEAVES; node++) begin
                argmax_valid[level][node] = 1'b0;
                argmax_class[level][node] = '0;
                argmax_score[level][node] = '0;
            end
        end

        // Keep the previous frame winner first so equal scores preserve the
        // original lowest-index/stable-winner behavior.
        argmax_valid[0][0] = frame_best_valid_q;
        argmax_class[0][0] = frame_best_class_q;
        argmax_score[0][0] = frame_best_score_q;
        for (integer engine = 0; engine < ENGINES; engine++) begin
            argmax_valid[0][engine + 1] = i_engine_mask[engine];
            argmax_class[0][engine + 1] = CLASS_WIDTH'(
                integer'(i_output_base) + engine);
            argmax_score[0][engine + 1] = li_next_score[engine];
        end

        for (integer level = 0; level < ARGMAX_LEVELS; level++) begin
            for (integer node = 0;
                 node < (ARGMAX_LEAVES >> (level + 1)); node++) begin
                if (argmax_valid[level][2*node + 1] &&
                    (!argmax_valid[level][2*node] ||
                     $signed(argmax_score[level][2*node + 1]) >
                     $signed(argmax_score[level][2*node]))) begin
                    argmax_valid[level + 1][node] =
                        argmax_valid[level][2*node + 1];
                    argmax_class[level + 1][node] =
                        argmax_class[level][2*node + 1];
                    argmax_score[level + 1][node] =
                        argmax_score[level][2*node + 1];
                end else begin
                    argmax_valid[level + 1][node] =
                        argmax_valid[level][2*node];
                    argmax_class[level + 1][node] =
                        argmax_class[level][2*node];
                    argmax_score[level + 1][node] =
                        argmax_score[level][2*node];
                end
            end
        end

        candidate_best_valid = argmax_valid[ARGMAX_LEVELS][0];
        candidate_best_class = argmax_class[ARGMAX_LEVELS][0];
        candidate_best_score = argmax_score[ARGMAX_LEVELS][0];
    end

    assign o_logit_wave_valid = i_raw_valid && i_final_input_group &&
        !sticky_error_q && !result_valid_q && !i_sample_reset &&
        (raw_error_code == ERROR_NONE);
    assign o_logit_wave_mask = i_engine_mask;
    assign o_logit_wave_base = i_output_base;
    assign o_logit_wave_count = effective_output_count;
    assign o_logit_wave_version = sample_active_q ? sample_version_q :
                                                      i_version;
    assign o_logit_wave_frame_index = frame_count_q;
    assign o_logit_wave_frame_last = i_frame_last;
    assign o_logit_wave_sample_last = i_sample_last;
    for (genvar logit_engine = 0; logit_engine < ENGINES; logit_engine++) begin : gen_logit_wave
        assign o_logit_wave_q8_8_flat[logit_engine*16 +: 16] =
            li_next_score[logit_engine];
    end

    assign o_raw_ready = !sticky_error_q && !result_valid_q &&
                         !i_sample_reset &&
                         (!i_raw_valid || !i_final_input_group ||
                          i_logit_wave_ready);
    assign raw_fire = i_raw_valid && o_raw_ready;

    assign o_result_valid = result_valid_q;
    assign o_result_class = result_class_q;
    assign o_result_score_q8_8 = result_score_q;
    assign o_result_version = result_version_q;
    assign o_result_frame_index = result_frame_index_q;
    assign o_result_sample_last = result_sample_last_q;
    assign o_locked_version = sample_version_q;
    assign o_frames_seen = frame_count_q;
    assign o_busy = sample_active_q || group_active_q || result_valid_q;
    assign o_idle = !o_busy;
    assign o_sticky_error = sticky_error_q;
    assign o_error_code = error_code_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            sample_active_q <= 1'b0;
            group_active_q <= 1'b0;
            sample_version_q <= '0;
            output_count_q <= '0;
            expected_output_base_q <= '0;
            group_output_base_q <= '0;
            group_engine_mask_q <= '0;
            frame_count_q <= '0;
            frame_best_valid_q <= 1'b0;
            frame_best_class_q <= '0;
            frame_best_score_q <= '0;
            result_valid_q <= 1'b0;
            result_class_q <= '0;
            result_score_q <= '0;
            result_version_q <= '0;
            result_frame_index_q <= '0;
            result_sample_last_q <= 1'b0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            for (engine_index = 0; engine_index < ENGINES;
                 engine_index = engine_index + 1)
                group_sum_q[engine_index] <= '0;
            for (state_index = 0; state_index < OUTPUT_MAX;
                 state_index = state_index + 1)
                li_state_q[state_index] <= '0;
        end else if (i_clear_error && o_idle) begin
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
        end else if (i_sample_reset) begin
            if (group_active_q || result_valid_q) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_RESET_WHILE_BUSY;
            end
            sample_active_q <= 1'b0;
            group_active_q <= 1'b0;
            sample_version_q <= '0;
            output_count_q <= '0;
            expected_output_base_q <= '0;
            group_output_base_q <= '0;
            group_engine_mask_q <= '0;
            frame_count_q <= '0;
            frame_best_valid_q <= 1'b0;
            frame_best_class_q <= '0;
            frame_best_score_q <= '0;
            result_valid_q <= 1'b0;
            result_sample_last_q <= 1'b0;
            for (engine_index = 0; engine_index < ENGINES;
                 engine_index = engine_index + 1)
                group_sum_q[engine_index] <= '0;
            for (state_index = 0; state_index < OUTPUT_MAX;
                 state_index = state_index + 1)
                li_state_q[state_index] <= '0;
        end else if (result_valid_q && i_result_ready) begin
            result_valid_q <= 1'b0;
            if (result_sample_last_q) begin
                sample_active_q <= 1'b0;
                sample_version_q <= '0;
                output_count_q <= '0;
                expected_output_base_q <= '0;
                frame_count_q <= '0;
                frame_best_valid_q <= 1'b0;
                frame_best_class_q <= '0;
                frame_best_score_q <= '0;
                result_sample_last_q <= 1'b0;
                for (state_index = 0; state_index < OUTPUT_MAX;
                     state_index = state_index + 1)
                    li_state_q[state_index] <= '0;
            end
        end else if (raw_fire) begin
            if (raw_error_code != ERROR_NONE) begin
                sticky_error_q <= 1'b1;
                error_code_q <= raw_error_code;
                sample_active_q <= 1'b0;
                group_active_q <= 1'b0;
                sample_version_q <= '0;
                output_count_q <= '0;
                expected_output_base_q <= '0;
                group_output_base_q <= '0;
                group_engine_mask_q <= '0;
                frame_count_q <= '0;
                frame_best_valid_q <= 1'b0;
                frame_best_class_q <= '0;
                frame_best_score_q <= '0;
                for (engine_index = 0; engine_index < ENGINES;
                     engine_index = engine_index + 1)
                    group_sum_q[engine_index] <= '0;
                for (state_index = 0; state_index < OUTPUT_MAX;
                     state_index = state_index + 1)
                    li_state_q[state_index] <= '0;
            end else begin
                if (!sample_active_q) begin
                    sample_active_q <= 1'b1;
                    sample_version_q <= i_version;
                    output_count_q <= i_output_count;
                    frame_count_q <= '0;
                end

                if (i_final_input_group) begin
                    group_active_q <= 1'b0;
                    group_output_base_q <= '0;
                    group_engine_mask_q <= '0;
                    for (engine_index = 0; engine_index < ENGINES;
                         engine_index = engine_index + 1) begin
                        group_sum_q[engine_index] <= '0;
                        if (i_engine_mask[engine_index]) begin
                            li_state_q[integer'(i_output_base) +
                                       engine_index] <= li_next[engine_index];
                        end
                    end

                    if (i_frame_last) begin
                        expected_output_base_q <= '0;
                        frame_count_q <= frame_count_q + 1'b1;
                        frame_best_valid_q <= 1'b0;
                        frame_best_class_q <= '0;
                        frame_best_score_q <= '0;
                        result_valid_q <= 1'b1;
                        result_class_q <= candidate_best_class;
                        result_score_q <= candidate_best_score;
                        result_version_q <= sample_active_q ?
                            sample_version_q : i_version;
                        result_frame_index_q <= frame_count_q;
                        result_sample_last_q <= i_sample_last;
                    end else begin
                        expected_output_base_q <= CLASS_WIDTH'(
                            integer'(i_output_base) + integer'(wave_span));
                        frame_best_valid_q <= candidate_best_valid;
                        frame_best_class_q <= candidate_best_class;
                        frame_best_score_q <= candidate_best_score;
                    end
                end else begin
                    group_active_q <= 1'b1;
                    group_output_base_q <= i_output_base;
                    group_engine_mask_q <= i_engine_mask;
                    for (engine_index = 0; engine_index < ENGINES;
                         engine_index = engine_index + 1)
                        group_sum_q[engine_index] <=
                            completed_group_sum[engine_index];
                end
            end
        end
    end

endmodule

`default_nettype wire
