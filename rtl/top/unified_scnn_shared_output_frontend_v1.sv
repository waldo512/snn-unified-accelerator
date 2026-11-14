`default_nettype none

// V2.2 OUTPUT request frontend for the externally owned shared K8 array.
// The fixed 2x2 final-feature layout needs two conflict-free K4 reads to form
// one logical K8 input group. Arithmetic and cross-frame LI state are external
// to this scheduler or retained by the existing OUTPUT/LI leaf, respectively.
module unified_scnn_shared_output_frontend_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int INPUT_FEATURES = 2048,
    parameter int OUTPUT_MAX = 32,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH),
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int ACC_WIDTH = 48,
    parameter int FEATURE_INDEX_WIDTH = INPUT_FEATURES <= 2 ?
        1 : $clog2(INPUT_FEATURES),
    parameter int CLASS_WIDTH = OUTPUT_MAX <= 1 ? 1 : $clog2(OUTPUT_MAX),
    parameter int OUTPUT_COUNT_WIDTH = $clog2(OUTPUT_MAX + 1)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_owner_enable,
    input  logic                                      i_clear_error,

    input  logic                                      i_layer_valid,
    output logic                                      o_layer_ready,
    input  logic [383:0]                              i_layer_words,
    input  logic                                      i_input_slot,
    input  logic                                      i_sample_reset,
    input  logic                                      i_sample_last,
    input  logic [VERSION_WIDTH-1:0]                  i_version,

    output logic                                      o_feature_read_valid,
    input  logic                                      i_feature_read_ready,
    output logic                                      o_feature_read_slot,
    output logic [31:0]                               o_feature_read_bank_flat,
    output logic [LANES*FEATURE_ADDR_WIDTH-1:0]        o_feature_read_addr_flat,
    output logic [LANES-1:0]                          o_feature_read_lane_mask,
    input  logic                                      i_feature_valid,
    output logic                                      o_feature_ready,
    input  logic [LANES*16-1:0]                       i_feature_flat,
    input  logic [LANES-1:0]                          i_feature_lane_mask,

    output logic                                      o_weight_read_valid,
    input  logic                                      i_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_weight_read_addr1,
    input  logic                                      i_weight_result_valid,
    output logic                                      o_weight_result_ready,
    input  logic [63:0]                               i_weight_data0,
    input  logic [63:0]                               i_weight_data1,

    output logic                                      o_feature_trace_wave_valid,
    input  logic                                      i_feature_trace_wave_ready,
    output logic [LANES-1:0]                          o_feature_trace_wave_mask,
    output logic [FEATURE_INDEX_WIDTH-1:0]            o_feature_trace_wave_base,
    output logic [LANES*16-1:0]                       o_feature_trace_wave_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]                  o_feature_trace_wave_version,
    output logic [FRAME_WIDTH-1:0]                    o_feature_trace_wave_frame_index,
    output logic                                      o_feature_trace_wave_first,
    output logic                                      o_feature_trace_wave_last,
    output logic                                      o_feature_trace_wave_sample_reset,
    output logic                                      o_feature_trace_wave_sample_last,

    output logic                                      o_side_request_valid,
    input  logic                                      i_side_request_ready,
    output logic [ENGINES-1:0]                        o_side_engine_mask,
    output logic [LANES*16-1:0]                       o_side_feature_flat,
    output logic [ENGINES*LANES*16-1:0]               o_side_weight_flat,
    output logic [LANES-1:0]                          o_side_lane_mask,
    output logic                                      o_side_first,
    output logic                                      o_side_last,
    output logic [9:0]                                o_side_output_channel_base,
    input  logic                                      i_side_result_valid,
    output logic                                      o_side_result_ready,
    input  logic [ENGINES-1:0]                        i_side_result_mask,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_side_result_accumulator_flat,
    input  logic                                      i_side_sticky_error,

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
    output logic                                      o_done,
    output logic                                      o_config_error,
    output logic                                      o_protocol_error,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam int INPUT_GROUPS = INPUT_FEATURES / LANES;
    localparam int INPUT_GROUP_WIDTH = INPUT_GROUPS <= 1 ?
        1 : $clog2(INPUT_GROUPS);
    localparam int INPUT_SHIFT = $clog2(INPUT_FEATURES);

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_DESCRIPTOR = 8'h01;
    localparam logic [7:0] ERROR_MIXED_VERSION = 8'h02;
    localparam logic [7:0] ERROR_FEATURE_PROTOCOL = 8'h03;
    localparam logic [7:0] ERROR_SIDE_PROTOCOL = 8'h04;
    localparam logic [7:0] ERROR_OWNER_REVOKED = 8'h05;
    localparam logic [7:0] ERROR_WEIGHT_PROTOCOL = 8'h06;
    localparam logic [7:0] ERROR_ABORT = 8'h07;

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_SAMPLE_RESET,
        STATE_FEATURE0_ISSUE,
        STATE_FEATURE0_WAIT,
        STATE_FEATURE1_ISSUE,
        STATE_FEATURE1_WAIT,
        STATE_FEATURE_OBSERVE,
        STATE_WEIGHT_ISSUE,
        STATE_WEIGHT_WAIT,
        STATE_COMMAND,
        STATE_RAW_RESULT,
        STATE_FINAL_RESULT
    } state_t;

    state_t state_q;
    logic input_slot_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [Q16_ADDR_WIDTH-1:0] weight_base_q;
    logic [OUTPUT_COUNT_WIDTH-1:0] output_count_q;
    logic [CLASS_WIDTH-1:0] output_base_q;
    logic [INPUT_GROUP_WIDTH-1:0] input_group_q;
    logic [LANES*16-1:0] feature_group_q;
    logic [ENGINES*LANES*16-1:0] weight_group_q;

    logic sample_lock_active_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic [OUTPUT_COUNT_WIDTH-1:0] sample_output_count_q;
    logic local_sticky_error_q;
    logic [7:0] local_error_code_q;

    logic descriptor_legal;
    logic descriptor_mixed_version;
    logic descriptor_mixed_count;
    logic [32:0] descriptor_weight_end;
    logic [OUTPUT_COUNT_WIDTH-1:0] descriptor_output_count;
    logic [ENGINES-1:0] current_engine_mask;
    logic [ENGINES*LANES*Q16_ADDR_WIDTH-1:0] gather_addr_flat;
    logic current_final_group;
    logic current_final_wave;
    logic layer_fire;
    logic command_fire;
    logic raw_result_fire;
    logic final_result_fire;

    logic gather_request_ready;
    logic gather_weight_valid;
    logic gather_weight_ready;
    logic gather_read_result_ready;
    logic [ENGINES*LANES*16-1:0] gather_weight_flat;
    logic [ENGINES-1:0] gather_engine_mask;
    logic [LANES-1:0] gather_lane_mask;
    logic gather_busy;
    logic gather_error;

    logic li_sample_reset;
    logic li_raw_valid;
    logic li_raw_ready;
    logic li_result_valid;
    logic [CLASS_WIDTH-1:0] li_result_class;
    logic signed [15:0] li_result_score;
    logic [VERSION_WIDTH-1:0] li_result_version;
    logic [FRAME_WIDTH-1:0] li_result_frame_index;
    logic li_result_sample_last;
    logic [VERSION_WIDTH-1:0] li_locked_version;
    logic [FRAME_WIDTH-1:0] li_frames_seen;
    logic li_busy;
    logic li_idle;
    logic li_sticky_error;
    logic [7:0] li_error_code;
    function automatic logic [ENGINES-1:0] engine_mask_for(
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

    always_comb begin
        descriptor_output_count = OUTPUT_COUNT_WIDTH'(
            i_layer_words[3*32 + 23 +: 6]);
        descriptor_weight_end = {1'b0, i_layer_words[7*32 +: 32]} +
            ({27'd0, i_layer_words[3*32 + 23 +: 6]} << INPUT_SHIFT);
        descriptor_legal = LANES == 8 && ENGINES == 12 &&
            INPUT_FEATURES >= LANES &&
            (INPUT_FEATURES & (INPUT_FEATURES - 1)) == 0 &&
            INPUT_FEATURES % LANES == 0 && INPUT_FEATURES % 4 == 0 &&
            i_layer_words[0*32 +: 32] == 32'h4e4d5232 &&
            i_layer_words[1*32 +: 32] == 32'h000c0202 &&
            i_layer_words[2*32 +: 4] == 4'd4 &&
            i_layer_words[3*32 +: 13] == 13'(INPUT_FEATURES) &&
            descriptor_output_count != 0 &&
            integer'(descriptor_output_count) <= OUTPUT_MAX &&
            descriptor_weight_end <= (33'd1 << Q16_ADDR_WIDTH) &&
            i_layer_words[11*32 +: 32] == 32'h55534e31 &&
            i_version != '0;
        descriptor_mixed_version = sample_lock_active_q &&
            !i_sample_reset && i_version != sample_version_q;
        descriptor_mixed_count = sample_lock_active_q &&
            !i_sample_reset &&
            descriptor_output_count != sample_output_count_q;

        current_engine_mask = engine_mask_for(output_base_q,
                                              output_count_q);
        current_final_group = integer'(input_group_q) + 1 >= INPUT_GROUPS;
        current_final_wave = integer'(output_base_q) + ENGINES >=
                             integer'(output_count_q);

        o_feature_read_slot = input_slot_q;
        o_feature_read_bank_flat = '0;
        o_feature_read_bank_flat[0*4 +: 4] = 4'd0;
        o_feature_read_bank_flat[1*4 +: 4] = 4'd1;
        o_feature_read_bank_flat[2*4 +: 4] = 4'd3;
        o_feature_read_bank_flat[3*4 +: 4] = 4'd4;
        o_feature_read_addr_flat = '0;
        for (integer lane = 0; lane < 4; lane++) begin
            o_feature_read_addr_flat[
                lane*FEATURE_ADDR_WIDTH +: FEATURE_ADDR_WIDTH
            ] = FEATURE_ADDR_WIDTH'(
                (integer'(input_group_q) << 1) +
                ((state_q == STATE_FEATURE1_ISSUE ||
                  state_q == STATE_FEATURE1_WAIT) ? 1 : 0));
        end
        o_feature_read_lane_mask = {{(LANES-4){1'b0}}, 4'hf};

        gather_addr_flat = '0;
        for (integer engine = 0; engine < ENGINES; engine++) begin
            for (integer lane = 0; lane < LANES; lane++) begin
                gather_addr_flat[
                    (engine*LANES + lane)*Q16_ADDR_WIDTH +:
                    Q16_ADDR_WIDTH
                ] = weight_base_q +
                    (Q16_ADDR_WIDTH'(
                        integer'(output_base_q) + engine) << INPUT_SHIFT) +
                    (Q16_ADDR_WIDTH'(input_group_q) << 3) +
                    Q16_ADDR_WIDTH'(lane);
            end
        end
    end

    assign o_layer_ready = !i_abort && i_owner_enable && state_q == STATE_IDLE &&
        gather_request_ready && !li_result_valid && !o_sticky_error;
    assign layer_fire = i_layer_valid && o_layer_ready;

    assign o_feature_read_valid =
        (state_q == STATE_FEATURE0_ISSUE ||
         state_q == STATE_FEATURE1_ISSUE) && !o_sticky_error && !i_abort;
    // The scratchpad may already own a response when the sample is canceled.
    assign o_feature_ready = i_abort || (
        (state_q == STATE_FEATURE0_WAIT ||
         state_q == STATE_FEATURE1_WAIT) && !o_sticky_error);

    assign gather_weight_ready = state_q == STATE_WEIGHT_WAIT &&
                                 !o_sticky_error;
    assign o_weight_result_ready = i_abort || gather_read_result_ready;

    assign o_feature_trace_wave_valid =
        state_q == STATE_FEATURE_OBSERVE && i_owner_enable &&
        !o_sticky_error && !i_abort;
    assign o_feature_trace_wave_mask = {LANES{1'b1}};
    assign o_feature_trace_wave_base = FEATURE_INDEX_WIDTH'(
        integer'(input_group_q) * LANES);
    assign o_feature_trace_wave_q8_8_flat = feature_group_q;
    assign o_feature_trace_wave_version = version_q;
    assign o_feature_trace_wave_frame_index = li_frames_seen;
    assign o_feature_trace_wave_first = input_group_q == 0;
    assign o_feature_trace_wave_last = current_final_group;
    assign o_feature_trace_wave_sample_reset = sample_reset_q &&
        o_feature_trace_wave_first;
    assign o_feature_trace_wave_sample_last = sample_last_q &&
        o_feature_trace_wave_last;

    assign o_side_request_valid = state_q == STATE_COMMAND &&
                                  i_owner_enable && !o_sticky_error && !i_abort;
    assign o_side_engine_mask = current_engine_mask;
    assign o_side_feature_flat = feature_group_q;
    assign o_side_weight_flat = weight_group_q;
    assign o_side_lane_mask = {LANES{1'b1}};
    assign o_side_first = 1'b1;
    assign o_side_last = 1'b1;
    assign o_side_output_channel_base = 10'(output_base_q);
    assign command_fire = o_side_request_valid && i_side_request_ready;

    assign li_raw_valid = state_q == STATE_RAW_RESULT &&
        i_side_result_valid && i_side_result_mask == current_engine_mask &&
        !o_sticky_error && !i_abort;
    assign o_side_result_ready = i_abort || (state_q == STATE_RAW_RESULT &&
        li_raw_ready && i_side_result_mask == current_engine_mask &&
        !o_sticky_error);
    assign raw_result_fire = li_raw_valid && li_raw_ready;

    assign li_sample_reset = i_abort || state_q == STATE_SAMPLE_RESET;
    assign o_result_valid = li_result_valid && state_q == STATE_FINAL_RESULT &&
                            !o_sticky_error && !i_abort;
    assign o_result_class = li_result_class;
    assign o_result_score_q8_8 = li_result_score;
    assign o_result_version = li_result_version;
    assign o_result_frame_index = li_result_frame_index;
    assign o_result_sample_last = li_result_sample_last;
    assign final_result_fire = o_result_valid && i_result_ready;

    assign o_locked_version = sample_lock_active_q ? sample_version_q :
                                                        li_locked_version;
    assign o_frames_seen = li_frames_seen;
    assign o_busy = state_q != STATE_IDLE || gather_busy || li_result_valid;
    assign o_sticky_error = local_sticky_error_q || gather_error ||
                            li_sticky_error || i_side_sticky_error;
    assign o_config_error = local_sticky_error_q &&
                            local_error_code_q == ERROR_DESCRIPTOR;
    assign o_protocol_error = (local_sticky_error_q &&
        local_error_code_q != ERROR_DESCRIPTOR) || gather_error ||
        li_sticky_error || i_side_sticky_error;

    always_comb begin
        if (local_sticky_error_q)
            o_error_code = local_error_code_q;
        else if (gather_error)
            o_error_code = 8'h40;
        else if (li_sticky_error)
            o_error_code = 8'h80 | li_error_code;
        else if (i_side_sticky_error)
            o_error_code = 8'hc0;
        else
            o_error_code = ERROR_NONE;
    end

    unified_scnn_q16_wave_weight_gather_soft_abort_v1 #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .Q64_ADDRESS_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDRESS_WIDTH(Q16_ADDR_WIDTH)
    ) u_weight_gather (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_request_valid(state_q == STATE_WEIGHT_ISSUE &&
                         !o_sticky_error),
        .o_request_ready(gather_request_ready),
        .i_engine_mask(current_engine_mask),
        .i_lane_mask({LANES{1'b1}}),
        .i_weight_q16_addr_flat(gather_addr_flat),
        .o_read_valid(o_weight_read_valid),
        .i_read_ready(i_weight_read_ready),
        .o_read_word_addr0(o_weight_read_addr0),
        .o_read_word_addr1(o_weight_read_addr1),
        .i_read_result_valid(i_weight_result_valid),
        .o_read_result_ready(gather_read_result_ready),
        .i_read_data0(i_weight_data0),
        .i_read_data1(i_weight_data1),
        .o_weight_valid(gather_weight_valid),
        .i_weight_ready(gather_weight_ready),
        .o_weight_flat(gather_weight_flat),
        .o_engine_mask(gather_engine_mask),
        .o_lane_mask(gather_lane_mask),
        .o_busy(gather_busy),
        .o_sticky_error(gather_error)
    );

    unified_scnn_shared_array_output_li_v1 #(
        .ENGINES(ENGINES),
        .OUTPUT_MAX(OUTPUT_MAX),
        .ACC_WIDTH(ACC_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH),
        .FRAME_WIDTH(FRAME_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH),
        .OUTPUT_COUNT_WIDTH(OUTPUT_COUNT_WIDTH)
    ) u_output_li (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_sample_reset(li_sample_reset),
        .i_raw_valid(li_raw_valid),
        .o_raw_ready(li_raw_ready),
        .i_engine_mask(current_engine_mask),
        .i_output_base(output_base_q),
        .i_output_count(output_count_q),
        .i_raw_partial_flat(i_side_result_accumulator_flat),
        .i_first_input_group(input_group_q == 0),
        .i_final_input_group(current_final_group),
        .i_frame_last(current_final_group && current_final_wave),
        .i_sample_last(current_final_group && current_final_wave &&
                       sample_last_q),
        .i_version(version_q),
        .o_logit_wave_valid(o_logit_wave_valid),
        .i_logit_wave_ready(i_logit_wave_ready),
        .o_logit_wave_mask(o_logit_wave_mask),
        .o_logit_wave_base(o_logit_wave_base),
        .o_logit_wave_count(o_logit_wave_count),
        .o_logit_wave_q8_8_flat(o_logit_wave_q8_8_flat),
        .o_logit_wave_version(o_logit_wave_version),
        .o_logit_wave_frame_index(o_logit_wave_frame_index),
        .o_logit_wave_frame_last(o_logit_wave_frame_last),
        .o_logit_wave_sample_last(o_logit_wave_sample_last),
        .o_result_valid(li_result_valid),
        .i_result_ready(i_result_ready && state_q == STATE_FINAL_RESULT &&
                        !o_sticky_error),
        .o_result_class(li_result_class),
        .o_result_score_q8_8(li_result_score),
        .o_result_version(li_result_version),
        .o_result_frame_index(li_result_frame_index),
        .o_result_sample_last(li_result_sample_last),
        .o_locked_version(li_locked_version),
        .o_frames_seen(li_frames_seen),
        .o_busy(li_busy),
        .o_idle(li_idle),
        .o_sticky_error(li_sticky_error),
        .o_error_code(li_error_code)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            input_slot_q <= 1'b0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            version_q <= '0;
            weight_base_q <= '0;
            output_count_q <= '0;
            output_base_q <= '0;
            input_group_q <= '0;
            feature_group_q <= '0;
            weight_group_q <= '0;
            sample_lock_active_q <= 1'b0;
            sample_version_q <= '0;
            sample_output_count_q <= '0;
            local_sticky_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
            o_done <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            sample_lock_active_q <= 1'b0;
            sample_version_q <= '0;
            sample_output_count_q <= '0;
            version_q <= '0;
            output_count_q <= '0;
            output_base_q <= '0;
            input_group_q <= '0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            o_done <= 1'b0;
            if (!local_sticky_error_q && (o_busy || sample_lock_active_q)) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_ABORT;
            end
        end else begin
            o_done <= 1'b0;

            if (i_clear_error && state_q == STATE_IDLE && !gather_busy &&
                !li_result_valid) begin
                local_sticky_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (state_q != STATE_IDLE && !i_owner_enable) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_OWNER_REVOKED;
                state_q <= STATE_IDLE;
            end else if (gather_error || li_sticky_error ||
                         i_side_sticky_error) begin
                state_q <= STATE_IDLE;
            end else begin
                case (state_q)
                    STATE_IDLE: begin
                        if (layer_fire) begin
                            if (!descriptor_legal ||
                                descriptor_mixed_count) begin
                                local_sticky_error_q <= 1'b1;
                                local_error_code_q <= ERROR_DESCRIPTOR;
                            end else if (descriptor_mixed_version) begin
                                local_sticky_error_q <= 1'b1;
                                local_error_code_q <= ERROR_MIXED_VERSION;
                            end else begin
                                input_slot_q <= i_input_slot;
                                sample_reset_q <= i_sample_reset;
                                sample_last_q <= i_sample_last;
                                version_q <= i_version;
                                weight_base_q <= i_layer_words[
                                    7*32 +: Q16_ADDR_WIDTH];
                                output_count_q <= descriptor_output_count;
                                output_base_q <= '0;
                                input_group_q <= '0;
                                feature_group_q <= '0;
                                weight_group_q <= '0;
                                sample_lock_active_q <= 1'b1;
                                sample_version_q <= i_version;
                                sample_output_count_q <=
                                    descriptor_output_count;
                                state_q <= i_sample_reset ?
                                    STATE_SAMPLE_RESET :
                                    STATE_FEATURE0_ISSUE;
                            end
                        end
                    end
                    STATE_SAMPLE_RESET: begin
                        state_q <= STATE_FEATURE0_ISSUE;
                    end
                    STATE_FEATURE0_ISSUE: begin
                        if (i_feature_read_ready)
                            state_q <= STATE_FEATURE0_WAIT;
                    end
                    STATE_FEATURE0_WAIT: begin
                        if (i_feature_valid) begin
                            if (i_feature_lane_mask !=
                                {{(LANES-4){1'b0}}, 4'hf}) begin
                                local_sticky_error_q <= 1'b1;
                                local_error_code_q <=
                                    ERROR_FEATURE_PROTOCOL;
                                state_q <= STATE_IDLE;
                            end else begin
                                feature_group_q[0 +: 4*16] <=
                                    i_feature_flat[0 +: 4*16];
                                state_q <= STATE_FEATURE1_ISSUE;
                            end
                        end
                    end
                    STATE_FEATURE1_ISSUE: begin
                        if (i_feature_read_ready)
                            state_q <= STATE_FEATURE1_WAIT;
                    end
                    STATE_FEATURE1_WAIT: begin
                        if (i_feature_valid) begin
                            if (i_feature_lane_mask !=
                                {{(LANES-4){1'b0}}, 4'hf}) begin
                                local_sticky_error_q <= 1'b1;
                                local_error_code_q <=
                                    ERROR_FEATURE_PROTOCOL;
                                state_q <= STATE_IDLE;
                            end else begin
                                feature_group_q[4*16 +: 4*16] <=
                                    i_feature_flat[0 +: 4*16];
                                state_q <= output_base_q == 0 ?
                                    STATE_FEATURE_OBSERVE :
                                    STATE_WEIGHT_ISSUE;
                            end
                        end
                    end
                    STATE_FEATURE_OBSERVE: begin
                        if (i_feature_trace_wave_ready)
                            state_q <= STATE_WEIGHT_ISSUE;
                    end
                    STATE_WEIGHT_ISSUE: begin
                        if (gather_request_ready)
                            state_q <= STATE_WEIGHT_WAIT;
                    end
                    STATE_WEIGHT_WAIT: begin
                        if (gather_weight_valid) begin
                            if (gather_engine_mask != current_engine_mask ||
                                gather_lane_mask != {LANES{1'b1}}) begin
                                local_sticky_error_q <= 1'b1;
                                local_error_code_q <= ERROR_SIDE_PROTOCOL;
                                state_q <= STATE_IDLE;
                            end else begin
                                weight_group_q <= gather_weight_flat;
                                state_q <= STATE_COMMAND;
                            end
                        end
                    end
                    STATE_COMMAND: begin
                        if (command_fire)
                            state_q <= STATE_RAW_RESULT;
                    end
                    STATE_RAW_RESULT: begin
                        if (i_side_result_valid &&
                            i_side_result_mask != current_engine_mask) begin
                            local_sticky_error_q <= 1'b1;
                            local_error_code_q <= ERROR_SIDE_PROTOCOL;
                            state_q <= STATE_IDLE;
                        end else if (raw_result_fire) begin
                            if (current_final_group) begin
                                input_group_q <= '0;
                                if (current_final_wave) begin
                                    state_q <= STATE_FINAL_RESULT;
                                end else begin
                                    output_base_q <= CLASS_WIDTH'(
                                        integer'(output_base_q) + ENGINES);
                                    state_q <= STATE_FEATURE0_ISSUE;
                                end
                            end else begin
                                input_group_q <= input_group_q + 1'b1;
                                state_q <= STATE_FEATURE0_ISSUE;
                            end
                        end
                    end
                    STATE_FINAL_RESULT: begin
                        if (final_result_fire) begin
                            o_done <= 1'b1;
                            state_q <= STATE_IDLE;
                            if (li_result_sample_last) begin
                                sample_lock_active_q <= 1'b0;
                                sample_version_q <= '0;
                                sample_output_count_q <= '0;
                            end
                        end
                    end
                    default: state_q <= STATE_IDLE;
                endcase
            end

            if (i_feature_valid && !o_feature_ready &&
                state_q != STATE_IDLE) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_FEATURE_PROTOCOL;
                state_q <= STATE_IDLE;
            end
            if (i_weight_result_valid && !o_weight_result_ready) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_WEIGHT_PROTOCOL;
                state_q <= STATE_IDLE;
            end
            if (i_side_result_valid && state_q != STATE_RAW_RESULT) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_SIDE_PROTOCOL;
                state_q <= STATE_IDLE;
            end
        end
    end

endmodule

`default_nettype wire
