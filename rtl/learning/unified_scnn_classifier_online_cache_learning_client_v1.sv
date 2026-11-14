`timescale 1ns/1ps
`default_nettype none

// Cache-coherent SCNN classifier learning client. The TP learner updates one
// persistent class-major parameter store. A completed learning window is not
// acknowledged until the updated matrix has been repacked into the inactive
// tile-cache pair; only then are active pair and version changed together.
module unified_scnn_classifier_online_cache_learning_client_v1 #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int MAX_FEATURES = 2048,
    parameter int MAX_CLASSES = 11,
    parameter int PARAM_DEPTH = MAX_FEATURES * MAX_CLASSES,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int ARRAY_WEIGHT_ADDR_WIDTH = 11,
    parameter int CLASSIFIER_CAPTURE_MAX_BATCH = MAX_BATCH,
    parameter int BATCH_INDEX_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int LEARNER_BATCH_INDEX_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int TIMESTEP_INDEX_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1),
    parameter int FEATURE_INDEX_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_INDEX_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter bit ENABLE_COMPUTED_TRACE_CAPTURE = 1'b0
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,
    input  wire logic                                  i_abort,
    input  wire logic                                  i_clear_error,

    input  wire logic                                  i_parameter_load_valid,
    output logic                                       o_parameter_load_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]            i_parameter_load_addr,
    input  wire logic signed [15:0]                     i_parameter_load_code,
    input  wire logic signed [15:0]                     i_parameter_load_residual,

    input  wire logic                                  i_cache_prime_valid,
    output logic                                       o_cache_prime_ready,
    input  wire logic                                  i_initial_pair,
    input  wire logic [VERSION_WIDTH-1:0]               i_initial_version,
    output logic                                       o_cache_prime_done,

    input  wire logic                                  i_cfg_valid,
    output logic                                       o_cfg_ready,
    input  wire logic [PROFILE_ID_WIDTH-1:0]            i_cfg_profile_id,
    input  wire logic [TENSOR_ID_WIDTH-1:0]             i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]            i_cfg_parameter_base,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]           i_cfg_batch_count,
    input  wire logic [TIMESTEP_INDEX_WIDTH-1:0]        i_cfg_timestep_base,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0]        i_cfg_timestep_count,
    input  wire logic                                  i_cfg_trace_state_reset,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0]         i_cfg_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]           i_cfg_class_count,
    input  wire logic                                  i_cfg_trace_leak_valid,
    input  wire logic signed [15:0]                     i_cfg_trace_leak_q8_8,
    input  wire logic                                  i_cfg_computed_trace_capture_enable,
    input  wire logic [15:0]                           i_cfg_vote_increment,
    input  wire logic [15:0]                           i_cfg_vote_threshold,
    input  wire logic signed [15:0]                    i_cfg_code_min,
    input  wire logic signed [15:0]                    i_cfg_code_max,
    output logic                                       o_active,
    input  wire logic                                  i_release,
    output logic                                       o_release_ready,
    input  wire logic                                  i_learning_enable,

    input  wire logic                                  i_feature_trace_wave_valid,
    output logic                                       o_feature_trace_wave_ready,
    input  wire logic [LANES-1:0]                      i_feature_trace_wave_mask,
    input  wire logic [FEATURE_INDEX_WIDTH-1:0]         i_feature_trace_wave_base,
    input  wire logic [LANES*16-1:0]                   i_feature_trace_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]               i_feature_trace_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]                 i_feature_trace_wave_frame_index,
    input  wire logic                                  i_feature_trace_wave_first,
    input  wire logic                                  i_feature_trace_wave_last,
    input  wire logic                                  i_feature_trace_wave_sample_reset,
    input  wire logic                                  i_feature_trace_wave_sample_last,
    input  wire logic [CLASS_INDEX_WIDTH-1:0]           i_sample_label,

    input  wire logic                                  i_logit_wave_valid,
    output logic                                       o_logit_wave_ready,
    input  wire logic [ENGINES-1:0]                    i_logit_wave_mask,
    input  wire logic [CLASS_INDEX_WIDTH-1:0]           i_logit_wave_base,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]           i_logit_wave_count,
    input  wire logic [ENGINES*16-1:0]                 i_logit_wave_q8_8_flat,
    input  wire logic [VERSION_WIDTH-1:0]               i_logit_wave_version,
    input  wire logic [FRAME_WIDTH-1:0]                 i_logit_wave_frame_index,
    input  wire logic                                  i_logit_wave_frame_last,
    input  wire logic                                  i_logit_wave_sample_last,

    input  wire logic                                  i_array_idle,
    output logic [3:0]                                 o_weight_fill_valid,
    input  wire logic [3:0]                            i_weight_fill_ready,
    output logic [7:0]                                 o_weight_fill_tile_flat,
    output logic [3:0]                                 o_weight_fill_bank,
    output logic [4*ARRAY_WEIGHT_ADDR_WIDTH-1:0]        o_weight_fill_addr_flat,
    output logic [799:0]                               o_weight_fill_data_flat,

    output logic                                       o_inference_ready,
    output logic                                       o_inference_hold,
    output logic                                       o_active_pair,
    output logic [VERSION_WIDTH-1:0]                   o_active_version,
    output logic                                       o_timestep_learning_done,
    output logic                                       o_batch_learning_done,
    output logic [31:0]                                o_commit_sequence,
    output logic                                       o_parameter_write_fire,
    output logic                                       o_cache_publish_busy,
    output logic                                       o_cache_valid,
    output logic                                       o_busy,
    output logic                                       o_sticky_error,

    output logic                                       o_computed_trace_valid,
    input  wire logic                                  i_computed_trace_ready,
    output logic [BATCH_INDEX_WIDTH-1:0]               o_computed_trace_sample_id,
    output logic [FEATURE_INDEX_WIDTH-1:0]             o_computed_trace_index,
    output logic [TIMESTEP_INDEX_WIDTH-1:0]            o_computed_trace_timestep,
    output logic [FRAME_WIDTH-1:0]                     o_computed_trace_frame_index,
    output logic [VERSION_WIDTH-1:0]                   o_computed_trace_version,
    output logic signed [15:0]                         o_computed_trace_q8_8
);
    typedef enum logic [1:0] {
        PUBLISH_IDLE,
        PUBLISH_PRIME,
        PUBLISH_COMMIT
    } publish_state_t;

    publish_state_t publish_state_q;
    logic cache_valid_q;
    logic active_pair_q;
    logic [VERSION_WIDTH-1:0] active_version_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;

    logic bridge_cfg_valid;
    logic bridge_cfg_ready;
    logic bridge_cfg_error;
    logic bridge_active;
    logic bridge_release_ready;
    logic bridge_commit_valid;
    logic bridge_commit_ready;
    logic [PROFILE_ID_WIDTH-1:0] bridge_commit_profile;
    logic [31:0] bridge_commit_sequence;
    logic bridge_window_busy;
    logic bridge_inference_hold;
    logic bridge_capture_error;
    logic bridge_context_error;
    logic bridge_gradient_overflow;
    logic bridge_parameter_error;
    logic bridge_protocol_error;
    logic bridge_read_valid;
    logic bridge_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] bridge_read_addr;
    logic [TAG_WIDTH-1:0] bridge_read_tag;
    logic bridge_read_response_valid;
    logic bridge_read_response_ready;
    logic signed [15:0] bridge_read_response_code;
    logic signed [15:0] bridge_read_response_residual;
    logic bridge_read_response_error;
    logic bridge_result_valid;
    logic bridge_result_ready;
    logic [PARAM_ADDR_WIDTH-1:0] bridge_result_addr;
    logic signed [15:0] bridge_result_code;
    logic signed [15:0] bridge_result_residual;
    logic bridge_write_code_enable;
    logic bridge_write_residual_enable;
    logic [LEARNER_BATCH_INDEX_WIDTH-1:0] bridge_computed_trace_sample_id;

    logic store_load_valid;
    logic store_load_ready;
    logic store_forward_read_valid;
    logic store_forward_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] store_forward_read_addr;
    logic [TAG_WIDTH-1:0] store_forward_read_tag;
    logic store_forward_response_valid;
    logic store_forward_response_ready;
    logic [TAG_WIDTH-1:0] store_forward_response_tag;
    logic signed [15:0] store_forward_response_code;
    logic store_forward_response_error;
    logic store_error;

    logic publisher_start_valid;
    logic publisher_start_ready;
    logic publisher_target_pair;
    logic [VERSION_WIDTH-1:0] publisher_target_version;
    logic publisher_done;
    logic publisher_pair;
    logic [VERSION_WIDTH-1:0] publisher_version;
    logic publisher_busy;
    logic publisher_error;
    logic cfg_matches_cache;
    logic cfg_fire;
    logic prime_fire;

    assign cfg_matches_cache = i_cfg_parameter_base == parameter_base_q &&
        i_cfg_feature_count == feature_count_q &&
        i_cfg_class_count == class_count_q;
    assign o_cfg_ready = cache_valid_q && publish_state_q == PUBLISH_IDLE &&
        !publisher_busy && cfg_matches_cache && bridge_cfg_ready;
    assign bridge_cfg_valid = i_cfg_valid && o_cfg_ready;
    assign cfg_fire = bridge_cfg_valid && bridge_cfg_ready;

    assign o_cache_prime_ready = publish_state_q == PUBLISH_IDLE &&
        !publisher_busy && !bridge_active && !bridge_commit_valid &&
        !i_parameter_load_valid && publisher_start_ready;
    assign prime_fire = i_cache_prime_valid && o_cache_prime_ready;
    assign publisher_start_valid = publish_state_q == PUBLISH_IDLE &&
        (prime_fire || (bridge_commit_valid && !i_cache_prime_valid));
    assign publisher_target_pair = prime_fire ? i_initial_pair :
        !active_pair_q;
    assign publisher_target_version = prime_fire ? i_initial_version :
        active_version_q + 1'b1;
    assign bridge_commit_ready = publish_state_q == PUBLISH_COMMIT &&
        publisher_done;

    assign store_load_valid = i_parameter_load_valid &&
        publish_state_q == PUBLISH_IDLE && !publisher_busy && !bridge_active;
    assign o_parameter_load_ready = store_load_ready &&
        publish_state_q == PUBLISH_IDLE && !publisher_busy && !bridge_active;

    assign o_active = bridge_active;
    assign o_release_ready = bridge_release_ready;
    assign o_inference_hold = bridge_inference_hold || publisher_busy ||
        publish_state_q != PUBLISH_IDLE || bridge_commit_valid;
    assign o_inference_ready = cache_valid_q && !o_inference_hold &&
        !o_sticky_error;
    assign o_active_pair = active_pair_q;
    assign o_active_version = active_version_q;
    assign o_commit_sequence = bridge_commit_sequence;
    assign o_parameter_write_fire = bridge_result_valid &&
        bridge_result_ready && bridge_write_code_enable;
    assign o_cache_publish_busy = publisher_busy ||
        publish_state_q != PUBLISH_IDLE;
    assign o_cache_valid = cache_valid_q;
    assign o_busy = bridge_window_busy || publisher_busy ||
        publish_state_q != PUBLISH_IDLE;

    unified_scnn_tp_classifier_final_core_learning_bridge #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .LANES(LANES), .ENGINES(ENGINES),
        .VERSION_WIDTH(VERSION_WIDTH), .FRAME_WIDTH(FRAME_WIDTH),
        .MAX_GRADIENT_ELEMENTS(PARAM_DEPTH),
        .BATCH_INDEX_WIDTH(LEARNER_BATCH_INDEX_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .TIMESTEP_INDEX_WIDTH(TIMESTEP_INDEX_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .FEATURE_INDEX_WIDTH(FEATURE_INDEX_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .WAVE_CLASS_INDEX_WIDTH(CLASS_INDEX_WIDTH),
        .WAVE_CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .ENABLE_COMPUTED_TRACE_CAPTURE(ENABLE_COMPUTED_TRACE_CAPTURE)
    ) u_learning_bridge (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_cfg_valid(bridge_cfg_valid), .o_cfg_ready(bridge_cfg_ready),
        .i_cfg_profile_id(i_cfg_profile_id),
        .i_cfg_tensor_id(i_cfg_tensor_id),
        .i_cfg_parameter_base(i_cfg_parameter_base),
        .i_cfg_batch_count(i_cfg_batch_count),
        .i_cfg_timestep_base(i_cfg_timestep_base),
        .i_cfg_timestep_count(i_cfg_timestep_count),
        .i_cfg_trace_state_reset(i_cfg_trace_state_reset),
        .i_cfg_feature_count(i_cfg_feature_count),
        .i_cfg_class_count(i_cfg_class_count),
        .i_cfg_trace_leak_valid(i_cfg_trace_leak_valid),
        .i_cfg_trace_leak_q8_8(i_cfg_trace_leak_q8_8),
        .i_cfg_computed_trace_capture_enable(
            i_cfg_computed_trace_capture_enable),
        .i_cfg_vote_increment(i_cfg_vote_increment),
        .i_cfg_vote_threshold(i_cfg_vote_threshold),
        .i_cfg_code_min(i_cfg_code_min), .i_cfg_code_max(i_cfg_code_max),
        .o_cfg_error(bridge_cfg_error), .o_active(bridge_active),
        .i_release(i_release), .o_release_ready(bridge_release_ready),
        .i_learning_enable(i_learning_enable),
        .i_feature_trace_wave_valid(i_feature_trace_wave_valid),
        .o_feature_trace_wave_ready(o_feature_trace_wave_ready),
        .i_feature_trace_wave_mask(i_feature_trace_wave_mask),
        .i_feature_trace_wave_base(i_feature_trace_wave_base),
        .i_feature_trace_wave_q8_8_flat(i_feature_trace_wave_q8_8_flat),
        .i_feature_trace_wave_version(i_feature_trace_wave_version),
        .i_feature_trace_wave_frame_index(i_feature_trace_wave_frame_index),
        .i_feature_trace_wave_first(i_feature_trace_wave_first),
        .i_feature_trace_wave_last(i_feature_trace_wave_last),
        .i_feature_trace_wave_sample_reset(i_feature_trace_wave_sample_reset),
        .i_feature_trace_wave_sample_last(i_feature_trace_wave_sample_last),
        .i_sample_label(i_sample_label),
        .i_logit_wave_valid(i_logit_wave_valid),
        .o_logit_wave_ready(o_logit_wave_ready),
        .i_logit_wave_mask(i_logit_wave_mask),
        .i_logit_wave_base(i_logit_wave_base),
        .i_logit_wave_count(i_logit_wave_count),
        .i_logit_wave_q8_8_flat(i_logit_wave_q8_8_flat),
        .i_logit_wave_version(i_logit_wave_version),
        .i_logit_wave_frame_index(i_logit_wave_frame_index),
        .i_logit_wave_frame_last(i_logit_wave_frame_last),
        .i_logit_wave_sample_last(i_logit_wave_sample_last),
        .o_window_commit_valid(bridge_commit_valid),
        .i_window_commit_ready(bridge_commit_ready),
        .o_window_commit_profile_id(bridge_commit_profile),
        .o_window_commit_sequence(bridge_commit_sequence),
        .o_timestep_learning_done(o_timestep_learning_done),
        .o_batch_learning_done(o_batch_learning_done),
        .o_completed_timestep_count(),
        .o_inference_hold(bridge_inference_hold),
        .o_window_busy(bridge_window_busy),
        .o_read_valid(bridge_read_valid), .i_read_ready(bridge_read_ready),
        .o_read_profile_id(), .o_read_tensor_id(),
        .o_read_parameter_addr(bridge_read_addr),
        .o_read_tag(bridge_read_tag),
        .i_read_response_valid(bridge_read_response_valid),
        .o_read_response_ready(bridge_read_response_ready),
        .i_read_response_code(bridge_read_response_code),
        .i_read_response_residual(bridge_read_response_residual),
        .i_read_response_error(bridge_read_response_error),
        .o_result_valid(bridge_result_valid),
        .i_result_ready(bridge_result_ready),
        .o_result_profile_id(), .o_result_tensor_id(),
        .o_result_parameter_addr(bridge_result_addr), .o_result_tag(),
        .o_result_code(bridge_result_code),
        .o_result_residual(bridge_result_residual),
        .o_write_code_enable(bridge_write_code_enable),
        .o_write_residual_enable(bridge_write_residual_enable),
        .o_write_code_fire(), .o_write_residual_fire(),
        .o_dirty(), .o_saturated(), .o_capture_complete(),
        .o_observation_error(bridge_capture_error),
        .o_context_error(bridge_context_error),
        .o_gradient_overflow(bridge_gradient_overflow),
        .o_parameter_error(bridge_parameter_error),
        .o_protocol_error(bridge_protocol_error),
        .o_computed_trace_valid(o_computed_trace_valid),
        .i_computed_trace_ready(i_computed_trace_ready),
        .o_computed_trace_sample_id(bridge_computed_trace_sample_id),
        .o_computed_trace_index(o_computed_trace_index),
        .o_computed_trace_timestep(o_computed_trace_timestep),
        .o_computed_trace_frame_index(o_computed_trace_frame_index),
        .o_computed_trace_version(o_computed_trace_version),
        .o_computed_trace_q8_8(o_computed_trace_q8_8)
    );

    // Computed-trace capture reports the original sample coordinate, never the
    // flattened BxT learner coordinate. Keep the public graph-facing width
    // independent from the wider learner index used by the TP reducer.
    assign o_computed_trace_sample_id =
        bridge_computed_trace_sample_id[BATCH_INDEX_WIDTH-1:0];

    unified_tp_shared_forward_rmw_store #(
        .DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) u_parameter_store (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_load_valid(store_load_valid), .o_load_ready(store_load_ready),
        .i_load_addr(i_parameter_load_addr),
        .i_load_code(i_parameter_load_code),
        .i_load_residual(i_parameter_load_residual),
        .i_learning_read_valid(bridge_read_valid),
        .o_learning_read_ready(bridge_read_ready),
        .i_learning_read_addr(bridge_read_addr),
        .i_learning_read_tag(bridge_read_tag),
        .o_learning_response_valid(bridge_read_response_valid),
        .i_learning_response_ready(bridge_read_response_ready),
        .o_learning_response_tag(),
        .o_learning_response_code(bridge_read_response_code),
        .o_learning_response_residual(bridge_read_response_residual),
        .o_learning_response_error(bridge_read_response_error),
        .i_learning_write_valid(bridge_result_valid),
        .o_learning_write_ready(bridge_result_ready),
        .i_learning_write_addr(bridge_result_addr),
        .i_learning_write_code(bridge_result_code),
        .i_learning_write_residual(bridge_result_residual),
        .i_learning_write_code_enable(bridge_write_code_enable),
        .i_learning_write_residual_enable(bridge_write_residual_enable),
        .i_forward_read_valid(store_forward_read_valid),
        .o_forward_read_ready(store_forward_read_ready),
        .i_forward_read_addr(store_forward_read_addr),
        .i_forward_read_tag(store_forward_read_tag),
        .o_forward_response_valid(store_forward_response_valid),
        .i_forward_response_ready(store_forward_response_ready),
        .o_forward_response_tag(store_forward_response_tag),
        .o_forward_response_code(store_forward_response_code),
        .o_forward_response_error(store_forward_response_error),
        .o_protocol_error(store_error)
    );

    unified_scnn_classifier_cache_refill_publisher_v1 #(
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .PARAM_DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH), .VERSION_WIDTH(VERSION_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(ARRAY_WEIGHT_ADDR_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_cache_publisher (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(publisher_start_valid),
        .o_start_ready(publisher_start_ready),
        .i_target_pair(publisher_target_pair),
        .i_parameter_base(prime_fire ? i_cfg_parameter_base :
            parameter_base_q),
        .i_feature_count(prime_fire ? i_cfg_feature_count : feature_count_q),
        .i_class_count(prime_fire ? i_cfg_class_count : class_count_q),
        .i_version(publisher_target_version), .i_array_idle(i_array_idle),
        .o_forward_read_valid(store_forward_read_valid),
        .i_forward_read_ready(store_forward_read_ready),
        .o_forward_read_addr(store_forward_read_addr),
        .o_forward_read_tag(store_forward_read_tag),
        .i_forward_response_valid(store_forward_response_valid),
        .o_forward_response_ready(store_forward_response_ready),
        .i_forward_response_tag(store_forward_response_tag),
        .i_forward_response_code(store_forward_response_code),
        .i_forward_response_error(store_forward_response_error),
        .o_weight_fill_valid(o_weight_fill_valid),
        .i_weight_fill_ready(i_weight_fill_ready),
        .o_weight_fill_tile_flat(o_weight_fill_tile_flat),
        .o_weight_fill_bank(o_weight_fill_bank),
        .o_weight_fill_addr_flat(o_weight_fill_addr_flat),
        .o_weight_fill_data_flat(o_weight_fill_data_flat),
        .o_done(publisher_done), .o_published_pair(publisher_pair),
        .o_published_version(publisher_version),
        .o_busy(publisher_busy), .o_sticky_error(publisher_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            publish_state_q <= PUBLISH_IDLE;
            cache_valid_q <= 1'b0;
            active_pair_q <= 1'b0;
            active_version_q <= '0;
            parameter_base_q <= '0;
            feature_count_q <= '0;
            class_count_q <= '0;
            o_cache_prime_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            publish_state_q <= PUBLISH_IDLE;
            cache_valid_q <= 1'b0;
            o_cache_prime_done <= 1'b0;
        end else begin
            o_cache_prime_done <= 1'b0;
            if (i_clear_error && publish_state_q == PUBLISH_IDLE &&
                    !publisher_busy)
                o_sticky_error <= 1'b0;
            if (bridge_cfg_error || bridge_capture_error ||
                bridge_context_error || bridge_gradient_overflow ||
                bridge_parameter_error || bridge_protocol_error ||
                publisher_error || store_error)
                o_sticky_error <= 1'b1;
            if (i_cfg_valid && cache_valid_q && !cfg_matches_cache)
                o_sticky_error <= 1'b1;

            if (cfg_fire) begin
                parameter_base_q <= i_cfg_parameter_base;
                feature_count_q <= i_cfg_feature_count;
                class_count_q <= i_cfg_class_count;
            end

            case (publish_state_q)
                PUBLISH_IDLE: begin
                    if (publisher_start_valid && publisher_start_ready) begin
                        if (prime_fire) begin
                            parameter_base_q <= i_cfg_parameter_base;
                            feature_count_q <= i_cfg_feature_count;
                            class_count_q <= i_cfg_class_count;
                            publish_state_q <= PUBLISH_PRIME;
                        end else begin
                            publish_state_q <= PUBLISH_COMMIT;
                        end
                    end
                end
                PUBLISH_PRIME: begin
                    if (publisher_done) begin
                        cache_valid_q <= 1'b1;
                        active_pair_q <= publisher_pair;
                        active_version_q <= publisher_version;
                        o_cache_prime_done <= 1'b1;
                        publish_state_q <= PUBLISH_IDLE;
                    end
                end
                PUBLISH_COMMIT: begin
                    if (publisher_done) begin
                        active_pair_q <= publisher_pair;
                        active_version_q <= publisher_version;
                        publish_state_q <= PUBLISH_IDLE;
                    end
                end
                default: begin
                    publish_state_q <= PUBLISH_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (MAX_CLASSES > 16 || LANES != 8)
            $error("SCNN cache-coherent learning client parameter mismatch");
    end
endmodule

`default_nettype wire
