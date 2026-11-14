`timescale 1ns/1ps
`default_nettype none

// Real-array SCNN graph plus cache-coherent TP classifier adaptation. Conv and
// classifier execute on the single shared 16xH40/P4 substrate. The learning
// client owns only the persistent parameter store and the cache-refill control
// plane; it never instantiates another MAC, LIF, state, or trace datapath.
module unified_scnn_h40_array16_graph_online_learning_system_v1 #(
    parameter int SCNN_WEIGHT_FRACTIONAL_BITS = 8,
    parameter bit ACCUMULATE_SAMPLE_LOGITS = 1'b0,
    parameter int WEIGHT_DEPTH = 1156,
    parameter int WEIGHT_ADDR_WIDTH = $clog2(WEIGHT_DEPTH),
    parameter int MEMORY_DEPTH = 1024,
    parameter int MEMORY_ADDR_WIDTH = $clog2(MEMORY_DEPTH),
    parameter int WORK_ADDR_WIDTH = $clog2(MEMORY_DEPTH / 2),
    parameter int FEATURE_WORD_DEPTH = 8192,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_WORD_DEPTH),
    parameter int MAX_FEATURE_CHANNELS = 512,
    parameter int CLASSIFIER_FEATURE_SPATIAL = 4,
    parameter int MAX_CLASSES = 11,
    // MAX_BATCH is the flattened learner capacity (B*T). Capture storage and
    // graph sample IDs use CLASSIFIER_CAPTURE_MAX_BATCH instead.
    parameter int MAX_BATCH = 10,
    parameter int CLASSIFIER_CAPTURE_MAX_BATCH = MAX_BATCH,
    parameter int MAX_TIMESTEPS = 20,
    parameter int MAX_FEATURES = MAX_FEATURE_CHANNELS * CLASSIFIER_FEATURE_SPATIAL,
    parameter int PARAM_DEPTH = MAX_FEATURES * MAX_CLASSES,
    parameter int PARAM_ADDR_WIDTH = $clog2(PARAM_DEPTH),
    parameter int CHANNEL_WIDTH = MAX_FEATURE_CHANNELS <= 2 ? 1 :
        $clog2(MAX_FEATURE_CHANNELS),
    parameter int CHANNEL_COUNT_WIDTH = $clog2(MAX_FEATURE_CHANNELS + 1),
    parameter int FEATURE_WIDTH = MAX_FEATURES <= 2 ? 1 : $clog2(MAX_FEATURES),
    parameter int FEATURE_COUNT_WIDTH = $clog2(MAX_FEATURES + 1),
    parameter int CLASS_WIDTH = MAX_CLASSES <= 2 ? 1 : $clog2(MAX_CLASSES),
    parameter int CLASS_COUNT_WIDTH = $clog2(MAX_CLASSES + 1),
    parameter int BATCH_COUNT_WIDTH = $clog2(MAX_BATCH + 1),
    parameter int SAMPLE_INDEX_WIDTH = CLASSIFIER_CAPTURE_MAX_BATCH <= 2 ? 1 :
        $clog2(CLASSIFIER_CAPTURE_MAX_BATCH),
    parameter int LEARNER_BATCH_INDEX_WIDTH = MAX_BATCH <= 2 ? 1 :
        $clog2(MAX_BATCH),
    parameter int TIMESTEP_INDEX_WIDTH = MAX_TIMESTEPS <= 2 ? 1 : $clog2(MAX_TIMESTEPS),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS + 1)
) (
    input  wire logic                           i_clk,
    input  wire logic                           i_rst_n,
    input  wire logic                           i_abort,
    input  wire logic                           i_clear_error,

    input  wire logic                           i_parameter_load_valid,
    output logic                                o_parameter_load_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]    i_parameter_load_addr,
    input  wire logic signed [15:0]             i_parameter_load_code,
    input  wire logic signed [15:0]             i_parameter_load_residual,
    input  wire logic                           i_classifier_cache_prime_valid,
    output logic                                o_classifier_cache_prime_ready,
    input  wire logic                           i_classifier_initial_pair,
    input  wire logic [15:0]                    i_classifier_initial_version,
    output logic                                o_classifier_cache_prime_done,

    input  wire logic                           i_learning_cfg_valid,
    output logic                                o_learning_cfg_ready,
    input  wire logic [7:0]                     i_learning_profile_id,
    input  wire logic [7:0]                     i_learning_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]    i_learning_parameter_base,
    input  wire logic [BATCH_COUNT_WIDTH-1:0]   i_learning_batch_count,
    input  wire logic [TIMESTEP_INDEX_WIDTH-1:0] i_learning_timestep_base,
    input  wire logic [TIMESTEP_COUNT_WIDTH-1:0] i_learning_timestep_count,
    input  wire logic                           i_learning_trace_state_reset,
    input  wire logic [FEATURE_COUNT_WIDTH-1:0] i_learning_feature_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]   i_learning_class_count,
    input  wire logic                           i_learning_trace_leak_valid,
    input  wire logic signed [15:0]             i_learning_trace_leak_q8_8,
    input  wire logic                           i_learning_computed_trace_capture_enable,
    input  wire logic [15:0]                    i_learning_vote_increment,
    input  wire logic [15:0]                    i_learning_vote_threshold,
    input  wire logic signed [15:0]             i_learning_code_min,
    input  wire logic signed [15:0]             i_learning_code_max,
    input  wire logic                           i_learning_enable,
    input  wire logic                           i_learning_release,
    output logic                                o_learning_release_ready,

    input  wire logic                           i_scnn_start_valid,
    output logic                                o_scnn_start_ready,
    input  wire logic [15:0]                    i_conv_in_width,
    input  wire logic [15:0]                    i_conv_in_height,
    input  wire logic [15:0]                    i_conv_in_channels,
    input  wire logic [15:0]                    i_conv_out_width,
    input  wire logic [15:0]                    i_conv_out_height,
    input  wire logic [15:0]                    i_conv_out_channels,
    input  wire logic [7:0]                     i_conv_stride,
    input  wire logic [7:0]                     i_conv_padding,
    input  wire logic [1:0]                     i_conv_pool_mode,
    input  wire logic                           i_conv_active_pair,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_conv_work_base_addr,
    input  wire logic [15:0]                    i_conv_leak,
    input  wire logic [31:0]                    i_conv_threshold,
    input  wire logic                           i_conv_reset_to_zero,
    input  wire logic [15:0]                    i_conv_trace_decay,
    input  wire logic [19:0]                    i_conv_layer_tag,
    input  wire logic [CHANNEL_WIDTH-1:0]       i_classifier_feature_channel_base,
    input  wire logic [CHANNEL_COUNT_WIDTH-1:0] i_classifier_feature_channel_count,
    input  wire logic [CLASS_COUNT_WIDTH-1:0]   i_classifier_class_count,
    input  wire logic [WORK_ADDR_WIDTH-1:0]     i_classifier_work_base_addr,
    input  wire logic [15:0]                    i_classifier_trace_decay,
    input  wire logic [19:0]                    i_classifier_layer_tag,
    input  wire logic [7:0]                     i_frame_index,
    input  wire logic                           i_sample_reset,
    input  wire logic                           i_sample_last,
    input  wire logic [CLASS_WIDTH-1:0]         i_sample_label,
    input  wire logic                           i_conv8_observation_target,
    input  wire logic [SAMPLE_INDEX_WIDTH-1:0]  i_sample_index,
    input  wire logic [TIMESTEP_INDEX_WIDTH-1:0] i_timestep_index,
    output logic                                o_scnn_done,

    output logic                                o_context_load_valid,
    input  wire logic                           i_context_load_ready,
    output logic                                o_context_load_weights,
    output logic [9:0]                          o_context_output_channel_base,
    output logic [9:0]                          o_context_output_channel_count,
    output logic [31:0]                         o_context_spatial_base,
    output logic [5:0]                          o_context_spatial_count,
    output logic                                o_context_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_context_work_base_addr,
    output logic                                o_context_release_valid,
    input  wire logic                           i_context_release_ready,
    output logic [9:0]                          o_release_output_channel_base,
    output logic [9:0]                          o_release_output_channel_count,
    output logic [31:0]                         o_release_spatial_base,
    output logic [5:0]                          o_release_spatial_count,
    output logic                                o_release_active_pair,
    output logic [WORK_ADDR_WIDTH-1:0]          o_release_work_base_addr,

    output logic                                o_feature_request_valid,
    input  wire logic                           i_feature_request_ready,
    output logic [3:0]                          o_feature_term_valid,
    output logic [39:0]                         o_feature_input_channel_flat,
    output logic [7:0]                          o_feature_kernel_row_flat,
    output logic [7:0]                          o_feature_kernel_col_flat,
    output logic [31:0]                         o_feature_spatial_base,
    output logic [5:0]                          o_feature_spatial_count,
    output logic [15:0]                         o_feature_output_row,
    output logic [15:0]                         o_feature_output_col,
    output logic [3:0]                          o_feature_value_bit,
    input  wire logic                           i_feature_response_valid,
    output logic                                o_feature_response_ready,
    input  wire logic [39:0]                    i_feature_mask0,
    input  wire logic [39:0]                    i_feature_mask1,
    input  wire logic [39:0]                    i_feature_mask2,
    input  wire logic [39:0]                    i_feature_mask3,
    input  wire logic                           i_feature_service_error,
    input  wire logic                           i_feature_write_slot,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_base_addr,
    input  wire logic [15:0]                    i_feature_write_row_stride_words,
    input  wire logic [FEATURE_ADDR_WIDTH-1:0]  i_feature_write_channel_stride_words,
    output logic                                o_feature_update_valid,
    input  wire logic                           i_feature_update_ready,
    output logic                                o_feature_update_slot,
    output logic [FEATURE_ADDR_WIDTH-1:0]       o_feature_update_addr,
    output logic [63:0]                         o_feature_update_data,
    output logic [63:0]                         o_feature_update_mask,
    input  wire logic                           i_feature_update_done,

    input  wire logic [3:0]                     i_external_weight_fill_valid,
    output logic [3:0]                          o_external_weight_fill_ready,
    input  wire logic [7:0]                     i_external_weight_fill_tile_flat,
    input  wire logic [3:0]                     i_external_weight_fill_bank,
    input  wire logic [4*WEIGHT_ADDR_WIDTH-1:0] i_external_weight_fill_addr_flat,
    input  wire logic [799:0]                   i_external_weight_fill_data_flat,
    input  wire logic [3:0]                     i_maintenance_tile,
    input  wire logic                           i_state_maintenance_write_valid,
    output logic                                o_state_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_write_addr,
    input  wire logic [319:0]                   i_state_maintenance_write_data,
    input  wire logic [39:0]                    i_state_maintenance_write_mask,
    input  wire logic                           i_state_maintenance_read_valid,
    output logic                                o_state_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_state_maintenance_read_addr,
    output logic                                o_state_maintenance_result_valid,
    input  wire logic                           i_state_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_state_maintenance_result_addr,
    output logic [319:0]                        o_state_maintenance_read_data,
    input  wire logic                           i_trace_maintenance_write_valid,
    output logic                                o_trace_maintenance_write_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_write_addr,
    input  wire logic [319:0]                   i_trace_maintenance_write_data,
    input  wire logic [39:0]                    i_trace_maintenance_write_mask,
    input  wire logic                           i_trace_maintenance_read_valid,
    output logic                                o_trace_maintenance_read_ready,
    input  wire logic [MEMORY_ADDR_WIDTH-1:0]   i_trace_maintenance_read_addr,
    output logic                                o_trace_maintenance_result_valid,
    input  wire logic                           i_trace_maintenance_result_ready,
    output logic [MEMORY_ADDR_WIDTH-1:0]        o_trace_maintenance_result_addr,
    output logic [319:0]                        o_trace_maintenance_read_data,

    output logic                                o_feature_wave_fire,
    output logic [7:0]                          o_feature_wave_mask,
    output logic [FEATURE_WIDTH-1:0]            o_feature_wave_base,
    output logic [127:0]                        o_feature_wave_q8_8_flat,
    output logic                                o_classifier_computed_trace_valid,
    input  wire logic                           i_classifier_computed_trace_ready,
    output logic [SAMPLE_INDEX_WIDTH-1:0]       o_classifier_computed_trace_sample,
    output logic [FEATURE_WIDTH-1:0]            o_classifier_computed_trace_index,
    output logic [TIMESTEP_INDEX_WIDTH-1:0]     o_classifier_computed_trace_timestep,
    output logic [7:0]                          o_classifier_computed_trace_frame_index,
    output logic [15:0]                         o_classifier_computed_trace_version,
    output logic signed [15:0]                  o_classifier_computed_trace_q8_8,
    output logic                                o_conv8_input_valid,
    input  wire logic                           i_conv8_input_ready,
    output logic                                o_conv8_input_target,
    output logic [SAMPLE_INDEX_WIDTH-1:0]       o_conv8_input_sample,
    output logic [TIMESTEP_INDEX_WIDTH-1:0]     o_conv8_input_timestep,
    output logic [15:0]                         o_conv8_input_version,
    output logic [19:0]                         o_conv8_input_layer_tag,
    output logic [9:0]                          o_conv8_input_channel,
    output logic [31:0]                         o_conv8_input_spatial,
    output logic signed [15:0]                  o_conv8_input_q8_8,
    output logic                                o_conv8_input_last,
    output logic                                o_conv8_surrogate_valid,
    input  wire logic                           i_conv8_surrogate_ready,
    output logic                                o_conv8_surrogate_target,
    output logic [SAMPLE_INDEX_WIDTH-1:0]       o_conv8_surrogate_sample,
    output logic [TIMESTEP_INDEX_WIDTH-1:0]     o_conv8_surrogate_timestep,
    output logic [15:0]                         o_conv8_surrogate_version,
    output logic [19:0]                         o_conv8_surrogate_layer_tag,
    output logic [9:0]                          o_conv8_surrogate_channel,
    output logic [31:0]                         o_conv8_surrogate_spatial,
    output logic signed [15:0]                  o_conv8_membrane_pre_q8_8,
    output logic signed [15:0]                  o_conv8_surrogate_q8_8,
    output logic                                o_logit_wave_fire,
    output logic [11:0]                         o_logit_wave_mask,
    output logic [CLASS_WIDTH-1:0]              o_logit_wave_base,
    output logic [CLASS_COUNT_WIDTH-1:0]        o_logit_wave_count,
    output logic [191:0]                        o_logit_wave_q8_8_flat,
    output logic                                o_classifier_active_pair,
    output logic [15:0]                         o_classifier_active_version,
    output logic                                o_classifier_cache_valid,
    output logic                                o_parameter_write_fire,
    output logic [31:0]                         o_learning_commit_sequence,
    output logic [1:0]                          o_active_owner,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic [15:0]                         o_tile_idle,
    output logic [15:0]                         o_tile_sticky_error,
    output logic                                o_sticky_error
);
    typedef enum logic [2:0] {
        TX_IDLE, TX_PRESCAN, TX_GRAPH, TX_LEARN
    } tx_state_t;
    tx_state_t tx_state_q;
    logic transaction_learning_q;
    logic transaction_observation_target_q;
    logic [SAMPLE_INDEX_WIDTH-1:0] transaction_sample_q;
    logic [TIMESTEP_INDEX_WIDTH-1:0] transaction_timestep_q;
    logic core_start_valid;
    logic core_start_ready;
    logic core_started_q;
    logic core_done;
    logic core_busy;
    logic core_idle;
    logic core_error;
    logic learning_active;
    logic learning_inference_ready;
    logic learning_inference_hold;
    logic learning_batch_done;
    logic learning_busy;
    logic learning_error;
    logic cache_publish_busy;
    logic classifier_cache_valid;
    logic classifier_pair;
    logic [15:0] classifier_version;
    logic feature_wave_valid;
    logic feature_wave_ready;
    logic [7:0] feature_wave_mask;
    logic [FEATURE_WIDTH-1:0] feature_wave_base;
    logic [127:0] feature_wave_flat;
    logic [15:0] feature_wave_version;
    logic [7:0] feature_wave_frame;
    logic feature_wave_first;
    logic feature_wave_last;
    logic feature_wave_sample_reset;
    logic feature_wave_sample_last;
    logic [CLASS_WIDTH-1:0] feature_sample_label;
    logic logit_wave_valid;
    logic logit_wave_ready;
    logic [11:0] logit_wave_mask;
    logic [CLASS_WIDTH-1:0] logit_wave_base;
    logic [CLASS_COUNT_WIDTH-1:0] logit_wave_count;
    logic [191:0] logit_wave_flat;
    logic [15:0] logit_wave_version;
    logic [7:0] logit_wave_frame;
    logic logit_wave_frame_last;
    logic logit_wave_sample_last;
    logic [3:0] learning_fill_valid;
    logic [3:0] learning_fill_ready;
    logic [7:0] learning_fill_tile_flat;
    logic [3:0] learning_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0] learning_fill_addr_flat;
    logic [799:0] learning_fill_data_flat;
    logic [3:0] core_fill_valid;
    logic [3:0] core_fill_ready;
    logic [7:0] core_fill_tile_flat;
    logic [3:0] core_fill_bank;
    logic [4*WEIGHT_ADDR_WIDTH-1:0] core_fill_addr_flat;
    logic [799:0] core_fill_data_flat;
    logic maintenance_lease_q;
    logic maintenance_request_present;
    logic core_maintenance_start_valid;
    logic core_maintenance_start_ready;
    logic core_maintenance_release_valid;
    logic core_maintenance_release_ready;
    logic parameter_publish_safe;
    logic core_context_load_valid;
    logic core_context_load_weights;
    logic core_context_active_pair;
    logic classifier_context_active;
    logic [3:0] external_fill_conflict;
    logic protocol_error_q;
    logic prescan_start_valid;
    logic prescan_start_ready;
    logic prescan_started_q;
    logic prescan_done;
    logic prescan_busy;
    logic prescan_error;
    logic prescan_feature_request_valid;
    logic prescan_feature_request_ready;
    logic [3:0] prescan_feature_term_valid;
    logic [39:0] prescan_feature_input_channel_flat;
    logic [7:0] prescan_feature_kernel_row_flat;
    logic [7:0] prescan_feature_kernel_col_flat;
    logic [31:0] prescan_feature_spatial_base;
    logic [5:0] prescan_feature_spatial_count;
    logic [15:0] prescan_feature_output_row;
    logic [15:0] prescan_feature_output_col;
    logic prescan_feature_response_valid;
    logic prescan_feature_response_ready;
    logic core_feature_request_valid;
    logic core_feature_request_ready;
    logic [3:0] core_feature_term_valid;
    logic [39:0] core_feature_input_channel_flat;
    logic [7:0] core_feature_kernel_row_flat;
    logic [7:0] core_feature_kernel_col_flat;
    logic [31:0] core_feature_spatial_base;
    logic [5:0] core_feature_spatial_count;
    logic [15:0] core_feature_output_row;
    logic [15:0] core_feature_output_col;
    logic [3:0] core_feature_value_bit;
    logic core_feature_response_valid;
    logic core_feature_response_ready;

    assign o_scnn_start_ready = tx_state_q == TX_IDLE && core_start_ready &&
        (!i_learning_enable || prescan_start_ready) &&
        learning_inference_ready && (!i_learning_enable || learning_active) &&
        i_conv_active_pair != classifier_pair && !protocol_error_q;
    assign prescan_start_valid = tx_state_q == TX_PRESCAN &&
        !prescan_started_q;
    assign core_start_valid = tx_state_q == TX_GRAPH && !core_started_q;
    assign classifier_context_active = classifier_cache_valid &&
        core_context_active_pair == classifier_pair;
    assign o_context_load_valid = core_context_load_valid;
    assign o_context_load_weights = core_context_load_weights &&
        !classifier_context_active;
    assign o_context_active_pair = core_context_active_pair;
    assign o_feature_wave_fire = feature_wave_valid && feature_wave_ready;
    assign o_feature_wave_mask = feature_wave_mask;
    assign o_feature_wave_base = feature_wave_base;
    assign o_feature_wave_q8_8_flat = feature_wave_flat;
    assign o_logit_wave_fire = logit_wave_valid && logit_wave_ready;
    assign o_logit_wave_mask = logit_wave_mask;
    assign o_logit_wave_base = logit_wave_base;
    assign o_logit_wave_count = logit_wave_count;
    assign o_logit_wave_q8_8_flat = logit_wave_flat;
    assign o_classifier_active_pair = classifier_pair;
    assign o_classifier_active_version = classifier_version;
    assign o_classifier_cache_valid = classifier_cache_valid;
    assign o_busy = tx_state_q != TX_IDLE || core_busy || prescan_busy ||
        cache_publish_busy || learning_inference_hold;
    assign o_idle = tx_state_q == TX_IDLE && core_idle &&
        !prescan_busy && !cache_publish_busy && !learning_inference_hold;
    assign o_sticky_error = core_error || prescan_error || learning_error ||
        protocol_error_q;
    assign maintenance_request_present = (|core_fill_valid) ||
        i_state_maintenance_write_valid || i_state_maintenance_read_valid ||
        o_state_maintenance_result_valid ||
        i_trace_maintenance_write_valid || i_trace_maintenance_read_valid ||
        o_trace_maintenance_result_valid || cache_publish_busy;
    assign core_maintenance_start_valid = !maintenance_lease_q &&
        maintenance_request_present;
    assign core_maintenance_release_valid = maintenance_lease_q &&
        !maintenance_request_present;
    // core_idle includes owner_idleness and therefore deasserts while the
    // maintenance lease is held.  The lease itself excludes all compute
    // owners, so cache publication remains safe for its whole lifetime.
    assign parameter_publish_safe = core_idle || maintenance_lease_q;

    always_comb begin
        o_feature_request_valid = core_feature_request_valid;
        o_feature_term_valid = core_feature_term_valid;
        o_feature_input_channel_flat = core_feature_input_channel_flat;
        o_feature_kernel_row_flat = core_feature_kernel_row_flat;
        o_feature_kernel_col_flat = core_feature_kernel_col_flat;
        o_feature_spatial_base = core_feature_spatial_base;
        o_feature_spatial_count = core_feature_spatial_count;
        o_feature_output_row = core_feature_output_row;
        o_feature_output_col = core_feature_output_col;
        o_feature_value_bit = core_feature_value_bit;
        core_feature_request_ready = i_feature_request_ready;
        prescan_feature_request_ready = 1'b0;
        core_feature_response_valid = i_feature_response_valid;
        prescan_feature_response_valid = 1'b0;
        o_feature_response_ready = core_feature_response_ready;
        if (tx_state_q == TX_PRESCAN) begin
            o_feature_request_valid = prescan_feature_request_valid;
            o_feature_term_valid = prescan_feature_term_valid;
            o_feature_input_channel_flat =
                prescan_feature_input_channel_flat;
            o_feature_kernel_row_flat = prescan_feature_kernel_row_flat;
            o_feature_kernel_col_flat = prescan_feature_kernel_col_flat;
            o_feature_spatial_base = prescan_feature_spatial_base;
            o_feature_spatial_count = prescan_feature_spatial_count;
            o_feature_output_row = prescan_feature_output_row;
            o_feature_output_col = prescan_feature_output_col;
            o_feature_value_bit = 4'd0;
            core_feature_request_ready = 1'b0;
            prescan_feature_request_ready = i_feature_request_ready;
            core_feature_response_valid = 1'b0;
            prescan_feature_response_valid = i_feature_response_valid;
            o_feature_response_ready = prescan_feature_response_ready;
        end else if (tx_state_q != TX_GRAPH) begin
            o_feature_request_valid = 1'b0;
            o_feature_value_bit = 4'd0;
            core_feature_request_ready = 1'b0;
            core_feature_response_valid = 1'b0;
            o_feature_response_ready = 1'b0;
        end
    end

    unified_scnn_h40_tp_input_prescan_v1 #(
        .MAX_CHANNELS(MAX_FEATURE_CHANNELS),
        .SAMPLE_WIDTH(SAMPLE_INDEX_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_INDEX_WIDTH), .VERSION_WIDTH(16)
    ) u_conv8_input_prescan (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_start_valid(prescan_start_valid),
        .o_start_ready(prescan_start_ready), .i_width(i_conv_in_width),
        .i_height(i_conv_in_height), .i_channels(i_conv_in_channels),
        .i_padding(i_conv_padding),
        .i_target(transaction_observation_target_q),
        .i_sample(transaction_sample_q),
        .i_timestep(transaction_timestep_q), .i_version(classifier_version),
        .i_layer_tag(i_conv_layer_tag),
        .o_feature_request_valid(prescan_feature_request_valid),
        .i_feature_request_ready(prescan_feature_request_ready),
        .o_feature_term_valid(prescan_feature_term_valid),
        .o_feature_input_channel_flat(
            prescan_feature_input_channel_flat),
        .o_feature_kernel_row_flat(prescan_feature_kernel_row_flat),
        .o_feature_kernel_col_flat(prescan_feature_kernel_col_flat),
        .o_feature_spatial_base(prescan_feature_spatial_base),
        .o_feature_spatial_count(prescan_feature_spatial_count),
        .o_feature_output_row(prescan_feature_output_row),
        .o_feature_output_col(prescan_feature_output_col),
        .i_feature_response_valid(prescan_feature_response_valid),
        .o_feature_response_ready(prescan_feature_response_ready),
        .i_feature_mask0(i_feature_mask0),
        .i_feature_mask1(i_feature_mask1),
        .i_feature_mask2(i_feature_mask2),
        .i_feature_mask3(i_feature_mask3),
        .i_feature_service_error(i_feature_service_error),
        .o_observation_valid(o_conv8_input_valid),
        .i_observation_ready(i_conv8_input_ready),
        .o_observation_target(o_conv8_input_target),
        .o_observation_sample(o_conv8_input_sample),
        .o_observation_timestep(o_conv8_input_timestep),
        .o_observation_version(o_conv8_input_version),
        .o_observation_layer_tag(o_conv8_input_layer_tag),
        .o_observation_channel(o_conv8_input_channel),
        .o_observation_spatial(o_conv8_input_spatial),
        .o_observation_q8_8(o_conv8_input_q8_8),
        .o_observation_last(o_conv8_input_last), .o_done(prescan_done),
        .o_busy(prescan_busy), .o_sticky_error(prescan_error)
    );

    always_comb begin
        external_fill_conflict = '0;
        for (int region = 0; region < 4; region++) begin
            external_fill_conflict[region] = classifier_cache_valid &&
                i_external_weight_fill_valid[region] &&
                i_external_weight_fill_addr_flat[
                    region*WEIGHT_ADDR_WIDTH + WEIGHT_ADDR_WIDTH-1] ==
                    classifier_pair;
        end
        core_fill_valid = i_external_weight_fill_valid &
            ~external_fill_conflict;
        core_fill_tile_flat = i_external_weight_fill_tile_flat;
        core_fill_bank = i_external_weight_fill_bank;
        core_fill_addr_flat = i_external_weight_fill_addr_flat;
        core_fill_data_flat = i_external_weight_fill_data_flat;
        o_external_weight_fill_ready = core_fill_ready &
            ~external_fill_conflict;
        learning_fill_ready = '0;
        if (cache_publish_busy) begin
            core_fill_valid = learning_fill_valid;
            core_fill_tile_flat = learning_fill_tile_flat;
            core_fill_bank = learning_fill_bank;
            core_fill_addr_flat = learning_fill_addr_flat;
            core_fill_data_flat = learning_fill_data_flat;
            learning_fill_ready = core_fill_ready;
            o_external_weight_fill_ready = '0;
        end
    end

    unified_scnn_classifier_online_cache_learning_client_v1 #(
        .MAX_BATCH(MAX_BATCH), .MAX_TIMESTEPS(MAX_TIMESTEPS),
        .CLASSIFIER_CAPTURE_MAX_BATCH(CLASSIFIER_CAPTURE_MAX_BATCH),
        .MAX_FEATURES(MAX_FEATURES), .MAX_CLASSES(MAX_CLASSES),
        .PARAM_DEPTH(PARAM_DEPTH), .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .VERSION_WIDTH(16), .FRAME_WIDTH(8),
        .FEATURE_INDEX_WIDTH(FEATURE_WIDTH),
        .FEATURE_COUNT_WIDTH(FEATURE_COUNT_WIDTH),
        .CLASS_INDEX_WIDTH(CLASS_WIDTH),
        .CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH),
        .BATCH_COUNT_WIDTH(BATCH_COUNT_WIDTH),
        .BATCH_INDEX_WIDTH(SAMPLE_INDEX_WIDTH),
        .LEARNER_BATCH_INDEX_WIDTH(LEARNER_BATCH_INDEX_WIDTH),
        .TIMESTEP_INDEX_WIDTH(TIMESTEP_INDEX_WIDTH),
        .TIMESTEP_COUNT_WIDTH(TIMESTEP_COUNT_WIDTH),
        .ARRAY_WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .ENABLE_COMPUTED_TRACE_CAPTURE(1'b1)
    ) u_learning_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_parameter_load_valid(i_parameter_load_valid),
        .o_parameter_load_ready(o_parameter_load_ready),
        .i_parameter_load_addr(i_parameter_load_addr),
        .i_parameter_load_code(i_parameter_load_code),
        .i_parameter_load_residual(i_parameter_load_residual),
        .i_cache_prime_valid(i_classifier_cache_prime_valid),
        .o_cache_prime_ready(o_classifier_cache_prime_ready),
        .i_initial_pair(i_classifier_initial_pair),
        .i_initial_version(i_classifier_initial_version),
        .o_cache_prime_done(o_classifier_cache_prime_done),
        .i_cfg_valid(i_learning_cfg_valid),
        .o_cfg_ready(o_learning_cfg_ready),
        .i_cfg_profile_id(i_learning_profile_id),
        .i_cfg_tensor_id(i_learning_tensor_id),
        .i_cfg_parameter_base(i_learning_parameter_base),
        .i_cfg_batch_count(i_learning_batch_count),
        .i_cfg_timestep_base(i_learning_timestep_base),
        .i_cfg_timestep_count(i_learning_timestep_count),
        .i_cfg_trace_state_reset(i_learning_trace_state_reset),
        .i_cfg_feature_count(i_learning_feature_count),
        .i_cfg_class_count(i_learning_class_count),
        .i_cfg_trace_leak_valid(i_learning_trace_leak_valid),
        .i_cfg_trace_leak_q8_8(i_learning_trace_leak_q8_8),
        .i_cfg_computed_trace_capture_enable(
            i_learning_computed_trace_capture_enable),
        .i_cfg_vote_increment(i_learning_vote_increment),
        .i_cfg_vote_threshold(i_learning_vote_threshold),
        .i_cfg_code_min(i_learning_code_min),
        .i_cfg_code_max(i_learning_code_max),
        .o_active(learning_active), .i_release(i_learning_release),
        .o_release_ready(o_learning_release_ready),
        .i_learning_enable(i_learning_enable),
        .i_feature_trace_wave_valid(feature_wave_valid),
        .o_feature_trace_wave_ready(feature_wave_ready),
        .i_feature_trace_wave_mask(feature_wave_mask),
        .i_feature_trace_wave_base(feature_wave_base),
        .i_feature_trace_wave_q8_8_flat(feature_wave_flat),
        .i_feature_trace_wave_version(feature_wave_version),
        .i_feature_trace_wave_frame_index(feature_wave_frame),
        .i_feature_trace_wave_first(feature_wave_first),
        .i_feature_trace_wave_last(feature_wave_last),
        .i_feature_trace_wave_sample_reset(feature_wave_sample_reset),
        .i_feature_trace_wave_sample_last(feature_wave_sample_last),
        .i_sample_label(feature_sample_label),
        .i_logit_wave_valid(logit_wave_valid),
        .o_logit_wave_ready(logit_wave_ready),
        .i_logit_wave_mask(logit_wave_mask),
        .i_logit_wave_base(logit_wave_base),
        .i_logit_wave_count(logit_wave_count),
        .i_logit_wave_q8_8_flat(logit_wave_flat),
        .i_logit_wave_version(logit_wave_version),
        .i_logit_wave_frame_index(logit_wave_frame),
        .i_logit_wave_frame_last(logit_wave_frame_last),
        .i_logit_wave_sample_last(logit_wave_sample_last),
        .i_array_idle(parameter_publish_safe),
        .o_weight_fill_valid(learning_fill_valid),
        .i_weight_fill_ready(learning_fill_ready),
        .o_weight_fill_tile_flat(learning_fill_tile_flat),
        .o_weight_fill_bank(learning_fill_bank),
        .o_weight_fill_addr_flat(learning_fill_addr_flat),
        .o_weight_fill_data_flat(learning_fill_data_flat),
        .o_inference_ready(learning_inference_ready),
        .o_inference_hold(learning_inference_hold),
        .o_active_pair(classifier_pair),
        .o_active_version(classifier_version),
        .o_timestep_learning_done(),
        .o_batch_learning_done(learning_batch_done),
        .o_commit_sequence(o_learning_commit_sequence),
        .o_parameter_write_fire(o_parameter_write_fire),
        .o_cache_publish_busy(cache_publish_busy),
        .o_cache_valid(classifier_cache_valid),
        .o_busy(learning_busy), .o_sticky_error(learning_error),
        .o_computed_trace_valid(o_classifier_computed_trace_valid),
        .i_computed_trace_ready(i_classifier_computed_trace_ready),
        .o_computed_trace_sample_id(o_classifier_computed_trace_sample),
        .o_computed_trace_index(o_classifier_computed_trace_index),
        .o_computed_trace_timestep(o_classifier_computed_trace_timestep),
        .o_computed_trace_frame_index(
            o_classifier_computed_trace_frame_index),
        .o_computed_trace_version(o_classifier_computed_trace_version),
        .o_computed_trace_q8_8(o_classifier_computed_trace_q8_8)
    );

    unified_h40_array16_three_topology_top_v1 #(
        .SCNN_WEIGHT_FRACTIONAL_BITS(SCNN_WEIGHT_FRACTIONAL_BITS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH), .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH), .MEMORY_ADDR_WIDTH(MEMORY_ADDR_WIDTH),
        .WORK_ADDR_WIDTH(WORK_ADDR_WIDTH),
        .FEATURE_WORD_DEPTH(FEATURE_WORD_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .ENABLE_SCNN_CLASSIFIER_GRAPH(1'b1),
        .ACCUMULATE_SCNN_SAMPLE_LOGITS(ACCUMULATE_SAMPLE_LOGITS),
        .MAX_SCNN_FEATURE_CHANNELS(MAX_FEATURE_CHANNELS),
        .SCNN_CLASSIFIER_FEATURE_SPATIAL(CLASSIFIER_FEATURE_SPATIAL),
        .MAX_SCNN_CLASSES(MAX_CLASSES), .SCNN_CHANNEL_WIDTH(CHANNEL_WIDTH),
        .SCNN_CHANNEL_COUNT_WIDTH(CHANNEL_COUNT_WIDTH),
        .SCNN_SAMPLE_WIDTH(SAMPLE_INDEX_WIDTH),
        .SCNN_TIMESTEP_WIDTH(TIMESTEP_INDEX_WIDTH),
        .SCNN_FEATURE_WIDTH(FEATURE_WIDTH), .SCNN_CLASS_WIDTH(CLASS_WIDTH),
        .SCNN_CLASS_COUNT_WIDTH(CLASS_COUNT_WIDTH)
    ) u_core (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_direct_command_valid(1'b0), .i_direct_mode(2'd0),
        .i_direct_neuron_count('0), .i_direct_first(1'b0),
        .i_direct_last(1'b0), .i_direct_source_valid('0),
        .i_direct_weight_addr0('0), .i_direct_weight_addr1('0),
        .i_direct_weight_addr2('0), .i_direct_weight_addr3('0),
        .i_direct_leak('0), .i_direct_threshold('0),
        .i_direct_reset_to_zero(1'b0), .i_direct_trace_decay('0),
        .i_direct_transaction_tag('0),
        .i_direct_work_base_addr('0),
        .i_direct_destination_tile_base(4'd0),
        .i_direct_observation_ready('1),
        .i_fc_srnn_work_base_addr('0),
        .i_fc_srnn_destination_tile_base(4'd0),
        .i_fc_srnn_observation_sink_idle(1'b1),
        .i_scnn_start_valid(core_start_valid),
        .o_scnn_start_ready(core_start_ready),
        .i_scnn_in_width(i_conv_in_width),
        .i_scnn_in_height(i_conv_in_height),
        .i_scnn_in_channels(i_conv_in_channels),
        .i_scnn_out_width(i_conv_out_width),
        .i_scnn_out_height(i_conv_out_height),
        .i_scnn_out_channels(i_conv_out_channels),
        .i_scnn_stride(i_conv_stride), .i_scnn_padding(i_conv_padding),
        .i_scnn_pool_mode(i_conv_pool_mode),
        .i_scnn_initial_active_pair(i_conv_active_pair),
        .i_scnn_work_base_addr(i_conv_work_base_addr),
        .i_scnn_leak(i_conv_leak), .i_scnn_threshold(i_conv_threshold),
        .i_scnn_reset_to_zero(i_conv_reset_to_zero),
        .i_scnn_trace_decay(i_conv_trace_decay),
        .i_scnn_layer_tag(i_conv_layer_tag),
        .i_scnn_classifier_feature_channel_base(
            i_classifier_feature_channel_base),
        .i_scnn_classifier_feature_channel_count(
            i_classifier_feature_channel_count),
        .i_scnn_classifier_class_count(i_classifier_class_count),
        .i_scnn_classifier_active_pair(classifier_pair),
        .i_scnn_classifier_work_base_addr(i_classifier_work_base_addr),
        .i_scnn_classifier_trace_decay(i_classifier_trace_decay),
        .i_scnn_classifier_layer_tag(i_classifier_layer_tag),
        .i_scnn_version(classifier_version), .i_scnn_frame_index(i_frame_index),
        .i_scnn_sample_reset(i_sample_reset),
        .i_scnn_sample_last(i_sample_last),
        .i_scnn_sample_label(i_sample_label),
        .i_scnn_learning_enable(i_learning_enable),
        .i_scnn_conv8_observation_enable(transaction_learning_q),
        .i_scnn_conv8_observation_target(transaction_observation_target_q),
        .i_scnn_sample_index(transaction_sample_q),
        .i_scnn_timestep_index(transaction_timestep_q),
        .o_scnn_context_load_valid(core_context_load_valid),
        .i_scnn_context_load_ready(i_context_load_ready),
        .o_scnn_context_load_weights(core_context_load_weights),
        .o_scnn_context_output_channel_base(o_context_output_channel_base),
        .o_scnn_context_output_channel_count(o_context_output_channel_count),
        .o_scnn_context_spatial_base(o_context_spatial_base),
        .o_scnn_context_spatial_count(o_context_spatial_count),
        .o_scnn_context_active_pair(core_context_active_pair),
        .o_scnn_context_work_base_addr(o_context_work_base_addr),
        .o_scnn_feature_request_valid(core_feature_request_valid),
        .i_scnn_feature_request_ready(core_feature_request_ready),
        .o_scnn_feature_term_valid(core_feature_term_valid),
        .o_scnn_feature_input_channel_flat(
            core_feature_input_channel_flat),
        .o_scnn_feature_kernel_row_flat(core_feature_kernel_row_flat),
        .o_scnn_feature_kernel_col_flat(core_feature_kernel_col_flat),
        .o_scnn_feature_spatial_base(core_feature_spatial_base),
        .o_scnn_feature_spatial_count(core_feature_spatial_count),
        .o_scnn_feature_output_row(core_feature_output_row),
        .o_scnn_feature_output_col(core_feature_output_col),
        .o_scnn_feature_value_bit(core_feature_value_bit),
        .i_scnn_feature_response_valid(core_feature_response_valid),
        .o_scnn_feature_response_ready(core_feature_response_ready),
        .i_scnn_feature_mask0(i_feature_mask0),
        .i_scnn_feature_mask1(i_feature_mask1),
        .i_scnn_feature_mask2(i_feature_mask2),
        .i_scnn_feature_mask3(i_feature_mask3),
        .i_scnn_feature_service_error(i_feature_service_error),
        .i_scnn_feature_write_slot(i_feature_write_slot),
        .i_scnn_feature_write_base_addr(i_feature_write_base_addr),
        .i_scnn_feature_write_row_stride_words(
            i_feature_write_row_stride_words),
        .i_scnn_feature_write_channel_stride_words(
            i_feature_write_channel_stride_words),
        .o_scnn_feature_update_valid(o_feature_update_valid),
        .i_scnn_feature_update_ready(i_feature_update_ready),
        .o_scnn_feature_update_slot(o_feature_update_slot),
        .o_scnn_feature_update_addr(o_feature_update_addr),
        .o_scnn_feature_update_data(o_feature_update_data),
        .o_scnn_feature_update_mask(o_feature_update_mask),
        .i_scnn_feature_update_done(i_feature_update_done),
        .o_scnn_context_release_valid(o_context_release_valid),
        .i_scnn_context_release_ready(i_context_release_ready),
        .o_scnn_release_output_channel_base(o_release_output_channel_base),
        .o_scnn_release_output_channel_count(o_release_output_channel_count),
        .o_scnn_release_spatial_base(o_release_spatial_base),
        .o_scnn_release_spatial_count(o_release_spatial_count),
        .o_scnn_release_active_pair(o_release_active_pair),
        .o_scnn_release_work_base_addr(o_release_work_base_addr),
        .i_scnn_observation_ready('1),
        .i_scnn_observation_sink_idle(1'b1),
        .o_scnn_done(core_done),
        .o_scnn_feature_trace_wave_valid(feature_wave_valid),
        .i_scnn_feature_trace_wave_ready(feature_wave_ready),
        .o_scnn_feature_trace_wave_mask(feature_wave_mask),
        .o_scnn_feature_trace_wave_base(feature_wave_base),
        .o_scnn_feature_trace_wave_q8_8_flat(feature_wave_flat),
        .o_scnn_feature_trace_wave_version(feature_wave_version),
        .o_scnn_feature_trace_wave_frame_index(feature_wave_frame),
        .o_scnn_feature_trace_wave_first(feature_wave_first),
        .o_scnn_feature_trace_wave_last(feature_wave_last),
        .o_scnn_feature_trace_wave_sample_reset(feature_wave_sample_reset),
        .o_scnn_feature_trace_wave_sample_last(feature_wave_sample_last),
        .o_scnn_feature_sample_label(feature_sample_label),
        .o_scnn_conv8_surrogate_valid(o_conv8_surrogate_valid),
        .i_scnn_conv8_surrogate_ready(i_conv8_surrogate_ready),
        .o_scnn_conv8_surrogate_target(o_conv8_surrogate_target),
        .o_scnn_conv8_surrogate_sample(o_conv8_surrogate_sample),
        .o_scnn_conv8_surrogate_timestep(o_conv8_surrogate_timestep),
        .o_scnn_conv8_surrogate_version(o_conv8_surrogate_version),
        .o_scnn_conv8_surrogate_layer_tag(o_conv8_surrogate_layer_tag),
        .o_scnn_conv8_surrogate_channel(o_conv8_surrogate_channel),
        .o_scnn_conv8_surrogate_spatial(o_conv8_surrogate_spatial),
        .o_scnn_conv8_membrane_pre_q8_8(o_conv8_membrane_pre_q8_8),
        .o_scnn_conv8_surrogate_q8_8(o_conv8_surrogate_q8_8),
        .o_scnn_logit_wave_valid(logit_wave_valid),
        .i_scnn_logit_wave_ready(logit_wave_ready),
        .o_scnn_logit_wave_mask(logit_wave_mask),
        .o_scnn_logit_wave_base(logit_wave_base),
        .o_scnn_logit_wave_count(logit_wave_count),
        .o_scnn_logit_wave_q8_8_flat(logit_wave_flat),
        .o_scnn_logit_wave_version(logit_wave_version),
        .o_scnn_logit_wave_frame_index(logit_wave_frame),
        .o_scnn_logit_wave_frame_last(logit_wave_frame_last),
        .o_scnn_logit_wave_sample_last(logit_wave_sample_last),
        .i_weight_fill_valid(core_fill_valid),
        .o_weight_fill_ready(core_fill_ready),
        .i_weight_fill_tile_flat(core_fill_tile_flat),
        .i_weight_fill_bank(core_fill_bank),
        .i_weight_fill_addr_flat(core_fill_addr_flat),
        .i_weight_fill_data_flat(core_fill_data_flat),
        .i_maintenance_start_valid(core_maintenance_start_valid),
        .o_maintenance_start_ready(core_maintenance_start_ready),
        .i_maintenance_release_valid(core_maintenance_release_valid),
        .o_maintenance_release_ready(core_maintenance_release_ready),
        .o_maintenance_done(),
        .i_maintenance_tile(i_maintenance_tile),
        .i_state_maintenance_write_valid(i_state_maintenance_write_valid),
        .o_state_maintenance_write_ready(o_state_maintenance_write_ready),
        .i_state_maintenance_write_addr(i_state_maintenance_write_addr),
        .i_state_maintenance_write_data(i_state_maintenance_write_data),
        .i_state_maintenance_write_mask(i_state_maintenance_write_mask),
        .i_state_maintenance_read_valid(i_state_maintenance_read_valid),
        .o_state_maintenance_read_ready(o_state_maintenance_read_ready),
        .i_state_maintenance_read_addr(i_state_maintenance_read_addr),
        .o_state_maintenance_result_valid(o_state_maintenance_result_valid),
        .i_state_maintenance_result_ready(i_state_maintenance_result_ready),
        .o_state_maintenance_result_addr(o_state_maintenance_result_addr),
        .o_state_maintenance_read_data(o_state_maintenance_read_data),
        .i_trace_maintenance_write_valid(i_trace_maintenance_write_valid),
        .o_trace_maintenance_write_ready(o_trace_maintenance_write_ready),
        .i_trace_maintenance_write_addr(i_trace_maintenance_write_addr),
        .i_trace_maintenance_write_data(i_trace_maintenance_write_data),
        .i_trace_maintenance_write_mask(i_trace_maintenance_write_mask),
        .i_trace_maintenance_read_valid(i_trace_maintenance_read_valid),
        .o_trace_maintenance_read_ready(o_trace_maintenance_read_ready),
        .i_trace_maintenance_read_addr(i_trace_maintenance_read_addr),
        .o_trace_maintenance_result_valid(o_trace_maintenance_result_valid),
        .i_trace_maintenance_result_ready(i_trace_maintenance_result_ready),
        .o_trace_maintenance_result_addr(o_trace_maintenance_result_addr),
        .o_trace_maintenance_read_data(o_trace_maintenance_read_data),
        .o_active_owner(o_active_owner), .o_busy(core_busy),
        .o_idle(core_idle), .o_tile_idle(o_tile_idle),
        .o_tile_sticky_error(o_tile_sticky_error),
        .o_sticky_error(core_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n || i_abort) begin
            maintenance_lease_q <= 1'b0;
        end else begin
            if (core_maintenance_start_valid && core_maintenance_start_ready)
                maintenance_lease_q <= 1'b1;
            else if (core_maintenance_release_valid &&
                    core_maintenance_release_ready)
                maintenance_lease_q <= 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            tx_state_q <= TX_IDLE;
            transaction_learning_q <= 1'b0;
            prescan_started_q <= 1'b0;
            core_started_q <= 1'b0;
            transaction_observation_target_q <= 1'b0;
            transaction_sample_q <= '0;
            transaction_timestep_q <= '0;
            o_scnn_done <= 1'b0;
            protocol_error_q <= 1'b0;
        end else if (i_abort) begin
            tx_state_q <= TX_IDLE;
            transaction_learning_q <= 1'b0;
            prescan_started_q <= 1'b0;
            core_started_q <= 1'b0;
            o_scnn_done <= 1'b0;
        end else begin
            o_scnn_done <= 1'b0;
            if (i_clear_error && tx_state_q == TX_IDLE && core_idle &&
                    !cache_publish_busy)
                protocol_error_q <= 1'b0;
            if ((i_scnn_start_valid && tx_state_q == TX_IDLE &&
                    classifier_cache_valid &&
                    i_conv_active_pair == classifier_pair) ||
                    |external_fill_conflict)
                protocol_error_q <= 1'b1;
            case (tx_state_q)
                TX_IDLE: begin
                    prescan_started_q <= 1'b0;
                    core_started_q <= 1'b0;
                    if (i_scnn_start_valid && o_scnn_start_ready) begin
                        transaction_learning_q <= i_learning_enable;
                        transaction_observation_target_q <=
                            i_conv8_observation_target;
                        transaction_sample_q <= i_sample_index;
                        transaction_timestep_q <= i_timestep_index;
                        tx_state_q <= i_learning_enable ?
                            TX_PRESCAN : TX_GRAPH;
                    end
                end
                TX_PRESCAN: begin
                    if (prescan_start_valid && prescan_start_ready)
                        prescan_started_q <= 1'b1;
                    if (prescan_done) begin
                        core_started_q <= 1'b0;
                        tx_state_q <= TX_GRAPH;
                    end
                end
                TX_GRAPH: begin
                    if (core_start_valid && core_start_ready)
                        core_started_q <= 1'b1;
                    if (core_done) begin
                        // A temporal learning window can span multiple graph
                        // invocations.  Return each nonfinal timestep to the
                        // caller; the replay bridge raises inference_hold only
                        // after the configured BxT observation window is
                        // complete and ready for learning/commit.
                        if (transaction_learning_q && learning_inference_hold)
                            tx_state_q <= TX_LEARN;
                        else begin
                            o_scnn_done <= 1'b1;
                            tx_state_q <= TX_IDLE;
                        end
                    end
                end
                TX_LEARN: begin
                    if (learning_batch_done) begin
                        o_scnn_done <= 1'b1;
                        tx_state_q <= TX_IDLE;
                    end
                end
                default: tx_state_q <= TX_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
