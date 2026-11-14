`timescale 1ns/1ps
`default_nettype none

// Paper-profile composition of the three actual TP learning clients.  The
// physical inference core is intentionally outside this module.  FC can retain
// its legacy native-bank boundary or route learning RMW through client 0 of the
// same externally backed scalar parameter service used by SRNN and SCNN.
module unified_tp_actual_learning_subsystem_v1 #(
    parameter bit FC_LEARN_W0 = 1'b1,
    parameter bit FC_USE_SHARED_PARAMETER_BACKEND = 1'b0,
    parameter bit SHARE_FC_SCNN_OUTPUT_LEARNER = 1'b0,
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int FC_MAX_HIDDEN = 480,
    parameter int FC_FEATURES = 2312,
    parameter int SRNN_HIDDEN = 256,
    parameter int SRNN_BACKBONE = 450,
    parameter int SRNN_CLASSES = 20,
    parameter int SCNN_CHANNELS = 512,
    parameter int SCNN_FEATURES = 2048,
    parameter int SCNN_CLASSES = 11,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter logic [PARAM_ADDR_WIDTH-1:0] FC_W0_BASE = 24'h000000,
    parameter logic [PARAM_ADDR_WIDTH-1:0] FC_W2_BASE = 24'h300000,
    parameter logic [PARAM_ADDR_WIDTH-1:0] SCNN_CONV8_BASE = 24'h000000,
    parameter logic [PARAM_ADDR_WIDTH-1:0] SCNN_CLASSIFIER_BASE = 24'h240000,
    parameter logic [PARAM_ADDR_WIDTH-1:0] SRNN_WH1_BASE = 24'h300000,
    parameter logic [PARAM_ADDR_WIDTH-1:0] SRNN_WOUT_BASE = 24'h31c200
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,

    input  logic                                i_profile_request_valid,
    input  logic [6:0]                          i_profile_request_code,
    input  logic                                i_profile_learning_enable,
    output logic                                o_profile_aligned,
    output logic                                o_start_allowed,
    output logic                                o_profile_switch_done,
    output logic                                o_profile_switch_busy,
    output logic [6:0]                          o_active_profile_code,
    output logic [15:0]                         o_profile_version,
    output logic                                o_active_learning_enable,
    input  logic                                i_run_valid,
    input  logic [6:0]                          i_run_profile_code,
    output logic                                o_run_ready,
    output logic                                o_run_done,
    output logic                                o_run_aborted,
    output logic                                o_run_busy,
    output logic [2:0]                          o_active_client,
    output logic [2:0]                          o_parameter_owner,
    output logic                                o_system_quiescent,
    output logic                                o_system_error,
    output logic [7:0]                          o_control_error_code,

    output logic                                o_core_start,
    input  logic                                i_core_start_ready,
    output logic [6:0]                          o_core_profile_code,
    output logic                                o_core_abort,
    input  logic                                i_core_sample_done,
    input  logic                                i_core_global_idle,
    input  logic                                i_core_error,
    output logic [2:0]                          o_core_active_source,

    input  logic                                i_fc_cfg_valid,
    output logic                                o_fc_cfg_ready,
    input  logic [8:0]                          i_fc_hidden_count,
    output logic                                o_fc_configured,
    output logic                                o_fc_cfg_error,
    input  logic                                i_fc_window_valid,
    output logic                                o_fc_window_ready,
    input  logic [LANES-1:0]                   i_fc_record_valid,
    input  logic [15:0]                         i_fc_window_sample_id,
    input  logic [8:0]                          i_fc_window_timestep,
    input  logic [LANES-1:0]                   i_fc_boundary_mask,
    input  logic                                i_fc_window_sample_last,
    input  logic [LANES*12-1:0]                i_fc_addr_flat,
    input  logic [LANES*8-1:0]                 i_fc_data_flat,
    input  logic [3:0]                          i_fc_snapshot_label,
    input  logic [FC_MAX_HIDDEN-1:0]           i_fc_snapshot_spikes_flat,
    input  logic [FC_MAX_HIDDEN*16-1:0]        i_fc_snapshot_trace_flat,
    input  logic [FC_MAX_HIDDEN*18-1:0]        i_fc_snapshot_vmem_flat,
    input  logic [10*24-1:0]                   i_fc_snapshot_logits_flat,
    input  logic                                i_fc_target_hidden_trace_valid,
    output logic                                o_fc_target_hidden_trace_ready,
    input  logic [3:0]                          i_fc_target_trace_sample,
    input  logic [8:0]                          i_fc_target_hidden_index,
    input  logic [16:0]                         i_fc_target_hidden_trace_q016,
    input  logic                                i_fc_target_input_trace_valid,
    output logic                                o_fc_target_input_trace_ready,
    input  logic [3:0]                          i_fc_target_input_index,
    input  logic [19:0]                         i_fc_target_input_trace_q016,
    input  logic                                i_fc_target_done,
    output logic                                o_fc_target_done_ready,
    output logic                                o_fc_event_window_valid,
    input  logic                                i_fc_event_window_ready,
    output logic [LANES-1:0]                   o_fc_event_record_valid,
    output logic [15:0]                         o_fc_event_window_sample_id,
    output logic [8:0]                          o_fc_event_window_timestep,
    output logic [LANES-1:0]                   o_fc_event_boundary_mask,
    output logic                                o_fc_event_window_sample_last,
    output logic [LANES*12-1:0]                o_fc_event_addr_flat,
    output logic [LANES*8-1:0]                 o_fc_event_data_flat,
    output logic [15:0]                         o_fc_core_sample_id,
    output logic                                o_fc_core_sample_state_reset,
    output logic                                o_fc_core_clear_state,
    output logic                                o_fc_core_sample_last,
    output logic                                o_fc_native_read_valid,
    input  logic                                i_fc_native_read_ready,
    output logic [7:0]                          o_fc_native_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_fc_native_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                o_fc_native_read_tag,
    input  logic                                i_fc_native_response_valid,
    output logic                                o_fc_native_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_fc_native_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_fc_native_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_fc_native_response_residual,
    input  logic                                i_fc_native_response_error,
    output logic                                o_fc_native_write_valid,
    input  logic                                i_fc_native_write_ready,
    output logic [7:0]                          o_fc_native_write_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_fc_native_write_parameter_addr,
    output logic signed [CODE_WIDTH-1:0]        o_fc_native_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_fc_native_write_residual,
    output logic                                o_fc_native_write_code_enable,
    output logic                                o_fc_native_write_residual_enable,
    input  logic                                i_fc_native_parameter_quiescent,
    output logic                                o_fc_learning_commit,
    output logic                                o_fc_learning_busy,
    input  logic                                i_fc_cache_forward_read_valid,
    output logic                                o_fc_cache_forward_read_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]         i_fc_cache_forward_read_addr,
    input  logic [TAG_WIDTH-1:0]                i_fc_cache_forward_read_tag,
    output logic                                o_fc_cache_forward_response_valid,
    input  logic                                i_fc_cache_forward_response_ready,
    output logic [TAG_WIDTH-1:0]                o_fc_cache_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]        o_fc_cache_forward_response_code,
    output logic                                o_fc_cache_forward_response_error,
    output logic                                o_fc_w0_refresh_valid,
    input  logic                                i_fc_w0_refresh_ready,
    output logic [((FC_FEATURES <= 2) ? 1 :
        $clog2(FC_FEATURES))-1:0]               o_fc_w0_refresh_feature,
    output logic [3:0]                          o_fc_w0_refresh_tile,
    input  logic                                i_fc_w0_refresh_done,
    input  logic                                i_fc_w0_refresh_error,

    input  logic                                i_srnn_cfg_valid,
    output logic                                o_srnn_cfg_ready,
    input  logic [15:0]                         i_srnn_cfg_commit_version,
    output logic                                o_srnn_configured,
    output logic                                o_srnn_cfg_error,
    input  logic                                i_srnn_parameter_load_valid,
    output logic                                o_srnn_parameter_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]         i_srnn_parameter_load_addr,
    input  logic signed [CODE_WIDTH-1:0]        i_srnn_parameter_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_srnn_parameter_load_residual,
    input  logic                                i_srnn_target_label_load_start,
    output logic                                o_srnn_target_label_load_start_ready,
    input  logic                                i_srnn_target_label_valid,
    output logic                                o_srnn_target_label_ready,
    input  logic                                i_srnn_target_label_sample,
    input  logic [4:0]                          i_srnn_target_label_class,
    input  logic                                i_srnn_target_label_last,
    output logic                                o_srnn_target_label_load_done,
    input  logic                                i_srnn_target_spike_word_write_valid,
    output logic                                o_srnn_target_spike_word_write_ready,
    input  logic [4:0]                          i_srnn_target_spike_word_class,
    input  logic                                i_srnn_target_spike_word_timestep,
    input  logic [3:0]                          i_srnn_target_spike_word_id,
    input  logic [31:0]                         i_srnn_target_spike_word_data,
    input  logic                                i_srnn_target_trace_write_valid,
    output logic                                o_srnn_target_trace_write_ready,
    input  logic [4:0]                          i_srnn_target_trace_class,
    input  logic                                i_srnn_target_trace_timestep,
    input  logic [8:0]                          i_srnn_target_trace_backbone,
    input  logic signed [31:0]                  i_srnn_target_trace_q14,
    input  logic                                i_srnn_backbone_sample_start_valid,
    output logic                                o_srnn_backbone_sample_start_ready,
    output logic                                o_srnn_backbone_sample_id,
    input  logic                                i_srnn_commit_valid,
    output logic                                o_srnn_commit_ready,
    input  logic [15:0]                         i_srnn_commit_version,
    input  logic                                i_srnn_commit_sample_last,
    input  logic [479:0]                        i_srnn_commit_spikes,
    output logic                                o_srnn_learning_commit,
    output logic                                o_srnn_detach,
    output logic [PARAM_ADDR_WIDTH:0]           o_srnn_update_count,

    input  logic                                i_scnn_configured,
    input  logic                                i_scnn_target_start_ready,
    output logic                                o_scnn_target_start_valid,
    output logic                                o_scnn_target_sample,
    output logic                                o_scnn_target_timestep,
    input  logic                                i_scnn_target_done,
    input  logic                                i_scnn_target_busy,
    input  logic                                i_scnn_target_error,
    output logic                                o_scnn_core_sample,
    output logic                                o_scnn_core_timestep,
    output logic                                o_scnn_core_post_forward,
    output logic                                o_scnn_classifier_capture_enable,
    input  logic                                i_scnn_selected_conv_done,
    input  logic                                i_scnn_sample_context_valid,
    output logic                                o_scnn_sample_context_ready,
    input  logic                                i_scnn_sample_context_target,
    input  logic                                i_scnn_sample_context_sample,
    input  logic [15:0]                         i_scnn_sample_context_version,
    input  logic                                i_scnn_input_valid,
    output logic                                o_scnn_input_ready,
    input  logic                                i_scnn_input_target,
    input  logic                                i_scnn_input_sample,
    input  logic                                i_scnn_input_timestep,
    input  logic [15:0]                         i_scnn_input_sample_version,
    input  logic [8:0]                          i_scnn_input_channel,
    input  logic [3:0]                          i_scnn_input_spatial,
    input  logic signed [15:0]                  i_scnn_input_q8_8,
    input  logic                                i_scnn_trace_valid,
    output logic                                o_scnn_trace_ready,
    input  logic                                i_scnn_trace_target,
    input  logic                                i_scnn_trace_sample,
    input  logic                                i_scnn_trace_timestep,
    input  logic [15:0]                         i_scnn_trace_sample_version,
    input  logic [8:0]                          i_scnn_trace_channel,
    input  logic [1:0]                          i_scnn_trace_spatial,
    input  logic signed [15:0]                  i_scnn_trace_q8_8,
    input  logic                                i_scnn_surrogate_valid,
    output logic                                o_scnn_surrogate_ready,
    input  logic                                i_scnn_surrogate_target,
    input  logic                                i_scnn_surrogate_sample,
    input  logic                                i_scnn_surrogate_timestep,
    input  logic [15:0]                         i_scnn_surrogate_sample_version,
    input  logic [8:0]                          i_scnn_surrogate_channel,
    input  logic [3:0]                          i_scnn_surrogate_spatial,
    input  logic signed [15:0]                  i_scnn_surrogate_q8_8,
    input  logic                                i_scnn_feature_wave_valid,
    output logic                                o_scnn_feature_wave_ready,
    input  logic [LANES-1:0]                   i_scnn_feature_wave_mask,
    input  logic [10:0]                         i_scnn_feature_wave_base,
    input  logic [LANES*16-1:0]                i_scnn_feature_wave_q8_8_flat,
    input  logic [15:0]                         i_scnn_feature_wave_version,
    input  logic [7:0]                          i_scnn_feature_wave_frame,
    input  logic                                i_scnn_feature_wave_first,
    input  logic                                i_scnn_feature_wave_last,
    input  logic                                i_scnn_feature_wave_sample_reset,
    input  logic                                i_scnn_feature_wave_sample_last,
    input  logic [3:0]                          i_scnn_sample_label,
    input  logic                                i_scnn_logit_wave_valid,
    output logic                                o_scnn_logit_wave_ready,
    input  logic [ENGINES-1:0]                 i_scnn_logit_wave_mask,
    input  logic [3:0]                          i_scnn_logit_wave_base,
    input  logic [3:0]                          i_scnn_logit_wave_count,
    input  logic [ENGINES*16-1:0]              i_scnn_logit_wave_q8_8_flat,
    input  logic [15:0]                         i_scnn_logit_wave_version,
    input  logic [7:0]                          i_scnn_logit_wave_frame,
    input  logic                                i_scnn_logit_wave_frame_last,
    input  logic                                i_scnn_logit_wave_sample_last,
    input  logic                                i_scnn_parameter_load_valid,
    output logic                                o_scnn_parameter_load_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]         i_scnn_parameter_load_addr,
    input  logic signed [CODE_WIDTH-1:0]        i_scnn_parameter_load_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_scnn_parameter_load_residual,
    input  logic                                i_scnn_forward_valid,
    output logic                                o_scnn_forward_ready,
    input  logic [PARAM_ADDR_WIDTH-1:0]         i_scnn_forward_addr,
    input  logic [TAG_WIDTH-1:0]                i_scnn_forward_tag,
    output logic                                o_scnn_forward_response_valid,
    input  logic                                i_scnn_forward_response_ready,
    output logic [TAG_WIDTH-1:0]                o_scnn_forward_response_tag,
    output logic signed [CODE_WIDTH-1:0]        o_scnn_forward_response_code,
    output logic                                o_scnn_forward_response_error,
    output logic                                o_scnn_learning_commit,
    output logic                                o_scnn_detach,
    output logic                                o_scnn_learning_busy,

    output logic                                o_store_load_valid,
    input  logic                                i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0]        o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_store_load_residual,
    output logic                                o_store_learning_read_valid,
    input  logic                                i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0]                o_store_learning_read_tag,
    input  logic                                i_store_learning_response_valid,
    output logic                                o_store_learning_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_store_learning_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_store_learning_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]    i_store_learning_response_residual,
    input  logic                                i_store_learning_response_error,
    output logic                                o_store_learning_write_valid,
    input  logic                                i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0]        o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0]    o_store_learning_write_residual,
    output logic                                o_store_learning_write_code_enable,
    output logic                                o_store_learning_write_residual_enable,
    output logic                                o_store_forward_read_valid,
    input  logic                                i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0]                o_store_forward_read_tag,
    input  logic                                i_store_forward_response_valid,
    output logic                                o_store_forward_response_ready,
    input  logic [TAG_WIDTH-1:0]                i_store_forward_response_tag,
    input  logic signed [CODE_WIDTH-1:0]        i_store_forward_response_code,
    input  logic                                i_store_forward_response_error,
    input  logic                                i_store_quiescent,
    input  logic                                i_store_protocol_error
);
    localparam int FC_BATCH_COUNT_WIDTH = 5;
    localparam int SRNN_TIMESTEP_COUNT_WIDTH = 2;
    localparam int SHARED_BATCH_INDEX_WIDTH = 4;
    localparam int SHARED_BATCH_COUNT_WIDTH = 5;
    localparam int SHARED_MAX_SOURCES =
        (SCNN_FEATURES > FC_MAX_HIDDEN) ? SCNN_FEATURES : FC_MAX_HIDDEN;
    localparam int SHARED_SOURCE_INDEX_WIDTH =
        (SHARED_MAX_SOURCES <= 2) ? 1 : $clog2(SHARED_MAX_SOURCES);
    localparam int SHARED_SOURCE_COUNT_WIDTH =
        $clog2(SHARED_MAX_SOURCES + 1);
    localparam int SHARED_OUTPUT_INDEX_WIDTH = 4;
    localparam int SHARED_OUTPUT_COUNT_WIDTH = 5;
    localparam int SHARED_MAX_GRADIENT_ELEMENTS =
        (SCNN_FEATURES * SCNN_CLASSES > FC_MAX_HIDDEN * 10) ?
        SCNN_FEATURES * SCNN_CLASSES : FC_MAX_HIDDEN * 10;
    localparam bit FC_EFFECTIVE_SHARED_BACKEND =
        FC_USE_SHARED_PARAMETER_BACKEND || SHARE_FC_SCNN_OUTPUT_LEARNER;

    logic [2:0] client_acquire_valid;
    logic [2:0] client_release_valid;
    logic [2:0] topology_learning_enable;
    logic [2:0] client_active;
    logic [2:0] client_quiescent;
    logic [2:0] client_error;
    logic control_quiescent;
    logic dispatch_protocol_error;
    logic backend_quiescent;
    logic backend_error;
    logic owner_encoding_error;
    logic router_protocol_error;

    logic fc_native_read_valid;
    logic fc_native_read_ready;
    logic [7:0] fc_native_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] fc_native_read_parameter_addr;
    logic [TAG_WIDTH-1:0] fc_native_read_tag;
    logic fc_native_response_valid;
    logic fc_native_response_ready;
    logic [TAG_WIDTH-1:0] fc_native_response_tag;
    logic signed [CODE_WIDTH-1:0] fc_native_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] fc_native_response_residual;
    logic fc_native_response_error;
    logic fc_native_write_valid;
    logic fc_native_write_ready;
    logic [7:0] fc_native_write_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] fc_native_write_parameter_addr;
    logic signed [CODE_WIDTH-1:0] fc_native_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] fc_native_write_residual;
    logic fc_native_write_code_enable;
    logic fc_native_write_residual_enable;
    logic fc_native_parameter_quiescent;
    logic fc_shared_write_valid;
    logic fc_shared_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] fc_shared_write_parameter_addr;
    logic signed [CODE_WIDTH-1:0] fc_shared_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] fc_shared_write_residual;
    logic fc_shared_write_code_enable;
    logic fc_shared_write_residual_enable;
    logic fc_cache_write_ready;
    logic fc_cache_quiescent;
    logic fc_cache_flush_ready;
    logic fc_cache_flush_done;
    logic fc_cache_protocol_error;
    logic fc_cache_flush_violation_q;
    logic fc_batch_done_pending_q;
    logic fc_batch_learning_done_raw;

    logic fc_release_ready;
    logic fc_execution_quiescent;
    logic fc_runtime_error;
    logic fc_runtime_sample_retired;
    logic fc_runtime_batch_learning_done;
    logic fc_runtime_batch_learning_done_raw;
    logic fc_runtime_abort_complete;
    logic fc_core_start_valid;
    logic fc_core_start_ready;
    logic fc_core_done;
    logic fc_core_aborted;
    logic fc_sample_start_allowed;
    logic fc_run_learning_enable;
    logic fc_abort_run;
    logic [FC_BATCH_COUNT_WIDTH-1:0] fc_started_samples;
    logic [FC_BATCH_COUNT_WIDTH-1:0] fc_retired_samples;

    logic srnn_runtime_timestep_start;
    logic srnn_runtime_timestep_start_ready;
    logic srnn_runtime_timestep_id;
    logic [15:0] srnn_runtime_sequence_version;
    logic srnn_runtime_learning_enable;
    logic srnn_runtime_abort_timestep;
    logic srnn_runtime_abort_ready;
    logic srnn_runtime_timestep_done;
    logic srnn_runtime_timestep_aborted;
    logic srnn_runtime_busy;
    logic srnn_runtime_error;
    logic srnn_core_start_valid;
    logic srnn_core_start_ready;
    logic srnn_core_done;
    logic srnn_core_aborted;
    logic srnn_core_abort;
    logic [SRNN_TIMESTEP_COUNT_WIDTH-1:0] srnn_completed_timesteps;
    logic srnn_quiescent;

    logic scnn_start_allowed;
    logic scnn_run_start_valid;
    logic scnn_run_start_ready;
    logic scnn_abort;
    logic scnn_run_done;
    logic scnn_run_aborted;
    logic scnn_busy;
    logic scnn_quiescent;
    logic scnn_error;
    logic scnn_core_start_valid;
    logic scnn_core_start_ready;
    logic scnn_core_done;
    logic scnn_core_aborted;
    logic scnn_core_abort;
    logic scnn_run_learning_enable;

    // FC and SCNN expose normalized observation clients.  When enabled, one
    // physical output learner consumes either client and emits one scalar RMW
    // stream; topology ownership remains locked for the complete window.
    logic fc_external_cfg_valid;
    logic fc_external_cfg_ready;
    logic [7:0] fc_external_cfg_profile_id;
    logic [7:0] fc_external_cfg_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] fc_external_cfg_parameter_base;
    logic [FC_BATCH_COUNT_WIDTH-1:0] fc_external_cfg_batch_count;
    logic [$clog2(FC_MAX_HIDDEN+1)-1:0] fc_external_cfg_source_count;
    logic [3:0] fc_external_cfg_output_count;
    logic [15:0] fc_external_cfg_vote_increment;
    logic [15:0] fc_external_cfg_vote_threshold;
    logic signed [CODE_WIDTH-1:0] fc_external_cfg_code_min;
    logic signed [CODE_WIDTH-1:0] fc_external_cfg_code_max;
    logic fc_external_cfg_error;
    logic fc_external_active;
    logic fc_external_release;
    logic fc_external_release_ready;
    logic fc_external_source_trace_valid;
    logic fc_external_source_trace_ready;
    logic [3:0] fc_external_source_trace_sample_id;
    logic [$clog2(FC_MAX_HIDDEN)-1:0] fc_external_source_trace_index;
    logic [15:0] fc_external_source_trace_q016;
    logic fc_external_source_trace_error;
    logic fc_external_logit_row_valid;
    logic fc_external_logit_row_ready;
    logic [3:0] fc_external_logit_sample_id;
    logic [3:0] fc_external_logit_label;
    logic fc_external_logit_valid;
    logic fc_external_logit_ready;
    logic signed [15:0] fc_external_logit_q6;
    logic fc_external_logit_error;
    logic fc_external_logit_row_done;
    logic fc_external_observation_done;
    logic fc_external_observation_done_ready;
    logic fc_external_window_done;
    logic fc_external_window_busy;
    logic fc_external_context_error;
    logic fc_external_gradient_overflow;
    logic fc_external_update_config_error;
    logic fc_external_request_error;
    logic fc_external_store_error;
    logic fc_external_internal_error;
    logic fc_external_dirty;
    logic fc_external_saturated;

    logic scnn_external_cfg_valid;
    logic scnn_external_cfg_ready;
    logic [7:0] scnn_external_cfg_profile_id;
    logic [7:0] scnn_external_cfg_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_external_cfg_parameter_base;
    logic [1:0] scnn_external_cfg_batch_count;
    logic [$clog2(SCNN_FEATURES+1)-1:0] scnn_external_cfg_source_count;
    logic [3:0] scnn_external_cfg_output_count;
    logic [15:0] scnn_external_cfg_vote_increment;
    logic [15:0] scnn_external_cfg_vote_threshold;
    logic signed [CODE_WIDTH-1:0] scnn_external_cfg_code_min;
    logic signed [CODE_WIDTH-1:0] scnn_external_cfg_code_max;
    logic scnn_external_cfg_error;
    logic scnn_external_active;
    logic scnn_external_release;
    logic scnn_external_release_ready;
    logic scnn_external_source_trace_valid;
    logic scnn_external_source_trace_ready;
    logic scnn_external_source_trace_sample_id;
    logic [$clog2(SCNN_FEATURES)-1:0] scnn_external_source_trace_index;
    logic [15:0] scnn_external_source_trace_q016;
    logic scnn_external_source_trace_error;
    logic scnn_external_logit_row_valid;
    logic scnn_external_logit_row_ready;
    logic scnn_external_logit_sample_id;
    logic [3:0] scnn_external_logit_label;
    logic scnn_external_logit_valid;
    logic scnn_external_logit_ready;
    logic signed [15:0] scnn_external_logit_q6;
    logic scnn_external_logit_error;
    logic scnn_external_logit_row_done;
    logic scnn_external_observation_done;
    logic scnn_external_observation_done_ready;
    logic scnn_external_window_done;
    logic scnn_external_window_busy;
    logic scnn_external_context_error;
    logic scnn_external_gradient_overflow;
    logic scnn_external_update_config_error;
    logic scnn_external_request_error;
    logic scnn_external_store_error;
    logic scnn_external_internal_error;

    logic [1:0] shared_output_owner;
    logic shared_output_busy;
    logic shared_output_protocol_error;
    logic shared_output_read_valid;
    logic shared_output_read_ready;
    logic [7:0] shared_output_read_profile_id;
    logic [7:0] shared_output_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] shared_output_read_addr;
    logic [TAG_WIDTH-1:0] shared_output_read_tag;
    logic shared_output_response_valid;
    logic shared_output_response_ready;
    logic signed [CODE_WIDTH-1:0] shared_output_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] shared_output_response_residual;
    logic shared_output_response_error;
    logic shared_output_result_valid;
    logic shared_output_result_ready;
    logic [7:0] shared_output_result_profile_id;
    logic [7:0] shared_output_result_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] shared_output_result_addr;
    logic [TAG_WIDTH-1:0] shared_output_result_tag;
    logic signed [CODE_WIDTH-1:0] shared_output_result_code;
    logic signed [RESIDUAL_WIDTH-1:0] shared_output_result_residual;
    logic shared_output_write_code_enable;
    logic shared_output_write_residual_enable;
    logic shared_output_write_code_fire;
    logic shared_output_write_residual_fire;
    logic shared_output_dirty;
    logic shared_output_saturated;
    logic shared_output_update_config_error;
    logic shared_output_request_error;
    logic shared_output_store_error;
    logic shared_output_internal_error;
    logic shared_scnn_read_ready;
    logic shared_scnn_response_valid;
    logic signed [CODE_WIDTH-1:0] shared_scnn_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] shared_scnn_response_residual;
    logic shared_scnn_response_error;
    logic shared_scnn_result_ready;

    logic srnn_store_load_valid;
    logic srnn_store_load_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_load_addr;
    logic signed [CODE_WIDTH-1:0] srnn_store_load_code;
    logic signed [RESIDUAL_WIDTH-1:0] srnn_store_load_residual;
    logic srnn_store_read_valid;
    logic srnn_store_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_read_addr;
    logic [TAG_WIDTH-1:0] srnn_store_read_tag;
    logic srnn_store_response_valid;
    logic srnn_store_response_ready;
    logic srnn_store_write_valid;
    logic srnn_store_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_write_addr;
    logic signed [CODE_WIDTH-1:0] srnn_store_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] srnn_store_write_residual;
    logic srnn_store_write_code_enable;
    logic srnn_store_write_residual_enable;
    logic srnn_store_forward_valid;
    logic srnn_store_forward_ready;
    logic [PARAM_ADDR_WIDTH-1:0] srnn_store_forward_addr;
    logic [TAG_WIDTH-1:0] srnn_store_forward_tag;
    logic srnn_store_forward_response_valid;
    logic srnn_store_forward_response_ready;

    logic scnn_read_valid;
    logic scnn_read_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_read_addr;
    logic [TAG_WIDTH-1:0] scnn_read_tag;
    logic scnn_response_valid;
    logic scnn_response_ready;
    logic scnn_write_valid;
    logic scnn_write_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_write_addr;
    logic [TAG_WIDTH-1:0] scnn_write_tag;
    logic signed [CODE_WIDTH-1:0] scnn_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] scnn_write_residual;
    logic scnn_write_code_enable;
    logic scnn_write_residual_enable;
    logic scnn_private_read_valid;
    logic scnn_private_read_ready;
    logic [7:0] scnn_private_read_profile_id;
    logic [7:0] scnn_private_read_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_private_read_addr;
    logic [TAG_WIDTH-1:0] scnn_private_read_tag;
    logic scnn_private_response_valid;
    logic scnn_private_response_ready;
    logic signed [CODE_WIDTH-1:0] scnn_private_response_code;
    logic signed [RESIDUAL_WIDTH-1:0] scnn_private_response_residual;
    logic scnn_private_response_error;
    logic scnn_private_write_valid;
    logic scnn_private_write_ready;
    logic [7:0] scnn_private_write_profile_id;
    logic [7:0] scnn_private_write_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_private_write_addr;
    logic [TAG_WIDTH-1:0] scnn_private_write_tag;
    logic signed [CODE_WIDTH-1:0] scnn_private_write_code;
    logic signed [RESIDUAL_WIDTH-1:0] scnn_private_write_residual;
    logic scnn_private_write_code_enable;
    logic scnn_private_write_residual_enable;
    logic [7:0] scnn_merged_read_profile_id;
    logic [7:0] scnn_merged_read_tensor_id;
    logic [7:0] scnn_merged_write_profile_id;
    logic [7:0] scnn_merged_write_tensor_id;
    logic scnn_merge_busy;
    logic scnn_merge_protocol_error;
    logic scnn_forward_valid;
    logic scnn_forward_ready;
    logic [PARAM_ADDR_WIDTH-1:0] scnn_forward_addr;
    logic [TAG_WIDTH-1:0] scnn_forward_tag;
    logic scnn_forward_response_valid;
    logic scnn_forward_response_ready;

    logic [2:0] load_ready;
    logic [2:0] learning_read_ready;
    logic [2:0] learning_response_valid;
    logic [3*TAG_WIDTH-1:0] learning_response_tag;
    logic signed [3*CODE_WIDTH-1:0] learning_response_code;
    logic signed [3*RESIDUAL_WIDTH-1:0] learning_response_residual;
    logic [2:0] learning_response_error;
    logic [2:0] learning_write_ready;
    logic [2:0] forward_read_ready;
    logic [2:0] forward_response_valid;
    logic [3*TAG_WIDTH-1:0] forward_response_tag;
    logic signed [3*CODE_WIDTH-1:0] forward_response_code;
    logic [2:0] forward_response_error;

    assign o_scnn_parameter_load_ready = load_ready[2];
    assign srnn_store_load_ready = load_ready[1];
    assign srnn_store_read_ready = learning_read_ready[1];
    assign srnn_store_response_valid = learning_response_valid[1];
    assign srnn_store_write_ready = learning_write_ready[1];
    assign srnn_store_forward_ready = forward_read_ready[1];
    assign srnn_store_forward_response_valid = forward_response_valid[1];
    assign scnn_read_ready = learning_read_ready[2];
    assign scnn_response_valid = learning_response_valid[2];
    assign scnn_write_ready = learning_write_ready[2];
    assign scnn_forward_valid = i_scnn_forward_valid;
    assign scnn_forward_ready = forward_read_ready[2];
    assign scnn_forward_response_valid = forward_response_valid[2];
    assign o_scnn_forward_ready = scnn_forward_ready;
    assign scnn_forward_addr = i_scnn_forward_addr;
    assign scnn_forward_tag = i_scnn_forward_tag;
    assign o_scnn_forward_response_valid = scnn_forward_response_valid;
    assign scnn_forward_response_ready = i_scnn_forward_response_ready;
    assign o_scnn_forward_response_tag =
        forward_response_tag[2*TAG_WIDTH +: TAG_WIDTH];
    assign o_scnn_forward_response_code =
        forward_response_code[2*CODE_WIDTH +: CODE_WIDTH];
    assign o_scnn_forward_response_error = forward_response_error[2];

    // Legacy mode preserves the resident-bank interface. Shared mode keeps
    // tensor IDs local to FC because the absolute parameter address is already
    // sufficient for the common scalar backend.
    assign o_fc_native_read_valid = FC_EFFECTIVE_SHARED_BACKEND ?
        1'b0 : fc_native_read_valid;
    assign o_fc_native_read_tensor_id = FC_EFFECTIVE_SHARED_BACKEND ?
        8'd0 : fc_native_read_tensor_id;
    assign o_fc_native_read_parameter_addr = FC_EFFECTIVE_SHARED_BACKEND ?
        {PARAM_ADDR_WIDTH{1'b0}} : fc_native_read_parameter_addr;
    assign o_fc_native_read_tag = FC_EFFECTIVE_SHARED_BACKEND ?
        {TAG_WIDTH{1'b0}} : fc_native_read_tag;
    assign fc_native_read_ready = SHARE_FC_SCNN_OUTPUT_LEARNER ? 1'b0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ? learning_read_ready[0] :
         i_fc_native_read_ready);

    assign fc_native_response_valid = SHARE_FC_SCNN_OUTPUT_LEARNER ? 1'b0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ? learning_response_valid[0] :
         i_fc_native_response_valid);
    assign o_fc_native_response_ready = FC_EFFECTIVE_SHARED_BACKEND ?
        1'b0 : fc_native_response_ready;
    assign fc_native_response_tag = SHARE_FC_SCNN_OUTPUT_LEARNER ? '0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ?
         learning_response_tag[0 +: TAG_WIDTH] : i_fc_native_response_tag);
    assign fc_native_response_code = SHARE_FC_SCNN_OUTPUT_LEARNER ? '0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ?
         learning_response_code[0 +: CODE_WIDTH] : i_fc_native_response_code);
    assign fc_native_response_residual = SHARE_FC_SCNN_OUTPUT_LEARNER ? '0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ?
         learning_response_residual[0 +: RESIDUAL_WIDTH] :
         i_fc_native_response_residual);
    assign fc_native_response_error = SHARE_FC_SCNN_OUTPUT_LEARNER ? 1'b0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ? learning_response_error[0] :
         i_fc_native_response_error);

    assign o_fc_native_write_valid = FC_EFFECTIVE_SHARED_BACKEND ?
        1'b0 : fc_native_write_valid;
    assign o_fc_native_write_tensor_id = FC_EFFECTIVE_SHARED_BACKEND ?
        8'd0 : fc_native_write_tensor_id;
    assign o_fc_native_write_parameter_addr = FC_EFFECTIVE_SHARED_BACKEND ?
        {PARAM_ADDR_WIDTH{1'b0}} : fc_native_write_parameter_addr;
    assign o_fc_native_write_code = FC_EFFECTIVE_SHARED_BACKEND ?
        {CODE_WIDTH{1'b0}} : fc_native_write_code;
    assign o_fc_native_write_residual = FC_EFFECTIVE_SHARED_BACKEND ?
        {RESIDUAL_WIDTH{1'b0}} : fc_native_write_residual;
    assign o_fc_native_write_code_enable = FC_EFFECTIVE_SHARED_BACKEND ?
        1'b0 : fc_native_write_code_enable;
    assign o_fc_native_write_residual_enable =
        FC_EFFECTIVE_SHARED_BACKEND ? 1'b0 :
        fc_native_write_residual_enable;
    assign fc_native_write_ready = SHARE_FC_SCNN_OUTPUT_LEARNER ? 1'b0 :
        (FC_USE_SHARED_PARAMETER_BACKEND ? fc_cache_write_ready :
         i_fc_native_write_ready);
    assign fc_native_parameter_quiescent =
        SHARE_FC_SCNN_OUTPUT_LEARNER ?
        (backend_quiescent && !fc_external_window_busy) :
        (FC_USE_SHARED_PARAMETER_BACKEND ?
         (backend_quiescent && fc_cache_quiescent) :
         i_fc_native_parameter_quiescent);
    assign fc_shared_write_ready = SHARE_FC_SCNN_OUTPUT_LEARNER ? 1'b0 :
        learning_write_ready[0];

    assign o_fc_cache_forward_read_ready =
        FC_EFFECTIVE_SHARED_BACKEND ? forward_read_ready[0] : 1'b0;
    assign o_fc_cache_forward_response_valid =
        FC_EFFECTIVE_SHARED_BACKEND ? forward_response_valid[0] : 1'b0;
    assign o_fc_cache_forward_response_tag = FC_EFFECTIVE_SHARED_BACKEND ?
        forward_response_tag[0 +: TAG_WIDTH] : {TAG_WIDTH{1'b0}};
    assign o_fc_cache_forward_response_code =
        FC_EFFECTIVE_SHARED_BACKEND ?
        forward_response_code[0 +: CODE_WIDTH] : {CODE_WIDTH{1'b0}};
    assign o_fc_cache_forward_response_error =
        FC_EFFECTIVE_SHARED_BACKEND ? forward_response_error[0] : 1'b0;

    generate
        if (FC_USE_SHARED_PARAMETER_BACKEND &&
            !SHARE_FC_SCNN_OUTPUT_LEARNER) begin : gen_fc_cache_coherence
            unified_fc_h40_w0_coherent_write_bridge_v1 #(
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
                .CODE_WIDTH(CODE_WIDTH),
                .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
                .TENSOR_ID_WIDTH(8), .MAX_FEATURES(FC_FEATURES),
                .HIDDEN_STRIDE(FC_MAX_HIDDEN), .HIDDEN_PER_TILE(40),
                .W0_TENSOR_ID(8'h10), .W2_TENSOR_ID(8'h11)
            ) u_fc_w0_cache_coherence (
                .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
                .i_clear_error(i_clear_error),
                .i_w0_parameter_base(FC_W0_BASE),
                .i_feature_count($clog2(FC_FEATURES + 1)'(FC_FEATURES)),
                .i_hidden_count($clog2(FC_MAX_HIDDEN + 1)'(
                    i_fc_hidden_count)),
                .i_write_valid(fc_native_write_valid),
                .o_write_ready(fc_cache_write_ready),
                .i_write_tensor_id(fc_native_write_tensor_id),
                .i_write_parameter_addr(fc_native_write_parameter_addr),
                .i_write_code(fc_native_write_code),
                .i_write_residual(fc_native_write_residual),
                .i_write_code_enable(fc_native_write_code_enable),
                .i_write_residual_enable(fc_native_write_residual_enable),
                .o_backend_write_valid(fc_shared_write_valid),
                .i_backend_write_ready(fc_shared_write_ready),
                .o_backend_write_parameter_addr(
                    fc_shared_write_parameter_addr),
                .o_backend_write_code(fc_shared_write_code),
                .o_backend_write_residual(fc_shared_write_residual),
                .o_backend_write_code_enable(
                    fc_shared_write_code_enable),
                .o_backend_write_residual_enable(
                    fc_shared_write_residual_enable),
                .i_flush_valid(fc_batch_learning_done_raw),
                .o_flush_ready(fc_cache_flush_ready),
                .o_flush_done(fc_cache_flush_done),
                .o_refresh_valid(o_fc_w0_refresh_valid),
                .i_refresh_ready(i_fc_w0_refresh_ready),
                .o_refresh_feature(o_fc_w0_refresh_feature),
                .o_refresh_tile(o_fc_w0_refresh_tile),
                .i_refresh_done(i_fc_w0_refresh_done),
                .i_refresh_error(i_fc_w0_refresh_error),
                .o_quiescent(fc_cache_quiescent),
                .o_sticky_error(fc_cache_protocol_error)
            );

            always_ff @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    fc_runtime_batch_learning_done <= 1'b0;
                    fc_batch_done_pending_q <= 1'b0;
                    fc_cache_flush_violation_q <= 1'b0;
                end else if (i_abort) begin
                    fc_runtime_batch_learning_done <= 1'b0;
                    fc_batch_done_pending_q <= 1'b0;
                end else begin
                    fc_runtime_batch_learning_done <= 1'b0;
                    if (i_clear_error)
                        fc_cache_flush_violation_q <= 1'b0;
                    if (fc_runtime_batch_learning_done_raw) begin
                        fc_batch_done_pending_q <= 1'b1;
                        if (!fc_cache_flush_ready)
                            fc_cache_flush_violation_q <= 1'b1;
                    end
                    if (fc_batch_done_pending_q && fc_cache_flush_done) begin
                        fc_runtime_batch_learning_done <= 1'b1;
                        fc_batch_done_pending_q <= 1'b0;
                    end
                end
            end
        end else begin : gen_fc_legacy_no_cache_coherence
            assign fc_cache_write_ready = 1'b0;
            assign fc_shared_write_valid = 1'b0;
            assign fc_shared_write_parameter_addr = '0;
            assign fc_shared_write_code = '0;
            assign fc_shared_write_residual = '0;
            assign fc_shared_write_code_enable = 1'b0;
            assign fc_shared_write_residual_enable = 1'b0;
            assign fc_cache_quiescent = 1'b1;
            assign fc_cache_flush_ready = 1'b0;
            assign fc_cache_flush_done = 1'b0;
            assign fc_cache_protocol_error = 1'b0;
            assign o_fc_w0_refresh_valid = 1'b0;
            assign o_fc_w0_refresh_feature = '0;
            assign o_fc_w0_refresh_tile = '0;
            always_comb begin
                fc_runtime_batch_learning_done =
                    fc_runtime_batch_learning_done_raw;
                fc_batch_done_pending_q = 1'b0;
                fc_cache_flush_violation_q = 1'b0;
            end
        end
    endgenerate

    generate
        if (SHARE_FC_SCNN_OUTPUT_LEARNER) begin : gen_shared_output_learner
            unified_tp_shared_output_learning_service_v1 #(
                .PROFILE_ID_WIDTH(8), .TENSOR_ID_WIDTH(8),
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
                .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
                .VOTE_WIDTH(16), .MAX_BATCH(16),
                .MAX_SOURCES(SHARED_MAX_SOURCES), .MAX_OUTPUTS(16),
                .MAX_GRADIENT_ELEMENTS(SHARED_MAX_GRADIENT_ELEMENTS),
                .BATCH_INDEX_WIDTH(SHARED_BATCH_INDEX_WIDTH),
                .BATCH_COUNT_WIDTH(SHARED_BATCH_COUNT_WIDTH),
                .SOURCE_INDEX_WIDTH(SHARED_SOURCE_INDEX_WIDTH),
                .SOURCE_COUNT_WIDTH(SHARED_SOURCE_COUNT_WIDTH),
                .OUTPUT_INDEX_WIDTH(SHARED_OUTPUT_INDEX_WIDTH),
                .OUTPUT_COUNT_WIDTH(SHARED_OUTPUT_COUNT_WIDTH)
            ) u_shared_output_learner (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_clear_protocol_error(i_clear_error),
                .i_fc_cfg_valid(fc_external_cfg_valid),
                .o_fc_cfg_ready(fc_external_cfg_ready),
                .i_fc_cfg_profile_id(fc_external_cfg_profile_id),
                .i_fc_cfg_tensor_id(fc_external_cfg_tensor_id),
                .i_fc_cfg_parameter_base(fc_external_cfg_parameter_base),
                .i_fc_cfg_batch_count(SHARED_BATCH_COUNT_WIDTH'(
                    fc_external_cfg_batch_count)),
                .i_fc_cfg_source_count(SHARED_SOURCE_COUNT_WIDTH'(
                    fc_external_cfg_source_count)),
                .i_fc_cfg_output_count(SHARED_OUTPUT_COUNT_WIDTH'(
                    fc_external_cfg_output_count)),
                .i_fc_cfg_output_major(1'b0),
                .i_fc_cfg_source_stride('0),
                .i_fc_cfg_output_stride(PARAM_ADDR_WIDTH'(10)),
                .i_fc_cfg_vote_increment(fc_external_cfg_vote_increment),
                .i_fc_cfg_vote_threshold(fc_external_cfg_vote_threshold),
                .i_fc_cfg_code_min(fc_external_cfg_code_min),
                .i_fc_cfg_code_max(fc_external_cfg_code_max),
                .o_fc_cfg_error(fc_external_cfg_error),
                .o_fc_active(fc_external_active),
                .i_fc_release(fc_external_release),
                .o_fc_release_ready(fc_external_release_ready),
                .i_fc_source_trace_valid(fc_external_source_trace_valid),
                .o_fc_source_trace_ready(fc_external_source_trace_ready),
                .i_fc_source_trace_sample_id(SHARED_BATCH_INDEX_WIDTH'(
                    fc_external_source_trace_sample_id)),
                .i_fc_source_trace_index(SHARED_SOURCE_INDEX_WIDTH'(
                    fc_external_source_trace_index)),
                .i_fc_source_trace_q016(fc_external_source_trace_q016),
                .o_fc_source_trace_error(fc_external_source_trace_error),
                .i_fc_logit_row_valid(fc_external_logit_row_valid),
                .o_fc_logit_row_ready(fc_external_logit_row_ready),
                .i_fc_logit_sample_id(SHARED_BATCH_INDEX_WIDTH'(
                    fc_external_logit_sample_id)),
                .i_fc_logit_label(SHARED_OUTPUT_INDEX_WIDTH'(
                    fc_external_logit_label)),
                .i_fc_logit_valid(fc_external_logit_valid),
                .o_fc_logit_ready(fc_external_logit_ready),
                .i_fc_logit_q6(fc_external_logit_q6),
                .o_fc_logit_error(fc_external_logit_error),
                .o_fc_logit_row_done(fc_external_logit_row_done),
                .i_fc_observation_done(fc_external_observation_done),
                .o_fc_observation_done_ready(
                    fc_external_observation_done_ready),
                .o_fc_window_done(fc_external_window_done),
                .o_fc_window_busy(fc_external_window_busy),
                .o_fc_context_error(fc_external_context_error),
                .o_fc_gradient_overflow(fc_external_gradient_overflow),
                .i_scnn_cfg_valid(scnn_external_cfg_valid),
                .o_scnn_cfg_ready(scnn_external_cfg_ready),
                .i_scnn_cfg_profile_id(scnn_external_cfg_profile_id),
                .i_scnn_cfg_tensor_id(scnn_external_cfg_tensor_id),
                .i_scnn_cfg_parameter_base(
                    scnn_external_cfg_parameter_base),
                .i_scnn_cfg_batch_count(SHARED_BATCH_COUNT_WIDTH'(
                    scnn_external_cfg_batch_count)),
                .i_scnn_cfg_source_count(SHARED_SOURCE_COUNT_WIDTH'(
                    scnn_external_cfg_source_count)),
                .i_scnn_cfg_output_count(SHARED_OUTPUT_COUNT_WIDTH'(
                    scnn_external_cfg_output_count)),
                .i_scnn_cfg_output_major(1'b1),
                .i_scnn_cfg_source_stride(PARAM_ADDR_WIDTH'(SCNN_FEATURES)),
                .i_scnn_cfg_output_stride('0),
                .i_scnn_cfg_vote_increment(
                    scnn_external_cfg_vote_increment),
                .i_scnn_cfg_vote_threshold(
                    scnn_external_cfg_vote_threshold),
                .i_scnn_cfg_code_min(scnn_external_cfg_code_min),
                .i_scnn_cfg_code_max(scnn_external_cfg_code_max),
                .o_scnn_cfg_error(scnn_external_cfg_error),
                .o_scnn_active(scnn_external_active),
                .i_scnn_release(scnn_external_release),
                .o_scnn_release_ready(scnn_external_release_ready),
                .i_scnn_source_trace_valid(
                    scnn_external_source_trace_valid),
                .o_scnn_source_trace_ready(
                    scnn_external_source_trace_ready),
                .i_scnn_source_trace_sample_id(SHARED_BATCH_INDEX_WIDTH'(
                    scnn_external_source_trace_sample_id)),
                .i_scnn_source_trace_index(SHARED_SOURCE_INDEX_WIDTH'(
                    scnn_external_source_trace_index)),
                .i_scnn_source_trace_q016(
                    scnn_external_source_trace_q016),
                .o_scnn_source_trace_error(
                    scnn_external_source_trace_error),
                .i_scnn_logit_row_valid(scnn_external_logit_row_valid),
                .o_scnn_logit_row_ready(scnn_external_logit_row_ready),
                .i_scnn_logit_sample_id(SHARED_BATCH_INDEX_WIDTH'(
                    scnn_external_logit_sample_id)),
                .i_scnn_logit_label(SHARED_OUTPUT_INDEX_WIDTH'(
                    scnn_external_logit_label)),
                .i_scnn_logit_valid(scnn_external_logit_valid),
                .o_scnn_logit_ready(scnn_external_logit_ready),
                .i_scnn_logit_q6(scnn_external_logit_q6),
                .o_scnn_logit_error(scnn_external_logit_error),
                .o_scnn_logit_row_done(scnn_external_logit_row_done),
                .i_scnn_observation_done(scnn_external_observation_done),
                .o_scnn_observation_done_ready(
                    scnn_external_observation_done_ready),
                .o_scnn_window_done(scnn_external_window_done),
                .o_scnn_window_busy(scnn_external_window_busy),
                .o_scnn_context_error(scnn_external_context_error),
                .o_scnn_gradient_overflow(
                    scnn_external_gradient_overflow),
                .o_active_owner(shared_output_owner),
                .o_busy(shared_output_busy),
                .o_protocol_error(shared_output_protocol_error),
                .o_read_valid(shared_output_read_valid),
                .i_read_ready(shared_output_read_ready),
                .o_read_profile_id(shared_output_read_profile_id),
                .o_read_tensor_id(shared_output_read_tensor_id),
                .o_read_parameter_addr(shared_output_read_addr),
                .o_read_tag(shared_output_read_tag),
                .i_read_response_valid(shared_output_response_valid),
                .o_read_response_ready(shared_output_response_ready),
                .i_read_response_code(shared_output_response_code),
                .i_read_response_residual(shared_output_response_residual),
                .i_read_response_error(shared_output_response_error),
                .o_result_valid(shared_output_result_valid),
                .i_result_ready(shared_output_result_ready),
                .o_result_profile_id(shared_output_result_profile_id),
                .o_result_tensor_id(shared_output_result_tensor_id),
                .o_result_parameter_addr(shared_output_result_addr),
                .o_result_tag(shared_output_result_tag),
                .o_result_code(shared_output_result_code),
                .o_result_residual(shared_output_result_residual),
                .o_write_code_enable(shared_output_write_code_enable),
                .o_write_residual_enable(
                    shared_output_write_residual_enable),
                .o_write_code_fire(shared_output_write_code_fire),
                .o_write_residual_fire(shared_output_write_residual_fire),
                .o_dirty(shared_output_dirty),
                .o_saturated(shared_output_saturated),
                .o_update_config_error(shared_output_update_config_error),
                .o_request_error(shared_output_request_error),
                .o_store_error(shared_output_store_error),
                .o_internal_error(shared_output_internal_error)
            );

            unified_tp_two_client_parameter_rmw_arbiter #(
                .PROFILE_ID_WIDTH(8), .TENSOR_ID_WIDTH(8),
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
                .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH)
            ) u_scnn_conv_classifier_rmw (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_c0_read_valid(scnn_private_read_valid),
                .o_c0_read_ready(scnn_private_read_ready),
                .i_c0_read_profile_id(scnn_private_read_profile_id),
                .i_c0_read_tensor_id(scnn_private_read_tensor_id),
                .i_c0_read_parameter_addr(scnn_private_read_addr),
                .i_c0_read_tag(scnn_private_read_tag),
                .o_c0_read_response_valid(scnn_private_response_valid),
                .i_c0_read_response_ready(scnn_private_response_ready),
                .o_c0_read_response_code(scnn_private_response_code),
                .o_c0_read_response_residual(
                    scnn_private_response_residual),
                .o_c0_read_response_error(scnn_private_response_error),
                .i_c0_result_valid(scnn_private_write_valid),
                .o_c0_result_ready(scnn_private_write_ready),
                .i_c0_result_profile_id(scnn_private_write_profile_id),
                .i_c0_result_tensor_id(scnn_private_write_tensor_id),
                .i_c0_result_parameter_addr(scnn_private_write_addr),
                .i_c0_result_tag(scnn_private_write_tag),
                .i_c0_result_code(scnn_private_write_code),
                .i_c0_result_residual(scnn_private_write_residual),
                .i_c0_write_code_enable(scnn_private_write_code_enable),
                .i_c0_write_residual_enable(
                    scnn_private_write_residual_enable),
                .i_c0_write_code_fire(scnn_private_write_valid &&
                    scnn_private_write_ready &&
                    scnn_private_write_code_enable),
                .i_c0_write_residual_fire(scnn_private_write_valid &&
                    scnn_private_write_ready &&
                    scnn_private_write_residual_enable),
                .i_c0_dirty(1'b0), .i_c0_saturated(1'b0),
                .i_c1_read_valid(shared_output_read_valid &&
                    shared_output_owner == 2'd2),
                .o_c1_read_ready(shared_scnn_read_ready),
                .i_c1_read_profile_id(shared_output_read_profile_id),
                .i_c1_read_tensor_id(shared_output_read_tensor_id),
                .i_c1_read_parameter_addr(shared_output_read_addr),
                .i_c1_read_tag(shared_output_read_tag),
                .o_c1_read_response_valid(shared_scnn_response_valid),
                .i_c1_read_response_ready(shared_output_response_ready),
                .o_c1_read_response_code(shared_scnn_response_code),
                .o_c1_read_response_residual(
                    shared_scnn_response_residual),
                .o_c1_read_response_error(shared_scnn_response_error),
                .i_c1_result_valid(shared_output_result_valid &&
                    shared_output_owner == 2'd2),
                .o_c1_result_ready(shared_scnn_result_ready),
                .i_c1_result_profile_id(shared_output_result_profile_id),
                .i_c1_result_tensor_id(shared_output_result_tensor_id),
                .i_c1_result_parameter_addr(shared_output_result_addr),
                .i_c1_result_tag(shared_output_result_tag),
                .i_c1_result_code(shared_output_result_code),
                .i_c1_result_residual(shared_output_result_residual),
                .i_c1_write_code_enable(shared_output_write_code_enable),
                .i_c1_write_residual_enable(
                    shared_output_write_residual_enable),
                .i_c1_write_code_fire(shared_output_write_code_fire),
                .i_c1_write_residual_fire(
                    shared_output_write_residual_fire),
                .i_c1_dirty(shared_output_dirty),
                .i_c1_saturated(shared_output_saturated),
                .o_read_valid(scnn_read_valid),
                .i_read_ready(scnn_read_ready),
                .o_read_profile_id(scnn_merged_read_profile_id),
                .o_read_tensor_id(scnn_merged_read_tensor_id),
                .o_read_parameter_addr(scnn_read_addr),
                .o_read_tag(scnn_read_tag),
                .i_read_response_valid(scnn_response_valid),
                .o_read_response_ready(scnn_response_ready),
                .i_read_response_code(
                    learning_response_code[2*CODE_WIDTH +: CODE_WIDTH]),
                .i_read_response_residual(
                    learning_response_residual[
                        2*RESIDUAL_WIDTH +: RESIDUAL_WIDTH]),
                .i_read_response_error(learning_response_error[2]),
                .o_result_valid(scnn_write_valid),
                .i_result_ready(scnn_write_ready),
                .o_result_profile_id(scnn_merged_write_profile_id),
                .o_result_tensor_id(scnn_merged_write_tensor_id),
                .o_result_parameter_addr(scnn_write_addr),
                .o_result_tag(scnn_write_tag),
                .o_result_code(scnn_write_code),
                .o_result_residual(scnn_write_residual),
                .o_write_code_enable(scnn_write_code_enable),
                .o_write_residual_enable(scnn_write_residual_enable),
                .o_write_code_fire(), .o_write_residual_fire(),
                .o_dirty(), .o_saturated(), .o_busy(scnn_merge_busy),
                .o_protocol_error(scnn_merge_protocol_error)
            );

            assign shared_output_read_ready =
                (shared_output_owner == 2'd1) ? learning_read_ready[0] :
                ((shared_output_owner == 2'd2) ?
                 shared_scnn_read_ready : 1'b0);
            assign shared_output_response_valid =
                (shared_output_owner == 2'd1) ? learning_response_valid[0] :
                ((shared_output_owner == 2'd2) ?
                 shared_scnn_response_valid : 1'b0);
            assign shared_output_response_code =
                (shared_output_owner == 2'd1) ?
                learning_response_code[0 +: CODE_WIDTH] :
                shared_scnn_response_code;
            assign shared_output_response_residual =
                (shared_output_owner == 2'd1) ?
                learning_response_residual[0 +: RESIDUAL_WIDTH] :
                shared_scnn_response_residual;
            assign shared_output_response_error =
                (shared_output_owner == 2'd1) ?
                learning_response_error[0] : shared_scnn_response_error;
            assign shared_output_result_ready =
                (shared_output_owner == 2'd1) ? learning_write_ready[0] :
                ((shared_output_owner == 2'd2) ?
                 shared_scnn_result_ready : 1'b0);
            assign fc_external_update_config_error =
                shared_output_update_config_error;
            assign fc_external_request_error = shared_output_request_error;
            assign fc_external_store_error = shared_output_store_error;
            assign fc_external_internal_error =
                shared_output_internal_error || shared_output_protocol_error;
            assign fc_external_dirty = shared_output_dirty;
            assign fc_external_saturated = shared_output_saturated;
            assign scnn_external_update_config_error =
                shared_output_update_config_error;
            assign scnn_external_request_error =
                shared_output_request_error;
            assign scnn_external_store_error = shared_output_store_error;
            assign scnn_external_internal_error =
                shared_output_internal_error || shared_output_protocol_error ||
                scnn_merge_protocol_error;
        end else begin : gen_private_output_learners
            assign fc_external_cfg_ready = 1'b0;
            assign fc_external_cfg_error = 1'b0;
            assign fc_external_active = 1'b0;
            assign fc_external_release_ready = 1'b0;
            assign fc_external_source_trace_ready = 1'b0;
            assign fc_external_source_trace_error = 1'b0;
            assign fc_external_logit_row_ready = 1'b0;
            assign fc_external_logit_ready = 1'b0;
            assign fc_external_logit_error = 1'b0;
            assign fc_external_logit_row_done = 1'b0;
            assign fc_external_observation_done_ready = 1'b0;
            assign fc_external_window_done = 1'b0;
            assign fc_external_window_busy = 1'b0;
            assign fc_external_context_error = 1'b0;
            assign fc_external_gradient_overflow = 1'b0;
            assign fc_external_update_config_error = 1'b0;
            assign fc_external_request_error = 1'b0;
            assign fc_external_store_error = 1'b0;
            assign fc_external_internal_error = 1'b0;
            assign fc_external_dirty = 1'b0;
            assign fc_external_saturated = 1'b0;
            assign scnn_external_cfg_ready = 1'b0;
            assign scnn_external_cfg_error = 1'b0;
            assign scnn_external_active = 1'b0;
            assign scnn_external_release_ready = 1'b0;
            assign scnn_external_source_trace_ready = 1'b0;
            assign scnn_external_source_trace_error = 1'b0;
            assign scnn_external_logit_row_ready = 1'b0;
            assign scnn_external_logit_ready = 1'b0;
            assign scnn_external_logit_error = 1'b0;
            assign scnn_external_logit_row_done = 1'b0;
            assign scnn_external_observation_done_ready = 1'b0;
            assign scnn_external_window_done = 1'b0;
            assign scnn_external_window_busy = 1'b0;
            assign scnn_external_context_error = 1'b0;
            assign scnn_external_gradient_overflow = 1'b0;
            assign scnn_external_update_config_error = 1'b0;
            assign scnn_external_request_error = 1'b0;
            assign scnn_external_store_error = 1'b0;
            assign scnn_external_internal_error = 1'b0;
            assign shared_output_owner = 2'd0;
            assign shared_output_busy = 1'b0;
            assign shared_output_protocol_error = 1'b0;
            assign shared_output_read_valid = 1'b0;
            assign shared_output_read_ready = 1'b0;
            assign shared_output_read_profile_id = '0;
            assign shared_output_read_tensor_id = '0;
            assign shared_output_read_addr = '0;
            assign shared_output_read_tag = '0;
            assign shared_output_response_valid = 1'b0;
            assign shared_output_response_ready = 1'b0;
            assign shared_output_response_code = '0;
            assign shared_output_response_residual = '0;
            assign shared_output_response_error = 1'b0;
            assign shared_output_result_valid = 1'b0;
            assign shared_output_result_ready = 1'b0;
            assign shared_output_result_profile_id = '0;
            assign shared_output_result_tensor_id = '0;
            assign shared_output_result_addr = '0;
            assign shared_output_result_tag = '0;
            assign shared_output_result_code = '0;
            assign shared_output_result_residual = '0;
            assign shared_output_write_code_enable = 1'b0;
            assign shared_output_write_residual_enable = 1'b0;
            assign shared_output_write_code_fire = 1'b0;
            assign shared_output_write_residual_fire = 1'b0;
            assign shared_output_dirty = 1'b0;
            assign shared_output_saturated = 1'b0;
            assign shared_output_update_config_error = 1'b0;
            assign shared_output_request_error = 1'b0;
            assign shared_output_store_error = 1'b0;
            assign shared_output_internal_error = 1'b0;
            assign shared_scnn_read_ready = 1'b0;
            assign shared_scnn_response_valid = 1'b0;
            assign shared_scnn_response_code = '0;
            assign shared_scnn_response_residual = '0;
            assign shared_scnn_response_error = 1'b0;
            assign shared_scnn_result_ready = 1'b0;
            assign scnn_private_read_ready = scnn_read_ready;
            assign scnn_private_response_valid = scnn_response_valid;
            assign scnn_private_response_code =
                learning_response_code[2*CODE_WIDTH +: CODE_WIDTH];
            assign scnn_private_response_residual =
                learning_response_residual[
                    2*RESIDUAL_WIDTH +: RESIDUAL_WIDTH];
            assign scnn_private_response_error =
                learning_response_error[2];
            assign scnn_private_write_ready = scnn_write_ready;
            assign scnn_read_valid = scnn_private_read_valid;
            assign scnn_read_addr = scnn_private_read_addr;
            assign scnn_read_tag = scnn_private_read_tag;
            assign scnn_response_ready = scnn_private_response_ready;
            assign scnn_write_valid = scnn_private_write_valid;
            assign scnn_write_addr = scnn_private_write_addr;
            assign scnn_write_tag = scnn_private_write_tag;
            assign scnn_write_code = scnn_private_write_code;
            assign scnn_write_residual = scnn_private_write_residual;
            assign scnn_write_code_enable =
                scnn_private_write_code_enable;
            assign scnn_write_residual_enable =
                scnn_private_write_residual_enable;
            assign scnn_merged_read_profile_id = '0;
            assign scnn_merged_read_tensor_id = '0;
            assign scnn_merged_write_profile_id = '0;
            assign scnn_merged_write_tensor_id = '0;
            assign scnn_merge_busy = 1'b0;
            assign scnn_merge_protocol_error = 1'b0;
        end
    endgenerate

    unified_tp_external_train_infer_fabric #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH), .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .FC_MAX_BATCH(16), .FC_BATCH_COUNT_WIDTH(FC_BATCH_COUNT_WIDTH),
        .SRNN_MAX_TIMESTEPS(2),
        .SRNN_TIMESTEP_COUNT_WIDTH(SRNN_TIMESTEP_COUNT_WIDTH),
        .SRNN_TIMESTEP_ID_WIDTH(1)
    ) u_fabric (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_profile_request_valid(i_profile_request_valid),
        .i_profile_request_code(i_profile_request_code),
        .i_profile_learning_enable(i_profile_learning_enable),
        .o_profile_aligned(o_profile_aligned),
        .o_start_allowed(o_start_allowed),
        .o_profile_switch_done(o_profile_switch_done),
        .o_profile_switch_busy(o_profile_switch_busy),
        .o_active_profile_code(o_active_profile_code),
        .o_profile_version(o_profile_version),
        .o_active_learning_enable(o_active_learning_enable),
        .i_run_valid(i_run_valid), .o_run_ready(o_run_ready),
        .i_run_profile_code(i_run_profile_code),
        .o_run_done(o_run_done), .o_run_aborted(o_run_aborted),
        .o_run_busy(o_run_busy),
        .i_parameter_quiescent({
            scnn_quiescent,
            srnn_quiescent,
            fc_execution_quiescent &&
                (FC_EFFECTIVE_SHARED_BACKEND ? 1'b1 :
                 i_fc_native_parameter_quiescent)
        }),
        .i_parameter_error({scnn_error,
            srnn_runtime_error, fc_runtime_error ||
                fc_cache_protocol_error || fc_cache_flush_violation_q}),
        .o_client_acquire_valid(client_acquire_valid),
        .o_client_release_valid(client_release_valid),
        .o_active_client(o_active_client),
        .o_parameter_owner(o_parameter_owner),
        .o_topology_learning_enable(topology_learning_enable),
        .o_client_active(client_active),
        .o_client_quiescent(client_quiescent),
        .o_client_error(client_error),
        .o_control_quiescent(control_quiescent),
        .o_system_quiescent(o_system_quiescent),
        .o_system_error(o_system_error),
        .o_control_error_code(o_control_error_code),
        .o_dispatch_protocol_error(dispatch_protocol_error),
        .i_fc_configured_profile_id(8'h02),
        .i_fc_bridge_configured(o_fc_configured),
        .i_fc_bridge_release_ready(fc_release_ready),
        .i_fc_execution_quiescent(fc_execution_quiescent),
        .i_fc_runtime_error(fc_runtime_error),
        .i_fc_run_sample_count(5'd16),
        .o_fc_sample_start_allowed(fc_sample_start_allowed),
        .i_fc_core_start_valid(fc_core_start_valid),
        .o_fc_core_start_ready(fc_core_start_ready),
        .o_fc_core_done(fc_core_done),
        .o_fc_core_aborted(fc_core_aborted),
        .i_fc_sample_retired(fc_runtime_sample_retired),
        .i_fc_batch_learning_done(fc_runtime_batch_learning_done),
        .o_fc_run_learning_enable(fc_run_learning_enable),
        .o_fc_abort_run(fc_abort_run),
        .i_fc_abort_complete(fc_runtime_abort_complete),
        .i_fc_core_abort(fc_abort_run),
        .o_fc_started_sample_count(fc_started_samples),
        .o_fc_retired_sample_count(fc_retired_samples),
        .i_srnn_coordinator_configured(o_srnn_configured),
        .i_srnn_timestep_count(2'd2),
        .i_srnn_sequence_version_base(i_srnn_cfg_commit_version),
        .o_srnn_timestep_start(srnn_runtime_timestep_start),
        .i_srnn_timestep_start_ready(srnn_runtime_timestep_start_ready),
        .o_srnn_timestep_id(srnn_runtime_timestep_id),
        .o_srnn_sequence_version(srnn_runtime_sequence_version),
        .o_srnn_timestep_learning_enable(srnn_runtime_learning_enable),
        .o_srnn_abort_timestep(srnn_runtime_abort_timestep),
        .i_srnn_abort_ready(srnn_runtime_abort_ready),
        .i_srnn_timestep_done(srnn_runtime_timestep_done),
        .i_srnn_timestep_aborted(srnn_runtime_timestep_aborted),
        .i_srnn_timestep_busy(srnn_runtime_busy),
        .i_srnn_timestep_error(srnn_runtime_error),
        .i_srnn_core_start_valid(srnn_core_start_valid),
        .o_srnn_core_start_ready(srnn_core_start_ready),
        .o_srnn_core_done(srnn_core_done),
        .o_srnn_core_aborted(srnn_core_aborted),
        .i_srnn_core_abort(srnn_core_abort),
        .o_srnn_completed_timestep_count(srnn_completed_timesteps),
        .i_scnn_configured_profile_id(8'h20),
        .i_scnn_native_configured(i_scnn_configured),
        .o_scnn_native_start_allowed(scnn_start_allowed),
        .o_scnn_native_run_start_valid(scnn_run_start_valid),
        .i_scnn_native_run_start_ready(scnn_run_start_ready),
        .o_scnn_native_abort(scnn_abort),
        .i_scnn_native_run_done(scnn_run_done),
        .i_scnn_native_run_aborted(scnn_run_aborted),
        .i_scnn_native_busy(scnn_busy),
        .i_scnn_native_quiescent(scnn_quiescent),
        .i_scnn_native_error(scnn_error),
        .i_scnn_core_start_valid(scnn_core_start_valid),
        .o_scnn_core_start_ready(scnn_core_start_ready),
        .o_scnn_core_done(scnn_core_done),
        .o_scnn_core_aborted(scnn_core_aborted),
        .i_scnn_core_abort(scnn_core_abort),
        .o_scnn_run_learning_enable(scnn_run_learning_enable),
        .o_core_start(o_core_start), .i_core_start_ready(i_core_start_ready),
        .o_core_profile_code(o_core_profile_code),
        .o_core_abort(o_core_abort),
        .i_core_sample_done(i_core_sample_done),
        .i_core_global_idle(i_core_global_idle),
        .i_core_error(i_core_error),
        .o_core_active_source(o_core_active_source),
        .o_core_busy(), .o_core_quiescent(), .o_core_protocol_error(),
        .i_load_valid({i_scnn_parameter_load_valid,
            srnn_store_load_valid, 1'b0}),
        .o_load_ready(load_ready),
        .i_load_addr({i_scnn_parameter_load_addr,
            srnn_store_load_addr, {PARAM_ADDR_WIDTH{1'b0}}}),
        .i_load_code({i_scnn_parameter_load_code,
            srnn_store_load_code, {CODE_WIDTH{1'b0}}}),
        .i_load_residual({i_scnn_parameter_load_residual,
            srnn_store_load_residual, {RESIDUAL_WIDTH{1'b0}}}),
        .i_learning_read_valid({scnn_read_valid, srnn_store_read_valid,
            SHARE_FC_SCNN_OUTPUT_LEARNER ?
                (shared_output_owner == 2'd1 && shared_output_read_valid) :
                (FC_USE_SHARED_PARAMETER_BACKEND &&
                 fc_native_read_valid)}),
        .o_learning_read_ready(learning_read_ready),
        .i_learning_read_addr({scnn_read_addr, srnn_store_read_addr,
            SHARE_FC_SCNN_OUTPUT_LEARNER ? shared_output_read_addr :
                fc_native_read_parameter_addr}),
        .i_learning_read_tag({scnn_read_tag, srnn_store_read_tag,
            SHARE_FC_SCNN_OUTPUT_LEARNER ? shared_output_read_tag :
                fc_native_read_tag}),
        .o_learning_response_valid(learning_response_valid),
        .i_learning_response_ready({scnn_response_ready,
            srnn_store_response_ready,
            SHARE_FC_SCNN_OUTPUT_LEARNER ?
                (shared_output_owner == 2'd1 &&
                 shared_output_response_ready) :
                (FC_USE_SHARED_PARAMETER_BACKEND &&
                 fc_native_response_ready)}),
        .o_learning_response_tag(learning_response_tag),
        .o_learning_response_code(learning_response_code),
        .o_learning_response_residual(learning_response_residual),
        .o_learning_response_error(learning_response_error),
        .i_learning_write_valid({scnn_write_valid,
            srnn_store_write_valid,
            SHARE_FC_SCNN_OUTPUT_LEARNER ?
                (shared_output_owner == 2'd1 &&
                 shared_output_result_valid) :
                (FC_USE_SHARED_PARAMETER_BACKEND &&
                 fc_shared_write_valid)}),
        .o_learning_write_ready(learning_write_ready),
        .i_learning_write_addr({scnn_write_addr, srnn_store_write_addr,
            SHARE_FC_SCNN_OUTPUT_LEARNER ? shared_output_result_addr :
                fc_shared_write_parameter_addr}),
        .i_learning_write_code({scnn_write_code, srnn_store_write_code,
            SHARE_FC_SCNN_OUTPUT_LEARNER ? shared_output_result_code :
                fc_shared_write_code}),
        .i_learning_write_residual({scnn_write_residual,
            srnn_store_write_residual,
            SHARE_FC_SCNN_OUTPUT_LEARNER ?
                shared_output_result_residual : fc_shared_write_residual}),
        .i_learning_write_code_enable({scnn_write_code_enable,
            srnn_store_write_code_enable,
            SHARE_FC_SCNN_OUTPUT_LEARNER ?
                shared_output_write_code_enable :
                (FC_USE_SHARED_PARAMETER_BACKEND &&
                 fc_shared_write_code_enable)}),
        .i_learning_write_residual_enable({scnn_write_residual_enable,
            srnn_store_write_residual_enable,
            SHARE_FC_SCNN_OUTPUT_LEARNER ?
                shared_output_write_residual_enable :
                (FC_USE_SHARED_PARAMETER_BACKEND &&
                 fc_shared_write_residual_enable)}),
        .i_forward_read_valid({scnn_forward_valid,
            srnn_store_forward_valid,
            FC_EFFECTIVE_SHARED_BACKEND &&
                i_fc_cache_forward_read_valid}),
        .o_forward_read_ready(forward_read_ready),
        .i_forward_read_addr({scnn_forward_addr,
            srnn_store_forward_addr, i_fc_cache_forward_read_addr}),
        .i_forward_read_tag({scnn_forward_tag, srnn_store_forward_tag,
            i_fc_cache_forward_read_tag}),
        .o_forward_response_valid(forward_response_valid),
        .i_forward_response_ready({scnn_forward_response_ready,
            srnn_store_forward_response_ready,
            FC_EFFECTIVE_SHARED_BACKEND &&
                i_fc_cache_forward_response_ready}),
        .o_forward_response_tag(forward_response_tag),
        .o_forward_response_code(forward_response_code),
        .o_forward_response_error(forward_response_error),
        .o_store_load_valid(o_store_load_valid),
        .i_store_load_ready(i_store_load_ready),
        .o_store_load_addr(o_store_load_addr),
        .o_store_load_code(o_store_load_code),
        .o_store_load_residual(o_store_load_residual),
        .o_store_learning_read_valid(o_store_learning_read_valid),
        .i_store_learning_read_ready(i_store_learning_read_ready),
        .o_store_learning_read_addr(o_store_learning_read_addr),
        .o_store_learning_read_tag(o_store_learning_read_tag),
        .i_store_learning_response_valid(i_store_learning_response_valid),
        .o_store_learning_response_ready(o_store_learning_response_ready),
        .i_store_learning_response_tag(i_store_learning_response_tag),
        .i_store_learning_response_code(i_store_learning_response_code),
        .i_store_learning_response_residual(
            i_store_learning_response_residual),
        .i_store_learning_response_error(i_store_learning_response_error),
        .o_store_learning_write_valid(o_store_learning_write_valid),
        .i_store_learning_write_ready(i_store_learning_write_ready),
        .o_store_learning_write_addr(o_store_learning_write_addr),
        .o_store_learning_write_code(o_store_learning_write_code),
        .o_store_learning_write_residual(o_store_learning_write_residual),
        .o_store_learning_write_code_enable(
            o_store_learning_write_code_enable),
        .o_store_learning_write_residual_enable(
            o_store_learning_write_residual_enable),
        .o_store_forward_read_valid(o_store_forward_read_valid),
        .i_store_forward_read_ready(i_store_forward_read_ready),
        .o_store_forward_read_addr(o_store_forward_read_addr),
        .o_store_forward_read_tag(o_store_forward_read_tag),
        .i_store_forward_response_valid(i_store_forward_response_valid),
        .o_store_forward_response_ready(o_store_forward_response_ready),
        .i_store_forward_response_tag(i_store_forward_response_tag),
        .i_store_forward_response_code(i_store_forward_response_code),
        .i_store_forward_response_error(i_store_forward_response_error),
        .i_store_quiescent(i_store_quiescent),
        .i_store_protocol_error(i_store_protocol_error),
        .o_backend_quiescent(backend_quiescent),
        .o_backend_error(backend_error),
        .o_owner_encoding_error(owner_encoding_error),
        .o_router_protocol_error(router_protocol_error)
    );

    unified_fc_tp_actual_runtime_binding #(
        .ENABLE_W0_LEARNING(FC_LEARN_W0),
        .USE_EXTERNAL_W2_LEARNER(SHARE_FC_SCNN_OUTPUT_LEARNER),
        .MAX_BATCH(16), .MAX_HIDDEN(FC_MAX_HIDDEN),
        .MAX_FEATURES(FC_FEATURES), .MAX_TARGET_DIMENSION(10),
        .MAX_OUTPUTS(10), .MAX_EVENTS(512),
        .MAX_WINDOWS_PER_SAMPLE(512),
        .W2_MAX_GRADIENT_ELEMENTS(FC_MAX_HIDDEN * 10),
        .W0_PARAMETER_HIDDEN_STRIDE(FC_MAX_HIDDEN),
        .W2_PARAMETER_OUTPUT_STRIDE(10)
    ) u_fc (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_cfg_valid(i_fc_cfg_valid), .o_cfg_ready(o_fc_cfg_ready),
        .i_cfg_profile_id(8'h02), .i_cfg_w0_tensor_id(8'h10),
        .i_cfg_w0_parameter_base(FC_W0_BASE),
        .i_cfg_w0_vote_increment(16'd1),
        .i_cfg_w0_vote_threshold(16'd1),
        .i_cfg_w2_tensor_id(8'h11),
        .i_cfg_w2_parameter_base(FC_W2_BASE),
        .i_cfg_w2_vote_increment(16'd1),
        .i_cfg_w2_vote_threshold(16'd1),
        .i_cfg_batch_count(5'd16),
        .i_cfg_hidden_count(i_fc_hidden_count),
        .i_cfg_feature_count(12'(FC_FEATURES)),
        .i_cfg_target_dimension(4'd10), .i_cfg_output_count(4'd10),
        .i_cfg_w0_code_min(-16'sd15), .i_cfg_w0_code_max(16'sd15),
        .i_cfg_w2_code_min(-16'sd127), .i_cfg_w2_code_max(16'sd127),
        .i_cfg_vth(18'sd1024), .i_cfg_logit_to_q6_scale(24'd65536),
        .o_cfg_error(o_fc_cfg_error), .o_configured(o_fc_configured),
        .i_release(client_release_valid[0]),
        .o_release_ready(fc_release_ready),
        .i_learning_enable(fc_run_learning_enable),
        .i_window_valid(i_fc_window_valid && o_active_client[0]),
        .o_window_ready(o_fc_window_ready),
        .i_record_valid(i_fc_record_valid),
        .i_window_sample_id(i_fc_window_sample_id),
        .i_window_timestep(i_fc_window_timestep),
        .i_boundary_mask(i_fc_boundary_mask),
        .i_window_sample_last(i_fc_window_sample_last),
        .i_addr_flat(i_fc_addr_flat), .i_data_flat(i_fc_data_flat),
        .i_snapshot_label(i_fc_snapshot_label),
        .i_snapshot_spikes_flat(i_fc_snapshot_spikes_flat),
        .i_snapshot_trace_flat(i_fc_snapshot_trace_flat),
        .i_snapshot_vmem_flat(i_fc_snapshot_vmem_flat),
        .i_snapshot_logits_flat(i_fc_snapshot_logits_flat),
        .i_target_hidden_trace_valid(i_fc_target_hidden_trace_valid),
        .o_target_hidden_trace_ready(o_fc_target_hidden_trace_ready),
        .i_target_hidden_trace_sample(i_fc_target_trace_sample),
        .i_target_hidden_trace_hidden(i_fc_target_hidden_index),
        .i_target_hidden_trace_q016(i_fc_target_hidden_trace_q016),
        .i_target_input_trace_valid(i_fc_target_input_trace_valid),
        .o_target_input_trace_ready(o_fc_target_input_trace_ready),
        .i_target_input_trace_sample(i_fc_target_trace_sample),
        .i_target_input_trace_index(i_fc_target_input_index),
        .i_target_input_trace_q016(i_fc_target_input_trace_q016),
        .i_target_done(i_fc_target_done),
        .o_target_done_ready(o_fc_target_done_ready),
        .o_window_commit_valid(o_fc_learning_commit),
        .i_window_commit_ready(1'b1),
        .o_batch_learning_done(fc_batch_learning_done_raw),
        .o_native_read_valid(fc_native_read_valid),
        .i_native_read_ready(fc_native_read_ready),
        .o_native_read_tensor_id(fc_native_read_tensor_id),
        .o_native_read_parameter_addr(fc_native_read_parameter_addr),
        .o_native_read_tag(fc_native_read_tag),
        .i_native_response_valid(fc_native_response_valid),
        .o_native_response_ready(fc_native_response_ready),
        .i_native_response_tag(fc_native_response_tag),
        .i_native_response_code(fc_native_response_code),
        .i_native_response_residual(fc_native_response_residual),
        .i_native_response_error(fc_native_response_error),
        .o_native_write_valid(fc_native_write_valid),
        .i_native_write_ready(fc_native_write_ready),
        .o_native_write_tensor_id(fc_native_write_tensor_id),
        .o_native_write_parameter_addr(fc_native_write_parameter_addr),
        .o_native_write_code(fc_native_write_code),
        .o_native_write_residual(fc_native_write_residual),
        .o_native_write_code_enable(fc_native_write_code_enable),
        .o_native_write_residual_enable(fc_native_write_residual_enable),
        .i_native_parameter_quiescent(fc_native_parameter_quiescent),
        .o_external_cfg_valid(fc_external_cfg_valid),
        .i_external_cfg_ready(fc_external_cfg_ready),
        .o_external_cfg_profile_id(fc_external_cfg_profile_id),
        .o_external_cfg_tensor_id(fc_external_cfg_tensor_id),
        .o_external_cfg_parameter_base(fc_external_cfg_parameter_base),
        .o_external_cfg_batch_count(fc_external_cfg_batch_count),
        .o_external_cfg_source_count(fc_external_cfg_source_count),
        .o_external_cfg_output_count(fc_external_cfg_output_count),
        .o_external_cfg_vote_increment(fc_external_cfg_vote_increment),
        .o_external_cfg_vote_threshold(fc_external_cfg_vote_threshold),
        .o_external_cfg_code_min(fc_external_cfg_code_min),
        .o_external_cfg_code_max(fc_external_cfg_code_max),
        .i_external_cfg_error(fc_external_cfg_error),
        .i_external_active(fc_external_active),
        .o_external_release(fc_external_release),
        .i_external_release_ready(fc_external_release_ready),
        .o_external_source_trace_valid(fc_external_source_trace_valid),
        .i_external_source_trace_ready(fc_external_source_trace_ready),
        .o_external_source_trace_sample_id(
            fc_external_source_trace_sample_id),
        .o_external_source_trace_index(fc_external_source_trace_index),
        .o_external_source_trace_q016(fc_external_source_trace_q016),
        .i_external_source_trace_error(fc_external_source_trace_error),
        .o_external_logit_row_valid(fc_external_logit_row_valid),
        .i_external_logit_row_ready(fc_external_logit_row_ready),
        .o_external_logit_sample_id(fc_external_logit_sample_id),
        .o_external_logit_label(fc_external_logit_label),
        .o_external_logit_valid(fc_external_logit_valid),
        .i_external_logit_ready(fc_external_logit_ready),
        .o_external_logit_q6(fc_external_logit_q6),
        .i_external_logit_error(fc_external_logit_error),
        .i_external_logit_row_done(fc_external_logit_row_done),
        .o_external_observation_done(fc_external_observation_done),
        .i_external_observation_done_ready(
            fc_external_observation_done_ready),
        .i_external_window_done(fc_external_window_done),
        .i_external_window_busy(fc_external_window_busy),
        .i_external_context_error(fc_external_context_error),
        .i_external_gradient_overflow(fc_external_gradient_overflow),
        .i_external_update_config_error(
            fc_external_update_config_error),
        .i_external_request_error(fc_external_request_error),
        .i_external_store_error(fc_external_store_error),
        .i_external_internal_error(fc_external_internal_error),
        .i_external_dirty(fc_external_dirty),
        .i_external_saturated(fc_external_saturated),
        .i_runtime_sample_start_allowed(fc_sample_start_allowed),
        .i_runtime_abort(fc_abort_run),
        .o_runtime_abort_complete(fc_runtime_abort_complete),
        .o_runtime_sample_retired(fc_runtime_sample_retired),
        .o_runtime_batch_learning_done(
            fc_runtime_batch_learning_done_raw),
        .o_execution_quiescent(fc_execution_quiescent),
        .o_runtime_error(fc_runtime_error),
        .o_core_event_window_valid(o_fc_event_window_valid),
        .i_core_event_window_ready(i_fc_event_window_ready),
        .o_core_event_record_valid(o_fc_event_record_valid),
        .o_core_event_window_sample_id(o_fc_event_window_sample_id),
        .o_core_event_window_timestep(o_fc_event_window_timestep),
        .o_core_event_boundary_mask(o_fc_event_boundary_mask),
        .o_core_event_window_sample_last(o_fc_event_window_sample_last),
        .o_core_event_addr_flat(o_fc_event_addr_flat),
        .o_core_event_data_flat(o_fc_event_data_flat),
        .o_core_start_valid(fc_core_start_valid),
        .i_core_start_ready(fc_core_start_ready),
        .o_core_sample_id(o_fc_core_sample_id),
        .o_core_sample_state_reset(o_fc_core_sample_state_reset),
        .o_core_fc_clear_state(o_fc_core_clear_state),
        .o_core_sample_last(o_fc_core_sample_last),
        .i_core_sample_done(fc_core_done), .i_core_aborted(fc_core_aborted),
        .o_window_busy(o_fc_learning_busy)
    );

    unified_srnn_m2_actual_runtime_binding #(
        .USE_INTERNAL_STORE(1'b0), .BATCH_MAX(2), .TIMESTEP_MAX(2),
        .HIDDEN_MAX(SRNN_HIDDEN), .BACKBONE_MAX(SRNN_BACKBONE),
        .CLASS_MAX(SRNN_CLASSES), .SOURCE_VECTOR_MAX(480)
    ) u_srnn (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_cfg_valid(i_srnn_cfg_valid), .o_cfg_ready(o_srnn_cfg_ready),
        .i_cfg_profile_id(8'h10), .i_cfg_wh1_tensor_id(8'h20),
        .i_cfg_wh1_base(SRNN_WH1_BASE), .i_cfg_wh1_increment(16'd1),
        .i_cfg_wh1_threshold(16'd1), .i_cfg_wout_tensor_id(8'h21),
        .i_cfg_wout_base(SRNN_WOUT_BASE), .i_cfg_wout_increment(16'd1),
        .i_cfg_wout_threshold(16'd1), .i_cfg_code_min(16'sh8000),
        .i_cfg_code_max(16'sh7fff), .i_cfg_batch_count(8'd2),
        .i_cfg_timestep_count(2'd2),
        .i_cfg_hidden_count(9'(SRNN_HIDDEN)),
        .i_cfg_backbone_count(10'(SRNN_BACKBONE)),
        .i_cfg_class_count(5'(SRNN_CLASSES)),
        .i_cfg_wout_element_count(24'(SRNN_HIDDEN * SRNN_CLASSES)),
        .i_cfg_commit_version(i_srnn_cfg_commit_version),
        .i_cfg_check_commit_version(1'b1),
        .o_configured(o_srnn_configured), .o_cfg_error(o_srnn_cfg_error),
        .i_parameter_load_valid(i_srnn_parameter_load_valid),
        .o_parameter_load_ready(o_srnn_parameter_load_ready),
        .i_parameter_load_addr(i_srnn_parameter_load_addr),
        .i_parameter_load_code(i_srnn_parameter_load_code),
        .i_parameter_load_residual(i_srnn_parameter_load_residual),
        .i_target_label_load_start(i_srnn_target_label_load_start),
        .o_target_label_load_start_ready(
            o_srnn_target_label_load_start_ready),
        .i_target_label_valid(i_srnn_target_label_valid),
        .o_target_label_ready(o_srnn_target_label_ready),
        .i_target_label_sample_id(i_srnn_target_label_sample),
        .i_target_label_class_id(i_srnn_target_label_class),
        .i_target_label_last(i_srnn_target_label_last),
        .o_target_label_load_done(o_srnn_target_label_load_done),
        .i_target_spike_word_write_valid(
            i_srnn_target_spike_word_write_valid),
        .o_target_spike_word_write_ready(
            o_srnn_target_spike_word_write_ready),
        .i_target_spike_word_class_id(i_srnn_target_spike_word_class),
        .i_target_spike_word_timestep_id(
            i_srnn_target_spike_word_timestep),
        .i_target_spike_word_id(i_srnn_target_spike_word_id),
        .i_target_spike_word_data(i_srnn_target_spike_word_data),
        .i_target_trace_write_valid(i_srnn_target_trace_write_valid),
        .o_target_trace_write_ready(o_srnn_target_trace_write_ready),
        .i_target_trace_class_id(i_srnn_target_trace_class),
        .i_target_trace_timestep_id(i_srnn_target_trace_timestep),
        .i_target_trace_backbone_id(i_srnn_target_trace_backbone),
        .i_target_trace_q14(i_srnn_target_trace_q14),
        .i_runtime_client_active(client_active[1]),
        .i_runtime_timestep_start(srnn_runtime_timestep_start),
        .o_runtime_timestep_start_ready(srnn_runtime_timestep_start_ready),
        .i_runtime_timestep_id(srnn_runtime_timestep_id),
        .i_runtime_sequence_version(srnn_runtime_sequence_version),
        .i_runtime_learning_enable(srnn_runtime_learning_enable),
        .i_runtime_abort_timestep(srnn_runtime_abort_timestep),
        .o_runtime_abort_ready(srnn_runtime_abort_ready),
        .o_runtime_timestep_done(srnn_runtime_timestep_done),
        .o_runtime_timestep_aborted(srnn_runtime_timestep_aborted),
        .o_runtime_busy(srnn_runtime_busy),
        .o_runtime_error(srnn_runtime_error),
        .i_backbone_sample_start_valid(i_srnn_backbone_sample_start_valid),
        .o_backbone_sample_start_ready(o_srnn_backbone_sample_start_ready),
        .o_backbone_sample_id(o_srnn_backbone_sample_id),
        .o_core_start_valid(srnn_core_start_valid),
        .i_core_start_ready(srnn_core_start_ready),
        .i_core_done(srnn_core_done), .i_core_aborted(srnn_core_aborted),
        .o_core_abort(srnn_core_abort),
        .i_commit_valid(i_srnn_commit_valid),
        .o_commit_ready(o_srnn_commit_ready),
        .i_commit_version(i_srnn_commit_version),
        .i_commit_sample_last(i_srnn_commit_sample_last),
        .i_commit_spikes(i_srnn_commit_spikes),
        .i_forward_code_ready(1'b1),
        .i_hidden_observation_ready(1'b1), .i_logit_ready(1'b1),
        .o_learning_commit_pulse(o_srnn_learning_commit),
        .o_detach_pulse(o_srnn_detach),
        .o_update_count(o_srnn_update_count), .o_quiescent(srnn_quiescent),
        .o_store_load_valid(srnn_store_load_valid),
        .i_store_load_ready(srnn_store_load_ready),
        .o_store_load_addr(srnn_store_load_addr),
        .o_store_load_code(srnn_store_load_code),
        .o_store_load_residual(srnn_store_load_residual),
        .o_store_learning_read_valid(srnn_store_read_valid),
        .i_store_learning_read_ready(srnn_store_read_ready),
        .o_store_learning_read_addr(srnn_store_read_addr),
        .o_store_learning_read_tag(srnn_store_read_tag),
        .i_store_learning_response_valid(srnn_store_response_valid),
        .o_store_learning_response_ready(srnn_store_response_ready),
        .i_store_learning_response_tag(
            learning_response_tag[TAG_WIDTH +: TAG_WIDTH]),
        .i_store_learning_response_code(
            learning_response_code[CODE_WIDTH +: CODE_WIDTH]),
        .i_store_learning_response_residual(
            learning_response_residual[RESIDUAL_WIDTH +: RESIDUAL_WIDTH]),
        .i_store_learning_response_error(learning_response_error[1]),
        .o_store_learning_write_valid(srnn_store_write_valid),
        .i_store_learning_write_ready(srnn_store_write_ready),
        .o_store_learning_write_addr(srnn_store_write_addr),
        .o_store_learning_write_code(srnn_store_write_code),
        .o_store_learning_write_residual(srnn_store_write_residual),
        .o_store_learning_write_code_enable(srnn_store_write_code_enable),
        .o_store_learning_write_residual_enable(
            srnn_store_write_residual_enable),
        .o_store_forward_read_valid(srnn_store_forward_valid),
        .i_store_forward_read_ready(srnn_store_forward_ready),
        .o_store_forward_read_addr(srnn_store_forward_addr),
        .o_store_forward_read_tag(srnn_store_forward_tag),
        .i_store_forward_response_valid(srnn_store_forward_response_valid),
        .o_store_forward_response_ready(srnn_store_forward_response_ready),
        .i_store_forward_response_tag(
            forward_response_tag[TAG_WIDTH +: TAG_WIDTH]),
        .i_store_forward_response_code(
            forward_response_code[CODE_WIDTH +: CODE_WIDTH]),
        .i_store_forward_response_error(forward_response_error[1]),
        .i_store_protocol_error(router_protocol_error ||
            i_store_protocol_error)
    );

    unified_scnn_tp_online_system_binding #(
        .MAX_BATCH(2), .MAX_TIMESTEPS(2),
        .MAX_INPUT_CHANNELS(8),
        .MAX_OUTPUT_CHANNELS(12),
        .OBS_INPUT_CHANNELS(SCNN_CHANNELS),
        .OBS_OUTPUT_CHANNELS(SCNN_CHANNELS),
        .SPATIAL(4), .POOLED_SPATIAL(2), .KERNEL_SIZE(3),
        .MAX_FEATURES(SCNN_FEATURES), .MAX_CLASSES(SCNN_CLASSES),
        .LANES(LANES), .ENGINES(ENGINES),
        .USE_SAMPLE_CONTEXT(1'b1),
        .USE_CLASSIFIER_STUDENT_TRACE(1'b0),
        .USE_INTERNAL_MODULATION(1'b1), .INTERNAL_MODULATION_BATCH(2),
        .USE_EXTERNAL_OUTPUT_LEARNER(
            SHARE_FC_SCNN_OUTPUT_LEARNER),
        .INTERNAL_MODULATION_TRACE_SPATIAL_MAJOR(1'b1)
    ) u_scnn (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error), .i_abort(scnn_abort),
        .i_profile_aligned(o_profile_aligned && o_parameter_owner[2]),
        .i_start_allowed(scnn_start_allowed),
        .i_scnn_parameter_owner(o_parameter_owner[2]),
        .i_scnn_learning_enable(scnn_run_learning_enable),
        .i_system_profile_id(8'h20),
        .i_run_start_valid(scnn_run_start_valid),
        .o_run_start_ready(scnn_run_start_ready),
        .i_batch_count(2'd2), .i_timestep_count(2'd2),
        .o_target_start_valid(o_scnn_target_start_valid),
        .i_target_start_ready(i_scnn_target_start_ready),
        .o_target_sample(o_scnn_target_sample),
        .o_target_timestep(o_scnn_target_timestep),
        .i_target_done(i_scnn_target_done),
        .i_target_busy(i_scnn_target_busy),
        .i_target_error(i_scnn_target_error),
        .o_core_start_valid(scnn_core_start_valid),
        .i_core_start_ready(scnn_core_start_ready),
        .o_core_sample(o_scnn_core_sample),
        .o_core_timestep(o_scnn_core_timestep),
        .o_core_post_forward(o_scnn_core_post_forward),
        .o_classifier_capture_enable(o_scnn_classifier_capture_enable),
        .i_selected_conv_done(i_scnn_selected_conv_done),
        .i_core_sample_done(scnn_core_done),
        // The physical core is shared.  Its reset/configuration activity must
        // not make an unowned SCNN client look active and block acquisition;
        // once SCNN owns the core, release still waits for true core idle.
        .i_core_quiescent(i_core_global_idle || !o_parameter_owner[2]),
        .i_core_error(i_core_error), .o_core_abort(scnn_core_abort),
        .i_profile_id(8'h20), .i_conv_tensor_id(8'h08),
        .i_classifier_tensor_id(8'h12),
        .i_total_input_channels(10'(SCNN_CHANNELS)),
        .i_total_output_channels(10'(SCNN_CHANNELS)),
        .i_conv_parameter_base(SCNN_CONV8_BASE),
        .i_conv_weight_q16_base(SCNN_CONV8_BASE),
        .i_classifier_parameter_base(SCNN_CLASSIFIER_BASE),
        .i_classifier_feature_count(12'(SCNN_FEATURES)),
        .i_classifier_class_count(4'(SCNN_CLASSES)),
        .i_classifier_trace_leak_valid(1'b1),
        .i_classifier_trace_leak_q8_8(16'sd250),
        .i_conv_vote_increment(16'd1), .i_conv_vote_threshold(16'd32),
        .i_classifier_vote_increment(16'd1),
        .i_classifier_vote_threshold(16'd40),
        .i_code_min(16'sh8000), .i_code_max(16'sh7fff),
        .i_sample_version(16'h7300),
        .i_context_load_valid(1'b0), .i_context_model_version(16'd0),
        .i_context_adapt_version(16'd0), .i_restore_complete(1'b0),
        .i_sample_context_valid(i_scnn_sample_context_valid),
        .o_sample_context_ready(o_scnn_sample_context_ready),
        .i_sample_context_target(i_scnn_sample_context_target),
        .i_sample_context_sample(i_scnn_sample_context_sample),
        .i_sample_context_version(i_scnn_sample_context_version),
        .i_input_valid(i_scnn_input_valid), .o_input_ready(o_scnn_input_ready),
        .i_input_target(i_scnn_input_target),
        .i_input_sample(i_scnn_input_sample),
        .i_input_timestep(i_scnn_input_timestep),
        .i_input_sample_version(i_scnn_input_sample_version),
        .i_input_channel(i_scnn_input_channel),
        .i_input_spatial(i_scnn_input_spatial),
        .i_input_q8_8(i_scnn_input_q8_8),
        .i_trace_valid(i_scnn_trace_valid), .o_trace_ready(o_scnn_trace_ready),
        .i_trace_target(i_scnn_trace_target),
        .i_trace_sample(i_scnn_trace_sample),
        .i_trace_timestep(i_scnn_trace_timestep),
        .i_trace_sample_version(i_scnn_trace_sample_version),
        .i_trace_channel(i_scnn_trace_channel),
        .i_trace_spatial(i_scnn_trace_spatial),
        .i_trace_q8_8(i_scnn_trace_q8_8),
        .i_surrogate_valid(i_scnn_surrogate_valid),
        .o_surrogate_ready(o_scnn_surrogate_ready),
        .i_surrogate_target(i_scnn_surrogate_target),
        .i_surrogate_sample(i_scnn_surrogate_sample),
        .i_surrogate_timestep(i_scnn_surrogate_timestep),
        .i_surrogate_sample_version(i_scnn_surrogate_sample_version),
        .i_surrogate_channel(i_scnn_surrogate_channel),
        .i_surrogate_spatial(i_scnn_surrogate_spatial),
        .i_surrogate_q8_8(i_scnn_surrogate_q8_8),
        .i_modulation_valid(1'b0), .i_modulation_row(1'b0),
        .i_modulation_col(1'b0), .i_modulation_q8_8(16'sd0),
        .i_feature_trace_wave_valid(i_scnn_feature_wave_valid),
        .o_feature_trace_wave_ready(o_scnn_feature_wave_ready),
        .i_feature_trace_wave_mask(i_scnn_feature_wave_mask),
        .i_feature_trace_wave_base(i_scnn_feature_wave_base),
        .i_feature_trace_wave_q8_8_flat(i_scnn_feature_wave_q8_8_flat),
        .i_feature_trace_wave_version(i_scnn_feature_wave_version),
        .i_feature_trace_wave_frame_index(i_scnn_feature_wave_frame),
        .i_feature_trace_wave_first(i_scnn_feature_wave_first),
        .i_feature_trace_wave_last(i_scnn_feature_wave_last),
        .i_feature_trace_wave_sample_reset(i_scnn_feature_wave_sample_reset),
        .i_feature_trace_wave_sample_last(i_scnn_feature_wave_sample_last),
        .i_sample_label(i_scnn_sample_label),
        .i_logit_wave_valid(i_scnn_logit_wave_valid),
        .o_logit_wave_ready(o_scnn_logit_wave_ready),
        .i_logit_wave_mask(i_scnn_logit_wave_mask),
        .i_logit_wave_base(i_scnn_logit_wave_base),
        .i_logit_wave_count(i_scnn_logit_wave_count),
        .i_logit_wave_q8_8_flat(i_scnn_logit_wave_q8_8_flat),
        .i_logit_wave_version(i_scnn_logit_wave_version),
        .i_logit_wave_frame_index(i_scnn_logit_wave_frame),
        .i_logit_wave_frame_last(i_scnn_logit_wave_frame_last),
        .i_logit_wave_sample_last(i_scnn_logit_wave_sample_last),
        .o_external_cfg_valid(scnn_external_cfg_valid),
        .i_external_cfg_ready(scnn_external_cfg_ready),
        .o_external_cfg_profile_id(scnn_external_cfg_profile_id),
        .o_external_cfg_tensor_id(scnn_external_cfg_tensor_id),
        .o_external_cfg_parameter_base(scnn_external_cfg_parameter_base),
        .o_external_cfg_batch_count(scnn_external_cfg_batch_count),
        .o_external_cfg_source_count(scnn_external_cfg_source_count),
        .o_external_cfg_output_count(scnn_external_cfg_output_count),
        .o_external_cfg_vote_increment(
            scnn_external_cfg_vote_increment),
        .o_external_cfg_vote_threshold(
            scnn_external_cfg_vote_threshold),
        .o_external_cfg_code_min(scnn_external_cfg_code_min),
        .o_external_cfg_code_max(scnn_external_cfg_code_max),
        .i_external_cfg_error(scnn_external_cfg_error),
        .i_external_active(scnn_external_active),
        .o_external_release(scnn_external_release),
        .i_external_release_ready(scnn_external_release_ready),
        .o_external_source_trace_valid(
            scnn_external_source_trace_valid),
        .i_external_source_trace_ready(
            scnn_external_source_trace_ready),
        .o_external_source_trace_sample_id(
            scnn_external_source_trace_sample_id),
        .o_external_source_trace_index(
            scnn_external_source_trace_index),
        .o_external_source_trace_q016(
            scnn_external_source_trace_q016),
        .i_external_source_trace_error(
            scnn_external_source_trace_error),
        .o_external_logit_row_valid(scnn_external_logit_row_valid),
        .i_external_logit_row_ready(scnn_external_logit_row_ready),
        .o_external_logit_sample_id(scnn_external_logit_sample_id),
        .o_external_logit_label(scnn_external_logit_label),
        .o_external_logit_valid(scnn_external_logit_valid),
        .i_external_logit_ready(scnn_external_logit_ready),
        .o_external_logit_q6(scnn_external_logit_q6),
        .i_external_logit_error(scnn_external_logit_error),
        .i_external_logit_row_done(scnn_external_logit_row_done),
        .o_external_observation_done(
            scnn_external_observation_done),
        .i_external_observation_done_ready(
            scnn_external_observation_done_ready),
        .i_external_window_done(scnn_external_window_done),
        .i_external_window_busy(scnn_external_window_busy),
        .i_external_context_error(scnn_external_context_error),
        .i_external_gradient_overflow(
            scnn_external_gradient_overflow),
        .i_external_update_config_error(
            scnn_external_update_config_error),
        .i_external_request_error(scnn_external_request_error),
        .i_external_store_error(scnn_external_store_error),
        .i_external_internal_error(scnn_external_internal_error),
        .o_read_valid(scnn_private_read_valid),
        .i_read_ready(scnn_private_read_ready),
        .o_read_profile_id(scnn_private_read_profile_id),
        .o_read_tensor_id(scnn_private_read_tensor_id),
        .o_read_parameter_addr(scnn_private_read_addr),
        .o_read_tag(scnn_private_read_tag),
        .i_read_response_valid(scnn_private_response_valid),
        .o_read_response_ready(scnn_private_response_ready),
        .i_read_response_code(scnn_private_response_code),
        .i_read_response_residual(scnn_private_response_residual),
        .i_read_response_error(scnn_private_response_error),
        .o_result_valid(scnn_private_write_valid),
        .i_result_ready(scnn_private_write_ready),
        .o_result_profile_id(scnn_private_write_profile_id),
        .o_result_tensor_id(scnn_private_write_tensor_id),
        .o_result_parameter_addr(scnn_private_write_addr),
        .o_result_tag(scnn_private_write_tag),
        .o_result_code(scnn_private_write_code),
        .o_result_residual(scnn_private_write_residual),
        .o_write_code_enable(scnn_private_write_code_enable),
        .o_write_residual_enable(
            scnn_private_write_residual_enable),
        .o_commit_pulse(o_scnn_learning_commit),
        .o_detach_pulse(o_scnn_detach), .o_run_done(scnn_run_done),
        .o_run_aborted(scnn_run_aborted), .o_busy(scnn_busy),
        .o_quiescent(scnn_quiescent), .o_error(scnn_error)
    );

    assign o_scnn_learning_busy = scnn_busy;

    initial begin
        if (SHARE_FC_SCNN_OUTPUT_LEARNER && FC_LEARN_W0)
            $error("shared FC/SCNN output learner requires W2-only FC learning");
        if (FC_FEATURES > 4095 || FC_MAX_HIDDEN > 511 ||
            SRNN_HIDDEN > 511 || SRNN_BACKBONE > 1023 ||
            SRNN_CLASSES > 31 || SCNN_CHANNELS > 512 ||
            SCNN_FEATURES > 4095 || SCNN_CLASSES > 15)
            $error("actual learning subsystem parameter width exceeded");
        if (64'(SCNN_CLASSIFIER_BASE) <
            64'(SCNN_CONV8_BASE) +
            64'(SCNN_CHANNELS)*64'(SCNN_CHANNELS)*64'd9)
            $error("SCNN Conv8 and classifier parameter windows overlap");
        if (64'(SRNN_WOUT_BASE) < 64'(SRNN_WH1_BASE) +
            64'(SRNN_BACKBONE)*64'(SRNN_HIDDEN))
            $error("SRNN Wh1 and Wout parameter windows overlap");
    end
endmodule

`default_nettype wire
