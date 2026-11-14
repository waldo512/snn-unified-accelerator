`timescale 1ns/1ps
`default_nettype none

// Paper-profile integration boundary.  One final inference core is shared by
// FC, SRNN and SCNN; online learning is admitted by the common runtime owner.
// FC updates the resident tile banks, while SRNN and SCNN use one externally
// backed scalar parameter service.  The production SCNN graph adapter drives
// the same physical core and routes selected-layer observations internally.
module unified_snn_train_infer_top_v1 #(
    parameter FC_WEIGHT_DIR =
        "../../../data/weights/r19b_h480_complete_fc_input2312",
    parameter FC_COMPAT_WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312",
    parameter int FC_PHYSICAL_TILES = 5,
    parameter bit FC_LEARN_W0 = 1'b1,
    parameter bit FC_USE_SHARED_PARAMETER_BACKEND = 1'b0,
    parameter bit SHARE_FC_SCNN_OUTPUT_LEARNER = 1'b0,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0,
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int OUTPUT_MAX = 32,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH),
    parameter int SCNN_SPATIAL_DEPTH = 1024,
    parameter int SCNN_SPATIAL_ADDR_WIDTH = 15,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int OUTPUT_ADDR_WIDTH = 19,
    parameter int SCNN_CLASSIFIER_FEATURES = 2048,
    parameter int DDR_ADDRESS_WIDTH = 32,
    parameter int DDR_BANK_WORDS = 8640,
    parameter int DDR_DESCRIPTOR_LIMIT = 0,
    parameter int SRNN_HIDDEN = 256,
    parameter int SRNN_BACKBONE = 450,
    parameter int SRNN_CLASSES = 20,
    parameter int SCNN_CHANNELS = 512,
    parameter int SCNN_CLASSES = 11,
    parameter int SCNN_TARGET_OUTPUT_CHANNELS = 64,
    parameter int SCNN_TARGET_OUTPUT_SPATIAL = 32,
    parameter bit ENABLE_SCNN_ACTUAL_GRAPH = 1'b1,
    parameter logic [Q16_ADDR_WIDTH-1:0] FC_W0_BASE = 24'h000000,
    parameter logic [Q16_ADDR_WIDTH-1:0] FC_W2_BASE = 24'h300000,
    parameter logic [Q16_ADDR_WIDTH-1:0] SCNN_CONV8_BASE = 24'h000000,
    parameter logic [Q16_ADDR_WIDTH-1:0] SCNN_CLASSIFIER_BASE = 24'h240000,
    parameter logic [Q16_ADDR_WIDTH-1:0] SRNN_WH1_BASE = 24'h300000,
    parameter logic [Q16_ADDR_WIDTH-1:0] SRNN_WOUT_BASE = 24'h31c200,
    parameter FC_TP_W0_RESIDUAL_INIT_FILE = "",
    parameter SCNN_STATE_INIT_FILE = ""
) (
    input  logic                                      i_core_clk,
    input  logic                                      i_core_rst_n,
    input  logic                                      i_mig_ui_clk,
    input  logic                                      i_mig_ui_rst_n,
    input  logic                                      i_ddr_calib_complete,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,

    // Runtime profile and command boundary.
    input  logic                                      i_profile_request_valid,
    input  logic [6:0]                                i_profile_request_code,
    input  logic                                      i_profile_learning_enable,
    output logic                                      o_profile_aligned,
    output logic                                      o_start_allowed,
    output logic                                      o_profile_switch_done,
    output logic                                      o_profile_switch_busy,
    output logic [6:0]                                o_active_profile_code,
    output logic [15:0]                               o_profile_version,
    output logic                                      o_active_learning_enable,
    input  logic                                      i_run_valid,
    input  logic [6:0]                                i_run_profile_code,
    output logic                                      o_run_ready,
    output logic                                      o_run_done,
    output logic                                      o_run_aborted,
    output logic                                      o_run_busy,
    output logic [2:0]                                o_active_client,
    output logic [2:0]                                o_parameter_owner,
    output logic [2:0]                                o_core_active_source,
    output logic                                      o_learning_system_quiescent,
    output logic                                      o_learning_system_error,
    output logic [7:0]                                o_learning_control_error_code,

    // Final-core graph/configuration boundary.
    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [1:0]                                i_cfg_profile,
    input  logic [1:0]                                i_cfg_table,
    input  logic [8:0]                                i_cfg_index,
    input  logic [31:0]                               i_cfg_data,
    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [1:0]                                i_commit_profile,
    input  logic [15:0]                               i_model_version,
    input  logic [15:0]                               i_adapt_version,

    // FC native event and target-trajectory boundary.
    input  logic                                      i_fc_cfg_valid,
    output logic                                      o_fc_cfg_ready,
    input  logic [8:0]                                i_fc_hidden_count,
    output logic                                      o_fc_configured,
    output logic                                      o_fc_cfg_error,
    input  logic                                      i_fc_window_valid,
    output logic                                      o_fc_window_ready,
    input  logic [LANES-1:0]                          i_fc_record_valid,
    input  logic [15:0]                               i_fc_window_sample_id,
    input  logic [8:0]                                i_fc_window_timestep,
    input  logic [LANES-1:0]                          i_fc_boundary_mask,
    input  logic                                      i_fc_window_sample_last,
    input  logic [LANES*12-1:0]                       i_fc_addr_flat,
    input  logic [LANES*8-1:0]                        i_fc_data_flat,
    input  logic [3:0]                                i_fc_snapshot_label,
    input  logic                                      i_fc_target_hidden_trace_valid,
    output logic                                      o_fc_target_hidden_trace_ready,
    input  logic [3:0]                                i_fc_target_trace_sample,
    input  logic [8:0]                                i_fc_target_hidden_index,
    input  logic [16:0]                               i_fc_target_hidden_trace_q016,
    input  logic                                      i_fc_target_input_trace_valid,
    output logic                                      o_fc_target_input_trace_ready,
    input  logic [3:0]                                i_fc_target_input_index,
    input  logic [19:0]                               i_fc_target_input_trace_q016,
    input  logic                                      i_fc_target_done,
    output logic                                      o_fc_target_done_ready,
    output logic                                      o_fc_learning_commit,
    output logic                                      o_fc_learning_busy,

    // Full 700->450 recurrent backbone and M2 learning services.
    input  logic                                      i_srnn_cfg_valid,
    output logic                                      o_srnn_cfg_ready,
    input  logic [15:0]                               i_srnn_cfg_commit_version,
    output logic                                      o_srnn_configured,
    output logic                                      o_srnn_cfg_error,
    input  logic                                      i_srnn_parameter_load_valid,
    output logic                                      o_srnn_parameter_load_ready,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_srnn_parameter_load_addr,
    input  logic signed [15:0]                        i_srnn_parameter_load_code,
    input  logic signed [15:0]                        i_srnn_parameter_load_residual,
    input  logic                                      i_srnn_target_label_load_start,
    output logic                                      o_srnn_target_label_load_start_ready,
    input  logic                                      i_srnn_target_label_valid,
    output logic                                      o_srnn_target_label_ready,
    input  logic                                      i_srnn_target_label_sample,
    input  logic [4:0]                                i_srnn_target_label_class,
    input  logic                                      i_srnn_target_label_last,
    output logic                                      o_srnn_target_label_load_done,
    input  logic                                      i_srnn_target_spike_word_write_valid,
    output logic                                      o_srnn_target_spike_word_write_ready,
    input  logic [4:0]                                i_srnn_target_spike_word_class,
    input  logic                                      i_srnn_target_spike_word_timestep,
    input  logic [3:0]                                i_srnn_target_spike_word_id,
    input  logic [31:0]                               i_srnn_target_spike_word_data,
    input  logic                                      i_srnn_target_trace_write_valid,
    output logic                                      o_srnn_target_trace_write_ready,
    input  logic [4:0]                                i_srnn_target_trace_class,
    input  logic                                      i_srnn_target_trace_timestep,
    input  logic [8:0]                                i_srnn_target_trace_backbone,
    input  logic signed [31:0]                        i_srnn_target_trace_q14,
    input  logic                                      i_srnn_backbone_sample_start_valid,
    output logic                                      o_srnn_backbone_sample_start_ready,
    output logic                                      o_srnn_backbone_sample_id,
    output logic                                      o_srnn_learning_commit,
    output logic                                      o_srnn_detach,
    output logic [Q16_ADDR_WIDTH:0]                   o_srnn_update_count,
    output logic                                      o_srnn_weight_req_valid,
    input  logic                                      i_srnn_weight_req_ready,
    output logic [15:0]                               o_srnn_weight_req_version,
    output logic [15:0]                               o_srnn_weight_req_id,
    output logic                                      o_srnn_weight_req_recurrent,
    output logic [31:0]                               o_srnn_weight_req_row_base,
    output logic [9:0]                                o_srnn_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_srnn_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_srnn_weight_req_lane_mask,
    input  logic                                      i_srnn_weight_rsp_valid,
    output logic                                      o_srnn_weight_rsp_ready,
    input  logic [15:0]                               i_srnn_weight_rsp_version,
    input  logic [15:0]                               i_srnn_weight_rsp_id,
    input  logic [9:0]                                i_srnn_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_srnn_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_srnn_weight_rsp_flat,
    output logic [ENGINES-1:0]                        o_srnn_w2_req_valid,
    input  logic [ENGINES-1:0]                        i_srnn_w2_req_ready,
    output logic [ENGINES*32-1:0]                     o_srnn_w2_addr_flat,
    input  logic [ENGINES*64-1:0]                     i_srnn_w2_data_flat,

    // SCNN runtime/target and selected Conv8 observation adapter ABI.
    input  logic                                      i_scnn_configured,
    input  logic                                      i_scnn_target_start_ready,
    output logic                                      o_scnn_target_start_valid,
    output logic                                      o_scnn_target_sample,
    output logic                                      o_scnn_target_timestep,
    input  logic                                      i_scnn_target_done,
    input  logic                                      i_scnn_target_busy,
    input  logic                                      i_scnn_target_error,
    output logic                                      o_scnn_core_sample,
    output logic                                      o_scnn_core_timestep,
    output logic                                      o_scnn_core_post_forward,
    output logic                                      o_scnn_classifier_capture_enable,
    input  logic                                      i_scnn_sample_context_valid,
    output logic                                      o_scnn_sample_context_ready,
    input  logic                                      i_scnn_sample_context_target,
    input  logic                                      i_scnn_sample_context_sample,
    input  logic [15:0]                               i_scnn_sample_context_version,
    input  logic                                      i_scnn_input_valid,
    output logic                                      o_scnn_input_ready,
    input  logic                                      i_scnn_input_target,
    input  logic                                      i_scnn_input_sample,
    input  logic                                      i_scnn_input_timestep,
    input  logic [15:0]                               i_scnn_input_sample_version,
    input  logic [8:0]                                i_scnn_input_channel,
    input  logic [3:0]                                i_scnn_input_spatial,
    input  logic signed [15:0]                        i_scnn_input_q8_8,
    input  logic                                      i_scnn_trace_valid,
    output logic                                      o_scnn_trace_ready,
    input  logic                                      i_scnn_trace_target,
    input  logic                                      i_scnn_trace_sample,
    input  logic                                      i_scnn_trace_timestep,
    input  logic [15:0]                               i_scnn_trace_sample_version,
    input  logic [8:0]                                i_scnn_trace_channel,
    input  logic [1:0]                                i_scnn_trace_spatial,
    input  logic signed [15:0]                        i_scnn_trace_q8_8,
    input  logic                                      i_scnn_surrogate_valid,
    output logic                                      o_scnn_surrogate_ready,
    input  logic                                      i_scnn_surrogate_target,
    input  logic                                      i_scnn_surrogate_sample,
    input  logic                                      i_scnn_surrogate_timestep,
    input  logic [15:0]                               i_scnn_surrogate_sample_version,
    input  logic [8:0]                                i_scnn_surrogate_channel,
    input  logic [3:0]                                i_scnn_surrogate_spatial,
    input  logic signed [15:0]                        i_scnn_surrogate_q8_8,
    input  logic [3:0]                                i_scnn_sample_label,
    input  logic                                      i_scnn_parameter_load_valid,
    output logic                                      o_scnn_parameter_load_ready,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_scnn_parameter_load_addr,
    input  logic signed [15:0]                        i_scnn_parameter_load_code,
    input  logic signed [15:0]                        i_scnn_parameter_load_residual,
    output logic                                      o_scnn_learning_commit,
    output logic                                      o_scnn_detach,
    output logic                                      o_scnn_learning_busy,

    // Production SCNN graph services. Descriptors and student input loading
    // remain host/control-plane services; graph execution and observation
    // routing are internal when ENABLE_SCNN_ACTUAL_GRAPH is set.
    input  logic                                      i_scnn_label_valid,
    output logic                                      o_scnn_label_ready,
    input  logic [3:0]                                i_scnn_label_class,
    input  logic signed [15:0]                        i_scnn_label_spike_q8_8,
    input  logic                                      i_scnn_label_last,
    output logic                                      o_scnn_student_input_start_valid,
    input  logic                                      i_scnn_student_input_start_ready,
    output logic                                      o_scnn_student_input_sample,
    output logic                                      o_scnn_student_input_timestep,
    output logic [15:0]                               o_scnn_student_input_state_version,
    input  logic                                      i_scnn_student_input_done,
    input  logic                                      i_scnn_student_input_error,
    output logic                                      o_scnn_descriptor_request_valid,
    input  logic                                      i_scnn_descriptor_request_ready,
    output logic                                      o_scnn_descriptor_request_target,
    output logic [7:0]                                o_scnn_descriptor_profile_id,
    output logic [3:0]                                o_scnn_descriptor_layer_index,
    input  logic                                      i_scnn_descriptor_valid,
    output logic                                      o_scnn_descriptor_ready,
    input  logic [7:0]                                i_scnn_descriptor_response_profile_id,
    input  logic [3:0]                                i_scnn_descriptor_response_layer_index,
    input  logic [12*32-1:0]                          i_scnn_descriptor_words,
    input  logic                                      i_scnn_descriptor_error,
    input  logic [7:0]                                i_scnn_projector_tensor_id,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_scnn_projector_parameter_base,
    input  logic                                      i_scnn_trace_state_reset,

    // Raw final-core SCNN observer/tap boundary for the graph adapter.
    input  logic                                      i_scnn_input_observation_enable,
    output logic                                      o_scnn_input_observation_valid,
    input  logic                                      i_scnn_input_observation_ready,
    output logic [3:0]                                o_scnn_input_observation_layer,
    output logic [15:0]                               o_scnn_input_observation_channel,
    output logic [15:0]                               o_scnn_input_observation_spatial,
    output logic [15:0]                               o_scnn_input_observation_version,
    output logic signed [15:0]                        o_scnn_input_observation_data,
    output logic                                      o_scnn_input_observation_last,
    input  logic                                      i_scnn_conv_observation_enable,
    input  logic [3:0]                                i_scnn_conv_selected_layer,
    output logic                                      o_scnn_conv_observation_valid,
    input  logic                                      i_scnn_conv_observation_ready,
    output logic [3:0]                                o_scnn_conv_observation_layer,
    output logic [15:0]                               o_scnn_conv_observation_channel,
    output logic [SCNN_SPATIAL_ADDR_WIDTH-1:0]        o_scnn_conv_observation_spatial,
    output logic [OUTPUT_ADDR_WIDTH-1:0]              o_scnn_conv_observation_addr,
    output logic [15:0]                               o_scnn_conv_observation_version,
    output logic signed [15:0]                        o_scnn_conv_observation_current,
    output logic [15:0]                               o_scnn_conv_observation_spike,
    output logic signed [15:0]                        o_scnn_conv_observation_membrane_pre,
    output logic signed [15:0]                        o_scnn_conv_observation_membrane_next,
    output logic [15:0]                               o_scnn_conv_observation_surrogate,
    output logic                                      o_scnn_selected_conv_done,
    output logic                                      o_scnn_feature_write_valid,
    input  logic                                      i_scnn_feature_write_ready,
    output logic                                      o_scnn_feature_write_slot,
    output logic [3:0]                                o_scnn_feature_write_bank,
    output logic [FEATURE_ADDR_WIDTH-1:0]             o_scnn_feature_write_addr,
    output logic signed [15:0]                        o_scnn_feature_write_data,

    input  logic                                      i_feature_load_valid,
    output logic                                      o_feature_load_ready,
    input  logic                                      i_feature_load_slot,
    input  logic [3:0]                                i_feature_load_bank,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_feature_load_addr,
    input  logic signed [15:0]                        i_feature_load_data,

    // One abstract external parameter backing store for SRNN and SCNN.
    output logic                                      o_store_load_valid,
    input  logic                                      i_store_load_ready,
    output logic [Q16_ADDR_WIDTH-1:0]                 o_store_load_addr,
    output logic signed [15:0]                        o_store_load_code,
    output logic signed [15:0]                        o_store_load_residual,
    output logic                                      o_store_learning_read_valid,
    input  logic                                      i_store_learning_read_ready,
    output logic [Q16_ADDR_WIDTH-1:0]                 o_store_learning_read_addr,
    output logic [15:0]                               o_store_learning_read_tag,
    input  logic                                      i_store_learning_response_valid,
    output logic                                      o_store_learning_response_ready,
    input  logic [15:0]                               i_store_learning_response_tag,
    input  logic signed [15:0]                        i_store_learning_response_code,
    input  logic signed [15:0]                        i_store_learning_response_residual,
    input  logic                                      i_store_learning_response_error,
    output logic                                      o_store_learning_write_valid,
    input  logic                                      i_store_learning_write_ready,
    output logic [Q16_ADDR_WIDTH-1:0]                 o_store_learning_write_addr,
    output logic signed [15:0]                        o_store_learning_write_code,
    output logic signed [15:0]                        o_store_learning_write_residual,
    output logic                                      o_store_learning_write_code_enable,
    output logic                                      o_store_learning_write_residual_enable,
    output logic                                      o_store_forward_read_valid,
    input  logic                                      i_store_forward_read_ready,
    output logic [Q16_ADDR_WIDTH-1:0]                 o_store_forward_read_addr,
    output logic [15:0]                               o_store_forward_read_tag,
    input  logic                                      i_store_forward_response_valid,
    output logic                                      o_store_forward_response_ready,
    input  logic [15:0]                               i_store_forward_response_tag,
    input  logic signed [15:0]                        i_store_forward_response_code,
    input  logic                                      i_store_forward_response_error,
    input  logic                                      i_store_quiescent,
    input  logic                                      i_store_protocol_error,

    // Public inference result and status.
    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [5:0]                                o_result_class,
    output logic signed [47:0]                        o_result_score,
    output logic [63:0]                               o_result_versions,
    output logic [1:0]                                o_result_owner,
    output logic                                      o_global_idle,
    output logic                                      o_busy,
    output logic [63:0]                               o_active_versions,
    output logic [1:0]                                o_core_active_profile,
    output logic [6:0]                                o_core_active_profile_code,
    output logic [4:0]                                o_active_layer_count,
    output logic [6:0]                                o_active_route_count,
    output logic [3:0]                                o_active_layer_index,
    output logic [3:0]                                o_active_layer_type,
    output logic [1:0]                                o_compute_owner,
    output logic [31:0]                               o_fc_sample_cycles,
    output logic [11:0]                               o_fc_locked_cluster_mask,
    output logic                                      o_sample_done,
    output logic                                      o_prefetch_start_ready,
    output logic                                      o_prefetch_done,
    output logic                                      o_core_sticky_error,
    output logic [7:0]                                o_core_error_code,
    output logic [2:0]                                o_core_error_source,

    // Read-only AXI interface used by the inference descriptor/weight path.
    output logic                                      o_m_axi_arvalid,
    input  logic                                      i_m_axi_arready,
    output logic [DDR_ADDRESS_WIDTH-1:0]              o_m_axi_araddr,
    output logic [7:0]                                o_m_axi_arlen,
    output logic [2:0]                                o_m_axi_arsize,
    output logic [1:0]                                o_m_axi_arburst,
    output logic                                      o_m_axi_arid,
    output logic [3:0]                                o_m_axi_arcache,
    output logic [2:0]                                o_m_axi_arprot,
    output logic [3:0]                                o_m_axi_arqos,
    input  logic                                      i_m_axi_rvalid,
    output logic                                      o_m_axi_rready,
    input  logic [127:0]                              i_m_axi_rdata,
    input  logic                                      i_m_axi_rlast,
    input  logic [1:0]                                i_m_axi_rresp,
    input  logic                                      i_m_axi_rid
);
    logic core_start;
    logic core_start_ready;
    logic [6:0] core_profile_code;
    logic core_abort;
    logic physical_core_abort;

    logic fc_event_window_valid;
    logic fc_event_window_ready;
    logic [LANES-1:0] fc_event_record_valid;
    logic [15:0] fc_event_window_sample_id;
    logic [8:0] fc_event_window_timestep;
    logic [LANES-1:0] fc_event_boundary_mask;
    logic fc_event_window_sample_last;
    logic [LANES*12-1:0] fc_event_addr_flat;
    logic [LANES*8-1:0] fc_event_data_flat;
    logic [15:0] fc_core_sample_id;
    logic fc_core_sample_state_reset;
    logic fc_core_clear_state;
    logic fc_core_sample_last;

    logic fc_native_read_valid;
    logic fc_native_read_ready;
    logic [7:0] fc_native_read_tensor_id;
    logic [Q16_ADDR_WIDTH-1:0] fc_native_read_parameter_addr;
    logic [15:0] fc_native_read_tag;
    logic fc_native_response_valid;
    logic fc_native_response_ready;
    logic [15:0] fc_native_response_tag;
    logic signed [15:0] fc_native_response_code;
    logic signed [15:0] fc_native_response_residual;
    logic fc_native_response_error;
    logic fc_native_write_valid;
    logic fc_native_write_ready;
    logic [7:0] fc_native_write_tensor_id;
    logic [Q16_ADDR_WIDTH-1:0] fc_native_write_parameter_addr;
    logic signed [15:0] fc_native_write_code;
    logic signed [15:0] fc_native_write_residual;
    logic fc_native_write_code_enable;
    logic fc_native_write_residual_enable;
    logic fc_native_parameter_quiescent;

    logic [10*24-1:0] core_fc_logits_flat;
    logic [ENGINES*40-1:0] core_fc_spikes_flat;
    logic [ENGINES*40*18-1:0] core_fc_vmem_flat;
    logic [ENGINES*40*16-1:0] core_fc_trace_flat;
    logic srnn_commit_valid;
    logic srnn_commit_ready;
    logic [15:0] srnn_commit_version;
    logic srnn_commit_sample_last;
    logic [HIDDEN_MAX-1:0] srnn_commit_spikes;

    logic scnn_forward_valid;
    logic scnn_forward_ready;
    logic [Q16_ADDR_WIDTH-1:0] scnn_forward_addr;
    logic [15:0] scnn_forward_tag;
    logic scnn_forward_response_valid;
    logic scnn_forward_response_ready;
    logic [15:0] scnn_forward_response_tag;
    logic signed [15:0] scnn_forward_response_code;
    logic scnn_forward_response_error;
    logic scnn_feature_valid;
    logic scnn_feature_ready;
    logic [LANES-1:0] scnn_feature_mask;
    logic [10:0] scnn_feature_base;
    logic [LANES*16-1:0] scnn_feature_q8_8_flat;
    logic [15:0] scnn_feature_version;
    logic [7:0] scnn_feature_frame;
    logic scnn_feature_first;
    logic scnn_feature_last;
    logic scnn_feature_sample_reset;
    logic scnn_feature_sample_last;
    logic scnn_logit_valid;
    logic scnn_logit_ready;
    logic [ENGINES-1:0] scnn_logit_mask;
    logic [$clog2(OUTPUT_MAX)-1:0] scnn_logit_base;
    logic [$clog2(OUTPUT_MAX+1)-1:0] scnn_logit_count;
    logic [ENGINES*16-1:0] scnn_logit_q8_8_flat;
    logic [15:0] scnn_logit_version;
    logic [7:0] scnn_logit_frame;
    logic scnn_logit_frame_last;
    logic scnn_logit_sample_last;
    logic scnn_classifier_capture_enable;
    logic physical_core_start;
    logic physical_core_start_ready;
    logic physical_core_sample_done;
    logic physical_core_global_idle;
    logic physical_core_busy;
    logic graph_command_selected;
    logic graph_start_pending_q;
    logic graph_start_sample_q;
    logic graph_start_timestep_q;
    logic graph_start_post_forward_q;
    logic graph_target_start_ready;
    logic graph_target_done, graph_target_busy;
    logic graph_core_start_ready, graph_core_sample_done, graph_core_aborted;
    logic graph_selected_conv_done;
    logic graph_busy, graph_quiescent, graph_error, graph_error_q;
    logic graph_core_session_active;
    logic graph_core_layer_valid, graph_core_layer_ready;
    logic [3:0] graph_core_layer_index;
    logic [12*32-1:0] graph_core_layer_words;
    logic graph_core_input_slot, graph_core_output_slot;
    logic [15:0] graph_core_state_version;
    logic graph_core_sample_reset, graph_core_sample_last;
    logic graph_core_observation_enable, graph_core_input_prescan_enable;
    logic graph_core_layer_done, graph_core_layer_busy, graph_core_layer_error;
    logic graph_core_weight_read_valid, graph_core_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0]
        graph_core_weight_read_addr0, graph_core_weight_read_addr1;
    logic graph_core_weight_result_valid, graph_core_weight_result_ready;
    logic [63:0] graph_core_weight_data0, graph_core_weight_data1;
    logic graph_core_weight_response_error;
    logic graph_target_feature_write_valid, graph_target_feature_write_ready;
    logic graph_target_feature_write_slot;
    logic [3:0] graph_target_feature_write_bank;
    logic [FEATURE_ADDR_WIDTH-1:0] graph_target_feature_write_addr;
    logic signed [15:0] graph_target_feature_write_data;
    logic core_feature_load_valid, core_feature_load_ready;
    logic core_feature_load_slot;
    logic [3:0] core_feature_load_bank;
    logic [FEATURE_ADDR_WIDTH-1:0] core_feature_load_addr;
    logic signed [15:0] core_feature_load_data;
    logic graph_context_valid, graph_context_ready, graph_context_target;
    logic graph_context_sample;
    logic [15:0] graph_context_version;
    logic graph_input_valid, graph_input_ready, graph_input_target;
    logic graph_input_sample, graph_input_timestep;
    logic [15:0] graph_input_version;
    logic [8:0] graph_input_channel;
    logic [3:0] graph_input_spatial;
    logic signed [15:0] graph_input_data;
    logic graph_trace_valid, graph_trace_ready, graph_trace_target;
    logic graph_trace_sample, graph_trace_timestep;
    logic [15:0] graph_trace_version;
    logic [8:0] graph_trace_channel;
    logic [1:0] graph_trace_spatial;
    logic signed [15:0] graph_trace_data;
    logic graph_surrogate_valid, graph_surrogate_ready;
    logic graph_surrogate_target, graph_surrogate_sample;
    logic graph_surrogate_timestep;
    logic [15:0] graph_surrogate_version;
    logic [8:0] graph_surrogate_channel;
    logic [3:0] graph_surrogate_spatial;
    logic signed [15:0] graph_surrogate_data;
    logic graph_forward_valid, graph_forward_ready;
    logic [Q16_ADDR_WIDTH-1:0] graph_forward_addr;
    logic [15:0] graph_forward_tag;
    logic graph_forward_response_valid, graph_forward_response_ready;
    logic [15:0] graph_forward_response_tag;
    logic signed [15:0] graph_forward_response_code;
    logic graph_forward_response_error;
    logic raw_scnn_input_observation_valid;
    logic raw_scnn_input_observation_ready;
    logic raw_scnn_conv_observation_valid;
    logic raw_scnn_conv_observation_ready;
    logic raw_scnn_selected_conv_done;
    logic raw_scnn_feature_write_valid;
    logic raw_scnn_feature_write_ready;
    logic graph_core_prescan_ready;
    logic graph_core_lif_ready;
    logic graph_core_feature_write_ready;
    logic scnn_target_start_valid_i;
    logic scnn_target_sample_i, scnn_target_timestep_i;
    logic scnn_core_sample_i, scnn_core_timestep_i, scnn_core_post_forward_i;
    logic scnn_context_ready_i, scnn_input_ready_i;
    logic scnn_trace_ready_i, scnn_surrogate_ready_i;

    assign scnn_feature_ready = !scnn_classifier_capture_enable ||
                                o_scnn_feature_wave_ready_i;
    assign scnn_logit_ready = !scnn_classifier_capture_enable ||
                              o_scnn_logit_wave_ready_i;

    logic o_scnn_feature_wave_ready_i;
    logic o_scnn_logit_wave_ready_i;

    unified_registered_core_abort_boundary_v1 u_core_abort_boundary (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_external_abort(i_abort), .i_owner_abort(core_abort),
        .o_physical_abort(physical_core_abort),
        .o_owner_abort_delayed()
    );

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            graph_error_q <= 1'b0;
        else if (i_clear_error)
            graph_error_q <= 1'b0;
        else if (ENABLE_SCNN_ACTUAL_GRAPH && graph_error)
            graph_error_q <= 1'b1;
    end

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n) begin
            graph_start_pending_q <= 1'b0;
            graph_start_sample_q <= 1'b0;
            graph_start_timestep_q <= 1'b0;
            graph_start_post_forward_q <= 1'b0;
        end else if (physical_core_abort) begin
            graph_start_pending_q <= 1'b0;
        end else begin
            if (graph_start_pending_q && graph_core_start_ready)
                graph_start_pending_q <= 1'b0;
            if (core_start && graph_command_selected && core_start_ready) begin
                graph_start_pending_q <= 1'b1;
                graph_start_sample_q <= scnn_core_sample_i;
                graph_start_timestep_q <= scnn_core_timestep_i;
                graph_start_post_forward_q <= scnn_core_post_forward_i;
            end
        end
    end

    assign graph_command_selected = ENABLE_SCNN_ACTUAL_GRAPH &&
        core_profile_code == 7'h20;
    assign physical_core_start = core_start && !graph_command_selected;
    assign core_start_ready = graph_command_selected ? !graph_start_pending_q :
        physical_core_start_ready;
    assign o_sample_done = (o_core_active_source[2] &&
        ENABLE_SCNN_ACTUAL_GRAPH) ? graph_core_sample_done :
        physical_core_sample_done;
    assign o_global_idle = physical_core_global_idle &&
        (!ENABLE_SCNN_ACTUAL_GRAPH ||
            (graph_quiescent && !graph_start_pending_q));
    assign o_busy = physical_core_busy ||
        (ENABLE_SCNN_ACTUAL_GRAPH && (graph_busy || graph_start_pending_q));

    assign core_feature_load_valid = ENABLE_SCNN_ACTUAL_GRAPH &&
        graph_target_feature_write_valid ? 1'b1 : i_feature_load_valid;
    assign core_feature_load_slot = ENABLE_SCNN_ACTUAL_GRAPH &&
        graph_target_feature_write_valid ? graph_target_feature_write_slot :
        i_feature_load_slot;
    assign core_feature_load_bank = ENABLE_SCNN_ACTUAL_GRAPH &&
        graph_target_feature_write_valid ? graph_target_feature_write_bank :
        i_feature_load_bank;
    assign core_feature_load_addr = ENABLE_SCNN_ACTUAL_GRAPH &&
        graph_target_feature_write_valid ? graph_target_feature_write_addr :
        i_feature_load_addr;
    assign core_feature_load_data = ENABLE_SCNN_ACTUAL_GRAPH &&
        graph_target_feature_write_valid ? graph_target_feature_write_data :
        i_feature_load_data;
    assign graph_target_feature_write_ready = ENABLE_SCNN_ACTUAL_GRAPH &&
        core_feature_load_ready;
    assign o_feature_load_ready = core_feature_load_ready &&
        !(ENABLE_SCNN_ACTUAL_GRAPH && graph_target_feature_write_valid);

    assign o_scnn_target_start_valid = scnn_target_start_valid_i;
    assign o_scnn_target_sample = scnn_target_sample_i;
    assign o_scnn_target_timestep = scnn_target_timestep_i;
    assign o_scnn_core_sample = scnn_core_sample_i;
    assign o_scnn_core_timestep = scnn_core_timestep_i;
    assign o_scnn_core_post_forward = scnn_core_post_forward_i;
    assign o_scnn_input_observation_valid =
        raw_scnn_input_observation_valid;
    assign raw_scnn_input_observation_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        graph_core_prescan_ready : i_scnn_input_observation_ready;
    assign o_scnn_conv_observation_valid = raw_scnn_conv_observation_valid;
    assign raw_scnn_conv_observation_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        graph_core_lif_ready : i_scnn_conv_observation_ready;
    assign o_scnn_selected_conv_done = ENABLE_SCNN_ACTUAL_GRAPH ?
        graph_selected_conv_done : raw_scnn_selected_conv_done;
    assign o_scnn_feature_write_valid = raw_scnn_feature_write_valid;
    assign raw_scnn_feature_write_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        graph_core_feature_write_ready : i_scnn_feature_write_ready;
    assign graph_context_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        (!o_active_learning_enable || scnn_context_ready_i) : 1'b0;
    assign graph_input_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        (!o_active_learning_enable || scnn_input_ready_i) : 1'b0;
    assign graph_trace_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        (!o_active_learning_enable || scnn_trace_ready_i) : 1'b0;
    assign graph_surrogate_ready = ENABLE_SCNN_ACTUAL_GRAPH ?
        (!o_active_learning_enable || scnn_surrogate_ready_i) : 1'b0;
    assign o_scnn_sample_context_ready = scnn_context_ready_i;
    assign o_scnn_input_ready = scnn_input_ready_i;
    assign o_scnn_trace_ready = scnn_trace_ready_i;
    assign o_scnn_surrogate_ready = scnn_surrogate_ready_i;
    assign scnn_forward_ready = ENABLE_SCNN_ACTUAL_GRAPH ? 1'b0 :
        graph_forward_ready;
    assign scnn_forward_response_valid = !ENABLE_SCNN_ACTUAL_GRAPH &&
        graph_forward_response_valid;
    assign scnn_forward_response_tag = graph_forward_response_tag;
    assign scnn_forward_response_code = graph_forward_response_code;
    assign scnn_forward_response_error = graph_forward_response_error;

    unified_snn_core_final_v1 #(
        .FC_WEIGHT_DIR(FC_WEIGHT_DIR),
        .FC_COMPAT_WEIGHT_DIR(FC_COMPAT_WEIGHT_DIR),
        .USE_LEGACY_CANONICAL_FC(1'b0),
        .USE_SHARED_FC_SRNN_STATE(USE_SHARED_FC_SRNN_STATE),
        .FC_PHYSICAL_TILES(FC_PHYSICAL_TILES),
        .FC_LEARN_W0(FC_LEARN_W0),
        .ENGINES(ENGINES), .LANES(LANES),
        .HIDDEN_MAX(HIDDEN_MAX), .INPUT_MAX(INPUT_MAX),
        .OUTPUT_MAX(OUTPUT_MAX), .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .SCNN_SPATIAL_DEPTH(SCNN_SPATIAL_DEPTH),
        .SCNN_SPATIAL_ADDR_WIDTH(SCNN_SPATIAL_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH),
        .SCNN_CLASSIFIER_FEATURES(SCNN_CLASSIFIER_FEATURES),
        .DDR_ADDRESS_WIDTH(DDR_ADDRESS_WIDTH),
        .DDR_BANK_WORDS(DDR_BANK_WORDS),
        .DDR_DESCRIPTOR_LIMIT(DDR_DESCRIPTOR_LIMIT),
        .ENABLE_TP_SYSTEM_BINDING(1'b1),
        .ENABLE_TP_SCNN_PARAMETER_READ(1'b1),
        .ENABLE_TP_SCNN_CONV_OBSERVER(1'b1),
        .ENABLE_TP_SCNN_INPUT_OBSERVER(1'b1),
        .ENABLE_TP_SCNN_FEATURE_WRITE_TAP(1'b1),
        .ENABLE_FC_TP_NATIVE_WEIGHTS(1'b1),
        .ENABLE_EXTERNAL_SCNN_LAYER_INGRESS(ENABLE_SCNN_ACTUAL_GRAPH),
        .FC_TP_W0_RESIDUAL_INIT_FILE(FC_TP_W0_RESIDUAL_INIT_FILE),
        .SCNN_STATE_INIT_FILE(SCNN_STATE_INIT_FILE)
    ) u_core (
        .i_core_clk(i_core_clk), .i_core_rst_n(i_core_rst_n),
        .i_mig_ui_clk(i_mig_ui_clk), .i_mig_ui_rst_n(i_mig_ui_rst_n),
        .i_ddr_calib_complete(i_ddr_calib_complete),
        .i_cfg_valid(i_cfg_valid), .o_cfg_ready(o_cfg_ready),
        .i_cfg_profile(i_cfg_profile), .i_cfg_table(i_cfg_table),
        .i_cfg_index(i_cfg_index), .i_cfg_data(i_cfg_data),
        .i_commit_valid(i_commit_valid), .o_commit_ready(o_commit_ready),
        .i_commit_profile(i_commit_profile),
        .i_start(physical_core_start),
        .o_start_ready(physical_core_start_ready),
        .i_profile_code(core_profile_code),
        .i_clear_error(i_clear_error), .i_abort(physical_core_abort),
        .i_model_version(i_model_version), .i_adapt_version(i_adapt_version),
        .i_tp_start_allowed(o_start_allowed),
        .i_tp_fc_w0_parameter_base(FC_W0_BASE),
        .i_tp_fc_w2_parameter_base(FC_W2_BASE),
        .i_tp_fc_learning_read_valid(fc_native_read_valid),
        .o_tp_fc_learning_read_ready(fc_native_read_ready),
        .i_tp_fc_learning_read_tensor_id(fc_native_read_tensor_id),
        .i_tp_fc_learning_read_parameter_addr(fc_native_read_parameter_addr),
        .i_tp_fc_learning_read_tag(fc_native_read_tag),
        .o_tp_fc_learning_response_valid(fc_native_response_valid),
        .i_tp_fc_learning_response_ready(fc_native_response_ready),
        .o_tp_fc_learning_response_tag(fc_native_response_tag),
        .o_tp_fc_learning_response_code(fc_native_response_code),
        .o_tp_fc_learning_response_residual(fc_native_response_residual),
        .o_tp_fc_learning_response_error(fc_native_response_error),
        .i_tp_fc_learning_write_valid(fc_native_write_valid),
        .o_tp_fc_learning_write_ready(fc_native_write_ready),
        .i_tp_fc_learning_write_tensor_id(fc_native_write_tensor_id),
        .i_tp_fc_learning_write_parameter_addr(fc_native_write_parameter_addr),
        .i_tp_fc_learning_write_code(fc_native_write_code),
        .i_tp_fc_learning_write_residual(fc_native_write_residual),
        .i_tp_fc_learning_write_code_enable(fc_native_write_code_enable),
        .i_tp_fc_learning_write_residual_enable(
            fc_native_write_residual_enable),
        .o_tp_fc_learning_quiescent(fc_native_parameter_quiescent),
        .o_tp_fc_compute_done(), .o_tp_fc_logits_flat(core_fc_logits_flat),
        .o_tp_fc_spikes_flat(core_fc_spikes_flat),
        .o_tp_fc_vmem_flat(core_fc_vmem_flat),
        .o_tp_fc_trace_flat(core_fc_trace_flat),
        .o_tp_srnn_timestep_commit(srnn_commit_valid),
        .o_tp_srnn_timestep_version(srnn_commit_version),
        .o_tp_srnn_timestep_sample_last(srnn_commit_sample_last),
        .o_tp_srnn_timestep_spikes(srnn_commit_spikes),
        .i_tp_scnn_parameter_base(SCNN_CLASSIFIER_BASE),
        .o_tp_scnn_forward_valid(scnn_forward_valid),
        .i_tp_scnn_forward_ready(scnn_forward_ready),
        .o_tp_scnn_forward_addr(scnn_forward_addr),
        .o_tp_scnn_forward_tag(scnn_forward_tag),
        .i_tp_scnn_forward_response_valid(scnn_forward_response_valid),
        .o_tp_scnn_forward_response_ready(scnn_forward_response_ready),
        .i_tp_scnn_forward_response_tag(scnn_forward_response_tag),
        .i_tp_scnn_forward_response_code(scnn_forward_response_code),
        .i_tp_scnn_forward_response_error(scnn_forward_response_error),
        .o_tp_scnn_feature_valid(scnn_feature_valid),
        .i_tp_scnn_feature_ready(scnn_feature_ready),
        .o_tp_scnn_feature_mask(scnn_feature_mask),
        .o_tp_scnn_feature_base(scnn_feature_base),
        .o_tp_scnn_feature_q8_8_flat(scnn_feature_q8_8_flat),
        .o_tp_scnn_feature_version(scnn_feature_version),
        .o_tp_scnn_feature_frame(scnn_feature_frame),
        .o_tp_scnn_feature_first(scnn_feature_first),
        .o_tp_scnn_feature_last(scnn_feature_last),
        .o_tp_scnn_feature_sample_reset(scnn_feature_sample_reset),
        .o_tp_scnn_feature_sample_last(scnn_feature_sample_last),
        .o_tp_scnn_logit_valid(scnn_logit_valid),
        .i_tp_scnn_logit_ready(scnn_logit_ready),
        .o_tp_scnn_logit_mask(scnn_logit_mask),
        .o_tp_scnn_logit_base(scnn_logit_base),
        .o_tp_scnn_logit_count(scnn_logit_count),
        .o_tp_scnn_logit_q8_8_flat(scnn_logit_q8_8_flat),
        .o_tp_scnn_logit_version(scnn_logit_version),
        .o_tp_scnn_logit_frame(scnn_logit_frame),
        .o_tp_scnn_logit_frame_last(scnn_logit_frame_last),
        .o_tp_scnn_logit_sample_last(scnn_logit_sample_last),
        .i_tp_scnn_input_observation_enable(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_core_input_prescan_enable : i_scnn_input_observation_enable),
        .o_tp_scnn_input_observation_valid(raw_scnn_input_observation_valid),
        .i_tp_scnn_input_observation_ready(raw_scnn_input_observation_ready),
        .o_tp_scnn_input_observation_layer(o_scnn_input_observation_layer),
        .o_tp_scnn_input_observation_channel(o_scnn_input_observation_channel),
        .o_tp_scnn_input_observation_spatial(o_scnn_input_observation_spatial),
        .o_tp_scnn_input_observation_version(o_scnn_input_observation_version),
        .o_tp_scnn_input_observation_data(o_scnn_input_observation_data),
        .o_tp_scnn_input_observation_last(o_scnn_input_observation_last),
        .i_tp_scnn_conv_observation_enable(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_core_observation_enable : i_scnn_conv_observation_enable),
        .i_tp_scnn_conv_selected_layer(ENABLE_SCNN_ACTUAL_GRAPH ?
            4'd7 : i_scnn_conv_selected_layer),
        .o_tp_scnn_conv_observation_valid(raw_scnn_conv_observation_valid),
        .i_tp_scnn_conv_observation_ready(raw_scnn_conv_observation_ready),
        .o_tp_scnn_conv_observation_layer(o_scnn_conv_observation_layer),
        .o_tp_scnn_conv_observation_channel(o_scnn_conv_observation_channel),
        .o_tp_scnn_conv_observation_spatial(o_scnn_conv_observation_spatial),
        .o_tp_scnn_conv_observation_addr(o_scnn_conv_observation_addr),
        .o_tp_scnn_conv_observation_version(o_scnn_conv_observation_version),
        .o_tp_scnn_conv_observation_current(o_scnn_conv_observation_current),
        .o_tp_scnn_conv_observation_spike(o_scnn_conv_observation_spike),
        .o_tp_scnn_conv_observation_membrane_pre(
            o_scnn_conv_observation_membrane_pre),
        .o_tp_scnn_conv_observation_membrane_next(
            o_scnn_conv_observation_membrane_next),
        .o_tp_scnn_conv_observation_surrogate(o_scnn_conv_observation_surrogate),
        .o_tp_scnn_selected_conv_done(raw_scnn_selected_conv_done),
        .o_tp_scnn_feature_write_valid(raw_scnn_feature_write_valid),
        .i_tp_scnn_feature_write_ready(raw_scnn_feature_write_ready),
        .o_tp_scnn_feature_write_slot(o_scnn_feature_write_slot),
        .o_tp_scnn_feature_write_bank(o_scnn_feature_write_bank),
        .o_tp_scnn_feature_write_addr(o_scnn_feature_write_addr),
        .o_tp_scnn_feature_write_data(o_scnn_feature_write_data),
        .i_external_scnn_session_active(graph_core_session_active),
        .i_external_scnn_layer_valid(graph_core_layer_valid),
        .o_external_scnn_layer_ready(graph_core_layer_ready),
        .i_external_scnn_layer_index(graph_core_layer_index),
        .i_external_scnn_layer_words(graph_core_layer_words),
        .i_external_scnn_input_slot(graph_core_input_slot),
        .i_external_scnn_output_slot(graph_core_output_slot),
        .i_external_scnn_state_version(graph_core_state_version),
        .i_external_scnn_sample_reset(graph_core_sample_reset),
        .i_external_scnn_sample_last(graph_core_sample_last),
        .i_external_scnn_observation_enable(graph_core_observation_enable),
        .i_external_scnn_input_prescan_enable(
            graph_core_input_prescan_enable),
        .o_external_scnn_layer_done(graph_core_layer_done),
        .o_external_scnn_layer_busy(graph_core_layer_busy),
        .o_external_scnn_layer_protocol_error(graph_core_layer_error),
        .o_external_scnn_weight_read_valid(graph_core_weight_read_valid),
        .i_external_scnn_weight_read_ready(graph_core_weight_read_ready),
        .o_external_scnn_weight_read_addr0(graph_core_weight_read_addr0),
        .o_external_scnn_weight_read_addr1(graph_core_weight_read_addr1),
        .i_external_scnn_weight_result_valid(graph_core_weight_result_valid),
        .o_external_scnn_weight_result_ready(graph_core_weight_result_ready),
        .i_external_scnn_weight_data0(graph_core_weight_data0),
        .i_external_scnn_weight_data1(graph_core_weight_data1),
        .i_external_scnn_weight_response_error(
            graph_core_weight_response_error),
        .i_sample_state_reset(fc_core_sample_state_reset),
        .i_sample_last(fc_core_sample_last),
        .i_fc_clear_state(fc_core_clear_state),
        .i_fc_sample_id(fc_core_sample_id),
        .i_event_window_valid(fc_event_window_valid),
        .o_event_window_ready(fc_event_window_ready),
        .i_event_record_valid(fc_event_record_valid),
        .i_event_window_sample_id(fc_event_window_sample_id),
        .i_event_window_timestep(fc_event_window_timestep),
        .i_event_boundary_mask(fc_event_boundary_mask),
        .i_event_window_sample_last(fc_event_window_sample_last),
        .i_event_addr_flat(fc_event_addr_flat),
        .i_event_data_flat(fc_event_data_flat),
        .o_srnn_weight_req_valid(o_srnn_weight_req_valid),
        .i_srnn_weight_req_ready(i_srnn_weight_req_ready),
        .o_srnn_weight_req_version(o_srnn_weight_req_version),
        .o_srnn_weight_req_id(o_srnn_weight_req_id),
        .o_srnn_weight_req_recurrent(o_srnn_weight_req_recurrent),
        .o_srnn_weight_req_row_base(o_srnn_weight_req_row_base),
        .o_srnn_weight_req_destination_base(o_srnn_weight_req_destination_base),
        .o_srnn_weight_req_engine_mask(o_srnn_weight_req_engine_mask),
        .o_srnn_weight_req_lane_mask(o_srnn_weight_req_lane_mask),
        .i_srnn_weight_rsp_valid(i_srnn_weight_rsp_valid),
        .o_srnn_weight_rsp_ready(o_srnn_weight_rsp_ready),
        .i_srnn_weight_rsp_version(i_srnn_weight_rsp_version),
        .i_srnn_weight_rsp_id(i_srnn_weight_rsp_id),
        .i_srnn_weight_rsp_destination_base(i_srnn_weight_rsp_destination_base),
        .i_srnn_weight_rsp_engine_mask(i_srnn_weight_rsp_engine_mask),
        .i_srnn_weight_rsp_flat(i_srnn_weight_rsp_flat),
        .o_srnn_w2_req_valid(o_srnn_w2_req_valid),
        .i_srnn_w2_req_ready(i_srnn_w2_req_ready),
        .o_srnn_w2_addr_flat(o_srnn_w2_addr_flat),
        .i_srnn_w2_data_flat(i_srnn_w2_data_flat),
        .i_feature_load_valid(core_feature_load_valid),
        .o_feature_load_ready(core_feature_load_ready),
        .i_feature_load_slot(core_feature_load_slot),
        .i_feature_load_bank(core_feature_load_bank),
        .i_feature_load_addr(core_feature_load_addr),
        .i_feature_load_data(core_feature_load_data),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_class(o_result_class), .o_result_score(o_result_score),
        .o_result_versions(o_result_versions), .o_result_owner(o_result_owner),
        .o_global_idle(physical_core_global_idle),
        .o_busy(physical_core_busy),
        .o_active_versions(o_active_versions),
        .o_active_profile(o_core_active_profile),
        .o_active_profile_code(o_core_active_profile_code),
        .o_active_layer_count(o_active_layer_count),
        .o_active_route_count(o_active_route_count),
        .o_active_layer_index(o_active_layer_index),
        .o_active_layer_type(o_active_layer_type),
        .o_compute_owner(o_compute_owner),
        .o_fc_sample_cycles(o_fc_sample_cycles),
        .o_fc_locked_cluster_mask(o_fc_locked_cluster_mask),
        .o_sample_done(physical_core_sample_done),
        .o_prefetch_start_ready(o_prefetch_start_ready),
        .o_prefetch_done(o_prefetch_done),
        .o_sticky_error(o_core_sticky_error),
        .o_error_code(o_core_error_code),
        .o_error_source(o_core_error_source),
        .o_m_axi_arvalid(o_m_axi_arvalid), .i_m_axi_arready(i_m_axi_arready),
        .o_m_axi_araddr(o_m_axi_araddr), .o_m_axi_arlen(o_m_axi_arlen),
        .o_m_axi_arsize(o_m_axi_arsize), .o_m_axi_arburst(o_m_axi_arburst),
        .o_m_axi_arid(o_m_axi_arid), .o_m_axi_arcache(o_m_axi_arcache),
        .o_m_axi_arprot(o_m_axi_arprot), .o_m_axi_arqos(o_m_axi_arqos),
        .i_m_axi_rvalid(i_m_axi_rvalid), .o_m_axi_rready(o_m_axi_rready),
        .i_m_axi_rdata(i_m_axi_rdata), .i_m_axi_rlast(i_m_axi_rlast),
        .i_m_axi_rresp(i_m_axi_rresp), .i_m_axi_rid(i_m_axi_rid)
    );

    unified_scnn_tp_actual_graph_system_adapter_v1 #(
        .LAYER_WORDS(12), .ENGINES(ENGINES), .MAX_BATCH(2),
        .MAX_TIMESTEPS(2), .MAX_INPUT_CHANNELS(SCNN_CHANNELS),
        .MAX_OUTPUT_CHANNELS(SCNN_CHANNELS), .INPUT_SPATIAL_COUNT(16),
        .SURROGATE_SPATIAL_COUNT(16), .TARGET_DIM(SCNN_CLASSES),
        .TARGET_OUTPUT_CHANNELS(SCNN_TARGET_OUTPUT_CHANNELS),
        .TARGET_OUTPUT_SPATIAL(SCNN_TARGET_OUTPUT_SPATIAL),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .PARAM_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .ALLOW_RECURRENT_TIMESTEPS(1'b1),
        .CLASSIFIER_FEATURE_COUNT(SCNN_CLASSIFIER_FEATURES),
        .CLASSIFIER_CLASS_COUNT(SCNN_CLASSES),
        .STUDENT_FIRST_LAYER(0), .STUDENT_LAST_LAYER(7),
        .SELECTED_LAYER(7)
    ) u_scnn_actual_graph (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_enable(ENABLE_SCNN_ACTUAL_GRAPH),
        .i_abort(physical_core_abort), .i_clear_error(i_clear_error),
        .i_target_start_valid(scnn_target_start_valid_i),
        .o_target_start_ready(graph_target_start_ready),
        .i_target_sample(scnn_target_sample_i),
        .i_target_timestep(scnn_target_timestep_i),
        .i_target_state_version({8'h71, o_profile_version[7:0]}),
        .i_target_profile_id(8'h20),
        .i_projector_tensor_id(i_scnn_projector_tensor_id),
        .i_projector_parameter_base(i_scnn_projector_parameter_base),
        .i_target_membrane_leak_q8_8(16'sd0),
        .i_target_trace_leak_q8_8(16'sd250),
        .i_target_threshold_q8_8(16'sd256),
        .i_target_reset_to_zero(1'b0), .i_observe_projector(1'b1),
        .i_trace_state_reset(i_scnn_trace_state_reset),
        .i_label_valid(i_scnn_label_valid),
        .o_label_ready(o_scnn_label_ready),
        .i_label_class(i_scnn_label_class),
        .i_label_spike_q8_8(i_scnn_label_spike_q8_8),
        .i_label_last(i_scnn_label_last),
        .o_target_feature_write_valid(graph_target_feature_write_valid),
        .i_target_feature_write_ready(graph_target_feature_write_ready),
        .o_target_feature_write_slot(graph_target_feature_write_slot),
        .o_target_feature_write_bank(graph_target_feature_write_bank),
        .o_target_feature_write_addr(graph_target_feature_write_addr),
        .o_target_feature_write_data_q8_8(graph_target_feature_write_data),
        .i_core_start_valid(graph_start_pending_q),
        .o_core_start_ready(graph_core_start_ready),
        .i_core_sample(graph_start_sample_q),
        .i_core_timestep(graph_start_timestep_q),
        .i_core_post_forward(graph_start_post_forward_q),
        .i_student_profile_id(8'h20),
        .i_student_state_version({8'h72, o_profile_version[7:0]}),
        .i_post_forward_state_version(
            {8'h73, o_profile_version[7:0]}),
        .o_student_input_start_valid(o_scnn_student_input_start_valid),
        .i_student_input_start_ready(i_scnn_student_input_start_ready),
        .o_student_input_sample(o_scnn_student_input_sample),
        .o_student_input_timestep(o_scnn_student_input_timestep),
        .o_student_input_state_version(o_scnn_student_input_state_version),
        .i_student_input_done(i_scnn_student_input_done),
        .i_student_input_error(i_scnn_student_input_error),
        .o_descriptor_request_valid(o_scnn_descriptor_request_valid),
        .i_descriptor_request_ready(i_scnn_descriptor_request_ready),
        .o_descriptor_request_target(o_scnn_descriptor_request_target),
        .o_descriptor_profile_id(o_scnn_descriptor_profile_id),
        .o_descriptor_layer_index(o_scnn_descriptor_layer_index),
        .i_descriptor_valid(i_scnn_descriptor_valid),
        .o_descriptor_ready(o_scnn_descriptor_ready),
        .i_descriptor_response_profile_id(
            i_scnn_descriptor_response_profile_id),
        .i_descriptor_response_layer_index(
            i_scnn_descriptor_response_layer_index),
        .i_descriptor_words(i_scnn_descriptor_words),
        .i_descriptor_error(i_scnn_descriptor_error),
        .i_classifier_parameter_base(SCNN_CLASSIFIER_BASE),
        .i_classifier_feature_count(
            $clog2(SCNN_CLASSIFIER_FEATURES+1)'(SCNN_CLASSIFIER_FEATURES)),
        .i_classifier_class_count(
            $clog2(SCNN_CLASSES+1)'(SCNN_CLASSES)),
        .i_classifier_input_slot(1'b0),
        .i_classifier_sample_reset(graph_start_timestep_q == 1'b0),
        .i_classifier_sample_last(1'b1),
        .i_classifier_observation_enable(1'b1),
        .i_classifier_input_prescan_enable(1'b0),
        .o_core_session_active(graph_core_session_active),
        .o_core_layer_valid(graph_core_layer_valid),
        .i_core_layer_ready(graph_core_layer_ready),
        .o_core_layer_index(graph_core_layer_index),
        .o_core_layer_words(graph_core_layer_words),
        .o_core_input_slot(graph_core_input_slot),
        .o_core_output_slot(graph_core_output_slot),
        .o_core_state_version(graph_core_state_version),
        .o_core_sample_reset(graph_core_sample_reset),
        .o_core_sample_last(graph_core_sample_last),
        .o_core_observation_enable(graph_core_observation_enable),
        .o_core_input_prescan_enable(graph_core_input_prescan_enable),
        .i_core_layer_done(graph_core_layer_done),
        .i_core_layer_busy(graph_core_layer_busy),
        .i_core_layer_error(graph_core_layer_error),
        .i_core_weight_read_valid(graph_core_weight_read_valid),
        .o_core_weight_read_ready(graph_core_weight_read_ready),
        .i_core_weight_read_addr0(graph_core_weight_read_addr0),
        .i_core_weight_read_addr1(graph_core_weight_read_addr1),
        .o_core_weight_result_valid(graph_core_weight_result_valid),
        .i_core_weight_result_ready(graph_core_weight_result_ready),
        .o_core_weight_data0(graph_core_weight_data0),
        .o_core_weight_data1(graph_core_weight_data1),
        .o_core_weight_response_error(graph_core_weight_response_error),
        .i_core_prescan_valid(raw_scnn_input_observation_valid),
        .o_core_prescan_ready(graph_core_prescan_ready),
        .i_core_prescan_channel(o_scnn_input_observation_channel),
        .i_core_prescan_spatial(o_scnn_input_observation_spatial),
        .i_core_prescan_layer(o_scnn_input_observation_layer),
        .i_core_prescan_version(o_scnn_input_observation_version),
        .i_core_prescan_data_q8_8(o_scnn_input_observation_data),
        .i_core_lif_valid(raw_scnn_conv_observation_valid),
        .o_core_lif_ready(graph_core_lif_ready),
        .i_core_lif_layer(o_scnn_conv_observation_layer),
        .i_core_lif_channel(o_scnn_conv_observation_channel),
        .i_core_lif_spatial({1'b0, o_scnn_conv_observation_spatial}),
        .i_core_lif_version(o_scnn_conv_observation_version),
        .i_core_lif_surrogate_q8_8(o_scnn_conv_observation_surrogate),
        .i_core_feature_write_valid(raw_scnn_feature_write_valid),
        .o_core_feature_write_ready(graph_core_feature_write_ready),
        .i_core_feature_write_slot(o_scnn_feature_write_slot),
        .i_core_feature_write_bank(o_scnn_feature_write_bank),
        .i_core_feature_write_addr(o_scnn_feature_write_addr),
        .i_core_feature_write_data_q8_8(o_scnn_feature_write_data),
        .o_sample_context_valid(graph_context_valid),
        .i_sample_context_ready(graph_context_ready),
        .o_sample_context_target(graph_context_target),
        .o_sample_context_sample(graph_context_sample),
        .o_sample_context_version(graph_context_version),
        .o_input_valid(graph_input_valid), .i_input_ready(graph_input_ready),
        .o_input_target(graph_input_target), .o_input_sample(graph_input_sample),
        .o_input_timestep(graph_input_timestep),
        .o_input_version(graph_input_version),
        .o_input_channel(graph_input_channel),
        .o_input_spatial(graph_input_spatial), .o_input_q8_8(graph_input_data),
        .o_trace_valid(graph_trace_valid), .i_trace_ready(graph_trace_ready),
        .o_trace_target(graph_trace_target), .o_trace_sample(graph_trace_sample),
        .o_trace_timestep(graph_trace_timestep),
        .o_trace_version(graph_trace_version),
        .o_trace_channel(graph_trace_channel),
        .o_trace_spatial(graph_trace_spatial), .o_trace_q8_8(graph_trace_data),
        .o_surrogate_valid(graph_surrogate_valid),
        .i_surrogate_ready(graph_surrogate_ready),
        .o_surrogate_target(graph_surrogate_target),
        .o_surrogate_sample(graph_surrogate_sample),
        .o_surrogate_timestep(graph_surrogate_timestep),
        .o_surrogate_version(graph_surrogate_version),
        .o_surrogate_channel(graph_surrogate_channel),
        .o_surrogate_spatial(graph_surrogate_spatial),
        .o_surrogate_q8_8(graph_surrogate_data),
        .o_forward_read_valid(graph_forward_valid),
        .i_forward_read_ready(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_forward_ready : 1'b0),
        .o_forward_read_addr(graph_forward_addr),
        .o_forward_read_tag(graph_forward_tag),
        .i_forward_response_valid(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_forward_response_valid : 1'b0),
        .o_forward_response_ready(graph_forward_response_ready),
        .i_forward_response_tag(graph_forward_response_tag),
        .i_forward_response_code(graph_forward_response_code),
        .i_forward_response_error(graph_forward_response_error),
        .o_selected_conv_done(graph_selected_conv_done),
        .o_core_sample_done(graph_core_sample_done),
        .o_core_aborted(graph_core_aborted),
        .o_target_done(graph_target_done),
        .o_target_busy(graph_target_busy), .o_busy(graph_busy),
        .o_quiescent(graph_quiescent), .o_protocol_error(graph_error)
    );

    unified_tp_actual_learning_subsystem_v1 #(
        .FC_LEARN_W0(FC_LEARN_W0),
        .FC_USE_SHARED_PARAMETER_BACKEND(FC_USE_SHARED_PARAMETER_BACKEND),
        .SHARE_FC_SCNN_OUTPUT_LEARNER(SHARE_FC_SCNN_OUTPUT_LEARNER),
        .LANES(LANES), .ENGINES(ENGINES),
        .FC_MAX_HIDDEN(HIDDEN_MAX), .FC_FEATURES(INPUT_MAX),
        .SRNN_HIDDEN(SRNN_HIDDEN), .SRNN_BACKBONE(SRNN_BACKBONE),
        .SRNN_CLASSES(SRNN_CLASSES), .SCNN_CHANNELS(SCNN_CHANNELS),
        .SCNN_FEATURES(SCNN_CLASSIFIER_FEATURES),
        .SCNN_CLASSES(SCNN_CLASSES), .PARAM_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .FC_W0_BASE(FC_W0_BASE), .FC_W2_BASE(FC_W2_BASE),
        .SCNN_CONV8_BASE(SCNN_CONV8_BASE),
        .SCNN_CLASSIFIER_BASE(SCNN_CLASSIFIER_BASE),
        .SRNN_WH1_BASE(SRNN_WH1_BASE), .SRNN_WOUT_BASE(SRNN_WOUT_BASE)
    ) u_learning (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
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
        .i_run_valid(i_run_valid), .i_run_profile_code(i_run_profile_code),
        .o_run_ready(o_run_ready), .o_run_done(o_run_done),
        .o_run_aborted(o_run_aborted), .o_run_busy(o_run_busy),
        .o_active_client(o_active_client),
        .o_parameter_owner(o_parameter_owner),
        .o_system_quiescent(o_learning_system_quiescent),
        .o_system_error(o_learning_system_error),
        .o_control_error_code(o_learning_control_error_code),
        .o_core_start(core_start), .i_core_start_ready(core_start_ready),
        .o_core_profile_code(core_profile_code), .o_core_abort(core_abort),
        .i_core_sample_done(o_sample_done), .i_core_global_idle(o_global_idle),
        .i_core_error(o_core_sticky_error ||
            (ENABLE_SCNN_ACTUAL_GRAPH && graph_error_q)),
        .o_core_active_source(o_core_active_source),
        .i_fc_cfg_valid(i_fc_cfg_valid), .o_fc_cfg_ready(o_fc_cfg_ready),
        .i_fc_hidden_count(i_fc_hidden_count),
        .o_fc_configured(o_fc_configured), .o_fc_cfg_error(o_fc_cfg_error),
        .i_fc_window_valid(i_fc_window_valid),
        .o_fc_window_ready(o_fc_window_ready),
        .i_fc_record_valid(i_fc_record_valid),
        .i_fc_window_sample_id(i_fc_window_sample_id),
        .i_fc_window_timestep(i_fc_window_timestep),
        .i_fc_boundary_mask(i_fc_boundary_mask),
        .i_fc_window_sample_last(i_fc_window_sample_last),
        .i_fc_addr_flat(i_fc_addr_flat), .i_fc_data_flat(i_fc_data_flat),
        .i_fc_snapshot_label(i_fc_snapshot_label),
        .i_fc_snapshot_spikes_flat(core_fc_spikes_flat[HIDDEN_MAX-1:0]),
        .i_fc_snapshot_trace_flat(core_fc_trace_flat[HIDDEN_MAX*16-1:0]),
        .i_fc_snapshot_vmem_flat(core_fc_vmem_flat[HIDDEN_MAX*18-1:0]),
        .i_fc_snapshot_logits_flat(core_fc_logits_flat),
        .i_fc_target_hidden_trace_valid(i_fc_target_hidden_trace_valid),
        .o_fc_target_hidden_trace_ready(o_fc_target_hidden_trace_ready),
        .i_fc_target_trace_sample(i_fc_target_trace_sample),
        .i_fc_target_hidden_index(i_fc_target_hidden_index),
        .i_fc_target_hidden_trace_q016(i_fc_target_hidden_trace_q016),
        .i_fc_target_input_trace_valid(i_fc_target_input_trace_valid),
        .o_fc_target_input_trace_ready(o_fc_target_input_trace_ready),
        .i_fc_target_input_index(i_fc_target_input_index),
        .i_fc_target_input_trace_q016(i_fc_target_input_trace_q016),
        .i_fc_target_done(i_fc_target_done),
        .o_fc_target_done_ready(o_fc_target_done_ready),
        .o_fc_event_window_valid(fc_event_window_valid),
        .i_fc_event_window_ready(fc_event_window_ready),
        .o_fc_event_record_valid(fc_event_record_valid),
        .o_fc_event_window_sample_id(fc_event_window_sample_id),
        .o_fc_event_window_timestep(fc_event_window_timestep),
        .o_fc_event_boundary_mask(fc_event_boundary_mask),
        .o_fc_event_window_sample_last(fc_event_window_sample_last),
        .o_fc_event_addr_flat(fc_event_addr_flat),
        .o_fc_event_data_flat(fc_event_data_flat),
        .o_fc_core_sample_id(fc_core_sample_id),
        .o_fc_core_sample_state_reset(fc_core_sample_state_reset),
        .o_fc_core_clear_state(fc_core_clear_state),
        .o_fc_core_sample_last(fc_core_sample_last),
        .o_fc_native_read_valid(fc_native_read_valid),
        .i_fc_native_read_ready(fc_native_read_ready),
        .o_fc_native_read_tensor_id(fc_native_read_tensor_id),
        .o_fc_native_read_parameter_addr(fc_native_read_parameter_addr),
        .o_fc_native_read_tag(fc_native_read_tag),
        .i_fc_native_response_valid(fc_native_response_valid),
        .o_fc_native_response_ready(fc_native_response_ready),
        .i_fc_native_response_tag(fc_native_response_tag),
        .i_fc_native_response_code(fc_native_response_code),
        .i_fc_native_response_residual(fc_native_response_residual),
        .i_fc_native_response_error(fc_native_response_error),
        .o_fc_native_write_valid(fc_native_write_valid),
        .i_fc_native_write_ready(fc_native_write_ready),
        .o_fc_native_write_tensor_id(fc_native_write_tensor_id),
        .o_fc_native_write_parameter_addr(fc_native_write_parameter_addr),
        .o_fc_native_write_code(fc_native_write_code),
        .o_fc_native_write_residual(fc_native_write_residual),
        .o_fc_native_write_code_enable(fc_native_write_code_enable),
        .o_fc_native_write_residual_enable(fc_native_write_residual_enable),
        .i_fc_native_parameter_quiescent(fc_native_parameter_quiescent),
        .o_fc_learning_commit(o_fc_learning_commit),
        .o_fc_learning_busy(o_fc_learning_busy),
        .i_srnn_cfg_valid(i_srnn_cfg_valid), .o_srnn_cfg_ready(o_srnn_cfg_ready),
        .i_srnn_cfg_commit_version(i_srnn_cfg_commit_version),
        .o_srnn_configured(o_srnn_configured),
        .o_srnn_cfg_error(o_srnn_cfg_error),
        .i_srnn_parameter_load_valid(i_srnn_parameter_load_valid),
        .o_srnn_parameter_load_ready(o_srnn_parameter_load_ready),
        .i_srnn_parameter_load_addr(i_srnn_parameter_load_addr),
        .i_srnn_parameter_load_code(i_srnn_parameter_load_code),
        .i_srnn_parameter_load_residual(i_srnn_parameter_load_residual),
        .i_srnn_target_label_load_start(i_srnn_target_label_load_start),
        .o_srnn_target_label_load_start_ready(
            o_srnn_target_label_load_start_ready),
        .i_srnn_target_label_valid(i_srnn_target_label_valid),
        .o_srnn_target_label_ready(o_srnn_target_label_ready),
        .i_srnn_target_label_sample(i_srnn_target_label_sample),
        .i_srnn_target_label_class(i_srnn_target_label_class),
        .i_srnn_target_label_last(i_srnn_target_label_last),
        .o_srnn_target_label_load_done(o_srnn_target_label_load_done),
        .i_srnn_target_spike_word_write_valid(
            i_srnn_target_spike_word_write_valid),
        .o_srnn_target_spike_word_write_ready(
            o_srnn_target_spike_word_write_ready),
        .i_srnn_target_spike_word_class(i_srnn_target_spike_word_class),
        .i_srnn_target_spike_word_timestep(i_srnn_target_spike_word_timestep),
        .i_srnn_target_spike_word_id(i_srnn_target_spike_word_id),
        .i_srnn_target_spike_word_data(i_srnn_target_spike_word_data),
        .i_srnn_target_trace_write_valid(i_srnn_target_trace_write_valid),
        .o_srnn_target_trace_write_ready(o_srnn_target_trace_write_ready),
        .i_srnn_target_trace_class(i_srnn_target_trace_class),
        .i_srnn_target_trace_timestep(i_srnn_target_trace_timestep),
        .i_srnn_target_trace_backbone(i_srnn_target_trace_backbone),
        .i_srnn_target_trace_q14(i_srnn_target_trace_q14),
        .i_srnn_backbone_sample_start_valid(
            i_srnn_backbone_sample_start_valid),
        .o_srnn_backbone_sample_start_ready(
            o_srnn_backbone_sample_start_ready),
        .o_srnn_backbone_sample_id(o_srnn_backbone_sample_id),
        .i_srnn_commit_valid(srnn_commit_valid),
        .o_srnn_commit_ready(srnn_commit_ready),
        .i_srnn_commit_version(srnn_commit_version),
        .i_srnn_commit_sample_last(srnn_commit_sample_last),
        .i_srnn_commit_spikes(srnn_commit_spikes[479:0]),
        .o_srnn_learning_commit(o_srnn_learning_commit),
        .o_srnn_detach(o_srnn_detach),
        .o_srnn_update_count(o_srnn_update_count),
        .i_scnn_configured(i_scnn_configured),
        .i_scnn_target_start_ready(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_target_start_ready : i_scnn_target_start_ready),
        .o_scnn_target_start_valid(scnn_target_start_valid_i),
        .o_scnn_target_sample(scnn_target_sample_i),
        .o_scnn_target_timestep(scnn_target_timestep_i),
        .i_scnn_target_done(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_target_done : i_scnn_target_done),
        .i_scnn_target_busy(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_target_busy : i_scnn_target_busy),
        .i_scnn_target_error(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_error_q : i_scnn_target_error),
        .o_scnn_core_sample(scnn_core_sample_i),
        .o_scnn_core_timestep(scnn_core_timestep_i),
        .o_scnn_core_post_forward(scnn_core_post_forward_i),
        .o_scnn_classifier_capture_enable(scnn_classifier_capture_enable),
        .i_scnn_selected_conv_done(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_selected_conv_done : raw_scnn_selected_conv_done),
        .i_scnn_sample_context_valid(ENABLE_SCNN_ACTUAL_GRAPH ?
            (o_active_learning_enable && graph_context_valid) :
            i_scnn_sample_context_valid),
        .o_scnn_sample_context_ready(scnn_context_ready_i),
        .i_scnn_sample_context_target(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_context_target : i_scnn_sample_context_target),
        .i_scnn_sample_context_sample(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_context_sample : i_scnn_sample_context_sample),
        .i_scnn_sample_context_version(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_context_version : i_scnn_sample_context_version),
        .i_scnn_input_valid(ENABLE_SCNN_ACTUAL_GRAPH ?
            (o_active_learning_enable && graph_input_valid) :
            i_scnn_input_valid),
        .o_scnn_input_ready(scnn_input_ready_i),
        .i_scnn_input_target(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_target : i_scnn_input_target),
        .i_scnn_input_sample(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_sample : i_scnn_input_sample),
        .i_scnn_input_timestep(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_timestep : i_scnn_input_timestep),
        .i_scnn_input_sample_version(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_version : i_scnn_input_sample_version),
        .i_scnn_input_channel(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_channel : i_scnn_input_channel),
        .i_scnn_input_spatial(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_spatial : i_scnn_input_spatial),
        .i_scnn_input_q8_8(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_input_data : i_scnn_input_q8_8),
        .i_scnn_trace_valid(ENABLE_SCNN_ACTUAL_GRAPH ?
            (o_active_learning_enable && graph_trace_valid) :
            i_scnn_trace_valid),
        .o_scnn_trace_ready(scnn_trace_ready_i),
        .i_scnn_trace_target(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_target : i_scnn_trace_target),
        .i_scnn_trace_sample(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_sample : i_scnn_trace_sample),
        .i_scnn_trace_timestep(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_timestep : i_scnn_trace_timestep),
        .i_scnn_trace_sample_version(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_version : i_scnn_trace_sample_version),
        .i_scnn_trace_channel(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_channel : i_scnn_trace_channel),
        .i_scnn_trace_spatial(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_spatial : i_scnn_trace_spatial),
        .i_scnn_trace_q8_8(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_trace_data : i_scnn_trace_q8_8),
        .i_scnn_surrogate_valid(ENABLE_SCNN_ACTUAL_GRAPH ?
            (o_active_learning_enable && graph_surrogate_valid) :
            i_scnn_surrogate_valid),
        .o_scnn_surrogate_ready(scnn_surrogate_ready_i),
        .i_scnn_surrogate_target(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_target : i_scnn_surrogate_target),
        .i_scnn_surrogate_sample(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_sample : i_scnn_surrogate_sample),
        .i_scnn_surrogate_timestep(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_timestep : i_scnn_surrogate_timestep),
        .i_scnn_surrogate_sample_version(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_version : i_scnn_surrogate_sample_version),
        .i_scnn_surrogate_channel(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_channel : i_scnn_surrogate_channel),
        .i_scnn_surrogate_spatial(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_spatial : i_scnn_surrogate_spatial),
        .i_scnn_surrogate_q8_8(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_surrogate_data : i_scnn_surrogate_q8_8),
        .i_scnn_feature_wave_valid(scnn_classifier_capture_enable &&
            scnn_feature_valid),
        .o_scnn_feature_wave_ready(o_scnn_feature_wave_ready_i),
        .i_scnn_feature_wave_mask(scnn_feature_mask),
        .i_scnn_feature_wave_base(scnn_feature_base),
        .i_scnn_feature_wave_q8_8_flat(scnn_feature_q8_8_flat),
        .i_scnn_feature_wave_version(scnn_feature_version),
        .i_scnn_feature_wave_frame(ENABLE_SCNN_ACTUAL_GRAPH ?
            8'(graph_start_timestep_q) : scnn_feature_frame),
        .i_scnn_feature_wave_first(scnn_feature_first),
        .i_scnn_feature_wave_last(scnn_feature_last),
        .i_scnn_feature_wave_sample_reset(scnn_feature_sample_reset),
        .i_scnn_feature_wave_sample_last(scnn_feature_sample_last),
        .i_scnn_sample_label(i_scnn_sample_label),
        .i_scnn_logit_wave_valid(scnn_classifier_capture_enable &&
            scnn_logit_valid),
        .o_scnn_logit_wave_ready(o_scnn_logit_wave_ready_i),
        .i_scnn_logit_wave_mask(scnn_logit_mask),
        .i_scnn_logit_wave_base(4'(scnn_logit_base)),
        .i_scnn_logit_wave_count(4'(scnn_logit_count)),
        .i_scnn_logit_wave_q8_8_flat(scnn_logit_q8_8_flat),
        .i_scnn_logit_wave_version(scnn_logit_version),
        .i_scnn_logit_wave_frame(ENABLE_SCNN_ACTUAL_GRAPH ?
            8'(graph_start_timestep_q) : scnn_logit_frame),
        .i_scnn_logit_wave_frame_last(scnn_logit_frame_last),
        .i_scnn_logit_wave_sample_last(scnn_logit_sample_last),
        .i_scnn_parameter_load_valid(i_scnn_parameter_load_valid),
        .o_scnn_parameter_load_ready(o_scnn_parameter_load_ready),
        .i_scnn_parameter_load_addr(i_scnn_parameter_load_addr),
        .i_scnn_parameter_load_code(i_scnn_parameter_load_code),
        .i_scnn_parameter_load_residual(i_scnn_parameter_load_residual),
        .i_scnn_forward_valid(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_forward_valid : scnn_forward_valid),
        .o_scnn_forward_ready(graph_forward_ready),
        .i_scnn_forward_addr(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_forward_addr : scnn_forward_addr),
        .i_scnn_forward_tag(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_forward_tag : scnn_forward_tag),
        .o_scnn_forward_response_valid(graph_forward_response_valid),
        .i_scnn_forward_response_ready(ENABLE_SCNN_ACTUAL_GRAPH ?
            graph_forward_response_ready : scnn_forward_response_ready),
        .o_scnn_forward_response_tag(graph_forward_response_tag),
        .o_scnn_forward_response_code(graph_forward_response_code),
        .o_scnn_forward_response_error(graph_forward_response_error),
        .o_scnn_learning_commit(o_scnn_learning_commit),
        .o_scnn_detach(o_scnn_detach),
        .o_scnn_learning_busy(o_scnn_learning_busy),
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
        .i_store_protocol_error(i_store_protocol_error)
    );

    assign o_scnn_classifier_capture_enable =
        scnn_classifier_capture_enable;

    initial begin
        if (LANES != 8 || ENGINES != 12 || HIDDEN_MAX != ENGINES*40 ||
            Q16_ADDR_WIDTH < 8 || SCNN_CLASSIFIER_FEATURES < LANES)
            $error("unified train/infer paper profile width mismatch");
    end
endmodule

`default_nettype wire
