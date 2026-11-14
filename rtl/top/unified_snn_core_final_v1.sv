`timescale 1ns/1ps
`default_nettype none

// Final-v1 FC-first integration boundary. The canonical single-layer FC path
// remains inside the sole shared datapath instance; SRNN, CONV and OUTPUT use
// the sole Wave3 numeric array. Configuration is runtime-selected and all
// physical drain state is generated internally.
module unified_snn_core_final_v1 #(
    parameter FC_WEIGHT_DIR =
        "../../../data/weights/r19b_h480_complete_fc_input2312",
    parameter FC_COMPAT_WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312",
    parameter bit USE_LEGACY_CANONICAL_FC = 1'b0,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0,
    parameter int FC_PHYSICAL_TILES = 5,
    parameter bit FC_LEARN_W0 = 1'b1,
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int OUTPUT_MAX = 32,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH),
    // VGG-9 uses at most one 32x32 spatial plane per output channel.
    // The 15-bit observation ABI is retained independently of RAM depth.
    parameter int SCNN_SPATIAL_DEPTH = 1024,
    parameter int SCNN_SPATIAL_ADDR_WIDTH = 15,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = 24,
    parameter int OUTPUT_ADDR_WIDTH = 19,
    parameter int ACC_WIDTH = 48,
    parameter int SCNN_CLASSIFIER_FEATURES = 2048,
    parameter int DDR_ADDRESS_WIDTH = 32,
    parameter int DDR_BANK_WORDS = 8640,
    parameter int DDR_DESCRIPTOR_LIMIT = 0,
    parameter bit SCNN_WAVE_GROUP_DESCRIPTORS = 1'b1,
    parameter bit ENABLE_TP_SYSTEM_BINDING = 1'b0,
    parameter bit ENABLE_TP_SCNN_PARAMETER_READ = 1'b0,
    parameter bit ENABLE_TP_SCNN_CONV_OBSERVER = 1'b0,
    parameter bit ENABLE_TP_SCNN_INPUT_OBSERVER = 1'b0,
    parameter bit ENABLE_TP_SCNN_FEATURE_WRITE_TAP = 1'b0,
    parameter bit ENABLE_FC_TP_NATIVE_WEIGHTS = 1'b0,
    parameter bit ENABLE_EXTERNAL_SCNN_LAYER_INGRESS = 1'b0,
    parameter FC_TP_W0_RESIDUAL_INIT_FILE = "",
    parameter SCNN_STATE_INIT_FILE = ""
) (
    input  logic                                      i_core_clk,
    input  logic                                      i_core_rst_n,
    input  logic                                      i_mig_ui_clk,
    input  logic                                      i_mig_ui_rst_n,
    input  logic                                      i_ddr_calib_complete,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [1:0]                                i_cfg_profile,
    input  logic [1:0]                                i_cfg_table,
    input  logic [8:0]                                i_cfg_index,
    input  logic [31:0]                               i_cfg_data,
    input  logic                                      i_commit_valid,
    output logic                                      o_commit_ready,
    input  logic [1:0]                                i_commit_profile,
    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic [6:0]                                i_profile_code,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,
    input  logic [15:0]                               i_model_version,
    input  logic [15:0]                               i_adapt_version,

    input  logic                                      i_tp_start_allowed,
    input  logic [23:0]                               i_tp_fc_w0_parameter_base,
    input  logic [23:0]                               i_tp_fc_w2_parameter_base,
    input  logic                                      i_tp_fc_learning_read_valid,
    output logic                                      o_tp_fc_learning_read_ready,
    input  logic [7:0]                                i_tp_fc_learning_read_tensor_id,
    input  logic [23:0]                               i_tp_fc_learning_read_parameter_addr,
    input  logic [15:0]                               i_tp_fc_learning_read_tag,
    output logic                                      o_tp_fc_learning_response_valid,
    input  logic                                      i_tp_fc_learning_response_ready,
    output logic [15:0]                               o_tp_fc_learning_response_tag,
    output logic signed [15:0]                        o_tp_fc_learning_response_code,
    output logic signed [15:0]                        o_tp_fc_learning_response_residual,
    output logic                                      o_tp_fc_learning_response_error,
    input  logic                                      i_tp_fc_learning_write_valid,
    output logic                                      o_tp_fc_learning_write_ready,
    input  logic [7:0]                                i_tp_fc_learning_write_tensor_id,
    input  logic [23:0]                               i_tp_fc_learning_write_parameter_addr,
    input  logic signed [15:0]                        i_tp_fc_learning_write_code,
    input  logic signed [15:0]                        i_tp_fc_learning_write_residual,
    input  logic                                      i_tp_fc_learning_write_code_enable,
    input  logic                                      i_tp_fc_learning_write_residual_enable,
    output logic                                      o_tp_fc_learning_quiescent,
    output logic                                      o_tp_fc_compute_done,
    output logic [10*24-1:0]                         o_tp_fc_logits_flat,
    output logic [ENGINES*40-1:0]                    o_tp_fc_spikes_flat,
    output logic [ENGINES*40*18-1:0]                 o_tp_fc_vmem_flat,
    output logic [ENGINES*40*16-1:0]                 o_tp_fc_trace_flat,
    output logic                                      o_tp_srnn_timestep_commit,
    output logic [15:0]                               o_tp_srnn_timestep_version,
    output logic                                      o_tp_srnn_timestep_sample_last,
    output logic [HIDDEN_MAX-1:0]                    o_tp_srnn_timestep_spikes,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_tp_scnn_parameter_base,
    output logic                                      o_tp_scnn_forward_valid,
    input  logic                                      i_tp_scnn_forward_ready,
    output logic [Q16_ADDR_WIDTH-1:0]                 o_tp_scnn_forward_addr,
    output logic [15:0]                               o_tp_scnn_forward_tag,
    input  logic                                      i_tp_scnn_forward_response_valid,
    output logic                                      o_tp_scnn_forward_response_ready,
    input  logic [15:0]                               i_tp_scnn_forward_response_tag,
    input  logic signed [15:0]                        i_tp_scnn_forward_response_code,
    input  logic                                      i_tp_scnn_forward_response_error,
    output logic                                      o_tp_scnn_feature_valid,
    input  logic                                      i_tp_scnn_feature_ready,
    output logic [LANES-1:0]                          o_tp_scnn_feature_mask,
    output logic [$clog2(SCNN_CLASSIFIER_FEATURES)-1:0]
                                                       o_tp_scnn_feature_base,
    output logic [LANES*16-1:0]                       o_tp_scnn_feature_q8_8_flat,
    output logic [15:0]                               o_tp_scnn_feature_version,
    output logic [7:0]                                o_tp_scnn_feature_frame,
    output logic                                      o_tp_scnn_feature_first,
    output logic                                      o_tp_scnn_feature_last,
    output logic                                      o_tp_scnn_feature_sample_reset,
    output logic                                      o_tp_scnn_feature_sample_last,
    output logic                                      o_tp_scnn_logit_valid,
    input  logic                                      i_tp_scnn_logit_ready,
    output logic [ENGINES-1:0]                        o_tp_scnn_logit_mask,
    output logic [$clog2(OUTPUT_MAX)-1:0]              o_tp_scnn_logit_base,
    output logic [$clog2(OUTPUT_MAX+1)-1:0]            o_tp_scnn_logit_count,
    output logic [ENGINES*16-1:0]                     o_tp_scnn_logit_q8_8_flat,
    output logic [15:0]                               o_tp_scnn_logit_version,
    output logic [7:0]                                o_tp_scnn_logit_frame,
    output logic                                      o_tp_scnn_logit_frame_last,
    output logic                                      o_tp_scnn_logit_sample_last,

    input  logic                                      i_tp_scnn_input_observation_enable,
    output logic                                      o_tp_scnn_input_observation_valid,
    input  logic                                      i_tp_scnn_input_observation_ready,
    output logic [3:0]                                o_tp_scnn_input_observation_layer,
    output logic [15:0]                               o_tp_scnn_input_observation_channel,
    output logic [15:0]                               o_tp_scnn_input_observation_spatial,
    output logic [15:0]                               o_tp_scnn_input_observation_version,
    output logic signed [15:0]                        o_tp_scnn_input_observation_data,
    output logic                                      o_tp_scnn_input_observation_last,
    input  logic                                      i_tp_scnn_conv_observation_enable,
    input  logic [3:0]                                i_tp_scnn_conv_selected_layer,
    output logic                                      o_tp_scnn_conv_observation_valid,
    input  logic                                      i_tp_scnn_conv_observation_ready,
    output logic [3:0]                                o_tp_scnn_conv_observation_layer,
    output logic [15:0]                               o_tp_scnn_conv_observation_channel,
    output logic [SCNN_SPATIAL_ADDR_WIDTH-1:0]        o_tp_scnn_conv_observation_spatial,
    output logic [OUTPUT_ADDR_WIDTH-1:0]              o_tp_scnn_conv_observation_addr,
    output logic [15:0]                               o_tp_scnn_conv_observation_version,
    output logic signed [15:0]                        o_tp_scnn_conv_observation_current,
    output logic [15:0]                               o_tp_scnn_conv_observation_spike,
    output logic signed [15:0]                        o_tp_scnn_conv_observation_membrane_pre,
    output logic signed [15:0]                        o_tp_scnn_conv_observation_membrane_next,
    output logic [15:0]                               o_tp_scnn_conv_observation_surrogate,
    output logic                                      o_tp_scnn_selected_conv_done,
    output logic                                      o_tp_scnn_feature_write_valid,
    input  logic                                      i_tp_scnn_feature_write_ready,
    output logic                                      o_tp_scnn_feature_write_slot,
    output logic [3:0]                                o_tp_scnn_feature_write_bank,
    output logic [FEATURE_ADDR_WIDTH-1:0]             o_tp_scnn_feature_write_addr,
    output logic signed [15:0]                        o_tp_scnn_feature_write_data,

    input  logic                                      i_external_scnn_session_active,
    input  logic                                      i_external_scnn_layer_valid,
    output logic                                      o_external_scnn_layer_ready,
    input  logic [3:0]                                i_external_scnn_layer_index,
    input  logic [383:0]                              i_external_scnn_layer_words,
    input  logic                                      i_external_scnn_input_slot,
    input  logic                                      i_external_scnn_output_slot,
    input  logic [15:0]                               i_external_scnn_state_version,
    input  logic                                      i_external_scnn_sample_reset,
    input  logic                                      i_external_scnn_sample_last,
    input  logic                                      i_external_scnn_observation_enable,
    input  logic                                      i_external_scnn_input_prescan_enable,
    output logic                                      o_external_scnn_layer_done,
    output logic                                      o_external_scnn_layer_busy,
    output logic                                      o_external_scnn_layer_protocol_error,
    output logic                                      o_external_scnn_weight_read_valid,
    input  logic                                      i_external_scnn_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_external_scnn_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_external_scnn_weight_read_addr1,
    input  logic                                      i_external_scnn_weight_result_valid,
    output logic                                      o_external_scnn_weight_result_ready,
    input  logic [63:0]                               i_external_scnn_weight_data0,
    input  logic [63:0]                               i_external_scnn_weight_data1,
    input  logic                                      i_external_scnn_weight_response_error,

    input  logic                                      i_sample_state_reset,
    input  logic                                      i_sample_last,
    input  logic                                      i_fc_clear_state,
    input  logic [15:0]                               i_fc_sample_id,

    input  logic                                      i_event_window_valid,
    output logic                                      o_event_window_ready,
    input  logic [7:0]                                i_event_record_valid,
    input  logic [15:0]                               i_event_window_sample_id,
    input  logic [8:0]                                i_event_window_timestep,
    input  logic [7:0]                                i_event_boundary_mask,
    input  logic                                      i_event_window_sample_last,
    input  logic [8*12-1:0]                           i_event_addr_flat,
    input  logic [63:0]                               i_event_data_flat,

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

    input  logic                                      i_feature_load_valid,
    output logic                                      o_feature_load_ready,
    input  logic                                      i_feature_load_slot,
    input  logic [3:0]                                i_feature_load_bank,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_feature_load_addr,
    input  logic signed [15:0]                        i_feature_load_data,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [5:0]                                o_result_class,
    output logic signed [47:0]                        o_result_score,
    output logic [63:0]                               o_result_versions,
    output logic [1:0]                                o_result_owner,

    output logic                                      o_global_idle,
    output logic                                      o_busy,
    output logic [63:0]                               o_active_versions,
    output logic [1:0]                                o_active_profile,
    output logic [6:0]                                o_active_profile_code,
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
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code,
    output logic [2:0]                                o_error_source,

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
    localparam logic [3:0] LAYER_SPARSE_FC = 4'd1;
    localparam logic [3:0] LAYER_CONV = 4'd2;
    localparam logic [3:0] LAYER_POOL = 4'd3;
    localparam logic [3:0] LAYER_OUTPUT = 4'd4;
    localparam logic [3:0] LAYER_RECURRENT_FC = 4'd5;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_SRNN = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    logic qualified_start;
    logic scnn_profile_request;
    logic control_start_ready;
    logic control_layer_valid;
    logic control_layer_ready;
    logic [3:0] control_layer_index;
    logic [383:0] control_layer_words;
    logic [3:0] control_inbound_valid;
    logic [15:0] control_inbound_sources_flat;
    logic [15:0] control_inbound_types_flat;
    logic control_layer_done;
    logic [3:0] current_layer_type;
    logic [63:0] live_versions;
    logic [63:0] locked_versions;
    logic control_profile_legal;
    logic control_sequence_busy;
    logic control_sample_active;
    logic control_sample_compute_done;
    logic control_commit_pending;
    logic control_commit_pulse;
    logic control_config_copy_busy;
    logic [25:0] control_blocker_mask;
    logic control_quiescent_for_commit;
    logic control_idle;
    logic control_busy;
    logic control_cfg_ready;
    logic control_commit_ready;
    logic control_global_idle;
    logic control_version_consistent;
    logic control_error;
    logic [7:0] control_error_code;
    logic versioned_accept_unused;
    logic versioned_stale_unused;
    logic versioned_mixed_unused;

    logic dispatch_layer_valid;
    logic dispatch_layer_ready;
    logic [3:0] dispatch_layer_index;
    logic [383:0] dispatch_layer_words;
    logic dispatch_input_slot;
    logic dispatch_output_slot;
    logic [15:0] dispatch_state_version;
    logic dispatch_sample_reset;
    logic dispatch_sample_last;
    logic dispatch_observation_enable;
    logic dispatch_input_prescan_enable;
    logic dispatch_layer_done;
    logic layer_ingress_active_external;
    logic layer_ingress_busy;
    logic layer_ingress_error;
    logic external_layer_request_allowed;
    logic external_scnn_session_active;
    logic [63:0] effective_locked_versions;
    logic external_conv_weight_owner;
    logic external_output_weight_owner;
    logic external_weight_owner;
    logic external_physical_owner;
    logic external_physical_owner_q;
    logic external_weight_request_fire;
    logic external_weight_response_fire;
    logic external_weight_pending_q;
    logic external_weight_error_q;

    logic system_fc_valid;
    logic system_fc_ready;
    logic system_fc_done;
    logic system_recurrent_valid;
    logic system_recurrent_ready;
    logic system_recurrent_done;
    logic system_scnn_valid;
    logic system_scnn_ready;
    logic system_scnn_done;
    logic system_output_valid;
    logic system_output_ready;
    logic system_output_done;
    logic system_layer_accept;
    logic [1:0] system_requested_owner;
    logic system_owner_active;
    logic system_dispatch_busy;
    logic system_datapath_quiescent;
    logic system_control_quiescent_q;
    logic system_layer_pending;
    logic system_drain_owner_active;
    logic system_owner_tracking_consistent;
    logic system_result_pending;
    logic system_abort_cleanup_pending;
    logic [18:0] system_physical_blocker_mask;
    logic system_dispatch_error;
    logic [7:0] system_dispatch_error_code;
    logic system_fail_stop;

    logic sample_state_reset_q;
    logic sample_last_q;
    logic fc_clear_state_q;
    logic [15:0] fc_sample_id_q;
    logic public_result_fire;
    logic fc_result_retire;
    logic srnn_result_retire;
    logic scnn_result_retire;

    logic srnn_engine_start_valid;
    logic srnn_engine_start_ready;
    logic srnn_adapter_layer_ready;
    logic srnn_adapter_done;
    logic srnn_adapter_busy;
    logic srnn_adapter_idle;
    logic srnn_adapter_error;
    logic [7:0] srnn_adapter_error_code;
    logic srnn_state_reset;
    logic [12:0] srnn_input_dim;
    logic [9:0] srnn_active_neurons;
    logic [3:0] srnn_active_tiles;
    logic [5:0] srnn_output_count;
    logic [31:0] srnn_w2_base;
    logic [5:0] srnn_w2_row_stride;
    logic [31:0] srnn_recurrent_base;
    logic srnn_recurrent_enable;
    logic [31:0] srnn_w0_base;
    logic signed [15:0] srnn_leak_q15;
    logic signed [31:0] srnn_threshold;
    logic [4:0] srnn_current_left_shift;
    logic srnn_reset_to_zero;
    logic [15:0] srnn_version;

    logic dp_fc_layer_ready;
    logic dp_scnn_start_valid;
    logic dp_scnn_start_ready;
    logic dp_scnn_external_mode;
    logic dp_scnn_feature_read_valid;
    logic dp_scnn_feature_read_ready;
    logic dp_scnn_feature_read_slot;
    logic [31:0] dp_scnn_feature_read_bank_flat;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] dp_scnn_feature_read_addr_flat;
    logic [LANES-1:0] dp_scnn_feature_read_lane_mask;
    logic dp_scnn_feature_valid;
    logic dp_scnn_feature_ready;
    logic [LANES*16-1:0] dp_scnn_feature_flat;
    logic [LANES-1:0] dp_scnn_feature_lane_mask;
    logic dp_scnn_weight_read_valid;
    logic dp_scnn_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0] dp_scnn_weight_read_addr0;
    logic [Q64_ADDR_WIDTH-1:0] dp_scnn_weight_read_addr1;
    logic dp_scnn_weight_result_valid;
    logic dp_scnn_weight_result_ready;
    logic [63:0] dp_scnn_weight_data0;
    logic [63:0] dp_scnn_weight_data1;
    logic dp_scnn_raw_result_valid;
    logic dp_scnn_raw_result_ready;
    logic [ENGINES-1:0] dp_scnn_raw_result_mask;
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0] dp_scnn_raw_result_addr_flat;
    logic [OUTPUT_ADDR_WIDTH-1:0] dp_scnn_raw_spatial_addr;
    logic [ENGINES*ACC_WIDTH-1:0] dp_scnn_raw_accumulator_flat;
    logic dp_scnn_external_request_valid;
    logic dp_scnn_external_request_ready;
    logic [ENGINES-1:0] dp_scnn_external_engine_mask;
    logic [LANES*16-1:0] dp_scnn_external_feature_flat;
    logic [ENGINES*LANES*16-1:0] dp_scnn_external_weight_flat;
    logic [LANES-1:0] dp_scnn_external_lane_mask;
    logic dp_scnn_external_first;
    logic dp_scnn_external_last;
    logic [9:0] dp_scnn_external_output_base;
    logic dp_scnn_external_result_valid;
    logic dp_scnn_external_result_ready;
    logic [ENGINES-1:0] dp_scnn_external_result_mask;
    logic [ENGINES*ACC_WIDTH-1:0] dp_scnn_external_result_accumulator_flat;
    logic dp_scnn_li_result_valid;
    logic dp_scnn_li_result_ready;
    logic [5:0] dp_scnn_li_result_class;
    logic signed [47:0] dp_scnn_li_result_score;
    logic dp_fc_compute_done;
    logic dp_srnn_compute_done;
    logic dp_scnn_compute_done;
    logic [10*24-1:0] dp_fc_logits_flat;
    logic [ENGINES*40-1:0] dp_fc_spikes_flat;
    logic [ENGINES*40*18-1:0] dp_fc_vmem_flat;
    logic [ENGINES*40*16-1:0] dp_fc_trace_flat;
    logic [OUTPUT_MAX*32-1:0] dp_srnn_logits_flat;
    logic [HIDDEN_MAX-1:0] dp_srnn_previous_spikes;
    logic [1:0] dp_extension_owner;
    logic dp_result_pending;
    logic dp_public_result_valid;
    logic dp_public_result_ready;
    logic [5:0] dp_public_result_class;
    logic signed [47:0] dp_public_result_score;
    logic [63:0] dp_public_result_versions;
    logic [1:0] dp_public_result_owner;
    logic dp_fc_busy;
    logic dp_fc_path_busy;
    logic dp_extension_busy;
    logic dp_dynamic_output_busy;
    logic dp_ingress_busy;
    logic dp_w2_commit_pending;
    logic dp_w2_snapshot_busy;
    logic dp_busy;
    logic dp_idle;
    logic dp_error;
    logic [7:0] dp_error_code;

    logic conv_layer_valid;
    logic conv_request_valid, conv_admission_ready;
    logic tp_input_enable_q, tp_input_busy, tp_input_error, tp_input_owner;
    logic tp_input_read_valid, tp_input_response_ready, tp_input_read_slot;
    logic [31:0] tp_input_read_banks;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] tp_input_read_addrs;
    logic [LANES-1:0] tp_input_read_mask;
    logic conv_memory_read_ready, conv_memory_result_valid;
    logic [LANES*16-1:0] conv_memory_result_data;
    logic [LANES-1:0] conv_memory_result_mask;
    logic conv_layer_ready;
    logic conv_layer_done;
    logic conv_layer_busy;
    logic conv_layer_idle;
    logic conv_layer_error;
    logic [7:0] conv_layer_error_code;
    logic conv_start_valid;
    logic conv_start_ready;
    logic [15:0] conv_in_width;
    logic [15:0] conv_in_height;
    logic [15:0] conv_total_in_channels;
    logic [15:0] conv_out_width;
    logic [15:0] conv_out_height;
    logic [15:0] conv_total_out_channels;
    logic [9:0] conv_input_channel_base;
    logic [9:0] conv_input_channel_count;
    logic [9:0] conv_output_channel_base;
    logic [9:0] conv_output_channel_count;
    logic [7:0] conv_stride;
    logic [7:0] conv_padding;
    logic conv_input_slot;
    logic [FEATURE_ADDR_WIDTH-1:0] conv_feature_base;
    logic [Q16_ADDR_WIDTH-1:0] conv_weight_q16_base;
    logic [OUTPUT_ADDR_WIDTH-1:0] conv_output_base;
    logic conv_unit_spike_features;
    logic conv_feature_write_valid;
    logic conv_feature_write_ready;
    logic conv_feature_write_slot;
    logic [3:0] conv_feature_write_bank;
    logic [FEATURE_ADDR_WIDTH-1:0] conv_feature_write_addr;
    logic signed [15:0] conv_feature_write_data;
    logic conv_store_write_valid;
    logic conv_store_write_ready;
    logic [15:0] conv_locked_version;

    logic pool_valid;
    logic pool_ready;
    logic pool_done;
    logic pool_busy;
    logic pool_error;
    logic [7:0] pool_error_code;

    logic output_adapter_ready;
    logic output_adapter_done;
    logic output_adapter_busy;
    logic output_adapter_idle;
    logic output_adapter_error;
    logic [7:0] output_adapter_error_code;
    logic output_wave3_start_valid;
    logic output_wave3_start_ready;
    logic output_wave3_external_mode;
    logic output_wave3_external_done;
    logic output_wave3_clear_error;
    logic output_frontend_owner_enable;
    logic output_frontend_clear_error;
    logic output_frontend_layer_valid;
    logic output_frontend_layer_ready;
    logic [383:0] output_frontend_layer_words;
    logic output_frontend_input_slot;
    logic output_frontend_sample_reset;
    logic output_frontend_sample_last;
    logic [15:0] output_frontend_version;
    logic output_frontend_busy;
    logic output_frontend_done;
    logic output_frontend_enqueued;
    logic output_result_enqueued;
    logic output_owner_held;
    logic output_frontend_error;
    logic [7:0] output_frontend_error_code;
    logic output_observation_enable;
    logic output_feature_trace_valid;
    logic output_logit_wave_valid;
    logic output_feature_read_valid;
    logic output_feature_read_ready;
    logic output_feature_read_slot;
    logic [31:0] output_feature_read_bank_flat;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] output_feature_read_addr_flat;
    logic [LANES-1:0] output_feature_read_lane_mask;
    logic output_feature_valid;
    logic output_feature_ready;
    logic [LANES*16-1:0] output_feature_flat;
    logic [LANES-1:0] output_feature_lane_mask;
    logic output_weight_read_valid;
    logic output_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0] output_weight_read_addr0;
    logic [Q64_ADDR_WIDTH-1:0] output_weight_read_addr1;
    logic output_weight_result_valid;
    logic output_weight_result_ready;
    logic [63:0] output_weight_data0;
    logic [63:0] output_weight_data1;
    logic output_result_valid;
    logic [4:0] output_result_class;
    logic signed [15:0] output_result_score;
    logic [15:0] output_result_version;
    logic [7:0] output_result_frame;
    logic output_result_sample_last;
    logic [15:0] output_locked_version;
    logic [7:0] output_frames_seen;
    logic output_config_error;
    logic output_protocol_error;
    logic output_version_error_q;
    logic tp_start_allowed_effective;
    logic tp_scnn_feature_ready_effective;
    logic tp_scnn_logit_ready_effective;
    logic tp_conv_observer_enable_q;
    logic [3:0] tp_conv_observer_layer_q;
    logic tp_weight_request_ready;
    logic tp_weight_response_valid;
    logic [63:0] tp_weight_data0, tp_weight_data1;
    logic tp_weight_busy, tp_weight_error;

    logic output_memory_owner;
    logic feature_memory_busy;
    logic feature_memory_error;
    logic [7:0] feature_memory_error_code;
    logic feature_host_enable;

    logic ddr_prefetch_start;
    logic ddr_busy;
    logic ddr_idle;
    logic ddr_done;
    logic ddr_compute_tile_valid;
    logic ddr_compute_tile_done;
    logic [3:0] ddr_compute_layer_index;
    logic [9:0] ddr_compute_output_base;
    logic [9:0] ddr_compute_output_count;
    logic [9:0] ddr_compute_input_base;
    logic [9:0] ddr_compute_input_count;
    logic [DDR_ADDRESS_WIDTH-1:0] ddr_compute_source;
    logic [$clog2(DDR_BANK_WORDS+1)-1:0] ddr_compute_total;
    logic [$clog2(DDR_BANK_WORDS+1)-1:0] ddr_compute_segment;
    logic [DDR_ADDRESS_WIDTH-1:0] ddr_compute_stride;
    logic [15:0] ddr_compute_tag;
    logic ddr_local_read_valid;
    logic ddr_local_read_ready;
    logic [$clog2(DDR_BANK_WORDS)-1:0] ddr_local_read_addr0;
    logic [$clog2(DDR_BANK_WORDS)-1:0] ddr_local_read_addr1;
    logic ddr_local_response_valid;
    logic ddr_local_response_ready;
    logic [63:0] ddr_local_data0;
    logic [63:0] ddr_local_data1;
    logic ddr_cmd_pending;
    logic ddr_response_pending;
    logic ddr_cdc_busy;
    logic [2:0] ddr_outstanding;
    logic [2:0] ddr_axi_outstanding;
    logic ddr_axi_busy_core;
    logic [2:0] ddr_physical_outstanding;
    logic ddr_memory_busy;
    logic ddr_prefetch_busy;
    logic ddr_scratchpad_busy;
    logic ddr_overlap_seen;
    logic ddr_error;
    logic [3:0] ddr_error_code;

    logic wave_start_valid;
    logic wave_start_ready;
    logic coordinator_wave_start_ready;
    logic coordinator_compute_start_valid;
    logic coordinator_compute_start_ready;
    logic [3:0] wave_layer_index;
    logic [9:0] wave_output_base;
    logic [9:0] wave_output_count;
    logic [9:0] wave_input_base;
    logic [9:0] wave_input_count;
    logic [DDR_ADDRESS_WIDTH-1:0] wave_source;
    logic wave_done;
    logic wave_output_mode_q;
    logic absolute_weight_read_valid;
    logic absolute_weight_read_ready;
    logic [DDR_ADDRESS_WIDTH-1:0] absolute_weight_addr0;
    logic [DDR_ADDRESS_WIDTH-1:0] absolute_weight_addr1;
    logic absolute_weight_response_valid;
    logic absolute_weight_response_ready;
    logic [63:0] absolute_weight_data0;
    logic [63:0] absolute_weight_data1;
    logic coordinator_busy;
    logic coordinator_idle;
    logic coordinator_descriptor_active;
    logic coordinator_wave_active;
    logic coordinator_metadata_match;
    logic coordinator_read_pending;
    logic coordinator_response_pending;
    logic coordinator_abort_request;
    logic coordinator_error;
    logic [3:0] coordinator_error_code;

    logic scnn_layer_error_combined;
    logic [7:0] scnn_layer_error_combined_code;
    logic scnn_output_error_combined;
    logic [7:0] scnn_output_error_combined_code;
    logic memory_error_combined;
    logic [7:0] memory_error_combined_code;
    logic scratchpad_blocker;
    logic extension_command_pending;
    logic writeback_busy;
    logic dp_abort_flush;
    logic abort_result_quarantine_q;

    assign current_layer_type = dispatch_layer_words[2*32 +: 4];
    assign o_active_layer_index = dispatch_layer_index;
    // requested_owner is a dispatch routing value and remains at the last
    // layer owner after retirement. Expose only an owner that is physically
    // active so global-idle is observably owner-free.
    assign o_compute_owner = system_owner_active ? system_requested_owner :
                                                   2'd0;
    assign effective_locked_versions = layer_ingress_active_external ?
        {dispatch_state_version, locked_versions[47:0]} : locked_versions;
    assign o_active_versions = effective_locked_versions;
    assign external_scnn_session_active =
        ENABLE_EXTERNAL_SCNN_LAYER_INGRESS &&
        i_external_scnn_session_active;
    assign o_cfg_ready = control_cfg_ready &&
        !external_scnn_session_active;
    assign o_commit_ready = control_commit_ready &&
        !external_scnn_session_active;
    assign o_global_idle = control_global_idle &&
        !external_scnn_session_active && !external_weight_pending_q;
    assign scnn_profile_request = i_profile_code == 7'h20;
    assign tp_start_allowed_effective =
        !(ENABLE_TP_SYSTEM_BINDING || ENABLE_TP_SCNN_PARAMETER_READ) ||
        i_tp_start_allowed;
    assign tp_scnn_feature_ready_effective = !ENABLE_TP_SYSTEM_BINDING ||
                                             i_tp_scnn_feature_ready;
    assign tp_scnn_logit_ready_effective = !ENABLE_TP_SYSTEM_BINDING ||
                                           i_tp_scnn_logit_ready;
    assign o_start_ready = control_start_ready && !system_fail_stop &&
        !layer_ingress_busy && !external_scnn_session_active &&
        tp_start_allowed_effective &&
        (!scnn_profile_request || o_prefetch_start_ready);
    assign qualified_start = i_start && o_start_ready;
    assign o_prefetch_done = ddr_done;
    assign o_busy = control_busy || layer_ingress_busy ||
        external_scnn_session_active || external_weight_pending_q;
    assign dp_public_result_ready = i_result_ready ||
                                    abort_result_quarantine_q;
    assign o_result_valid = dp_public_result_valid &&
                            !abort_result_quarantine_q;
    assign o_result_class = dp_public_result_class;
    assign o_result_score = dp_public_result_score;
    assign o_result_versions = dp_public_result_versions;
    assign o_result_owner = dp_public_result_owner;

    assign public_result_fire = o_result_valid && i_result_ready;
    assign fc_result_retire = public_result_fire &&
                              o_result_owner == OWNER_FC;
    assign srnn_result_retire = public_result_fire &&
                                o_result_owner == OWNER_SRNN;
    assign scnn_result_retire = public_result_fire &&
                                 o_result_owner == OWNER_SCNN;

    // The physical drain tree spans every compute and memory owner. Register
    // its idle observation before the version/start control plane so activity
    // cannot form a device-wide combinational path into sequencer enables.
    // Admission is conservatively delayed by one cycle after drain completes.
    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            system_control_quiescent_q <= 1'b0;
        else if (i_abort)
            system_control_quiescent_q <= 1'b0;
        else
            system_control_quiescent_q <= system_datapath_quiescent;
    end

    // DDR prefetch contributes to the global-idle tree. Delay only this start
    // side effect so the public admission handshake cannot feed back through
    // prefetch pending/busy state into its own ready decision.
    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            ddr_prefetch_start <= 1'b0;
        else if (i_abort || system_fail_stop)
            ddr_prefetch_start <= 1'b0;
        else
            ddr_prefetch_start <= qualified_start && scnn_profile_request;
    end

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n) begin
            sample_state_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            fc_clear_state_q <= 1'b0;
            fc_sample_id_q <= '0;
            output_version_error_q <= 1'b0;
        end else begin
            if (qualified_start) begin
                sample_state_reset_q <= i_sample_state_reset;
                sample_last_q <= i_sample_last;
                fc_clear_state_q <= i_fc_clear_state;
                fc_sample_id_q <= i_fc_sample_id;
            end
            if (i_clear_error && o_global_idle)
                output_version_error_q <= 1'b0;
            else if (output_result_valid &&
                     output_result_version != dispatch_state_version)
                output_version_error_q <= 1'b1;
        end
    end

    unified_final_control_plane_v1 u_control_plane (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_cfg_valid(i_cfg_valid && !external_scnn_session_active),
        .o_cfg_ready(control_cfg_ready),
        .i_cfg_profile(i_cfg_profile), .i_cfg_table(i_cfg_table),
        .i_cfg_index(i_cfg_index), .i_cfg_data(i_cfg_data),
        .i_commit_valid(i_commit_valid && !external_scnn_session_active),
        .o_commit_ready(control_commit_ready),
        .i_commit_profile(i_commit_profile), .i_start(qualified_start),
        .o_start_ready(control_start_ready), .i_profile_code(i_profile_code),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .o_layer_valid(control_layer_valid),
        .i_layer_ready(control_layer_ready),
        .o_layer_index(control_layer_index),
        .o_layer_words(control_layer_words),
        .o_inbound_valid(control_inbound_valid),
        .o_inbound_sources_flat(control_inbound_sources_flat),
        .o_inbound_types_flat(control_inbound_types_flat),
        .i_layer_done(control_layer_done),
        .i_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .i_versioned_command_valid(1'b0),
        .i_versioned_command_versions(64'd0),
        .o_versioned_command_accept(versioned_accept_unused),
        .o_versioned_command_reject_stale(versioned_stale_unused),
        .o_versioned_command_reject_mixed(versioned_mixed_unused),
        .i_model_version(i_model_version),
        .i_adapt_version(i_adapt_version),
        .i_model_activation_pending(1'b0),
        .i_adapt_activation_pending(1'b0),
        .i_datapath_quiescent(system_control_quiescent_q),
        .i_fc_busy(dp_fc_path_busy), .i_extension_busy(dp_extension_busy),
        .i_dynamic_output_busy(dp_dynamic_output_busy),
        .i_scnn_li_busy(output_frontend_busy),
        .i_layer_pending(system_layer_pending),
        .i_scratchpad_busy(scratchpad_blocker),
        .i_prefetch_busy(ddr_memory_busy),
        .i_writeback_busy(writeback_busy),
        .i_ddr_cmd_pending(ddr_cmd_pending),
        .i_ddr_response_pending(ddr_response_pending),
        .i_ddr_cdc_busy(ddr_cdc_busy),
        .i_ddr_outstanding(ddr_physical_outstanding),
        .i_w2_commit_pending(dp_w2_commit_pending),
        .i_w2_snapshot_busy(dp_w2_snapshot_busy),
        .i_owner_active(system_drain_owner_active),
        .i_extension_command_pending(extension_command_pending),
        .i_abort_cleanup_pending(system_abort_cleanup_pending),
        .o_active_profile(o_active_profile),
        .o_active_profile_code(o_active_profile_code),
        .o_active_layer_count(o_active_layer_count),
        .o_active_route_count(o_active_route_count),
        .o_live_versions(live_versions),
        .o_locked_versions(locked_versions),
        .o_profile_code_legal(control_profile_legal),
        .o_sequence_busy(control_sequence_busy),
        .o_sample_active(control_sample_active),
        .o_sample_compute_done(control_sample_compute_done),
        .o_sample_done(o_sample_done),
        .o_commit_pending(control_commit_pending),
        .o_commit_pulse(control_commit_pulse),
        .o_config_copy_busy(control_config_copy_busy),
        .o_blocker_mask(control_blocker_mask),
        .o_datapath_quiescent_for_commit(control_quiescent_for_commit),
        .o_control_idle(control_idle),
        .o_global_idle(control_global_idle),
        .o_busy(control_busy),
        .o_version_observer_consistent(control_version_consistent),
        .o_sticky_error(control_error), .o_error_code(control_error_code)
    );

    assign external_layer_request_allowed =
        external_scnn_session_active && !control_sequence_busy &&
        !control_sample_active && !control_commit_pending &&
        !system_fail_stop && coordinator_idle && !ddr_memory_busy &&
        !ddr_cmd_pending && !ddr_response_pending && !tp_weight_busy &&
        !external_weight_pending_q;

    unified_final_layer_ingress_arbiter_v1 u_layer_ingress (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_abort(dp_abort_flush), .i_clear_error(i_clear_error),
        .i_internal_valid(control_layer_valid),
        .o_internal_ready(control_layer_ready),
        .i_internal_layer_index(control_layer_index),
        .i_internal_layer_words(control_layer_words),
        .i_internal_input_slot(control_layer_index[0]),
        .i_internal_output_slot(!control_layer_index[0]),
        .i_internal_state_version(locked_versions[63:48]),
        .i_internal_sample_reset(sample_state_reset_q),
        .i_internal_sample_last(sample_last_q),
        .i_internal_observation_enable(tp_conv_observer_enable_q &&
            control_layer_index == tp_conv_observer_layer_q),
        .i_internal_input_prescan_enable(tp_input_enable_q &&
            control_layer_index == tp_conv_observer_layer_q),
        .o_internal_done(control_layer_done),
        .i_external_valid(i_external_scnn_layer_valid &&
            external_layer_request_allowed),
        .i_external_session_active(external_scnn_session_active),
        .o_external_ready(o_external_scnn_layer_ready),
        .i_external_layer_index(i_external_scnn_layer_index),
        .i_external_layer_words(i_external_scnn_layer_words),
        .i_external_input_slot(i_external_scnn_input_slot),
        .i_external_output_slot(i_external_scnn_output_slot),
        .i_external_state_version(i_external_scnn_state_version),
        .i_external_sample_reset(i_external_scnn_sample_reset),
        .i_external_sample_last(i_external_scnn_sample_last),
        .i_external_observation_enable(
            i_external_scnn_observation_enable),
        .i_external_input_prescan_enable(
            i_external_scnn_input_prescan_enable),
        .o_external_done(o_external_scnn_layer_done),
        .o_shared_valid(dispatch_layer_valid),
        .i_shared_ready(dispatch_layer_ready),
        .o_shared_layer_index(dispatch_layer_index),
        .o_shared_layer_words(dispatch_layer_words),
        .o_shared_input_slot(dispatch_input_slot),
        .o_shared_output_slot(dispatch_output_slot),
        .o_shared_state_version(dispatch_state_version),
        .o_shared_sample_reset(dispatch_sample_reset),
        .o_shared_sample_last(dispatch_sample_last),
        .o_shared_observation_enable(dispatch_observation_enable),
        .o_shared_input_prescan_enable(dispatch_input_prescan_enable),
        .i_shared_done(dispatch_layer_done),
        .i_shared_error(1'b0),
        .o_active_external(layer_ingress_active_external),
        .o_busy(layer_ingress_busy),
        .o_protocol_error(layer_ingress_error)
    );

    assign o_external_scnn_layer_busy = layer_ingress_busy;
    assign o_external_scnn_layer_protocol_error = layer_ingress_error;

    unified_final_system_control_v1 u_system_control (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_layer_valid(dispatch_layer_valid),
        .o_layer_ready(dispatch_layer_ready),
        .i_layer_type(current_layer_type),
        .o_fc_valid(system_fc_valid), .i_fc_ready(system_fc_ready),
        .i_fc_done(system_fc_done),
        .o_recurrent_valid(system_recurrent_valid),
        .i_recurrent_ready(system_recurrent_ready),
        .i_recurrent_done(system_recurrent_done),
        .o_scnn_valid(system_scnn_valid), .i_scnn_ready(system_scnn_ready),
        .i_scnn_done(system_scnn_done),
        .o_output_valid(system_output_valid),
        .i_output_ready(system_output_ready),
        .i_output_done(system_output_done),
        .o_layer_done(dispatch_layer_done),
        .o_layer_accept(system_layer_accept),
        .o_requested_owner(system_requested_owner),
        .o_active_layer_type(o_active_layer_type),
        .o_owner_active(system_owner_active),
        .o_dispatch_busy(system_dispatch_busy),
        .i_fc_busy(dp_fc_path_busy), .i_extension_busy(dp_extension_busy),
        .i_dynamic_output_busy(dp_dynamic_output_busy),
        .i_scnn_finalize_busy(conv_layer_busy || pool_busy || tp_input_busy),
        .i_scnn_li_busy(output_frontend_busy || output_adapter_busy),
        .i_scratchpad_busy(scratchpad_blocker),
        .i_prefetch_busy(ddr_memory_busy),
        .i_writeback_busy(writeback_busy),
        .i_ddr_cmd_pending(ddr_cmd_pending),
        .i_ddr_response_pending(ddr_response_pending),
        .i_ddr_cdc_busy(ddr_cdc_busy),
        .i_ddr_outstanding(ddr_physical_outstanding),
        .i_w2_commit_pending(dp_w2_commit_pending),
        .i_w2_snapshot_busy(dp_w2_snapshot_busy),
        .i_extension_command_pending(extension_command_pending),
        .i_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .i_control_error(control_error), .i_control_code(control_error_code),
        .i_datapath_error(dp_error || srnn_adapter_error ||
                          layer_ingress_error),
        .i_datapath_code(dp_error ? dp_error_code : srnn_adapter_error ?
                         srnn_adapter_error_code : 8'h95),
        .i_scnn_layer_error(scnn_layer_error_combined),
        .i_scnn_layer_code(scnn_layer_error_combined_code),
        .i_scnn_output_error(scnn_output_error_combined),
        .i_scnn_output_code(scnn_output_error_combined_code),
        .i_memory_error(memory_error_combined),
        .i_memory_code(memory_error_combined_code),
        .o_datapath_quiescent(system_datapath_quiescent),
        .o_layer_pending(system_layer_pending),
        .o_drain_owner_active(system_drain_owner_active),
        .o_owner_tracking_consistent(system_owner_tracking_consistent),
        .o_result_pending(system_result_pending),
        .o_abort_cleanup_pending(system_abort_cleanup_pending),
        .o_physical_blocker_mask(system_physical_blocker_mask),
        .o_dispatch_error(system_dispatch_error),
        .o_dispatch_error_code(system_dispatch_error_code),
        .o_fail_stop(system_fail_stop), .o_sticky_error(o_sticky_error),
        .o_error_source(o_error_source), .o_error_code(o_error_code)
    );

    assign system_fc_ready = dp_fc_layer_ready;
    assign system_fc_done = fc_result_retire;
    assign system_recurrent_ready = srnn_adapter_layer_ready;
    assign system_recurrent_done = srnn_adapter_done;
    assign conv_request_valid = system_scnn_valid &&
                              current_layer_type == LAYER_CONV;
    assign conv_layer_valid = conv_request_valid;
    assign pool_valid = system_scnn_valid &&
                        current_layer_type == LAYER_POOL;
    assign system_scnn_ready = current_layer_type == LAYER_POOL ?
        pool_ready : conv_admission_ready;
    assign system_scnn_done = current_layer_type == LAYER_POOL ?
        pool_done : conv_layer_done;
    assign system_output_ready = output_adapter_ready;
    assign system_output_done = output_adapter_done;

    unified_srnn_graph_layer_adapter_v1 u_srnn_adapter (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_layer_valid(system_recurrent_valid),
        .o_layer_ready(srnn_adapter_layer_ready),
        .i_layer_words(dispatch_layer_words),
        .i_sample_state_reset(dispatch_sample_reset),
        .i_sample_version(dispatch_state_version),
        .o_engine_start_valid(srnn_engine_start_valid),
        .i_engine_start_ready(srnn_engine_start_ready),
        .i_engine_done(srnn_result_retire), .i_engine_error(dp_error),
        .o_state_reset(srnn_state_reset), .o_input_dim(srnn_input_dim),
        .o_active_neurons(srnn_active_neurons),
        .o_active_tiles(srnn_active_tiles),
        .o_output_count(srnn_output_count), .o_w2_base(srnn_w2_base),
        .o_w2_row_stride(srnn_w2_row_stride),
        .o_recurrent_base(srnn_recurrent_base),
        .o_recurrent_enable(srnn_recurrent_enable),
        .o_w0_base(srnn_w0_base),
        .o_leak_q15(srnn_leak_q15), .o_threshold(srnn_threshold),
        .o_current_left_shift(srnn_current_left_shift),
        .o_reset_to_zero(srnn_reset_to_zero), .o_version(srnn_version),
        .o_layer_done(srnn_adapter_done), .o_busy(srnn_adapter_busy),
        .o_idle(srnn_adapter_idle), .o_sticky_error(srnn_adapter_error),
        .o_error_code(srnn_adapter_error_code)
    );

    unified_fc_srnn_shared_datapath_v1 #(
        .FC_WEIGHT_DIR(FC_WEIGHT_DIR),
        .FC_COMPAT_WEIGHT_DIR(FC_COMPAT_WEIGHT_DIR),
        .USE_LEGACY_CANONICAL_FC(USE_LEGACY_CANONICAL_FC),
        .USE_SHARED_FC_SRNN_STATE(USE_SHARED_FC_SRNN_STATE),
        .FC_PHYSICAL_TILES(FC_PHYSICAL_TILES),
        .ENGINES(ENGINES), .LANES(LANES),
        .HIDDEN_MAX(HIDDEN_MAX), .INPUT_MAX(INPUT_MAX),
        .OUTPUT_MAX(OUTPUT_MAX), .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .ENABLE_FC_TP_NATIVE_WEIGHTS(ENABLE_FC_TP_NATIVE_WEIGHTS),
        .FC_LEARN_W0(FC_LEARN_W0),
        .FC_TP_W0_RESIDUAL_INIT_FILE(FC_TP_W0_RESIDUAL_INIT_FILE)
    ) u_shared_datapath (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_abort(dp_abort_flush),
        .i_fc_tp_w0_parameter_base(i_tp_fc_w0_parameter_base),
        .i_fc_tp_w2_parameter_base(i_tp_fc_w2_parameter_base),
        .i_fc_tp_learning_read_valid(i_tp_fc_learning_read_valid),
        .o_fc_tp_learning_read_ready(o_tp_fc_learning_read_ready),
        .i_fc_tp_learning_read_tensor_id(i_tp_fc_learning_read_tensor_id),
        .i_fc_tp_learning_read_parameter_addr(
            i_tp_fc_learning_read_parameter_addr),
        .i_fc_tp_learning_read_tag(i_tp_fc_learning_read_tag),
        .o_fc_tp_learning_response_valid(o_tp_fc_learning_response_valid),
        .i_fc_tp_learning_response_ready(i_tp_fc_learning_response_ready),
        .o_fc_tp_learning_response_tag(o_tp_fc_learning_response_tag),
        .o_fc_tp_learning_response_code(o_tp_fc_learning_response_code),
        .o_fc_tp_learning_response_residual(
            o_tp_fc_learning_response_residual),
        .o_fc_tp_learning_response_error(o_tp_fc_learning_response_error),
        .i_fc_tp_learning_write_valid(i_tp_fc_learning_write_valid),
        .o_fc_tp_learning_write_ready(o_tp_fc_learning_write_ready),
        .i_fc_tp_learning_write_tensor_id(i_tp_fc_learning_write_tensor_id),
        .i_fc_tp_learning_write_parameter_addr(
            i_tp_fc_learning_write_parameter_addr),
        .i_fc_tp_learning_write_code(i_tp_fc_learning_write_code),
        .i_fc_tp_learning_write_residual(i_tp_fc_learning_write_residual),
        .i_fc_tp_learning_write_code_enable(
            i_tp_fc_learning_write_code_enable),
        .i_fc_tp_learning_write_residual_enable(
            i_tp_fc_learning_write_residual_enable),
        .o_fc_tp_learning_quiescent(o_tp_fc_learning_quiescent),
        .i_requested_owner(system_requested_owner),
        .i_locked_versions(effective_locked_versions),
        .i_fc_layer_valid(system_fc_valid),
        .o_fc_layer_ready(dp_fc_layer_ready),
        .i_fc_layer_words(dispatch_layer_words),
        .i_fc_clear_state(fc_clear_state_q),
        .i_fc_sample_id(fc_sample_id_q),
        .i_srnn_start_valid(srnn_engine_start_valid),
        .o_srnn_start_ready(srnn_engine_start_ready),
        .i_srnn_state_reset(srnn_state_reset),
        .i_srnn_input_dim(srnn_input_dim),
        .i_srnn_active_neurons(srnn_active_neurons),
        .i_srnn_active_tiles(srnn_active_tiles),
        .i_srnn_w0_base(srnn_w0_base),
        .i_srnn_recurrent_base(srnn_recurrent_base),
        .i_srnn_recurrent_enable(srnn_recurrent_enable),
        .i_srnn_leak_q15(srnn_leak_q15),
        .i_srnn_threshold(srnn_threshold),
        .i_srnn_current_left_shift(srnn_current_left_shift),
        .i_srnn_reset_to_zero(srnn_reset_to_zero),
        .i_scnn_start_valid(dp_scnn_start_valid),
        .o_scnn_start_ready(dp_scnn_start_ready),
        .i_scnn_in_width(output_memory_owner ? 16'd1 : conv_in_width),
        .i_scnn_in_height(output_memory_owner ? 16'd1 : conv_in_height),
        .i_scnn_total_in_channels(output_memory_owner ? 16'd1 :
                                  conv_total_in_channels),
        .i_scnn_out_width(output_memory_owner ? 16'd1 : conv_out_width),
        .i_scnn_out_height(output_memory_owner ? 16'd1 : conv_out_height),
        .i_scnn_total_out_channels(output_memory_owner ? 16'd1 :
                                   conv_total_out_channels),
        .i_scnn_input_channel_base(output_memory_owner ? 10'd0 :
                                   conv_input_channel_base),
        .i_scnn_input_channel_count(output_memory_owner ? 10'd1 :
                                    conv_input_channel_count),
        .i_scnn_output_channel_base(output_memory_owner ? 10'd0 :
                                    conv_output_channel_base),
        .i_scnn_output_channel_count(output_memory_owner ? 10'd1 :
                                     conv_output_channel_count),
        .i_scnn_stride(output_memory_owner ? 8'd1 : conv_stride),
        .i_scnn_padding(output_memory_owner ? 8'd0 : conv_padding),
        .i_scnn_input_slot(output_memory_owner ? output_frontend_input_slot :
                           conv_input_slot),
        .i_scnn_feature_base(output_memory_owner ? '0 : conv_feature_base),
        .i_scnn_weight_q16_base(output_memory_owner ?
            output_frontend_layer_words[7*32 +: Q16_ADDR_WIDTH] :
            conv_weight_q16_base),
        .i_scnn_output_base(output_memory_owner ? '0 : conv_output_base),
        .i_scnn_unit_spike_features(output_memory_owner ? 1'b0 :
                                    conv_unit_spike_features),
        .i_scnn_external_mode(dp_scnn_external_mode),
        .i_event_window_valid(i_event_window_valid),
        .o_event_window_ready(o_event_window_ready),
        .i_event_record_valid(i_event_record_valid),
        .i_event_window_sample_id(i_event_window_sample_id),
        .i_event_window_timestep(i_event_window_timestep),
        .i_event_boundary_mask(i_event_boundary_mask),
        .i_event_window_sample_last(i_event_window_sample_last),
        .i_event_addr_flat(i_event_addr_flat),
        .i_event_data_flat(i_event_data_flat),
        .o_srnn_weight_req_valid(o_srnn_weight_req_valid),
        .i_srnn_weight_req_ready(i_srnn_weight_req_ready),
        .o_srnn_weight_req_version(o_srnn_weight_req_version),
        .o_srnn_weight_req_id(o_srnn_weight_req_id),
        .o_srnn_weight_req_recurrent(o_srnn_weight_req_recurrent),
        .o_srnn_weight_req_row_base(o_srnn_weight_req_row_base),
        .o_srnn_weight_req_destination_base(
            o_srnn_weight_req_destination_base),
        .o_srnn_weight_req_engine_mask(o_srnn_weight_req_engine_mask),
        .o_srnn_weight_req_lane_mask(o_srnn_weight_req_lane_mask),
        .i_srnn_weight_rsp_valid(i_srnn_weight_rsp_valid),
        .o_srnn_weight_rsp_ready(o_srnn_weight_rsp_ready),
        .i_srnn_weight_rsp_version(i_srnn_weight_rsp_version),
        .i_srnn_weight_rsp_id(i_srnn_weight_rsp_id),
        .i_srnn_weight_rsp_destination_base(
            i_srnn_weight_rsp_destination_base),
        .i_srnn_weight_rsp_engine_mask(i_srnn_weight_rsp_engine_mask),
        .i_srnn_weight_rsp_flat(i_srnn_weight_rsp_flat),
        .o_srnn_w2_req_valid(o_srnn_w2_req_valid),
        .i_srnn_w2_req_ready(i_srnn_w2_req_ready),
        .o_srnn_w2_addr_flat(o_srnn_w2_addr_flat),
        .i_srnn_w2_data_flat(i_srnn_w2_data_flat),
        .i_srnn_w2_base(srnn_w2_base),
        .i_srnn_w2_row_stride(srnn_w2_row_stride),
        .i_srnn_output_count(srnn_output_count),
        .o_scnn_feature_read_valid(dp_scnn_feature_read_valid),
        .i_scnn_feature_read_ready(dp_scnn_feature_read_ready),
        .o_scnn_feature_read_slot(dp_scnn_feature_read_slot),
        .o_scnn_feature_read_bank_flat(dp_scnn_feature_read_bank_flat),
        .o_scnn_feature_read_addr_flat(dp_scnn_feature_read_addr_flat),
        .o_scnn_feature_read_lane_mask(dp_scnn_feature_read_lane_mask),
        .i_scnn_feature_valid(dp_scnn_feature_valid),
        .o_scnn_feature_ready(dp_scnn_feature_ready),
        .i_scnn_feature_flat(dp_scnn_feature_flat),
        .i_scnn_feature_lane_mask(dp_scnn_feature_lane_mask),
        .o_scnn_weight_read_valid(dp_scnn_weight_read_valid),
        .i_scnn_weight_read_ready(dp_scnn_weight_read_ready),
        .o_scnn_weight_read_addr0(dp_scnn_weight_read_addr0),
        .o_scnn_weight_read_addr1(dp_scnn_weight_read_addr1),
        .i_scnn_weight_result_valid(dp_scnn_weight_result_valid),
        .o_scnn_weight_result_ready(dp_scnn_weight_result_ready),
        .i_scnn_weight_data0(dp_scnn_weight_data0),
        .i_scnn_weight_data1(dp_scnn_weight_data1),
        .o_scnn_raw_result_valid(dp_scnn_raw_result_valid),
        .i_scnn_raw_result_ready(dp_scnn_raw_result_ready),
        .o_scnn_raw_result_mask(dp_scnn_raw_result_mask),
        .o_scnn_raw_result_output_addr_flat(dp_scnn_raw_result_addr_flat),
        .o_scnn_raw_result_spatial_addr(dp_scnn_raw_spatial_addr),
        .o_scnn_raw_result_accumulator_flat(dp_scnn_raw_accumulator_flat),
        .i_scnn_external_request_valid(dp_scnn_external_request_valid),
        .o_scnn_external_request_ready(dp_scnn_external_request_ready),
        .i_scnn_external_engine_mask(dp_scnn_external_engine_mask),
        .i_scnn_external_feature_flat(dp_scnn_external_feature_flat),
        .i_scnn_external_weight_flat(dp_scnn_external_weight_flat),
        .i_scnn_external_lane_mask(dp_scnn_external_lane_mask),
        .i_scnn_external_first(dp_scnn_external_first),
        .i_scnn_external_last(dp_scnn_external_last),
        .i_scnn_external_output_channel_base(dp_scnn_external_output_base),
        .o_scnn_external_result_valid(dp_scnn_external_result_valid),
        .i_scnn_external_result_ready(dp_scnn_external_result_ready),
        .o_scnn_external_result_mask(dp_scnn_external_result_mask),
        .o_scnn_external_result_accumulator_flat(
            dp_scnn_external_result_accumulator_flat),
        .i_scnn_external_done(output_wave3_external_done),
        .i_scnn_external_sticky_error(output_adapter_error ||
                                      output_frontend_error),
        .i_scnn_li_result_valid(dp_scnn_li_result_valid),
        .o_scnn_li_result_ready(dp_scnn_li_result_ready),
        .i_scnn_li_result_class(dp_scnn_li_result_class),
        .i_scnn_li_result_score(dp_scnn_li_result_score),
        .o_result_valid(dp_public_result_valid),
        .i_result_ready(dp_public_result_ready),
        .o_result_class(dp_public_result_class),
        .o_result_score(dp_public_result_score),
        .o_result_versions(dp_public_result_versions),
        .o_result_owner(dp_public_result_owner),
        .o_fc_compute_done(dp_fc_compute_done),
        .o_srnn_compute_done(dp_srnn_compute_done),
        .o_scnn_compute_done(dp_scnn_compute_done),
        .o_fc_logits_flat(dp_fc_logits_flat),
        .o_fc_spikes_flat(dp_fc_spikes_flat),
        .o_fc_vmem_flat(dp_fc_vmem_flat),
        .o_fc_trace_flat(dp_fc_trace_flat),
        .o_srnn_logits_flat(dp_srnn_logits_flat),
        .o_fc_sample_cycles(o_fc_sample_cycles),
        .o_fc_locked_cluster_mask(o_fc_locked_cluster_mask),
        .o_srnn_previous_spikes(dp_srnn_previous_spikes),
        .o_tp_srnn_timestep_commit(o_tp_srnn_timestep_commit),
        .o_tp_srnn_timestep_version(o_tp_srnn_timestep_version),
        .o_tp_srnn_timestep_sample_last(
            o_tp_srnn_timestep_sample_last),
        .o_tp_srnn_timestep_spikes(o_tp_srnn_timestep_spikes),
        .o_extension_owner(dp_extension_owner),
        .o_result_pending(dp_result_pending), .o_fc_busy(dp_fc_busy),
        .o_extension_busy(dp_extension_busy),
        .o_dynamic_output_busy(dp_dynamic_output_busy),
        .o_ingress_busy(dp_ingress_busy),
        .o_fc_w2_commit_pending(dp_w2_commit_pending),
        .o_fc_w2_snapshot_busy(dp_w2_snapshot_busy),
        .o_busy(dp_busy), .o_idle(dp_idle),
        .o_sticky_error(dp_error), .o_error_code(dp_error_code)
    );

    assign o_tp_fc_compute_done = dp_fc_compute_done;
    assign o_tp_fc_logits_flat = dp_fc_logits_flat;
    assign o_tp_fc_spikes_flat = dp_fc_spikes_flat;
    assign o_tp_fc_vmem_flat = dp_fc_vmem_flat;
    assign o_tp_fc_trace_flat = dp_fc_trace_flat;

    // Lock the tap selection with the admitted inference, not live host pins.
    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n) begin
            tp_conv_observer_enable_q <= 1'b0;
            tp_input_enable_q <= 1'b0;
            tp_conv_observer_layer_q <= '0;
        end else if (i_start && o_start_ready) begin
            tp_conv_observer_enable_q <= ENABLE_TP_SCNN_CONV_OBSERVER &&
                i_tp_scnn_conv_observation_enable;
            tp_input_enable_q <= ENABLE_TP_SCNN_INPUT_OBSERVER &&
                i_tp_scnn_input_observation_enable;
            tp_conv_observer_layer_q <=
                (ENABLE_TP_SCNN_CONV_OBSERVER || ENABLE_TP_SCNN_INPUT_OBSERVER) ?
                i_tp_scnn_conv_selected_layer : '0;
        end
    end
    assign o_tp_scnn_conv_observation_layer =
        layer_ingress_active_external ? dispatch_layer_index :
        tp_conv_observer_layer_q;
    assign o_tp_scnn_selected_conv_done = conv_layer_done &&
        dispatch_observation_enable;

    generate
        if (ENABLE_TP_SCNN_INPUT_OBSERVER) begin : g_tp_input_observer
            unified_scnn_tp_input_prescan #(
                .ADDR_WIDTH(FEATURE_ADDR_WIDTH), .BANK_DEPTH(FEATURE_DEPTH)
            ) u_prescan (
                .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
                .i_abort(dp_abort_flush), .i_clear_error(i_clear_error),
                .i_enable(dispatch_input_prescan_enable),
                // Dispatch emits an acceptance pulse, not a held request.
                // Start from the pending descriptor; only dispatch starts CONV.
                .i_layer_valid(dispatch_layer_valid && !system_dispatch_busy &&
                    !system_fail_stop && current_layer_type == LAYER_CONV),
                .o_layer_ready(conv_admission_ready),
                .o_layer_valid(), .i_layer_ready(conv_layer_ready),
                .i_width({10'd0, dispatch_layer_words[5*32 +: 6]}),
                .i_height({10'd0, dispatch_layer_words[5*32+6 +: 6]}),
                .i_channels({6'd0, dispatch_layer_words[5*32+12 +: 10]}),
                .i_slot(dispatch_input_slot),
                .i_base({FEATURE_ADDR_WIDTH{1'b0}}),
                .i_layer(dispatch_layer_index),
                .i_version(dispatch_state_version),
                .o_read_valid(tp_input_read_valid), .i_read_ready(conv_memory_read_ready),
                .o_read_slot(tp_input_read_slot), .o_read_banks(tp_input_read_banks),
                .o_read_addrs(tp_input_read_addrs), .o_read_mask(tp_input_read_mask),
                .i_response_valid(conv_memory_result_valid && tp_input_owner),
                .o_response_ready(tp_input_response_ready),
                .i_response_data(conv_memory_result_data), .i_response_mask(conv_memory_result_mask),
                .o_observation_valid(o_tp_scnn_input_observation_valid),
                .i_observation_ready(i_tp_scnn_input_observation_ready),
                .o_channel(o_tp_scnn_input_observation_channel),
                .o_spatial(o_tp_scnn_input_observation_spatial),
                .o_layer(o_tp_scnn_input_observation_layer),
                .o_version(o_tp_scnn_input_observation_version),
                .o_data_q8_8(o_tp_scnn_input_observation_data),
                .o_last(o_tp_scnn_input_observation_last),
                .o_read_owner(tp_input_owner), .o_busy(tp_input_busy), .o_error(tp_input_error)
            );
        end else begin : g_no_tp_input_observer
            assign conv_admission_ready = conv_layer_ready;
            assign tp_input_read_valid = 1'b0;
            assign tp_input_read_slot = 1'b0;
            assign tp_input_read_banks = '0;
            assign tp_input_read_addrs = '0;
            assign tp_input_read_mask = '0;
            assign tp_input_response_ready = 1'b0;
            assign tp_input_owner = 1'b0;
            assign tp_input_busy = 1'b0;
            assign tp_input_error = 1'b0;
            assign o_tp_scnn_input_observation_valid = 1'b0;
            assign o_tp_scnn_input_observation_channel = '0;
            assign o_tp_scnn_input_observation_spatial = '0;
            assign o_tp_scnn_input_observation_layer = '0;
            assign o_tp_scnn_input_observation_version = '0;
            assign o_tp_scnn_input_observation_data = '0;
            assign o_tp_scnn_input_observation_last = 1'b0;
        end
    endgenerate

    unified_scnn_external_shared_layer_v1 #(
        .ENGINES(ENGINES), .LANES(LANES),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .SPATIAL_DEPTH(SCNN_SPATIAL_DEPTH),
        .SPATIAL_ADDR_WIDTH(SCNN_SPATIAL_ADDR_WIDTH),
        .STATE_ADDR_WIDTH(OUTPUT_ADDR_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .BANK_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .ENABLE_TP_OBSERVER(ENABLE_TP_SCNN_CONV_OBSERVER),
        .STATE_INIT_FILE(SCNN_STATE_INIT_FILE)
    ) u_scnn_conv_layer (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_abort(dp_abort_flush),
        .i_clear_error(i_clear_error), .i_layer_valid(conv_layer_valid),
        .o_layer_ready(conv_layer_ready),
        .i_layer_words(dispatch_layer_words),
        .i_input_slot(dispatch_input_slot),
        .i_output_slot(dispatch_output_slot),
        .i_sample_version(dispatch_state_version),
        .o_conv_start_valid(conv_start_valid),
        .i_conv_start_ready(conv_start_ready),
        .o_conv_in_width(conv_in_width), .o_conv_in_height(conv_in_height),
        .o_conv_total_in_channels(conv_total_in_channels),
        .o_conv_out_width(conv_out_width),
        .o_conv_out_height(conv_out_height),
        .o_conv_total_out_channels(conv_total_out_channels),
        .o_conv_input_channel_base(conv_input_channel_base),
        .o_conv_input_channel_count(conv_input_channel_count),
        .o_conv_output_channel_base(conv_output_channel_base),
        .o_conv_output_channel_count(conv_output_channel_count),
        .o_conv_stride(conv_stride), .o_conv_padding(conv_padding),
        .o_conv_input_slot(conv_input_slot),
        .o_conv_feature_base(conv_feature_base),
        .o_conv_weight_q16_base(conv_weight_q16_base),
        .o_conv_output_base(conv_output_base),
        .o_conv_unit_spike_features(conv_unit_spike_features),
        .i_conv_done(dp_scnn_compute_done),
        .i_conv_idle(!dp_extension_busy),
        .i_conv_sticky_error(dp_error), .i_conv_error_code(dp_error_code),
        .i_raw_result_valid(dp_scnn_raw_result_valid),
        .o_raw_result_ready(dp_scnn_raw_result_ready),
        .i_raw_result_mask(dp_scnn_raw_result_mask),
        .i_raw_result_output_addr_flat(dp_scnn_raw_result_addr_flat),
        .i_raw_result_spatial_addr(dp_scnn_raw_spatial_addr),
        .i_raw_result_accumulator_flat(dp_scnn_raw_accumulator_flat),
        .o_raw_result_fire(), .o_raw_result_group_first(),
        .o_raw_result_group_final(),
        .o_feature_write_valid(conv_feature_write_valid),
        .i_feature_write_ready(conv_feature_write_ready),
        .o_feature_write_slot(conv_feature_write_slot),
        .o_feature_write_bank(conv_feature_write_bank),
        .o_feature_write_addr(conv_feature_write_addr),
        .o_feature_write_data_q8_8(conv_feature_write_data),
        .i_observation_enable(dispatch_observation_enable),
        .o_observation_valid(o_tp_scnn_conv_observation_valid),
        .i_observation_ready(!ENABLE_TP_SCNN_CONV_OBSERVER ||
            i_tp_scnn_conv_observation_ready),
        .o_observation_channel(o_tp_scnn_conv_observation_channel),
        .o_observation_spatial_addr(o_tp_scnn_conv_observation_spatial),
        .o_observation_output_addr(o_tp_scnn_conv_observation_addr),
        .o_observation_sample_version(o_tp_scnn_conv_observation_version),
        .o_observation_current_q8_8(o_tp_scnn_conv_observation_current),
        .o_observation_spike_q8_8(o_tp_scnn_conv_observation_spike),
        .o_observation_membrane_pre_q8_8(
            o_tp_scnn_conv_observation_membrane_pre),
        .o_observation_membrane_next_q8_8(
            o_tp_scnn_conv_observation_membrane_next),
        .o_observation_surrogate_q8_8(o_tp_scnn_conv_observation_surrogate),
        .o_locked_sample_version(conv_locked_version),
        .o_layer_done(conv_layer_done), .o_busy(conv_layer_busy),
        .o_idle(conv_layer_idle), .o_sticky_error(conv_layer_error),
        .o_error_code(conv_layer_error_code)
    );

    assign o_tp_scnn_feature_write_slot = conv_feature_write_slot;
    assign o_tp_scnn_feature_write_bank = conv_feature_write_bank;
    assign o_tp_scnn_feature_write_addr = conv_feature_write_addr;
    assign o_tp_scnn_feature_write_data = conv_feature_write_data;
    generate
        if (ENABLE_TP_SCNN_FEATURE_WRITE_TAP) begin : g_scnn_feature_write_tap
            assign o_tp_scnn_feature_write_valid =
                conv_feature_write_valid && conv_store_write_ready;
            assign conv_store_write_valid =
                conv_feature_write_valid && i_tp_scnn_feature_write_ready;
            assign conv_feature_write_ready =
                conv_store_write_ready && i_tp_scnn_feature_write_ready;
        end else begin : g_no_scnn_feature_write_tap
            assign o_tp_scnn_feature_write_valid = 1'b0;
            assign conv_store_write_valid = conv_feature_write_valid;
            assign conv_feature_write_ready = conv_store_write_ready;
        end
    endgenerate

    unified_final_pool_reject_v1 u_pool_reject (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_layer_valid(pool_valid), .o_layer_ready(pool_ready),
        .o_layer_done(pool_done), .o_busy(pool_busy),
        .o_sticky_error(pool_error), .o_error_code(pool_error_code)
    );

    unified_scnn_output_owner_adapter_v1 u_output_owner_adapter (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_abort(dp_abort_flush),
        .i_layer_valid(system_output_valid),
        .o_layer_ready(output_adapter_ready),
        .i_layer_words(dispatch_layer_words),
        .i_input_slot(dispatch_input_slot),
        .i_sample_reset(dispatch_sample_reset),
        .i_sample_last(dispatch_sample_last),
        .i_version(dispatch_state_version),
        .o_wave3_start_valid(output_wave3_start_valid),
        .i_wave3_start_ready(output_wave3_start_ready),
        .o_wave3_external_mode(output_wave3_external_mode),
        .o_wave3_external_done(output_wave3_external_done),
        .o_wave3_clear_error(output_wave3_clear_error),
        // Datapath errors are aggregated by system control. Feeding the
        // aggregate back here would create a loop through external error.
        .i_wave3_sticky_error(1'b0),
        .i_wave3_error_code(8'h00),
        .o_frontend_owner_enable(output_frontend_owner_enable),
        .o_frontend_clear_error(output_frontend_clear_error),
        .o_frontend_layer_valid(output_frontend_layer_valid),
        .i_frontend_layer_ready(output_frontend_layer_ready),
        .o_frontend_layer_words(output_frontend_layer_words),
        .o_frontend_input_slot(output_frontend_input_slot),
        .o_frontend_sample_reset(output_frontend_sample_reset),
        .o_frontend_sample_last(output_frontend_sample_last),
        .o_frontend_version(output_frontend_version),
        .i_frontend_busy(output_frontend_busy),
        .i_frontend_done(output_frontend_done),
        .i_frontend_sticky_error(output_frontend_error),
        .i_frontend_error_code(output_frontend_error_code),
        .i_public_result_retire(scnn_result_retire),
        .o_frontend_enqueued(output_frontend_enqueued),
        .o_result_enqueued(output_result_enqueued),
        .o_layer_done(output_adapter_done),
        .o_owner_held(output_owner_held), .o_busy(output_adapter_busy),
        .o_idle(output_adapter_idle), .o_sticky_error(output_adapter_error),
        .o_error_code(output_adapter_error_code)
    );

    unified_scnn_shared_output_frontend_v1 #(
        .ENGINES(ENGINES), .LANES(LANES), .OUTPUT_MAX(OUTPUT_MAX),
        .INPUT_FEATURES(SCNN_CLASSIFIER_FEATURES),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_scnn_output_frontend (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_abort(dp_abort_flush),
        .i_owner_enable(output_frontend_owner_enable),
        .i_clear_error(output_frontend_clear_error),
        .i_layer_valid(output_frontend_layer_valid),
        .o_layer_ready(output_frontend_layer_ready),
        .i_layer_words(output_frontend_layer_words),
        .i_input_slot(output_frontend_input_slot),
        .i_sample_reset(output_frontend_sample_reset),
        .i_sample_last(output_frontend_sample_last),
        .i_version(output_frontend_version),
        .o_feature_read_valid(output_feature_read_valid),
        .i_feature_read_ready(output_feature_read_ready),
        .o_feature_read_slot(output_feature_read_slot),
        .o_feature_read_bank_flat(output_feature_read_bank_flat),
        .o_feature_read_addr_flat(output_feature_read_addr_flat),
        .o_feature_read_lane_mask(output_feature_read_lane_mask),
        .i_feature_valid(output_feature_valid),
        .o_feature_ready(output_feature_ready),
        .i_feature_flat(output_feature_flat),
        .i_feature_lane_mask(output_feature_lane_mask),
        .o_weight_read_valid(output_weight_read_valid),
        .i_weight_read_ready(output_weight_read_ready),
        .o_weight_read_addr0(output_weight_read_addr0),
        .o_weight_read_addr1(output_weight_read_addr1),
        .i_weight_result_valid(output_weight_result_valid),
        .o_weight_result_ready(output_weight_result_ready),
        .i_weight_data0(output_weight_data0),
        .i_weight_data1(output_weight_data1),
        .o_feature_trace_wave_valid(output_feature_trace_valid),
        .i_feature_trace_wave_ready(output_observation_enable ?
            tp_scnn_feature_ready_effective : 1'b1),
        .o_feature_trace_wave_mask(o_tp_scnn_feature_mask),
        .o_feature_trace_wave_base(o_tp_scnn_feature_base),
        .o_feature_trace_wave_q8_8_flat(o_tp_scnn_feature_q8_8_flat),
        .o_feature_trace_wave_version(o_tp_scnn_feature_version),
        .o_feature_trace_wave_frame_index(o_tp_scnn_feature_frame),
        .o_feature_trace_wave_first(o_tp_scnn_feature_first),
        .o_feature_trace_wave_last(o_tp_scnn_feature_last),
        .o_feature_trace_wave_sample_reset(
            o_tp_scnn_feature_sample_reset),
        .o_feature_trace_wave_sample_last(o_tp_scnn_feature_sample_last),
        .o_side_request_valid(dp_scnn_external_request_valid),
        .i_side_request_ready(dp_scnn_external_request_ready),
        .o_side_engine_mask(dp_scnn_external_engine_mask),
        .o_side_feature_flat(dp_scnn_external_feature_flat),
        .o_side_weight_flat(dp_scnn_external_weight_flat),
        .o_side_lane_mask(dp_scnn_external_lane_mask),
        .o_side_first(dp_scnn_external_first),
        .o_side_last(dp_scnn_external_last),
        .o_side_output_channel_base(dp_scnn_external_output_base),
        .i_side_result_valid(dp_scnn_external_result_valid),
        .o_side_result_ready(dp_scnn_external_result_ready),
        .i_side_result_mask(dp_scnn_external_result_mask),
        .i_side_result_accumulator_flat(
            dp_scnn_external_result_accumulator_flat),
        .i_side_sticky_error(1'b0),
        .o_logit_wave_valid(output_logit_wave_valid),
        .i_logit_wave_ready(output_observation_enable ?
            tp_scnn_logit_ready_effective : 1'b1),
        .o_logit_wave_mask(o_tp_scnn_logit_mask),
        .o_logit_wave_base(o_tp_scnn_logit_base),
        .o_logit_wave_count(o_tp_scnn_logit_count),
        .o_logit_wave_q8_8_flat(o_tp_scnn_logit_q8_8_flat),
        .o_logit_wave_version(o_tp_scnn_logit_version),
        .o_logit_wave_frame_index(o_tp_scnn_logit_frame),
        .o_logit_wave_frame_last(o_tp_scnn_logit_frame_last),
        .o_logit_wave_sample_last(o_tp_scnn_logit_sample_last),
        .o_result_valid(output_result_valid),
        .i_result_ready(dp_scnn_li_result_ready),
        .o_result_class(output_result_class),
        .o_result_score_q8_8(output_result_score),
        .o_result_version(output_result_version),
        .o_result_frame_index(output_result_frame),
        .o_result_sample_last(output_result_sample_last),
        .o_locked_version(output_locked_version),
        .o_frames_seen(output_frames_seen),
        .o_busy(output_frontend_busy), .o_done(output_frontend_done),
        .o_config_error(output_config_error),
        .o_protocol_error(output_protocol_error),
        .o_sticky_error(output_frontend_error),
        .o_error_code(output_frontend_error_code)
    );

    assign dp_scnn_li_result_valid = output_result_valid;
    assign dp_scnn_li_result_class = {1'b0, output_result_class};
    assign dp_scnn_li_result_score = {{32{output_result_score[15]}},
                                      output_result_score};
    assign output_memory_owner = system_dispatch_busy &&
                                 o_active_layer_type == LAYER_OUTPUT;
    // Internal profile execution preserves the established classifier taps.
    // External graph clients explicitly select whether this OUTPUT command
    // contributes feature/logit observations to online learning.
    assign output_observation_enable = !layer_ingress_active_external ||
                                       dispatch_observation_enable;
    assign o_tp_scnn_feature_valid = output_feature_trace_valid &&
                                     output_observation_enable;
    assign o_tp_scnn_logit_valid = output_logit_wave_valid &&
                                   output_observation_enable;

    assign feature_host_enable = !control_sample_active &&
        !system_dispatch_busy && !control_sequence_busy;
    assign dp_scnn_feature_read_ready = conv_memory_read_ready && !tp_input_owner;
    assign dp_scnn_feature_valid = conv_memory_result_valid && !tp_input_owner;
    assign dp_scnn_feature_flat = conv_memory_result_data;
    assign dp_scnn_feature_lane_mask = conv_memory_result_mask;
    unified_scnn_feature_memory_router_v1 #(
        .LANES(LANES), .FEATURE_DEPTH(FEATURE_DEPTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH)
    ) u_feature_memory (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_output_owner(output_memory_owner),
        .i_conv_read_valid(tp_input_owner ? tp_input_read_valid : dp_scnn_feature_read_valid),
        .o_conv_read_ready(conv_memory_read_ready),
        .i_conv_read_slot(tp_input_owner ? tp_input_read_slot : dp_scnn_feature_read_slot),
        .i_conv_read_bank_flat(tp_input_owner ? tp_input_read_banks : dp_scnn_feature_read_bank_flat),
        .i_conv_read_addr_flat(tp_input_owner ? tp_input_read_addrs : dp_scnn_feature_read_addr_flat),
        .i_conv_read_lane_mask(tp_input_owner ? tp_input_read_mask : dp_scnn_feature_read_lane_mask),
        .o_conv_result_valid(conv_memory_result_valid),
        .i_conv_result_ready(tp_input_owner ? tp_input_response_ready : dp_scnn_feature_ready),
        .o_conv_result_flat(conv_memory_result_data),
        .o_conv_result_lane_mask(conv_memory_result_mask),
        .i_output_read_valid(output_feature_read_valid),
        .o_output_read_ready(output_feature_read_ready),
        .i_output_read_slot(output_feature_read_slot),
        .i_output_read_bank_flat(output_feature_read_bank_flat),
        .i_output_read_addr_flat(output_feature_read_addr_flat),
        .i_output_read_lane_mask(output_feature_read_lane_mask),
        .o_output_result_valid(output_feature_valid),
        .i_output_result_ready(output_feature_ready),
        .o_output_result_flat(output_feature_flat),
        .o_output_result_lane_mask(output_feature_lane_mask),
        .i_conv_write_valid(conv_store_write_valid),
        .o_conv_write_ready(conv_store_write_ready),
        .i_conv_write_slot(conv_feature_write_slot),
        .i_conv_write_bank(conv_feature_write_bank),
        .i_conv_write_addr(conv_feature_write_addr),
        .i_conv_write_data(conv_feature_write_data),
        .i_host_enable(feature_host_enable),
        .i_host_write_valid(i_feature_load_valid),
        .o_host_write_ready(o_feature_load_ready),
        .i_host_write_slot(i_feature_load_slot),
        .i_host_write_bank(i_feature_load_bank),
        .i_host_write_addr(i_feature_load_addr),
        .i_host_write_data(i_feature_load_data),
        .o_busy(feature_memory_busy),
        .o_sticky_error(feature_memory_error),
        .o_error_code(feature_memory_error_code)
    );

    assign wave_start_valid = output_memory_owner ?
                              output_wave3_start_valid : conv_start_valid;
    // External graph layers already carry their descriptor and provide the
    // paired Q64 weight service. They therefore start the shared datapath
    // directly instead of waiting for an internal DDR prefetch descriptor.
    assign external_physical_owner = layer_ingress_active_external ||
        external_physical_owner_q;
    assign wave_start_ready = layer_ingress_active_external ?
                              dp_scnn_start_ready :
                              coordinator_wave_start_ready;
    assign dp_scnn_start_valid = layer_ingress_active_external ?
                                 wave_start_valid :
                                 coordinator_compute_start_valid;
    assign coordinator_compute_start_ready =
        !layer_ingress_active_external && dp_scnn_start_ready;

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            external_physical_owner_q <= 1'b0;
        else if (dp_abort_flush)
            external_physical_owner_q <= 1'b0;
        else if (layer_ingress_active_external &&
                 wave_start_valid && wave_start_ready)
            external_physical_owner_q <= 1'b1;
        else if (external_physical_owner_q &&
                 (wave_output_mode_q ? dp_scnn_compute_done : wave_done))
            external_physical_owner_q <= 1'b0;
    end
    assign conv_start_ready = !output_memory_owner && wave_start_ready;
    assign output_wave3_start_ready = output_memory_owner && wave_start_ready;
    assign wave_layer_index = dispatch_layer_index;
    assign wave_output_base = output_memory_owner ? 10'd0 :
                              conv_output_channel_base;
    assign wave_output_count = output_memory_owner ?
        {4'd0, output_frontend_layer_words[3*32 + 23 +: 6]} :
        conv_output_channel_count;
    assign wave_input_base = output_memory_owner ? 10'd0 :
                             conv_input_channel_base;
    assign wave_input_count = output_memory_owner ? 10'd512 :
                              conv_input_channel_count;
    // The compute engine keeps the layer-wide Q16 base and generates absolute
    // addresses. The coordinator must describe the currently prefetched
    // wave/group slice, whose Q64 origin advances for every v2 descriptor.
    assign wave_source = ddr_compute_source;
    // Completion belongs to the owner captured when the DDR wave starts.
    // The live OUTPUT owner drops one cycle before Wave3 publishes its own
    // external-owner release pulse, so a live mux would report one OUTPUT
    // completion again through the CONV path.
    assign wave_done = wave_output_mode_q ? output_wave3_external_done :
                                            dp_scnn_compute_done;
    assign dp_scnn_external_mode = output_memory_owner &&
                                   output_wave3_external_mode;

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            wave_output_mode_q <= 1'b0;
        else if (i_abort || coordinator_abort_request)
            wave_output_mode_q <= 1'b0;
        else if (wave_start_valid && wave_start_ready)
            wave_output_mode_q <= output_memory_owner;
        // OUTPUT transaction retirement reaches this wrapper one cycle before
        // Wave3 publishes its physical SCNN-owner completion.  Retain the
        // mode through that trailing pulse so it cannot be mistaken for an
        // internal coordinator completion.  CONV already completes directly
        // on dp_scnn_compute_done through wave_done.
        else if (ddr_compute_tile_done ||
                 (external_physical_owner &&
                  (wave_output_mode_q ? dp_scnn_compute_done : wave_done)))
            wave_output_mode_q <= 1'b0;
    end

    // OUTPUT still retires its DDR wave descriptor, but in TP mode only CONV
    // consumes the prefetched image. Classifier codes come from the RMW store.
    assign external_conv_weight_owner = external_physical_owner &&
        current_layer_type == LAYER_CONV;
    assign external_output_weight_owner = external_physical_owner &&
        current_layer_type == LAYER_OUTPUT;
    assign external_weight_owner = external_conv_weight_owner ||
        external_output_weight_owner;
    assign o_external_scnn_weight_read_valid =
        external_conv_weight_owner ? dp_scnn_weight_read_valid :
        external_output_weight_owner ? output_weight_read_valid : 1'b0;
    assign o_external_scnn_weight_read_addr0 = external_conv_weight_owner ?
        dp_scnn_weight_read_addr0 : output_weight_read_addr0;
    assign o_external_scnn_weight_read_addr1 = external_conv_weight_owner ?
        dp_scnn_weight_read_addr1 : output_weight_read_addr1;
    assign o_external_scnn_weight_result_ready =
        external_conv_weight_owner ? dp_scnn_weight_result_ready :
        external_output_weight_owner ? output_weight_result_ready :
        (dp_abort_flush && external_weight_pending_q);
    assign external_weight_request_fire =
        o_external_scnn_weight_read_valid &&
        i_external_scnn_weight_read_ready;
    assign external_weight_response_fire =
        i_external_scnn_weight_result_valid &&
        o_external_scnn_weight_result_ready;

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            external_weight_pending_q <= 1'b0;
        else begin
            unique case ({external_weight_request_fire,
                          external_weight_response_fire})
                2'b10,
                2'b11: external_weight_pending_q <= 1'b1;
                2'b01: external_weight_pending_q <= 1'b0;
                default: external_weight_pending_q <=
                    external_weight_pending_q;
            endcase
        end
    end

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            external_weight_error_q <= 1'b0;
        else if (i_clear_error && o_global_idle)
            external_weight_error_q <= 1'b0;
        else if (i_external_scnn_weight_result_valid &&
                 !dp_abort_flush &&
                 (!external_weight_owner ||
                  i_external_scnn_weight_response_error))
            external_weight_error_q <= 1'b1;
    end

    assign absolute_weight_read_valid = output_memory_owner ?
        (output_weight_read_valid && !external_output_weight_owner &&
         !ENABLE_TP_SCNN_PARAMETER_READ) :
        (dp_scnn_weight_read_valid && !external_conv_weight_owner);
    assign absolute_weight_addr0 = output_memory_owner ?
        DDR_ADDRESS_WIDTH'(output_weight_read_addr0) :
        DDR_ADDRESS_WIDTH'(dp_scnn_weight_read_addr0);
    assign absolute_weight_addr1 = output_memory_owner ?
        DDR_ADDRESS_WIDTH'(output_weight_read_addr1) :
        DDR_ADDRESS_WIDTH'(dp_scnn_weight_read_addr1);
    assign output_weight_read_ready = output_memory_owner &&
        (external_output_weight_owner ? i_external_scnn_weight_read_ready :
         ENABLE_TP_SCNN_PARAMETER_READ ? tp_weight_request_ready :
                                          absolute_weight_read_ready);
    assign dp_scnn_weight_read_ready = !output_memory_owner &&
        (external_conv_weight_owner ?
            i_external_scnn_weight_read_ready :
            absolute_weight_read_ready);
    assign absolute_weight_response_ready = output_memory_owner ?
        (output_weight_result_ready && !external_output_weight_owner &&
         !ENABLE_TP_SCNN_PARAMETER_READ) :
        (external_conv_weight_owner ? 1'b0 : dp_scnn_weight_result_ready);
    assign output_weight_result_valid = output_memory_owner &&
        (external_output_weight_owner ? i_external_scnn_weight_result_valid :
         ENABLE_TP_SCNN_PARAMETER_READ ? tp_weight_response_valid :
                                          absolute_weight_response_valid);
    assign dp_scnn_weight_result_valid = !output_memory_owner &&
        (external_conv_weight_owner ?
            i_external_scnn_weight_result_valid :
            absolute_weight_response_valid);
    assign output_weight_data0 = ENABLE_TP_SCNN_PARAMETER_READ ?
        (external_output_weight_owner ? i_external_scnn_weight_data0 :
         tp_weight_data0) :
        (external_output_weight_owner ? i_external_scnn_weight_data0 :
         absolute_weight_data0);
    assign output_weight_data1 = ENABLE_TP_SCNN_PARAMETER_READ ?
        (external_output_weight_owner ? i_external_scnn_weight_data1 :
         tp_weight_data1) :
        (external_output_weight_owner ? i_external_scnn_weight_data1 :
         absolute_weight_data1);
    assign dp_scnn_weight_data0 = external_conv_weight_owner ?
        i_external_scnn_weight_data0 : absolute_weight_data0;
    assign dp_scnn_weight_data1 = external_conv_weight_owner ?
        i_external_scnn_weight_data1 : absolute_weight_data1;

    generate
        if (ENABLE_TP_SCNN_PARAMETER_READ) begin : g_tp_classifier_read
            logic [Q16_ADDR_WIDTH-1:0] parameter_base_q;
            logic [Q64_ADDR_WIDTH-1:0] image_base_q;
            logic [Q64_ADDR_WIDTH:0] word_count_q, offset0, offset1;
            logic [Q16_ADDR_WIDTH-1:0] relative_parameter_addr;
            logic bridge_ready, bridge_response_valid, bridge_error;
            logic address_error_q, cancelled_q, request_legal;
            logic [63:0] parameter_end;

            assign parameter_end = 64'(parameter_base_q) +
                64'(output_frontend_layer_words[3*32+23 +: 6]) *
                64'(SCNN_CLASSIFIER_FEATURES);
            assign offset0 = {1'b0, output_weight_read_addr0} -
                             {1'b0, image_base_q};
            assign offset1 = {1'b0, output_weight_read_addr1} -
                             {1'b0, image_base_q};
            assign request_legal = !offset0[Q64_ADDR_WIDTH] &&
                !offset1[Q64_ADDR_WIDTH] && offset0 < word_count_q &&
                offset1 < word_count_q;
            assign tp_weight_request_ready = bridge_ready && request_legal &&
                !address_error_q && !cancelled_q && !dp_abort_flush;
            assign tp_weight_response_valid = bridge_response_valid &&
                !cancelled_q && !dp_abort_flush;
            assign tp_weight_error = address_error_q || bridge_error;
            assign o_tp_scnn_forward_addr = parameter_base_q +
                                            relative_parameter_addr;

            always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
                if (!i_core_rst_n) begin
                    parameter_base_q <= '0;
                    image_base_q <= '0;
                    word_count_q <= '0;
                    address_error_q <= 1'b0;
                    cancelled_q <= 1'b0;
                end else begin
                    if (i_clear_error && o_global_idle)
                        address_error_q <= 1'b0;
                    if (qualified_start) begin
                        parameter_base_q <= i_tp_scnn_parameter_base;
                        cancelled_q <= 1'b0;
                    end
                    if (dp_abort_flush || system_fail_stop)
                        cancelled_q <= 1'b1;
                    if (output_frontend_layer_valid && output_frontend_layer_ready) begin
                        image_base_q <= Q64_ADDR_WIDTH'(
                            output_frontend_layer_words[7*32 +: 32] >> 2);
                        word_count_q <= (Q64_ADDR_WIDTH+1)'(
                            (64'(output_frontend_layer_words[3*32+23 +: 6]) *
                             64'(SCNN_CLASSIFIER_FEATURES)) >> 2);
                        if (output_frontend_layer_words[7*32 +: 2] != 0 ||
                            parameter_end > (64'd1 << Q16_ADDR_WIDTH))
                            address_error_q <= 1'b1;
                    end
                    if (output_memory_owner && !external_output_weight_owner &&
                        output_weight_read_valid &&
                        !request_legal && !dp_abort_flush)
                        address_error_q <= 1'b1;
                end
            end

            // Never reset a bridge with an accepted scalar read in flight.
            // On cancellation, finish that pair and discard its Q64 response.
            unified_scnn_classifier_q64_forward_bridge #(
                .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
                .PARAM_ADDR_WIDTH(Q16_ADDR_WIDTH)
            ) u_bridge (
                .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
                .i_clear_error(i_clear_error && o_global_idle),
                .i_request_valid(output_memory_owner &&
                    !external_output_weight_owner && output_weight_read_valid &&
                    request_legal && !address_error_q && !cancelled_q && !dp_abort_flush),
                .o_request_ready(bridge_ready),
                .i_request_addr0(offset0[Q64_ADDR_WIDTH-1:0]),
                .i_request_addr1(offset1[Q64_ADDR_WIDTH-1:0]),
                .o_response_valid(bridge_response_valid),
                .i_response_ready((output_memory_owner &&
                    !external_output_weight_owner && output_weight_result_ready) ||
                    cancelled_q || dp_abort_flush),
                .o_response_data0(tp_weight_data0), .o_response_data1(tp_weight_data1),
                .o_forward_read_valid(o_tp_scnn_forward_valid),
                .i_forward_read_ready(i_tp_scnn_forward_ready),
                .o_forward_read_addr(relative_parameter_addr),
                .o_forward_read_tag(o_tp_scnn_forward_tag),
                .i_forward_response_valid(i_tp_scnn_forward_response_valid),
                .o_forward_response_ready(o_tp_scnn_forward_response_ready),
                .i_forward_response_tag(i_tp_scnn_forward_response_tag),
                .i_forward_response_code(i_tp_scnn_forward_response_code),
                .i_forward_response_error(i_tp_scnn_forward_response_error),
                .o_busy(tp_weight_busy), .o_protocol_error(bridge_error)
            );
        end else begin : g_ddr_classifier_read
            assign tp_weight_request_ready = 1'b0;
            assign tp_weight_response_valid = 1'b0;
            assign tp_weight_data0 = '0;
            assign tp_weight_data1 = '0;
            assign tp_weight_busy = 1'b0;
            assign tp_weight_error = 1'b0;
            assign o_tp_scnn_forward_valid = 1'b0;
            assign o_tp_scnn_forward_addr = '0;
            assign o_tp_scnn_forward_tag = '0;
            assign o_tp_scnn_forward_response_ready = 1'b0;
        end
    endgenerate

    unified_scnn_ddr_tile_coordinator_v1 #(
        .ADDRESS_WIDTH(DDR_ADDRESS_WIDTH), .BANK_WORDS(DDR_BANK_WORDS),
        .MAX_SEGMENTS(ENGINES)
    ) u_ddr_tile_coordinator (
        .i_clk(i_core_clk), .i_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error), .i_abort(i_abort),
        .i_service_compute_tile_valid(ddr_compute_tile_valid),
        .o_service_compute_tile_done(ddr_compute_tile_done),
        .i_service_layer_index(ddr_compute_layer_index),
        .i_service_output_channel_base(ddr_compute_output_base),
        .i_service_output_channel_count(ddr_compute_output_count),
        .i_service_input_channel_base(ddr_compute_input_base),
        .i_service_input_channel_count(ddr_compute_input_count),
        .i_service_source_q64_word(ddr_compute_source),
        .i_service_total_q64_words(ddr_compute_total),
        .i_service_segment_q64_words(ddr_compute_segment),
        .i_service_source_stride_q64_words(ddr_compute_stride),
        .i_service_tile_tag(ddr_compute_tag),
        .i_wave_start_valid(wave_start_valid &&
            !external_physical_owner),
        .o_wave_start_ready(coordinator_wave_start_ready),
        .i_wave_layer_index(wave_layer_index),
        .i_wave_output_channel_base(wave_output_base),
        .i_wave_output_channel_count(wave_output_count),
        .i_wave_input_channel_base(wave_input_base),
        .i_wave_input_channel_count(wave_input_count),
        .i_wave_source_q64_word(wave_source),
        .i_wave_total_q64_words(ddr_compute_total),
        .i_wave_segment_q64_words(ddr_compute_segment),
        .i_wave_source_stride_q64_words(ddr_compute_stride),
        .i_wave_tile_tag(ddr_compute_tag),
        .o_compute_start_valid(coordinator_compute_start_valid),
        .i_compute_start_ready(coordinator_compute_start_ready),
        .i_wave_done(wave_done && !external_physical_owner),
        .i_wave_done_tag(ddr_compute_tag),
        .i_absolute_read_valid(absolute_weight_read_valid),
        .o_absolute_read_ready(absolute_weight_read_ready),
        .i_absolute_q64_addr0(absolute_weight_addr0),
        .i_absolute_q64_addr1(absolute_weight_addr1),
        .o_absolute_response_valid(absolute_weight_response_valid),
        .i_absolute_response_ready(absolute_weight_response_ready),
        .o_absolute_data0(absolute_weight_data0),
        .o_absolute_data1(absolute_weight_data1),
        .o_service_read_valid(ddr_local_read_valid),
        .i_service_read_ready(ddr_local_read_ready),
        .o_service_read_word_addr0(ddr_local_read_addr0),
        .o_service_read_word_addr1(ddr_local_read_addr1),
        .i_service_response_valid(ddr_local_response_valid),
        .o_service_response_ready(ddr_local_response_ready),
        .i_service_read_data0(ddr_local_data0),
        .i_service_read_data1(ddr_local_data1),
        .i_service_sticky_error(ddr_error),
        .i_service_error_code(ddr_error_code),
        .o_busy(coordinator_busy), .o_idle(coordinator_idle),
        .o_descriptor_active(coordinator_descriptor_active),
        .o_wave_active(coordinator_wave_active),
        .o_metadata_match(coordinator_metadata_match),
        .o_read_pending(coordinator_read_pending),
        .o_response_pending(coordinator_response_pending),
        .o_abort_request(coordinator_abort_request),
        .o_sticky_error(coordinator_error),
        .o_error_code(coordinator_error_code)
    );

    unified_scnn_ddr3a_core_service_v1 #(
        .ADDRESS_WIDTH(DDR_ADDRESS_WIDTH), .BANK_WORDS(DDR_BANK_WORDS),
        .DESCRIPTOR_LIMIT(DDR_DESCRIPTOR_LIMIT),
        .WAVE_GROUP_DESCRIPTORS(SCNN_WAVE_GROUP_DESCRIPTORS),
        .DDR_WORD_LIMIT(DDR_ADDRESS_WIDTH'(2309920))
    ) u_ddr3a_service (
        .i_core_clk(i_core_clk), .i_core_rst_n(i_core_rst_n),
        .i_mig_clk(i_mig_ui_clk), .i_mig_rst_n(i_mig_ui_rst_n),
        .i_calib_complete(i_ddr_calib_complete),
        .i_clear_error(i_clear_error),
        // Keep one external abort transaction level-contiguous while the
        // physical drain FSM is active. Otherwise the delayed coordinator
        // sticky error can create a second rising edge after i_abort drops.
        .i_soft_abort(i_abort || system_abort_cleanup_pending ||
                      coordinator_abort_request),
        .i_prefetch_start(ddr_prefetch_start),
        .o_prefetch_start_ready(o_prefetch_start_ready),
        .o_busy(ddr_busy), .o_idle(ddr_idle), .o_done(ddr_done),
        .i_core_req_valid(1'b0), .o_core_req_ready(),
        .i_core_req_word_addr('0), .i_core_req_words('0),
        .o_core_rsp_valid(), .i_core_rsp_ready(1'b1),
        .o_core_rsp_data(), .o_core_rsp_last(), .o_core_rsp_error(),
        .o_compute_tile_valid(ddr_compute_tile_valid),
        .i_compute_tile_done(ddr_compute_tile_done),
        .o_compute_layer_index(ddr_compute_layer_index),
        .o_compute_output_channel_base(ddr_compute_output_base),
        .o_compute_output_channel_count(ddr_compute_output_count),
        .o_compute_input_channel_base(ddr_compute_input_base),
        .o_compute_input_channel_count(ddr_compute_input_count),
        .o_compute_source_q64_word(ddr_compute_source),
        .o_compute_total_q64_words(ddr_compute_total),
        .o_compute_segment_q64_words(ddr_compute_segment),
        .o_compute_source_stride_q64_words(ddr_compute_stride),
        .o_compute_tile_tag(ddr_compute_tag),
        .i_read_valid(ddr_local_read_valid),
        .o_read_ready(ddr_local_read_ready),
        .i_read_word_addr0(ddr_local_read_addr0),
        .i_read_word_addr1(ddr_local_read_addr1),
        .o_read_valid(ddr_local_response_valid),
        .i_read_result_ready(ddr_local_response_ready),
        .o_read_data0(ddr_local_data0), .o_read_data1(ddr_local_data1),
        .o_cmd_pending(ddr_cmd_pending),
        .o_response_pending(ddr_response_pending),
        .o_cdc_busy(ddr_cdc_busy), .o_outstanding(ddr_outstanding),
        .o_axi_outstanding(ddr_axi_outstanding),
        .o_axi_busy_core(ddr_axi_busy_core),
        .o_prefetch_busy(ddr_prefetch_busy),
        .o_scratchpad_busy(ddr_scratchpad_busy),
        .o_prefetch_compute_overlap_seen(ddr_overlap_seen),
        .o_sticky_error(ddr_error), .o_error_code(ddr_error_code),
        .o_m_axi_arvalid(o_m_axi_arvalid),
        .i_m_axi_arready(i_m_axi_arready), .o_m_axi_araddr(o_m_axi_araddr),
        .o_m_axi_arlen(o_m_axi_arlen), .o_m_axi_arsize(o_m_axi_arsize),
        .o_m_axi_arburst(o_m_axi_arburst), .o_m_axi_arid(o_m_axi_arid),
        .o_m_axi_arcache(o_m_axi_arcache),
        .o_m_axi_arprot(o_m_axi_arprot), .o_m_axi_arqos(o_m_axi_arqos),
        .i_m_axi_rvalid(i_m_axi_rvalid), .o_m_axi_rready(o_m_axi_rready),
        .i_m_axi_rdata(i_m_axi_rdata), .i_m_axi_rlast(i_m_axi_rlast),
        .i_m_axi_rresp(i_m_axi_rresp), .i_m_axi_rid(i_m_axi_rid)
    );

    assign scnn_layer_error_combined = conv_layer_error || pool_error || tp_input_error;
    assign scnn_layer_error_combined_code = conv_layer_error ?
        conv_layer_error_code : pool_error ? pool_error_code : 8'h94;
    assign scnn_output_error_combined = output_adapter_error ||
        output_frontend_error || output_version_error_q;
    assign scnn_output_error_combined_code = output_adapter_error ?
        output_adapter_error_code : output_frontend_error ?
        output_frontend_error_code : 8'h93;
    assign memory_error_combined = feature_memory_error || ddr_error ||
                                   coordinator_error || tp_weight_error ||
                                   external_weight_error_q;
    assign memory_error_combined_code = feature_memory_error ?
        feature_memory_error_code : ddr_error ? {4'hd, ddr_error_code} :
        coordinator_error ? {4'he, coordinator_error_code} :
        tp_weight_error ? 8'hf1 : 8'hf2;
    assign scratchpad_blocker = feature_memory_busy || ddr_scratchpad_busy ||
                                coordinator_busy || tp_weight_busy || tp_input_busy;
    assign ddr_memory_busy = ddr_prefetch_busy || ddr_busy;
    assign ddr_physical_outstanding = ddr_outstanding |
                                      {2'b00, ddr_axi_busy_core};
    assign dp_fc_path_busy = dp_fc_busy || dp_ingress_busy;
    assign dp_abort_flush = i_abort || system_abort_cleanup_pending;

    assign extension_command_pending = srnn_engine_start_valid ||
        conv_start_valid || output_wave3_start_valid ||
        coordinator_read_pending || coordinator_response_pending ||
        external_weight_pending_q;
    assign writeback_busy = conv_feature_write_valid || i_feature_load_valid;

    // An aborted producer may retire its internal busy state one cycle before
    // its registered result-valid reaches the shared arbiter. Keep results in
    // ready/drop quarantine until the next accepted sample establishes a new
    // version/owner epoch.
    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n)
            abort_result_quarantine_q <= 1'b0;
        else if (i_abort)
            abort_result_quarantine_q <= 1'b1;
        else if (qualified_start)
            abort_result_quarantine_q <= 1'b0;
    end

    logic unused_status;
    assign unused_status = &{1'b0, control_inbound_valid,
        control_inbound_sources_flat, control_inbound_types_flat,
        live_versions, control_profile_legal, control_sample_compute_done,
        control_commit_pending, control_commit_pulse,
        control_config_copy_busy, control_blocker_mask,
        control_quiescent_for_commit, control_idle,
        control_version_consistent, versioned_accept_unused,
        versioned_stale_unused, versioned_mixed_unused,
        system_layer_accept, system_owner_active,
        system_owner_tracking_consistent, system_result_pending,
        system_physical_blocker_mask, system_dispatch_error,
        system_dispatch_error_code, srnn_adapter_busy, srnn_version,
        dp_fc_compute_done, dp_srnn_compute_done, dp_fc_logits_flat,
        dp_fc_spikes_flat, dp_fc_vmem_flat, dp_fc_trace_flat,
        dp_srnn_logits_flat, dp_srnn_previous_spikes, dp_extension_owner,
        dp_result_pending, dp_ingress_busy, dp_busy, dp_idle,
        conv_layer_idle, conv_locked_version, output_wave3_clear_error,
        output_frontend_enqueued, output_result_enqueued,
        output_owner_held, output_adapter_idle, output_result_frame,
        output_result_sample_last, output_locked_version, output_frames_seen,
        output_config_error, output_protocol_error, ddr_busy, ddr_idle,
        ddr_axi_outstanding, ddr_overlap_seen, coordinator_idle,
        coordinator_descriptor_active, coordinator_wave_active,
        coordinator_metadata_match};
endmodule

`default_nettype wire
