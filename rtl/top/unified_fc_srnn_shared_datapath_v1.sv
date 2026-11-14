`timescale 1ns/1ps
`default_nettype none

// FC/SRNN/SCNN shared datapath composition. The FC owner can select the
// paper-oriented FC_COMPAT tile path or the retained canonical reference;
// SRNN and SCNN continue to share the Wave3 numeric array.
module unified_fc_srnn_shared_datapath_v1 #(
    parameter FC_WEIGHT_DIR =
        "../../../data/weights/r19b_h480_complete_fc_input2312",
    parameter FC_COMPAT_WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312",
    parameter bit USE_LEGACY_CANONICAL_FC = 1'b0,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0,
    parameter int FC_PHYSICAL_TILES = 5,
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int OUTPUT_MAX = 32,
    parameter int VERSION_WIDTH = 16,
    parameter int REQUEST_ID_WIDTH = 16,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int FEATURE_DEPTH = 9216,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int ACC_WIDTH = 48,
    parameter bit ENABLE_FC_TP_NATIVE_WEIGHTS = 1'b0,
    parameter bit FC_LEARN_W0 = 1'b1,
    parameter int FC_TP_TENSOR_ID_WIDTH = 8,
    parameter int FC_TP_PARAM_ADDR_WIDTH = 24,
    parameter int FC_TP_TAG_WIDTH = 16,
    parameter int FC_TP_CODE_WIDTH = 16,
    parameter int FC_TP_RESIDUAL_WIDTH = 16,
    parameter FC_TP_W0_RESIDUAL_INIT_FILE = ""
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,
    input  logic [1:0]                                i_requested_owner,
    input  logic [63:0]                               i_locked_versions,

    input  logic                                      i_fc_layer_valid,
    output logic                                      o_fc_layer_ready,
    input  logic [383:0]                              i_fc_layer_words,
    input  logic                                      i_fc_clear_state,
    input  logic [15:0]                               i_fc_sample_id,

    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_w0_parameter_base,
    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_w2_parameter_base,
    input  logic                                      i_fc_tp_learning_read_valid,
    output logic                                      o_fc_tp_learning_read_ready,
    input  logic [FC_TP_TENSOR_ID_WIDTH-1:0]          i_fc_tp_learning_read_tensor_id,
    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_learning_read_parameter_addr,
    input  logic [FC_TP_TAG_WIDTH-1:0]                i_fc_tp_learning_read_tag,
    output logic                                      o_fc_tp_learning_response_valid,
    input  logic                                      i_fc_tp_learning_response_ready,
    output logic [FC_TP_TAG_WIDTH-1:0]                o_fc_tp_learning_response_tag,
    output logic signed [FC_TP_CODE_WIDTH-1:0]        o_fc_tp_learning_response_code,
    output logic signed [FC_TP_RESIDUAL_WIDTH-1:0]    o_fc_tp_learning_response_residual,
    output logic                                      o_fc_tp_learning_response_error,
    input  logic                                      i_fc_tp_learning_write_valid,
    output logic                                      o_fc_tp_learning_write_ready,
    input  logic [FC_TP_TENSOR_ID_WIDTH-1:0]          i_fc_tp_learning_write_tensor_id,
    input  logic [FC_TP_PARAM_ADDR_WIDTH-1:0]         i_fc_tp_learning_write_parameter_addr,
    input  logic signed [FC_TP_CODE_WIDTH-1:0]        i_fc_tp_learning_write_code,
    input  logic signed [FC_TP_RESIDUAL_WIDTH-1:0]    i_fc_tp_learning_write_residual,
    input  logic                                      i_fc_tp_learning_write_code_enable,
    input  logic                                      i_fc_tp_learning_write_residual_enable,
    output logic                                      o_fc_tp_learning_quiescent,

    input  logic                                      i_srnn_start_valid,
    output logic                                      o_srnn_start_ready,
    input  logic                                      i_srnn_state_reset,
    input  logic [12:0]                               i_srnn_input_dim,
    input  logic [9:0]                                i_srnn_active_neurons,
    input  logic [3:0]                                i_srnn_active_tiles,
    input  logic [31:0]                               i_srnn_w0_base,
    input  logic [31:0]                               i_srnn_recurrent_base,
    input  logic                                      i_srnn_recurrent_enable,
    input  logic signed [15:0]                        i_srnn_leak_q15,
    input  logic signed [31:0]                        i_srnn_threshold,
    input  logic [4:0]                                i_srnn_current_left_shift,
    input  logic                                      i_srnn_reset_to_zero,

    input  logic                                      i_scnn_start_valid,
    output logic                                      o_scnn_start_ready,
    input  logic [15:0]                               i_scnn_in_width,
    input  logic [15:0]                               i_scnn_in_height,
    input  logic [15:0]                               i_scnn_total_in_channels,
    input  logic [15:0]                               i_scnn_out_width,
    input  logic [15:0]                               i_scnn_out_height,
    input  logic [15:0]                               i_scnn_total_out_channels,
    input  logic [9:0]                                i_scnn_input_channel_base,
    input  logic [9:0]                                i_scnn_input_channel_count,
    input  logic [9:0]                                i_scnn_output_channel_base,
    input  logic [9:0]                                i_scnn_output_channel_count,
    input  logic [7:0]                                i_scnn_stride,
    input  logic [7:0]                                i_scnn_padding,
    input  logic                                      i_scnn_input_slot,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_scnn_feature_base,
    input  logic [Q16_ADDR_WIDTH-1:0]                 i_scnn_weight_q16_base,
    input  logic [OUTPUT_ADDR_WIDTH-1:0]              i_scnn_output_base,
    input  logic                                      i_scnn_unit_spike_features,
    input  logic                                      i_scnn_external_mode,

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
    output logic [VERSION_WIDTH-1:0]                  o_srnn_weight_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_srnn_weight_req_id,
    output logic                                      o_srnn_weight_req_recurrent,
    output logic [31:0]                               o_srnn_weight_req_row_base,
    output logic [9:0]                                o_srnn_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_srnn_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_srnn_weight_req_lane_mask,
    input  logic                                      i_srnn_weight_rsp_valid,
    output logic                                      o_srnn_weight_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_srnn_weight_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_srnn_weight_rsp_id,
    input  logic [9:0]                                i_srnn_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_srnn_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_srnn_weight_rsp_flat,

    output logic [ENGINES-1:0]                        o_srnn_w2_req_valid,
    input  logic [ENGINES-1:0]                        i_srnn_w2_req_ready,
    output logic [ENGINES*32-1:0]                     o_srnn_w2_addr_flat,
    input  logic [ENGINES*64-1:0]                     i_srnn_w2_data_flat,
    input  logic [31:0]                               i_srnn_w2_base,
    input  logic [5:0]                                i_srnn_w2_row_stride,
    input  logic [5:0]                                i_srnn_output_count,

    output logic                                      o_scnn_feature_read_valid,
    input  logic                                      i_scnn_feature_read_ready,
    output logic                                      o_scnn_feature_read_slot,
    output logic [31:0]                               o_scnn_feature_read_bank_flat,
    output logic [LANES*FEATURE_ADDR_WIDTH-1:0]        o_scnn_feature_read_addr_flat,
    output logic [LANES-1:0]                          o_scnn_feature_read_lane_mask,
    input  logic                                      i_scnn_feature_valid,
    output logic                                      o_scnn_feature_ready,
    input  logic [LANES*16-1:0]                       i_scnn_feature_flat,
    input  logic [LANES-1:0]                          i_scnn_feature_lane_mask,
    output logic                                      o_scnn_weight_read_valid,
    input  logic                                      i_scnn_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_scnn_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]                 o_scnn_weight_read_addr1,
    input  logic                                      i_scnn_weight_result_valid,
    output logic                                      o_scnn_weight_result_ready,
    input  logic [63:0]                               i_scnn_weight_data0,
    input  logic [63:0]                               i_scnn_weight_data1,
    output logic                                      o_scnn_raw_result_valid,
    input  logic                                      i_scnn_raw_result_ready,
    output logic [ENGINES-1:0]                        o_scnn_raw_result_mask,
    output logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      o_scnn_raw_result_output_addr_flat,
    output logic [OUTPUT_ADDR_WIDTH-1:0]              o_scnn_raw_result_spatial_addr,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_scnn_raw_result_accumulator_flat,

    input  logic                                      i_scnn_external_request_valid,
    output logic                                      o_scnn_external_request_ready,
    input  logic [ENGINES-1:0]                        i_scnn_external_engine_mask,
    input  logic [LANES*16-1:0]                       i_scnn_external_feature_flat,
    input  logic [ENGINES*LANES*16-1:0]               i_scnn_external_weight_flat,
    input  logic [LANES-1:0]                          i_scnn_external_lane_mask,
    input  logic                                      i_scnn_external_first,
    input  logic                                      i_scnn_external_last,
    input  logic [9:0]                                i_scnn_external_output_channel_base,
    output logic                                      o_scnn_external_result_valid,
    input  logic                                      i_scnn_external_result_ready,
    output logic [ENGINES-1:0]                        o_scnn_external_result_mask,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_scnn_external_result_accumulator_flat,
    input  logic                                      i_scnn_external_done,
    input  logic                                      i_scnn_external_sticky_error,

    input  logic                                      i_scnn_li_result_valid,
    output logic                                      o_scnn_li_result_ready,
    input  logic [5:0]                                i_scnn_li_result_class,
    input  logic signed [47:0]                        i_scnn_li_result_score,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [5:0]                                o_result_class,
    output logic signed [47:0]                        o_result_score,
    output logic [63:0]                               o_result_versions,
    output logic [1:0]                                o_result_owner,

    output logic                                      o_fc_compute_done,
    output logic                                      o_srnn_compute_done,
    output logic                                      o_scnn_compute_done,
    output logic [10*24-1:0]                         o_fc_logits_flat,
    output logic [ENGINES*40-1:0]                    o_fc_spikes_flat,
    output logic [ENGINES*40*18-1:0]                 o_fc_vmem_flat,
    output logic [ENGINES*40*16-1:0]                 o_fc_trace_flat,
    output logic [OUTPUT_MAX*32-1:0]                 o_srnn_logits_flat,
    output logic [31:0]                               o_fc_sample_cycles,
    output logic [11:0]                               o_fc_locked_cluster_mask,
    output logic [HIDDEN_MAX-1:0]                    o_srnn_previous_spikes,
    output logic                                      o_tp_srnn_timestep_commit,
    output logic [VERSION_WIDTH-1:0]                  o_tp_srnn_timestep_version,
    output logic                                      o_tp_srnn_timestep_sample_last,
    output logic [HIDDEN_MAX-1:0]                    o_tp_srnn_timestep_spikes,
    output logic [1:0]                                o_extension_owner,
    output logic                                      o_result_pending,
    output logic                                      o_fc_busy,
    output logic                                      o_extension_busy,
    output logic                                      o_dynamic_output_busy,
    output logic                                      o_ingress_busy,
    output logic                                      o_fc_w2_commit_pending,
    output logic                                      o_fc_w2_snapshot_busy,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_SRNN = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER_COMMAND = 8'h01;
    localparam logic [7:0] ERROR_FC = 8'h20;
    localparam logic [7:0] ERROR_INGRESS = 8'h30;
    localparam logic [7:0] ERROR_EXTENSION = 8'h40;
    localparam logic [7:0] ERROR_OUTPUT = 8'h50;
    localparam logic [7:0] ERROR_RESULT = 8'h60;

    logic fc_window_valid;
    logic fc_window_ready;
    logic [7:0] fc_record_valid;
    logic [15:0] fc_window_sample_id;
    logic [8:0] fc_window_timestep;
    logic [7:0] fc_boundary_mask;
    logic fc_window_sample_last;
    logic [8*12-1:0] fc_addr_flat;
    logic [63:0] fc_data_flat;
    logic srnn_event_valid;
    logic srnn_event_ready;
    logic [12:0] srnn_event_addr;
    logic signed [7:0] srnn_event_data;
    logic srnn_timestep_last;
    logic srnn_sample_last;
    logic ingress_busy;
    logic ingress_idle;
    logic ingress_error;
    logic [7:0] ingress_error_code;

    logic fc_result_valid;
    logic fc_result_ready;
    logic [5:0] fc_result_class;
    logic signed [31:0] fc_result_score;
    logic fc_busy;
    logic fc_idle;
    logic fc_error;
    logic [7:0] fc_error_code;
    logic fc_shared_sample_start;
    logic fc_shared_sample_ready;
    logic fc_shared_sample_clear_state;
    logic [12:0] fc_shared_active_input_dim;
    logic [ENGINES-1:0] fc_shared_requested_tile_mask;
    logic fc_shared_window_valid;
    logic fc_shared_window_ready;
    logic [7:0] fc_shared_record_valid;
    logic [15:0] fc_shared_sample_id;
    logic [8:0] fc_shared_window_timestep;
    logic [7:0] fc_shared_boundary_mask;
    logic fc_shared_sample_last;
    logic [8*12-1:0] fc_shared_fi_flat;
    logic [63:0] fc_shared_x_flat;
    logic fc_shared_timestep_done;
    logic fc_shared_sample_done;
    logic [10*24-1:0] fc_shared_logits_flat;
    logic [3:0] fc_shared_pred;
    logic [63:0] fc_shared_sample_cycles;
    logic [31:0] fc_shared_accepted_windows;
    logic [31:0] fc_shared_accepted_events;
    logic [31:0] fc_shared_issue_groups;
    logic [31:0] fc_shared_boundary_count;
    logic [31:0] fc_shared_dropped_events;
    logic fc_shared_protocol_error;
    logic fc_shared_abort;

    logic srnn_timestep_valid;
    logic srnn_timestep_ready;
    logic [VERSION_WIDTH-1:0] srnn_timestep_version;
    logic srnn_timestep_sample_last;
    logic [HIDDEN_MAX-1:0] srnn_timestep_spikes;
    logic srnn_leaf_done;
    logic extension_busy;
    logic extension_idle;
    logic extension_error;
    logic [7:0] extension_error_code;
    logic [1:0] shared_active_owner;
    logic shared_busy;
    logic shared_idle;
    logic shared_error;
    logic [7:0] shared_error_code;

    logic dynamic_result_valid;
    logic dynamic_result_ready;
    logic [5:0] dynamic_result_class;
    logic signed [31:0] dynamic_result_score;
    logic dynamic_busy;
    logic dynamic_error;
    logic [2:0] result_pending_mask;
    logic result_error;
    logic [7:0] result_error_code;
    logic local_error_q;

    logic fc_layer_valid_gated;
    logic srnn_start_valid_gated;
    logic scnn_start_valid_gated;
    logic owner_command_error;
    logic srnn_shape_valid;
    logic fc_w2_commit_pending;
    logic fc_w2_snapshot_busy;

    function automatic logic state_dim_fits_h40_tiles(
        input logic [9:0] state_dim,
        input logic [3:0] tile_count
    );
        integer capacity;
        begin
            state_dim_fits_h40_tiles = 1'b0;
            if (tile_count != 0) begin
                capacity = integer'(tile_count) * 40;
                state_dim_fits_h40_tiles =
                    integer'(state_dim) > capacity - 40 &&
                    integer'(state_dim) <= capacity;
            end
        end
    endfunction

    assign fc_layer_valid_gated = i_fc_layer_valid &&
                                  i_requested_owner == OWNER_FC;
    assign srnn_shape_valid = i_srnn_active_tiles != 0 &&
        i_srnn_active_tiles <= 4'd12 &&
        state_dim_fits_h40_tiles(
            i_srnn_active_neurons, i_srnn_active_tiles);
    assign srnn_start_valid_gated = i_srnn_start_valid &&
                                    i_requested_owner == OWNER_SRNN &&
                                    srnn_shape_valid;
    assign scnn_start_valid_gated = i_scnn_start_valid &&
                                    i_requested_owner == OWNER_SCNN;
    assign o_fc_layer_ready = i_requested_owner == OWNER_FC &&
                              fc_layer_ready_internal;
    assign o_srnn_start_ready = i_requested_owner == OWNER_SRNN &&
                                srnn_start_ready_internal;
    assign o_scnn_start_ready = i_requested_owner == OWNER_SCNN &&
                                scnn_start_ready_internal;
    assign owner_command_error =
        (i_fc_layer_valid && i_requested_owner != OWNER_FC) ||
        (i_srnn_start_valid && i_requested_owner != OWNER_SRNN) ||
        (i_scnn_start_valid && i_requested_owner != OWNER_SCNN) ||
        (i_srnn_start_valid && i_requested_owner == OWNER_SRNN &&
         !srnn_shape_valid);
    assign o_result_pending = |result_pending_mask;
    assign o_fc_busy = fc_busy;
    assign o_extension_busy = extension_busy;
    assign o_dynamic_output_busy = dynamic_busy;
    assign o_ingress_busy = ingress_busy;
    assign o_fc_w2_commit_pending = fc_w2_commit_pending;
    assign o_fc_w2_snapshot_busy = fc_w2_snapshot_busy;
    assign o_busy = fc_busy || extension_busy || dynamic_busy ||
                    ingress_busy || o_result_pending;
    assign o_idle = !o_busy;
    assign o_sticky_error = local_error_q || fc_error || ingress_error ||
                            extension_error || dynamic_error || result_error;
    assign o_error_code = local_error_q ? ERROR_OWNER_COMMAND :
        fc_error ? (ERROR_FC | fc_error_code) :
        ingress_error ? (ERROR_INGRESS | ingress_error_code) :
        extension_error ? (ERROR_EXTENSION | extension_error_code) :
        dynamic_error ? ERROR_OUTPUT :
        result_error ? (ERROR_RESULT | result_error_code) : ERROR_NONE;

    logic fc_layer_ready_internal;
    logic srnn_start_ready_internal;
    logic scnn_start_ready_internal;

    unified_fc_first_event_ingress_adapter_v1 u_ingress (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(i_abort || fc_shared_abort),
        .i_requested_owner(i_requested_owner), .o_active_owner(),
        .i_window_valid(i_event_window_valid),
        .o_window_ready(o_event_window_ready),
        .i_record_valid(i_event_record_valid),
        .i_window_sample_id(i_event_window_sample_id),
        .i_window_timestep(i_event_window_timestep),
        .i_boundary_mask(i_event_boundary_mask),
        .i_window_sample_last(i_event_window_sample_last),
        .i_fi_flat(i_event_addr_flat), .i_x_flat(i_event_data_flat),
        .o_fc_window_valid(fc_window_valid),
        .i_fc_window_ready(fc_window_ready),
        .o_fc_record_valid(fc_record_valid),
        .o_fc_window_sample_id(fc_window_sample_id),
        .o_fc_window_timestep(fc_window_timestep),
        .o_fc_boundary_mask(fc_boundary_mask),
        .o_fc_window_sample_last(fc_window_sample_last),
        .o_fc_fi_flat(fc_addr_flat), .o_fc_x_flat(fc_data_flat),
        .o_srnn_event_valid(srnn_event_valid),
        .i_srnn_event_ready(srnn_event_ready),
        .o_srnn_event_addr(srnn_event_addr),
        .o_srnn_event_data(srnn_event_data),
        .o_srnn_timestep_last(srnn_timestep_last),
        .o_srnn_sample_last(srnn_sample_last),
        .o_srnn_boundary_token(), .o_srnn_window_sample_id(),
        .o_srnn_window_timestep(), .o_busy(ingress_busy),
        .o_idle(ingress_idle), .o_sticky_error(ingress_error),
        .o_error_code(ingress_error_code)
    );

    unified_fc_owner_selector_v1 #(
        .USE_LEGACY_CANONICAL_FC(USE_LEGACY_CANONICAL_FC),
        .LEGACY_WEIGHT_DIR(FC_WEIGHT_DIR),
        .FC_COMPAT_WEIGHT_DIR(FC_COMPAT_WEIGHT_DIR),
        .FC_PHYSICAL_TILES(FC_PHYSICAL_TILES),
        .ENGINES(ENGINES),
        .INPUT_MAX(INPUT_MAX)
    ) u_fc (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(i_abort),
        .i_layer_valid(fc_layer_valid_gated),
        .o_layer_ready(fc_layer_ready_internal),
        .i_layer_words(i_fc_layer_words),
        .i_clear_state(i_fc_clear_state),
        .i_start_sample_id(i_fc_sample_id),
        .i_window_valid(fc_window_valid), .o_window_ready(fc_window_ready),
        .i_record_valid(fc_record_valid),
        .i_window_sample_id(fc_window_sample_id),
        .i_window_timestep(fc_window_timestep),
        .i_boundary_mask(fc_boundary_mask),
        .i_window_sample_last(fc_window_sample_last),
        .i_fi_flat(fc_addr_flat), .i_x_flat(fc_data_flat),
        .o_timestep_done(), .o_layer_compute_done(o_fc_compute_done),
        .o_result_valid(fc_result_valid), .i_result_ready(fc_result_ready),
        .o_result_class(fc_result_class), .o_result_score(fc_result_score),
        .o_logits_flat(o_fc_logits_flat),
        .o_sample_cycles(o_fc_sample_cycles), .o_accepted_windows(),
        .o_accepted_events(), .o_issue_groups(), .o_boundary_count(),
        .o_runtime_dropped_events(),
        .o_locked_cluster_mask(o_fc_locked_cluster_mask),
        .o_locked_input_dim(), .o_active_bank(), .o_active_version(),
        .o_update_window_ready(),
        .o_commit_pending(fc_w2_commit_pending),
        .o_snapshot_busy(fc_w2_snapshot_busy),
        .i_debug_cluster('0), .i_debug_word_addr('0),
        .o_debug_active_word(), .o_busy(fc_busy), .o_idle(fc_idle),
        .o_sticky_error(fc_error), .o_error_code(fc_error_code),
        .o_shared_sample_start(fc_shared_sample_start),
        .i_shared_sample_ready(fc_shared_sample_ready),
        .o_shared_sample_clear_state(fc_shared_sample_clear_state),
        .o_shared_active_input_dim(fc_shared_active_input_dim),
        .o_shared_requested_tile_mask(fc_shared_requested_tile_mask),
        .o_shared_window_valid(fc_shared_window_valid),
        .i_shared_window_ready(fc_shared_window_ready),
        .o_shared_record_valid(fc_shared_record_valid),
        .o_shared_sample_id(fc_shared_sample_id),
        .o_shared_window_timestep(fc_shared_window_timestep),
        .o_shared_boundary_mask(fc_shared_boundary_mask),
        .o_shared_sample_last(fc_shared_sample_last),
        .o_shared_fi_flat(fc_shared_fi_flat),
        .o_shared_x_flat(fc_shared_x_flat),
        .i_shared_timestep_done(fc_shared_timestep_done),
        .i_shared_sample_done(fc_shared_sample_done),
        .i_shared_logits_flat(fc_shared_logits_flat),
        .i_shared_pred(fc_shared_pred),
        .i_shared_sample_cycles(fc_shared_sample_cycles),
        .i_shared_accepted_windows(fc_shared_accepted_windows),
        .i_shared_accepted_events(fc_shared_accepted_events),
        .i_shared_issue_groups(fc_shared_issue_groups),
        .i_shared_boundary_count(fc_shared_boundary_count),
        .i_shared_dropped_events(fc_shared_dropped_events),
        .i_shared_protocol_error(fc_shared_protocol_error),
        .i_shared_fc_active(shared_active_owner == OWNER_FC),
        .o_shared_abort(fc_shared_abort)
    );

    unified_shared_h40_inference_top_v1 #(
        .ENGINES(ENGINES), .FC_TILES(FC_PHYSICAL_TILES),
        .LANES(LANES), .HIDDEN_MAX(HIDDEN_MAX),
        .INPUT_MAX(INPUT_MAX), .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH), .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .INCLUDE_FC_COMPAT(!USE_LEGACY_CANONICAL_FC),
        .USE_SHARED_FC_SRNN_STATE(USE_SHARED_FC_SRNN_STATE),
        .ENABLE_FC_TP_NATIVE_WEIGHTS(ENABLE_FC_TP_NATIVE_WEIGHTS),
        .FC_LEARN_W0(FC_LEARN_W0),
        .FC_TP_TENSOR_ID_WIDTH(FC_TP_TENSOR_ID_WIDTH),
        .FC_TP_PARAM_ADDR_WIDTH(FC_TP_PARAM_ADDR_WIDTH),
        .FC_TP_TAG_WIDTH(FC_TP_TAG_WIDTH),
        .FC_TP_CODE_WIDTH(FC_TP_CODE_WIDTH),
        .FC_TP_RESIDUAL_WIDTH(FC_TP_RESIDUAL_WIDTH),
        .FC_WEIGHT_DIR(FC_COMPAT_WEIGHT_DIR),
        .FC_W0_RESIDUAL_INIT_FILE(FC_TP_W0_RESIDUAL_INIT_FILE)
    ) u_shared_h40 (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_abort(i_abort || fc_shared_abort),
        .i_clear_error(i_clear_error),

        .i_fc_sample_start(fc_shared_sample_start),
        .o_fc_sample_ready(fc_shared_sample_ready),
        .i_fc_sample_clear_state(fc_shared_sample_clear_state),
        .i_fc_active_input_dim(fc_shared_active_input_dim),
        .i_fc_requested_tile_mask(fc_shared_requested_tile_mask),
        .i_fc_window_valid(fc_shared_window_valid),
        .o_fc_window_ready(fc_shared_window_ready),
        .i_fc_record_valid(fc_shared_record_valid),
        .i_fc_sample_id(fc_shared_sample_id),
        .i_fc_window_timestep(fc_shared_window_timestep),
        .i_fc_boundary_mask(fc_shared_boundary_mask),
        .i_fc_sample_last(fc_shared_sample_last),
        .i_fc_fi_flat(fc_shared_fi_flat),
        .i_fc_x_flat(fc_shared_x_flat),
        .o_fc_timestep_done(fc_shared_timestep_done),
        .o_fc_sample_done(fc_shared_sample_done),
        .o_fc_completed_timestep(),
        .o_fc_logits_flat(fc_shared_logits_flat),
        .o_fc_pred(fc_shared_pred),
        .o_fc_spikes_flat(o_fc_spikes_flat),
        .o_fc_vmem_flat(o_fc_vmem_flat),
        .o_fc_trace_flat(o_fc_trace_flat),
        .o_fc_accepted_windows(fc_shared_accepted_windows),
        .o_fc_accepted_events(fc_shared_accepted_events),
        .o_fc_issue_groups(fc_shared_issue_groups),
        .o_fc_boundary_count(fc_shared_boundary_count),
        .o_fc_dropped_events(fc_shared_dropped_events),
        .o_fc_sample_cycles(fc_shared_sample_cycles),
        .o_fc_protocol_error(fc_shared_protocol_error),
        .i_fc_tp_w0_parameter_base(i_fc_tp_w0_parameter_base),
        .i_fc_tp_w2_parameter_base(i_fc_tp_w2_parameter_base),
        .i_fc_tp_learning_read_valid(i_fc_tp_learning_read_valid),
        .o_fc_tp_learning_read_ready(o_fc_tp_learning_read_ready),
        .i_fc_tp_learning_read_tensor_id(
            i_fc_tp_learning_read_tensor_id),
        .i_fc_tp_learning_read_parameter_addr(
            i_fc_tp_learning_read_parameter_addr),
        .i_fc_tp_learning_read_tag(i_fc_tp_learning_read_tag),
        .o_fc_tp_learning_response_valid(
            o_fc_tp_learning_response_valid),
        .i_fc_tp_learning_response_ready(i_fc_tp_learning_response_ready),
        .o_fc_tp_learning_response_tag(o_fc_tp_learning_response_tag),
        .o_fc_tp_learning_response_code(o_fc_tp_learning_response_code),
        .o_fc_tp_learning_response_residual(
            o_fc_tp_learning_response_residual),
        .o_fc_tp_learning_response_error(
            o_fc_tp_learning_response_error),
        .i_fc_tp_learning_write_valid(i_fc_tp_learning_write_valid),
        .o_fc_tp_learning_write_ready(o_fc_tp_learning_write_ready),
        .i_fc_tp_learning_write_tensor_id(
            i_fc_tp_learning_write_tensor_id),
        .i_fc_tp_learning_write_parameter_addr(
            i_fc_tp_learning_write_parameter_addr),
        .i_fc_tp_learning_write_code(i_fc_tp_learning_write_code),
        .i_fc_tp_learning_write_residual(i_fc_tp_learning_write_residual),
        .i_fc_tp_learning_write_code_enable(
            i_fc_tp_learning_write_code_enable),
        .i_fc_tp_learning_write_residual_enable(
            i_fc_tp_learning_write_residual_enable),
        .o_fc_tp_learning_quiescent(o_fc_tp_learning_quiescent),

        .i_srnn_start_valid(srnn_start_valid_gated),
        .o_srnn_start_ready(srnn_start_ready_internal),
        .i_srnn_state_reset(i_srnn_state_reset),
        .i_srnn_input_dim(i_srnn_input_dim),
        .i_srnn_active_neurons(i_srnn_active_neurons),
        .i_srnn_w0_base(i_srnn_w0_base),
        .i_srnn_recurrent_base(i_srnn_recurrent_base),
        .i_srnn_recurrent_enable(i_srnn_recurrent_enable),
        .i_srnn_leak_q15(i_srnn_leak_q15),
        .i_srnn_threshold(i_srnn_threshold),
        .i_srnn_current_left_shift(i_srnn_current_left_shift),
        .i_srnn_reset_to_zero(i_srnn_reset_to_zero),
        .i_srnn_version(i_locked_versions[63:48]),
        .i_srnn_event_valid(srnn_event_valid),
        .o_srnn_event_ready(srnn_event_ready),
        .i_srnn_event_addr(srnn_event_addr),
        .i_srnn_event_data(srnn_event_data),
        .i_srnn_timestep_last(srnn_timestep_last),
        .i_srnn_sample_last(srnn_sample_last),
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
        .o_srnn_result_valid(srnn_timestep_valid),
        .i_srnn_result_ready(srnn_timestep_ready),
        .o_srnn_result_version(srnn_timestep_version),
        .o_srnn_result_sample_last(srnn_timestep_sample_last),
        .o_srnn_result_spikes(srnn_timestep_spikes),
        .o_srnn_previous_spikes(o_srnn_previous_spikes),
        .o_srnn_done(srnn_leaf_done),
        .i_scnn_start_valid(scnn_start_valid_gated),
        .o_scnn_start_ready(scnn_start_ready_internal),
        .i_scnn_in_width(i_scnn_in_width),
        .i_scnn_in_height(i_scnn_in_height),
        .i_scnn_total_in_channels(i_scnn_total_in_channels),
        .i_scnn_out_width(i_scnn_out_width),
        .i_scnn_out_height(i_scnn_out_height),
        .i_scnn_total_out_channels(i_scnn_total_out_channels),
        .i_scnn_input_channel_base(i_scnn_input_channel_base),
        .i_scnn_input_channel_count(i_scnn_input_channel_count),
        .i_scnn_output_channel_base(i_scnn_output_channel_base),
        .i_scnn_output_channel_count(i_scnn_output_channel_count),
        .i_scnn_stride(i_scnn_stride), .i_scnn_padding(i_scnn_padding),
        .i_scnn_input_slot(i_scnn_input_slot),
        .i_scnn_feature_base(i_scnn_feature_base),
        .i_scnn_weight_q16_base(i_scnn_weight_q16_base),
        .i_scnn_output_base(i_scnn_output_base),
        .i_scnn_unit_spike_features(i_scnn_unit_spike_features),
        .i_scnn_external_mode(i_scnn_external_mode),
        .o_scnn_feature_read_valid(o_scnn_feature_read_valid),
        .i_scnn_feature_read_ready(i_scnn_feature_read_ready),
        .o_scnn_feature_read_slot(o_scnn_feature_read_slot),
        .o_scnn_feature_read_bank_flat(o_scnn_feature_read_bank_flat),
        .o_scnn_feature_read_addr_flat(o_scnn_feature_read_addr_flat),
        .o_scnn_feature_read_lane_mask(o_scnn_feature_read_lane_mask),
        .i_scnn_feature_valid(i_scnn_feature_valid),
        .o_scnn_feature_ready(o_scnn_feature_ready),
        .i_scnn_feature_flat(i_scnn_feature_flat),
        .i_scnn_feature_lane_mask(i_scnn_feature_lane_mask),
        .o_scnn_weight_read_valid(o_scnn_weight_read_valid),
        .i_scnn_weight_read_ready(i_scnn_weight_read_ready),
        .o_scnn_weight_read_addr0(o_scnn_weight_read_addr0),
        .o_scnn_weight_read_addr1(o_scnn_weight_read_addr1),
        .i_scnn_weight_result_valid(i_scnn_weight_result_valid),
        .o_scnn_weight_result_ready(o_scnn_weight_result_ready),
        .i_scnn_weight_data0(i_scnn_weight_data0),
        .i_scnn_weight_data1(i_scnn_weight_data1),
        .o_scnn_result_valid(o_scnn_raw_result_valid),
        .i_scnn_result_ready(i_scnn_raw_result_ready),
        .o_scnn_result_mask(o_scnn_raw_result_mask),
        .o_scnn_result_output_addr_flat(
            o_scnn_raw_result_output_addr_flat),
        .o_scnn_result_spatial_addr(o_scnn_raw_result_spatial_addr),
        .o_scnn_result_accumulator_flat(
            o_scnn_raw_result_accumulator_flat),
        .o_scnn_done(o_scnn_compute_done),
        .i_scnn_external_request_valid(
            i_scnn_external_request_valid),
        .o_scnn_external_request_ready(
            o_scnn_external_request_ready),
        .i_scnn_external_engine_mask(i_scnn_external_engine_mask),
        .i_scnn_external_feature_flat(i_scnn_external_feature_flat),
        .i_scnn_external_weight_flat(i_scnn_external_weight_flat),
        .i_scnn_external_lane_mask(i_scnn_external_lane_mask),
        .i_scnn_external_first(i_scnn_external_first),
        .i_scnn_external_last(i_scnn_external_last),
        .i_scnn_external_output_channel_base(
            i_scnn_external_output_channel_base),
        .o_scnn_external_result_valid(
            o_scnn_external_result_valid),
        .i_scnn_external_result_ready(
            i_scnn_external_result_ready),
        .o_scnn_external_result_mask(o_scnn_external_result_mask),
        .o_scnn_external_result_accumulator_flat(
            o_scnn_external_result_accumulator_flat),
        .i_scnn_external_done(i_scnn_external_done),
        .i_scnn_external_sticky_error(
            i_scnn_external_sticky_error),
        .o_active_owner(shared_active_owner),
        .o_shared_request_fire(), .o_shared_result_fire(),
        .o_scnn_zero_lane_normalized(), .o_busy(shared_busy),
        .o_idle(shared_idle), .o_sticky_error(shared_error),
        .o_error_code(shared_error_code)
    );

    // Preserve the historical extension-status ABI while the shared array uses
    // the unified FC/SRNN/SCNN owner encoding internally.
    assign o_extension_owner = shared_active_owner == OWNER_SRNN ? 2'd1 :
        shared_active_owner == OWNER_SCNN ? 2'd2 : 2'd0;
    assign extension_busy = shared_busy &&
        (shared_active_owner == OWNER_SRNN ||
         shared_active_owner == OWNER_SCNN);
    assign extension_idle = !extension_busy;
    assign extension_error = shared_error;
    assign extension_error_code = shared_error_code;

    unified_recurrent_output_bridge #(
        .TILE_COUNT(ENGINES), .NEURONS_PER_TILE(40),
        .OUTPUT_MAX(OUTPUT_MAX), .ACC_BITS(32)
    ) u_srnn_output (
        .clk(i_clk), .rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(i_abort),
        .i_timestep_valid(srnn_timestep_valid),
        .o_timestep_ready(srnn_timestep_ready),
        .i_timestep_sample_last(srnn_timestep_sample_last),
        .i_spikes_flat(srnn_timestep_spikes),
        .i_active_tiles(i_srnn_active_tiles),
        .i_output_count(i_srnn_output_count),
        .i_w2_row_stride(i_srnn_w2_row_stride),
        .i_w2_base(i_srnn_w2_base),
        .i_version(i_locked_versions[63:48]),
        .o_w2_req_valid(o_srnn_w2_req_valid),
        .i_w2_req_ready(i_srnn_w2_req_ready),
        .o_w2_addr_flat(o_srnn_w2_addr_flat),
        .i_w2_data_flat(i_srnn_w2_data_flat),
        .o_result_valid(dynamic_result_valid),
        .i_result_ready(dynamic_result_ready),
        .o_result_class(dynamic_result_class),
        .o_result_score(dynamic_result_score),
        .o_result_logits_flat(o_srnn_logits_flat),
        .o_busy(dynamic_busy), .o_sticky_error(dynamic_error)
    );
    assign o_srnn_compute_done = srnn_leaf_done;
    // Passive TP observation boundary: expose only accepted recurrent states.
    assign o_tp_srnn_timestep_commit =
        srnn_timestep_valid && srnn_timestep_ready;
    assign o_tp_srnn_timestep_version = srnn_timestep_version;
    assign o_tp_srnn_timestep_sample_last = srnn_timestep_sample_last;
    assign o_tp_srnn_timestep_spikes = srnn_timestep_spikes;

    unified_result_owner_arbiter_v1 #(
        .CLASS_WIDTH(6), .SCORE_WIDTH(48), .VERSION_WIDTH(64)
    ) u_result_owner (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error && !o_result_pending),
        .i_flush(i_abort || fc_shared_abort),
        .i_expected_owner(i_requested_owner),
        .i_fc_valid(fc_result_valid), .o_fc_ready(fc_result_ready),
        .i_fc_class(fc_result_class),
        .i_fc_score({{16{fc_result_score[31]}}, fc_result_score}),
        .i_fc_versions(i_locked_versions),
        .i_dynamic_valid(dynamic_result_valid),
        .o_dynamic_ready(dynamic_result_ready),
        .i_dynamic_class(dynamic_result_class),
        .i_dynamic_score({{16{dynamic_result_score[31]}},
                          dynamic_result_score}),
        .i_dynamic_versions(i_locked_versions),
        .i_scnn_valid(i_scnn_li_result_valid),
        .o_scnn_ready(o_scnn_li_result_ready),
        .i_scnn_class(i_scnn_li_result_class),
        .i_scnn_score(i_scnn_li_result_score),
        .i_scnn_versions(i_locked_versions),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_class(o_result_class), .o_result_score(o_result_score),
        .o_result_versions(o_result_versions),
        .o_result_owner(o_result_owner),
        .o_pending_mask(result_pending_mask), .o_pending(),
        .o_sticky_error(result_error),
        .o_error_code(result_error_code)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            local_error_q <= 1'b0;
        else if (i_abort)
            local_error_q <= 1'b0;
        else begin
            if (i_clear_error && o_idle)
                local_error_q <= 1'b0;
            if (owner_command_error)
                local_error_q <= 1'b1;
        end
    end

    logic unused_status;
    assign unused_status = &{1'b0, ingress_idle, fc_idle,
                             srnn_timestep_version};
endmodule

// Stable FC-owner boundary used by the shared datapath. The legacy branch is
// retained for differential regressions; new paper-oriented integrations use
// the FC_COMPAT branch without changing graph, ingress, or result protocols.
module unified_fc_owner_selector_v1 #(
    parameter bit USE_LEGACY_CANONICAL_FC = 1'b0,
    parameter LEGACY_WEIGHT_DIR =
        "../../../data/weights/r19b_h480_complete_fc_input2312",
    parameter FC_COMPAT_WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312",
    parameter int FC_PHYSICAL_TILES = 5,
    parameter int ENGINES = 12,
    parameter int INPUT_MAX = 2312
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,
    input  logic                         i_abort,
    input  logic                         i_layer_valid,
    output logic                         o_layer_ready,
    input  logic [383:0]                 i_layer_words,
    input  logic                         i_clear_state,
    input  logic [15:0]                  i_start_sample_id,
    input  logic                         i_window_valid,
    output logic                         o_window_ready,
    input  logic [7:0]                   i_record_valid,
    input  logic [15:0]                  i_window_sample_id,
    input  logic [8:0]                   i_window_timestep,
    input  logic [7:0]                   i_boundary_mask,
    input  logic                         i_window_sample_last,
    input  logic [8*12-1:0]              i_fi_flat,
    input  logic [63:0]                  i_x_flat,
    output logic                         o_timestep_done,
    output logic                         o_layer_compute_done,
    output logic                         o_result_valid,
    input  logic                         i_result_ready,
    output logic [5:0]                   o_result_class,
    output logic signed [31:0]           o_result_score,
    output logic [10*24-1:0]             o_logits_flat,
    output logic [31:0]                  o_sample_cycles,
    output logic [31:0]                  o_accepted_windows,
    output logic [31:0]                  o_accepted_events,
    output logic [31:0]                  o_issue_groups,
    output logic [31:0]                  o_boundary_count,
    output logic [31:0]                  o_runtime_dropped_events,
    output logic [11:0]                  o_locked_cluster_mask,
    output logic [12:0]                  o_locked_input_dim,
    output logic                         o_active_bank,
    output logic [15:0]                  o_active_version,
    output logic                         o_update_window_ready,
    output logic                         o_commit_pending,
    output logic                         o_snapshot_busy,
    input  logic [3:0]                   i_debug_cluster,
    input  logic [6:0]                   i_debug_word_addr,
    output logic [39:0]                  o_debug_active_word,
    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code,

    output logic                         o_shared_sample_start,
    input  logic                         i_shared_sample_ready,
    output logic                         o_shared_sample_clear_state,
    output logic [12:0]                  o_shared_active_input_dim,
    output logic [ENGINES-1:0]           o_shared_requested_tile_mask,
    output logic                         o_shared_window_valid,
    input  logic                         i_shared_window_ready,
    output logic [7:0]                   o_shared_record_valid,
    output logic [15:0]                  o_shared_sample_id,
    output logic [8:0]                   o_shared_window_timestep,
    output logic [7:0]                   o_shared_boundary_mask,
    output logic                         o_shared_sample_last,
    output logic [8*12-1:0]              o_shared_fi_flat,
    output logic [63:0]                  o_shared_x_flat,
    input  logic                         i_shared_timestep_done,
    input  logic                         i_shared_sample_done,
    input  logic [10*24-1:0]             i_shared_logits_flat,
    input  logic [3:0]                   i_shared_pred,
    input  logic [63:0]                  i_shared_sample_cycles,
    input  logic [31:0]                  i_shared_accepted_windows,
    input  logic [31:0]                  i_shared_accepted_events,
    input  logic [31:0]                  i_shared_issue_groups,
    input  logic [31:0]                  i_shared_boundary_count,
    input  logic [31:0]                  i_shared_dropped_events,
    input  logic                         i_shared_protocol_error,
    input  logic                         i_shared_fc_active,
    output logic                         o_shared_abort
);
    generate
        if (USE_LEGACY_CANONICAL_FC) begin : g_legacy
            unified_fc_graph_layer_adapter_v1 #(
                .WEIGHT_DIR(LEGACY_WEIGHT_DIR)
            ) u_owner (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_clear_error(i_clear_error),
                .i_layer_valid(i_layer_valid), .o_layer_ready(o_layer_ready),
                .i_layer_words(i_layer_words), .i_clear_state(i_clear_state),
                .i_start_sample_id(i_start_sample_id),
                .i_window_valid(i_window_valid),
                .o_window_ready(o_window_ready),
                .i_record_valid(i_record_valid),
                .i_window_sample_id(i_window_sample_id),
                .i_window_timestep(i_window_timestep),
                .i_boundary_mask(i_boundary_mask),
                .i_window_sample_last(i_window_sample_last),
                .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
                .o_timestep_done(o_timestep_done),
                .o_layer_compute_done(o_layer_compute_done),
                .o_result_valid(o_result_valid),
                .i_result_ready(i_result_ready),
                .o_result_class(o_result_class),
                .o_result_score(o_result_score),
                .o_logits_flat(o_logits_flat),
                .o_sample_cycles(o_sample_cycles),
                .o_accepted_windows(o_accepted_windows),
                .o_accepted_events(o_accepted_events),
                .o_issue_groups(o_issue_groups),
                .o_boundary_count(o_boundary_count),
                .o_runtime_dropped_events(o_runtime_dropped_events),
                .o_locked_cluster_mask(o_locked_cluster_mask),
                .o_locked_input_dim(o_locked_input_dim),
                .o_active_bank(o_active_bank),
                .o_active_version(o_active_version),
                .o_update_window_ready(o_update_window_ready),
                .o_commit_pending(o_commit_pending),
                .o_snapshot_busy(o_snapshot_busy),
                .i_debug_cluster(i_debug_cluster),
                .i_debug_word_addr(i_debug_word_addr),
                .o_debug_active_word(o_debug_active_word),
                .o_busy(o_busy), .o_idle(o_idle),
                .o_sticky_error(o_sticky_error),
                .o_error_code(o_error_code)
            );

            logic unused_abort;
            assign unused_abort = i_abort;
            assign o_shared_sample_start = 1'b0;
            assign o_shared_sample_clear_state = 1'b0;
            assign o_shared_active_input_dim = '0;
            assign o_shared_requested_tile_mask = '0;
            assign o_shared_window_valid = 1'b0;
            assign o_shared_record_valid = '0;
            assign o_shared_sample_id = '0;
            assign o_shared_window_timestep = '0;
            assign o_shared_boundary_mask = '0;
            assign o_shared_sample_last = 1'b0;
            assign o_shared_fi_flat = '0;
            assign o_shared_x_flat = '0;
            assign o_shared_abort = 1'b0;

            logic unused_shared_inputs;
            assign unused_shared_inputs = &{1'b0,
                i_shared_sample_ready, i_shared_window_ready,
                i_shared_timestep_done, i_shared_sample_done,
                i_shared_logits_flat, i_shared_pred,
                i_shared_sample_cycles, i_shared_accepted_windows,
                i_shared_accepted_events, i_shared_issue_groups,
                i_shared_boundary_count, i_shared_dropped_events,
                i_shared_protocol_error, i_shared_fc_active};
        end else begin : g_fc_compat
            unified_fc_shared_h40_graph_adapter_v1 #(
                .PHYSICAL_TILES(FC_PHYSICAL_TILES),
                .ENGINES(ENGINES),
                .INPUT_MAX(INPUT_MAX)
            ) u_owner (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_clear_error(i_clear_error), .i_abort(i_abort),
                .i_layer_valid(i_layer_valid), .o_layer_ready(o_layer_ready),
                .i_layer_words(i_layer_words), .i_clear_state(i_clear_state),
                .i_start_sample_id(i_start_sample_id),
                .i_window_valid(i_window_valid),
                .o_window_ready(o_window_ready),
                .i_record_valid(i_record_valid),
                .i_window_sample_id(i_window_sample_id),
                .i_window_timestep(i_window_timestep),
                .i_boundary_mask(i_boundary_mask),
                .i_window_sample_last(i_window_sample_last),
                .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
                .o_timestep_done(o_timestep_done),
                .o_layer_compute_done(o_layer_compute_done),
                .o_result_valid(o_result_valid),
                .i_result_ready(i_result_ready),
                .o_result_class(o_result_class),
                .o_result_score(o_result_score),
                .o_logits_flat(o_logits_flat),
                .o_sample_cycles(o_sample_cycles),
                .o_accepted_windows(o_accepted_windows),
                .o_accepted_events(o_accepted_events),
                .o_issue_groups(o_issue_groups),
                .o_boundary_count(o_boundary_count),
                .o_runtime_dropped_events(o_runtime_dropped_events),
                .o_locked_cluster_mask(o_locked_cluster_mask),
                .o_locked_input_dim(o_locked_input_dim),
                .o_active_bank(o_active_bank),
                .o_active_version(o_active_version),
                .o_update_window_ready(o_update_window_ready),
                .o_commit_pending(o_commit_pending),
                .o_snapshot_busy(o_snapshot_busy),
                .i_debug_cluster(i_debug_cluster),
                .i_debug_word_addr(i_debug_word_addr),
                .o_debug_active_word(o_debug_active_word),
                .o_busy(o_busy), .o_idle(o_idle),
                .o_sticky_error(o_sticky_error),
                .o_error_code(o_error_code),
                .o_shared_sample_start(o_shared_sample_start),
                .i_shared_sample_ready(i_shared_sample_ready),
                .o_shared_sample_clear_state(o_shared_sample_clear_state),
                .o_shared_active_input_dim(o_shared_active_input_dim),
                .o_shared_requested_tile_mask(
                    o_shared_requested_tile_mask),
                .o_shared_window_valid(o_shared_window_valid),
                .i_shared_window_ready(i_shared_window_ready),
                .o_shared_record_valid(o_shared_record_valid),
                .o_shared_sample_id(o_shared_sample_id),
                .o_shared_window_timestep(o_shared_window_timestep),
                .o_shared_boundary_mask(o_shared_boundary_mask),
                .o_shared_sample_last(o_shared_sample_last),
                .o_shared_fi_flat(o_shared_fi_flat),
                .o_shared_x_flat(o_shared_x_flat),
                .i_shared_timestep_done(i_shared_timestep_done),
                .i_shared_sample_done(i_shared_sample_done),
                .i_shared_logits_flat(i_shared_logits_flat),
                .i_shared_pred(i_shared_pred),
                .i_shared_sample_cycles(i_shared_sample_cycles),
                .i_shared_accepted_windows(i_shared_accepted_windows),
                .i_shared_accepted_events(i_shared_accepted_events),
                .i_shared_issue_groups(i_shared_issue_groups),
                .i_shared_boundary_count(i_shared_boundary_count),
                .i_shared_dropped_events(i_shared_dropped_events),
                .i_shared_protocol_error(i_shared_protocol_error),
                .i_shared_fc_active(i_shared_fc_active),
                .o_shared_abort(o_shared_abort)
            );
        end
    endgenerate

    initial begin
        if (!USE_LEGACY_CANONICAL_FC &&
            (FC_PHYSICAL_TILES < 1 || FC_PHYSICAL_TILES > 12))
            $error("FC_COMPAT owner requires 1..12 H40 physical tiles");
    end
endmodule

`default_nettype wire
