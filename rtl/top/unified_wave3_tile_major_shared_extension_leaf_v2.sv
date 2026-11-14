`timescale 1ns/1ps
`default_nettype none

// Production inference leaf for the shared-compute paper architecture. SRNN
// and SCNN retain their external protocols while sharing one physical H40
// array. Persistent recurrent state is owned only by that array.
module unified_wave3_tile_major_shared_extension_leaf_v2 #(
    parameter int ENGINES = 12,
    parameter int FC_TILES = ENGINES,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int VERSION_WIDTH = 16,
    parameter int REQUEST_ID_WIDTH = 16,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int FEATURE_DEPTH = 1 << FEATURE_ADDR_WIDTH,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48,
    parameter bit INCLUDE_FC_COMPAT = 1'b1,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,

    input  logic                                      i_fc_start_valid,
    output logic                                      o_fc_start_ready,
    input  logic                                      i_fc_clear_state,
    input  logic [ENGINES-1:0]                        i_fc_active_tile_mask,
    input  logic                                      i_fc_step_start,
    output logic                                      o_fc_step_ready,
    input  logic [8:0]                                i_fc_timestep,
    input  logic                                      i_fc_row_valid,
    output logic                                      o_fc_row_ready,
    input  logic [LANES-1:0]                          i_fc_lane_valid,
    input  logic [ENGINES*LANES*40*5-1:0]             i_fc_w0_weight_rows_flat,
    input  logic                                      i_fc_step_close,
    input  logic [ENGINES*40*10*8-1:0]                i_fc_w2_weight_flat,
    output logic                                      o_fc_result_valid,
    input  logic                                      i_fc_result_ready,
    output logic [8:0]                                o_fc_result_timestep,
    output logic [31:0]                               o_fc_result_events,
    output logic [ENGINES*40-1:0]                     o_fc_result_spikes,
    output logic [10*24-1:0]                          o_fc_result_logits_flat,
    output logic [3:0]                                o_fc_result_pred,
    output logic [ENGINES*40*18-1:0]                  o_fc_vmem_flat,
    output logic [ENGINES*40*16-1:0]                  o_fc_trace_flat,
    input  logic                                      i_fc_finish,
    output logic                                      o_fc_finish_ready,
    output logic                                      o_fc_done,

    input  logic                                      i_srnn_start_valid,
    output logic                                      o_srnn_start_ready,
    input  logic                                      i_srnn_state_reset,
    input  logic [12:0]                               i_srnn_input_dim,
    input  logic [9:0]                                i_srnn_active_neurons,
    input  logic [31:0]                               i_srnn_w0_base,
    input  logic [31:0]                               i_srnn_recurrent_base,
    input  logic                                      i_srnn_recurrent_enable,
    input  logic signed [15:0]                       i_srnn_leak_q15,
    input  logic signed [31:0]                       i_srnn_threshold,
    input  logic [4:0]                                i_srnn_current_left_shift,
    input  logic                                      i_srnn_reset_to_zero,
    input  logic [VERSION_WIDTH-1:0]                  i_srnn_version,
    input  logic                                      i_srnn_event_valid,
    output logic                                      o_srnn_event_ready,
    input  logic [12:0]                               i_srnn_event_addr,
    input  logic signed [7:0]                        i_srnn_event_data,
    input  logic                                      i_srnn_timestep_last,
    input  logic                                      i_srnn_sample_last,

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

    output logic                                      o_srnn_result_valid,
    input  logic                                      i_srnn_result_ready,
    output logic [VERSION_WIDTH-1:0]                  o_srnn_result_version,
    output logic                                      o_srnn_result_sample_last,
    output logic [HIDDEN_MAX-1:0]                     o_srnn_result_spikes,
    output logic [HIDDEN_MAX-1:0]                     o_srnn_previous_spikes,
    output logic                                      o_srnn_done,

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

    output logic                                      o_scnn_result_valid,
    input  logic                                      i_scnn_result_ready,
    output logic [ENGINES-1:0]                        o_scnn_result_mask,
    output logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      o_scnn_result_output_addr_flat,
    output logic [OUTPUT_ADDR_WIDTH-1:0]              o_scnn_result_spatial_addr,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_scnn_result_accumulator_flat,
    output logic                                      o_scnn_done,

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

    output logic                                      o_owner_active,
    output logic [1:0]                                o_active_owner,
    output logic                                      o_shared_request_fire,
    output logic                                      o_shared_result_fire,
    output logic                                      o_scnn_zero_lane_normalized,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_array_sticky_error,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_SRNN = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;
    localparam int TILE_NEURONS = 40;
    localparam int PHYSICAL_HIDDEN = ENGINES * TILE_NEURONS;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER_CONFLICT = 8'h01;
    localparam logic [7:0] ERROR_MIXED_OWNER = 8'h02;
    localparam logic [7:0] ERROR_WEIGHT_PROTOCOL = 8'h03;
    localparam logic [7:0] ERROR_ARRAY_OWNER = 8'h04;
    localparam logic [7:0] ERROR_SCNN_DONE_ORDER = 8'h05;

    typedef enum logic [3:0] {
        OWNER_STATE_IDLE,
        OWNER_STATE_FC_ACQUIRE,
        OWNER_STATE_FC,
        OWNER_STATE_FC_RELEASE,
        OWNER_STATE_SRNN_ACQUIRE,
        OWNER_STATE_SRNN,
        OWNER_STATE_SRNN_RELEASE,
        OWNER_STATE_SCNN_ACQUIRE,
        OWNER_STATE_SCNN,
        OWNER_STATE_SCNN_RELEASE
    } owner_state_t;

    owner_state_t owner_state_q;

    logic local_sticky_error_q;
    logic [7:0] local_error_code_q;
    logic operational_idle;
    logic srnn_shape_legal;

    logic array_owner_valid;
    logic array_owner_ready;
    logic [1:0] array_owner;
    logic array_owner_release_valid;
    logic array_owner_release_ready;
    logic array_owner_active;
    logic [1:0] array_active_owner;
    logic array_fc_step_ready;
    logic array_fc_row_ready;
    logic array_valid;
    logic array_ready;
    logic [ENGINES-1:0] array_engine_mask;
    logic [LANES-1:0] array_lane_mask;
    logic [LANES*16-1:0] array_q8_feature_flat;
    logic [ENGINES*LANES*16-1:0] array_q8_weight_flat;
    logic [LANES*8-1:0] array_int8_lhs_flat;
    logic [ENGINES*LANES*8-1:0] array_int8_rhs_flat;
    logic array_first;
    logic array_last;
    logic [2:0] array_local_word;
    logic [CHANNEL_WIDTH-1:0] array_output_channel_base;
    logic array_result_valid;
    logic array_result_ready;
    logic [1:0] array_result_owner;
    logic array_result_mode_q8_8;
    logic [ENGINES-1:0] array_result_engine_mask;
    logic [LANES-1:0] array_result_lane_mask;
    logic [ENGINES*CHANNEL_WIDTH-1:0] array_result_channel_flat;
    logic [ENGINES*ACC_WIDTH-1:0] array_result_q8_accumulator_flat;
    logic [ENGINES*LANES*32-1:0] array_result_int8_contribution_flat;
    logic [8:0] array_result_fc_timestep;
    logic [31:0] array_result_fc_events;
    logic [ENGINES*40-1:0] array_result_fc_spikes;
    logic [10*24-1:0] array_result_fc_logits_flat;
    logic [3:0] array_result_fc_pred;
    logic [PHYSICAL_HIDDEN-1:0] array_previous_spikes;
    logic [PHYSICAL_HIDDEN-1:0] array_current_spikes;
    logic array_busy;
    logic array_idle;
    logic array_sticky_error;

    logic srnn_start_pending_q;
    logic srnn_done_seen_q;
    logic srnn_reset_only_q;
    logic srnn_reset_direct_active_q;
    logic [2:0] srnn_reset_word_q;
    logic [12:0] srnn_input_dim_q;
    logic [9:0] srnn_active_neurons_q;
    logic [31:0] srnn_w0_base_q;
    logic [31:0] srnn_recurrent_base_q;
    logic srnn_recurrent_enable_q;
    logic signed [15:0] srnn_leak_q15_q;
    logic signed [31:0] srnn_threshold_q;
    logic [4:0] srnn_current_left_shift_q;
    logic srnn_reset_to_zero_q;
    logic [VERSION_WIDTH-1:0] srnn_version_q;
    logic srnn_frontend_start_ready;
    logic srnn_frontend_start_fire;
    logic srnn_frontend_array_valid;
    logic srnn_frontend_array_ready;
    logic [ENGINES-1:0] srnn_frontend_array_engine_mask;
    logic [LANES-1:0] srnn_frontend_array_lane_mask;
    logic [LANES*8-1:0] srnn_frontend_array_lhs_flat;
    logic [ENGINES*LANES*8-1:0] srnn_frontend_array_rhs_flat;
    logic srnn_frontend_array_first;
    logic srnn_frontend_array_last;
    logic [2:0] srnn_frontend_array_local_word;
    logic srnn_frontend_array_result_ready;
    logic srnn_state_clear_spikes;
    logic srnn_state_clear_word_valid;
    logic [2:0] srnn_state_clear_local_word;
    logic [9:0] srnn_state_active_neurons;
    logic srnn_state_lif_begin;
    logic srnn_state_lif_word_valid;
    logic [2:0] srnn_state_lif_local_word;
    logic signed [15:0] srnn_state_leak_q15;
    logic signed [31:0] srnn_state_threshold;
    logic [4:0] srnn_state_current_left_shift;
    logic srnn_state_reset_to_zero;
    logic srnn_state_commit_timestep;
    logic srnn_timestep_valid;
    logic srnn_timestep_ready;
    logic [VERSION_WIDTH-1:0] srnn_timestep_version;
    logic srnn_timestep_sample_last;
    logic [PHYSICAL_HIDDEN-1:0] srnn_timestep_spikes;
    logic [PHYSICAL_HIDDEN-1:0] srnn_frontend_previous_spikes;
    logic srnn_frontend_busy;
    logic srnn_frontend_idle;
    logic srnn_frontend_sticky_error;
    logic [7:0] srnn_frontend_error_code;

    logic srnn_result_valid_q;
    logic [VERSION_WIDTH-1:0] srnn_result_version_q;
    logic srnn_result_sample_last_q;
    logic [HIDDEN_MAX-1:0] srnn_result_spikes_q;
    logic srnn_result_push;
    logic srnn_result_pop;
    logic srnn_result_queue_ready;

    logic scnn_start_pending_q;
    logic scnn_done_seen_q;
    logic scnn_result_enqueued_q;
    logic scnn_external_mode_q;
    logic [15:0] scnn_in_width_q;
    logic [15:0] scnn_in_height_q;
    logic [15:0] scnn_total_in_channels_q;
    logic [15:0] scnn_out_width_q;
    logic [15:0] scnn_out_height_q;
    logic [15:0] scnn_total_out_channels_q;
    logic [9:0] scnn_input_channel_base_q;
    logic [9:0] scnn_input_channel_count_q;
    logic [9:0] scnn_output_channel_base_q;
    logic [9:0] scnn_output_channel_count_q;
    logic [7:0] scnn_stride_q;
    logic [7:0] scnn_padding_q;
    logic scnn_input_slot_q;
    logic [FEATURE_ADDR_WIDTH-1:0] scnn_feature_base_q;
    logic [Q16_ADDR_WIDTH-1:0] scnn_weight_q16_base_q;
    logic [OUTPUT_ADDR_WIDTH-1:0] scnn_output_base_q;
    logic scnn_unit_spike_features_q;
    logic scnn_frontend_start_ready;
    logic scnn_frontend_start_fire;
    logic scnn_side_request_valid;
    logic scnn_side_request_ready;
    logic [ENGINES-1:0] scnn_side_engine_mask;
    logic [LANES*16-1:0] scnn_side_feature_flat;
    logic [ENGINES*LANES*16-1:0] scnn_side_weight_flat;
    logic [LANES-1:0] scnn_side_lane_mask;
    logic scnn_side_first;
    logic scnn_side_last;
    logic [9:0] scnn_side_output_channel_base;
    logic scnn_side_result_valid;
    logic scnn_side_result_ready;
    logic [ENGINES-1:0] scnn_frontend_result_mask;
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0] scnn_frontend_result_output_addr_flat;
    logic [OUTPUT_ADDR_WIDTH-1:0] scnn_frontend_result_spatial_addr;
    logic [ENGINES*ACC_WIDTH-1:0] scnn_frontend_result_accumulator_flat;
    logic scnn_frontend_result_valid;
    logic scnn_frontend_result_ready;
    logic scnn_frontend_busy;
    logic scnn_frontend_done;
    logic scnn_frontend_config_error;
    logic scnn_frontend_protocol_error;

    logic scnn_result_valid_q;
    logic [ENGINES-1:0] scnn_result_mask_q;
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0] scnn_result_output_addr_flat_q;
    logic [OUTPUT_ADDR_WIDTH-1:0] scnn_result_spatial_addr_q;
    logic [ENGINES*ACC_WIDTH-1:0] scnn_result_accumulator_flat_q;
    logic scnn_result_push;
    logic scnn_result_pop;
    logic scnn_result_queue_ready;

    assign operational_idle = owner_state_q == OWNER_STATE_IDLE &&
        !srnn_result_valid_q && !scnn_result_valid_q &&
        srnn_frontend_idle && !scnn_frontend_busy && array_idle;
    assign o_idle = operational_idle;
    assign o_busy = !operational_idle;
    assign o_owner_active = array_owner_active;

    always_comb begin
        unique case (owner_state_q)
            OWNER_STATE_FC_ACQUIRE,
            OWNER_STATE_FC,
            OWNER_STATE_FC_RELEASE: o_active_owner = OWNER_FC;
            OWNER_STATE_SRNN_ACQUIRE,
            OWNER_STATE_SRNN,
            OWNER_STATE_SRNN_RELEASE: o_active_owner = OWNER_SRNN;
            OWNER_STATE_SCNN_ACQUIRE,
            OWNER_STATE_SCNN,
            OWNER_STATE_SCNN_RELEASE: o_active_owner = OWNER_SCNN;
            default: o_active_owner = OWNER_IDLE;
        endcase
    end

    assign o_sticky_error = local_sticky_error_q ||
        srnn_frontend_sticky_error || scnn_frontend_config_error ||
        scnn_frontend_protocol_error || array_sticky_error ||
        (scnn_external_mode_q && i_scnn_external_sticky_error);
    assign o_array_sticky_error = array_sticky_error;

    always_comb begin
        if (local_sticky_error_q)
            o_error_code = local_error_code_q;
        else if (srnn_frontend_sticky_error)
            o_error_code = 8'h40 | srnn_frontend_error_code;
        else if (scnn_frontend_protocol_error)
            o_error_code = 8'h81;
        else if (scnn_frontend_config_error)
            o_error_code = 8'h82;
        else if (array_sticky_error)
            o_error_code = 8'h83;
        else if (scnn_external_mode_q && i_scnn_external_sticky_error)
            o_error_code = 8'h84;
        else
            o_error_code = ERROR_NONE;
    end

    assign srnn_shape_legal = i_srnn_active_neurons != 0 &&
        i_srnn_active_neurons <= 10'(HIDDEN_MAX);
    assign o_fc_start_ready = !i_abort &&
        owner_state_q == OWNER_STATE_IDLE &&
        !o_sticky_error && !i_srnn_start_valid && !i_scnn_start_valid &&
        array_owner_ready;
    assign o_srnn_start_ready = !i_abort &&
        owner_state_q == OWNER_STATE_IDLE &&
        !o_sticky_error && !i_fc_start_valid && !i_scnn_start_valid &&
        array_owner_ready && srnn_shape_legal;
    assign o_scnn_start_ready = !i_abort &&
        owner_state_q == OWNER_STATE_IDLE &&
        !o_sticky_error && !i_fc_start_valid && !i_srnn_start_valid &&
        array_owner_ready;

    assign array_owner_valid = !i_abort && (
        owner_state_q == OWNER_STATE_FC_ACQUIRE ||
        owner_state_q == OWNER_STATE_SRNN_ACQUIRE ||
        owner_state_q == OWNER_STATE_SCNN_ACQUIRE);
    assign array_owner = owner_state_q == OWNER_STATE_FC_ACQUIRE ? OWNER_FC :
        (owner_state_q == OWNER_STATE_SRNN_ACQUIRE ? OWNER_SRNN : OWNER_SCNN);
    assign array_owner_release_valid = !i_abort && (
        owner_state_q == OWNER_STATE_FC_RELEASE ||
        owner_state_q == OWNER_STATE_SRNN_RELEASE ||
        owner_state_q == OWNER_STATE_SCNN_RELEASE);

    assign o_fc_step_ready = !i_abort &&
        owner_state_q == OWNER_STATE_FC &&
        array_fc_step_ready;
    assign o_fc_row_ready = !i_abort &&
        owner_state_q == OWNER_STATE_FC &&
        array_fc_row_ready;
    assign o_fc_result_valid = !i_abort &&
        owner_state_q == OWNER_STATE_FC &&
        array_result_valid;
    assign o_fc_result_timestep = array_result_fc_timestep;
    assign o_fc_result_events = array_result_fc_events;
    assign o_fc_result_spikes = array_result_fc_spikes;
    assign o_fc_result_logits_flat = array_result_fc_logits_flat;
    assign o_fc_result_pred = array_result_fc_pred;
    assign o_fc_finish_ready = !i_abort &&
        owner_state_q == OWNER_STATE_FC &&
        !array_result_valid && array_owner_release_ready;

    assign srnn_frontend_start_fire = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        srnn_start_pending_q && srnn_frontend_start_ready;
    assign o_srnn_event_ready = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        !srnn_start_pending_q && !srnn_reset_only_q &&
        srnn_frontend_event_ready;

    // The explicit alias keeps event gating readable without leaking the
    // frontend's owner-enable protocol into the public interface.
    logic srnn_frontend_event_ready;

    assign srnn_result_queue_ready = !srnn_result_valid_q ||
        i_srnn_result_ready;
    assign srnn_timestep_ready = owner_state_q == OWNER_STATE_SRNN &&
        srnn_result_queue_ready;
    assign srnn_result_push = srnn_timestep_valid && srnn_timestep_ready;
    assign srnn_result_pop = srnn_result_valid_q && i_srnn_result_ready &&
        !srnn_result_push;

    assign o_srnn_result_valid = !i_abort && srnn_result_valid_q;
    assign o_srnn_result_version = srnn_result_version_q;
    assign o_srnn_result_sample_last = srnn_result_sample_last_q;
    assign o_srnn_result_spikes = srnn_result_spikes_q;
    // A new sample and an explicit reset both clear five tile-local state
    // words. Hide the previous sample while that atomic clear is in flight so
    // the control-plane ABI never exposes a partially-cleared spike vector.
    assign o_srnn_previous_spikes =
        (srnn_start_pending_q || srnn_reset_direct_active_q) ? '0 :
        srnn_result_push ? srnn_timestep_spikes[HIDDEN_MAX-1:0] :
        array_previous_spikes[HIDDEN_MAX-1:0];

    assign scnn_frontend_start_fire = !i_abort &&
        owner_state_q == OWNER_STATE_SCNN &&
        !scnn_external_mode_q && scnn_start_pending_q &&
        scnn_frontend_start_ready;
    assign scnn_result_queue_ready = !scnn_result_valid_q ||
        i_scnn_result_ready;
    assign scnn_frontend_result_ready = !i_abort &&
        scnn_result_queue_ready &&
        owner_state_q == OWNER_STATE_SCNN && !scnn_external_mode_q;
    assign scnn_result_push = scnn_frontend_result_valid &&
        scnn_frontend_result_ready;
    assign scnn_result_pop = scnn_result_valid_q && i_scnn_result_ready &&
        !scnn_result_push;

    assign o_scnn_result_valid = !i_abort && scnn_result_valid_q;
    assign o_scnn_result_mask = scnn_result_mask_q;
    assign o_scnn_result_output_addr_flat = scnn_result_output_addr_flat_q;
    assign o_scnn_result_spatial_addr = scnn_result_spatial_addr_q;
    assign o_scnn_result_accumulator_flat = scnn_result_accumulator_flat_q;

    assign srnn_frontend_array_ready = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        array_ready;
    assign scnn_side_request_ready = !i_abort &&
        owner_state_q == OWNER_STATE_SCNN &&
        !scnn_external_mode_q && array_ready;
    assign o_scnn_external_request_ready =
        !i_abort && owner_state_q == OWNER_STATE_SCNN && scnn_external_mode_q &&
        array_ready;

    always_comb begin
        array_valid = 1'b0;
        array_engine_mask = '0;
        array_lane_mask = '0;
        array_q8_feature_flat = '0;
        array_q8_weight_flat = '0;
        array_int8_lhs_flat = '0;
        array_int8_rhs_flat = '0;
        array_first = 1'b0;
        array_last = 1'b0;
        array_local_word = '0;
        array_output_channel_base = '0;

        if (!i_abort && owner_state_q == OWNER_STATE_SRNN) begin
            array_valid = srnn_frontend_array_valid;
            array_engine_mask = srnn_frontend_array_engine_mask;
            array_lane_mask = srnn_frontend_array_lane_mask;
            array_int8_lhs_flat = srnn_frontend_array_lhs_flat;
            array_int8_rhs_flat = srnn_frontend_array_rhs_flat;
            array_first = srnn_frontend_array_first;
            array_last = srnn_frontend_array_last;
            array_local_word = srnn_frontend_array_local_word;
        end else if (!i_abort && owner_state_q == OWNER_STATE_SCNN) begin
            if (scnn_external_mode_q) begin
                array_valid = i_scnn_external_request_valid;
                array_engine_mask = i_scnn_external_engine_mask;
                array_lane_mask = i_scnn_external_lane_mask == '0 ?
                    {{(LANES-1){1'b0}}, 1'b1} :
                    i_scnn_external_lane_mask;
                array_q8_feature_flat = i_scnn_external_feature_flat;
                array_q8_weight_flat = i_scnn_external_weight_flat;
                array_first = i_scnn_external_first;
                array_last = i_scnn_external_last;
                array_output_channel_base =
                    i_scnn_external_output_channel_base;
            end else begin
                array_valid = scnn_side_request_valid;
                array_engine_mask = scnn_side_engine_mask;
                array_lane_mask = scnn_side_lane_mask == '0 ?
                    {{(LANES-1){1'b0}}, 1'b1} : scnn_side_lane_mask;
                array_q8_feature_flat = scnn_side_feature_flat;
                array_q8_weight_flat = scnn_side_weight_flat;
                array_first = scnn_side_first;
                array_last = scnn_side_last;
                array_output_channel_base = scnn_side_output_channel_base;
            end
        end
    end

    always_comb begin
        array_result_ready = 1'b0;
        if (!i_abort && owner_state_q == OWNER_STATE_FC)
            array_result_ready = i_fc_result_ready;
        else if (!i_abort && owner_state_q == OWNER_STATE_SRNN)
            array_result_ready = srnn_frontend_array_result_ready;
        else if (!i_abort && owner_state_q == OWNER_STATE_SCNN &&
                 scnn_external_mode_q)
            array_result_ready = i_scnn_external_result_ready;
        else if (!i_abort && owner_state_q == OWNER_STATE_SCNN)
            array_result_ready = scnn_side_result_ready;
    end

    assign scnn_side_result_valid = !i_abort &&
        owner_state_q == OWNER_STATE_SCNN &&
        !scnn_external_mode_q && array_result_valid;
    assign o_scnn_external_result_valid =
        !i_abort && owner_state_q == OWNER_STATE_SCNN && scnn_external_mode_q &&
        array_result_valid;
    assign o_scnn_external_result_mask = array_result_engine_mask;
    assign o_scnn_external_result_accumulator_flat =
        array_result_q8_accumulator_flat;

    assign o_shared_request_fire = !i_abort &&
        ((array_valid && array_ready) ||
        (owner_state_q == OWNER_STATE_FC &&
         i_fc_row_valid && o_fc_row_ready));
    assign o_shared_result_fire = !i_abort &&
        array_result_valid && array_result_ready;
    assign o_scnn_zero_lane_normalized = o_shared_request_fire &&
        owner_state_q == OWNER_STATE_SCNN &&
        (scnn_external_mode_q ? i_scnn_external_lane_mask == '0 :
                               scnn_side_lane_mask == '0);

    unified_srnn_tile_major_frontend_v1 #(
        .HIDDEN_MAX(PHYSICAL_HIDDEN),
        .INPUT_MAX(INPUT_MAX),
        .TILES(ENGINES),
        .TILE_NEURONS(TILE_NEURONS),
        .ENGINES(ENGINES),
        .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH)
    ) u_srnn_frontend (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_owner_enable(owner_state_q == OWNER_STATE_SRNN),
        .i_clear_error(i_clear_error),
        .i_state_reset(i_srnn_state_reset &&
            owner_state_q == OWNER_STATE_SRNN && !srnn_reset_only_q),
        .i_start_valid(srnn_frontend_start_fire),
        .o_start_ready(srnn_frontend_start_ready),
        .i_input_dim(srnn_input_dim_q),
        .i_active_neurons(srnn_active_neurons_q),
        .i_w0_base(srnn_w0_base_q),
        .i_recurrent_base(srnn_recurrent_base_q),
        .i_recurrent_enable(srnn_recurrent_enable_q),
        .i_leak_q15(srnn_leak_q15_q),
        .i_threshold(srnn_threshold_q),
        .i_current_left_shift(srnn_current_left_shift_q),
        .i_reset_to_zero(srnn_reset_to_zero_q),
        .i_version(srnn_version_q),
        .i_event_valid(i_srnn_event_valid &&
            owner_state_q == OWNER_STATE_SRNN),
        .o_event_ready(srnn_frontend_event_ready),
        .i_event_addr(i_srnn_event_addr),
        .i_event_data(i_srnn_event_data),
        .i_timestep_last(i_srnn_timestep_last),
        .i_sample_last(i_srnn_sample_last),
        .o_weight_req_valid(o_srnn_weight_req_valid),
        .i_weight_req_ready(i_srnn_weight_req_ready),
        .o_weight_req_version(o_srnn_weight_req_version),
        .o_weight_req_id(o_srnn_weight_req_id),
        .o_weight_req_recurrent(o_srnn_weight_req_recurrent),
        .o_weight_req_row_base(o_srnn_weight_req_row_base),
        .o_weight_req_destination_base(o_srnn_weight_req_destination_base),
        .o_weight_req_engine_mask(o_srnn_weight_req_engine_mask),
        .o_weight_req_lane_mask(o_srnn_weight_req_lane_mask),
        .i_weight_rsp_valid(i_srnn_weight_rsp_valid),
        .o_weight_rsp_ready(o_srnn_weight_rsp_ready),
        .i_weight_rsp_version(i_srnn_weight_rsp_version),
        .i_weight_rsp_id(i_srnn_weight_rsp_id),
        .i_weight_rsp_destination_base(i_srnn_weight_rsp_destination_base),
        .i_weight_rsp_engine_mask(i_srnn_weight_rsp_engine_mask),
        .i_weight_rsp_flat(i_srnn_weight_rsp_flat),
        .o_array_valid(srnn_frontend_array_valid),
        .i_array_ready(srnn_frontend_array_ready),
        .o_array_engine_mask(srnn_frontend_array_engine_mask),
        .o_array_lane_mask(srnn_frontend_array_lane_mask),
        .o_array_int8_lhs_flat(srnn_frontend_array_lhs_flat),
        .o_array_int8_rhs_flat(srnn_frontend_array_rhs_flat),
        .o_array_first(srnn_frontend_array_first),
        .o_array_last(srnn_frontend_array_last),
        .o_array_local_word(srnn_frontend_array_local_word),
        .i_array_result_valid(array_result_valid &&
            owner_state_q == OWNER_STATE_SRNN),
        .o_array_result_ready(srnn_frontend_array_result_ready),
        .i_array_result_owner(array_result_owner),
        .i_array_result_mode_q8_8(array_result_mode_q8_8),
        .i_array_result_engine_mask(array_result_engine_mask),
        .i_array_result_lane_mask(array_result_lane_mask),
        .o_state_clear_spikes(srnn_state_clear_spikes),
        .o_state_clear_word_valid(srnn_state_clear_word_valid),
        .o_state_clear_local_word(srnn_state_clear_local_word),
        .o_state_active_neurons(srnn_state_active_neurons),
        .o_state_lif_begin(srnn_state_lif_begin),
        .o_state_lif_word_valid(srnn_state_lif_word_valid),
        .o_state_lif_local_word(srnn_state_lif_local_word),
        .o_state_leak_q15(srnn_state_leak_q15),
        .o_state_threshold(srnn_state_threshold),
        .o_state_current_left_shift(srnn_state_current_left_shift),
        .o_state_reset_to_zero(srnn_state_reset_to_zero),
        .o_state_commit_timestep(srnn_state_commit_timestep),
        .i_state_previous_spikes(array_previous_spikes),
        .i_state_current_spikes(array_current_spikes),
        .o_timestep_valid(srnn_timestep_valid),
        .i_timestep_ready(srnn_timestep_ready),
        .o_timestep_version(srnn_timestep_version),
        .o_timestep_sample_last(srnn_timestep_sample_last),
        .o_timestep_spikes(srnn_timestep_spikes),
        .o_previous_spikes(srnn_frontend_previous_spikes),
        .o_busy(srnn_frontend_busy),
        .o_idle(srnn_frontend_idle),
        .o_sticky_error(srnn_frontend_sticky_error),
        .o_error_code(srnn_frontend_error_code)
    );

    unified_scnn_external_side_frontend_soft_abort_v1 #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .FEATURE_DEPTH(FEATURE_DEPTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_scnn_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_owner_enable(owner_state_q == OWNER_STATE_SCNN &&
            !scnn_external_mode_q),
        .i_clear_error(i_clear_error),
        .i_start(scnn_frontend_start_fire),
        .o_start_ready(scnn_frontend_start_ready),
        .i_in_width(scnn_in_width_q),
        .i_in_height(scnn_in_height_q),
        .i_total_in_channels(scnn_total_in_channels_q),
        .i_out_width(scnn_out_width_q),
        .i_out_height(scnn_out_height_q),
        .i_total_out_channels(scnn_total_out_channels_q),
        .i_input_channel_base(scnn_input_channel_base_q),
        .i_input_channel_count(scnn_input_channel_count_q),
        .i_output_channel_base(scnn_output_channel_base_q),
        .i_output_channel_count(scnn_output_channel_count_q),
        .i_stride(scnn_stride_q),
        .i_padding(scnn_padding_q),
        .i_input_slot(scnn_input_slot_q),
        .i_feature_base(scnn_feature_base_q),
        .i_weight_q16_base(scnn_weight_q16_base_q),
        .i_output_base(scnn_output_base_q),
        .i_unit_spike_features(scnn_unit_spike_features_q),
        .o_feature_read_valid(o_scnn_feature_read_valid),
        .i_feature_read_ready(i_scnn_feature_read_ready),
        .o_feature_read_slot(o_scnn_feature_read_slot),
        .o_feature_read_bank_flat(o_scnn_feature_read_bank_flat),
        .o_feature_read_addr_flat(o_scnn_feature_read_addr_flat),
        .o_feature_read_lane_mask(o_scnn_feature_read_lane_mask),
        .i_feature_valid(i_scnn_feature_valid),
        .o_feature_ready(o_scnn_feature_ready),
        .i_feature_flat(i_scnn_feature_flat),
        .i_feature_lane_mask(i_scnn_feature_lane_mask),
        .o_weight_read_valid(o_scnn_weight_read_valid),
        .i_weight_read_ready(i_scnn_weight_read_ready),
        .o_weight_read_addr0(o_scnn_weight_read_addr0),
        .o_weight_read_addr1(o_scnn_weight_read_addr1),
        .i_weight_result_valid(i_scnn_weight_result_valid),
        .o_weight_result_ready(o_scnn_weight_result_ready),
        .i_weight_data0(i_scnn_weight_data0),
        .i_weight_data1(i_scnn_weight_data1),
        .o_side_request_valid(scnn_side_request_valid),
        .i_side_request_ready(scnn_side_request_ready),
        .o_side_engine_mask(scnn_side_engine_mask),
        .o_side_feature_flat(scnn_side_feature_flat),
        .o_side_weight_flat(scnn_side_weight_flat),
        .o_side_lane_mask(scnn_side_lane_mask),
        .o_side_first(scnn_side_first),
        .o_side_last(scnn_side_last),
        .o_side_output_channel_base(scnn_side_output_channel_base),
        .i_side_result_valid(scnn_side_result_valid),
        .o_side_result_ready(scnn_side_result_ready),
        .i_side_result_mask(array_result_engine_mask),
        .i_side_result_accumulator_flat(array_result_q8_accumulator_flat),
        .i_side_busy(owner_state_q == OWNER_STATE_SCNN &&
            !array_owner_release_ready),
        .i_side_idle(array_owner_release_ready && array_owner_active &&
            array_active_owner == OWNER_SCNN),
        .i_side_sticky_error(array_sticky_error),
        .o_result_valid(scnn_frontend_result_valid),
        .i_result_ready(scnn_frontend_result_ready),
        .o_result_mask(scnn_frontend_result_mask),
        .o_result_output_addr_flat(scnn_frontend_result_output_addr_flat),
        .o_result_spatial_addr(scnn_frontend_result_spatial_addr),
        .o_result_accumulator_flat(scnn_frontend_result_accumulator_flat),
        .o_busy(scnn_frontend_busy),
        .o_done(scnn_frontend_done),
        .o_config_error(scnn_frontend_config_error),
        .o_protocol_error(scnn_frontend_protocol_error)
    );

    unified_h40_shared_inference_array_v1 #(
        .TILES(ENGINES),
        .FC_TILES(FC_TILES),
        .H_PER_TILE(TILE_NEURONS),
        .LANES(LANES),
        .K8_ACC_BITS(ACC_WIDTH),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .INCLUDE_FC_COMPAT(INCLUDE_FC_COMPAT),
        .USE_SHARED_FC_SRNN_STATE(USE_SHARED_FC_SRNN_STATE)
    ) u_shared_array (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_owner_valid(array_owner_valid),
        .o_owner_ready(array_owner_ready),
        .i_owner(array_owner),
        .i_owner_release_valid(array_owner_release_valid),
        .o_owner_release_ready(array_owner_release_ready),
        .o_owner_active(array_owner_active),
        .o_active_owner(array_active_owner),
        .i_fc_clear_state(i_fc_clear_state &&
            owner_state_q == OWNER_STATE_FC),
        .i_fc_active_tile_mask(i_fc_active_tile_mask),
        .i_fc_step_start(i_fc_step_start &&
            owner_state_q == OWNER_STATE_FC),
        .o_fc_step_ready(array_fc_step_ready),
        .i_fc_timestep(i_fc_timestep),
        .i_fc_row_valid(i_fc_row_valid &&
            owner_state_q == OWNER_STATE_FC),
        .o_fc_row_ready(array_fc_row_ready),
        .i_fc_lane_valid(i_fc_lane_valid),
        .i_fc_w0_weight_rows_flat(i_fc_w0_weight_rows_flat),
        .i_fc_step_close(i_fc_step_close &&
            owner_state_q == OWNER_STATE_FC),
        .i_fc_w2_weight_flat(i_fc_w2_weight_flat),
        .i_k8_valid(array_valid),
        .o_k8_ready(array_ready),
        .i_k8_engine_mask(array_engine_mask),
        .i_k8_lane_mask(array_lane_mask),
        .i_k8_q8_8_feature_flat(array_q8_feature_flat),
        .i_k8_q8_8_weight_flat(array_q8_weight_flat),
        .i_k8_int8_lhs_flat(array_int8_lhs_flat),
        .i_k8_int8_rhs_flat(array_int8_rhs_flat),
        .i_k8_first(array_first),
        .i_k8_last(array_last),
        .i_k8_local_word(array_local_word),
        .i_k8_output_channel_base(array_output_channel_base),
        .i_srnn_clear_spikes(owner_state_q == OWNER_STATE_SRNN &&
            (srnn_reset_only_q ?
                (srnn_reset_direct_active_q && srnn_reset_word_q == 0) :
                srnn_state_clear_spikes)),
        .i_srnn_clear_word_valid(owner_state_q == OWNER_STATE_SRNN &&
            (srnn_reset_only_q ? srnn_reset_direct_active_q :
                srnn_state_clear_word_valid)),
        .i_srnn_clear_local_word(srnn_reset_only_q ? srnn_reset_word_q :
            srnn_state_clear_local_word),
        .i_srnn_active_neurons(srnn_reset_only_q ? srnn_active_neurons_q :
            srnn_state_active_neurons),
        .i_srnn_lif_begin(srnn_state_lif_begin &&
            owner_state_q == OWNER_STATE_SRNN),
        .i_srnn_lif_word_valid(srnn_state_lif_word_valid &&
            owner_state_q == OWNER_STATE_SRNN),
        .i_srnn_lif_local_word(srnn_state_lif_local_word),
        .i_srnn_leak_q15(srnn_state_leak_q15),
        .i_srnn_threshold(srnn_state_threshold),
        .i_srnn_current_left_shift(srnn_state_current_left_shift),
        .i_srnn_reset_to_zero(srnn_state_reset_to_zero),
        .i_srnn_commit_timestep(srnn_state_commit_timestep &&
            owner_state_q == OWNER_STATE_SRNN),
        .o_result_valid(array_result_valid),
        .i_result_ready(array_result_ready),
        .o_result_owner(array_result_owner),
        .o_result_mode_q8_8(array_result_mode_q8_8),
        .o_result_engine_mask(array_result_engine_mask),
        .o_result_lane_mask(array_result_lane_mask),
        .o_result_channel_flat(array_result_channel_flat),
        .o_result_k8_q8_8_accumulator_flat(array_result_q8_accumulator_flat),
        .o_result_k8_int8_contribution_flat(
            array_result_int8_contribution_flat),
        .o_result_fc_timestep(array_result_fc_timestep),
        .o_result_fc_events(array_result_fc_events),
        .o_result_fc_spikes(array_result_fc_spikes),
        .o_result_fc_tile_partials_flat(),
        .o_result_fc_logits_flat(array_result_fc_logits_flat),
        .o_result_fc_pred(array_result_fc_pred),
        .o_fc_vmem_flat(o_fc_vmem_flat),
        .o_fc_trace_flat(o_fc_trace_flat),
        .o_srnn_previous_spikes(array_previous_spikes),
        .o_srnn_current_spikes(array_current_spikes),
        .o_srnn_accumulator_flat(),
        .o_srnn_membrane_flat(),
        .o_busy(array_busy),
        .o_idle(array_idle),
        .o_sticky_error(array_sticky_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_state_q <= OWNER_STATE_IDLE;
            local_sticky_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
            srnn_start_pending_q <= 1'b0;
            srnn_done_seen_q <= 1'b0;
            srnn_reset_only_q <= 1'b0;
            srnn_reset_direct_active_q <= 1'b0;
            srnn_reset_word_q <= '0;
            scnn_start_pending_q <= 1'b0;
            scnn_done_seen_q <= 1'b0;
            scnn_result_enqueued_q <= 1'b0;
            scnn_external_mode_q <= 1'b0;
            o_fc_done <= 1'b0;
            o_srnn_done <= 1'b0;
            o_scnn_done <= 1'b0;
            srnn_result_valid_q <= 1'b0;
            srnn_result_version_q <= '0;
            srnn_result_sample_last_q <= 1'b0;
            srnn_result_spikes_q <= '0;
            scnn_result_valid_q <= 1'b0;
            scnn_result_mask_q <= '0;
            scnn_result_output_addr_flat_q <= '0;
            scnn_result_spatial_addr_q <= '0;
            scnn_result_accumulator_flat_q <= '0;
            srnn_input_dim_q <= '0;
            srnn_active_neurons_q <= '0;
            srnn_w0_base_q <= '0;
            srnn_recurrent_base_q <= '0;
            srnn_recurrent_enable_q <= 1'b0;
            srnn_leak_q15_q <= '0;
            srnn_threshold_q <= '0;
            srnn_current_left_shift_q <= '0;
            srnn_reset_to_zero_q <= 1'b0;
            srnn_version_q <= '0;
            scnn_in_width_q <= '0;
            scnn_in_height_q <= '0;
            scnn_total_in_channels_q <= '0;
            scnn_out_width_q <= '0;
            scnn_out_height_q <= '0;
            scnn_total_out_channels_q <= '0;
            scnn_input_channel_base_q <= '0;
            scnn_input_channel_count_q <= '0;
            scnn_output_channel_base_q <= '0;
            scnn_output_channel_count_q <= '0;
            scnn_stride_q <= '0;
            scnn_padding_q <= '0;
            scnn_input_slot_q <= 1'b0;
            scnn_feature_base_q <= '0;
            scnn_weight_q16_base_q <= '0;
            scnn_output_base_q <= '0;
            scnn_unit_spike_features_q <= 1'b0;
        end else if (i_abort) begin
            owner_state_q <= OWNER_STATE_IDLE;
            srnn_start_pending_q <= 1'b0;
            srnn_done_seen_q <= 1'b0;
            srnn_reset_only_q <= 1'b0;
            srnn_reset_direct_active_q <= 1'b0;
            srnn_reset_word_q <= '0;
            scnn_start_pending_q <= 1'b0;
            scnn_done_seen_q <= 1'b0;
            scnn_result_enqueued_q <= 1'b0;
            scnn_external_mode_q <= 1'b0;
            o_fc_done <= 1'b0;
            o_srnn_done <= 1'b0;
            o_scnn_done <= 1'b0;
            srnn_result_valid_q <= 1'b0;
            srnn_result_version_q <= '0;
            srnn_result_sample_last_q <= 1'b0;
            srnn_result_spikes_q <= '0;
            scnn_result_valid_q <= 1'b0;
            scnn_result_mask_q <= '0;
            scnn_result_output_addr_flat_q <= '0;
            scnn_result_spatial_addr_q <= '0;
            scnn_result_accumulator_flat_q <= '0;
            srnn_input_dim_q <= '0;
            srnn_active_neurons_q <= '0;
            srnn_w0_base_q <= '0;
            srnn_recurrent_base_q <= '0;
            srnn_recurrent_enable_q <= 1'b0;
            srnn_leak_q15_q <= '0;
            srnn_threshold_q <= '0;
            srnn_current_left_shift_q <= '0;
            srnn_reset_to_zero_q <= 1'b0;
            srnn_version_q <= '0;
            scnn_in_width_q <= '0;
            scnn_in_height_q <= '0;
            scnn_total_in_channels_q <= '0;
            scnn_out_width_q <= '0;
            scnn_out_height_q <= '0;
            scnn_total_out_channels_q <= '0;
            scnn_input_channel_base_q <= '0;
            scnn_input_channel_count_q <= '0;
            scnn_output_channel_base_q <= '0;
            scnn_output_channel_count_q <= '0;
            scnn_stride_q <= '0;
            scnn_padding_q <= '0;
            scnn_input_slot_q <= 1'b0;
            scnn_feature_base_q <= '0;
            scnn_weight_q16_base_q <= '0;
            scnn_output_base_q <= '0;
            scnn_unit_spike_features_q <= 1'b0;
        end else begin
            o_fc_done <= 1'b0;
            o_srnn_done <= 1'b0;
            o_scnn_done <= 1'b0;

            if (i_clear_error && operational_idle) begin
                local_sticky_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (owner_state_q == OWNER_STATE_IDLE &&
                ((i_fc_start_valid && i_srnn_start_valid) ||
                 (i_fc_start_valid && i_scnn_start_valid) ||
                 (i_srnn_start_valid && i_scnn_start_valid))) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_OWNER_CONFLICT;
            end else if (owner_state_q == OWNER_STATE_IDLE &&
                         i_srnn_start_valid && !srnn_shape_legal) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_WEIGHT_PROTOCOL;
            end else if (owner_state_q == OWNER_STATE_IDLE &&
                         i_srnn_state_reset && !srnn_shape_legal) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_WEIGHT_PROTOCOL;
            end else if ((owner_state_q == OWNER_STATE_FC_ACQUIRE ||
                         owner_state_q == OWNER_STATE_FC ||
                         owner_state_q == OWNER_STATE_FC_RELEASE) &&
                        (i_srnn_start_valid || i_scnn_start_valid)) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_MIXED_OWNER;
            end else if ((owner_state_q == OWNER_STATE_SRNN_ACQUIRE ||
                         owner_state_q == OWNER_STATE_SRNN ||
                         owner_state_q == OWNER_STATE_SRNN_RELEASE) &&
                        (i_fc_start_valid || i_scnn_start_valid)) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_MIXED_OWNER;
            end else if ((owner_state_q == OWNER_STATE_SCNN_ACQUIRE ||
                         owner_state_q == OWNER_STATE_SCNN ||
                         owner_state_q == OWNER_STATE_SCNN_RELEASE) &&
                        (i_fc_start_valid || i_srnn_start_valid)) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_MIXED_OWNER;
            end else if (array_result_valid &&
                ((owner_state_q == OWNER_STATE_FC &&
                  array_result_owner != OWNER_FC) ||
                 (owner_state_q == OWNER_STATE_SRNN &&
                  (array_result_owner != OWNER_SRNN ||
                   array_result_mode_q8_8)) ||
                 (owner_state_q == OWNER_STATE_SCNN &&
                  (array_result_owner != OWNER_SCNN ||
                   !array_result_mode_q8_8)))) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_ARRAY_OWNER;
            end else if (scnn_frontend_done && !scnn_external_mode_q &&
                         !(scnn_result_enqueued_q || scnn_result_push)) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_SCNN_DONE_ORDER;
            end else if (i_scnn_external_done &&
                         (owner_state_q != OWNER_STATE_SCNN ||
                          !scnn_external_mode_q)) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= ERROR_SCNN_DONE_ORDER;
            end

            if (srnn_result_push) begin
                srnn_result_valid_q <= 1'b1;
                srnn_result_version_q <= srnn_timestep_version;
                srnn_result_sample_last_q <= srnn_timestep_sample_last;
                srnn_result_spikes_q <=
                    srnn_timestep_spikes[HIDDEN_MAX-1:0];
                if (srnn_timestep_sample_last) begin
                    srnn_done_seen_q <= 1'b1;
                    o_srnn_done <= 1'b1;
                end
            end else if (srnn_result_pop) begin
                srnn_result_valid_q <= 1'b0;
            end

            if (scnn_result_push) begin
                scnn_result_valid_q <= 1'b1;
                scnn_result_mask_q <= scnn_frontend_result_mask;
                scnn_result_output_addr_flat_q <=
                    scnn_frontend_result_output_addr_flat;
                scnn_result_spatial_addr_q <=
                    scnn_frontend_result_spatial_addr;
                scnn_result_accumulator_flat_q <=
                    scnn_frontend_result_accumulator_flat;
                scnn_result_enqueued_q <= 1'b1;
            end else if (scnn_result_pop) begin
                scnn_result_valid_q <= 1'b0;
            end

            if (scnn_frontend_done &&
                (scnn_result_enqueued_q || scnn_result_push)) begin
                scnn_done_seen_q <= 1'b1;
                o_scnn_done <= 1'b1;
            end
            if (i_scnn_external_done && scnn_external_mode_q &&
                owner_state_q == OWNER_STATE_SCNN) begin
                scnn_done_seen_q <= 1'b1;
                o_scnn_done <= 1'b1;
            end

            if (srnn_frontend_start_fire)
                srnn_start_pending_q <= 1'b0;
            if (scnn_frontend_start_fire)
                scnn_start_pending_q <= 1'b0;

            if (owner_state_q == OWNER_STATE_SRNN && srnn_reset_only_q &&
                srnn_reset_direct_active_q) begin
                if (srnn_reset_word_q == 3'd4)
                    srnn_reset_direct_active_q <= 1'b0;
                else
                    srnn_reset_word_q <= srnn_reset_word_q + 1'b1;
            end

            unique case (owner_state_q)
                OWNER_STATE_IDLE: begin
                    srnn_done_seen_q <= 1'b0;
                    scnn_done_seen_q <= 1'b0;
                    scnn_result_enqueued_q <= 1'b0;
                    if (i_srnn_state_reset && !i_fc_start_valid &&
                        !i_srnn_start_valid && !i_scnn_start_valid &&
                        srnn_shape_legal && !o_sticky_error &&
                        array_owner_ready) begin
                        owner_state_q <= OWNER_STATE_SRNN_ACQUIRE;
                        srnn_start_pending_q <= 1'b0;
                        srnn_reset_only_q <= 1'b1;
                        srnn_active_neurons_q <= i_srnn_active_neurons;
                    end else if (i_fc_start_valid && o_fc_start_ready) begin
                        owner_state_q <= OWNER_STATE_FC_ACQUIRE;
                    end else if (i_srnn_start_valid && o_srnn_start_ready) begin
                        owner_state_q <= OWNER_STATE_SRNN_ACQUIRE;
                        srnn_start_pending_q <= 1'b1;
                        srnn_input_dim_q <= i_srnn_input_dim;
                        srnn_active_neurons_q <= i_srnn_active_neurons;
                        srnn_w0_base_q <= i_srnn_w0_base;
                        srnn_recurrent_base_q <= i_srnn_recurrent_base;
                        srnn_recurrent_enable_q <=
                            i_srnn_recurrent_enable;
                        srnn_leak_q15_q <= i_srnn_leak_q15;
                        srnn_threshold_q <= i_srnn_threshold;
                        srnn_current_left_shift_q <=
                            i_srnn_current_left_shift;
                        srnn_reset_to_zero_q <= i_srnn_reset_to_zero;
                        srnn_version_q <= i_srnn_version;
                    end else if (i_scnn_start_valid && o_scnn_start_ready) begin
                        owner_state_q <= OWNER_STATE_SCNN_ACQUIRE;
                        scnn_start_pending_q <= !i_scnn_external_mode;
                        scnn_external_mode_q <= i_scnn_external_mode;
                        scnn_in_width_q <= i_scnn_in_width;
                        scnn_in_height_q <= i_scnn_in_height;
                        scnn_total_in_channels_q <=
                            i_scnn_total_in_channels;
                        scnn_out_width_q <= i_scnn_out_width;
                        scnn_out_height_q <= i_scnn_out_height;
                        scnn_total_out_channels_q <=
                            i_scnn_total_out_channels;
                        scnn_input_channel_base_q <=
                            i_scnn_input_channel_base;
                        scnn_input_channel_count_q <=
                            i_scnn_input_channel_count;
                        scnn_output_channel_base_q <=
                            i_scnn_output_channel_base;
                        scnn_output_channel_count_q <=
                            i_scnn_output_channel_count;
                        scnn_stride_q <= i_scnn_stride;
                        scnn_padding_q <= i_scnn_padding;
                        scnn_input_slot_q <= i_scnn_input_slot;
                        scnn_feature_base_q <= i_scnn_feature_base;
                        scnn_weight_q16_base_q <= i_scnn_weight_q16_base;
                        scnn_output_base_q <= i_scnn_output_base;
                        scnn_unit_spike_features_q <=
                            i_scnn_unit_spike_features;
                    end
                end
                OWNER_STATE_FC_ACQUIRE: begin
                    if (array_owner_valid && array_owner_ready)
                        owner_state_q <= OWNER_STATE_FC;
                end
                OWNER_STATE_FC: begin
                    if (i_fc_finish && o_fc_finish_ready)
                        owner_state_q <= OWNER_STATE_FC_RELEASE;
                end
                OWNER_STATE_FC_RELEASE: begin
                    if (array_owner_release_valid &&
                        array_owner_release_ready) begin
                        owner_state_q <= OWNER_STATE_IDLE;
                        o_fc_done <= 1'b1;
                    end
                end
                OWNER_STATE_SRNN_ACQUIRE: begin
                    if (array_owner_valid && array_owner_ready) begin
                        owner_state_q <= OWNER_STATE_SRNN;
                        if (srnn_reset_only_q) begin
                            srnn_reset_direct_active_q <= 1'b1;
                            srnn_reset_word_q <= '0;
                        end
                    end
                end
                OWNER_STATE_SRNN: begin
                    if (((srnn_reset_only_q &&
                          !srnn_reset_direct_active_q) ||
                         (!srnn_reset_only_q && srnn_done_seen_q &&
                          !srnn_result_valid_q)) &&
                        srnn_frontend_idle && array_owner_release_ready)
                        owner_state_q <= OWNER_STATE_SRNN_RELEASE;
                end
                OWNER_STATE_SRNN_RELEASE: begin
                    if (array_owner_release_valid &&
                        array_owner_release_ready) begin
                        owner_state_q <= OWNER_STATE_IDLE;
                        srnn_start_pending_q <= 1'b0;
                        srnn_reset_only_q <= 1'b0;
                        srnn_reset_direct_active_q <= 1'b0;
                    end
                end
                OWNER_STATE_SCNN_ACQUIRE: begin
                    if (array_owner_valid && array_owner_ready)
                        owner_state_q <= OWNER_STATE_SCNN;
                end
                OWNER_STATE_SCNN: begin
                    if ((scnn_done_seen_q ||
                         (!scnn_external_mode_q &&
                          scnn_frontend_config_error)) &&
                        (scnn_external_mode_q || !scnn_frontend_busy) &&
                        (scnn_external_mode_q || !scnn_result_valid_q) &&
                        array_owner_release_ready)
                        owner_state_q <= OWNER_STATE_SCNN_RELEASE;
                end
                OWNER_STATE_SCNN_RELEASE: begin
                    if (array_owner_release_valid &&
                        array_owner_release_ready) begin
                        owner_state_q <= OWNER_STATE_IDLE;
                        scnn_start_pending_q <= 1'b0;
                        scnn_external_mode_q <= 1'b0;
                    end
                end
                default: owner_state_q <= OWNER_STATE_IDLE;
            endcase
        end
    end

    initial begin
        if (ENGINES != 12 || LANES != 8)
            $error("tile-major shared leaf requires 12 H40/K8 engines");
        if (HIDDEN_MAX < 1 || HIDDEN_MAX > PHYSICAL_HIDDEN)
            $error("logical HIDDEN_MAX must fit the physical H40 array");
        if (ACC_WIDTH != 48)
            $error("SCNN Q8.8 accumulation requires ACC_WIDTH=48");
    end
endmodule

`default_nettype wire
