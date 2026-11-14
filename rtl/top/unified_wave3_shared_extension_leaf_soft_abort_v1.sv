`default_nettype none

// Wave3 integration leaf. SRNN and SCNN retain their established control
// semantics while sharing exactly one twelve-engine K8 numeric array.
module unified_wave3_shared_extension_leaf_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int HIDDEN_MAX = 480,
    parameter int INPUT_MAX = 2312,
    parameter int VERSION_WIDTH = 16,
    parameter int REQUEST_ID_WIDTH = 16,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int Q16_ADDR_WIDTH = Q64_ADDR_WIDTH + 2,
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_abort,

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

    // Optional SCNN-OUTPUT request source. It acquires the same SCNN owner as
    // the convolution frontend and therefore cannot create a second array.
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

    output logic [1:0]                                o_active_owner,
    output logic                                      o_shared_request_fire,
    output logic                                      o_shared_result_fire,
    output logic                                      o_scnn_zero_lane_normalized,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_SRNN = 2'd1;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER_CONFLICT = 8'h01;
    localparam logic [7:0] ERROR_MIXED_OWNER = 8'h02;
    localparam logic [7:0] ERROR_WEIGHT_PROTOCOL = 8'h03;
    localparam logic [7:0] ERROR_ARRAY_OWNER = 8'h04;
    localparam logic [7:0] ERROR_SCNN_DONE_ORDER = 8'h05;
    localparam int RESPONSE_ENGINE_WIDTH =
        ENGINES <= 1 ? 1 : $clog2(ENGINES);

    typedef enum logic [2:0] {
        OWNER_STATE_IDLE,
        OWNER_STATE_SRNN,
        OWNER_STATE_SRNN_RELEASE,
        OWNER_STATE_SCNN,
        OWNER_STATE_SCNN_RELEASE
    } owner_state_t;

    typedef enum logic [2:0] {
        ROW_STATE_IDLE,
        ROW_STATE_WEIGHT_REQUEST,
        ROW_STATE_WEIGHT_RESPONSE,
        ROW_STATE_ARRAY_REQUEST,
        ROW_STATE_ARRAY_RESULT,
        ROW_STATE_SERIAL_RESPONSE
    } row_state_t;

    owner_state_t owner_state_q;
    row_state_t row_state_q;

    logic local_sticky_error_q;
    logic [7:0] local_error_code_q;
    logic local_error_set;
    logic [7:0] local_error_set_code;
    logic operational_idle;

    logic array_owner_valid;
    logic array_owner_ready;
    logic [1:0] array_owner;
    logic array_mode_q8_8;
    logic array_owner_release_valid;
    logic array_owner_release_ready;
    logic array_owner_active;
    logic [1:0] array_active_owner;
    logic array_active_mode_q8_8;
    logic array_valid;
    logic array_ready;
    logic [ENGINES-1:0] array_engine_mask;
    logic [LANES-1:0] array_lane_mask;
    logic [LANES*16-1:0] array_q8_8_feature_flat;
    logic [ENGINES*LANES*16-1:0] array_q8_8_weight_flat;
    logic [LANES*8-1:0] array_int8_lhs_flat;
    logic [ENGINES*LANES*8-1:0] array_int8_rhs_flat;
    logic array_first;
    logic array_last;
    logic [CHANNEL_WIDTH-1:0] array_output_channel_base;
    logic array_result_valid;
    logic array_result_ready;
    logic [1:0] array_result_owner;
    logic array_result_mode_q8_8;
    logic [ENGINES-1:0] array_result_engine_mask;
    logic [LANES-1:0] array_result_lane_mask;
    logic [ENGINES*CHANNEL_WIDTH-1:0] array_result_channel_flat;
    logic [ENGINES*ACC_WIDTH-1:0] array_result_q8_8_accumulator_flat;
    logic [ENGINES*LANES*32-1:0] array_result_int8_contribution_flat;
    logic array_busy;
    logic array_idle;
    logic array_sticky_error;

    logic srnn_start_pending_q;
    logic srnn_done_seen_q;
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
    logic srnn_sidecar_start_ready;
    logic srnn_sidecar_start_fire;
    logic srnn_sidecar_req_valid;
    logic srnn_sidecar_req_ready;
    logic [VERSION_WIDTH-1:0] srnn_sidecar_req_version;
    logic [REQUEST_ID_WIDTH-1:0] srnn_sidecar_req_id;
    logic srnn_sidecar_req_recurrent;
    logic [31:0] srnn_sidecar_req_row_base;
    logic [9:0] srnn_sidecar_req_destination_count;
    logic signed [7:0] srnn_sidecar_req_activation;
    logic [2:0] srnn_sidecar_req_numeric_profile;
    logic srnn_sidecar_rsp_valid;
    logic srnn_sidecar_rsp_ready;
    logic [VERSION_WIDTH-1:0] srnn_sidecar_rsp_version;
    logic [REQUEST_ID_WIDTH-1:0] srnn_sidecar_rsp_id;
    logic [9:0] srnn_sidecar_rsp_destination_base;
    logic [LANES-1:0] srnn_sidecar_rsp_lane_mask;
    logic [LANES*32-1:0] srnn_sidecar_rsp_contribution_flat;
    logic srnn_sidecar_rsp_last;
    logic srnn_timestep_valid;
    logic srnn_timestep_ready;
    logic [VERSION_WIDTH-1:0] srnn_timestep_version;
    logic srnn_timestep_sample_last;
    logic [HIDDEN_MAX-1:0] srnn_timestep_spikes;
    logic [HIDDEN_MAX-1:0] srnn_previous_spikes;
    logic srnn_sidecar_busy;
    logic srnn_sidecar_idle;
    logic srnn_sidecar_sticky_error;
    logic [7:0] srnn_sidecar_error_code;

    logic [VERSION_WIDTH-1:0] row_version_q;
    logic [REQUEST_ID_WIDTH-1:0] row_id_q;
    logic row_recurrent_q;
    logic [31:0] row_base_q;
    logic [9:0] row_destination_count_q;
    logic signed [7:0] row_activation_q;
    logic [9:0] row_wave_base_q;
    logic [ENGINES*LANES*8-1:0] row_weight_flat_q;
    logic [ENGINES*LANES*32-1:0] row_contribution_flat_q;
    logic [ENGINES-1:0] row_result_engine_mask_q;
    logic [RESPONSE_ENGINE_WIDTH-1:0] row_response_engine_q;
    logic [ENGINES-1:0] row_expected_engine_mask;
    logic row_next_engine_valid;
    logic [9:0] row_response_destination_base;
    logic [LANES-1:0] row_response_lane_mask;
    logic row_response_last;
    logic row_weight_response_legal;

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
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]
        scnn_frontend_result_output_addr_flat;
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

    function automatic logic [ENGINES-1:0] engine_mask_for(
        input logic [9:0] destination_base,
        input logic [9:0] destination_count
    );
        begin
            engine_mask_for = '0;
            for (integer engine = 0; engine < ENGINES; engine++)
                if (integer'(destination_base) + engine*LANES <
                    integer'(destination_count))
                    engine_mask_for[engine] = 1'b1;
        end
    endfunction

    function automatic logic [LANES-1:0] lane_mask_for(
        input logic [9:0] destination_base,
        input logic [9:0] destination_count
    );
        begin
            lane_mask_for = '0;
            for (integer lane = 0; lane < LANES; lane++)
                if (integer'(destination_base) + lane <
                    integer'(destination_count))
                    lane_mask_for[lane] = 1'b1;
        end
    endfunction

    assign operational_idle = owner_state_q == OWNER_STATE_IDLE &&
        row_state_q == ROW_STATE_IDLE && !srnn_result_valid_q &&
        !scnn_result_valid_q && srnn_sidecar_idle &&
        !scnn_frontend_busy && array_idle;
    assign o_idle = operational_idle;
    assign o_busy = !operational_idle;

    always_comb begin
        unique case (owner_state_q)
            OWNER_STATE_SRNN,
            OWNER_STATE_SRNN_RELEASE: o_active_owner = OWNER_SRNN;
            OWNER_STATE_SCNN,
            OWNER_STATE_SCNN_RELEASE: o_active_owner = OWNER_SCNN;
            default: o_active_owner = OWNER_IDLE;
        endcase
    end

    assign o_sticky_error = local_sticky_error_q ||
        srnn_sidecar_sticky_error || scnn_frontend_config_error ||
        scnn_frontend_protocol_error || array_sticky_error ||
        (scnn_external_mode_q && i_scnn_external_sticky_error);
    always_comb begin
        if (local_sticky_error_q)
            o_error_code = local_error_code_q;
        else if (srnn_sidecar_sticky_error)
            o_error_code = 8'h40 | srnn_sidecar_error_code;
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

    assign array_owner_valid = !i_abort &&
        owner_state_q == OWNER_STATE_IDLE &&
        !o_sticky_error && (i_srnn_start_valid ^ i_scnn_start_valid);
    assign array_owner = i_srnn_start_valid ? OWNER_SRNN : OWNER_SCNN;
    assign array_mode_q8_8 = i_scnn_start_valid;
    assign o_srnn_start_ready = !i_abort &&
        owner_state_q == OWNER_STATE_IDLE &&
        !o_sticky_error && !i_scnn_start_valid && array_owner_ready;
    assign o_scnn_start_ready = !i_abort &&
        owner_state_q == OWNER_STATE_IDLE &&
        !o_sticky_error && !i_srnn_start_valid && array_owner_ready;
    assign array_owner_release_valid =
        owner_state_q == OWNER_STATE_SRNN_RELEASE ||
        owner_state_q == OWNER_STATE_SCNN_RELEASE;

    assign srnn_sidecar_start_fire = owner_state_q == OWNER_STATE_SRNN &&
        srnn_start_pending_q && srnn_sidecar_start_ready;
    assign scnn_frontend_start_fire = owner_state_q == OWNER_STATE_SCNN &&
        !scnn_external_mode_q && scnn_start_pending_q &&
        scnn_frontend_start_ready;

    assign o_srnn_event_ready = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        srnn_sidecar_event_ready;

    // The sidecar event-ready signal is kept separate to make owner gating
    // explicit at the wrapper boundary.
    logic srnn_sidecar_event_ready;

    assign srnn_sidecar_req_ready = owner_state_q == OWNER_STATE_SRNN &&
        row_state_q == ROW_STATE_IDLE;
    assign row_expected_engine_mask = engine_mask_for(
        row_wave_base_q, row_destination_count_q);
    assign o_srnn_weight_req_valid = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        row_state_q == ROW_STATE_WEIGHT_REQUEST;
    assign o_srnn_weight_req_version = row_version_q;
    assign o_srnn_weight_req_id = row_id_q;
    assign o_srnn_weight_req_recurrent = row_recurrent_q;
    assign o_srnn_weight_req_row_base = row_base_q;
    assign o_srnn_weight_req_destination_base = row_wave_base_q;
    assign o_srnn_weight_req_engine_mask = row_expected_engine_mask;
    assign o_srnn_weight_req_lane_mask = {LANES{1'b1}};
    assign o_srnn_weight_rsp_ready = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        row_state_q == ROW_STATE_WEIGHT_RESPONSE;
    assign row_weight_response_legal =
        i_srnn_weight_rsp_version == row_version_q &&
        i_srnn_weight_rsp_id == row_id_q &&
        i_srnn_weight_rsp_destination_base == row_wave_base_q &&
        i_srnn_weight_rsp_engine_mask == row_expected_engine_mask;

    assign row_response_destination_base = row_wave_base_q +
        10'(integer'(row_response_engine_q) * LANES);
    assign row_response_lane_mask = lane_mask_for(
        row_response_destination_base, row_destination_count_q);
    assign row_response_last = integer'(row_response_destination_base) +
        LANES >= integer'(row_destination_count_q);
    always_comb begin
        row_next_engine_valid = 1'b0;
        for (integer engine = 0; engine < ENGINES-1; engine++)
            if (integer'(row_response_engine_q) == engine)
                row_next_engine_valid =
                    row_result_engine_mask_q[engine + 1];
    end
    assign srnn_sidecar_rsp_valid = !i_abort &&
        owner_state_q == OWNER_STATE_SRNN &&
        row_state_q == ROW_STATE_SERIAL_RESPONSE;
    assign srnn_sidecar_rsp_version = row_version_q;
    assign srnn_sidecar_rsp_id = row_id_q;
    assign srnn_sidecar_rsp_destination_base = row_response_destination_base;
    assign srnn_sidecar_rsp_lane_mask = row_response_lane_mask;
    assign srnn_sidecar_rsp_contribution_flat = row_contribution_flat_q[
        integer'(row_response_engine_q)*LANES*32 +: LANES*32];
    assign srnn_sidecar_rsp_last = row_response_last;

    assign srnn_result_queue_ready = !srnn_result_valid_q ||
        i_srnn_result_ready;
    assign srnn_timestep_ready = owner_state_q == OWNER_STATE_SRNN &&
        srnn_result_queue_ready;
    assign srnn_result_push = srnn_timestep_valid && srnn_timestep_ready;
    assign srnn_result_pop = srnn_result_valid_q && i_srnn_result_ready;
    assign o_srnn_result_valid = !i_abort && srnn_result_valid_q;
    assign o_srnn_result_version = srnn_result_version_q;
    assign o_srnn_result_sample_last = srnn_result_sample_last_q;
    assign o_srnn_result_spikes = srnn_result_spikes_q;
    assign o_srnn_previous_spikes = srnn_previous_spikes;

    assign scnn_result_queue_ready = !scnn_result_valid_q ||
        i_scnn_result_ready;
    assign scnn_frontend_result_ready = !scnn_external_mode_q &&
        scnn_result_queue_ready;
    assign scnn_result_push = !scnn_external_mode_q &&
        scnn_frontend_result_valid &&
        scnn_frontend_result_ready;
    assign scnn_result_pop = scnn_result_valid_q && i_scnn_result_ready;
    assign o_scnn_result_valid = !i_abort && scnn_result_valid_q;
    assign o_scnn_result_mask = scnn_result_mask_q;
    assign o_scnn_result_output_addr_flat =
        scnn_result_output_addr_flat_q;
    assign o_scnn_result_spatial_addr = scnn_result_spatial_addr_q;
    assign o_scnn_result_accumulator_flat =
        scnn_result_accumulator_flat_q;

    always_comb begin
        array_valid = 1'b0;
        array_engine_mask = '0;
        array_lane_mask = '0;
        array_q8_8_feature_flat = '0;
        array_q8_8_weight_flat = '0;
        array_int8_lhs_flat = '0;
        array_int8_rhs_flat = '0;
        array_first = 1'b1;
        array_last = 1'b1;
        array_output_channel_base = '0;
        if (!i_abort && owner_state_q == OWNER_STATE_SRNN &&
            row_state_q == ROW_STATE_ARRAY_REQUEST) begin
            array_valid = 1'b1;
            array_engine_mask = row_expected_engine_mask;
            array_lane_mask = {LANES{1'b1}};
            for (integer lane = 0; lane < LANES; lane++)
                array_int8_lhs_flat[lane*8 +: 8] = row_activation_q;
            array_int8_rhs_flat = row_weight_flat_q;
            array_output_channel_base = CHANNEL_WIDTH'(row_wave_base_q);
        end else if (!i_abort && owner_state_q == OWNER_STATE_SCNN) begin
            if (scnn_external_mode_q) begin
                array_valid = i_scnn_external_request_valid;
                array_engine_mask = i_scnn_external_engine_mask;
                array_lane_mask = i_scnn_external_lane_mask == '0 ?
                    {{(LANES-1){1'b0}}, 1'b1} :
                    i_scnn_external_lane_mask;
                array_q8_8_feature_flat = i_scnn_external_feature_flat;
                array_q8_8_weight_flat = i_scnn_external_weight_flat;
                array_first = i_scnn_external_first;
                array_last = i_scnn_external_last;
                array_output_channel_base = CHANNEL_WIDTH'(
                    i_scnn_external_output_channel_base);
            end else begin
                array_valid = scnn_side_request_valid;
                array_engine_mask = scnn_side_engine_mask;
                array_lane_mask = scnn_side_lane_mask == '0 ?
                    {{(LANES-1){1'b0}}, 1'b1} : scnn_side_lane_mask;
                array_q8_8_feature_flat = scnn_side_feature_flat;
                array_q8_8_weight_flat = scnn_side_weight_flat;
                array_first = scnn_side_first;
                array_last = scnn_side_last;
                array_output_channel_base =
                    CHANNEL_WIDTH'(scnn_side_output_channel_base);
            end
        end
    end

    assign scnn_side_request_ready = !i_abort &&
        owner_state_q == OWNER_STATE_SCNN &&
        !scnn_external_mode_q && array_ready;
    assign o_scnn_external_request_ready =
        !i_abort && owner_state_q == OWNER_STATE_SCNN && scnn_external_mode_q &&
        array_ready;
    assign o_scnn_zero_lane_normalized =
        owner_state_q == OWNER_STATE_SCNN && array_valid && array_ready &&
        (scnn_external_mode_q ?
         i_scnn_external_lane_mask == '0 : scnn_side_lane_mask == '0);

    always_comb begin
        array_result_ready = 1'b0;
        if (!i_abort && owner_state_q == OWNER_STATE_SRNN &&
            row_state_q == ROW_STATE_ARRAY_RESULT)
            array_result_ready = 1'b1;
        else if (!i_abort && owner_state_q == OWNER_STATE_SCNN)
            array_result_ready = scnn_external_mode_q ?
                i_scnn_external_result_ready : scnn_side_result_ready;
    end
    assign scnn_side_result_valid = !i_abort &&
        owner_state_q == OWNER_STATE_SCNN &&
        !scnn_external_mode_q && array_result_valid;
    assign o_scnn_external_result_valid =
        !i_abort && owner_state_q == OWNER_STATE_SCNN && scnn_external_mode_q &&
        array_result_valid;
    assign o_scnn_external_result_mask = array_result_engine_mask;
    assign o_scnn_external_result_accumulator_flat =
        array_result_q8_8_accumulator_flat;
    assign o_shared_request_fire = array_valid && array_ready;
    assign o_shared_result_fire = array_result_valid && array_result_ready;

    always_comb begin
        local_error_set = 1'b0;
        local_error_set_code = ERROR_NONE;
        if (owner_state_q == OWNER_STATE_IDLE &&
            i_srnn_start_valid && i_scnn_start_valid) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_OWNER_CONFLICT;
        end else if ((owner_state_q == OWNER_STATE_SRNN ||
                      owner_state_q == OWNER_STATE_SRNN_RELEASE) &&
                     i_scnn_start_valid) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_MIXED_OWNER;
        end else if ((owner_state_q == OWNER_STATE_SCNN ||
                      owner_state_q == OWNER_STATE_SCNN_RELEASE) &&
                     i_srnn_start_valid) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_MIXED_OWNER;
        end else if (i_srnn_weight_rsp_valid &&
                     row_state_q != ROW_STATE_WEIGHT_RESPONSE) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_WEIGHT_PROTOCOL;
        end else if (i_srnn_weight_rsp_valid && o_srnn_weight_rsp_ready &&
                     !row_weight_response_legal) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_WEIGHT_PROTOCOL;
        end else if (array_result_valid &&
                     ((owner_state_q == OWNER_STATE_SRNN &&
                       (array_result_owner != OWNER_SRNN ||
                        array_result_mode_q8_8)) ||
                      (owner_state_q == OWNER_STATE_SCNN &&
                       (array_result_owner != OWNER_SCNN ||
                        !array_result_mode_q8_8)))) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_ARRAY_OWNER;
        end else if (scnn_frontend_done && !scnn_external_mode_q &&
                     !(scnn_result_enqueued_q || scnn_result_push)) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_SCNN_DONE_ORDER;
        end else if (i_scnn_external_done &&
                     (owner_state_q != OWNER_STATE_SCNN ||
                      !scnn_external_mode_q)) begin
            local_error_set = 1'b1;
            local_error_set_code = ERROR_SCNN_DONE_ORDER;
        end
    end

    unified_srnn_recurrent_sidecar_soft_abort_v1 #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .INPUT_MAX(INPUT_MAX),
        .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH)
    ) u_srnn_sidecar (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_owner_enable(owner_state_q == OWNER_STATE_SRNN),
        .i_clear_error(i_clear_error),
        .i_state_reset(i_srnn_state_reset),
        .i_start_valid(srnn_sidecar_start_fire),
        .o_start_ready(srnn_sidecar_start_ready),
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
        .o_event_ready(srnn_sidecar_event_ready),
        .i_event_addr(i_srnn_event_addr),
        .i_event_data(i_srnn_event_data),
        .i_timestep_last(i_srnn_timestep_last),
        .i_sample_last(i_srnn_sample_last),
        .o_k8_req_valid(srnn_sidecar_req_valid),
        .i_k8_req_ready(srnn_sidecar_req_ready),
        .o_k8_req_version(srnn_sidecar_req_version),
        .o_k8_req_id(srnn_sidecar_req_id),
        .o_k8_req_recurrent(srnn_sidecar_req_recurrent),
        .o_k8_req_row_base(srnn_sidecar_req_row_base),
        .o_k8_req_destination_count(srnn_sidecar_req_destination_count),
        .o_k8_req_activation(srnn_sidecar_req_activation),
        .o_k8_req_numeric_profile(srnn_sidecar_req_numeric_profile),
        .i_k8_rsp_valid(srnn_sidecar_rsp_valid),
        .o_k8_rsp_ready(srnn_sidecar_rsp_ready),
        .i_k8_rsp_version(srnn_sidecar_rsp_version),
        .i_k8_rsp_id(srnn_sidecar_rsp_id),
        .i_k8_rsp_destination_base(srnn_sidecar_rsp_destination_base),
        .i_k8_rsp_lane_mask(srnn_sidecar_rsp_lane_mask),
        .i_k8_rsp_contribution_flat(srnn_sidecar_rsp_contribution_flat),
        .i_k8_rsp_last(srnn_sidecar_rsp_last),
        .o_timestep_valid(srnn_timestep_valid),
        .i_timestep_ready(srnn_timestep_ready),
        .o_timestep_version(srnn_timestep_version),
        .o_timestep_sample_last(srnn_timestep_sample_last),
        .o_timestep_spikes(srnn_timestep_spikes),
        .o_previous_spikes(srnn_previous_spikes),
        .o_busy(srnn_sidecar_busy),
        .o_idle(srnn_sidecar_idle),
        .o_sticky_error(srnn_sidecar_sticky_error),
        .o_error_code(srnn_sidecar_error_code)
    );

    unified_scnn_external_side_frontend_soft_abort_v1 #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .FEATURE_ADDR_WIDTH(FEATURE_ADDR_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .Q16_ADDR_WIDTH(Q16_ADDR_WIDTH),
        .OUTPUT_ADDR_WIDTH(OUTPUT_ADDR_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_scnn_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_owner_enable(owner_state_q == OWNER_STATE_SCNN),
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
        .i_unit_spike_features(1'b0),
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
        .i_side_result_accumulator_flat(
            array_result_q8_8_accumulator_flat),
        .i_side_busy(owner_state_q == OWNER_STATE_SCNN &&
                     !array_owner_release_ready),
        .i_side_idle(array_owner_release_ready &&
                     array_owner_active &&
                     array_active_owner == OWNER_SCNN),
        .i_side_sticky_error(array_sticky_error),
        .o_result_valid(scnn_frontend_result_valid),
        .i_result_ready(scnn_frontend_result_ready),
        .o_result_mask(scnn_frontend_result_mask),
        .o_result_output_addr_flat(
            scnn_frontend_result_output_addr_flat),
        .o_result_spatial_addr(scnn_frontend_result_spatial_addr),
        .o_result_accumulator_flat(
            scnn_frontend_result_accumulator_flat),
        .o_busy(scnn_frontend_busy),
        .o_done(scnn_frontend_done),
        .o_config_error(scnn_frontend_config_error),
        .o_protocol_error(scnn_frontend_protocol_error)
    );

    unified_shared_k8_numeric_v1_array_soft_abort_v1 #(
        .ENGINES(ENGINES),
        .LANES(LANES),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_shared_numeric_array (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_owner_valid(array_owner_valid),
        .o_owner_ready(array_owner_ready),
        .i_owner(array_owner),
        .i_mode_q8_8(array_mode_q8_8),
        .i_owner_release_valid(array_owner_release_valid),
        .o_owner_release_ready(array_owner_release_ready),
        .o_owner_active(array_owner_active),
        .o_active_owner(array_active_owner),
        .o_active_mode_q8_8(array_active_mode_q8_8),
        .i_valid(array_valid),
        .o_ready(array_ready),
        .i_engine_mask(array_engine_mask),
        .i_lane_mask(array_lane_mask),
        .i_q8_8_feature_flat(array_q8_8_feature_flat),
        .i_q8_8_weight_flat(array_q8_8_weight_flat),
        .i_int8_lhs_flat(array_int8_lhs_flat),
        .i_int8_rhs_flat(array_int8_rhs_flat),
        .i_first(array_first),
        .i_last(array_last),
        .i_output_channel_base(array_output_channel_base),
        .o_valid(array_result_valid),
        .i_ready(array_result_ready),
        .o_result_owner(array_result_owner),
        .o_result_mode_q8_8(array_result_mode_q8_8),
        .o_result_engine_mask(array_result_engine_mask),
        .o_result_lane_mask(array_result_lane_mask),
        .o_result_channel_flat(array_result_channel_flat),
        .o_q8_8_accumulator_flat(
            array_result_q8_8_accumulator_flat),
        .o_int8_contribution_flat(
            array_result_int8_contribution_flat),
        .o_busy(array_busy),
        .o_idle(array_idle),
        .o_sticky_error(array_sticky_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            local_sticky_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else if (!i_abort) begin
            if (i_clear_error && operational_idle) begin
                local_sticky_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end else if (local_error_set && !local_sticky_error_q) begin
                local_sticky_error_q <= 1'b1;
                local_error_code_q <= local_error_set_code;
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_state_q <= OWNER_STATE_IDLE;
            srnn_start_pending_q <= 1'b0;
            srnn_done_seen_q <= 1'b0;
            scnn_start_pending_q <= 1'b0;
            scnn_done_seen_q <= 1'b0;
            scnn_result_enqueued_q <= 1'b0;
            scnn_external_mode_q <= 1'b0;
            o_srnn_done <= 1'b0;
            o_scnn_done <= 1'b0;
        end else if (i_abort) begin
            owner_state_q <= OWNER_STATE_IDLE;
            srnn_start_pending_q <= 1'b0;
            srnn_done_seen_q <= 1'b0;
            scnn_start_pending_q <= 1'b0;
            scnn_done_seen_q <= 1'b0;
            scnn_result_enqueued_q <= 1'b0;
            scnn_external_mode_q <= 1'b0;
            o_srnn_done <= 1'b0;
            o_scnn_done <= 1'b0;
        end else begin
            o_srnn_done <= 1'b0;
            o_scnn_done <= 1'b0;

            if (srnn_sidecar_start_fire)
                srnn_start_pending_q <= 1'b0;
            if (scnn_frontend_start_fire)
                scnn_start_pending_q <= 1'b0;

            if (srnn_result_push && srnn_timestep_sample_last) begin
                srnn_done_seen_q <= 1'b1;
                o_srnn_done <= 1'b1;
            end
            if (scnn_result_push)
                scnn_result_enqueued_q <= 1'b1;
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

            unique case (owner_state_q)
                OWNER_STATE_IDLE: begin
                    srnn_done_seen_q <= 1'b0;
                    scnn_done_seen_q <= 1'b0;
                    scnn_result_enqueued_q <= 1'b0;
                    if (i_srnn_start_valid && o_srnn_start_ready) begin
                        owner_state_q <= OWNER_STATE_SRNN;
                        srnn_start_pending_q <= 1'b1;
                    end else if (i_scnn_start_valid && o_scnn_start_ready) begin
                        owner_state_q <= OWNER_STATE_SCNN;
                        scnn_start_pending_q <= !i_scnn_external_mode;
                        scnn_external_mode_q <= i_scnn_external_mode;
                    end
                end
                OWNER_STATE_SRNN: begin
                    if (srnn_done_seen_q && srnn_sidecar_idle &&
                        row_state_q == ROW_STATE_IDLE &&
                        !srnn_result_valid_q && array_owner_release_ready)
                        owner_state_q <= OWNER_STATE_SRNN_RELEASE;
                end
                OWNER_STATE_SRNN_RELEASE: begin
                    if (array_owner_release_valid &&
                        array_owner_release_ready) begin
                        owner_state_q <= OWNER_STATE_IDLE;
                        srnn_start_pending_q <= 1'b0;
                    end
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

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
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
        end else if (i_abort) begin
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
        end else begin
            if (i_srnn_start_valid && o_srnn_start_ready) begin
                srnn_input_dim_q <= i_srnn_input_dim;
                srnn_active_neurons_q <= i_srnn_active_neurons;
                srnn_w0_base_q <= i_srnn_w0_base;
                srnn_recurrent_base_q <= i_srnn_recurrent_base;
                srnn_recurrent_enable_q <= i_srnn_recurrent_enable;
                srnn_leak_q15_q <= i_srnn_leak_q15;
                srnn_threshold_q <= i_srnn_threshold;
                srnn_current_left_shift_q <= i_srnn_current_left_shift;
                srnn_reset_to_zero_q <= i_srnn_reset_to_zero;
                srnn_version_q <= i_srnn_version;
            end
            if (i_scnn_start_valid && o_scnn_start_ready) begin
                scnn_in_width_q <= i_scnn_in_width;
                scnn_in_height_q <= i_scnn_in_height;
                scnn_total_in_channels_q <= i_scnn_total_in_channels;
                scnn_out_width_q <= i_scnn_out_width;
                scnn_out_height_q <= i_scnn_out_height;
                scnn_total_out_channels_q <= i_scnn_total_out_channels;
                scnn_input_channel_base_q <= i_scnn_input_channel_base;
                scnn_input_channel_count_q <= i_scnn_input_channel_count;
                scnn_output_channel_base_q <= i_scnn_output_channel_base;
                scnn_output_channel_count_q <= i_scnn_output_channel_count;
                scnn_stride_q <= i_scnn_stride;
                scnn_padding_q <= i_scnn_padding;
                scnn_input_slot_q <= i_scnn_input_slot;
                scnn_feature_base_q <= i_scnn_feature_base;
                scnn_weight_q16_base_q <= i_scnn_weight_q16_base;
                scnn_output_base_q <= i_scnn_output_base;
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            row_state_q <= ROW_STATE_IDLE;
            row_version_q <= '0;
            row_id_q <= '0;
            row_recurrent_q <= 1'b0;
            row_base_q <= '0;
            row_destination_count_q <= '0;
            row_activation_q <= '0;
            row_wave_base_q <= '0;
            row_weight_flat_q <= '0;
            row_contribution_flat_q <= '0;
            row_result_engine_mask_q <= '0;
            row_response_engine_q <= '0;
        end else if (i_abort) begin
            row_state_q <= ROW_STATE_IDLE;
            row_version_q <= '0;
            row_id_q <= '0;
            row_recurrent_q <= 1'b0;
            row_base_q <= '0;
            row_destination_count_q <= '0;
            row_activation_q <= '0;
            row_wave_base_q <= '0;
            row_weight_flat_q <= '0;
            row_contribution_flat_q <= '0;
            row_result_engine_mask_q <= '0;
            row_response_engine_q <= '0;
        end else begin
            unique case (row_state_q)
                ROW_STATE_IDLE: begin
                    if (srnn_sidecar_req_valid && srnn_sidecar_req_ready) begin
                        row_version_q <= srnn_sidecar_req_version;
                        row_id_q <= srnn_sidecar_req_id;
                        row_recurrent_q <= srnn_sidecar_req_recurrent;
                        row_base_q <= srnn_sidecar_req_row_base;
                        row_destination_count_q <=
                            srnn_sidecar_req_destination_count;
                        row_activation_q <= srnn_sidecar_req_activation;
                        row_wave_base_q <= 10'd0;
                        row_state_q <= ROW_STATE_WEIGHT_REQUEST;
                    end
                end
                ROW_STATE_WEIGHT_REQUEST: begin
                    if (o_srnn_weight_req_valid &&
                        i_srnn_weight_req_ready)
                        row_state_q <= ROW_STATE_WEIGHT_RESPONSE;
                end
                ROW_STATE_WEIGHT_RESPONSE: begin
                    if (i_srnn_weight_rsp_valid &&
                        o_srnn_weight_rsp_ready) begin
                        row_weight_flat_q <= row_weight_response_legal ?
                            i_srnn_weight_rsp_flat : '0;
                        row_state_q <= ROW_STATE_ARRAY_REQUEST;
                    end
                end
                ROW_STATE_ARRAY_REQUEST: begin
                    if (array_valid && array_ready)
                        row_state_q <= ROW_STATE_ARRAY_RESULT;
                end
                ROW_STATE_ARRAY_RESULT: begin
                    if (array_result_valid && array_result_ready) begin
                        row_contribution_flat_q <=
                            array_result_int8_contribution_flat;
                        row_result_engine_mask_q <=
                            array_result_engine_mask;
                        row_response_engine_q <= '0;
                        row_state_q <= ROW_STATE_SERIAL_RESPONSE;
                    end
                end
                ROW_STATE_SERIAL_RESPONSE: begin
                    if (srnn_sidecar_rsp_valid &&
                        srnn_sidecar_rsp_ready) begin
                        if (row_response_last) begin
                            row_state_q <= ROW_STATE_IDLE;
                        end else if (row_next_engine_valid) begin
                            row_response_engine_q <=
                                row_response_engine_q + 1'b1;
                        end else begin
                            row_wave_base_q <= row_wave_base_q +
                                10'(ENGINES*LANES);
                            row_state_q <= ROW_STATE_WEIGHT_REQUEST;
                        end
                    end
                end
                default: row_state_q <= ROW_STATE_IDLE;
            endcase
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            srnn_result_valid_q <= 1'b0;
            srnn_result_version_q <= '0;
            srnn_result_sample_last_q <= 1'b0;
            srnn_result_spikes_q <= '0;
        end else if (i_abort) begin
            srnn_result_valid_q <= 1'b0;
            srnn_result_version_q <= '0;
            srnn_result_sample_last_q <= 1'b0;
            srnn_result_spikes_q <= '0;
        end else begin
            unique case ({srnn_result_push, srnn_result_pop})
                2'b10, 2'b11: begin
                    srnn_result_valid_q <= 1'b1;
                    srnn_result_version_q <= srnn_timestep_version;
                    srnn_result_sample_last_q <=
                        srnn_timestep_sample_last;
                    srnn_result_spikes_q <= srnn_timestep_spikes;
                end
                2'b01: srnn_result_valid_q <= 1'b0;
                default: begin end
            endcase
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            scnn_result_valid_q <= 1'b0;
            scnn_result_mask_q <= '0;
            scnn_result_output_addr_flat_q <= '0;
            scnn_result_spatial_addr_q <= '0;
            scnn_result_accumulator_flat_q <= '0;
        end else if (i_abort) begin
            scnn_result_valid_q <= 1'b0;
            scnn_result_mask_q <= '0;
            scnn_result_output_addr_flat_q <= '0;
            scnn_result_spatial_addr_q <= '0;
            scnn_result_accumulator_flat_q <= '0;
        end else begin
            unique case ({scnn_result_push, scnn_result_pop})
                2'b10, 2'b11: begin
                    scnn_result_valid_q <= 1'b1;
                    scnn_result_mask_q <= scnn_frontend_result_mask;
                    scnn_result_output_addr_flat_q <=
                        scnn_frontend_result_output_addr_flat;
                    scnn_result_spatial_addr_q <=
                        scnn_frontend_result_spatial_addr;
                    scnn_result_accumulator_flat_q <=
                        scnn_frontend_result_accumulator_flat;
                end
                2'b01: scnn_result_valid_q <= 1'b0;
                default: begin end
            endcase
        end
    end
endmodule

`default_nettype wire
