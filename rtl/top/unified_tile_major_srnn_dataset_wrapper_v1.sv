`timescale 1ns/1ps
`default_nettype none

// SRNN dataset closure on the shared H40 physical array. This wrapper is an
// inference validation endpoint; FC/SCNN ports remain physically present in
// the array and are reserved for the unified owner wrapper.
module unified_tile_major_srnn_dataset_wrapper_v1 #(
    parameter integer ENGINES = 12,
    parameter integer LANES = 8,
    parameter integer HIDDEN_MAX = 480,
    parameter integer INPUT_MAX = 2312,
    parameter integer OUTPUT_MAX = 32,
    parameter integer VERSION_WIDTH = 16,
    parameter integer REQUEST_ID_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_start_valid,
    output logic                                      o_start_ready,
    input  logic                                      i_state_reset,
    input  logic [12:0]                               i_input_dim,
    input  logic [9:0]                                i_active_neurons,
    input  logic [31:0]                               i_w0_base,
    input  logic [31:0]                               i_recurrent_base,
    input  logic                                      i_recurrent_enable,
    input  logic signed [15:0]                       i_leak_q15,
    input  logic signed [31:0]                       i_threshold,
    input  logic [4:0]                                i_current_left_shift,
    input  logic                                      i_reset_to_zero,
    input  logic [VERSION_WIDTH-1:0]                  i_version,

    input  logic                                      i_event_valid,
    output logic                                      o_event_ready,
    input  logic [12:0]                               i_event_addr,
    input  logic signed [7:0]                        i_event_data,
    input  logic                                      i_timestep_last,
    input  logic                                      i_sample_last,

    output logic                                      o_weight_req_valid,
    input  logic                                      i_weight_req_ready,
    output logic [VERSION_WIDTH-1:0]                  o_weight_req_version,
    output logic [REQUEST_ID_WIDTH-1:0]               o_weight_req_id,
    output logic                                      o_weight_req_recurrent,
    output logic [31:0]                               o_weight_req_row_base,
    output logic [9:0]                                o_weight_req_destination_base,
    output logic [ENGINES-1:0]                        o_weight_req_engine_mask,
    output logic [LANES-1:0]                          o_weight_req_lane_mask,
    input  logic                                      i_weight_rsp_valid,
    output logic                                      o_weight_rsp_ready,
    input  logic [VERSION_WIDTH-1:0]                  i_weight_rsp_version,
    input  logic [REQUEST_ID_WIDTH-1:0]               i_weight_rsp_id,
    input  logic [9:0]                                i_weight_rsp_destination_base,
    input  logic [ENGINES-1:0]                        i_weight_rsp_engine_mask,
    input  logic [ENGINES*LANES*8-1:0]                i_weight_rsp_flat,

    output logic [ENGINES-1:0]                        o_w2_req_valid,
    input  logic [ENGINES-1:0]                        i_w2_req_ready,
    output logic [ENGINES*32-1:0]                     o_w2_addr_flat,
    input  logic [ENGINES*64-1:0]                     i_w2_data_flat,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [5:0]                                o_result_class,
    output logic signed [31:0]                       o_result_score,
    output logic [OUTPUT_MAX*32-1:0]                  o_result_logits_flat,
    output logic [VERSION_WIDTH-1:0]                  o_result_version,
    output logic                                      o_done,

    output logic                                      o_timestep_commit,
    output logic [VERSION_WIDTH-1:0]                  o_timestep_version,
    output logic                                      o_timestep_sample_last,
    output logic [HIDDEN_MAX-1:0]                     o_timestep_spikes,
    output logic [HIDDEN_MAX-1:0]                     o_previous_spikes,
    output logic                                      o_leaf_done,
    output logic [1:0]                                o_active_owner,
    output logic                                      o_shared_request_fire,
    output logic                                      o_shared_result_fire,
    output logic                                      o_busy,
    output logic                                      o_idle,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam integer TILE_NEURONS = 40;
    localparam logic [1:0] OWNER_RECURRENT_INT8 = 2'd2;
    localparam logic [2:0] STATE_IDLE = 3'd0;
    localparam logic [2:0] STATE_ACQUIRE = 3'd1;
    localparam logic [2:0] STATE_RUN = 3'd2;
    localparam logic [2:0] STATE_RELEASE = 3'd3;

    logic [2:0] state_q;
    logic start_pending_q;
    logic sample_done_seen_q;
    logic leaf_done_q;
    logic local_error_q;
    logic [12:0] input_dim_q;
    logic [9:0] active_neurons_q;
    logic [31:0] w0_base_q;
    logic [31:0] recurrent_base_q;
    logic recurrent_enable_q;
    logic signed [15:0] leak_q15_q;
    logic signed [31:0] threshold_q;
    logic [4:0] current_left_shift_q;
    logic reset_to_zero_q;
    logic [VERSION_WIDTH-1:0] version_q;

    logic array_owner_ready;
    logic array_owner_release_ready;
    logic array_owner_active;
    logic [1:0] array_active_owner;
    logic array_result_valid;
    logic array_result_ready;
    logic [1:0] array_result_owner;
    logic array_result_mode_q8_8;
    logic [ENGINES-1:0] array_result_engine_mask;
    logic [LANES-1:0] array_result_lane_mask;
    logic array_request_valid;
    logic array_request_ready;
    logic [ENGINES-1:0] array_request_engine_mask;
    logic [LANES-1:0] array_request_lane_mask;
    logic [LANES*8-1:0] array_request_lhs_flat;
    logic [ENGINES*LANES*8-1:0] array_request_rhs_flat;
    logic array_request_first;
    logic array_request_last;
    logic [2:0] array_request_local_word;
    logic [HIDDEN_MAX-1:0] array_previous_spikes;
    logic [HIDDEN_MAX-1:0] array_current_spikes;
    logic array_busy;
    logic array_idle;
    logic array_error;

    logic state_clear_spikes;
    logic state_clear_word_valid;
    logic [2:0] state_clear_local_word;
    logic [9:0] state_active_neurons;
    logic state_lif_begin;
    logic state_lif_word_valid;
    logic [2:0] state_lif_local_word;
    logic signed [15:0] state_leak_q15;
    logic signed [31:0] state_threshold;
    logic [4:0] state_current_left_shift;
    logic state_reset_to_zero;
    logic state_commit;

    logic frontend_start_ready;
    logic frontend_event_ready;
    logic frontend_timestep_valid;
    logic frontend_timestep_ready;
    logic [VERSION_WIDTH-1:0] frontend_timestep_version;
    logic frontend_timestep_sample_last;
    logic [HIDDEN_MAX-1:0] frontend_timestep_spikes;
    logic frontend_busy;
    logic frontend_idle;
    logic frontend_error;
    logic [7:0] frontend_error_code;
    logic bridge_result_valid;
    logic bridge_timestep_ready;
    logic bridge_busy;
    logic bridge_error;

    wire start_fire = i_start_valid && o_start_ready;
    wire frontend_start_fire = state_q == STATE_RUN &&
        start_pending_q && frontend_start_ready;
    wire timestep_fire = frontend_timestep_valid &&
        frontend_timestep_ready;

    assign o_start_ready = state_q == STATE_IDLE && frontend_idle &&
        array_idle && !bridge_busy && !bridge_result_valid &&
        !o_sticky_error;
    assign o_event_ready = state_q == STATE_RUN && !start_pending_q &&
        !sample_done_seen_q && frontend_event_ready;
    assign frontend_timestep_ready = bridge_timestep_ready;
    assign o_timestep_commit = timestep_fire;
    assign o_timestep_version = frontend_timestep_version;
    assign o_timestep_sample_last = frontend_timestep_sample_last;
    assign o_timestep_spikes = frontend_timestep_spikes;
    // Preserve the Wave3 observation contract: on the commit handshake the
    // newly committed vector is visible immediately, while the physical
    // array updates its previous-spike registers on that clock edge.
    assign o_previous_spikes = o_timestep_commit ?
        frontend_timestep_spikes : array_previous_spikes;
    assign o_leaf_done = leaf_done_q;
    assign o_active_owner = array_active_owner;
    assign o_shared_request_fire = array_request_valid &&
        array_request_ready;
    assign o_shared_result_fire = array_result_valid &&
        array_result_ready;
    assign o_result_valid = bridge_result_valid;
    assign o_result_version = version_q;
    assign o_done = bridge_result_valid && i_result_ready;
    assign o_busy = state_q != STATE_IDLE || frontend_busy || bridge_busy;
    assign o_idle = state_q == STATE_IDLE && frontend_idle && array_idle &&
        !bridge_busy && !bridge_result_valid;
    assign o_sticky_error = local_error_q || frontend_error ||
        array_error || bridge_error;
    assign o_error_code = local_error_q ? 8'h01 :
        (frontend_error ? frontend_error_code :
         (array_error ? 8'h02 : (bridge_error ? 8'h03 : 8'h00)));

    unified_srnn_tile_major_frontend_v1 #(
        .HIDDEN_MAX(HIDDEN_MAX),
        .INPUT_MAX(INPUT_MAX),
        .TILES(ENGINES),
        .TILE_NEURONS(TILE_NEURONS),
        .ENGINES(ENGINES),
        .LANES(LANES),
        .VERSION_WIDTH(VERSION_WIDTH),
        .REQUEST_ID_WIDTH(REQUEST_ID_WIDTH)
    ) u_frontend (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_abort(1'b0),
        .i_owner_enable(state_q == STATE_RUN && array_owner_active &&
            array_active_owner == OWNER_RECURRENT_INT8),
        .i_clear_error(i_clear_error),
        .i_state_reset(i_state_reset && state_q == STATE_RUN),
        .i_start_valid(state_q == STATE_RUN && start_pending_q),
        .o_start_ready(frontend_start_ready),
        .i_input_dim(input_dim_q),
        .i_active_neurons(active_neurons_q),
        .i_w0_base(w0_base_q),
        .i_recurrent_base(recurrent_base_q),
        .i_recurrent_enable(recurrent_enable_q),
        .i_leak_q15(leak_q15_q),
        .i_threshold(threshold_q),
        .i_current_left_shift(current_left_shift_q),
        .i_reset_to_zero(reset_to_zero_q),
        .i_version(version_q),
        .i_event_valid(i_event_valid && state_q == STATE_RUN &&
            !start_pending_q && !sample_done_seen_q),
        .o_event_ready(frontend_event_ready),
        .i_event_addr(i_event_addr),
        .i_event_data(i_event_data),
        .i_timestep_last(i_timestep_last),
        .i_sample_last(i_sample_last),
        .o_weight_req_valid(o_weight_req_valid),
        .i_weight_req_ready(i_weight_req_ready),
        .o_weight_req_version(o_weight_req_version),
        .o_weight_req_id(o_weight_req_id),
        .o_weight_req_recurrent(o_weight_req_recurrent),
        .o_weight_req_row_base(o_weight_req_row_base),
        .o_weight_req_destination_base(o_weight_req_destination_base),
        .o_weight_req_engine_mask(o_weight_req_engine_mask),
        .o_weight_req_lane_mask(o_weight_req_lane_mask),
        .i_weight_rsp_valid(i_weight_rsp_valid),
        .o_weight_rsp_ready(o_weight_rsp_ready),
        .i_weight_rsp_version(i_weight_rsp_version),
        .i_weight_rsp_id(i_weight_rsp_id),
        .i_weight_rsp_destination_base(i_weight_rsp_destination_base),
        .i_weight_rsp_engine_mask(i_weight_rsp_engine_mask),
        .i_weight_rsp_flat(i_weight_rsp_flat),
        .o_array_valid(array_request_valid),
        .i_array_ready(array_request_ready),
        .o_array_engine_mask(array_request_engine_mask),
        .o_array_lane_mask(array_request_lane_mask),
        .o_array_int8_lhs_flat(array_request_lhs_flat),
        .o_array_int8_rhs_flat(array_request_rhs_flat),
        .o_array_first(array_request_first),
        .o_array_last(array_request_last),
        .o_array_local_word(array_request_local_word),
        .i_array_result_valid(array_result_valid),
        .o_array_result_ready(array_result_ready),
        .i_array_result_owner(array_result_owner),
        .i_array_result_mode_q8_8(array_result_mode_q8_8),
        .i_array_result_engine_mask(array_result_engine_mask),
        .i_array_result_lane_mask(array_result_lane_mask),
        .o_state_clear_spikes(state_clear_spikes),
        .o_state_clear_word_valid(state_clear_word_valid),
        .o_state_clear_local_word(state_clear_local_word),
        .o_state_active_neurons(state_active_neurons),
        .o_state_lif_begin(state_lif_begin),
        .o_state_lif_word_valid(state_lif_word_valid),
        .o_state_lif_local_word(state_lif_local_word),
        .o_state_leak_q15(state_leak_q15),
        .o_state_threshold(state_threshold),
        .o_state_current_left_shift(state_current_left_shift),
        .o_state_reset_to_zero(state_reset_to_zero),
        .o_state_commit_timestep(state_commit),
        .i_state_previous_spikes(array_previous_spikes),
        .i_state_current_spikes(array_current_spikes),
        .o_timestep_valid(frontend_timestep_valid),
        .i_timestep_ready(frontend_timestep_ready),
        .o_timestep_version(frontend_timestep_version),
        .o_timestep_sample_last(frontend_timestep_sample_last),
        .o_timestep_spikes(frontend_timestep_spikes),
        .o_previous_spikes(),
        .o_busy(frontend_busy),
        .o_idle(frontend_idle),
        .o_sticky_error(frontend_error),
        .o_error_code(frontend_error_code)
    );

    unified_h40_shared_inference_array_v1 #(
        .TILES(ENGINES),
        .H_PER_TILE(TILE_NEURONS),
        .LANES(LANES)
    ) u_array (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_abort(1'b0),
        .i_clear_error(i_clear_error),
        .i_owner_valid(state_q == STATE_ACQUIRE),
        .o_owner_ready(array_owner_ready),
        .i_owner(OWNER_RECURRENT_INT8),
        .i_owner_release_valid(state_q == STATE_RELEASE),
        .o_owner_release_ready(array_owner_release_ready),
        .o_owner_active(array_owner_active),
        .o_active_owner(array_active_owner),
        .i_fc_clear_state(1'b0),
        .i_fc_active_tile_mask('0),
        .i_fc_step_start(1'b0),
        .o_fc_step_ready(),
        .i_fc_timestep('0),
        .i_fc_row_valid(1'b0),
        .o_fc_row_ready(),
        .i_fc_lane_valid('0),
        .i_fc_w0_weight_rows_flat('0),
        .i_fc_step_close(1'b0),
        .i_fc_w2_weight_flat('0),
        .i_k8_valid(array_request_valid),
        .o_k8_ready(array_request_ready),
        .i_k8_engine_mask(array_request_engine_mask),
        .i_k8_lane_mask(array_request_lane_mask),
        .i_k8_q8_8_feature_flat('0),
        .i_k8_q8_8_weight_flat('0),
        .i_k8_int8_lhs_flat(array_request_lhs_flat),
        .i_k8_int8_rhs_flat(array_request_rhs_flat),
        .i_k8_first(array_request_first),
        .i_k8_last(array_request_last),
        .i_k8_local_word(array_request_local_word),
        .i_k8_output_channel_base('0),
        .i_srnn_clear_spikes(state_clear_spikes),
        .i_srnn_clear_word_valid(state_clear_word_valid),
        .i_srnn_clear_local_word(state_clear_local_word),
        .i_srnn_active_neurons(state_active_neurons),
        .i_srnn_lif_begin(state_lif_begin),
        .i_srnn_lif_word_valid(state_lif_word_valid),
        .i_srnn_lif_local_word(state_lif_local_word),
        .i_srnn_leak_q15(state_leak_q15),
        .i_srnn_threshold(state_threshold),
        .i_srnn_current_left_shift(state_current_left_shift),
        .i_srnn_reset_to_zero(state_reset_to_zero),
        .i_srnn_commit_timestep(state_commit),
        .o_result_valid(array_result_valid),
        .i_result_ready(array_result_ready),
        .o_result_owner(array_result_owner),
        .o_result_mode_q8_8(array_result_mode_q8_8),
        .o_result_engine_mask(array_result_engine_mask),
        .o_result_lane_mask(array_result_lane_mask),
        .o_result_channel_flat(),
        .o_result_k8_q8_8_accumulator_flat(),
        .o_result_k8_int8_contribution_flat(),
        .o_result_fc_timestep(),
        .o_result_fc_events(),
        .o_result_fc_spikes(),
        .o_result_fc_tile_partials_flat(),
        .o_result_fc_logits_flat(),
        .o_result_fc_pred(),
        .o_fc_vmem_flat(),
        .o_fc_trace_flat(),
        .o_srnn_previous_spikes(array_previous_spikes),
        .o_srnn_current_spikes(array_current_spikes),
        .o_srnn_accumulator_flat(),
        .o_srnn_membrane_flat(),
        .o_busy(array_busy),
        .o_idle(array_idle),
        .o_sticky_error(array_error)
    );

    unified_recurrent_output_bridge #(
        .TILE_COUNT(ENGINES),
        .NEURONS_PER_TILE(TILE_NEURONS),
        .OUTPUT_MAX(OUTPUT_MAX),
        .ACC_BITS(32)
    ) u_output (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_abort(1'b0),
        .i_timestep_valid(frontend_timestep_valid),
        .o_timestep_ready(bridge_timestep_ready),
        .i_timestep_sample_last(frontend_timestep_sample_last),
        .i_spikes_flat(frontend_timestep_spikes),
        .i_active_tiles(4'(ENGINES)),
        .i_output_count(6'd20),
        .i_w2_row_stride(6'd20),
        .i_w2_base(32'd0),
        .i_version(version_q),
        .o_w2_req_valid(o_w2_req_valid),
        .i_w2_req_ready(i_w2_req_ready),
        .o_w2_addr_flat(o_w2_addr_flat),
        .i_w2_data_flat(i_w2_data_flat),
        .o_result_valid(bridge_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_class(o_result_class),
        .o_result_score(o_result_score),
        .o_result_logits_flat(o_result_logits_flat),
        .o_busy(bridge_busy),
        .o_sticky_error(bridge_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            start_pending_q <= 1'b0;
            sample_done_seen_q <= 1'b0;
            leaf_done_q <= 1'b0;
            local_error_q <= 1'b0;
            input_dim_q <= '0;
            active_neurons_q <= '0;
            w0_base_q <= '0;
            recurrent_base_q <= '0;
            recurrent_enable_q <= 1'b0;
            leak_q15_q <= '0;
            threshold_q <= '0;
            current_left_shift_q <= '0;
            reset_to_zero_q <= 1'b0;
            version_q <= '0;
        end else begin
            leaf_done_q <= 1'b0;
            if (i_clear_error && o_idle)
                local_error_q <= 1'b0;
            if (array_owner_active && state_q != STATE_IDLE &&
                array_active_owner != OWNER_RECURRENT_INT8)
                local_error_q <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (start_fire) begin
                        input_dim_q <= i_input_dim;
                        active_neurons_q <= i_active_neurons;
                        w0_base_q <= i_w0_base;
                        recurrent_base_q <= i_recurrent_base;
                        recurrent_enable_q <= i_recurrent_enable;
                        leak_q15_q <= i_leak_q15;
                        threshold_q <= i_threshold;
                        current_left_shift_q <= i_current_left_shift;
                        reset_to_zero_q <= i_reset_to_zero;
                        version_q <= i_version;
                        start_pending_q <= 1'b1;
                        sample_done_seen_q <= 1'b0;
                        state_q <= STATE_ACQUIRE;
                    end
                end
                STATE_ACQUIRE: begin
                    if (array_owner_ready)
                        state_q <= STATE_RUN;
                end
                STATE_RUN: begin
                    if (frontend_start_fire)
                        start_pending_q <= 1'b0;
                    if (timestep_fire && frontend_timestep_sample_last)
                        sample_done_seen_q <= 1'b1;
                    if (sample_done_seen_q && frontend_idle)
                        state_q <= STATE_RELEASE;
                end
                STATE_RELEASE: begin
                    if (array_owner_release_ready) begin
                        state_q <= STATE_IDLE;
                        sample_done_seen_q <= 1'b0;
                        leaf_done_q <= 1'b1;
                    end
                end
                default: begin
                    state_q <= STATE_IDLE;
                    local_error_q <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (ENGINES != 12 || LANES != 8 || HIDDEN_MAX != 480)
            $error("dataset wrapper requires the H480 12xH40 target");
    end
endmodule

`default_nettype wire
