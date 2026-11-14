`timescale 1ns/1ps
`default_nettype none

// First physical integration slice for the paper architecture. One owner and
// one tagged result channel select either the exact FC_COMPAT H40 datapath or
// one generic K8 INT8/Q8.8 engine. The two arithmetic frontends are mutually
// exclusive. This module does not yet claim that FC_COMPAT ACC18 state and the
// recurrent ACC32 state store are physically the same memory.
module unified_h40_shared_inference_tile_v1 #(
    parameter integer TILE_BASE = 0,
    parameter integer H_COUNT = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer LIF_LANES = 20,
    parameter integer O_DIM = 10,
    parameter integer FC_ACC_BITS = 18,
    parameter integer FC_OUT_ACC_BITS = 24,
    parameter integer T_BITS = 9,
    parameter integer K8_ACC_BITS = 48,
    parameter bit INCLUDE_FC_COMPAT = 1'b1,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0
) (
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         i_abort,
    input  logic                                         i_clear_error,

    input  logic                                         i_owner_valid,
    output logic                                         o_owner_ready,
    input  logic [1:0]                                   i_owner,
    input  logic                                         i_owner_release_valid,
    output logic                                         o_owner_release_ready,
    output logic                                         o_owner_active,
    output logic [1:0]                                   o_active_owner,

    input  logic                                         i_fc_clear_state,
    input  logic                                         i_fc_step_start,
    output logic                                         o_fc_step_ready,
    input  logic [T_BITS-1:0]                            i_fc_timestep,
    input  logic                                         i_fc_row_valid,
    output logic                                         o_fc_row_ready,
    input  logic [SOURCE_LANES-1:0]                      i_fc_lane_valid,
    input  logic [SOURCE_LANES*H_COUNT*5-1:0]            i_fc_w0_weight_rows_flat,
    input  logic                                         i_fc_step_close,
    input  logic [H_COUNT*O_DIM*8-1:0]                   i_fc_w2_weight_flat,

    input  logic                                         i_k8_valid,
    output logic                                         o_k8_ready,
    input  logic [SOURCE_LANES-1:0]                      i_k8_lane_mask,
    input  logic [SOURCE_LANES*16-1:0]                   i_k8_q8_8_feature_flat,
    input  logic [SOURCE_LANES*16-1:0]                   i_k8_q8_8_weight_flat,
    input  logic [SOURCE_LANES*8-1:0]                    i_k8_int8_lhs_flat,
    input  logic [SOURCE_LANES*8-1:0]                    i_k8_int8_rhs_flat,
    input  logic                                         i_k8_first,
    input  logic                                         i_k8_last,
    input  logic [9:0]                                   i_k8_destination_base,

    input  logic                                         i_srnn_clear_spikes,
    input  logic                                         i_srnn_clear_word_valid,
    input  logic [9:0]                                   i_srnn_clear_word_base,
    input  logic [9:0]                                   i_srnn_active_neurons,
    input  logic                                         i_srnn_lif_begin,
    input  logic                                         i_srnn_lif_word_valid,
    input  logic [9:0]                                   i_srnn_lif_word_base,
    input  logic signed [15:0]                          i_srnn_leak_q15,
    input  logic signed [31:0]                          i_srnn_threshold,
    input  logic [4:0]                                   i_srnn_current_left_shift,
    input  logic                                         i_srnn_reset_to_zero,
    input  logic                                         i_srnn_commit_timestep,

    output logic                                         o_result_valid,
    input  logic                                         i_result_ready,
    output logic [1:0]                                   o_result_owner,
    output logic                                         o_result_mode_q8_8,
    output logic [T_BITS-1:0]                            o_result_fc_timestep,
    output logic [31:0]                                  o_result_fc_events,
    output logic [H_COUNT-1:0]                           o_result_fc_spikes,
    output logic [O_DIM*FC_OUT_ACC_BITS-1:0]             o_result_fc_logits_flat,
    output logic signed [K8_ACC_BITS-1:0]                o_result_k8_q8_8_accumulator,
    output logic [SOURCE_LANES*32-1:0]                   o_result_k8_int8_contribution_flat,

    output logic [H_COUNT-1:0]                           o_srnn_previous_spikes,
    output logic [H_COUNT-1:0]                           o_srnn_current_spikes,
    output logic [H_COUNT*32-1:0]                        o_srnn_accumulator_flat,
    output logic [H_COUNT*32-1:0]                        o_srnn_membrane_flat,

    output logic [H_COUNT*FC_ACC_BITS-1:0]               o_fc_vmem_flat,
    output logic [H_COUNT*16-1:0]                        o_fc_trace_flat,
    output logic                                         o_busy,
    output logic                                         o_idle,
    output logic                                         o_sticky_error
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC_COMPAT = 2'd1;
    localparam logic [1:0] OWNER_RECURRENT_INT8 = 2'd2;
    localparam logic [1:0] OWNER_SCNN_Q8_8 = 2'd3;

    logic owner_active_q;
    logic [1:0] owner_q;
    logic local_error_q;
    logic q8_sequence_active_q;

    logic fc_owner;
    logic recurrent_owner;
    logic k8_owner;
    logic k8_mode_q8_8;
    logic selected_datapath_busy;
    logic owner_request_legal;
    logic fc_command_present;
    logic k8_request_legal;
    logic k8_launch;

    logic fc_clear_state;
    logic fc_step_ready;
    logic fc_row_ready;
    logic fc_result_valid;
    logic [T_BITS-1:0] fc_result_timestep;
    logic [31:0] fc_result_events;
    logic [H_COUNT-1:0] fc_result_spikes;
    logic [O_DIM*FC_OUT_ACC_BITS-1:0] fc_result_logits_flat;
    logic fc_busy;
    logic fc_protocol_error;

    logic k8_ready;
    logic k8_result_valid;
    logic k8_result_mode_q8_8;
    logic signed [K8_ACC_BITS-1:0] k8_q8_8_accumulator;
    logic [SOURCE_LANES*32-1:0] k8_int8_contribution_flat;
    logic k8_busy;
    logic [9:0] k8_destination_base_q;
    logic [SOURCE_LANES-1:0] k8_lane_mask_q;
    logic srnn_accumulate_valid;
    logic srnn_state_command_present;

    assign fc_owner = owner_active_q && owner_q == OWNER_FC_COMPAT;
    assign recurrent_owner = owner_active_q &&
        owner_q == OWNER_RECURRENT_INT8;
    assign k8_owner = owner_active_q &&
        (owner_q == OWNER_RECURRENT_INT8 || owner_q == OWNER_SCNN_Q8_8);
    assign k8_mode_q8_8 = owner_q == OWNER_SCNN_Q8_8;
    // FC-disabled tiles still join the array-wide ownership transaction. They
    // remain inert for FC commands, while retaining their SRNN/SCNN datapaths.
    assign owner_request_legal = i_owner == OWNER_FC_COMPAT ||
        i_owner == OWNER_RECURRENT_INT8 || i_owner == OWNER_SCNN_Q8_8;

    assign selected_datapath_busy =
        (fc_owner && fc_busy) ||
        (k8_owner && (k8_busy || q8_sequence_active_q));
    assign o_owner_ready = !owner_active_q && !fc_busy && !k8_busy &&
        !q8_sequence_active_q && !o_sticky_error;
    assign o_owner_release_ready = owner_active_q &&
        !selected_datapath_busy;
    assign o_owner_active = owner_active_q;
    assign o_active_owner = owner_q;
    assign o_busy = owner_active_q || fc_busy || k8_busy ||
        q8_sequence_active_q;
    assign o_idle = !o_busy;
    assign o_sticky_error = local_error_q || fc_protocol_error;

    assign fc_command_present = i_fc_step_start || i_fc_row_valid ||
        i_fc_step_close;
    assign fc_clear_state = i_abort ||
        (fc_owner && i_fc_clear_state) ||
        (i_clear_error && !owner_active_q && !fc_busy && !k8_busy);
    assign o_fc_step_ready = fc_owner && !o_sticky_error && fc_step_ready;
    assign o_fc_row_ready = fc_owner && !o_sticky_error && fc_row_ready;

    // An INT8 request is a complete one-cycle contribution transaction.
    // Q8.8 may span several chunks and therefore has explicit first/last
    // sequence legality at this shared boundary.
    always_comb begin
        if (owner_q == OWNER_RECURRENT_INT8)
            k8_request_legal = i_k8_first && i_k8_last;
        else if (owner_q == OWNER_SCNN_Q8_8)
            k8_request_legal =
                (i_k8_first && !q8_sequence_active_q) ||
                (!i_k8_first && q8_sequence_active_q);
        else
            k8_request_legal = 1'b0;
    end

    assign o_k8_ready = k8_owner && !o_sticky_error && k8_ready;
    assign k8_launch = i_k8_valid && o_k8_ready && k8_request_legal;
    assign srnn_accumulate_valid = recurrent_owner && k8_result_valid &&
        i_result_ready;
    assign srnn_state_command_present = i_srnn_clear_spikes ||
        i_srnn_clear_word_valid || i_srnn_lif_begin ||
        i_srnn_lif_word_valid || i_srnn_commit_timestep;

    assign o_result_valid =
        (fc_owner && fc_result_valid) ||
        (k8_owner && k8_result_valid);
    assign o_result_owner = o_result_valid ? owner_q : OWNER_IDLE;
    assign o_result_mode_q8_8 = k8_owner && k8_result_mode_q8_8;
    assign o_result_fc_timestep = fc_owner ? fc_result_timestep : '0;
    assign o_result_fc_events = fc_owner ? fc_result_events : 32'd0;
    assign o_result_fc_spikes = fc_owner ? fc_result_spikes : '0;
    assign o_result_fc_logits_flat = fc_owner ? fc_result_logits_flat : '0;
    // Result validity owns these payloads. Keeping the registered engine data
    // visible while invalid avoids replicating owner-controlled zero muxes on
    // every wide result bit.
    assign o_result_k8_q8_8_accumulator = k8_q8_8_accumulator;
    assign o_result_k8_int8_contribution_flat =
        k8_int8_contribution_flat;

    generate
        if (USE_SHARED_FC_SRNN_STATE && INCLUDE_FC_COMPAT) begin : g_shared_state
            unified_h40_fc_srnn_shared_compute_tile_candidate_v1 #(
                .TILE_BASE(TILE_BASE),
                .H_COUNT(H_COUNT),
                .SOURCE_LANES(SOURCE_LANES),
                .W0_BITS(5),
                .FC_ACC_BITS(FC_ACC_BITS),
                .W0_INPUT_SHIFT(6),
                .O_DIM(O_DIM),
                .W2_BITS(8),
                .OUT_ACC_BITS(FC_OUT_ACC_BITS),
                .T_BITS(T_BITS)
            ) u_shared_compute (
                .clk(clk),
                .rst_n(rst_n),
                .i_abort(i_abort),
                .i_mode_fc(fc_owner),
                .i_fc_clear_state(fc_clear_state),
                .i_fc_step_start(
                    i_fc_step_start && fc_owner && !o_sticky_error),
                .o_fc_step_ready(fc_step_ready),
                .i_fc_timestep(i_fc_timestep),
                .i_fc_row_valid(
                    i_fc_row_valid && fc_owner && !o_sticky_error),
                .o_fc_row_ready(fc_row_ready),
                .i_fc_lane_valid(i_fc_lane_valid),
                .i_fc_w0_weight_rows_flat(i_fc_w0_weight_rows_flat),
                .i_fc_step_close(
                    i_fc_step_close && fc_owner && !o_sticky_error),
                .i_fc_w2_weight_flat(i_fc_w2_weight_flat),
                .o_fc_result_valid(fc_result_valid),
                .i_fc_result_ready(i_result_ready && fc_owner),
                .o_fc_result_timestep(fc_result_timestep),
                .o_fc_result_events(fc_result_events),
                .o_fc_result_spikes(fc_result_spikes),
                .o_fc_partial_logits_flat(fc_result_logits_flat),
                .o_fc_vmem_flat(o_fc_vmem_flat),
                .o_fc_trace_flat(o_fc_trace_flat),
                .i_srnn_clear_spikes(
                    recurrent_owner && i_srnn_clear_spikes),
                .i_srnn_clear_word_valid(
                    recurrent_owner && i_srnn_clear_word_valid),
                .i_srnn_clear_word_base(i_srnn_clear_word_base),
                .i_srnn_accumulate_valid(srnn_accumulate_valid),
                .i_srnn_accumulate_base(k8_destination_base_q),
                .i_srnn_accumulate_lane_mask(
                    k8_result_valid ? k8_lane_mask_q : '0),
                .i_srnn_accumulate_contribution_flat(
                    k8_int8_contribution_flat),
                .i_srnn_active_neurons(i_srnn_active_neurons),
                .i_srnn_lif_begin(
                    recurrent_owner && i_srnn_lif_begin),
                .i_srnn_lif_word_valid(
                    recurrent_owner && i_srnn_lif_word_valid),
                .i_srnn_lif_word_base(i_srnn_lif_word_base),
                .i_srnn_leak_q15(i_srnn_leak_q15),
                .i_srnn_threshold(i_srnn_threshold),
                .i_srnn_current_left_shift(i_srnn_current_left_shift),
                .i_srnn_reset_to_zero(i_srnn_reset_to_zero),
                .i_srnn_commit_timestep(
                    recurrent_owner && i_srnn_commit_timestep),
                .o_srnn_previous_spikes(o_srnn_previous_spikes),
                .o_srnn_current_spikes(o_srnn_current_spikes),
                .o_srnn_accumulator_flat(o_srnn_accumulator_flat),
                .o_srnn_membrane_flat(o_srnn_membrane_flat),
                .o_busy(fc_busy),
                .o_protocol_error(fc_protocol_error)
            );
        end else if (INCLUDE_FC_COMPAT) begin : g_fc_compat
            unified_h40_fc_compat_compute_tile #(
                .H_COUNT(H_COUNT),
                .SOURCE_LANES(SOURCE_LANES),
                .LIF_LANES(LIF_LANES),
                .O_DIM(O_DIM),
                .ACC_BITS(FC_ACC_BITS),
                .OUT_ACC_BITS(FC_OUT_ACC_BITS),
                .T_BITS(T_BITS)
            ) u_fc_compat (
                .clk(clk),
                .rst_n(rst_n),
                .i_clear_state(fc_clear_state),
                .i_step_start(
                    i_fc_step_start && fc_owner && !o_sticky_error),
                .o_step_ready(fc_step_ready),
                .i_timestep(i_fc_timestep),
                .i_row_valid(i_fc_row_valid && fc_owner && !o_sticky_error),
                .o_row_ready(fc_row_ready),
                .i_lane_valid(i_fc_lane_valid),
                .i_w0_weight_rows_flat(i_fc_w0_weight_rows_flat),
                .i_step_close(
                    i_fc_step_close && fc_owner && !o_sticky_error),
                .i_w2_weight_flat(i_fc_w2_weight_flat),
                .o_result_valid(fc_result_valid),
                .i_result_ready(i_result_ready && fc_owner),
                .o_result_timestep(fc_result_timestep),
                .o_result_events(fc_result_events),
                .o_result_spikes(fc_result_spikes),
                .o_partial_logits_flat(fc_result_logits_flat),
                .o_vmem_flat(o_fc_vmem_flat),
                .o_trace_flat(o_fc_trace_flat),
                .o_busy(fc_busy),
                .o_protocol_error(fc_protocol_error)
            );
        end else begin : g_no_fc_compat
            assign fc_step_ready = 1'b0;
            assign fc_row_ready = 1'b0;
            assign fc_result_valid = 1'b0;
            assign fc_result_timestep = '0;
            assign fc_result_events = '0;
            assign fc_result_spikes = '0;
            assign fc_result_logits_flat = '0;
            assign o_fc_vmem_flat = '0;
            assign o_fc_trace_flat = '0;
            assign fc_busy = 1'b0;
            assign fc_protocol_error = 1'b0;
        end
    endgenerate

    unified_shared_k8_numeric_v1_engine_soft_abort_v1 #(
        .LANES(SOURCE_LANES),
        .ACC_WIDTH(K8_ACC_BITS)
    ) u_k8 (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_abort(i_abort),
        .i_valid(k8_launch),
        .o_ready(k8_ready),
        .i_mode_q8_8(k8_mode_q8_8),
        .i_lane_mask(i_k8_lane_mask),
        .i_q8_8_feature_flat(i_k8_q8_8_feature_flat),
        .i_q8_8_weight_flat(i_k8_q8_8_weight_flat),
        .i_int8_lhs_flat(i_k8_int8_lhs_flat),
        .i_int8_rhs_flat(i_k8_int8_rhs_flat),
        .i_first(i_k8_first),
        .i_last(i_k8_last),
        .o_valid(k8_result_valid),
        .i_ready(i_result_ready && k8_owner),
        .o_result_mode_q8_8(k8_result_mode_q8_8),
        .o_q8_8_accumulator(k8_q8_8_accumulator),
        .o_int8_contribution_flat(k8_int8_contribution_flat),
        .o_busy(k8_busy)
    );

    generate
        if (!(USE_SHARED_FC_SRNN_STATE && INCLUDE_FC_COMPAT)) begin : g_srnn_state
            unified_srnn_h40_state_shard_v1 #(
                .TILE_BASE(TILE_BASE),
                .TILE_NEURONS(H_COUNT),
                .LANES(SOURCE_LANES)
            ) u_srnn_state (
                .clk(clk),
                .rst_n(rst_n),
                .i_abort(i_abort),
                .i_clear_spikes(recurrent_owner && i_srnn_clear_spikes),
                .i_clear_word_valid(
                    recurrent_owner && i_srnn_clear_word_valid),
                .i_clear_word_base(i_srnn_clear_word_base),
                .i_accumulate_valid(srnn_accumulate_valid),
                .i_accumulate_base(k8_destination_base_q),
                .i_accumulate_lane_mask(
                    k8_result_valid ? k8_lane_mask_q : '0),
                .i_accumulate_contribution_flat(k8_int8_contribution_flat),
                .i_active_neurons(i_srnn_active_neurons),
                .i_lif_begin(recurrent_owner && i_srnn_lif_begin),
                .i_lif_word_valid(
                    recurrent_owner && i_srnn_lif_word_valid),
                .i_lif_word_base(i_srnn_lif_word_base),
                .i_leak_q15(i_srnn_leak_q15),
                .i_threshold(i_srnn_threshold),
                .i_current_left_shift(i_srnn_current_left_shift),
                .i_reset_to_zero(i_srnn_reset_to_zero),
                .i_commit_timestep(
                    recurrent_owner && i_srnn_commit_timestep),
                .o_previous_spikes(o_srnn_previous_spikes),
                .o_current_spikes(o_srnn_current_spikes),
                .o_accumulator_flat(o_srnn_accumulator_flat),
                .o_membrane_flat(o_srnn_membrane_flat)
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            owner_active_q <= 1'b0;
            owner_q <= OWNER_IDLE;
            local_error_q <= 1'b0;
            q8_sequence_active_q <= 1'b0;
            k8_destination_base_q <= '0;
            k8_lane_mask_q <= '0;
        end else if (i_abort) begin
            owner_active_q <= 1'b0;
            owner_q <= OWNER_IDLE;
            q8_sequence_active_q <= 1'b0;
            k8_destination_base_q <= '0;
            k8_lane_mask_q <= '0;
        end else begin
            if (i_clear_error && !owner_active_q && !fc_busy && !k8_busy) begin
                local_error_q <= 1'b0;
                q8_sequence_active_q <= 1'b0;
            end

            if (i_owner_valid && o_owner_ready) begin
                if (owner_request_legal) begin
                    owner_active_q <= 1'b1;
                    owner_q <= i_owner;
                end else begin
                    local_error_q <= 1'b1;
                end
            end else if (i_owner_valid && owner_active_q &&
                         i_owner != owner_q) begin
                local_error_q <= 1'b1;
            end

            if (i_owner_release_valid) begin
                if (o_owner_release_ready) begin
                    owner_active_q <= 1'b0;
                    owner_q <= OWNER_IDLE;
                end else begin
                    local_error_q <= 1'b1;
                end
            end

            if (fc_command_present && !fc_owner)
                local_error_q <= 1'b1;
            if (i_k8_valid && !k8_owner)
                local_error_q <= 1'b1;
            if (i_k8_valid && o_k8_ready && !k8_request_legal)
                local_error_q <= 1'b1;
            if (srnn_state_command_present && !recurrent_owner)
                local_error_q <= 1'b1;

            if (k8_launch) begin
                if (recurrent_owner)
                    k8_destination_base_q <= i_k8_destination_base;
                if (recurrent_owner)
                    k8_lane_mask_q <= i_k8_lane_mask;
                if (k8_mode_q8_8)
                    q8_sequence_active_q <= !i_k8_last;
            end
        end
    end

    initial begin
        if (SOURCE_LANES != 8)
            $error("shared inference tile currently requires eight lanes");
        if (H_COUNT != 40)
            $error("shared inference tile currently requires one H40 state tile");
        if (TILE_BASE % H_COUNT != 0)
            $error("shared inference tile base must align to H40");
    end
endmodule

`default_nettype wire
