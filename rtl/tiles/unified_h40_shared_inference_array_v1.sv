`timescale 1ns/1ps
`default_nettype none

// Tile-major physical array for the paper architecture. Each engine is fixed
// beside one H40 state shard. A recurrent row is evaluated in five local-word
// issues across all active tiles, while FC_COMPAT and SCNN retain their exact
// arithmetic frontends and share the same array owner/result boundary.
module unified_h40_shared_inference_array_v1 #(
    parameter integer TILES = 12,
    parameter integer FC_TILES = TILES,
    parameter integer H_PER_TILE = 40,
    parameter integer LANES = 8,
    parameter integer O_DIM = 10,
    parameter integer FC_ACC_BITS = 18,
    parameter integer FC_OUT_ACC_BITS = 24,
    parameter integer T_BITS = 9,
    parameter integer K8_ACC_BITS = 48,
    parameter integer CHANNEL_WIDTH = 10,
    parameter bit INCLUDE_FC_COMPAT = 1'b1,
    parameter bit USE_SHARED_FC_SRNN_STATE = 1'b0
) (
    input  logic                                             clk,
    input  logic                                             rst_n,
    input  logic                                             i_abort,
    input  logic                                             i_clear_error,

    input  logic                                             i_owner_valid,
    output logic                                             o_owner_ready,
    input  logic [1:0]                                       i_owner,
    input  logic                                             i_owner_release_valid,
    output logic                                             o_owner_release_ready,
    output logic                                             o_owner_active,
    output logic [1:0]                                       o_active_owner,

    input  logic                                             i_fc_clear_state,
    input  logic [TILES-1:0]                                i_fc_active_tile_mask,
    input  logic                                             i_fc_step_start,
    output logic                                             o_fc_step_ready,
    input  logic [T_BITS-1:0]                                i_fc_timestep,
    input  logic                                             i_fc_row_valid,
    output logic                                             o_fc_row_ready,
    input  logic [LANES-1:0]                                 i_fc_lane_valid,
    input  logic [TILES*LANES*H_PER_TILE*5-1:0]             i_fc_w0_weight_rows_flat,
    input  logic                                             i_fc_step_close,
    input  logic [TILES*H_PER_TILE*O_DIM*8-1:0]             i_fc_w2_weight_flat,

    input  logic                                             i_k8_valid,
    output logic                                             o_k8_ready,
    input  logic [TILES-1:0]                                i_k8_engine_mask,
    input  logic [LANES-1:0]                                 i_k8_lane_mask,
    input  logic [LANES*16-1:0]                              i_k8_q8_8_feature_flat,
    input  logic [TILES*LANES*16-1:0]                       i_k8_q8_8_weight_flat,
    input  logic [LANES*8-1:0]                               i_k8_int8_lhs_flat,
    input  logic [TILES*LANES*8-1:0]                        i_k8_int8_rhs_flat,
    input  logic                                             i_k8_first,
    input  logic                                             i_k8_last,
    input  logic [2:0]                                       i_k8_local_word,
    input  logic [CHANNEL_WIDTH-1:0]                         i_k8_output_channel_base,

    input  logic                                             i_srnn_clear_spikes,
    input  logic                                             i_srnn_clear_word_valid,
    input  logic [2:0]                                       i_srnn_clear_local_word,
    input  logic [9:0]                                       i_srnn_active_neurons,
    input  logic                                             i_srnn_lif_begin,
    input  logic                                             i_srnn_lif_word_valid,
    input  logic [2:0]                                       i_srnn_lif_local_word,
    input  logic signed [15:0]                              i_srnn_leak_q15,
    input  logic signed [31:0]                              i_srnn_threshold,
    input  logic [4:0]                                       i_srnn_current_left_shift,
    input  logic                                             i_srnn_reset_to_zero,
    input  logic                                             i_srnn_commit_timestep,

    output logic                                             o_result_valid,
    input  logic                                             i_result_ready,
    output logic [1:0]                                       o_result_owner,
    output logic                                             o_result_mode_q8_8,
    output logic [TILES-1:0]                                o_result_engine_mask,
    output logic [LANES-1:0]                                 o_result_lane_mask,
    output logic [TILES*CHANNEL_WIDTH-1:0]                   o_result_channel_flat,
    output logic [TILES*K8_ACC_BITS-1:0]                    o_result_k8_q8_8_accumulator_flat,
    output logic [TILES*LANES*32-1:0]                       o_result_k8_int8_contribution_flat,

    output logic [T_BITS-1:0]                                o_result_fc_timestep,
    output logic [31:0]                                      o_result_fc_events,
    output logic [TILES*H_PER_TILE-1:0]                     o_result_fc_spikes,
    output logic [TILES*O_DIM*FC_OUT_ACC_BITS-1:0]          o_result_fc_tile_partials_flat,
    output logic [O_DIM*FC_OUT_ACC_BITS-1:0]                o_result_fc_logits_flat,
    output logic [3:0]                                       o_result_fc_pred,
    output logic [TILES*H_PER_TILE*FC_ACC_BITS-1:0]         o_fc_vmem_flat,
    output logic [TILES*H_PER_TILE*16-1:0]                  o_fc_trace_flat,

    output logic [TILES*H_PER_TILE-1:0]                     o_srnn_previous_spikes,
    output logic [TILES*H_PER_TILE-1:0]                     o_srnn_current_spikes,
    output logic [TILES*H_PER_TILE*32-1:0]                  o_srnn_accumulator_flat,
    output logic [TILES*H_PER_TILE*32-1:0]                  o_srnn_membrane_flat,

    output logic                                             o_busy,
    output logic                                             o_idle,
    output logic                                             o_sticky_error
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC_COMPAT = 2'd1;
    localparam logic [1:0] OWNER_RECURRENT_INT8 = 2'd2;
    localparam logic [1:0] OWNER_SCNN_Q8_8 = 2'd3;
    localparam integer LOCAL_WORDS = H_PER_TILE / LANES;
    localparam logic [TILES-1:0] FC_CAPABLE_MASK =
        {TILES{1'b1}} >> (TILES - FC_TILES);

    logic owner_active_q;
    logic [1:0] owner_q;
    logic local_error_q;
    logic q8_sequence_active_q;
    logic [TILES-1:0] q8_sequence_engine_mask_q;
    logic [CHANNEL_WIDTH-1:0] q8_sequence_channel_base_q;

    logic [TILES-1:0] tile_owner_ready;
    logic [TILES-1:0] tile_owner_release_ready;
    logic [TILES-1:0] tile_owner_active;
    logic [TILES*2-1:0] tile_active_owner_flat;
    logic tile_owner_valid;
    logic tile_owner_release_valid;
    logic [TILES-1:0] tile_result_valid;
    logic [TILES-1:0] tile_result_ready;
    logic [TILES*2-1:0] tile_result_owner_flat;
    logic [TILES-1:0] tile_result_mode_q8_8;
    logic [TILES*T_BITS-1:0] tile_fc_timestep_flat;
    logic [TILES*32-1:0] tile_fc_events_flat;
    logic [TILES*H_PER_TILE-1:0] tile_fc_spikes_flat;
    logic [TILES*O_DIM*FC_OUT_ACC_BITS-1:0] tile_fc_partials_flat;
    logic [TILES*K8_ACC_BITS-1:0] tile_k8_q8_8_accumulator_flat;
    logic [TILES*LANES*32-1:0] tile_k8_int8_contribution_flat;
    logic [TILES*H_PER_TILE*FC_ACC_BITS-1:0] tile_fc_vmem_flat;
    logic [TILES*H_PER_TILE*16-1:0] tile_fc_trace_flat;
    logic [TILES*H_PER_TILE-1:0] tile_srnn_previous_spikes;
    logic [TILES*H_PER_TILE-1:0] tile_srnn_current_spikes;
    logic [TILES*H_PER_TILE*32-1:0] tile_srnn_accumulator_flat;
    logic [TILES*H_PER_TILE*32-1:0] tile_srnn_membrane_flat;
    logic [TILES-1:0] tile_busy;
    logic [TILES-1:0] tile_idle;
    logic [TILES-1:0] tile_sticky_error;
    logic [TILES-1:0] tile_fc_step_ready;
    logic [TILES-1:0] tile_fc_row_ready;
    logic [TILES-1:0] tile_k8_ready;

    logic owner_request_legal;
    logic owner_acquire_fire;
    logic owner_release_fire;
    logic fc_owner;
    logic recurrent_owner;
    logic scnn_owner;
    logic k8_owner;
    logic fc_command_present;
    logic srnn_state_command_present;
    logic active_tile_mask_legal;
    logic tile_owner_mismatch;

    logic [TILES-1:0] fc_active_tile_mask_q;
    logic all_fc_results_valid;
    logic fc_metadata_mismatch;
    logic fc_reduction_start;
    logic fc_reduction_valid;
    logic [T_BITS-1:0] fc_reduction_timestep;
    logic [O_DIM*FC_OUT_ACC_BITS-1:0] fc_reduction_logits;
    logic fc_reduction_inflight_q;
    logic fc_result_valid_q;
    logic [T_BITS-1:0] fc_result_timestep_q;
    logic [31:0] fc_result_events_q;
    logic [TILES*H_PER_TILE-1:0] fc_result_spikes_q;
    logic [O_DIM*FC_OUT_ACC_BITS-1:0] fc_result_logits_q;
    logic [31:0] fc_pending_events_q;
    logic [TILES*H_PER_TILE-1:0] fc_pending_spikes_q;

    logic k8_request_legal;
    logic all_requested_k8_ready;
    logic k8_launch;
    logic k8_produces_result;
    logic [TILES-1:0] k8_result_engine_mask_q;
    logic [LANES-1:0] k8_result_lane_mask_q;
    logic [CHANNEL_WIDTH-1:0] k8_result_channel_base_q;
    logic all_k8_results_valid;
    logic k8_result_valid;
    logic k8_result_retire;
    logic wrapper_pending;

    function automatic logic prefix_mask_legal(
        input logic [TILES-1:0] mask
    );
        logic zero_seen;
        begin
            prefix_mask_legal = mask[0];
            zero_seen = 1'b0;
            for (integer tile_i = 0; tile_i < TILES; tile_i++) begin
                if (!mask[tile_i])
                    zero_seen = 1'b1;
                else if (zero_seen)
                    prefix_mask_legal = 1'b0;
            end
        end
    endfunction

    assign owner_request_legal =
        (INCLUDE_FC_COMPAT && i_owner == OWNER_FC_COMPAT) ||
        i_owner == OWNER_RECURRENT_INT8 || i_owner == OWNER_SCNN_Q8_8;
    assign wrapper_pending = fc_reduction_inflight_q || fc_result_valid_q ||
        (k8_result_engine_mask_q != '0) || q8_sequence_active_q;
    assign o_owner_ready = !owner_active_q && !wrapper_pending &&
        (&tile_owner_ready) && !o_sticky_error;
    assign owner_acquire_fire = i_owner_valid && o_owner_ready &&
        owner_request_legal;
    assign tile_owner_valid = owner_acquire_fire;
    assign o_owner_release_ready = owner_active_q && !wrapper_pending &&
        !srnn_state_command_present && (&tile_owner_release_ready);
    assign owner_release_fire = i_owner_release_valid &&
        o_owner_release_ready;
    assign tile_owner_release_valid = owner_release_fire;
    assign o_owner_active = owner_active_q;
    assign o_active_owner = owner_q;

    assign fc_owner = owner_active_q && owner_q == OWNER_FC_COMPAT;
    assign recurrent_owner = owner_active_q &&
        owner_q == OWNER_RECURRENT_INT8;
    assign scnn_owner = owner_active_q && owner_q == OWNER_SCNN_Q8_8;
    assign k8_owner = recurrent_owner || scnn_owner;
    assign fc_command_present = i_fc_step_start || i_fc_row_valid ||
        i_fc_step_close || i_fc_clear_state;
    assign srnn_state_command_present = i_srnn_clear_spikes ||
        i_srnn_clear_word_valid || i_srnn_lif_begin ||
        i_srnn_lif_word_valid || i_srnn_commit_timestep;
    assign active_tile_mask_legal = prefix_mask_legal(
        i_fc_active_tile_mask) &&
        ((i_fc_active_tile_mask & ~FC_CAPABLE_MASK) == '0);

    always_comb begin
        tile_owner_mismatch = 1'b0;
        for (integer tile_i = 0; tile_i < TILES; tile_i++) begin
            if (tile_owner_active[tile_i] != owner_active_q)
                tile_owner_mismatch = 1'b1;
            if (owner_active_q && tile_owner_active[tile_i] &&
                tile_active_owner_flat[tile_i*2 +: 2] != owner_q)
                tile_owner_mismatch = 1'b1;
            if (tile_result_valid[tile_i] &&
                tile_result_owner_flat[tile_i*2 +: 2] != owner_q)
                tile_owner_mismatch = 1'b1;
            if (tile_result_valid[tile_i] &&
                tile_result_mode_q8_8[tile_i] != scnn_owner)
                tile_owner_mismatch = 1'b1;
        end
    end

    assign o_fc_step_ready = fc_owner && active_tile_mask_legal &&
        (&(tile_fc_step_ready | ~i_fc_active_tile_mask));
    assign o_fc_row_ready = fc_owner &&
        (&(tile_fc_row_ready | ~fc_active_tile_mask_q));
    assign all_fc_results_valid = |fc_active_tile_mask_q &&
        (&(tile_result_valid | ~fc_active_tile_mask_q));
    assign fc_reduction_start = fc_owner && all_fc_results_valid &&
        !fc_reduction_inflight_q && !fc_result_valid_q;

    always_comb begin
        fc_metadata_mismatch = 1'b0;
        for (integer tile_i = 1; tile_i < TILES; tile_i++) begin
            if (all_fc_results_valid && fc_active_tile_mask_q[tile_i] &&
                (tile_fc_timestep_flat[tile_i*T_BITS +: T_BITS] !=
                    tile_fc_timestep_flat[0 +: T_BITS] ||
                 tile_fc_events_flat[tile_i*32 +: 32] !=
                    tile_fc_events_flat[0 +: 32]))
                fc_metadata_mismatch = 1'b1;
        end
    end

    always_comb begin
        k8_request_legal = k8_owner && i_k8_engine_mask != '0 &&
            i_k8_lane_mask != '0;
        if (recurrent_owner)
            k8_request_legal = k8_request_legal && i_k8_first &&
                i_k8_last && integer'(i_k8_local_word) < LOCAL_WORDS;
        else if (scnn_owner)
            k8_request_legal = k8_request_legal &&
                ((i_k8_first && !q8_sequence_active_q) ||
                 (!i_k8_first && q8_sequence_active_q &&
                  i_k8_engine_mask == q8_sequence_engine_mask_q &&
                  i_k8_output_channel_base ==
                    q8_sequence_channel_base_q));
    end

    assign all_requested_k8_ready =
        &(tile_k8_ready | ~i_k8_engine_mask);
    assign o_k8_ready = k8_owner && !o_sticky_error &&
        (k8_result_engine_mask_q == '0) && all_requested_k8_ready &&
        k8_request_legal;
    assign k8_launch = i_k8_valid && o_k8_ready;
    assign k8_produces_result = recurrent_owner || i_k8_last;
    assign all_k8_results_valid = k8_result_engine_mask_q != '0 &&
        (&(tile_result_valid | ~k8_result_engine_mask_q));
    assign k8_result_valid = k8_owner && all_k8_results_valid;
    assign k8_result_retire = k8_result_valid && i_result_ready;

    always_comb begin
        tile_result_ready = '0;
        if (fc_owner && fc_reduction_start)
            tile_result_ready = fc_active_tile_mask_q;
        else if (k8_result_retire)
            tile_result_ready = k8_result_engine_mask_q;
    end

    assign o_result_valid = fc_owner ? fc_result_valid_q : k8_result_valid;
    assign o_result_owner = o_result_valid ? owner_q : OWNER_IDLE;
    assign o_result_mode_q8_8 = scnn_owner && k8_result_valid;
    assign o_result_engine_mask = k8_result_valid ?
        k8_result_engine_mask_q : '0;
    assign o_result_lane_mask = k8_result_valid ?
        k8_result_lane_mask_q : '0;
    // The ready-valid contract makes payload values irrelevant while invalid.
    // Forwarding the registered tile payload directly keeps owner/valid
    // control out of the wide arithmetic path into the SCNN LI backend.
    assign o_result_k8_q8_8_accumulator_flat =
        tile_k8_q8_8_accumulator_flat;
    assign o_result_k8_int8_contribution_flat =
        tile_k8_int8_contribution_flat;
    assign o_result_fc_timestep = fc_result_timestep_q;
    assign o_result_fc_events = fc_result_events_q;
    assign o_result_fc_spikes = fc_result_spikes_q;
    assign o_result_fc_tile_partials_flat = tile_fc_partials_flat;
    assign o_result_fc_logits_flat = fc_result_logits_q;
    assign o_fc_vmem_flat = tile_fc_vmem_flat;
    assign o_fc_trace_flat = tile_fc_trace_flat;
    assign o_srnn_previous_spikes = tile_srnn_previous_spikes;
    assign o_srnn_current_spikes = tile_srnn_current_spikes;
    assign o_srnn_accumulator_flat = tile_srnn_accumulator_flat;
    assign o_srnn_membrane_flat = tile_srnn_membrane_flat;

    for (genvar channel_tile = 0; channel_tile < TILES;
         channel_tile++) begin : g_result_channel
        assign o_result_channel_flat[
            channel_tile*CHANNEL_WIDTH +: CHANNEL_WIDTH
        ] = k8_result_channel_base_q + CHANNEL_WIDTH'(channel_tile);
    end

    assign o_busy = owner_active_q || wrapper_pending || (|tile_busy);
    assign o_idle = !o_busy && (&tile_idle);
    // Cross-tile comparisons can span the full array. Capture them locally so
    // the error/status network never becomes a combinational return path into
    // an active producer's ready or write-enable logic.
    assign o_sticky_error = local_error_q || (|tile_sticky_error);

    unified_fc_compat_w2_reduction_tree #(
        .TILES(TILES),
        .O_DIM(O_DIM),
        .ACC_BITS(FC_OUT_ACC_BITS),
        .META_BITS(T_BITS)
    ) u_fc_reduction (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(fc_reduction_start),
        .i_partials_flat(tile_fc_partials_flat),
        .i_meta(tile_fc_timestep_flat[0 +: T_BITS]),
        .o_valid(fc_reduction_valid),
        .o_reduced_flat(fc_reduction_logits),
        .o_meta(fc_reduction_timestep)
    );

    argmax10 u_fc_argmax (
        .in0(o_result_fc_logits_flat[0*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in1(o_result_fc_logits_flat[1*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in2(o_result_fc_logits_flat[2*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in3(o_result_fc_logits_flat[3*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in4(o_result_fc_logits_flat[4*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in5(o_result_fc_logits_flat[5*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in6(o_result_fc_logits_flat[6*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in7(o_result_fc_logits_flat[7*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in8(o_result_fc_logits_flat[8*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .in9(o_result_fc_logits_flat[9*FC_OUT_ACC_BITS +: FC_OUT_ACC_BITS]),
        .idx(o_result_fc_pred)
    );

    for (genvar tile = 0; tile < TILES; tile++) begin : g_tile
        logic [9:0] k8_destination_base;
        logic [9:0] srnn_clear_word_base;
        logic [9:0] srnn_lif_word_base;

        assign k8_destination_base = 10'(
            tile*H_PER_TILE + integer'(i_k8_local_word)*LANES);
        assign srnn_clear_word_base = 10'(
            tile*H_PER_TILE + integer'(i_srnn_clear_local_word)*LANES);
        assign srnn_lif_word_base = 10'(
            tile*H_PER_TILE + integer'(i_srnn_lif_local_word)*LANES);

        unified_h40_shared_inference_tile_v1 #(
            .TILE_BASE(tile*H_PER_TILE),
            .H_COUNT(H_PER_TILE),
            .SOURCE_LANES(LANES),
            .LIF_LANES(20),
            .O_DIM(O_DIM),
            .FC_ACC_BITS(FC_ACC_BITS),
            .FC_OUT_ACC_BITS(FC_OUT_ACC_BITS),
            .T_BITS(T_BITS),
            .K8_ACC_BITS(K8_ACC_BITS),
            .INCLUDE_FC_COMPAT(INCLUDE_FC_COMPAT && (tile < FC_TILES)),
            .USE_SHARED_FC_SRNN_STATE(USE_SHARED_FC_SRNN_STATE)
        ) u_tile (
            .clk(clk),
            .rst_n(rst_n),
            .i_abort(i_abort),
            .i_clear_error(i_clear_error),
            .i_owner_valid(tile_owner_valid),
            .o_owner_ready(tile_owner_ready[tile]),
            .i_owner(i_owner),
            .i_owner_release_valid(tile_owner_release_valid),
            .o_owner_release_ready(tile_owner_release_ready[tile]),
            .o_owner_active(tile_owner_active[tile]),
            .o_active_owner(tile_active_owner_flat[tile*2 +: 2]),
            .i_fc_clear_state(i_abort ||
                (fc_owner && i_fc_clear_state)),
            .i_fc_step_start(fc_owner && i_fc_step_start &&
                i_fc_active_tile_mask[tile]),
            .o_fc_step_ready(tile_fc_step_ready[tile]),
            .i_fc_timestep(i_fc_timestep),
            .i_fc_row_valid(fc_owner && i_fc_row_valid &&
                fc_active_tile_mask_q[tile]),
            .o_fc_row_ready(tile_fc_row_ready[tile]),
            .i_fc_lane_valid(i_fc_lane_valid),
            .i_fc_w0_weight_rows_flat(i_fc_w0_weight_rows_flat[
                tile*LANES*H_PER_TILE*5 +: LANES*H_PER_TILE*5]),
            .i_fc_step_close(fc_owner && i_fc_step_close &&
                fc_active_tile_mask_q[tile]),
            .i_fc_w2_weight_flat(i_fc_w2_weight_flat[
                tile*H_PER_TILE*O_DIM*8 +: H_PER_TILE*O_DIM*8]),
            .i_k8_valid(k8_launch && i_k8_engine_mask[tile]),
            .o_k8_ready(tile_k8_ready[tile]),
            .i_k8_lane_mask(i_k8_lane_mask),
            .i_k8_q8_8_feature_flat(i_k8_q8_8_feature_flat),
            .i_k8_q8_8_weight_flat(i_k8_q8_8_weight_flat[
                tile*LANES*16 +: LANES*16]),
            .i_k8_int8_lhs_flat(i_k8_int8_lhs_flat),
            .i_k8_int8_rhs_flat(i_k8_int8_rhs_flat[
                tile*LANES*8 +: LANES*8]),
            .i_k8_first(i_k8_first),
            .i_k8_last(i_k8_last),
            .i_k8_destination_base(k8_destination_base),
            .i_srnn_clear_spikes(i_srnn_clear_spikes),
            .i_srnn_clear_word_valid(i_srnn_clear_word_valid),
            .i_srnn_clear_word_base(srnn_clear_word_base),
            .i_srnn_active_neurons(i_srnn_active_neurons),
            .i_srnn_lif_begin(i_srnn_lif_begin),
            .i_srnn_lif_word_valid(i_srnn_lif_word_valid),
            .i_srnn_lif_word_base(srnn_lif_word_base),
            .i_srnn_leak_q15(i_srnn_leak_q15),
            .i_srnn_threshold(i_srnn_threshold),
            .i_srnn_current_left_shift(i_srnn_current_left_shift),
            .i_srnn_reset_to_zero(i_srnn_reset_to_zero),
            .i_srnn_commit_timestep(i_srnn_commit_timestep),
            .o_result_valid(tile_result_valid[tile]),
            .i_result_ready(tile_result_ready[tile]),
            .o_result_owner(tile_result_owner_flat[tile*2 +: 2]),
            .o_result_mode_q8_8(tile_result_mode_q8_8[tile]),
            .o_result_fc_timestep(tile_fc_timestep_flat[
                tile*T_BITS +: T_BITS]),
            .o_result_fc_events(tile_fc_events_flat[tile*32 +: 32]),
            .o_result_fc_spikes(tile_fc_spikes_flat[
                tile*H_PER_TILE +: H_PER_TILE]),
            .o_result_fc_logits_flat(tile_fc_partials_flat[
                tile*O_DIM*FC_OUT_ACC_BITS +: O_DIM*FC_OUT_ACC_BITS]),
            .o_result_k8_q8_8_accumulator(
                tile_k8_q8_8_accumulator_flat[
                    tile*K8_ACC_BITS +: K8_ACC_BITS]),
            .o_result_k8_int8_contribution_flat(
                tile_k8_int8_contribution_flat[
                    tile*LANES*32 +: LANES*32]),
            .o_srnn_previous_spikes(tile_srnn_previous_spikes[
                tile*H_PER_TILE +: H_PER_TILE]),
            .o_srnn_current_spikes(tile_srnn_current_spikes[
                tile*H_PER_TILE +: H_PER_TILE]),
            .o_srnn_accumulator_flat(tile_srnn_accumulator_flat[
                tile*H_PER_TILE*32 +: H_PER_TILE*32]),
            .o_srnn_membrane_flat(tile_srnn_membrane_flat[
                tile*H_PER_TILE*32 +: H_PER_TILE*32]),
            .o_fc_vmem_flat(tile_fc_vmem_flat[
                tile*H_PER_TILE*FC_ACC_BITS +: H_PER_TILE*FC_ACC_BITS]),
            .o_fc_trace_flat(tile_fc_trace_flat[
                tile*H_PER_TILE*16 +: H_PER_TILE*16]),
            .o_busy(tile_busy[tile]),
            .o_idle(tile_idle[tile]),
            .o_sticky_error(tile_sticky_error[tile])
        );
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            owner_active_q <= 1'b0;
            owner_q <= OWNER_IDLE;
            local_error_q <= 1'b0;
            q8_sequence_active_q <= 1'b0;
            q8_sequence_engine_mask_q <= '0;
            q8_sequence_channel_base_q <= '0;
            fc_active_tile_mask_q <= '0;
            fc_reduction_inflight_q <= 1'b0;
            fc_result_valid_q <= 1'b0;
            fc_result_timestep_q <= '0;
            fc_result_events_q <= '0;
            fc_result_spikes_q <= '0;
            fc_result_logits_q <= '0;
            fc_pending_events_q <= '0;
            fc_pending_spikes_q <= '0;
            k8_result_engine_mask_q <= '0;
            k8_result_lane_mask_q <= '0;
            k8_result_channel_base_q <= '0;
        end else if (i_abort) begin
            owner_active_q <= 1'b0;
            owner_q <= OWNER_IDLE;
            q8_sequence_active_q <= 1'b0;
            q8_sequence_engine_mask_q <= '0;
            q8_sequence_channel_base_q <= '0;
            fc_active_tile_mask_q <= '0;
            fc_reduction_inflight_q <= 1'b0;
            fc_result_valid_q <= 1'b0;
            fc_result_timestep_q <= '0;
            fc_result_events_q <= '0;
            fc_result_spikes_q <= '0;
            fc_result_logits_q <= '0;
            fc_pending_events_q <= '0;
            fc_pending_spikes_q <= '0;
            k8_result_engine_mask_q <= '0;
            k8_result_lane_mask_q <= '0;
            k8_result_channel_base_q <= '0;
        end else begin
            if (i_clear_error && !owner_active_q && !wrapper_pending)
                local_error_q <= 1'b0;

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
            if (i_k8_valid && k8_owner && all_requested_k8_ready &&
                !k8_request_legal)
                local_error_q <= 1'b1;
            if (srnn_state_command_present && !recurrent_owner)
                local_error_q <= 1'b1;
            if (tile_owner_mismatch || fc_metadata_mismatch)
                local_error_q <= 1'b1;

            if (fc_owner && i_fc_step_start) begin
                if (!active_tile_mask_legal)
                    local_error_q <= 1'b1;
                else if (o_fc_step_ready)
                    fc_active_tile_mask_q <= i_fc_active_tile_mask;
            end

            if (fc_reduction_start) begin
                fc_reduction_inflight_q <= 1'b1;
                fc_pending_events_q <= tile_fc_events_flat[0 +: 32];
                for (integer tile_i = 0; tile_i < TILES; tile_i++)
                    fc_pending_spikes_q[
                        tile_i*H_PER_TILE +: H_PER_TILE
                    ] <= fc_active_tile_mask_q[tile_i] ?
                        tile_fc_spikes_flat[
                            tile_i*H_PER_TILE +: H_PER_TILE] : '0;
            end
            if (fc_reduction_valid && fc_reduction_inflight_q) begin
                fc_reduction_inflight_q <= 1'b0;
                fc_result_valid_q <= 1'b1;
                fc_result_timestep_q <= fc_reduction_timestep;
                fc_result_events_q <= fc_pending_events_q;
                fc_result_spikes_q <= fc_pending_spikes_q;
                fc_result_logits_q <= fc_reduction_logits;
            end else if (fc_result_valid_q && i_result_ready) begin
                fc_result_valid_q <= 1'b0;
            end

            if (k8_result_retire)
                k8_result_engine_mask_q <= '0;

            if (k8_launch) begin
                if (scnn_owner) begin
                    if (i_k8_first) begin
                        q8_sequence_engine_mask_q <= i_k8_engine_mask;
                        q8_sequence_channel_base_q <=
                            i_k8_output_channel_base;
                    end
                    q8_sequence_active_q <= !i_k8_last;
                end
                if (k8_produces_result) begin
                    k8_result_engine_mask_q <= i_k8_engine_mask;
                    k8_result_lane_mask_q <= i_k8_lane_mask;
                    k8_result_channel_base_q <=
                        i_k8_output_channel_base;
                end
            end
        end
    end

    initial begin
        if (TILES < 1 || TILES > 12)
            $error("shared H40 array supports 1..12 tiles");
        if (H_PER_TILE != 40 || LANES != 8 || LOCAL_WORDS != 5)
            $error("shared H40 array requires H40/K8 tile geometry");
        if (O_DIM != 10 || FC_OUT_ACC_BITS != 24)
            $error("shared H40 array FC reduction requires 10 ACC24 outputs");
    end

    initial begin
        if (FC_TILES < 1 || FC_TILES > TILES)
            $error("FC_TILES must be in the range 1..TILES");
    end
endmodule

`default_nettype wire
