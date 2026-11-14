`timescale 1ns/1ps
`default_nettype none

module unified_fc_compat_tile_array #(
    parameter integer TILES = 4,
    parameter integer H_PER_TILE = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer O_DIM = 10,
    parameter integer W0_BITS = 5,
    parameter integer ACC_BITS = 18,
    parameter integer W2_BITS = 8,
    parameter integer OUT_ACC_BITS = 24,
    parameter integer T_BITS = 9
) (
    input  logic                                             clk,
    input  logic                                             rst_n,
    input  logic                                             i_clear_state,
    input  logic [TILES-1:0]                                i_active_tile_mask,
    input  logic                                             i_step_start,
    output logic                                             o_step_ready,
    input  logic [T_BITS-1:0]                                i_timestep,
    input  logic                                             i_row_valid,
    output logic                                             o_row_ready,
    input  logic [SOURCE_LANES-1:0]                          i_lane_valid,
    input  logic [TILES*SOURCE_LANES*H_PER_TILE*W0_BITS-1:0] i_w0_weight_rows_flat,
    input  logic                                             i_step_close,
    input  logic [TILES*H_PER_TILE*O_DIM*W2_BITS-1:0]         i_w2_weight_flat,
    output logic                                             o_result_valid,
    input  logic                                             i_result_ready,
    output logic [T_BITS-1:0]                                o_result_timestep,
    output logic [31:0]                                      o_result_events,
    output logic [TILES*H_PER_TILE-1:0]                      o_result_spikes,
    output logic [TILES*O_DIM*OUT_ACC_BITS-1:0]              o_tile_partials_flat,
    output logic [O_DIM*OUT_ACC_BITS-1:0]                    o_logits_flat,
    output logic [3:0]                                       o_pred,
    output logic [TILES*H_PER_TILE*ACC_BITS-1:0]             o_vmem_flat,
    output logic [TILES*H_PER_TILE*16-1:0]                   o_trace_flat,
    output logic                                             o_busy,
    output logic                                             o_protocol_error
);
    localparam integer META_BITS = T_BITS;
    logic [TILES-1:0] tile_step_ready;
    logic [TILES-1:0] tile_row_ready;
    logic [TILES-1:0] tile_result_valid;
    logic [TILES-1:0] tile_result_ready;
    logic [TILES*T_BITS-1:0] tile_timestep_flat;
    logic [TILES*32-1:0] tile_events_flat;
    logic [TILES*H_PER_TILE-1:0] tile_spikes_flat;
    logic [TILES*O_DIM*OUT_ACC_BITS-1:0] tile_partials_flat;
    logic [TILES*H_PER_TILE*ACC_BITS-1:0] tile_vmem_flat;
    logic [TILES*H_PER_TILE*16-1:0] tile_trace_flat;
    logic [TILES-1:0] tile_busy;
    logic [TILES-1:0] tile_protocol_error;
    logic all_tile_results_valid;
    logic reduction_start;
    logic reduction_valid;
    logic [META_BITS-1:0] reduction_timestep;
    logic [O_DIM*OUT_ACC_BITS-1:0] reduction_logits;
    logic reduction_inflight_q;
    logic [31:0] pending_events_q;
    logic [TILES*H_PER_TILE-1:0] pending_spikes_q;
    logic [TILES-1:0] active_tile_mask_q;
    logic metadata_mismatch;
    logic local_protocol_error_q;

    assign o_step_ready = &(tile_step_ready | ~i_active_tile_mask);
    assign o_row_ready = &(tile_row_ready | ~active_tile_mask_q);
    assign all_tile_results_valid =
        &(tile_result_valid | ~active_tile_mask_q);
    assign reduction_start = (|active_tile_mask_q) &&
        all_tile_results_valid && !reduction_inflight_q &&
        (!o_result_valid || i_result_ready);
    assign tile_result_ready = active_tile_mask_q & {TILES{reduction_start}};
    assign o_busy = (|(tile_busy & active_tile_mask_q)) ||
        reduction_inflight_q || o_result_valid;
    assign o_protocol_error = (|(tile_protocol_error & active_tile_mask_q)) ||
        local_protocol_error_q || metadata_mismatch;

    function automatic logic prefix_mask_legal(
        input logic [TILES-1:0] mask
    );
        logic zero_seen;
        integer mask_i;
        begin
            prefix_mask_legal = mask[0];
            zero_seen = 1'b0;
            for (mask_i = 0; mask_i < TILES; mask_i = mask_i + 1) begin
                if (!mask[mask_i])
                    zero_seen = 1'b1;
                else if (zero_seen)
                    prefix_mask_legal = 1'b0;
            end
        end
    endfunction

    genvar tile;
    generate
        for (tile = 0; tile < TILES; tile = tile + 1) begin : gen_tile
            unified_h40_fc_compat_compute_tile #(
                .H_COUNT(H_PER_TILE),
                .SOURCE_LANES(SOURCE_LANES),
                .LIF_LANES(20),
                .W0_BITS(W0_BITS),
                .ACC_BITS(ACC_BITS),
                .O_DIM(O_DIM),
                .W2_BITS(W2_BITS),
                .OUT_ACC_BITS(OUT_ACC_BITS),
                .T_BITS(T_BITS)
            ) u_compute_tile (
                .clk(clk),
                .rst_n(rst_n),
                .i_clear_state(i_clear_state),
                .i_step_start(i_step_start && i_active_tile_mask[tile]),
                .o_step_ready(tile_step_ready[tile]),
                .i_timestep(i_timestep),
                .i_row_valid(i_row_valid && active_tile_mask_q[tile]),
                .o_row_ready(tile_row_ready[tile]),
                .i_lane_valid(i_lane_valid),
                .i_w0_weight_rows_flat(i_w0_weight_rows_flat[
                    tile*SOURCE_LANES*H_PER_TILE*W0_BITS +:
                    SOURCE_LANES*H_PER_TILE*W0_BITS
                ]),
                .i_step_close(i_step_close && active_tile_mask_q[tile]),
                .i_w2_weight_flat(i_w2_weight_flat[
                    tile*H_PER_TILE*O_DIM*W2_BITS +:
                    H_PER_TILE*O_DIM*W2_BITS
                ]),
                .o_result_valid(tile_result_valid[tile]),
                .i_result_ready(tile_result_ready[tile]),
                .o_result_timestep(tile_timestep_flat[tile*T_BITS +: T_BITS]),
                .o_result_events(tile_events_flat[tile*32 +: 32]),
                .o_result_spikes(tile_spikes_flat[
                    tile*H_PER_TILE +: H_PER_TILE
                ]),
                .o_partial_logits_flat(tile_partials_flat[
                    tile*O_DIM*OUT_ACC_BITS +: O_DIM*OUT_ACC_BITS
                ]),
                .o_vmem_flat(tile_vmem_flat[
                    tile*H_PER_TILE*ACC_BITS +: H_PER_TILE*ACC_BITS
                ]),
                .o_trace_flat(tile_trace_flat[
                    tile*H_PER_TILE*16 +: H_PER_TILE*16
                ]),
                .o_busy(tile_busy[tile]),
                .o_protocol_error(tile_protocol_error[tile])
            );

            assign o_tile_partials_flat[
                tile*O_DIM*OUT_ACC_BITS +: O_DIM*OUT_ACC_BITS
            ] = active_tile_mask_q[tile] ?
                tile_partials_flat[
                    tile*O_DIM*OUT_ACC_BITS +: O_DIM*OUT_ACC_BITS
                ] : '0;
            assign o_vmem_flat[
                tile*H_PER_TILE*ACC_BITS +: H_PER_TILE*ACC_BITS
            ] = active_tile_mask_q[tile] ?
                tile_vmem_flat[
                    tile*H_PER_TILE*ACC_BITS +: H_PER_TILE*ACC_BITS
                ] : '0;
            assign o_trace_flat[
                tile*H_PER_TILE*16 +: H_PER_TILE*16
            ] = active_tile_mask_q[tile] ?
                tile_trace_flat[
                    tile*H_PER_TILE*16 +: H_PER_TILE*16
                ] : '0;
        end
    endgenerate

    always_comb begin
        metadata_mismatch = 1'b0;
        for (integer tile_i = 1; tile_i < TILES; tile_i = tile_i + 1)
            if (all_tile_results_valid && active_tile_mask_q[tile_i] &&
                (tile_timestep_flat[tile_i*T_BITS +: T_BITS] !=
                    tile_timestep_flat[0 +: T_BITS] ||
                 tile_events_flat[tile_i*32 +: 32] !=
                    tile_events_flat[0 +: 32]))
                metadata_mismatch = 1'b1;
    end

    unified_fc_compat_w2_reduction_tree #(
        .TILES(TILES),
        .O_DIM(O_DIM),
        .ACC_BITS(OUT_ACC_BITS),
        .META_BITS(META_BITS)
    ) u_reduction (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(reduction_start),
        .i_partials_flat(o_tile_partials_flat),
        .i_meta(tile_timestep_flat[0 +: T_BITS]),
        .o_valid(reduction_valid),
        .o_reduced_flat(reduction_logits),
        .o_meta(reduction_timestep)
    );

    argmax10 u_argmax (
        .in0(o_logits_flat[0*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in1(o_logits_flat[1*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in2(o_logits_flat[2*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in3(o_logits_flat[3*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in4(o_logits_flat[4*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in5(o_logits_flat[5*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in6(o_logits_flat[6*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in7(o_logits_flat[7*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in8(o_logits_flat[8*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .in9(o_logits_flat[9*OUT_ACC_BITS +: OUT_ACC_BITS]),
        .idx(o_pred)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reduction_inflight_q <= 1'b0;
            pending_events_q <= 32'd0;
            pending_spikes_q <= '0;
            active_tile_mask_q <= '0;
            o_result_valid <= 1'b0;
            o_result_timestep <= '0;
            o_result_events <= 32'd0;
            o_result_spikes <= '0;
            o_logits_flat <= '0;
            local_protocol_error_q <= 1'b0;
        end else if (i_clear_state) begin
            reduction_inflight_q <= 1'b0;
            pending_events_q <= 32'd0;
            pending_spikes_q <= '0;
            active_tile_mask_q <= '0;
            o_result_valid <= 1'b0;
            o_result_timestep <= '0;
            o_result_events <= 32'd0;
            o_result_spikes <= '0;
            o_logits_flat <= '0;
            local_protocol_error_q <= 1'b0;
        end else begin
            if (o_result_valid && i_result_ready)
                o_result_valid <= 1'b0;

            if (i_step_start) begin
                if (!prefix_mask_legal(i_active_tile_mask))
                    local_protocol_error_q <= 1'b1;
                else if (o_step_ready)
                    active_tile_mask_q <= i_active_tile_mask;
            end

            if (reduction_start) begin
                reduction_inflight_q <= 1'b1;
                pending_events_q <= tile_events_flat[0 +: 32];
                for (integer pending_tile_i = 0;
                     pending_tile_i < TILES;
                     pending_tile_i = pending_tile_i + 1)
                    pending_spikes_q[
                        pending_tile_i*H_PER_TILE +: H_PER_TILE
                    ] <= active_tile_mask_q[pending_tile_i] ?
                        tile_spikes_flat[
                            pending_tile_i*H_PER_TILE +: H_PER_TILE
                        ] : '0;
            end

            // A clear invalidates the array transaction, so accept only a
            // completion that still has an authoritative in-flight request.
            if (reduction_valid && reduction_inflight_q) begin
                if (o_result_valid && !i_result_ready)
                    local_protocol_error_q <= 1'b1;
                reduction_inflight_q <= 1'b0;
                o_result_valid <= 1'b1;
                o_result_timestep <= reduction_timestep;
                o_result_events <= pending_events_q;
                o_result_spikes <= pending_spikes_q;
                o_logits_flat <= reduction_logits;
            end
        end
    end

    initial begin
        if (TILES < 1 || TILES > 12)
            $error("FC_COMPAT tile array supports 1..12 H40 tiles");
        if (O_DIM != 10 || OUT_ACC_BITS != 24)
            $error("FC_COMPAT argmax currently requires 10 ACC24 outputs");
    end
endmodule

`default_nettype wire
