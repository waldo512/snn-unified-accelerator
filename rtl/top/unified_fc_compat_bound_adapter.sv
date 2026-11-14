`timescale 1ns/1ps
`default_nettype none

// Runtime boundary around the paper-oriented FC_COMPAT inference path. The
// current implementation uses resident inference weights; the update/version
// ports deliberately remain inactive until the TP learning service is bound.
module unified_fc_compat_bound_adapter #(
    parameter integer PHYSICAL_TILES = 5,
    parameter WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312"
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,
    input  logic                         i_abort,

    input  logic                         i_start,
    output logic                         o_start_ready,
    input  logic                         i_clear_state,
    input  logic [15:0]                  i_start_sample_id,
    input  logic [12:0]                  i_active_input_dim,
    input  logic [4:0]                   i_active_h40_tiles,

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
    output logic                         o_compute_done,
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
    output logic                         o_global_idle,
    output logic                         o_busy,
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
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_ILLEGAL_CONFIG = 8'h01;
    localparam logic [7:0] ERROR_START_NOT_READY = 8'h02;
    localparam logic [7:0] ERROR_FC_PROTOCOL = 8'h03;
    localparam logic [7:0] ERROR_RESULT_OVERRUN = 8'h04;
    localparam logic [7:0] ERROR_SAMPLE_ID = 8'h05;

    logic launch_pending_q;
    logic core_start_q;
    logic clear_state_q;
    logic windows_enabled_q;
    logic [15:0] sample_id_q;
    logic [12:0] active_input_dim_q;
    logic [PHYSICAL_TILES-1:0] active_tile_mask_q;
    logic [11:0] active_cluster_mask_q;
    logic result_pending_q;
    logic [5:0] result_class_q;
    logic signed [31:0] result_score_q;
    logic [10*24-1:0] result_logits_q;
    logic sticky_error_q;
    logic [7:0] error_code_q;

    logic config_legal;
    logic [PHYSICAL_TILES-1:0] requested_tile_mask;
    logic [11:0] requested_cluster_mask;
    logic core_sample_ready;
    logic core_window_ready;
    logic core_sample_done;
    logic [3:0] core_pred;
    logic [10*24-1:0] core_logits_flat;
    logic [63:0] core_sample_cycles;
    logic core_global_idle;
    logic core_protocol_error;
    logic window_fire;
    logic sample_id_match;
    logic sample_id_mismatch;
    logic core_abort;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [PHYSICAL_TILES*40-1:0] core_spikes_flat;
    logic [PHYSICAL_TILES*40*18-1:0] core_vmem_flat;
    logic [PHYSICAL_TILES*40*16-1:0] core_trace_flat;
    /* verilator lint_on UNUSEDSIGNAL */

    function automatic logic [PHYSICAL_TILES-1:0] tile_prefix_mask(
        input logic [4:0] tile_count
    );
        integer tile_i;
        begin
            tile_prefix_mask = '0;
            for (tile_i = 0; tile_i < PHYSICAL_TILES; tile_i = tile_i + 1)
                if (tile_i < tile_count)
                    tile_prefix_mask[tile_i] = 1'b1;
        end
    endfunction

    function automatic logic [11:0] cluster_prefix_mask(
        input logic [4:0] tile_count
    );
        integer tile_i;
        begin
            cluster_prefix_mask = '0;
            for (tile_i = 0; tile_i < 12; tile_i = tile_i + 1)
                if (tile_i < tile_count)
                    cluster_prefix_mask[tile_i] = 1'b1;
        end
    endfunction

    function automatic logic signed [31:0] selected_score(
        input logic [10*24-1:0] logits,
        input logic [3:0] prediction
    );
        logic signed [23:0] score24;
        begin
            score24 = logits[prediction*24 +: 24];
            selected_score = {{8{score24[23]}}, score24};
        end
    endfunction

    always_comb begin
        requested_tile_mask = tile_prefix_mask(i_active_h40_tiles);
        requested_cluster_mask = cluster_prefix_mask(i_active_h40_tiles);
        config_legal = i_active_input_dim != 0 &&
            i_active_input_dim <= 13'd2312 &&
            i_active_h40_tiles != 0 &&
            i_active_h40_tiles <= PHYSICAL_TILES;
        o_start_ready = core_global_idle && core_sample_ready &&
            !launch_pending_q && !result_pending_q &&
            !sticky_error_q && config_legal;
    end

    assign sample_id_match = i_window_sample_id == sample_id_q;
    assign o_window_ready = windows_enabled_q && core_window_ready;
    assign window_fire = i_window_valid && o_window_ready;
    assign sample_id_mismatch = window_fire && !sample_id_match;
    assign core_abort = i_abort || sample_id_mismatch;
    assign o_compute_done = core_sample_done;
    assign o_result_valid = result_pending_q;
    assign o_result_class = result_class_q;
    assign o_result_score = result_score_q;
    assign o_logits_flat = result_logits_q;
    assign o_sample_cycles = core_sample_cycles[31:0];
    assign o_global_idle = core_global_idle && !launch_pending_q &&
        !result_pending_q;
    assign o_busy = !o_global_idle;
    assign o_locked_cluster_mask = active_cluster_mask_q;
    assign o_locked_input_dim = active_input_dim_q;
    assign o_sticky_error = sticky_error_q || core_protocol_error;
    assign o_error_code = sticky_error_q ? error_code_q :
        core_protocol_error ? ERROR_FC_PROTOCOL : ERROR_NONE;

    // Reserved TP learning boundary. Inference-only integration never claims
    // that resident weights are writable or version switched.
    assign o_active_bank = 1'b0;
    assign o_active_version = 16'd0;
    assign o_update_window_ready = 1'b0;
    assign o_commit_pending = 1'b0;
    assign o_snapshot_busy = 1'b0;
    assign o_debug_active_word = 40'd0;

    unified_fc_compat_h160_window_top #(
        .TILES(PHYSICAL_TILES),
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_fc (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_abort(core_abort),
        .i_clear_state(clear_state_q),
        .i_sample_start(core_start_q),
        .o_sample_ready(core_sample_ready),
        .i_active_input_dim(active_input_dim_q),
        .i_active_tile_mask(active_tile_mask_q),
        .i_window_valid(i_window_valid && windows_enabled_q &&
                        sample_id_match),
        .o_window_ready(core_window_ready),
        .i_record_valid(i_record_valid),
        .i_sample_id(i_window_sample_id),
        .i_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_sample_last(i_window_sample_last),
        .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat),
        .o_timestep_done(o_timestep_done),
        .o_sample_done(core_sample_done),
        .o_completed_timestep(),
        .o_logits_flat(core_logits_flat),
        .o_pred(core_pred),
        .o_spikes_flat(core_spikes_flat),
        .o_vmem_flat(core_vmem_flat),
        .o_trace_flat(core_trace_flat),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups),
        .o_boundary_count(o_boundary_count),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_sample_cycles(core_sample_cycles),
        .o_global_idle(core_global_idle),
        .o_protocol_error(core_protocol_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            launch_pending_q <= 1'b0;
            core_start_q <= 1'b0;
            clear_state_q <= 1'b1;
            windows_enabled_q <= 1'b0;
            sample_id_q <= '0;
            active_input_dim_q <= 13'd2312;
            active_tile_mask_q <= {PHYSICAL_TILES{1'b1}};
            active_cluster_mask_q <= '0;
            result_pending_q <= 1'b0;
            result_class_q <= '0;
            result_score_q <= '0;
            result_logits_q <= '0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
        end else if (i_abort) begin
            launch_pending_q <= 1'b0;
            core_start_q <= 1'b0;
            windows_enabled_q <= 1'b0;
            result_pending_q <= 1'b0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
        end else begin
            core_start_q <= 1'b0;

            if (core_start_q)
                windows_enabled_q <= 1'b1;
            if (core_sample_done)
                windows_enabled_q <= 1'b0;

            if (i_clear_error && o_global_idle && !core_protocol_error) begin
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
            end

            if (i_start) begin
                if (!config_legal) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_ILLEGAL_CONFIG;
                end else if (!o_start_ready) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_START_NOT_READY;
                end else begin
                    launch_pending_q <= 1'b1;
                    clear_state_q <= i_clear_state;
                    sample_id_q <= i_start_sample_id;
                    active_input_dim_q <= i_active_input_dim;
                    active_tile_mask_q <= requested_tile_mask;
                    active_cluster_mask_q <= requested_cluster_mask;
                end
            end

            if (launch_pending_q && core_sample_ready) begin
                launch_pending_q <= 1'b0;
                core_start_q <= 1'b1;
            end

            if (result_pending_q && i_result_ready)
                result_pending_q <= 1'b0;

            if (core_sample_done) begin
                if (result_pending_q && !i_result_ready) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_RESULT_OVERRUN;
                end else begin
                    result_pending_q <= 1'b1;
                    result_class_q <= {2'b00, core_pred};
                    result_score_q <= selected_score(
                        core_logits_flat, core_pred);
                    result_logits_q <= core_logits_flat;
                end
            end

            if (core_protocol_error) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_FC_PROTOCOL;
            end

            if (sample_id_mismatch) begin
                launch_pending_q <= 1'b0;
                windows_enabled_q <= 1'b0;
                result_pending_q <= 1'b0;
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_SAMPLE_ID;
            end
        end
    end

    logic unused_inputs;
    assign unused_inputs = &{1'b0, i_debug_cluster, i_debug_word_addr};

    initial begin
        if (PHYSICAL_TILES < 1 || PHYSICAL_TILES > 12)
            $error("FC_COMPAT bound adapter supports 1..12 H40 tiles");
    end
endmodule

`default_nettype wire
