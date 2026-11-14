`timescale 1ns/1ps
`default_nettype none

// Boundary around the sole canonical H480 FC implementation. Runtime profile
// control is kept outside the FC datapath: it is sampled before the canonical
// sample-start pulse and only changes the active H40 prefix mask.
module unified_fc_h480_bound_adapter #(
    parameter WEIGHT_DIR =
        "../../../data/weights/r19b_h480_complete_fc_input2312"
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,

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

    logic launch_pending_q;
    logic clear_before_launch_q;
    logic fc_start_q;
    logic fc_clear_state_q;
    logic windows_enabled_q;
    logic [15:0] sample_id_q;
    logic [12:0] active_input_dim_q;
    logic [11:0] active_cluster_mask_q;
    logic result_pending_q;
    logic [5:0] result_class_q;
    logic signed [31:0] result_score_q;
    logic [10*24-1:0] result_logits_q;
    logic sticky_error_q;
    logic [7:0] error_code_q;

    logic config_legal;
    logic [11:0] requested_cluster_mask;
    logic fc_sample_ready;
    logic fc_window_ready;
    logic fc_sample_done;
    logic [3:0] fc_pred;
    logic [10*24-1:0] fc_logits_flat;
    logic fc_global_idle;
    logic fc_protocol_error;

    function automatic logic [11:0] prefix_mask(
        input logic [4:0] tile_count
    );
        integer tile_index;
        begin
            prefix_mask = 12'h000;
            for (tile_index = 0; tile_index < 12; tile_index = tile_index + 1)
                if (tile_index < tile_count)
                    prefix_mask[tile_index] = 1'b1;
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
        requested_cluster_mask = prefix_mask(i_active_h40_tiles);
        config_legal = (i_active_input_dim != 13'd0) &&
                       (i_active_input_dim <= 13'd2312) &&
                       (i_active_h40_tiles != 5'd0) &&
                       (i_active_h40_tiles <= 5'd12);
        o_start_ready = fc_global_idle && fc_sample_ready &&
                        !launch_pending_q && !result_pending_q &&
                        !sticky_error_q && config_legal;
    end

    assign o_window_ready = windows_enabled_q && fc_window_ready;
    assign o_compute_done = fc_sample_done;
    assign o_result_valid = result_pending_q;
    assign o_result_class = result_class_q;
    assign o_result_score = result_score_q;
    assign o_logits_flat = result_logits_q;
    assign o_global_idle = fc_global_idle && !launch_pending_q &&
                           !result_pending_q;
    assign o_busy = !o_global_idle;
    assign o_locked_cluster_mask = active_cluster_mask_q;
    assign o_locked_input_dim = active_input_dim_q;
    assign o_sticky_error = sticky_error_q || fc_protocol_error;
    assign o_error_code = sticky_error_q ? error_code_q :
                          fc_protocol_error ? ERROR_FC_PROTOCOL : ERROR_NONE;

    unified_fc2312_h480_complete #(
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_fc (
        .clk(i_clk),
        .rst_n(i_rst_n),
        .i_sample_start(fc_start_q),
        .o_sample_ready(fc_sample_ready),
        .i_clear_state(fc_clear_state_q),
        .i_start_sample_id(sample_id_q),
        .i_active_input_dim(active_input_dim_q),
        .i_active_cluster_mask(active_cluster_mask_q),
        .i_window_valid(i_window_valid && windows_enabled_q),
        .o_window_ready(fc_window_ready),
        .i_record_valid(i_record_valid),
        .i_window_sample_id(i_window_sample_id),
        .i_window_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_window_sample_last(i_window_sample_last),
        .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat),
        .o_timestep_done(o_timestep_done),
        .o_sample_done(fc_sample_done),
        .o_completed_timestep(),
        .o_logits_flat(fc_logits_flat),
        .o_pred(fc_pred),
        .o_sample_cycles(o_sample_cycles),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups),
        .o_boundary_count(o_boundary_count),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_global_idle(fc_global_idle),
        .o_active_bank(o_active_bank),
        .o_active_version(o_active_version),
        .o_update_window_ready(o_update_window_ready),
        .o_commit_pending(o_commit_pending),
        .o_snapshot_busy(o_snapshot_busy),
        .i_debug_cluster(i_debug_cluster),
        .i_debug_word_addr(i_debug_word_addr),
        .o_debug_active_word(o_debug_active_word),
        .o_protocol_error(fc_protocol_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            launch_pending_q <= 1'b0;
            clear_before_launch_q <= 1'b0;
            fc_start_q <= 1'b0;
            fc_clear_state_q <= 1'b0;
            windows_enabled_q <= 1'b0;
            sample_id_q <= '0;
            active_input_dim_q <= 13'd2312;
            active_cluster_mask_q <= 12'hfff;
            result_pending_q <= 1'b0;
            result_class_q <= '0;
            result_score_q <= '0;
            result_logits_q <= '0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
        end else begin
            fc_start_q <= 1'b0;
            fc_clear_state_q <= 1'b0;

            // Enable traffic only after the canonical block has sampled the
            // registered sample-start pulse on the preceding active edge.
            if (fc_start_q)
                windows_enabled_q <= 1'b1;
            if (fc_sample_done)
                windows_enabled_q <= 1'b0;

            if (i_clear_error && o_global_idle && !fc_protocol_error) begin
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
                    clear_before_launch_q <= i_clear_state;
                    sample_id_q <= i_start_sample_id;
                    active_input_dim_q <= i_active_input_dim;
                    active_cluster_mask_q <= requested_cluster_mask;
                end
            end

            if (launch_pending_q) begin
                if (clear_before_launch_q) begin
                    // The canonical H480 state clear must occupy a full clock
                    // before sample start.  Keeping it high during the sample
                    // would erase accumulated membrane/W2 state every cycle.
                    fc_clear_state_q <= 1'b1;
                    clear_before_launch_q <= 1'b0;
                end else if (fc_sample_ready) begin
                    launch_pending_q <= 1'b0;
                    fc_start_q <= 1'b1;
                end
            end

            if (result_pending_q && i_result_ready)
                result_pending_q <= 1'b0;

            if (fc_sample_done) begin
                if (result_pending_q && !i_result_ready) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_RESULT_OVERRUN;
                end else begin
                    result_pending_q <= 1'b1;
                    result_class_q <= {2'b00, fc_pred};
                    result_score_q <= selected_score(fc_logits_flat, fc_pred);
                    result_logits_q <= fc_logits_flat;
                end
            end

            if (fc_protocol_error) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_FC_PROTOCOL;
            end
        end
    end
endmodule

`default_nettype wire
