`timescale 1ns/1ps
`default_nettype none

// FC graph/control adapter for the single shared H40 array. This module owns
// descriptor validation, launch sequencing and the public result queue, but
// deliberately owns no neuron or arithmetic array.
module unified_fc_shared_h40_graph_adapter_v1 #(
    parameter int PHYSICAL_TILES = 5,
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
    output logic                         o_shared_abort,

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
    output logic [7:0]                   o_error_code
);
    localparam logic [31:0] LAYER_MAGIC = 32'h4e4d5232;
    localparam logic [31:0] ABI_V21 = 32'h000c0102;
    localparam logic [31:0] ABI_V22 = 32'h000c0202;
    localparam logic [31:0] ABI_V23 = 32'h000c0302;
    localparam logic [3:0] LAYER_SPARSE_FC = 4'd1;
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_DESCRIPTOR = 8'h01;
    localparam logic [7:0] ERROR_LAYER_WHILE_BUSY = 8'h02;
    localparam logic [7:0] ERROR_SAMPLE_ID = 8'h03;
    localparam logic [7:0] ERROR_RESULT_OVERRUN = 8'h04;
    localparam logic [7:0] ERROR_FC_PROTOCOL = 8'h05;

    logic [31:0] word0;
    logic [31:0] word1;
    logic [31:0] word2;
    logic [31:0] word3;
    logic [31:0] word4;
    logic [12:0] descriptor_input_dim;
    logic [9:0] descriptor_state_dim;
    logic [5:0] descriptor_output_dim;
    logic [4:0] descriptor_active_tiles;
    logic [3:0] descriptor_issue_k;
    logic descriptor_legal;
    logic layer_fire;

    logic launch_pending_q;
    logic windows_enabled_q;
    logic clear_state_q;
    logic [15:0] sample_id_q;
    logic [12:0] active_input_dim_q;
    logic [ENGINES-1:0] active_tile_mask_q;
    logic [11:0] active_cluster_mask_q;
    logic result_pending_q;
    logic [5:0] result_class_q;
    logic signed [31:0] result_score_q;
    logic [10*24-1:0] result_logits_q;
    logic sticky_error_q;
    logic [7:0] error_code_q;
    logic clear_error_accept;
    logic shared_start_fire;
    logic window_fire;
    logic sample_id_match;
    logic sample_id_mismatch;
    logic shared_abort_q;

    function automatic logic [ENGINES-1:0] tile_prefix_mask(
        input logic [4:0] tile_count
    );
        integer tile_i;
        begin
            tile_prefix_mask = '0;
            for (tile_i = 0; tile_i < ENGINES; tile_i = tile_i + 1)
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
        word0 = i_layer_words[0*32 +: 32];
        word1 = i_layer_words[1*32 +: 32];
        word2 = i_layer_words[2*32 +: 32];
        word3 = i_layer_words[3*32 +: 32];
        word4 = i_layer_words[4*32 +: 32];
        descriptor_input_dim = word3[12:0];
        descriptor_state_dim = word3[22:13];
        descriptor_output_dim = word3[28:23];
        descriptor_active_tiles = word4[4:0];
        descriptor_issue_k = word4[17:14];
        descriptor_legal = word0 == LAYER_MAGIC &&
            (word1 == ABI_V21 || word1 == ABI_V22 || word1 == ABI_V23) &&
            word2[3:0] == LAYER_SPARSE_FC &&
            descriptor_input_dim != 0 &&
            descriptor_input_dim <= INPUT_MAX &&
            descriptor_active_tiles != 0 &&
            descriptor_active_tiles <= PHYSICAL_TILES &&
            descriptor_active_tiles <= ENGINES &&
            descriptor_state_dim == 10'(descriptor_active_tiles * 40) &&
            descriptor_output_dim == 6'd10 &&
            descriptor_issue_k == 4'd8;
    end

    assign o_layer_ready = !i_abort && !launch_pending_q &&
        !windows_enabled_q && !result_pending_q && !sticky_error_q &&
        !i_shared_protocol_error && i_shared_sample_ready;
    assign layer_fire = i_layer_valid && o_layer_ready;

    assign o_shared_sample_start = !i_abort && launch_pending_q;
    assign shared_start_fire = o_shared_sample_start &&
        i_shared_sample_ready;
    assign o_shared_sample_clear_state = clear_state_q;
    assign o_shared_active_input_dim = active_input_dim_q;
    assign o_shared_requested_tile_mask = active_tile_mask_q;

    assign sample_id_match = i_window_sample_id == sample_id_q;
    assign o_window_ready = !i_abort && windows_enabled_q &&
        i_shared_window_ready;
    assign window_fire = i_window_valid && o_window_ready;
    assign sample_id_mismatch = window_fire && !sample_id_match;
    assign o_shared_abort = shared_abort_q;
    assign o_shared_window_valid = !i_abort && windows_enabled_q &&
        sample_id_match && i_window_valid;
    assign o_shared_record_valid = i_record_valid;
    assign o_shared_sample_id = i_window_sample_id;
    assign o_shared_window_timestep = i_window_timestep;
    assign o_shared_boundary_mask = i_boundary_mask;
    assign o_shared_sample_last = i_window_sample_last;
    assign o_shared_fi_flat = i_fi_flat;
    assign o_shared_x_flat = i_x_flat;

    assign o_timestep_done = i_shared_timestep_done;
    assign o_layer_compute_done = i_shared_sample_done;
    assign o_result_valid = !i_abort && result_pending_q;
    assign o_result_class = result_class_q;
    assign o_result_score = result_score_q;
    assign o_logits_flat = result_logits_q;
    assign o_sample_cycles = i_shared_sample_cycles[31:0];
    assign o_accepted_windows = i_shared_accepted_windows;
    assign o_accepted_events = i_shared_accepted_events;
    assign o_issue_groups = i_shared_issue_groups;
    assign o_boundary_count = i_shared_boundary_count;
    assign o_runtime_dropped_events = i_shared_dropped_events;
    assign o_locked_cluster_mask = active_cluster_mask_q;
    assign o_locked_input_dim = active_input_dim_q;
    assign o_busy = launch_pending_q || windows_enabled_q ||
        result_pending_q || i_shared_fc_active;
    assign o_idle = !o_busy && !sticky_error_q &&
        !i_shared_protocol_error;
    assign o_sticky_error = sticky_error_q || i_shared_protocol_error;
    assign o_error_code = sticky_error_q ? error_code_q :
        i_shared_protocol_error ? ERROR_FC_PROTOCOL : ERROR_NONE;
    assign clear_error_accept = i_clear_error && !o_busy;

    // Reserved TP-learning boundary. The shared inference implementation does
    // not claim writable resident weights or a committed learning bank yet.
    assign o_active_bank = 1'b0;
    assign o_active_version = 16'd0;
    assign o_update_window_ready = 1'b0;
    assign o_commit_pending = 1'b0;
    assign o_snapshot_busy = 1'b0;
    assign o_debug_active_word = 40'd0;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            launch_pending_q <= 1'b0;
            windows_enabled_q <= 1'b0;
            clear_state_q <= 1'b1;
            sample_id_q <= '0;
            active_input_dim_q <= 13'(INPUT_MAX);
            active_tile_mask_q <= '0;
            active_cluster_mask_q <= '0;
            result_pending_q <= 1'b0;
            result_class_q <= '0;
            result_score_q <= '0;
            result_logits_q <= '0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            shared_abort_q <= 1'b0;
        end else if (i_abort) begin
            launch_pending_q <= 1'b0;
            windows_enabled_q <= 1'b0;
            result_pending_q <= 1'b0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            shared_abort_q <= 1'b0;
        end else begin
            shared_abort_q <= 1'b0;
            if (clear_error_accept) begin
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
            end

            if (layer_fire) begin
                if (!descriptor_legal) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_DESCRIPTOR;
                end else begin
                    launch_pending_q <= 1'b1;
                    clear_state_q <= i_clear_state;
                    sample_id_q <= i_start_sample_id;
                    active_input_dim_q <= descriptor_input_dim;
                    active_tile_mask_q <=
                        tile_prefix_mask(descriptor_active_tiles);
                    active_cluster_mask_q <=
                        cluster_prefix_mask(descriptor_active_tiles);
                end
            end else if (i_layer_valid && !o_layer_ready &&
                         !sticky_error_q) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_LAYER_WHILE_BUSY;
            end

            if (shared_start_fire) begin
                launch_pending_q <= 1'b0;
                windows_enabled_q <= 1'b1;
            end

            if (result_pending_q && i_result_ready)
                result_pending_q <= 1'b0;

            if (i_shared_sample_done) begin
                windows_enabled_q <= 1'b0;
                if (result_pending_q && !i_result_ready) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_RESULT_OVERRUN;
                end else begin
                    result_pending_q <= 1'b1;
                    result_class_q <= {2'b00, i_shared_pred};
                    result_score_q <= selected_score(
                        i_shared_logits_flat, i_shared_pred);
                    result_logits_q <= i_shared_logits_flat;
                end
            end

            if (i_shared_protocol_error && !clear_error_accept) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_FC_PROTOCOL;
            end

            if (sample_id_mismatch) begin
                launch_pending_q <= 1'b0;
                windows_enabled_q <= 1'b0;
                result_pending_q <= 1'b0;
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_SAMPLE_ID;
                shared_abort_q <= 1'b1;
            end
        end
    end

    logic unused_debug;
    assign unused_debug = &{1'b0, i_debug_cluster, i_debug_word_addr};

    initial begin
        if (PHYSICAL_TILES < 1 || PHYSICAL_TILES > ENGINES || ENGINES > 12)
            $error("shared-H40 FC owner requires 1..ENGINES<=12 tiles");
    end
endmodule

`default_nettype wire
