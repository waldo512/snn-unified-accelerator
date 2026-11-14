`timescale 1ns/1ps
`default_nettype none

// Descriptor boundary for the FC_COMPAT inference owner. It keeps the public
// FC owner protocol stable while replacing the private canonical H480 core.
module unified_fc_compat_graph_layer_adapter_v1 #(
    parameter integer PHYSICAL_TILES = 5,
    parameter WEIGHT_DIR =
        "data/weights/p3a_nmnist_h160_h480_active4_input2312"
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
    logic fc_start;
    logic fc_start_ready;
    logic fc_busy;
    logic fc_idle;
    logic fc_error;
    logic [7:0] fc_error_code;
    logic local_error_q;
    logic [7:0] local_error_code_q;

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
            descriptor_input_dim <= 13'd2312 &&
            descriptor_active_tiles != 0 &&
            descriptor_active_tiles <= PHYSICAL_TILES &&
            descriptor_state_dim == 10'(descriptor_active_tiles * 40) &&
            descriptor_output_dim == 6'd10 &&
            descriptor_issue_k == 4'd8;
        // An invalid descriptor is still admitted at an idle boundary so it
        // can be reported as a descriptor error rather than as false busy.
        o_layer_ready = fc_idle && !local_error_q && !fc_error;
        layer_fire = i_layer_valid && o_layer_ready;
        fc_start = layer_fire && descriptor_legal;
        o_busy = fc_busy;
        o_idle = fc_idle && !local_error_q && !fc_error;
        o_sticky_error = local_error_q || fc_error;
        o_error_code = local_error_q ? local_error_code_q :
            fc_error ? fc_error_code : ERROR_NONE;
    end

    unified_fc_compat_bound_adapter #(
        .PHYSICAL_TILES(PHYSICAL_TILES),
        .WEIGHT_DIR(WEIGHT_DIR)
    ) u_fc_adapter (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_clear_error(i_clear_error),
        .i_abort(i_abort),
        .i_start(fc_start), .o_start_ready(fc_start_ready),
        .i_clear_state(i_clear_state),
        .i_start_sample_id(i_start_sample_id),
        .i_active_input_dim(descriptor_input_dim),
        .i_active_h40_tiles(descriptor_active_tiles),
        .i_window_valid(i_window_valid), .o_window_ready(o_window_ready),
        .i_record_valid(i_record_valid),
        .i_window_sample_id(i_window_sample_id),
        .i_window_timestep(i_window_timestep),
        .i_boundary_mask(i_boundary_mask),
        .i_window_sample_last(i_window_sample_last),
        .i_fi_flat(i_fi_flat), .i_x_flat(i_x_flat),
        .o_timestep_done(o_timestep_done),
        .o_compute_done(o_layer_compute_done),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_class(o_result_class), .o_result_score(o_result_score),
        .o_logits_flat(o_logits_flat), .o_sample_cycles(o_sample_cycles),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events),
        .o_issue_groups(o_issue_groups),
        .o_boundary_count(o_boundary_count),
        .o_runtime_dropped_events(o_runtime_dropped_events),
        .o_global_idle(fc_idle), .o_busy(fc_busy),
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
        .o_sticky_error(fc_error), .o_error_code(fc_error_code)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else if (i_abort) begin
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else begin
            if (i_clear_error && fc_idle) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end
            if (layer_fire && !descriptor_legal && !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_DESCRIPTOR;
            end else if (layer_fire && descriptor_legal &&
                         !fc_start_ready && !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_LAYER_WHILE_BUSY;
            end else if (i_layer_valid && !o_layer_ready &&
                         !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_LAYER_WHILE_BUSY;
            end
        end
    end

    initial begin
        if (PHYSICAL_TILES < 1 || PHYSICAL_TILES > 12)
            $error("FC_COMPAT graph adapter supports 1..12 H40 tiles");
    end
endmodule

`default_nettype wire
