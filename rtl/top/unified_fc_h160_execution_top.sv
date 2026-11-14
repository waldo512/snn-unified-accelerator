`timescale 1ns/1ps
`default_nettype none

// Complete trained H160 FC path behind the unified mode boundary. No FC
// transaction is translated into generic row commands: the verified P3A
// ingress, W0/LIF overlap, W2 reduction and argmax remain intact.
module unified_fc_h160_execution_top (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [1:0]                   i_mode_q,
    input  wire                         i_clear_error,
    input  wire                         i_sample_start,
    input  wire                         i_window_valid,
    output wire                         o_window_ready,
    input  wire [7:0]                   i_record_valid,
    input  wire [15:0]                  i_sample_id,
    input  wire [8:0]                   i_timestep,
    input  wire [7:0]                   i_boundary_mask,
    input  wire                         i_sample_last,
    input  wire [8*12-1:0]              i_fi_flat,
    input  wire [63:0]                  i_x_flat,
    output wire                         o_sample_done,
    output wire [3:0]                   o_pred,
    output wire [10*24-1:0]             o_logits_flat,
    output wire [4*40-1:0]              o_spikes_flat,
    output wire [4*40*18-1:0]           o_vmem_flat,
    output wire [4*40*16-1:0]           o_trace_flat,
    output wire [31:0]                  o_accepted_windows,
    output wire [31:0]                  o_accepted_events,
    output wire [31:0]                  o_issue_beats,
    output wire [31:0]                  o_process_beats,
    output wire [31:0]                  o_boundary_count,
    output wire [31:0]                  o_input_stall_cycles,
    output wire [31:0]                  o_output_stall_cycles,
    output wire [63:0]                  o_sample_cycles,
    output wire [63:0]                  o_boundary_wait_cycles,
    output wire [63:0]                  o_cluster_wait_cycles,
    output wire [63:0]                  o_reduction_wait_cycles,
    output wire                         o_protocol_error,
    output wire                         o_fc_active,
    output wire                         o_extension_active,
    output logic                        o_mode_error
);
    localparam logic [1:0] MODE_SPARSE_FC = 2'd0;

    logic mode_locked_q;
    logic sample_active_q;
    wire inner_sample_done;
    wire inner_window_ready;
    wire fc_selected = sample_active_q ? mode_locked_q :
                       (i_mode_q == MODE_SPARSE_FC);

    assign o_sample_done = inner_sample_done;
    assign o_window_ready = fc_selected && inner_window_ready;
    assign o_fc_active = sample_active_q;
    assign o_extension_active = 1'b0;

    unified_fc2312_h160_fixed_forward u_fc (
        .clk(clk), .rst_n(rst_n),
        .i_sample_start(i_sample_start && fc_selected),
        .i_window_valid(i_window_valid && fc_selected),
        .o_window_ready(inner_window_ready),
        .i_record_valid(i_record_valid), .i_sample_id(i_sample_id),
        .i_timestep(i_timestep), .i_boundary_mask(i_boundary_mask),
        .i_sample_last(i_sample_last), .i_fi_flat(i_fi_flat),
        .i_x_flat(i_x_flat), .o_sample_done(inner_sample_done),
        .o_pred(o_pred), .o_logits_flat(o_logits_flat),
        .o_spikes_flat(o_spikes_flat), .o_vmem_flat(o_vmem_flat),
        .o_trace_flat(o_trace_flat),
        .o_accepted_windows(o_accepted_windows),
        .o_accepted_events(o_accepted_events), .o_issue_beats(o_issue_beats),
        .o_process_beats(o_process_beats), .o_boundary_count(o_boundary_count),
        .o_input_stall_cycles(o_input_stall_cycles),
        .o_output_stall_cycles(o_output_stall_cycles),
        .o_sample_cycles(o_sample_cycles),
        .o_boundary_wait_cycles(o_boundary_wait_cycles),
        .o_cluster_wait_cycles(o_cluster_wait_cycles),
        .o_reduction_wait_cycles(o_reduction_wait_cycles),
        .o_protocol_error(o_protocol_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_locked_q <= 1'b0;
            sample_active_q <= 1'b0;
            o_mode_error <= 1'b0;
        end else begin
            if (i_clear_error)
                o_mode_error <= 1'b0;
            if (i_sample_start && !sample_active_q) begin
                if (i_mode_q == MODE_SPARSE_FC) begin
                    mode_locked_q <= 1'b1;
                    sample_active_q <= 1'b1;
                end else begin
                    mode_locked_q <= 1'b0;
                    o_mode_error <= 1'b1;
                end
            end
            if (sample_active_q && (i_mode_q != MODE_SPARSE_FC))
                o_mode_error <= 1'b1;
            if (inner_sample_done) begin
                mode_locked_q <= 1'b0;
                sample_active_q <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
