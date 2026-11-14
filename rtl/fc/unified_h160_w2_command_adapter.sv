`timescale 1ns/1ps
`default_nettype none

// Convert one global H160 update into one or two canonical five-class writes.
// A guarded command retires only after the final packed-SDP write cycle ends.
module unified_h160_w2_command_adapter (
    input  wire logic         clk,
    input  wire logic         rst_n,
    input  wire logic         i_guarded_valid,
    output logic              o_guarded_ready,
    output logic              o_command_complete,
    input  wire logic [31:0]  i_word_addr,
    input  wire logic [9:0]   i_lane_mask,
    input  wire logic signed [7:0] i_signed_delta,
    input  wire logic [15:0]  i_target_version,
    input  wire logic         i_commit_last,
    input  wire logic         i_active_bank,
    output logic [3:0]        o_cmd_valid,
    input  wire logic [3:0]   i_cmd_ready,
    output logic              o_cmd_target_bank,
    output logic [6:0]        o_cmd_local_word_addr,
    output logic [9:0]        o_cmd_deltas,
    output logic [15:0]       o_cmd_update_version,
    output logic              o_cmd_update_last,
    output logic              o_protocol_error
);
    localparam logic [2:0] S_IDLE = 3'd0;
    localparam logic [2:0] S_ISSUE_LOW = 3'd1;
    localparam logic [2:0] S_WAIT_LOW = 3'd2;
    localparam logic [2:0] S_ISSUE_HIGH = 3'd3;
    localparam logic [2:0] S_WAIT_HIGH = 3'd4;
    localparam logic [2:0] S_COMPLETE = 3'd5;

    logic [2:0] state_q;
    logic lower_selected;
    logic upper_selected;
    logic selected_upper;
    logic [1:0] cluster_index;
    logic [5:0] local_hidden;
    logic [4:0] selected_mask;
    logic [1:0] encoded_delta;
    logic selected_ready;
    logic issue_fire;
    logic guarded_payload_legal;
    integer lane;

    always_comb begin
        lower_selected = |i_lane_mask[4:0];
        upper_selected = |i_lane_mask[9:5];
        selected_upper = state_q == S_ISSUE_HIGH;
        if (i_word_addr < 32'd40) begin
            cluster_index = 2'd0;
            local_hidden = i_word_addr[5:0];
        end else if (i_word_addr < 32'd80) begin
            cluster_index = 2'd1;
            local_hidden = i_word_addr[5:0] - 6'd40;
        end else if (i_word_addr < 32'd120) begin
            cluster_index = 2'd2;
            local_hidden = i_word_addr[5:0] - 6'd16;
        end else begin
            cluster_index = 2'd3;
            local_hidden = i_word_addr[5:0] - 6'd56;
        end
        selected_mask = selected_upper ? i_lane_mask[9:5] : i_lane_mask[4:0];
        encoded_delta = (i_signed_delta == 8'sd1) ? 2'b01 : 2'b11;
        guarded_payload_legal = (i_word_addr < 32'd160) &&
            (i_lane_mask != 10'd0) &&
            ((i_signed_delta == 8'sd1) ||
             (i_signed_delta == -8'sd1));

        o_cmd_valid = 4'b0000;
        if (i_guarded_valid && guarded_payload_legal &&
            ((state_q == S_ISSUE_LOW) || (state_q == S_ISSUE_HIGH)))
            o_cmd_valid[cluster_index] = 1'b1;
        selected_ready = i_cmd_ready[cluster_index];
        issue_fire = (|o_cmd_valid) && selected_ready;
        o_guarded_ready = (state_q == S_COMPLETE) && i_guarded_valid;
        o_command_complete = o_guarded_ready;
        o_cmd_target_bank = ~i_active_bank;
        o_cmd_local_word_addr = {local_hidden, selected_upper};
        o_cmd_deltas = 10'd0;
        for (lane = 0; lane < 5; lane = lane + 1)
            if (selected_mask[lane])
                o_cmd_deltas[lane*2 +: 2] = encoded_delta;
        o_cmd_update_version = i_target_version;
        o_cmd_update_last = i_commit_last &&
            (selected_upper || !upper_selected);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= S_IDLE;
            o_protocol_error <= 1'b0;
        end else begin
            case (state_q)
                S_IDLE: begin
                    if (i_guarded_valid) begin
                        if (!guarded_payload_legal) begin
                            o_protocol_error <= 1'b1;
                        end else if (lower_selected) begin
                            state_q <= S_ISSUE_LOW;
                        end else begin
                            state_q <= S_ISSUE_HIGH;
                        end
                    end
                end
                S_ISSUE_LOW: begin
                    if (!i_guarded_valid || !guarded_payload_legal)
                        o_protocol_error <= 1'b1;
                    else if (issue_fire)
                        state_q <= S_WAIT_LOW;
                end
                S_WAIT_LOW: begin
                    if (!i_guarded_valid)
                        o_protocol_error <= 1'b1;
                    else if (selected_ready)
                        state_q <= upper_selected ? S_ISSUE_HIGH : S_COMPLETE;
                end
                S_ISSUE_HIGH: begin
                    if (!i_guarded_valid || !guarded_payload_legal)
                        o_protocol_error <= 1'b1;
                    else if (issue_fire)
                        state_q <= S_WAIT_HIGH;
                end
                S_WAIT_HIGH: begin
                    if (!i_guarded_valid)
                        o_protocol_error <= 1'b1;
                    else if (selected_ready)
                        state_q <= S_COMPLETE;
                end
                S_COMPLETE: begin
                    if (!i_guarded_valid)
                        o_protocol_error <= 1'b1;
                    else
                        state_q <= S_IDLE;
                end
                default: begin
                    state_q <= S_IDLE;
                    o_protocol_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
