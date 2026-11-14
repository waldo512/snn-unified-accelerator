`timescale 1ns/1ps
`default_nettype none

// FC-first event ingress leaf. Canonical FC windows are a combinational
// ready-valid pass-through. Recurrent FC owns one accepted window at a time
// and serializes its event lanes for the Wave3 scalar-event interface.
module unified_fc_first_event_ingress_adapter_v1 (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,
    input  logic                         i_abort,

    // 0: IDLE, 1: canonical FC, 2: recurrent FC, 3: SCNN.
    input  logic [1:0]                   i_requested_owner,
    output logic [1:0]                   o_active_owner,

    input  logic                         i_window_valid,
    output logic                         o_window_ready,
    input  logic [7:0]                   i_record_valid,
    input  logic [15:0]                  i_window_sample_id,
    input  logic [8:0]                   i_window_timestep,
    input  logic [7:0]                   i_boundary_mask,
    input  logic                         i_window_sample_last,
    input  logic [8*12-1:0]              i_fi_flat,
    input  logic [63:0]                  i_x_flat,

    // Direct canonical-FC window interface. These payload signals are never
    // registered or reordered by this leaf.
    output logic                         o_fc_window_valid,
    input  logic                         i_fc_window_ready,
    output logic [7:0]                   o_fc_record_valid,
    output logic [15:0]                  o_fc_window_sample_id,
    output logic [8:0]                   o_fc_window_timestep,
    output logic [7:0]                   o_fc_boundary_mask,
    output logic                         o_fc_window_sample_last,
    output logic [8*12-1:0]              o_fc_fi_flat,
    output logic [63:0]                  o_fc_x_flat,

    // Scalar recurrent-FC interface consumed by the Wave3 shared-extension
    // leaf. Boundary-only windows emit one neutral data=0 token.
    output logic                         o_srnn_event_valid,
    input  logic                         i_srnn_event_ready,
    output logic [12:0]                  o_srnn_event_addr,
    output logic signed [7:0]           o_srnn_event_data,
    output logic                         o_srnn_timestep_last,
    output logic                         o_srnn_sample_last,
    output logic                         o_srnn_boundary_token,
    output logic [15:0]                  o_srnn_window_sample_id,
    output logic [8:0]                   o_srnn_window_timestep,

    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_sticky_error,
    output logic [7:0]                   o_error_code
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_RECURRENT_FC = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER_SWITCH_BUSY = 8'h01;
    localparam logic [7:0] ERROR_IDLE_WINDOW = 8'h02;
    localparam logic [7:0] ERROR_SCNN_WINDOW = 8'h03;
    localparam logic [7:0] ERROR_BOUNDARY_SHAPE = 8'h04;

    logic [1:0] active_owner_q;
    logic recurrent_window_q;
    logic [7:0] recurrent_pending_mask_q;
    logic [8*12-1:0] recurrent_fi_flat_q;
    logic [63:0] recurrent_x_flat_q;
    logic recurrent_boundary_q;
    logic recurrent_sample_last_q;
    logic [15:0] recurrent_sample_id_q;
    logic [8:0] recurrent_timestep_q;

    logic [7:0] input_event_mask;
    logic input_boundary;
    logic input_shape_valid;
    logic [2:0] selected_lane;
    logic [7:0] selected_lane_mask;
    logic selected_is_last;
    logic recurrent_accept;
    logic recurrent_ready;
    logic recurrent_fire;
    logic owner_matches;

    function automatic logic [2:0] first_valid_lane(
        input logic [7:0] mask
    );
        integer lane;
        logic found;
        begin
            first_valid_lane = 3'd0;
            found = 1'b0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                if (mask[lane] && !found) begin
                    first_valid_lane = lane[2:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    always_comb begin
        input_event_mask = i_record_valid & ~i_boundary_mask;
        input_boundary = |(i_record_valid & i_boundary_mask);
        input_shape_valid = (i_boundary_mask & ~i_record_valid) == 8'd0;
        owner_matches = i_requested_owner == active_owner_q;

        // Canonical FC is intentionally a zero-buffer combinational path.
        o_fc_window_valid = !i_abort && i_window_valid && owner_matches &&
                            (active_owner_q == OWNER_FC);
        o_fc_record_valid = i_record_valid;
        o_fc_window_sample_id = i_window_sample_id;
        o_fc_window_timestep = i_window_timestep;
        o_fc_boundary_mask = i_boundary_mask;
        o_fc_window_sample_last = i_window_sample_last;
        o_fc_fi_flat = i_fi_flat;
        o_fc_x_flat = i_x_flat;

        recurrent_ready = !i_abort && owner_matches &&
            (active_owner_q == OWNER_RECURRENT_FC) &&
            !recurrent_window_q && !o_sticky_error && input_shape_valid;
        o_window_ready = 1'b0;
        if (!i_abort && owner_matches && (active_owner_q == OWNER_FC))
            o_window_ready = i_fc_window_ready;
        else if (recurrent_ready)
            o_window_ready = 1'b1;

        selected_lane = first_valid_lane(recurrent_pending_mask_q);
        selected_lane_mask = 8'b00000001 << selected_lane;
        selected_is_last =
            (recurrent_pending_mask_q & ~selected_lane_mask) == 8'd0;

        o_srnn_event_valid = recurrent_window_q && !i_abort;
        o_srnn_event_addr = 13'd0;
        o_srnn_event_data = 8'sd0;
        o_srnn_boundary_token =
            recurrent_window_q && (recurrent_pending_mask_q == 8'd0);
        if (recurrent_pending_mask_q != 8'd0) begin
            o_srnn_event_addr = {
                1'b0,
                recurrent_fi_flat_q[selected_lane*12 +: 12]
            };
            o_srnn_event_data = $signed(
                recurrent_x_flat_q[selected_lane*8 +: 8]
            );
            o_srnn_boundary_token = 1'b0;
        end
        o_srnn_timestep_last = recurrent_window_q &&
            recurrent_boundary_q &&
            ((recurrent_pending_mask_q == 8'd0) || selected_is_last);
        o_srnn_sample_last = o_srnn_timestep_last &&
                             recurrent_sample_last_q;
        o_srnn_window_sample_id = recurrent_sample_id_q;
        o_srnn_window_timestep = recurrent_timestep_q;

        // Keep the recurrent capture enable independent of the FC ready path.
        // Both expressions are handshake-equivalent in recurrent mode, but
        // this form prevents synthesis from building a cross-owner ready mux
        // into the recurrent payload register enables.
        recurrent_accept = i_window_valid && recurrent_ready;
        recurrent_fire = o_srnn_event_valid && i_srnn_event_ready;

        o_busy = !i_abort && (recurrent_window_q ||
                  (((active_owner_q == OWNER_FC) ||
                    (active_owner_q == OWNER_RECURRENT_FC)) &&
                   i_window_valid));
        o_idle = !o_busy;
        o_active_owner = active_owner_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_owner_q <= OWNER_IDLE;
            recurrent_window_q <= 1'b0;
            recurrent_pending_mask_q <= 8'd0;
            recurrent_fi_flat_q <= '0;
            recurrent_x_flat_q <= 64'd0;
            recurrent_boundary_q <= 1'b0;
            recurrent_sample_last_q <= 1'b0;
            recurrent_sample_id_q <= 16'd0;
            recurrent_timestep_q <= 9'd0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else if (i_abort) begin
            active_owner_q <= OWNER_IDLE;
            recurrent_window_q <= 1'b0;
            recurrent_pending_mask_q <= 8'd0;
            recurrent_fi_flat_q <= '0;
            recurrent_x_flat_q <= 64'd0;
            recurrent_boundary_q <= 1'b0;
            recurrent_sample_last_q <= 1'b0;
            recurrent_sample_id_q <= 16'd0;
            recurrent_timestep_q <= 9'd0;
        end else begin
            if (i_clear_error) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end

            if (!o_sticky_error &&
                (i_requested_owner != active_owner_q)) begin
                if (o_idle)
                    active_owner_q <= i_requested_owner;
                else begin
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_OWNER_SWITCH_BUSY;
                end
            end else if (!o_sticky_error && owner_matches &&
                         i_window_valid &&
                         (active_owner_q == OWNER_IDLE)) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_IDLE_WINDOW;
            end else if (!o_sticky_error && owner_matches &&
                         i_window_valid &&
                         (active_owner_q == OWNER_SCNN)) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_SCNN_WINDOW;
            end else if (!o_sticky_error && owner_matches &&
                         i_window_valid &&
                         (active_owner_q == OWNER_RECURRENT_FC) &&
                         !input_shape_valid) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_BOUNDARY_SHAPE;
            end

            if (recurrent_accept) begin
                recurrent_window_q <= 1'b1;
                recurrent_pending_mask_q <= input_event_mask;
                recurrent_fi_flat_q <= i_fi_flat;
                recurrent_x_flat_q <= i_x_flat;
                // A window with no event lane is itself a boundary token.
                recurrent_boundary_q <= input_boundary ||
                                        i_window_sample_last ||
                                        (input_event_mask == 8'd0);
                recurrent_sample_last_q <= i_window_sample_last;
                recurrent_sample_id_q <= i_window_sample_id;
                recurrent_timestep_q <= i_window_timestep;
            end else if (recurrent_fire) begin
                if ((recurrent_pending_mask_q == 8'd0) || selected_is_last) begin
                    recurrent_window_q <= 1'b0;
                    recurrent_pending_mask_q <= 8'd0;
                end else begin
                    recurrent_pending_mask_q <=
                        recurrent_pending_mask_q & ~selected_lane_mask;
                end
            end
        end
    end
endmodule

`default_nettype wire
