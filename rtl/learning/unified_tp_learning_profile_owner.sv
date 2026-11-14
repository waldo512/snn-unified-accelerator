`timescale 1ns/1ps
`default_nettype none

// Serializes FC, SRNN and SCNN access to the shared learning resources.  A
// profile switch is accepted only after the active owner is quiescent.  The
// old owner is then released before the destination configuration is allowed
// to acquire the datapath and parameter-context service.
module unified_tp_learning_profile_owner #(
    parameter logic [7:0] PROFILE_FC_H160 = 8'h01,
    parameter logic [7:0] PROFILE_FC_H480 = 8'h02,
    parameter logic [7:0] PROFILE_SRNN_M2 = 8'h10,
    parameter logic [7:0] PROFILE_SCNN_VGG9 = 8'h20
) (
    input  logic       i_clk,
    input  logic       i_rst_n,
    input  logic       i_clear_error,

    input  logic       i_switch_valid,
    output logic       o_switch_ready,
    input  logic [7:0] i_switch_profile_id,
    output logic       o_switch_done,

    output logic       o_fc_acquire_valid,
    input  logic       i_fc_acquire_ready,
    output logic       o_srnn_acquire_valid,
    input  logic       i_srnn_acquire_ready,
    output logic       o_scnn_acquire_valid,
    input  logic       i_scnn_acquire_ready,

    output logic       o_fc_release_valid,
    input  logic       i_fc_release_ready,
    output logic       o_srnn_release_valid,
    input  logic       i_srnn_release_ready,
    output logic       o_scnn_release_valid,
    input  logic       i_scnn_release_ready,

    input  logic       i_fc_quiescent,
    input  logic       i_srnn_quiescent,
    input  logic       i_scnn_quiescent,
    input  logic       i_fc_error,
    input  logic       i_srnn_error,
    input  logic       i_scnn_error,

    output logic       o_active,
    output logic [7:0] o_active_profile_id,
    output logic [1:0] o_active_owner,
    output logic       o_fc_owner,
    output logic       o_srnn_owner,
    output logic       o_scnn_owner,
    output logic [15:0] o_profile_version,
    output logic       o_busy,
    output logic       o_sticky_error,
    output logic [7:0] o_error_code
);

    localparam logic [1:0] OWNER_NONE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_SRNN = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_ILLEGAL_PROFILE = 8'h01;
    localparam logic [7:0] ERROR_OWNER = 8'h02;

    typedef enum logic [1:0] {
        STATE_STABLE,
        STATE_RELEASE,
        STATE_ACQUIRE
    } state_t;

    state_t state_q;
    logic active_q;
    logic [1:0] active_owner_q;
    logic [7:0] active_profile_q;
    logic [1:0] pending_owner_q;
    logic [7:0] pending_profile_q;
    logic [15:0] profile_version_q;
    logic sticky_error_q;
    logic [7:0] error_code_q;

    logic switch_profile_valid;
    logic [1:0] switch_owner;
    logic switch_fire;
    logic active_owner_quiescent;
    logic selected_owner_error;
    logic release_fire;
    logic acquire_fire;

    always_comb begin
        switch_profile_valid = 1'b1;
        switch_owner = OWNER_NONE;
        unique case (i_switch_profile_id)
            PROFILE_FC_H160,
            PROFILE_FC_H480: switch_owner = OWNER_FC;
            PROFILE_SRNN_M2: switch_owner = OWNER_SRNN;
            PROFILE_SCNN_VGG9: switch_owner = OWNER_SCNN;
            default: switch_profile_valid = 1'b0;
        endcase

        unique case (active_owner_q)
            OWNER_FC: active_owner_quiescent = i_fc_quiescent;
            OWNER_SRNN: active_owner_quiescent = i_srnn_quiescent;
            OWNER_SCNN: active_owner_quiescent = i_scnn_quiescent;
            default: active_owner_quiescent = 1'b1;
        endcase

        selected_owner_error = 1'b0;
        unique case (state_q)
            STATE_RELEASE: begin
                unique case (active_owner_q)
                    OWNER_FC: selected_owner_error = i_fc_error;
                    OWNER_SRNN: selected_owner_error = i_srnn_error;
                    OWNER_SCNN: selected_owner_error = i_scnn_error;
                    default: selected_owner_error = 1'b0;
                endcase
            end
            STATE_ACQUIRE: begin
                unique case (pending_owner_q)
                    OWNER_FC: selected_owner_error = i_fc_error;
                    OWNER_SRNN: selected_owner_error = i_srnn_error;
                    OWNER_SCNN: selected_owner_error = i_scnn_error;
                    default: selected_owner_error = 1'b0;
                endcase
            end
            default: begin
                unique case (active_owner_q)
                    OWNER_FC: selected_owner_error = active_q && i_fc_error;
                    OWNER_SRNN: selected_owner_error = active_q && i_srnn_error;
                    OWNER_SCNN: selected_owner_error = active_q && i_scnn_error;
                    default: selected_owner_error = 1'b0;
                endcase
            end
        endcase

        o_switch_ready = (state_q == STATE_STABLE) && !sticky_error_q &&
            (!active_q || active_owner_quiescent);
        switch_fire = i_switch_valid && o_switch_ready;

        o_fc_acquire_valid = (state_q == STATE_ACQUIRE) &&
            (pending_owner_q == OWNER_FC) && !sticky_error_q;
        o_srnn_acquire_valid = (state_q == STATE_ACQUIRE) &&
            (pending_owner_q == OWNER_SRNN) && !sticky_error_q;
        o_scnn_acquire_valid = (state_q == STATE_ACQUIRE) &&
            (pending_owner_q == OWNER_SCNN) && !sticky_error_q;
        acquire_fire = (o_fc_acquire_valid && i_fc_acquire_ready) ||
            (o_srnn_acquire_valid && i_srnn_acquire_ready) ||
            (o_scnn_acquire_valid && i_scnn_acquire_ready);

        o_fc_release_valid = (state_q == STATE_RELEASE) &&
            (active_owner_q == OWNER_FC) && !sticky_error_q;
        o_srnn_release_valid = (state_q == STATE_RELEASE) &&
            (active_owner_q == OWNER_SRNN) && !sticky_error_q;
        o_scnn_release_valid = (state_q == STATE_RELEASE) &&
            (active_owner_q == OWNER_SCNN) && !sticky_error_q;
        release_fire = (o_fc_release_valid && i_fc_release_ready) ||
            (o_srnn_release_valid && i_srnn_release_ready) ||
            (o_scnn_release_valid && i_scnn_release_ready);

        o_active = active_q;
        o_active_profile_id = active_profile_q;
        o_active_owner = active_owner_q;
        o_fc_owner = active_q && (active_owner_q == OWNER_FC);
        o_srnn_owner = active_q && (active_owner_q == OWNER_SRNN);
        o_scnn_owner = active_q && (active_owner_q == OWNER_SCNN);
        o_profile_version = profile_version_q;
        o_busy = state_q != STATE_STABLE;
        o_sticky_error = sticky_error_q;
        o_error_code = error_code_q;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_STABLE;
            active_q <= 1'b0;
            active_owner_q <= OWNER_NONE;
            active_profile_q <= '0;
            pending_owner_q <= OWNER_NONE;
            pending_profile_q <= '0;
            profile_version_q <= '0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            o_switch_done <= 1'b0;
        end else begin
            o_switch_done <= 1'b0;

            if (i_clear_error && !selected_owner_error) begin
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
            end
            if (selected_owner_error) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_OWNER;
            end

            unique case (state_q)
                STATE_STABLE: begin
                    if (switch_fire) begin
                        if (!switch_profile_valid) begin
                            sticky_error_q <= 1'b1;
                            error_code_q <= ERROR_ILLEGAL_PROFILE;
                        end else if (active_q &&
                                (i_switch_profile_id == active_profile_q)) begin
                            o_switch_done <= 1'b1;
                        end else begin
                            pending_owner_q <= switch_owner;
                            pending_profile_q <= i_switch_profile_id;
                            if (active_q)
                                state_q <= STATE_RELEASE;
                            else
                                state_q <= STATE_ACQUIRE;
                        end
                    end
                end

                STATE_RELEASE: begin
                    if (release_fire) begin
                        active_q <= 1'b0;
                        active_owner_q <= OWNER_NONE;
                        active_profile_q <= '0;
                        state_q <= STATE_ACQUIRE;
                    end
                end

                STATE_ACQUIRE: begin
                    if (acquire_fire) begin
                        active_q <= 1'b1;
                        active_owner_q <= pending_owner_q;
                        active_profile_q <= pending_profile_q;
                        profile_version_q <= profile_version_q + 1'b1;
                        o_switch_done <= 1'b1;
                        state_q <= STATE_STABLE;
                    end
                end

                default: state_q <= STATE_STABLE;
            endcase
        end
    end

endmodule

`default_nettype wire
