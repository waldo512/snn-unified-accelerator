`timescale 1ns/1ps
`default_nettype none

// Serializes topology-native sample/timestep commands onto the single shared
// inference core. The accepted source and profile remain locked until the core
// reports completion and returns physically idle.
module unified_tp_shared_inference_run_arbiter (
    input  logic       i_clk,
    input  logic       i_rst_n,
    input  logic       i_clear_error,
    input  logic       i_abort,

    input  logic [2:0] i_active_client,
    input  logic [6:0] i_active_profile_code,

    input  logic       i_fc_start_valid,
    output logic       o_fc_start_ready,
    output logic       o_fc_done,
    output logic       o_fc_aborted,

    input  logic       i_srnn_start_valid,
    output logic       o_srnn_start_ready,
    output logic       o_srnn_done,
    output logic       o_srnn_aborted,

    input  logic       i_scnn_start_valid,
    output logic       o_scnn_start_ready,
    output logic       o_scnn_done,
    output logic       o_scnn_aborted,

    output logic       o_core_start,
    input  logic       i_core_start_ready,
    output logic [6:0] o_core_profile_code,
    output logic       o_core_abort,
    input  logic       i_core_sample_done,
    input  logic       i_core_global_idle,
    input  logic       i_core_error,

    output logic [2:0] o_active_source,
    output logic       o_busy,
    output logic       o_quiescent,
    output logic       o_protocol_error
);
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_ACTIVE,
        ST_DRAIN
    } state_t;

    state_t state_q;
    logic [2:0] source_q;
    logic [6:0] profile_q;
    logic abort_q;
    logic [2:0] request_vector;
    logic [2:0] selected_request;
    logic any_request;
    logic request_onehot;
    logic request_from_inactive;
    logic start_fire;
    logic normal_done;
    logic abort_done;
    logic owner_mismatch;

    assign request_vector = {
        i_scnn_start_valid, i_srnn_start_valid, i_fc_start_valid
    };
    assign selected_request = request_vector & i_active_client;
    assign any_request = |request_vector;
    assign request_onehot = $onehot(request_vector);
    assign request_from_inactive = |(request_vector & ~i_active_client);
    // The profile input is request context, not live execution state. Once a
    // command is accepted, profile_q is authoritative until physical drain.
    assign owner_mismatch = state_q != ST_IDLE && i_active_client != source_q;

    assign o_core_start = state_q == ST_IDLE && request_onehot &&
        !request_from_inactive && !o_protocol_error;
    assign o_core_profile_code = state_q == ST_IDLE ?
        i_active_profile_code : profile_q;
    assign start_fire = o_core_start && i_core_start_ready;

    assign o_fc_start_ready = state_q == ST_IDLE && i_active_client[0] &&
        !request_from_inactive && !o_protocol_error && i_core_start_ready;
    assign o_srnn_start_ready = state_q == ST_IDLE && i_active_client[1] &&
        !request_from_inactive && !o_protocol_error && i_core_start_ready;
    assign o_scnn_start_ready = state_q == ST_IDLE && i_active_client[2] &&
        !request_from_inactive && !o_protocol_error && i_core_start_ready;

    assign normal_done = state_q == ST_ACTIVE && i_core_sample_done &&
        !abort_q && !i_abort && !i_core_error;
    assign abort_done = state_q != ST_IDLE &&
        (abort_q || i_abort || i_core_error) && i_core_global_idle;
    assign o_fc_done = normal_done && source_q[0];
    assign o_srnn_done = normal_done && source_q[1];
    assign o_scnn_done = normal_done && source_q[2];
    assign o_fc_aborted = abort_done && source_q[0];
    assign o_srnn_aborted = abort_done && source_q[1];
    assign o_scnn_aborted = abort_done && source_q[2];

    // The final core accepts an abort event and retains its own cleanup state.
    // Keep the transaction locked here, but pulse the physical abort only once.
    assign o_core_abort = state_q != ST_IDLE && !abort_q &&
        (i_abort || i_core_error);
    assign o_active_source = source_q;
    assign o_busy = state_q != ST_IDLE;
    assign o_quiescent = state_q == ST_IDLE && i_core_global_idle &&
        !o_protocol_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            source_q <= '0;
            profile_q <= '0;
            abort_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_clear_error && state_q == ST_IDLE)
                o_protocol_error <= 1'b0;

            if ((state_q == ST_IDLE && any_request &&
                 (!request_onehot || request_from_inactive)) ||
                owner_mismatch ||
                (state_q != ST_IDLE && i_core_error &&
                 !abort_q && !i_abort) ||
                (state_q == ST_ACTIVE && i_core_global_idle &&
                 !i_core_sample_done && !abort_q && !i_abort) ||
                (state_q == ST_DRAIN && i_core_sample_done))
                o_protocol_error <= 1'b1;

            if (state_q != ST_IDLE && (i_abort || i_core_error))
                abort_q <= 1'b1;

            case (state_q)
                ST_IDLE: begin
                    source_q <= '0;
                    profile_q <= '0;
                    abort_q <= 1'b0;
                    if (start_fire) begin
                        source_q <= selected_request;
                        profile_q <= i_active_profile_code;
                        state_q <= ST_ACTIVE;
                    end
                end
                ST_ACTIVE: begin
                    if (abort_done) begin
                        source_q <= '0;
                        profile_q <= '0;
                        abort_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end else if (normal_done) begin
                        if (i_core_global_idle) begin
                            source_q <= '0;
                            profile_q <= '0;
                            state_q <= ST_IDLE;
                        end else begin
                            state_q <= ST_DRAIN;
                        end
                    end
                end
                ST_DRAIN: begin
                    if (abort_done) begin
                        source_q <= '0;
                        profile_q <= '0;
                        abort_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end else if (i_core_global_idle) begin
                        source_q <= '0;
                        profile_q <= '0;
                        state_q <= ST_IDLE;
                    end
                end
                default: begin
                    state_q <= ST_IDLE;
                    source_q <= '0;
                    profile_q <= '0;
                    abort_q <= 1'b0;
                    o_protocol_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
