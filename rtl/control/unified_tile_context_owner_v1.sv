`timescale 1ns/1ps
`default_nettype none

// Owns the single state/trace context resident in the homogeneous tile array.
// FC, SRNN and SCNN use the same activation protocol; topology changes only
// alter the context tag and geometry, never the physical state memories.
module unified_tile_context_owner_v1 #(
    parameter int CONTEXT_WIDTH = 64,
    parameter int EXT_ADDR_WIDTH = 32,
    parameter int ACTIVE_UNITS_WIDTH = 10,
    parameter int ROWS_PER_UNIT_WIDTH = 7
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,
    input  wire logic                              i_abort,
    input  wire logic                              i_clear_error,

    input  wire logic                              i_request_valid,
    output logic                                   o_request_ready,
    input  wire logic [CONTEXT_WIDTH-1:0]          i_request_context,
    input  wire logic [ACTIVE_UNITS_WIDTH-1:0]     i_request_active_units,
    input  wire logic [ROWS_PER_UNIT_WIDTH-1:0]    i_request_rows_per_unit,
    input  wire logic [EXT_ADDR_WIDTH-1:0]         i_request_backing_base,
    input  wire logic                              i_request_restore,
    input  wire logic                              i_request_save_on_evict,
    output logic                                   o_request_complete,

    input  wire logic                              i_release_valid,
    output logic                                   o_release_ready,
    input  wire logic                              i_release_writeback,
    output logic                                   o_release_complete,

    output logic                                   o_transport_command_valid,
    input  wire logic                              i_transport_command_ready,
    output logic [1:0]                             o_transport_command_op,
    output logic [CONTEXT_WIDTH-1:0]               o_transport_command_context,
    output logic [ACTIVE_UNITS_WIDTH-1:0]          o_transport_command_active_units,
    output logic [ROWS_PER_UNIT_WIDTH-1:0]         o_transport_command_rows_per_unit,
    output logic [EXT_ADDR_WIDTH-1:0]              o_transport_command_backing_base,
    output logic                                   o_transport_command_zero_initialize,
    input  wire logic                              i_transport_done,
    input  wire logic                              i_transport_busy,
    input  wire logic                              i_transport_sticky_error,
    input  wire logic                              i_transport_resident_valid,
    input  wire logic [CONTEXT_WIDTH-1:0]          i_transport_resident_context,
    input  wire logic [ACTIVE_UNITS_WIDTH-1:0]     i_transport_resident_active_units,
    input  wire logic [ROWS_PER_UNIT_WIDTH-1:0]    i_transport_resident_rows_per_unit,

    output logic                                   o_context_active,
    output logic [CONTEXT_WIDTH-1:0]               o_active_context,
    output logic [ACTIVE_UNITS_WIDTH-1:0]          o_active_units,
    output logic [ROWS_PER_UNIT_WIDTH-1:0]         o_active_rows_per_unit,
    output logic                                   o_busy,
    output logic                                   o_sticky_error
);
    localparam logic [1:0] COMMAND_ACQUIRE = 2'd0;
    localparam logic [1:0] COMMAND_RELEASE_WRITEBACK = 2'd1;
    localparam logic [1:0] COMMAND_RELEASE_DROP = 2'd2;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_EVICT_COMMAND,
        STATE_EVICT_WAIT,
        STATE_ACQUIRE_COMMAND,
        STATE_ACQUIRE_WAIT,
        STATE_RELEASE_COMMAND,
        STATE_RELEASE_WAIT
    } state_t;

    state_t state_q;
    logic [CONTEXT_WIDTH-1:0] pending_context_q;
    logic [ACTIVE_UNITS_WIDTH-1:0] pending_active_units_q;
    logic [ROWS_PER_UNIT_WIDTH-1:0] pending_rows_per_unit_q;
    logic [EXT_ADDR_WIDTH-1:0] pending_backing_base_q;
    logic pending_restore_q;
    logic pending_save_on_evict_q;

    logic current_metadata_valid_q;
    logic [CONTEXT_WIDTH-1:0] current_context_q;
    logic [ACTIVE_UNITS_WIDTH-1:0] current_active_units_q;
    logic [ROWS_PER_UNIT_WIDTH-1:0] current_rows_per_unit_q;
    logic [EXT_ADDR_WIDTH-1:0] current_backing_base_q;
    logic current_save_on_evict_q;
    logic release_writeback_q;
    logic local_error_q;

    logic request_fire;
    logic release_fire;
    logic exact_context_hit;
    logic context_id_hit;
    logic resident_metadata_consistent;
    logic transport_command_fire;

    assign context_id_hit = current_metadata_valid_q &&
        i_transport_resident_valid &&
        i_request_context == current_context_q &&
        i_request_context == i_transport_resident_context;
    assign exact_context_hit = context_id_hit &&
        i_request_active_units == current_active_units_q &&
        i_request_rows_per_unit == current_rows_per_unit_q &&
        i_request_backing_base == current_backing_base_q &&
        i_transport_resident_active_units == current_active_units_q &&
        i_transport_resident_rows_per_unit == current_rows_per_unit_q;
    assign resident_metadata_consistent =
        current_metadata_valid_q == i_transport_resident_valid &&
        (!current_metadata_valid_q ||
         (current_context_q == i_transport_resident_context &&
          current_active_units_q == i_transport_resident_active_units &&
          current_rows_per_unit_q == i_transport_resident_rows_per_unit));

    assign o_request_ready = !i_abort && !o_sticky_error &&
        state_q == STATE_IDLE && !i_transport_busy;
    assign o_release_ready = o_request_ready && !i_request_valid;
    assign request_fire = i_request_valid && o_request_ready;
    assign release_fire = i_release_valid && o_release_ready;

    assign o_transport_command_valid = !i_abort && !o_sticky_error &&
        (state_q == STATE_EVICT_COMMAND ||
         state_q == STATE_ACQUIRE_COMMAND ||
         state_q == STATE_RELEASE_COMMAND);
    assign transport_command_fire = o_transport_command_valid &&
        i_transport_command_ready;

    always_comb begin
        o_transport_command_op = COMMAND_ACQUIRE;
        o_transport_command_context = pending_context_q;
        o_transport_command_active_units = pending_active_units_q;
        o_transport_command_rows_per_unit = pending_rows_per_unit_q;
        o_transport_command_backing_base = pending_backing_base_q;
        o_transport_command_zero_initialize = !pending_restore_q;

        if (state_q == STATE_EVICT_COMMAND) begin
            o_transport_command_op = current_save_on_evict_q ?
                COMMAND_RELEASE_WRITEBACK : COMMAND_RELEASE_DROP;
            o_transport_command_context = current_context_q;
            o_transport_command_active_units = current_active_units_q;
            o_transport_command_rows_per_unit = current_rows_per_unit_q;
            o_transport_command_backing_base = current_backing_base_q;
            o_transport_command_zero_initialize = 1'b0;
        end else if (state_q == STATE_RELEASE_COMMAND) begin
            o_transport_command_op = release_writeback_q ?
                COMMAND_RELEASE_WRITEBACK : COMMAND_RELEASE_DROP;
            o_transport_command_context = current_context_q;
            o_transport_command_active_units = current_active_units_q;
            o_transport_command_rows_per_unit = current_rows_per_unit_q;
            o_transport_command_backing_base = current_backing_base_q;
            o_transport_command_zero_initialize = 1'b0;
        end
    end

    assign o_context_active = state_q == STATE_IDLE &&
        current_metadata_valid_q && resident_metadata_consistent &&
        !o_sticky_error;
    assign o_active_context = current_context_q;
    assign o_active_units = current_active_units_q;
    assign o_active_rows_per_unit = current_rows_per_unit_q;
    assign o_busy = state_q != STATE_IDLE || i_transport_busy;
    assign o_sticky_error = local_error_q || i_transport_sticky_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            pending_context_q <= '0;
            pending_active_units_q <= '0;
            pending_rows_per_unit_q <= '0;
            pending_backing_base_q <= '0;
            pending_restore_q <= 1'b0;
            pending_save_on_evict_q <= 1'b0;
            current_metadata_valid_q <= 1'b0;
            current_context_q <= '0;
            current_active_units_q <= '0;
            current_rows_per_unit_q <= '0;
            current_backing_base_q <= '0;
            current_save_on_evict_q <= 1'b0;
            release_writeback_q <= 1'b0;
            local_error_q <= 1'b0;
            o_request_complete <= 1'b0;
            o_release_complete <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            current_metadata_valid_q <= 1'b0;
            local_error_q <= 1'b0;
            o_request_complete <= 1'b0;
            o_release_complete <= 1'b0;
        end else begin
            o_request_complete <= 1'b0;
            o_release_complete <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE &&
                !i_transport_busy)
                local_error_q <= 1'b0;

            if (state_q == STATE_IDLE && !resident_metadata_consistent)
                local_error_q <= 1'b1;
            if (i_transport_sticky_error) begin
                state_q <= STATE_IDLE;
                current_metadata_valid_q <= 1'b0;
            end else begin
                case (state_q)
                    STATE_IDLE: begin
                        if (request_fire) begin
                            if (context_id_hit && !exact_context_hit) begin
                                local_error_q <= 1'b1;
                            end else if (exact_context_hit) begin
                                current_save_on_evict_q <=
                                    i_request_save_on_evict;
                                o_request_complete <= 1'b1;
                            end else begin
                                pending_context_q <= i_request_context;
                                pending_active_units_q <=
                                    i_request_active_units;
                                pending_rows_per_unit_q <=
                                    i_request_rows_per_unit;
                                pending_backing_base_q <=
                                    i_request_backing_base;
                                pending_restore_q <= i_request_restore;
                                pending_save_on_evict_q <=
                                    i_request_save_on_evict;
                                if (current_metadata_valid_q)
                                    state_q <= STATE_EVICT_COMMAND;
                                else
                                    state_q <= STATE_ACQUIRE_COMMAND;
                            end
                        end else if (release_fire) begin
                            if (!current_metadata_valid_q) begin
                                o_release_complete <= 1'b1;
                            end else begin
                                release_writeback_q <=
                                    i_release_writeback;
                                state_q <= STATE_RELEASE_COMMAND;
                            end
                        end
                    end

                    STATE_EVICT_COMMAND: begin
                        if (transport_command_fire)
                            state_q <= STATE_EVICT_WAIT;
                    end

                    STATE_EVICT_WAIT: begin
                        if (i_transport_done) begin
                            current_metadata_valid_q <= 1'b0;
                            state_q <= STATE_ACQUIRE_COMMAND;
                        end
                    end

                    STATE_ACQUIRE_COMMAND: begin
                        if (transport_command_fire)
                            state_q <= STATE_ACQUIRE_WAIT;
                    end

                    STATE_ACQUIRE_WAIT: begin
                        if (i_transport_done) begin
                            if (!i_transport_resident_valid ||
                                i_transport_resident_context !=
                                    pending_context_q ||
                                i_transport_resident_active_units !=
                                    pending_active_units_q ||
                                i_transport_resident_rows_per_unit !=
                                    pending_rows_per_unit_q) begin
                                local_error_q <= 1'b1;
                                current_metadata_valid_q <= 1'b0;
                            end else begin
                                current_metadata_valid_q <= 1'b1;
                                current_context_q <= pending_context_q;
                                current_active_units_q <=
                                    pending_active_units_q;
                                current_rows_per_unit_q <=
                                    pending_rows_per_unit_q;
                                current_backing_base_q <=
                                    pending_backing_base_q;
                                current_save_on_evict_q <=
                                    pending_save_on_evict_q;
                                o_request_complete <= 1'b1;
                            end
                            state_q <= STATE_IDLE;
                        end
                    end

                    STATE_RELEASE_COMMAND: begin
                        if (transport_command_fire)
                            state_q <= STATE_RELEASE_WAIT;
                    end

                    STATE_RELEASE_WAIT: begin
                        if (i_transport_done) begin
                            current_metadata_valid_q <= 1'b0;
                            o_release_complete <= 1'b1;
                            state_q <= STATE_IDLE;
                        end
                    end

                    default: begin
                        state_q <= STATE_IDLE;
                        current_metadata_valid_q <= 1'b0;
                        local_error_q <= 1'b1;
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
