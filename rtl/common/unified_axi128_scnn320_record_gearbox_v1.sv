`timescale 1ns/1ps
`default_nettype none

// Shared bidirectional gearbox for SCNN state/trace worksets. Restore packs
// five 128-bit AXI beats into two 320-bit records; writeback performs the
// inverse mapping. Load and store are phase-exclusive, so one reset-free
// 640-bit payload buffer serves both directions.
module unified_axi128_scnn320_record_gearbox_v1 (
    input  wire logic          i_clk,
    input  wire logic          i_rst_n,
    input  wire logic          i_abort,
    input  wire logic          i_clear_error,

    input  wire logic          i_start_valid,
    output logic               o_start_ready,
    input  wire logic          i_direction_store,
    input  wire logic [31:0]   i_record_count,

    input  wire logic          i_axi_valid,
    output logic               o_axi_ready,
    input  wire logic [127:0]  i_axi_data,
    input  wire logic          i_axi_last,
    output logic               o_axi_valid,
    input  wire logic          i_axi_ready,
    output logic [127:0]       o_axi_data,
    output logic               o_axi_last,

    input  wire logic          i_record_valid,
    output logic               o_record_ready,
    input  wire logic [319:0]  i_record_data,
    input  wire logic          i_record_last,
    output logic               o_record_valid,
    input  wire logic          i_record_ready,
    output logic [319:0]       o_record_data,
    output logic               o_record_last,

    output logic               o_busy,
    output logic               o_done,
    output logic               o_sticky_error
);
    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_LOAD_CAPTURE,
        STATE_LOAD_EMIT,
        STATE_STORE_CAPTURE,
        STATE_STORE_EMIT
    } state_t;

    state_t state_q;
    logic [639:0] payload_q;
    logic [31:0] records_remaining_q;
    logic [2:0] beat_q;
    logic record_slot_q;
    logic start_fire;
    logic axi_input_fire;
    logic axi_output_fire;
    logic record_input_fire;
    logic record_output_fire;
    logic start_legal;
    logic expected_axi_input_last;
    logic expected_record_input_last;

    assign start_legal = i_record_count != 0 && !i_record_count[0];
    assign o_start_ready = !i_abort && !o_sticky_error &&
        state_q == STATE_IDLE;
    assign start_fire = i_start_valid && o_start_ready;
    assign o_busy = state_q != STATE_IDLE;

    assign o_axi_ready = !i_abort && state_q == STATE_LOAD_CAPTURE;
    assign axi_input_fire = i_axi_valid && o_axi_ready;
    assign expected_axi_input_last = records_remaining_q == 2 && beat_q == 4;

    assign o_record_valid = !i_abort && state_q == STATE_LOAD_EMIT;
    assign o_record_data = record_slot_q ? payload_q[639:320] :
        payload_q[319:0];
    assign o_record_last = records_remaining_q == 1;
    assign record_output_fire = o_record_valid && i_record_ready;

    assign o_record_ready = !i_abort && state_q == STATE_STORE_CAPTURE;
    assign record_input_fire = i_record_valid && o_record_ready;
    assign expected_record_input_last = records_remaining_q == 1;

    assign o_axi_valid = !i_abort && state_q == STATE_STORE_EMIT;
    always_comb begin
        case (beat_q)
            3'd0: o_axi_data = payload_q[127:0];
            3'd1: o_axi_data = payload_q[255:128];
            3'd2: o_axi_data = payload_q[383:256];
            3'd3: o_axi_data = payload_q[511:384];
            default: o_axi_data = payload_q[639:512];
        endcase
    end
    assign o_axi_last = records_remaining_q == 0 && beat_q == 4;
    assign axi_output_fire = o_axi_valid && i_axi_ready;

    // Payload validity is carried only by state_q; resetting 640 data bits
    // would add a high-fanout reset for values that are never observed invalid.
    always_ff @(posedge i_clk) begin
        if (state_q == STATE_LOAD_CAPTURE && axi_input_fire) begin
            case (beat_q)
                3'd0: payload_q[127:0] <= i_axi_data;
                3'd1: payload_q[255:128] <= i_axi_data;
                3'd2: payload_q[383:256] <= i_axi_data;
                3'd3: payload_q[511:384] <= i_axi_data;
                default: payload_q[639:512] <= i_axi_data;
            endcase
        end
        if (state_q == STATE_STORE_CAPTURE && record_input_fire) begin
            if (record_slot_q)
                payload_q[639:320] <= i_record_data;
            else
                payload_q[319:0] <= i_record_data;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            records_remaining_q <= '0;
            beat_q <= '0;
            record_slot_q <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            records_remaining_q <= '0;
            beat_q <= '0;
            record_slot_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (start_fire) begin
                        if (!start_legal) begin
                            o_sticky_error <= 1'b1;
                        end else begin
                            records_remaining_q <= i_record_count;
                            beat_q <= '0;
                            record_slot_q <= 1'b0;
                            if (i_direction_store)
                                state_q <= STATE_STORE_CAPTURE;
                            else
                                state_q <= STATE_LOAD_CAPTURE;
                        end
                    end
                end

                STATE_LOAD_CAPTURE: begin
                    if (axi_input_fire) begin
                        if (i_axi_last != expected_axi_input_last)
                            o_sticky_error <= 1'b1;
                        if (beat_q == 4) begin
                            beat_q <= '0;
                            record_slot_q <= 1'b0;
                            state_q <= STATE_LOAD_EMIT;
                        end else begin
                            beat_q <= beat_q + 3'd1;
                        end
                    end
                end

                STATE_LOAD_EMIT: begin
                    if (record_output_fire) begin
                        records_remaining_q <= records_remaining_q - 1;
                        if (records_remaining_q == 1) begin
                            state_q <= STATE_IDLE;
                            record_slot_q <= 1'b0;
                            o_done <= 1'b1;
                        end else if (record_slot_q) begin
                            record_slot_q <= 1'b0;
                            beat_q <= '0;
                            state_q <= STATE_LOAD_CAPTURE;
                        end else begin
                            record_slot_q <= 1'b1;
                        end
                    end
                end

                STATE_STORE_CAPTURE: begin
                    if (record_input_fire) begin
                        if (i_record_last != expected_record_input_last)
                            o_sticky_error <= 1'b1;
                        records_remaining_q <= records_remaining_q - 1;
                        if (record_slot_q) begin
                            record_slot_q <= 1'b0;
                            beat_q <= '0;
                            state_q <= STATE_STORE_EMIT;
                        end else begin
                            record_slot_q <= 1'b1;
                        end
                    end
                end

                STATE_STORE_EMIT: begin
                    if (axi_output_fire) begin
                        if (beat_q == 4) begin
                            beat_q <= '0;
                            if (records_remaining_q == 0) begin
                                state_q <= STATE_IDLE;
                                o_done <= 1'b1;
                            end else begin
                                state_q <= STATE_STORE_CAPTURE;
                            end
                        end else begin
                            beat_q <= beat_q + 3'd1;
                        end
                    end
                end

                default: begin
                    state_q <= STATE_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
