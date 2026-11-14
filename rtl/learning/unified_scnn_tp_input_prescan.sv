`timescale 1ns/1ps
`default_nettype none

// Before a selected CONV descriptor is accepted, read its scratchpad input
// once in CHW order. The upstream descriptor must remain valid until ready.
// This borrows the existing CONV read port; there is no second feature RAM.
module unified_scnn_tp_input_prescan #(
    parameter int ADDR_WIDTH = 14,
    parameter int BANK_DEPTH = 9216,
    parameter int MAX_WIDTH = 32,
    parameter int MAX_CHANNELS = 512,
    parameter int VERSION_WIDTH = 16
) (
    input logic i_clk, i_rst_n, i_abort, i_clear_error,
    input logic i_enable,
    input logic i_layer_valid,
    output logic o_layer_ready,
    output logic o_layer_valid,
    input logic i_layer_ready,
    input logic [15:0] i_width, i_height, i_channels,
    input logic i_slot,
    input logic [ADDR_WIDTH-1:0] i_base,
    input logic [3:0] i_layer,
    input logic [VERSION_WIDTH-1:0] i_version,

    output logic o_read_valid,
    input logic i_read_ready,
    output logic o_read_slot,
    output logic [31:0] o_read_banks,
    output logic [8*ADDR_WIDTH-1:0] o_read_addrs,
    output logic [7:0] o_read_mask,
    input logic i_response_valid,
    output logic o_response_ready,
    input logic [127:0] i_response_data,
    input logic [7:0] i_response_mask,

    output logic o_observation_valid,
    input logic i_observation_ready,
    output logic [15:0] o_channel,
    output logic [15:0] o_spatial,
    output logic [3:0] o_layer,
    output logic [VERSION_WIDTH-1:0] o_version,
    output logic signed [15:0] o_data_q8_8,
    output logic o_last,
    output logic o_read_owner,
    output logic o_busy,
    output logic o_error
);
    typedef enum logic [2:0] {IDLE, PREP_CONFIG, READ_INPUT, WAIT_INPUT,
                              SEND_INPUT, DISPATCH, DRAIN, FAULT} state_t;
    state_t state_q;
    logic [15:0] width_q, height_q, channels_q;
    logic [15:0] channel_q, row_q, col_q;
    logic [ADDR_WIDTH-1:0] base_q;
    logic slot_q;
    logic [3:0] layer_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic signed [15:0] data_q;
    logic config_legal, selected, final_element;
    logic [63:0] requested_end;
    integer bank_rows, bank_cols, address;

    assign requested_end = 64'(base_q) + 64'(channels_q) *
        ((64'(height_q)+2)/3) * ((64'(width_q)+2)/3);
    assign config_legal = width_q != 0 && width_q <= 16'(MAX_WIDTH) &&
        height_q != 0 && height_q <= 16'(MAX_WIDTH) &&
        channels_q != 0 && channels_q <= 16'(MAX_CHANNELS) &&
        requested_end <= 64'(BANK_DEPTH);
    assign selected = i_enable && i_layer_valid;
    assign o_layer_valid = i_rst_n && !i_abort && !o_error && i_layer_valid &&
        (state_q == DISPATCH || (state_q == IDLE && !i_enable));
    assign o_layer_ready = i_rst_n && !i_abort && !o_error && i_layer_ready &&
        (state_q == DISPATCH || (state_q == IDLE && !i_enable));
    assign o_read_owner = state_q == READ_INPUT || state_q == WAIT_INPUT ||
        state_q == SEND_INPUT || state_q == DRAIN;
    // Busy is state-owned. Looking ahead to an unaccepted layer-valid input
    // creates a combinational dependency on the system-wide idle decision.
    assign o_busy = state_q == PREP_CONFIG || o_read_owner ||
        state_q == DISPATCH;
    assign o_read_valid = state_q == READ_INPUT && !i_abort;
    assign o_response_ready = state_q == WAIT_INPUT || state_q == DRAIN;
    assign o_read_slot = slot_q;
    assign o_observation_valid = state_q == SEND_INPUT && !i_abort;
    assign o_channel = channel_q;
    assign o_spatial = 16'(int'(row_q)*int'(width_q)+int'(col_q));
    assign o_layer = layer_q;
    assign o_version = version_q;
    assign o_data_q8_8 = data_q;
    assign final_element = channel_q+16'd1 == channels_q &&
        row_q+16'd1 == height_q && col_q+16'd1 == width_q;
    assign o_last = final_element;

    always_comb begin
        bank_rows = (int'(height_q)+2)/3;
        bank_cols = (int'(width_q)+2)/3;
        address = int'(base_q) + int'(channel_q)*bank_rows*bank_cols +
            (int'(row_q)/3)*bank_cols + int'(col_q)/3;
        o_read_banks = '0;
        o_read_addrs = '0;
        o_read_mask = '0;
        if (o_read_owner) begin
            o_read_banks[3:0] = 4'((int'(row_q)%3)*3+int'(col_q)%3);
            o_read_addrs[ADDR_WIDTH-1:0] = ADDR_WIDTH'(address);
            o_read_mask = 8'h01;
        end
    end

    // A validity state owns the unreset payload. An abort in WAIT still drains
    // the accepted RAM response; no observation or descriptor can escape it.
    always_ff @(posedge i_clk) begin
        if (state_q == WAIT_INPUT && i_response_valid && !i_abort)
            data_q <= $signed(i_response_data[15:0]);
    end
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            width_q <= '0; height_q <= '0; channels_q <= '0;
            channel_q <= '0; row_q <= '0; col_q <= '0;
            base_q <= '0; slot_q <= 1'b0; layer_q <= '0; version_q <= '0;
            o_error <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy) begin
                o_error <= 1'b0;
                if (state_q == FAULT) state_q <= IDLE;
            end
            if (i_abort) begin
                if ((state_q == WAIT_INPUT || state_q == DRAIN) && !i_response_valid)
                    state_q <= DRAIN;
                else state_q <= IDLE;
            end else begin
                case (state_q)
                    IDLE: if (selected && !o_error) begin
                        width_q <= i_width; height_q <= i_height;
                        channels_q <= i_channels; base_q <= i_base;
                        slot_q <= i_slot; layer_q <= i_layer; version_q <= i_version;
                        channel_q <= '0; row_q <= '0; col_q <= '0;
                        state_q <= PREP_CONFIG;
                    end
                    PREP_CONFIG: begin
                        if (!config_legal) begin
                            o_error <= 1'b1;
                            state_q <= FAULT;
                        end else begin
                            state_q <= READ_INPUT;
                        end
                    end
                    READ_INPUT: if (i_read_ready) state_q <= WAIT_INPUT;
                    WAIT_INPUT: if (i_response_valid) begin
                        if (i_response_mask != 8'h01) begin
                            o_error <= 1'b1;
                            state_q <= FAULT;
                        end else state_q <= SEND_INPUT;
                    end
                    SEND_INPUT: if (i_observation_ready) begin
                        if (final_element) state_q <= DISPATCH;
                        else begin
                            if (col_q+16'd1 == width_q) begin
                                col_q <= '0;
                                if (row_q+16'd1 == height_q) begin
                                    row_q <= '0; channel_q <= channel_q+16'd1;
                                end else row_q <= row_q+16'd1;
                            end else col_q <= col_q+16'd1;
                            state_q <= READ_INPUT;
                        end
                    end
                    DISPATCH: if (i_layer_valid && i_layer_ready) state_q <= IDLE;
                    DRAIN: if (i_response_valid) state_q <= IDLE;
                    default: begin end
                endcase
            end
        end
    end
endmodule
`default_nettype wire
