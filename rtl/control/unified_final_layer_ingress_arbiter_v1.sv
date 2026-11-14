`timescale 1ns/1ps
`default_nettype none

// Serializes the resident graph sequencer and an external SCNN graph client
// onto one final-core layer dispatcher. The accepted command metadata remains
// stable until the physical layer reports completion.
module unified_final_layer_ingress_arbiter_v1 #(
    parameter int LAYER_WORDS = 12,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_internal_valid,
    output logic                                  o_internal_ready,
    input  logic [3:0]                            i_internal_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_internal_layer_words,
    input  logic                                  i_internal_input_slot,
    input  logic                                  i_internal_output_slot,
    input  logic [VERSION_WIDTH-1:0]              i_internal_state_version,
    input  logic                                  i_internal_sample_reset,
    input  logic                                  i_internal_sample_last,
    input  logic                                  i_internal_observation_enable,
    input  logic                                  i_internal_input_prescan_enable,
    output logic                                  o_internal_done,

    input  logic                                  i_external_valid,
    input  logic                                  i_external_session_active,
    output logic                                  o_external_ready,
    input  logic [3:0]                            i_external_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_external_layer_words,
    input  logic                                  i_external_input_slot,
    input  logic                                  i_external_output_slot,
    input  logic [VERSION_WIDTH-1:0]              i_external_state_version,
    input  logic                                  i_external_sample_reset,
    input  logic                                  i_external_sample_last,
    input  logic                                  i_external_observation_enable,
    input  logic                                  i_external_input_prescan_enable,
    output logic                                  o_external_done,

    output logic                                  o_shared_valid,
    input  logic                                  i_shared_ready,
    output logic [3:0]                            o_shared_layer_index,
    output logic [LAYER_WORDS*32-1:0]             o_shared_layer_words,
    output logic                                  o_shared_input_slot,
    output logic                                  o_shared_output_slot,
    output logic [VERSION_WIDTH-1:0]              o_shared_state_version,
    output logic                                  o_shared_sample_reset,
    output logic                                  o_shared_sample_last,
    output logic                                  o_shared_observation_enable,
    output logic                                  o_shared_input_prescan_enable,
    input  logic                                  i_shared_done,
    input  logic                                  i_shared_error,

    output logic                                  o_active_external,
    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    localparam logic [3:0] LAYER_CONV = 4'd2;
    localparam logic [3:0] LAYER_OUTPUT = 4'd4;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_INTERNAL_ACTIVE,
        ST_EXTERNAL_ACTIVE,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [3:0] layer_index_q;
    logic [LAYER_WORDS*32-1:0] layer_words_q;
    logic input_slot_q;
    logic output_slot_q;
    logic [VERSION_WIDTH-1:0] state_version_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic observation_enable_q;
    logic input_prescan_enable_q;
    logic simultaneous_request;
    logic internal_selected;
    logic external_selected;
    logic [3:0] external_layer_type;
    logic external_type_legal;

    assign simultaneous_request = i_internal_valid && i_external_valid;
    assign external_layer_type = i_external_layer_words[2*32 +: 4];
    assign external_type_legal = external_layer_type == LAYER_CONV ||
        external_layer_type == LAYER_OUTPUT;
    assign internal_selected = state_q == ST_IDLE && i_internal_valid &&
        !i_external_valid;
    assign external_selected = state_q == ST_IDLE &&
        i_external_session_active && i_external_valid &&
        !i_internal_valid && external_type_legal;

    assign o_shared_valid = (internal_selected || external_selected) &&
        !o_protocol_error;
    assign o_internal_ready = internal_selected && i_shared_ready;
    assign o_external_ready = external_selected && i_shared_ready;
    assign o_internal_done = state_q == ST_INTERNAL_ACTIVE && i_shared_done;
    assign o_external_done = state_q == ST_EXTERNAL_ACTIVE && i_shared_done;
    assign o_active_external = state_q == ST_EXTERNAL_ACTIVE ||
        external_selected;
    assign o_busy = state_q != ST_IDLE;

    always_comb begin
        o_shared_layer_index = layer_index_q;
        o_shared_layer_words = layer_words_q;
        o_shared_input_slot = input_slot_q;
        o_shared_output_slot = output_slot_q;
        o_shared_state_version = state_version_q;
        o_shared_sample_reset = sample_reset_q;
        o_shared_sample_last = sample_last_q;
        o_shared_observation_enable = observation_enable_q;
        o_shared_input_prescan_enable = input_prescan_enable_q;

        if (state_q == ST_IDLE) begin
            if (external_selected) begin
                o_shared_layer_index = i_external_layer_index;
                o_shared_layer_words = i_external_layer_words;
                o_shared_input_slot = i_external_input_slot;
                o_shared_output_slot = i_external_output_slot;
                o_shared_state_version = i_external_state_version;
                o_shared_sample_reset = i_external_sample_reset;
                o_shared_sample_last = i_external_sample_last;
                o_shared_observation_enable =
                    i_external_observation_enable;
                o_shared_input_prescan_enable =
                    i_external_input_prescan_enable;
            end else begin
                o_shared_layer_index = i_internal_layer_index;
                o_shared_layer_words = i_internal_layer_words;
                o_shared_input_slot = i_internal_input_slot;
                o_shared_output_slot = i_internal_output_slot;
                o_shared_state_version = i_internal_state_version;
                o_shared_sample_reset = i_internal_sample_reset;
                o_shared_sample_last = i_internal_sample_last;
                o_shared_observation_enable =
                    i_internal_observation_enable;
                o_shared_input_prescan_enable =
                    i_internal_input_prescan_enable;
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            layer_index_q <= '0;
            layer_words_q <= '0;
            input_slot_q <= 1'b0;
            output_slot_q <= 1'b0;
            state_version_q <= '0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            observation_enable_q <= 1'b0;
            input_prescan_enable_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= ST_IDLE;
        end else begin
            if (i_clear_error && state_q == ST_ERROR) begin
                o_protocol_error <= 1'b0;
                state_q <= ST_IDLE;
            end else if (simultaneous_request ||
                         (i_external_valid &&
                          !i_external_session_active) ||
                         (state_q == ST_EXTERNAL_ACTIVE &&
                          !i_external_session_active) ||
                         (i_external_valid && !external_type_legal) ||
                         (i_shared_error && state_q != ST_IDLE)) begin
                o_protocol_error <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        if (o_internal_ready) begin
                            layer_index_q <= i_internal_layer_index;
                            layer_words_q <= i_internal_layer_words;
                            input_slot_q <= i_internal_input_slot;
                            output_slot_q <= i_internal_output_slot;
                            state_version_q <= i_internal_state_version;
                            sample_reset_q <= i_internal_sample_reset;
                            sample_last_q <= i_internal_sample_last;
                            observation_enable_q <=
                                i_internal_observation_enable;
                            input_prescan_enable_q <=
                                i_internal_input_prescan_enable;
                            state_q <= ST_INTERNAL_ACTIVE;
                        end else if (o_external_ready) begin
                            layer_index_q <= i_external_layer_index;
                            layer_words_q <= i_external_layer_words;
                            input_slot_q <= i_external_input_slot;
                            output_slot_q <= i_external_output_slot;
                            state_version_q <= i_external_state_version;
                            sample_reset_q <= i_external_sample_reset;
                            sample_last_q <= i_external_sample_last;
                            observation_enable_q <=
                                i_external_observation_enable;
                            input_prescan_enable_q <=
                                i_external_input_prescan_enable;
                            state_q <= ST_EXTERNAL_ACTIVE;
                        end
                    end

                    ST_INTERNAL_ACTIVE,
                    ST_EXTERNAL_ACTIVE: begin
                        if (i_shared_done)
                            state_q <= ST_IDLE;
                    end

                    default: begin
                        o_protocol_error <= 1'b1;
                        state_q <= ST_ERROR;
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
