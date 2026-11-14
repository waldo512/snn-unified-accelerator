`default_nettype none

// Locks one target or student layer command onto the shared SCNN layer
// executor until the complete layer, including its observation fork, retires.
module unified_scnn_tp_target_student_layer_arbiter #(
    parameter int LAYER_WORDS = 12,
    parameter int VERSION_WIDTH = 16
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_target_valid,
    output logic                                  o_target_ready,
    input  logic [3:0]                            i_target_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_target_layer_words,
    input  logic                                  i_target_input_slot,
    input  logic                                  i_target_output_slot,
    input  logic [VERSION_WIDTH-1:0]              i_target_state_version,
    output logic                                  o_target_done,

    input  logic                                  i_student_valid,
    output logic                                  o_student_ready,
    input  logic [3:0]                            i_student_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_student_layer_words,
    input  logic                                  i_student_input_slot,
    input  logic                                  i_student_output_slot,
    input  logic [VERSION_WIDTH-1:0]              i_student_state_version,
    output logic                                  o_student_done,

    output logic                                  o_shared_valid,
    input  logic                                  i_shared_ready,
    output logic [3:0]                            o_shared_layer_index,
    output logic [LAYER_WORDS*32-1:0]             o_shared_layer_words,
    output logic                                  o_shared_input_slot,
    output logic                                  o_shared_output_slot,
    output logic [VERSION_WIDTH-1:0]              o_shared_state_version,
    input  logic                                  i_shared_done,
    input  logic                                  i_shared_error,

    output logic                                  o_active_target,
    output logic                                  o_active_student,
    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_TARGET_ACTIVE,
        ST_STUDENT_ACTIVE,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic protocol_error_q;
    logic simultaneous_request;
    logic target_selected;
    logic student_selected;
    logic [3:0] layer_index_q;
    logic [LAYER_WORDS*32-1:0] layer_words_q;
    logic input_slot_q;
    logic output_slot_q;
    logic [VERSION_WIDTH-1:0] state_version_q;

    assign simultaneous_request = i_target_valid && i_student_valid;
    assign target_selected = state_q == ST_IDLE && i_target_valid &&
        !i_student_valid;
    assign student_selected = state_q == ST_IDLE && i_student_valid &&
        !i_target_valid;
    assign o_shared_valid = (target_selected || student_selected) &&
        !protocol_error_q;
    assign o_target_ready = target_selected && i_shared_ready;
    assign o_student_ready = student_selected && i_shared_ready;
    assign o_shared_layer_index = state_q == ST_IDLE ?
        (student_selected ? i_student_layer_index :
            i_target_layer_index) : layer_index_q;
    assign o_shared_layer_words = state_q == ST_IDLE ?
        (student_selected ? i_student_layer_words :
            i_target_layer_words) : layer_words_q;
    assign o_shared_input_slot = state_q == ST_IDLE ?
        (student_selected ? i_student_input_slot :
            i_target_input_slot) : input_slot_q;
    assign o_shared_output_slot = state_q == ST_IDLE ?
        (student_selected ? i_student_output_slot :
            i_target_output_slot) : output_slot_q;
    assign o_shared_state_version = state_q == ST_IDLE ?
        (student_selected ? i_student_state_version :
            i_target_state_version) : state_version_q;
    assign o_target_done = state_q == ST_TARGET_ACTIVE && i_shared_done;
    assign o_student_done = state_q == ST_STUDENT_ACTIVE && i_shared_done;
    assign o_active_target = state_q == ST_TARGET_ACTIVE || target_selected;
    assign o_active_student = state_q == ST_STUDENT_ACTIVE || student_selected;
    assign o_busy = state_q != ST_IDLE;
    assign o_protocol_error = protocol_error_q || i_shared_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            protocol_error_q <= 1'b0;
            layer_index_q <= '0;
            layer_words_q <= '0;
            input_slot_q <= 1'b0;
            output_slot_q <= 1'b0;
            state_version_q <= '0;
        end else if (i_abort) begin
            state_q <= ST_IDLE;
        end else begin
            if (i_clear_error && state_q == ST_ERROR) begin
                protocol_error_q <= 1'b0;
                state_q <= ST_IDLE;
            end else if ((i_shared_error && state_q != ST_IDLE) ||
                         simultaneous_request) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        if (o_target_ready && i_target_valid) begin
                            layer_index_q <= i_target_layer_index;
                            layer_words_q <= i_target_layer_words;
                            input_slot_q <= i_target_input_slot;
                            output_slot_q <= i_target_output_slot;
                            state_version_q <= i_target_state_version;
                            state_q <= ST_TARGET_ACTIVE;
                        end else if (o_student_ready && i_student_valid) begin
                            layer_index_q <= i_student_layer_index;
                            layer_words_q <= i_student_layer_words;
                            input_slot_q <= i_student_input_slot;
                            output_slot_q <= i_student_output_slot;
                            state_version_q <= i_student_state_version;
                            state_q <= ST_STUDENT_ACTIVE;
                        end
                    end

                    ST_TARGET_ACTIVE,
                    ST_STUDENT_ACTIVE: begin
                        if (i_shared_done)
                            state_q <= ST_IDLE;
                    end

                    default: state_q <= ST_ERROR;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
