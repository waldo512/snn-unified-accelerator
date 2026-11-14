`default_nettype none

// Locks a descriptor request to its target or student requester until the
// corresponding response is consumed. The owner bit lets one external
// descriptor service select branch-specific tensor and state mappings.
module unified_scnn_tp_target_student_descriptor_arbiter #(
    parameter int LAYER_WORDS = 12,
    parameter int PROFILE_ID_WIDTH = 8
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_target_request_valid,
    output logic                                  o_target_request_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_target_profile_id,
    input  logic [3:0]                            i_target_layer_index,
    output logic                                  o_target_response_valid,
    input  logic                                  i_target_response_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_target_response_profile_id,
    output logic [3:0]                            o_target_response_layer_index,
    output logic [LAYER_WORDS*32-1:0]             o_target_response_words,
    output logic                                  o_target_response_error,

    input  logic                                  i_student_request_valid,
    output logic                                  o_student_request_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_student_profile_id,
    input  logic [3:0]                            i_student_layer_index,
    output logic                                  o_student_response_valid,
    input  logic                                  i_student_response_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_student_response_profile_id,
    output logic [3:0]                            o_student_response_layer_index,
    output logic [LAYER_WORDS*32-1:0]             o_student_response_words,
    output logic                                  o_student_response_error,

    output logic                                  o_common_request_valid,
    input  logic                                  i_common_request_ready,
    output logic                                  o_common_request_target,
    output logic [PROFILE_ID_WIDTH-1:0]           o_common_profile_id,
    output logic [3:0]                            o_common_layer_index,
    input  logic                                  i_common_response_valid,
    output logic                                  o_common_response_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_common_response_profile_id,
    input  logic [3:0]                            i_common_response_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_common_response_words,
    input  logic                                  i_common_response_error,

    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_TARGET_RESPONSE,
        ST_STUDENT_RESPONSE,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic protocol_error_q;
    logic target_selected;
    logic student_selected;
    logic simultaneous_request;
    logic response_mismatch;
    logic [PROFILE_ID_WIDTH-1:0] requested_profile_q;
    logic [3:0] requested_layer_q;

    assign simultaneous_request = state_q == ST_IDLE &&
        i_target_request_valid && i_student_request_valid;
    assign target_selected = state_q == ST_IDLE &&
        i_target_request_valid && !i_student_request_valid;
    assign student_selected = state_q == ST_IDLE &&
        i_student_request_valid && !i_target_request_valid;

    assign o_common_request_valid = (target_selected || student_selected) &&
        !protocol_error_q;
    assign o_common_request_target = state_q == ST_TARGET_RESPONSE ||
        target_selected;
    assign o_common_profile_id = state_q == ST_IDLE ?
        (student_selected ? i_student_profile_id : i_target_profile_id) :
        requested_profile_q;
    assign o_common_layer_index = state_q == ST_IDLE ?
        (student_selected ? i_student_layer_index : i_target_layer_index) :
        requested_layer_q;
    assign o_target_request_ready = target_selected &&
        i_common_request_ready && !protocol_error_q;
    assign o_student_request_ready = student_selected &&
        i_common_request_ready && !protocol_error_q;

    assign response_mismatch =
        i_common_response_profile_id != requested_profile_q ||
        i_common_response_layer_index != requested_layer_q;
    assign o_target_response_valid = state_q == ST_TARGET_RESPONSE &&
        i_common_response_valid;
    assign o_student_response_valid = state_q == ST_STUDENT_RESPONSE &&
        i_common_response_valid;
    assign o_common_response_ready =
        (state_q == ST_TARGET_RESPONSE && i_target_response_ready) ||
        (state_q == ST_STUDENT_RESPONSE && i_student_response_ready);

    assign o_target_response_profile_id = i_common_response_profile_id;
    assign o_target_response_layer_index = i_common_response_layer_index;
    assign o_target_response_words = i_common_response_words;
    assign o_target_response_error = o_target_response_valid &&
        (i_common_response_error || response_mismatch);
    assign o_student_response_profile_id = i_common_response_profile_id;
    assign o_student_response_layer_index = i_common_response_layer_index;
    assign o_student_response_words = i_common_response_words;
    assign o_student_response_error = o_student_response_valid &&
        (i_common_response_error || response_mismatch);
    assign o_busy = state_q != ST_IDLE;
    assign o_protocol_error = protocol_error_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            protocol_error_q <= 1'b0;
            requested_profile_q <= '0;
            requested_layer_q <= '0;
        end else if (i_abort) begin
            state_q <= ST_IDLE;
            requested_profile_q <= '0;
            requested_layer_q <= '0;
        end else begin
            if (i_clear_error && state_q == ST_ERROR) begin
                protocol_error_q <= 1'b0;
                state_q <= ST_IDLE;
            end else if (simultaneous_request ||
                         (state_q == ST_IDLE && i_common_response_valid)) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        if (o_target_request_ready) begin
                            requested_profile_q <= i_target_profile_id;
                            requested_layer_q <= i_target_layer_index;
                            state_q <= ST_TARGET_RESPONSE;
                        end else if (o_student_request_ready) begin
                            requested_profile_q <= i_student_profile_id;
                            requested_layer_q <= i_student_layer_index;
                            state_q <= ST_STUDENT_RESPONSE;
                        end
                    end

                    ST_TARGET_RESPONSE,
                    ST_STUDENT_RESPONSE: begin
                        if (i_common_response_valid &&
                            o_common_response_ready) begin
                            if (i_common_response_error ||
                                response_mismatch) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                state_q <= ST_IDLE;
                            end
                        end
                    end

                    default: state_q <= ST_ERROR;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
