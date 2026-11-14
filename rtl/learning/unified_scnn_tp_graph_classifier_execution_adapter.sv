`default_nettype none

// Expands one SCNN core transaction into either student-graph then classifier
// execution, or a post-commit classifier-only forward. Numeric datapaths and
// storage stay outside this control-only adapter.
module unified_scnn_tp_graph_classifier_execution_adapter #(
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_abort,
    input  logic                            i_clear_error,

    input  logic                            i_core_start_valid,
    output logic                            o_core_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]         i_core_sample,
    input  logic [TIMESTEP_WIDTH-1:0]       i_core_timestep,
    input  logic                            i_core_post_forward,

    output logic                            o_student_start_valid,
    input  logic                            i_student_start_ready,
    output logic [SAMPLE_WIDTH-1:0]         o_student_sample,
    output logic [TIMESTEP_WIDTH-1:0]       o_student_timestep,
    input  logic                            i_student_done,
    input  logic                            i_student_error,

    output logic                            o_classifier_start_valid,
    input  logic                            i_classifier_start_ready,
    output logic [SAMPLE_WIDTH-1:0]         o_classifier_sample,
    output logic [TIMESTEP_WIDTH-1:0]       o_classifier_timestep,
    input  logic                            i_classifier_done,
    input  logic                            i_classifier_error,

    output logic                            o_post_forward_start_valid,
    input  logic                            i_post_forward_start_ready,
    output logic [SAMPLE_WIDTH-1:0]         o_post_forward_sample,
    output logic [TIMESTEP_WIDTH-1:0]       o_post_forward_timestep,
    input  logic                            i_post_forward_done,
    input  logic                            i_post_forward_error,

    output logic                            o_selected_conv_done,
    output logic                            o_core_sample_done,
    output logic                            o_aborted,
    output logic                            o_busy,
    output logic                            o_quiescent,
    output logic                            o_protocol_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_STUDENT_WAIT,
        ST_CLASSIFIER_REQUEST,
        ST_CLASSIFIER_WAIT,
        ST_POST_WAIT,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic protocol_error_q;
    logic selected_start_ready;
    logic child_error;

    assign selected_start_ready = i_core_post_forward ?
        i_post_forward_start_ready : i_student_start_ready;
    assign o_core_start_ready = state_q == ST_IDLE &&
        selected_start_ready && !protocol_error_q && !child_error;
    assign o_student_start_valid = state_q == ST_IDLE &&
        i_core_start_valid && !i_core_post_forward &&
        !protocol_error_q && !child_error;
    assign o_post_forward_start_valid = state_q == ST_IDLE &&
        i_core_start_valid && i_core_post_forward &&
        !protocol_error_q && !child_error;
    assign o_classifier_start_valid = state_q == ST_CLASSIFIER_REQUEST &&
        !protocol_error_q;
    assign o_student_sample = state_q == ST_IDLE ? i_core_sample : sample_q;
    assign o_student_timestep = state_q == ST_IDLE ?
        i_core_timestep : timestep_q;
    assign o_classifier_sample = sample_q;
    assign o_classifier_timestep = timestep_q;
    assign o_post_forward_sample = state_q == ST_IDLE ?
        i_core_sample : sample_q;
    assign o_post_forward_timestep = state_q == ST_IDLE ?
        i_core_timestep : timestep_q;
    assign child_error = i_student_error || i_classifier_error ||
        i_post_forward_error;
    assign o_busy = state_q != ST_IDLE && state_q != ST_ERROR;
    assign o_quiescent = state_q == ST_IDLE && !protocol_error_q;
    assign o_protocol_error = protocol_error_q || child_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            sample_q <= '0;
            timestep_q <= '0;
            protocol_error_q <= 1'b0;
            o_selected_conv_done <= 1'b0;
            o_core_sample_done <= 1'b0;
            o_aborted <= 1'b0;
        end else begin
            o_selected_conv_done <= 1'b0;
            o_core_sample_done <= 1'b0;
            o_aborted <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR && !child_error) begin
                protocol_error_q <= 1'b0;
                state_q <= ST_IDLE;
            end else if (child_error) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else if (i_abort && state_q != ST_IDLE &&
                         state_q != ST_ERROR) begin
                state_q <= ST_IDLE;
                o_aborted <= 1'b1;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        if (i_core_start_valid && o_core_start_ready) begin
                            sample_q <= i_core_sample;
                            timestep_q <= i_core_timestep;
                            state_q <= i_core_post_forward ?
                                ST_POST_WAIT : ST_STUDENT_WAIT;
                        end else if (i_student_done || i_classifier_done ||
                                     i_post_forward_done) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                    end

                    ST_STUDENT_WAIT: begin
                        if (i_student_done) begin
                            o_selected_conv_done <= 1'b1;
                            state_q <= ST_CLASSIFIER_REQUEST;
                        end else if (i_classifier_done ||
                                     i_post_forward_done) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                    end

                    ST_CLASSIFIER_REQUEST: begin
                        if (i_classifier_done || i_student_done ||
                            i_post_forward_done) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (o_classifier_start_valid &&
                                     i_classifier_start_ready) begin
                            state_q <= ST_CLASSIFIER_WAIT;
                        end
                    end

                    ST_CLASSIFIER_WAIT: begin
                        if (i_classifier_done) begin
                            o_core_sample_done <= 1'b1;
                            state_q <= ST_IDLE;
                        end else if (i_student_done ||
                                     i_post_forward_done) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                    end

                    ST_POST_WAIT: begin
                        if (i_post_forward_done) begin
                            o_core_sample_done <= 1'b1;
                            state_q <= ST_IDLE;
                        end else if (i_student_done || i_classifier_done) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                    end

                    default: state_q <= ST_ERROR;
                endcase
            end
        end
    end

    initial begin
        if (SAMPLE_WIDTH < 1 || TIMESTEP_WIDTH < 1)
            $error("SCNN graph/classifier execution widths are invalid");
    end
endmodule

`default_nettype wire
