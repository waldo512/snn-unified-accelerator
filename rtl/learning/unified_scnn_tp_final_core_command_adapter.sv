`timescale 1ns/1ps
`default_nettype none

// Maps the SCNN coordinator's student/classifier phases onto one continuous
// final-core sample. The student request starts the graph; the classifier
// request joins that same in-flight graph after the selected CONV retires.
module unified_scnn_tp_final_core_command_adapter #(
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 7
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,
    input  logic                            i_abort,

    input  logic                            i_student_start_valid,
    output logic                            o_student_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]         i_student_sample,
    input  logic [TIMESTEP_WIDTH-1:0]       i_student_timestep,
    output logic                            o_student_done,

    input  logic                            i_classifier_start_valid,
    output logic                            o_classifier_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]         i_classifier_sample,
    input  logic [TIMESTEP_WIDTH-1:0]       i_classifier_timestep,
    output logic                            o_classifier_done,

    output logic                            o_core_start_valid,
    input  logic                            i_core_start_ready,
    input  logic                            i_selected_conv_done,
    input  logic                            i_core_sample_done,

    output logic [SAMPLE_WIDTH-1:0]         o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]       o_active_timestep,
    output logic                            o_busy,
    output logic                            o_aborted,
    output logic                            o_protocol_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_WAIT_CONV,
        ST_WAIT_CLASSIFIER_REQUEST,
        ST_WAIT_SAMPLE_DONE,
        ST_REPORT_CLASSIFIER_DONE,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic sample_done_seen_q;

    assign o_core_start_valid = state_q == ST_IDLE &&
        i_student_start_valid && !o_protocol_error;
    assign o_student_start_ready = state_q == ST_IDLE &&
        i_core_start_ready && !o_protocol_error;
    assign o_classifier_start_ready = state_q ==
        ST_WAIT_CLASSIFIER_REQUEST && !o_protocol_error;
    assign o_active_sample = state_q == ST_IDLE ? i_student_sample : sample_q;
    assign o_active_timestep = state_q == ST_IDLE ?
        i_student_timestep : timestep_q;
    assign o_busy = state_q != ST_IDLE && state_q != ST_ERROR;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            sample_q <= '0;
            timestep_q <= '0;
            sample_done_seen_q <= 1'b0;
            o_student_done <= 1'b0;
            o_classifier_done <= 1'b0;
            o_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_student_done <= 1'b0;
            o_classifier_done <= 1'b0;
            o_aborted <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR) begin
                state_q <= ST_IDLE;
                sample_done_seen_q <= 1'b0;
                o_protocol_error <= 1'b0;
            end else if (i_abort && state_q != ST_IDLE &&
                         state_q != ST_ERROR) begin
                sample_done_seen_q <= 1'b0;
                o_aborted <= 1'b1;
                state_q <= ST_IDLE;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        sample_done_seen_q <= 1'b0;
                        // Other topology owners may retire through the same
                        // final-core status wires while this adapter is idle.
                        if (o_core_start_valid && i_core_start_ready) begin
                            sample_q <= i_student_sample;
                            timestep_q <= i_student_timestep;
                            state_q <= ST_WAIT_CONV;
                        end
                    end

                    ST_WAIT_CONV: begin
                        if (i_core_sample_done && !i_selected_conv_done) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (i_selected_conv_done) begin
                            o_student_done <= 1'b1;
                            sample_done_seen_q <= i_core_sample_done;
                            state_q <= ST_WAIT_CLASSIFIER_REQUEST;
                        end
                    end

                    ST_WAIT_CLASSIFIER_REQUEST: begin
                        if (i_selected_conv_done) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else begin
                            if (i_core_sample_done)
                                sample_done_seen_q <= 1'b1;
                            if (i_classifier_start_valid &&
                                o_classifier_start_ready) begin
                                if (i_classifier_sample != sample_q ||
                                    i_classifier_timestep != timestep_q) begin
                                    o_protocol_error <= 1'b1;
                                    state_q <= ST_ERROR;
                                end else begin
                                    state_q <= (sample_done_seen_q ||
                                                i_core_sample_done) ?
                                        ST_REPORT_CLASSIFIER_DONE :
                                        ST_WAIT_SAMPLE_DONE;
                                end
                            end
                        end
                    end

                    ST_WAIT_SAMPLE_DONE: begin
                        if (i_selected_conv_done) begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (i_core_sample_done) begin
                            state_q <= ST_REPORT_CLASSIFIER_DONE;
                        end
                    end

                    ST_REPORT_CLASSIFIER_DONE: begin
                        o_classifier_done <= 1'b1;
                        sample_done_seen_q <= 1'b0;
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

    initial begin
        if (SAMPLE_WIDTH < 1 || TIMESTEP_WIDTH < 1)
            $error("SCNN final-core command adapter widths are invalid");
    end
endmodule

`default_nettype wire
