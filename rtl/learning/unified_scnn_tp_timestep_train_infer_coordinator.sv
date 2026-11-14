`timescale 1ns/1ps
`default_nettype none

// Runtime scheduler for one SCNN profile. It orders the existing target,
// student, classifier and joint-learning clients without owning any numeric
// datapath or parameter memory.
module unified_scnn_tp_timestep_train_infer_coordinator #(
    parameter int MAX_BATCH = 16,
    parameter int MAX_TIMESTEPS = 128,
    parameter int BATCH_WIDTH = $clog2(MAX_BATCH+1),
    parameter int TIMESTEP_COUNT_WIDTH = $clog2(MAX_TIMESTEPS+1),
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS)
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,

    input  logic                                i_start_valid,
    output logic                                o_start_ready,
    input  logic [BATCH_WIDTH-1:0]              i_batch_count,
    input  logic [TIMESTEP_COUNT_WIDTH-1:0]     i_timestep_count,
    input  logic                                i_learning_enable,

    output logic                                o_learner_start_valid,
    input  logic                                i_learner_start_ready,
    input  logic                                i_learner_done,

    output logic                                o_target_start_valid,
    input  logic                                i_target_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_target_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_target_timestep,
    input  logic                                i_target_done,

    output logic                                o_student_start_valid,
    input  logic                                i_student_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_student_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_student_timestep,
    input  logic                                i_student_done,

    output logic                                o_classifier_start_valid,
    input  logic                                i_classifier_start_ready,
    output logic [SAMPLE_WIDTH-1:0]             o_classifier_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_classifier_timestep,
    output logic                                o_classifier_capture_enable,
    input  logic                                i_classifier_done,

    output logic                                o_post_forward_start_valid,
    input  logic                                i_post_forward_start_ready,
    output logic [TIMESTEP_WIDTH-1:0]           o_post_forward_timestep,
    input  logic                                i_post_forward_done,

    output logic [SAMPLE_WIDTH-1:0]             o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]           o_active_timestep,
    output logic                                o_busy,
    output logic                                o_done,
    output logic                                o_aborted,
    output logic                                o_protocol_error
);
    typedef enum logic [3:0] {
        ST_IDLE,
        ST_LEARNER_REQUEST,
        ST_TARGET_REQUEST,
        ST_TARGET_WAIT,
        ST_STUDENT_REQUEST,
        ST_STUDENT_WAIT,
        ST_CLASSIFIER_REQUEST,
        ST_CLASSIFIER_WAIT,
        ST_LEARNER_WAIT,
        ST_POST_REQUEST,
        ST_POST_WAIT,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [BATCH_WIDTH-1:0] batch_count_q;
    logic [TIMESTEP_COUNT_WIDTH-1:0] timestep_count_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic learning_q;
    logic start_legal;

    assign start_legal = i_batch_count != 0 && i_timestep_count != 0 &&
        i_batch_count <= BATCH_WIDTH'(MAX_BATCH) &&
        i_timestep_count <= TIMESTEP_COUNT_WIDTH'(MAX_TIMESTEPS);
    assign o_start_ready = state_q == ST_IDLE && !o_protocol_error;
    assign o_learner_start_valid = state_q == ST_LEARNER_REQUEST;
    assign o_target_start_valid = state_q == ST_TARGET_REQUEST;
    assign o_student_start_valid = state_q == ST_STUDENT_REQUEST;
    assign o_classifier_start_valid = state_q == ST_CLASSIFIER_REQUEST;
    assign o_post_forward_start_valid = state_q == ST_POST_REQUEST;
    assign o_target_sample = sample_q;
    assign o_target_timestep = timestep_q;
    assign o_student_sample = sample_q;
    assign o_student_timestep = timestep_q;
    assign o_classifier_sample = sample_q;
    assign o_classifier_timestep = timestep_q;
    assign o_classifier_capture_enable = learning_q;
    assign o_post_forward_timestep = timestep_q;
    assign o_active_sample = sample_q;
    assign o_active_timestep = timestep_q;
    assign o_busy = state_q != ST_IDLE && state_q != ST_ERROR;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            batch_count_q <= '0;
            timestep_count_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            learning_q <= 1'b0;
            o_done <= 1'b0;
            o_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            o_aborted <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR) begin
                o_protocol_error <= 1'b0;
                state_q <= ST_IDLE;
            end else if (i_abort && state_q != ST_IDLE &&
                         state_q != ST_ERROR) begin
                sample_q <= '0;
                timestep_q <= '0;
                learning_q <= 1'b0;
                o_aborted <= 1'b1;
                state_q <= ST_IDLE;
            end else begin
                if ((i_target_done && state_q != ST_TARGET_WAIT) ||
                    (i_student_done && state_q != ST_STUDENT_WAIT) ||
                    (i_classifier_done && state_q != ST_CLASSIFIER_WAIT) ||
                    (i_learner_done && state_q != ST_LEARNER_WAIT) ||
                    (i_post_forward_done && state_q != ST_POST_WAIT)) begin
                    o_protocol_error <= 1'b1;
                    state_q <= ST_ERROR;
                end else begin
                    unique case (state_q)
                        ST_IDLE: begin
                            if (i_start_valid && o_start_ready) begin
                                if (!start_legal) begin
                                    o_protocol_error <= 1'b1;
                                    state_q <= ST_ERROR;
                                end else begin
                                    batch_count_q <= i_batch_count;
                                    timestep_count_q <= i_timestep_count;
                                    sample_q <= '0;
                                    timestep_q <= '0;
                                    learning_q <= i_learning_enable;
                                    state_q <= i_learning_enable ?
                                        ST_LEARNER_REQUEST : ST_STUDENT_REQUEST;
                                end
                            end
                        end

                        ST_LEARNER_REQUEST:
                            if (i_learner_start_ready)
                                state_q <= ST_TARGET_REQUEST;

                        ST_TARGET_REQUEST:
                            if (i_target_start_ready)
                                state_q <= ST_TARGET_WAIT;

                        ST_TARGET_WAIT: begin
                            if (i_target_done) begin
                                if ({1'b0, sample_q}+1 < batch_count_q) begin
                                    sample_q <= sample_q + 1'b1;
                                    state_q <= ST_TARGET_REQUEST;
                                end else begin
                                    sample_q <= '0;
                                    state_q <= ST_STUDENT_REQUEST;
                                end
                            end
                        end

                        ST_STUDENT_REQUEST:
                            if (i_student_start_ready)
                                state_q <= ST_STUDENT_WAIT;

                        ST_STUDENT_WAIT:
                            if (i_student_done)
                                state_q <= ST_CLASSIFIER_REQUEST;

                        ST_CLASSIFIER_REQUEST:
                            if (i_classifier_start_ready)
                                state_q <= ST_CLASSIFIER_WAIT;

                        ST_CLASSIFIER_WAIT: begin
                            if (i_classifier_done) begin
                                if ({1'b0, sample_q}+1 < batch_count_q) begin
                                    sample_q <= sample_q + 1'b1;
                                    state_q <= ST_STUDENT_REQUEST;
                                end else if (learning_q) begin
                                    state_q <= ST_LEARNER_WAIT;
                                end else if ({1'b0, timestep_q}+1 <
                                             timestep_count_q) begin
                                    sample_q <= '0;
                                    timestep_q <= timestep_q + 1'b1;
                                    state_q <= ST_STUDENT_REQUEST;
                                end else begin
                                    o_done <= 1'b1;
                                    state_q <= ST_IDLE;
                                end
                            end
                        end

                        ST_LEARNER_WAIT:
                            if (i_learner_done)
                                state_q <= ST_POST_REQUEST;

                        ST_POST_REQUEST:
                            if (i_post_forward_start_ready)
                                state_q <= ST_POST_WAIT;

                        ST_POST_WAIT: begin
                            if (i_post_forward_done) begin
                                if ({1'b0, timestep_q}+1 <
                                    timestep_count_q) begin
                                    sample_q <= '0;
                                    timestep_q <= timestep_q + 1'b1;
                                    state_q <= ST_LEARNER_REQUEST;
                                end else begin
                                    o_done <= 1'b1;
                                    state_q <= ST_IDLE;
                                end
                            end
                        end

                        default: begin
                            o_protocol_error <= 1'b1;
                            state_q <= ST_ERROR;
                        end
                    endcase
                end
            end
        end
    end

    initial begin
        if (MAX_BATCH < 1 || MAX_TIMESTEPS < 1)
            $error("SCNN timestep coordinator dimensions are invalid");
    end
endmodule

`default_nettype wire
