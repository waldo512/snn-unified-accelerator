`timescale 1ns/1ps
`default_nettype none

// Sequences one SCNN transaction on a single graph engine. Learning executes a
// target pass followed by a student pass under one runtime capture window;
// inference executes only the student pass. Conv observations are admitted on
// both passes, while classifier features/logits are admitted only on student.
module unified_scnn_selected_layers_graph_pair_coordinator_v1 #(
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int VERSION_WIDTH = 16,
    parameter int CLASS_WIDTH = 4
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_abort,
    input  wire logic                         i_clear_error,

    input  wire logic                         i_request_valid,
    output logic                              o_request_ready,
    input  wire logic                         i_learning_enable,
    input  wire logic [SAMPLE_WIDTH-1:0]       i_sample,
    input  wire logic [TIMESTEP_WIDTH-1:0]     i_timestep,
    input  wire logic [VERSION_WIDTH-1:0]      i_version,
    input  wire logic [CLASS_WIDTH-1:0]        i_label,
    output logic                              o_done,

    output logic                              o_runtime_run_valid,
    input  wire logic                         i_runtime_run_ready,
    output logic [TIMESTEP_WIDTH-1:0]          o_runtime_timestep,
    input  wire logic                         i_runtime_run_done,

    output logic                              o_graph_start_valid,
    input  wire logic                         i_graph_start_ready,
    output logic                              o_graph_target,
    output logic [SAMPLE_WIDTH-1:0]            o_graph_sample,
    output logic [TIMESTEP_WIDTH-1:0]          o_graph_timestep,
    output logic [VERSION_WIDTH-1:0]           o_graph_version,
    output logic [CLASS_WIDTH-1:0]             o_graph_label,
    input  wire logic                         i_graph_done,

    output logic                              o_conv_observation_enable,
    output logic                              o_classifier_observation_enable,
    output logic                              o_observation_target,
    output logic                              o_busy,
    output logic                              o_sticky_error
);
    typedef enum logic [3:0] {
        COORD_IDLE,
        COORD_RUNTIME_START,
        COORD_TARGET_START,
        COORD_TARGET_WAIT,
        COORD_STUDENT_START,
        COORD_STUDENT_WAIT,
        COORD_LEARNING_WAIT,
        COORD_RESPOND
    } coord_state_t;

    coord_state_t state_q;
    logic learning_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic [CLASS_WIDTH-1:0] label_q;
    logic runtime_started_q;
    logic runtime_done_seen_q;

    logic request_fire;
    logic runtime_start_fire;
    logic graph_start_fire;

    assign o_request_ready = state_q == COORD_IDLE && !o_sticky_error;
    assign request_fire = i_request_valid && o_request_ready;
    assign o_runtime_run_valid = state_q == COORD_RUNTIME_START;
    assign runtime_start_fire = o_runtime_run_valid && i_runtime_run_ready;
    assign o_graph_start_valid = state_q == COORD_TARGET_START ||
        state_q == COORD_STUDENT_START;
    assign graph_start_fire = o_graph_start_valid && i_graph_start_ready;
    assign o_graph_target = state_q == COORD_TARGET_START ||
        state_q == COORD_TARGET_WAIT;
    assign o_runtime_timestep = timestep_q;
    assign o_graph_sample = sample_q;
    assign o_graph_timestep = timestep_q;
    assign o_graph_version = version_q;
    assign o_graph_label = label_q;
    assign o_conv_observation_enable = learning_q &&
        (state_q == COORD_TARGET_START || state_q == COORD_TARGET_WAIT ||
         state_q == COORD_STUDENT_START || state_q == COORD_STUDENT_WAIT);
    assign o_classifier_observation_enable = learning_q &&
        (state_q == COORD_STUDENT_START || state_q == COORD_STUDENT_WAIT);
    assign o_observation_target = state_q == COORD_TARGET_START ||
        state_q == COORD_TARGET_WAIT;
    assign o_busy = state_q != COORD_IDLE;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= COORD_IDLE;
            learning_q <= 1'b0;
            sample_q <= '0;
            timestep_q <= '0;
            version_q <= '0;
            label_q <= '0;
            runtime_started_q <= 1'b0;
            runtime_done_seen_q <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            state_q <= COORD_IDLE;
            runtime_started_q <= 1'b0;
            runtime_done_seen_q <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == COORD_IDLE)
                o_sticky_error <= 1'b0;

            if (i_runtime_run_done) begin
                if (!learning_q || !runtime_started_q)
                    o_sticky_error <= 1'b1;
                else
                    runtime_done_seen_q <= 1'b1;
            end
            if (i_graph_done && state_q != COORD_TARGET_WAIT &&
                    state_q != COORD_STUDENT_WAIT)
                o_sticky_error <= 1'b1;

            case (state_q)
                COORD_IDLE: begin
                    if (request_fire) begin
                        learning_q <= i_learning_enable;
                        sample_q <= i_sample;
                        timestep_q <= i_timestep;
                        version_q <= i_version;
                        label_q <= i_label;
                        runtime_started_q <= 1'b0;
                        runtime_done_seen_q <= 1'b0;
                        state_q <= i_learning_enable ?
                            COORD_RUNTIME_START : COORD_STUDENT_START;
                    end
                end

                COORD_RUNTIME_START: begin
                    if (runtime_start_fire) begin
                        runtime_started_q <= 1'b1;
                        state_q <= COORD_TARGET_START;
                    end
                end

                COORD_TARGET_START: begin
                    if (graph_start_fire)
                        state_q <= COORD_TARGET_WAIT;
                end

                COORD_TARGET_WAIT: begin
                    if (i_graph_done)
                        state_q <= COORD_STUDENT_START;
                end

                COORD_STUDENT_START: begin
                    if (graph_start_fire)
                        state_q <= COORD_STUDENT_WAIT;
                end

                COORD_STUDENT_WAIT: begin
                    if (i_graph_done) begin
                        if (!learning_q || runtime_done_seen_q ||
                                i_runtime_run_done)
                            state_q <= COORD_RESPOND;
                        else
                            state_q <= COORD_LEARNING_WAIT;
                    end
                end

                COORD_LEARNING_WAIT: begin
                    if (runtime_done_seen_q || i_runtime_run_done)
                        state_q <= COORD_RESPOND;
                end

                COORD_RESPOND: begin
                    o_done <= 1'b1;
                    state_q <= COORD_IDLE;
                end

                default: begin
                    state_q <= COORD_IDLE;
                    o_sticky_error <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
