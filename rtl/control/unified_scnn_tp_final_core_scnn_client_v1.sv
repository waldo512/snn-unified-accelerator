`timescale 1ns/1ps
`default_nettype none

// Shared external client for the final core.  Target/student graph layers
// and the graph classifier OUTPUT command use one layer port and one Q64
// service; no SCNN arithmetic or parameter storage is instantiated here.
module unified_scnn_tp_final_core_scnn_client_v1 #(
    parameter int LAYER_WORDS = 12,
    parameter int VERSION_WIDTH = 16,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int FEATURE_COUNT = 2048,
    parameter int CLASS_COUNT = 11
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_graph_busy,
    input  logic                                  i_graph_protocol_error,
    input  logic                                  i_graph_layer_valid,
    output logic                                  o_graph_layer_ready,
    input  logic                                  i_graph_layer_target,
    input  logic [3:0]                            i_graph_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_graph_layer_words,
    input  logic                                  i_graph_input_slot,
    input  logic                                  i_graph_output_slot,
    input  logic [VERSION_WIDTH-1:0]              i_graph_state_version,
    input  logic                                  i_graph_sample_reset,
    input  logic                                  i_graph_sample_last,
    input  logic                                  i_graph_target_observation_enable,
    input  logic                                  i_graph_target_input_prescan_enable,
    input  logic                                  i_graph_student_observation_enable,
    input  logic                                  i_graph_student_input_prescan_enable,
    output logic                                  o_graph_layer_done,
    output logic                                  o_graph_layer_error,

    input  logic                                  i_classifier_start_valid,
    output logic                                  o_classifier_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]               i_classifier_sample,
    input  logic [TIMESTEP_WIDTH-1:0]             i_classifier_timestep,
    input  logic                                  i_post_forward_start_valid,
    output logic                                  o_post_forward_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]               i_post_forward_sample,
    input  logic [TIMESTEP_WIDTH-1:0]             i_post_forward_timestep,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_classifier_parameter_base,
    input  logic [$clog2(FEATURE_COUNT+1)-1:0]    i_classifier_feature_count,
    input  logic [$clog2(CLASS_COUNT+1)-1:0]      i_classifier_class_count,
    input  logic [VERSION_WIDTH-1:0]              i_classifier_state_version,
    input  logic [VERSION_WIDTH-1:0]              i_post_forward_state_version,
    input  logic                                  i_classifier_input_slot,
    input  logic                                  i_classifier_sample_reset,
    input  logic                                  i_classifier_sample_last,
    input  logic                                  i_classifier_observation_enable,
    input  logic                                  i_classifier_input_prescan_enable,
    output logic                                  o_classifier_done,
    output logic                                  o_post_forward_done,
    output logic                                  o_classifier_error,
    output logic                                  o_post_forward_error,

    output logic                                  o_core_session_active,
    output logic                                  o_classifier_session_active,
    output logic                                  o_core_layer_valid,
    input  logic                                  i_core_layer_ready,
    output logic [3:0]                            o_core_layer_index,
    output logic [LAYER_WORDS*32-1:0]             o_core_layer_words,
    output logic                                  o_core_input_slot,
    output logic                                  o_core_output_slot,
    output logic [VERSION_WIDTH-1:0]              o_core_state_version,
    output logic                                  o_core_sample_reset,
    output logic                                  o_core_sample_last,
    output logic                                  o_core_observation_enable,
    output logic                                  o_core_input_prescan_enable,
    input  logic                                  i_core_layer_done,
    input  logic                                  i_core_layer_busy,
    input  logic                                  i_core_layer_error,

    input  logic                                  i_core_weight_read_valid,
    output logic                                  o_core_weight_read_ready,
    input  logic [Q64_ADDR_WIDTH-1:0]             i_core_weight_read_addr0,
    input  logic [Q64_ADDR_WIDTH-1:0]             i_core_weight_read_addr1,
    output logic                                  o_weight_read_valid,
    input  logic                                  i_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]             o_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]             o_weight_read_addr1,
    input  logic                                  i_weight_result_valid,
    output logic                                  o_weight_result_ready,
    input  logic                                  i_core_weight_result_ready,
    output logic                                  o_core_weight_result_valid,
    output logic [63:0]                           o_core_weight_data0,
    output logic [63:0]                           o_core_weight_data1,
    output logic                                  o_core_weight_response_error,
    input  logic [63:0]                           i_weight_data0,
    input  logic [63:0]                           i_weight_data1,
    input  logic                                  i_weight_response_error,

    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    logic graph_session_active;
    logic graph_core_layer_valid;
    logic graph_core_layer_ready;
    logic [3:0] graph_core_layer_index;
    logic [LAYER_WORDS*32-1:0] graph_core_layer_words;
    logic graph_core_input_slot;
    logic graph_core_output_slot;
    logic [VERSION_WIDTH-1:0] graph_core_state_version;
    logic graph_core_sample_reset;
    logic graph_core_sample_last;
    logic graph_core_observation_enable;
    logic graph_core_input_prescan_enable;
    logic graph_core_layer_error;
    logic graph_busy;
    logic graph_protocol_error;
    logic graph_core_weight_read_valid;
    logic graph_core_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0] graph_weight_read_addr0;
    logic [Q64_ADDR_WIDTH-1:0] graph_weight_read_addr1;
    logic graph_weight_read_valid;
    logic graph_weight_result_ready;
    logic graph_core_weight_result_valid;
    logic [63:0] graph_core_weight_data0;
    logic [63:0] graph_core_weight_data1;
    logic graph_core_weight_response_error;

    logic classifier_session_active;
    logic classifier_core_layer_valid;
    logic classifier_core_layer_ready;
    logic [3:0] classifier_core_layer_index;
    logic [LAYER_WORDS*32-1:0] classifier_core_layer_words;
    logic classifier_core_input_slot;
    logic classifier_core_output_slot;
    logic [VERSION_WIDTH-1:0] classifier_core_state_version;
    logic classifier_core_sample_reset;
    logic classifier_core_sample_last;
    logic classifier_core_observation_enable;
    logic classifier_core_input_prescan_enable;
    logic classifier_core_layer_error;
    logic classifier_core_done;
    logic classifier_busy;
    logic classifier_protocol_error;
    logic classifier_core_weight_read_valid;
    logic classifier_core_weight_read_ready;
    logic [Q64_ADDR_WIDTH-1:0] classifier_weight_read_addr0;
    logic [Q64_ADDR_WIDTH-1:0] classifier_weight_read_addr1;
    logic classifier_weight_read_valid;
    logic classifier_weight_result_ready;
    logic classifier_core_weight_result_valid;
    logic [63:0] classifier_core_weight_data0;
    logic [63:0] classifier_core_weight_data1;
    logic classifier_core_weight_response_error;
    logic classifier_start_valid_gated;
    logic classifier_start_ready_i;
    logic classifier_selected_post;
    logic classifier_fire;
    logic classifier_kind_q;
    logic classifier_request_conflict;
    logic [SAMPLE_WIDTH-1:0] classifier_selected_sample;
    logic [TIMESTEP_WIDTH-1:0] classifier_selected_timestep;
    logic [VERSION_WIDTH-1:0] classifier_selected_state_version;
    logic graph_owner;
    logic classifier_owner;

    assign graph_owner = graph_session_active;
    assign classifier_owner = classifier_session_active;
    assign classifier_request_conflict = i_classifier_start_valid &&
        i_post_forward_start_valid;
    assign classifier_selected_post = i_post_forward_start_valid &&
        !i_classifier_start_valid;
    assign classifier_selected_sample = classifier_selected_post ?
        i_post_forward_sample : i_classifier_sample;
    assign classifier_selected_timestep = classifier_selected_post ?
        i_post_forward_timestep : i_classifier_timestep;
    assign classifier_selected_state_version = classifier_selected_post ?
        i_post_forward_state_version : i_classifier_state_version;
    assign classifier_start_valid_gated =
        (i_classifier_start_valid || i_post_forward_start_valid) &&
        !classifier_request_conflict && !graph_owner &&
        !i_graph_protocol_error;
    assign classifier_fire = classifier_start_valid_gated &&
        classifier_start_ready_i;

    unified_scnn_tp_final_core_layer_client_v1 #(
        .LAYER_WORDS(LAYER_WORDS), .VERSION_WIDTH(VERSION_WIDTH),
        .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH)
    ) u_graph_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error), .i_graph_busy(i_graph_busy),
        .i_graph_protocol_error(i_graph_protocol_error),
        .i_graph_layer_valid(i_graph_layer_valid),
        .o_graph_layer_ready(o_graph_layer_ready),
        .i_graph_layer_target(i_graph_layer_target),
        .i_graph_layer_index(i_graph_layer_index),
        .i_graph_layer_words(i_graph_layer_words),
        .i_graph_input_slot(i_graph_input_slot),
        .i_graph_output_slot(i_graph_output_slot),
        .i_graph_state_version(i_graph_state_version),
        .i_graph_sample_reset(i_graph_sample_reset),
        .i_graph_sample_last(i_graph_sample_last),
        .i_graph_target_observation_enable(
            i_graph_target_observation_enable),
        .i_graph_target_input_prescan_enable(
            i_graph_target_input_prescan_enable),
        .i_graph_student_observation_enable(
            i_graph_student_observation_enable),
        .i_graph_student_input_prescan_enable(
            i_graph_student_input_prescan_enable),
        .o_graph_layer_done(o_graph_layer_done),
        .o_graph_layer_error(graph_core_layer_error),
        .o_core_session_active(graph_session_active),
        .o_core_layer_valid(graph_core_layer_valid),
        .i_core_layer_ready(graph_core_layer_ready),
        .o_core_layer_index(graph_core_layer_index),
        .o_core_layer_words(graph_core_layer_words),
        .o_core_input_slot(graph_core_input_slot),
        .o_core_output_slot(graph_core_output_slot),
        .o_core_state_version(graph_core_state_version),
        .o_core_sample_reset(graph_core_sample_reset),
        .o_core_sample_last(graph_core_sample_last),
        .o_core_observation_enable(graph_core_observation_enable),
        .o_core_input_prescan_enable(graph_core_input_prescan_enable),
        .i_core_layer_done(i_core_layer_done && graph_owner),
        .i_core_layer_busy(i_core_layer_busy && graph_owner),
        .i_core_layer_error(i_core_layer_error && graph_owner),
        .i_core_weight_read_valid(
            i_core_weight_read_valid && graph_owner),
        .o_core_weight_read_ready(graph_core_weight_read_ready),
        .i_core_weight_read_addr0(i_core_weight_read_addr0),
        .i_core_weight_read_addr1(i_core_weight_read_addr1),
        .o_weight_read_valid(graph_weight_read_valid),
        .i_weight_read_ready(i_weight_read_ready),
        .o_weight_read_addr0(graph_weight_read_addr0),
        .o_weight_read_addr1(graph_weight_read_addr1),
        .i_weight_result_valid(i_weight_result_valid && graph_owner),
        .i_core_weight_result_ready(i_core_weight_result_ready),
        .o_weight_result_ready(graph_weight_result_ready),
        .i_weight_data0(i_weight_data0), .i_weight_data1(i_weight_data1),
        .i_weight_response_error(i_weight_response_error && graph_owner),
        .o_core_weight_result_valid(graph_core_weight_result_valid),
        .o_core_weight_data0(graph_core_weight_data0),
        .o_core_weight_data1(graph_core_weight_data1),
        .o_core_weight_response_error(graph_core_weight_response_error),
        .o_busy(graph_busy), .o_protocol_error(graph_protocol_error)
    );

    unified_scnn_tp_graph_classifier_output_adapter_v1 #(
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH), .Q64_ADDR_WIDTH(Q64_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .TIMESTEP_WIDTH(TIMESTEP_WIDTH), .FEATURE_COUNT(FEATURE_COUNT),
        .CLASS_COUNT(CLASS_COUNT)
    ) u_classifier_client (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_classifier_start_valid(classifier_start_valid_gated),
        .o_classifier_start_ready(classifier_start_ready_i),
        .i_classifier_sample(classifier_selected_sample),
        .i_classifier_timestep(classifier_selected_timestep),
        .i_parameter_base(i_classifier_parameter_base),
        .i_feature_count(i_classifier_feature_count),
        .i_class_count(i_classifier_class_count),
        .i_state_version(classifier_selected_state_version),
        .i_input_slot(i_classifier_input_slot),
        .i_sample_reset(i_classifier_sample_reset),
        .i_sample_last(i_classifier_sample_last),
        .i_observation_enable(i_classifier_observation_enable),
        .i_input_prescan_enable(i_classifier_input_prescan_enable),
        .o_core_session_active(classifier_session_active),
        .o_core_layer_valid(classifier_core_layer_valid),
        .i_core_layer_ready(classifier_core_layer_ready),
        .o_core_layer_index(classifier_core_layer_index),
        .o_core_layer_words(classifier_core_layer_words),
        .o_core_input_slot(classifier_core_input_slot),
        .o_core_output_slot(classifier_core_output_slot),
        .o_core_state_version(classifier_core_state_version),
        .o_core_sample_reset(classifier_core_sample_reset),
        .o_core_sample_last(classifier_core_sample_last),
        .o_core_observation_enable(classifier_core_observation_enable),
        .o_core_input_prescan_enable(classifier_core_input_prescan_enable),
        .i_core_layer_done(i_core_layer_done && classifier_owner),
        .i_core_layer_busy(i_core_layer_busy && classifier_owner),
        .i_core_layer_error(i_core_layer_error && classifier_owner),
        .i_core_weight_read_valid(
            i_core_weight_read_valid && classifier_owner),
        .o_core_weight_read_ready(classifier_core_weight_read_ready),
        .i_core_weight_read_addr0(i_core_weight_read_addr0),
        .i_core_weight_read_addr1(i_core_weight_read_addr1),
        .o_weight_read_valid(classifier_weight_read_valid),
        .i_weight_read_ready(i_weight_read_ready),
        .o_weight_read_addr0(classifier_weight_read_addr0),
        .o_weight_read_addr1(classifier_weight_read_addr1),
        .i_weight_result_valid(i_weight_result_valid && classifier_owner),
        .i_core_weight_result_ready(i_core_weight_result_ready),
        .o_weight_result_ready(classifier_weight_result_ready),
        .i_weight_data0(i_weight_data0), .i_weight_data1(i_weight_data1),
        .i_weight_response_error(i_weight_response_error && classifier_owner),
        .o_core_weight_result_valid(classifier_core_weight_result_valid),
        .o_core_weight_data0(classifier_core_weight_data0),
        .o_core_weight_data1(classifier_core_weight_data1),
        .o_core_weight_response_error(classifier_core_weight_response_error),
        .o_active_sample(), .o_active_timestep(),
        .o_classifier_done(classifier_core_done),
        .o_classifier_error(classifier_core_layer_error),
        .o_busy(classifier_busy), .o_protocol_error(classifier_protocol_error)
    );

    assign graph_core_layer_ready = i_core_layer_ready && graph_owner;
    assign classifier_core_layer_ready = i_core_layer_ready &&
        classifier_owner;
    assign o_classifier_start_ready = classifier_start_ready_i &&
        !graph_owner && !i_graph_protocol_error &&
        !i_post_forward_start_valid && !classifier_request_conflict;
    assign o_post_forward_start_ready = classifier_start_ready_i &&
        !graph_owner && !i_graph_protocol_error &&
        !i_classifier_start_valid && !classifier_request_conflict;
    assign o_classifier_done = classifier_core_done && !classifier_kind_q;
    assign o_post_forward_done = classifier_core_done && classifier_kind_q;
    assign o_graph_layer_error = graph_core_layer_error;
    assign o_classifier_error = classifier_core_layer_error &&
        !classifier_kind_q;
    assign o_post_forward_error = classifier_core_layer_error &&
        classifier_kind_q;
    assign o_core_session_active = graph_session_active ||
        classifier_session_active;
    assign o_classifier_session_active = classifier_session_active;
    assign o_core_layer_valid = graph_core_layer_valid ||
        classifier_core_layer_valid;
    assign o_core_layer_index = graph_owner ? graph_core_layer_index :
        classifier_core_layer_index;
    assign o_core_layer_words = graph_owner ? graph_core_layer_words :
        classifier_core_layer_words;
    assign o_core_input_slot = graph_owner ? graph_core_input_slot :
        classifier_core_input_slot;
    assign o_core_output_slot = graph_owner ? graph_core_output_slot :
        classifier_core_output_slot;
    assign o_core_state_version = graph_owner ? graph_core_state_version :
        classifier_core_state_version;
    assign o_core_sample_reset = graph_owner ? graph_core_sample_reset :
        classifier_core_sample_reset;
    assign o_core_sample_last = graph_owner ? graph_core_sample_last :
        classifier_core_sample_last;
    assign o_core_observation_enable = graph_owner ?
        graph_core_observation_enable : classifier_core_observation_enable;
    assign o_core_input_prescan_enable = graph_owner ?
        graph_core_input_prescan_enable :
        classifier_core_input_prescan_enable;

    assign o_core_weight_read_ready = graph_owner ?
        graph_core_weight_read_ready : classifier_core_weight_read_ready;
    assign o_weight_read_valid = graph_owner ? graph_weight_read_valid :
        classifier_weight_read_valid;
    assign o_weight_read_addr0 = graph_owner ? graph_weight_read_addr0 :
        classifier_weight_read_addr0;
    assign o_weight_read_addr1 = graph_owner ? graph_weight_read_addr1 :
        classifier_weight_read_addr1;
    assign o_weight_result_ready = graph_owner ? graph_weight_result_ready :
        classifier_weight_result_ready;
    assign o_core_weight_result_valid = graph_owner ?
        graph_core_weight_result_valid : classifier_core_weight_result_valid;
    assign o_core_weight_data0 = graph_owner ? graph_core_weight_data0 :
        classifier_core_weight_data0;
    assign o_core_weight_data1 = graph_owner ? graph_core_weight_data1 :
        classifier_core_weight_data1;
    assign o_core_weight_response_error = graph_owner ?
        graph_core_weight_response_error : classifier_core_weight_response_error;
    assign o_busy = graph_busy || classifier_busy;
    assign o_protocol_error = graph_protocol_error ||
        classifier_protocol_error || graph_core_layer_error ||
        classifier_core_layer_error || classifier_request_conflict;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            classifier_kind_q <= 1'b0;
        end else if (i_abort || i_clear_error) begin
            classifier_kind_q <= 1'b0;
        end else begin
            if (classifier_fire)
                classifier_kind_q <= classifier_selected_post;
            else if (classifier_core_done)
                classifier_kind_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
