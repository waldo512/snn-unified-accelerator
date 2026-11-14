`timescale 1ns/1ps
`default_nettype none

// Adapts the production SCNN target/student graph boundary to the external
// layer client of unified_snn_core_final_v1.  This module owns only protocol
// lifetime: the graph remains the source of descriptors and the final core
// remains the sole physical CONV executor.
module unified_scnn_tp_final_core_layer_client_v1 #(
    parameter int LAYER_WORDS = 12,
    parameter int VERSION_WIDTH = 16,
    parameter int Q64_ADDR_WIDTH = 22
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

    output logic                                  o_core_session_active,
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
    input  logic                                  i_core_weight_result_ready,
    output logic                                  o_weight_result_ready,
    input  logic [63:0]                           i_weight_data0,
    input  logic [63:0]                           i_weight_data1,
    input  logic                                  i_weight_response_error,
    output logic                                  o_core_weight_result_valid,
    output logic [63:0]                           o_core_weight_data0,
    output logic [63:0]                           o_core_weight_data1,
    output logic                                  o_core_weight_response_error,

    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    logic session_q;
    logic layer_active_q;
    logic protocol_error_q;
    logic layer_accept;

    assign o_core_session_active = session_q || i_graph_busy ||
        i_graph_layer_valid || layer_active_q;
    assign o_core_layer_valid = i_graph_layer_valid &&
        o_core_session_active && !protocol_error_q;
    assign o_graph_layer_ready = i_core_layer_ready &&
        o_core_session_active && !protocol_error_q;
    assign layer_accept = o_core_layer_valid && o_graph_layer_ready;

    assign o_core_layer_index = i_graph_layer_index;
    assign o_core_layer_words = i_graph_layer_words;
    assign o_core_input_slot = i_graph_input_slot;
    assign o_core_output_slot = i_graph_output_slot;
    assign o_core_state_version = i_graph_state_version;
    assign o_core_sample_reset = i_graph_sample_reset;
    assign o_core_sample_last = i_graph_sample_last;
    assign o_core_observation_enable = i_graph_layer_target ?
        i_graph_target_observation_enable :
        i_graph_student_observation_enable;
    assign o_core_input_prescan_enable = i_graph_layer_target ?
        i_graph_target_input_prescan_enable :
        i_graph_student_input_prescan_enable;

    assign o_graph_layer_done = i_core_layer_done && layer_active_q;
    assign o_graph_layer_error = protocol_error_q || i_core_layer_error;

    // Q64 service is a transparent single-client boundary.  The final core
    // still owns request timing; the external store owns only the response.
    assign o_weight_read_valid = i_core_weight_read_valid &&
        o_core_session_active && !protocol_error_q;
    assign o_weight_read_addr0 = i_core_weight_read_addr0;
    assign o_weight_read_addr1 = i_core_weight_read_addr1;
    assign o_core_weight_read_ready = o_core_session_active &&
        !protocol_error_q && i_weight_read_ready;
    assign o_weight_result_ready = i_core_weight_result_ready;
    assign o_core_weight_result_valid = i_weight_result_valid &&
        o_core_session_active && !protocol_error_q;
    assign o_core_weight_data0 = i_weight_data0;
    assign o_core_weight_data1 = i_weight_data1;
    assign o_core_weight_response_error = i_weight_response_error;

    assign o_busy = o_core_session_active || i_core_layer_busy ||
        layer_active_q;
    assign o_protocol_error = protocol_error_q || i_graph_protocol_error ||
        i_core_layer_error || i_weight_response_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            session_q <= 1'b0;
            layer_active_q <= 1'b0;
            protocol_error_q <= 1'b0;
        end else if (i_abort) begin
            session_q <= 1'b0;
            layer_active_q <= 1'b0;
        end else begin
            if (i_clear_error && !o_busy)
                protocol_error_q <= 1'b0;
            else if (i_graph_protocol_error || i_core_layer_error ||
                     i_weight_response_error)
                protocol_error_q <= 1'b1;

            if (i_graph_busy)
                session_q <= 1'b1;
            if (layer_accept)
                layer_active_q <= 1'b1;
            if (i_core_layer_done)
                layer_active_q <= 1'b0;

            // Do not drop the external session between the last graph command
            // and its physical retirement.  The graph's busy indication is
            // the transaction-level lifetime; the layer/core flags cover the
            // final completion cycle.
            if (session_q && !i_graph_busy && !layer_active_q &&
                !i_core_layer_busy && !i_graph_layer_valid)
                session_q <= 1'b0;
        end
    end
endmodule

`default_nettype wire
