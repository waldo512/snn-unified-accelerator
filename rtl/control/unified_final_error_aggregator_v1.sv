`timescale 1ns/1ps
`default_nettype none

// Priority encoder for already-sticky leaf errors. Quiescence is deliberately
// handled elsewhere so an error cannot prevent the idle state needed to clear it.
module unified_final_error_aggregator_v1 (
    input  logic       i_control_error,
    input  logic [7:0] i_control_code,
    input  logic       i_dispatch_error,
    input  logic [7:0] i_dispatch_code,
    input  logic       i_datapath_error,
    input  logic [7:0] i_datapath_code,
    input  logic       i_scnn_layer_error,
    input  logic [7:0] i_scnn_layer_code,
    input  logic       i_scnn_output_error,
    input  logic [7:0] i_scnn_output_code,
    input  logic       i_memory_error,
    input  logic [7:0] i_memory_code,
    output logic       o_fail_stop,
    output logic       o_sticky_error,
    output logic [2:0] o_error_source,
    output logic [7:0] o_error_code
);
    localparam logic [2:0] SOURCE_NONE = 3'd0;
    localparam logic [2:0] SOURCE_CONTROL = 3'd1;
    localparam logic [2:0] SOURCE_DISPATCH = 3'd2;
    localparam logic [2:0] SOURCE_DATAPATH = 3'd3;
    localparam logic [2:0] SOURCE_SCNN_LAYER = 3'd4;
    localparam logic [2:0] SOURCE_SCNN_OUTPUT = 3'd5;
    localparam logic [2:0] SOURCE_MEMORY = 3'd6;

    always_comb begin
        o_sticky_error = i_control_error || i_dispatch_error ||
            i_datapath_error || i_scnn_layer_error ||
            i_scnn_output_error || i_memory_error;
        o_fail_stop = o_sticky_error;
        o_error_source = SOURCE_NONE;
        o_error_code = 8'h00;
        if (i_control_error) begin
            o_error_source = SOURCE_CONTROL;
            o_error_code = i_control_code;
        end else if (i_dispatch_error) begin
            o_error_source = SOURCE_DISPATCH;
            o_error_code = i_dispatch_code;
        end else if (i_datapath_error) begin
            o_error_source = SOURCE_DATAPATH;
            o_error_code = i_datapath_code;
        end else if (i_scnn_layer_error) begin
            o_error_source = SOURCE_SCNN_LAYER;
            o_error_code = i_scnn_layer_code;
        end else if (i_scnn_output_error) begin
            o_error_source = SOURCE_SCNN_OUTPUT;
            o_error_code = i_scnn_output_code;
        end else if (i_memory_error) begin
            o_error_source = SOURCE_MEMORY;
            o_error_code = i_memory_code;
        end
    end
endmodule

`default_nettype wire
