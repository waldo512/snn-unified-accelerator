`timescale 1ns/1ps
`default_nettype none

// Standalone POOL is deliberately rejected in final-v1. The validated VGG-9
// path performs pooling in the CONV finalizer. This adapter retires the
// offending layer once, then holds a sticky error instead of silently hanging.
module unified_final_pool_reject_v1 (
    input  logic       i_clk,
    input  logic       i_rst_n,
    input  logic       i_clear_error,
    input  logic       i_abort,
    input  logic       i_layer_valid,
    output logic       o_layer_ready,
    output logic       o_layer_done,
    output logic       o_busy,
    output logic       o_sticky_error,
    output logic [7:0] o_error_code
);
    logic retire_q;

    assign o_layer_ready = !retire_q && !o_sticky_error;
    assign o_layer_done = retire_q;
    assign o_busy = retire_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            retire_q <= 1'b0;
            o_sticky_error <= 1'b0;
            o_error_code <= 8'h00;
        end else begin
            if (retire_q)
                retire_q <= 1'b0;
            if (i_layer_valid && o_layer_ready) begin
                retire_q <= 1'b1;
                o_sticky_error <= 1'b1;
                o_error_code <= 8'h31;
            end
            if (i_abort)
                retire_q <= 1'b0;
            if (i_clear_error && !retire_q) begin
                o_sticky_error <= 1'b0;
                o_error_code <= 8'h00;
            end
        end
    end
endmodule

`default_nettype wire
