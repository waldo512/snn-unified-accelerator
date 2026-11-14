`timescale 1ns/1ps
`default_nettype none

// A selected-layer learning transaction updates the shared parameter image in
// place and cannot be cut in half. Abort is therefore immediate only while no
// learning transaction is active; otherwise it is delivered to the scheduler
// at the first completed, quiescent transaction boundary.
module unified_scnn_tp_learning_abort_guard (
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_abort_request,
    input  logic i_learning_transaction_active,
    input  logic i_learning_transaction_safe,
    output logic o_scheduler_abort,
    output logic o_child_abort,
    output logic o_abort_pending,
    output logic o_deferred_abort_fire
);
    logic abort_pending_q;

    assign o_deferred_abort_fire = abort_pending_q &&
        i_learning_transaction_safe;
    assign o_scheduler_abort =
        (i_abort_request && !i_learning_transaction_active &&
         !abort_pending_q) || o_deferred_abort_fire;
    assign o_child_abort = i_abort_request &&
        !i_learning_transaction_active && !abort_pending_q;
    assign o_abort_pending = abort_pending_q;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            abort_pending_q <= 1'b0;
        else begin
            if (o_deferred_abort_fire)
                abort_pending_q <= 1'b0;
            else if (i_abort_request && i_learning_transaction_active)
                abort_pending_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
