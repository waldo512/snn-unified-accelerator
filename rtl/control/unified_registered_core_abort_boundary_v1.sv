`timescale 1ns/1ps
`default_nettype none

// The owner arbiter derives i_owner_abort from sticky core diagnostics. Delay
// that cleanup request by one cycle so a leaf diagnostic cannot become a
// same-cycle control path into every other topology. External abort remains
// immediate and suppresses a redundant delayed pulse on the next cycle.
module unified_registered_core_abort_boundary_v1 (
    input  logic i_clk,
    input  logic i_rst_n,
    input  logic i_external_abort,
    input  logic i_owner_abort,
    output logic o_physical_abort,
    output logic o_owner_abort_delayed
);
    logic owner_abort_q;

    assign o_owner_abort_delayed = owner_abort_q;
    assign o_physical_abort = i_external_abort || owner_abort_q;

    // This register fans into memory-side abort gating, so its reset is
    // synchronous with the memory control clock.
    always_ff @(posedge i_clk) begin
        if (!i_rst_n)
            owner_abort_q <= 1'b0;
        else if (i_external_abort)
            owner_abort_q <= 1'b0;
        else
            owner_abort_q <= i_owner_abort;
    end
endmodule

`default_nettype wire
