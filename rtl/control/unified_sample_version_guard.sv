`timescale 1ns/1ps
`default_nettype none

module unified_sample_version_guard (
    input  wire logic        clk,
    input  wire logic        rst_n,
    input  wire logic        i_sample_start,
    input  wire logic        i_sample_done,
    input  wire logic [15:0] i_profile_version,
    input  wire logic [15:0] i_graph_version,
    input  wire logic [15:0] i_model_version,
    input  wire logic [15:0] i_adapt_version,
    input  wire logic        i_clear_error,
    output logic             o_sample_active,
    output logic [63:0]      o_locked_versions,
    output logic             o_version_error
);
    logic [63:0] active_versions;

    always_comb begin
        active_versions = {i_profile_version, i_graph_version,
                           i_model_version, i_adapt_version};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_sample_active <= 1'b0;
            o_locked_versions <= 64'd0;
            o_version_error <= 1'b0;
        end else begin
            if (i_clear_error)
                o_version_error <= 1'b0;
            if (i_sample_start) begin
                if (o_sample_active)
                    o_version_error <= 1'b1;
                else begin
                    o_sample_active <= 1'b1;
                    o_locked_versions <= active_versions;
                end
            end
            if (o_sample_active && (active_versions != o_locked_versions))
                o_version_error <= 1'b1;
            if (i_sample_done)
                o_sample_active <= 1'b0;
        end
    end
endmodule

`default_nettype wire
