`timescale 1ns/1ps
`default_nettype none

// Selects a row-command source at the registered descriptor boundary. The
// canonical FC source is a same-cycle ready/valid bypass; extension sources
// cannot create memory traffic while MODE_SPARSE_FC is selected.
module unified_fc_first_row_arbiter #(
    parameter integer NUMERIC_PROFILE_BITS = 2
) (
    input  wire logic                                  clk,
    input  wire logic                                  rst_n,
    input  wire logic                                  i_clear_error,
    input  wire logic [1:0]                            i_mode_q,

    input  wire logic                                  i_fc_valid,
    output logic                                       o_fc_ready,
    input  wire logic [31:0]                           i_fc_row_base,
    input  wire logic [3:0]                            i_fc_active_tiles,
    input  wire logic signed [7:0]                     i_fc_activation,
    input  wire logic [NUMERIC_PROFILE_BITS-1:0]       i_fc_numeric_profile,
    input  wire logic                                  i_fc_timestep_last,
    input  wire logic                                  i_fc_sample_last,

    input  wire logic                                  i_recurrent_valid,
    output logic                                       o_recurrent_ready,
    input  wire logic [31:0]                           i_recurrent_row_base,
    input  wire logic [3:0]                            i_recurrent_active_tiles,
    input  wire logic signed [7:0]                     i_recurrent_activation,
    input  wire logic [NUMERIC_PROFILE_BITS-1:0]       i_recurrent_numeric_profile,
    input  wire logic                                  i_recurrent_timestep_last,
    input  wire logic                                  i_recurrent_sample_last,

    input  wire logic                                  i_conv_valid,
    output logic                                       o_conv_ready,
    input  wire logic [31:0]                           i_conv_row_base,
    input  wire logic [3:0]                            i_conv_active_tiles,
    input  wire logic signed [7:0]                     i_conv_activation,
    input  wire logic [NUMERIC_PROFILE_BITS-1:0]       i_conv_numeric_profile,
    input  wire logic                                  i_conv_timestep_last,
    input  wire logic                                  i_conv_sample_last,

    output logic                                       o_row_valid,
    input  wire logic                                  i_row_ready,
    output logic [31:0]                                o_row_base,
    output logic [3:0]                                 o_active_tiles,
    output logic signed [7:0]                          o_activation,
    output logic [NUMERIC_PROFILE_BITS-1:0]            o_numeric_profile,
    output logic                                       o_timestep_last,
    output logic                                       o_sample_last,
    output logic                                       o_extension_active,
    output logic                                       o_sticky_error
);
    localparam logic [1:0] MODE_SPARSE_FC = 2'd0;
    localparam logic [1:0] MODE_RECURRENT_FC = 2'd1;
    localparam logic [1:0] MODE_CONV = 2'd2;

    logic source_violation;

    always_comb begin
        o_fc_ready = 1'b0;
        o_recurrent_ready = 1'b0;
        o_conv_ready = 1'b0;
        o_row_valid = 1'b0;
        o_row_base = 32'd0;
        o_active_tiles = 4'd0;
        o_activation = 8'sd0;
        o_numeric_profile = '0;
        o_timestep_last = 1'b0;
        o_sample_last = 1'b0;
        o_extension_active = 1'b0;
        source_violation = 1'b0;

        unique case (i_mode_q)
            MODE_SPARSE_FC: begin
                o_fc_ready = i_row_ready;
                o_row_valid = i_fc_valid;
                o_row_base = i_fc_row_base;
                o_active_tiles = i_fc_active_tiles;
                o_activation = i_fc_activation;
                o_numeric_profile = i_fc_numeric_profile;
                o_timestep_last = i_fc_timestep_last;
                o_sample_last = i_fc_sample_last;
                source_violation = i_recurrent_valid || i_conv_valid;
            end
            MODE_RECURRENT_FC: begin
                o_recurrent_ready = i_row_ready;
                o_row_valid = i_recurrent_valid;
                o_row_base = i_recurrent_row_base;
                o_active_tiles = i_recurrent_active_tiles;
                o_activation = i_recurrent_activation;
                o_numeric_profile = i_recurrent_numeric_profile;
                o_timestep_last = i_recurrent_timestep_last;
                o_sample_last = i_recurrent_sample_last;
                o_extension_active = i_recurrent_valid;
                source_violation = i_fc_valid || i_conv_valid;
            end
            MODE_CONV: begin
                o_conv_ready = i_row_ready;
                o_row_valid = i_conv_valid;
                o_row_base = i_conv_row_base;
                o_active_tiles = i_conv_active_tiles;
                o_activation = i_conv_activation;
                o_numeric_profile = i_conv_numeric_profile;
                o_timestep_last = i_conv_timestep_last;
                o_sample_last = i_conv_sample_last;
                o_extension_active = i_conv_valid;
                source_violation = i_fc_valid || i_recurrent_valid;
            end
            default: source_violation = i_fc_valid || i_recurrent_valid || i_conv_valid;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            o_sticky_error <= 1'b0;
        else if (i_clear_error)
            o_sticky_error <= 1'b0;
        else if (source_violation)
            o_sticky_error <= 1'b1;
    end
endmodule

`default_nettype wire
