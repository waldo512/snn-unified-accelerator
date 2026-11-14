`timescale 1ns/1ps
`default_nettype none

module unified_pool2d_unit (
    input  wire logic                       clk,
    input  wire logic                       rst_n,
    input  wire logic                       i_valid,
    output logic                            o_ready,
    input  wire logic signed [31:0]         i_value,
    input  wire logic                       i_window_first,
    input  wire logic                       i_window_last,
    input  wire logic                       i_average_mode,
    input  wire logic [4:0]                 i_average_shift,
    output logic                            o_valid,
    input  wire logic                       i_ready,
    output logic signed [31:0]              o_value,
    output logic                            o_protocol_error
);
    logic signed [39:0] sum_q;
    logic signed [31:0] max_q;
    logic window_active_q;
    logic signed [39:0] next_sum;
    logic signed [39:0] shifted_sum;
    logic signed [31:0] next_max;

    always_comb begin
        o_ready = !o_valid || i_ready;
        next_sum = i_window_first ? {{8{i_value[31]}}, i_value} :
            sum_q + {{8{i_value[31]}}, i_value};
        shifted_sum = next_sum >>> i_average_shift;
        if (i_window_first || (i_value > max_q))
            next_max = i_value;
        else
            next_max = max_q;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_q <= 0;
            max_q <= 0;
            window_active_q <= 1'b0;
            o_valid <= 1'b0;
            o_value <= 0;
            o_protocol_error <= 1'b0;
        end else begin
            if (o_valid && i_ready)
                o_valid <= 1'b0;

            if (i_valid && o_ready) begin
                if (i_window_first) begin
                    window_active_q <= 1'b1;
                end else if (!window_active_q) begin
                    o_protocol_error <= 1'b1;
                end
                sum_q <= next_sum;
                max_q <= next_max;
                if (i_window_last) begin
                    if (!window_active_q && !i_window_first)
                        o_protocol_error <= 1'b1;
                    if (i_average_mode)
                        o_value <= shifted_sum[31:0];
                    else
                        o_value <= next_max;
                    o_valid <= 1'b1;
                    window_active_q <= 1'b0;
                end
            end
        end
    end
endmodule

`default_nettype wire
