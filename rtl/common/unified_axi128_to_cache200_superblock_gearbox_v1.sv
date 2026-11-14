`timescale 1ns/1ps
`default_nettype none

// Dense 400-byte transport block:
//   25 x 128-bit AXI payload beats == 16 x 200-bit cache rows.
//
// The deliberately non-overlapped buffer favors a small, deterministic
// boundary over a barrel-shifting streaming datapath. At 66.667 MHz it still
// carries 0.650 GB/s of useful cache-row data, above the modeled layer-major
// SCNN weight demand. Upstream AXI buffering is provided by the DataMover/XPM
// boundary; this block owns only width conversion and backpressure.
module unified_axi128_to_cache200_superblock_gearbox_v1 (
    input  wire logic         i_clk,
    input  wire logic         i_rst_n,
    input  wire logic         i_abort,

    input  wire logic         i_valid,
    output logic              o_ready,
    input  wire logic [127:0] i_data,

    output logic              o_valid,
    input  wire logic         i_ready,
    output logic [199:0]      o_data,
    output logic [3:0]        o_row_index,
    output logic              o_row_first,
    output logic              o_row_last
);
    logic [127:0] beat_q [0:24];
    logic [4:0] write_index_q;
    logic [3:0] row_index_q;
    logic capturing_q;
    logic emitting_q;
    logic input_fire;
    logic output_fire;

    assign o_ready = !i_abort && capturing_q;
    assign o_valid = !i_abort && emitting_q;
    assign o_row_index = row_index_q;
    assign o_row_first = o_valid && row_index_q == 0;
    assign o_row_last = o_valid && row_index_q == 15;
    assign input_fire = i_valid && o_ready;
    assign output_fire = o_valid && i_ready;

    // Each arm is a fixed wiring pattern. This avoids a variable 3200-bit
    // shift while preserving byte-dense external storage.
    always_comb begin
        unique case (row_index_q)
            4'd0:  o_data = {beat_q[1][71:0], beat_q[0]};
            4'd1:  o_data = {beat_q[3][15:0], beat_q[2], beat_q[1][127:72]};
            4'd2:  o_data = {beat_q[4][87:0], beat_q[3][127:16]};
            4'd3:  o_data = {beat_q[6][31:0], beat_q[5], beat_q[4][127:88]};
            4'd4:  o_data = {beat_q[7][103:0], beat_q[6][127:32]};
            4'd5:  o_data = {beat_q[9][47:0], beat_q[8], beat_q[7][127:104]};
            4'd6:  o_data = {beat_q[10][119:0], beat_q[9][127:48]};
            4'd7:  o_data = {beat_q[12][63:0], beat_q[11], beat_q[10][127:120]};
            4'd8:  o_data = {beat_q[14][7:0], beat_q[13], beat_q[12][127:64]};
            4'd9:  o_data = {beat_q[15][79:0], beat_q[14][127:8]};
            4'd10: o_data = {beat_q[17][23:0], beat_q[16], beat_q[15][127:80]};
            4'd11: o_data = {beat_q[18][95:0], beat_q[17][127:24]};
            4'd12: o_data = {beat_q[20][39:0], beat_q[19], beat_q[18][127:96]};
            4'd13: o_data = {beat_q[21][111:0], beat_q[20][127:40]};
            4'd14: o_data = {beat_q[23][55:0], beat_q[22], beat_q[21][127:112]};
            4'd15: o_data = {beat_q[24], beat_q[23][127:56]};
            default: o_data = '0;
        endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            write_index_q <= '0;
            row_index_q <= '0;
            capturing_q <= 1'b1;
            emitting_q <= 1'b0;
        end else if (i_abort) begin
            write_index_q <= '0;
            row_index_q <= '0;
            capturing_q <= 1'b1;
            emitting_q <= 1'b0;
        end else begin
            if (input_fire) begin
                if (write_index_q == 24) begin
                    write_index_q <= '0;
                    row_index_q <= '0;
                    capturing_q <= 1'b0;
                    emitting_q <= 1'b1;
                end else begin
                    write_index_q <= write_index_q + 1'b1;
                end
            end

            if (output_fire) begin
                if (row_index_q == 15) begin
                    row_index_q <= '0;
                    capturing_q <= 1'b1;
                    emitting_q <= 1'b0;
                end else begin
                    row_index_q <= row_index_q + 1'b1;
                end
            end
        end
    end

    // Payload reset is unnecessary because the capture/emission state is the
    // sole validity authority.
    always_ff @(posedge i_clk) begin
        if (input_fire)
            beat_q[write_index_q] <= i_data;
    end
endmodule

`default_nettype wire
