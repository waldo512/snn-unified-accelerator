`timescale 1ns/1ps
`default_nettype none

// Couples one layer-workset command to a DataMover descriptor and the shared
// 128<->320 gearbox. The three parties accept the command atomically, so a
// descriptor can never run without a matching payload conversion phase.
module unified_scnn_layer_workset_dma_bridge_v1 #(
    parameter int EXT_ADDR_WIDTH = 32,
    parameter int BTT_WIDTH = 23
) (
    input  wire logic                      i_clk,
    input  wire logic                      i_rst_n,
    input  wire logic                      i_abort,
    input  wire logic                      i_clear_error,

    input  wire logic                      i_workset_command_valid,
    output logic                           o_workset_command_ready,
    input  wire logic                      i_workset_command_write,
    input  wire logic [EXT_ADDR_WIDTH-1:0] i_workset_command_base,
    input  wire logic [31:0]               i_workset_command_records,

    output logic                           o_dma_command_valid,
    input  wire logic                      i_dma_command_ready,
    output logic                           o_dma_command_write,
    output logic [EXT_ADDR_WIDTH-1:0]      o_dma_command_base,
    output logic [31:0]                    o_dma_command_records,
    output logic [31:0]                    o_dma_command_beats,
    output logic [BTT_WIDTH-1:0]           o_dma_command_bytes,
    output logic                           o_dma_abort,

    input  wire logic                      i_dma_read_valid,
    output logic                           o_dma_read_ready,
    input  wire logic [127:0]              i_dma_read_data,
    input  wire logic                      i_dma_read_last,
    output logic                           o_dma_write_valid,
    input  wire logic                      i_dma_write_ready,
    output logic [127:0]                   o_dma_write_data,
    output logic                           o_dma_write_last,

    output logic                           o_load_record_valid,
    input  wire logic                      i_load_record_ready,
    output logic [319:0]                   o_load_record_data,
    output logic                           o_load_record_last,
    input  wire logic                      i_store_record_valid,
    output logic                           o_store_record_ready,
    input  wire logic [319:0]              i_store_record_data,
    input  wire logic                      i_store_record_last,

    output logic                           o_busy,
    output logic                           o_done,
    output logic                           o_sticky_error
);
    logic gearbox_start_valid;
    logic gearbox_start_ready;
    logic gearbox_busy;
    logic gearbox_done;
    logic gearbox_error;
    logic local_error_q;
    logic command_legal;
    logic invalid_command_fire;
    logic [34:0] command_bytes_wide;
    logic [34:0] command_beats_wide;

    assign command_bytes_wide =
        ({3'b000, i_workset_command_records} << 5) +
        ({3'b000, i_workset_command_records} << 3);
    assign command_beats_wide =
        (({3'b000, i_workset_command_records} << 2) +
         {3'b000, i_workset_command_records}) >> 1;
    assign command_legal = i_workset_command_records != 0 &&
        !i_workset_command_records[0] &&
        command_bytes_wide[34:BTT_WIDTH] == '0 &&
        command_beats_wide[34:32] == '0;

    // A legal command fires only when both sinks can take it in this cycle.
    // An illegal command is consumed locally and never reaches DataMover.
    assign o_workset_command_ready = !i_abort && !o_sticky_error &&
        !gearbox_busy &&
        (command_legal ?
            (gearbox_start_ready && i_dma_command_ready) : 1'b1);
    assign invalid_command_fire = i_workset_command_valid &&
        o_workset_command_ready && !command_legal;

    assign o_dma_command_valid = !i_abort && !o_sticky_error &&
        i_workset_command_valid && command_legal && gearbox_start_ready;
    assign gearbox_start_valid = !i_abort && !o_sticky_error &&
        i_workset_command_valid && command_legal && i_dma_command_ready;
    assign o_dma_command_write = i_workset_command_write;
    assign o_dma_command_base = i_workset_command_base;
    assign o_dma_command_records = i_workset_command_records;
    assign o_dma_command_beats = command_beats_wide[31:0];
    assign o_dma_command_bytes = command_bytes_wide[BTT_WIDTH-1:0];
    assign o_dma_abort = i_abort && gearbox_busy;

    assign o_busy = gearbox_busy;
    assign o_done = gearbox_done;
    assign o_sticky_error = local_error_q || gearbox_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            local_error_q <= 1'b0;
        else if (i_clear_error && !gearbox_busy)
            local_error_q <= 1'b0;
        else if (invalid_command_fire)
            local_error_q <= 1'b1;
    end

    unified_axi128_scnn320_record_gearbox_v1 u_gearbox (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_start_valid(gearbox_start_valid),
        .o_start_ready(gearbox_start_ready),
        .i_direction_store(i_workset_command_write),
        .i_record_count(i_workset_command_records),
        .i_axi_valid(i_dma_read_valid),
        .o_axi_ready(o_dma_read_ready),
        .i_axi_data(i_dma_read_data),
        .i_axi_last(i_dma_read_last),
        .o_axi_valid(o_dma_write_valid),
        .i_axi_ready(i_dma_write_ready),
        .o_axi_data(o_dma_write_data),
        .o_axi_last(o_dma_write_last),
        .i_record_valid(i_store_record_valid),
        .o_record_ready(o_store_record_ready),
        .i_record_data(i_store_record_data),
        .i_record_last(i_store_record_last),
        .o_record_valid(o_load_record_valid),
        .i_record_ready(i_load_record_ready),
        .o_record_data(o_load_record_data),
        .o_record_last(o_load_record_last),
        .o_busy(gearbox_busy),
        .o_done(gearbox_done),
        .o_sticky_error(gearbox_error)
    );
endmodule

`default_nettype wire
