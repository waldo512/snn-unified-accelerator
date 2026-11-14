`default_nettype none

// Binds the full H480 descriptor lifecycle to the SCNN channel-wave engines.
// Feature/weight generation remains an external ready/valid producer.
module unified_scnn_h480_pass_compute_bridge #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_scnn_owner_enable,
    input  logic                                i_clear_error,
    input  logic                                i_sequence_start,
    output logic                                o_sequence_start_ready,

    input  logic                                i_tile_valid,
    output logic                                o_tile_ready,
    input  logic [3:0]                          i_layer_index,
    input  logic [9:0]                          i_output_channel_base,
    input  logic [9:0]                          i_output_channel_count,
    input  logic [9:0]                          i_input_channel_base,
    input  logic [9:0]                          i_input_channel_count,
    input  logic [15:0]                         i_tile_tag,

    input  logic                                i_chunk_valid,
    output logic                                o_chunk_ready,
    input  logic [LANES*16-1:0]                 i_feature_flat,
    input  logic [ENGINES*LANES*16-1:0]         i_weight_flat,
    input  logic [LANES-1:0]                    i_lane_mask,
    input  logic                                i_chunk_first,
    input  logic                                i_chunk_last,

    output logic [ENGINES-1:0]                  o_wave_engine_mask,
    output logic [CHANNEL_WIDTH-1:0]            o_wave_channel_base,
    output logic                                o_result_valid,
    input  logic                                i_result_ready,
    output logic [ENGINES-1:0]                  o_result_mask,
    output logic [ENGINES*CHANNEL_WIDTH-1:0]    o_result_channel_flat,
    output logic [ENGINES*ACC_WIDTH-1:0]        o_result_accumulator_flat,

    output logic                                o_tile_done,
    output logic                                o_sequence_done,
    output logic                                o_busy,
    output logic                                o_sticky_error
);
    logic scheduler_compute_start;
    logic executor_start_ready;
    logic executor_done;
    logic executor_busy;
    logic scheduler_tile_busy;
    logic scheduler_sequence_busy;
    logic scheduler_error;
    logic executor_error;
    logic integration_error_q;
    logic [3:0] active_layer;
    logic [9:0] active_output_base;
    logic [9:0] active_output_count;
    logic [9:0] active_input_base;
    logic [9:0] active_input_count;
    logic [15:0] active_tile_tag;
    logic [3:0] active_h40_tiles;
    logic input_slot;
    logic output_slot;
    logic accumulator_clear;
    logic accumulator_finalize;
    logic lif_pool_commit;
    logic output_pass_first;
    logic output_pass_last;
    logic layer_first_tile;
    logic layer_last_tile;
    logic classifier_tile;

    assign o_busy = scheduler_sequence_busy || scheduler_tile_busy || executor_busy;
    assign o_sticky_error = scheduler_error || executor_error || integration_error_q;

    unified_scnn_h480_pass_scheduler u_scheduler (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_sequence_start(i_sequence_start && i_scnn_owner_enable),
        .o_sequence_start_ready(o_sequence_start_ready),
        .i_clear_error(i_clear_error), .i_abort(1'b0),
        .i_tile_valid(i_tile_valid), .o_tile_ready(o_tile_ready),
        .i_layer_index(i_layer_index),
        .i_output_channel_base(i_output_channel_base),
        .i_output_channel_count(i_output_channel_count),
        .i_input_channel_base(i_input_channel_base),
        .i_input_channel_count(i_input_channel_count), .i_tile_tag(i_tile_tag),
        .o_compute_start(scheduler_compute_start),
        .i_compute_done(executor_done), .o_layer_index(active_layer),
        .o_output_channel_base(active_output_base),
        .o_output_channel_count(active_output_count),
        .o_input_channel_base(active_input_base),
        .o_input_channel_count(active_input_count), .o_tile_tag(active_tile_tag),
        .o_active_h40_tiles(active_h40_tiles), .o_input_slot(input_slot),
        .o_output_slot(output_slot), .o_accumulator_clear(accumulator_clear),
        .o_accumulator_finalize(accumulator_finalize),
        .o_lif_pool_commit(lif_pool_commit),
        .o_output_pass_first(output_pass_first),
        .o_output_pass_last(output_pass_last),
        .o_layer_first_tile(layer_first_tile),
        .o_layer_last_tile(layer_last_tile),
        .o_classifier_tile(classifier_tile), .o_tile_busy(scheduler_tile_busy),
        .o_sequence_busy(scheduler_sequence_busy), .o_tile_done(o_tile_done),
        .o_sequence_done(o_sequence_done), .o_sticky_error(scheduler_error)
    );

    unified_scnn_h480_channel_wave_executor #(
        .ENGINES(ENGINES), .LANES(LANES), .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_executor (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_owner_enable(i_scnn_owner_enable), .i_clear_error(i_clear_error),
        .i_start(scheduler_compute_start), .o_start_ready(executor_start_ready),
        .i_output_channel_base(active_output_base),
        .i_output_channel_count(active_output_count),
        .i_chunk_valid(i_chunk_valid), .o_chunk_ready(o_chunk_ready),
        .i_feature_flat(i_feature_flat), .i_weight_flat(i_weight_flat),
        .i_lane_mask(i_lane_mask), .i_chunk_first(i_chunk_first),
        .i_chunk_last(i_chunk_last), .o_wave_engine_mask(o_wave_engine_mask),
        .o_wave_channel_base(o_wave_channel_base),
        .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
        .o_result_mask(o_result_mask),
        .o_result_channel_flat(o_result_channel_flat),
        .o_result_accumulator_flat(o_result_accumulator_flat),
        .o_busy(executor_busy), .o_done(executor_done),
        .o_sticky_error(executor_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            integration_error_q <= 1'b0;
        else begin
            if (i_clear_error && !o_busy)
                integration_error_q <= 1'b0;
            if ((i_sequence_start && !i_scnn_owner_enable) ||
                (scheduler_compute_start && !executor_start_ready))
                integration_error_q <= 1'b1;
        end
    end

    logic unused_status;
    assign unused_status = &{1'b0, active_layer, active_input_base,
        active_input_count, active_tile_tag, active_h40_tiles, input_slot,
        output_slot, accumulator_clear, accumulator_finalize, lif_pool_commit,
        output_pass_first, output_pass_last, layer_first_tile, layer_last_tile,
        classifier_tile};
endmodule

`default_nettype wire
