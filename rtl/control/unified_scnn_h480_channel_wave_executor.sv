`default_nettype none

// Maps one H480 scheduler output-channel pass onto twelve SCNN side engines.
// Spatial traversal and memory addressing remain outside this module.
module unified_scnn_h480_channel_wave_executor #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48,
    parameter int OUTPUT_LIMIT = 512
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_owner_enable,
    input  logic                                i_clear_error,
    input  logic                                i_start,
    output logic                                o_start_ready,
    input  logic [CHANNEL_WIDTH-1:0]            i_output_channel_base,
    input  logic [CHANNEL_WIDTH-1:0]            i_output_channel_count,

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
    output logic                                o_busy,
    output logic                                o_done,
    output logic                                o_sticky_error
);
    logic [CHANNEL_WIDTH-1:0] remaining_q;
    logic [CHANNEL_WIDTH-1:0] wave_base_q;
    logic [3:0] wave_engine_count_q;
    logic [ENGINES-1:0] wave_mask_q;
    logic side_ready;
    logic side_valid;
    logic side_idle;
    logic side_busy;
    logic side_error;
    logic start_legal;
    logic result_retire;
    logic [CHANNEL_WIDTH-1:0] remaining_after_wave;

    function automatic logic [ENGINES-1:0] prefix_mask(
        input logic [CHANNEL_WIDTH-1:0] count
    );
        logic [ENGINES-1:0] mask;
        begin
            mask = '0;
            for (integer index = 0; index < ENGINES; index++)
                if (index < count)
                    mask[index] = 1'b1;
            prefix_mask = mask;
        end
    endfunction

    function automatic logic [3:0] limited_engine_count(
        input logic [CHANNEL_WIDTH-1:0] count
    );
        begin
            if (count > ENGINES)
                limited_engine_count = 4'(ENGINES);
            else
                limited_engine_count = count[3:0];
        end
    endfunction

    assign start_legal = i_output_channel_count != 0 &&
        i_output_channel_count <= 10'd480 &&
        i_output_channel_base + i_output_channel_count <= OUTPUT_LIMIT;
    assign o_start_ready = !o_busy && side_idle && i_owner_enable &&
                           !o_sticky_error;
    assign o_chunk_ready = o_busy && side_ready;
    assign o_wave_engine_mask = wave_mask_q;
    assign o_wave_channel_base = wave_base_q;
    assign o_result_valid = side_valid;
    assign result_retire = side_valid && i_result_ready;
    assign remaining_after_wave = remaining_q - wave_engine_count_q;

    unified_scnn_q8_8_k8_side_array #(
        .ENGINES(ENGINES), .LANES(LANES), .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_side_array (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_owner_enable(i_owner_enable),
        .i_clear_error(i_clear_error),
        .i_valid(i_chunk_valid && o_busy), .o_ready(side_ready),
        .i_engine_mask(wave_mask_q), .i_feature_flat(i_feature_flat),
        .i_weight_flat(i_weight_flat), .i_lane_mask(i_lane_mask),
        .i_first(i_chunk_first), .i_last(i_chunk_last),
        .i_output_channel_base(wave_base_q), .o_valid(side_valid),
        .i_ready(i_result_ready), .o_result_mask(o_result_mask),
        .o_result_channel_flat(o_result_channel_flat),
        .o_result_accumulator_flat(o_result_accumulator_flat),
        .o_busy(side_busy), .o_idle(side_idle), .o_sticky_error(side_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            remaining_q <= '0;
            wave_base_q <= '0;
            wave_engine_count_q <= '0;
            wave_mask_q <= '0;
            o_busy <= 1'b0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && !o_busy && side_idle)
                o_sticky_error <= 1'b0;
            if (side_error)
                o_sticky_error <= 1'b1;

            if (i_start && o_start_ready) begin
                if (!start_legal) begin
                    o_sticky_error <= 1'b1;
                end else begin
                    remaining_q <= i_output_channel_count;
                    wave_base_q <= i_output_channel_base;
                    wave_engine_count_q <=
                        limited_engine_count(i_output_channel_count);
                    wave_mask_q <= prefix_mask(i_output_channel_count);
                    o_busy <= 1'b1;
                end
            end else if (i_start && !o_start_ready) begin
                o_sticky_error <= 1'b1;
            end

            if (i_chunk_valid && !o_busy)
                o_sticky_error <= 1'b1;

            if (result_retire) begin
                if (remaining_after_wave == 0) begin
                    remaining_q <= '0;
                    wave_mask_q <= '0;
                    o_busy <= 1'b0;
                    o_done <= 1'b1;
                end else begin
                    remaining_q <= remaining_after_wave;
                    wave_base_q <= wave_base_q + wave_engine_count_q;
                    wave_engine_count_q <=
                        limited_engine_count(remaining_after_wave);
                    wave_mask_q <= prefix_mask(remaining_after_wave);
                end
            end
        end
    end

    logic unused_status;
    assign unused_status = &{1'b0, side_busy};
endmodule

`default_nettype wire
