`default_nettype none

// SCNN-only arithmetic beside the immutable canonical FC datapath. One engine
// serves one output channel at a time; twelve engines cover one channel wave.
module unified_scnn_q8_8_k8_side_array #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_owner_enable,
    input  logic                                i_clear_error,
    input  logic                                i_valid,
    output logic                                o_ready,
    input  logic [ENGINES-1:0]                  i_engine_mask,
    input  logic [LANES*16-1:0]                 i_feature_flat,
    input  logic [ENGINES*LANES*16-1:0]         i_weight_flat,
    input  logic [LANES-1:0]                    i_lane_mask,
    input  logic                                i_first,
    input  logic                                i_last,
    input  logic [CHANNEL_WIDTH-1:0]            i_output_channel_base,
    output logic                                o_valid,
    input  logic                                i_ready,
    output logic [ENGINES-1:0]                  o_result_mask,
    output logic [ENGINES*CHANNEL_WIDTH-1:0]    o_result_channel_flat,
    output logic [ENGINES*ACC_WIDTH-1:0]        o_result_accumulator_flat,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    logic [ENGINES-1:0] engine_ready;
    logic [ENGINES-1:0] engine_valid;
    logic [ENGINES*ACC_WIDTH-1:0] engine_accumulator_flat;
    logic [ENGINES-1:0] active_mask_q;
    logic [CHANNEL_WIDTH-1:0] output_channel_base_q;
    logic sequence_active_q;
    logic all_active_ready;
    logic all_active_valid;
    logic request_legal;
    logic launch;
    logic retire;

    assign all_active_ready = &(engine_ready | ~i_engine_mask);
    assign all_active_valid = &(engine_valid | ~active_mask_q) &&
                              (active_mask_q != '0);
    assign request_legal = (i_engine_mask != '0) &&
        ((i_first && !sequence_active_q) ||
         (!i_first && sequence_active_q &&
          i_engine_mask == active_mask_q &&
          i_output_channel_base == output_channel_base_q));
    assign o_ready = i_owner_enable && !o_sticky_error && all_active_ready;
    assign launch = i_valid && o_ready && request_legal;
    assign o_valid = all_active_valid;
    assign retire = o_valid && i_ready;
    assign o_result_mask = active_mask_q;
    assign o_result_accumulator_flat = engine_accumulator_flat;
    assign o_busy = sequence_active_q || (|engine_valid) ||
                    (|((~engine_ready) & active_mask_q));
    assign o_idle = !o_busy;

    genvar engine;
    generate
        for (engine = 0; engine < ENGINES; engine++) begin : g_engine
            assign o_result_channel_flat[
                engine*CHANNEL_WIDTH +: CHANNEL_WIDTH
            ] = output_channel_base_q + CHANNEL_WIDTH'(engine);

            unified_q8_8_k8_accumulator #(
                .LANES(LANES), .ACC_WIDTH(ACC_WIDTH)
            ) u_accumulator (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_valid(launch && i_engine_mask[engine]),
                .o_ready(engine_ready[engine]),
                .i_a_flat(i_feature_flat),
                .i_b_flat(i_weight_flat[engine*LANES*16 +: LANES*16]),
                .i_lane_mask(i_lane_mask), .i_first(i_first), .i_last(i_last),
                .o_valid(engine_valid[engine]), .i_ready(retire),
                .o_accumulator(engine_accumulator_flat[
                    engine*ACC_WIDTH +: ACC_WIDTH
                ])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_mask_q <= '0;
            output_channel_base_q <= '0;
            sequence_active_q <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            if (i_clear_error && o_idle)
                o_sticky_error <= 1'b0;

            if (i_valid && !i_owner_enable)
                o_sticky_error <= 1'b1;
            else if (i_valid && o_ready && !request_legal)
                o_sticky_error <= 1'b1;

            if (launch) begin
                if (i_first) begin
                    active_mask_q <= i_engine_mask;
                    output_channel_base_q <= i_output_channel_base;
                end
                sequence_active_q <= !i_last;
            end

            if (retire)
                active_mask_q <= '0;
        end
    end
endmodule

`default_nettype wire
