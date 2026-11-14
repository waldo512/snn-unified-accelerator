`default_nettype none

// Shared extension arithmetic. Canonical single-layer FC bypasses this array.
// SRNN and multi-layer FC use INT8 mode; SCNN uses exact Q8.8 multipass mode.
module unified_shared_k8_numeric_v1_array_soft_abort_v1 #(
    parameter int ENGINES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_abort,
    input  logic                                i_clear_error,

    input  logic                                i_owner_valid,
    output logic                                o_owner_ready,
    input  logic [1:0]                          i_owner,
    input  logic                                i_mode_q8_8,
    input  logic                                i_owner_release_valid,
    output logic                                o_owner_release_ready,
    output logic                                o_owner_active,
    output logic [1:0]                          o_active_owner,
    output logic                                o_active_mode_q8_8,

    input  logic                                i_valid,
    output logic                                o_ready,
    input  logic [ENGINES-1:0]                  i_engine_mask,
    input  logic [LANES-1:0]                    i_lane_mask,
    input  logic [LANES*16-1:0]                 i_q8_8_feature_flat,
    input  logic [ENGINES*LANES*16-1:0]         i_q8_8_weight_flat,
    input  logic [LANES*8-1:0]                  i_int8_lhs_flat,
    input  logic [ENGINES*LANES*8-1:0]          i_int8_rhs_flat,
    input  logic                                i_first,
    input  logic                                i_last,
    input  logic [CHANNEL_WIDTH-1:0]            i_output_channel_base,

    output logic                                o_valid,
    input  logic                                i_ready,
    output logic [1:0]                          o_result_owner,
    output logic                                o_result_mode_q8_8,
    output logic [ENGINES-1:0]                  o_result_engine_mask,
    output logic [LANES-1:0]                    o_result_lane_mask,
    output logic [ENGINES*CHANNEL_WIDTH-1:0]    o_result_channel_flat,
    output logic [ENGINES*ACC_WIDTH-1:0]        o_q8_8_accumulator_flat,
    output logic [ENGINES*LANES*32-1:0]         o_int8_contribution_flat,
    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_SRNN = 2'd1;
    localparam logic [1:0] OWNER_MULTILAYER_FC = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    logic owner_active_q;
    logic [1:0] owner_q;
    logic mode_q8_8_q;
    logic sequence_active_q;
    logic [ENGINES-1:0] sequence_engine_mask_q;
    logic [CHANNEL_WIDTH-1:0] sequence_channel_base_q;
    logic [ENGINES-1:0] result_engine_mask_q;
    logic [LANES-1:0] result_lane_mask_q;
    logic [CHANNEL_WIDTH-1:0] result_channel_base_q;
    logic [1:0] result_owner_q;
    logic result_mode_q8_8_q;

    logic [ENGINES-1:0] engine_ready;
    logic [ENGINES-1:0] engine_valid;
    logic [ENGINES-1:0] engine_result_mode;
    logic [ENGINES-1:0] engine_busy;
    logic [ENGINES*ACC_WIDTH-1:0] engine_q8_8_accumulator_flat;
    logic [ENGINES*LANES*32-1:0] engine_int8_contribution_flat;
    logic owner_request_legal;
    logic request_legal;
    logic all_requested_ready;
    logic all_result_valid;
    logic any_engine_busy;
    logic datapath_busy;
    logic launch;
    logic retire;
    logic produces_result;

    function automatic logic owner_mode_legal(
        input logic [1:0] owner,
        input logic mode_q8_8
    );
        begin
            unique case (owner)
                OWNER_SRNN, OWNER_MULTILAYER_FC:
                    owner_mode_legal = !mode_q8_8;
                OWNER_SCNN:
                    owner_mode_legal = mode_q8_8;
                default:
                    owner_mode_legal = 1'b0;
            endcase
        end
    endfunction

    assign owner_request_legal = owner_mode_legal(i_owner, i_mode_q8_8);
    assign all_requested_ready = &(engine_ready | ~i_engine_mask);
    assign all_result_valid = &(engine_valid | ~result_engine_mask_q) &&
        &(~result_engine_mask_q |
          ~(engine_result_mode ^ {ENGINES{result_mode_q8_8_q}})) &&
        (result_engine_mask_q != '0);
    assign any_engine_busy = |engine_busy;
    assign retire = o_valid && i_ready;
    assign datapath_busy = sequence_active_q || any_engine_busy ||
        (result_engine_mask_q != '0);
    assign o_owner_ready = !owner_active_q && !datapath_busy &&
        !o_sticky_error;
    assign o_owner_release_ready = owner_active_q && !datapath_busy;
    assign o_owner_active = owner_active_q;
    assign o_active_owner = owner_q;
    assign o_active_mode_q8_8 = mode_q8_8_q;
    assign o_busy = owner_active_q || datapath_busy;
    assign o_idle = !o_busy;

    always_comb begin
        request_legal = owner_active_q &&
            owner_mode_legal(owner_q, mode_q8_8_q) &&
            (i_engine_mask != '0) && (i_lane_mask != '0);
        if (mode_q8_8_q) begin
            request_legal = request_legal &&
                ((i_first && !sequence_active_q) ||
                 (!i_first && sequence_active_q &&
                  i_engine_mask == sequence_engine_mask_q &&
                  i_output_channel_base == sequence_channel_base_q));
        end
    end

    assign o_ready = owner_active_q && !o_sticky_error &&
        all_requested_ready && (!any_engine_busy || retire);
    assign launch = i_valid && o_ready && request_legal;
    assign produces_result = !mode_q8_8_q || i_last;
    assign o_valid = all_result_valid;
    assign o_result_owner = result_owner_q;
    assign o_result_mode_q8_8 = result_mode_q8_8_q;
    assign o_result_engine_mask = result_engine_mask_q;
    assign o_result_lane_mask = result_lane_mask_q;

    always_comb begin
        o_q8_8_accumulator_flat = '0;
        o_int8_contribution_flat = '0;
        for (int result_engine = 0;
             result_engine < ENGINES; result_engine++) begin
            if (result_engine_mask_q[result_engine]) begin
                o_q8_8_accumulator_flat[
                    result_engine*ACC_WIDTH +: ACC_WIDTH
                ] = engine_q8_8_accumulator_flat[
                    result_engine*ACC_WIDTH +: ACC_WIDTH];
                o_int8_contribution_flat[
                    result_engine*LANES*32 +: LANES*32
                ] = engine_int8_contribution_flat[
                    result_engine*LANES*32 +: LANES*32];
            end
        end
    end

    genvar engine;
    generate
        for (engine = 0; engine < ENGINES; engine++) begin : g_engine
            assign o_result_channel_flat[
                engine*CHANNEL_WIDTH +: CHANNEL_WIDTH
            ] = result_channel_base_q + CHANNEL_WIDTH'(engine);

            unified_shared_k8_numeric_v1_engine_soft_abort_v1 #(
                .LANES(LANES),
                .ACC_WIDTH(ACC_WIDTH)
            ) u_engine (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_abort(i_abort),
                .i_valid(launch && i_engine_mask[engine]),
                .o_ready(engine_ready[engine]),
                .i_mode_q8_8(mode_q8_8_q),
                .i_lane_mask(i_lane_mask),
                .i_q8_8_feature_flat(i_q8_8_feature_flat),
                .i_q8_8_weight_flat(i_q8_8_weight_flat[
                    engine*LANES*16 +: LANES*16]),
                .i_int8_lhs_flat(i_int8_lhs_flat),
                .i_int8_rhs_flat(i_int8_rhs_flat[
                    engine*LANES*8 +: LANES*8]),
                .i_first(i_first),
                .i_last(i_last),
                .o_valid(engine_valid[engine]),
                .i_ready(retire),
                .o_result_mode_q8_8(engine_result_mode[engine]),
                .o_q8_8_accumulator(engine_q8_8_accumulator_flat[
                    engine*ACC_WIDTH +: ACC_WIDTH]),
                .o_int8_contribution_flat(engine_int8_contribution_flat[
                    engine*LANES*32 +: LANES*32]),
                .o_busy(engine_busy[engine])
            );
        end
    endgenerate

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_active_q <= 1'b0;
            owner_q <= OWNER_IDLE;
            mode_q8_8_q <= 1'b0;
            sequence_active_q <= 1'b0;
            sequence_engine_mask_q <= '0;
            sequence_channel_base_q <= '0;
            result_engine_mask_q <= '0;
            result_lane_mask_q <= '0;
            result_channel_base_q <= '0;
            result_owner_q <= OWNER_IDLE;
            result_mode_q8_8_q <= 1'b0;
            o_sticky_error <= 1'b0;
        end else if (i_abort) begin
            owner_active_q <= 1'b0;
            owner_q <= OWNER_IDLE;
            mode_q8_8_q <= 1'b0;
            sequence_active_q <= 1'b0;
            sequence_engine_mask_q <= '0;
            sequence_channel_base_q <= '0;
            result_engine_mask_q <= '0;
            result_lane_mask_q <= '0;
            result_channel_base_q <= '0;
            result_owner_q <= OWNER_IDLE;
            result_mode_q8_8_q <= 1'b0;
        end else begin
            if (i_clear_error && !owner_active_q && !datapath_busy)
                o_sticky_error <= 1'b0;

            if (i_owner_valid && o_owner_ready) begin
                if (owner_request_legal) begin
                    owner_active_q <= 1'b1;
                    owner_q <= i_owner;
                    mode_q8_8_q <= i_mode_q8_8;
                end else begin
                    o_sticky_error <= 1'b1;
                end
            end else if (i_owner_valid && owner_active_q &&
                         (i_owner != owner_q ||
                          i_mode_q8_8 != mode_q8_8_q)) begin
                o_sticky_error <= 1'b1;
            end

            if (i_owner_release_valid) begin
                if (o_owner_release_ready) begin
                    owner_active_q <= 1'b0;
                    owner_q <= OWNER_IDLE;
                    mode_q8_8_q <= 1'b0;
                end else begin
                    o_sticky_error <= 1'b1;
                end
            end

            if (i_valid && !owner_active_q)
                o_sticky_error <= 1'b1;
            else if (i_valid && o_ready && !request_legal)
                o_sticky_error <= 1'b1;

            if (retire)
                result_engine_mask_q <= '0;

            if (launch) begin
                if (mode_q8_8_q) begin
                    if (i_first) begin
                        sequence_engine_mask_q <= i_engine_mask;
                        sequence_channel_base_q <= i_output_channel_base;
                    end
                    sequence_active_q <= !i_last;
                end

                if (produces_result) begin
                    result_engine_mask_q <= i_engine_mask;
                    result_lane_mask_q <= i_lane_mask;
                    result_channel_base_q <= i_output_channel_base;
                    result_owner_q <= owner_q;
                    result_mode_q8_8_q <= mode_q8_8_q;
                end
            end
        end
    end
endmodule

`default_nettype wire
