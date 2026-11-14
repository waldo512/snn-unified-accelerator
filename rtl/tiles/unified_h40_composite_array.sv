`default_nettype none

// Logical pairing of the immutable twelve-cluster canonical FC fabric with
// twelve compact SCNN K8 side engines. The canonical FC datapath remains
// outside this module so that no SCNN mux enters its timing paths.
module unified_h40_composite_array #(
    parameter int TILES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,

    input  logic [1:0]                          i_requested_owner,
    output logic                                o_owner_ready,
    output logic [1:0]                          o_active_owner,

    input  logic                                i_fc_request,
    output logic                                o_fc_start,
    input  logic                                i_fc_busy,
    input  logic                                i_recurrent_request,
    output logic                                o_recurrent_start,
    input  logic                                i_recurrent_busy,

    input  logic                                i_scnn_valid,
    output logic                                o_scnn_ready,
    input  logic [TILES-1:0]                    i_scnn_tile_mask,
    input  logic [LANES*16-1:0]                 i_scnn_feature_flat,
    input  logic [TILES*LANES*16-1:0]           i_scnn_weight_flat,
    input  logic [LANES-1:0]                    i_scnn_lane_mask,
    input  logic                                i_scnn_first,
    input  logic                                i_scnn_last,
    input  logic [CHANNEL_WIDTH-1:0]            i_scnn_output_channel_base,
    output logic                                o_scnn_result_valid,
    input  logic                                i_scnn_result_ready,
    output logic [TILES-1:0]                    o_scnn_result_mask,
    output logic [TILES*CHANNEL_WIDTH-1:0]      o_scnn_result_channel_flat,
    output logic [TILES*ACC_WIDTH-1:0]          o_scnn_result_accumulator_flat,

    output logic                                o_busy,
    output logic                                o_idle,
    output logic                                o_sticky_error
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_RECURRENT_FC = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    logic [1:0] owner_q;
    logic scnn_busy;
    logic scnn_idle;
    logic scnn_error;
    logic local_error_q;
    logic request_conflict;
    logic owner_mismatch;
    logic side_owner_enable;
    logic side_valid;

    assign request_conflict =
        (i_fc_request && i_recurrent_request) ||
        (i_fc_request && i_scnn_valid) ||
        (i_recurrent_request && i_scnn_valid);
    assign owner_mismatch =
        (i_fc_request && owner_q != OWNER_FC) ||
        (i_recurrent_request && owner_q != OWNER_RECURRENT_FC) ||
        (i_scnn_valid && owner_q != OWNER_SCNN);

    assign side_owner_enable = owner_q == OWNER_SCNN;
    assign side_valid = i_scnn_valid && !request_conflict &&
                        !owner_mismatch && !local_error_q;

    // These are command admissions only. FC data and state never enter this
    // module, preserving the canonical fast path exactly.
    assign o_fc_start = i_fc_request && !request_conflict &&
                        !owner_mismatch && !local_error_q && !scnn_error;
    assign o_recurrent_start = i_recurrent_request && !request_conflict &&
                               !owner_mismatch && !local_error_q && !scnn_error;

    assign o_busy = i_fc_busy || i_recurrent_busy || scnn_busy;
    assign o_idle = !o_busy;
    assign o_owner_ready = o_idle && !i_fc_request &&
                           !i_recurrent_request && !i_scnn_valid;
    assign o_active_owner = owner_q;
    assign o_sticky_error = local_error_q || scnn_error;

    unified_scnn_q8_8_k8_side_array #(
        .ENGINES(TILES),
        .LANES(LANES),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_scnn_side_array (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_owner_enable(side_owner_enable),
        .i_clear_error(i_clear_error && o_idle),
        .i_valid(side_valid),
        .o_ready(o_scnn_ready),
        .i_engine_mask(i_scnn_tile_mask),
        .i_feature_flat(i_scnn_feature_flat),
        .i_weight_flat(i_scnn_weight_flat),
        .i_lane_mask(i_scnn_lane_mask),
        .i_first(i_scnn_first),
        .i_last(i_scnn_last),
        .i_output_channel_base(i_scnn_output_channel_base),
        .o_valid(o_scnn_result_valid),
        .i_ready(i_scnn_result_ready),
        .o_result_mask(o_scnn_result_mask),
        .o_result_channel_flat(o_scnn_result_channel_flat),
        .o_result_accumulator_flat(o_scnn_result_accumulator_flat),
        .o_busy(scnn_busy),
        .o_idle(scnn_idle),
        .o_sticky_error(scnn_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            owner_q <= OWNER_IDLE;
            local_error_q <= 1'b0;
        end else begin
            if (i_clear_error && o_idle)
                local_error_q <= 1'b0;

            if (o_owner_ready)
                owner_q <= i_requested_owner;
            else if (i_requested_owner != owner_q)
                local_error_q <= 1'b1;

            if (request_conflict || owner_mismatch)
                local_error_q <= 1'b1;
        end
    end

endmodule

`default_nettype wire
