`default_nettype none

// Final-image orchestration skeleton. There is one canonical H480 FC engine
// behind the FC adapter ports; SRNN reuses that path. The SCNN arithmetic port
// terminates in the twelve-engine side array. Engine datapaths remain separate
// adapters so this control shell cannot insert muxes into the FC fast path.
module unified_snn_core #(
    parameter int TILES = 12,
    parameter int LANES = 8,
    parameter int CHANNEL_WIDTH = 10,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_clear_error,
    input  logic                                i_abort,

    input  logic                                i_start,
    output logic                                o_start_ready,
    input  logic [7:0]                          i_opcode,
    input  logic [15:0]                         i_model_version,
    input  logic [15:0]                         i_graph_version,
    input  logic [15:0]                         i_adapt_version,

    output logic                                o_fc_start,
    input  logic                                i_fc_start_ready,
    input  logic                                i_fc_busy,
    input  logic                                i_fc_done,
    input  logic                                i_fc_error,
    output logic                                o_recurrent_start,
    input  logic                                i_recurrent_start_ready,
    input  logic                                i_recurrent_busy,
    input  logic                                i_recurrent_done,
    input  logic                                i_recurrent_error,
    output logic                                o_scnn_start,
    input  logic                                i_scnn_start_ready,
    input  logic                                i_scnn_busy,
    input  logic                                i_scnn_done,
    input  logic                                i_scnn_error,

    input  logic                                i_scnn_mac_valid,
    output logic                                o_scnn_mac_ready,
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

    input  logic                                i_output_busy,
    input  logic                                i_layer_pending,
    input  logic                                i_ddr_calib_complete,
    input  logic                                i_ddr_cmd_pending,
    input  logic                                i_ddr_response_pending,
    input  logic                                i_ddr_cdc_busy,
    input  logic [2:0]                          i_ddr_outstanding,
    input  logic                                i_prefetch_busy,
    input  logic                                i_scratchpad_busy,
    input  logic                                i_writeback_busy,
    input  logic                                i_w2_commit_pending,
    input  logic                                i_w2_snapshot_busy,
    input  logic                                i_version_activation_pending,

    output logic                                o_global_idle,
    output logic                                o_busy,
    output logic                                o_sample_done,
    output logic [1:0]                          o_tile_owner,
    output logic [7:0]                          o_active_opcode,
    output logic [63:0]                         o_locked_versions,
    output logic [12:0]                         o_input_dim,
    output logic [9:0]                          o_logical_hidden_or_channels,
    output logic [5:0]                          o_output_dim,
    output logic [4:0]                          o_active_h40_tiles,
    output logic [4:0]                          o_layer_count,
    output logic [7:0]                          o_fixed_timesteps,
    output logic [2:0]                          o_numeric_profile,
    output logic                                o_fc_w2_update_enable,
    output logic                                o_sticky_error,
    output logic [7:0]                          o_error_code
);
    localparam logic [1:0] OWNER_IDLE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_RECURRENT_FC = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    localparam logic [7:0] PROFILE_FC_H160 = 8'h01;
    localparam logic [7:0] PROFILE_FC_H480 = 8'h02;
    localparam logic [7:0] PROFILE_SRNN_H480 = 8'h10;
    localparam logic [7:0] PROFILE_SCNN_VGG9 = 8'h20;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_START_NOT_READY = 8'h40;
    localparam logic [7:0] ERROR_DONE_OWNER = 8'h41;
    localparam logic [7:0] ERROR_ENGINE = 8'h42;
    localparam logic [7:0] ERROR_SCNN_DDR_NOT_READY = 8'h43;
    localparam logic [7:0] ERROR_ABORT = 8'h44;
    localparam logic [7:0] ERROR_COMPOSITE_ARRAY = 8'h45;

    logic decoder_start_ready;
    logic decoder_start_pulse;
    logic decoder_busy;
    logic [15:0] profile_version;
    logic fc_fast_path;
    logic recurrent_extension;
    logic conv_extension;
    logic decoder_error;
    logic [7:0] decoder_error_code;
    logic decoder_profile_done;

    logic [1:0] requested_tile_owner;
    logic tile_owner_ready;
    logic composite_busy;
    logic composite_idle;
    logic composite_error;
    logic fc_start_internal;
    logic recurrent_start_internal;

    logic local_error_q;
    logic [7:0] local_error_code_q;
    logic completion_pending_q;
    logic retire_pending_q;
    logic sample_done_q;
    logic start_accept;
    logic start_dependencies_ready;
    logic selected_engine_ready;
    logic engine_done_for_owner;
    logic any_engine_done;
    logic done_owner_mismatch;
    logic drains_idle;
    logic [63:0] locked_versions_q;

    function automatic logic [1:0] opcode_owner(input logic [7:0] opcode);
        case (opcode)
            PROFILE_FC_H160, PROFILE_FC_H480: opcode_owner = OWNER_FC;
            PROFILE_SRNN_H480: opcode_owner = OWNER_RECURRENT_FC;
            PROFILE_SCNN_VGG9: opcode_owner = OWNER_SCNN;
            default: opcode_owner = OWNER_IDLE;
        endcase
    endfunction

    always_comb begin
        case (opcode_owner(i_opcode))
            OWNER_FC: selected_engine_ready = i_fc_start_ready;
            OWNER_RECURRENT_FC:
                selected_engine_ready = i_recurrent_start_ready;
            OWNER_SCNN: selected_engine_ready = i_scnn_start_ready;
            default: selected_engine_ready = 1'b1;
        endcase
    end
    assign start_dependencies_ready = selected_engine_ready &&
        ((i_opcode != PROFILE_SCNN_VGG9) || i_ddr_calib_complete);
    assign start_accept = i_start && o_start_ready;

    always_comb begin
        if (decoder_busy) begin
            if (fc_fast_path)
                requested_tile_owner = OWNER_FC;
            else if (recurrent_extension)
                requested_tile_owner = OWNER_RECURRENT_FC;
            else if (conv_extension)
                requested_tile_owner = OWNER_SCNN;
            else
                requested_tile_owner = OWNER_IDLE;
        end else if (start_accept) begin
            requested_tile_owner = opcode_owner(i_opcode);
        end else begin
            requested_tile_owner = OWNER_IDLE;
        end
    end

    assign any_engine_done = i_fc_done || i_recurrent_done || i_scnn_done;
    assign engine_done_for_owner =
        (o_tile_owner == OWNER_FC && i_fc_done) ||
        (o_tile_owner == OWNER_RECURRENT_FC && i_recurrent_done) ||
        (o_tile_owner == OWNER_SCNN && i_scnn_done);
    assign done_owner_mismatch = any_engine_done &&
        (!decoder_busy || !engine_done_for_owner ||
         (i_fc_done && (i_recurrent_done || i_scnn_done)) ||
         (i_recurrent_done && i_scnn_done));

    assign drains_idle = !i_fc_busy && !i_recurrent_busy && !i_scnn_busy &&
        !i_output_busy && !i_layer_pending && composite_idle &&
        !i_ddr_cmd_pending && !i_ddr_response_pending && !i_ddr_cdc_busy &&
        (i_ddr_outstanding == 3'd0) && !i_prefetch_busy &&
        !i_scratchpad_busy && !i_writeback_busy &&
        !i_w2_commit_pending && !i_w2_snapshot_busy &&
        !i_version_activation_pending;
    assign decoder_profile_done = decoder_busy && completion_pending_q &&
                                  drains_idle;

    assign o_global_idle = !decoder_busy && !completion_pending_q &&
        !retire_pending_q && (o_tile_owner == OWNER_IDLE) && drains_idle;
    assign o_start_ready = o_global_idle && decoder_start_ready &&
        !local_error_q && !decoder_error && !composite_error &&
        start_dependencies_ready;
    assign o_busy = !o_global_idle;
    assign o_sample_done = sample_done_q;
    assign o_locked_versions = locked_versions_q;
    assign o_sticky_error = local_error_q || decoder_error || composite_error;
    assign o_error_code = local_error_q ? local_error_code_q :
        decoder_error ? decoder_error_code :
        composite_error ? ERROR_COMPOSITE_ARRAY : ERROR_NONE;
    assign o_scnn_start = decoder_start_pulse && conv_extension;
    assign o_fc_start = fc_start_internal;
    assign o_recurrent_start = recurrent_start_internal;

    unified_runtime_profile_decoder u_profile_decoder (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(start_accept),
        .i_opcode(i_opcode),
        .i_profile_done(decoder_profile_done),
        .i_clear_error(i_clear_error && o_global_idle),
        .o_start_ready(decoder_start_ready),
        .o_start_pulse(decoder_start_pulse),
        .o_busy(decoder_busy),
        .o_active_opcode(o_active_opcode),
        .o_profile_version(profile_version),
        .o_fc_fast_path(fc_fast_path),
        .o_recurrent_extension(recurrent_extension),
        .o_conv_extension(conv_extension),
        .o_input_dim(o_input_dim),
        .o_logical_hidden_or_channels(o_logical_hidden_or_channels),
        .o_output_dim(o_output_dim),
        .o_active_h40_tiles(o_active_h40_tiles),
        .o_layer_count(o_layer_count),
        .o_fixed_timesteps(o_fixed_timesteps),
        .o_numeric_profile(o_numeric_profile),
        .o_fc_w2_update_enable(o_fc_w2_update_enable),
        .o_config_error(decoder_error),
        .o_error_code(decoder_error_code)
    );

    unified_h40_composite_array #(
        .TILES(TILES),
        .LANES(LANES),
        .CHANNEL_WIDTH(CHANNEL_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_composite_array (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error && o_global_idle),
        .i_requested_owner(requested_tile_owner),
        .o_owner_ready(tile_owner_ready),
        .o_active_owner(o_tile_owner),
        .i_fc_request(decoder_start_pulse && fc_fast_path),
        .o_fc_start(fc_start_internal),
        .i_fc_busy(i_fc_busy),
        .i_recurrent_request(decoder_start_pulse && recurrent_extension),
        .o_recurrent_start(recurrent_start_internal),
        .i_recurrent_busy(i_recurrent_busy),
        .i_scnn_valid(i_scnn_mac_valid),
        .o_scnn_ready(o_scnn_mac_ready),
        .i_scnn_tile_mask(i_scnn_tile_mask),
        .i_scnn_feature_flat(i_scnn_feature_flat),
        .i_scnn_weight_flat(i_scnn_weight_flat),
        .i_scnn_lane_mask(i_scnn_lane_mask),
        .i_scnn_first(i_scnn_first),
        .i_scnn_last(i_scnn_last),
        .i_scnn_output_channel_base(i_scnn_output_channel_base),
        .o_scnn_result_valid(o_scnn_result_valid),
        .i_scnn_result_ready(i_scnn_result_ready),
        .o_scnn_result_mask(o_scnn_result_mask),
        .o_scnn_result_channel_flat(o_scnn_result_channel_flat),
        .o_scnn_result_accumulator_flat(o_scnn_result_accumulator_flat),
        .o_busy(composite_busy),
        .o_idle(composite_idle),
        .o_sticky_error(composite_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
            completion_pending_q <= 1'b0;
            retire_pending_q <= 1'b0;
            sample_done_q <= 1'b0;
            locked_versions_q <= '0;
        end else begin
            sample_done_q <= 1'b0;

            if (i_clear_error && o_global_idle) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (i_start && !o_start_ready) begin
                local_error_q <= 1'b1;
                local_error_code_q <=
                    (i_opcode == PROFILE_SCNN_VGG9 && !i_ddr_calib_complete) ?
                    ERROR_SCNN_DDR_NOT_READY : ERROR_START_NOT_READY;
            end
            if (done_owner_mismatch) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_DONE_OWNER;
            end
            if (i_fc_error || i_recurrent_error || i_scnn_error) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_ENGINE;
            end
            if (i_abort) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_ABORT;
                if (decoder_busy)
                    completion_pending_q <= 1'b1;
            end

            if (decoder_start_pulse) begin
                locked_versions_q <= {
                    profile_version,
                    i_model_version,
                    i_graph_version,
                    i_adapt_version
                };
            end
            if (engine_done_for_owner)
                completion_pending_q <= 1'b1;
            if (decoder_profile_done) begin
                completion_pending_q <= 1'b0;
                retire_pending_q <= 1'b1;
            end
            if (retire_pending_q && o_tile_owner == OWNER_IDLE && drains_idle) begin
                retire_pending_q <= 1'b0;
                sample_done_q <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
