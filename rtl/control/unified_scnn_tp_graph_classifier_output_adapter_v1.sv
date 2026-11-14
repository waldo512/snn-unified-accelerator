`timescale 1ns/1ps
`default_nettype none

// Converts the graph classifier start protocol into the final-core OUTPUT
// layer protocol.  It builds one ABI-v22 descriptor and keeps the external
// session alive until the shared OUTPUT frontend retires.
module unified_scnn_tp_graph_classifier_output_adapter_v1 #(
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int Q64_ADDR_WIDTH = 22,
    parameter int VERSION_WIDTH = 16,
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int FEATURE_COUNT = 2048,
    parameter int CLASS_COUNT = 11,
    parameter int FEATURE_COUNT_WIDTH = FEATURE_COUNT <= 2 ? 1 : $clog2(FEATURE_COUNT + 1),
    parameter int CLASS_COUNT_WIDTH = CLASS_COUNT <= 2 ? 1 : $clog2(CLASS_COUNT + 1),
    parameter logic [3:0] OUTPUT_LAYER_INDEX = 4'd4
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_classifier_start_valid,
    output logic                                  o_classifier_start_ready,
    input  logic [SAMPLE_WIDTH-1:0]               i_classifier_sample,
    input  logic [TIMESTEP_WIDTH-1:0]             i_classifier_timestep,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_parameter_base,
    input  logic [FEATURE_COUNT_WIDTH-1:0]        i_feature_count,
    input  logic [CLASS_COUNT_WIDTH-1:0]          i_class_count,
    input  logic [VERSION_WIDTH-1:0]              i_state_version,
    input  logic                                  i_input_slot,
    input  logic                                  i_sample_reset,
    input  logic                                  i_sample_last,
    input  logic                                  i_observation_enable,
    input  logic                                  i_input_prescan_enable,

    output logic                                  o_core_session_active,
    output logic                                  o_core_layer_valid,
    input  logic                                  i_core_layer_ready,
    output logic [3:0]                            o_core_layer_index,
    output logic [383:0]                          o_core_layer_words,
    output logic                                  o_core_input_slot,
    output logic                                  o_core_output_slot,
    output logic [VERSION_WIDTH-1:0]              o_core_state_version,
    output logic                                  o_core_sample_reset,
    output logic                                  o_core_sample_last,
    output logic                                  o_core_observation_enable,
    output logic                                  o_core_input_prescan_enable,
    input  logic                                  i_core_layer_done,
    input  logic                                  i_core_layer_busy,
    input  logic                                  i_core_layer_error,

    input  logic                                  i_core_weight_read_valid,
    output logic                                  o_core_weight_read_ready,
    input  logic [Q64_ADDR_WIDTH-1:0]              i_core_weight_read_addr0,
    input  logic [Q64_ADDR_WIDTH-1:0]              i_core_weight_read_addr1,
    output logic                                  o_weight_read_valid,
    input  logic                                  i_weight_read_ready,
    output logic [Q64_ADDR_WIDTH-1:0]              o_weight_read_addr0,
    output logic [Q64_ADDR_WIDTH-1:0]              o_weight_read_addr1,
    input  logic                                  i_weight_result_valid,
    input  logic                                  i_core_weight_result_ready,
    output logic                                  o_weight_result_ready,
    input  logic [63:0]                           i_weight_data0,
    input  logic [63:0]                           i_weight_data1,
    input  logic                                  i_weight_response_error,
    output logic                                  o_core_weight_result_valid,
    output logic [63:0]                           o_core_weight_data0,
    output logic [63:0]                           o_core_weight_data1,
    output logic                                  o_core_weight_response_error,

    output logic [SAMPLE_WIDTH-1:0]               o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]             o_active_timestep,
    output logic                                  o_classifier_done,
    output logic                                  o_classifier_error,
    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    localparam logic [31:0] LAYER_MAGIC = 32'h4e4d5232;
    localparam logic [31:0] ABI_V22 = 32'h000c0202;
    localparam logic [31:0] GRAPH_MAGIC = 32'h55534e31;
    localparam logic [3:0] LAYER_OUTPUT = 4'd4;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_LAYER,
        ST_RUN,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [FEATURE_COUNT_WIDTH-1:0] feature_count_q;
    logic [CLASS_COUNT_WIDTH-1:0] class_count_q;
    logic [VERSION_WIDTH-1:0] state_version_q;
    logic input_slot_q;
    logic sample_reset_q;
    logic sample_last_q;
    logic observation_enable_q;
    logic input_prescan_enable_q;
    logic [383:0] descriptor;
    logic protocol_error_q;
    logic start_fire;
    logic layer_fire;

    always_comb begin
        descriptor = '0;
        descriptor[0*32 +: 32] = LAYER_MAGIC;
        descriptor[1*32 +: 32] = ABI_V22;
        descriptor[2*32 +: 4] = LAYER_OUTPUT;
        descriptor[3*32 +: 13] = feature_count_q;
        descriptor[3*32 + 23 +: 6] = class_count_q;
        descriptor[4*32 +: 32] = 32'd1 | (32'd40 << 5) |
            (32'd8 << 14) | (32'd20 << 18);
        descriptor[7*32 +: PARAM_ADDR_WIDTH] = parameter_base_q;
        descriptor[11*32 +: 32] = GRAPH_MAGIC;
    end

    assign o_classifier_start_ready = state_q == ST_IDLE &&
        !protocol_error_q && !i_abort;
    assign start_fire = i_classifier_start_valid &&
        o_classifier_start_ready;

    assign o_core_session_active = state_q != ST_IDLE;
    assign o_core_layer_valid = state_q == ST_LAYER &&
        !protocol_error_q && !i_abort;
    assign layer_fire = o_core_layer_valid && i_core_layer_ready;
    assign o_core_layer_index = OUTPUT_LAYER_INDEX;
    assign o_core_layer_words = descriptor;
    assign o_core_input_slot = input_slot_q;
    assign o_core_output_slot = 1'b0;
    assign o_core_state_version = state_version_q;
    assign o_core_sample_reset = sample_reset_q;
    assign o_core_sample_last = sample_last_q;
    assign o_core_observation_enable = observation_enable_q;
    assign o_core_input_prescan_enable = input_prescan_enable_q;

    assign o_active_sample = sample_q;
    assign o_active_timestep = timestep_q;
    assign o_classifier_done = state_q == ST_RUN && i_core_layer_done;
    assign o_classifier_error = protocol_error_q || i_core_layer_error ||
        i_weight_response_error;
    assign o_busy = state_q != ST_IDLE || i_core_layer_busy;
    assign o_protocol_error = o_classifier_error;

    // The final core owns request timing; this module only preserves the
    // classifier session while the external Q64 service owns the response.
    assign o_weight_read_valid = i_core_weight_read_valid &&
        o_core_session_active && !protocol_error_q;
    assign o_weight_read_addr0 = i_core_weight_read_addr0;
    assign o_weight_read_addr1 = i_core_weight_read_addr1;
    assign o_core_weight_read_ready = o_core_session_active &&
        !protocol_error_q && i_weight_read_ready;
    assign o_weight_result_ready = i_core_weight_result_ready;
    assign o_core_weight_result_valid = i_weight_result_valid &&
        o_core_session_active && !protocol_error_q;
    assign o_core_weight_data0 = i_weight_data0;
    assign o_core_weight_data1 = i_weight_data1;
    assign o_core_weight_response_error = i_weight_response_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            sample_q <= '0;
            timestep_q <= '0;
            parameter_base_q <= '0;
            feature_count_q <= '0;
            class_count_q <= '0;
            state_version_q <= '0;
            input_slot_q <= 1'b0;
            sample_reset_q <= 1'b0;
            sample_last_q <= 1'b0;
            observation_enable_q <= 1'b0;
            input_prescan_enable_q <= 1'b0;
            protocol_error_q <= 1'b0;
        end else if (i_abort) begin
            state_q <= ST_IDLE;
        end else if (i_clear_error && state_q == ST_ERROR) begin
            state_q <= ST_IDLE;
            protocol_error_q <= 1'b0;
        end else if (i_core_layer_error || i_weight_response_error) begin
            protocol_error_q <= 1'b1;
            state_q <= ST_ERROR;
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    if (i_classifier_start_valid && !o_classifier_start_ready) begin
                        protocol_error_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end else if (start_fire) begin
                        sample_q <= i_classifier_sample;
                        timestep_q <= i_classifier_timestep;
                        parameter_base_q <= i_parameter_base;
                        feature_count_q <= i_feature_count;
                        class_count_q <= i_class_count;
                        state_version_q <= i_state_version;
                        input_slot_q <= i_input_slot;
                        sample_reset_q <= i_sample_reset;
                        sample_last_q <= i_sample_last;
                        observation_enable_q <= i_observation_enable;
                        input_prescan_enable_q <= i_input_prescan_enable;
                        state_q <= ST_LAYER;
                    end else if (i_core_layer_done) begin
                        protocol_error_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end
                end
                ST_LAYER: begin
                    if (i_core_layer_done) begin
                        protocol_error_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end else if (layer_fire) begin
                        state_q <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (i_core_layer_done)
                        state_q <= ST_IDLE;
                end
                default: begin
                    state_q <= ST_ERROR;
                end
            endcase
        end
    end

    initial begin
        if (FEATURE_COUNT < 1 || FEATURE_COUNT > 8191 ||
            CLASS_COUNT < 1 || CLASS_COUNT > 63)
            $error("SCNN classifier OUTPUT descriptor dimensions are invalid");
    end
endmodule

`default_nettype wire
