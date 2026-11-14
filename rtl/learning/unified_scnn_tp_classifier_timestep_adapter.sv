`default_nettype none

// Maps an independently evaluated classifier frame onto one absolute TP
// timestep. The classifier frontend is reset for each sample and therefore
// emits native frame zero; temporal trace state belongs to the learner and is
// reset only for absolute timestep zero.
module unified_scnn_tp_classifier_timestep_adapter #(
    parameter int LANES = 8,
    parameter int ENGINES = 12,
    parameter int FEATURE_INDEX_WIDTH = 11,
    parameter int CLASS_INDEX_WIDTH = 4,
    parameter int CLASS_COUNT_WIDTH = 4,
    parameter int VERSION_WIDTH = 16,
    parameter int FRAME_WIDTH = 8
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_enable,
    input  logic [FRAME_WIDTH-1:0]                    i_absolute_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_expected_version,

    input  logic                                      i_feature_valid,
    output logic                                      o_feature_ready,
    input  logic [LANES-1:0]                          i_feature_mask,
    input  logic [FEATURE_INDEX_WIDTH-1:0]            i_feature_base,
    input  logic [LANES*16-1:0]                       i_feature_q8_8_flat,
    input  logic [VERSION_WIDTH-1:0]                  i_feature_version,
    input  logic [FRAME_WIDTH-1:0]                    i_feature_native_frame,
    input  logic                                      i_feature_first,
    input  logic                                      i_feature_last,
    input  logic                                      i_feature_native_reset,
    input  logic                                      i_feature_sample_last,
    output logic                                      o_feature_valid,
    input  logic                                      i_feature_ready,
    output logic [LANES-1:0]                          o_feature_mask,
    output logic [FEATURE_INDEX_WIDTH-1:0]            o_feature_base,
    output logic [LANES*16-1:0]                       o_feature_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]                  o_feature_version,
    output logic [FRAME_WIDTH-1:0]                    o_feature_frame,
    output logic                                      o_feature_first,
    output logic                                      o_feature_last,
    output logic                                      o_feature_sample_reset,
    output logic                                      o_feature_sample_last,

    input  logic                                      i_logit_valid,
    output logic                                      o_logit_ready,
    input  logic [ENGINES-1:0]                        i_logit_mask,
    input  logic [CLASS_INDEX_WIDTH-1:0]              i_logit_base,
    input  logic [CLASS_COUNT_WIDTH-1:0]              i_logit_count,
    input  logic [ENGINES*16-1:0]                     i_logit_q8_8_flat,
    input  logic [VERSION_WIDTH-1:0]                  i_logit_version,
    input  logic [FRAME_WIDTH-1:0]                    i_logit_native_frame,
    input  logic                                      i_logit_frame_last,
    input  logic                                      i_logit_sample_last,
    output logic                                      o_logit_valid,
    input  logic                                      i_logit_ready,
    output logic [ENGINES-1:0]                        o_logit_mask,
    output logic [CLASS_INDEX_WIDTH-1:0]              o_logit_base,
    output logic [CLASS_COUNT_WIDTH-1:0]              o_logit_count,
    output logic [ENGINES*16-1:0]                     o_logit_q8_8_flat,
    output logic [VERSION_WIDTH-1:0]                  o_logit_version,
    output logic [FRAME_WIDTH-1:0]                    o_logit_frame,
    output logic                                      o_logit_frame_last,
    output logic                                      o_logit_sample_last,
    output logic                                      o_protocol_error
);
    logic feature_legal;
    logic logit_legal;

    assign feature_legal = i_feature_version == i_expected_version &&
        i_feature_native_frame == 0 &&
        i_feature_native_reset == i_feature_first;
    assign logit_legal = i_logit_version == i_expected_version &&
        i_logit_native_frame == 0;

    assign o_feature_valid = i_enable && !o_protocol_error &&
        i_feature_valid && feature_legal;
    assign o_feature_ready = i_enable && !o_protocol_error &&
        (feature_legal ? i_feature_ready : 1'b1);
    assign o_feature_mask = i_feature_mask;
    assign o_feature_base = i_feature_base;
    assign o_feature_q8_8_flat = i_feature_q8_8_flat;
    assign o_feature_version = i_feature_version;
    assign o_feature_frame = i_absolute_timestep;
    assign o_feature_first = i_feature_first;
    assign o_feature_last = i_feature_last;
    assign o_feature_sample_reset = i_absolute_timestep == 0 &&
        i_feature_native_reset;
    assign o_feature_sample_last = i_feature_sample_last;

    assign o_logit_valid = i_enable && !o_protocol_error &&
        i_logit_valid && logit_legal;
    assign o_logit_ready = i_enable && !o_protocol_error &&
        (logit_legal ? i_logit_ready : 1'b1);
    assign o_logit_mask = i_logit_mask;
    assign o_logit_base = i_logit_base;
    assign o_logit_count = i_logit_count;
    assign o_logit_q8_8_flat = i_logit_q8_8_flat;
    assign o_logit_version = i_logit_version;
    assign o_logit_frame = i_absolute_timestep;
    assign o_logit_frame_last = i_logit_frame_last;
    assign o_logit_sample_last = i_logit_sample_last;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            o_protocol_error <= 1'b0;
        else begin
            if (i_clear_error)
                o_protocol_error <= 1'b0;
            if (i_enable &&
                ((i_feature_valid && !feature_legal) ||
                 (i_logit_valid && !logit_legal)))
                o_protocol_error <= 1'b1;
        end
    end
endmodule

`default_nettype wire
