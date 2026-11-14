`timescale 1ns/1ps
`default_nettype none

// Aligns the shared learning/parameter owner with the profile requested by the
// final inference core. It contains no topology datapath: FC, SRNN and SCNN
// retain separate adapters around one shared parameter and learning backend.
module unified_tp_train_infer_system_control #(
    parameter logic [6:0] PROFILE_FC_H160 = 7'h01,
    parameter logic [6:0] PROFILE_FC_H480 = 7'h02,
    parameter logic [6:0] PROFILE_SRNN_M2 = 7'h10,
    parameter logic [6:0] PROFILE_SCNN_VGG9 = 7'h20
) (
    input  logic        i_clk,
    input  logic        i_rst_n,
    input  logic        i_clear_error,

    input  logic        i_requested_profile_valid,
    input  logic [6:0]  i_requested_profile_code,
    input  logic        i_inference_global_idle,
    input  logic        i_learning_enable,

    input  logic        i_backend_quiescent,
    input  logic        i_backend_error,

    output logic        o_fc_acquire_valid,
    input  logic        i_fc_acquire_ready,
    output logic        o_srnn_acquire_valid,
    input  logic        i_srnn_acquire_ready,
    output logic        o_scnn_acquire_valid,
    input  logic        i_scnn_acquire_ready,

    output logic        o_fc_release_valid,
    input  logic        i_fc_release_ready,
    output logic        o_srnn_release_valid,
    input  logic        i_srnn_release_ready,
    output logic        o_scnn_release_valid,
    input  logic        i_scnn_release_ready,

    input  logic        i_fc_quiescent,
    input  logic        i_srnn_quiescent,
    input  logic        i_scnn_quiescent,
    input  logic        i_fc_error,
    input  logic        i_srnn_error,
    input  logic        i_scnn_error,

    output logic        o_fc_parameter_owner,
    output logic        o_srnn_parameter_owner,
    output logic        o_scnn_parameter_owner,
    output logic        o_fc_learning_enable,
    output logic        o_srnn_learning_enable,
    output logic        o_scnn_learning_enable,
    output logic        o_profile_aligned,
    output logic        o_start_allowed,
    output logic        o_profile_switch_done,
    output logic        o_switch_busy,
    output logic [1:0]  o_active_owner,
    output logic [6:0]  o_active_profile_code,
    output logic [15:0] o_profile_version,
    output logic        o_sticky_error,
    output logic [7:0]  o_error_code
);
    logic owner_switch_valid;
    logic owner_switch_ready;
    logic owner_active;
    logic [7:0] owner_active_profile;
    logic owner_busy;
    logic owner_fc;
    logic owner_srnn;
    logic owner_scnn;
    logic common_switch_quiescent;

    assign common_switch_quiescent = i_inference_global_idle &&
        i_backend_quiescent;

    assign owner_switch_valid = i_requested_profile_valid &&
        common_switch_quiescent && owner_switch_ready && !owner_busy &&
        !o_sticky_error &&
        (!owner_active ||
         owner_active_profile != {1'b0, i_requested_profile_code});

    assign o_fc_parameter_owner = owner_fc;
    assign o_srnn_parameter_owner = owner_srnn;
    assign o_scnn_parameter_owner = owner_scnn;
    assign o_fc_learning_enable = i_learning_enable && owner_fc;
    assign o_srnn_learning_enable = i_learning_enable && owner_srnn;
    assign o_scnn_learning_enable = i_learning_enable && owner_scnn;
    assign o_active_profile_code = owner_active_profile[6:0];
    assign o_profile_aligned = i_requested_profile_valid && owner_active &&
        !owner_busy && !o_sticky_error &&
        owner_active_profile == {1'b0, i_requested_profile_code};
    // A topology may intentionally retain sequence state across timesteps.
    // Full topology quiescence is required for release, not for another start
    // in the already aligned profile; the topology's own ready remains final.
    assign o_start_allowed = o_profile_aligned && common_switch_quiescent;
    assign o_switch_busy = owner_busy ||
        (i_requested_profile_valid && !o_profile_aligned);

    unified_tp_learning_profile_owner #(
        .PROFILE_FC_H160({1'b0, PROFILE_FC_H160}),
        .PROFILE_FC_H480({1'b0, PROFILE_FC_H480}),
        .PROFILE_SRNN_M2({1'b0, PROFILE_SRNN_M2}),
        .PROFILE_SCNN_VGG9({1'b0, PROFILE_SCNN_VGG9})
    ) u_profile_owner (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_clear_error(i_clear_error),
        .i_switch_valid(owner_switch_valid),
        .o_switch_ready(owner_switch_ready),
        .i_switch_profile_id({1'b0, i_requested_profile_code}),
        .o_switch_done(o_profile_switch_done),
        .o_fc_acquire_valid(o_fc_acquire_valid),
        .i_fc_acquire_ready(i_fc_acquire_ready &&
                            common_switch_quiescent),
        .o_srnn_acquire_valid(o_srnn_acquire_valid),
        .i_srnn_acquire_ready(i_srnn_acquire_ready &&
                              common_switch_quiescent),
        .o_scnn_acquire_valid(o_scnn_acquire_valid),
        .i_scnn_acquire_ready(i_scnn_acquire_ready &&
                              common_switch_quiescent),
        .o_fc_release_valid(o_fc_release_valid),
        .i_fc_release_ready(i_fc_release_ready &&
                            common_switch_quiescent),
        .o_srnn_release_valid(o_srnn_release_valid),
        .i_srnn_release_ready(i_srnn_release_ready &&
                              common_switch_quiescent),
        .o_scnn_release_valid(o_scnn_release_valid),
        .i_scnn_release_ready(i_scnn_release_ready &&
                              common_switch_quiescent),
        .i_fc_quiescent(i_fc_quiescent && common_switch_quiescent),
        .i_srnn_quiescent(i_srnn_quiescent && common_switch_quiescent),
        .i_scnn_quiescent(i_scnn_quiescent && common_switch_quiescent),
        .i_fc_error(i_fc_error || i_backend_error),
        .i_srnn_error(i_srnn_error || i_backend_error),
        .i_scnn_error(i_scnn_error || i_backend_error),
        .o_active(owner_active),
        .o_active_profile_id(owner_active_profile),
        .o_active_owner(o_active_owner),
        .o_fc_owner(owner_fc),
        .o_srnn_owner(owner_srnn),
        .o_scnn_owner(owner_scnn),
        .o_profile_version(o_profile_version),
        .o_busy(owner_busy),
        .o_sticky_error(o_sticky_error),
        .o_error_code(o_error_code)
    );

endmodule

`default_nettype wire
