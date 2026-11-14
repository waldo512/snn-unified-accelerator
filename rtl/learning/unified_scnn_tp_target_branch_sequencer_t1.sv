`default_nettype none

// Control boundary for a serialized SCNN target pass. The label projector
// replaces Conv1, then Conv2..Conv8 are issued to the existing shared
// convolution/LIF/pool datapath. This module never instantiates a CNN engine.
//
// Later timesteps are opt-in. They are legal only when the integration gives
// target and student branches disjoint state addresses and preserves those
// addresses across transactions.
module unified_scnn_tp_target_branch_sequencer_t1 #(
    parameter int LAYER_WORDS = 12,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int VERSION_WIDTH = 16,
    parameter int FIRST_SHARED_LAYER = 1,
    parameter int LAST_SHARED_LAYER = 7,
    parameter int FIRST_OBSERVED_LAYER = 6,
    parameter bit ALLOW_RECURRENT_TIMESTEPS = 1'b0
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_start_valid,
    output logic                                  o_start_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]            i_projector_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]           i_projector_parameter_base,
    input  logic [SAMPLE_WIDTH-1:0]               i_sample,
    input  logic [TIMESTEP_WIDTH-1:0]             i_timestep,
    input  logic [VERSION_WIDTH-1:0]              i_target_state_version,

    output logic                                  o_projector_cfg_valid,
    input  logic                                  i_projector_cfg_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_projector_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]            o_projector_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]           o_projector_parameter_base,
    output logic [SAMPLE_WIDTH-1:0]               o_projector_sample,
    output logic [TIMESTEP_WIDTH-1:0]             o_projector_timestep,
    output logic [VERSION_WIDTH-1:0]              o_projector_state_version,
    input  logic                                  i_projector_done,
    input  logic                                  i_projector_error,

    output logic                                  o_descriptor_request_valid,
    input  logic                                  i_descriptor_request_ready,
    output logic [PROFILE_ID_WIDTH-1:0]           o_descriptor_profile_id,
    output logic [3:0]                            o_descriptor_layer_index,
    input  logic                                  i_descriptor_valid,
    output logic                                  o_descriptor_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_descriptor_profile_id,
    input  logic [3:0]                            i_descriptor_layer_index,
    input  logic [LAYER_WORDS*32-1:0]             i_descriptor_words,
    input  logic                                  i_descriptor_error,

    output logic                                  o_layer_valid,
    input  logic                                  i_layer_ready,
    output logic [3:0]                            o_layer_index,
    output logic [LAYER_WORDS*32-1:0]             o_layer_words,
    output logic                                  o_layer_input_slot,
    output logic                                  o_layer_output_slot,
    output logic [VERSION_WIDTH-1:0]              o_layer_state_version,
    input  logic                                  i_layer_done,
    input  logic                                  i_layer_error,

    output logic                                  o_target_observation_enable,
    output logic                                  o_target_input_prescan_enable,
    output logic                                  o_student_state_invalidated,
    output logic                                  o_done,
    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_PROJECTOR_REQUEST,
        ST_PROJECTOR_WAIT,
        ST_DESCRIPTOR_REQUEST,
        ST_DESCRIPTOR_WAIT,
        ST_LAYER_REQUEST,
        ST_LAYER_WAIT,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] projector_tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] projector_parameter_base_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] target_state_version_q;
    logic [3:0] layer_index_q;
    logic [LAYER_WORDS*32-1:0] layer_words_q;
    logic protocol_error_q;
    logic layer_phase_active;

    assign o_start_ready = state_q == ST_IDLE && !protocol_error_q;
    assign o_projector_cfg_valid = state_q == ST_PROJECTOR_REQUEST;
    assign o_projector_profile_id = profile_id_q;
    assign o_projector_tensor_id = projector_tensor_id_q;
    assign o_projector_parameter_base = projector_parameter_base_q;
    assign o_projector_sample = sample_q;
    assign o_projector_timestep = timestep_q;
    assign o_projector_state_version = target_state_version_q;

    assign o_descriptor_request_valid = state_q == ST_DESCRIPTOR_REQUEST;
    assign o_descriptor_profile_id = profile_id_q;
    assign o_descriptor_layer_index = layer_index_q;
    assign o_descriptor_ready = state_q == ST_DESCRIPTOR_WAIT;

    assign o_layer_valid = state_q == ST_LAYER_REQUEST;
    assign o_layer_index = layer_index_q;
    assign o_layer_words = layer_words_q;
    assign o_layer_input_slot = layer_index_q[0];
    assign o_layer_output_slot = !layer_index_q[0];
    assign o_layer_state_version = target_state_version_q;
    assign layer_phase_active = state_q == ST_LAYER_REQUEST ||
        state_q == ST_LAYER_WAIT;
    assign o_target_observation_enable = layer_phase_active &&
        layer_index_q >= 4'(FIRST_OBSERVED_LAYER);
    assign o_target_input_prescan_enable = layer_phase_active &&
        layer_index_q == 4'(LAST_SHARED_LAYER);
    assign o_busy = state_q != ST_IDLE;
    assign o_protocol_error = protocol_error_q || i_projector_error ||
        i_layer_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            profile_id_q <= '0;
            projector_tensor_id_q <= '0;
            projector_parameter_base_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            target_state_version_q <= '0;
            layer_index_q <= 4'(FIRST_SHARED_LAYER);
            layer_words_q <= '0;
            protocol_error_q <= 1'b0;
            o_student_state_invalidated <= 1'b0;
            o_done <= 1'b0;
        end else if (i_abort) begin
            state_q <= ST_IDLE;
            profile_id_q <= '0;
            projector_tensor_id_q <= '0;
            projector_parameter_base_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            target_state_version_q <= '0;
            layer_index_q <= 4'(FIRST_SHARED_LAYER);
            layer_words_q <= '0;
            o_student_state_invalidated <= 1'b0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR) begin
                protocol_error_q <= 1'b0;
                state_q <= ST_IDLE;
            end else if ((i_projector_error || i_layer_error) &&
                         state_q != ST_IDLE) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        o_student_state_invalidated <= 1'b0;
                        if (i_start_valid && o_start_ready) begin
                            if ((!ALLOW_RECURRENT_TIMESTEPS &&
                                 i_timestep != 0) ||
                                i_target_state_version == 0) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                profile_id_q <= i_profile_id;
                                projector_tensor_id_q <= i_projector_tensor_id;
                                projector_parameter_base_q <=
                                    i_projector_parameter_base;
                                sample_q <= i_sample;
                                timestep_q <= i_timestep;
                                target_state_version_q <=
                                    i_target_state_version;
                                layer_index_q <= 4'(FIRST_SHARED_LAYER);
                                state_q <= ST_PROJECTOR_REQUEST;
                            end
                        end
                    end

                    ST_PROJECTOR_REQUEST: begin
                        if (o_projector_cfg_valid && i_projector_cfg_ready)
                            state_q <= ST_PROJECTOR_WAIT;
                    end

                    ST_PROJECTOR_WAIT: begin
                        if (i_projector_done) begin
                            o_student_state_invalidated <= 1'b1;
                            layer_index_q <= 4'(FIRST_SHARED_LAYER);
                            state_q <= ST_DESCRIPTOR_REQUEST;
                        end
                    end

                    ST_DESCRIPTOR_REQUEST: begin
                        if (o_descriptor_request_valid &&
                            i_descriptor_request_ready)
                            state_q <= ST_DESCRIPTOR_WAIT;
                    end

                    ST_DESCRIPTOR_WAIT: begin
                        if (i_descriptor_valid && o_descriptor_ready) begin
                            if (i_descriptor_error ||
                                i_descriptor_profile_id != profile_id_q ||
                                i_descriptor_layer_index != layer_index_q) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                layer_words_q <= i_descriptor_words;
                                state_q <= ST_LAYER_REQUEST;
                            end
                        end
                    end

                    ST_LAYER_REQUEST: begin
                        if (o_layer_valid && i_layer_ready)
                            state_q <= ST_LAYER_WAIT;
                    end

                    ST_LAYER_WAIT: begin
                        if (i_layer_done) begin
                            if (layer_index_q == 4'(LAST_SHARED_LAYER)) begin
                                o_done <= 1'b1;
                                state_q <= ST_IDLE;
                            end else begin
                                layer_index_q <= layer_index_q + 1'b1;
                                state_q <= ST_DESCRIPTOR_REQUEST;
                            end
                        end
                    end

                    default: begin
                        protocol_error_q <= 1'b1;
                        state_q <= ST_ERROR;
                    end
                endcase
            end
        end
    end

    initial begin
        if (FIRST_SHARED_LAYER < 1 || LAST_SHARED_LAYER > 7 ||
            FIRST_SHARED_LAYER > LAST_SHARED_LAYER ||
            FIRST_OBSERVED_LAYER < FIRST_SHARED_LAYER ||
            FIRST_OBSERVED_LAYER > LAST_SHARED_LAYER)
            $error("SCNN target T1 shared-layer range is invalid");
    end
endmodule

`default_nettype wire
