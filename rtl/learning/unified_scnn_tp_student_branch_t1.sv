`default_nettype none

// Serialized student graph controller. Input loading and the shared layer
// executor remain external so the same controller can use either a streamed
// sensor frontend or a test/data-set adapter without copying the CONV core.
module unified_scnn_tp_student_branch_t1 #(
    parameter int LAYER_WORDS = 12,
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int SAMPLE_WIDTH = 4,
    parameter int TIMESTEP_WIDTH = 5,
    parameter int VERSION_WIDTH = 16,
    parameter int FIRST_LAYER = 0,
    parameter int LAST_LAYER = 7,
    parameter int SELECTED_LAYER = 7
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_abort,
    input  logic                                  i_clear_error,

    input  logic                                  i_start_valid,
    output logic                                  o_start_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]           i_profile_id,
    input  logic [SAMPLE_WIDTH-1:0]               i_sample,
    input  logic [TIMESTEP_WIDTH-1:0]             i_timestep,
    input  logic [VERSION_WIDTH-1:0]              i_state_version,

    output logic                                  o_input_start_valid,
    input  logic                                  i_input_start_ready,
    output logic [SAMPLE_WIDTH-1:0]               o_input_sample,
    output logic [TIMESTEP_WIDTH-1:0]             o_input_timestep,
    output logic [VERSION_WIDTH-1:0]              o_input_state_version,
    input  logic                                  i_input_done,
    input  logic                                  i_input_error,

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

    output logic                                  o_selected_observation_enable,
    output logic                                  o_selected_input_prescan_enable,
    output logic [SAMPLE_WIDTH-1:0]               o_active_sample,
    output logic [TIMESTEP_WIDTH-1:0]             o_active_timestep,
    output logic                                  o_done,
    output logic                                  o_busy,
    output logic                                  o_protocol_error
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_INPUT_REQUEST,
        ST_INPUT_WAIT,
        ST_DESCRIPTOR_REQUEST,
        ST_DESCRIPTOR_WAIT,
        ST_LAYER_REQUEST,
        ST_LAYER_WAIT,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] state_version_q;
    logic [3:0] layer_index_q;
    logic [LAYER_WORDS*32-1:0] layer_words_q;
    logic protocol_error_q;
    logic layer_phase_active;

    assign o_start_ready = state_q == ST_IDLE && !protocol_error_q;
    assign o_input_start_valid = state_q == ST_INPUT_REQUEST;
    assign o_input_sample = sample_q;
    assign o_input_timestep = timestep_q;
    assign o_input_state_version = state_version_q;

    assign o_descriptor_request_valid = state_q == ST_DESCRIPTOR_REQUEST;
    assign o_descriptor_profile_id = profile_id_q;
    assign o_descriptor_layer_index = layer_index_q;
    assign o_descriptor_ready = state_q == ST_DESCRIPTOR_WAIT;

    assign o_layer_valid = state_q == ST_LAYER_REQUEST;
    assign o_layer_index = layer_index_q;
    assign o_layer_words = layer_words_q;
    assign o_layer_input_slot = layer_index_q[0];
    assign o_layer_output_slot = !layer_index_q[0];
    assign o_layer_state_version = state_version_q;
    assign layer_phase_active = state_q == ST_LAYER_REQUEST ||
        state_q == ST_LAYER_WAIT;
    assign o_selected_observation_enable = layer_phase_active &&
        layer_index_q == 4'(SELECTED_LAYER);
    assign o_selected_input_prescan_enable =
        o_selected_observation_enable;
    assign o_active_sample = sample_q;
    assign o_active_timestep = timestep_q;
    assign o_busy = state_q != ST_IDLE;
    assign o_protocol_error = protocol_error_q || i_input_error ||
        i_descriptor_error || i_layer_error;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            profile_id_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            state_version_q <= '0;
            layer_index_q <= 4'(FIRST_LAYER);
            layer_words_q <= '0;
            protocol_error_q <= 1'b0;
            o_done <= 1'b0;
        end else if (i_abort) begin
            state_q <= ST_IDLE;
            profile_id_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            state_version_q <= '0;
            layer_index_q <= 4'(FIRST_LAYER);
            layer_words_q <= '0;
            o_done <= 1'b0;
        end else begin
            o_done <= 1'b0;

            if (i_clear_error && state_q == ST_ERROR) begin
                protocol_error_q <= 1'b0;
                state_q <= ST_IDLE;
            end else if ((i_input_error || i_descriptor_error ||
                         i_layer_error) && state_q != ST_IDLE) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else begin
                unique case (state_q)
                    ST_IDLE: begin
                        if (i_start_valid && o_start_ready) begin
                            if (i_state_version == 0) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end else begin
                                profile_id_q <= i_profile_id;
                                sample_q <= i_sample;
                                timestep_q <= i_timestep;
                                state_version_q <= i_state_version;
                                layer_index_q <= 4'(FIRST_LAYER);
                                state_q <= ST_INPUT_REQUEST;
                            end
                        end
                    end

                    ST_INPUT_REQUEST: begin
                        if (o_input_start_valid && i_input_start_ready)
                            state_q <= ST_INPUT_WAIT;
                    end

                    ST_INPUT_WAIT: begin
                        if (i_input_done) begin
                            layer_index_q <= 4'(FIRST_LAYER);
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
                            if (layer_index_q == 4'(LAST_LAYER)) begin
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
        if (FIRST_LAYER < 0 || LAST_LAYER > 15 ||
            FIRST_LAYER > LAST_LAYER || SELECTED_LAYER < FIRST_LAYER ||
            SELECTED_LAYER > LAST_LAYER)
            $error("SCNN student layer range is invalid");
    end
endmodule

`default_nettype wire
