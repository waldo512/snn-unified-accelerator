`timescale 1ns/1ps
`default_nettype none

// Label-driven start of the SCNN TP target path. Bank 0 is fixed, so the
// projector streams one output row through the shared forward-read service
// instead of instantiating a 65536x11 register matrix. LIF and trace state are
// indexed by sample and persist across timestep transactions.
module unified_scnn_tp_target_projector_q8_8 #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int VERSION_WIDTH = 16,
    parameter int MAX_BATCH = 10,
    parameter int MAX_TIMESTEPS = 20,
    parameter int TARGET_DIM = 11,
    parameter int OUTPUT_CHANNELS = 64,
    parameter int OUTPUT_SPATIAL = 32,
    parameter int SAMPLE_WIDTH = MAX_BATCH <= 2 ? 1 : $clog2(MAX_BATCH),
    parameter int TIMESTEP_WIDTH = MAX_TIMESTEPS <= 2 ? 1 :
        $clog2(MAX_TIMESTEPS),
    parameter int CLASS_WIDTH = TARGET_DIM <= 2 ? 1 : $clog2(TARGET_DIM),
    parameter int OUTPUT_COUNT = OUTPUT_CHANNELS*OUTPUT_SPATIAL*OUTPUT_SPATIAL,
    parameter int OUTPUT_INDEX_WIDTH = OUTPUT_COUNT <= 2 ? 1 :
        $clog2(OUTPUT_COUNT),
    parameter int CHANNEL_WIDTH = OUTPUT_CHANNELS <= 2 ? 1 :
        $clog2(OUTPUT_CHANNELS),
    parameter int SPATIAL_INDEX_WIDTH = OUTPUT_SPATIAL*OUTPUT_SPATIAL <= 2 ? 1 :
        $clog2(OUTPUT_SPATIAL*OUTPUT_SPATIAL),
    parameter TARGET_SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_sample_state_reset,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_parameter_base,
    input  logic [SAMPLE_WIDTH-1:0]                   i_cfg_sample,
    input  logic [TIMESTEP_WIDTH-1:0]                 i_cfg_timestep,
    input  logic [VERSION_WIDTH-1:0]                  i_cfg_sample_version,
    input  logic signed [15:0]                        i_cfg_membrane_leak_q8_8,
    input  logic signed [15:0]                        i_cfg_trace_leak_q8_8,
    input  logic signed [15:0]                        i_cfg_threshold_q8_8,
    input  logic                                      i_cfg_reset_to_zero,

    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [CLASS_WIDTH-1:0]                    i_label_class,
    input  logic signed [15:0]                        i_label_spike_q8_8,
    input  logic                                      i_label_last,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_read_tag,
    input  logic                                      i_read_response_valid,
    output logic                                      o_read_response_ready,
    input  logic [TAG_WIDTH-1:0]                      i_read_response_tag,
    input  logic signed [15:0]                        i_read_response_code,
    input  logic                                      i_read_response_error,

    output logic                                      o_observation_valid,
    input  logic                                      i_observation_ready,
    output logic [SAMPLE_WIDTH-1:0]                   o_observation_sample,
    output logic [TIMESTEP_WIDTH-1:0]                 o_observation_timestep,
    output logic [VERSION_WIDTH-1:0]                  o_observation_sample_version,
    output logic [OUTPUT_INDEX_WIDTH-1:0]             o_observation_output_index,
    output logic [CHANNEL_WIDTH-1:0]                  o_observation_channel,
    output logic [SPATIAL_INDEX_WIDTH-1:0]            o_observation_spatial,
    output logic signed [15:0]                        o_observation_current_q8_8,
    output logic        [15:0]                        o_observation_spike_q8_8,
    output logic signed [15:0]                        o_observation_membrane_pre_q8_8,
    output logic signed [15:0]                        o_observation_membrane_next_q8_8,
    output logic signed [15:0]                        o_observation_trace_q8_8,
    output logic        [15:0]                        o_observation_surrogate_q8_8,
    output logic                                      o_observation_last,
    output logic                                      o_done,
    output logic                                      o_busy,
    output logic                                      o_protocol_error
);
    localparam int OUTPUT_PLANE = OUTPUT_SPATIAL*OUTPUT_SPATIAL;
    localparam int STATE_COUNT = MAX_BATCH*OUTPUT_COUNT;
    localparam int STATE_ADDR_WIDTH = STATE_COUNT <= 2 ? 1 : $clog2(STATE_COUNT);
    localparam int LABEL_COUNT_WIDTH = $clog2(TARGET_DIM+1);
    localparam int ACC_WIDTH = 40;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LABEL,
        ST_READ_REQUEST,
        ST_READ_RESPONSE,
        ST_LIF_REQUEST,
        ST_LIF_RESPONSE,
        ST_OBSERVATION,
        ST_ERROR
    } state_t;

    state_t state_q;
    logic signed [15:0] label_mem [0:TARGET_DIM-1];
    logic [TARGET_DIM-1:0] label_seen_q;
    logic [LABEL_COUNT_WIDTH-1:0] label_count_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] parameter_base_q;
    logic [SAMPLE_WIDTH-1:0] sample_q;
    logic [TIMESTEP_WIDTH-1:0] timestep_q;
    logic [VERSION_WIDTH-1:0] sample_version_q;
    logic signed [15:0] membrane_leak_q;
    logic signed [15:0] trace_leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic [OUTPUT_INDEX_WIDTH-1:0] output_index_q;
    logic [CLASS_WIDTH-1:0] class_index_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [15:0] projected_current_q;
    logic [TAG_WIDTH-1:0] next_tag_q;
    logic [TAG_WIDTH-1:0] expected_tag_q;

    logic signed [31:0] weight_product;
    logic signed [31:0] weight_term;
    logic signed [ACC_WIDTH-1:0] weight_term_extended;
    logic signed [ACC_WIDTH-1:0] accumulator_next;
    logic [PARAM_ADDR_WIDTH-1:0] row_offset;
    logic [STATE_ADDR_WIDTH-1:0] state_addr;

    logic lif_input_ready;
    logic lif_output_valid;
    logic lif_output_ready;
    logic [STATE_ADDR_WIDTH-1:0] lif_output_addr;
    logic [15:0] lif_spike;
    logic signed [15:0] lif_membrane_pre;
    logic signed [15:0] lif_membrane_next;
    logic lif_idle;
    logic lif_address_error;
    logic lif_zero_epoch_error;

    logic trace_input_ready;
    logic trace_output_valid;
    logic trace_output_ready;
    logic [STATE_ADDR_WIDTH-1:0] trace_output_addr;
    logic signed [15:0] trace_output;
    logic trace_idle;
    logic trace_address_error;
    logic trace_zero_epoch_error;

    logic surrogate_input_ready;
    logic surrogate_output_valid;
    logic surrogate_output_ready;
    logic [15:0] surrogate_output;
    logic branch_launch;

    logic [OUTPUT_INDEX_WIDTH-1:0] observation_output_index_q;
    logic [CHANNEL_WIDTH-1:0] observation_channel_q;
    logic [SPATIAL_INDEX_WIDTH-1:0] observation_spatial_q;
    logic signed [15:0] observation_current_q;
    logic [15:0] observation_spike_q;
    logic signed [15:0] observation_membrane_pre_q;
    logic signed [15:0] observation_membrane_next_q;
    logic protocol_error_q;
    logic [VERSION_WIDTH-1:0] state_epoch_q;
    integer label_index;

    function automatic logic signed [15:0] saturate_q8_8(
        input logic signed [ACC_WIDTH-1:0] value
    );
        begin
            if (value > ACC_WIDTH'(32767))
                saturate_q8_8 = 16'sh7fff;
            else if (value < -ACC_WIDTH'(32768))
                saturate_q8_8 = 16'sh8000;
            else
                saturate_q8_8 = value[15:0];
        end
    endfunction

    always_comb begin
        row_offset = PARAM_ADDR_WIDTH'(
            int'(output_index_q)*TARGET_DIM + int'(class_index_q));
        state_addr = STATE_ADDR_WIDTH'(
            int'(sample_q)*OUTPUT_COUNT + int'(output_index_q));
        weight_product = $signed(label_mem[class_index_q]) *
            $signed(i_read_response_code);
        weight_term = weight_product >>> 8;
        weight_term_extended = {{(ACC_WIDTH-32){weight_term[31]}}, weight_term};
        accumulator_next = accumulator_q + weight_term_extended;
    end

    assign o_cfg_ready = state_q == ST_IDLE && lif_idle && trace_idle &&
        !i_sample_state_reset && !protocol_error_q;
    assign o_label_ready = state_q == ST_LABEL && !protocol_error_q;
    assign o_read_valid = state_q == ST_READ_REQUEST;
    assign o_read_profile_id = profile_id_q;
    assign o_read_tensor_id = tensor_id_q;
    assign o_read_parameter_addr = parameter_base_q + row_offset;
    assign o_read_tag = next_tag_q;
    assign o_read_response_ready = state_q == ST_READ_RESPONSE;
    assign o_busy = state_q != ST_IDLE;
    assign o_protocol_error = protocol_error_q || lif_address_error ||
        lif_zero_epoch_error || trace_address_error || trace_zero_epoch_error;

    assign branch_launch = state_q == ST_LIF_RESPONSE && lif_output_valid &&
        trace_input_ready && surrogate_input_ready;
    assign lif_output_ready = branch_launch;

    assign o_observation_valid = state_q == ST_OBSERVATION &&
        trace_output_valid && surrogate_output_valid;
    assign trace_output_ready = o_observation_valid && i_observation_ready;
    assign surrogate_output_ready = o_observation_valid && i_observation_ready;
    assign o_observation_sample = sample_q;
    assign o_observation_timestep = timestep_q;
    assign o_observation_sample_version = sample_version_q;
    assign o_observation_output_index = observation_output_index_q;
    assign o_observation_channel = observation_channel_q;
    assign o_observation_spatial = observation_spatial_q;
    assign o_observation_current_q8_8 = observation_current_q;
    assign o_observation_spike_q8_8 = observation_spike_q;
    assign o_observation_membrane_pre_q8_8 = observation_membrane_pre_q;
    assign o_observation_membrane_next_q8_8 = observation_membrane_next_q;
    assign o_observation_trace_q8_8 = trace_output;
    assign o_observation_surrogate_q8_8 = surrogate_output;
    assign o_observation_last = observation_output_index_q ==
        OUTPUT_INDEX_WIDTH'(OUTPUT_COUNT-1);

    unified_scnn_q8_8_versioned_state_engine_soft_abort_v1 #(
        .NUM_NEURONS(STATE_COUNT), .ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .STATE_INIT_FILE("NONE")
    ) u_target_lif_state (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_clear_error),
        .i_valid(state_q == ST_LIF_REQUEST), .o_ready(lif_input_ready),
        .i_addr(state_addr), .i_sample_version(state_epoch_q),
        .i_current(projected_current_q),
        .i_leak_q8_8(membrane_leak_q), .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .o_valid(lif_output_valid), .i_ready(lif_output_ready),
        .o_addr(lif_output_addr), .o_sample_version(),
        .o_spike_q8_8(lif_spike),
        .o_membrane_pre(lif_membrane_pre),
        .o_membrane_next(lif_membrane_next), .o_idle(lif_idle),
        .o_address_error(lif_address_error),
        .o_zero_version_error(lif_zero_epoch_error)
    );

    unified_scnn_q8_8_versioned_trace_state_engine #(
        .NUM_NEURONS(STATE_COUNT), .ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH), .STATE_INIT_FILE("NONE")
    ) u_target_trace_state (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_clear_error),
        .i_valid(branch_launch), .o_ready(trace_input_ready),
        .i_addr(lif_output_addr), .i_state_epoch(state_epoch_q),
        .i_spike_q8_8($signed(lif_spike)),
        .i_leak_q8_8(trace_leak_q),
        .o_valid(trace_output_valid), .i_ready(trace_output_ready),
        .o_addr(trace_output_addr), .o_state_epoch(),
        .o_trace_q8_8(trace_output), .o_idle(trace_idle),
        .o_address_error(trace_address_error),
        .o_zero_epoch_error(trace_zero_epoch_error)
    );

    unified_scnn_tp_surrogate_type1_q8_8 #(
        .ROM_FILE(TARGET_SURROGATE_ROM_FILE)
    ) u_target_surrogate (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_clear_error),
        .i_valid(branch_launch), .o_ready(surrogate_input_ready),
        .i_membrane_pre_q8_8(lif_membrane_pre),
        .i_threshold_q8_8(threshold_q),
        .o_valid(surrogate_output_valid), .i_ready(surrogate_output_ready),
        .o_surrogate_q8_8(surrogate_output)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= ST_IDLE;
            label_seen_q <= '0;
            label_count_q <= '0;
            profile_id_q <= '0;
            tensor_id_q <= '0;
            parameter_base_q <= '0;
            sample_q <= '0;
            timestep_q <= '0;
            sample_version_q <= '0;
            membrane_leak_q <= '0;
            trace_leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            output_index_q <= '0;
            class_index_q <= '0;
            accumulator_q <= '0;
            projected_current_q <= '0;
            next_tag_q <= '0;
            expected_tag_q <= '0;
            observation_output_index_q <= '0;
            observation_channel_q <= '0;
            observation_spatial_q <= '0;
            observation_current_q <= '0;
            observation_spike_q <= '0;
            observation_membrane_pre_q <= '0;
            observation_membrane_next_q <= '0;
            protocol_error_q <= 1'b0;
            state_epoch_q <= VERSION_WIDTH'(1);
            o_done <= 1'b0;
            for (label_index = 0; label_index < TARGET_DIM;
                 label_index = label_index + 1)
                label_mem[label_index] <= '0;
        end else begin
            o_done <= 1'b0;

            if (i_clear_error) begin
                protocol_error_q <= 1'b0;
                if (state_q == ST_ERROR)
                    state_q <= ST_IDLE;
            end

            if (i_sample_state_reset && state_q != ST_IDLE) begin
                protocol_error_q <= 1'b1;
                state_q <= ST_ERROR;
            end else if (i_sample_state_reset) begin
                if (&state_epoch_q) begin
                    protocol_error_q <= 1'b1;
                    state_q <= ST_ERROR;
                end else begin
                    state_epoch_q <= state_epoch_q + 1'b1;
                end
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (i_cfg_valid && o_cfg_ready) begin
                        profile_id_q <= i_cfg_profile_id;
                        tensor_id_q <= i_cfg_tensor_id;
                        parameter_base_q <= i_cfg_parameter_base;
                        sample_q <= i_cfg_sample;
                        timestep_q <= i_cfg_timestep;
                        sample_version_q <= i_cfg_sample_version;
                        membrane_leak_q <= i_cfg_membrane_leak_q8_8;
                        trace_leak_q <= i_cfg_trace_leak_q8_8;
                        threshold_q <= i_cfg_threshold_q8_8;
                        reset_to_zero_q <= i_cfg_reset_to_zero;
                        label_seen_q <= '0;
                        label_count_q <= '0;
                        output_index_q <= '0;
                        class_index_q <= '0;
                        accumulator_q <= '0;
                        state_q <= ST_LABEL;
                    end
                end

                ST_LABEL: begin
                    if (i_label_valid && o_label_ready) begin
                        if (i_label_class >= CLASS_WIDTH'(TARGET_DIM) ||
                            label_seen_q[i_label_class]) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else begin
                            label_mem[i_label_class] <= i_label_spike_q8_8;
                            label_seen_q[i_label_class] <= 1'b1;
                            label_count_q <= label_count_q + 1'b1;
                            if (i_label_last) begin
                                if (label_count_q == LABEL_COUNT_WIDTH'(TARGET_DIM-1)) begin
                                    output_index_q <= '0;
                                    class_index_q <= '0;
                                    accumulator_q <= '0;
                                    state_q <= ST_READ_REQUEST;
                                end else begin
                                    protocol_error_q <= 1'b1;
                                    state_q <= ST_ERROR;
                                end
                            end else if (label_count_q ==
                                         LABEL_COUNT_WIDTH'(TARGET_DIM-1)) begin
                                protocol_error_q <= 1'b1;
                                state_q <= ST_ERROR;
                            end
                        end
                    end
                end

                ST_READ_REQUEST: begin
                    if (o_read_valid && i_read_ready) begin
                        expected_tag_q <= next_tag_q;
                        next_tag_q <= next_tag_q + 1'b1;
                        state_q <= ST_READ_RESPONSE;
                    end
                end

                ST_READ_RESPONSE: begin
                    if (i_read_response_valid && o_read_response_ready) begin
                        if (i_read_response_error ||
                            i_read_response_tag != expected_tag_q) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (class_index_q == CLASS_WIDTH'(TARGET_DIM-1)) begin
                            projected_current_q <= saturate_q8_8(accumulator_next);
                            class_index_q <= '0;
                            accumulator_q <= '0;
                            state_q <= ST_LIF_REQUEST;
                        end else begin
                            accumulator_q <= accumulator_next;
                            class_index_q <= class_index_q + 1'b1;
                            state_q <= ST_READ_REQUEST;
                        end
                    end
                end

                ST_LIF_REQUEST: begin
                    if (lif_input_ready)
                        state_q <= ST_LIF_RESPONSE;
                end

                ST_LIF_RESPONSE: begin
                    if (branch_launch) begin
                        observation_output_index_q <= output_index_q;
                        observation_channel_q <= CHANNEL_WIDTH'(
                            int'(output_index_q) / OUTPUT_PLANE);
                        observation_spatial_q <= SPATIAL_INDEX_WIDTH'(
                            int'(output_index_q) % OUTPUT_PLANE);
                        observation_current_q <= projected_current_q;
                        observation_spike_q <= lif_spike;
                        observation_membrane_pre_q <= lif_membrane_pre;
                        observation_membrane_next_q <= lif_membrane_next;
                        state_q <= ST_OBSERVATION;
                    end
                end

                ST_OBSERVATION: begin
                    if (o_observation_valid && i_observation_ready) begin
                        if (trace_output_addr != lif_output_addr) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else if (output_index_q ==
                                     OUTPUT_INDEX_WIDTH'(OUTPUT_COUNT-1)) begin
                            o_done <= 1'b1;
                            state_q <= ST_IDLE;
                        end else begin
                            output_index_q <= output_index_q + 1'b1;
                            class_index_q <= '0;
                            accumulator_q <= '0;
                            state_q <= ST_READ_REQUEST;
                        end
                    end
                end

                default: begin
                    state_q <= ST_ERROR;
                end
            endcase
        end
    end

    initial begin
        if (MAX_BATCH < 1 || MAX_TIMESTEPS < 1 || TARGET_DIM < 2 ||
            OUTPUT_CHANNELS < 1 || OUTPUT_SPATIAL < 1 || OUTPUT_COUNT < 1)
            $error("SCNN TP target projector dimensions are invalid");
        if (ACC_WIDTH < 33)
            $error("SCNN TP target projector accumulator is too narrow");
        if (STATE_COUNT > (1 << STATE_ADDR_WIDTH))
            $error("SCNN TP target projector state address is too narrow");
    end
endmodule

`default_nettype wire
