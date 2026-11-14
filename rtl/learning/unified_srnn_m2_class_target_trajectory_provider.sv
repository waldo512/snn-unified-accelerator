`timescale 1ns/1ps
`default_nettype none

// Replays one exact target trajectory per class and maps it through the current
// batch labels.  Spike vectors feed the H450 observation adapter; trace replay
// supplies the exact target trace stream used by the TP distance/opposite-trace
// path without rebuilding it from rounded Q14 recurrence.
module unified_srnn_m2_class_target_trajectory_provider #(
    parameter int BATCH_MAX = 64,
    parameter int CLASS_MAX = 20,
    parameter int TIMESTEP_MAX = 128,
    parameter int BACKBONE_MAX = 450,
    parameter int SOURCE_VECTOR_MAX = 480,
    parameter int WORD_WIDTH = 32,
    parameter int TRACE_WIDTH = 32,
    parameter int TRACE_STORAGE_WIDTH = 18,
    parameter int WORD_INDEX_WIDTH =
        (((SOURCE_VECTOR_MAX+WORD_WIDTH-1)/WORD_WIDTH) <= 1) ?
            1 : $clog2((SOURCE_VECTOR_MAX+WORD_WIDTH-1)/WORD_WIDTH)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [$clog2(BATCH_MAX+1)-1:0]            i_cfg_batch_count,
    input  logic [$clog2(CLASS_MAX+1)-1:0]            i_cfg_class_count,
    input  logic [$clog2(TIMESTEP_MAX+1)-1:0]         i_cfg_timestep_count,
    input  logic [$clog2(BACKBONE_MAX+1)-1:0]         i_cfg_backbone_count,
    output logic                                      o_cfg_error,

    input  logic                                      i_label_load_start,
    output logic                                      o_label_load_start_ready,
    input  logic                                      i_label_valid,
    output logic                                      o_label_ready,
    input  logic [$clog2(BATCH_MAX)-1:0]              i_label_sample_id,
    input  logic [$clog2(CLASS_MAX)-1:0]              i_label_class_id,
    input  logic                                      i_label_last,
    output logic                                      o_label_load_done,

    input  logic                                      i_label_replay_start,
    output logic                                      o_label_replay_start_ready,
    output logic                                      o_label_replay_valid,
    input  logic                                      i_label_replay_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_label_replay_sample_id,
    output logic [$clog2(CLASS_MAX)-1:0]              o_label_replay_class_id,
    output logic                                      o_label_replay_last,
    output logic                                      o_label_replay_done,

    input  logic                                      i_spike_word_write_valid,
    output logic                                      o_spike_word_write_ready,
    input  logic [$clog2(CLASS_MAX)-1:0]              i_spike_word_class_id,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_spike_word_timestep_id,
    input  logic [WORD_INDEX_WIDTH-1:0]                 i_spike_word_id,
    input  logic [WORD_WIDTH-1:0]                     i_spike_word_data,

    input  logic                                      i_trace_write_valid,
    output logic                                      o_trace_write_ready,
    input  logic [$clog2(CLASS_MAX)-1:0]              i_trace_class_id,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_trace_timestep_id,
    input  logic [$clog2(BACKBONE_MAX)-1:0]           i_trace_backbone_id,
    input  logic signed [TRACE_WIDTH-1:0]             i_trace_q14,

    input  logic                                      i_vector_replay_start,
    output logic                                      o_vector_replay_start_ready,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_vector_replay_timestep_id,
    output logic                                      o_vector_valid,
    input  logic                                      i_vector_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_vector_sample_id,
    output logic [SOURCE_VECTOR_MAX-1:0]              o_vector_spikes,
    output logic                                      o_vector_last,
    output logic                                      o_vector_done,

    input  logic                                      i_trace_replay_start,
    output logic                                      o_trace_replay_start_ready,
    input  logic [$clog2(TIMESTEP_MAX)-1:0]           i_trace_replay_timestep_id,
    output logic                                      o_trace_valid,
    input  logic                                      i_trace_ready,
    output logic [$clog2(BATCH_MAX)-1:0]              o_trace_sample_id,
    output logic [$clog2(BACKBONE_MAX)-1:0]           o_trace_backbone_id,
    output logic signed [TRACE_WIDTH-1:0]             o_trace_q14,
    output logic                                      o_trace_last,
    output logic                                      o_trace_done,

    input  logic                                      i_abort,
    output logic                                      o_busy,
    output logic                                      o_configured,
    output logic                                      o_labels_loaded,
    output logic                                      o_protocol_error
);

    localparam int BATCH_W = $clog2(BATCH_MAX);
    localparam int CLASS_W = $clog2(CLASS_MAX);
    localparam int TIMESTEP_W = $clog2(TIMESTEP_MAX);
    localparam int BACKBONE_W = $clog2(BACKBONE_MAX);
    localparam int WORDS_PER_VECTOR =
        (SOURCE_VECTOR_MAX + WORD_WIDTH - 1) / WORD_WIDTH;
    localparam int WORD_INDEX_W = WORD_INDEX_WIDTH;
    localparam int PADDED_VECTOR_WIDTH = WORDS_PER_VECTOR * WORD_WIDTH;
    localparam int SPIKE_DEPTH = CLASS_MAX * TIMESTEP_MAX * WORDS_PER_VECTOR;
    localparam int TRACE_DEPTH = CLASS_MAX * TIMESTEP_MAX * BACKBONE_MAX;
    localparam int SPIKE_ADDR_W = (SPIKE_DEPTH <= 1) ? 1 : $clog2(SPIKE_DEPTH);
    localparam int TRACE_ADDR_W = (TRACE_DEPTH <= 1) ? 1 : $clog2(TRACE_DEPTH);

    (* ram_style = "block" *) logic [WORD_WIDTH-1:0]
        spike_word_mem [0:SPIKE_DEPTH-1];
    (* ram_style = "block" *) logic signed [TRACE_STORAGE_WIDTH-1:0]
        trace_mem [0:TRACE_DEPTH-1];
    logic [CLASS_W-1:0] label_mem [0:BATCH_MAX-1];

    logic configured_q;
    logic labels_loaded_q;
    logic label_load_open_q;
    logic [$clog2(BATCH_MAX+1)-1:0] batch_count_q;
    logic [$clog2(CLASS_MAX+1)-1:0] class_count_q;
    logic [$clog2(TIMESTEP_MAX+1)-1:0] timestep_count_q;
    logic [$clog2(BACKBONE_MAX+1)-1:0] backbone_count_q;
    logic [BATCH_W-1:0] label_expected_sample_q;
    logic label_replay_active_q;
    logic [BATCH_W-1:0] label_replay_sample_q;

    logic vector_active_q;
    logic vector_read_pending_q;
    logic vector_output_valid_q;
    logic [TIMESTEP_W-1:0] vector_timestep_q;
    logic [BATCH_W-1:0] vector_sample_q;
    logic [WORD_INDEX_W-1:0] vector_word_q;
    logic [WORD_WIDTH-1:0] vector_word_data_q;
    logic [PADDED_VECTOR_WIDTH-1:0] vector_data_q;

    logic trace_active_q;
    logic trace_read_pending_q;
    logic trace_output_valid_q;
    logic [TIMESTEP_W-1:0] trace_timestep_q;
    logic [BATCH_W-1:0] trace_sample_q;
    logic [BACKBONE_W-1:0] trace_backbone_q;
    logic signed [TRACE_WIDTH-1:0] trace_data_q;

    logic protocol_error_q;

    logic cfg_legal;
    logic idle_for_cfg;
    logic label_fire;
    logic label_last_expected;
    logic label_input_legal;
    logic label_replay_fire;
    logic label_replay_is_last;
    logic write_idle;
    logic spike_word_write_legal;
    logic trace_write_legal;
    logic trace_write_fits;
    logic vector_start_legal;
    logic trace_start_legal;
    logic vector_fire;
    logic vector_is_last;
    logic trace_fire;
    logic trace_is_last;
    logic [SOURCE_VECTOR_MAX-1:0] vector_mask;

    function automatic logic [SPIKE_ADDR_W-1:0] spike_address(
        input logic [CLASS_W-1:0] class_id,
        input logic [TIMESTEP_W-1:0] timestep_id,
        input logic [WORD_INDEX_W-1:0] word_id
    );
        begin
            spike_address = SPIKE_ADDR_W'(
                ((int'(class_id) * TIMESTEP_MAX) + int'(timestep_id)) *
                WORDS_PER_VECTOR + int'(word_id));
        end
    endfunction

    function automatic logic [TRACE_ADDR_W-1:0] trace_address(
        input logic [CLASS_W-1:0] class_id,
        input logic [TIMESTEP_W-1:0] timestep_id,
        input logic [BACKBONE_W-1:0] backbone_id
    );
        begin
            trace_address = TRACE_ADDR_W'(
                ((int'(class_id) * TIMESTEP_MAX) + int'(timestep_id)) *
                BACKBONE_MAX + int'(backbone_id));
        end
    endfunction

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (int'(i_cfg_batch_count) <= BATCH_MAX) &&
            (i_cfg_class_count != '0) &&
            (int'(i_cfg_class_count) <= CLASS_MAX) &&
            (i_cfg_timestep_count != '0) &&
            (int'(i_cfg_timestep_count) <= TIMESTEP_MAX) &&
            (i_cfg_backbone_count != '0) &&
            (int'(i_cfg_backbone_count) <= BACKBONE_MAX) &&
            (int'(i_cfg_backbone_count) <= SOURCE_VECTOR_MAX);

        idle_for_cfg = !label_load_open_q && !label_replay_active_q &&
            !vector_active_q &&
            !vector_output_valid_q && !trace_active_q &&
            !trace_output_valid_q;
        o_cfg_ready = idle_for_cfg;
        o_label_load_start_ready = configured_q && idle_for_cfg &&
            !protocol_error_q && !i_cfg_valid;
        o_label_ready = label_load_open_q && !protocol_error_q;
        label_fire = i_label_valid && o_label_ready;
        label_last_expected = label_expected_sample_q ==
            BATCH_W'(batch_count_q - 1'b1);
        label_input_legal = (i_label_sample_id == label_expected_sample_q) &&
            (int'(i_label_class_id) < int'(class_count_q)) &&
            (i_label_last == label_last_expected);
        o_label_replay_start_ready = configured_q && labels_loaded_q &&
            !label_load_open_q && !label_replay_active_q &&
            !protocol_error_q && !i_cfg_valid;
        o_label_replay_valid = label_replay_active_q;
        o_label_replay_sample_id = label_replay_sample_q;
        o_label_replay_class_id = label_mem[label_replay_sample_q];
        label_replay_is_last = label_replay_sample_q ==
            BATCH_W'(batch_count_q - 1'b1);
        o_label_replay_last = o_label_replay_valid &&
            label_replay_is_last;
        label_replay_fire = o_label_replay_valid && i_label_replay_ready;

        write_idle = configured_q && !label_load_open_q &&
            !vector_active_q && !vector_output_valid_q &&
            !trace_active_q && !trace_output_valid_q && !protocol_error_q;
        o_spike_word_write_ready = write_idle;
        spike_word_write_legal =
            (int'(i_spike_word_class_id) < int'(class_count_q)) &&
            (int'(i_spike_word_timestep_id) < int'(timestep_count_q)) &&
            (int'(i_spike_word_id) < WORDS_PER_VECTOR);
        o_trace_write_ready = write_idle;
        trace_write_fits = i_trace_q14 ==
            {{(TRACE_WIDTH-TRACE_STORAGE_WIDTH){
                i_trace_q14[TRACE_STORAGE_WIDTH-1]}},
             i_trace_q14[TRACE_STORAGE_WIDTH-1:0]};
        trace_write_legal =
            (int'(i_trace_class_id) < int'(class_count_q)) &&
            (int'(i_trace_timestep_id) < int'(timestep_count_q)) &&
            (int'(i_trace_backbone_id) < int'(backbone_count_q)) &&
            trace_write_fits;

        o_vector_replay_start_ready = configured_q && labels_loaded_q &&
            idle_for_cfg && !protocol_error_q && !i_cfg_valid;
        vector_start_legal =
            int'(i_vector_replay_timestep_id) < int'(timestep_count_q);
        o_vector_valid = vector_output_valid_q;
        o_vector_sample_id = vector_sample_q;
        vector_mask = '0;
        for (int bit_i = 0; bit_i < SOURCE_VECTOR_MAX; bit_i++) begin
            if (bit_i < int'(backbone_count_q))
                vector_mask[bit_i] = 1'b1;
        end
        o_vector_spikes = vector_data_q[SOURCE_VECTOR_MAX-1:0] & vector_mask;
        vector_is_last = vector_sample_q == BATCH_W'(batch_count_q - 1'b1);
        o_vector_last = o_vector_valid && vector_is_last;
        vector_fire = o_vector_valid && i_vector_ready;

        o_trace_replay_start_ready = configured_q && labels_loaded_q &&
            idle_for_cfg && !protocol_error_q && !i_cfg_valid;
        trace_start_legal =
            int'(i_trace_replay_timestep_id) < int'(timestep_count_q);
        o_trace_valid = trace_output_valid_q;
        o_trace_sample_id = trace_sample_q;
        o_trace_backbone_id = trace_backbone_q;
        o_trace_q14 = trace_data_q;
        trace_is_last =
            (trace_sample_q == BATCH_W'(batch_count_q - 1'b1)) &&
            (trace_backbone_q == BACKBONE_W'(backbone_count_q - 1'b1));
        o_trace_last = o_trace_valid && trace_is_last;
        trace_fire = o_trace_valid && i_trace_ready;

        o_busy = label_load_open_q || label_replay_active_q || vector_active_q ||
            vector_output_valid_q || trace_active_q || trace_output_valid_q;
        o_configured = configured_q;
        o_labels_loaded = labels_loaded_q;
        o_protocol_error = protocol_error_q;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            configured_q <= 1'b0;
            labels_loaded_q <= 1'b0;
            label_load_open_q <= 1'b0;
            batch_count_q <= '0;
            class_count_q <= '0;
            timestep_count_q <= '0;
            backbone_count_q <= '0;
            label_expected_sample_q <= '0;
            label_replay_active_q <= 1'b0;
            label_replay_sample_q <= '0;
            vector_active_q <= 1'b0;
            vector_read_pending_q <= 1'b0;
            vector_output_valid_q <= 1'b0;
            vector_timestep_q <= '0;
            vector_sample_q <= '0;
            vector_word_q <= '0;
            vector_word_data_q <= '0;
            vector_data_q <= '0;
            trace_active_q <= 1'b0;
            trace_read_pending_q <= 1'b0;
            trace_output_valid_q <= 1'b0;
            trace_timestep_q <= '0;
            trace_sample_q <= '0;
            trace_backbone_q <= '0;
            trace_data_q <= '0;
            protocol_error_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_label_load_done <= 1'b0;
            o_label_replay_done <= 1'b0;
            o_vector_done <= 1'b0;
            o_trace_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_label_load_done <= 1'b0;
            o_label_replay_done <= 1'b0;
            o_vector_done <= 1'b0;
            o_trace_done <= 1'b0;

            if (i_clear_error)
                protocol_error_q <= 1'b0;

            if (i_abort) begin
                label_load_open_q <= 1'b0;
                label_replay_active_q <= 1'b0;
                vector_active_q <= 1'b0;
                vector_read_pending_q <= 1'b0;
                vector_output_valid_q <= 1'b0;
                trace_active_q <= 1'b0;
                trace_read_pending_q <= 1'b0;
                trace_output_valid_q <= 1'b0;
            end else begin
                if (i_cfg_valid && o_cfg_ready) begin
                    if (!cfg_legal) begin
                        configured_q <= 1'b0;
                        labels_loaded_q <= 1'b0;
                        o_cfg_error <= 1'b1;
                    end else begin
                        configured_q <= 1'b1;
                        labels_loaded_q <= 1'b0;
                        batch_count_q <= i_cfg_batch_count;
                        class_count_q <= i_cfg_class_count;
                        timestep_count_q <= i_cfg_timestep_count;
                        backbone_count_q <= i_cfg_backbone_count;
                        protocol_error_q <= 1'b0;
                    end
                end

                if (i_label_load_start && o_label_load_start_ready) begin
                    label_load_open_q <= 1'b1;
                    labels_loaded_q <= 1'b0;
                    label_expected_sample_q <= '0;
                    protocol_error_q <= 1'b0;
                end

                if (label_fire) begin
                    if (!label_input_legal) begin
                        protocol_error_q <= 1'b1;
                        label_load_open_q <= 1'b0;
                    end else begin
                        label_mem[i_label_sample_id] <= i_label_class_id;
                        if (label_last_expected) begin
                            label_load_open_q <= 1'b0;
                            labels_loaded_q <= 1'b1;
                            o_label_load_done <= 1'b1;
                        end else begin
                            label_expected_sample_q <=
                                label_expected_sample_q + 1'b1;
                        end
                    end
                end

                if (i_label_replay_start &&
                        o_label_replay_start_ready) begin
                    label_replay_active_q <= 1'b1;
                    label_replay_sample_q <= '0;
                end

                if (label_replay_fire) begin
                    if (label_replay_is_last) begin
                        label_replay_active_q <= 1'b0;
                        o_label_replay_done <= 1'b1;
                    end else begin
                        label_replay_sample_q <=
                            label_replay_sample_q + 1'b1;
                    end
                end

                if (i_spike_word_write_valid && o_spike_word_write_ready) begin
                    if (!spike_word_write_legal) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        spike_word_mem[spike_address(
                            i_spike_word_class_id,
                            i_spike_word_timestep_id,
                            i_spike_word_id)] <= i_spike_word_data;
                    end
                end

                if (i_trace_write_valid && o_trace_write_ready) begin
                    if (!trace_write_legal) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        trace_mem[trace_address(
                            i_trace_class_id,
                            i_trace_timestep_id,
                            i_trace_backbone_id)] <=
                            i_trace_q14[TRACE_STORAGE_WIDTH-1:0];
                    end
                end

                if (i_vector_replay_start && o_vector_replay_start_ready) begin
                    if (!vector_start_legal) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        vector_active_q <= 1'b1;
                        vector_read_pending_q <= 1'b0;
                        vector_output_valid_q <= 1'b0;
                        vector_timestep_q <= i_vector_replay_timestep_id;
                        vector_sample_q <= '0;
                        vector_word_q <= '0;
                        vector_data_q <= '0;
                    end
                end

                if (vector_active_q && !vector_output_valid_q) begin
                    if (!vector_read_pending_q) begin
                        vector_word_data_q <= spike_word_mem[spike_address(
                            label_mem[vector_sample_q],
                            vector_timestep_q,
                            vector_word_q)];
                        vector_read_pending_q <= 1'b1;
                    end else begin
                        vector_data_q[vector_word_q*WORD_WIDTH +: WORD_WIDTH]
                            <= vector_word_data_q;
                        vector_read_pending_q <= 1'b0;
                        if (vector_word_q == WORD_INDEX_W'(WORDS_PER_VECTOR-1)) begin
                            vector_output_valid_q <= 1'b1;
                            vector_word_q <= '0;
                        end else begin
                            vector_word_q <= vector_word_q + 1'b1;
                        end
                    end
                end

                if (vector_fire) begin
                    vector_output_valid_q <= 1'b0;
                    if (vector_is_last) begin
                        vector_active_q <= 1'b0;
                        o_vector_done <= 1'b1;
                    end else begin
                        vector_sample_q <= vector_sample_q + 1'b1;
                        vector_word_q <= '0;
                        vector_data_q <= '0;
                    end
                end

                if (i_trace_replay_start && o_trace_replay_start_ready) begin
                    if (!trace_start_legal) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        trace_active_q <= 1'b1;
                        trace_read_pending_q <= 1'b0;
                        trace_output_valid_q <= 1'b0;
                        trace_timestep_q <= i_trace_replay_timestep_id;
                        trace_sample_q <= '0;
                        trace_backbone_q <= '0;
                    end
                end

                if (trace_active_q && !trace_output_valid_q) begin
                    if (!trace_read_pending_q) begin
                        trace_data_q <= TRACE_WIDTH'($signed(
                            trace_mem[trace_address(
                                label_mem[trace_sample_q],
                                trace_timestep_q,
                                trace_backbone_q)]));
                        trace_read_pending_q <= 1'b1;
                    end else begin
                        trace_output_valid_q <= 1'b1;
                        trace_read_pending_q <= 1'b0;
                    end
                end

                if (trace_fire) begin
                    trace_output_valid_q <= 1'b0;
                    if (trace_is_last) begin
                        trace_active_q <= 1'b0;
                        o_trace_done <= 1'b1;
                    end else if (trace_backbone_q ==
                        BACKBONE_W'(backbone_count_q - 1'b1)) begin
                        trace_backbone_q <= '0;
                        trace_sample_q <= trace_sample_q + 1'b1;
                    end else begin
                        trace_backbone_q <= trace_backbone_q + 1'b1;
                    end
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (TRACE_STORAGE_WIDTH < 2 || TRACE_STORAGE_WIDTH > TRACE_WIDTH)
            $fatal(1, "TRACE_STORAGE_WIDTH must be within [2, TRACE_WIDTH]");
    end
`endif

endmodule

`default_nettype wire
