`timescale 1ns/1ps
`default_nettype none

// FC W0 gradient adapter. It stores one BxH hidden-error window and a sparse
// feature-to-sample mask, then reduces one feature/hidden address at a time.
// No full W0 gradient tensor is materialized.
module unified_fc_tp_w0_sparse_sign_generator #(
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int ERROR_WIDTH = 18,
    parameter int ACC_WIDTH = 48,
    parameter int MAX_BATCH = 16,
    parameter int MAX_HIDDEN = 400,
    parameter int MAX_FEATURES = 2312,
    parameter int PARAMETER_HIDDEN_STRIDE = 0,
    parameter int SAMPLE_ID_WIDTH = (MAX_BATCH <= 2) ? 1 : $clog2(MAX_BATCH),
    parameter int HIDDEN_INDEX_WIDTH = (MAX_HIDDEN <= 2) ? 1 : $clog2(MAX_HIDDEN),
    parameter int FEATURE_INDEX_WIDTH = (MAX_FEATURES <= 2) ? 1 : $clog2(MAX_FEATURES)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,

    input  logic                                  i_cfg_valid,
    output logic                                  o_cfg_ready,
    input  logic [TENSOR_ID_WIDTH-1:0]             i_cfg_tensor_id,
    input  logic [SAMPLE_ID_WIDTH:0]               i_cfg_batch_count,
    input  logic [HIDDEN_INDEX_WIDTH:0]            i_cfg_hidden_count,
    input  logic [FEATURE_INDEX_WIDTH:0]           i_cfg_feature_count,
    output logic                                  o_cfg_error,
    output logic                                  o_active,
    output logic [TENSOR_ID_WIDTH-1:0]             o_active_tensor_id,
    input  logic                                  i_release,
    output logic                                  o_release_ready,

    input  logic                                  i_hidden_error_valid,
    output logic                                  o_hidden_error_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]             i_hidden_error_sample_id,
    input  logic [HIDDEN_INDEX_WIDTH-1:0]          i_hidden_error_hidden_index,
    input  logic signed [ERROR_WIDTH-1:0]          i_hidden_error_value,
    output logic                                  o_hidden_error_error,

    input  logic                                  i_event_valid,
    output logic                                  o_event_ready,
    input  logic [SAMPLE_ID_WIDTH-1:0]             i_event_sample_id,
    input  logic [FEATURE_INDEX_WIDTH-1:0]         i_event_feature_index,
    output logic                                  o_event_error,

    input  logic                                  i_observation_done,
    output logic                                  o_observation_done_ready,
    input  logic                                  i_emit_start,
    output logic                                  o_emit_ready,
    output logic                                  o_sign_valid,
    input  logic                                  i_sign_ready,
    output logic [TENSOR_ID_WIDTH-1:0]             o_sign_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]            o_sign_local_addr,
    output logic signed [1:0]                     o_gradient_sign,
    output logic signed [ACC_WIDTH-1:0]           o_gradient_accumulator,
    output logic                                  o_emit_done,
    output logic                                  o_busy
);

    localparam int HIDDEN_MEM_DEPTH = MAX_BATCH * MAX_HIDDEN;
    localparam int HIDDEN_MEM_ADDR_WIDTH =
        (HIDDEN_MEM_DEPTH <= 2) ? 1 : $clog2(HIDDEN_MEM_DEPTH);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_CONFIG_CLEAR,
        STATE_LOAD,
        STATE_EVENT_READ,
        STATE_EVENT_WRITE,
        STATE_FEATURE_READ,
        STATE_MASK_READ,
        STATE_SUM_INIT,
        STATE_ERROR_READ,
        STATE_SUM_ADD,
        STATE_SIGN_HOLD,
        STATE_WINDOW_CLEAR
    } state_t;

    state_t state_q;
    logic signed [ERROR_WIDTH-1:0] hidden_error_mem [0:HIDDEN_MEM_DEPTH-1];
    logic [MAX_BATCH-1:0] feature_mask_mem [0:MAX_FEATURES-1];
    logic [FEATURE_INDEX_WIDTH-1:0] active_feature_mem [0:MAX_FEATURES-1];

    logic [TENSOR_ID_WIDTH-1:0] tensor_id_q;
    logic [SAMPLE_ID_WIDTH:0] batch_count_q;
    logic [HIDDEN_INDEX_WIDTH:0] hidden_count_q;
    logic [FEATURE_INDEX_WIDTH:0] feature_count_q;
    logic [FEATURE_INDEX_WIDTH:0] clear_index_q;

    logic [SAMPLE_ID_WIDTH-1:0] expected_sample_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] expected_hidden_q;
    logic hidden_error_complete_q;
    logic observation_sealed_q;
    logic [FEATURE_INDEX_WIDTH:0] active_feature_count_q;

    logic [SAMPLE_ID_WIDTH-1:0] pending_event_sample_q;
    logic [FEATURE_INDEX_WIDTH-1:0] pending_event_feature_q;
    logic [MAX_BATCH-1:0] pending_event_mask_q;

    logic [FEATURE_INDEX_WIDTH-1:0] feature_list_index_q;
    logic [FEATURE_INDEX_WIDTH-1:0] current_feature_q;
    logic [MAX_BATCH-1:0] current_feature_mask_q;
    logic [HIDDEN_INDEX_WIDTH-1:0] current_hidden_q;
    logic [SAMPLE_ID_WIDTH-1:0] current_sample_q;
    logic signed [ERROR_WIDTH-1:0] error_read_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [ACC_WIDTH-1:0] sign_accumulator_q;

    logic cfg_legal;
    logic hidden_error_fire;
    logic event_fire;
    logic observation_done_fire;
    logic emit_fire;
    logic sign_fire;
    logic hidden_error_order_legal;
    logic event_legal;
    logic hidden_last;
    logic sample_last;
    logic feature_last;
    logic signed [ACC_WIDTH-1:0] error_extended;
    logic signed [ACC_WIDTH-1:0] sum_next;
    logic [HIDDEN_MEM_ADDR_WIDTH-1:0] hidden_write_addr;
    logic [HIDDEN_MEM_ADDR_WIDTH-1:0] hidden_read_addr;
    logic [PARAM_ADDR_WIDTH-1:0] feature_addr_extended;
    logic [PARAM_ADDR_WIDTH-1:0] hidden_count_extended;
    logic [PARAM_ADDR_WIDTH-1:0] hidden_index_extended;

    always_comb begin
        cfg_legal = (i_cfg_batch_count != '0) &&
            (i_cfg_batch_count <= (SAMPLE_ID_WIDTH+1)'(MAX_BATCH)) &&
            (i_cfg_hidden_count != '0) &&
            (i_cfg_hidden_count <= (HIDDEN_INDEX_WIDTH+1)'(MAX_HIDDEN)) &&
            (i_cfg_feature_count != '0) &&
            (i_cfg_feature_count <= (FEATURE_INDEX_WIDTH+1)'(MAX_FEATURES)) &&
            ((PARAMETER_HIDDEN_STRIDE == 0) ||
             ((PARAM_ADDR_WIDTH+1)'(PARAMETER_HIDDEN_STRIDE) >=
              (PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count))) &&
            ((PARAM_ADDR_WIDTH+1)'(i_cfg_hidden_count) *
             (PARAM_ADDR_WIDTH+1)'(i_cfg_feature_count) <=
             {1'b1, {PARAM_ADDR_WIDTH{1'b0}}});

        o_cfg_ready = state_q == STATE_IDLE;
        o_release_ready = (state_q == STATE_LOAD) && o_active;
        o_hidden_error_ready = (state_q == STATE_LOAD) && o_active &&
            !hidden_error_complete_q && !i_release;
        hidden_error_fire = i_hidden_error_valid && o_hidden_error_ready;
        hidden_error_order_legal =
            (i_hidden_error_sample_id == expected_sample_q) &&
            (i_hidden_error_hidden_index == expected_hidden_q);

        // Sparse events may arrive while the TP frontend is still producing
        // hidden errors. Hidden errors have handshake priority when both
        // sources assert valid in the same cycle.
        o_event_ready = (state_q == STATE_LOAD) && o_active &&
            !observation_sealed_q && !i_release && !i_observation_done &&
            !i_hidden_error_valid;
        event_fire = i_event_valid && o_event_ready;
        event_legal =
            ({1'b0, i_event_sample_id} < batch_count_q) &&
            ({1'b0, i_event_feature_index} < feature_count_q);

        o_observation_done_ready = (state_q == STATE_LOAD) && o_active &&
            hidden_error_complete_q && !observation_sealed_q &&
            !i_event_valid && !i_release;
        observation_done_fire = i_observation_done && o_observation_done_ready;
        o_emit_ready = (state_q == STATE_LOAD) && o_active &&
            observation_sealed_q && !i_release;
        emit_fire = i_emit_start && o_emit_ready;

        o_sign_valid = state_q == STATE_SIGN_HOLD;
        sign_fire = o_sign_valid && i_sign_ready;
        o_sign_tensor_id = tensor_id_q;
        o_gradient_accumulator = sign_accumulator_q;
        o_gradient_sign = (sign_accumulator_q > 0) ? 2'sd1 : -2'sd1;
        feature_addr_extended = PARAM_ADDR_WIDTH'(current_feature_q);
        hidden_count_extended = (PARAMETER_HIDDEN_STRIDE == 0) ?
            PARAM_ADDR_WIDTH'(hidden_count_q) :
            PARAM_ADDR_WIDTH'(PARAMETER_HIDDEN_STRIDE);
        hidden_index_extended = PARAM_ADDR_WIDTH'(current_hidden_q);
        o_sign_local_addr = feature_addr_extended * hidden_count_extended +
            hidden_index_extended;
        o_active_tensor_id = tensor_id_q;
        o_busy = (state_q != STATE_IDLE) && (state_q != STATE_LOAD);

        hidden_last = ({1'b0, current_hidden_q} + 1'b1) >= hidden_count_q;
        sample_last = ({1'b0, current_sample_q} + 1'b1) >= batch_count_q;
        feature_last = ({1'b0, feature_list_index_q} + 1'b1) >=
            active_feature_count_q;
        error_extended = {{(ACC_WIDTH-ERROR_WIDTH){error_read_q[ERROR_WIDTH-1]}},
                          error_read_q};
        sum_next = accumulator_q;
        if (current_feature_mask_q[current_sample_q])
            sum_next = accumulator_q + error_extended;

        hidden_write_addr = HIDDEN_MEM_ADDR_WIDTH'(
            (i_hidden_error_sample_id * MAX_HIDDEN) +
            i_hidden_error_hidden_index);
        hidden_read_addr = HIDDEN_MEM_ADDR_WIDTH'(
            (current_sample_q * MAX_HIDDEN) + current_hidden_q);
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            tensor_id_q <= '0;
            batch_count_q <= '0;
            hidden_count_q <= '0;
            feature_count_q <= '0;
            clear_index_q <= '0;
            expected_sample_q <= '0;
            expected_hidden_q <= '0;
            hidden_error_complete_q <= 1'b0;
            observation_sealed_q <= 1'b0;
            active_feature_count_q <= '0;
            pending_event_sample_q <= '0;
            pending_event_feature_q <= '0;
            pending_event_mask_q <= '0;
            feature_list_index_q <= '0;
            current_feature_q <= '0;
            current_feature_mask_q <= '0;
            current_hidden_q <= '0;
            current_sample_q <= '0;
            error_read_q <= '0;
            accumulator_q <= '0;
            sign_accumulator_q <= '0;
            o_active <= 1'b0;
            o_cfg_error <= 1'b0;
            o_hidden_error_error <= 1'b0;
            o_event_error <= 1'b0;
            o_emit_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_hidden_error_error <= 1'b0;
            o_event_error <= 1'b0;
            o_emit_done <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (i_cfg_valid) begin
                        if (!cfg_legal) begin
                            o_cfg_error <= 1'b1;
                        end else begin
                            tensor_id_q <= i_cfg_tensor_id;
                            batch_count_q <= i_cfg_batch_count;
                            hidden_count_q <= i_cfg_hidden_count;
                            feature_count_q <= i_cfg_feature_count;
                            clear_index_q <= '0;
                            o_active <= 1'b0;
                            state_q <= STATE_CONFIG_CLEAR;
                        end
                    end
                end

                STATE_CONFIG_CLEAR: begin
                    feature_mask_mem[clear_index_q[FEATURE_INDEX_WIDTH-1:0]] <= '0;
                    if ((clear_index_q + 1'b1) >= feature_count_q) begin
                        clear_index_q <= '0;
                        expected_sample_q <= '0;
                        expected_hidden_q <= '0;
                        hidden_error_complete_q <= 1'b0;
                        observation_sealed_q <= 1'b0;
                        active_feature_count_q <= '0;
                        o_active <= 1'b1;
                        state_q <= STATE_LOAD;
                    end else begin
                        clear_index_q <= clear_index_q + 1'b1;
                    end
                end

                STATE_LOAD: begin
                    if (i_release && o_release_ready) begin
                        o_active <= 1'b0;
                        tensor_id_q <= '0;
                        batch_count_q <= '0;
                        hidden_count_q <= '0;
                        feature_count_q <= '0;
                        state_q <= STATE_IDLE;
                    end else if (hidden_error_fire) begin
                        if (!hidden_error_order_legal) begin
                            o_hidden_error_error <= 1'b1;
                        end else begin
                            hidden_error_mem[hidden_write_addr] <= i_hidden_error_value;
                            if (({1'b0, expected_hidden_q} + 1'b1) >= hidden_count_q) begin
                                expected_hidden_q <= '0;
                                if (({1'b0, expected_sample_q} + 1'b1) >=
                                    batch_count_q) begin
                                    hidden_error_complete_q <= 1'b1;
                                end else begin
                                    expected_sample_q <= expected_sample_q + 1'b1;
                                end
                            end else begin
                                expected_hidden_q <= expected_hidden_q + 1'b1;
                            end
                        end
                    end else if (event_fire) begin
                        if (!event_legal) begin
                            o_event_error <= 1'b1;
                        end else begin
                            pending_event_sample_q <= i_event_sample_id;
                            pending_event_feature_q <= i_event_feature_index;
                            state_q <= STATE_EVENT_READ;
                        end
                    end else if (observation_done_fire) begin
                        observation_sealed_q <= 1'b1;
                    end else if (emit_fire) begin
                        if (active_feature_count_q == 0) begin
                            clear_index_q <= '0;
                            o_emit_done <= 1'b1;
                            state_q <= STATE_WINDOW_CLEAR;
                        end else begin
                            feature_list_index_q <= '0;
                            state_q <= STATE_FEATURE_READ;
                        end
                    end
                end

                STATE_EVENT_READ: begin
                    pending_event_mask_q <= feature_mask_mem[pending_event_feature_q];
                    state_q <= STATE_EVENT_WRITE;
                end

                STATE_EVENT_WRITE: begin
                    feature_mask_mem[pending_event_feature_q] <=
                        pending_event_mask_q |
                        ({{(MAX_BATCH-1){1'b0}}, 1'b1} << pending_event_sample_q);
                    if (pending_event_mask_q == '0) begin
                        active_feature_mem[
                            active_feature_count_q[FEATURE_INDEX_WIDTH-1:0]] <=
                            pending_event_feature_q;
                        active_feature_count_q <= active_feature_count_q + 1'b1;
                    end
                    state_q <= STATE_LOAD;
                end

                STATE_FEATURE_READ: begin
                    current_feature_q <= active_feature_mem[feature_list_index_q];
                    state_q <= STATE_MASK_READ;
                end

                STATE_MASK_READ: begin
                    current_feature_mask_q <= feature_mask_mem[current_feature_q];
                    current_hidden_q <= '0;
                    state_q <= STATE_SUM_INIT;
                end

                STATE_SUM_INIT: begin
                    current_sample_q <= '0;
                    accumulator_q <= '0;
                    state_q <= STATE_ERROR_READ;
                end

                STATE_ERROR_READ: begin
                    error_read_q <= hidden_error_mem[hidden_read_addr];
                    state_q <= STATE_SUM_ADD;
                end

                STATE_SUM_ADD: begin
                    accumulator_q <= sum_next;
                    if (sample_last) begin
                        if (sum_next != 0) begin
                            sign_accumulator_q <= sum_next;
                            state_q <= STATE_SIGN_HOLD;
                        end else if (hidden_last) begin
                            if (feature_last) begin
                                clear_index_q <= '0;
                                o_emit_done <= 1'b1;
                                state_q <= STATE_WINDOW_CLEAR;
                            end else begin
                                feature_list_index_q <= feature_list_index_q + 1'b1;
                                state_q <= STATE_FEATURE_READ;
                            end
                        end else begin
                            current_hidden_q <= current_hidden_q + 1'b1;
                            state_q <= STATE_SUM_INIT;
                        end
                    end else begin
                        current_sample_q <= current_sample_q + 1'b1;
                        state_q <= STATE_ERROR_READ;
                    end
                end

                STATE_SIGN_HOLD: begin
                    if (sign_fire) begin
                        if (hidden_last) begin
                            if (feature_last) begin
                                clear_index_q <= '0;
                                o_emit_done <= 1'b1;
                                state_q <= STATE_WINDOW_CLEAR;
                            end else begin
                                feature_list_index_q <= feature_list_index_q + 1'b1;
                                state_q <= STATE_FEATURE_READ;
                            end
                        end else begin
                            current_hidden_q <= current_hidden_q + 1'b1;
                            state_q <= STATE_SUM_INIT;
                        end
                    end
                end

                STATE_WINDOW_CLEAR: begin
                    feature_mask_mem[clear_index_q[FEATURE_INDEX_WIDTH-1:0]] <= '0;
                    if ((clear_index_q + 1'b1) >= feature_count_q) begin
                        clear_index_q <= '0;
                        expected_sample_q <= '0;
                        expected_hidden_q <= '0;
                        hidden_error_complete_q <= 1'b0;
                        observation_sealed_q <= 1'b0;
                        active_feature_count_q <= '0;
                        state_q <= STATE_LOAD;
                    end else begin
                        clear_index_q <= clear_index_q + 1'b1;
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
