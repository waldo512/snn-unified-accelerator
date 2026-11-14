`timescale 1ns/1ps
`default_nettype none

// Topology-neutral two-tensor backend for the hardware-native TP direct update.
// Dense gradients arrive in tensor/address order. Zero gradients are retired
// locally; nonzero gradients share one parameter RMW service. Tensor completion
// is reported only after the final writeback has drained and the context is
// released.
module unified_tp_two_tensor_direct_rmw_backend #(
    parameter int PROFILE_ID_WIDTH = 8,
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int TAG_WIDTH = 16,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter int VOTE_WIDTH = 16
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    input  logic                                      i_cfg_valid,
    output logic                                      o_cfg_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_cfg_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_tensor0_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_tensor0_base,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_tensor0_count,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_tensor0_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_tensor0_threshold,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_cfg_tensor1_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_tensor1_base,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_cfg_tensor1_count,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_tensor1_increment,
    input  logic [VOTE_WIDTH-1:0]                     i_cfg_tensor1_threshold,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_min,
    input  logic signed [CODE_WIDTH-1:0]              i_cfg_code_max,
    output logic                                      o_cfg_error,

    input  logic                                      i_start,
    output logic                                      o_start_ready,
    input  logic                                      i_abort,

    input  logic                                      i_gradient_valid,
    output logic                                      o_gradient_ready,
    input  logic [PROFILE_ID_WIDTH-1:0]               i_gradient_profile_id,
    input  logic [TENSOR_ID_WIDTH-1:0]                i_gradient_tensor_id,
    input  logic [PARAM_ADDR_WIDTH-1:0]               i_gradient_local_addr,
    input  logic signed [63:0]                        i_gradient_q40,
    input  logic                                      i_gradient_nonzero,
    input  logic                                      i_gradient_tensor_last,
    input  logic                                      i_gradient_window_last,

    output logic                                      o_tensor_complete_valid,
    input  logic                                      i_tensor_complete_ready,
    output logic [TENSOR_ID_WIDTH-1:0]                o_tensor_complete_id,
    output logic                                      o_tensor_complete_error,

    output logic                                      o_read_valid,
    input  logic                                      i_read_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_read_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_read_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_read_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_read_tag,
    input  logic                                      i_read_response_valid,
    output logic                                      o_read_response_ready,
    input  logic signed [CODE_WIDTH-1:0]              i_read_response_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]          i_read_response_residual,
    input  logic                                      i_read_response_error,

    output logic                                      o_result_valid,
    input  logic                                      i_result_ready,
    output logic [PROFILE_ID_WIDTH-1:0]               o_result_profile_id,
    output logic [TENSOR_ID_WIDTH-1:0]                o_result_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]               o_result_parameter_addr,
    output logic [TAG_WIDTH-1:0]                      o_result_tag,
    output logic signed [CODE_WIDTH-1:0]              o_result_code,
    output logic signed [RESIDUAL_WIDTH-1:0]          o_result_residual,
    output logic                                      o_write_code_enable,
    output logic                                      o_write_residual_enable,
    output logic                                      o_write_code_fire,
    output logic                                      o_write_residual_fire,
    output logic                                      o_dirty,
    output logic                                      o_saturated,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_aborted,
    output logic                                      o_protocol_error,
    output logic                                      o_store_error,
    output logic                                      o_internal_error,
    output logic [PARAM_ADDR_WIDTH:0]                 o_zero_gradient_count,
    output logic [PARAM_ADDR_WIDTH:0]                 o_update_count
);

    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_LOAD0,
        STATE_STREAM0,
        STATE_DRAIN0,
        STATE_RELEASE0,
        STATE_COMPLETE0,
        STATE_LOAD1,
        STATE_STREAM1,
        STATE_DRAIN1,
        STATE_RELEASE1,
        STATE_COMPLETE1,
        STATE_ABORT_DRAIN,
        STATE_ABORT_RELEASE,
        STATE_ABORT_DONE
    } state_t;

    state_t state_q;
    logic configured_q;
    logic [PROFILE_ID_WIDTH-1:0] profile_id_q;
    logic [TENSOR_ID_WIDTH-1:0] tensor0_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] tensor0_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] tensor0_count_q;
    logic [VOTE_WIDTH-1:0] tensor0_increment_q;
    logic [VOTE_WIDTH-1:0] tensor0_threshold_q;
    logic [TENSOR_ID_WIDTH-1:0] tensor1_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] tensor1_base_q;
    logic [PARAM_ADDR_WIDTH-1:0] tensor1_count_q;
    logic [VOTE_WIDTH-1:0] tensor1_increment_q;
    logic [VOTE_WIDTH-1:0] tensor1_threshold_q;
    logic signed [CODE_WIDTH-1:0] code_min_q;
    logic signed [CODE_WIDTH-1:0] code_max_q;
    logic [PARAM_ADDR_WIDTH-1:0] expected_addr_q;
    logic abort_pending_q;

    logic cfg_legal;
    logic tensor0_ranges_before_tensor1;
    logic tensor1_ranges_before_tensor0;
    logic [PARAM_ADDR_WIDTH:0] tensor0_end_ext;
    logic [PARAM_ADDR_WIDTH:0] tensor1_end_ext;
    logic [VOTE_WIDTH:0] tensor0_twice_threshold;
    logic [VOTE_WIDTH:0] tensor1_twice_threshold;
    logic selected_tensor0;
    logic selected_tensor1;
    logic selected_stream;
    logic selected_last_expected;
    logic gradient_metadata_legal;
    logic gradient_fire;
    logic gradient_is_zero;
    logic rmw_cfg_valid;
    logic rmw_cfg_ready;
    logic [TENSOR_ID_WIDTH-1:0] rmw_cfg_tensor_id;
    logic [PARAM_ADDR_WIDTH-1:0] rmw_cfg_base;
    logic [PARAM_ADDR_WIDTH-1:0] rmw_cfg_count;
    logic [VOTE_WIDTH-1:0] rmw_cfg_increment;
    logic [VOTE_WIDTH-1:0] rmw_cfg_threshold;
    logic rmw_cfg_error;
    logic rmw_release;
    logic rmw_release_ready;
    logic rmw_active;
    logic [PROFILE_ID_WIDTH-1:0] rmw_active_profile_id;
    logic [TENSOR_ID_WIDTH-1:0] rmw_active_tensor_id;
    logic rmw_req_valid;
    logic rmw_req_ready;
    logic signed [1:0] rmw_req_direction;
    logic rmw_req_error;
    logic rmw_update_config_error;
    logic rmw_store_error;
    logic rmw_internal_error;
    logic rmw_busy;

    always_comb begin
        tensor0_end_ext = {1'b0, i_cfg_tensor0_base} +
            {1'b0, i_cfg_tensor0_count};
        tensor1_end_ext = {1'b0, i_cfg_tensor1_base} +
            {1'b0, i_cfg_tensor1_count};
        tensor0_ranges_before_tensor1 =
            tensor0_end_ext <= {1'b0, i_cfg_tensor1_base};
        tensor1_ranges_before_tensor0 =
            tensor1_end_ext <= {1'b0, i_cfg_tensor0_base};
        tensor0_twice_threshold = {1'b0, i_cfg_tensor0_threshold} +
            {1'b0, i_cfg_tensor0_threshold};
        tensor1_twice_threshold = {1'b0, i_cfg_tensor1_threshold} +
            {1'b0, i_cfg_tensor1_threshold};
        cfg_legal = (i_cfg_tensor0_id != i_cfg_tensor1_id) &&
            (i_cfg_tensor0_count != '0) && (i_cfg_tensor1_count != '0) &&
            !tensor0_end_ext[PARAM_ADDR_WIDTH] &&
            !tensor1_end_ext[PARAM_ADDR_WIDTH] &&
            (tensor0_ranges_before_tensor1 || tensor1_ranges_before_tensor0) &&
            (i_cfg_tensor0_threshold != '0) &&
            ({1'b0, i_cfg_tensor0_increment} < tensor0_twice_threshold) &&
            (i_cfg_tensor1_threshold != '0) &&
            ({1'b0, i_cfg_tensor1_increment} < tensor1_twice_threshold) &&
            ($signed(i_cfg_code_min) <= $signed(i_cfg_code_max));

        o_cfg_ready = state_q == STATE_IDLE;
        o_start_ready = (state_q == STATE_IDLE) && configured_q &&
            !i_cfg_valid;
        o_busy = state_q != STATE_IDLE;

        selected_tensor0 = (state_q == STATE_STREAM0) ||
            (state_q == STATE_DRAIN0) || (state_q == STATE_RELEASE0) ||
            (state_q == STATE_COMPLETE0);
        selected_tensor1 = (state_q == STATE_STREAM1) ||
            (state_q == STATE_DRAIN1) || (state_q == STATE_RELEASE1) ||
            (state_q == STATE_COMPLETE1);
        selected_stream = (state_q == STATE_STREAM0) ||
            (state_q == STATE_STREAM1);
        selected_last_expected = selected_tensor0 ?
            (expected_addr_q == tensor0_count_q - 1'b1) :
            (expected_addr_q == tensor1_count_q - 1'b1);
        gradient_is_zero = i_gradient_q40 == 64'sd0;
        gradient_metadata_legal =
            rmw_active &&
            (rmw_active_profile_id == profile_id_q) &&
            (rmw_active_tensor_id ==
                (selected_tensor0 ? tensor0_id_q : tensor1_id_q)) &&
            (i_gradient_profile_id == profile_id_q) &&
            (i_gradient_tensor_id ==
                (selected_tensor0 ? tensor0_id_q : tensor1_id_q)) &&
            (i_gradient_local_addr == expected_addr_q) &&
            (i_gradient_nonzero == !gradient_is_zero) &&
            (i_gradient_tensor_last == selected_last_expected) &&
            (i_gradient_window_last ==
                (selected_tensor1 && selected_last_expected));

        rmw_req_valid = selected_stream && i_gradient_valid &&
            gradient_metadata_legal && !gradient_is_zero && !abort_pending_q;
        rmw_req_direction = i_gradient_q40 > 0 ? -2'sd1 : 2'sd1;
        o_gradient_ready = selected_stream && !abort_pending_q &&
            (gradient_metadata_legal ?
                (gradient_is_zero ? 1'b1 : rmw_req_ready) : 1'b1);
        gradient_fire = i_gradient_valid && o_gradient_ready;

        rmw_cfg_valid = (state_q == STATE_LOAD0) ||
            (state_q == STATE_LOAD1);
        rmw_cfg_tensor_id = state_q == STATE_LOAD0 ? tensor0_id_q :
            tensor1_id_q;
        rmw_cfg_base = state_q == STATE_LOAD0 ? tensor0_base_q :
            tensor1_base_q;
        rmw_cfg_count = state_q == STATE_LOAD0 ? tensor0_count_q :
            tensor1_count_q;
        rmw_cfg_increment = state_q == STATE_LOAD0 ? tensor0_increment_q :
            tensor1_increment_q;
        rmw_cfg_threshold = state_q == STATE_LOAD0 ? tensor0_threshold_q :
            tensor1_threshold_q;
        rmw_release = (state_q == STATE_RELEASE0) ||
            (state_q == STATE_RELEASE1) || (state_q == STATE_ABORT_RELEASE);

        o_tensor_complete_valid = (state_q == STATE_COMPLETE0) ||
            (state_q == STATE_COMPLETE1);
        o_tensor_complete_id = state_q == STATE_COMPLETE0 ? tensor0_id_q :
            tensor1_id_q;
        o_tensor_complete_error = 1'b0;
    end

    unified_tp_parameter_rmw_service #(
        .PROFILE_ID_WIDTH(PROFILE_ID_WIDTH),
        .TENSOR_ID_WIDTH(TENSOR_ID_WIDTH),
        .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
        .TAG_WIDTH(TAG_WIDTH),
        .CODE_WIDTH(CODE_WIDTH),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .VOTE_WIDTH(VOTE_WIDTH)
    ) rmw_service (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_cfg_valid(rmw_cfg_valid),
        .o_cfg_ready(rmw_cfg_ready),
        .i_cfg_profile_id(profile_id_q),
        .i_cfg_tensor_id(rmw_cfg_tensor_id),
        .i_cfg_parameter_base(rmw_cfg_base),
        .i_cfg_element_count(rmw_cfg_count),
        .i_cfg_vote_increment(rmw_cfg_increment),
        .i_cfg_vote_threshold(rmw_cfg_threshold),
        .i_cfg_code_min(code_min_q),
        .i_cfg_code_max(code_max_q),
        .o_cfg_error(rmw_cfg_error),
        .i_release(rmw_release),
        .o_release_ready(rmw_release_ready),
        .o_active(rmw_active),
        .o_active_profile_id(rmw_active_profile_id),
        .o_active_tensor_id(rmw_active_tensor_id),
        .i_req_valid(rmw_req_valid),
        .o_req_ready(rmw_req_ready),
        .i_req_local_parameter_addr(i_gradient_local_addr),
        .i_req_tag(TAG_WIDTH'(i_gradient_local_addr)),
        .i_req_update_direction(rmw_req_direction),
        .i_req_magnitude_mode(1'b0),
        .i_req_scaled_residual_delta('0),
        .o_req_error(rmw_req_error),
        .o_read_valid(o_read_valid),
        .i_read_ready(i_read_ready),
        .o_read_profile_id(o_read_profile_id),
        .o_read_tensor_id(o_read_tensor_id),
        .o_read_parameter_addr(o_read_parameter_addr),
        .o_read_tag(o_read_tag),
        .i_read_response_valid(i_read_response_valid),
        .o_read_response_ready(o_read_response_ready),
        .i_read_response_code(i_read_response_code),
        .i_read_response_residual(i_read_response_residual),
        .i_read_response_error(i_read_response_error),
        .o_result_valid(o_result_valid),
        .i_result_ready(i_result_ready),
        .o_result_profile_id(o_result_profile_id),
        .o_result_tensor_id(o_result_tensor_id),
        .o_result_parameter_addr(o_result_parameter_addr),
        .o_result_tag(o_result_tag),
        .o_result_code(o_result_code),
        .o_result_residual(o_result_residual),
        .o_write_code_enable(o_write_code_enable),
        .o_write_residual_enable(o_write_residual_enable),
        .o_write_code_fire(o_write_code_fire),
        .o_write_residual_fire(o_write_residual_fire),
        .o_dirty(o_dirty),
        .o_saturated(o_saturated),
        .o_update_config_error(rmw_update_config_error),
        .o_store_error(rmw_store_error),
        .o_internal_error(rmw_internal_error),
        .o_busy(rmw_busy)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            configured_q <= 1'b0;
            profile_id_q <= '0;
            tensor0_id_q <= '0;
            tensor0_base_q <= '0;
            tensor0_count_q <= '0;
            tensor0_increment_q <= '0;
            tensor0_threshold_q <= '0;
            tensor1_id_q <= '0;
            tensor1_base_q <= '0;
            tensor1_count_q <= '0;
            tensor1_increment_q <= '0;
            tensor1_threshold_q <= '0;
            code_min_q <= '0;
            code_max_q <= '0;
            expected_addr_q <= '0;
            abort_pending_q <= 1'b0;
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_aborted <= 1'b0;
            o_protocol_error <= 1'b0;
            o_store_error <= 1'b0;
            o_internal_error <= 1'b0;
            o_zero_gradient_count <= '0;
            o_update_count <= '0;
        end else begin
            o_cfg_error <= 1'b0;
            o_done <= 1'b0;
            o_aborted <= 1'b0;

            if (i_cfg_valid && o_cfg_ready) begin
                if (!cfg_legal) begin
                    configured_q <= 1'b0;
                    o_cfg_error <= 1'b1;
                end else begin
                    configured_q <= 1'b1;
                    profile_id_q <= i_cfg_profile_id;
                    tensor0_id_q <= i_cfg_tensor0_id;
                    tensor0_base_q <= i_cfg_tensor0_base;
                    tensor0_count_q <= i_cfg_tensor0_count;
                    tensor0_increment_q <= i_cfg_tensor0_increment;
                    tensor0_threshold_q <= i_cfg_tensor0_threshold;
                    tensor1_id_q <= i_cfg_tensor1_id;
                    tensor1_base_q <= i_cfg_tensor1_base;
                    tensor1_count_q <= i_cfg_tensor1_count;
                    tensor1_increment_q <= i_cfg_tensor1_increment;
                    tensor1_threshold_q <= i_cfg_tensor1_threshold;
                    code_min_q <= i_cfg_code_min;
                    code_max_q <= i_cfg_code_max;
                end
            end

            if (rmw_cfg_error || rmw_req_error || rmw_update_config_error ||
                rmw_store_error || rmw_internal_error) begin
                abort_pending_q <= 1'b1;
                if (rmw_store_error)
                    o_store_error <= 1'b1;
                if (rmw_cfg_error || rmw_req_error || rmw_update_config_error ||
                    rmw_internal_error)
                    o_internal_error <= 1'b1;
            end
            if (i_abort && (state_q != STATE_IDLE))
                abort_pending_q <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (i_start && o_start_ready) begin
                        expected_addr_q <= '0;
                        abort_pending_q <= 1'b0;
                        o_protocol_error <= 1'b0;
                        o_store_error <= 1'b0;
                        o_internal_error <= 1'b0;
                        o_zero_gradient_count <= '0;
                        o_update_count <= '0;
                        state_q <= STATE_LOAD0;
                    end
                end
                STATE_LOAD0: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (rmw_cfg_valid && rmw_cfg_ready)
                        state_q <= STATE_STREAM0;
                end
                STATE_STREAM0: begin
                    if (abort_pending_q) begin
                        state_q <= STATE_ABORT_DRAIN;
                    end else if (gradient_fire) begin
                        if (!gradient_metadata_legal) begin
                            o_protocol_error <= 1'b1;
                            abort_pending_q <= 1'b1;
                            state_q <= STATE_ABORT_DRAIN;
                        end else begin
                            if (gradient_is_zero)
                                o_zero_gradient_count <=
                                    o_zero_gradient_count + 1'b1;
                            else
                                o_update_count <= o_update_count + 1'b1;
                            if (selected_last_expected)
                                state_q <= STATE_DRAIN0;
                            else
                                expected_addr_q <= expected_addr_q + 1'b1;
                        end
                    end
                end
                STATE_DRAIN0: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (!rmw_busy)
                        state_q <= STATE_RELEASE0;
                end
                STATE_RELEASE0: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (rmw_release && rmw_release_ready)
                        state_q <= STATE_COMPLETE0;
                end
                STATE_COMPLETE0: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (o_tensor_complete_valid &&
                             i_tensor_complete_ready) begin
                        expected_addr_q <= '0;
                        state_q <= STATE_LOAD1;
                    end
                end
                STATE_LOAD1: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (rmw_cfg_valid && rmw_cfg_ready)
                        state_q <= STATE_STREAM1;
                end
                STATE_STREAM1: begin
                    if (abort_pending_q) begin
                        state_q <= STATE_ABORT_DRAIN;
                    end else if (gradient_fire) begin
                        if (!gradient_metadata_legal) begin
                            o_protocol_error <= 1'b1;
                            abort_pending_q <= 1'b1;
                            state_q <= STATE_ABORT_DRAIN;
                        end else begin
                            if (gradient_is_zero)
                                o_zero_gradient_count <=
                                    o_zero_gradient_count + 1'b1;
                            else
                                o_update_count <= o_update_count + 1'b1;
                            if (selected_last_expected)
                                state_q <= STATE_DRAIN1;
                            else
                                expected_addr_q <= expected_addr_q + 1'b1;
                        end
                    end
                end
                STATE_DRAIN1: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (!rmw_busy)
                        state_q <= STATE_RELEASE1;
                end
                STATE_RELEASE1: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (rmw_release && rmw_release_ready)
                        state_q <= STATE_COMPLETE1;
                end
                STATE_COMPLETE1: begin
                    if (abort_pending_q)
                        state_q <= STATE_ABORT_DRAIN;
                    else if (o_tensor_complete_valid &&
                             i_tensor_complete_ready) begin
                        o_done <= 1'b1;
                        state_q <= STATE_IDLE;
                    end
                end
                STATE_ABORT_DRAIN: begin
                    if (!rmw_busy)
                        state_q <= rmw_active ? STATE_ABORT_RELEASE :
                            STATE_ABORT_DONE;
                end
                STATE_ABORT_RELEASE: begin
                    if (rmw_release && rmw_release_ready)
                        state_q <= STATE_ABORT_DONE;
                end
                STATE_ABORT_DONE: begin
                    o_done <= 1'b1;
                    o_aborted <= 1'b1;
                    state_q <= STATE_IDLE;
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
