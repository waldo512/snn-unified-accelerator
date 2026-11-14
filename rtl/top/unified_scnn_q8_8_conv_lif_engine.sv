`default_nettype none

module unified_scnn_q8_8_conv_lif_engine #(
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16
) (
    input  logic                           i_clk,
    input  logic                           i_rst_n,
    input  logic                           i_valid,
    output logic                           o_ready,
    input  logic [127:0]                   i_feature_flat,
    input  logic [127:0]                   i_weight_flat,
    input  logic [7:0]                     i_lane_mask,
    input  logic                           i_first,
    input  logic                           i_last,
    input  logic [STATE_ADDR_WIDTH-1:0]    i_output_addr,
    input  logic [VERSION_WIDTH-1:0]       i_sample_version,
    input  logic signed [15:0]             i_leak_q8_8,
    input  logic signed [15:0]             i_threshold_q8_8,
    input  logic                           i_reset_to_zero,
    output logic                           o_accumulator_commit,
    output logic                           o_valid,
    input  logic                           i_ready,
    output logic [STATE_ADDR_WIDTH-1:0]    o_output_addr,
    output logic [VERSION_WIDTH-1:0]       o_sample_version,
    output logic signed [15:0]             o_current_q8_8,
    output logic [15:0]                    o_spike_q8_8,
    output logic signed [15:0]             o_membrane_next,
    output logic                           o_idle,
    output logic                           o_protocol_error,
    output logic                           o_state_error
);

    logic accumulator_ready;
    logic accumulator_valid;
    logic signed [47:0] accumulator_result;
    logic result_pending_q;
    logic sequence_active_q;
    logic [STATE_ADDR_WIDTH-1:0] output_addr_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic signed [15:0] leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic signed [47:0] shifted_accumulator;
    logic signed [15:0] saturated_current;
    logic state_ready;
    logic state_idle;
    logic state_address_error;
    logic state_zero_version_error;
    logic signed [15:0] state_membrane_pre;
    logic unused_status;

    function automatic logic signed [15:0] saturate_q8_8(
        input logic signed [47:0] value
    );
        begin
            if (value > 48'sd32767)
                saturate_q8_8 = 16'sh7fff;
            else if (value < -48'sd32768)
                saturate_q8_8 = 16'sh8000;
            else
                saturate_q8_8 = value[15:0];
        end
    endfunction

    assign o_ready = accumulator_ready && !result_pending_q;
    assign shifted_accumulator = accumulator_result >>> 8;
    assign saturated_current = saturate_q8_8(shifted_accumulator);
    assign o_accumulator_commit = accumulator_valid && state_ready;
    assign o_current_q8_8 = saturated_current;
    assign o_idle = !sequence_active_q && !result_pending_q &&
                    !accumulator_valid && state_idle;
    assign o_state_error = state_address_error || state_zero_version_error;

    unified_q8_8_k8_accumulator #(
        .LANES(8), .ACC_WIDTH(48)
    ) u_accumulator (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(i_valid && o_ready), .o_ready(accumulator_ready),
        .i_a_flat(i_feature_flat), .i_b_flat(i_weight_flat),
        .i_lane_mask(i_lane_mask), .i_first(i_first), .i_last(i_last),
        .o_valid(accumulator_valid), .i_ready(state_ready),
        .o_accumulator(accumulator_result)
    );

    unified_scnn_q8_8_versioned_state_engine #(
        .NUM_NEURONS(NUM_NEURONS), .ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_state (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(accumulator_valid), .o_ready(state_ready),
        .i_addr(output_addr_q), .i_sample_version(version_q),
        .i_current(saturated_current), .i_leak_q8_8(leak_q),
        .i_threshold_q8_8(threshold_q),
        .i_reset_to_zero(reset_to_zero_q),
        .o_valid(o_valid), .i_ready(i_ready), .o_addr(o_output_addr),
        .o_sample_version(o_sample_version), .o_spike_q8_8(o_spike_q8_8),
        .o_membrane_pre(state_membrane_pre),
        .o_membrane_next(o_membrane_next),
        .o_idle(state_idle), .o_address_error(state_address_error),
        .o_zero_version_error(state_zero_version_error)
    );

    assign unused_status = &{1'b0, state_membrane_pre};

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            result_pending_q <= 1'b0;
            sequence_active_q <= 1'b0;
            output_addr_q <= '0;
            version_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            if (i_valid && o_ready) begin
                if (i_first) begin
                    if (sequence_active_q)
                        o_protocol_error <= 1'b1;
                    output_addr_q <= i_output_addr;
                    version_q <= i_sample_version;
                    leak_q <= i_leak_q8_8;
                    threshold_q <= i_threshold_q8_8;
                    reset_to_zero_q <= i_reset_to_zero;
                    sequence_active_q <= !i_last;
                end else begin
                    if (!sequence_active_q || i_output_addr != output_addr_q ||
                        i_sample_version != version_q)
                        o_protocol_error <= 1'b1;
                    if (i_last)
                        sequence_active_q <= 1'b0;
                end
                if (i_last)
                    result_pending_q <= 1'b1;
            end
            if (o_accumulator_commit)
                result_pending_q <= 1'b0;
        end
    end

endmodule

`default_nettype wire
