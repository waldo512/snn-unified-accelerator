`default_nettype none

// Serializes one finalized output-channel wave through one shared versioned
// membrane store. The resulting stream is spatial-major, then engine-major.
module unified_scnn_wave_partial_lif_serializer #(
    parameter int ENGINES = 12,
    parameter int SPATIAL_ADDR_WIDTH = 15,
    parameter int NUM_NEURONS = 409600,
    parameter int STATE_ADDR_WIDTH = $clog2(NUM_NEURONS),
    parameter int VERSION_WIDTH = 16,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_valid,
    output logic                                      o_ready,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]             i_spatial_addr,
    input  logic [ENGINES*STATE_ADDR_WIDTH-1:0]       i_output_addr_flat,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_accumulator_flat,
    input  logic [VERSION_WIDTH-1:0]                  i_sample_version,
    input  logic signed [15:0]                        i_leak_q8_8,
    input  logic signed [15:0]                        i_threshold_q8_8,
    input  logic                                      i_reset_to_zero,
    output logic                                      o_valid,
    input  logic                                      i_ready,
    output logic [$clog2(ENGINES)-1:0]                o_engine_index,
    output logic [SPATIAL_ADDR_WIDTH-1:0]             o_spatial_addr,
    output logic [STATE_ADDR_WIDTH-1:0]               o_output_addr,
    output logic [VERSION_WIDTH-1:0]                  o_sample_version,
    output logic signed [15:0]                        o_current_q8_8,
    output logic [15:0]                               o_spike_q8_8,
    output logic signed [15:0]                        o_membrane_pre,
    output logic signed [15:0]                        o_membrane_next,
    output logic                                      o_wave_done,
    output logic                                      o_idle,
    output logic                                      o_sticky_error
);
    localparam int ENGINE_INDEX_WIDTH = $clog2(ENGINES);
    localparam logic signed [ACC_WIDTH-1:0] Q8_8_MAX = 32767;
    localparam logic signed [ACC_WIDTH-1:0] Q8_8_MIN = -32768;

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_ISSUE,
        STATE_WAIT
    } state_t;

    state_t state_q;
    logic [ENGINES-1:0] engine_mask_q;
    logic [ENGINE_INDEX_WIDTH-1:0] engine_index_q;
    logic [SPATIAL_ADDR_WIDTH-1:0] spatial_addr_q;
    logic [ENGINES*STATE_ADDR_WIDTH-1:0] output_addr_flat_q;
    logic [ENGINES*ACC_WIDTH-1:0] accumulator_flat_q;
    logic [VERSION_WIDTH-1:0] version_q;
    logic signed [15:0] leak_q;
    logic signed [15:0] threshold_q;
    logic reset_to_zero_q;
    logic request_legal;
    logic first_found;
    logic [ENGINE_INDEX_WIDTH-1:0] first_engine;
    logic next_found;
    logic [ENGINE_INDEX_WIDTH-1:0] next_engine;
    logic [STATE_ADDR_WIDTH-1:0] selected_addr;
    logic signed [ACC_WIDTH-1:0] selected_raw;
    logic signed [ACC_WIDTH-1:0] selected_shifted;
    logic signed [15:0] selected_current;
    logic state_input_valid;
    logic state_input_ready;
    logic state_output_valid;
    logic state_output_ready;
    logic [STATE_ADDR_WIDTH-1:0] state_output_addr;
    logic [VERSION_WIDTH-1:0] state_output_version;
    logic [15:0] state_output_spike;
    logic signed [15:0] state_membrane_pre;
    logic signed [15:0] state_membrane_next;
    logic state_idle;
    logic state_address_error;
    logic state_version_error;
    logic state_issue_fire;
    logic output_fire;

    function automatic logic signed [15:0] saturate_q8_8(
        input logic signed [ACC_WIDTH-1:0] value
    );
        begin
            if (value > Q8_8_MAX)
                saturate_q8_8 = 16'sh7fff;
            else if (value < Q8_8_MIN)
                saturate_q8_8 = 16'sh8000;
            else
                saturate_q8_8 = value[15:0];
        end
    endfunction

    always_comb begin
        request_legal = i_engine_mask != '0 && i_sample_version != 0;
        first_found = 1'b0;
        first_engine = '0;
        next_found = 1'b0;
        next_engine = '0;
        for (integer engine = 0; engine < ENGINES; engine++) begin
            if (i_engine_mask[engine] &&
                {1'b0, i_output_addr_flat[
                    engine*STATE_ADDR_WIDTH +: STATE_ADDR_WIDTH]} >=
                    (STATE_ADDR_WIDTH+1)'(NUM_NEURONS))
                request_legal = 1'b0;
            if (i_engine_mask[engine] && !first_found) begin
                first_found = 1'b1;
                first_engine = ENGINE_INDEX_WIDTH'(engine);
            end
            if (engine > integer'(engine_index_q) &&
                engine_mask_q[engine] && !next_found) begin
                next_found = 1'b1;
                next_engine = ENGINE_INDEX_WIDTH'(engine);
            end
        end
    end

    assign selected_addr = output_addr_flat_q[
        engine_index_q*STATE_ADDR_WIDTH +: STATE_ADDR_WIDTH];
    assign selected_raw = $signed(accumulator_flat_q[
        engine_index_q*ACC_WIDTH +: ACC_WIDTH]);
    assign selected_shifted = selected_raw >>> 8;
    assign selected_current = saturate_q8_8(selected_shifted);
    assign o_ready = state_q == STATE_IDLE && !o_sticky_error;
    assign state_input_valid = state_q == STATE_ISSUE;
    assign state_issue_fire = state_input_valid && state_input_ready;
    assign o_valid = state_q == STATE_WAIT && state_output_valid;
    assign state_output_ready = state_q == STATE_WAIT && i_ready;
    assign output_fire = o_valid && i_ready;
    assign o_engine_index = engine_index_q;
    assign o_spatial_addr = spatial_addr_q;
    assign o_output_addr = state_output_addr;
    assign o_sample_version = state_output_version;
    assign o_current_q8_8 = selected_current;
    assign o_spike_q8_8 = state_output_spike;
    assign o_membrane_pre = state_membrane_pre;
    assign o_membrane_next = state_membrane_next;
    assign o_idle = state_q == STATE_IDLE && state_idle;

    unified_scnn_q8_8_versioned_state_engine #(
        .NUM_NEURONS(NUM_NEURONS), .ADDR_WIDTH(STATE_ADDR_WIDTH),
        .VERSION_WIDTH(VERSION_WIDTH)
    ) u_state (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_valid(state_input_valid), .o_ready(state_input_ready),
        .i_addr(selected_addr), .i_sample_version(version_q),
        .i_current(selected_current), .i_leak_q8_8(leak_q),
        .i_threshold_q8_8(threshold_q), .i_reset_to_zero(reset_to_zero_q),
        .o_valid(state_output_valid), .i_ready(state_output_ready),
        .o_addr(state_output_addr), .o_sample_version(state_output_version),
        .o_spike_q8_8(state_output_spike),
        .o_membrane_pre(state_membrane_pre),
        .o_membrane_next(state_membrane_next), .o_idle(state_idle),
        .o_address_error(state_address_error),
        .o_zero_version_error(state_version_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            engine_mask_q <= '0;
            engine_index_q <= '0;
            spatial_addr_q <= '0;
            output_addr_flat_q <= '0;
            accumulator_flat_q <= '0;
            version_q <= '0;
            leak_q <= '0;
            threshold_q <= '0;
            reset_to_zero_q <= 1'b0;
            o_wave_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_wave_done <= 1'b0;
            if (i_clear_error && o_idle)
                o_sticky_error <= 1'b0;
            if (state_address_error || state_version_error)
                o_sticky_error <= 1'b1;
            case (state_q)
                STATE_IDLE: begin
                    if (i_valid && !request_legal) begin
                        o_sticky_error <= 1'b1;
                    end else if (i_valid && o_ready) begin
                        engine_mask_q <= i_engine_mask;
                        engine_index_q <= first_engine;
                        spatial_addr_q <= i_spatial_addr;
                        output_addr_flat_q <= i_output_addr_flat;
                        accumulator_flat_q <= i_accumulator_flat;
                        version_q <= i_sample_version;
                        leak_q <= i_leak_q8_8;
                        threshold_q <= i_threshold_q8_8;
                        reset_to_zero_q <= i_reset_to_zero;
                        state_q <= STATE_ISSUE;
                    end
                end
                STATE_ISSUE: begin
                    if (state_issue_fire)
                        state_q <= STATE_WAIT;
                end
                STATE_WAIT: begin
                    if (output_fire) begin
                        if (next_found) begin
                            engine_index_q <= next_engine;
                            state_q <= STATE_ISSUE;
                        end else begin
                            engine_mask_q <= '0;
                            state_q <= STATE_IDLE;
                            o_wave_done <= 1'b1;
                        end
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
