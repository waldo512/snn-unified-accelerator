`default_nettype none

// Inserts a read-through binary-activity cache between the SCNN convolution
// frontend and the banked feature store. The cache is enabled only for layers
// whose inputs are unit spikes; analog Conv1 features always bypass it.
module unified_scnn_feature_activity_cache_bridge_v1 #(
    parameter int LANES = 8,
    parameter int BANK_WIDTH = 4,
    parameter int BANKS = 9,
    parameter int FEATURE_ADDR_WIDTH = 14,
    parameter int FEATURE_DEPTH = 9216
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_abort,
    input  logic                                      i_clear_error,
    input  logic                                      i_cache_enable,

    input  logic                                      i_map_clear_start,
    output logic                                      o_map_clear_ready,
    output logic                                      o_map_clear_busy,
    output logic                                      o_map_clear_done,

    input  logic                                      i_core_read_valid,
    output logic                                      o_core_read_ready,
    input  logic                                      i_core_read_slot,
    input  logic [LANES*BANK_WIDTH-1:0]               i_core_read_bank_flat,
    input  logic [LANES*FEATURE_ADDR_WIDTH-1:0]       i_core_read_addr_flat,
    input  logic [LANES-1:0]                          i_core_read_lane_mask,
    output logic                                      o_core_result_valid,
    input  logic                                      i_core_result_ready,
    output logic [LANES*16-1:0]                       o_core_result_flat,
    output logic [LANES-1:0]                          o_core_result_lane_mask,

    output logic                                      o_memory_read_valid,
    input  logic                                      i_memory_read_ready,
    output logic                                      o_memory_read_slot,
    output logic [LANES*BANK_WIDTH-1:0]               o_memory_read_bank_flat,
    output logic [LANES*FEATURE_ADDR_WIDTH-1:0]       o_memory_read_addr_flat,
    output logic [LANES-1:0]                          o_memory_read_lane_mask,
    input  logic                                      i_memory_result_valid,
    output logic                                      o_memory_result_ready,
    input  logic [LANES*16-1:0]                       i_memory_result_flat,
    input  logic [LANES-1:0]                          i_memory_result_lane_mask,

    output logic                                      o_busy,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_RESULT_MASK = 8'h01;
    localparam logic [7:0] ERROR_MODE_CHANGE = 8'h02;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_WAIT_LOOKUP,
        STATE_SEND_MEMORY,
        STATE_WAIT_MEMORY,
        STATE_SEND_CORE
    } state_t;

    state_t state_q;
    logic request_slot_q;
    logic [LANES*BANK_WIDTH-1:0] request_bank_flat_q;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] request_addr_flat_q;
    logic [LANES-1:0] request_lane_mask_q;
    logic [LANES*16-1:0] response_flat_q;
    logic [LANES-1:0] response_lane_mask_q;
    logic accepted_cache_enable_q;

    logic cache_lookup_ready;
    logic cache_lookup_result_valid;
    logic cache_lookup_result_ready;
    logic [LANES-1:0] cache_known_mask;
    logic [LANES-1:0] cache_active_mask;
    logic cache_fill_valid;
    logic cache_fill_ready;
    logic cache_busy;
    logic cache_sticky_error;
    logic [7:0] cache_error_code;
    logic cache_hit;
    logic core_request_fire;
    logic memory_request_fire;
    logic memory_result_fire;
    logic [LANES-1:0] memory_active_mask;
    logic local_error_q;
    logic [7:0] local_error_code_q;

    always_comb begin
        cache_hit = (cache_known_mask & request_lane_mask_q) ==
                    request_lane_mask_q;
        memory_active_mask = '0;
        for (integer lane = 0; lane < LANES; lane++) begin
            memory_active_mask[lane] = request_lane_mask_q[lane] &&
                i_memory_result_lane_mask[lane] &&
                i_memory_result_flat[lane*16 +: 16] != 16'h0000;
        end

        o_core_read_ready = 1'b0;
        o_core_result_valid = 1'b0;
        o_core_result_flat = '0;
        o_core_result_lane_mask = '0;
        o_memory_read_valid = 1'b0;
        o_memory_read_slot = i_core_read_slot;
        o_memory_read_bank_flat = i_core_read_bank_flat;
        o_memory_read_addr_flat = i_core_read_addr_flat;
        o_memory_read_lane_mask = i_core_read_lane_mask;
        o_memory_result_ready = 1'b0;
        cache_lookup_result_ready = 1'b0;
        cache_fill_valid = 1'b0;

        if (i_abort) begin
        end else if (!i_cache_enable) begin
            o_core_read_ready = i_memory_read_ready;
            o_core_result_valid = i_memory_result_valid;
            o_core_result_flat = i_memory_result_flat;
            o_core_result_lane_mask = i_memory_result_lane_mask;
            o_memory_read_valid = i_core_read_valid;
            o_memory_result_ready = i_core_result_ready;
        end else begin
            o_memory_read_slot = request_slot_q;
            o_memory_read_bank_flat = request_bank_flat_q;
            o_memory_read_addr_flat = request_addr_flat_q;
            o_memory_read_lane_mask = request_lane_mask_q;
            unique case (state_q)
                STATE_IDLE: begin
                    o_core_read_ready = cache_lookup_ready &&
                                        !o_sticky_error;
                end
                STATE_WAIT_LOOKUP: begin
                    if (cache_lookup_result_valid) begin
                        if (cache_hit) begin
                            o_core_result_valid = 1'b1;
                            o_core_result_lane_mask = request_lane_mask_q;
                            for (integer lane = 0; lane < LANES; lane++)
                                if (cache_active_mask[lane])
                                    o_core_result_flat[lane*16 +: 16] =
                                        16'h0100;
                            cache_lookup_result_ready = i_core_result_ready;
                        end else begin
                            cache_lookup_result_ready = 1'b1;
                        end
                    end
                end
                STATE_SEND_MEMORY: begin
                    o_memory_read_valid = 1'b1;
                end
                STATE_WAIT_MEMORY: begin
                    o_memory_result_ready = cache_fill_ready;
                    cache_fill_valid = i_memory_result_valid;
                end
                STATE_SEND_CORE: begin
                    o_core_result_valid = 1'b1;
                    o_core_result_flat = response_flat_q;
                    o_core_result_lane_mask = response_lane_mask_q;
                end
                default: begin
                end
            endcase
        end
    end

    assign core_request_fire = i_core_read_valid && o_core_read_ready;
    assign memory_request_fire = o_memory_read_valid && i_memory_read_ready;
    assign memory_result_fire = i_memory_result_valid &&
                                o_memory_result_ready;
    assign o_busy = state_q != STATE_IDLE || cache_busy;
    assign o_sticky_error = local_error_q || cache_sticky_error;
    assign o_error_code = local_error_q ? local_error_code_q :
                          cache_sticky_error ? cache_error_code : ERROR_NONE;

    unified_scnn_banked_feature_activity_cache_v1 #(
        .LANES(LANES), .BANKS(BANKS), .BANK_WIDTH(BANK_WIDTH),
        .DEPTH(FEATURE_DEPTH), .ADDR_WIDTH(FEATURE_ADDR_WIDTH)
    ) u_activity_cache (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_clear_error(i_clear_error),
        .i_map_clear_start(i_map_clear_start && !i_abort),
        .o_map_clear_ready(o_map_clear_ready),
        .o_map_clear_busy(o_map_clear_busy),
        .o_map_clear_done(o_map_clear_done),
        .i_lookup_valid(i_cache_enable && state_q == STATE_IDLE &&
                        i_core_read_valid && !o_sticky_error),
        .o_lookup_ready(cache_lookup_ready),
        .i_lookup_bank_flat(i_core_read_bank_flat),
        .i_lookup_addr_flat(i_core_read_addr_flat),
        .i_lookup_lane_mask(i_core_read_lane_mask),
        .o_lookup_result_valid(cache_lookup_result_valid),
        .i_lookup_result_ready(cache_lookup_result_ready),
        .o_lookup_known_mask(cache_known_mask),
        .o_lookup_active_mask(cache_active_mask),
        .i_fill_valid(cache_fill_valid),
        .o_fill_ready(cache_fill_ready),
        .i_fill_bank_flat(request_bank_flat_q),
        .i_fill_addr_flat(request_addr_flat_q),
        .i_fill_lane_mask(request_lane_mask_q),
        .i_fill_active_mask(memory_active_mask),
        .o_busy(cache_busy),
        .o_sticky_error(cache_sticky_error),
        .o_error_code(cache_error_code)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            request_slot_q <= 1'b0;
            request_bank_flat_q <= '0;
            request_addr_flat_q <= '0;
            request_lane_mask_q <= '0;
            response_flat_q <= '0;
            response_lane_mask_q <= '0;
            accepted_cache_enable_q <= 1'b0;
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else if (i_abort) begin
            state_q <= STATE_IDLE;
            request_slot_q <= 1'b0;
            request_bank_flat_q <= '0;
            request_addr_flat_q <= '0;
            request_lane_mask_q <= '0;
            response_flat_q <= '0;
            response_lane_mask_q <= '0;
            accepted_cache_enable_q <= 1'b0;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE && !cache_busy) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (state_q != STATE_IDLE &&
                i_cache_enable != accepted_cache_enable_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_MODE_CHANGE;
            end

            if (core_request_fire && i_cache_enable) begin
                request_slot_q <= i_core_read_slot;
                request_bank_flat_q <= i_core_read_bank_flat;
                request_addr_flat_q <= i_core_read_addr_flat;
                request_lane_mask_q <= i_core_read_lane_mask;
                accepted_cache_enable_q <= i_cache_enable;
                state_q <= STATE_WAIT_LOOKUP;
            end

            if (state_q == STATE_WAIT_LOOKUP &&
                cache_lookup_result_valid) begin
                if (cache_hit) begin
                    if (i_core_result_ready)
                        state_q <= STATE_IDLE;
                end else begin
                    state_q <= STATE_SEND_MEMORY;
                end
            end

            if (state_q == STATE_SEND_MEMORY && memory_request_fire)
                state_q <= STATE_WAIT_MEMORY;

            if (state_q == STATE_WAIT_MEMORY && memory_result_fire) begin
                response_flat_q <= i_memory_result_flat;
                response_lane_mask_q <= i_memory_result_lane_mask;
                if (i_memory_result_lane_mask != request_lane_mask_q) begin
                    local_error_q <= 1'b1;
                    local_error_code_q <= ERROR_RESULT_MASK;
                end
                state_q <= STATE_SEND_CORE;
            end

            if (state_q == STATE_SEND_CORE && i_core_result_ready)
                state_q <= STATE_IDLE;
        end
    end
endmodule

`default_nettype wire
