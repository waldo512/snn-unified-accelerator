`timescale 1ns/1ps
`default_nettype none

// Shared output-layer gradient primitive. It accumulates fixed-point
// error*source_trace contributions by tensor-local address and later emits the
// sign of each nonzero gradient in ascending address order.
module unified_tp_outer_product_sign_accumulator #(
    parameter int TENSOR_ID_WIDTH = 8,
    parameter int PARAM_ADDR_WIDTH = 24,
    parameter int SOURCE_WIDTH = 16,
    parameter int ERROR_WIDTH = 18,
    parameter int ACC_WIDTH = 48,
    parameter int MAX_ELEMENTS = 32768
) (
    input  wire logic                               i_clk,
    input  wire logic                               i_rst_n,

    input  wire logic                               i_cfg_valid,
    output logic                               o_cfg_ready,
    input  wire logic [TENSOR_ID_WIDTH-1:0]          i_cfg_tensor_id,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_cfg_element_count,
    output logic                               o_cfg_error,
    output logic                               o_active,
    output logic [TENSOR_ID_WIDTH-1:0]          o_active_tensor_id,
    input  wire logic                               i_release,
    output logic                               o_release_ready,

    input  wire logic                               i_contribution_valid,
    output logic                               o_contribution_ready,
    input  wire logic [PARAM_ADDR_WIDTH-1:0]         i_contribution_local_addr,
    input  wire logic [SOURCE_WIDTH-1:0]             i_contribution_source_trace,
    input  wire logic signed [ERROR_WIDTH-1:0]       i_contribution_error,
    output logic                               o_contribution_error,
    output logic                               o_accumulator_overflow,

    input  wire logic                               i_emit_start,
    output logic                               o_emit_ready,
    output logic                               o_sign_valid,
    input  wire logic                               i_sign_ready,
    output logic [TENSOR_ID_WIDTH-1:0]          o_sign_tensor_id,
    output logic [PARAM_ADDR_WIDTH-1:0]         o_sign_local_addr,
    output logic signed [1:0]                  o_gradient_sign,
    output logic signed [ACC_WIDTH-1:0]        o_gradient_accumulator,
    output logic                               o_emit_done,
    output logic                               o_busy
);

    localparam int MEM_ADDR_WIDTH = (MAX_ELEMENTS <= 2) ? 1 : $clog2(MAX_ELEMENTS);
    localparam int PRODUCT_WIDTH = SOURCE_WIDTH + 1 + ERROR_WIDTH;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_CLEAR,
        STATE_ACCUMULATE,
        STATE_ACCUMULATE_WRITE,
        STATE_SCAN_READ,
        STATE_SCAN_CHECK,
        STATE_SIGN_HOLD
    } state_t;

    state_t state_q;
    (* ram_style = "block" *) logic signed [ACC_WIDTH-1:0]
        gradient_mem [0:MAX_ELEMENTS-1];
    logic [TENSOR_ID_WIDTH-1:0] tensor_id_q;
    logic [PARAM_ADDR_WIDTH-1:0] element_count_q;
    logic [PARAM_ADDR_WIDTH-1:0] clear_addr_q;
    logic [MEM_ADDR_WIDTH-1:0] pending_addr_q;
    logic signed [PRODUCT_WIDTH-1:0] pending_product_q;
    logic [PARAM_ADDR_WIDTH-1:0] scan_addr_q;
    logic signed [ACC_WIDTH-1:0] memory_read_q;
    logic memory_read_enable;
    logic [MEM_ADDR_WIDTH-1:0] memory_read_addr;
    logic memory_write_enable;
    logic [MEM_ADDR_WIDTH-1:0] memory_write_addr;
    logic signed [ACC_WIDTH-1:0] memory_write_data;

    logic contribution_fire;
    logic emit_fire;
    logic sign_fire;
    logic scan_last;
    logic signed [ACC_WIDTH:0] sum_extended;
    logic signed [ACC_WIDTH:0] accumulator_max_extended;
    logic signed [ACC_WIDTH:0] accumulator_min_extended;
    logic signed [ACC_WIDTH-1:0] accumulated_value;
    logic accumulated_overflow;

    always_comb begin
        o_cfg_ready = state_q == STATE_IDLE;
        o_release_ready = (state_q == STATE_ACCUMULATE) && o_active;
        o_contribution_ready = (state_q == STATE_ACCUMULATE) && o_active &&
            !i_emit_start && !i_release;
        contribution_fire = i_contribution_valid && o_contribution_ready;
        o_emit_ready = (state_q == STATE_ACCUMULATE) && o_active && !i_release;
        emit_fire = i_emit_start && o_emit_ready;
        o_sign_valid = state_q == STATE_SIGN_HOLD;
        sign_fire = o_sign_valid && i_sign_ready;
        o_sign_tensor_id = tensor_id_q;
        o_sign_local_addr = scan_addr_q;
        o_gradient_sign = (memory_read_q > 0) ? 2'sd1 : -2'sd1;
        o_gradient_accumulator = memory_read_q;
        o_active_tensor_id = tensor_id_q;
        o_busy = (state_q != STATE_IDLE) && (state_q != STATE_ACCUMULATE);
        scan_last = (scan_addr_q + {{(PARAM_ADDR_WIDTH-1){1'b0}}, 1'b1}) >=
            element_count_q;

        sum_extended = $signed({memory_read_q[ACC_WIDTH-1], memory_read_q}) +
            $signed({{(ACC_WIDTH + 1 - PRODUCT_WIDTH){pending_product_q[PRODUCT_WIDTH-1]}},
                     pending_product_q});
        accumulator_max_extended = $signed({2'b00, {(ACC_WIDTH-1){1'b1}}});
        accumulator_min_extended = $signed({2'b11, {(ACC_WIDTH-1){1'b0}}});
        accumulated_overflow = 1'b0;
        if (sum_extended > accumulator_max_extended) begin
            accumulated_value = {1'b0, {(ACC_WIDTH-1){1'b1}}};
            accumulated_overflow = 1'b1;
        end else if (sum_extended < accumulator_min_extended) begin
            accumulated_value = {1'b1, {(ACC_WIDTH-1){1'b0}}};
            accumulated_overflow = 1'b1;
        end else begin
            accumulated_value = sum_extended[ACC_WIDTH-1:0];
        end

        memory_read_enable = 1'b0;
        memory_read_addr = '0;
        if ((state_q == STATE_ACCUMULATE) && contribution_fire &&
            (i_contribution_local_addr < element_count_q)) begin
            memory_read_enable = 1'b1;
            memory_read_addr =
                i_contribution_local_addr[MEM_ADDR_WIDTH-1:0];
        end else if (state_q == STATE_SCAN_READ) begin
            memory_read_enable = 1'b1;
            memory_read_addr = scan_addr_q[MEM_ADDR_WIDTH-1:0];
        end

        memory_write_enable = 1'b0;
        memory_write_addr = '0;
        memory_write_data = '0;
        unique case (state_q)
            STATE_CLEAR: begin
                memory_write_enable = 1'b1;
                memory_write_addr = clear_addr_q[MEM_ADDR_WIDTH-1:0];
            end
            STATE_ACCUMULATE_WRITE: begin
                memory_write_enable = 1'b1;
                memory_write_addr = pending_addr_q;
                memory_write_data = accumulated_value;
            end
            STATE_SCAN_CHECK: begin
                if (memory_read_q == 0) begin
                    memory_write_enable = 1'b1;
                    memory_write_addr = scan_addr_q[MEM_ADDR_WIDTH-1:0];
                end
            end
            STATE_SIGN_HOLD: begin
                if (sign_fire) begin
                    memory_write_enable = 1'b1;
                    memory_write_addr = scan_addr_q[MEM_ADDR_WIDTH-1:0];
                end
            end
            default: begin
            end
        endcase
    end

    // The payload RAM has one synchronous port. State controls when its
    // unreset contents are observable, so reset never expands it into flops.
    always_ff @(posedge i_clk) begin
        if (memory_write_enable)
            gradient_mem[memory_write_addr] <= memory_write_data;
        if (memory_read_enable)
            memory_read_q <= gradient_mem[memory_read_addr];
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            tensor_id_q <= '0;
            element_count_q <= '0;
            clear_addr_q <= '0;
            pending_addr_q <= '0;
            pending_product_q <= '0;
            scan_addr_q <= '0;
            o_active <= 1'b0;
            o_cfg_error <= 1'b0;
            o_contribution_error <= 1'b0;
            o_accumulator_overflow <= 1'b0;
            o_emit_done <= 1'b0;
        end else begin
            o_cfg_error <= 1'b0;
            o_contribution_error <= 1'b0;
            o_accumulator_overflow <= 1'b0;
            o_emit_done <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (i_cfg_valid) begin
            if ((i_cfg_element_count == 0) ||
                ({1'b0, i_cfg_element_count} >
                 (PARAM_ADDR_WIDTH+1)'(MAX_ELEMENTS))) begin
                            o_cfg_error <= 1'b1;
                        end else begin
                            tensor_id_q <= i_cfg_tensor_id;
                            element_count_q <= i_cfg_element_count;
                            clear_addr_q <= '0;
                            o_active <= 1'b0;
                            state_q <= STATE_CLEAR;
                        end
                    end
                end

                STATE_CLEAR: begin
                    if ((clear_addr_q + 1'b1) >= element_count_q) begin
                        clear_addr_q <= '0;
                        o_active <= 1'b1;
                        state_q <= STATE_ACCUMULATE;
                    end else begin
                        clear_addr_q <= clear_addr_q + 1'b1;
                    end
                end

                STATE_ACCUMULATE: begin
                    if (i_release && o_release_ready) begin
                        o_active <= 1'b0;
                        tensor_id_q <= '0;
                        element_count_q <= '0;
                        state_q <= STATE_IDLE;
                    end else if (emit_fire) begin
                        scan_addr_q <= '0;
                        state_q <= STATE_SCAN_READ;
                    end else if (contribution_fire) begin
                        if (i_contribution_local_addr >= element_count_q) begin
                            o_contribution_error <= 1'b1;
                        end else begin
                            pending_addr_q <=
                                i_contribution_local_addr[MEM_ADDR_WIDTH-1:0];
                            pending_product_q <=
                                $signed({1'b0, i_contribution_source_trace}) *
                                $signed(i_contribution_error);
                            state_q <= STATE_ACCUMULATE_WRITE;
                        end
                    end
                end

                STATE_ACCUMULATE_WRITE: begin
                    o_accumulator_overflow <= accumulated_overflow;
                    state_q <= STATE_ACCUMULATE;
                end

                STATE_SCAN_READ: begin
                    state_q <= STATE_SCAN_CHECK;
                end

                STATE_SCAN_CHECK: begin
                    if (memory_read_q != 0) begin
                        state_q <= STATE_SIGN_HOLD;
                    end else begin
                        if (scan_last) begin
                            o_emit_done <= 1'b1;
                            state_q <= STATE_ACCUMULATE;
                        end else begin
                            scan_addr_q <= scan_addr_q + 1'b1;
                            state_q <= STATE_SCAN_READ;
                        end
                    end
                end

                STATE_SIGN_HOLD: begin
                    if (sign_fire) begin
                        if (scan_last) begin
                            o_emit_done <= 1'b1;
                            state_q <= STATE_ACCUMULATE;
                        end else begin
                            scan_addr_q <= scan_addr_q + 1'b1;
                            state_q <= STATE_SCAN_READ;
                        end
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
