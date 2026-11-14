`default_nettype none

// Twelve independent banks retain one output-channel wave across scheduler
// input groups. Each bank needs one synchronous read/write port only.
module unified_scnn_wave_partial_sum_banks #(
    parameter int ENGINES = 12,
    parameter int SPATIAL_DEPTH = 32768,
    parameter int SPATIAL_ADDR_WIDTH = $clog2(SPATIAL_DEPTH),
    parameter int OUTPUT_ADDR_WIDTH = 20,
    parameter int ACC_WIDTH = 48
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_valid,
    output logic                                      o_ready,
    input  logic [ENGINES-1:0]                        i_engine_mask,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]             i_spatial_addr,
    input  logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      i_output_addr_flat,
    input  logic [ENGINES*ACC_WIDTH-1:0]              i_partial_flat,
    input  logic                                      i_accumulator_clear,
    input  logic                                      i_accumulator_finalize,
    output logic                                      o_valid,
    input  logic                                      i_ready,
    output logic [ENGINES-1:0]                        o_engine_mask,
    output logic [SPATIAL_ADDR_WIDTH-1:0]             o_spatial_addr,
    output logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0]      o_output_addr_flat,
    output logic [ENGINES*ACC_WIDTH-1:0]              o_accumulator_flat,
    output logic                                      o_idle,
    output logic                                      o_sticky_error
);
    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_READ,
        STATE_ACCUMULATE,
        STATE_OUTPUT
    } state_t;

    (* ram_style = "block" *) logic signed [ACC_WIDTH-1:0]
        partial_memory [0:ENGINES-1][0:SPATIAL_DEPTH-1];
    logic signed [ACC_WIDTH-1:0] read_data_q [0:ENGINES-1];
    logic [ENGINES-1:0] engine_mask_q;
    logic [SPATIAL_ADDR_WIDTH-1:0] spatial_addr_q;
    logic [ENGINES*OUTPUT_ADDR_WIDTH-1:0] output_addr_flat_q;
    logic [ENGINES*ACC_WIDTH-1:0] partial_flat_q;
    logic finalize_q;
    state_t state_q;
    logic request_legal;

    assign request_legal = i_engine_mask != '0 &&
        {1'b0, i_spatial_addr} <
            (SPATIAL_ADDR_WIDTH+1)'(SPATIAL_DEPTH);
    assign o_ready = state_q == STATE_IDLE && !o_sticky_error;
    assign o_valid = state_q == STATE_OUTPUT;
    assign o_engine_mask = engine_mask_q;
    assign o_spatial_addr = spatial_addr_q;
    assign o_output_addr_flat = output_addr_flat_q;
    assign o_idle = state_q == STATE_IDLE;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            engine_mask_q <= '0;
            spatial_addr_q <= '0;
            output_addr_flat_q <= '0;
            partial_flat_q <= '0;
            o_accumulator_flat <= '0;
            finalize_q <= 1'b0;
            state_q <= STATE_IDLE;
            o_sticky_error <= 1'b0;
            for (integer engine = 0; engine < ENGINES; engine++)
                read_data_q[engine] <= '0;
        end else begin
            if (i_clear_error && state_q == STATE_IDLE)
                o_sticky_error <= 1'b0;

            case (state_q)
                STATE_IDLE: begin
                    if (i_valid && !request_legal) begin
                        o_sticky_error <= 1'b1;
                    end else if (i_valid && o_ready) begin
                        engine_mask_q <= i_engine_mask;
                        spatial_addr_q <= i_spatial_addr;
                        output_addr_flat_q <= i_output_addr_flat;
                        partial_flat_q <= i_partial_flat;
                        finalize_q <= i_accumulator_finalize;
                        if (i_accumulator_finalize)
                            o_accumulator_flat <= '0;
                        if (i_accumulator_clear) begin
                            if (i_accumulator_finalize) begin
                                o_accumulator_flat <= i_partial_flat;
                                state_q <= STATE_OUTPUT;
                            end else begin
                                for (integer engine = 0; engine < ENGINES;
                                     engine++) begin
                                    if (i_engine_mask[engine])
                                        partial_memory[engine][i_spatial_addr] <=
                                            i_partial_flat[
                                                engine*ACC_WIDTH +: ACC_WIDTH];
                                end
                            end
                        end else begin
                            state_q <= STATE_READ;
                        end
                    end
                end
                STATE_READ: begin
                    for (integer engine = 0; engine < ENGINES; engine++) begin
                        if (engine_mask_q[engine])
                            read_data_q[engine] <=
                                partial_memory[engine][spatial_addr_q];
                    end
                    state_q <= STATE_ACCUMULATE;
                end
                STATE_ACCUMULATE: begin
                    for (integer engine = 0; engine < ENGINES; engine++) begin
                        if (engine_mask_q[engine]) begin
                            if (finalize_q)
                                o_accumulator_flat[
                                    engine*ACC_WIDTH +: ACC_WIDTH] <=
                                    read_data_q[engine] + $signed(partial_flat_q[
                                        engine*ACC_WIDTH +: ACC_WIDTH]);
                            else
                                partial_memory[engine][spatial_addr_q] <=
                                    read_data_q[engine] + $signed(partial_flat_q[
                                        engine*ACC_WIDTH +: ACC_WIDTH]);
                        end
                    end
                    if (finalize_q)
                        state_q <= STATE_OUTPUT;
                    else
                        state_q <= STATE_IDLE;
                end
                STATE_OUTPUT: begin
                    if (i_ready) begin
                        engine_mask_q <= '0;
                        state_q <= STATE_IDLE;
                    end
                end
                default: state_q <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
