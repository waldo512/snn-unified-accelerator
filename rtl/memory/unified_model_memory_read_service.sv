`default_nettype none

module unified_model_memory_read_service #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int DEST_WIDTH = 20,
    parameter int MAX_BURST_BEATS = 64,
    parameter int MAX_OUTSTANDING = 4
) (
    input  logic                     i_clk,
    input  logic                     i_rst_n,
    input  logic                     i_abort,
    input  logic [ADDRESS_WIDTH-1:0] i_region_base_word,
    input  logic [ADDRESS_WIDTH-1:0] i_region_limit_word,
    input  logic                     i_req_valid,
    output logic                     o_req_ready,
    input  logic [ADDRESS_WIDTH-1:0] i_req_word_addr,
    input  logic [7:0]               i_req_beats,
    input  logic [DEST_WIDTH-1:0]    i_req_destination,
    input  logic [DEST_WIDTH:0]      i_destination_limit,
    output logic                     o_ddr_cmd_valid,
    input  logic                     i_ddr_cmd_ready,
    output logic [ADDRESS_WIDTH-1:0] o_ddr_cmd_word_addr,
    output logic [7:0]               o_ddr_cmd_beats,
    input  logic                     i_ddr_data_valid,
    output logic                     o_ddr_data_ready,
    input  logic [63:0]              i_ddr_data,
    input  logic                     i_ddr_data_last,
    input  logic                     i_ddr_data_error,
    output logic                     o_fill_valid,
    input  logic                     i_fill_ready,
    output logic [DEST_WIDTH-1:0]    o_fill_addr,
    output logic [63:0]              o_fill_data,
    output logic                     o_fill_last,
    output logic [$clog2(MAX_OUTSTANDING+1)-1:0] o_outstanding,
    output logic                     o_idle,
    output logic                     o_bounds_error,
    output logic                     o_response_protocol_error,
    output logic                     o_ddr_error
);

    localparam int POINTER_WIDTH = MAX_OUTSTANDING <= 1 ? 1 : $clog2(MAX_OUTSTANDING);
    localparam int COUNT_WIDTH = $clog2(MAX_OUTSTANDING + 1);

    logic [DEST_WIDTH-1:0] destination_queue [0:MAX_OUTSTANDING-1];
    logic [7:0] beats_queue [0:MAX_OUTSTANDING-1];
    logic [7:0] received_queue [0:MAX_OUTSTANDING-1];
    logic [POINTER_WIDTH-1:0] head_q;
    logic [POINTER_WIDTH-1:0] tail_q;
    logic [COUNT_WIDTH-1:0] count_q;
    logic [ADDRESS_WIDTH:0] request_end;
    logic [DEST_WIDTH:0] destination_end;
    logic request_bounds_valid;
    logic queue_has_space;
    logic enqueue;
    logic response_handshake;
    logic expected_last;
    logic dequeue;
    logic response_fault;
    logic response_poison_q;
    logic cancel_q;
    logic cancel_dequeue;

    function automatic logic [POINTER_WIDTH-1:0] next_pointer(
        input logic [POINTER_WIDTH-1:0] pointer
    );
        if (pointer == POINTER_WIDTH'(MAX_OUTSTANDING - 1))
            next_pointer = '0;
        else
            next_pointer = pointer + 1'b1;
    endfunction

    assign request_end = {1'b0, i_req_word_addr} +
                         (ADDRESS_WIDTH+1)'(i_req_beats);
    assign destination_end = {1'b0, i_req_destination} +
                             (DEST_WIDTH+1)'(i_req_beats);
    assign request_bounds_valid = i_req_beats != 0 &&
        i_req_beats <= 8'(MAX_BURST_BEATS) &&
        i_req_word_addr >= i_region_base_word &&
        request_end <= {1'b0, i_region_limit_word} &&
        destination_end <= i_destination_limit;
    assign queue_has_space = count_q < COUNT_WIDTH'(MAX_OUTSTANDING);
    assign o_ddr_cmd_valid = i_req_valid && !i_abort && !cancel_q &&
                             !response_poison_q && queue_has_space &&
                             request_bounds_valid;
    assign o_req_ready = !i_abort && !cancel_q && !response_poison_q &&
                         queue_has_space &&
                         (!request_bounds_valid || i_ddr_cmd_ready);
    assign o_ddr_cmd_word_addr = i_req_word_addr;
    assign o_ddr_cmd_beats = i_req_beats;
    assign enqueue = o_ddr_cmd_valid && i_ddr_cmd_ready;

    assign expected_last = count_q != 0 &&
                           received_queue[head_q] + 1'b1 >= beats_queue[head_q];
    assign response_fault = count_q == 0 || i_ddr_data_error ||
                            i_ddr_data_last != expected_last;
    assign o_fill_valid = count_q != 0 && i_ddr_data_valid && !cancel_q &&
                          !response_poison_q && !response_fault;
    assign o_fill_addr = count_q != 0 ?
                         destination_queue[head_q] +
                         DEST_WIDTH'(received_queue[head_q]) : '0;
    assign o_fill_data = i_ddr_data;
    assign o_fill_last = expected_last;
    assign o_ddr_data_ready = cancel_q || response_poison_q ||
                              (i_ddr_data_valid && response_fault) ||
                              (count_q != 0 && i_fill_ready);
    assign response_handshake = i_ddr_data_valid && o_ddr_data_ready;
    assign cancel_dequeue = response_handshake && cancel_q &&
                            count_q != 0 && i_ddr_data_last;
    assign dequeue = cancel_dequeue ||
        (response_handshake && expected_last && !response_fault &&
         !response_poison_q && !cancel_q);
    assign o_outstanding = count_q;
    assign o_idle = count_q == 0 && !o_fill_valid && !o_ddr_cmd_valid &&
                    !i_req_valid && !i_ddr_data_valid &&
                    !response_poison_q && !cancel_q;

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            head_q <= '0;
            tail_q <= '0;
            count_q <= '0;
            o_bounds_error <= 1'b0;
            o_response_protocol_error <= 1'b0;
            o_ddr_error <= 1'b0;
            response_poison_q <= 1'b0;
            cancel_q <= 1'b0;
        end else begin
            if (i_abort)
                cancel_q <= 1'b1;
            else if (cancel_q && count_q == 0 && !i_ddr_data_valid)
                cancel_q <= 1'b0;

            if (i_req_valid && o_req_ready && !request_bounds_valid)
                o_bounds_error <= 1'b1;

            if (enqueue) begin
                destination_queue[tail_q] <= i_req_destination;
                beats_queue[tail_q] <= i_req_beats;
                received_queue[tail_q] <= 8'd0;
                tail_q <= next_pointer(tail_q);
            end

            if (i_ddr_data_valid && count_q == 0 && !cancel_q) begin
                o_response_protocol_error <= 1'b1;
                response_poison_q <= 1'b1;
            end

            if (response_handshake) begin
                if (response_fault && !cancel_q) begin
                    o_response_protocol_error <= 1'b1;
                    response_poison_q <= 1'b1;
                end
                if (i_ddr_data_error && !cancel_q)
                    o_ddr_error <= 1'b1;
                if (dequeue) begin
                    head_q <= next_pointer(head_q);
                end else begin
                    if (!response_fault && !response_poison_q)
                        received_queue[head_q] <= received_queue[head_q] + 1'b1;
                end
            end

            case ({enqueue, dequeue})
                2'b10: count_q <= count_q + 1'b1;
                2'b01: count_q <= count_q - 1'b1;
                default: count_q <= count_q;
            endcase
        end
    end

endmodule

`default_nettype wire
