`timescale 1ns/1ps
`default_nettype none

// Expands one FC_COMPAT K8 event row into commands for the shared eight-lane
// INT8 engine. All H40 neuron state remains resident; only synaptic products
// are scheduled over the five eight-neuron destination words.
module unified_fc_compat_row_to_shared_k8_v1 #(
    parameter integer H_COUNT = 40,
    parameter integer SOURCE_LANES = 8,
    parameter integer K8_LANES = 8,
    parameter integer W_BITS = 5,
    parameter integer INPUT_SHIFT = 6,
    parameter integer METADATA_DEPTH = 4
) (
    input  logic                                             clk,
    input  logic                                             rst_n,
    input  logic                                             i_abort,
    input  logic                                             i_clear_error,

    input  logic                                             i_row_valid,
    output logic                                             o_row_ready,
    input  logic [SOURCE_LANES-1:0]                          i_source_valid,
    input  logic [SOURCE_LANES*H_COUNT*W_BITS-1:0]           i_weight_rows_flat,
    output logic                                             o_row_done,

    output logic                                             o_k8_valid,
    input  logic                                             i_k8_ready,
    output logic [K8_LANES-1:0]                              o_k8_lane_mask,
    output logic [K8_LANES*8-1:0]                            o_k8_lhs_flat,
    output logic [K8_LANES*8-1:0]                            o_k8_rhs_flat,
    output logic                                             o_k8_first,
    output logic                                             o_k8_last,
    output logic [9:0]                                       o_k8_destination_base,

    input  logic                                             i_k8_retire,
    output logic                                             o_retire_metadata_valid,
    output logic [K8_LANES-1:0]                              o_retire_lane_mask,
    output logic [9:0]                                       o_retire_destination_base,

    output logic                                             o_busy,
    output logic                                             o_protocol_error
);
    localparam integer DEST_WORDS = (H_COUNT + K8_LANES - 1) / K8_LANES;
    localparam integer SOURCE_INDEX_WIDTH =
        (SOURCE_LANES <= 1) ? 1 : $clog2(SOURCE_LANES);
    localparam integer WORD_INDEX_WIDTH =
        (DEST_WORDS <= 1) ? 1 : $clog2(DEST_WORDS);
    localparam integer METADATA_PTR_WIDTH =
        (METADATA_DEPTH <= 1) ? 1 : $clog2(METADATA_DEPTH);
    localparam integer METADATA_COUNT_WIDTH = $clog2(METADATA_DEPTH + 1);
    localparam logic signed [7:0] FC_ACTIVATION = 8'sd1 <<< INPUT_SHIFT;

    logic active_q;
    logic issue_complete_q;
    logic [SOURCE_LANES-1:0] source_valid_q;
    logic [SOURCE_LANES*H_COUNT*W_BITS-1:0] weight_rows_q;
    logic [SOURCE_INDEX_WIDTH-1:0] source_index_q;
    logic [WORD_INDEX_WIDTH-1:0] destination_word_q;

    logic [K8_LANES-1:0] metadata_mask_q [0:METADATA_DEPTH-1];
    logic [9:0] metadata_base_q [0:METADATA_DEPTH-1];
    logic [METADATA_PTR_WIDTH-1:0] metadata_read_ptr_q;
    logic [METADATA_PTR_WIDTH-1:0] metadata_write_ptr_q;
    logic [METADATA_COUNT_WIDTH-1:0] metadata_count_q;

    logic current_source_valid;
    logic metadata_full;
    logic row_accept;
    logic k8_accept;

    function automatic logic [METADATA_PTR_WIDTH-1:0] next_metadata_ptr(
        input logic [METADATA_PTR_WIDTH-1:0] current
    );
        begin
            if (current == METADATA_PTR_WIDTH'(METADATA_DEPTH-1))
                next_metadata_ptr = '0;
            else
                next_metadata_ptr = current + 1'b1;
        end
    endfunction

    always_comb begin
        current_source_valid = source_valid_q[source_index_q];
        metadata_full = metadata_count_q == METADATA_COUNT_WIDTH'(METADATA_DEPTH);
        o_row_ready = !active_q && metadata_count_q == '0;
        row_accept = i_row_valid && o_row_ready;

        o_k8_valid = active_q && !issue_complete_q &&
            current_source_valid && !metadata_full;
        k8_accept = o_k8_valid && i_k8_ready;
        o_k8_lane_mask = '0;
        o_k8_lhs_flat = '0;
        o_k8_rhs_flat = '0;
        o_k8_destination_base = 10'(destination_word_q * K8_LANES);
        for (int lane = 0; lane < K8_LANES; lane++) begin
            int hidden_index;
            logic signed [W_BITS-1:0] packed_weight;
            hidden_index = destination_word_q * K8_LANES + lane;
            packed_weight = '0;
            if (hidden_index < H_COUNT) begin
                packed_weight = $signed(weight_rows_q[
                    source_index_q*H_COUNT*W_BITS + hidden_index*W_BITS +:
                    W_BITS]);
                o_k8_lane_mask[lane] = 1'b1;
                o_k8_lhs_flat[lane*8 +: 8] = FC_ACTIVATION;
                o_k8_rhs_flat[lane*8 +: 8] =
                    {{(8-W_BITS){packed_weight[W_BITS-1]}}, packed_weight};
            end
        end
        o_k8_first = 1'b1;
        o_k8_last = 1'b1;

        o_retire_metadata_valid = metadata_count_q != '0;
        o_retire_lane_mask = metadata_mask_q[metadata_read_ptr_q];
        o_retire_destination_base = metadata_base_q[metadata_read_ptr_q];
        o_busy = active_q || metadata_count_q != '0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q <= 1'b0;
            issue_complete_q <= 1'b0;
            source_valid_q <= '0;
            weight_rows_q <= '0;
            source_index_q <= '0;
            destination_word_q <= '0;
            metadata_read_ptr_q <= '0;
            metadata_write_ptr_q <= '0;
            metadata_count_q <= '0;
            o_row_done <= 1'b0;
            o_protocol_error <= 1'b0;
        end else if (i_abort) begin
            active_q <= 1'b0;
            issue_complete_q <= 1'b0;
            source_valid_q <= '0;
            source_index_q <= '0;
            destination_word_q <= '0;
            metadata_read_ptr_q <= '0;
            metadata_write_ptr_q <= '0;
            metadata_count_q <= '0;
            o_row_done <= 1'b0;
        end else begin
            o_row_done <= 1'b0;
            if (i_clear_error && !o_busy)
                o_protocol_error <= 1'b0;

            if (row_accept) begin
                active_q <= 1'b1;
                issue_complete_q <= 1'b0;
                source_valid_q <= i_source_valid;
                weight_rows_q <= i_weight_rows_flat;
                source_index_q <= '0;
                destination_word_q <= '0;
            end

            if (active_q && !issue_complete_q && !current_source_valid) begin
                destination_word_q <= '0;
                if (source_index_q == SOURCE_INDEX_WIDTH'(SOURCE_LANES-1))
                    issue_complete_q <= 1'b1;
                else
                    source_index_q <= source_index_q + 1'b1;
            end else if (k8_accept) begin
                if (destination_word_q == WORD_INDEX_WIDTH'(DEST_WORDS-1)) begin
                    destination_word_q <= '0;
                    if (source_index_q == SOURCE_INDEX_WIDTH'(SOURCE_LANES-1))
                        issue_complete_q <= 1'b1;
                    else
                        source_index_q <= source_index_q + 1'b1;
                end else begin
                    destination_word_q <= destination_word_q + 1'b1;
                end
            end

            unique case ({k8_accept, i_k8_retire})
                2'b10: begin
                    metadata_count_q <= metadata_count_q + 1'b1;
                    metadata_write_ptr_q <=
                        next_metadata_ptr(metadata_write_ptr_q);
                end
                2'b01: begin
                    if (metadata_count_q == '0) begin
                        o_protocol_error <= 1'b1;
                    end else begin
                        metadata_count_q <= metadata_count_q - 1'b1;
                        metadata_read_ptr_q <=
                            next_metadata_ptr(metadata_read_ptr_q);
                    end
                end
                2'b11: begin
                    if (metadata_count_q == '0) begin
                        o_protocol_error <= 1'b1;
                        metadata_count_q <= metadata_count_q + 1'b1;
                        metadata_write_ptr_q <=
                            next_metadata_ptr(metadata_write_ptr_q);
                    end else begin
                        metadata_read_ptr_q <=
                            next_metadata_ptr(metadata_read_ptr_q);
                        metadata_write_ptr_q <=
                            next_metadata_ptr(metadata_write_ptr_q);
                    end
                end
                default: begin
                end
            endcase

            if (k8_accept) begin
                metadata_mask_q[metadata_write_ptr_q] <= o_k8_lane_mask;
                metadata_base_q[metadata_write_ptr_q] <=
                    o_k8_destination_base;
            end

            if (active_q && issue_complete_q && metadata_count_q == '0) begin
                active_q <= 1'b0;
                issue_complete_q <= 1'b0;
                o_row_done <= 1'b1;
            end

            if (i_k8_retire && metadata_count_q == '0)
                o_protocol_error <= 1'b1;
            if (i_row_valid && o_row_ready && i_source_valid == '0) begin
                // An empty sparse row is legal and completes without issuing.
            end
        end
    end

    initial begin
        if (SOURCE_LANES < 1 || K8_LANES != 8 || H_COUNT < 1 ||
            W_BITS < 2 || W_BITS > 8 || INPUT_SHIFT > 6 ||
            METADATA_DEPTH < 2)
            $error("FC-to-shared-K8 bridge parameters are invalid");
    end
endmodule

`default_nettype wire
