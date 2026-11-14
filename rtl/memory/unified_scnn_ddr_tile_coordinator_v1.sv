`default_nettype none

// Binds one prefetched DDR weight descriptor to one or more contiguous compute
// subwaves. Absolute Q64 addresses from the compute frontend are relocated into
// the packed ping-pong scratchpad address space. This leaf owns no compute or
// memory.
module unified_scnn_ddr_tile_coordinator_v1 #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int BANK_WORDS = 8640,
    parameter int TAG_WIDTH = 16,
    parameter int MAX_SEGMENTS = 12
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_clear_error,
    input  logic                         i_abort,

    // Descriptor held by unified_scnn_ddr3a_core_service_v1 until the
    // coordinator returns o_service_compute_tile_done.
    input  logic                         i_service_compute_tile_valid,
    output logic                         o_service_compute_tile_done,
    input  logic [3:0]                   i_service_layer_index,
    input  logic [9:0]                   i_service_output_channel_base,
    input  logic [9:0]                   i_service_output_channel_count,
    input  logic [9:0]                   i_service_input_channel_base,
    input  logic [9:0]                   i_service_input_channel_count,
    input  logic [ADDRESS_WIDTH-1:0]     i_service_source_q64_word,
    input  logic [$clog2(BANK_WORDS+1)-1:0]
                                                i_service_total_q64_words,
    input  logic [$clog2(BANK_WORDS+1)-1:0]
                                                i_service_segment_q64_words,
    input  logic [ADDRESS_WIDTH-1:0]
                                                i_service_source_stride_q64_words,
    input  logic [TAG_WIDTH-1:0]         i_service_tile_tag,

    // Compute-wave request. Layer, input range, source layout and tag must
    // exactly match the resident descriptor. The output range may be one
    // contiguous, Q64-aligned subrange of that descriptor.
    input  logic                         i_wave_start_valid,
    output logic                         o_wave_start_ready,
    input  logic [3:0]                   i_wave_layer_index,
    input  logic [9:0]                   i_wave_output_channel_base,
    input  logic [9:0]                   i_wave_output_channel_count,
    input  logic [9:0]                   i_wave_input_channel_base,
    input  logic [9:0]                   i_wave_input_channel_count,
    input  logic [ADDRESS_WIDTH-1:0]     i_wave_source_q64_word,
    input  logic [$clog2(BANK_WORDS+1)-1:0]
                                                i_wave_total_q64_words,
    input  logic [$clog2(BANK_WORDS+1)-1:0]
                                                i_wave_segment_q64_words,
    input  logic [ADDRESS_WIDTH-1:0]
                                                i_wave_source_stride_q64_words,
    input  logic [TAG_WIDTH-1:0]         i_wave_tile_tag,
    output logic                         o_compute_start_valid,
    input  logic                         i_compute_start_ready,
    input  logic                         i_wave_done,
    input  logic [TAG_WIDTH-1:0]         i_wave_done_tag,

    // Absolute Q64 read pair from CONV or OUTPUT.
    input  logic                         i_absolute_read_valid,
    output logic                         o_absolute_read_ready,
    input  logic [ADDRESS_WIDTH-1:0]     i_absolute_q64_addr0,
    input  logic [ADDRESS_WIDTH-1:0]     i_absolute_q64_addr1,
    output logic                         o_absolute_response_valid,
    input  logic                         i_absolute_response_ready,
    output logic [63:0]                  o_absolute_data0,
    output logic [63:0]                  o_absolute_data1,

    // Packed bank-local read pair returned to the DDR scratchpad service.
    output logic                         o_service_read_valid,
    input  logic                         i_service_read_ready,
    output logic [$clog2(BANK_WORDS)-1:0] o_service_read_word_addr0,
    output logic [$clog2(BANK_WORDS)-1:0] o_service_read_word_addr1,
    input  logic                         i_service_response_valid,
    output logic                         o_service_response_ready,
    input  logic [63:0]                  i_service_read_data0,
    input  logic [63:0]                  i_service_read_data1,
    input  logic                         i_service_sticky_error,
    input  logic [3:0]                   i_service_error_code,

    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_descriptor_active,
    output logic                         o_wave_active,
    output logic                         o_metadata_match,
    output logic                         o_read_pending,
    output logic                         o_response_pending,
    output logic                         o_abort_request,
    output logic                         o_sticky_error,
    output logic [3:0]                   o_error_code
);

    localparam int COUNT_WIDTH = $clog2(BANK_WORDS + 1);
    localparam int LOCAL_ADDR_WIDTH = $clog2(BANK_WORDS);
    localparam int PREP_INDEX_WIDTH = $clog2(MAX_SEGMENTS + 1);
    localparam int Q16_COUNT_WIDTH = COUNT_WIDTH + 2;
    localparam int DIV_STEP_WIDTH = $clog2(Q16_COUNT_WIDTH + 1);

    localparam logic [3:0] ERROR_NONE = 4'h0;
    localparam logic [3:0] ERROR_DESCRIPTOR = 4'h1;
    localparam logic [3:0] ERROR_METADATA = 4'h2;
    localparam logic [3:0] ERROR_START_PROTOCOL = 4'h3;
    localparam logic [3:0] ERROR_ADDRESS = 4'h4;
    localparam logic [3:0] ERROR_SERVICE_PROTOCOL = 4'h5;
    localparam logic [3:0] ERROR_DONE_PROTOCOL = 4'h6;
    localparam logic [3:0] ERROR_ABORT = 4'h7;
    localparam logic [3:0] ERROR_SERVICE = 4'h8;
    localparam logic [3:0] ERROR_DESCRIPTOR_DROP = 4'h9;

    typedef struct packed {
        logic valid;
        logic [LOCAL_ADDR_WIDTH-1:0] local_addr;
    } relocation_t;

    logic descriptor_active_q;
    logic descriptor_preparing_q;
    logic descriptor_wait_drop_q;
    logic wave_active_q;
    logic wave_done_seen_q;
    logic command_pending_q;
    logic read_outstanding_q;
    logic response_valid_q;
    logic sticky_error_q;
    logic [3:0] error_code_q;

    logic [3:0] descriptor_layer_q;
    logic [9:0] descriptor_output_base_q;
    logic [9:0] descriptor_output_count_q;
    logic [9:0] descriptor_input_base_q;
    logic [9:0] descriptor_input_count_q;
    logic [ADDRESS_WIDTH-1:0] descriptor_source_q;
    logic [COUNT_WIDTH-1:0] descriptor_total_q;
    logic [COUNT_WIDTH-1:0] descriptor_segment_q;
    logic [ADDRESS_WIDTH-1:0] descriptor_stride_q;
    logic [COUNT_WIDTH-1:0] descriptor_segment_count_q;
    logic [TAG_WIDTH-1:0] descriptor_tag_q;
    logic [9:0] descriptor_next_output_q;
    logic [Q16_COUNT_WIDTH-1:0] descriptor_q16_per_output_q;
    logic descriptor_q16_layout_exact_q;
    logic segment_prepare_active_q;
    logic [PREP_INDEX_WIDTH-1:0] prepare_index_q;
    logic [ADDRESS_WIDTH-1:0] prepare_source_q;
    logic [ADDRESS_WIDTH-1:0] segment_source_q [0:MAX_SEGMENTS-1];
    logic flat_divide_active_q;
    logic [Q16_COUNT_WIDTH-1:0] divide_dividend_q;
    logic [Q16_COUNT_WIDTH-1:0] divide_quotient_q;
    logic [Q16_COUNT_WIDTH:0] divide_remainder_q;
    logic [9:0] divide_divisor_q;
    logic [DIV_STEP_WIDTH-1:0] divide_steps_q;

    logic [LOCAL_ADDR_WIDTH-1:0] command_addr0_q;
    logic [LOCAL_ADDR_WIDTH-1:0] command_addr1_q;
    logic [63:0] response_data0_q;
    logic [63:0] response_data1_q;
    logic [LOCAL_ADDR_WIDTH:0] wave_local_base_q;
    logic [LOCAL_ADDR_WIDTH:0] wave_local_limit_q;
    logic [9:0] wave_next_output_q;
    logic wave_finishes_descriptor_q;

    logic descriptor_shape_valid;
    logic [63:0] descriptor_segment_count_wide;
    logic [63:0] descriptor_last_exclusive;
    logic [10:0] service_output_end_wide;
    logic [10:0] service_input_end_wide;
    logic [10:0] descriptor_output_end_wide;
    logic [10:0] wave_output_end_wide;
    logic [10:0] wave_output_offset_wide;
    logic [63:0] wave_local_q16_base_wide;
    logic [63:0] wave_local_q16_limit_wide;
    logic [63:0] wave_local_q64_base_wide;
    logic [63:0] wave_local_q64_limit_wide;
    logic wave_output_range_contained;
    logic wave_output_range_contiguous;
    logic wave_output_layout_valid;
    logic metadata_match;
    logic compute_start_fire;
    logic absolute_read_fire;
    logic service_read_fire;
    logic service_response_fire;
    logic absolute_response_fire;
    relocation_t relocation0;
    relocation_t relocation1;
    logic [Q16_COUNT_WIDTH:0] divider_shifted_remainder;
    logic [Q16_COUNT_WIDTH:0] divider_remainder_next;
    logic [Q16_COUNT_WIDTH-1:0] divider_quotient_next;

    function automatic relocation_t relocate_address(
        input logic [ADDRESS_WIDTH-1:0] absolute_addr
    );
        relocation_t result;
        logic [ADDRESS_WIDTH:0] address_wide;
        logic [ADDRESS_WIDTH:0] segment_begin_wide;
        logic [ADDRESS_WIDTH:0] segment_end_wide;
        logic [ADDRESS_WIDTH:0] segment_offset_wide;
        logic [LOCAL_ADDR_WIDTH:0] local_wide;
        begin
            result = '0;
            address_wide = {1'b0, absolute_addr};
            segment_begin_wide = '0;
            segment_end_wide = '0;
            segment_offset_wide = '0;
            local_wide = '0;
            for (integer segment_index = 0;
                 segment_index < MAX_SEGMENTS; segment_index++) begin
                segment_begin_wide = {
                    1'b0, segment_source_q[segment_index]};
                segment_end_wide = segment_begin_wide +
                    (ADDRESS_WIDTH+1)'(descriptor_segment_q);
                if (!result.valid && descriptor_active_q &&
                    segment_index < descriptor_segment_count_q &&
                    address_wide >= segment_begin_wide &&
                    address_wide < segment_end_wide) begin
                    segment_offset_wide = address_wide -
                        segment_begin_wide;
                    local_wide = (LOCAL_ADDR_WIDTH+1)'(segment_index) *
                        (LOCAL_ADDR_WIDTH+1)'(descriptor_segment_q) +
                        (LOCAL_ADDR_WIDTH+1)'(segment_offset_wide);
                    if (local_wide < descriptor_total_q &&
                        local_wide < BANK_WORDS &&
                        local_wide >= wave_local_base_q &&
                        local_wide < wave_local_limit_q) begin
                        result.valid = 1'b1;
                        result.local_addr = LOCAL_ADDR_WIDTH'(local_wide);
                    end
                end
            end
            return result;
        end
    endfunction

    always_comb begin
        divider_shifted_remainder = {
            divide_remainder_q[Q16_COUNT_WIDTH-1:0],
            divide_dividend_q[Q16_COUNT_WIDTH-1]
        };
        divider_remainder_next = divider_shifted_remainder;
        divider_quotient_next = divide_quotient_q << 1;
        if (divider_shifted_remainder >=
            (Q16_COUNT_WIDTH+1)'(divide_divisor_q)) begin
            divider_remainder_next = divider_shifted_remainder -
                (Q16_COUNT_WIDTH+1)'(divide_divisor_q);
            divider_quotient_next[0] = 1'b1;
        end
    end

    always_comb begin
        descriptor_segment_count_wide = 64'd0;
        descriptor_last_exclusive = 64'd0;
        service_output_end_wide = {1'b0, i_service_output_channel_base} +
                                  {1'b0, i_service_output_channel_count};
        service_input_end_wide = {1'b0, i_service_input_channel_base} +
                                 {1'b0, i_service_input_channel_count};
        descriptor_shape_valid = 1'b0;
        if (i_service_segment_q64_words != 0 &&
            i_service_total_q64_words != 0) begin
            descriptor_segment_count_wide =
                i_service_segment_q64_words == i_service_total_q64_words ?
                    64'd1 : 64'(i_service_output_channel_count);
            if (descriptor_segment_count_wide != 0) begin
                descriptor_last_exclusive =
                    64'(i_service_source_q64_word) +
                    (descriptor_segment_count_wide - 1'b1) *
                    64'(i_service_source_stride_q64_words) +
                    64'(i_service_segment_q64_words);
            end
        end
        descriptor_shape_valid =
            i_service_output_channel_count != 0 &&
            i_service_input_channel_count != 0 &&
            service_output_end_wide <= 11'd1024 &&
            service_input_end_wide <= 11'd1024 &&
            i_service_total_q64_words != 0 &&
            i_service_segment_q64_words != 0 &&
            i_service_total_q64_words <= COUNT_WIDTH'(BANK_WORDS) &&
            i_service_segment_q64_words <= i_service_total_q64_words &&
            64'(i_service_total_q64_words) ==
                descriptor_segment_count_wide *
                64'(i_service_segment_q64_words) &&
            descriptor_segment_count_wide <= MAX_SEGMENTS &&
            i_service_source_stride_q64_words >=
                ADDRESS_WIDTH'(i_service_segment_q64_words) &&
            descriptor_last_exclusive <= (64'h1 << ADDRESS_WIDTH);
    end

    // The descriptor preparation phase computes Q16 words/output once. Wave
    // admission then needs only multiplies and alignment checks, rather than a
    // dynamic divider on every read-control path.
    always_comb begin
        descriptor_output_end_wide =
            {1'b0, descriptor_output_base_q} +
            {1'b0, descriptor_output_count_q};
        wave_output_end_wide = {1'b0, i_wave_output_channel_base} +
                               {1'b0, i_wave_output_channel_count};
        wave_output_offset_wide = 11'd0;
        wave_local_q16_base_wide = 64'd0;
        wave_local_q16_limit_wide = 64'd0;
        wave_local_q64_base_wide = 64'd0;
        wave_local_q64_limit_wide = 64'd0;
        wave_output_range_contained = 1'b0;
        wave_output_range_contiguous = 1'b0;
        wave_output_layout_valid = 1'b0;
        if (i_wave_output_channel_base >= descriptor_output_base_q) begin
            wave_output_offset_wide =
                {1'b0, i_wave_output_channel_base} -
                {1'b0, descriptor_output_base_q};
        end
        wave_local_q16_base_wide =
            64'(wave_output_offset_wide) *
            64'(descriptor_q16_per_output_q);
        wave_local_q16_limit_wide =
            64'(wave_output_offset_wide +
                {1'b0, i_wave_output_channel_count}) *
            64'(descriptor_q16_per_output_q);
        wave_local_q64_base_wide = wave_local_q16_base_wide >> 2;
        wave_local_q64_limit_wide = wave_local_q16_limit_wide >> 2;

        wave_output_range_contained =
            i_wave_output_channel_count != 0 &&
            i_wave_output_channel_base >= descriptor_output_base_q &&
            wave_output_end_wide <= descriptor_output_end_wide;
        wave_output_range_contiguous =
            i_wave_output_channel_base == descriptor_next_output_q;
        wave_output_layout_valid =
            descriptor_q16_layout_exact_q &&
            descriptor_q16_per_output_q != 0 &&
            64'(descriptor_q16_per_output_q) *
                64'(descriptor_output_count_q) ==
                (64'(descriptor_total_q) << 2) &&
            wave_local_q16_base_wide[1:0] == 2'b00 &&
            wave_local_q16_limit_wide[1:0] == 2'b00 &&
            wave_local_q64_base_wide < wave_local_q64_limit_wide &&
            wave_local_q64_limit_wide <= descriptor_total_q &&
            wave_local_q64_limit_wide <= BANK_WORDS;
    end

    assign metadata_match = descriptor_active_q &&
        i_wave_layer_index == descriptor_layer_q &&
        wave_output_range_contained &&
        wave_output_range_contiguous &&
        wave_output_layout_valid &&
        i_wave_input_channel_base == descriptor_input_base_q &&
        i_wave_input_channel_count == descriptor_input_count_q &&
        i_wave_source_q64_word == descriptor_source_q &&
        i_wave_total_q64_words == descriptor_total_q &&
        i_wave_segment_q64_words == descriptor_segment_q &&
        i_wave_source_stride_q64_words == descriptor_stride_q &&
        i_wave_tile_tag == descriptor_tag_q;

    assign o_compute_start_valid = i_wave_start_valid && metadata_match &&
        !wave_active_q && !sticky_error_q && !i_abort &&
        !i_service_sticky_error;
    assign o_wave_start_ready = o_compute_start_valid &&
        i_compute_start_ready;
    assign compute_start_fire = o_compute_start_valid &&
        i_compute_start_ready;

    assign relocation0 = relocate_address(i_absolute_q64_addr0);
    assign relocation1 = relocate_address(i_absolute_q64_addr1);
    assign o_absolute_read_ready = wave_active_q && !wave_done_seen_q &&
        !command_pending_q && !read_outstanding_q && !response_valid_q &&
        !sticky_error_q && !i_abort && !i_service_sticky_error;
    assign absolute_read_fire = i_absolute_read_valid &&
        o_absolute_read_ready;

    assign o_service_read_valid = command_pending_q && !sticky_error_q &&
        !i_abort && !i_service_sticky_error;
    assign o_service_read_word_addr0 = command_addr0_q;
    assign o_service_read_word_addr1 = command_addr1_q;
    assign service_read_fire = o_service_read_valid && i_service_read_ready;

    assign o_service_response_ready = read_outstanding_q &&
        (sticky_error_q || i_abort || !response_valid_q);
    assign service_response_fire = i_service_response_valid &&
        o_service_response_ready;

    assign o_absolute_response_valid = response_valid_q;
    assign o_absolute_data0 = response_data0_q;
    assign o_absolute_data1 = response_data1_q;
    assign absolute_response_fire = response_valid_q &&
        i_absolute_response_ready;

    assign o_busy = descriptor_preparing_q || descriptor_active_q || wave_active_q ||
        command_pending_q || read_outstanding_q || response_valid_q;
    assign o_idle = !o_busy;
    assign o_descriptor_active = descriptor_active_q;
    assign o_wave_active = wave_active_q;
    assign o_metadata_match = metadata_match;
    assign o_read_pending = command_pending_q || read_outstanding_q;
    assign o_response_pending = response_valid_q;
    assign o_abort_request = sticky_error_q;
    assign o_sticky_error = sticky_error_q;
    assign o_error_code = error_code_q;

    // Payload state is reset-free; descriptor_preparing_q is its validity.
    always_ff @(posedge i_clk) begin
        if (descriptor_preparing_q && segment_prepare_active_q &&
            integer'(prepare_index_q) < MAX_SEGMENTS)
            segment_source_q[prepare_index_q] <= prepare_source_q;
    end

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            descriptor_active_q <= 1'b0;
            descriptor_preparing_q <= 1'b0;
            descriptor_wait_drop_q <= 1'b0;
            wave_active_q <= 1'b0;
            wave_done_seen_q <= 1'b0;
            command_pending_q <= 1'b0;
            read_outstanding_q <= 1'b0;
            response_valid_q <= 1'b0;
            sticky_error_q <= 1'b0;
            error_code_q <= ERROR_NONE;
            descriptor_layer_q <= '0;
            descriptor_output_base_q <= '0;
            descriptor_output_count_q <= '0;
            descriptor_input_base_q <= '0;
            descriptor_input_count_q <= '0;
            descriptor_source_q <= '0;
            descriptor_total_q <= '0;
            descriptor_segment_q <= '0;
            descriptor_stride_q <= '0;
            descriptor_segment_count_q <= '0;
            descriptor_tag_q <= '0;
            descriptor_next_output_q <= '0;
            descriptor_q16_per_output_q <= '0;
            descriptor_q16_layout_exact_q <= 1'b0;
            segment_prepare_active_q <= 1'b0;
            prepare_index_q <= '0;
            prepare_source_q <= '0;
            flat_divide_active_q <= 1'b0;
            divide_dividend_q <= '0;
            divide_quotient_q <= '0;
            divide_remainder_q <= '0;
            divide_divisor_q <= '0;
            divide_steps_q <= '0;
            command_addr0_q <= '0;
            command_addr1_q <= '0;
            response_data0_q <= '0;
            response_data1_q <= '0;
            wave_local_base_q <= '0;
            wave_local_limit_q <= '0;
            wave_next_output_q <= '0;
            wave_finishes_descriptor_q <= 1'b0;
            o_service_compute_tile_done <= 1'b0;
        end else begin
            o_service_compute_tile_done <= 1'b0;

            if (!i_service_compute_tile_valid)
                descriptor_wait_drop_q <= 1'b0;

            if (i_clear_error && !o_busy &&
                !i_service_compute_tile_valid &&
                !i_abort) begin
                // The service and coordinator clear on the same edge. Do not
                // require the coordinator to observe the service's old sticky
                // value deasserted first; a persistent service fault is
                // sampled again on the following cycle.
                sticky_error_q <= 1'b0;
                error_code_q <= ERROR_NONE;
                descriptor_wait_drop_q <= 1'b0;
            end else if (i_abort) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_ABORT;
                descriptor_active_q <= 1'b0;
                descriptor_preparing_q <= 1'b0;
                segment_prepare_active_q <= 1'b0;
                flat_divide_active_q <= 1'b0;
                descriptor_wait_drop_q <= 1'b1;
                wave_active_q <= 1'b0;
                wave_done_seen_q <= 1'b0;
                wave_finishes_descriptor_q <= 1'b0;
                command_pending_q <= 1'b0;
                response_valid_q <= 1'b0;
                if (response_valid_q || service_response_fire)
                    read_outstanding_q <= 1'b0;
            end else if (i_service_sticky_error) begin
                sticky_error_q <= 1'b1;
                error_code_q <= ERROR_SERVICE;
                descriptor_active_q <= 1'b0;
                descriptor_preparing_q <= 1'b0;
                segment_prepare_active_q <= 1'b0;
                flat_divide_active_q <= 1'b0;
                descriptor_wait_drop_q <= 1'b1;
                wave_active_q <= 1'b0;
                wave_done_seen_q <= 1'b0;
                wave_finishes_descriptor_q <= 1'b0;
                command_pending_q <= 1'b0;
                response_valid_q <= 1'b0;
                if (response_valid_q || service_response_fire)
                    read_outstanding_q <= 1'b0;
            end else if (sticky_error_q) begin
                descriptor_active_q <= 1'b0;
                descriptor_preparing_q <= 1'b0;
                segment_prepare_active_q <= 1'b0;
                flat_divide_active_q <= 1'b0;
                wave_active_q <= 1'b0;
                wave_done_seen_q <= 1'b0;
                wave_finishes_descriptor_q <= 1'b0;
                command_pending_q <= 1'b0;
                if (response_valid_q) begin
                    response_valid_q <= 1'b0;
                    read_outstanding_q <= 1'b0;
                end else if (service_response_fire) begin
                    read_outstanding_q <= 1'b0;
                end
            end else begin
                if (!descriptor_active_q && !descriptor_preparing_q &&
                    !descriptor_wait_drop_q &&
                    i_service_compute_tile_valid) begin
                    if (!descriptor_shape_valid) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_DESCRIPTOR;
                        descriptor_wait_drop_q <= 1'b1;
                    end else begin
                        descriptor_preparing_q <= 1'b1;
                        segment_prepare_active_q <= 1'b1;
                        prepare_index_q <= '0;
                        prepare_source_q <= i_service_source_q64_word;
                        descriptor_layer_q <= i_service_layer_index;
                        descriptor_output_base_q <=
                            i_service_output_channel_base;
                        descriptor_output_count_q <=
                            i_service_output_channel_count;
                        descriptor_input_base_q <=
                            i_service_input_channel_base;
                        descriptor_input_count_q <=
                            i_service_input_channel_count;
                        descriptor_source_q <= i_service_source_q64_word;
                        descriptor_total_q <= i_service_total_q64_words;
                        descriptor_segment_q <= i_service_segment_q64_words;
                        descriptor_stride_q <=
                            i_service_source_stride_q64_words;
                        descriptor_segment_count_q <= COUNT_WIDTH'(
                            descriptor_segment_count_wide);
                        descriptor_tag_q <= i_service_tile_tag;
                        if (i_service_segment_q64_words ==
                            i_service_total_q64_words) begin
                            descriptor_q16_layout_exact_q <= 1'b0;
                            flat_divide_active_q <= 1'b1;
                            divide_dividend_q <= Q16_COUNT_WIDTH'(
                                i_service_total_q64_words) << 2;
                            divide_quotient_q <= '0;
                            divide_remainder_q <= '0;
                            divide_divisor_q <=
                                i_service_output_channel_count;
                            divide_steps_q <= DIV_STEP_WIDTH'(
                                Q16_COUNT_WIDTH);
                        end else begin
                            descriptor_q16_per_output_q <=
                                Q16_COUNT_WIDTH'(
                                    i_service_segment_q64_words) << 2;
                            descriptor_q16_layout_exact_q <= 1'b1;
                            flat_divide_active_q <= 1'b0;
                        end
                    end
                end

                if (descriptor_preparing_q &&
                    i_service_compute_tile_valid) begin
                    if (segment_prepare_active_q) begin
                        if (COUNT_WIDTH'(prepare_index_q) + 1 >=
                            descriptor_segment_count_q) begin
                            segment_prepare_active_q <= 1'b0;
                        end else begin
                            prepare_index_q <= prepare_index_q + 1'b1;
                            prepare_source_q <= prepare_source_q +
                                descriptor_stride_q;
                        end
                    end

                    if (flat_divide_active_q) begin
                        divide_dividend_q <= divide_dividend_q << 1;
                        divide_quotient_q <= divider_quotient_next;
                        divide_remainder_q <= divider_remainder_next;
                        divide_steps_q <= divide_steps_q - 1'b1;
                        if (divide_steps_q == 1) begin
                            flat_divide_active_q <= 1'b0;
                            descriptor_q16_per_output_q <=
                                divider_quotient_next;
                            descriptor_q16_layout_exact_q <=
                                divider_quotient_next != 0 &&
                                divider_remainder_next == 0;
                        end
                    end

                    if (!segment_prepare_active_q &&
                        !flat_divide_active_q) begin
                        descriptor_preparing_q <= 1'b0;
                        if (!descriptor_q16_layout_exact_q) begin
                            sticky_error_q <= 1'b1;
                            error_code_q <= ERROR_DESCRIPTOR;
                            descriptor_wait_drop_q <= 1'b1;
                        end else begin
                            descriptor_active_q <= 1'b1;
                            descriptor_next_output_q <=
                                descriptor_output_base_q;
                        end
                    end
                end

                if ((descriptor_active_q || descriptor_preparing_q) &&
                    !i_service_compute_tile_valid) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_DESCRIPTOR_DROP;
                    descriptor_active_q <= 1'b0;
                    descriptor_preparing_q <= 1'b0;
                    segment_prepare_active_q <= 1'b0;
                    flat_divide_active_q <= 1'b0;
                end

                if (i_wave_start_valid && descriptor_active_q &&
                    !wave_active_q && !metadata_match) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_METADATA;
                end else if (i_wave_start_valid && wave_active_q) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_START_PROTOCOL;
                end

                if (compute_start_fire) begin
                    wave_active_q <= 1'b1;
                    wave_done_seen_q <= 1'b0;
                    wave_local_base_q <= (LOCAL_ADDR_WIDTH+1)'(
                        wave_local_q64_base_wide);
                    wave_local_limit_q <= (LOCAL_ADDR_WIDTH+1)'(
                        wave_local_q64_limit_wide);
                    wave_next_output_q <= 10'(wave_output_end_wide);
                    wave_finishes_descriptor_q <=
                        wave_output_end_wide == descriptor_output_end_wide;
                end

                if (absolute_read_fire) begin
                    if (!relocation0.valid || !relocation1.valid) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_ADDRESS;
                    end else begin
                        command_pending_q <= 1'b1;
                        command_addr0_q <= relocation0.local_addr;
                        command_addr1_q <= relocation1.local_addr;
                    end
                end

                if (service_read_fire) begin
                    command_pending_q <= 1'b0;
                    read_outstanding_q <= 1'b1;
                end

                if (i_service_response_valid &&
                    !read_outstanding_q) begin
                    sticky_error_q <= 1'b1;
                    error_code_q <= ERROR_SERVICE_PROTOCOL;
                end else if (service_response_fire) begin
                    response_valid_q <= 1'b1;
                    response_data0_q <= i_service_read_data0;
                    response_data1_q <= i_service_read_data1;
                end

                if (absolute_response_fire) begin
                    response_valid_q <= 1'b0;
                    read_outstanding_q <= 1'b0;
                end

                if (i_wave_done) begin
                    if (!wave_active_q || wave_done_seen_q ||
                        i_wave_done_tag != descriptor_tag_q) begin
                        sticky_error_q <= 1'b1;
                        error_code_q <= ERROR_DONE_PROTOCOL;
                    end else begin
                        wave_done_seen_q <= 1'b1;
                    end
                end

                if (descriptor_active_q && wave_active_q &&
                    wave_done_seen_q && !command_pending_q &&
                    !read_outstanding_q && !response_valid_q) begin
                    wave_active_q <= 1'b0;
                    wave_done_seen_q <= 1'b0;
                    wave_finishes_descriptor_q <= 1'b0;
                    if (wave_finishes_descriptor_q) begin
                        o_service_compute_tile_done <= 1'b1;
                        descriptor_active_q <= 1'b0;
                        descriptor_preparing_q <= 1'b0;
                        descriptor_wait_drop_q <= 1'b1;
                    end else begin
                        descriptor_next_output_q <= wave_next_output_q;
                    end
                end
            end
        end
    end

    logic unused_service_code;
    assign unused_service_code = &{1'b0, i_service_error_code};

    initial begin
        if (MAX_SEGMENTS < 1)
            $fatal(1, "MAX_SEGMENTS must be positive");
    end

endmodule

`default_nettype wire
