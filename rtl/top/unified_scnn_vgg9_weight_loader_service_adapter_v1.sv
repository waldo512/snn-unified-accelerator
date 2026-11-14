`default_nettype none

// Service-facing adaptation of the existing VGG-9 descriptor loader. This
// wrapper keeps the established sequencer/prefetch/scratchpad implementation
// while exposing the memory blockers needed by the final control plane.
module unified_scnn_vgg9_weight_loader_service_adapter_v1 #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int BANK_WORDS = 8640,
    parameter int TAG_WIDTH = 16,
    parameter int MAX_BURST_BEATS = 64,
    parameter int MAX_OUTSTANDING = 4,
    parameter int DESCRIPTOR_LIMIT = 0,
    parameter bit WAVE_GROUP_DESCRIPTORS = 1'b0
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_start,
    input  logic                         i_abort,
    input  logic                         i_clear_error,
    output logic                         o_busy,
    output logic                         o_done,

    input  logic [ADDRESS_WIDTH-1:0]     i_region_base_word,
    input  logic [ADDRESS_WIDTH-1:0]     i_region_limit_word,
    output logic                         o_ddr_cmd_valid,
    input  logic                         i_ddr_cmd_ready,
    output logic [ADDRESS_WIDTH-1:0]     o_ddr_cmd_word_addr,
    output logic [7:0]                   o_ddr_cmd_beats,
    input  logic                         i_ddr_data_valid,
    output logic                         o_ddr_data_ready,
    input  logic [63:0]                  i_ddr_data,
    input  logic                         i_ddr_data_last,
    input  logic                         i_ddr_data_error,

    output logic                         o_compute_tile_valid,
    input  logic                         i_compute_tile_done,
    output logic [3:0]                   o_compute_layer_index,
    output logic [9:0]                   o_compute_output_channel_base,
    output logic [9:0]                   o_compute_output_channel_count,
    output logic [9:0]                   o_compute_input_channel_base,
    output logic [9:0]                   o_compute_input_channel_count,
    output logic [ADDRESS_WIDTH-1:0]     o_compute_source_q64_word,
    output logic [$clog2(BANK_WORDS+1)-1:0] o_compute_total_q64_words,
    output logic [$clog2(BANK_WORDS+1)-1:0] o_compute_segment_q64_words,
    output logic [ADDRESS_WIDTH-1:0]     o_compute_source_stride_q64_words,
    output logic [TAG_WIDTH-1:0]         o_compute_tile_tag,

    input  logic                         i_read_valid,
    output logic                         o_read_ready,
    input  logic [$clog2(BANK_WORDS)-1:0] i_read_word_addr0,
    input  logic [$clog2(BANK_WORDS)-1:0] i_read_word_addr1,
    output logic                         o_read_valid,
    input  logic                         i_read_result_ready,
    output logic [63:0]                  o_read_data0,
    output logic [63:0]                  o_read_data1,

    output logic                         o_prefetch_busy,
    output logic                         o_scratchpad_busy,
    output logic                         o_prefetch_compute_overlap_seen,
    output logic                         o_sticky_error,
    output logic [3:0]                   o_error_code
);

    localparam int COUNT_WIDTH = $clog2(BANK_WORDS + 1);
    localparam logic [3:0] ERROR_NONE = 4'h0;
    localparam logic [3:0] ERROR_ABORT = 4'h4;
    localparam logic [3:0] ERROR_PREFETCH = 4'h6;

    logic seq_start;
    logic seq_rst_n;
    logic seq_busy;
    logic seq_done;
    logic seq_valid;
    logic seq_ready;
    logic [3:0] seq_layer;
    logic [9:0] seq_output_base;
    logic [9:0] seq_output_count;
    logic [9:0] seq_input_base;
    logic [9:0] seq_input_count;
    logic [ADDRESS_WIDTH-1:0] seq_source;
    logic [COUNT_WIDTH-1:0] seq_total;
    logic [COUNT_WIDTH-1:0] seq_segment;
    logic [ADDRESS_WIDTH-1:0] seq_stride;
    logic [TAG_WIDTH-1:0] seq_tag;
    logic seq_last;

    logic [3:0] fifo_layer [0:1];
    logic [9:0] fifo_output_base [0:1];
    logic [9:0] fifo_output_count [0:1];
    logic [9:0] fifo_input_base [0:1];
    logic [9:0] fifo_input_count [0:1];
    logic [ADDRESS_WIDTH-1:0] fifo_source [0:1];
    logic [COUNT_WIDTH-1:0] fifo_total [0:1];
    logic [COUNT_WIDTH-1:0] fifo_segment [0:1];
    logic [ADDRESS_WIDTH-1:0] fifo_stride [0:1];
    logic [TAG_WIDTH-1:0] fifo_tag [0:1];
    logic [1:0] loaded_q;
    logic head_q;
    logic tail_q;
    logic [1:0] fifo_count_q;
    logic active_prefetch_slot_q;
    logic compute_present_q;
    logic sequence_exhausted_q;
    logic [15:0] accepted_descriptors_q;
    logic loader_busy_q;
    logic prefetch_active_q;
    logic local_error_q;
    logic [3:0] local_error_code_q;
    logic cancel_pending_q;

    logic memory_prefetch_ready;
    logic memory_prefetch_done;
    logic memory_acquire_ready;
    logic memory_compute_bank;
    logic [TAG_WIDTH-1:0] memory_compute_tag;
    logic [1:0] memory_bank_ready;
    logic memory_busy;
    logic memory_idle;
    logic memory_error;
    logic push;
    logic pop;
    logic acquire_valid;
    logic acquire_accept;
    logic descriptor_limit_reached;
    logic completion_condition;

    generate
        if (WAVE_GROUP_DESCRIPTORS) begin : g_wave_group_sequencer
            unified_scnn_vgg9_wave_group_sequencer_v2 #(
                .ADDRESS_WIDTH(ADDRESS_WIDTH), .BANK_WORDS(BANK_WORDS),
                .TAG_WIDTH(TAG_WIDTH)
            ) u_sequencer (
                .i_clk(i_clk), .i_rst_n(seq_rst_n), .i_start(seq_start),
                .o_busy(seq_busy), .o_done(seq_done),
                .o_tile_valid(seq_valid), .i_tile_ready(seq_ready),
                .o_layer_index(seq_layer),
                .o_output_channel_base(seq_output_base),
                .o_output_channel_count(seq_output_count),
                .o_input_channel_base(seq_input_base),
                .o_input_channel_count(seq_input_count),
                .o_source_q64_word(seq_source),
                .o_total_q64_words(seq_total),
                .o_segment_q64_words(seq_segment),
                .o_source_stride_q64_words(seq_stride),
                .o_tile_tag(seq_tag), .o_last_tile(seq_last)
            );
        end else begin : g_legacy_tile_sequencer
            unified_scnn_vgg9_weight_tile_sequencer #(
                .ADDRESS_WIDTH(ADDRESS_WIDTH), .BANK_WORDS(BANK_WORDS),
                .TAG_WIDTH(TAG_WIDTH)
            ) u_sequencer (
                .i_clk(i_clk), .i_rst_n(seq_rst_n), .i_start(seq_start),
                .o_busy(seq_busy), .o_done(seq_done),
                .o_tile_valid(seq_valid), .i_tile_ready(seq_ready),
                .o_layer_index(seq_layer),
                .o_output_channel_base(seq_output_base),
                .o_output_channel_count(seq_output_count),
                .o_input_channel_base(seq_input_base),
                .o_input_channel_count(seq_input_count),
                .o_source_q64_word(seq_source),
                .o_total_q64_words(seq_total),
                .o_segment_q64_words(seq_segment),
                .o_source_stride_q64_words(seq_stride),
                .o_tile_tag(seq_tag), .o_last_tile(seq_last)
            );
        end
    endgenerate

    assign descriptor_limit_reached = DESCRIPTOR_LIMIT != 0 &&
        accepted_descriptors_q >= 16'(DESCRIPTOR_LIMIT);
    assign seq_ready = memory_prefetch_ready && fifo_count_q < 2 &&
                       !descriptor_limit_reached && !o_sticky_error;
    assign push = seq_valid && seq_ready && !i_abort && !cancel_pending_q;

    unified_model_prefetch_path #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .BANK_WORDS(BANK_WORDS),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_BURST_BEATS(MAX_BURST_BEATS),
        .MAX_OUTSTANDING(MAX_OUTSTANDING)
    ) u_prefetch_path (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_region_base_word(i_region_base_word),
        .i_region_limit_word(i_region_limit_word),
        .i_prefetch_valid(seq_valid && fifo_count_q < 2 &&
                          !descriptor_limit_reached && !o_sticky_error),
        .o_prefetch_ready(memory_prefetch_ready),
        .i_prefetch_source_word(seq_source),
        .i_prefetch_word_count(seq_total),
        .i_prefetch_segment_words(seq_segment),
        .i_prefetch_source_stride_words(seq_stride),
        .i_prefetch_tag(seq_tag),
        .o_prefetch_done(memory_prefetch_done),
        .o_ddr_cmd_valid(o_ddr_cmd_valid),
        .i_ddr_cmd_ready(i_ddr_cmd_ready),
        .o_ddr_cmd_word_addr(o_ddr_cmd_word_addr),
        .o_ddr_cmd_beats(o_ddr_cmd_beats),
        .i_ddr_data_valid(i_ddr_data_valid),
        .o_ddr_data_ready(o_ddr_data_ready),
        .i_ddr_data(i_ddr_data),
        .i_ddr_data_last(i_ddr_data_last),
        .i_ddr_data_error(i_ddr_data_error),
        .i_compute_acquire_valid(acquire_valid),
        .o_compute_acquire_ready(memory_acquire_ready),
        .i_compute_acquire_tag(fifo_tag[head_q]),
        .o_compute_bank(memory_compute_bank),
        .o_compute_tag(memory_compute_tag),
        .i_compute_release(pop),
        .i_read_valid(i_read_valid),
        .o_read_ready(o_read_ready),
        .i_read_word_addr0(i_read_word_addr0),
        .i_read_word_addr1(i_read_word_addr1),
        .o_read_valid(o_read_valid),
        .i_read_result_ready(i_read_result_ready),
        .o_read_data0(o_read_data0),
        .o_read_data1(o_read_data1),
        .o_bank_ready(memory_bank_ready),
        .o_busy(memory_busy),
        .o_idle(memory_idle),
        .o_error(memory_error)
    );

    assign acquire_valid = fifo_count_q != 0 && loaded_q[head_q] &&
                           !compute_present_q && !o_sticky_error;
    assign acquire_accept = acquire_valid && memory_acquire_ready;
    assign o_compute_tile_valid = compute_present_q;
    assign pop = compute_present_q && i_compute_tile_done && !i_abort &&
                 !cancel_pending_q;
    assign o_compute_layer_index = fifo_layer[head_q];
    assign o_compute_output_channel_base = fifo_output_base[head_q];
    assign o_compute_output_channel_count = fifo_output_count[head_q];
    assign o_compute_input_channel_base = fifo_input_base[head_q];
    assign o_compute_input_channel_count = fifo_input_count[head_q];
    assign o_compute_source_q64_word = fifo_source[head_q];
    assign o_compute_total_q64_words = fifo_total[head_q];
    assign o_compute_segment_q64_words = fifo_segment[head_q];
    assign o_compute_source_stride_q64_words = fifo_stride[head_q];
    assign o_compute_tile_tag = fifo_tag[head_q];

    assign completion_condition = loader_busy_q && sequence_exhausted_q &&
        fifo_count_q == 0 && !compute_present_q && memory_idle &&
        !prefetch_active_q && !o_sticky_error;
    assign o_busy = loader_busy_q;
    assign o_prefetch_busy = prefetch_active_q;
    // A reserved/filling/ready/compute bank remains a scratchpad blocker.
    assign o_scratchpad_busy = prefetch_active_q || compute_present_q ||
                               o_read_valid || (|memory_bank_ready);
    assign o_sticky_error = local_error_q || memory_error;
    assign o_error_code = local_error_q ? local_error_code_q :
                          memory_error ? ERROR_PREFETCH : ERROR_NONE;
    // Both descriptor sequencers use synchronous reset. Abort restarts the
    // next explicit run at descriptor zero instead of resuming stale state.
    assign seq_rst_n = i_rst_n && !i_abort;
    assign seq_start = i_start && !i_abort && !cancel_pending_q &&
                       !loader_busy_q && !o_sticky_error;

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            loaded_q <= 2'b00;
            head_q <= 1'b0;
            tail_q <= 1'b0;
            fifo_count_q <= '0;
            active_prefetch_slot_q <= 1'b0;
            compute_present_q <= 1'b0;
            sequence_exhausted_q <= 1'b0;
            accepted_descriptors_q <= '0;
            loader_busy_q <= 1'b0;
            prefetch_active_q <= 1'b0;
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
            cancel_pending_q <= 1'b0;
            o_done <= 1'b0;
            o_prefetch_compute_overlap_seen <= 1'b0;
        end else begin
            o_done <= 1'b0;

            if (seq_start) begin
                loaded_q <= 2'b00;
                head_q <= 1'b0;
                tail_q <= 1'b0;
                fifo_count_q <= '0;
                compute_present_q <= 1'b0;
                sequence_exhausted_q <= 1'b0;
                accepted_descriptors_q <= '0;
                loader_busy_q <= 1'b1;
                prefetch_active_q <= 1'b0;
                o_prefetch_compute_overlap_seen <= 1'b0;
                cancel_pending_q <= 1'b0;
            end

            if (i_abort) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_ABORT;
                loaded_q <= 2'b00;
                head_q <= 1'b0;
                tail_q <= 1'b0;
                fifo_count_q <= '0;
                compute_present_q <= 1'b0;
                sequence_exhausted_q <= 1'b1;
                prefetch_active_q <= 1'b0;
                cancel_pending_q <= 1'b1;
            end else if (memory_error && !local_error_q) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_PREFETCH;
            end
            if (cancel_pending_q && memory_idle) begin
                loader_busy_q <= 1'b0;
                cancel_pending_q <= 1'b0;
            end
            if (i_clear_error && !loader_busy_q && memory_idle) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end

            if (!i_abort && !cancel_pending_q) begin
                if (push) begin
                    fifo_layer[tail_q] <= seq_layer;
                    fifo_output_base[tail_q] <= seq_output_base;
                    fifo_output_count[tail_q] <= seq_output_count;
                    fifo_input_base[tail_q] <= seq_input_base;
                    fifo_input_count[tail_q] <= seq_input_count;
                    fifo_source[tail_q] <= seq_source;
                    fifo_total[tail_q] <= seq_total;
                    fifo_segment[tail_q] <= seq_segment;
                    fifo_stride[tail_q] <= seq_stride;
                    fifo_tag[tail_q] <= seq_tag;
                    loaded_q[tail_q] <= 1'b0;
                    active_prefetch_slot_q <= tail_q;
                    tail_q <= ~tail_q;
                    accepted_descriptors_q <= accepted_descriptors_q + 1'b1;
                    if (DESCRIPTOR_LIMIT != 0 &&
                        accepted_descriptors_q + 1'b1 >= 16'(DESCRIPTOR_LIMIT))
                        sequence_exhausted_q <= 1'b1;
                end

                if (memory_prefetch_done)
                    loaded_q[active_prefetch_slot_q] <= 1'b1;
                case ({push, memory_prefetch_done})
                    2'b10, 2'b11: prefetch_active_q <= 1'b1;
                    2'b01: prefetch_active_q <= 1'b0;
                    default: prefetch_active_q <= prefetch_active_q;
                endcase
                if (acquire_accept)
                    compute_present_q <= 1'b1;
                if (pop) begin
                    compute_present_q <= 1'b0;
                    loaded_q[head_q] <= 1'b0;
                    head_q <= ~head_q;
                end
                case ({push, pop})
                    2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                    2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                    default: fifo_count_q <= fifo_count_q;
                endcase

                if (seq_done)
                    sequence_exhausted_q <= 1'b1;
                if (prefetch_active_q && compute_present_q)
                    o_prefetch_compute_overlap_seen <= 1'b1;
                if (completion_condition) begin
                    loader_busy_q <= 1'b0;
                    o_done <= 1'b1;
                end
            end
        end
    end

    logic unused_status;
    assign unused_status = &{1'b0, seq_busy, seq_last, memory_compute_bank,
        memory_compute_tag, memory_busy};

endmodule

`default_nettype wire
