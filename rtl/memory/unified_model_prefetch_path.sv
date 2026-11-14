`default_nettype none

module unified_model_prefetch_path #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int BANK_WORDS = 16384,
    parameter int TAG_WIDTH = 16,
    parameter int MAX_BURST_BEATS = 64,
    parameter int MAX_OUTSTANDING = 4
) (
    input  logic                         i_clk,
    input  logic                         i_rst_n,
    input  logic                         i_abort,
    input  logic [ADDRESS_WIDTH-1:0]     i_region_base_word,
    input  logic [ADDRESS_WIDTH-1:0]     i_region_limit_word,

    input  logic                         i_prefetch_valid,
    output logic                         o_prefetch_ready,
    input  logic [ADDRESS_WIDTH-1:0]     i_prefetch_source_word,
    input  logic [$clog2(BANK_WORDS+1)-1:0] i_prefetch_word_count,
    input  logic [$clog2(BANK_WORDS+1)-1:0] i_prefetch_segment_words,
    input  logic [ADDRESS_WIDTH-1:0]     i_prefetch_source_stride_words,
    input  logic [TAG_WIDTH-1:0]         i_prefetch_tag,
    output logic                         o_prefetch_done,

    output logic                         o_ddr_cmd_valid,
    input  logic                         i_ddr_cmd_ready,
    output logic [ADDRESS_WIDTH-1:0]     o_ddr_cmd_word_addr,
    output logic [7:0]                   o_ddr_cmd_beats,
    input  logic                         i_ddr_data_valid,
    output logic                         o_ddr_data_ready,
    input  logic [63:0]                  i_ddr_data,
    input  logic                         i_ddr_data_last,
    input  logic                         i_ddr_data_error,

    input  logic                         i_compute_acquire_valid,
    output logic                         o_compute_acquire_ready,
    input  logic [TAG_WIDTH-1:0]         i_compute_acquire_tag,
    output logic                         o_compute_bank,
    output logic [TAG_WIDTH-1:0]         o_compute_tag,
    input  logic                         i_compute_release,
    input  logic                         i_read_valid,
    output logic                         o_read_ready,
    input  logic [$clog2(BANK_WORDS)-1:0] i_read_word_addr0,
    input  logic [$clog2(BANK_WORDS)-1:0] i_read_word_addr1,
    output logic                         o_read_valid,
    input  logic                         i_read_result_ready,
    output logic [63:0]                  o_read_data0,
    output logic [63:0]                  o_read_data1,

    output logic [1:0]                   o_bank_ready,
    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_error
);

    localparam int COUNT_WIDTH = $clog2(BANK_WORDS + 1);
    localparam int DEST_WIDTH = $clog2(BANK_WORDS);

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_START_BANK,
        STATE_ISSUE,
        STATE_WAIT,
        STATE_CANCEL,
        STATE_ERROR
    } state_t;

    state_t state_q;
    logic [ADDRESS_WIDTH-1:0] segment_source_q;
    logic [ADDRESS_WIDTH-1:0] source_stride_q;
    logic [COUNT_WIDTH-1:0] word_count_q;
    logic [COUNT_WIDTH-1:0] segment_words_q;
    logic [COUNT_WIDTH-1:0] segment_progress_q;
    logic [TAG_WIDTH-1:0] tag_q;
    logic [COUNT_WIDTH-1:0] words_issued_q;
    logic [7:0] next_burst_beats;

    logic scratch_prefetch_valid;
    logic scratch_prefetch_ready;
    logic scratch_prefetch_bank;
    logic scratch_fill_ready;
    logic scratch_prefetch_active;
    logic scratch_compute_active;
    logic scratch_idle;
    logic scratch_bounds_error;
    logic scratch_protocol_error;
    logic scratch_abort;
    logic service_req_valid;
    logic service_req_ready;
    logic service_fill_valid;
    logic service_fill_ready;
    logic [DEST_WIDTH-1:0] service_fill_addr;
    logic [63:0] service_fill_data;
    logic service_fill_last;
    logic [$clog2(MAX_OUTSTANDING+1)-1:0] service_outstanding;
    logic service_idle;
    logic service_bounds_error;
    logic service_protocol_error;
    logic service_ddr_error;
    logic service_error;
    logic issue_accept;
    logic all_words_issued;
    logic [COUNT_WIDTH-1:0] total_words_remaining;
    logic [COUNT_WIDTH-1:0] segment_words_remaining;
    logic prefetch_contract_valid;

    always_comb begin
        total_words_remaining = word_count_q - words_issued_q;
        segment_words_remaining = segment_words_q - segment_progress_q;
        if (total_words_remaining <= segment_words_remaining &&
            total_words_remaining <= COUNT_WIDTH'(MAX_BURST_BEATS))
            next_burst_beats = 8'(total_words_remaining);
        else if (segment_words_remaining <= total_words_remaining &&
                 segment_words_remaining <= COUNT_WIDTH'(MAX_BURST_BEATS))
            next_burst_beats = 8'(segment_words_remaining);
        else
            next_burst_beats = 8'(MAX_BURST_BEATS);
    end

    assign prefetch_contract_valid = word_count_q != 0 &&
        word_count_q <= COUNT_WIDTH'(BANK_WORDS) && segment_words_q != 0 &&
        segment_words_q <= word_count_q &&
        word_count_q % segment_words_q == 0 &&
        source_stride_q >= ADDRESS_WIDTH'(segment_words_q);
    assign o_prefetch_ready = state_q == STATE_IDLE && !i_abort &&
                              scratch_prefetch_ready &&
                              !service_error && !scratch_bounds_error &&
                              !scratch_protocol_error;
    assign scratch_prefetch_valid = state_q == STATE_START_BANK &&
                                    prefetch_contract_valid;
    assign service_req_valid = state_q == STATE_ISSUE && !all_words_issued;
    assign issue_accept = service_req_valid && service_req_ready;
    assign all_words_issued = words_issued_q == word_count_q;
    assign service_error = service_bounds_error || service_protocol_error ||
                           service_ddr_error;
    assign scratch_abort = state_q != STATE_ERROR &&
                           (service_error || scratch_bounds_error ||
                            scratch_protocol_error);
    assign service_fill_ready = scratch_fill_ready;

    unified_model_memory_read_service #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .DEST_WIDTH(DEST_WIDTH),
        .MAX_BURST_BEATS(MAX_BURST_BEATS),
        .MAX_OUTSTANDING(MAX_OUTSTANDING)
    ) u_read_service (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_abort(i_abort),
        .i_region_base_word(i_region_base_word),
        .i_region_limit_word(i_region_limit_word),
        .i_req_valid(service_req_valid), .o_req_ready(service_req_ready),
        .i_req_word_addr(segment_source_q + ADDRESS_WIDTH'(segment_progress_q)),
        .i_req_beats(next_burst_beats),
        .i_req_destination(DEST_WIDTH'(words_issued_q)),
        .i_destination_limit((DEST_WIDTH+1)'(BANK_WORDS)),
        .o_ddr_cmd_valid(o_ddr_cmd_valid), .i_ddr_cmd_ready(i_ddr_cmd_ready),
        .o_ddr_cmd_word_addr(o_ddr_cmd_word_addr),
        .o_ddr_cmd_beats(o_ddr_cmd_beats),
        .i_ddr_data_valid(i_ddr_data_valid),
        .o_ddr_data_ready(o_ddr_data_ready),
        .i_ddr_data(i_ddr_data), .i_ddr_data_last(i_ddr_data_last),
        .i_ddr_data_error(i_ddr_data_error),
        .o_fill_valid(service_fill_valid), .i_fill_ready(service_fill_ready),
        .o_fill_addr(service_fill_addr), .o_fill_data(service_fill_data),
        .o_fill_last(service_fill_last), .o_outstanding(service_outstanding),
        .o_idle(service_idle), .o_bounds_error(service_bounds_error),
        .o_response_protocol_error(service_protocol_error),
        .o_ddr_error(service_ddr_error)
    );

    unified_model_pingpong_scratchpad #(
        .WORD_WIDTH(64), .BANK_WORDS(BANK_WORDS), .TAG_WIDTH(TAG_WIDTH)
    ) u_scratchpad (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_prefetch_valid(scratch_prefetch_valid),
        .o_prefetch_ready(scratch_prefetch_ready),
        .i_prefetch_tag(tag_q), .i_prefetch_word_count(word_count_q),
        .o_prefetch_bank(scratch_prefetch_bank),
        .i_fill_valid(service_fill_valid), .o_fill_ready(scratch_fill_ready),
        .i_fill_word_addr(service_fill_addr), .i_fill_data(service_fill_data),
        .i_fill_last(({1'b0, service_fill_addr} + 1'b1) ==
                     (DEST_WIDTH+1)'(word_count_q)),
        .i_prefetch_abort(scratch_abort),
        .i_cancel(i_abort || state_q == STATE_CANCEL),
        .i_compute_acquire_valid(i_compute_acquire_valid),
        .o_compute_acquire_ready(o_compute_acquire_ready),
        .i_compute_acquire_tag(i_compute_acquire_tag),
        .o_compute_bank(o_compute_bank), .o_compute_tag(o_compute_tag),
        .i_compute_release(i_compute_release),
        .i_read_valid(i_read_valid), .o_read_ready(o_read_ready),
        .i_read_word_addr0(i_read_word_addr0),
        .i_read_word_addr1(i_read_word_addr1),
        .o_read_valid(o_read_valid), .i_read_ready(i_read_result_ready),
        .o_read_data0(o_read_data0), .o_read_data1(o_read_data1),
        .o_bank_ready(o_bank_ready), .o_prefetch_active(scratch_prefetch_active),
        .o_compute_active(scratch_compute_active), .o_idle(scratch_idle),
        .o_bounds_error(scratch_bounds_error),
        .o_protocol_error(scratch_protocol_error)
    );

    assign o_busy = state_q != STATE_IDLE || scratch_prefetch_active ||
                    scratch_compute_active || service_outstanding != 0;
    assign o_idle = state_q == STATE_IDLE && service_idle && scratch_idle;
    assign o_error = state_q == STATE_ERROR || service_error ||
                     scratch_bounds_error || scratch_protocol_error;

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            segment_source_q <= '0;
            source_stride_q <= '0;
            word_count_q <= '0;
            segment_words_q <= '0;
            segment_progress_q <= '0;
            tag_q <= '0;
            words_issued_q <= '0;
            o_prefetch_done <= 1'b0;
        end else begin
            o_prefetch_done <= 1'b0;
            if (i_abort)
                state_q <= STATE_CANCEL;
            else if (scratch_abort)
                state_q <= STATE_ERROR;
            else begin
                case (state_q)
                    STATE_IDLE: begin
                        if (i_prefetch_valid && o_prefetch_ready) begin
                            segment_source_q <= i_prefetch_source_word;
                            source_stride_q <= i_prefetch_source_stride_words;
                            word_count_q <= i_prefetch_word_count;
                            segment_words_q <= i_prefetch_segment_words;
                            segment_progress_q <= '0;
                            tag_q <= i_prefetch_tag;
                            words_issued_q <= '0;
                            state_q <= STATE_START_BANK;
                        end
                    end
                    STATE_START_BANK: begin
                        if (!prefetch_contract_valid)
                            state_q <= STATE_ERROR;
                        else if (scratch_prefetch_valid && scratch_prefetch_ready)
                            state_q <= STATE_ISSUE;
                    end
                    STATE_ISSUE: begin
                        if (issue_accept) begin
                            words_issued_q <= words_issued_q +
                                              COUNT_WIDTH'(next_burst_beats);
                            if (segment_progress_q + COUNT_WIDTH'(next_burst_beats) ==
                                segment_words_q) begin
                                segment_progress_q <= '0;
                                segment_source_q <= segment_source_q + source_stride_q;
                            end else begin
                                segment_progress_q <= segment_progress_q +
                                                      COUNT_WIDTH'(next_burst_beats);
                            end
                        end
                        if (all_words_issued)
                            state_q <= STATE_WAIT;
                    end
                    STATE_WAIT: begin
                        if (!scratch_prefetch_active && service_idle) begin
                            o_prefetch_done <= 1'b1;
                            state_q <= STATE_IDLE;
                        end
                    end
                    STATE_CANCEL: begin
                        if (service_idle && scratch_idle)
                            state_q <= STATE_IDLE;
                    end
                    default: state_q <= STATE_ERROR;
                endcase
            end
        end
    end

    logic unused_status;
    assign unused_status = &{1'b0, service_fill_last, scratch_prefetch_bank};

endmodule

`default_nettype wire
