`default_nettype none

// Read-only DDR3A bridge for SCNN weight prefetch. Core commands are 64-bit
// word addressed and ordered. MIG AXI is 128-bit and may run asynchronously.
module unified_netfpga_ddr3a_axi_read_bridge #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int MAX_BURST_WORDS = 64,
    parameter int MAX_OUTSTANDING = 4,
    parameter logic [ADDRESS_WIDTH-1:0] DDR_WORD_LIMIT = 32'h2000_0000
) (
    input  wire logic                    i_core_clk,
    input  wire logic                    i_core_rst_n,
    input  wire logic                    i_clear_error,
    input  wire logic                    i_soft_abort,
    input  wire logic                    i_cmd_valid,
    output logic                         o_cmd_ready,
    input  wire logic [ADDRESS_WIDTH-1:0] i_cmd_word_addr,
    input  wire logic [7:0]              i_cmd_words,
    output logic                         o_data_valid,
    input  wire logic                    i_data_ready,
    output logic [63:0]                  o_data,
    output logic                         o_data_last,
    output logic                         o_data_error,
    output logic [2:0]                   o_core_outstanding,
    output logic                         o_axi_busy_core,
    output logic                         o_cdc_busy,
    output logic                         o_idle,
    output logic                         o_sticky_error,
    output logic [3:0]                   o_error_code,

    input  wire logic                    i_mig_clk,
    input  wire logic                    i_mig_rst_n,
    input  wire logic                    i_calib_complete,
    output logic                         o_m_axi_arvalid,
    input  wire logic                    i_m_axi_arready,
    output logic [ADDRESS_WIDTH-1:0]     o_m_axi_araddr,
    output logic [7:0]                   o_m_axi_arlen,
    output logic [2:0]                   o_m_axi_arsize,
    output logic [1:0]                   o_m_axi_arburst,
    output logic                         o_m_axi_arid,
    output logic [3:0]                   o_m_axi_arcache,
    output logic [2:0]                   o_m_axi_arprot,
    output logic [3:0]                   o_m_axi_arqos,
    input  wire logic                    i_m_axi_rvalid,
    output logic                         o_m_axi_rready,
    input  wire logic [127:0]            i_m_axi_rdata,
    input  wire logic                    i_m_axi_rlast,
    input  wire logic [1:0]              i_m_axi_rresp,
    input  wire logic                    i_m_axi_rid,
    output logic [2:0]                   o_axi_outstanding
);
    localparam int CMD_WIDTH = ADDRESS_WIDTH + 8;
    localparam int RESP_WIDTH = 66;
    localparam logic [3:0] ERROR_NONE = 4'd0;
    localparam logic [3:0] ERROR_COMMAND = 4'd1;
    localparam logic [3:0] ERROR_AXI_RESPONSE = 4'd2;
    localparam logic [3:0] ERROR_AXI_LAST = 4'd3;
    localparam logic [3:0] ERROR_ABORT = 4'd4;
    localparam logic [3:0] ERROR_UNEXPECTED_R = 4'd5;

    logic cmd_fifo_write_ready;
    logic cmd_fifo_read_valid;
    logic cmd_fifo_read_ready;
    logic [CMD_WIDTH-1:0] cmd_fifo_read_data;
    logic resp_fifo_write_valid;
    logic resp_fifo_write_ready;
    logic [RESP_WIDTH-1:0] resp_fifo_write_data;
    logic resp_fifo_read_valid;
    logic resp_fifo_read_ready;
    logic [RESP_WIDTH-1:0] resp_fifo_read_data;

    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic calib_core_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic calib_core_sync2_q;
    logic abort_toggle_core_q;
    logic soft_abort_q;
    logic clear_toggle_core_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic abort_toggle_mig_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic abort_toggle_mig_sync2_q;
    logic abort_toggle_mig_seen_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic abort_toggle_ack_core_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic abort_toggle_ack_core_sync2_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic clear_toggle_mig_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic clear_toggle_mig_sync2_q;
    logic clear_toggle_mig_seen_q;
    logic clear_request_pending_q;
    logic axi_busy_mig_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic axi_busy_core_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic axi_busy_core_sync2_q;
    logic core_outstanding_nonzero_core_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic core_outstanding_nonzero_mig_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic core_outstanding_nonzero_mig_sync2_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic mig_error_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic mig_error_sync2_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic [3:0] mig_error_code_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic [3:0] mig_error_code_sync2_q;
    logic core_error_q;
    logic [3:0] core_error_code_q;
    logic [2:0] core_outstanding_q;
    logic [2:0] core_outstanding_next;
    logic command_legal;
    logic [ADDRESS_WIDTH:0] command_end;
    logic command_enqueue;
    logic response_retire;

    logic command_active_q;
    logic [ADDRESS_WIDTH-1:0] current_word_q;
    logic [8:0] words_remaining_q;
    logic segment_active_q;
    logic [8:0] segment_words_q;
    logic [8:0] segment_words_emitted_q;
    logic [8:0] segment_axi_beats_q;
    logic [8:0] r_beats_received_q;
    logic segment_leading_skip_q;
    logic arvalid_q;
    logic [ADDRESS_WIDTH-1:0] araddr_q;
    logic [7:0] arlen_q;
    logic r_buffer_valid_q;
    logic [127:0] r_buffer_data_q;
    logic r_buffer_low_pending_q;
    logic r_buffer_high_pending_q;
    logic error_response_pending_q;
    logic abort_drain_mig_q;
    logic mig_error_q;
    logic [3:0] mig_error_code_q;

    logic [ADDRESS_WIDTH-1:0] current_byte_addr;
    logic [ADDRESS_WIDTH-1:0] segment_aligned_addr;
    logic [12:0] bytes_to_boundary;
    logic [8:0] axi_beats_to_boundary;
    logic [9:0] words_to_boundary;
    logic [8:0] segment_words_calc;
    logic [8:0] segment_axi_beats_calc;
    logic [9:0] segment_word_span;
    logic [9:0] segment_word_span_rounded;
    logic expected_axi_last;
    logic first_axi_beat;
    logic [9:0] segment_words_left_for_buffer;
    logic low_needed_for_r;
    logic high_needed_for_r;
    logic ar_handshake;
    logic r_handshake;
    logic resp_write_handshake;

    assign command_end = {1'b0, i_cmd_word_addr} +
                         (ADDRESS_WIDTH+1)'(i_cmd_words);
    assign command_legal = i_cmd_words != 0 &&
        i_cmd_words <= 8'(MAX_BURST_WORDS) &&
        command_end <= {1'b0, DDR_WORD_LIMIT};
    assign o_cmd_ready = calib_core_sync2_q && !o_sticky_error &&
        !i_soft_abort && !o_axi_busy_core &&
        (command_legal ?
            (core_outstanding_q < 3'(MAX_OUTSTANDING) && cmd_fifo_write_ready) :
            1'b1);
    assign command_enqueue = i_cmd_valid && o_cmd_ready && command_legal;
    assign response_retire = o_data_valid && i_data_ready && o_data_last;
    assign core_outstanding_next = command_enqueue && !response_retire ?
        core_outstanding_q + 1'b1 :
        !command_enqueue && response_retire ? core_outstanding_q - 1'b1 :
        core_outstanding_q;

    assign o_data_valid = resp_fifo_read_valid;
    assign resp_fifo_read_ready = i_data_ready;
    assign o_data = resp_fifo_read_data[63:0];
    assign o_data_last = resp_fifo_read_data[64];
    assign o_data_error = resp_fifo_read_data[65];
    assign o_core_outstanding = core_outstanding_q;
    assign o_axi_busy_core = axi_busy_core_sync2_q;
    assign o_cdc_busy = core_outstanding_q != 0;
    assign o_idle = core_outstanding_q == 0 && !o_data_valid &&
                    !i_cmd_valid && !o_axi_busy_core &&
                    !clear_request_pending_q;
    assign o_sticky_error = core_error_q || mig_error_sync2_q;
    assign o_error_code = core_error_q ? core_error_code_q :
                          mig_error_sync2_q ? mig_error_code_sync2_q : ERROR_NONE;

    unified_async_fifo #(.WIDTH(CMD_WIDTH), .DEPTH(4)) u_command_fifo (
        .i_write_clk(i_core_clk), .i_write_rst_n(i_core_rst_n),
        .i_write_valid(command_enqueue), .o_write_ready(cmd_fifo_write_ready),
        .i_write_data({i_cmd_words, i_cmd_word_addr}),
        .i_read_clk(i_mig_clk), .i_read_rst_n(i_mig_rst_n),
        .o_read_valid(cmd_fifo_read_valid), .i_read_ready(cmd_fifo_read_ready),
        .o_read_data(cmd_fifo_read_data)
    );

    unified_async_fifo #(.WIDTH(RESP_WIDTH), .DEPTH(8)) u_response_fifo (
        .i_write_clk(i_mig_clk), .i_write_rst_n(i_mig_rst_n),
        .i_write_valid(resp_fifo_write_valid),
        .o_write_ready(resp_fifo_write_ready),
        .i_write_data(resp_fifo_write_data),
        .i_read_clk(i_core_clk), .i_read_rst_n(i_core_rst_n),
        .o_read_valid(resp_fifo_read_valid),
        .i_read_ready(resp_fifo_read_ready),
        .o_read_data(resp_fifo_read_data)
    );

    // Core-side status contributes to scratchpad memory control. Keep the
    // complete core-domain control state on a synchronous reset boundary.
    always_ff @(posedge i_core_clk) begin
        if (!i_core_rst_n) begin
            calib_core_sync1_q <= 1'b0;
            calib_core_sync2_q <= 1'b0;
            mig_error_sync1_q <= 1'b0;
            mig_error_sync2_q <= 1'b0;
            mig_error_code_sync1_q <= ERROR_NONE;
            mig_error_code_sync2_q <= ERROR_NONE;
            core_error_q <= 1'b0;
            core_error_code_q <= ERROR_NONE;
            core_outstanding_q <= '0;
            core_outstanding_nonzero_core_q <= 1'b0;
            abort_toggle_core_q <= 1'b0;
            soft_abort_q <= 1'b0;
            clear_toggle_core_q <= 1'b0;
            abort_toggle_ack_core_sync1_q <= 1'b0;
            abort_toggle_ack_core_sync2_q <= 1'b0;
            clear_request_pending_q <= 1'b0;
            axi_busy_core_sync1_q <= 1'b0;
            axi_busy_core_sync2_q <= 1'b0;
        end else begin
            calib_core_sync1_q <= i_calib_complete;
            calib_core_sync2_q <= calib_core_sync1_q;
            mig_error_sync1_q <= mig_error_q;
            mig_error_sync2_q <= mig_error_sync1_q;
            mig_error_code_sync1_q <= mig_error_code_q;
            mig_error_code_sync2_q <= mig_error_code_sync1_q;
            abort_toggle_ack_core_sync1_q <= abort_toggle_mig_seen_q;
            abort_toggle_ack_core_sync2_q <=
                abort_toggle_ack_core_sync1_q;

            soft_abort_q <= i_soft_abort;
            axi_busy_core_sync1_q <= axi_busy_mig_q;
            axi_busy_core_sync2_q <= axi_busy_core_sync1_q;
            core_outstanding_q <= core_outstanding_next;
            core_outstanding_nonzero_core_q <= core_outstanding_next != 0;

            if (i_soft_abort && !soft_abort_q)
                abort_toggle_core_q <= ~abort_toggle_core_q;

            // A clear requested immediately after physical drain can otherwise
            // overtake the abort toggle in the MIG clock domain. Hold it until
            // MIG has observed the current abort epoch, then send one clear
            // toggle. The pending bit also keeps the bridge non-idle.
            if (i_clear_error)
                clear_request_pending_q <= 1'b1;
            if (clear_request_pending_q && core_outstanding_q == 0 &&
                !o_axi_busy_core && !o_data_valid &&
                abort_toggle_ack_core_sync2_q == abort_toggle_core_q) begin
                clear_toggle_core_q <= ~clear_toggle_core_q;
                clear_request_pending_q <= 1'b0;
            end

            if (i_clear_error && core_outstanding_q == 0 &&
                !o_axi_busy_core && !o_data_valid) begin
                core_error_q <= 1'b0;
                core_error_code_q <= ERROR_NONE;
            end
            if (i_cmd_valid && o_cmd_ready && !command_legal) begin
                core_error_q <= 1'b1;
                core_error_code_q <= ERROR_COMMAND;
            end

        end
    end

    always_ff @(posedge i_mig_clk or negedge i_mig_rst_n) begin
        if (!i_mig_rst_n) begin
            abort_toggle_mig_sync1_q <= 1'b0;
            abort_toggle_mig_sync2_q <= 1'b0;
            clear_toggle_mig_sync1_q <= 1'b0;
            clear_toggle_mig_sync2_q <= 1'b0;
            axi_busy_mig_q <= 1'b0;
            core_outstanding_nonzero_mig_sync1_q <= 1'b0;
            core_outstanding_nonzero_mig_sync2_q <= 1'b0;
        end else begin
            abort_toggle_mig_sync1_q <= abort_toggle_core_q;
            abort_toggle_mig_sync2_q <= abort_toggle_mig_sync1_q;
            clear_toggle_mig_sync1_q <= clear_toggle_core_q;
            clear_toggle_mig_sync2_q <= clear_toggle_mig_sync1_q;
            axi_busy_mig_q <= command_active_q || segment_active_q ||
                              arvalid_q || abort_drain_mig_q;
            core_outstanding_nonzero_mig_sync1_q <=
                core_outstanding_nonzero_core_q;
            core_outstanding_nonzero_mig_sync2_q <=
                core_outstanding_nonzero_mig_sync1_q;
        end
    end

    assign current_byte_addr = current_word_q << 3;
    assign segment_aligned_addr = {
        current_byte_addr[ADDRESS_WIDTH-1:4], 4'b0000
    };
    assign bytes_to_boundary = 13'd4096 -
                               {1'b0, segment_aligned_addr[11:0]};
    assign axi_beats_to_boundary = bytes_to_boundary[12:4];
    assign words_to_boundary = {axi_beats_to_boundary, 1'b0} -
                               10'(current_word_q[0]);
    assign segment_words_calc =
        {1'b0, words_remaining_q} <= words_to_boundary ? words_remaining_q :
        words_to_boundary[8:0];
    assign segment_word_span = {1'b0, segment_words_calc} +
                               10'(current_word_q[0]);
    assign segment_word_span_rounded = segment_word_span + 1'b1;
    assign segment_axi_beats_calc = segment_word_span_rounded[9:1];
    assign ar_handshake = o_m_axi_arvalid && i_m_axi_arready;
    assign expected_axi_last = r_beats_received_q + 1'b1 >=
                               segment_axi_beats_q;
    assign first_axi_beat = r_beats_received_q == 0;
    assign segment_words_left_for_buffer =
        {1'b0, segment_words_q} - {1'b0, segment_words_emitted_q};
    assign low_needed_for_r = !(first_axi_beat && segment_leading_skip_q);
    assign high_needed_for_r = low_needed_for_r ?
        (segment_words_left_for_buffer >= 2) :
        (segment_words_left_for_buffer >= 1);

    assign o_m_axi_arvalid = arvalid_q;
    assign o_m_axi_araddr = araddr_q;
    assign o_m_axi_arlen = arlen_q;
    assign o_m_axi_arsize = 3'd4;
    assign o_m_axi_arburst = 2'b01;
    assign o_m_axi_arid = 1'b0;
    assign o_m_axi_arcache = 4'b0011;
    assign o_m_axi_arprot = 3'b000;
    assign o_m_axi_arqos = 4'b0000;
    assign o_m_axi_rready = segment_active_q &&
        (abort_drain_mig_q || mig_error_q || !r_buffer_valid_q);
    assign r_handshake = i_m_axi_rvalid && o_m_axi_rready;
    assign o_axi_outstanding = segment_active_q || arvalid_q ? 3'd1 : 3'd0;

    assign resp_fifo_write_valid = error_response_pending_q ||
        (!abort_drain_mig_q && !mig_error_q && r_buffer_valid_q &&
         (r_buffer_low_pending_q || r_buffer_high_pending_q));
    assign resp_fifo_write_data = error_response_pending_q ?
        {1'b1, 1'b1, 64'd0} :
        {
            1'b0,
            words_remaining_q == 1,
            r_buffer_low_pending_q ? r_buffer_data_q[63:0] :
                                     r_buffer_data_q[127:64]
        };
    assign resp_write_handshake = resp_fifo_write_valid &&
                                  resp_fifo_write_ready;
    assign cmd_fifo_read_ready = i_calib_complete && !command_active_q &&
        !segment_active_q && !arvalid_q && !error_response_pending_q &&
        (abort_drain_mig_q || !mig_error_q);

    always_ff @(posedge i_mig_clk or negedge i_mig_rst_n) begin
        if (!i_mig_rst_n) begin
            command_active_q <= 1'b0;
            current_word_q <= '0;
            words_remaining_q <= '0;
            segment_active_q <= 1'b0;
            segment_words_q <= '0;
            segment_words_emitted_q <= '0;
            segment_axi_beats_q <= '0;
            r_beats_received_q <= '0;
            segment_leading_skip_q <= 1'b0;
            arvalid_q <= 1'b0;
            araddr_q <= '0;
            arlen_q <= '0;
            r_buffer_valid_q <= 1'b0;
            r_buffer_data_q <= '0;
            r_buffer_low_pending_q <= 1'b0;
            r_buffer_high_pending_q <= 1'b0;
            error_response_pending_q <= 1'b0;
            abort_drain_mig_q <= 1'b0;
            mig_error_q <= 1'b0;
            mig_error_code_q <= ERROR_NONE;
            abort_toggle_mig_seen_q <= 1'b0;
            clear_toggle_mig_seen_q <= 1'b0;
        end else begin
            abort_toggle_mig_seen_q <= abort_toggle_mig_sync2_q;
            if (cmd_fifo_read_valid && cmd_fifo_read_ready) begin
                if (abort_drain_mig_q) begin
                    error_response_pending_q <= 1'b1;
                end else begin
                    command_active_q <= 1'b1;
                    current_word_q <= cmd_fifo_read_data[ADDRESS_WIDTH-1:0];
                    words_remaining_q <= {1'b0, cmd_fifo_read_data[
                        CMD_WIDTH-1 -: 8]};
                end
            end

            if (command_active_q && !segment_active_q && !arvalid_q &&
                !mig_error_q) begin
                arvalid_q <= 1'b1;
                araddr_q <= segment_aligned_addr;
                arlen_q <= segment_axi_beats_calc[7:0] - 1'b1;
                segment_words_q <= segment_words_calc;
                segment_axi_beats_q <= segment_axi_beats_calc;
                segment_leading_skip_q <= current_word_q[0];
                segment_words_emitted_q <= '0;
            end
            if (ar_handshake) begin
                arvalid_q <= 1'b0;
                segment_active_q <= 1'b1;
                r_beats_received_q <= '0;
            end

            if (i_m_axi_rvalid && !segment_active_q) begin
                mig_error_q <= 1'b1;
                mig_error_code_q <= ERROR_UNEXPECTED_R;
                error_response_pending_q <= command_active_q;
            end

            if (r_handshake) begin
                if (abort_drain_mig_q) begin
                    r_buffer_valid_q <= 1'b0;
                    r_buffer_low_pending_q <= 1'b0;
                    r_buffer_high_pending_q <= 1'b0;
                    if (i_m_axi_rlast) begin
                        segment_active_q <= 1'b0;
                        command_active_q <= 1'b0;
                        error_response_pending_q <= 1'b1;
                    end
                end else begin
                    r_beats_received_q <= r_beats_received_q + 1'b1;
                    if (i_m_axi_rresp != 2'b00 || i_m_axi_rid != 1'b0) begin
                    mig_error_q <= 1'b1;
                    mig_error_code_q <= ERROR_AXI_RESPONSE;
                    error_response_pending_q <= 1'b1;
                    end else if (i_m_axi_rlast != expected_axi_last) begin
                    mig_error_q <= 1'b1;
                    mig_error_code_q <= ERROR_AXI_LAST;
                    error_response_pending_q <= 1'b1;
                    end else begin
                    r_buffer_valid_q <= 1'b1;
                    r_buffer_data_q <= i_m_axi_rdata;
                    r_buffer_low_pending_q <= low_needed_for_r;
                    r_buffer_high_pending_q <= high_needed_for_r;
                    end
                end
            end

            if (resp_write_handshake) begin
                if (error_response_pending_q) begin
                    error_response_pending_q <= 1'b0;
                end else begin
                    words_remaining_q <= words_remaining_q - 1'b1;
                    current_word_q <= current_word_q + 1'b1;
                    segment_words_emitted_q <=
                        segment_words_emitted_q + 1'b1;
                    if (r_buffer_low_pending_q) begin
                        r_buffer_low_pending_q <= 1'b0;
                        if (!r_buffer_high_pending_q)
                            r_buffer_valid_q <= 1'b0;
                    end else begin
                        r_buffer_high_pending_q <= 1'b0;
                        r_buffer_valid_q <= 1'b0;
                    end

                    if (segment_words_emitted_q + 1'b1 >= segment_words_q)
                        segment_active_q <= 1'b0;
                    if (words_remaining_q == 1)
                        command_active_q <= 1'b0;
                end
            end

            if (abort_toggle_mig_sync2_q != abort_toggle_mig_seen_q) begin
                abort_drain_mig_q <= 1'b1;
                mig_error_q <= 1'b1;
                mig_error_code_q <= ERROR_ABORT;
                r_buffer_valid_q <= 1'b0;
                r_buffer_low_pending_q <= 1'b0;
                r_buffer_high_pending_q <= 1'b0;
                if (segment_active_q || ar_handshake) begin
                    segment_active_q <= 1'b1;
                    arvalid_q <= 1'b0;
                end else if (command_active_q || arvalid_q) begin
                    command_active_q <= 1'b0;
                    arvalid_q <= 1'b0;
                    error_response_pending_q <= 1'b1;
                end
            end

            // A command FIFO read can coincide with the first observed abort
            // toggle. In that race, command_active_q becomes set after the
            // abort branch sampled its old value. Consume that command on the
            // following drain cycle and return one terminal error response;
            // otherwise it can neither launch AR (mig_error_q is set) nor let
            // the FIFO advance.
            if (abort_drain_mig_q && command_active_q &&
                !segment_active_q && !arvalid_q &&
                !error_response_pending_q) begin
                command_active_q <= 1'b0;
                error_response_pending_q <= 1'b1;
            end

            if (abort_drain_mig_q &&
                !core_outstanding_nonzero_mig_sync2_q &&
                !cmd_fifo_read_valid && !command_active_q &&
                !segment_active_q && !arvalid_q &&
                !error_response_pending_q && !r_buffer_valid_q)
                abort_drain_mig_q <= 1'b0;

            if (clear_toggle_mig_sync2_q != clear_toggle_mig_seen_q &&
                !abort_drain_mig_q && !command_active_q &&
                !segment_active_q && !arvalid_q &&
                !error_response_pending_q && !r_buffer_valid_q) begin
                clear_toggle_mig_seen_q <= clear_toggle_mig_sync2_q;
                mig_error_q <= 1'b0;
                mig_error_code_q <= ERROR_NONE;
            end
        end
    end

endmodule

`default_nettype wire
