`default_nettype none

// Final-top instantiable DDR3A service leaf. It contains exactly one VGG-9
// descriptor prefetch/scratchpad chain and one 64-bit-core to 128-bit-AXI
// bridge. No compute engine is instantiated here.
module unified_scnn_ddr3a_core_service_v1 #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int BANK_WORDS = 8640,
    parameter int TAG_WIDTH = 16,
    parameter int MAX_BURST_WORDS = 64,
    parameter int MAX_OUTSTANDING = 4,
    parameter int DESCRIPTOR_LIMIT = 0,
    parameter bit WAVE_GROUP_DESCRIPTORS = 1'b0,
    parameter logic [ADDRESS_WIDTH-1:0] DDR_WORD_LIMIT = 32'd2309920
) (
    input  logic                         i_core_clk,
    input  logic                         i_core_rst_n,
    input  logic                         i_mig_clk,
    input  logic                         i_mig_rst_n,
    input  logic                         i_calib_complete,
    input  logic                         i_clear_error,
    input  logic                         i_soft_abort,

    input  logic                         i_prefetch_start,
    output logic                         o_prefetch_start_ready,
    output logic                         o_busy,
    output logic                         o_idle,
    output logic                         o_done,

    // Direct ordered 64-bit core read service. The prefetch run owns this
    // shared bridge for its complete lifetime.
    input  logic                         i_core_req_valid,
    output logic                         o_core_req_ready,
    input  logic [ADDRESS_WIDTH-1:0]     i_core_req_word_addr,
    input  logic [7:0]                   i_core_req_words,
    output logic                         o_core_rsp_valid,
    input  logic                         i_core_rsp_ready,
    output logic [63:0]                  o_core_rsp_data,
    output logic                         o_core_rsp_last,
    output logic                         o_core_rsp_error,

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

    output logic                         o_cmd_pending,
    output logic                         o_response_pending,
    output logic                         o_cdc_busy,
    output logic [2:0]                   o_outstanding,
    output logic [2:0]                   o_axi_outstanding,
    output logic                         o_axi_busy_core,
    output logic                         o_prefetch_busy,
    output logic                         o_scratchpad_busy,
    output logic                         o_prefetch_compute_overlap_seen,
    output logic                         o_sticky_error,
    output logic [3:0]                   o_error_code,

    output logic                         o_m_axi_arvalid,
    input  logic                         i_m_axi_arready,
    output logic [ADDRESS_WIDTH-1:0]     o_m_axi_araddr,
    output logic [7:0]                   o_m_axi_arlen,
    output logic [2:0]                   o_m_axi_arsize,
    output logic [1:0]                   o_m_axi_arburst,
    output logic                         o_m_axi_arid,
    output logic [3:0]                   o_m_axi_arcache,
    output logic [2:0]                   o_m_axi_arprot,
    output logic [3:0]                   o_m_axi_arqos,
    input  logic                         i_m_axi_rvalid,
    output logic                         o_m_axi_rready,
    input  logic [127:0]                 i_m_axi_rdata,
    input  logic                         i_m_axi_rlast,
    input  logic [1:0]                   i_m_axi_rresp,
    input  logic                         i_m_axi_rid
);

    logic loader_start;
    logic loader_busy;
    logic loader_done;
    logic loader_cmd_valid;
    logic loader_cmd_ready;
    logic [ADDRESS_WIDTH-1:0] loader_cmd_word_addr;
    logic [7:0] loader_cmd_words;
    logic loader_data_valid;
    logic loader_data_ready;
    logic loader_prefetch_busy;
    logic loader_scratchpad_busy;
    logic loader_error;
    logic [3:0] loader_error_code;

    logic bridge_cmd_valid;
    logic bridge_cmd_ready;
    logic [ADDRESS_WIDTH-1:0] bridge_cmd_word_addr;
    logic [7:0] bridge_cmd_words;
    logic bridge_data_valid;
    logic bridge_data_ready;
    logic [63:0] bridge_data;
    logic bridge_data_last;
    logic bridge_data_error;
    logic bridge_idle;
    logic bridge_error;
    logic [3:0] bridge_error_code;
    logic soft_abort_q;
    logic soft_abort_pulse;
    logic clear_in_progress_q;
    logic [7:0] clear_wait_q;
    logic calib_core_sync1_q;
    logic calib_core_sync2_q;
    logic start_accept;
    logic direct_owner;
    logic service_idle_raw;

    assign o_prefetch_start_ready = calib_core_sync2_q &&
        !clear_in_progress_q && !loader_busy &&
        !loader_error && !bridge_error && bridge_idle &&
        !i_core_req_valid && !o_cmd_pending && !o_response_pending &&
        !o_cdc_busy && (o_outstanding == 3'd0);
    assign start_accept = i_prefetch_start && o_prefetch_start_ready;
    assign loader_start = start_accept;
    assign direct_owner = !loader_busy && !i_prefetch_start;

    assign bridge_cmd_valid = loader_busy ? loader_cmd_valid :
                              direct_owner ? i_core_req_valid : 1'b0;
    assign bridge_cmd_word_addr = loader_busy ? loader_cmd_word_addr :
                                  i_core_req_word_addr;
    assign bridge_cmd_words = loader_busy ? loader_cmd_words :
                              i_core_req_words;
    assign loader_cmd_ready = loader_busy && bridge_cmd_ready;
    assign o_core_req_ready = direct_owner && !clear_in_progress_q &&
                              bridge_cmd_ready &&
                              !loader_error;

    assign loader_data_valid = loader_busy && bridge_data_valid;
    assign bridge_data_ready = loader_busy ? loader_data_ready :
                               i_core_rsp_ready;
    assign o_core_rsp_valid = !loader_busy && bridge_data_valid;
    assign o_core_rsp_data = bridge_data;
    assign o_core_rsp_last = bridge_data_last;
    assign o_core_rsp_error = bridge_data_error;

    assign o_cmd_pending = bridge_cmd_valid;
    assign o_response_pending = bridge_data_valid;
    assign o_prefetch_busy = loader_prefetch_busy;
    assign o_scratchpad_busy = loader_scratchpad_busy;
    assign o_done = loader_done;
    // A clear can retire loader-local state immediately while its memory or
    // MIG acknowledgement crosses clock boundaries. Mask the complete retired
    // service fault during that bounded handshake so the coordinator cannot
    // relatch an old component error on the following cycle.
    assign o_sticky_error = (loader_error || bridge_error) &&
                            !clear_in_progress_q;
    assign o_error_code = loader_error ? loader_error_code :
                          bridge_error ? bridge_error_code : 4'h0;
    assign service_idle_raw = !loader_busy && !o_cmd_pending &&
        !o_response_pending && !o_cdc_busy && (o_outstanding == 3'd0) &&
        !o_prefetch_busy && !o_scratchpad_busy && bridge_idle &&
        !o_axi_busy_core;
    assign o_idle = service_idle_raw && !clear_in_progress_q;
    assign o_busy = !o_idle;
    assign soft_abort_pulse = i_soft_abort && !soft_abort_q;

    // Clear state gates the prefetch scratchpad BRAM interface.
    always_ff @(posedge i_core_clk) begin
        if (!i_core_rst_n) begin
            calib_core_sync1_q <= 1'b0;
            calib_core_sync2_q <= 1'b0;
            soft_abort_q <= 1'b0;
            clear_in_progress_q <= 1'b0;
            clear_wait_q <= '0;
        end else begin
            calib_core_sync1_q <= i_calib_complete;
            calib_core_sync2_q <= calib_core_sync1_q;
            if (i_clear_error)
                soft_abort_q <= 1'b0;
            else if (i_soft_abort)
                soft_abort_q <= 1'b1;

            // Enter a closed clear transaction even if a late CDC error makes
            // the raw idle view change on the request edge. Child blocks clear
            // only at their own safe points; this wrapper remains busy until
            // both error views have retired or the bounded timeout expires.
            if (i_clear_error) begin
                clear_in_progress_q <= loader_error || bridge_error;
                clear_wait_q <= '0;
            end else if (clear_in_progress_q) begin
                if (!loader_error && !bridge_error) begin
                    clear_in_progress_q <= 1'b0;
                    clear_wait_q <= '0;
                end else if (&clear_wait_q) begin
                    clear_in_progress_q <= 1'b0;
                end else begin
                    clear_wait_q <= clear_wait_q + 1'b1;
                end
            end
        end
    end

    unified_scnn_vgg9_weight_loader_service_adapter_v1 #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .BANK_WORDS(BANK_WORDS),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_BURST_BEATS(MAX_BURST_WORDS),
        .MAX_OUTSTANDING(MAX_OUTSTANDING),
        .DESCRIPTOR_LIMIT(DESCRIPTOR_LIMIT),
        .WAVE_GROUP_DESCRIPTORS(WAVE_GROUP_DESCRIPTORS)
    ) u_loader_service (
        .i_clk(i_core_clk),
        .i_rst_n(i_core_rst_n),
        .i_start(loader_start),
        // System cleanup may hold abort high. The loader/prefetch cancel FSMs
        // consume one pulse, then must be allowed to drain to physical idle.
        .i_abort(soft_abort_pulse),
        .i_clear_error(i_clear_error),
        .o_busy(loader_busy),
        .o_done(loader_done),
        .i_region_base_word('0),
        .i_region_limit_word(DDR_WORD_LIMIT),
        .o_ddr_cmd_valid(loader_cmd_valid),
        .i_ddr_cmd_ready(loader_cmd_ready),
        .o_ddr_cmd_word_addr(loader_cmd_word_addr),
        .o_ddr_cmd_beats(loader_cmd_words),
        .i_ddr_data_valid(loader_data_valid),
        .o_ddr_data_ready(loader_data_ready),
        .i_ddr_data(bridge_data),
        .i_ddr_data_last(bridge_data_last),
        .i_ddr_data_error(bridge_data_error),
        .o_compute_tile_valid(o_compute_tile_valid),
        .i_compute_tile_done(i_compute_tile_done),
        .o_compute_layer_index(o_compute_layer_index),
        .o_compute_output_channel_base(o_compute_output_channel_base),
        .o_compute_output_channel_count(o_compute_output_channel_count),
        .o_compute_input_channel_base(o_compute_input_channel_base),
        .o_compute_input_channel_count(o_compute_input_channel_count),
        .o_compute_source_q64_word(o_compute_source_q64_word),
        .o_compute_total_q64_words(o_compute_total_q64_words),
        .o_compute_segment_q64_words(o_compute_segment_q64_words),
        .o_compute_source_stride_q64_words(
            o_compute_source_stride_q64_words
        ),
        .o_compute_tile_tag(o_compute_tile_tag),
        .i_read_valid(i_read_valid),
        .o_read_ready(o_read_ready),
        .i_read_word_addr0(i_read_word_addr0),
        .i_read_word_addr1(i_read_word_addr1),
        .o_read_valid(o_read_valid),
        .i_read_result_ready(i_read_result_ready),
        .o_read_data0(o_read_data0),
        .o_read_data1(o_read_data1),
        .o_prefetch_busy(loader_prefetch_busy),
        .o_scratchpad_busy(loader_scratchpad_busy),
        .o_prefetch_compute_overlap_seen(
            o_prefetch_compute_overlap_seen
        ),
        .o_sticky_error(loader_error),
        .o_error_code(loader_error_code)
    );

    unified_netfpga_ddr3a_axi_read_bridge #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .MAX_BURST_WORDS(MAX_BURST_WORDS),
        .MAX_OUTSTANDING(MAX_OUTSTANDING),
        .DDR_WORD_LIMIT(DDR_WORD_LIMIT)
    ) u_ddr3a_bridge (
        .i_core_clk(i_core_clk),
        .i_core_rst_n(i_core_rst_n),
        .i_clear_error(i_clear_error),
        .i_soft_abort(soft_abort_pulse),
        .i_cmd_valid(bridge_cmd_valid),
        .o_cmd_ready(bridge_cmd_ready),
        .i_cmd_word_addr(bridge_cmd_word_addr),
        .i_cmd_words(bridge_cmd_words),
        .o_data_valid(bridge_data_valid),
        .i_data_ready(bridge_data_ready),
        .o_data(bridge_data),
        .o_data_last(bridge_data_last),
        .o_data_error(bridge_data_error),
        .o_core_outstanding(o_outstanding),
        .o_axi_busy_core(o_axi_busy_core),
        .o_cdc_busy(o_cdc_busy),
        .o_idle(bridge_idle),
        .o_sticky_error(bridge_error),
        .o_error_code(bridge_error_code),
        .i_mig_clk(i_mig_clk),
        .i_mig_rst_n(i_mig_rst_n),
        .i_calib_complete(i_calib_complete),
        .o_m_axi_arvalid(o_m_axi_arvalid),
        .i_m_axi_arready(i_m_axi_arready),
        .o_m_axi_araddr(o_m_axi_araddr),
        .o_m_axi_arlen(o_m_axi_arlen),
        .o_m_axi_arsize(o_m_axi_arsize),
        .o_m_axi_arburst(o_m_axi_arburst),
        .o_m_axi_arid(o_m_axi_arid),
        .o_m_axi_arcache(o_m_axi_arcache),
        .o_m_axi_arprot(o_m_axi_arprot),
        .o_m_axi_arqos(o_m_axi_arqos),
        .i_m_axi_rvalid(i_m_axi_rvalid),
        .o_m_axi_rready(o_m_axi_rready),
        .i_m_axi_rdata(i_m_axi_rdata),
        .i_m_axi_rlast(i_m_axi_rlast),
        .i_m_axi_rresp(i_m_axi_rresp),
        .i_m_axi_rid(i_m_axi_rid),
        .o_axi_outstanding(o_axi_outstanding)
    );

endmodule

`default_nettype wire
