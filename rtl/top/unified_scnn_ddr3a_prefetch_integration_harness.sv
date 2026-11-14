`default_nettype none

// Verification harness that closes the existing VGG-9 weight loader onto the
// read-only NetFPGA-SUME DDR3A bridge. It is not the final unified core.
module unified_scnn_ddr3a_prefetch_integration_harness #(
    parameter int ADDRESS_WIDTH = 32,
    parameter int BANK_WORDS = 8640,
    parameter int TAG_WIDTH = 16,
    parameter int MAX_BURST_WORDS = 64,
    parameter int MAX_OUTSTANDING = 4,
    parameter int DESCRIPTOR_LIMIT = 3,
    parameter logic [ADDRESS_WIDTH-1:0] DDR_WORD_LIMIT = 32'd2309920
) (
    input  logic                         i_core_clk,
    input  logic                         i_core_rst_n,
    input  logic                         i_mig_clk,
    input  logic                         i_mig_rst_n,
    input  logic                         i_calib_complete,
    input  logic                         i_clear_error,
    input  logic                         i_soft_abort,

    input  logic                         i_start,
    output logic                         o_start_ready,
    output logic                         o_busy,
    output logic                         o_done,
    output logic                         o_idle,

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

    // Verification-only idle diagnostic command. This exercises odd Q64
    // starts without changing the authoritative VGG-9 descriptor schedule.
    input  logic                         i_probe_cmd_valid,
    output logic                         o_probe_cmd_ready,
    input  logic [ADDRESS_WIDTH-1:0]     i_probe_word_addr,
    input  logic [7:0]                   i_probe_words,
    output logic                         o_probe_data_valid,
    input  logic                         i_probe_data_ready,
    output logic [63:0]                  o_probe_data,
    output logic                         o_probe_data_last,
    output logic                         o_probe_data_error,

    // Verification-only response throttling between the bridge and loader.
    input  logic                         i_force_core_data_stall,

    output logic                         o_prefetch_compute_overlap_seen,
    output logic                         o_prefetch_cmd_fire,
    output logic [ADDRESS_WIDTH-1:0]     o_prefetch_cmd_word_addr,
    output logic [7:0]                   o_prefetch_cmd_words,
    output logic [2:0]                   o_core_outstanding,
    output logic                         o_cdc_busy,
    output logic [2:0]                   o_axi_outstanding,
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
    logic loader_error;
    logic loader_cmd_valid;
    logic loader_cmd_ready;
    logic [ADDRESS_WIDTH-1:0] loader_cmd_word_addr;
    logic [7:0] loader_cmd_words;
    logic loader_data_valid;
    logic loader_data_ready;
    logic bridge_cmd_valid;
    logic bridge_cmd_ready;
    logic [ADDRESS_WIDTH-1:0] bridge_cmd_word_addr;
    logic [7:0] bridge_cmd_words;
    logic bridge_data_valid;
    logic bridge_data_ready;
    logic [63:0] bridge_data;
    logic bridge_data_last;
    logic bridge_data_error;
    logic bridge_axi_busy_core_unused;
    logic bridge_idle;
    logic bridge_sticky_error;
    logic [3:0] bridge_error_code;
    logic probe_active_q;
    logic probe_accept;
    logic probe_retire;
    logic calib_core_sync1_q;
    logic calib_core_sync2_q;

    assign o_start_ready = calib_core_sync2_q && !loader_busy &&
                           !probe_active_q && bridge_idle && !loader_error &&
                           !bridge_sticky_error;
    assign loader_start = i_start && o_start_ready;

    // Loader owns the bridge for its complete transaction lifetime. A probe is
    // accepted only while the loader and bridge are both idle.
    assign bridge_cmd_valid = loader_busy ? loader_cmd_valid :
                              (i_probe_cmd_valid && !probe_active_q);
    assign bridge_cmd_word_addr = loader_busy ? loader_cmd_word_addr :
                                  i_probe_word_addr;
    assign bridge_cmd_words = loader_busy ? loader_cmd_words : i_probe_words;
    assign loader_cmd_ready = loader_busy && bridge_cmd_ready;
    assign o_probe_cmd_ready = !loader_busy && !probe_active_q && bridge_cmd_ready;
    assign probe_accept = i_probe_cmd_valid && o_probe_cmd_ready;

    // Gate valid and ready together during an injected core-side stall so the
    // scratchpad cannot consume the same retained bridge word more than once.
    assign loader_data_valid = bridge_data_valid && loader_busy &&
                               !i_force_core_data_stall;
    assign bridge_data_ready = loader_busy ?
        (loader_data_ready && !i_force_core_data_stall) :
        (probe_active_q && i_probe_data_ready);
    assign o_probe_data_valid = bridge_data_valid && probe_active_q;
    assign o_probe_data = bridge_data;
    assign o_probe_data_last = bridge_data_last;
    assign o_probe_data_error = bridge_data_error;
    assign probe_retire = o_probe_data_valid && i_probe_data_ready &&
                          o_probe_data_last;

    assign o_busy = loader_busy || probe_active_q || !bridge_idle;
    assign o_idle = !loader_busy && !probe_active_q && bridge_idle;
    assign o_done = loader_done;
    assign o_sticky_error = loader_error || bridge_sticky_error;
    assign o_error_code = bridge_sticky_error ? bridge_error_code :
                          loader_error ? 4'hf : 4'h0;
    assign o_prefetch_cmd_fire = loader_cmd_valid && loader_cmd_ready;
    assign o_prefetch_cmd_word_addr = loader_cmd_word_addr;
    assign o_prefetch_cmd_words = loader_cmd_words;

    always_ff @(posedge i_core_clk or negedge i_core_rst_n) begin
        if (!i_core_rst_n) begin
            probe_active_q <= 1'b0;
            calib_core_sync1_q <= 1'b0;
            calib_core_sync2_q <= 1'b0;
        end else begin
            calib_core_sync1_q <= i_calib_complete;
            calib_core_sync2_q <= calib_core_sync1_q;
            if (probe_accept)
                probe_active_q <= 1'b1;
            if (probe_retire)
                probe_active_q <= 1'b0;
        end
    end

    unified_scnn_vgg9_weight_loader #(
        .ADDRESS_WIDTH(ADDRESS_WIDTH),
        .BANK_WORDS(BANK_WORDS),
        .TAG_WIDTH(TAG_WIDTH),
        .MAX_BURST_BEATS(MAX_BURST_WORDS),
        .MAX_OUTSTANDING(MAX_OUTSTANDING),
        .DESCRIPTOR_LIMIT(DESCRIPTOR_LIMIT)
    ) u_loader (
        .i_clk(i_core_clk),
        .i_rst_n(i_core_rst_n),
        .i_start(loader_start),
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
        .o_compute_source_stride_q64_words(o_compute_source_stride_q64_words),
        .o_compute_tile_tag(o_compute_tile_tag),
        .i_read_valid(i_read_valid),
        .o_read_ready(o_read_ready),
        .i_read_word_addr0(i_read_word_addr0),
        .i_read_word_addr1(i_read_word_addr1),
        .o_read_valid(o_read_valid),
        .i_read_result_ready(i_read_result_ready),
        .o_read_data0(o_read_data0),
        .o_read_data1(o_read_data1),
        .o_prefetch_compute_overlap_seen(o_prefetch_compute_overlap_seen),
        .o_error(loader_error)
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
        .i_soft_abort(i_soft_abort),
        .i_cmd_valid(bridge_cmd_valid),
        .o_cmd_ready(bridge_cmd_ready),
        .i_cmd_word_addr(bridge_cmd_word_addr),
        .i_cmd_words(bridge_cmd_words),
        .o_data_valid(bridge_data_valid),
        .i_data_ready(bridge_data_ready),
        .o_data(bridge_data),
        .o_data_last(bridge_data_last),
        .o_data_error(bridge_data_error),
        .o_core_outstanding(o_core_outstanding),
        .o_cdc_busy(o_cdc_busy),
        .o_axi_busy_core(bridge_axi_busy_core_unused),
        .o_idle(bridge_idle),
        .o_sticky_error(bridge_sticky_error),
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
