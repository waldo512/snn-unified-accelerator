`default_nettype none

// Small asynchronous FIFO. Vivado uses the vendor CDC primitive; portable
// simulation keeps the equivalent Gray-pointer model below. DEPTH must be a
// power of two and at least four.
module unified_async_fifo #(
    parameter int WIDTH = 8,
    parameter int DEPTH = 4
) (
    input  wire logic            i_write_clk,
    input  wire logic            i_write_rst_n,
    input  wire logic            i_write_valid,
    output logic                 o_write_ready,
    input  wire logic [WIDTH-1:0] i_write_data,
    input  wire logic            i_read_clk,
    input  wire logic            i_read_rst_n,
    output logic                 o_read_valid,
    input  wire logic            i_read_ready,
    output logic [WIDTH-1:0]     o_read_data
);
    logic fifo_rst;

    // Either domain reset invalidates the complete dual-clock transaction.
    // This also gives the XPM primitive the single reset contract it requires.
    assign fifo_rst = !i_write_rst_n || !i_read_rst_n;

`ifdef UNIFIED_USE_XPM_MEMORY
    localparam int XPM_FIFO_DEPTH = DEPTH < 16 ? 16 : DEPTH;

    logic xpm_full;
    logic xpm_empty;
    logic xpm_write_reset_busy;
    logic xpm_read_reset_busy;
    logic write_fire;
    logic read_fire;

    assign o_write_ready = !xpm_full && !xpm_write_reset_busy;
    assign o_read_valid = !xpm_empty && !xpm_read_reset_busy;
    assign write_fire = i_write_valid && o_write_ready;
    assign read_fire = o_read_valid && i_read_ready;

    // Vivado 2022.2 requires a minimum XPM FIFO depth of 16. The bridge still
    // enforces its architectural outstanding limit before this queue.
    xpm_fifo_async #(
        .FIFO_MEMORY_TYPE("distributed"),
        .ECC_MODE("no_ecc"),
        .RELATED_CLOCKS(0),
        .SIM_ASSERT_CHK(0),
        .CASCADE_HEIGHT(0),
        .FIFO_WRITE_DEPTH(XPM_FIFO_DEPTH),
        .WRITE_DATA_WIDTH(WIDTH),
        .WR_DATA_COUNT_WIDTH(1),
        .PROG_FULL_THRESH(10),
        .FULL_RESET_VALUE(0),
        .USE_ADV_FEATURES("0000"),
        .READ_MODE("fwft"),
        .FIFO_READ_LATENCY(0),
        .READ_DATA_WIDTH(WIDTH),
        .RD_DATA_COUNT_WIDTH(1),
        .PROG_EMPTY_THRESH(10),
        .DOUT_RESET_VALUE("0"),
        .CDC_SYNC_STAGES(2),
        .WAKEUP_TIME(0)
    ) u_xpm_async_fifo (
        .sleep(1'b0),
        .rst(fifo_rst),
        .wr_clk(i_write_clk),
        .wr_en(write_fire),
        .din(i_write_data),
        .full(xpm_full),
        .prog_full(),
        .wr_data_count(),
        .overflow(),
        .wr_rst_busy(xpm_write_reset_busy),
        .almost_full(),
        .wr_ack(),
        .rd_clk(i_read_clk),
        .rd_en(read_fire),
        .dout(o_read_data),
        .empty(xpm_empty),
        .prog_empty(),
        .rd_data_count(),
        .underflow(),
        .rd_rst_busy(xpm_read_reset_busy),
        .almost_empty(),
        .data_valid(),
        .injectsbiterr(1'b0),
        .injectdbiterr(1'b0),
        .sbiterr(),
        .dbiterr()
    );
`else
    localparam int ADDR_WIDTH = $clog2(DEPTH);
    localparam int PTR_WIDTH = ADDR_WIDTH + 1;

    logic [WIDTH-1:0] memory [0:DEPTH-1];
    logic [PTR_WIDTH-1:0] write_binary_q;
    logic [PTR_WIDTH-1:0] write_gray_q;
    logic [PTR_WIDTH-1:0] read_binary_q;
    logic [PTR_WIDTH-1:0] read_gray_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic [PTR_WIDTH-1:0] read_gray_write_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic [PTR_WIDTH-1:0] read_gray_write_sync2_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic [PTR_WIDTH-1:0] write_gray_read_sync1_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *)
    logic [PTR_WIDTH-1:0] write_gray_read_sync2_q;
    logic [PTR_WIDTH-1:0] write_binary_next;
    logic [PTR_WIDTH-1:0] write_gray_next;
    logic [PTR_WIDTH-1:0] read_binary_next;
    logic [PTR_WIDTH-1:0] read_gray_next;
    logic write_full_q;
    logic read_empty_q;
    logic write_fire;
    logic read_fire;

    assign o_write_ready = !write_full_q;
    assign o_read_valid = !read_empty_q;
    assign o_read_data = memory[read_binary_q[ADDR_WIDTH-1:0]];
    assign write_fire = i_write_valid && o_write_ready;
    assign read_fire = o_read_valid && i_read_ready;
    assign write_binary_next = write_binary_q + PTR_WIDTH'(write_fire);
    assign write_gray_next = (write_binary_next >> 1) ^ write_binary_next;
    assign read_binary_next = read_binary_q + PTR_WIDTH'(read_fire);
    assign read_gray_next = (read_binary_next >> 1) ^ read_binary_next;

    always_ff @(posedge i_write_clk or posedge fifo_rst) begin
        if (fifo_rst) begin
            write_binary_q <= '0;
            write_gray_q <= '0;
            read_gray_write_sync1_q <= '0;
            read_gray_write_sync2_q <= '0;
            write_full_q <= 1'b0;
        end else begin
            read_gray_write_sync1_q <= read_gray_q;
            read_gray_write_sync2_q <= read_gray_write_sync1_q;
            write_binary_q <= write_binary_next;
            write_gray_q <= write_gray_next;
            write_full_q <= write_gray_next == {
                ~read_gray_write_sync2_q[PTR_WIDTH-1:PTR_WIDTH-2],
                read_gray_write_sync2_q[PTR_WIDTH-3:0]
            };
        end
    end

    // Keep memory outside the asynchronously reset pointer process so Vivado
    // can infer the intended small dual-clock distributed storage.
    always_ff @(posedge i_write_clk) begin
        if (write_fire)
            memory[write_binary_q[ADDR_WIDTH-1:0]] <= i_write_data;
    end

    always_ff @(posedge i_read_clk or posedge fifo_rst) begin
        if (fifo_rst) begin
            read_binary_q <= '0;
            read_gray_q <= '0;
            write_gray_read_sync1_q <= '0;
            write_gray_read_sync2_q <= '0;
            read_empty_q <= 1'b1;
        end else begin
            write_gray_read_sync1_q <= write_gray_q;
            write_gray_read_sync2_q <= write_gray_read_sync1_q;
            read_binary_q <= read_binary_next;
            read_gray_q <= read_gray_next;
            read_empty_q <= read_gray_next == write_gray_read_sync2_q;
        end
    end
`endif

endmodule

`default_nettype wire
