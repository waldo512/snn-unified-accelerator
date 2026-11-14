`timescale 1ns/1ps
`default_nettype none

// One native FC W0 feature bank. Port A preserves the H40-wide inference row;
// port B serializes TP scalar reads and whole-row read/modify/write updates.
// The code memory is therefore the single source of truth for both paths.
module unified_fc_tp_w0_native_bank_ram #(
    parameter int HIDDEN_PER_TILE = 40,
    parameter int W0_BITS = 5,
    parameter int DEPTH = 289,
    parameter int CODE_WIDTH = 16,
    parameter int RESIDUAL_WIDTH = 16,
    parameter INIT_FILE = "",
    parameter RESIDUAL_INIT_FILE = "",
    parameter int ADDR_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH),
    parameter int LANE_WIDTH =
        (HIDDEN_PER_TILE <= 2) ? 1 : $clog2(HIDDEN_PER_TILE)
) (
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,
    input  logic                                  i_clear_error,

    input  logic                                  i_forward_enable,
    input  logic [ADDR_WIDTH-1:0]                 i_forward_addr,
    output logic [HIDDEN_PER_TILE*W0_BITS-1:0]    o_forward_row,

    input  logic                                  i_learning_read_valid,
    output logic                                  o_learning_read_ready,
    input  logic [ADDR_WIDTH-1:0]                 i_learning_read_addr,
    input  logic [LANE_WIDTH-1:0]                 i_learning_read_lane,
    output logic                                  o_learning_response_valid,
    input  logic                                  i_learning_response_ready,
    output logic signed [CODE_WIDTH-1:0]          o_learning_response_code,
    output logic signed [RESIDUAL_WIDTH-1:0]      o_learning_response_residual,

    input  logic                                  i_learning_write_valid,
    output logic                                  o_learning_write_ready,
    input  logic [ADDR_WIDTH-1:0]                 i_learning_write_addr,
    input  logic [LANE_WIDTH-1:0]                 i_learning_write_lane,
    input  logic signed [CODE_WIDTH-1:0]          i_learning_write_code,
    input  logic signed [RESIDUAL_WIDTH-1:0]      i_learning_write_residual,
    input  logic                                  i_learning_write_code_enable,
    input  logic                                  i_learning_write_residual_enable,
    output logic                                  o_learning_write_done,
    output logic                                  o_protocol_error
);
    localparam int ROW_BITS = HIDDEN_PER_TILE * W0_BITS;
    localparam int RESIDUAL_DEPTH = DEPTH * HIDDEN_PER_TILE;
    localparam int RESIDUAL_ADDR_WIDTH =
        (RESIDUAL_DEPTH <= 2) ? 1 : $clog2(RESIDUAL_DEPTH);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_READ_RESPONSE,
        STATE_WRITE_COMMIT
    } state_t;

`ifndef UNIFIED_USE_XPM_MEMORY
    (* ram_style = "block" *) logic [ROW_BITS-1:0] code_mem [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [RESIDUAL_WIDTH-1:0]
        residual_mem [0:RESIDUAL_DEPTH-1];
`endif
    state_t state_q;
    logic [ROW_BITS-1:0] learning_row_q;
    logic signed [RESIDUAL_WIDTH-1:0] learning_residual_q;
    logic [ADDR_WIDTH-1:0] selected_addr_q;
    logic [LANE_WIDTH-1:0] selected_lane_q;
    logic [RESIDUAL_ADDR_WIDTH-1:0] selected_residual_addr_q;
    logic signed [W0_BITS-1:0] write_code_q;
    logic signed [RESIDUAL_WIDTH-1:0] write_residual_q;
    logic write_code_enable_q;
    logic write_residual_enable_q;
    logic learning_port_enable;
    logic learning_port_write_code;
    logic [ADDR_WIDTH-1:0] learning_port_addr;
    logic [ROW_BITS-1:0] learning_port_write_data;
    logic [ROW_BITS-1:0] patched_row;
    logic [RESIDUAL_ADDR_WIDTH-1:0] request_residual_addr;
`ifdef UNIFIED_USE_XPM_MEMORY
    logic residual_memory_enable;
    logic residual_memory_write;
    logic [RESIDUAL_ADDR_WIDTH-1:0] residual_memory_addr;
`endif
    logic read_fire;
    logic write_fire;
    logic response_fire;
    logic write_code_fits;

    always_comb begin
        o_learning_read_ready = state_q == STATE_IDLE &&
            !i_learning_write_valid;
        o_learning_write_ready = state_q == STATE_IDLE &&
            !i_learning_read_valid;
        read_fire = i_learning_read_valid && o_learning_read_ready;
        write_fire = i_learning_write_valid && o_learning_write_ready;
        o_learning_response_valid = state_q == STATE_READ_RESPONSE;
        response_fire = o_learning_response_valid &&
            i_learning_response_ready;
        write_code_fits = i_learning_write_code == CODE_WIDTH'($signed(
            i_learning_write_code[W0_BITS-1:0]));

        o_learning_response_code = CODE_WIDTH'($signed(
            learning_row_q[selected_lane_q*W0_BITS +: W0_BITS]));
        o_learning_response_residual = learning_residual_q;

        patched_row = learning_row_q;
        patched_row[selected_lane_q*W0_BITS +: W0_BITS] =
            write_code_q;
        learning_port_enable = read_fire || write_fire ||
            state_q == STATE_WRITE_COMMIT;
        learning_port_write_code = state_q == STATE_WRITE_COMMIT &&
            write_code_enable_q;
        learning_port_addr = state_q == STATE_WRITE_COMMIT ?
            selected_addr_q :
            read_fire ? i_learning_read_addr : i_learning_write_addr;
        learning_port_write_data = patched_row;
        request_residual_addr = RESIDUAL_ADDR_WIDTH'(
            $unsigned(read_fire ? i_learning_read_addr :
                i_learning_write_addr) * HIDDEN_PER_TILE +
            $unsigned(read_fire ? i_learning_read_lane :
                i_learning_write_lane));
`ifdef UNIFIED_USE_XPM_MEMORY
        residual_memory_write = state_q == STATE_WRITE_COMMIT &&
            write_residual_enable_q;
        residual_memory_enable = read_fire || residual_memory_write;
        residual_memory_addr = residual_memory_write ?
            selected_residual_addr_q : request_residual_addr;
`endif
    end

`ifndef UNIFIED_USE_XPM_MEMORY
    initial begin
        o_forward_row = '0;
        learning_row_q = '0;
        learning_residual_q = '0;
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, code_mem);
    end
`endif

`ifdef UNIFIED_USE_XPM_MEMORY
    localparam XPM_CODE_INIT_FILE =
        (INIT_FILE == "" || INIT_FILE == "NONE") ? "none" : INIT_FILE;
    localparam integer XPM_CODE_USE_MEM_INIT =
        (INIT_FILE == "" || INIT_FILE == "NONE") ? 0 : 1;
    localparam XPM_RESIDUAL_INIT_FILE =
        (RESIDUAL_INIT_FILE == "" || RESIDUAL_INIT_FILE == "NONE") ?
        "none" : RESIDUAL_INIT_FILE;
    localparam integer XPM_RESIDUAL_USE_MEM_INIT =
        (RESIDUAL_INIT_FILE == "" || RESIDUAL_INIT_FILE == "NONE") ? 0 : 1;

    logic [ROW_BITS-1:0] xpm_code_forward_data;
    logic [ROW_BITS-1:0] xpm_code_learning_data;
    logic [RESIDUAL_WIDTH-1:0] xpm_residual_read_data;

    assign o_forward_row = xpm_code_forward_data;
    assign learning_row_q = xpm_code_learning_data;
    assign learning_residual_q = $signed(xpm_residual_read_data);

    xpm_memory_tdpram #(
        .MEMORY_SIZE(DEPTH * ROW_BITS),
        .MEMORY_PRIMITIVE("block"),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(XPM_CODE_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .USE_MEM_INIT(XPM_CODE_USE_MEM_INIT),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep"),
        .AUTO_SLEEP_TIME(0),
        .MESSAGE_CONTROL(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .MEMORY_OPTIMIZATION("true"),
        .CASCADE_HEIGHT(0),
        .SIM_ASSERT_CHK(0),
        .WRITE_PROTECT(1),
        .IGNORE_INIT_SYNTH(0),
        .WRITE_DATA_WIDTH_A(ROW_BITS),
        .READ_DATA_WIDTH_A(ROW_BITS),
        .BYTE_WRITE_WIDTH_A(ROW_BITS),
        .ADDR_WIDTH_A(ADDR_WIDTH),
        .READ_RESET_VALUE_A("0"),
        .READ_LATENCY_A(1),
        .WRITE_MODE_A("read_first"),
        .RST_MODE_A("SYNC"),
        .WRITE_DATA_WIDTH_B(ROW_BITS),
        .READ_DATA_WIDTH_B(ROW_BITS),
        .BYTE_WRITE_WIDTH_B(ROW_BITS),
        .ADDR_WIDTH_B(ADDR_WIDTH),
        .READ_RESET_VALUE_B("0"),
        .READ_LATENCY_B(1),
        .WRITE_MODE_B("read_first"),
        .RST_MODE_B("SYNC")
    ) u_code_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .rsta(1'b0),
        .ena(i_forward_enable),
        .regcea(1'b1),
        .wea(1'b0),
        .addra(i_forward_addr),
        .dina('0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .douta(xpm_code_forward_data),
        .sbiterra(),
        .dbiterra(),
        .clkb(i_clk),
        .rstb(1'b0),
        .enb(learning_port_enable),
        .regceb(1'b1),
        .web(learning_port_write_code),
        .addrb(learning_port_addr),
        .dinb(learning_port_write_data),
        .injectsbiterrb(1'b0),
        .injectdbiterrb(1'b0),
        .doutb(xpm_code_learning_data),
        .sbiterrb(),
        .dbiterrb()
    );

    xpm_memory_spram #(
        .MEMORY_SIZE(RESIDUAL_DEPTH * RESIDUAL_WIDTH),
        .MEMORY_PRIMITIVE("block"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(XPM_RESIDUAL_INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .USE_MEM_INIT(XPM_RESIDUAL_USE_MEM_INIT),
        .USE_MEM_INIT_MMI(0),
        .WAKEUP_TIME("disable_sleep"),
        .AUTO_SLEEP_TIME(0),
        .MESSAGE_CONTROL(0),
        .MEMORY_OPTIMIZATION("true"),
        .CASCADE_HEIGHT(0),
        .SIM_ASSERT_CHK(0),
        .WRITE_PROTECT(1),
        .IGNORE_INIT_SYNTH(0),
        .WRITE_DATA_WIDTH_A(RESIDUAL_WIDTH),
        .READ_DATA_WIDTH_A(RESIDUAL_WIDTH),
        .BYTE_WRITE_WIDTH_A(RESIDUAL_WIDTH),
        .ADDR_WIDTH_A(RESIDUAL_ADDR_WIDTH),
        .READ_RESET_VALUE_A("0"),
        .READ_LATENCY_A(1),
        .WRITE_MODE_A("read_first"),
        .RST_MODE_A("SYNC")
    ) u_residual_memory (
        .sleep(1'b0),
        .clka(i_clk),
        .rsta(1'b0),
        .ena(residual_memory_enable),
        .regcea(1'b1),
        .wea(residual_memory_write),
        .addra(residual_memory_addr),
        .dina(write_residual_q),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0),
        .douta(xpm_residual_read_data),
        .sbiterra(),
        .dbiterra()
    );
`else
    generate
        if (RESIDUAL_INIT_FILE == "NONE") begin : g_default_zero_residual
            // 7-series block RAM configuration defaults to all-zero INIT.
        end else if (RESIDUAL_INIT_FILE != "") begin : g_file_residual_init
            initial $readmemb(RESIDUAL_INIT_FILE, residual_mem);
        end else begin : g_simulation_zero_residual
            initial begin
                for (int parameter_index = 0;
                     parameter_index < RESIDUAL_DEPTH; parameter_index++)
                    residual_mem[parameter_index] = '0;
            end
        end
    endgenerate
`endif

`ifndef UNIFIED_USE_XPM_MEMORY
    always_ff @(posedge i_clk) begin
        if (i_forward_enable)
            o_forward_row <= code_mem[i_forward_addr];
        if (learning_port_enable) begin
            learning_row_q <= code_mem[learning_port_addr];
            if (learning_port_write_code)
                code_mem[learning_port_addr] <= learning_port_write_data;
        end
        if (read_fire)
            learning_residual_q <= residual_mem[request_residual_addr];
        if (state_q == STATE_WRITE_COMMIT && write_residual_enable_q)
            residual_mem[selected_residual_addr_q] <= write_residual_q;
    end
`endif

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= STATE_IDLE;
            selected_addr_q <= '0;
            selected_lane_q <= '0;
            selected_residual_addr_q <= '0;
            write_code_q <= '0;
            write_residual_q <= '0;
            write_code_enable_q <= 1'b0;
            write_residual_enable_q <= 1'b0;
            o_learning_write_done <= 1'b0;
            o_protocol_error <= 1'b0;
        end else begin
            o_learning_write_done <= 1'b0;
            if (i_clear_error)
                o_protocol_error <= 1'b0;
            if (i_learning_read_valid && i_learning_write_valid)
                o_protocol_error <= 1'b1;

            case (state_q)
                STATE_IDLE: begin
                    if (read_fire) begin
                        selected_addr_q <= i_learning_read_addr;
                        selected_lane_q <= i_learning_read_lane;
                        selected_residual_addr_q <= request_residual_addr;
                        state_q <= STATE_READ_RESPONSE;
                    end else if (write_fire) begin
                        if (i_learning_write_code_enable &&
                            !write_code_fits) begin
                            o_protocol_error <= 1'b1;
                            o_learning_write_done <= 1'b1;
                        end else begin
                            selected_addr_q <= i_learning_write_addr;
                            selected_lane_q <= i_learning_write_lane;
                            selected_residual_addr_q <=
                                request_residual_addr;
                            write_code_q <=
                                i_learning_write_code[W0_BITS-1:0];
                            write_residual_q <= i_learning_write_residual;
                            write_code_enable_q <=
                                i_learning_write_code_enable;
                            write_residual_enable_q <=
                                i_learning_write_residual_enable;
                            state_q <= STATE_WRITE_COMMIT;
                        end
                    end
                end
                STATE_READ_RESPONSE: begin
                    if (response_fire)
                        state_q <= STATE_IDLE;
                end
                STATE_WRITE_COMMIT: begin
                    o_learning_write_done <= 1'b1;
                    state_q <= STATE_IDLE;
                end
                default: begin
                    state_q <= STATE_IDLE;
                    o_protocol_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (HIDDEN_PER_TILE < 1 || W0_BITS < 2 || DEPTH < 1 ||
            CODE_WIDTH < W0_BITS || RESIDUAL_WIDTH < 1)
            $error("FC TP native W0 bank parameters are invalid");
    end
endmodule

`default_nettype wire
