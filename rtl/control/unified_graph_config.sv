`timescale 1ns/1ps
`default_nettype none

module unified_graph_config #(
    parameter integer PROFILE_COUNT = 4,
    parameter integer MAX_LAYERS = 16,
    parameter integer LAYER_WORDS = 12,
    parameter integer MAX_ROUTES = 64,
    parameter integer INPUT_MAX = 2312,
    parameter integer HIDDEN_MAX = 480,
    parameter integer OUTPUT_MAX = 32,
    parameter integer NEURONS_PER_TILE = 40
) (
    input  wire logic         clk,
    input  wire logic         rst_n,

    input  wire logic         i_cfg_valid,
    output logic              o_cfg_ready,
    input  wire logic [1:0]   i_cfg_profile,
    input  wire logic [1:0]   i_cfg_table,
    input  wire logic [8:0]   i_cfg_index,
    input  wire logic [31:0]  i_cfg_data,

    input  wire logic         i_commit_valid,
    output logic              o_commit_ready,
    input  wire logic [1:0]   i_commit_profile,
    input  wire logic         i_global_idle,
    input  wire logic         i_clear_error,

    input  wire logic [3:0]   i_active_layer_read_index,
    output logic [LAYER_WORDS*32-1:0] o_active_layer_words,
    input  wire logic [5:0]   i_active_route_read_index,
    output logic [63:0]       o_active_route_words,

    output logic [1:0]        o_active_profile,
    output logic [4:0]        o_active_layer_count,
    output logic [6:0]        o_active_route_count,
    output logic [15:0]       o_profile_version,
    output logic [15:0]       o_graph_version,
    output logic              o_commit_pending,
    output logic              o_commit_pulse,
    output logic              o_config_error,
    output logic [7:0]        o_error_code
);
    localparam integer LAYER_DEPTH = MAX_LAYERS * LAYER_WORDS;
    localparam integer ROUTE_DEPTH = MAX_ROUTES * 2;
    localparam integer SHADOW_LAYER_DEPTH = PROFILE_COUNT * LAYER_DEPTH;
    localparam integer SHADOW_ROUTE_DEPTH = PROFILE_COUNT * ROUTE_DEPTH;

    localparam logic [1:0] TABLE_HEADER = 2'd0;
    localparam logic [1:0] TABLE_LAYER  = 2'd1;
    localparam logic [1:0] TABLE_ROUTE  = 2'd2;

    localparam logic [31:0] GRAPH_MAGIC = 32'h55534e31;
    localparam logic [31:0] LAYER_MAGIC = 32'h4e4d5232;
    localparam logic [31:0] ABI_V21 = 32'h000c0102;
    localparam logic [31:0] ABI_V22 = 32'h000c0202;
    localparam logic [31:0] ABI_V23 = 32'h000c0302;
    localparam logic [31:0] CONTRACT_N8E1 = 32'h4e384531;
    localparam logic [31:0] CONTRACT_USN1 = 32'h55534e31;

    localparam logic [3:0] LAYER_SPARSE_FC = 4'd1;
    localparam logic [3:0] LAYER_CONV = 4'd2;
    localparam logic [3:0] LAYER_POOL = 4'd3;
    localparam logic [3:0] LAYER_OUTPUT = 4'd4;
    localparam logic [3:0] LAYER_RECURRENT_FC = 4'd5;
    localparam logic [3:0] EDGE_PREVIOUS_TIMESTEP = 4'd3;
    localparam logic [1:0] POOL_VALUE_BYPASS = 2'd3;

    localparam logic [7:0] ERROR_NONE = 8'd0;
    localparam logic [7:0] ERROR_CFG_ADDRESS = 8'd1;
    localparam logic [7:0] ERROR_INCOMPLETE = 8'd2;
    localparam logic [7:0] ERROR_HEADER = 8'd3;
    localparam logic [7:0] ERROR_LAYER = 8'd4;
    localparam logic [7:0] ERROR_ROUTE = 8'd5;
    localparam logic [7:0] ERROR_CYCLE = 8'd6;
    localparam logic [7:0] ERROR_FANIN = 8'd7;
    localparam logic [7:0] ERROR_CAPACITY = 8'd8;

    localparam logic [3:0] STATE_IDLE = 4'd0;
    localparam logic [3:0] STATE_WAIT_IDLE = 4'd1;
    localparam logic [3:0] STATE_LAYER_READ = 4'd2;
    localparam logic [3:0] STATE_LAYER_CHECK = 4'd3;
    localparam logic [3:0] STATE_ROUTE_READ = 4'd4;
    localparam logic [3:0] STATE_ROUTE_CHECK = 4'd5;
    localparam logic [3:0] STATE_COPY_LAYER_READ = 4'd6;
    localparam logic [3:0] STATE_COPY_LAYER_CAPTURE = 4'd7;
    localparam logic [3:0] STATE_COPY_LAYER_STORE = 4'd8;
    localparam logic [3:0] STATE_COPY_ROUTE_READ = 4'd9;
    localparam logic [3:0] STATE_COPY_ROUTE_CAPTURE = 4'd10;
    localparam logic [3:0] STATE_COPY_ROUTE_STORE = 4'd11;
    localparam logic [3:0] STATE_ACTIVATE = 4'd12;

    logic [31:0] shadow_header [0:PROFILE_COUNT-1][0:1];
    (* ram_style = "block" *)
    logic [31:0] shadow_layer_mem [0:SHADOW_LAYER_DEPTH-1];
    (* ram_style = "block" *)
    logic [31:0] shadow_route_mem [0:SHADOW_ROUTE_DEPTH-1];

    logic [1:0] header_written [0:PROFILE_COUNT-1];
    logic [LAYER_DEPTH-1:0] layer_written [0:PROFILE_COUNT-1];
    logic [ROUTE_DEPTH-1:0] route_written [0:PROFILE_COUNT-1];

    logic [3:0] state_q;
    logic [1:0] pending_profile_q;
    logic [4:0] pending_layer_count_q;
    logic [6:0] pending_route_count_q;
    logic [3:0] layer_index_q;
    logic [3:0] layer_word_q;
    logic [5:0] route_index_q;
    logic       route_word_q;
    logic [31:0] layer_read_q;
    logic [31:0] route_read_q;
    logic [31:0] route_word0_q;
    logic [31:0] copy_layer_words_q [0:LAYER_WORDS-1];
    logic [31:0] copy_route_words_q [0:1];
    logic [3:0] layer_type_q [0:MAX_LAYERS-1];
    logic [31:0] layer_abi_q [0:MAX_LAYERS-1];
    logic [9:0] layer_state_dim_q [0:MAX_LAYERS-1];
    logic [2:0] fanin_count_q [0:MAX_LAYERS-1];

    logic cfg_profile_valid;
    logic cfg_address_valid;
    logic cfg_fire;
    logic cfg_shadow_write_q;
    logic [1:0] cfg_shadow_profile_q;
    logic [1:0] cfg_shadow_table_q;
    logic [8:0] cfg_shadow_index_q;
    logic [31:0] cfg_shadow_data_q;
    logic commit_fire;
    logic [LAYER_WORDS*32-1:0] active_layer_read_words;
    logic [63:0] active_route_read_words;
    integer reset_profile;
    integer reset_index;
    genvar active_bank;

    generate
        for (active_bank = 0; active_bank < LAYER_WORDS;
             active_bank = active_bank + 1) begin : g_active_layer
            (* ram_style = "distributed" *)
            logic [31:0] memory [0:MAX_LAYERS-1];
            always_ff @(posedge clk)
                if (state_q == STATE_COPY_LAYER_STORE)
                    memory[layer_index_q] <= copy_layer_words_q[active_bank];
            assign active_layer_read_words[active_bank*32 +: 32] =
                memory[i_active_layer_read_index];
        end
        for (active_bank = 0; active_bank < 2;
             active_bank = active_bank + 1) begin : g_active_route
            (* ram_style = "distributed" *)
            logic [31:0] memory [0:MAX_ROUTES-1];
            always_ff @(posedge clk)
                if (state_q == STATE_COPY_ROUTE_STORE)
                    memory[route_index_q] <= copy_route_words_q[active_bank];
            assign active_route_read_words[active_bank*32 +: 32] =
                memory[i_active_route_read_index];
        end
    endgenerate

    function automatic logic valid_layer_type(input logic [3:0] layer_type);
        begin
            valid_layer_type =
                (layer_type == LAYER_SPARSE_FC) ||
                (layer_type == LAYER_CONV) ||
                (layer_type == LAYER_POOL) ||
                (layer_type == LAYER_OUTPUT) ||
                (layer_type == LAYER_RECURRENT_FC);
        end
    endfunction

    function automatic logic [7:0] validate_layer_word(
        input logic [3:0] layer_index,
        input logic [3:0] word_index,
        input logic [31:0] word_value,
        input logic [4:0] layer_count,
        input logic [3:0] layer_type,
        input logic [31:0] layer_abi
    );
        begin
            validate_layer_word = ERROR_NONE;
            case (word_index)
                0: if (word_value != LAYER_MAGIC)
                    validate_layer_word = ERROR_LAYER;
                1: if ((word_value != ABI_V21) && (word_value != ABI_V22) &&
                       (word_value != ABI_V23))
                    validate_layer_word = ERROR_LAYER;
                2: if (!valid_layer_type(word_value[3:0]))
                    validate_layer_word = ERROR_LAYER;
                3: if ((word_value[12:0] == 0) ||
                       (word_value[12:0] > INPUT_MAX) ||
                       (word_value[22:13] == 0) ||
                       (word_value[22:13] > HIDDEN_MAX) ||
                       (word_value[28:23] == 0) ||
                       (word_value[28:23] > OUTPUT_MAX))
                    validate_layer_word = ERROR_CAPACITY;
                4: if ((word_value[4:0] == 0) || (word_value[4:0] > 12))
                    validate_layer_word = ERROR_CAPACITY;
                   else if ((word_value[13:5] == 0) ||
                            (word_value[17:14] != 4'd8))
                    validate_layer_word = ERROR_LAYER;
                5: if ((layer_type == LAYER_CONV) &&
                       (layer_abi == ABI_V23) &&
                       ((word_value[5:0] == 0) || (word_value[5:0] > 32) ||
                        (word_value[11:6] == 0) || (word_value[11:6] > 32) ||
                        (word_value[21:12] == 0) ||
                        (word_value[21:12] > 512) ||
                        (word_value[23:22] == 0) ||
                        (word_value[23:22] > 2) ||
                        (word_value[25:24] > 1)))
                    validate_layer_word = ERROR_CAPACITY;
                6: if ((layer_type == LAYER_CONV) &&
                       (layer_abi == ABI_V23) &&
                       ((word_value[5:0] == 0) || (word_value[5:0] > 32) ||
                        (word_value[11:6] == 0) || (word_value[11:6] > 32) ||
                        (word_value[21:12] == 0) ||
                        (word_value[21:12] > 512) ||
                        ((word_value[23:22] == POOL_VALUE_BYPASS) &&
                         (layer_index != 0))))
                    validate_layer_word = ERROR_CAPACITY;
                10: if ((!word_value[8]) && (word_value[7:0] >= layer_count))
                    validate_layer_word = ERROR_LAYER;
                11: if ((word_value != CONTRACT_N8E1) &&
                        (word_value != CONTRACT_USN1))
                    validate_layer_word = ERROR_LAYER;
                default: begin end
            endcase
        end
    endfunction

    task automatic fail_commit(input logic [7:0] code);
        begin
            if (!o_config_error) begin
                o_config_error <= 1'b1;
                o_error_code <= code;
            end
            o_commit_pending <= 1'b0;
            state_q <= STATE_IDLE;
        end
    endtask

    always_comb begin
        cfg_profile_valid = ({1'b0, i_cfg_profile} < PROFILE_COUNT);
        case (i_cfg_table)
            TABLE_HEADER: cfg_address_valid = i_cfg_index < 2;
            TABLE_LAYER: cfg_address_valid = i_cfg_index < LAYER_DEPTH;
            TABLE_ROUTE: cfg_address_valid = i_cfg_index < ROUTE_DEPTH;
            default: cfg_address_valid = 1'b0;
        endcase

        o_cfg_ready = (state_q == STATE_IDLE) && !o_commit_pending &&
            !i_commit_valid;
        o_commit_ready = (state_q == STATE_IDLE) && !o_commit_pending &&
            !i_cfg_valid;
        cfg_fire = i_cfg_valid && o_cfg_ready;
        commit_fire = i_commit_valid && o_commit_ready;

        o_active_layer_words =
            (i_active_layer_read_index < o_active_layer_count) ?
            active_layer_read_words : '0;
        o_active_route_words =
            (i_active_route_read_index < o_active_route_count) ?
            active_route_read_words : '0;
    end

    // Register accepted writes before the RAM controls. This keeps asynchronous
    // upstream reset cones away from RAM address and write-enable pins.
    always_ff @(posedge clk) begin : p_shadow_write_command
        if (!rst_n) begin
            cfg_shadow_write_q <= 1'b0;
            cfg_shadow_profile_q <= '0;
            cfg_shadow_table_q <= TABLE_HEADER;
            cfg_shadow_index_q <= '0;
            cfg_shadow_data_q <= '0;
        end else begin
            cfg_shadow_write_q <= cfg_fire && cfg_profile_valid &&
                cfg_address_valid && ((i_cfg_table == TABLE_LAYER) ||
                                      (i_cfg_table == TABLE_ROUTE));
            if (cfg_fire && cfg_profile_valid && cfg_address_valid) begin
                cfg_shadow_profile_q <= i_cfg_profile;
                cfg_shadow_table_q <= i_cfg_table;
                cfg_shadow_index_q <= i_cfg_index;
                cfg_shadow_data_q <= i_cfg_data;
            end
        end
    end

    // Shadow tables remain unreset so Vivado can infer true memories. The
    // written-bit maps define validity; reset only clears the registered RAM
    // command boundary and the control state that drives synchronous reads.
    always_ff @(posedge clk) begin : p_shadow_layer_memory
        if (cfg_shadow_write_q && (cfg_shadow_table_q == TABLE_LAYER))
            shadow_layer_mem[cfg_shadow_profile_q*LAYER_DEPTH +
                             cfg_shadow_index_q] <= cfg_shadow_data_q;
        if ((state_q == STATE_LAYER_READ) ||
            (state_q == STATE_COPY_LAYER_READ))
            layer_read_q <= shadow_layer_mem[
                pending_profile_q*LAYER_DEPTH +
                layer_index_q*LAYER_WORDS + layer_word_q];
    end

    always_ff @(posedge clk) begin : p_shadow_route_memory
        if (cfg_shadow_write_q && (cfg_shadow_table_q == TABLE_ROUTE))
            shadow_route_mem[cfg_shadow_profile_q*ROUTE_DEPTH +
                             cfg_shadow_index_q] <= cfg_shadow_data_q;
        if ((state_q == STATE_ROUTE_READ) ||
            (state_q == STATE_COPY_ROUTE_READ))
            route_read_q <= shadow_route_mem[
                pending_profile_q*ROUTE_DEPTH +
                route_index_q*2 + route_word_q];
    end

    always_ff @(posedge clk) begin : p_unreset_working_words
        if ((state_q == STATE_LAYER_CHECK) && (layer_word_q == 1))
            layer_abi_q[layer_index_q] <= layer_read_q;
        if ((state_q == STATE_LAYER_CHECK) && (layer_word_q == 2))
            layer_type_q[layer_index_q] <= layer_read_q[3:0];
        if ((state_q == STATE_LAYER_CHECK) && (layer_word_q == 3))
            layer_state_dim_q[layer_index_q] <= layer_read_q[22:13];
        if ((state_q == STATE_ROUTE_CHECK) && !route_word_q)
            route_word0_q <= route_read_q;
        if (state_q == STATE_COPY_LAYER_CAPTURE)
            copy_layer_words_q[layer_word_q] <= layer_read_q;
        if (state_q == STATE_COPY_ROUTE_CAPTURE)
            copy_route_words_q[route_word_q] <= route_read_q;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (reset_profile = 0; reset_profile < PROFILE_COUNT;
                 reset_profile = reset_profile + 1) begin
                shadow_header[reset_profile][0] <= 32'd0;
                shadow_header[reset_profile][1] <= 32'd0;
                header_written[reset_profile] <= '0;
                layer_written[reset_profile] <= '0;
                route_written[reset_profile] <= '0;
            end
            for (reset_index = 0; reset_index < MAX_LAYERS;
                 reset_index = reset_index + 1) begin
                fanin_count_q[reset_index] <= 3'd0;
            end
            state_q <= STATE_IDLE;
            pending_profile_q <= 2'd0;
            pending_layer_count_q <= 5'd0;
            pending_route_count_q <= 7'd0;
            layer_index_q <= 4'd0;
            layer_word_q <= 4'd0;
            route_index_q <= 6'd0;
            route_word_q <= 1'b0;
            o_active_profile <= 2'd0;
            o_active_layer_count <= 5'd0;
            o_active_route_count <= 7'd0;
            o_profile_version <= 16'd0;
            o_graph_version <= 16'd0;
            o_commit_pending <= 1'b0;
            o_commit_pulse <= 1'b0;
            o_config_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else begin
            o_commit_pulse <= 1'b0;
            if (i_clear_error) begin
                o_config_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end

            if (cfg_fire) begin
                if (!cfg_profile_valid || !cfg_address_valid) begin
                    if (!o_config_error) begin
                        o_config_error <= 1'b1;
                        o_error_code <= ERROR_CFG_ADDRESS;
                    end
                end else begin
                    case (i_cfg_table)
                        TABLE_HEADER: begin
                            shadow_header[i_cfg_profile][i_cfg_index] <= i_cfg_data;
                            header_written[i_cfg_profile][i_cfg_index] <= 1'b1;
                        end
                        TABLE_LAYER: begin
                            layer_written[i_cfg_profile][i_cfg_index] <= 1'b1;
                        end
                        TABLE_ROUTE: begin
                            route_written[i_cfg_profile][i_cfg_index] <= 1'b1;
                        end
                        default: begin end
                    endcase
                end
            end

            if (commit_fire) begin
                if ({1'b0, i_commit_profile} >= PROFILE_COUNT) begin
                    if (!o_config_error) begin
                        o_config_error <= 1'b1;
                        o_error_code <= ERROR_CFG_ADDRESS;
                    end
                end else begin
                    pending_profile_q <= i_commit_profile;
                    o_commit_pending <= 1'b1;
                    state_q <= STATE_WAIT_IDLE;
                end
            end

            case (state_q)
                STATE_IDLE: begin end

                STATE_WAIT_IDLE: begin
                    if (i_global_idle) begin
                        if (header_written[pending_profile_q] != 2'b11) begin
                            fail_commit(ERROR_INCOMPLETE);
                        end else if ((shadow_header[pending_profile_q][0] != GRAPH_MAGIC) ||
                                     (shadow_header[pending_profile_q][1][31:13] != 0) ||
                                     (shadow_header[pending_profile_q][1][4:0] == 0) ||
                                     (shadow_header[pending_profile_q][1][4:0] > MAX_LAYERS) ||
                                     (shadow_header[pending_profile_q][1][12:6] > MAX_ROUTES)) begin
                            fail_commit(ERROR_HEADER);
                        end else begin
                            pending_layer_count_q <=
                                shadow_header[pending_profile_q][1][4:0];
                            pending_route_count_q <=
                                shadow_header[pending_profile_q][1][12:6];
                            layer_index_q <= 4'd0;
                            layer_word_q <= 4'd0;
                            for (reset_index = 0; reset_index < MAX_LAYERS;
                                 reset_index = reset_index + 1) begin
                                fanin_count_q[reset_index] <= 3'd0;
                            end
                            state_q <= STATE_LAYER_READ;
                        end
                    end
                end

                STATE_LAYER_READ: begin
                    state_q <= STATE_LAYER_CHECK;
                end

                STATE_LAYER_CHECK: begin
                    if (!layer_written[pending_profile_q]
                                      [layer_index_q*LAYER_WORDS + layer_word_q]) begin
                        fail_commit(ERROR_INCOMPLETE);
                    end else if (validate_layer_word(
                                     layer_index_q, layer_word_q, layer_read_q,
                                     pending_layer_count_q,
                                     layer_type_q[layer_index_q],
                                     layer_abi_q[layer_index_q]) != ERROR_NONE) begin
                        fail_commit(validate_layer_word(
                            layer_index_q, layer_word_q, layer_read_q,
                            pending_layer_count_q,
                            layer_type_q[layer_index_q],
                            layer_abi_q[layer_index_q]));
                    end else if ((layer_word_q == 4) &&
                                 ((layer_type_q[layer_index_q] ==
                                   LAYER_SPARSE_FC) ||
                                  (layer_type_q[layer_index_q] ==
                                   LAYER_RECURRENT_FC)) &&
                                 (layer_state_dim_q[layer_index_q] !=
                                  layer_read_q[4:0]*NEURONS_PER_TILE)) begin
                        fail_commit(ERROR_CAPACITY);
                    end else begin
                        if (layer_word_q + 1 >= LAYER_WORDS) begin
                            layer_word_q <= 4'd0;
                            if (layer_index_q + 1 >= pending_layer_count_q) begin
                                route_index_q <= 6'd0;
                                route_word_q <= 1'b0;
                                if (pending_route_count_q == 0) begin
                                    layer_index_q <= 4'd0;
                                    state_q <= STATE_COPY_LAYER_READ;
                                end else begin
                                    state_q <= STATE_ROUTE_READ;
                                end
                            end else begin
                                layer_index_q <= layer_index_q + 1'b1;
                                state_q <= STATE_LAYER_READ;
                            end
                        end else begin
                            layer_word_q <= layer_word_q + 1'b1;
                            state_q <= STATE_LAYER_READ;
                        end
                    end
                end

                STATE_ROUTE_READ: begin
                    state_q <= STATE_ROUTE_CHECK;
                end

                STATE_ROUTE_CHECK: begin
                    if (!route_written[pending_profile_q]
                                      [route_index_q*2 + route_word_q]) begin
                        fail_commit(ERROR_INCOMPLETE);
                    end else if (!route_word_q) begin
                        route_word_q <= 1'b1;
                        state_q <= STATE_ROUTE_READ;
                    end else if ((route_word0_q[31:15] != 0) ||
                                 (route_read_q != 0) ||
                                 (route_word0_q[4:1] > EDGE_PREVIOUS_TIMESTEP) ||
                                 (route_word0_q[8:5] >= pending_layer_count_q) ||
                                 (route_word0_q[12:9] >= pending_layer_count_q)) begin
                        fail_commit(ERROR_ROUTE);
                    end else if (route_word0_q[0] &&
                                 (route_word0_q[4:1] != EDGE_PREVIOUS_TIMESTEP) &&
                                 (route_word0_q[8:5] >= route_word0_q[12:9])) begin
                        fail_commit(ERROR_CYCLE);
                    end else if (route_word0_q[0] &&
                                 (route_word0_q[4:1] == EDGE_PREVIOUS_TIMESTEP) &&
                                 (layer_type_q[route_word0_q[12:9]] !=
                                  LAYER_RECURRENT_FC)) begin
                        fail_commit(ERROR_ROUTE);
                    end else if (route_word0_q[0] &&
                                 (fanin_count_q[route_word0_q[12:9]] >= 4)) begin
                        fail_commit(ERROR_FANIN);
                    end else begin
                        if (route_word0_q[0])
                            fanin_count_q[route_word0_q[12:9]] <=
                                fanin_count_q[route_word0_q[12:9]] + 1'b1;
                        route_word_q <= 1'b0;
                        if (route_index_q + 1 >= pending_route_count_q) begin
                            layer_index_q <= 4'd0;
                            layer_word_q <= 4'd0;
                            state_q <= STATE_COPY_LAYER_READ;
                        end else begin
                            route_index_q <= route_index_q + 1'b1;
                            state_q <= STATE_ROUTE_READ;
                        end
                    end
                end

                STATE_COPY_LAYER_READ: begin
                    state_q <= STATE_COPY_LAYER_CAPTURE;
                end

                STATE_COPY_LAYER_CAPTURE: begin
                    if (layer_word_q + 1 >= LAYER_WORDS) begin
                        layer_word_q <= 4'd0;
                        state_q <= STATE_COPY_LAYER_STORE;
                    end else begin
                        layer_word_q <= layer_word_q + 1'b1;
                        state_q <= STATE_COPY_LAYER_READ;
                    end
                end

                STATE_COPY_LAYER_STORE: begin
                    if (layer_index_q + 1 >= pending_layer_count_q) begin
                        route_index_q <= 6'd0;
                        route_word_q <= 1'b0;
                        state_q <= (pending_route_count_q == 0) ?
                            STATE_ACTIVATE : STATE_COPY_ROUTE_READ;
                    end else begin
                        layer_index_q <= layer_index_q + 1'b1;
                        state_q <= STATE_COPY_LAYER_READ;
                    end
                end

                STATE_COPY_ROUTE_READ: begin
                    state_q <= STATE_COPY_ROUTE_CAPTURE;
                end

                STATE_COPY_ROUTE_CAPTURE: begin
                    if (route_word_q) begin
                        route_word_q <= 1'b0;
                        state_q <= STATE_COPY_ROUTE_STORE;
                    end else begin
                        route_word_q <= 1'b1;
                        state_q <= STATE_COPY_ROUTE_READ;
                    end
                end

                STATE_COPY_ROUTE_STORE: begin
                    if (route_index_q + 1 >= pending_route_count_q) begin
                        state_q <= STATE_ACTIVATE;
                    end else begin
                        route_index_q <= route_index_q + 1'b1;
                        state_q <= STATE_COPY_ROUTE_READ;
                    end
                end

                STATE_ACTIVATE: begin
                    o_active_profile <= pending_profile_q;
                    o_active_layer_count <= pending_layer_count_q;
                    o_active_route_count <= pending_route_count_q;
                    o_profile_version <= o_profile_version + 1'b1;
                    o_graph_version <= o_graph_version + 1'b1;
                    o_commit_pending <= 1'b0;
                    o_commit_pulse <= 1'b1;
                    state_q <= STATE_IDLE;
                end

                default: begin
                    o_commit_pending <= 1'b0;
                    state_q <= STATE_IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
