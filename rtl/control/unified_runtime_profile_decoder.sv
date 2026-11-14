`default_nettype none

module unified_runtime_profile_decoder (
    input  logic          i_clk,
    input  logic          i_rst_n,
    input  logic          i_start,
    input  logic [7:0]    i_opcode,
    input  logic          i_profile_done,
    input  logic          i_clear_error,
    output logic          o_start_ready,
    output logic          o_start_pulse,
    output logic          o_busy,
    output logic [7:0]    o_active_opcode,
    output logic [15:0]   o_profile_version,
    output logic          o_fc_fast_path,
    output logic          o_recurrent_extension,
    output logic          o_conv_extension,
    output logic [12:0]   o_input_dim,
    output logic [9:0]    o_logical_hidden_or_channels,
    output logic [5:0]    o_output_dim,
    output logic [4:0]    o_active_h40_tiles,
    output logic [4:0]    o_layer_count,
    output logic [7:0]    o_fixed_timesteps,
    output logic [2:0]    o_numeric_profile,
    output logic          o_fc_w2_update_enable,
    output logic          o_config_error,
    output logic [7:0]    o_error_code
);
    localparam logic [7:0] PROFILE_FC_H160 = 8'h01;
    localparam logic [7:0] PROFILE_FC_H480 = 8'h02;
    localparam logic [7:0] PROFILE_SRNN_H480 = 8'h10;
    localparam logic [7:0] PROFILE_SCNN_VGG9 = 8'h20;

    localparam logic [2:0] NUMERIC_FC_INT5_ACC24 = 3'd1;
    localparam logic [2:0] NUMERIC_SRNN_INT8_ACC32 = 3'd2;
    localparam logic [2:0] NUMERIC_SCNN_Q8_8_ACC48 = 3'd3;

    localparam logic [7:0] ERROR_NONE = 8'd0;
    localparam logic [7:0] ERROR_ILLEGAL_OPCODE = 8'd1;
    localparam logic [7:0] ERROR_START_WHILE_BUSY = 8'd2;
    localparam logic [7:0] ERROR_DONE_WHILE_IDLE = 8'd3;

    logic opcode_valid;
    logic selected_fc_fast;
    logic selected_recurrent;
    logic selected_conv;
    logic [12:0] selected_input_dim;
    logic [9:0] selected_hidden;
    logic [5:0] selected_output_dim;
    logic [4:0] selected_tiles;
    logic [4:0] selected_layers;
    logic [7:0] selected_timesteps;
    logic [2:0] selected_numeric;
    logic selected_update;

    always_comb begin
        opcode_valid = 1'b1;
        selected_fc_fast = 1'b0;
        selected_recurrent = 1'b0;
        selected_conv = 1'b0;
        selected_input_dim = 13'd0;
        selected_hidden = 10'd0;
        selected_output_dim = 6'd0;
        selected_tiles = 5'd0;
        selected_layers = 5'd0;
        selected_timesteps = 8'd0;
        selected_numeric = 3'd0;
        selected_update = 1'b0;
        case (i_opcode)
            PROFILE_FC_H160: begin
                selected_fc_fast = 1'b1;
                selected_input_dim = 13'd2312;
                selected_hidden = 10'd160;
                selected_output_dim = 6'd10;
                selected_tiles = 5'd4;
                selected_layers = 5'd1;
                selected_numeric = NUMERIC_FC_INT5_ACC24;
                selected_update = 1'b1;
            end
            PROFILE_FC_H480: begin
                selected_fc_fast = 1'b1;
                selected_input_dim = 13'd2312;
                selected_hidden = 10'd480;
                selected_output_dim = 6'd10;
                selected_tiles = 5'd12;
                selected_layers = 5'd1;
                selected_numeric = NUMERIC_FC_INT5_ACC24;
                selected_update = 1'b1;
            end
            PROFILE_SRNN_H480: begin
                selected_recurrent = 1'b1;
                selected_input_dim = 13'd700;
                selected_hidden = 10'd480;
                selected_output_dim = 6'd20;
                selected_tiles = 5'd12;
                selected_layers = 5'd1;
                selected_numeric = NUMERIC_SRNN_INT8_ACC32;
            end
            PROFILE_SCNN_VGG9: begin
                selected_conv = 1'b1;
                selected_input_dim = 13'd2048;
                selected_hidden = 10'd512;
                selected_output_dim = 6'd11;
                selected_tiles = 5'd12;
                selected_layers = 5'd9;
                selected_timesteps = 8'd20;
                selected_numeric = NUMERIC_SCNN_Q8_8_ACC48;
            end
            default: opcode_valid = 1'b0;
        endcase
        o_start_ready = !o_busy;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_start_pulse <= 1'b0;
            o_busy <= 1'b0;
            o_active_opcode <= '0;
            o_profile_version <= '0;
            o_fc_fast_path <= 1'b0;
            o_recurrent_extension <= 1'b0;
            o_conv_extension <= 1'b0;
            o_input_dim <= '0;
            o_logical_hidden_or_channels <= '0;
            o_output_dim <= '0;
            o_active_h40_tiles <= '0;
            o_layer_count <= '0;
            o_fixed_timesteps <= '0;
            o_numeric_profile <= '0;
            o_fc_w2_update_enable <= 1'b0;
            o_config_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else begin
            o_start_pulse <= 1'b0;
            if (i_clear_error) begin
                o_config_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end
            if (i_start) begin
                if (o_busy) begin
                    o_config_error <= 1'b1;
                    o_error_code <= ERROR_START_WHILE_BUSY;
                end else if (!opcode_valid) begin
                    o_config_error <= 1'b1;
                    o_error_code <= ERROR_ILLEGAL_OPCODE;
                end else begin
                    o_start_pulse <= 1'b1;
                    o_busy <= 1'b1;
                    o_active_opcode <= i_opcode;
                    o_profile_version <= o_profile_version + 1'b1;
                    o_fc_fast_path <= selected_fc_fast;
                    o_recurrent_extension <= selected_recurrent;
                    o_conv_extension <= selected_conv;
                    o_input_dim <= selected_input_dim;
                    o_logical_hidden_or_channels <= selected_hidden;
                    o_output_dim <= selected_output_dim;
                    o_active_h40_tiles <= selected_tiles;
                    o_layer_count <= selected_layers;
                    o_fixed_timesteps <= selected_timesteps;
                    o_numeric_profile <= selected_numeric;
                    o_fc_w2_update_enable <= selected_update;
                end
            end
            if (i_profile_done) begin
                if (o_busy)
                    o_busy <= 1'b0;
                else begin
                    o_config_error <= 1'b1;
                    o_error_code <= ERROR_DONE_WHILE_IDLE;
                end
            end
        end
    end
endmodule

`default_nettype wire
