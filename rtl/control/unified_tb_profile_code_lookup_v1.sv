`default_nettype none

// TB/JTAG-facing resident-profile selector. A valid request is backpressured
// until the complete system is idle, then all metadata changes atomically.
module unified_tb_profile_code_lookup_v1 (
    input  logic         i_clk,
    input  logic         i_rst_n,
    input  logic         i_clear_error,
    input  logic         i_global_idle,
    input  logic         i_profile_valid,
    output logic         o_profile_ready,
    input  logic [6:0]   i_profile_code,
    output logic         o_lookup_legal,
    output logic         o_profile_activate_pulse,
    output logic         o_profile_reject_pulse,
    output logic         o_active_valid,
    output logic [6:0]   o_active_profile_code,
    output logic [1:0]   o_active_slot,
    output logic [2:0]   o_active_mode,
    output logic [11:0]  o_active_cluster_mask,
    output logic [12:0]  o_input_dim,
    output logic [9:0]   o_hidden_or_channel_dim,
    output logic [5:0]   o_output_dim,
    output logic [4:0]   o_layer_count,
    output logic [7:0]   o_fixed_timesteps,
    output logic [2:0]   o_numeric_profile,
    output logic         o_fc_w2_update_enable,
    output logic [15:0]  o_profile_version,
    output logic         o_sticky_error,
    output logic [7:0]   o_error_code
);
    localparam logic [6:0] PROFILE_FC_H160 = 7'h01;
    localparam logic [6:0] PROFILE_FC_H480 = 7'h02;
    localparam logic [6:0] PROFILE_SRNN_H480 = 7'h10;
    localparam logic [6:0] PROFILE_SCNN_VGG9 = 7'h20;

    localparam logic [2:0] MODE_NONE = 3'd0;
    localparam logic [2:0] MODE_SPARSE_FC = 3'd1;
    localparam logic [2:0] MODE_RECURRENT_FC = 3'd2;
    localparam logic [2:0] MODE_SCNN = 3'd3;

    localparam logic [2:0] NUMERIC_NONE = 3'd0;
    localparam logic [2:0] NUMERIC_FC_INT5_ACC24 = 3'd1;
    localparam logic [2:0] NUMERIC_SRNN_INT8_ACC32 = 3'd2;
    localparam logic [2:0] NUMERIC_SCNN_Q8_8_ACC48 = 3'd3;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_ILLEGAL_PROFILE = 8'h01;

    logic [1:0] selected_slot;
    logic [2:0] selected_mode;
    logic [11:0] selected_cluster_mask;
    logic [12:0] selected_input_dim;
    logic [9:0] selected_hidden_or_channel_dim;
    logic [5:0] selected_output_dim;
    logic [4:0] selected_layer_count;
    logic [7:0] selected_fixed_timesteps;
    logic [2:0] selected_numeric_profile;
    logic selected_fc_w2_update_enable;
    logic profile_fire;

    always_comb begin
        o_lookup_legal = 1'b1;
        selected_slot = 2'd0;
        selected_mode = MODE_NONE;
        selected_cluster_mask = 12'h000;
        selected_input_dim = 13'd0;
        selected_hidden_or_channel_dim = 10'd0;
        selected_output_dim = 6'd0;
        selected_layer_count = 5'd0;
        selected_fixed_timesteps = 8'd0;
        selected_numeric_profile = NUMERIC_NONE;
        selected_fc_w2_update_enable = 1'b0;
        case (i_profile_code)
            PROFILE_FC_H160: begin
                selected_slot = 2'd0;
                selected_mode = MODE_SPARSE_FC;
                selected_cluster_mask = 12'h00f;
                selected_input_dim = 13'd2312;
                selected_hidden_or_channel_dim = 10'd160;
                selected_output_dim = 6'd10;
                selected_layer_count = 5'd1;
                selected_numeric_profile = NUMERIC_FC_INT5_ACC24;
                selected_fc_w2_update_enable = 1'b1;
            end
            PROFILE_FC_H480: begin
                selected_slot = 2'd1;
                selected_mode = MODE_SPARSE_FC;
                selected_cluster_mask = 12'hfff;
                selected_input_dim = 13'd2312;
                selected_hidden_or_channel_dim = 10'd480;
                selected_output_dim = 6'd10;
                selected_layer_count = 5'd1;
                selected_numeric_profile = NUMERIC_FC_INT5_ACC24;
                selected_fc_w2_update_enable = 1'b1;
            end
            PROFILE_SRNN_H480: begin
                selected_slot = 2'd2;
                selected_mode = MODE_RECURRENT_FC;
                selected_cluster_mask = 12'hfff;
                selected_input_dim = 13'd700;
                selected_hidden_or_channel_dim = 10'd480;
                selected_output_dim = 6'd20;
                selected_layer_count = 5'd1;
                selected_numeric_profile = NUMERIC_SRNN_INT8_ACC32;
            end
            PROFILE_SCNN_VGG9: begin
                selected_slot = 2'd3;
                selected_mode = MODE_SCNN;
                selected_cluster_mask = 12'hfff;
                selected_input_dim = 13'd2048;
                selected_hidden_or_channel_dim = 10'd512;
                selected_output_dim = 6'd11;
                selected_layer_count = 5'd9;
                selected_fixed_timesteps = 8'd20;
                selected_numeric_profile = NUMERIC_SCNN_Q8_8_ACC48;
            end
            default: o_lookup_legal = 1'b0;
        endcase
        o_profile_ready = i_global_idle;
        profile_fire = i_profile_valid && o_profile_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_profile_activate_pulse <= 1'b0;
            o_profile_reject_pulse <= 1'b0;
            o_active_valid <= 1'b0;
            o_active_profile_code <= '0;
            o_active_slot <= '0;
            o_active_mode <= MODE_NONE;
            o_active_cluster_mask <= '0;
            o_input_dim <= '0;
            o_hidden_or_channel_dim <= '0;
            o_output_dim <= '0;
            o_layer_count <= '0;
            o_fixed_timesteps <= '0;
            o_numeric_profile <= NUMERIC_NONE;
            o_fc_w2_update_enable <= 1'b0;
            o_profile_version <= '0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else begin
            o_profile_activate_pulse <= 1'b0;
            o_profile_reject_pulse <= 1'b0;
            if (i_clear_error) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end
            if (profile_fire) begin
                if (!o_lookup_legal) begin
                    o_profile_reject_pulse <= 1'b1;
                    o_sticky_error <= 1'b1;
                    o_error_code <= ERROR_ILLEGAL_PROFILE;
                end else begin
                    o_profile_activate_pulse <= 1'b1;
                    o_active_valid <= 1'b1;
                    o_active_profile_code <= i_profile_code;
                    o_active_slot <= selected_slot;
                    o_active_mode <= selected_mode;
                    o_active_cluster_mask <= selected_cluster_mask;
                    o_input_dim <= selected_input_dim;
                    o_hidden_or_channel_dim <=
                        selected_hidden_or_channel_dim;
                    o_output_dim <= selected_output_dim;
                    o_layer_count <= selected_layer_count;
                    o_fixed_timesteps <= selected_fixed_timesteps;
                    o_numeric_profile <= selected_numeric_profile;
                    o_fc_w2_update_enable <=
                        selected_fc_w2_update_enable;
                    o_profile_version <= o_profile_version + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
