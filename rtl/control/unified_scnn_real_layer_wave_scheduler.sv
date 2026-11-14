`default_nettype none

// Expands one accepted CONV layer into output-channel waves and K8 input
// groups. The child datapath owns arithmetic and storage handshakes.
module unified_scnn_real_layer_wave_scheduler #(
    parameter int ENGINES = 12,
    parameter int LANES = 8
) (
    input  logic         i_clk,
    input  logic         i_rst_n,
    input  logic         i_clear_error,
    input  logic         i_start,
    output logic         o_start_ready,
    input  logic [15:0]  i_in_width,
    input  logic [15:0]  i_in_height,
    input  logic [15:0]  i_in_channels,
    input  logic [15:0]  i_out_width,
    input  logic [15:0]  i_out_height,
    input  logic [15:0]  i_out_channels,

    output logic         o_wave_start,
    input  logic         i_wave_start_ready,
    output logic [9:0]   o_output_channel_base,
    output logic [9:0]   o_output_channel_count,
    output logic         o_group_start,
    input  logic         i_group_start_ready,
    output logic [9:0]   o_input_channel_base,
    output logic [9:0]   o_input_channel_count,
    output logic         o_first_input_group,
    output logic         o_final_input_group,
    input  logic         i_group_done,
    input  logic         i_wave_done,
    input  logic         i_child_error,

    output logic         o_busy,
    output logic         o_done,
    output logic         o_sticky_error
);
    typedef enum logic [2:0] {
        IDLE,
        WAVE_REQUEST,
        GROUP_REQUEST,
        GROUP_WAIT,
        WAVE_WAIT
    } state_t;

    state_t state_q;
    logic [10:0] in_channels_q;
    logic [10:0] out_channels_q;
    logic [9:0] output_base_q;
    logic [9:0] input_base_q;
    logic [10:0] output_remaining;
    logic [10:0] input_remaining;
    logic config_legal;
    logic start_fire;
    logic wave_fire;
    logic group_fire;

    always_comb begin
        output_remaining = out_channels_q - {1'b0, output_base_q};
        input_remaining = in_channels_q - {1'b0, input_base_q};
        o_output_channel_base = output_base_q;
        o_output_channel_count = output_remaining > 11'(ENGINES) ?
            10'(ENGINES) : output_remaining[9:0];
        o_input_channel_base = input_base_q;
        o_input_channel_count = input_remaining > 11'(LANES) ?
            10'(LANES) : input_remaining[9:0];
        o_first_input_group = input_base_q == 0;
        o_final_input_group = input_remaining <= 11'(LANES);
        config_legal = i_in_width != 0 && i_in_height != 0 &&
            i_in_channels != 0 && i_in_channels <= 16'd512 &&
            i_out_width != 0 && i_out_height != 0 &&
            i_out_channels != 0 && i_out_channels <= 16'd512;
        o_start_ready = state_q == IDLE && !o_sticky_error;
        o_wave_start = state_q == WAVE_REQUEST;
        o_group_start = state_q == GROUP_REQUEST;
        o_busy = state_q != IDLE;
        start_fire = i_start && o_start_ready;
        wave_fire = o_wave_start && i_wave_start_ready;
        group_fire = o_group_start && i_group_start_ready;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_q <= IDLE;
            in_channels_q <= '0;
            out_channels_q <= '0;
            output_base_q <= '0;
            input_base_q <= '0;
            o_done <= 1'b0;
            o_sticky_error <= 1'b0;
        end else begin
            o_done <= 1'b0;
            if (i_clear_error && state_q == IDLE)
                o_sticky_error <= 1'b0;

            if (i_child_error && state_q != IDLE) begin
                state_q <= IDLE;
                o_sticky_error <= 1'b1;
            end else begin
                case (state_q)
                    IDLE: begin
                        if (start_fire) begin
                            if (!config_legal) begin
                                o_sticky_error <= 1'b1;
                            end else begin
                                in_channels_q <= 11'(i_in_channels);
                                out_channels_q <= 11'(i_out_channels);
                                output_base_q <= '0;
                                input_base_q <= '0;
                                state_q <= WAVE_REQUEST;
                            end
                        end
                    end
                    WAVE_REQUEST: begin
                        if (wave_fire)
                            state_q <= GROUP_REQUEST;
                    end
                    GROUP_REQUEST: begin
                        if (group_fire)
                            state_q <= GROUP_WAIT;
                    end
                    GROUP_WAIT: begin
                        if (i_group_done) begin
                            if (o_final_input_group) begin
                                state_q <= WAVE_WAIT;
                            end else begin
                                input_base_q <= input_base_q +
                                                o_input_channel_count;
                                state_q <= GROUP_REQUEST;
                            end
                        end
                    end
                    WAVE_WAIT: begin
                        if (i_wave_done) begin
                            if ({1'b0, output_base_q} +
                                {1'b0, o_output_channel_count} >=
                                out_channels_q) begin
                                state_q <= IDLE;
                                o_done <= 1'b1;
                            end else begin
                                output_base_q <= output_base_q +
                                                 o_output_channel_count;
                                input_base_q <= '0;
                                state_q <= WAVE_REQUEST;
                            end
                        end
                    end
                    default: begin
                        state_q <= IDLE;
                        o_sticky_error <= 1'b1;
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
