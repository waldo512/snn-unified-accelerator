`default_nettype none

module unified_layer_owner_dispatch (
    input  logic          i_clk,
    input  logic          i_rst_n,
    input  logic          i_clear_error,
    input  logic          i_abort,
    input  logic          i_layer_valid,
    output logic          o_layer_ready,
    input  logic [3:0]    i_layer_type,
    output logic          o_fc_valid,
    input  logic          i_fc_ready,
    input  logic          i_fc_done,
    output logic          o_recurrent_valid,
    input  logic          i_recurrent_ready,
    input  logic          i_recurrent_done,
    output logic          o_scnn_valid,
    input  logic          i_scnn_ready,
    input  logic          i_scnn_done,
    output logic          o_output_valid,
    input  logic          i_output_ready,
    input  logic          i_output_done,
    output logic          o_layer_done,
    output logic          o_busy,
    output logic          o_dispatch_error
);
    localparam logic [3:0] LAYER_SPARSE_FC = 4'd1;
    localparam logic [3:0] LAYER_CONV = 4'd2;
    localparam logic [3:0] LAYER_POOL = 4'd3;
    localparam logic [3:0] LAYER_OUTPUT = 4'd4;
    localparam logic [3:0] LAYER_RECURRENT_FC = 4'd5;

    localparam logic [2:0] OWNER_NONE = 3'd0;
    localparam logic [2:0] OWNER_FC = 3'd1;
    localparam logic [2:0] OWNER_RECURRENT = 3'd2;
    localparam logic [2:0] OWNER_SCNN = 3'd3;
    localparam logic [2:0] OWNER_OUTPUT = 3'd4;

    logic [2:0] selected_owner;
    logic [2:0] active_owner_q;
    logic done_armed_q;
    logic selected_ready;
    logic active_done;

    always_comb begin
        case (i_layer_type)
            LAYER_SPARSE_FC: selected_owner = OWNER_FC;
            LAYER_RECURRENT_FC: selected_owner = OWNER_RECURRENT;
            LAYER_CONV, LAYER_POOL: selected_owner = OWNER_SCNN;
            LAYER_OUTPUT: selected_owner = OWNER_OUTPUT;
            default: selected_owner = OWNER_NONE;
        endcase
        case (selected_owner)
            OWNER_FC: selected_ready = i_fc_ready;
            OWNER_RECURRENT: selected_ready = i_recurrent_ready;
            OWNER_SCNN: selected_ready = i_scnn_ready;
            OWNER_OUTPUT: selected_ready = i_output_ready;
            default: selected_ready = 1'b0;
        endcase
        case (active_owner_q)
            OWNER_FC: active_done = i_fc_done;
            OWNER_RECURRENT: active_done = i_recurrent_done;
            OWNER_SCNN: active_done = i_scnn_done;
            OWNER_OUTPUT: active_done = i_output_done;
            default: active_done = 1'b0;
        endcase
        o_layer_ready = !o_busy && selected_owner != OWNER_NONE && selected_ready;
        o_fc_valid = i_layer_valid && !o_busy && selected_owner == OWNER_FC;
        o_recurrent_valid = i_layer_valid && !o_busy &&
                            selected_owner == OWNER_RECURRENT;
        o_scnn_valid = i_layer_valid && !o_busy && selected_owner == OWNER_SCNN;
        o_output_valid = i_layer_valid && !o_busy &&
                         selected_owner == OWNER_OUTPUT;
        o_layer_done = o_busy && done_armed_q && active_done;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            active_owner_q <= OWNER_NONE;
            done_armed_q <= 1'b0;
            o_busy <= 1'b0;
            o_dispatch_error <= 1'b0;
        end else begin
            if (i_clear_error)
                o_dispatch_error <= 1'b0;
            if (i_layer_valid && !o_busy && selected_owner == OWNER_NONE)
                o_dispatch_error <= 1'b1;
            if (i_layer_valid && o_layer_ready) begin
                active_owner_q <= selected_owner;
                done_armed_q <= 1'b0;
                o_busy <= 1'b1;
            end
            if (o_busy && !active_done)
                done_armed_q <= 1'b1;
            if (o_layer_done) begin
                active_owner_q <= OWNER_NONE;
                done_armed_q <= 1'b0;
                o_busy <= 1'b0;
            end
            if (i_abort) begin
                active_owner_q <= OWNER_NONE;
                done_armed_q <= 1'b0;
                o_busy <= 1'b0;
                o_dispatch_error <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
