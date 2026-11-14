`default_nettype none

// Three one-entry input queues isolate result producers from top-level
// backpressure. Fixed priority only controls drain order; simultaneous claims
// are retained and reported as an ownership error instead of being dropped.
module unified_result_owner_arbiter_v1 #(
    parameter int CLASS_WIDTH = 6,
    parameter int SCORE_WIDTH = 48,
    parameter int VERSION_WIDTH = 64
) (
    input  logic                            i_clk,
    input  logic                            i_rst_n,
    input  logic                            i_clear_error,
    input  logic                            i_flush,
    input  logic [1:0]                      i_expected_owner,

    input  logic                            i_fc_valid,
    output logic                            o_fc_ready,
    input  logic [CLASS_WIDTH-1:0]          i_fc_class,
    input  logic signed [SCORE_WIDTH-1:0]   i_fc_score,
    input  logic [VERSION_WIDTH-1:0]        i_fc_versions,

    input  logic                            i_dynamic_valid,
    output logic                            o_dynamic_ready,
    input  logic [CLASS_WIDTH-1:0]          i_dynamic_class,
    input  logic signed [SCORE_WIDTH-1:0]   i_dynamic_score,
    input  logic [VERSION_WIDTH-1:0]        i_dynamic_versions,

    input  logic                            i_scnn_valid,
    output logic                            o_scnn_ready,
    input  logic [CLASS_WIDTH-1:0]          i_scnn_class,
    input  logic signed [SCORE_WIDTH-1:0]   i_scnn_score,
    input  logic [VERSION_WIDTH-1:0]        i_scnn_versions,

    output logic                            o_result_valid,
    input  logic                            i_result_ready,
    output logic [CLASS_WIDTH-1:0]          o_result_class,
    output logic signed [SCORE_WIDTH-1:0]   o_result_score,
    output logic [VERSION_WIDTH-1:0]        o_result_versions,
    output logic [1:0]                      o_result_owner,
    output logic [2:0]                      o_pending_mask,
    output logic                            o_pending,
    output logic                            o_sticky_error,
    output logic [7:0]                      o_error_code
);
    localparam logic [1:0] OWNER_NONE = 2'd0;
    localparam logic [1:0] OWNER_FC = 2'd1;
    localparam logic [1:0] OWNER_DYNAMIC = 2'd2;
    localparam logic [1:0] OWNER_SCNN = 2'd3;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_OWNER_CONFLICT = 8'h01;
    localparam logic [7:0] ERROR_UNEXPECTED_OWNER = 8'h02;

    logic fc_pending_q;
    logic dynamic_pending_q;
    logic scnn_pending_q;
    logic [CLASS_WIDTH-1:0] fc_class_q;
    logic [CLASS_WIDTH-1:0] dynamic_class_q;
    logic [CLASS_WIDTH-1:0] scnn_class_q;
    logic signed [SCORE_WIDTH-1:0] fc_score_q;
    logic signed [SCORE_WIDTH-1:0] dynamic_score_q;
    logic signed [SCORE_WIDTH-1:0] scnn_score_q;
    logic [VERSION_WIDTH-1:0] fc_versions_q;
    logic [VERSION_WIDTH-1:0] dynamic_versions_q;
    logic [VERSION_WIDTH-1:0] scnn_versions_q;

    logic [1:0] selected_owner;
    logic result_fire;
    logic fc_accept;
    logic dynamic_accept;
    logic scnn_accept;
    logic [2:0] claim_mask;
    logic owner_conflict;
    logic unexpected_owner;

    always_comb begin
        selected_owner = OWNER_NONE;
        if (!i_flush) begin
            if (fc_pending_q)
                selected_owner = OWNER_FC;
            else if (dynamic_pending_q)
                selected_owner = OWNER_DYNAMIC;
            else if (scnn_pending_q)
                selected_owner = OWNER_SCNN;
        end

        o_result_valid = selected_owner != OWNER_NONE;
        o_result_owner = selected_owner;
        o_result_class = '0;
        o_result_score = '0;
        o_result_versions = '0;
        case (selected_owner)
            OWNER_FC: begin
                o_result_class = fc_class_q;
                o_result_score = fc_score_q;
                o_result_versions = fc_versions_q;
            end
            OWNER_DYNAMIC: begin
                o_result_class = dynamic_class_q;
                o_result_score = dynamic_score_q;
                o_result_versions = dynamic_versions_q;
            end
            OWNER_SCNN: begin
                o_result_class = scnn_class_q;
                o_result_score = scnn_score_q;
                o_result_versions = scnn_versions_q;
            end
            default: begin end
        endcase

        result_fire = o_result_valid && i_result_ready;
        o_fc_ready = i_flush || !fc_pending_q ||
                     (result_fire && selected_owner == OWNER_FC);
        o_dynamic_ready = i_flush || !dynamic_pending_q ||
                          (result_fire && selected_owner == OWNER_DYNAMIC);
        o_scnn_ready = i_flush || !scnn_pending_q ||
                       (result_fire && selected_owner == OWNER_SCNN);
        fc_accept = !i_flush && i_fc_valid && o_fc_ready;
        dynamic_accept = !i_flush && i_dynamic_valid && o_dynamic_ready;
        scnn_accept = !i_flush && i_scnn_valid && o_scnn_ready;

        o_pending_mask = i_flush ? 3'b000 :
            {scnn_pending_q, dynamic_pending_q, fc_pending_q};
        o_pending = |o_pending_mask;

        claim_mask = i_flush ? 3'b000 : {
            scnn_pending_q || i_scnn_valid,
            dynamic_pending_q || i_dynamic_valid,
            fc_pending_q || i_fc_valid
        };
        owner_conflict = (claim_mask[0] && claim_mask[1]) ||
                         (claim_mask[0] && claim_mask[2]) ||
                         (claim_mask[1] && claim_mask[2]);
        unexpected_owner =
            (claim_mask[0] && i_expected_owner != OWNER_FC) ||
            (claim_mask[1] && i_expected_owner != OWNER_DYNAMIC) ||
            (claim_mask[2] && i_expected_owner != OWNER_SCNN);
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            fc_pending_q <= 1'b0;
            dynamic_pending_q <= 1'b0;
            scnn_pending_q <= 1'b0;
            fc_class_q <= '0;
            dynamic_class_q <= '0;
            scnn_class_q <= '0;
            fc_score_q <= '0;
            dynamic_score_q <= '0;
            scnn_score_q <= '0;
            fc_versions_q <= '0;
            dynamic_versions_q <= '0;
            scnn_versions_q <= '0;
            o_sticky_error <= 1'b0;
            o_error_code <= ERROR_NONE;
        end else begin
            if (i_flush) begin
                fc_pending_q <= 1'b0;
                dynamic_pending_q <= 1'b0;
                scnn_pending_q <= 1'b0;
                fc_class_q <= '0;
                dynamic_class_q <= '0;
                scnn_class_q <= '0;
                fc_score_q <= '0;
                dynamic_score_q <= '0;
                scnn_score_q <= '0;
                fc_versions_q <= '0;
                dynamic_versions_q <= '0;
                scnn_versions_q <= '0;
            end else begin
                if (fc_accept) begin
                    fc_pending_q <= 1'b1;
                    fc_class_q <= i_fc_class;
                    fc_score_q <= i_fc_score;
                    fc_versions_q <= i_fc_versions;
                end else if (result_fire && selected_owner == OWNER_FC) begin
                    fc_pending_q <= 1'b0;
                end

                if (dynamic_accept) begin
                    dynamic_pending_q <= 1'b1;
                    dynamic_class_q <= i_dynamic_class;
                    dynamic_score_q <= i_dynamic_score;
                    dynamic_versions_q <= i_dynamic_versions;
                end else if (result_fire && selected_owner == OWNER_DYNAMIC) begin
                    dynamic_pending_q <= 1'b0;
                end

                if (scnn_accept) begin
                    scnn_pending_q <= 1'b1;
                    scnn_class_q <= i_scnn_class;
                    scnn_score_q <= i_scnn_score;
                    scnn_versions_q <= i_scnn_versions;
                end else if (result_fire && selected_owner == OWNER_SCNN) begin
                    scnn_pending_q <= 1'b0;
                end
            end

            if (i_clear_error) begin
                o_sticky_error <= 1'b0;
                o_error_code <= ERROR_NONE;
            end else if (!o_sticky_error && owner_conflict) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_OWNER_CONFLICT;
            end else if (!o_sticky_error && unexpected_owner) begin
                o_sticky_error <= 1'b1;
                o_error_code <= ERROR_UNEXPECTED_OWNER;
            end
        end
    end
endmodule

`default_nettype wire
