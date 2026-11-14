`timescale 1ns/1ps
// Generated from official1024 calibration samples 0..511.
module unified_input2312_replica_map_bucketed(
    input wire [11:0] i_feature,
    output reg o_has_replica,
    output reg [2:0] o_replica_bank,
    output reg [8:0] o_replica_addr
);
    always @* begin
        o_has_replica = 1'b0;
        o_replica_bank = 3'd0;
        o_replica_addr = 9'd0;
        case (i_feature[11:5])
            7'd11: begin
                case (i_feature[4:0])
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd300; end
                    default: begin end
                endcase
            end
            7'd12: begin
                case (i_feature[4:0])
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd302; end
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd290; end
                    default: begin end
                endcase
            end
            7'd13: begin
                case (i_feature[4:0])
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd293; end
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd290; end
                    default: begin end
                endcase
            end
            7'd14: begin
                case (i_feature[4:0])
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd295; end
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd294; end
                    5'd9: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd304; end
                    5'd11: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd300; end
                    default: begin end
                endcase
            end
            7'd15: begin
                case (i_feature[4:0])
                    5'd9: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd312; end
                    5'd10: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd308; end
                    5'd11: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd301; end
                    default: begin end
                endcase
            end
            7'd17: begin
                case (i_feature[4:0])
                    5'd12: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd310; end
                    default: begin end
                endcase
            end
            7'd23: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd292; end
                    default: begin end
                endcase
            end
            7'd24: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd304; end
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd298; end
                    default: begin end
                endcase
            end
            7'd25: begin
                case (i_feature[4:0])
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd291; end
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd290; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd306; end
                    5'd25: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd309; end
                    default: begin end
                endcase
            end
            7'd26: begin
                case (i_feature[4:0])
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd293; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd291; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd292; end
                    5'd27: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd311; end
                    5'd28: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd295; end
                    5'd29: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd291; end
                    5'd31: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd293; end
                    default: begin end
                endcase
            end
            7'd27: begin
                case (i_feature[4:0])
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd308; end
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd298; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd296; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd292; end
                    5'd5: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd291; end
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd5; o_replica_addr = 9'd290; end
                    5'd29: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd301; end
                    5'd30: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd296; end
                    5'd31: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd289; end
                    default: begin end
                endcase
            end
            7'd28: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd289; end
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd290; end
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd297; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd299; end
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd303; end
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd293; end
                    default: begin end
                endcase
            end
            7'd29: begin
                case (i_feature[4:0])
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd294; end
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd310; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd294; end
                    5'd5: begin o_has_replica = 1'b1; o_replica_bank = 3'd5; o_replica_addr = 9'd292; end
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd289; end
                    default: begin end
                endcase
            end
            7'd30: begin
                case (i_feature[4:0])
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd314; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd306; end
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd5; o_replica_addr = 9'd289; end
                    5'd9: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd291; end
                    default: begin end
                endcase
            end
            7'd31: begin
                case (i_feature[4:0])
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd3; o_replica_addr = 9'd291; end
                    5'd9: begin o_has_replica = 1'b1; o_replica_bank = 3'd3; o_replica_addr = 9'd293; end
                    default: begin end
                endcase
            end
            7'd44: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd292; end
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd300; end
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd301; end
                    default: begin end
                endcase
            end
            7'd45: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd295; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd305; end
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd289; end
                    default: begin end
                endcase
            end
            7'd46: begin
                case (i_feature[4:0])
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd303; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd302; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd289; end
                    5'd5: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd294; end
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd298; end
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd305; end
                    default: begin end
                endcase
            end
            7'd47: begin
                case (i_feature[4:0])
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd292; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd297; end
                    5'd5: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd290; end
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd290; end
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd293; end
                    5'd10: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd313; end
                    5'd11: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd295; end
                    5'd12: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd303; end
                    default: begin end
                endcase
            end
            7'd48: begin
                case (i_feature[4:0])
                    5'd5: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd297; end
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd304; end
                    default: begin end
                endcase
            end
            7'd49: begin
                case (i_feature[4:0])
                    5'd6: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd295; end
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd3; o_replica_addr = 9'd292; end
                    default: begin end
                endcase
            end
            7'd50: begin
                case (i_feature[4:0])
                    5'd9: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd294; end
                    default: begin end
                endcase
            end
            7'd55: begin
                case (i_feature[4:0])
                    5'd16: begin o_has_replica = 1'b1; o_replica_bank = 3'd4; o_replica_addr = 9'd294; end
                    default: begin end
                endcase
            end
            7'd57: begin
                case (i_feature[4:0])
                    5'd31: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd296; end
                    default: begin end
                endcase
            end
            7'd59: begin
                case (i_feature[4:0])
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd295; end
                    default: begin end
                endcase
            end
            7'd60: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd293; end
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd299; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd3; o_replica_addr = 9'd290; end
                    5'd30: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd307; end
                    5'd31: begin o_has_replica = 1'b1; o_replica_bank = 3'd5; o_replica_addr = 9'd291; end
                    default: begin end
                endcase
            end
            7'd61: begin
                case (i_feature[4:0])
                    5'd0: begin o_has_replica = 1'b1; o_replica_bank = 3'd6; o_replica_addr = 9'd292; end
                    5'd1: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd296; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd307; end
                    5'd5: begin o_has_replica = 1'b1; o_replica_bank = 3'd2; o_replica_addr = 9'd289; end
                    default: begin end
                endcase
            end
            7'd62: begin
                case (i_feature[4:0])
                    5'd2: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd299; end
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd7; o_replica_addr = 9'd291; end
                    5'd4: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd305; end
                    default: begin end
                endcase
            end
            7'd63: begin
                case (i_feature[4:0])
                    5'd3: begin o_has_replica = 1'b1; o_replica_bank = 3'd0; o_replica_addr = 9'd302; end
                    5'd7: begin o_has_replica = 1'b1; o_replica_bank = 3'd3; o_replica_addr = 9'd289; end
                    default: begin end
                endcase
            end
            7'd64: begin
                case (i_feature[4:0])
                    5'd8: begin o_has_replica = 1'b1; o_replica_bank = 3'd1; o_replica_addr = 9'd309; end
                    default: begin end
                endcase
            end
            default: begin end
        endcase
    end
endmodule
