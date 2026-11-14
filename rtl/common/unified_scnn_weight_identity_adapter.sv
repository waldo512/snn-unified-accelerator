`default_nettype none

module unified_scnn_weight_identity_adapter #(
    parameter int DATA_WIDTH = 16,
    parameter int ADDR_WIDTH = 17
) (
    input  logic                  i_enable,
    input  logic                  i_valid,
    output logic                  o_ready,
    input  logic [DATA_WIDTH-1:0] i_data,
    input  logic [ADDR_WIDTH-1:0] i_addr,
    output logic                  o_valid,
    input  logic                  i_ready,
    output logic [DATA_WIDTH-1:0] o_data,
    output logic [ADDR_WIDTH-1:0] o_addr
);

    assign o_ready = i_ready;
    assign o_valid = i_enable && i_valid;
    assign o_data = o_valid ? i_data : '0;
    assign o_addr = o_valid ? i_addr : '0;

endmodule

`default_nettype wire
