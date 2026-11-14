`timescale 1ns/1ps
`default_nettype none

// One physical two-slot/nine-bank feature scratchpad shared by the CONV and
// OUTPUT frontends. The response owner is captured with the accepted request,
// so a later owner transition cannot redirect a stalled response.
module unified_scnn_feature_memory_router_v1 #(
    parameter int LANES = 8,
    parameter int FEATURE_DEPTH = 9216,
    parameter int FEATURE_ADDR_WIDTH = $clog2(FEATURE_DEPTH)
) (
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,
    input  logic                                      i_clear_error,
    input  logic                                      i_output_owner,

    input  logic                                      i_conv_read_valid,
    output logic                                      o_conv_read_ready,
    input  logic                                      i_conv_read_slot,
    input  logic [31:0]                               i_conv_read_bank_flat,
    input  logic [LANES*FEATURE_ADDR_WIDTH-1:0]       i_conv_read_addr_flat,
    input  logic [LANES-1:0]                          i_conv_read_lane_mask,
    output logic                                      o_conv_result_valid,
    input  logic                                      i_conv_result_ready,
    output logic [LANES*16-1:0]                       o_conv_result_flat,
    output logic [LANES-1:0]                          o_conv_result_lane_mask,

    input  logic                                      i_output_read_valid,
    output logic                                      o_output_read_ready,
    input  logic                                      i_output_read_slot,
    input  logic [31:0]                               i_output_read_bank_flat,
    input  logic [LANES*FEATURE_ADDR_WIDTH-1:0]       i_output_read_addr_flat,
    input  logic [LANES-1:0]                          i_output_read_lane_mask,
    output logic                                      o_output_result_valid,
    input  logic                                      i_output_result_ready,
    output logic [LANES*16-1:0]                       o_output_result_flat,
    output logic [LANES-1:0]                          o_output_result_lane_mask,

    input  logic                                      i_conv_write_valid,
    output logic                                      o_conv_write_ready,
    input  logic                                      i_conv_write_slot,
    input  logic [3:0]                                i_conv_write_bank,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_conv_write_addr,
    input  logic signed [15:0]                        i_conv_write_data,

    input  logic                                      i_host_enable,
    input  logic                                      i_host_write_valid,
    output logic                                      o_host_write_ready,
    input  logic                                      i_host_write_slot,
    input  logic [3:0]                                i_host_write_bank,
    input  logic [FEATURE_ADDR_WIDTH-1:0]             i_host_write_addr,
    input  logic signed [15:0]                        i_host_write_data,

    output logic                                      o_busy,
    output logic                                      o_sticky_error,
    output logic [7:0]                                o_error_code
);
    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_READ_OWNER = 8'h01;
    localparam logic [7:0] ERROR_READ_CONFLICT = 8'h02;
    localparam logic [7:0] ERROR_WRITE_CONFLICT = 8'h03;
    localparam logic [7:0] ERROR_BANK_CONFLICT = 8'h04;
    localparam logic [7:0] ERROR_BOUNDS = 8'h05;

    logic scratch_read_valid;
    logic scratch_read_ready;
    logic scratch_read_slot;
    logic [31:0] scratch_read_bank_flat;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] scratch_read_addr_flat;
    logic [LANES-1:0] scratch_read_lane_mask;
    logic scratch_result_valid;
    logic scratch_result_ready;
    logic [LANES*16-1:0] scratch_result_flat;
    logic [LANES-1:0] scratch_result_lane_mask;

    logic read_command_valid_q;
    logic read_command_ready;
    logic read_command_output_owner_q;
    logic read_command_slot_q;
    logic [31:0] read_command_bank_flat_q;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] read_command_addr_flat_q;
    logic [LANES-1:0] read_command_lane_mask_q;
    logic selected_read_valid;
    logic selected_read_slot;
    logic [31:0] selected_read_bank_flat;
    logic [LANES*FEATURE_ADDR_WIDTH-1:0] selected_read_addr_flat;
    logic [LANES-1:0] selected_read_lane_mask;

    logic scratch_write_valid;
    logic scratch_write_ready;
    logic scratch_write_slot;
    logic [3:0] scratch_write_bank;
    logic [FEATURE_ADDR_WIDTH-1:0] scratch_write_addr;
    logic signed [15:0] scratch_write_data;

    logic write_command_valid_q;
    logic write_command_ready;
    logic write_command_slot_q;
    logic [3:0] write_command_bank_q;
    logic [FEATURE_ADDR_WIDTH-1:0] write_command_addr_q;
    logic signed [15:0] write_command_data_q;
    logic selected_write_valid;
    logic selected_write_slot;
    logic [3:0] selected_write_bank;
    logic [FEATURE_ADDR_WIDTH-1:0] selected_write_addr;
    logic signed [15:0] selected_write_data;

    logic scratch_response_output_owner_q;
    logic [1:0] response_count_q;
    logic response_read_ptr_q;
    logic response_write_ptr_q;
    logic response_output_owner_q [0:1];
    logic [LANES*16-1:0] response_flat_q [0:1];
    logic [LANES-1:0] response_lane_mask_q [0:1];
    logic response_valid;
    logic response_pop;
    logic response_push;
    logic read_accept;
    logic bank_conflict_error;
    logic bounds_error;
    logic local_error_q;
    logic [7:0] local_error_code_q;
    logic read_owner_error;
    logic read_conflict_error;
    logic write_conflict_error;

    always_comb begin
        selected_read_valid = i_output_owner ? i_output_read_valid :
                                              i_conv_read_valid;
        selected_read_slot = i_output_owner ? i_output_read_slot :
                                             i_conv_read_slot;
        selected_read_bank_flat = i_output_owner ? i_output_read_bank_flat :
                                                  i_conv_read_bank_flat;
        selected_read_addr_flat = i_output_owner ? i_output_read_addr_flat :
                                                  i_conv_read_addr_flat;
        selected_read_lane_mask = i_output_owner ? i_output_read_lane_mask :
                                                  i_conv_read_lane_mask;
        read_command_ready = !read_command_valid_q || scratch_read_ready;
        o_conv_read_ready = !i_output_owner && read_command_ready;
        o_output_read_ready = i_output_owner && read_command_ready;

        scratch_read_valid = read_command_valid_q;
        scratch_read_slot = read_command_slot_q;
        scratch_read_bank_flat = read_command_bank_flat_q;
        scratch_read_addr_flat = read_command_addr_flat_q;
        scratch_read_lane_mask = read_command_lane_mask_q;
        read_accept = scratch_read_valid && scratch_read_ready;

        response_valid = response_count_q != 0;
        response_pop = response_valid &&
            (response_output_owner_q[response_read_ptr_q] ?
                i_output_result_ready : i_conv_result_ready);
        // Count is registered, so external result backpressure cannot reach
        // the scratchpad BRAM read enables combinationally.
        scratch_result_ready = response_count_q != 2;
        response_push = scratch_result_valid && scratch_result_ready;
        o_conv_result_valid = response_valid &&
            !response_output_owner_q[response_read_ptr_q];
        o_output_result_valid = response_valid &&
            response_output_owner_q[response_read_ptr_q];
        o_conv_result_flat = response_flat_q[response_read_ptr_q];
        o_output_result_flat = response_flat_q[response_read_ptr_q];
        o_conv_result_lane_mask =
            response_lane_mask_q[response_read_ptr_q];
        o_output_result_lane_mask =
            response_lane_mask_q[response_read_ptr_q];

        selected_write_valid = i_conv_write_valid ||
            (i_host_write_valid && i_host_enable && !i_conv_write_valid);
        selected_write_slot = i_conv_write_valid ? i_conv_write_slot :
                                                  i_host_write_slot;
        selected_write_bank = i_conv_write_valid ? i_conv_write_bank :
                                                  i_host_write_bank;
        selected_write_addr = i_conv_write_valid ? i_conv_write_addr :
                                                  i_host_write_addr;
        selected_write_data = i_conv_write_valid ? i_conv_write_data :
                                                  i_host_write_data;
        write_command_ready = !write_command_valid_q || scratch_write_ready;
        o_conv_write_ready = write_command_ready;
        o_host_write_ready = i_host_enable && !i_conv_write_valid &&
                             write_command_ready;

        scratch_write_valid = write_command_valid_q;
        scratch_write_slot = write_command_slot_q;
        scratch_write_bank = write_command_bank_q;
        scratch_write_addr = write_command_addr_q;
        scratch_write_data = write_command_data_q;

        read_owner_error = (i_output_owner && i_conv_read_valid) ||
                           (!i_output_owner && i_output_read_valid);
        read_conflict_error = i_conv_read_valid && i_output_read_valid;
        write_conflict_error = i_conv_write_valid && i_host_write_valid;
        o_busy = response_valid || scratch_result_valid ||
                  read_command_valid_q ||
                  write_command_valid_q;
        o_sticky_error = local_error_q || bank_conflict_error || bounds_error;
        o_error_code = local_error_q ? local_error_code_q :
            bank_conflict_error ? ERROR_BANK_CONFLICT :
            bounds_error ? ERROR_BOUNDS : ERROR_NONE;
    end

    unified_scnn_k8_feature_scratchpad #(
        .BANK_DEPTH(FEATURE_DEPTH),
        .ADDR_WIDTH(FEATURE_ADDR_WIDTH)
    ) u_feature_scratchpad (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_write_valid(scratch_write_valid),
        .i_write_slot(scratch_write_slot),
        .i_write_bank(scratch_write_bank),
        .i_write_addr(scratch_write_addr),
        .i_write_data(scratch_write_data),
        .o_write_ready(scratch_write_ready),
        .i_read_valid(scratch_read_valid),
        .o_read_ready(scratch_read_ready),
        .i_read_slot(scratch_read_slot),
        .i_read_bank_flat(scratch_read_bank_flat),
        .i_read_addr_flat(scratch_read_addr_flat),
        .i_read_lane_mask(scratch_read_lane_mask),
        .o_read_valid(scratch_result_valid),
        .i_read_result_ready(scratch_result_ready),
        .o_read_data_flat(scratch_result_flat),
        .o_read_lane_mask(scratch_result_lane_mask),
        .o_bank_conflict_error(bank_conflict_error),
        .o_bounds_error(bounds_error)
    );

    always_ff @(posedge i_clk) begin
        if (!i_rst_n) begin
            read_command_valid_q <= 1'b0;
            write_command_valid_q <= 1'b0;
            response_count_q <= '0;
            response_read_ptr_q <= 1'b0;
            response_write_ptr_q <= 1'b0;
            scratch_response_output_owner_q <= 1'b0;
            local_error_q <= 1'b0;
            local_error_code_q <= ERROR_NONE;
        end else begin
            if (read_command_ready)
                read_command_valid_q <= selected_read_valid;
            if (write_command_ready)
                write_command_valid_q <= selected_write_valid;
            unique case ({response_push, response_pop})
                2'b10: response_count_q <= response_count_q + 1'b1;
                2'b01: response_count_q <= response_count_q - 1'b1;
                default: response_count_q <= response_count_q;
            endcase
            if (response_push)
                response_write_ptr_q <= response_write_ptr_q + 1'b1;
            if (response_pop)
                response_read_ptr_q <= response_read_ptr_q + 1'b1;
            if (read_accept)
                scratch_response_output_owner_q <=
                    read_command_output_owner_q;
            if (i_clear_error && !o_busy) begin
                local_error_q <= 1'b0;
                local_error_code_q <= ERROR_NONE;
            end else if (!local_error_q && read_conflict_error) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_READ_CONFLICT;
            end else if (!local_error_q && read_owner_error) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_READ_OWNER;
            end else if (!local_error_q && write_conflict_error) begin
                local_error_q <= 1'b1;
                local_error_code_q <= ERROR_WRITE_CONFLICT;
            end
        end
    end

    // Payload registers are reset-free. Their valid bits are the sole authority,
    // which avoids distributing reset onto the wide request buses.
    always_ff @(posedge i_clk) begin
        if (read_command_ready && selected_read_valid) begin
            read_command_output_owner_q <= i_output_owner;
            read_command_slot_q <= selected_read_slot;
            read_command_bank_flat_q <= selected_read_bank_flat;
            read_command_addr_flat_q <= selected_read_addr_flat;
            read_command_lane_mask_q <= selected_read_lane_mask;
        end
        if (write_command_ready && selected_write_valid) begin
            write_command_slot_q <= selected_write_slot;
            write_command_bank_q <= selected_write_bank;
            write_command_addr_q <= selected_write_addr;
            write_command_data_q <= selected_write_data;
        end
        if (response_push) begin
            response_output_owner_q[response_write_ptr_q] <=
                scratch_response_output_owner_q;
            response_flat_q[response_write_ptr_q] <= scratch_result_flat;
            response_lane_mask_q[response_write_ptr_q] <=
                scratch_result_lane_mask;
        end
    end
endmodule

`default_nettype wire
