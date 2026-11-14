`timescale 1ns/1ps

// R19A inserts a registered effective-depth snapshot before the two R17
// balancing stages. The scheduler owns committed depth; this adapter adds the
// exact reservations of all in-flight windows before capturing the next S0.
module unified_input2312_depth_snapshot_replica_ingress_adapter8 #(
    parameter integer SAMPLE_BITS = 16,
    parameter integer T_BITS = 9,
    parameter integer RESERVATION_MODE = 2,
    parameter integer USE_PREFIX_BALANCE4 = 0
) (
    input wire clk,
    input wire rst_n,
    input wire i_window_valid,
    output wire o_window_ready,
    input wire [7:0] i_record_valid,
    input wire [SAMPLE_BITS-1:0] i_sample_id,
    input wire [T_BITS-1:0] i_timestep,
    input wire [7:0] i_boundary_mask,
    input wire i_sample_last,
    input wire [8*12-1:0] i_fi_flat,
    input wire [8*8-1:0] i_x_flat,
    input wire [8*8-1:0] i_scheduler_queue_depth_flat,
    output wire o_scheduler_window_valid,
    input wire i_scheduler_window_ready,
    output wire [7:0] o_record_valid,
    output wire [SAMPLE_BITS-1:0] o_sample_id,
    output wire [T_BITS-1:0] o_timestep,
    output wire [7:0] o_boundary_mask,
    output wire o_sample_last,
    output wire [8*12-1:0] o_fi_flat,
    output wire [8*8-1:0] o_x_flat
);
    reg s0_valid;
    reg [7:0] s0_record_valid;
    reg [SAMPLE_BITS-1:0] s0_sample_id;
    reg [T_BITS-1:0] s0_timestep;
    reg [7:0] s0_boundary_mask;
    reg s0_sample_last;
    reg [8*12-1:0] s0_fi_flat;
    reg [8*8-1:0] s0_x_flat;
    reg [8*8-1:0] s0_start_depth_flat;

    reg s1_valid;
    reg [7:0] s1_record_valid;
    reg [SAMPLE_BITS-1:0] s1_sample_id;
    reg [T_BITS-1:0] s1_timestep;
    reg [7:0] s1_boundary_mask;
    reg s1_sample_last;
    reg [8*8-1:0] s1_x_flat;
    reg [4*12-1:0] s1_low_physical_fi_flat;
    reg [3:0] s1_high_event_valid;
    reg [3:0] s1_high_has_replica;
    reg [4*3-1:0] s1_high_primary_bank_flat;
    reg [4*9-1:0] s1_high_primary_addr_flat;
    reg [4*3-1:0] s1_high_replica_bank_flat;
    reg [4*9-1:0] s1_high_replica_addr_flat;
    reg [8*8-1:0] s1_depth_after_low_flat;
    reg [8*4-1:0] s1_low_increment_flat;
    reg [8*4-1:0] s1_high_predicted_increment_flat;

    reg s2_valid;
    reg [7:0] s2_record_valid;
    reg [SAMPLE_BITS-1:0] s2_sample_id;
    reg [T_BITS-1:0] s2_timestep;
    reg [7:0] s2_boundary_mask;
    reg s2_sample_last;
    reg [8*8-1:0] s2_x_flat;
    reg [8*12-1:0] s2_physical_fi_flat;
    reg [8*4-1:0] s2_bank_increment_flat;

    wire s2_ready = !s2_valid || i_scheduler_window_ready;
    wire s1_ready = !s1_valid || s2_ready;
    wire s0_ready = !s0_valid || s1_ready;

    wire [7:0] s0_event_valid = s0_record_valid & ~s0_boundary_mask;
    wire [7:0] map_has_replica;
    wire [8*3-1:0] map_primary_bank_flat;
    wire [8*9-1:0] map_primary_addr_flat;
    wire [8*3-1:0] map_replica_bank_flat;
    wire [8*9-1:0] map_replica_addr_flat;
    wire [4*12-1:0] low_physical_fi_flat;
    wire [8*8-1:0] low_end_depth_flat;
    wire [8*4-1:0] low_increment_flat;
    wire [4*12-1:0] high_physical_fi_flat;
    wire [8*8-1:0] high_end_depth_flat;
    wire [8*4-1:0] high_increment_flat;

    reg [8*4-1:0] s0_high_predicted_increment_flat;
    reg [3:0] s0_high_predicted_increment [0:7];
    reg [2:0] s0_high_predicted_bank;
    reg [8*4-1:0] combined_increment_flat;
    reg [8*8-1:0] capture_start_depth_flat;
    integer bank;
    integer lane;

    genvar map_lane;
    generate
        for (map_lane = 0; map_lane < 8; map_lane = map_lane + 1) begin : gen_map
            assign map_primary_bank_flat[map_lane*3 +: 3] =
                s0_fi_flat[map_lane*12 +: 3] ^ {
                    s0_fi_flat[map_lane*12 + 6],
                    s0_fi_flat[map_lane*12 + 8],
                    s0_fi_flat[map_lane*12 + 7]
                };
            assign map_primary_addr_flat[map_lane*9 +: 9] =
                s0_fi_flat[map_lane*12 + 3 +: 9];
            unified_input2312_replica_map_bucketed u_map (
                .i_feature(s0_fi_flat[map_lane*12 +: 12]),
                .o_has_replica(map_has_replica[map_lane]),
                .o_replica_bank(map_replica_bank_flat[map_lane*3 +: 3]),
                .o_replica_addr(map_replica_addr_flat[map_lane*9 +: 9])
            );
        end
    endgenerate

    always @* begin
        s0_high_predicted_increment_flat = 0;
        s0_high_predicted_bank = 0;
        for (bank = 0; bank < 8; bank = bank + 1)
            s0_high_predicted_increment[bank] = 0;
        for (lane = 4; lane < 8; lane = lane + 1) begin
            if (s0_event_valid[lane]) begin
                if (map_has_replica[lane] &&
                    s0_start_depth_flat[map_replica_bank_flat[lane*3 +: 3]*8 +: 8] <
                    s0_start_depth_flat[map_primary_bank_flat[lane*3 +: 3]*8 +: 8])
                    s0_high_predicted_bank = map_replica_bank_flat[lane*3 +: 3];
                else
                    s0_high_predicted_bank = map_primary_bank_flat[lane*3 +: 3];
                s0_high_predicted_increment[s0_high_predicted_bank] =
                    s0_high_predicted_increment[s0_high_predicted_bank] + 1'b1;
            end
        end
        for (bank = 0; bank < 8; bank = bank + 1)
            s0_high_predicted_increment_flat[bank*4 +: 4] =
                s0_high_predicted_increment[bank];
    end

    always @* begin
        combined_increment_flat = 0;
        capture_start_depth_flat = i_scheduler_queue_depth_flat;
        for (bank = 0; bank < 8; bank = bank + 1) begin
            combined_increment_flat[bank*4 +: 4] =
                s1_low_increment_flat[bank*4 +: 4] +
                high_increment_flat[bank*4 +: 4];
            if (s2_valid)
                capture_start_depth_flat[bank*8 +: 8] =
                    capture_start_depth_flat[bank*8 +: 8] +
                    s2_bank_increment_flat[bank*4 +: 4];
            if (s1_valid)
                capture_start_depth_flat[bank*8 +: 8] =
                    capture_start_depth_flat[bank*8 +: 8] +
                    s1_low_increment_flat[bank*4 +: 4] +
                    s1_high_predicted_increment_flat[bank*4 +: 4];
            if (s0_valid)
                capture_start_depth_flat[bank*8 +: 8] =
                    capture_start_depth_flat[bank*8 +: 8] +
                    low_increment_flat[bank*4 +: 4] +
                    s0_high_predicted_increment_flat[bank*4 +: 4];
        end
    end

    generate
        if (USE_PREFIX_BALANCE4 != 0) begin : gen_prefix_balance4
            newarch_n9r19b_prefix_replica_balance4 u_balance_low (
                .i_event_valid(s0_event_valid[3:0]),
                .i_primary_bank_flat(map_primary_bank_flat[4*3-1:0]),
                .i_primary_addr_flat(map_primary_addr_flat[4*9-1:0]),
                .i_has_replica(map_has_replica[3:0]),
                .i_replica_bank_flat(map_replica_bank_flat[4*3-1:0]),
                .i_replica_addr_flat(map_replica_addr_flat[4*9-1:0]),
                .i_start_depth_flat(s0_start_depth_flat),
                .o_physical_fi_flat(low_physical_fi_flat),
                .o_end_depth_flat(low_end_depth_flat),
                .o_bank_increment_flat(low_increment_flat)
            );

            newarch_n9r19b_prefix_replica_balance4 u_balance_high (
                .i_event_valid(s1_high_event_valid),
                .i_primary_bank_flat(s1_high_primary_bank_flat),
                .i_primary_addr_flat(s1_high_primary_addr_flat),
                .i_has_replica(s1_high_has_replica),
                .i_replica_bank_flat(s1_high_replica_bank_flat),
                .i_replica_addr_flat(s1_high_replica_addr_flat),
                .i_start_depth_flat(s1_depth_after_low_flat),
                .o_physical_fi_flat(high_physical_fi_flat),
                .o_end_depth_flat(high_end_depth_flat),
                .o_bank_increment_flat(high_increment_flat)
            );
        end else begin : gen_serial_balance4
            newarch_n9r17_replica_balance4 u_balance_low (
                .i_event_valid(s0_event_valid[3:0]),
                .i_primary_bank_flat(map_primary_bank_flat[4*3-1:0]),
                .i_primary_addr_flat(map_primary_addr_flat[4*9-1:0]),
                .i_has_replica(map_has_replica[3:0]),
                .i_replica_bank_flat(map_replica_bank_flat[4*3-1:0]),
                .i_replica_addr_flat(map_replica_addr_flat[4*9-1:0]),
                .i_start_depth_flat(s0_start_depth_flat),
                .o_physical_fi_flat(low_physical_fi_flat),
                .o_end_depth_flat(low_end_depth_flat),
                .o_bank_increment_flat(low_increment_flat)
            );

            newarch_n9r17_replica_balance4 u_balance_high (
                .i_event_valid(s1_high_event_valid),
                .i_primary_bank_flat(s1_high_primary_bank_flat),
                .i_primary_addr_flat(s1_high_primary_addr_flat),
                .i_has_replica(s1_high_has_replica),
                .i_replica_bank_flat(s1_high_replica_bank_flat),
                .i_replica_addr_flat(s1_high_replica_addr_flat),
                .i_start_depth_flat(s1_depth_after_low_flat),
                .o_physical_fi_flat(high_physical_fi_flat),
                .o_end_depth_flat(high_end_depth_flat),
                .o_bank_increment_flat(high_increment_flat)
            );
        end
    endgenerate

    assign o_window_ready = s0_ready;
    assign o_scheduler_window_valid = s2_valid;
    assign o_record_valid = s2_record_valid;
    assign o_sample_id = s2_sample_id;
    assign o_timestep = s2_timestep;
    assign o_boundary_mask = s2_boundary_mask;
    assign o_sample_last = s2_sample_last;
    assign o_fi_flat = s2_physical_fi_flat;
    assign o_x_flat = s2_x_flat;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0;
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
            s0_record_valid <= 0;
            s0_sample_id <= 0;
            s0_timestep <= 0;
            s0_boundary_mask <= 0;
            s0_sample_last <= 1'b0;
            s0_fi_flat <= 0;
            s0_x_flat <= 0;
            s0_start_depth_flat <= 0;
            s1_record_valid <= 0;
            s1_sample_id <= 0;
            s1_timestep <= 0;
            s1_boundary_mask <= 0;
            s1_sample_last <= 1'b0;
            s1_x_flat <= 0;
            s1_low_physical_fi_flat <= 0;
            s1_high_event_valid <= 0;
            s1_high_has_replica <= 0;
            s1_high_primary_bank_flat <= 0;
            s1_high_primary_addr_flat <= 0;
            s1_high_replica_bank_flat <= 0;
            s1_high_replica_addr_flat <= 0;
            s1_depth_after_low_flat <= 0;
            s1_low_increment_flat <= 0;
            s1_high_predicted_increment_flat <= 0;
            s2_record_valid <= 0;
            s2_sample_id <= 0;
            s2_timestep <= 0;
            s2_boundary_mask <= 0;
            s2_sample_last <= 1'b0;
            s2_x_flat <= 0;
            s2_physical_fi_flat <= 0;
            s2_bank_increment_flat <= 0;
        end else begin
            if (s2_ready) begin
                s2_valid <= s1_valid;
                if (s1_valid) begin
                    s2_record_valid <= s1_record_valid;
                    s2_sample_id <= s1_sample_id;
                    s2_timestep <= s1_timestep;
                    s2_boundary_mask <= s1_boundary_mask;
                    s2_sample_last <= s1_sample_last;
                    s2_x_flat <= s1_x_flat;
                    s2_physical_fi_flat <= {high_physical_fi_flat, s1_low_physical_fi_flat};
                    s2_bank_increment_flat <= combined_increment_flat;
                end
            end
            if (s1_ready) begin
                s1_valid <= s0_valid;
                if (s0_valid) begin
                    s1_record_valid <= s0_record_valid;
                    s1_sample_id <= s0_sample_id;
                    s1_timestep <= s0_timestep;
                    s1_boundary_mask <= s0_boundary_mask;
                    s1_sample_last <= s0_sample_last;
                    s1_x_flat <= s0_x_flat;
                    s1_low_physical_fi_flat <= low_physical_fi_flat;
                    s1_high_event_valid <= s0_event_valid[7:4];
                    s1_high_has_replica <= map_has_replica[7:4];
                    s1_high_primary_bank_flat <= map_primary_bank_flat[8*3-1:4*3];
                    s1_high_primary_addr_flat <= map_primary_addr_flat[8*9-1:4*9];
                    s1_high_replica_bank_flat <= map_replica_bank_flat[8*3-1:4*3];
                    s1_high_replica_addr_flat <= map_replica_addr_flat[8*9-1:4*9];
                    s1_depth_after_low_flat <= low_end_depth_flat;
                    s1_low_increment_flat <= low_increment_flat;
                    s1_high_predicted_increment_flat <= s0_high_predicted_increment_flat;
                end
            end
            if (s0_ready) begin
                s0_valid <= i_window_valid;
                if (i_window_valid) begin
                    s0_record_valid <= i_record_valid;
                    s0_sample_id <= i_sample_id;
                    s0_timestep <= i_timestep;
                    s0_boundary_mask <= i_boundary_mask;
                    s0_sample_last <= i_sample_last;
                    s0_fi_flat <= i_fi_flat;
                    s0_x_flat <= i_x_flat;
                    s0_start_depth_flat <= capture_start_depth_flat;
                end
            end
        end
    end

    initial begin
        if (RESERVATION_MODE != 2)
            $error("R19A depth snapshot requires balanced provisional reservation mode 2");
    end
endmodule
