`default_nettype none

// Atomically forks one serialized SCNN LIF result to the inference pool and
// the optional TP observer. Each sink may accept on a different cycle; the
// upstream token retires only after every enabled sink has accepted it once.
module unified_scnn_tp_lif_observation_fork #(
    parameter int ENGINE_INDEX_WIDTH = 4,
    parameter int SPATIAL_ADDR_WIDTH = 15,
    parameter int STATE_ADDR_WIDTH = 19,
    parameter int VERSION_WIDTH = 16,
    parameter SURROGATE_ROM_FILE =
        "rtl/learning/data/unified_scnn_tp_surrogate_type1_q8_8.mem"
) (
    input  logic                                i_clk,
    input  logic                                i_rst_n,
    input  logic                                i_abort,
    input  logic                                i_observation_enable,
    input  logic                                i_valid,
    output logic                                o_ready,
    input  logic [ENGINE_INDEX_WIDTH-1:0]       i_engine_index,
    input  logic [SPATIAL_ADDR_WIDTH-1:0]       i_spatial_addr,
    input  logic [STATE_ADDR_WIDTH-1:0]         i_output_addr,
    input  logic [VERSION_WIDTH-1:0]            i_sample_version,
    input  logic signed [15:0]                  i_current_q8_8,
    input  logic        [15:0]                  i_spike_q8_8,
    input  logic signed [15:0]                  i_membrane_pre_q8_8,
    input  logic signed [15:0]                  i_membrane_next_q8_8,
    input  logic signed [15:0]                  i_threshold_q8_8,
    input  logic [15:0]                         i_channel_base,

    output logic                                o_pool_valid,
    input  logic                                i_pool_ready,
    output logic [ENGINE_INDEX_WIDTH-1:0]       o_pool_engine_index,
    output logic [SPATIAL_ADDR_WIDTH-1:0]       o_pool_spatial_addr,
    output logic        [15:0]                  o_pool_spike_q8_8,

    output logic                                o_observation_valid,
    input  logic                                i_observation_ready,
    output logic [15:0]                         o_observation_channel,
    output logic [SPATIAL_ADDR_WIDTH-1:0]       o_observation_spatial_addr,
    output logic [STATE_ADDR_WIDTH-1:0]         o_observation_output_addr,
    output logic [VERSION_WIDTH-1:0]            o_observation_sample_version,
    output logic signed [15:0]                  o_observation_current_q8_8,
    output logic        [15:0]                  o_observation_spike_q8_8,
    output logic signed [15:0]                  o_observation_membrane_pre_q8_8,
    output logic signed [15:0]                  o_observation_membrane_next_q8_8,
    output logic        [15:0]                  o_observation_surrogate_q8_8,
    output logic                                o_protocol_error
);
    logic pool_accepted_q;
    logic observer_accepted_q;
    logic pool_fire;
    logic observer_issue_valid;
    logic observer_issue_fire;
    logic metadata_ready;
    logic surrogate_ready;
    logic surrogate_valid;
    logic [15:0] surrogate_value;
    logic observation_valid_q;

    logic [15:0] observation_channel_q;
    logic [SPATIAL_ADDR_WIDTH-1:0] observation_spatial_q;
    logic [STATE_ADDR_WIDTH-1:0] observation_output_addr_q;
    logic [VERSION_WIDTH-1:0] observation_version_q;
    logic signed [15:0] observation_current_q;
    logic [15:0] observation_spike_q;
    logic signed [15:0] observation_membrane_pre_q;
    logic signed [15:0] observation_membrane_next_q;

    assign o_pool_valid = i_valid && !pool_accepted_q;
    assign o_pool_engine_index = i_engine_index;
    assign o_pool_spatial_addr = i_spatial_addr;
    assign o_pool_spike_q8_8 = i_spike_q8_8;
    assign pool_fire = o_pool_valid && i_pool_ready;

    assign metadata_ready = !observation_valid_q || i_observation_ready;
    assign observer_issue_valid = i_valid && !observer_accepted_q &&
                                  i_observation_enable;
    assign observer_issue_fire = observer_issue_valid && metadata_ready &&
                                 surrogate_ready;
    assign o_ready = (pool_accepted_q || i_pool_ready) &&
                     (observer_accepted_q || !i_observation_enable ||
                      (metadata_ready && surrogate_ready));

    assign o_observation_valid = observation_valid_q && surrogate_valid;
    assign o_observation_channel = observation_channel_q;
    assign o_observation_spatial_addr = observation_spatial_q;
    assign o_observation_output_addr = observation_output_addr_q;
    assign o_observation_sample_version = observation_version_q;
    assign o_observation_current_q8_8 = observation_current_q;
    assign o_observation_spike_q8_8 = observation_spike_q;
    assign o_observation_membrane_pre_q8_8 = observation_membrane_pre_q;
    assign o_observation_membrane_next_q8_8 = observation_membrane_next_q;
    assign o_observation_surrogate_q8_8 = surrogate_value;
    assign o_protocol_error = observation_valid_q != surrogate_valid;

    unified_scnn_tp_surrogate_type1_q8_8 #(
        .ROM_FILE(SURROGATE_ROM_FILE)
    ) u_surrogate (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_abort(i_abort),
        .i_valid(observer_issue_fire), .o_ready(surrogate_ready),
        .i_membrane_pre_q8_8(i_membrane_pre_q8_8),
        .i_threshold_q8_8(i_threshold_q8_8),
        .o_valid(surrogate_valid), .i_ready(i_observation_ready),
        .o_surrogate_q8_8(surrogate_value)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            pool_accepted_q <= 1'b0;
            observer_accepted_q <= 1'b0;
            observation_valid_q <= 1'b0;
            observation_channel_q <= '0;
            observation_spatial_q <= '0;
            observation_output_addr_q <= '0;
            observation_version_q <= '0;
            observation_current_q <= '0;
            observation_spike_q <= '0;
            observation_membrane_pre_q <= '0;
            observation_membrane_next_q <= '0;
        end else if (i_abort) begin
            pool_accepted_q <= 1'b0;
            observer_accepted_q <= 1'b0;
            observation_valid_q <= 1'b0;
            observation_channel_q <= '0;
            observation_spatial_q <= '0;
            observation_output_addr_q <= '0;
            observation_version_q <= '0;
            observation_current_q <= '0;
            observation_spike_q <= '0;
            observation_membrane_pre_q <= '0;
            observation_membrane_next_q <= '0;
        end else begin
            if (i_valid && o_ready) begin
                pool_accepted_q <= 1'b0;
                observer_accepted_q <= 1'b0;
            end else begin
                if (pool_fire)
                    pool_accepted_q <= 1'b1;
                if (observer_issue_fire)
                    observer_accepted_q <= 1'b1;
            end

            if (metadata_ready) begin
                observation_valid_q <= observer_issue_fire;
                if (observer_issue_fire) begin
                    observation_channel_q <= i_channel_base +
                        16'(i_engine_index);
                    observation_spatial_q <= i_spatial_addr;
                    observation_output_addr_q <= i_output_addr;
                    observation_version_q <= i_sample_version;
                    observation_current_q <= i_current_q8_8;
                    observation_spike_q <= i_spike_q8_8;
                    observation_membrane_pre_q <= i_membrane_pre_q8_8;
                    observation_membrane_next_q <= i_membrane_next_q8_8;
                end
            end
        end
    end
endmodule

`default_nettype wire
