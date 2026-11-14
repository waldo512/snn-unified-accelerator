`default_nettype none

// Maps a flattened CHW target spike into the nine-bank feature scratchpad
// layout used by the shared K8 convolution reader. No feature image is kept
// here; accepted values are written directly into the selected scratchpad slot.
module unified_scnn_tp_target_feature_writer #(
    parameter int OUTPUT_CHANNELS = 64,
    parameter int OUTPUT_SPATIAL = 32,
    parameter int CHANNEL_WIDTH = OUTPUT_CHANNELS <= 2 ? 1 :
        $clog2(OUTPUT_CHANNELS),
    parameter int SPATIAL_INDEX_WIDTH = OUTPUT_SPATIAL*OUTPUT_SPATIAL <= 2 ? 1 :
        $clog2(OUTPUT_SPATIAL*OUTPUT_SPATIAL),
    parameter int BANK_DEPTH = 9216,
    parameter int BANK_ADDR_WIDTH = $clog2(BANK_DEPTH)
) (
    input  logic                                  i_valid,
    output logic                                  o_ready,
    input  logic [CHANNEL_WIDTH-1:0]              i_output_channel,
    input  logic [SPATIAL_INDEX_WIDTH-1:0]        i_spatial_index,
    input  logic                                  i_output_slot,
    input  logic [15:0]                           i_spike_q8_8,

    output logic                                  o_write_valid,
    input  logic                                  i_write_ready,
    output logic                                  o_write_slot,
    output logic [3:0]                            o_write_bank,
    output logic [BANK_ADDR_WIDTH-1:0]            o_write_addr,
    output logic [15:0]                           o_write_data_q8_8,
    output logic                                  o_mapping_error
);
    localparam int BANK_GRID = (OUTPUT_SPATIAL+2)/3;
    localparam int CHANNEL_BANK_WORDS = BANK_GRID*BANK_GRID;
    localparam int OUTPUT_PLANE = OUTPUT_SPATIAL*OUTPUT_SPATIAL;

    integer row_index;
    integer column_index;
    integer bank_index;
    integer local_address;
    logic mapping_in_range;

    always_comb begin
        row_index = int'(i_spatial_index) / OUTPUT_SPATIAL;
        column_index = int'(i_spatial_index) % OUTPUT_SPATIAL;
        bank_index = (row_index % 3)*3 + column_index % 3;
        local_address = int'(i_output_channel)*CHANNEL_BANK_WORDS +
            (row_index/3)*BANK_GRID + column_index/3;
        mapping_in_range = int'(i_output_channel) < OUTPUT_CHANNELS &&
            int'(i_spatial_index) < OUTPUT_PLANE && bank_index < 9 &&
            local_address < BANK_DEPTH;

        o_ready = mapping_in_range && i_write_ready;
        o_write_valid = i_valid && mapping_in_range;
        o_write_slot = i_output_slot;
        o_write_bank = 4'(bank_index);
        o_write_addr = BANK_ADDR_WIDTH'(local_address);
        o_write_data_q8_8 = i_spike_q8_8;
        o_mapping_error = i_valid && !mapping_in_range;
    end

    initial begin
        if (OUTPUT_CHANNELS < 1 || OUTPUT_SPATIAL < 1 || BANK_DEPTH < 1)
            $error("SCNN TP target feature-writer dimensions are invalid");
        if (OUTPUT_CHANNELS*CHANNEL_BANK_WORDS > BANK_DEPTH)
            $error("SCNN TP target feature image exceeds one scratchpad bank");
    end
endmodule

`default_nettype wire
