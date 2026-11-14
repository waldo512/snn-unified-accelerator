$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $root 'sim_work\unified_h40_array16_three_topology_graph_top_v1'
$obj = Join-Path $work 'obj_dir'
$verilator = 'C:\msys64\mingw64\bin\verilator_bin.exe'
$previousVerilatorRoot = $env:VERILATOR_ROOT
$previousPath = $env:PATH
New-Item -ItemType Directory -Force -Path $work | Out-Null
$env:VERILATOR_ROOT = 'C:\msys64\mingw64\share\verilator'
$env:PATH = "C:\msys64\mingw64\bin;C:\msys64\usr\bin;$env:PATH"
try {
  $sources = @(
    'rtl\common\unified_elastic_pipeline.sv',
    'rtl\common\unified_elastic_fifo2.sv',
    'rtl\memory\unified_dual_role_weight_store_candidate_v1.sv',
    'rtl\memory\unified_dual_role_weight_store_p4fc_candidate_v1.sv',
    'rtl\memory\unified_dual_role_weight_store_p4all_candidate_v1.sv',
    'rtl\memory\unified_dual_role_state_trace_store_candidate_v2.sv',
    'rtl\tiles\unified_dual_role_dsp_lane_candidate_v3.sv',
    'rtl\tiles\unified_dual_role_h40_tile_candidate_v3.sv',
    'rtl\tiles\unified_dual_role_h40_tile_p4fc_dual_accum_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_tile_p4all_dual_accum_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_dsp_lane_p4_cascade_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_tile_p4all_cascade_accum_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_dsp_add48_lane_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_p4fc_pair_merge10_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_current_normalizer10_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_lif10_arithmetic_candidate_v3.sv',
    'rtl\tiles\unified_dual_role_trace10_arithmetic_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_neuron_region_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_neuron_region_p4fc_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_group_sequencer_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_stationary_tile_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_stationary_region4_candidate_v1.sv',
    'rtl\tiles\unified_dual_role_h40_stationary_array12_candidate_v1.sv',
    'rtl\control\unified_h40_topology_geometry_v1.sv',
    'rtl\control\unified_fc_srnn_h40_p4_layer_client_v1.sv',
    'rtl\control\unified_binary_p4_destination_walker_v1.sv',
    'rtl\control\unified_scnn_h40_spatial_wave_scheduler_v1.sv',
    'rtl\control\unified_scnn_h40_classifier_p4_scheduler_v1.sv',
    'rtl\control\unified_h40_array16_topology_owner_v1.sv',
    'rtl\tiles\unified_binary_p4_h40_array12_path_v1.sv',
    'rtl\learning\unified_fc_srnn_h40_tp_observation_serializer_v1.sv',
    'rtl\learning\unified_fc_srnn_h40_tp_observation_reorder_v1.sv',
    'rtl\learning\unified_fc_srnn_h40_tp_replay_router_v1.sv',
    'rtl\learning\unified_fc_srnn_h40_tp_observation_ingress_v1.sv',
    'rtl\learning\unified_scnn_h40_tp_observation_serializer_v1.sv',
    'rtl\learning\unified_scnn_tp_surrogate_type1_q8_8.sv',
    'rtl\learning\unified_scnn_h40_conv8_scalar_observation_fork_v1.sv',
    'rtl\learning\unified_scnn_h40_classifier_feature_commit_adapter_v1.sv',
    'rtl\learning\unified_scnn_h40_classifier_logit_wave_adapter_v1.sv',
    'rtl\top\unified_scnn_h40_observation_writeback_v1.sv',
    'rtl\top\unified_scnn_h40_array16_layer_client_v1.sv',
    'rtl\top\unified_scnn_h40_classifier_tail_client_v1.sv',
    'rtl\top\unified_scnn_h40_array16_graph_client_v1.sv',
    'rtl\top\unified_h40_array16_three_topology_top_v1.sv',
    'tb\tb_unified_h40_array16_three_topology_graph_top_v1.sv'
  ) | ForEach-Object { Join-Path $root $_ }
  & $verilator --binary -j 0 --timing --assert -Wall -Wno-fatal `
    -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-PINCONNECTEMPTY `
    -Wno-PINMISSING -Wno-WIDTHCONCAT -Wno-WIDTHTRUNC `
    -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL `
    -CFLAGS '-D_GLIBCXX_USE_CXX11_ABI=0' `
    --top-module tb_unified_h40_array16_three_topology_graph_top_v1 `
    --Mdir $obj $sources
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & (Join-Path $obj 'Vtb_unified_h40_array16_three_topology_graph_top_v1.exe')
  exit $LASTEXITCODE
} finally {
  $env:VERILATOR_ROOT = $previousVerilatorRoot
  $env:PATH = $previousPath
}
