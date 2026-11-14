param(
    [switch]$LearningOff,

    [ValidateSet(1, 2, 4, 7)]
    [int]$TopologyMask = 7
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Bash = 'C:\msys64\usr\bin\bash.exe'
$LearningVariant = if ($LearningOff) { 'learning_off' } else { 'learning_on' }
$Variant = "mask${TopologyMask}_$LearningVariant"
$Build = Join-Path $Root "sim_work\unified_h40_final_composition_v1_lint_$Variant"
New-Item -ItemType Directory -Force -Path $Build | Out-Null

function Convert-ToMsysPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if ($FullPath -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert non-drive Windows path: $FullPath"
    }
    return "/$($Matches[1].ToLowerInvariant())/$(($Matches[2] -replace '\\','/'))"
}

& python (Join-Path $Root 'scripts\generate_unified_h40_final_composition_top_v1.py')
if ($LASTEXITCODE -ne 0) {
    throw "Final-composition top generation failed: $LASTEXITCODE"
}

$UnixRoot = Convert-ToMsysPath $Root
$UnixBuild = Convert-ToMsysPath $Build
$Generics = "-GTOPOLOGY_MASK=$TopologyMask"
if ($LearningOff) {
    $Generics += ' -GENABLE_ONLINE_LEARNING=0'
}
$Command = @"
set -euo pipefail
export PATH=/mingw64/bin:/usr/bin:`$PATH
export VERILATOR_ROOT=/mingw64/share/verilator
cd '$UnixRoot'
verilator --lint-only --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-PINCONNECTEMPTY \
  -Wno-PINMISSING -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-BLKSEQ -Wno-CMPCONST \
  -Wno-SYNCASYNCNET -Wno-TIMESCALEMOD -Wno-WIDTHCONCAT \
  --top-module unified_h40_final_composition_top_v1 \
  $Generics \
  `$(find rtl -name '*.sv') \
  2>&1 | tee '$UnixBuild/lint.log'
"@

& $Bash -lc $Command
if ($LASTEXITCODE -ne 0) {
    throw "Unified H40 final-composition lint failed: $LASTEXITCODE"
}
Write-Output (
    'UNIFIED_H40_FINAL_COMPOSITION_V1_LINT_PASS ' +
    "topology_mask=$TopologyMask variant=$LearningVariant"
)
