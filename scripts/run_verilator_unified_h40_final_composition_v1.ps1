$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Bash = 'C:\msys64\usr\bin\bash.exe'
$Build = Join-Path $Root 'sim_work\unified_h40_final_composition_v1'
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
& python (Join-Path $Root 'scripts\generate_tb_unified_h40_final_composition_v1.py')
if ($LASTEXITCODE -ne 0) {
    throw "Final-composition testbench generation failed: $LASTEXITCODE"
}

$UnixRoot = Convert-ToMsysPath $Root
$UnixBuild = Convert-ToMsysPath $Build
$Command = @"
set -euo pipefail
export PATH=/mingw64/bin:/usr/bin:`$PATH
export VERILATOR_ROOT=/mingw64/share/verilator
cd '$UnixRoot'
verilator --binary -j 0 --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-PINCONNECTEMPTY \
  -Wno-PINMISSING -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
  -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-BLKSEQ -Wno-CMPCONST \
  -Wno-SYNCASYNCNET -Wno-TIMESCALEMOD -Wno-WIDTHCONCAT \
  -CFLAGS '-D_GLIBCXX_USE_CXX11_ABI=0' \
  --top-module tb_unified_h40_final_composition_v1 \
  --Mdir '$UnixBuild/obj_dir' \
  `$(find rtl -name '*.sv') \
  tb/tb_unified_h40_final_composition_v1.sv
'$UnixBuild/obj_dir/Vtb_unified_h40_final_composition_v1.exe'
"@

& $Bash -lc $Command
if ($LASTEXITCODE -ne 0) {
    throw "Unified H40 final-composition regression failed: $LASTEXITCODE"
}
Write-Output 'UNIFIED_H40_FINAL_COMPOSITION_V1_PASS'
