#!/usr/bin/env python3
"""Summarize deterministic current-source SRNN M2 online cycle evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


MARKER = "UNIFIED_SNN_CORE_FINAL_V1_SRNN_M2_ONLINE_PASS"


def parse_fields(line: str) -> dict[str, int | str]:
    fields: dict[str, int | str] = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", line):
        fields[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return fields


def read_marker(path: Path) -> tuple[str, dict[str, int | str]]:
    matches = [
        line
        for line in path.read_text(encoding="ascii", errors="strict").splitlines()
        if line.startswith(MARKER + " ")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected one PASS marker, found {len(matches)}")
    return matches[0], parse_fields(matches[0])


def require_int(fields: dict[str, int | str], key: str) -> int:
    value = fields.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"missing integer field {key}: {fields}")
    return value


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "results" / "learning_reference" / "actual_srnn_m2_online"
    logs = [log_dir / "run1.log", log_dir / "run2.log"]
    markers = [read_marker(path) for path in logs]
    if markers[0][0] != markers[1][0]:
        raise RuntimeError("SRNN M2 PASS markers are not deterministic")

    fields = markers[0][1]
    train_cycles = require_int(fields, "train_cycles")
    inference_cycles = require_int(fields, "inference_cycles")
    incremental_cycles = require_int(fields, "incremental_learning_cycles")
    if train_cycles - inference_cycles != incremental_cycles:
        raise RuntimeError("incremental learning cycle arithmetic is inconsistent")
    if require_int(fields, "changed_after_training") <= 0:
        raise RuntimeError("SRNN M2 evidence contains no changed parameter")
    if require_int(fields, "learning_off_mismatch") != 0:
        raise RuntimeError("updated-weight learning-off replay mismatch")

    summary = {
        "schema": "srnn_m2_current_online_cycles_v1",
        "status": "current_bounded_rtl_simulated",
        "topology": "SRNN",
        "selected_mask": "Wout-only",
        "workload": "two real SHD samples x two captured timesteps",
        "runs": 2,
        "deterministic": True,
        "train_cycles": train_cycles,
        "learning_off_cycles": inference_cycles,
        "incremental_learning_cycles": incremental_cycles,
        "changed_parameters": require_int(fields, "changed_after_training"),
        "commits": require_int(fields, "m2_commits"),
        "model_version": require_int(fields, "model_version"),
        "adapt_version": require_int(fields, "adapt_version"),
        "evidence_label": "Verilator RTL simulated",
        "not_evidence_for": [
            "complete support=128 adaptation episode latency",
            "post-route functional timing",
            "board-measured throughput",
            "cross-topology speed ranking",
        ],
        "source_logs": [str(path.relative_to(root)).replace("\\", "/") for path in logs],
        "marker": markers[0][0],
    }

    out_dir = (
        root
        / "results"
        / "paper_evidence"
        / "rtl"
        / "srnn_m2_current_cycles_20260928"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    with (out_dir / "summary.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topology",
                "selected_mask",
                "workload",
                "train_cycles",
                "learning_off_cycles",
                "incremental_learning_cycles",
                "changed_parameters",
                "runs",
                "evidence_label",
            ],
        )
        writer.writeheader()
        writer.writerow({key: summary[key] for key in writer.fieldnames})

    readme = f"""# Current SRNN M2 Wout-Only Cycle Evidence

Updated: 2026-09-28

Artifact ID: `RTL-SRNN-M2-CYCLES-20260928`

Status: `CURRENT/BOUNDED RTL SIMULATED`

Two deterministic Verilator runs exercise the current production SRNN M2
Wout-only path with two real SHD samples and two captured timesteps per sample.
The learning transaction takes {train_cycles:,} cycles. The matched
learning-disabled updated-weight transaction takes {inference_cycles:,} cycles,
leaving {incremental_cycles:,} incremental learning cycles for this directed
boundary. Eight parameters change, two commits complete, model/adaptation
versions advance to one, and the next learning-disabled replay has zero
mismatches.

This is not the latency of the complete support=128 algorithm report. It is a
bounded current-source RTL transaction used to characterize the implemented
Wout-only forward/learn/commit path. It is not post-route simulation, board
throughput, or a cross-topology speed comparison.

Sources:

- `results/learning_reference/actual_srnn_m2_online/run1.log`
- `results/learning_reference/actual_srnn_m2_online/run2.log`
- `tb/tb_unified_snn_core_final_v1_srnn_m2_online.sv`
- `scripts/run_verilator_unified_snn_core_final_v1_srnn_m2_online.ps1`
"""
    (out_dir / "README.md").write_text(readme, encoding="ascii")
    print(
        "SRNN_M2_CURRENT_CYCLES_SUMMARY_PASS "
        f"runs=2 train={train_cycles} inference={inference_cycles} "
        f"incremental={incremental_cycles}"
    )


if __name__ == "__main__":
    main()
