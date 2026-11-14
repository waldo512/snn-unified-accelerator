#!/usr/bin/env python3
"""Run one locked FC H160 candidate on the untouched official N-MNIST test split."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EPISODE_RUNNER = ROOT / "scripts" / "run_fc_tp_h160_mask_ablation.py"
DEFAULT_LOCK = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "fc_h160_w2_int8_magnitude_rtl_uq6_10_lr2m14_selection"
    / "locked_report_protocol.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--traces-root", type=Path, default=Path(r"D:\Desktop\traces_propagation-main"))
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\datasets"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=(
            ROOT
            / "imports"
            / "fc"
            / "p3a_h160"
            / "data"
            / "weights"
            / "p3a_nmnist_h160_k8_int5_w2_int8_hash8"
            / "weights_h160_int5_w2_int8.npz"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("FC report lock has no selected candidate")
    required = (
        "trainable_mask",
        "optimizer",
        "learning_rate",
        "vote_increment",
        "vote_threshold",
        "source_trace_bits",
        "source_trace_format",
        "batch_size",
        "support_samples_per_class",
    )
    missing = [key for key in required if key not in selected]
    if missing:
        raise RuntimeError(f"FC report lock is missing fields: {missing}")
    if selected["trainable_mask"] != "w2":
        raise RuntimeError("FC report must use the selected W2-only mask")
    if selected["optimizer"] != "int8_magnitude_residual":
        raise RuntimeError("FC report must use the selected magnitude-residual rule")
    protocol = payload.get("report_protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("FC report lock has no report protocol")
    if not bool(protocol.get("all_official_test_samples")):
        raise RuntimeError("FC report protocol must use the complete official test split")
    return payload, selected


def run_episode(
    *,
    args: argparse.Namespace,
    selected: dict[str, Any],
    seed: int,
    dx: int,
    dy: int,
    output: Path,
) -> None:
    command = [
        sys.executable,
        str(EPISODE_RUNNER),
        "--traces-root",
        str(args.traces_root),
        "--weights",
        str(args.weights),
        "--data-root",
        str(args.data_root),
        "--output",
        str(output),
        "--batch-size",
        str(int(selected["batch_size"])),
        "--adapt-samples-per-class",
        str(int(selected["support_samples_per_class"])),
        "--episode-seed",
        str(seed),
        "--query-split",
        "test",
        "--all-query-samples",
        "--shift-x",
        str(dx),
        "--shift-y",
        str(dy),
        "--trainable-masks",
        str(selected["trainable_mask"]),
        "--optimizer",
        str(selected["optimizer"]),
        "--learning-rate",
        str(selected["learning_rate"]),
        "--vote-increment",
        str(int(selected["vote_increment"])),
        "--vote-threshold",
        str(int(selected["vote_threshold"])),
        "--source-trace-bits",
        str(int(selected["source_trace_bits"])),
        "--source-trace-format",
        str(selected["source_trace_format"]),
        "--device",
        args.device,
    ]
    print("FC_H160_REPORT_RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    lock, selected = load_lock(args.selection_lock)
    protocol = lock["report_protocol"]
    seeds = [int(value) for value in protocol["support_seeds"]]
    shifts = [(int(item["dx"]), int(item["dy"])) for item in protocol["shifts"]]
    output_dir = args.output_dir or args.selection_lock.parent / "official_test_report"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths: list[Path] = []

    for seed in seeds:
        for dx, dy in shifts:
            sx = f"m{abs(dx)}" if dx < 0 else f"p{dx}"
            sy = f"m{abs(dy)}" if dy < 0 else f"p{dy}"
            raw_path = raw_dir / f"seed{seed}_x{sx}_y{sy}.json"
            raw_paths.append(raw_path)
            if args.force or not raw_path.exists():
                run_episode(
                    args=args,
                    selected=selected,
                    seed=seed,
                    dx=dx,
                    dy=dy,
                    output=raw_path,
                )

    rows: list[dict[str, Any]] = []
    query_count: int | None = None
    for raw_path in raw_paths:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("query_split") != "test":
            raise RuntimeError(f"invalid FC report result: {raw_path}")
        if not bool(payload.get("all_query_samples")):
            raise RuntimeError(f"FC report did not use the full query split: {raw_path}")
        if int(payload.get("support_query_overlap", -1)) != 0:
            raise RuntimeError(f"FC report support/query overlap is nonzero: {raw_path}")
        masks = payload.get("masks", [])
        if len(masks) != 1 or masks[0]["trainable_mask"] != selected["trainable_mask"]:
            raise RuntimeError(f"FC report candidate drift: {raw_path}")
        candidate = masks[0]
        count = int(payload["query_samples"])
        if query_count is None:
            query_count = count
        elif count != query_count:
            raise RuntimeError("FC report episodes use different query counts")
        rows.append(
            {
                "seed": int(payload["episode_seed"]),
                "dx": int(payload["shift"]["dx_pixels"]),
                "dy": int(payload["shift"]["dy_pixels"]),
                "query_samples": count,
                "baseline_clean_accuracy_percent": float(candidate["baseline_clean"]["accuracy_percent"]),
                "baseline_shifted_accuracy_percent": float(candidate["baseline_shifted"]["accuracy_percent"]),
                "after_clean_accuracy_percent": float(candidate["clean_after"]["accuracy_percent"]),
                "after_shifted_accuracy_percent": float(candidate["shifted_after"]["accuracy_percent"]),
                "shifted_gain_points": float(candidate["shifted_gain_points"]),
                "clean_retention_points": float(candidate["clean_retention_points"]),
                "code_writes": int(candidate["optimizer_diagnostics"]["code_writes"]),
                "residual_writes": int(candidate["optimizer_diagnostics"]["residual_writes"]),
            }
        )

    by_shift: dict[str, Any] = {}
    for dx, dy in shifts:
        shift_rows = [row for row in rows if row["dx"] == dx and row["dy"] == dy]
        by_shift[f"{dx},{dy}"] = {
            "episodes": len(shift_rows),
            "shifted_gain_points": mean_std([row["shifted_gain_points"] for row in shift_rows]),
            "clean_retention_points": mean_std([row["clean_retention_points"] for row in shift_rows]),
            "positive_gain_episodes": sum(row["shifted_gain_points"] > 0 for row in shift_rows),
            "clean_loss_within_1pt_episodes": sum(row["clean_retention_points"] >= -1 for row in shift_rows),
        }

    summary = {
        "status": "pass",
        "claim_boundary": (
            "One pre-locked FC H160 W2-only candidate evaluated on every official "
            "N-MNIST test sample. No report-stage mask, shift, rate or numeric-format "
            "selection is performed."
        ),
        "selection_lock": str(args.selection_lock),
        "locked_candidate": selected,
        "support_seeds": seeds,
        "shifts": [{"dx": dx, "dy": dy} for dx, dy in shifts],
        "query_split": "official_test_all",
        "query_samples_per_episode": query_count,
        "episodes": len(rows),
        "by_shift": by_shift,
        "pooled_shifted_gain_points": mean_std([row["shifted_gain_points"] for row in rows]),
        "pooled_clean_retention_points": mean_std([row["clean_retention_points"] for row in rows]),
        "positive_gain_episodes": sum(row["shifted_gain_points"] > 0 for row in rows),
        "clean_loss_within_1pt_episodes": sum(row["clean_retention_points"] >= -1 for row in rows),
        "episodes_detail": rows,
        "raw_results": [str(path) for path in raw_paths],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "report_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
