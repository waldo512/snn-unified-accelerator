#!/usr/bin/env python3
"""Run the locked SRNN Wout-only candidate on fixed SHD report episodes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EPISODE_RUNNER = ROOT / "scripts" / "compare_srnn_shd_shifted_optimizers.py"
DEFAULT_LOCK = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "srnn_wout_q2_14_direct_selection"
    / "locked_report_protocol.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--handoff",
        type=Path,
        default=Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\datasets"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument(
        "--dataset-partition",
        choices=("official_test", "train_support_official_test"),
        help=(
            "evaluation boundary; the default preserves the locked official-test "
            "support/query protocol, while train_support_official_test uses labelled "
            "support from the training-validation split and queries the full test set"
        ),
    )
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
        raise RuntimeError("SRNN report lock has no selected candidate")
    expected = {
        "trainable_mask": "wout",
        "optimizer": "direct_response",
        "weight_format": "signed_q2_14",
        "gradient_format": "signed_q40",
        "wout_increment": 31,
        "wout_threshold": 65,
        "mutable_parameters": 5120,
    }
    drift = {
        key: {"expected": value, "actual": selected.get(key)}
        for key, value in expected.items()
        if selected.get(key) != value
    }
    if drift:
        raise RuntimeError(f"SRNN report lock candidate drift: {drift}")
    protocol = payload.get("report_protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("SRNN report lock has no report protocol")
    if protocol.get("dataset_partition") != "official_test":
        raise RuntimeError("SRNN report must use the official SHD test partition")
    return payload, selected


def run_episode(
    *,
    args: argparse.Namespace,
    selected: dict[str, Any],
    support_size: int,
    seed: int,
    shift: int,
    dataset_partition: str,
    output: Path,
) -> None:
    command = [
        sys.executable,
        str(EPISODE_RUNNER),
        "--handoff",
        str(args.handoff),
        "--dataset-root",
        str(args.dataset_root),
        "--dataset-partition",
        dataset_partition,
        "--episode-seed",
        str(seed),
        "--support-size",
        str(support_size),
        "--shift-channels",
        str(shift),
        "--trainable-masks",
        str(selected["trainable_mask"]),
        "--optimizers",
        str(selected["optimizer"]),
        "--output",
        str(output),
        "--device",
        args.device,
    ]
    print("SRNN_SHD_REPORT_RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    lock, selected = load_lock(args.selection_lock)
    protocol = lock["report_protocol"]
    seeds = [int(value) for value in protocol["support_seeds"]]
    shifts = [int(value) for value in protocol["shift_channels"]]
    support_size = int(protocol["support_samples"])
    dataset_partition = args.dataset_partition or str(protocol["dataset_partition"])
    default_report_dir = (
        "official_test_report"
        if dataset_partition == "official_test"
        else "train_support_official_test_report"
    )
    output_dir = args.output_dir or args.selection_lock.parent / default_report_dir
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    for seed in seeds:
        for shift in shifts:
            shift_name = f"m{abs(shift)}" if shift < 0 else f"p{shift}"
            raw_path = raw_dir / f"seed{seed}_shift{shift_name}.json"
            raw_paths.append(raw_path)
            if args.force or not raw_path.exists():
                run_episode(
                    args=args,
                    selected=selected,
                    support_size=support_size,
                    seed=seed,
                    shift=shift,
                    dataset_partition=dataset_partition,
                    output=raw_path,
                )

    rows: list[dict[str, Any]] = []
    query_count: int | None = None
    for raw_path in raw_paths:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        protocol_row = payload.get("protocol", {})
        if payload.get("status") != "pass":
            raise RuntimeError(f"invalid SRNN report result: {raw_path}")
        if protocol_row.get("dataset_partition") != dataset_partition:
            raise RuntimeError(f"SRNN report partition drift: {raw_path}")
        if int(protocol_row.get("support_query_overlap", -1)) != 0:
            raise RuntimeError(f"SRNN report support/query overlap: {raw_path}")
        expected_spaces = "training_validation_vs_official_test"
        if (
            dataset_partition == "train_support_official_test"
            and protocol_row.get("support_query_index_spaces") != expected_spaces
        ):
            raise RuntimeError(f"SRNN report index-space drift: {raw_path}")
        optimizers = payload.get("optimizers", [])
        if len(optimizers) != 1:
            raise RuntimeError(f"SRNN report evaluated more than one profile: {raw_path}")
        candidate = optimizers[0]
        configuration = candidate.get("configuration", {})
        if (
            candidate.get("optimizer") != selected["optimizer"]
            or candidate.get("trainable_mask") != selected["trainable_mask"]
            or int(configuration.get("wout_increment", -1))
            != int(selected["wout_increment"])
            or int(configuration.get("wout_threshold", -1))
            != int(selected["wout_threshold"])
        ):
            raise RuntimeError(f"SRNN report candidate drift: {raw_path}")
        evaluations = candidate.get("evaluations", [])
        if len(evaluations) != int(protocol_row["support_batches"]) + 1:
            raise RuntimeError(f"SRNN report recovery-curve length mismatch: {raw_path}")
        before = evaluations[0]
        after = evaluations[-1]
        count = int(protocol_row["query_samples"])
        if query_count is None:
            query_count = count
        elif count != query_count:
            raise RuntimeError("SRNN report episodes use different query counts")
        shift = int(protocol_row["shift_channels"])
        if shift not in shifts:
            raise RuntimeError(f"SRNN report shift drift: {raw_path}")
        traffic = candidate.get("traffic") or {}
        rows.append(
            {
                "seed": int(protocol_row["episode_seed"]),
                "shift_channels": shift,
                "support_samples": int(protocol_row["support_samples"]),
                "query_samples": count,
                "baseline_clean_accuracy_percent": 100.0 * float(before["clean"]["accuracy"]),
                "baseline_shifted_accuracy_percent": 100.0 * float(before["shifted"]["accuracy"]),
                "after_clean_accuracy_percent": 100.0 * float(after["clean"]["accuracy"]),
                "after_shifted_accuracy_percent": 100.0 * float(after["shifted"]["accuracy"]),
                "shifted_gain_points": 100.0 * float(candidate["shifted_accuracy_improvement"]),
                "clean_retention_points": 100.0 * float(candidate["clean_accuracy_change"]),
                "rmw": int(traffic.get("rmw", 0)),
                "code_writes": int(traffic.get("code_writes", 0)),
                "residual_writes": int(traffic.get("residual_writes", 0)),
            }
        )

    by_shift: dict[str, Any] = {}
    for shift in shifts:
        shift_rows = [row for row in rows if row["shift_channels"] == shift]
        by_shift[str(shift)] = {
            "episodes": len(shift_rows),
            "shifted_gain_points": mean_std(
                [float(row["shifted_gain_points"]) for row in shift_rows]
            ),
            "clean_retention_points": mean_std(
                [float(row["clean_retention_points"]) for row in shift_rows]
            ),
            "positive_gain_episodes": sum(
                float(row["shifted_gain_points"]) > 0.0 for row in shift_rows
            ),
            "clean_loss_within_1pt_episodes": sum(
                float(row["clean_retention_points"]) >= -1.0 for row in shift_rows
            ),
        }

    if dataset_partition == "official_test":
        claim_boundary = (
            "One validation-selected SRNN Wout-only Q2.14 direct-residual profile "
            "evaluated on predeclared, disjoint support/query partitions of the "
            "official SHD test set. No report-stage mask or update-rule selection "
            "is performed. The synthetic frequency shifts are robustness tests, "
            "not natural domain shifts or RTL evidence."
        )
        support_query_index_spaces = "official_test"
    else:
        claim_boundary = (
            "One validation-selected SRNN Wout-only Q2.14 direct-residual profile "
            "adapted only with labelled support from the SHD training-validation "
            "index space and evaluated on the complete official SHD test partition. "
            "No test sample participates in adaptation and no report-stage mask or "
            "update-rule selection is performed. The synthetic frequency shifts are "
            "robustness tests, not natural domain shifts or RTL evidence."
        )
        support_query_index_spaces = "training_validation_vs_official_test"

    summary = {
        "status": "pass",
        "claim_boundary": claim_boundary,
        "selection_lock": str(args.selection_lock),
        "locked_candidate": selected,
        "dataset_partition": dataset_partition,
        "support_query_index_spaces": support_query_index_spaces,
        "support_seeds": seeds,
        "shift_channels": shifts,
        "support_samples_per_episode": support_size,
        "query_samples_per_episode": query_count,
        "episodes": len(rows),
        "by_shift": by_shift,
        "pooled_shifted_gain_points": mean_std(
            [float(row["shifted_gain_points"]) for row in rows]
        ),
        "pooled_clean_retention_points": mean_std(
            [float(row["clean_retention_points"]) for row in rows]
        ),
        "positive_gain_episodes": sum(
            float(row["shifted_gain_points"]) > 0.0 for row in rows
        ),
        "clean_loss_within_1pt_episodes": sum(
            float(row["clean_retention_points"]) >= -1.0 for row in rows
        ),
        "episodes_detail": rows,
        "raw_results": [str(path) for path in raw_paths],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
