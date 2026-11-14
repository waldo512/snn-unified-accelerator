#!/usr/bin/env python3
"""Aggregate the locked SCNN Q0.15 classifier across event corruptions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "scnn_q1_15_cross_corruption_report_20260927"
)
JITTER_SUMMARY = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "scnn_event_fixed_sgd_q1_15_report_subjects_20260924"
    / "report_summary.json"
)
EXPECTED_SEEDS = {1729, 38472, 65537, 99891}
EXPECTED_PAIRS = {(28, 29), (29, 28)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jitter-summary", type=Path, default=JITTER_SUMMARY)
    return parser.parse_args()


def metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def corruption_name(config: dict[str, Any]) -> str:
    signature = (
        float(config.get("event_drop_probability", 0.0)),
        float(config.get("polarity_flip_probability", 0.0)),
        int(config.get("temporal_jitter", 0)),
    )
    names = {
        (0.4, 0.0, 0): "drop40",
        (0.0, 0.4, 0): "flip40",
    }
    if signature not in names:
        raise RuntimeError(f"unexpected corruption signature: {signature}")
    return names[signature]


def validate_config(path: Path, config: dict[str, Any]) -> str:
    expected = {
        "masks": ["classifier"],
        "conv_thresholds": [1],
        "classifier_thresholds": [1],
        "windows": [4],
        "adapt_batch_size": 4,
        "eval_batch_size": 4,
        "weight_fractional_bits": 15,
        "update_rule": "fixed_sgd",
        "fixed_sgd_residual_bits": 15,
        "full_adaptation_batches_only": True,
        "force_class_mixed_batches": False,
        "shift_x": 0,
        "shift_y": 0,
    }
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise RuntimeError(f"protocol drift in {path}: {drift}")
    pairs = {tuple(pair) for pair in config["subject_pairs"]}
    if pairs != EXPECTED_PAIRS:
        raise RuntimeError(f"subject-pair drift in {path}: {sorted(pairs)}")
    return corruption_name(config)


def main() -> int:
    args = parse_args()
    raw_paths = sorted((args.input_dir / "raw").glob("*.json"))
    if len(raw_paths) != 8:
        raise RuntimeError(f"expected eight raw results, found {len(raw_paths)}")

    rows: list[dict[str, Any]] = []
    files_by_corruption: dict[str, list[str]] = defaultdict(list)
    seeds_by_corruption: dict[str, set[int]] = defaultdict(set)
    for path in raw_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise RuntimeError(f"incomplete SCNN result: {path}")
        config = payload["configuration"]
        corruption = validate_config(path, config)
        seed = int(config["corruption_seed"])
        files_by_corruption[corruption].append(str(path))
        seeds_by_corruption[corruption].add(seed)
        for episode in payload["episodes"]:
            pair = (int(episode["support_subject"]), int(episode["query_subject"]))
            if pair not in EXPECTED_PAIRS:
                raise RuntimeError(f"unexpected pair in {path}: {pair}")
            candidates = episode["candidates"]
            if len(candidates) != 1 or candidates[0]["mask"] != "classifier":
                raise RuntimeError(f"candidate drift in {path}")
            candidate = candidates[0]
            update = candidate["update"]
            rows.append(
                {
                    "corruption": corruption,
                    "corruption_seed": seed,
                    "support_subject": pair[0],
                    "query_subject": pair[1],
                    "baseline_clean_accuracy_percent": float(
                        episode["baseline_clean"]["accuracy_percent"]
                    ),
                    "baseline_corrupted_accuracy_percent": float(
                        episode["baseline_shifted"]["accuracy_percent"]
                    ),
                    "after_clean_accuracy_percent": float(
                        candidate["clean_after"]["accuracy_percent"]
                    ),
                    "after_corrupted_accuracy_percent": float(
                        candidate["shifted_after"]["accuracy_percent"]
                    ),
                    "corrupted_gain_points": float(candidate["shifted_gain_points"]),
                    "clean_change_points": float(candidate["clean_retention_points"]),
                    "code_writes": int(candidate["code_writes"]),
                    "mutable_parameters": int(candidate["mutable_parameters"]),
                    "mutable_state_bytes": int(candidate["mutable_state_bytes"]),
                    "adaptation_samples_used": int(update["adaptation_samples_used"]),
                    "adaptation_samples_dropped": int(update["adaptation_samples_dropped"]),
                    "mask_realized": bool(candidate["mask_realized"]),
                }
            )

    if set(files_by_corruption) != {"drop40", "flip40"}:
        raise RuntimeError(f"missing corruption group: {sorted(files_by_corruption)}")
    for corruption, seeds in seeds_by_corruption.items():
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"seed drift for {corruption}: {sorted(seeds)}")

    by_corruption: dict[str, Any] = {}
    for corruption in sorted(files_by_corruption):
        group = [row for row in rows if row["corruption"] == corruption]
        gains = [float(row["corrupted_gain_points"]) for row in group]
        clean = [float(row["clean_change_points"]) for row in group]
        by_corruption[corruption] = {
            "episodes": len(group),
            "baseline_clean_accuracy_percent": metric(
                [float(row["baseline_clean_accuracy_percent"]) for row in group]
            ),
            "baseline_corrupted_accuracy_percent": metric(
                [float(row["baseline_corrupted_accuracy_percent"]) for row in group]
            ),
            "after_clean_accuracy_percent": metric(
                [float(row["after_clean_accuracy_percent"]) for row in group]
            ),
            "after_corrupted_accuracy_percent": metric(
                [float(row["after_corrupted_accuracy_percent"]) for row in group]
            ),
            "corrupted_gain_points": metric(gains),
            "clean_change_points": metric(clean),
            "code_writes": metric([float(row["code_writes"]) for row in group]),
            "positive_gain_episodes": sum(value > 0.0 for value in gains),
            "nondegrading_gain_episodes": sum(value >= 0.0 for value in gains),
            "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
            "mask_realized_episodes": sum(bool(row["mask_realized"]) for row in group),
            "raw_results": sorted(files_by_corruption[corruption]),
        }

    jitter = json.loads(args.jitter_summary.read_text(encoding="utf-8"))
    if jitter.get("status") != "pass":
        raise RuntimeError("locked temporal-jitter summary is not complete")
    by_corruption["jitter1"] = {
        "episodes": int(jitter["episodes"]),
        "baseline_clean_accuracy_percent": {
            "mean": float(jitter["baseline_clean_accuracy_mean"])
        },
        "baseline_corrupted_accuracy_percent": {
            "mean": float(jitter["baseline_shifted_accuracy_mean"])
        },
        "after_clean_accuracy_percent": {
            "mean": float(jitter["after_clean_accuracy_mean"])
        },
        "after_corrupted_accuracy_percent": {
            "mean": float(jitter["after_shifted_accuracy_mean"])
        },
        "corrupted_gain_points": {
            "mean": float(jitter["report_shifted_gain_mean"]),
            "stdev": float(jitter["report_shifted_gain_std"]),
        },
        "clean_change_points": {
            "mean": float(jitter["report_clean_retention_mean"]),
            "stdev": float(jitter["report_clean_retention_std"]),
        },
        "code_writes": {
            "mean": float(jitter["code_writes_mean"]),
            "stdev": float(jitter["code_writes_std"]),
        },
        "positive_gain_episodes": sum(
            float(row["shifted_gain_points"]) > 0.0
            for row in jitter["episodes_detail"]
        ),
        "clean_loss_within_1pt_episodes": sum(
            float(row["clean_retention_points"]) >= -1.0
            for row in jitter["episodes_detail"]
        ),
        "source_summary": str(args.jitter_summary),
    }

    mutable_parameters = {int(row["mutable_parameters"]) for row in rows}
    mutable_state_bytes = {int(row["mutable_state_bytes"]) for row in rows}
    if len(mutable_parameters) != 1 or len(mutable_state_bytes) != 1:
        raise RuntimeError("mutable-state accounting changed across episodes")

    all_gains = [float(row["corrupted_gain_points"]) for row in rows]
    all_clean = [float(row["clean_change_points"]) for row in rows]
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Post-lock report-subject robustness extension for the single predeclared "
            "SCNN classifier-only Q0.15 candidate. Drop40 and flip40 change only the "
            "bounded event corruption. Jitter1 is imported from the earlier locked "
            "report. This is not candidate reselection, natural domain shift, RTL "
            "cycle evidence, power measurement, or a cross-dataset claim."
        ),
        "protocol": {
            "dataset": "DVS-Gesture",
            "partition": "report subjects 28 and 29",
            "subject_pairs": [list(pair) for pair in sorted(EXPECTED_PAIRS)],
            "corruption_seeds": sorted(EXPECTED_SEEDS),
            "mask": "classifier",
            "adapt_batch_size": 4,
            "window_timesteps": 4,
            "weight_format": "signed Q0.15",
            "update_rule": "magnitude-residual fixed SGD",
            "optimizer_lr": 1.1920928955078125e-6,
            "mutable_parameters": mutable_parameters.pop(),
            "mutable_state_bytes": mutable_state_bytes.pop(),
        },
        "by_corruption": by_corruption,
        "new_cross_corruption_episodes": len(rows),
        "new_cross_corruption_gain_points": metric(all_gains),
        "new_cross_corruption_clean_change_points": metric(all_clean),
        "new_positive_gain_episodes": sum(value > 0.0 for value in all_gains),
        "new_clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in all_clean),
        "episodes": rows,
    }

    args.input_dir.mkdir(parents=True, exist_ok=True)
    (args.input_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.input_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
