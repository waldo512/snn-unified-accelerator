#!/usr/bin/env python3
"""Aggregate matched SCNN classifier and Conv8 mask costs by corruption."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "paper_evidence" / "algorithm" / "mask_ablation_20260919"
DEFAULT_INPUT = BASE / "scnn_q1_15_cross_corruption_mask_cost_20260927"
JITTER_SUMMARY = BASE / "scnn_q1_15_mask_cost_benefit_20260927" / "summary.json"
MASKS = {"classifier", "conv8_classifier"}
SEEDS = {1729, 38472, 65537, 99891}
PAIRS = {(24, 25), (25, 24), (26, 27), (27, 26)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jitter-summary", type=Path, default=JITTER_SUMMARY)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def corruption(config: dict[str, Any]) -> str:
    signature = (
        float(config.get("event_drop_probability", 0.0)),
        float(config.get("polarity_flip_probability", 0.0)),
        int(config.get("temporal_jitter", 0)),
    )
    mapping = {(0.4, 0.0, 0): "drop40", (0.0, 0.4, 0): "flip40"}
    if signature not in mapping:
        raise RuntimeError(f"unexpected corruption: {signature}")
    return mapping[signature]


def validate(path: Path, config: dict[str, Any]) -> str:
    expected = {
        "masks": ["classifier", "conv8_classifier"],
        "conv_thresholds": [1],
        "classifier_thresholds": [1],
        "windows": [4],
        "adapt_batch_size": 4,
        "eval_batch_size": 4,
        "weight_fractional_bits": 15,
        "update_rule": "fixed_sgd",
        "fixed_sgd_residual_bits": 15,
        "full_adaptation_batches_only": True,
        "force_class_mixed_batches": True,
    }
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise RuntimeError(f"protocol drift in {path}: {drift}")
    if {tuple(pair) for pair in config["subject_pairs"]} != PAIRS:
        raise RuntimeError(f"subject-pair drift in {path}")
    return corruption(config)


def main() -> int:
    args = parse_args()
    raw_paths = sorted((args.input_dir / "raw").glob("*.json"))
    if len(raw_paths) != 8:
        raise RuntimeError(f"expected eight raw results, found {len(raw_paths)}")

    rows: list[dict[str, Any]] = []
    seen_seeds: dict[str, set[int]] = defaultdict(set)
    for path in raw_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise RuntimeError(f"incomplete result: {path}")
        config = payload["configuration"]
        name = validate(path, config)
        seed = int(config["corruption_seed"])
        seen_seeds[name].add(seed)
        for episode in payload["episodes"]:
            pair = (int(episode["support_subject"]), int(episode["query_subject"]))
            candidates = {candidate["mask"]: candidate for candidate in episode["candidates"]}
            if pair not in PAIRS or set(candidates) != MASKS:
                raise RuntimeError(f"episode drift in {path}: {pair}, {sorted(candidates)}")
            for mask, candidate in candidates.items():
                update = candidate["update"]
                rows.append(
                    {
                        "corruption": name,
                        "corruption_seed": seed,
                        "support_subject": pair[0],
                        "query_subject": pair[1],
                        "mask": mask,
                        "shifted_gain_points": float(candidate["shifted_gain_points"]),
                        "clean_change_points": float(candidate["clean_retention_points"]),
                        "mutable_parameters": int(candidate["mutable_parameters"]),
                        "mutable_state_bytes": int(candidate["mutable_state_bytes"]),
                        "code_writes": int(candidate["code_writes"]),
                        "conv8_code_writes": int(update["code_writes"].get("conv8", 0)),
                        "conv8_nonzero_residuals": int(
                            update["nonzero_residuals"].get("conv8", 0)
                        ),
                        "conv8_gradient_active_windows": int(
                            update["gradient_active_windows"].get("conv8", 0)
                        ),
                        "mask_realized": bool(candidate["mask_realized"]),
                    }
                )

    if set(seen_seeds) != {"drop40", "flip40"}:
        raise RuntimeError(f"corruption groups missing: {sorted(seen_seeds)}")
    for name, seeds in seen_seeds.items():
        if seeds != SEEDS:
            raise RuntimeError(f"seed drift for {name}: {sorted(seeds)}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["corruption"]), str(row["mask"]))].append(row)
        key = (
            str(row["corruption"]),
            int(row["corruption_seed"]),
            int(row["support_subject"]),
            int(row["query_subject"]),
        )
        paired[key][str(row["mask"])] = row

    by_corruption: dict[str, Any] = {}
    for name in ("drop40", "flip40"):
        by_mask: dict[str, Any] = {}
        for mask in sorted(MASKS):
            group = grouped[(name, mask)]
            gains = [float(row["shifted_gain_points"]) for row in group]
            clean = [float(row["clean_change_points"]) for row in group]
            by_mask[mask] = {
                "episodes": len(group),
                "shifted_gain_points": stats(gains),
                "clean_change_points": stats(clean),
                "positive_gain_episodes": sum(value > 0.0 for value in gains),
                "nondegrading_gain_episodes": sum(value >= 0.0 for value in gains),
                "mutable_parameters": int(group[0]["mutable_parameters"]),
                "mutable_state_bytes": int(group[0]["mutable_state_bytes"]),
                "code_writes": stats([float(row["code_writes"]) for row in group]),
                "conv8_code_writes": stats(
                    [float(row["conv8_code_writes"]) for row in group]
                ),
                "conv8_nonzero_residuals": stats(
                    [float(row["conv8_nonzero_residuals"]) for row in group]
                ),
            }
        differences = []
        for key, candidates in paired.items():
            if key[0] != name:
                continue
            if set(candidates) != MASKS:
                raise RuntimeError(f"unpaired episode: {key}")
            differences.append(
                float(candidates["conv8_classifier"]["shifted_gain_points"])
                - float(candidates["classifier"]["shifted_gain_points"])
            )
        by_corruption[name] = {
            "by_mask": by_mask,
            "conv8_incremental_gain_points": stats(differences),
            "positive_incremental_episodes": sum(value > 0.0 for value in differences),
            "nonzero_incremental_episodes": sum(value != 0.0 for value in differences),
        }

    jitter = json.loads(args.jitter_summary.read_text(encoding="utf-8"))
    if jitter.get("status") != "pass":
        raise RuntimeError("jitter mask-cost source is not complete")
    by_corruption["jitter1"] = {
        "by_mask": jitter["by_mask"],
        "conv8_incremental_gain_points": jitter["paired_conv8_incremental_gain_points"],
        "positive_incremental_episodes": int(jitter["paired_positive_incremental_episodes"]),
        "nonzero_incremental_episodes": sum(
            float(row["conv8_plus_classifier_minus_classifier_gain_points"]) != 0.0
            for row in jitter["paired_episodes"]
        ),
        "source_summary": str(args.jitter_summary),
    }

    all_differences = []
    for name in ("drop40", "flip40"):
        for key, candidates in paired.items():
            if key[0] == name:
                all_differences.append(
                    float(candidates["conv8_classifier"]["shifted_gain_points"])
                    - float(candidates["classifier"]["shifted_gain_points"])
                )
    all_differences.extend(
        float(row["conv8_plus_classifier_minus_classifier_gain_points"])
        for row in jitter["paired_episodes"]
    )

    classifier_state = int(
        next(row["mutable_state_bytes"] for row in rows if row["mask"] == "classifier")
    )
    joint_state = int(
        next(row["mutable_state_bytes"] for row in rows if row["mask"] == "conv8_classifier")
    )
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Development-only matched-mask audit after locking the main classifier "
            "candidate. It tests whether Conv8 becomes useful under two additional "
            "event corruptions at the same shared Q0.15 scale. It does not reopen "
            "report-subject selection or exclude separately scaled/longer Conv8 learning."
        ),
        "protocol": {
            "dataset": "DVS-Gesture development subjects 24-27",
            "masks": sorted(MASKS),
            "corruptions": ["jitter1", "drop40", "flip40"],
            "weight_format": "signed Q0.15",
            "update_rule": "magnitude-residual fixed SGD",
            "optimizer_lr": 1.1920928955078125e-6,
            "adapt_batch_size": 4,
            "window_timesteps": 4,
            "force_class_mixed_batches": True,
        },
        "by_corruption": by_corruption,
        "all_48_paired_episodes_conv8_incremental_gain_points": stats(all_differences),
        "all_positive_incremental_episodes": sum(value > 0.0 for value in all_differences),
        "all_nonzero_incremental_episodes": sum(value != 0.0 for value in all_differences),
        "classifier_mutable_state_bytes": classifier_state,
        "joint_mutable_state_bytes": joint_state,
        "joint_over_classifier_state_ratio": joint_state / classifier_state,
        "episodes": rows,
        "raw_results": [str(path) for path in raw_paths],
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
