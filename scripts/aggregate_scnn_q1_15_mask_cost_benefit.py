#!/usr/bin/env python3
"""Aggregate the matched Q0.15 classifier and Conv8+classifier comparison."""

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
    / "scnn_q1_15_mask_cost_benefit_20260927"
)
EXPECTED_MASKS = {"classifier", "conv8_classifier"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    args = parse_args()
    raw_paths = sorted((args.input_dir / "raw").glob("*.json"))
    if len(raw_paths) != 4:
        raise RuntimeError(f"expected four corruption-seed results, found {len(raw_paths)}")

    rows: list[dict[str, Any]] = []
    for path in raw_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise RuntimeError(f"incomplete SCNN result: {path}")
        config = payload["configuration"]
        if set(config["masks"]) != EXPECTED_MASKS:
            raise RuntimeError(f"mask drift in {path}: {config['masks']}")
        expected = {
            "adapt_batch_size": 4,
            "weight_fractional_bits": 15,
            "update_rule": "fixed_sgd",
            "fixed_sgd_residual_bits": 15,
            "full_adaptation_batches_only": True,
            "force_class_mixed_batches": True,
            "temporal_jitter": 1,
        }
        drift = {
            key: {"expected": value, "actual": config.get(key)}
            for key, value in expected.items()
            if config.get(key) != value
        }
        if drift:
            raise RuntimeError(f"protocol drift in {path}: {drift}")

        seed = int(config["corruption_seed"])
        for episode in payload["episodes"]:
            support_subject = int(episode["support_subject"])
            query_subject = int(episode["query_subject"])
            candidates = {candidate["mask"]: candidate for candidate in episode["candidates"]}
            if set(candidates) != EXPECTED_MASKS:
                raise RuntimeError(f"candidate drift in {path}: {sorted(candidates)}")
            for mask, candidate in candidates.items():
                update = candidate["update"]
                rows.append(
                    {
                        "corruption_seed": seed,
                        "support_subject": support_subject,
                        "query_subject": query_subject,
                        "mask": mask,
                        "shifted_gain_points": float(candidate["shifted_gain_points"]),
                        "clean_retention_points": float(candidate["clean_retention_points"]),
                        "mutable_parameters": int(candidate["mutable_parameters"]),
                        "mutable_state_bytes": int(candidate["mutable_state_bytes"]),
                        "code_writes": int(candidate["code_writes"]),
                        "mask_realized": bool(candidate["mask_realized"]),
                        "adaptation_samples_used": int(update["adaptation_samples_used"]),
                        "adaptation_samples_dropped": int(update["adaptation_samples_dropped"]),
                        "conv8_code_writes": int(update["code_writes"].get("conv8", 0)),
                        "conv8_nonzero_residuals": int(
                            update["nonzero_residuals"].get("conv8", 0)
                        ),
                        "conv8_gradient_active_windows": int(
                            update["gradient_active_windows"].get("conv8", 0)
                        ),
                    }
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    indexed: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["mask"])].append(row)
        key = (
            int(row["corruption_seed"]),
            int(row["support_subject"]),
            int(row["query_subject"]),
        )
        indexed[key][str(row["mask"])] = row

    by_mask: dict[str, Any] = {}
    for mask in sorted(EXPECTED_MASKS):
        mask_rows = grouped[mask]
        mutable_parameters = {int(row["mutable_parameters"]) for row in mask_rows}
        mutable_state_bytes = {int(row["mutable_state_bytes"]) for row in mask_rows}
        if len(mutable_parameters) != 1 or len(mutable_state_bytes) != 1:
            raise RuntimeError(f"inconsistent mutable-state accounting for {mask}")
        gains = [float(row["shifted_gain_points"]) for row in mask_rows]
        clean = [float(row["clean_retention_points"]) for row in mask_rows]
        writes = [float(row["code_writes"]) for row in mask_rows]
        by_mask[mask] = {
            "episodes": len(mask_rows),
            "shifted_gain_points": stats(gains),
            "clean_retention_points": stats(clean),
            "code_writes": stats(writes),
            "positive_gain_episodes": sum(value > 0.0 for value in gains),
            "nondegrading_gain_episodes": sum(value >= 0.0 for value in gains),
            "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
            "mask_realized_episodes": sum(bool(row["mask_realized"]) for row in mask_rows),
            "conv8_code_writes": stats(
                [float(row["conv8_code_writes"]) for row in mask_rows]
            ),
            "conv8_nonzero_residuals": stats(
                [float(row["conv8_nonzero_residuals"]) for row in mask_rows]
            ),
            "mutable_parameters": mutable_parameters.pop(),
            "mutable_state_bytes": mutable_state_bytes.pop(),
        }

    paired_differences: list[float] = []
    paired_rows: list[dict[str, Any]] = []
    for key, candidates in sorted(indexed.items()):
        if set(candidates) != EXPECTED_MASKS:
            raise RuntimeError(f"unpaired episode: {key}")
        difference = (
            float(candidates["conv8_classifier"]["shifted_gain_points"])
            - float(candidates["classifier"]["shifted_gain_points"])
        )
        paired_differences.append(difference)
        paired_rows.append(
            {
                "corruption_seed": key[0],
                "support_subject": key[1],
                "query_subject": key[2],
                "conv8_plus_classifier_minus_classifier_gain_points": difference,
            }
        )

    classifier = by_mask["classifier"]
    joint = by_mask["conv8_classifier"]
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Matched development-partition comparison under one Q0.15 fixed-SGD, "
            "class-mixed B4, temporal-jitter protocol. It measures the marginal "
            "algorithm benefit and mutable-state cost of adding Conv8 to the "
            "classifier mask; it is not a report-set result or RTL measurement."
        ),
        "protocol": {
            "dataset": "DVS-Gesture",
            "partition": "development subjects 24-27",
            "corruption": "one-bin temporal jitter",
            "corruption_seeds": sorted({int(row["corruption_seed"]) for row in rows}),
            "adapt_batch_size": 4,
            "weight_format": "signed Q0.15",
            "update_rule": "magnitude-residual fixed SGD",
            "optimizer_lr": 1.1920928955078125e-6,
            "masks": sorted(EXPECTED_MASKS),
        },
        "by_mask": by_mask,
        "paired_conv8_incremental_gain_points": stats(paired_differences),
        "paired_positive_incremental_episodes": sum(value > 0.0 for value in paired_differences),
        "mutable_parameter_ratio_joint_over_classifier": (
            float(joint["mutable_parameters"]) / float(classifier["mutable_parameters"])
        ),
        "mutable_state_ratio_joint_over_classifier": (
            float(joint["mutable_state_bytes"]) / float(classifier["mutable_state_bytes"])
        ),
        "episodes": rows,
        "paired_episodes": paired_rows,
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
