#!/usr/bin/env python3
"""Aggregate fixed-profile SRNN Wout support-budget sensitivity evidence."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "srnn_wout_support_sensitivity_20260927"
)
RAW = OUTPUT / "raw"
SEEDS = (228474, 228475, 228476, 228477, 228478)
SHIFTS = (16, -16)
SUPPORT_SIZES = (64, 128, 256)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def parse(path: Path, support_size: int, seed: int, shift: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    candidates = payload.get("optimizers", [])
    if payload.get("status") != "pass" or len(candidates) != 1:
        raise RuntimeError(f"invalid SRNN sensitivity result: {path}")
    if protocol.get("dataset_partition") != "train_support_official_test":
        raise RuntimeError(f"partition drift: {path}")
    if protocol.get("support_query_index_spaces") != "training_validation_vs_official_test":
        raise RuntimeError(f"index-space drift: {path}")
    if int(protocol.get("support_query_overlap", -1)) != 0:
        raise RuntimeError(f"support/query overlap: {path}")
    if (
        int(protocol.get("support_samples", -1)) != support_size
        or int(protocol.get("episode_seed", -1)) != seed
        or int(protocol.get("shift_channels", 0)) != shift
    ):
        raise RuntimeError(f"protocol drift: {path}")
    candidate = candidates[0]
    config = candidate.get("configuration", {})
    if (
        candidate.get("trainable_mask") != "wout"
        or candidate.get("optimizer") != "direct_response"
        or int(config.get("wout_increment", -1)) != 31
        or int(config.get("wout_threshold", -1)) != 65
    ):
        raise RuntimeError(f"candidate drift: {path}")
    before = candidate["evaluations"][0]
    after = candidate["evaluations"][-1]
    traffic = candidate["traffic"]
    return {
        "support_samples": support_size,
        "support_batches": int(protocol["support_batches"]),
        "seed": seed,
        "shift_channels": shift,
        "query_samples": int(protocol["query_samples"]),
        "baseline_clean_accuracy_percent": 100.0 * float(before["clean"]["accuracy"]),
        "baseline_shifted_accuracy_percent": 100.0 * float(before["shifted"]["accuracy"]),
        "after_clean_accuracy_percent": 100.0 * float(after["clean"]["accuracy"]),
        "after_shifted_accuracy_percent": 100.0 * float(after["shifted"]["accuracy"]),
        "shifted_gain_points": 100.0 * float(candidate["shifted_accuracy_improvement"]),
        "clean_change_points": 100.0 * float(candidate["clean_accuracy_change"]),
        "rmw": int(traffic["rmw"]),
        "code_writes": int(traffic["code_writes"]),
        "residual_writes": int(traffic["residual_writes"]),
        "raw_result": str(path.relative_to(ROOT)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gain = [float(row["shifted_gain_points"]) for row in rows]
    clean = [float(row["clean_change_points"]) for row in rows]
    return {
        "episodes": len(rows),
        "shifted_gain_points": stats(gain),
        "clean_change_points": stats(clean),
        "positive_gain_episodes": sum(value > 0.0 for value in gain),
        "nondegrading_gain_episodes": sum(value >= 0.0 for value in gain),
        "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
        "mean_rmw": statistics.fmean(int(row["rmw"]) for row in rows),
        "mean_code_writes": statistics.fmean(int(row["code_writes"]) for row in rows),
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    for support_size in SUPPORT_SIZES:
        for seed in SEEDS:
            for shift in SHIFTS:
                suffix = f"m{abs(shift)}" if shift < 0 else f"p{shift}"
                path = RAW / f"support{support_size}_seed{seed}_shift{suffix}.json"
                if not path.exists():
                    raise RuntimeError(f"missing sensitivity result: {path}")
                rows.append(parse(path, support_size, seed, shift))

    by_support = {
        str(size): summarize([row for row in rows if row["support_samples"] == size])
        for size in SUPPORT_SIZES
    }
    paired: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["seed"]), int(row["shift_channels"]))][
            int(row["support_samples"])
        ] = row
    if any(set(values) != set(SUPPORT_SIZES) for values in paired.values()):
        raise RuntimeError("support-size pairing is incomplete")

    deltas_vs_128: dict[str, Any] = {}
    for support_size in SUPPORT_SIZES:
        if support_size == 128:
            continue
        gain_delta = [
            values[support_size]["shifted_gain_points"]
            - values[128]["shifted_gain_points"]
            for values in paired.values()
        ]
        clean_delta = [
            values[support_size]["clean_change_points"]
            - values[128]["clean_change_points"]
            for values in paired.values()
        ]
        rmw_ratio = [
            values[support_size]["rmw"] / values[128]["rmw"]
            for values in paired.values()
        ]
        deltas_vs_128[str(support_size)] = {
            "paired_episodes": len(gain_delta),
            "gain_delta_points": stats(gain_delta),
            "clean_delta_points": stats(clean_delta),
            "rmw_ratio": stats(rmw_ratio),
        }

    payload = {
        "status": "pass",
        "claim_boundary": (
            "Post-lock support-budget sensitivity for the fixed SRNN Wout-only "
            "Q2.14 direct-response profile. Only the number of labelled training-"
            "validation support samples changes; all queries use the complete official "
            "test partition. This does not reselect the mask or update rule."
        ),
        "fixed_profile": {
            "mask": "wout",
            "optimizer": "direct_response",
            "wout_increment": 31,
            "wout_threshold": 65,
            "support_sizes": list(SUPPORT_SIZES),
            "shift_channels": list(SHIFTS),
        },
        "episodes": len(rows),
        "paired_episode_keys": len(paired),
        "by_support_size": by_support,
        "paired_deltas_vs_128": deltas_vs_128,
        "episode_rows": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SRNN Wout Support-budget Sensitivity",
        "",
        "Status: PASS. This is a post-lock sensitivity, not a new profile-selection run.",
        "",
        "| Support samples | Episodes | Gain (points) | Positive / nondegrading | Clean budget | Mean RMW | Mean code writes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SUPPORT_SIZES:
        item = by_support[str(size)]
        gain = item["shifted_gain_points"]
        lines.append(
            f"| {size} | {item['episodes']} | {gain['mean']:+.3f} +/- {gain['stdev']:.3f} | "
            f"{item['positive_gain_episodes']}/{item['episodes']} / "
            f"{item['nondegrading_gain_episodes']}/{item['episodes']} | "
            f"{item['clean_loss_within_1pt_episodes']}/{item['episodes']} | "
            f"{item['mean_rmw']:.1f} | {item['mean_code_writes']:.1f} |"
        )
    lines.extend(
        [
            "",
            "All rows use the same five support seeds and two frequency shifts. Paired",
            "deltas and RMW ratios against the locked 128-sample point are in",
            "`summary.json`. The experiment changes runtime support work, not hardware",
            "mask capacity, and remains a synthetic-shift software reference.",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
