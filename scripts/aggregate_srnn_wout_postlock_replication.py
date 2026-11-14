#!/usr/bin/env python3
"""Aggregate a no-retuning SRNN Wout post-lock replication."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "paper_evidence" / "algorithm" / "mask_ablation_20260919"
ORIGINAL = BASE / "srnn_wout_train_support_official_test_20260927" / "report_summary.json"
OUTPUT = BASE / "srnn_wout_postlock_replication_20260927"
RAW = OUTPUT / "raw"
EXPECTED_SEEDS = (228479, 228480, 228481, 228482, 228483)
EXPECTED_SHIFTS = (16, -16)


def mean_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def one_sided_sign_test(values: list[float]) -> dict[str, float | int]:
    positives = sum(value > 0.0 for value in values)
    negatives = sum(value < 0.0 for value in values)
    nonzero = positives + negatives
    p_value = (
        sum(math.comb(nonzero, k) for k in range(positives, nonzero + 1))
        / (2**nonzero)
        if nonzero
        else 1.0
    )
    return {
        "positive_clusters": positives,
        "negative_clusters": negatives,
        "nonzero_clusters": nonzero,
        "one_sided_exact_p": p_value,
    }


def cluster_bootstrap(values: list[float], *, seed: int = 20260927) -> dict[str, float]:
    rng = random.Random(seed)
    samples = [
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(20000)
    ]
    samples.sort()
    return {
        "mean": statistics.fmean(values),
        "ci95_low": samples[int(0.025 * len(samples))],
        "ci95_high": samples[int(0.975 * len(samples)) - 1],
        "bootstrap_unit": "support_seed",
        "bootstrap_replicates": len(samples),
    }


def parse_raw(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    candidates = payload.get("optimizers", [])
    if payload.get("status") != "pass" or len(candidates) != 1:
        raise RuntimeError(f"invalid SRNN replication result: {path}")
    if protocol.get("dataset_partition") != "train_support_official_test":
        raise RuntimeError(f"partition drift: {path}")
    if protocol.get("support_query_index_spaces") != "training_validation_vs_official_test":
        raise RuntimeError(f"index-space drift: {path}")
    if int(protocol.get("support_query_overlap", -1)) != 0:
        raise RuntimeError(f"support/query overlap: {path}")
    if int(protocol.get("support_samples", -1)) != 128:
        raise RuntimeError(f"support-size drift: {path}")
    candidate = candidates[0]
    config = candidate.get("configuration", {})
    if (
        candidate.get("trainable_mask") != "wout"
        or candidate.get("optimizer") != "direct_response"
        or int(config.get("wout_increment", -1)) != 31
        or int(config.get("wout_threshold", -1)) != 65
    ):
        raise RuntimeError(f"candidate drift: {path}")
    evaluations = candidate["evaluations"]
    before, after = evaluations[0], evaluations[-1]
    traffic = candidate["traffic"]
    return {
        "source": "postlock_replication",
        "seed": int(protocol["episode_seed"]),
        "shift_channels": int(protocol["shift_channels"]),
        "support_samples": int(protocol["support_samples"]),
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
    gains = [float(row["shifted_gain_points"]) for row in rows]
    clean = [float(row["clean_change_points"]) for row in rows]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(float(row["shifted_gain_points"]))
    seed_means = [statistics.fmean(by_seed[seed]) for seed in sorted(by_seed)]
    return {
        "episodes": len(rows),
        "support_seed_clusters": len(seed_means),
        "shifted_gain_points": mean_stats(gains),
        "clean_change_points": mean_stats(clean),
        "positive_gain_episodes": sum(value > 0.0 for value in gains),
        "nondegrading_gain_episodes": sum(value >= 0.0 for value in gains),
        "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
        "mean_rmw": statistics.fmean(int(row["rmw"]) for row in rows),
        "mean_code_writes": statistics.fmean(int(row["code_writes"]) for row in rows),
        "seed_mean_shifted_gain_points": seed_means,
        "seed_cluster_bootstrap": cluster_bootstrap(seed_means),
        "seed_cluster_sign_test": one_sided_sign_test(seed_means),
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for shift in EXPECTED_SHIFTS:
            suffix = f"m{abs(shift)}" if shift < 0 else f"p{shift}"
            path = RAW / f"seed{seed}_shift{suffix}.json"
            if not path.exists():
                raise RuntimeError(f"missing replication result: {path}")
            row = parse_raw(path)
            if row["seed"] != seed or row["shift_channels"] != shift:
                raise RuntimeError(f"seed/shift drift: {path}")
            rows.append(row)

    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    original_rows = []
    for row in original["episodes_detail"]:
        original_rows.append(
            {
                "source": "prelocked_report",
                "seed": int(row["seed"]),
                "shift_channels": int(row["shift_channels"]),
                "support_samples": int(row["support_samples"]),
                "query_samples": int(row["query_samples"]),
                "baseline_clean_accuracy_percent": float(row["baseline_clean_accuracy_percent"]),
                "baseline_shifted_accuracy_percent": float(row["baseline_shifted_accuracy_percent"]),
                "after_clean_accuracy_percent": float(row["after_clean_accuracy_percent"]),
                "after_shifted_accuracy_percent": float(row["after_shifted_accuracy_percent"]),
                "shifted_gain_points": float(row["shifted_gain_points"]),
                "clean_change_points": float(row["clean_retention_points"]),
                "rmw": int(row["rmw"]),
                "code_writes": int(row["code_writes"]),
                "residual_writes": int(row["residual_writes"]),
                "raw_result": "see original report_summary.json",
            }
        )

    combined = original_rows + rows
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Post-lock no-retuning replication of the selected SRNN Wout-only Q2.14 "
            "direct-response profile. New support seeds remain in the training-validation "
            "index space and every query uses the complete official test partition. The "
            "replication strengthens precision but does not replace the pre-locked report, "
            "prove natural-domain robustness, or provide RTL evidence."
        ),
        "candidate": {
            "mask": "wout",
            "optimizer": "direct_response",
            "wout_increment": 31,
            "wout_threshold": 65,
            "weight_format": "signed_q2_14",
            "support_samples": 128,
            "shift_channels": list(EXPECTED_SHIFTS),
        },
        "replication_seeds": list(EXPECTED_SEEDS),
        "replication": summarize(rows),
        "combined_with_prelocked_report": summarize(combined),
        "episodes": combined,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    rep = summary["replication"]
    total = summary["combined_with_prelocked_report"]
    lines = [
        "# SRNN Wout Post-lock Replication",
        "",
        "Status: PASS. The selected Wout/Q2.14 direct-response profile was not retuned.",
        "",
        "| Scope | Episodes | Seed clusters | Gain (points) | Positive | Clean budget | Seed-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| New replication | {rep['episodes']} | {rep['support_seed_clusters']} | "
            f"{rep['shifted_gain_points']['mean']:+.3f} +/- {rep['shifted_gain_points']['stdev']:.3f} | "
            f"{rep['positive_gain_episodes']}/{rep['episodes']} | "
            f"{rep['clean_loss_within_1pt_episodes']}/{rep['episodes']} | "
            f"[{rep['seed_cluster_bootstrap']['ci95_low']:+.3f}, "
            f"{rep['seed_cluster_bootstrap']['ci95_high']:+.3f}] |"
        ),
        (
            f"| Combined descriptive | {total['episodes']} | {total['support_seed_clusters']} | "
            f"{total['shifted_gain_points']['mean']:+.3f} +/- {total['shifted_gain_points']['stdev']:.3f} | "
            f"{total['positive_gain_episodes']}/{total['episodes']} | "
            f"{total['clean_loss_within_1pt_episodes']}/{total['episodes']} | "
            f"[{total['seed_cluster_bootstrap']['ci95_low']:+.3f}, "
            f"{total['seed_cluster_bootstrap']['ci95_high']:+.3f}] |"
        ),
        "",
        "The combined row is a post-lock descriptive replication, not a replacement",
        "for the original pre-locked report. Bootstrap units are support seeds so the",
        "two shift directions from one seed are not treated as independent samples.",
        "Synthetic frequency shifts are not natural-domain evidence, and these are",
        "software-reference results rather than RTL dataset accuracy.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

