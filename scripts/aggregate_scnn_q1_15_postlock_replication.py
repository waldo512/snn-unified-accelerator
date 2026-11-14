#!/usr/bin/env python3
"""Aggregate no-retuning SCNN classifier post-lock replication episodes."""

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
OUTPUT = BASE / "scnn_q1_15_postlock_replication_20260927"
RAW = OUTPUT / "raw"
SEEDS = (104729, 130363, 155921, 196613)
CORRUPTIONS = ("jitter1", "drop40", "flip40")


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def bootstrap(values: list[float]) -> dict[str, float | int | str]:
    rng = random.Random(20260927)
    draws = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(20000)]
    draws.sort()
    return {
        "mean": statistics.fmean(values),
        "ci95_low": draws[int(0.025 * len(draws))],
        "ci95_high": draws[int(0.975 * len(draws)) - 1],
        "replicates": len(draws),
        "unit": "corruption_seed",
    }


def sign_test(values: list[float]) -> dict[str, float | int]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    nonzero = positive + negative
    return {
        "positive_clusters": positive,
        "negative_clusters": negative,
        "zero_clusters": len(values) - nonzero,
        "one_sided_exact_p": (
            sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1)) / 2**nonzero
            if nonzero
            else 1.0
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [float(row["corrupted_gain_points"]) for row in rows]
    clean = [float(row["clean_change_points"]) for row in rows]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["corruption_seed"])].append(float(row["corrupted_gain_points"]))
    seed_means = [statistics.fmean(by_seed[key]) for key in sorted(by_seed)]
    return {
        "episodes": len(rows),
        "corruption_seed_clusters": len(seed_means),
        "gain_points": stats(gains),
        "clean_change_points": stats(clean),
        "positive_episodes": sum(value > 0.0 for value in gains),
        "nondegrading_episodes": sum(value >= 0.0 for value in gains),
        "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
        "mean_code_writes": statistics.fmean(int(row["code_writes"]) for row in rows),
        "seed_mean_gain_points": seed_means,
        "seed_cluster_bootstrap": bootstrap(seed_means),
        "seed_cluster_sign_test": sign_test(seed_means),
    }


def parse_new(path: Path, corruption: str, seed: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("configuration", {})
    if payload.get("status") != "pass" or payload.get("partition") != "report":
        raise RuntimeError(f"invalid SCNN replication result: {path}")
    expected = {
        "masks": ["classifier"],
        "classifier_thresholds": [1],
        "windows": [4],
        "adapt_batch_size": 4,
        "weight_fractional_bits": 15,
        "update_rule": "fixed_sgd",
        "fixed_sgd_residual_bits": 15,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"configuration drift for {key}: {path}")
    rows = []
    for episode in payload["episodes"]:
        candidates = episode.get("candidates", [])
        if len(candidates) != 1:
            raise RuntimeError(f"candidate-count drift: {path}")
        candidate = candidates[0]
        if candidate.get("mask") != "classifier" or not candidate.get("mask_realized"):
            raise RuntimeError(f"candidate realization drift: {path}")
        rows.append(
            {
                "source": "postlock_replication",
                "corruption": corruption,
                "corruption_seed": seed,
                "support_subject": int(episode["support_subject"]),
                "query_subject": int(episode["query_subject"]),
                "corrupted_gain_points": float(candidate["shifted_gain_points"]),
                "clean_change_points": float(candidate["clean_retention_points"]),
                "code_writes": int(candidate["code_writes"]),
                "raw_result": str(path.relative_to(ROOT)),
            }
        )
    return rows


def original_rows() -> list[dict[str, Any]]:
    jitter_path = BASE / "scnn_event_fixed_sgd_q1_15_report_subjects_20260924" / "report_summary.json"
    jitter = json.loads(jitter_path.read_text(encoding="utf-8"))
    cross_path = BASE / "scnn_q1_15_cross_corruption_report_20260927" / "summary.json"
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    rows = [
        {
            "source": "prelocked_report",
            "corruption": "jitter1",
            "corruption_seed": int(row["corruption_seed"]),
            "support_subject": int(row["support_subject"]),
            "query_subject": int(row["query_subject"]),
            "corrupted_gain_points": float(row["shifted_gain_points"]),
            "clean_change_points": float(row["clean_retention_points"]),
            "code_writes": int(row["code_writes"]),
            "raw_result": str(jitter_path.relative_to(ROOT)),
        }
        for row in jitter["episodes_detail"]
    ]
    rows.extend(
        {
            "source": "postlock_cross_corruption_extension",
            "corruption": str(row["corruption"]),
            "corruption_seed": int(row["corruption_seed"]),
            "support_subject": int(row["support_subject"]),
            "query_subject": int(row["query_subject"]),
            "corrupted_gain_points": float(row["corrupted_gain_points"]),
            "clean_change_points": float(row["clean_change_points"]),
            "code_writes": int(row["code_writes"]),
            "raw_result": str(cross_path.relative_to(ROOT)),
        }
        for row in cross["episodes"]
    )
    return rows


def main() -> int:
    new_rows: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            path = RAW / f"scnn_{corruption}_q1_15_classifier_seed{seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing SCNN replication result: {path}")
            new_rows.extend(parse_new(path, corruption, seed))
    combined = original_rows() + new_rows
    by_corruption = {
        corruption: summarize([row for row in combined if row["corruption"] == corruption])
        for corruption in CORRUPTIONS
    }
    payload = {
        "status": "pass",
        "claim_boundary": (
            "Post-lock no-retuning replication of the selected classifier-only Q0.15 "
            "profile on the already-open report subjects. New corruption seeds improve "
            "precision but do not replace the pre-locked report, select a new operating "
            "point, establish natural-domain robustness, or provide RTL evidence."
        ),
        "candidate": {
            "mask": "classifier",
            "window_timesteps": 4,
            "adapt_batch_size": 4,
            "update_rule": "fixed_sgd",
            "optimizer_lr": 1.1920928955078125e-6,
            "weight_fractional_bits": 15,
            "fixed_sgd_residual_bits": 15,
        },
        "replication_seeds": list(SEEDS),
        "replication": summarize(new_rows),
        "combined_with_original": summarize(combined),
        "combined_by_corruption": by_corruption,
        "episodes": combined,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    rep = payload["replication"]
    total = payload["combined_with_original"]
    lines = [
        "# SCNN Classifier Post-lock Replication",
        "",
        "Status: PASS. The classifier/Q0.15 operating point was not retuned.",
        "",
        "| Scope | Episodes | Seed clusters | Gain (points) | Positive / nondegrading | Clean budget | Seed-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| New replication | {rep['episodes']} | {rep['corruption_seed_clusters']} | "
            f"{rep['gain_points']['mean']:+.3f} +/- {rep['gain_points']['stdev']:.3f} | "
            f"{rep['positive_episodes']}/{rep['episodes']} / {rep['nondegrading_episodes']}/{rep['episodes']} | "
            f"{rep['clean_loss_within_1pt_episodes']}/{rep['episodes']} | "
            f"[{rep['seed_cluster_bootstrap']['ci95_low']:+.3f}, {rep['seed_cluster_bootstrap']['ci95_high']:+.3f}] |"
        ),
        (
            f"| Combined descriptive | {total['episodes']} | {total['corruption_seed_clusters']} | "
            f"{total['gain_points']['mean']:+.3f} +/- {total['gain_points']['stdev']:.3f} | "
            f"{total['positive_episodes']}/{total['episodes']} / {total['nondegrading_episodes']}/{total['episodes']} | "
            f"{total['clean_loss_within_1pt_episodes']}/{total['episodes']} | "
            f"[{total['seed_cluster_bootstrap']['ci95_low']:+.3f}, {total['seed_cluster_bootstrap']['ci95_high']:+.3f}] |"
        ),
        "",
        "The combined row is descriptive post-lock evidence. Bootstrap units are",
        "corruption seeds, so two subject directions and three corruptions sharing one",
        "seed are not treated as independent. Report subjects were already open before",
        "the replication seeds were added; the original pre-locked report remains the",
        "primary claim source.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

