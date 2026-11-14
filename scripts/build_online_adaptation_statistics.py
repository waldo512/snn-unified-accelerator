#!/usr/bin/env python3
"""Build seed-clustered statistics for the locked online-adaptation reports."""

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
OUTPUT = ROOT / "results" / "paper_evidence" / "algorithm" / "online_adaptation_statistics_20260927"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"source evidence is not complete: {path}")
    return payload


def mean_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def bootstrap(values: list[float], seed: int) -> dict[str, float | int | str]:
    rng = random.Random(seed)
    draws = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(20000)]
    draws.sort()
    return {
        "mean": statistics.fmean(values),
        "ci95_low": draws[int(0.025 * len(draws))],
        "ci95_high": draws[int(0.975 * len(draws)) - 1],
        "replicates": len(draws),
        "unit": "support_or_corruption_seed",
    }


def sign_test(values: list[float]) -> dict[str, float | int]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    nonzero = positive + negative
    p_value = (
        sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1)) / 2**nonzero
        if nonzero
        else 1.0
    )
    return {
        "positive_clusters": positive,
        "negative_clusters": negative,
        "zero_clusters": len(values) - nonzero,
        "one_sided_exact_p": p_value,
    }


def summarize(
    topology: str,
    rows: list[dict[str, Any]],
    *,
    cluster_key: str,
    source: str,
    scope: str,
) -> dict[str, Any]:
    gains = [float(row["gain"]) for row in rows]
    clean = [float(row["clean"]) for row in rows]
    clusters: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        clusters[int(row[cluster_key])].append(float(row["gain"]))
    cluster_means = [statistics.fmean(clusters[key]) for key in sorted(clusters)]
    return {
        "topology": topology,
        "scope": scope,
        "episodes": len(rows),
        "clusters": len(cluster_means),
        "episode_gain_points": mean_stats(gains),
        "episode_clean_change_points": mean_stats(clean),
        "positive_episodes": sum(value > 0.0 for value in gains),
        "nondegrading_episodes": sum(value >= 0.0 for value in gains),
        "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
        "cluster_mean_gain_points": cluster_means,
        "cluster_bootstrap": bootstrap(cluster_means, 20260927 + len(rows)),
        "cluster_sign_test": sign_test(cluster_means),
        "source": source,
    }


def fc_rows() -> list[dict[str, Any]]:
    payload = load(
        BASE
        / "fc_h160_w2_int8_magnitude_uq6_10_official_test_report_20260924"
        / "report_summary.json"
    )
    return [
        {
            "seed": int(row["seed"]),
            "gain": float(row["shifted_gain_points"]),
            "clean": float(row["clean_retention_points"]),
        }
        for row in payload["episodes_detail"]
    ]


def srnn_rows() -> tuple[list[dict[str, Any]], str, str]:
    replication = BASE / "srnn_wout_postlock_replication_20260927" / "summary.json"
    if replication.exists():
        payload = load(replication)
        rows = payload["episodes"]
        return (
            [
                {
                    "seed": int(row["seed"]),
                    "gain": float(row["shifted_gain_points"]),
                    "clean": float(row["clean_change_points"]),
                }
                for row in rows
            ],
            str(replication.relative_to(ROOT)),
            "pre-locked report plus no-retuning post-lock replication",
        )
    report = BASE / "srnn_wout_train_support_official_test_20260927" / "report_summary.json"
    payload = load(report)
    return (
        [
            {
                "seed": int(row["seed"]),
                "gain": float(row["shifted_gain_points"]),
                "clean": float(row["clean_retention_points"]),
            }
            for row in payload["episodes_detail"]
        ],
        str(report.relative_to(ROOT)),
        "pre-locked cross-partition report",
    )


def scnn_rows() -> tuple[list[dict[str, Any]], str, str]:
    replication = BASE / "scnn_q1_15_postlock_replication_20260927" / "summary.json"
    if replication.exists():
        payload = load(replication)
        return (
            [
                {
                    "seed": int(row["corruption_seed"]),
                    "gain": float(row["corrupted_gain_points"]),
                    "clean": float(row["clean_change_points"]),
                }
                for row in payload["episodes"]
            ],
            str(replication.relative_to(ROOT)),
            "pre-locked report plus no-retuning post-lock replication",
        )

    jitter = load(
        BASE
        / "scnn_event_fixed_sgd_q1_15_report_subjects_20260924"
        / "report_summary.json"
    )
    cross = load(BASE / "scnn_q1_15_cross_corruption_report_20260927" / "summary.json")
    rows = [
        {
            "seed": int(row["corruption_seed"]),
            "gain": float(row["shifted_gain_points"]),
            "clean": float(row["clean_retention_points"]),
        }
        for row in jitter["episodes_detail"]
    ]
    rows.extend(
        {
            "seed": int(row["corruption_seed"]),
            "gain": float(row["corrupted_gain_points"]),
            "clean": float(row["clean_change_points"]),
        }
        for row in cross["episodes"]
    )
    return (
        rows,
        str(
            (BASE / "scnn_q1_15_cross_corruption_report_20260927" / "summary.json").relative_to(
                ROOT
            )
        ),
        "locked report plus post-lock cross-corruption extension",
    )


def main() -> int:
    srnn, srnn_source, srnn_scope = srnn_rows()
    scnn, scnn_source, scnn_scope = scnn_rows()
    summaries = [
        summarize(
            "FC",
            fc_rows(),
            cluster_key="seed",
            source=str(
                (
                    BASE
                    / "fc_h160_w2_int8_magnitude_uq6_10_official_test_report_20260924"
                    / "report_summary.json"
                ).relative_to(ROOT)
            ),
            scope="pre-locked official-test report",
        ),
        summarize(
            "SRNN",
            srnn,
            cluster_key="seed",
            source=srnn_source,
            scope=srnn_scope,
        ),
        summarize(
            "SCNN",
            scnn,
            cluster_key="seed",
            source=scnn_source,
            scope=scnn_scope,
        ),
    ]
    payload = {
        "status": "pass",
        "claim_boundary": (
            "Descriptive episode statistics plus a seed-cluster bootstrap and exact "
            "one-sided sign test. Multiple shifts, directions, or corruptions sharing "
            "one seed are clustered before resampling. Small cluster counts and fixed "
            "query populations limit inferential strength; p-values are supporting "
            "diagnostics, not standalone proof of generalization."
        ),
        "topologies": summaries,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "topology",
                "scope",
                "episodes",
                "clusters",
                "gain_mean",
                "gain_stdev",
                "positive_episodes",
                "nondegrading_episodes",
                "clean_budget_episodes",
                "cluster_ci95_low",
                "cluster_ci95_high",
                "cluster_sign_p",
                "source",
            ),
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "topology": row["topology"],
                    "scope": row["scope"],
                    "episodes": row["episodes"],
                    "clusters": row["clusters"],
                    "gain_mean": row["episode_gain_points"]["mean"],
                    "gain_stdev": row["episode_gain_points"]["stdev"],
                    "positive_episodes": row["positive_episodes"],
                    "nondegrading_episodes": row["nondegrading_episodes"],
                    "clean_budget_episodes": row["clean_loss_within_1pt_episodes"],
                    "cluster_ci95_low": row["cluster_bootstrap"]["ci95_low"],
                    "cluster_ci95_high": row["cluster_bootstrap"]["ci95_high"],
                    "cluster_sign_p": row["cluster_sign_test"]["one_sided_exact_p"],
                    "source": row["source"],
                }
            )

    lines = [
        "# Seed-clustered Online-Adaptation Statistics",
        "",
        "Status: PASS. These statistics summarize locked candidates; they do not",
        "select new masks, rates, or corruption settings.",
        "",
        "| Topology | Scope | Episodes / seed clusters | Gain (points) | Positive / nondegrading | Clean budget | Cluster-bootstrap 95% CI | Exact sign p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        gain = row["episode_gain_points"]
        boot = row["cluster_bootstrap"]
        sign = row["cluster_sign_test"]
        lines.append(
            f"| {row['topology']} | {row['scope']} | {row['episodes']} / {row['clusters']} | "
            f"{gain['mean']:+.3f} +/- {gain['stdev']:.3f} | "
            f"{row['positive_episodes']}/{row['episodes']} / "
            f"{row['nondegrading_episodes']}/{row['episodes']} | "
            f"{row['clean_loss_within_1pt_episodes']}/{row['episodes']} | "
            f"[{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}] | "
            f"{sign['one_sided_exact_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The bootstrap resamples support/corruption seeds, not individual",
            "shift-direction episodes. This prevents repeated conditions from being",
            "treated as fully independent. Cluster counts remain small, query subjects",
            "are fixed within each topology, and the perturbations are synthetic. The",
            "table therefore supports result stability and uncertainty reporting, not a",
            "universal generalization claim or a cross-topology accuracy ranking.",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
