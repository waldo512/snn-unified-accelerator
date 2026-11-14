#!/usr/bin/env python3
"""Aggregate subject-disjoint SCNN corruption experiments across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask", default="conv8")
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else 0.0


def population_std(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.pstdev(items) if len(items) > 1 else 0.0


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mask"],
        int(row["conv_vote_threshold"]),
        int(row["classifier_vote_threshold"]),
        int(row["window_timesteps"]),
        str(row["support_domain"]),
        str(row["update_rule"]),
    )


def load_observations(paths: list[Path], mask: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"{path}: expected status=pass")
        if payload.get("partition") != "development":
            raise ValueError(f"{path}: only development evidence may select a mask")
        configuration = payload["configuration"]
        seed = int(configuration["corruption_seed"])
        for episode in payload["episodes"]:
            support_subject = int(episode["support_subject"])
            query_subject = int(episode["query_subject"])
            for candidate in episode["candidates"]:
                if candidate["mask"] != mask:
                    continue
                observations.append(
                    {
                        "source_file": path.name,
                        "corruption_seed": seed,
                        "support_subject": support_subject,
                        "query_subject": query_subject,
                        "mask": candidate["mask"],
                        "conv_vote_threshold": int(
                            candidate["conv_vote_threshold"]
                        ),
                        "classifier_vote_threshold": int(
                            candidate["classifier_vote_threshold"]
                        ),
                        "window_timesteps": int(candidate["window_timesteps"]),
                        "support_domain": str(
                            candidate["update"]["support_domain"]
                        ),
                        "update_rule": str(candidate["update"]["update_rule"]),
                        "baseline_clean_percent": float(
                            episode["baseline_clean"]["accuracy_percent"]
                        ),
                        "baseline_corrupt_percent": float(
                            episode["baseline_shifted"]["accuracy_percent"]
                        ),
                        "corrupt_gain_points": float(
                            candidate["shifted_gain_points"]
                        ),
                        "clean_retention_points": float(
                            candidate["clean_retention_points"]
                        ),
                        "code_writes": int(candidate["code_writes"]),
                        "mask_realized": bool(candidate["mask_realized"]),
                        "mutable_parameters": int(candidate["mutable_parameters"]),
                        "mutable_state_bytes": int(candidate["mutable_state_bytes"]),
                    }
                )
    if not observations:
        raise ValueError(f"no {mask!r} observations found")
    return observations


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [float(row["corrupt_gain_points"]) for row in rows]
    retention = [float(row["clean_retention_points"]) for row in rows]
    writes = [int(row["code_writes"]) for row in rows]
    realized = [bool(row["mask_realized"]) for row in rows]
    seeds = {int(row["corruption_seed"]) for row in rows}
    pairs = {
        (int(row["support_subject"]), int(row["query_subject"])) for row in rows
    }
    positive = sum(value > 0.0 for value in gains)
    forgetting = sum(value < 0.0 for value in retention)
    return {
        "observations": len(rows),
        "seeds": len(seeds),
        "subject_pairs": len(pairs),
        "corrupt_gain_mean": mean(gains),
        "corrupt_gain_std": population_std(gains),
        "corrupt_gain_min": min(gains),
        "corrupt_gain_max": max(gains),
        "positive_gain_observations": positive,
        "positive_gain_fraction": positive / len(rows),
        "clean_retention_mean": mean(retention),
        "clean_retention_std": population_std(retention),
        "clean_retention_min": min(retention),
        "clean_retention_max": max(retention),
        "forgetting_observations": forgetting,
        "forgetting_fraction": forgetting / len(rows),
        "code_writes_mean": mean(writes),
        "code_writes_std": population_std(writes),
        "mask_realized_observations": sum(realized),
        "mask_realized_fraction": sum(realized) / len(rows),
        "mutable_parameters": int(rows[0]["mutable_parameters"]),
        "mutable_state_bytes": int(rows[0]["mutable_state_bytes"]),
    }


def add_key_fields(summary: dict[str, Any], key: tuple[Any, ...]) -> dict[str, Any]:
    (
        mask,
        conv_threshold,
        classifier_threshold,
        window,
        support_domain,
        update_rule,
    ) = key
    return {
        "mask": mask,
        "conv_vote_threshold": conv_threshold,
        "classifier_vote_threshold": classifier_threshold,
        "window_timesteps": window,
        "support_domain": support_domain,
        "update_rule": update_rule,
        **summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = sorted({path.resolve() for path in args.inputs})
    observations = load_observations(paths, args.mask)

    by_candidate: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_seed_candidate: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_pair_candidate: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        key = candidate_key(row)
        by_candidate[key].append(row)
        by_seed_candidate[(int(row["corruption_seed"]), *key)].append(row)
        by_pair_candidate[
            (int(row["support_subject"]), int(row["query_subject"]), *key)
        ].append(row)

    candidate_rows = [
        add_key_fields(summarize(rows), key)
        for key, rows in sorted(by_candidate.items())
    ]
    seed_rows = []
    for compound_key, rows in sorted(by_seed_candidate.items()):
        seed, *key = compound_key
        seed_rows.append(
            {"corruption_seed": seed, **add_key_fields(summarize(rows), tuple(key))}
        )
    pair_rows = []
    for compound_key, rows in sorted(by_pair_candidate.items()):
        support_subject, query_subject, *key = compound_key
        pair_rows.append(
            {
                "support_subject": support_subject,
                "query_subject": query_subject,
                **add_key_fields(summarize(rows), tuple(key)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episode_observations.csv", observations)
    write_csv(args.output_dir / "candidate_summary.csv", candidate_rows)
    write_csv(args.output_dir / "seed_summary.csv", seed_rows)
    write_csv(args.output_dir / "subject_pair_summary.csv", pair_rows)
    aggregate = {
        "status": "pass",
        "claim_boundary": (
            "Development subject pairs and bounded synthetic event corruption only. "
            "Report subjects remain unread; this evidence may select but not report "
            "the final SCNN learning scope."
        ),
        "input_files": [str(path) for path in paths],
        "candidate_summary": candidate_rows,
        "seed_summary": seed_rows,
        "subject_pair_summary": pair_rows,
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
