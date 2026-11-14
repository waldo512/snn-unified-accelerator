#!/usr/bin/env python3
"""Aggregate fixed-profile SCNN adaptation-window sensitivity evidence."""

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
    / "scnn_q1_15_window_sensitivity_20260927"
)
RAW = OUTPUT / "raw"
SEEDS = (38472, 1729, 65537, 99891)
CORRUPTIONS = ("jitter1", "drop40", "flip40")
WINDOWS = (1, 2, 4, 8)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def parse(path: Path, corruption: str, seed: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("configuration", {})
    if payload.get("status") != "pass" or payload.get("partition") != "development":
        raise RuntimeError(f"invalid sensitivity result: {path}")
    expected = {
        "masks": ["classifier"],
        "conv_thresholds": [1],
        "classifier_thresholds": [1],
        "windows": list(WINDOWS),
        "adapt_batch_size": 4,
        "weight_fractional_bits": 15,
        "update_rule": "fixed_sgd",
        "fixed_sgd_residual_bits": 15,
        "full_adaptation_batches_only": True,
        "force_class_mixed_batches": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"configuration drift for {key}: {path}")

    rows: list[dict[str, Any]] = []
    for episode in payload["episodes"]:
        candidates = episode.get("candidates", [])
        if len(candidates) != len(WINDOWS):
            raise RuntimeError(f"candidate-count drift: {path}")
        for candidate in candidates:
            if candidate.get("mask") != "classifier" or not candidate.get("mask_realized"):
                raise RuntimeError(f"classifier realization drift: {path}")
            window = int(candidate["window_timesteps"])
            if window not in WINDOWS:
                raise RuntimeError(f"unexpected window {window}: {path}")
            update = candidate["update"]
            rows.append(
                {
                    "corruption": corruption,
                    "corruption_seed": seed,
                    "support_subject": int(episode["support_subject"]),
                    "query_subject": int(episode["query_subject"]),
                    "window_timesteps": window,
                    "corrupted_gain_points": float(candidate["shifted_gain_points"]),
                    "clean_change_points": float(candidate["clean_retention_points"]),
                    "code_writes": int(candidate["code_writes"]),
                    "update_windows": int(update["windows"]),
                    "adaptation_samples_used": int(update["adaptation_samples_used"]),
                    "raw_result": str(path.relative_to(ROOT)),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gain = [float(row["corrupted_gain_points"]) for row in rows]
    clean = [float(row["clean_change_points"]) for row in rows]
    return {
        "episodes": len(rows),
        "gain_points": stats(gain),
        "clean_change_points": stats(clean),
        "positive_episodes": sum(value > 0.0 for value in gain),
        "nondegrading_episodes": sum(value >= 0.0 for value in gain),
        "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in clean),
        "mean_code_writes": statistics.fmean(int(row["code_writes"]) for row in rows),
        "mean_update_windows": statistics.fmean(int(row["update_windows"]) for row in rows),
        "mean_adaptation_samples_used": statistics.fmean(
            int(row["adaptation_samples_used"]) for row in rows
        ),
    }


def main() -> int:
    rows: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            path = RAW / f"scnn_{corruption}_q1_15_window_sensitivity_seed{seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing sensitivity result: {path}")
            rows.extend(parse(path, corruption, seed))

    expected_rows = len(CORRUPTIONS) * len(SEEDS) * 4 * len(WINDOWS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(rows)}")

    by_window = {
        str(window): summarize([row for row in rows if row["window_timesteps"] == window])
        for window in WINDOWS
    }
    paired: dict[tuple[str, int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["corruption"]),
            int(row["corruption_seed"]),
            int(row["support_subject"]),
            int(row["query_subject"]),
        )
        paired[key][int(row["window_timesteps"])] = row
    if any(set(values) != set(WINDOWS) for values in paired.values()):
        raise RuntimeError("window pairing is incomplete")

    deltas_vs_window4: dict[str, Any] = {}
    for window in WINDOWS:
        if window == 4:
            continue
        gain_delta = [
            values[window]["corrupted_gain_points"]
            - values[4]["corrupted_gain_points"]
            for values in paired.values()
        ]
        clean_delta = [
            values[window]["clean_change_points"] - values[4]["clean_change_points"]
            for values in paired.values()
        ]
        write_ratio = [
            values[window]["code_writes"] / values[4]["code_writes"]
            if values[4]["code_writes"]
            else 0.0
            for values in paired.values()
        ]
        deltas_vs_window4[str(window)] = {
            "paired_episodes": len(gain_delta),
            "gain_delta_points": stats(gain_delta),
            "clean_delta_points": stats(clean_delta),
            "code_write_ratio": stats(write_ratio),
        }

    payload = {
        "status": "pass",
        "claim_boundary": (
            "Post-lock development-set sensitivity of the fixed classifier-only "
            "Q0.15 profile. The grid changes only observation-window length and "
            "does not reselect a mask, threshold, learning rate, batch size, report "
            "subjects, or corruption definition. Results are descriptive and do not "
            "replace the pre-locked report."
        ),
        "fixed_profile": {
            "mask": "classifier",
            "adapt_batch_size": 4,
            "update_rule": "fixed_sgd",
            "optimizer_lr": 1.1920928955078125e-6,
            "weight_fractional_bits": 15,
            "fixed_sgd_residual_bits": 15,
            "windows": list(WINDOWS),
        },
        "rows": len(rows),
        "paired_episode_keys": len(paired),
        "by_window": by_window,
        "paired_deltas_vs_window4": deltas_vs_window4,
        "episodes": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SCNN Q0.15 Adaptation-window Sensitivity",
        "",
        "Status: PASS. This is a post-lock development sensitivity, not a new selection run.",
        "",
        "| Window T | Episodes | Gain (points) | Positive / nondegrading | Clean budget | Mean code writes | Mean update windows |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in WINDOWS:
        item = by_window[str(window)]
        gain = item["gain_points"]
        lines.append(
            f"| {window} | {item['episodes']} | {gain['mean']:+.3f} +/- {gain['stdev']:.3f} | "
            f"{item['positive_episodes']}/{item['episodes']} / "
            f"{item['nondegrading_episodes']}/{item['episodes']} | "
            f"{item['clean_loss_within_1pt_episodes']}/{item['episodes']} | "
            f"{item['mean_code_writes']:.1f} | {item['mean_update_windows']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Each row uses the same four development subject directions, four seeds,",
            "and three corruptions. Paired deltas against T=4 are in `summary.json`.",
            "The report subjects remain untouched by this sensitivity analysis.",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
