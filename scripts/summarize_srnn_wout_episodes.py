#!/usr/bin/env python3
"""Aggregate independent SRNN Wout-only mask episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results" / "paper_evidence" / "algorithm" / "mask_ablation_20260919"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "srnn_wout_independent_episodes_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pattern",
        default="srnn_wout_episode_seed*_shift16.json",
    )
    parser.add_argument("--shift-channels", type=int, default=16)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError("no independent SRNN Wout episode files found")
    grouped: dict[str, list[dict[str, object]]] = {}
    seeds: list[int] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds.append(int(payload["protocol"]["episode_seed"]))
        for item in payload["optimizers"]:
            grouped.setdefault(str(item["optimizer"]), []).append(item)

    rows: list[dict[str, object]] = []
    for optimizer, episodes in sorted(grouped.items()):
        shifted = 100.0 * np.asarray([
            float(item["shifted_accuracy_improvement"]) for item in episodes
        ])
        clean = 100.0 * np.asarray([
            float(item["clean_accuracy_change"]) for item in episodes
        ])
        rmw_values = [
            int(item["traffic"]["rmw"])
            for item in episodes
            if item.get("traffic") is not None
        ]
        rows.append({
            "optimizer": optimizer,
            "episodes": len(episodes),
            "shifted_gain_points_mean": float(shifted.mean()),
            "shifted_gain_points_std": float(shifted.std(ddof=0)),
            "shifted_positive_episodes": int((shifted > 0.0).sum()),
            "shifted_gain_points_min": float(shifted.min()),
            "shifted_gain_points_max": float(shifted.max()),
            "clean_retention_points_mean": float(clean.mean()),
            "clean_retention_points_std": float(clean.std(ddof=0)),
            "rmw_mean": float(np.mean(rmw_values)) if rmw_values else None,
            "mutable_parameters": int(episodes[0]["mutable_parameters"]),
            "mutable_code_residual_bytes": int(
                episodes[0]["q2_14_code_plus_residual_bytes"]
            ),
        })
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Five validation-split support episodes for the Wout-only mask. "
            "This is software algorithm evidence, not RTL or official-test accuracy."
        ),
        "shift_channels": args.shift_channels,
        "episode_seeds": seeds,
        "source_files": [str(path) for path in files],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
