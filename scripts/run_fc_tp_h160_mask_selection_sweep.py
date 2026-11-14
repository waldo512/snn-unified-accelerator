#!/usr/bin/env python3
"""Run repeated FC H160 mask selection without reading the official test split."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EPISODE_SCRIPT = ROOT / "scripts" / "run_fc_tp_h160_mask_ablation.py"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "fc_h160_independent_selection"
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def parse_shift(text: str) -> tuple[int, int]:
    fields = text.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("shift must use dx,dy syntax")
    def parse_component(field: str) -> int:
        # ``m2`` is accepted because argparse otherwise treats ``-2,0`` as a
        # new option instead of one value of a nargs argument.
        return -int(field[1:]) if field.startswith("m") else int(field)

    return parse_component(fields[0]), parse_component(fields[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episode-seeds",
        nargs="+",
        type=int,
        default=[38472, 38473, 38474, 38475, 38476],
    )
    parser.add_argument(
        "--shifts",
        nargs="+",
        type=parse_shift,
        default=[(2, 0), (-2, 0), (0, 2)],
        help="Zero-fill shifts as dx,dy; use m2,0 for a negative component.",
    )
    parser.add_argument("--support-per-class", type=int, default=16)
    parser.add_argument("--selection-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--trainable-masks",
        nargs="+",
        choices=("off", "w2", "w0", "w0_w2"),
        default=["off", "w2", "w0", "w0_w2"],
    )
    parser.add_argument("--learning-rate", type=float, default=2.0**-14)
    parser.add_argument(
        "--optimizer",
        choices=(
            "direct_sign",
            "fixed_sgd",
            "int8_direct_residual",
            "int8_magnitude_residual",
            "adam",
            "sgd_momentum",
        ),
        default="direct_sign",
    )
    parser.add_argument("--vote-increment", type=int, default=1)
    parser.add_argument("--vote-threshold", type=int, default=1)
    parser.add_argument(
        "--source-trace-bits",
        type=int,
        choices=(16, 17),
        default=16,
    )
    parser.add_argument(
        "--source-trace-format",
        choices=("rtl_u16_integer", "rtl_uq6_10", "q0_16_saturating"),
        default="rtl_u16_integer",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for dx, dy in args.shifts:
        shift_name = f"x{'m' if dx < 0 else 'p'}{abs(dx)}_y{'m' if dy < 0 else 'p'}{abs(dy)}"
        for seed in args.episode_seeds:
            output = raw_dir / f"seed{seed}_{shift_name}.json"
            command = [
                sys.executable,
                str(EPISODE_SCRIPT),
                "--episode-seed",
                str(seed),
                "--query-split",
                "train",
                "--adapt-samples-per-class",
                str(args.support_per_class),
                "--test-samples-per-class",
                str(args.selection_per_class),
                "--batch-size",
                str(args.batch_size),
                "--shift-x",
                str(dx),
                "--shift-y",
                str(dy),
                "--trainable-masks",
                *args.trainable_masks,
                "--learning-rate",
                str(args.learning_rate),
                "--optimizer",
                args.optimizer,
                "--vote-increment",
                str(args.vote_increment),
                "--vote-threshold",
                str(args.vote_threshold),
                "--source-trace-bits",
                str(args.source_trace_bits),
                "--source-trace-format",
                args.source_trace_format,
                "--device",
                args.device,
                "--output",
                str(output),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            records.append(json.loads(output.read_text(encoding="utf-8")))

    summary: dict = {
        "status": "pass",
        "claim_boundary": (
            "Selection-only FC H160 mask repetitions. Support and selection are "
            "disjoint class-balanced subsets of the N-MNIST training split; the "
            "official test split is not read by this runner."
        ),
        "episode_seeds": args.episode_seeds,
        "shifts": [{"dx": dx, "dy": dy} for dx, dy in args.shifts],
        "support_per_class": args.support_per_class,
        "selection_per_class": args.selection_per_class,
        "trainable_masks": args.trainable_masks,
        "optimizer": args.optimizer,
        "vote_increment": args.vote_increment,
        "vote_threshold": args.vote_threshold,
        "source_trace_bits": args.source_trace_bits,
        "source_trace_format": args.source_trace_format,
        "learning_rate_or_step": args.learning_rate,
        "device": args.device,
        "masks": {},
    }
    for dx, dy in args.shifts:
        shift_key = f"{dx},{dy}"
        shift_records = [
            record
            for record in records
            if record["shift"]["dx_pixels"] == dx
            and record["shift"]["dy_pixels"] == dy
        ]
        per_mask: dict[str, dict] = {}
        for mask_name in args.trainable_masks:
            items = [
                next(item for item in record["masks"] if item["trainable_mask"] == mask_name)
                for record in shift_records
            ]
            gains = [float(item["shifted_gain_points"]) for item in items]
            retention = [float(item["clean_retention_points"]) for item in items]
            per_mask[mask_name] = {
                "shifted_gain_points": stats(gains),
                "clean_retention_points": stats(retention),
                "positive_gain_episodes": sum(value > 0.0 for value in gains),
                "clean_loss_within_1pt_episodes": sum(value >= -1.0 for value in retention),
                "all_episodes_positive_and_clean_within_1pt": all(
                    gain > 0.0 and clean >= -1.0
                    for gain, clean in zip(gains, retention)
                ),
                "mutable_parameters": items[0]["state_cost"]["mutable_parameters"],
                "code_plus_residual_bytes": items[0]["state_cost"][
                    "code_plus_residual_bytes"
                ],
            }
        summary["masks"][shift_key] = per_mask

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"FC_TP_H160_SELECTION_SWEEP_PASS summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
