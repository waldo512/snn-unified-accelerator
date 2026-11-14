#!/usr/bin/env python3
"""Run report-only SCNN subject-disjoint evaluation for a locked candidate.

This wrapper keeps the selection/report boundary explicit.  It reads the
candidate chosen by a development-only subject-pair sweep, runs the existing
subject-pair ablation script on the report partition with exactly that
candidate, and writes a compact summary for paper tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PAIR_SCRIPT = ROOT / "scripts" / "run_scnn_tp_subject_pair_ablation.py"
DEFAULT_SELECTION = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "scnn_subject_disjoint_locked_selection"
    / "selection_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-summary", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_selected(selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("selection summary has no locked candidate")
    required = (
        "mask",
        "window_timesteps",
        "adapt_batch_size",
        "classifier_row_mask",
        "classifier_sample_gate",
        "commit_scope",
        "support_domain",
        "update_rule",
    )
    missing = [key for key in required if key not in selected]
    if missing:
        raise RuntimeError(f"locked candidate is missing fields: {missing}")
    if selected["update_rule"] not in ("direct", "fixed_sgd"):
        raise RuntimeError(
            "subject-pair hardware report supports direct or fixed_sgd updates"
        )
    legacy_threshold = selected.get("threshold")
    if selected.get("conv_vote_threshold", legacy_threshold) is None:
        raise RuntimeError("locked candidate has no convolution vote threshold")
    if selected.get("classifier_vote_threshold", legacy_threshold) is None:
        raise RuntimeError("locked candidate has no classifier vote threshold")
    if selected["update_rule"] == "fixed_sgd":
        fixed_required = (
            "optimizer_lr",
            "fixed_sgd_residual_bits",
            "weight_fractional_bits",
            "full_adaptation_batches_only",
        )
        missing = [key for key in fixed_required if key not in selected]
        if missing:
            raise RuntimeError(
                f"locked fixed_sgd candidate is missing fields: {missing}"
            )
    return selected


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def summarize_report(
    *,
    selection: dict[str, Any],
    selection_path: Path,
    selected: dict[str, Any],
    raw_paths: list[Path],
    summary_path: Path,
) -> dict[str, Any]:
    shifted_gain: list[float] = []
    clean_retention: list[float] = []
    writes: list[int] = []
    baseline_clean: list[float] = []
    baseline_shifted: list[float] = []
    after_clean: list[float] = []
    after_shifted: list[float] = []
    episode_rows: list[dict[str, Any]] = []

    report_seeds: list[int] = []
    report_pairs: list[list[int]] | None = None
    for raw_path in raw_paths:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload.get("partition") != "report":
            raise RuntimeError(f"{raw_path} is not a report-partition result")
        if payload.get("status") != "pass":
            raise RuntimeError(
                f"{raw_path} did not pass: {payload.get('status')}"
            )
        configuration = payload["configuration"]
        seed = int(configuration["corruption_seed"])
        report_seeds.append(seed)
        current_pairs = configuration["subject_pairs"]
        if report_pairs is None:
            report_pairs = current_pairs
        elif report_pairs != current_pairs:
            raise RuntimeError("report files use different subject pairs")
        if str(configuration["update_rule"]) != str(selected["update_rule"]):
            raise RuntimeError("report update rule differs from locked selection")
        if int(configuration["weight_fractional_bits"]) != int(
            selected.get("weight_fractional_bits", 8)
        ):
            raise RuntimeError("report weight format differs from locked selection")
        if selected["update_rule"] == "fixed_sgd":
            if not np.isclose(
                float(configuration["optimizer_lr"]),
                float(selected["optimizer_lr"]),
                rtol=0.0,
                atol=1e-18,
            ):
                raise RuntimeError("report learning rate differs from locked selection")
            if int(configuration["fixed_sgd_residual_bits"]) != int(
                selected["fixed_sgd_residual_bits"]
            ):
                raise RuntimeError(
                    "report residual precision differs from locked selection"
                )
            if bool(configuration["full_adaptation_batches_only"]) != bool(
                selected["full_adaptation_batches_only"]
            ):
                raise RuntimeError(
                    "report batch-tail policy differs from locked selection"
                )

        for episode in payload["episodes"]:
            candidates = episode.get("candidates", [])
            if len(candidates) != 1:
                raise RuntimeError(
                    f"report episode {episode.get('support_subject')}->"
                    f"{episode.get('query_subject')} has {len(candidates)} candidates"
                )
            candidate = candidates[0]
            if candidate["mask"] != selected["mask"]:
                raise RuntimeError(
                    "report candidate mask differs from locked selection"
                )
            if int(candidate["window_timesteps"]) != int(
                selected["window_timesteps"]
            ):
                raise RuntimeError(
                    "report candidate window differs from locked selection"
                )
            legacy_threshold = selected.get("threshold")
            selected_conv_threshold = int(
                selected.get("conv_vote_threshold", legacy_threshold)
            )
            selected_classifier_threshold = int(
                selected.get("classifier_vote_threshold", legacy_threshold)
            )
            if int(candidate["conv_vote_threshold"]) != selected_conv_threshold:
                raise RuntimeError(
                    "report convolution threshold differs from locked selection"
                )
            if (
                int(candidate["classifier_vote_threshold"])
                != selected_classifier_threshold
            ):
                raise RuntimeError(
                    "report classifier threshold differs from locked selection"
                )
            if not bool(candidate.get("mask_realized")):
                raise RuntimeError(
                    "report candidate did not realize every selected tensor"
                )

            gain = float(candidate["shifted_gain_points"])
            retention = float(candidate["clean_retention_points"])
            write_count = int(candidate["code_writes"])
            shifted_gain.append(gain)
            clean_retention.append(retention)
            writes.append(write_count)

            base_clean = float(episode["baseline_clean"]["accuracy_percent"])
            base_shifted = float(episode["baseline_shifted"]["accuracy_percent"])
            clean = float(candidate["clean_after"]["accuracy_percent"])
            shifted = float(candidate["shifted_after"]["accuracy_percent"])
            baseline_clean.append(base_clean)
            baseline_shifted.append(base_shifted)
            after_clean.append(clean)
            after_shifted.append(shifted)
            episode_rows.append(
                {
                    "corruption_seed": seed,
                    "support_subject": int(episode["support_subject"]),
                    "query_subject": int(episode["query_subject"]),
                    "baseline_clean_accuracy_percent": base_clean,
                    "baseline_shifted_accuracy_percent": base_shifted,
                    "after_clean_accuracy_percent": clean,
                    "after_shifted_accuracy_percent": shifted,
                    "shifted_gain_points": gain,
                    "clean_retention_points": retention,
                    "code_writes": write_count,
                }
            )

    shifted_mean, shifted_std = mean_std(shifted_gain)
    retention_mean, retention_std = mean_std(clean_retention)
    writes_mean, writes_std = mean_std([float(value) for value in writes])
    base_clean_mean, _ = mean_std(baseline_clean)
    base_shifted_mean, _ = mean_std(baseline_shifted)
    after_clean_mean, _ = mean_std(after_clean)
    after_shifted_mean, _ = mean_std(after_shifted)
    summary = {
        "status": "pass",
        "claim_boundary": (
            "Report metrics use the locked subject-disjoint SCNN candidate. "
            "No report-stage mask, threshold or window comparison is performed."
        ),
        "selection_lock": str(selection_path),
        "selection_source": str(selection["source"]),
        "locked_candidate": selected,
        "report_subject_pairs": report_pairs,
        "report_corruption_seeds": report_seeds,
        "episodes": len(episode_rows),
        "baseline_clean_accuracy_mean": base_clean_mean,
        "baseline_shifted_accuracy_mean": base_shifted_mean,
        "after_clean_accuracy_mean": after_clean_mean,
        "after_shifted_accuracy_mean": after_shifted_mean,
        "report_shifted_gain_mean": shifted_mean,
        "report_shifted_gain_std": shifted_std,
        "report_clean_retention_mean": retention_mean,
        "report_clean_retention_std": retention_std,
        "code_writes_mean": writes_mean,
        "code_writes_std": writes_std,
        "episodes_detail": episode_rows,
        "raw_results": [str(path) for path in raw_paths],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    selected = require_selected(selection)
    output_dir = args.output_dir or args.selection_summary.parent / "report_subject_disjoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "report_summary.json"
    report_protocol = selection.get("report_protocol", {})
    report_seeds = report_protocol.get("corruption_seeds", [38472])
    report_pairs = report_protocol.get("subject_pairs")
    raw_paths: list[Path] = []

    for report_seed in report_seeds:
        raw_path = output_dir / f"raw_report_seed{int(report_seed)}.json"
        raw_paths.append(raw_path)
        if raw_path.exists() and not args.force:
            continue
        legacy_threshold = selected.get("threshold")
        conv_threshold = selected.get("conv_vote_threshold", legacy_threshold)
        classifier_threshold = selected.get(
            "classifier_vote_threshold", legacy_threshold
        )
        command = [
            sys.executable,
            str(SUBJECT_PAIR_SCRIPT),
            "--partition",
            "report",
            "--masks",
            str(selected["mask"]),
            "--conv-thresholds",
            str(conv_threshold),
            "--classifier-thresholds",
            str(classifier_threshold),
            "--windows",
            str(selected["window_timesteps"]),
            "--adapt-batch-size",
            str(selected["adapt_batch_size"]),
            "--classifier-row-mask",
            str(selected["classifier_row_mask"]),
            "--classifier-sample-gate",
            str(selected["classifier_sample_gate"]),
            "--commit-scope",
            str(selected["commit_scope"]),
            "--support-domain",
            str(selected["support_domain"]),
            "--update-rule",
            str(selected["update_rule"]),
            "--weight-fractional-bits",
            str(selected.get("weight_fractional_bits", 8)),
            "--shift-x",
            str(selected.get("shift_x", 0)),
            "--shift-y",
            str(selected.get("shift_y", 0)),
            "--event-drop-probability",
            str(selected.get("event_drop_probability", 0.0)),
            "--polarity-flip-probability",
            str(selected.get("polarity_flip_probability", 0.0)),
            "--temporal-jitter",
            str(selected.get("temporal_jitter", 0)),
            "--corruption-seed",
            str(int(report_seed)),
            "--traces-root",
            str(args.traces_root),
            "--checkpoint",
            str(args.checkpoint),
            "--inputs",
            str(args.inputs),
            "--labels",
            str(args.labels),
            "--metadata",
            str(args.metadata),
            "--output",
            str(raw_path),
            "--device",
            args.device,
        ]
        if report_pairs:
            command.extend(
                [
                    "--subject-pairs",
                    *[f"{int(pair[0])}:{int(pair[1])}" for pair in report_pairs],
                ]
            )
        if "eval_batch_size" in selected:
            command.extend(
                ["--eval-batch-size", str(int(selected["eval_batch_size"]))]
            )
        if selected["update_rule"] == "fixed_sgd":
            command.extend(
                [
                    "--optimizer-lr",
                    str(selected["optimizer_lr"]),
                    "--fixed-sgd-residual-bits",
                    str(int(selected["fixed_sgd_residual_bits"])),
                ]
            )
            if bool(selected["full_adaptation_batches_only"]):
                command.append("--full-adaptation-batches-only")
        print("SCNN_SUBJECT_REPORT_RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    summary = summarize_report(
        selection=selection,
        selection_path=args.selection_summary,
        selected=selected,
        raw_paths=raw_paths,
        summary_path=summary_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
