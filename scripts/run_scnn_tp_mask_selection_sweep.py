#!/usr/bin/env python3
"""Run reproducible SCNN mask/threshold selection without reading report metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "run_scnn_tp_selected_layer_shift_adaptation.py"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "paper_evidence"
    / "algorithm"
    / "mask_ablation_20260919"
    / "scnn_independent_selection"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(38472, 38477)))
    parser.add_argument("--thresholds", nargs="+", type=int, default=[28, 32, 36, 40, 44])
    parser.add_argument(
        "--conv-thresholds",
        nargs="+",
        type=int,
        help="independent Conv vote thresholds; defaults to --thresholds",
    )
    parser.add_argument(
        "--classifier-thresholds",
        nargs="+",
        type=int,
        help="independent classifier vote thresholds; defaults to --thresholds",
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[2, 4])
    parser.add_argument(
        "--masks",
        nargs="+",
        choices=("classifier", "conv8", "conv8_classifier"),
        default=["classifier", "conv8", "conv8_classifier"],
    )
    parser.add_argument(
        "--classifier-row-masks",
        nargs="+",
        choices=("all", "target", "target_competitor"),
        default=["all"],
    )
    parser.add_argument(
        "--classifier-sample-gates",
        nargs="+",
        choices=("all", "misclassified"),
        default=["all"],
    )
    parser.add_argument(
        "--commit-scopes",
        nargs="+",
        choices=("window", "support"),
        default=["window"],
    )
    parser.add_argument(
        "--support-domains",
        nargs="+",
        choices=("shifted", "mixed"),
        default=["shifted"],
    )
    parser.add_argument(
        "--update-rules",
        nargs="+",
        choices=("direct", "sgd", "adam"),
        default=["direct"],
    )
    parser.add_argument("--optimizer-lrs", nargs="+", type=float, default=[1.0e-4])
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--selection-per-class", type=int, default=3)
    parser.add_argument("--shift-x", type=int, default=8)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--traces-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    *,
    seed: int,
    threshold: int,
    conv_threshold: int,
    classifier_threshold: int,
    window: int,
    classifier_row_mask: str,
    classifier_sample_gate: str,
    commit_scope: str,
    support_domain: str,
    update_rule: str,
    optimizer_lr: float,
) -> Path:
    result_path = (
        args.output_dir
        / "raw"
        / (
            f"seed{seed}_ct{conv_threshold}_ft{classifier_threshold}"
            f"_w{window}_r{classifier_row_mask}"
            f"_g{classifier_sample_gate}_c{commit_scope}_d{support_domain}"
            f"_u{update_rule}_lr{optimizer_lr:g}.json"
        )
    )
    if result_path.exists() and not args.force:
        return result_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--output",
        str(result_path),
        "--episode-seed",
        str(seed),
        "--support-per-class",
        str(args.support_per_class),
        "--selection-per-class",
        str(args.selection_per_class),
        "--evaluation-split",
        "selection",
        "--shift-x",
        str(args.shift_x),
        "--shift-y",
        str(args.shift_y),
        "--vote-threshold",
        str(threshold),
        "--conv-vote-threshold",
        str(conv_threshold),
        "--classifier-vote-threshold",
        str(classifier_threshold),
        "--window-timesteps",
        str(window),
        "--classifier-row-mask",
        classifier_row_mask,
        "--classifier-sample-gate",
        classifier_sample_gate,
        "--commit-scope",
        commit_scope,
        "--support-domain",
        support_domain,
        "--update-rule",
        update_rule,
        "--optimizer-lr",
        str(optimizer_lr),
        "--trainable-masks",
        *args.masks,
        "--device",
        args.device,
    ]
    for flag, value in (
        ("--traces-root", args.traces_root),
        ("--checkpoint", args.checkpoint),
        ("--inputs", args.inputs),
        ("--labels", args.labels),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    print(
        f"SCNN_SELECTION_RUN seed={seed} conv_threshold={conv_threshold} "
        f"classifier_threshold={classifier_threshold} window={window} "
        f"row_mask={classifier_row_mask} sample_gate={classifier_sample_gate} "
        f"commit_scope={commit_scope} support_domain={support_domain}",
        f"update_rule={update_rule} optimizer_lr={optimizer_lr:g}",
        flush=True,
    )
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return result_path


def summarize(paths: list[Path]) -> dict[str, object]:
    grouped: dict[
        tuple[str, int, int, int, str, str, str, str, str, float],
        list[dict[str, object]],
    ] = defaultdict(list)
    configurations: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["configuration"]
        configurations.append(config)
        for mask in payload["masks"]:
            key = (
                str(mask["trainable_mask"]),
                int(config["conv_vote_threshold"]),
                int(config["classifier_vote_threshold"]),
                int(config["window_timesteps"]),
                str(config["classifier_row_mask"]),
                str(config["classifier_sample_gate"]),
                str(config["commit_scope"]),
                str(config["support_domain"]),
                str(config["update_rule"]),
                float(config["optimizer_lr"] or 0.0),
            )
            grouped[key].append(mask)

    rows: list[dict[str, object]] = []
    for (
        mask_name,
        conv_threshold,
        classifier_threshold,
        window,
        row_mask,
        sample_gate,
        commit_scope,
        support_domain,
        update_rule,
        optimizer_lr,
    ), episodes in sorted(grouped.items()):
        shifted = np.asarray(
            [float(item["selection_shifted_gain_points"]) for item in episodes]
        )
        retention = np.asarray(
            [float(item["selection_clean_retention_points"]) for item in episodes]
        )
        writes = np.asarray([
            sum(int(value) for value in item["update"]["code_writes"].values())
            for item in episodes
        ])
        realized = np.asarray([
            bool(item["update"].get("mask_realized", all(
                int(delta["nonzero"]) != 0
                for delta in item["update"]["parameter_deltas"].values()
                if bool(delta["selected"])
            )))
            for item in episodes
        ], dtype=bool)
        feasible = (shifted > 0.0) & (retention >= -1.0) & realized
        rows.append({
            "mask": mask_name,
            "conv_vote_threshold": conv_threshold,
            "classifier_vote_threshold": classifier_threshold,
            "window_timesteps": window,
            "classifier_row_mask": row_mask,
            "classifier_sample_gate": sample_gate,
            "commit_scope": commit_scope,
            "support_domain": support_domain,
            "update_rule": update_rule,
            "optimizer_lr": optimizer_lr if update_rule != "direct" else None,
            "episodes": len(episodes),
            "selection_shifted_gain_mean": float(shifted.mean()),
            "selection_shifted_gain_std": float(shifted.std(ddof=0)),
            "selection_clean_retention_mean": float(retention.mean()),
            "selection_clean_retention_std": float(retention.std(ddof=0)),
            "code_writes_mean": float(writes.mean()),
            "mask_realized_episodes": int(realized.sum()),
            "all_episodes_mask_realized": bool(realized.all()),
            "feasible_episodes": int(feasible.sum()),
            "all_episodes_feasible": bool(feasible.all()),
            "mutable_parameters": int(episodes[0]["update"]["mutable_parameters"]),
            "optimizer_steps_mean": float(np.mean([
                int(item["update"]["optimizer_steps"]) for item in episodes
            ])),
        })

    feasible_rows = [
        row
        for row in rows
        if row["all_episodes_feasible"]
        and float(row["selection_shifted_gain_mean"]) > 0.0
    ]
    selected = None
    if feasible_rows:
        selected = sorted(
            feasible_rows,
            key=lambda row: (
                -float(row["selection_shifted_gain_mean"]),
                int(row["mutable_parameters"]),
                float(row["code_writes_mean"]),
            ),
        )[0]

    first = configurations[0]
    return {
        "status": "pass",
        "claim_boundary": (
            "Hyperparameter and trainable-mask selection uses only disjoint "
            "selection metrics. No report-partition metric is read or stored "
            "by this sweep. Final report evaluation must be a separate run."
        ),
        "protocol": {
            "seeds": sorted({int(item["episode_seed"]) for item in configurations}),
            "support_per_class": int(first["support_per_class"]),
            "selection_per_class": int(first["selection_per_class"]),
            "shift_x": int(first["shift_x"]),
            "shift_y": int(first["shift_y"]),
            "conv_vote_thresholds": sorted(
                {int(item["conv_vote_threshold"]) for item in configurations}
            ),
            "classifier_vote_thresholds": sorted(
                {int(item["classifier_vote_threshold"]) for item in configurations}
            ),
            "classifier_row_masks": sorted(
                {str(item["classifier_row_mask"]) for item in configurations}
            ),
            "classifier_sample_gates": sorted(
                {str(item["classifier_sample_gate"]) for item in configurations}
            ),
            "commit_scopes": sorted(
                {str(item["commit_scope"]) for item in configurations}
            ),
            "support_domains": sorted(
                {str(item["support_domain"]) for item in configurations}
            ),
            "update_rules": sorted(
                {str(item["update_rule"]) for item in configurations}
            ),
            "feasibility": "shifted gain > 0 and clean retention >= -1 point in every episode",
        },
        "rows": rows,
        "selected": selected,
    }


def main() -> None:
    args = parse_args()
    conv_thresholds = args.conv_thresholds or args.thresholds
    classifier_thresholds = args.classifier_thresholds or args.thresholds
    if any(
        value <= 0
        for value in (*conv_thresholds, *classifier_thresholds, *args.windows)
    ):
        raise ValueError("thresholds and windows must be positive")
    paths: list[Path] = []
    for seed in args.seeds:
        for conv_threshold in conv_thresholds:
            for classifier_threshold in classifier_thresholds:
                threshold = classifier_threshold
                for window in args.windows:
                    for classifier_row_mask in args.classifier_row_masks:
                        for classifier_sample_gate in args.classifier_sample_gates:
                            for commit_scope in args.commit_scopes:
                                for support_domain in args.support_domains:
                                    for update_rule in args.update_rules:
                                        for optimizer_lr in args.optimizer_lrs:
                                            paths.append(
                                                run_one(
                                                    args,
                                                    seed=seed,
                                                    threshold=threshold,
                                                    conv_threshold=conv_threshold,
                                                    classifier_threshold=classifier_threshold,
                                                    window=window,
                                                    classifier_row_mask=classifier_row_mask,
                                                    classifier_sample_gate=classifier_sample_gate,
                                                    commit_scope=commit_scope,
                                                    support_domain=support_domain,
                                                    update_rule=update_rule,
                                                    optimizer_lr=optimizer_lr,
                                                )
                                            )
    summary = summarize(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
