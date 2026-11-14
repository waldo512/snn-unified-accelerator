#!/usr/bin/env python3
"""Build the cross-topology selective-adaptation evidence table."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "paper_evidence" / "algorithm" / "mask_ablation_20260919"
OUTPUT = ROOT / "results" / "paper_evidence" / "algorithm" / "selective_adaptation_pareto_20260927"

FC_REPORT = BASE / "fc_h160_w2_int8_magnitude_uq6_10_official_test_report_20260924" / "report_summary.json"
SRNN_REPORT = BASE / "srnn_wout_train_support_official_test_20260927" / "report_summary.json"
SCNN_REPORT = BASE / "scnn_q1_15_cross_corruption_report_20260927" / "summary.json"
SCNN_JITTER = BASE / "scnn_event_fixed_sgd_q1_15_report_subjects_20260924" / "report_summary.json"
SCNN_MASK_COST = BASE / "scnn_q1_15_mask_cost_benefit_20260927" / "summary.json"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"source evidence is not complete: {path}")
    return payload


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> int:
    fc = load(FC_REPORT)
    srnn = load(SRNN_REPORT)
    scnn = load(SCNN_REPORT)
    jitter = load(SCNN_JITTER)
    scnn_cost = load(SCNN_MASK_COST)

    scnn_rows = list(scnn["episodes"])
    scnn_rows.extend(
        {
            "corrupted_gain_points": float(row["shifted_gain_points"]),
            "clean_change_points": float(row["clean_retention_points"]),
            "code_writes": int(row["code_writes"]),
        }
        for row in jitter["episodes_detail"]
    )

    fc_selected = int(fc["locked_candidate"]["mutable_parameters"])
    srnn_selected = int(srnn["locked_candidate"]["mutable_parameters"])
    scnn_selected = int(scnn["protocol"]["mutable_parameters"])
    scnn_joint = int(scnn_cost["by_mask"]["conv8_classifier"]["mutable_parameters"])

    rows = [
        {
            "topology": "FC",
            "dataset": "N-MNIST",
            "model": "2312-160-10",
            "selected_mask": "W2",
            "excluded_larger_mask": "W0+W2",
            "selected_mutable_parameters": fc_selected,
            "selected_mutable_state_bytes": int(fc["locked_candidate"]["mutable_state_bytes"]),
            "larger_mask_parameters": 371520,
            "larger_over_selected_state_ratio": 371520 / fc_selected,
            "report_episodes": int(fc["episodes"]),
            "perturbations": "spatial shifts (+2,0),(-2,0),(0,+2)",
            "adaptation_gain_points_mean": float(fc["pooled_shifted_gain_points"]["mean"]),
            "adaptation_gain_points_stdev": float(fc["pooled_shifted_gain_points"]["stdev"]),
            "clean_change_points_mean": float(fc["pooled_clean_retention_points"]["mean"]),
            "clean_change_points_stdev": float(fc["pooled_clean_retention_points"]["stdev"]),
            "positive_gain_episodes": int(fc["positive_gain_episodes"]),
            "nondegrading_gain_episodes": sum(
                float(row["shifted_gain_points"]) >= 0.0 for row in fc["episodes_detail"]
            ),
            "clean_loss_within_1pt_episodes": int(fc["clean_loss_within_1pt_episodes"]),
            "mean_code_writes_per_episode": mean(fc["episodes_detail"], "code_writes"),
            "mean_rmw_or_residual_writes_per_episode": mean(
                fc["episodes_detail"], "residual_writes"
            ),
            "source": str(FC_REPORT.relative_to(ROOT)),
        },
        {
            "topology": "SRNN",
            "dataset": "SHD",
            "model": "frozen 700x450 recurrent backbone + 450-256-20 M2",
            "selected_mask": "Wout",
            "excluded_larger_mask": "Wh1+Wout",
            "selected_mutable_parameters": srnn_selected,
            "selected_mutable_state_bytes": int(
                srnn["locked_candidate"]["mutable_code_residual_bytes"]
            ),
            "larger_mask_parameters": 120320,
            "larger_over_selected_state_ratio": 120320 / srnn_selected,
            "report_episodes": int(srnn["episodes"]),
            "perturbations": "frequency shifts +16 and -16",
            "adaptation_gain_points_mean": float(srnn["pooled_shifted_gain_points"]["mean"]),
            "adaptation_gain_points_stdev": float(srnn["pooled_shifted_gain_points"]["stdev"]),
            "clean_change_points_mean": float(srnn["pooled_clean_retention_points"]["mean"]),
            "clean_change_points_stdev": float(srnn["pooled_clean_retention_points"]["stdev"]),
            "positive_gain_episodes": int(srnn["positive_gain_episodes"]),
            "nondegrading_gain_episodes": sum(
                float(row["shifted_gain_points"]) >= 0.0 for row in srnn["episodes_detail"]
            ),
            "clean_loss_within_1pt_episodes": int(srnn["clean_loss_within_1pt_episodes"]),
            "mean_code_writes_per_episode": mean(srnn["episodes_detail"], "code_writes"),
            "mean_rmw_or_residual_writes_per_episode": mean(srnn["episodes_detail"], "rmw"),
            "source": str(SRNN_REPORT.relative_to(ROOT)),
        },
        {
            "topology": "SCNN",
            "dataset": "DVS-Gesture",
            "model": "eight-convolution VGG-9 SNN + 2048-11 classifier",
            "selected_mask": "classifier",
            "excluded_larger_mask": "Conv8+classifier",
            "selected_mutable_parameters": scnn_selected,
            "selected_mutable_state_bytes": int(scnn["protocol"]["mutable_state_bytes"]),
            "larger_mask_parameters": scnn_joint,
            "larger_over_selected_state_ratio": scnn_joint / scnn_selected,
            "report_episodes": len(scnn_rows),
            "perturbations": "jitter1, event-drop40, polarity-flip40",
            "adaptation_gain_points_mean": mean(scnn_rows, "corrupted_gain_points"),
            "adaptation_gain_points_stdev": statistics.pstdev(
                float(row["corrupted_gain_points"]) for row in scnn_rows
            ),
            "clean_change_points_mean": mean(scnn_rows, "clean_change_points"),
            "clean_change_points_stdev": statistics.pstdev(
                float(row["clean_change_points"]) for row in scnn_rows
            ),
            "positive_gain_episodes": sum(
                float(row["corrupted_gain_points"]) > 0.0 for row in scnn_rows
            ),
            "nondegrading_gain_episodes": sum(
                float(row["corrupted_gain_points"]) >= 0.0 for row in scnn_rows
            ),
            "clean_loss_within_1pt_episodes": sum(
                float(row["clean_change_points"]) >= -1.0 for row in scnn_rows
            ),
            "mean_code_writes_per_episode": mean(scnn_rows, "code_writes"),
            "mean_rmw_or_residual_writes_per_episode": None,
            "source": str(SCNN_REPORT.relative_to(ROOT)),
        },
    ]

    summary = {
        "status": "pass",
        "claim_boundary": (
            "Cross-topology summary of independently locked, topology-specific "
            "adaptation reports. Accuracy changes are not directly comparable across "
            "datasets or perturbations. The table supports selective mutable-state "
            "provisioning in one accelerator, not global mask optimality."
        ),
        "rows": rows,
        "architecture_consequence": {
            "shared_requirement": (
                "one configurable observation/scaling/contribution/RMW protocol with "
                "topology-specific numeric frontends"
            ),
            "selected_parameter_banks": ["FC W2", "SRNN Wout", "SCNN classifier"],
            "excluded_from_main_profile": ["FC W0", "SRNN Wh1", "SCNN Conv8"],
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Selective Online Adaptation Pareto Across Topologies",
        "",
        "Status: PASS on 2026-09-27.",
        "",
        "| Topology | Dataset | Selected / larger mask | Mutable state | State ratio | Gain (points) | Clean change | Positive / nondegrading |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {topology} | {dataset} | {selected_mask} / {excluded_larger_mask} | "
            "{selected_mutable_state_bytes:,} B | {larger_over_selected_state_ratio:.2f}x | "
            "+{adaptation_gain_points_mean:.3f} +/- {adaptation_gain_points_stdev:.3f} | "
            "{clean_change_points_mean:+.3f} +/- {clean_change_points_stdev:.3f} | "
            "{positive_gain_episodes}/{report_episodes} / "
            "{nondegrading_gain_episodes}/{report_episodes} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The ratios compare the selected mutable bank with the nearest larger mask",
            "evaluated during topology-specific selection: FC `W0+W2`, SRNN",
            "`Wh1+Wout`, and SCNN `Conv8+classifier`. The selected banks reduce mutable",
            "state by 232.20x, 23.50x, and 105.73x, respectively. These are parameter-",
            "state ratios, not whole-FPGA area reductions.",
            "",
            "The architectural consequence is the main result: the integrated",
            "accelerator needs configurable topology-specific observation and numeric",
            "frontends, but only three compact writable parameter banks feeding a common",
            "contribution/RMW protocol. FC W0, SRNN Wh1, and SCNN Conv8 remain optional",
            "capabilities rather than permanently provisioned main-profile state.",
            "",
            "Do not rank the three gain values against each other. They use different",
            "datasets, perturbations, support/query protocols, and model sizes. See",
            "`summary.json` for exact source paths and machine-readable values.",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
