#!/usr/bin/env python3
"""Build the paper-facing table of locked algorithm and hardware profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = ROOT / "results" / "paper_evidence" / "algorithm"
MASK_BASE = ALGORITHM / "mask_ablation_20260919"
OUTPUT = ALGORITHM / "final_algorithm_profiles_20260927"

SOURCES = {
    "pareto": ALGORITHM / "selective_adaptation_pareto_20260927" / "summary.json",
    "statistics": ALGORITHM / "online_adaptation_statistics_20260927" / "summary.json",
    "fc": (
        MASK_BASE
        / "fc_h160_w2_int8_magnitude_uq6_10_official_test_report_20260924"
        / "report_summary.json"
    ),
    "srnn": MASK_BASE / "srnn_wout_postlock_replication_20260927" / "summary.json",
    "scnn": MASK_BASE / "scnn_q1_15_postlock_replication_20260927" / "summary.json",
    "srnn_support": (
        MASK_BASE / "srnn_wout_support_sensitivity_20260927" / "summary.json"
    ),
    "scnn_window": (
        MASK_BASE / "scnn_q1_15_window_sensitivity_20260927" / "summary.json"
    ),
}


def load(name: str) -> dict[str, Any]:
    path = SOURCES[name]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"source evidence is not complete: {path}")
    return payload


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pareto = load("pareto")
    statistics = load("statistics")
    fc = load("fc")
    srnn = load("srnn")
    scnn = load("scnn")
    srnn_support = load("srnn_support")
    scnn_window = load("scnn_window")

    pareto_by_topology = {row["topology"]: row for row in pareto["rows"]}
    stats_by_topology = {
        row["topology"]: row for row in statistics["topologies"]
    }

    profiles = {
        "FC": {
            "numeric_contract": (
                "INT8 forward code; UQ6.10 trace; Q0.15 residual; "
                "delta=-RNE(G*11/2^26)"
            ),
            "learning_rule": "INT8 magnitude-residual update",
            "runtime_point": "batch=16; support=16 samples/class",
            "hardware_decision": (
                "Keep W0 immutable and update W2 only; W0+W2 is retained as an "
                "algorithm ablation rather than the publication hardware profile."
            ),
            "limitation": (
                "N-MNIST only; synthetic spatial shifts; five support-seed clusters."
            ),
            "profile_source": relative(SOURCES["fc"]),
            "profile": fc["locked_candidate"],
        },
        "SRNN": {
            "numeric_contract": "signed Q2.14 Wout code/residual",
            "learning_rule": "direct-response update (increment=31, threshold=65)",
            "runtime_point": "support=128 labelled samples; shifts=+/-16 channels",
            "hardware_decision": (
                "Keep the pre-locked 128-sample report point. Support=64 is a "
                "lower-work option and support=256 is an accuracy-oriented option; "
                "official-test sensitivity is not used for post-hoc reselection."
            ),
            "limitation": (
                "SHD only; frozen 700x450 recurrent backbone; synthetic frequency "
                "shifts; one regressing episode remains."
            ),
            "profile_source": relative(SOURCES["srnn"]),
            "profile": srnn["candidate"],
        },
        "SCNN": {
            "numeric_contract": (
                "Q8.8 pooled observation; Q0.15 classifier code/residual; "
                "shared-port multiplier=640"
            ),
            "learning_rule": "fixed-SGD magnitude-residual update",
            "runtime_point": "batch=4; temporal window T=4",
            "hardware_decision": (
                "Keep classifier-only as the main profile. T=4 is the engineering "
                "balance: it matches T=2/T=8 outcomes in the bounded sensitivity "
                "while halving update windows versus T=2 without T=8 residency."
            ),
            "limitation": (
                "DVS-Gesture only; bounded synthetic corruptions; Conv8's zero "
                "incremental gain does not exclude longer-horizon or independently "
                "scaled deeper-layer adaptation."
            ),
            "profile_source": relative(SOURCES["scnn"]),
            "profile": scnn["candidate"],
        },
    }

    rows: list[dict[str, Any]] = []
    for topology in ("FC", "SRNN", "SCNN"):
        cost = pareto_by_topology[topology]
        stats = stats_by_topology[topology]
        profile = profiles[topology]
        rows.append(
            {
                "topology": topology,
                "dataset": cost["dataset"],
                "model": cost["model"],
                "selected_mask": cost["selected_mask"],
                "numeric_contract": profile["numeric_contract"],
                "learning_rule": profile["learning_rule"],
                "runtime_point": profile["runtime_point"],
                "mutable_parameters": cost["selected_mutable_parameters"],
                "mutable_state_bytes": cost["selected_mutable_state_bytes"],
                "larger_mask": cost["excluded_larger_mask"],
                "larger_over_selected_state_ratio": cost[
                    "larger_over_selected_state_ratio"
                ],
                "evaluation_scope": stats["scope"],
                "episodes": stats["episodes"],
                "independent_seed_clusters": stats["clusters"],
                "gain_points_mean": stats["episode_gain_points"]["mean"],
                "gain_points_stdev": stats["episode_gain_points"]["stdev"],
                "cluster_ci95_low": stats["cluster_bootstrap"]["ci95_low"],
                "cluster_ci95_high": stats["cluster_bootstrap"]["ci95_high"],
                "positive_episodes": stats["positive_episodes"],
                "nondegrading_episodes": stats["nondegrading_episodes"],
                "clean_budget_episodes": stats["clean_loss_within_1pt_episodes"],
                "perturbations": cost["perturbations"],
                "hardware_decision": profile["hardware_decision"],
                "limitation": profile["limitation"],
                "profile_source": profile["profile_source"],
                "statistics_source": stats["source"],
                "cost_source": relative(SOURCES["pareto"]),
            }
        )

    sensitivity = {
        "srnn_support": {
            "locked_main_point": 128,
            "alternatives": srnn_support["by_support_size"],
            "paired_deltas_vs_128": srnn_support["paired_deltas_vs_128"],
            "decision": profiles["SRNN"]["hardware_decision"],
            "source": relative(SOURCES["srnn_support"]),
        },
        "scnn_window": {
            "locked_main_point": 4,
            "alternatives": scnn_window["by_window"],
            "paired_deltas_vs_window4": scnn_window[
                "paired_deltas_vs_window4"
            ],
            "decision": profiles["SCNN"]["hardware_decision"],
            "source": relative(SOURCES["scnn_window"]),
        },
    }
    return rows, sensitivity


def write_readme(rows: list[dict[str, Any]], sensitivity: dict[str, Any]) -> None:
    lines = [
        "# Final Algorithm Profiles",
        "",
        "This directory is the compact paper-facing view of the locked FC, SRNN and",
        "SCNN selective online-adaptation profiles. It is generated only from the",
        "authoritative passed evidence listed in `summary.json`; it does not rerun or",
        "reinterpret the experiments.",
        "",
        "## Locked profiles",
        "",
        "| Topology | Dataset/model | Main mutable mask | Numeric/update contract | Runtime point | Mutable state | Larger-mask ratio | Report evidence |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        result = (
            f"{row['gain_points_mean']:+.3f} +/- {row['gain_points_stdev']:.3f} pp; "
            f"CI [{row['cluster_ci95_low']:+.3f},{row['cluster_ci95_high']:+.3f}]; "
            f"{row['positive_episodes']}/{row['episodes']} positive, "
            f"{row['nondegrading_episodes']}/{row['episodes']} non-degrading"
        )
        lines.append(
            "| {topology} | {dataset}; {model} | {selected_mask} | "
            "{numeric_contract}; {learning_rule} | {runtime_point} | "
            "{mutable_state_bytes:,} B | {larger_over_selected_state_ratio:.2f}x | "
            "{result} |".format(result=result, **row)
        )

    srnn = sensitivity["srnn_support"]["alternatives"]
    scnn = sensitivity["scnn_window"]["alternatives"]
    lines.extend(
        [
            "",
            "## Runtime sensitivity decisions",
            "",
            "- SRNN keeps support=128 as the pre-locked publication point. The matched",
            "  support sweep gives gains "
            f"{fmt(srnn['64']['shifted_gain_points']['mean'])}/"
            f"{fmt(srnn['128']['shifted_gain_points']['mean'])}/"
            f"{fmt(srnn['256']['shifted_gain_points']['mean'])} points at "
            "support=64/128/256. Relative to 128, mean RMW is 0.494x at 64 and",
            "  2.083x at 256. The sweep is reported as a runtime tradeoff, not used",
            "  to reselect the official-test operating point.",
            "- SCNN keeps T=4. In 48 matched development episodes per setting, T=2,",
            "  T=4 and T=8 have identical gain and clean-retention outcomes; mean",
            "  update windows are "
            f"{scnn['2']['mean_update_windows']:.0f}, "
            f"{scnn['4']['mean_update_windows']:.0f} and "
            f"{scnn['8']['mean_update_windows']:.0f}, respectively. T=1 is 0.047",
            "  points lower than T=4 on average.",
            "",
            "## Claim boundary",
            "",
            "The three rows use different datasets, perturbations, model structures and",
            "evaluation populations. They establish topology-specific feasibility and",
            "hardware-driving mask choices; they must not be used to rank one topology",
            "against another. The confidence intervals resample seed-cluster means and",
            "remain limited by small cluster counts. The table does not prove global mask",
            "optimality, natural-domain generalization, board operation or measured power.",
            "",
            "`summary.csv` is the manuscript-table input. `summary.json` additionally",
            "retains source paths, sensitivity results and decision text for audit.",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows, sensitivity = build_rows()
    payload = {
        "status": "pass",
        "claim_boundary": (
            "Paper-facing join of already accepted topology-specific algorithm evidence. "
            "Rows are not a cross-topology accuracy ranking and do not establish global "
            "mask optimality or natural-domain generalization."
        ),
        "profiles": rows,
        "runtime_sensitivity": sensitivity,
        "source_files": {name: relative(path) for name, path in SOURCES.items()},
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_readme(rows, sensitivity)
    print(f"wrote {len(rows)} locked profiles to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
