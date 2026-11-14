#!/usr/bin/env python3
"""Deterministic self-test for summarize_fixed_shell_comparison.py."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import summarize_fixed_shell_comparison as target


def utilization_report(
    top: str,
    luts: int,
    registers: int,
    ramb36: int,
    ramb18: int | None,
    dsps: int,
) -> str:
    ramb18_row = "" if ramb18 is None else f"| RAMB18 | {ramb18} | 0 | 0 | 2940 | 0.00 |\n"
    return (
        "| Tool Version : Vivado v.2022.2 |\n"
        f"| Design       : {top}\n"
        "| Device       : xc7vx690tffg1761-3\n"
        "| Design State : Synthesized\n"
        f"| Slice LUTs* | {luts} | 0 | 0 | 433200 | 0.00 |\n"
        f"| Slice Registers | {registers} | 0 | 0 | 866400 | 0.00 |\n"
        f"| RAMB36/FIFO* | {ramb36} | 0 | 0 | 1470 | 0.00 |\n"
        f"{ramb18_row}"
        f"| DSPs | {dsps} | 0 | 0 | 3600 | 0.00 |\n"
    )


def timing_report(wns: float, whs: float) -> str:
    return (
        "Clock Waveform(ns) Period(ns) Frequency(MHz)\n"
        "----- ------------ ---------- --------------\n"
        "core_clk {0.000 7.500} 15.000 66.667\n\n"
        "WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints "
        "WHS(ns) THS(ns) THS Failing Endpoints THS Total Endpoints\n"
        "------- ------- --------------------- ------------------- "
        "------- ------- --------------------- -------------------\n"
        f"{wns:.3f} 0.000 0 100 {whs:.3f} 0.000 0 100\n"
    )


def write_run(
    root: Path,
    name: str,
    resources: tuple[int, int, int, int | None, int],
    timing: tuple[float, float],
    *,
    top: str = "fixed_shell_top",
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    run_dir.joinpath("utilization_post_synth.rpt").write_text(
        utilization_report(top, *resources), encoding="utf-8"
    )
    run_dir.joinpath("timing_summary_post_synth.rpt").write_text(
        timing_report(*timing), encoding="utf-8"
    )
    run_dir.joinpath("vivado_status.txt").write_text(
        "\n".join(
            (
                "status=pass",
                f"top={top}",
                "part=xc7vx690tffg1761-3",
                "period_ns=15.000",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


class FixedShellComparisonTest(unittest.TestCase):
    def test_formulas_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = {
                "Unified": write_run(root, "unified", (100, 200, 10, 4, 20), (1.0, 0.1)),
                "FC-only": write_run(root, "fc", (60, 100, 6, 2, 12), (1.1, 0.2)),
                "SRNN-only": write_run(root, "srnn", (70, 120, 5, 4, 15), (1.2, 0.3)),
                "SCNN-only": write_run(root, "scnn", (80, 140, 8, 2, 16), (1.3, 0.4)),
            }

            summary = target.summarize_comparison(run_dirs)
            self.assertEqual(summary.overall_comparability, "PASS")
            self.assertEqual(target.parse_clock_period(timing_report(1.0, 0.1)), 15.0)
            self.assertEqual(summary.runs["Unified"].metrics["bram36eq"], 12.0)
            self.assertEqual(summary.derived["slice_luts"].independent_sum, 210.0)
            self.assertEqual(summary.derived["slice_luts"].avoided, 110.0)
            self.assertAlmostEqual(
                summary.derived["slice_luts"].saving, 1.0 - 100.0 / 210.0
            )
            self.assertAlmostEqual(
                summary.derived["slice_luts"].reuse_factor, 2.1
            )
            self.assertAlmostEqual(
                summary.derived["slice_luts"].unified_premium, 0.25
            )
            self.assertEqual(summary.derived["bram36eq"].independent_sum, 23.0)

            csv_path = root / "comparison.csv"
            markdown_path = root / "comparison.md"
            target.write_csv(summary, csv_path)
            target.write_markdown(summary, markdown_path)

            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            timing_row = next(row for row in rows if row["Metric"] == "WNS")
            self.assertEqual(timing_row["IndependentSum"], "NA")
            self.assertEqual(timing_row["Unified"], "1")
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Overall comparability: **PASS**", markdown)
            self.assertIn("No weighted area score", markdown)

    def test_missing_metric_is_na_and_metadata_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = {
                "Unified": write_run(root, "unified", (100, 200, 10, 4, 20), (1.0, 0.1)),
                "FC-only": write_run(root, "fc", (60, 100, 6, 2, 12), (1.1, 0.2)),
                "SRNN-only": write_run(root, "srnn", (70, 120, 5, 4, 15), (1.2, 0.3)),
                "SCNN-only": write_run(
                    root,
                    "scnn",
                    (80, 140, 8, None, 16),
                    (1.3, 0.4),
                    top="mismatched_top",
                ),
            }

            summary = target.summarize_comparison(run_dirs)
            self.assertEqual(summary.overall_comparability, "FAIL")
            top_check = next(
                check for check in summary.comparability if check.name == "Top"
            )
            self.assertEqual(top_check.status, "FAIL")
            self.assertIsNone(summary.runs["SCNN-only"].metrics["ramb18"])
            self.assertIsNone(summary.runs["SCNN-only"].metrics["bram36eq"])
            self.assertIsNone(summary.derived["ramb18"].independent_sum)
            self.assertIsNone(summary.derived["bram36eq"].saving)

            csv_path = root / "comparison.csv"
            target.write_csv(summary, csv_path)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            ramb18_row = next(row for row in rows if row["Metric"] == "RAMB18")
            self.assertEqual(ramb18_row["SCNN-only"], "NA")
            self.assertEqual(ramb18_row["IndependentSum"], "NA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
