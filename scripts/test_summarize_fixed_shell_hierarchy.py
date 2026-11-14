#!/usr/bin/env python3
"""Focused tests for summarize_fixed_shell_hierarchy.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import summarize_fixed_shell_hierarchy as target


CLIENTS = {
    "fc": "fixed_shell_fc_client_v1",
    "srnn": "fixed_shell_srnn_client_v1",
    "scnn": "fixed_shell_scnn_client_v1",
    "fc_scnn": "fixed_shell_fc_scnn_shared_client_v1",
}


def hierarchy_row(depth: int, instance: str, module: str, value: int = 0) -> str:
    fields = [
        " " * (1 + 2 * depth) + instance,
        f" {module} ",
        *[f" {value} " for _ in range(8)],
    ]
    return "|" + "|".join(fields) + "|\n"


def hierarchy_report(mask: int, *, tile_count: int = 12, leak_srnn: bool = False) -> str:
    enabled = {
        CLIENTS["fc"]: bool(mask & 1),
        CLIENTS["srnn"]: bool(mask & 2),
        CLIENTS["scnn"]: bool(mask & 4),
        CLIENTS["fc_scnn"]: bool(mask & (1 | 4)),
    }
    lines = [
        "| Instance | Module | Total LUTs | Logic LUTs | LUTRAMs | SRLs | "
        "FFs | RAMB36 | RAMB18 | DSP Blocks |\n",
        hierarchy_row(0, "fixed_shell_top", "(top)"),
        hierarchy_row(
            1,
            "u_array",
            target.PHYSICAL_ARRAY_MODULE + "_91",
            100,
        ),
        hierarchy_row(
            2,
            "(u_array)",
            target.PHYSICAL_ARRAY_MODULE + "_91",
            1,
        ),
    ]
    for index in range(tile_count):
        suffix = "" if index == 0 else f"_{100 + index}"
        lines.append(
            hierarchy_row(
                2,
                f"g_tile[{index}].u_tile",
                target.H40_TILE_MODULE + suffix,
                10,
            )
        )
        lines.append(
            hierarchy_row(
                3,
                f"(g_tile[{index}].u_tile)",
                target.H40_TILE_MODULE + suffix,
                1,
            )
        )
    for index, (module, present) in enumerate(enabled.items()):
        if present or (leak_srnn and module == CLIENTS["srnn"]):
            elaborated = (
                module + "__parameterized3_44" if index == 0 else module + f"_{index}"
            )
            lines.append(hierarchy_row(1, f"u_client_{index}", elaborated, 5))
            lines.append(hierarchy_row(2, f"(u_client_{index})", elaborated, 1))
    return "".join(lines)


def expected_payload() -> dict[str, object]:
    return {
        "schema": "fixed_shell_expected_clients_v1",
        "clients": {
            "fc": [CLIENTS["fc"], CLIENTS["fc_scnn"]],
            "srnn": [CLIENTS["srnn"]],
            "scnn": [CLIENTS["scnn"], CLIENTS["fc_scnn"]],
        },
    }


def write_reports(root: Path, **overrides: str) -> dict[int, Path]:
    reports: dict[int, Path] = {}
    for mask in target.MASK_ORDER:
        run_dir = root / f"mask{mask}"
        run_dir.mkdir()
        report = run_dir / "utilization_hierarchical_post_synth.rpt"
        report.write_text(
            overrides.get(str(mask), hierarchy_report(mask)), encoding="utf-8"
        )
        reports[mask] = run_dir
    return reports


class FixedShellHierarchyTest(unittest.TestCase):
    def test_normalizes_only_recognized_terminal_suffixes(self) -> None:
        self.assertEqual(
            target.normalize_module_name("client_v1__parameterized3_617"),
            "client_v1",
        )
        self.assertEqual(
            target.normalize_module_name("client_v1__parameterized5__1"),
            "client_v1",
        )
        self.assertEqual(target.normalize_module_name("client_v1_406"), "client_v1")
        self.assertEqual(target.normalize_module_name("client_v1"), "client_v1")
        self.assertEqual(
            target.normalize_module_name("client_v1_variant"),
            "client_v1_variant",
        )

    def test_parser_excludes_parenthesized_self_rows(self) -> None:
        parsed = target.parse_hierarchy_report(hierarchy_report(7))
        counts = {
            module: sum(row.normalized_module == module for row in parsed.rows)
            for module in (target.PHYSICAL_ARRAY_MODULE, target.H40_TILE_MODULE)
        }
        self.assertEqual(counts[target.PHYSICAL_ARRAY_MODULE], 1)
        self.assertEqual(counts[target.H40_TILE_MODULE], 12)
        self.assertEqual(parsed.parenthesized_self_rows_excluded, 17)
        tile_paths = [
            row.path
            for row in parsed.rows
            if row.normalized_module == target.H40_TILE_MODULE
        ]
        self.assertTrue(all(path.startswith("fixed_shell_top/u_array/") for path in tile_paths))

    def test_all_masks_validate_and_emit_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = write_reports(root)
            expectations = target.parse_expected_clients(expected_payload())
            summary = target.summarize_hierarchies(reports, expectations)

            self.assertEqual(summary["status"], "pass")
            masks = {row["mask"]: row for row in summary["masks"]}
            self.assertEqual(masks[7]["counts"]["physical_arrays"], 1)
            self.assertEqual(masks[1]["counts"]["h40_tiles"], 12)
            self.assertTrue(masks[1]["checks"]["all_forbidden_clients_absent"])
            self.assertEqual(
                next(
                    client["count"]
                    for client in masks[1]["required_clients"]
                    if client["module"] == CLIENTS["fc"]
                ),
                1,
            )
            self.assertIn(
                CLIENTS["fc_scnn"],
                [client["module"] for client in masks[4]["required_clients"]],
            )

            json_path = root / "summary.json"
            markdown_path = root / "summary.md"
            target.write_json(summary, json_path)
            target.write_markdown(summary, markdown_path)
            encoded = json.loads(json_path.read_text(encoding="ascii"))
            self.assertEqual(encoded["schema"], "fixed_shell_hierarchy_summary_v1")
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Overall hierarchy validation: **PASS**", markdown)
            self.assertIn("| 7 | Unified | 1 | 12 |", markdown)

    def test_missing_tile_and_forbidden_client_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = write_reports(
                root,
                **{"1": hierarchy_report(1, tile_count=11, leak_srnn=True)},
            )
            summary = target.summarize_hierarchies(
                reports, target.parse_expected_clients(expected_payload())
            )
            self.assertEqual(summary["status"], "fail")
            mask1 = next(row for row in summary["masks"] if row["mask"] == 1)
            self.assertFalse(mask1["checks"]["exactly_twelve_h40_tiles"])
            self.assertFalse(mask1["checks"]["all_forbidden_clients_absent"])
            leaked = next(
                client
                for client in mask1["forbidden_clients"]
                if client["module"] == CLIENTS["srnn"]
            )
            self.assertEqual(leaked["count"], 1)
            self.assertTrue(leaked["paths"])

    def test_explicit_mask_expectations_derive_forbidden_union(self) -> None:
        payload = {
            "masks": {
                "7": [CLIENTS["fc"], CLIENTS["srnn"], CLIENTS["scnn"]],
                "1": [CLIENTS["fc"]],
                "2": [CLIENTS["srnn"]],
                "4": [CLIENTS["scnn"]],
            }
        }
        expectations = target.parse_expected_clients(payload)
        self.assertEqual(expectations[1].required, (CLIENTS["fc"],))
        self.assertEqual(
            set(expectations[1].forbidden),
            {CLIENTS["srnn"], CLIENTS["scnn"]},
        )

    def test_malformed_expectations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing topology client groups"):
            target.parse_expected_clients(
                {"clients": {"fc": [CLIENTS["fc"]], "srnn": []}}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
