#!/usr/bin/env python3
"""Audit storage-safe replay options against formal SHD M2 vectors.

The handoff vectors expose two Bx450 backbone spike paths and the Bx450
target-path trace consumed by the M2 TP distance engine at every timestep.
This tool checks whether that trace can be rebuilt from spikes and whether
samples with the same class can share one exact target trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "learning_reference" / "srnn_m2_replay_audit.json"
LEAK_Q14 = 13_926
SPIKE_Q14 = 16_384


def read_hex_values(path: Path, *, signed_bits: int | None = None) -> np.ndarray:
    """Read one hexadecimal value per line into an int64 array."""
    values = np.fromiter(
        (int(line, 16) for line in path.read_text(encoding="ascii").splitlines()),
        dtype=np.uint64,
    )
    if signed_bits is None:
        return values.astype(np.int64)
    sign = np.uint64(1 << (signed_bits - 1))
    modulus = 1 << signed_bits
    signed = values.astype(object)
    signed[values >= sign] -= modulus
    return signed.astype(np.int64)


def round_shift_even(values: np.ndarray | int, shift: int) -> np.ndarray:
    """Signed round-to-nearest-even followed by a right shift."""
    array = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return array.copy()
    if shift < 0:
        raise ValueError("shift must be nonnegative")
    negative = array < 0
    magnitude = np.abs(array)
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    rounded = quotient + increment.astype(np.int64)
    return np.where(negative, -rounded, rounded)


def advance_q14_trace(trace_q14: np.ndarray, spike_q14: np.ndarray) -> np.ndarray:
    return round_shift_even(trace_q14 * LEAK_Q14, 14) + spike_q14


def advance_fp32_trace(trace: np.ndarray, spike_q14: np.ndarray) -> np.ndarray:
    spike = (spike_q14 // SPIKE_Q14).astype(np.float32)
    return np.float32(np.float32(0.85) * trace + spike)


def quantize_fp32_q14(trace: np.ndarray) -> np.ndarray:
    return np.rint(trace.astype(np.float64) * SPIKE_Q14).astype(np.int64)


def class_dictionary_mismatches(
    labels: np.ndarray,
    target_spikes_q14: np.ndarray,
    target_trace_q14: np.ndarray,
) -> tuple[int, int]:
    spike_mismatches = 0
    trace_mismatches = 0
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        reference = members[0]
        spike_mismatches += int(np.count_nonzero(
            target_spikes_q14[members] != target_spikes_q14[reference]
        ))
        trace_mismatches += int(np.count_nonzero(
            target_trace_q14[members] != target_trace_q14[reference]
        ))
    return spike_mismatches, trace_mismatches


def _first_mismatch(
    timestep: int,
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, int] | None:
    mismatches = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    if not mismatches.size:
        return None
    flat_index = int(mismatches[0])
    return {
        "timestep": timestep,
        "sample": flat_index // expected.shape[1],
        "neuron": flat_index % expected.shape[1],
        "actual": int(actual.reshape(-1)[flat_index]),
        "expected": int(expected.reshape(-1)[flat_index]),
    }


def audit_prefix(
    vector_root: Path,
    prefix: str,
    *,
    batch: int = 64,
    backbone: int = 450,
) -> dict[str, Any]:
    actual_t_path = vector_root / f"{prefix}_actual_t.mem"
    labels_path = vector_root / f"{prefix}_labels.mem"
    if not actual_t_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"missing {prefix} metadata under {vector_root}")

    timesteps = int(actual_t_path.read_text(encoding="ascii").strip(), 16)
    labels = read_hex_values(labels_path)
    if labels.size != batch:
        raise ValueError(f"{prefix}: expected {batch} labels, found {labels.size}")

    classes = sorted(int(value) for value in np.unique(labels))
    q14_trace = np.zeros((batch, backbone), dtype=np.int64)
    fp32_trace = np.zeros((batch, backbone), dtype=np.float32)
    q14_mismatches = 0
    fp32_mismatches = 0
    domain_errors = 0
    class_spike_mismatches = 0
    class_trace_mismatches = 0
    first_q14_mismatch: dict[str, int] | None = None
    first_fp32_mismatch: dict[str, int] | None = None
    q14_residuals: Counter[int] = Counter()
    target_trace_min = 0
    target_trace_max = 0

    for timestep in range(timesteps):
        backbone_values = read_hex_values(
            vector_root / f"{prefix}_t{timestep:03d}_backbone_q14.mem",
            signed_bits=32,
        )
        expected_trace = read_hex_values(
            vector_root / f"{prefix}_t{timestep:03d}_target_q14.mem",
            signed_bits=32,
        )
        expected_backbone_values = 2 * batch * backbone
        if backbone_values.size != expected_backbone_values:
            raise ValueError(
                f"{prefix} t{timestep}: expected {expected_backbone_values} backbone values, "
                f"found {backbone_values.size}"
            )
        if expected_trace.size != batch * backbone:
            raise ValueError(
                f"{prefix} t{timestep}: expected {batch * backbone} target trace values, "
                f"found {expected_trace.size}"
            )

        backbone_paths = backbone_values.reshape(2 * batch, backbone)
        expected_trace = expected_trace.reshape(batch, backbone)
        target_spikes = backbone_paths[batch:]
        domain_errors += int(np.count_nonzero(
            (backbone_paths != 0) & (backbone_paths != SPIKE_Q14)
        ))

        q14_trace = advance_q14_trace(q14_trace, target_spikes)
        fp32_trace = advance_fp32_trace(fp32_trace, target_spikes)
        fp32_q14 = quantize_fp32_q14(fp32_trace)

        q14_delta = q14_trace - expected_trace
        q14_count = int(np.count_nonzero(q14_delta))
        fp32_count = int(np.count_nonzero(fp32_q14 - expected_trace))
        q14_mismatches += q14_count
        fp32_mismatches += fp32_count
        if first_q14_mismatch is None and q14_count:
            first_q14_mismatch = _first_mismatch(timestep, q14_trace, expected_trace)
        if first_fp32_mismatch is None and fp32_count:
            first_fp32_mismatch = _first_mismatch(timestep, fp32_q14, expected_trace)
        residual_values, residual_counts = np.unique(q14_delta, return_counts=True)
        q14_residuals.update({
            int(value): int(count)
            for value, count in zip(residual_values, residual_counts, strict=True)
            if value != 0
        })

        spike_mismatches, trace_mismatches = class_dictionary_mismatches(
            labels, target_spikes, expected_trace
        )
        class_spike_mismatches += spike_mismatches
        class_trace_mismatches += trace_mismatches
        target_trace_min = min(target_trace_min, int(expected_trace.min()))
        target_trace_max = max(target_trace_max, int(expected_trace.max()))

    trace_bits_unsigned = max(1, target_trace_max.bit_length())
    batch_trace_values = batch * timesteps * backbone
    class_trace_values = len(classes) * timesteps * backbone
    input_spike_bits = batch * timesteps * backbone
    class_target_spike_bits = len(classes) * timesteps * backbone
    replay_cache_bytes = len(classes) * backbone * 4
    q14_largest_error = max((abs(value) for value in q14_residuals), default=0)

    source_pass = domain_errors == 0 and fp32_mismatches == 0
    class_dictionary_pass = class_spike_mismatches == 0 and class_trace_mismatches == 0
    return {
        "prefix": prefix,
        "status": "pass" if source_pass and class_dictionary_pass else "fail",
        "shape": {
            "batch": batch,
            "timesteps": timesteps,
            "backbone_neurons": backbone,
            "classes_present": classes,
            "class_count": len(classes),
        },
        "source_domain": {
            "allowed_backbone_values_q14": [0, SPIKE_Q14],
            "invalid_value_count": domain_errors,
        },
        "q14_spike_only_rebuild": {
            "status": "not_bit_exact" if q14_mismatches else "bit_exact",
            "mismatch_count": q14_mismatches,
            "largest_absolute_error_lsb": q14_largest_error,
            "first_mismatch": first_q14_mismatch,
            "nonzero_residual_histogram": {
                str(value): count for value, count in sorted(q14_residuals.items())
            },
        },
        "original_fp32_rebuild": {
            "status": "bit_exact" if fp32_mismatches == 0 else "not_bit_exact",
            "mismatch_count": fp32_mismatches,
            "first_mismatch": first_fp32_mismatch,
        },
        "class_dictionary": {
            "status": "bit_exact" if class_dictionary_pass else "not_bit_exact",
            "same_class_spike_mismatch_count": class_spike_mismatches,
            "same_class_trace_mismatch_count": class_trace_mismatches,
            "current_timestep_cache_bytes_stored32": replay_cache_bytes,
        },
        "storage": {
            "batch_target_trace_tape_bytes_stored32": batch_trace_values * 4,
            "present_class_target_trace_tape_bytes_stored32": class_trace_values * 4,
            "present_class_target_trace_tape_bytes_min_packed": math.ceil(
                class_trace_values * trace_bits_unsigned / 8
            ),
            "target_trace_unsigned_bits_observed": trace_bits_unsigned,
            "target_trace_min_q14": target_trace_min,
            "target_trace_max_q14": target_trace_max,
            "input_spike_tape_bytes_bitpacked": math.ceil(input_spike_bits / 8),
            "present_class_target_spike_tape_bytes_bitpacked": math.ceil(
                class_target_spike_bits / 8
            ),
        },
        "architecture_conclusion": (
            "Do not rebuild the formal target trace with a Q14 recurrence. "
            "Generate or capture one exact target H450 spike/trace trajectory per class, "
            "store it in training scratch memory, and cache only the classes used by the "
            "current timestep during M2 replay."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--prefix", action="append", dest="prefixes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    prefixes = args.prefixes or ["V28", "V29"]
    audits = [audit_prefix(args.vector_root, prefix) for prefix in prefixes]
    result = {
        "schema_version": 1,
        "status": "pass" if all(item["status"] == "pass" for item in audits) else "fail",
        "evidence": "read-only audit of external formal SHD M2 vectors",
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "status": result["status"],
        "prefixes": [item["prefix"] for item in audits],
        "q14_mismatches": {
            item["prefix"]: item["q14_spike_only_rebuild"]["mismatch_count"]
            for item in audits
        },
        "class_dictionary": {
            item["prefix"]: item["class_dictionary"]["status"] for item in audits
        },
        "output": str(args.output),
    }, separators=(",", ":")))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
