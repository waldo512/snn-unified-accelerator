#!/usr/bin/env python3
"""Build a bounded real DVS-Gesture B4xT4 learn/recompute fixture.

The production Q0.15 learner retains sub-LSB updates in a signed residual.
Starting from the complete code/residual state produced by eleven prior
identical real B4xT4 windows gives the RTL test one bounded twelfth physical
forward/learn/commit transaction followed by a genuinely changed forward.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/paper_evidence/rtl/scnn_q015_dataset_parity"
BASE_FIXTURE = SOURCE / "b4t4_update_fixture"
OUTPUT = (
    ROOT
    / "results/paper_evidence/rtl"
    / "scnn_q015_real_b4t4_forward_learn_recompute_20260928"
    / "fixture"
)
WEIGHTS = ROOT / "models/scnn_dvsgesture_vgg9_q0_15/bank_12.hex"

BATCH = 4
TIMESTEPS = 4
FEATURES = 2048
CLASSES = 11
PARAMS = FEATURES * CLASSES
RESIDUAL_BITS = 15
WARM_WINDOWS = 11


def read_signed_hex(path: Path, expected: int) -> np.ndarray:
    words = np.fromiter(
        (int(line, 16) for line in path.read_text(encoding="ascii").splitlines()
         if line.strip()),
        dtype=np.uint16,
    )
    if words.size != expected:
        raise ValueError(f"{path}: got {words.size}, expected {expected}")
    return words.view(np.int16).astype(np.int64)


def write_signed_hex(path: Path, values: np.ndarray) -> None:
    words = np.asarray(values, dtype=np.int16).view(np.uint16).reshape(-1)
    path.write_text(
        "".join(f"{int(word):04x}\n" for word in words), encoding="ascii"
    )


def apply_delta(
    codes: np.ndarray, residuals: np.ndarray, delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = residuals + delta
    magnitude = np.abs(total)
    code_delta = magnitude >> RESIDUAL_BITS
    code_delta = np.where(total < 0, -code_delta, code_delta)
    residual_magnitude = magnitude & ((1 << RESIDUAL_BITS) - 1)
    next_residual = np.where(total < 0, -residual_magnitude, residual_magnitude)
    next_code = codes + code_delta
    if np.any((next_code < -32768) | (next_code > 32767)):
        raise ValueError("warm-state update saturates a classifier code")
    return next_code, next_residual


def classifier_currents(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    raw = features.astype(np.int64) @ weights.astype(np.int64).T
    return np.clip(raw >> 15, -32768, 32767).astype(np.int64)


def temporal_center(currents: np.ndarray) -> np.ndarray:
    result = np.zeros_like(currents)
    for sample in range(BATCH):
        cumulative = np.zeros(CLASSES, dtype=np.int64)
        for timestep in range(TIMESTEPS):
            cumulative += currents[sample, timestep]
            centered = cumulative - cumulative.max()
            result[sample, timestep] = np.clip(centered, -32768, 32767)
    return result


def main() -> None:
    features = read_signed_hex(
        BASE_FIXTURE / "features_q8_8.hex", BATCH * TIMESTEPS * FEATURES
    ).reshape(BATCH, TIMESTEPS, FEATURES)
    baseline_fixture_logits = read_signed_hex(
        BASE_FIXTURE / "temporal_logits_q8_8.hex", BATCH * TIMESTEPS * CLASSES
    ).reshape(BATCH, TIMESTEPS, CLASSES)
    delta = read_signed_hex(
        BASE_FIXTURE / "expected_residual_q15.hex", PARAMS
    )
    weights = read_signed_hex(WEIGHTS, PARAMS).reshape(CLASSES, FEATURES)

    baseline_currents = classifier_currents(features, weights)
    baseline_logits = temporal_center(baseline_currents)
    if not np.array_equal(baseline_logits, baseline_fixture_logits):
        mismatch = int(np.count_nonzero(baseline_logits != baseline_fixture_logits))
        raise AssertionError(f"baseline temporal fixture mismatch: {mismatch}")

    warm_code_delta = np.zeros(PARAMS, dtype=np.int64)
    warm_residuals = np.zeros(PARAMS, dtype=np.int64)
    for _ in range(WARM_WINDOWS):
        warm_code_delta, warm_residuals = apply_delta(
            warm_code_delta, warm_residuals, delta
        )
    warm_code = weights.reshape(-1) + warm_code_delta
    if np.any((warm_code < -32768) | (warm_code > 32767)):
        raise AssertionError("warm real classifier weight exceeds int16")
    warm_currents = classifier_currents(
        features, warm_code.reshape(CLASSES, FEATURES)
    )
    warm_logits = temporal_center(warm_currents)
    if not np.array_equal(warm_currents, baseline_currents):
        raise AssertionError("eleven-window warm state already changes forward data")

    expected_code_delta, expected_residual = apply_delta(
        warm_code_delta, warm_residuals, delta
    )
    expected_code = weights.reshape(-1) + expected_code_delta
    if np.any((expected_code < -32768) | (expected_code > 32767)):
        raise AssertionError("updated real classifier weight exceeds int16")
    changed_code_count = int(np.count_nonzero(expected_code != warm_code))
    if changed_code_count == 0:
        raise AssertionError("twelfth window produced no visible code update")

    updated_weights = expected_code.reshape(CLASSES, FEATURES)
    updated_currents = classifier_currents(features, updated_weights)
    updated_logits = temporal_center(updated_currents)
    changed_current_count = int(np.count_nonzero(updated_currents != warm_currents))
    changed_logit_count = int(np.count_nonzero(updated_logits != warm_logits))
    if changed_current_count == 0 or changed_logit_count == 0:
        raise AssertionError("visible code update did not change recomputed forward data")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_signed_hex(OUTPUT / "warm_code_after_11_windows_q15.hex", warm_code)
    write_signed_hex(OUTPUT / "warm_residual_after_11_windows_q15.hex", warm_residuals)
    write_signed_hex(OUTPUT / "expected_code_after_commit_q15.hex", expected_code)
    write_signed_hex(OUTPUT / "expected_residual_after_commit_q15.hex", expected_residual)
    write_signed_hex(OUTPUT / "baseline_current_q8_8.hex", warm_currents)
    write_signed_hex(OUTPUT / "baseline_temporal_logits_q8_8.hex", warm_logits)
    write_signed_hex(OUTPUT / "updated_current_q8_8.hex", updated_currents)
    write_signed_hex(OUTPUT / "updated_temporal_logits_q8_8.hex", updated_logits)

    summary = {
        "scope": (
            "One real DVS-Gesture classifier-boundary B4xT4 transaction from a "
            "legitimate eleven-window persistent code/residual warm state; not full "
            "Conv1-to-Conv8 RTL parity or dataset accuracy."
        ),
        "batch": BATCH,
        "timesteps": TIMESTEPS,
        "features": FEATURES,
        "classes": CLASSES,
        "warm_windows": WARM_WINDOWS,
        "transaction_window": WARM_WINDOWS + 1,
        "changed_classifier_codes": changed_code_count,
        "changed_forward_currents": changed_current_count,
        "changed_temporal_logits": changed_logit_count,
        "warm_code_changes_from_pretrained": int(np.count_nonzero(warm_code_delta)),
        "source_feature_fixture": str(BASE_FIXTURE / "features_q8_8.hex"),
        "source_weight_bank": str(WEIGHTS),
    }
    (OUTPUT / "contract.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
