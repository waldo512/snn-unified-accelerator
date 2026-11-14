#!/usr/bin/env python3
"""Compare FC H160 TP trainable masks from the RTL quantized checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fc_tp_h400_b16_reference import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_TRACES_ROOT,
    OUTPUT_DIM,
)
from scripts.run_fc_tp_supervised_shift_adaptation import (  # noqa: E402
    evaluate,
    interleaved_class_indices,
    load_dataset_batches,
    shift_nmnist_frames,
)
from scripts.screen_fc_tp_update_candidates import (  # noqa: E402
    Candidate,
    _delta_stats,
    _snn_params,
    build_optimizers,
)


HIDDEN_DIM = 160
IN_DIM = 2312
RTL_TRACE_DECAY_Q16 = 64225
SOURCE_TRACE_FORMATS = (
    "rtl_u16_integer",
    "rtl_uq6_10",
    "q0_16_saturating",
)
DEFAULT_WEIGHTS = (
    ROOT
    / "imports"
    / "fc"
    / "p3a_h160"
    / "data"
    / "weights"
    / "p3a_nmnist_h160_k8_int5_w2_int8_hash8"
    / "weights_h160_int5_w2_int8.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "learning_reference"
    / "fc_tp_h160_trainable_mask_ablation.json"
)
TRAINABLE_MASKS: dict[str, tuple[bool, bool]] = {
    "off": (False, False),
    "w2": (False, True),
    "w0": (True, False),
    "w0_w2": (True, True),
}


def advance_rtl_u16_trace(
    state: torch.Tensor,
    spike: torch.Tensor,
    *,
    fractional_bits: int = 0,
) -> torch.Tensor:
    """Advance the physical H40 unsigned 16-bit trace state by one timestep."""
    if fractional_bits < 0 or fractional_bits > 10:
        raise ValueError("16-bit FC trace fractional bits must be in [0,10]")
    state_i64 = state.to(torch.int64)
    spike_i64 = (
        torch.round(spike).to(torch.int64) * (1 << fractional_bits)
    )
    return (
        (((state_i64 * RTL_TRACE_DECAY_Q16 + 32768) >> 16) & 0xFFFF)
        + spike_i64
    ) & 0xFFFF


class FixedSgdResidualOptimizer:
    """Q2.14 code updates with a signed 16-bit sub-LSB residual.

    The parameter itself is the forward-visible Q2.14 code value. Gradient
    magnitude is retained in an integer residual with 16 fractional bits per
    code LSB. This mirrors the shared code/residual RMW contract without
    introducing momentum or an FP32 optimizer state.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float,
        *,
        weight_fractional_bits: int = 14,
        residual_fractional_bits: int = 16,
    ):
        if learning_rate <= 0.0:
            raise ValueError("fixed SGD learning rate must be positive")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)
        self.weight_fractional_bits = int(weight_fractional_bits)
        self.residual_fractional_bits = int(residual_fractional_bits)
        self.residuals = [
            torch.zeros_like(parameter, dtype=torch.int32)
            for parameter in self.parameters
        ]
        self.code_writes = 0
        self.code_lsb_steps = 0
        self.increment_abs_max = 0
        self.residual_abs_max = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        code_scale = 1 << self.weight_fractional_bits
        residual_scale = 1 << self.residual_fractional_bits
        code_min = -(1 << 15)
        code_max = (1 << 15) - 1
        for parameter, residual in zip(self.parameters, self.residuals):
            if parameter.grad is None:
                continue
            scaled_delta = torch.round(
                -parameter.grad.to(torch.float64)
                * self.learning_rate
                * float(code_scale * residual_scale)
            ).to(torch.int64)
            total = residual.to(torch.int64) + scaled_delta
            code_delta = torch.sign(total) * torch.div(
                total.abs(), residual_scale, rounding_mode="floor"
            )
            code = torch.round(
                parameter.detach().to(torch.float64) * code_scale
            ).to(torch.int64)
            code_next = torch.clamp(code + code_delta, code_min, code_max)
            saturated = code_next != code + code_delta
            residual_next = total - (code_next - code) * residual_scale
            residual_next = torch.where(
                saturated, torch.zeros_like(residual_next), residual_next
            )
            self.code_writes += int(torch.count_nonzero(code_next != code).item())
            self.code_lsb_steps += int((code_next - code).abs().sum().item())
            self.increment_abs_max = max(
                self.increment_abs_max,
                int(scaled_delta.abs().max().item()),
            )
            self.residual_abs_max = max(
                self.residual_abs_max,
                int(residual_next.abs().max().item()),
            )
            residual.copy_(residual_next.to(torch.int32))
            parameter.copy_(code_next.to(parameter.dtype) / float(code_scale))


class Int8DirectResidualOptimizer:
    """Bit-exact INT8 code/residual update used by the FC W2 RTL path."""

    def __init__(
        self,
        parameters,
        *,
        weight_scale: float,
        vote_increment: int,
        vote_threshold: int,
        code_min: int = -127,
        code_max: int = 127,
    ):
        if weight_scale <= 0.0:
            raise ValueError("INT8 weight scale must be positive")
        if vote_threshold <= 0:
            raise ValueError("vote threshold must be positive")
        if vote_increment < 0 or vote_increment >= 2 * vote_threshold:
            raise ValueError("vote increment must be in [0, 2*threshold)")
        self.parameters = list(parameters)
        self.weight_scale = float(weight_scale)
        self.vote_increment = int(vote_increment)
        self.vote_threshold = int(vote_threshold)
        self.code_min = int(code_min)
        self.code_max = int(code_max)
        self.codes = [
            torch.clamp(
                torch.round(parameter.detach().to(torch.float64) / weight_scale),
                code_min,
                code_max,
            ).to(torch.int16)
            for parameter in self.parameters
        ]
        self.residuals = [
            torch.zeros_like(parameter, dtype=torch.int32)
            for parameter in self.parameters
        ]
        self.code_writes = 0
        self.code_lsb_steps = 0
        self.residual_writes = 0
        self.residual_abs_max = 0
        for parameter, code in zip(self.parameters, self.codes):
            parameter.data.copy_(
                code.to(parameter.dtype) * self.weight_scale
            )

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, raw_gradient: torch.Tensor) -> None:
        for parameter, code, residual in zip(
            self.parameters, self.codes, self.residuals
        ):
            if parameter.grad is None:
                continue
            if raw_gradient.shape != parameter.shape:
                raise ValueError(
                    "raw-gradient shape does not match INT8 parameter"
                )
            direction = -torch.sign(raw_gradient).to(torch.int64)
            total = residual.to(torch.int64) + direction * self.vote_increment
            threshold = self.vote_threshold
            code_delta = torch.where(
                total >= threshold,
                torch.div(total, threshold, rounding_mode="floor"),
                torch.where(
                    total <= -threshold,
                    -torch.div(-total, threshold, rounding_mode="floor"),
                    torch.zeros_like(total),
                ),
            )
            code_i64 = code.to(torch.int64)
            unclamped = code_i64 + code_delta
            code_next = torch.clamp(unclamped, self.code_min, self.code_max)
            saturated = code_next != unclamped
            residual_next = total - (code_next - code_i64) * threshold
            residual_next = torch.where(
                saturated, torch.zeros_like(residual_next), residual_next
            )
            code_changed = code_next != code_i64
            residual_changed = residual_next != residual.to(torch.int64)
            self.code_writes += int(torch.count_nonzero(code_changed).item())
            self.code_lsb_steps += int(
                (code_next - code_i64).abs().sum().item()
            )
            self.residual_writes += int(
                torch.count_nonzero(residual_changed).item()
            )
            self.residual_abs_max = max(
                self.residual_abs_max,
                int(residual_next.abs().max().item()),
            )
            code.copy_(code_next.to(torch.int16))
            residual.copy_(residual_next.to(torch.int32))
            parameter.copy_(code_next.to(parameter.dtype) * self.weight_scale)


class Int8MagnitudeResidualOptimizer:
    """INT8 code plus Q0.15 sub-LSB residual magnitude update.

    The floating-point TP model supplies the batch-mean gradient.  Each update
    is converted into units of one INT8 weight-code LSB, then retained with 15
    residual fractional bits.  ``torch.round`` provides the ties-to-even rule
    used by the intended RTL scaler.
    """

    def __init__(
        self,
        parameters,
        *,
        weight_scale: float,
        learning_rate: float,
        batch_count: int,
        raw_gradient_fractional_bits: int,
        residual_fractional_bits: int = 15,
        code_min: int = -127,
        code_max: int = 127,
    ):
        if weight_scale <= 0.0:
            raise ValueError("INT8 weight scale must be positive")
        if learning_rate <= 0.0:
            raise ValueError("magnitude residual learning rate must be positive")
        if batch_count <= 0:
            raise ValueError("magnitude residual batch count must be positive")
        self.parameters = list(parameters)
        self.weight_scale = float(weight_scale)
        self.learning_rate = float(learning_rate)
        self.batch_count = int(batch_count)
        self.raw_gradient_fractional_bits = int(raw_gradient_fractional_bits)
        self.residual_fractional_bits = int(residual_fractional_bits)
        self.raw_gradient_multiplier = int(round(
            self.learning_rate
            * float(1 << self.residual_fractional_bits)
            / (self.weight_scale * self.batch_count)
        ))
        if self.raw_gradient_multiplier <= 0:
            raise ValueError("raw-gradient multiplier rounded to zero")
        self.raw_gradient_shift = self.raw_gradient_fractional_bits
        self.code_min = int(code_min)
        self.code_max = int(code_max)
        self.codes = [
            torch.clamp(
                torch.round(parameter.detach().to(torch.float64) / weight_scale),
                code_min,
                code_max,
            ).to(torch.int16)
            for parameter in self.parameters
        ]
        self.residuals = [
            torch.zeros_like(parameter, dtype=torch.int32)
            for parameter in self.parameters
        ]
        self.code_writes = 0
        self.code_lsb_steps = 0
        self.residual_writes = 0
        self.scaled_delta_abs_max = 0
        self.residual_abs_max = 0
        for parameter, code in zip(self.parameters, self.codes):
            parameter.data.copy_(code.to(parameter.dtype) * self.weight_scale)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, raw_gradient: torch.Tensor) -> None:
        residual_scale = 1 << self.residual_fractional_bits
        for parameter, code, residual in zip(
            self.parameters, self.codes, self.residuals
        ):
            if parameter.grad is None:
                continue
            if raw_gradient.shape != parameter.shape:
                raise ValueError(
                    "raw-gradient shape does not match INT8 parameter"
                )
            product = raw_gradient.to(torch.int64) * self.raw_gradient_multiplier
            product_abs = product.abs()
            quotient = product_abs >> self.raw_gradient_shift
            remainder = product_abs & ((1 << self.raw_gradient_shift) - 1)
            half = 1 << (self.raw_gradient_shift - 1)
            round_up = (remainder > half) | (
                (remainder == half) & ((quotient & 1) != 0)
            )
            rounded_magnitude = quotient + round_up.to(torch.int64)
            scaled_delta = torch.where(
                product < 0, rounded_magnitude, -rounded_magnitude
            )
            total = residual.to(torch.int64) + scaled_delta
            code_delta = torch.sign(total) * torch.div(
                total.abs(), residual_scale, rounding_mode="floor"
            )
            code_i64 = code.to(torch.int64)
            unclamped = code_i64 + code_delta
            code_next = torch.clamp(unclamped, self.code_min, self.code_max)
            saturated = code_next != unclamped
            residual_next = total - (code_next - code_i64) * residual_scale
            residual_next = torch.where(
                saturated, torch.zeros_like(residual_next), residual_next
            )
            code_changed = code_next != code_i64
            residual_changed = residual_next != residual.to(torch.int64)
            self.code_writes += int(torch.count_nonzero(code_changed).item())
            self.code_lsb_steps += int(
                (code_next - code_i64).abs().sum().item()
            )
            self.residual_writes += int(
                torch.count_nonzero(residual_changed).item()
            )
            self.scaled_delta_abs_max = max(
                self.scaled_delta_abs_max,
                int(scaled_delta.abs().max().item()),
            )
            self.residual_abs_max = max(
                self.residual_abs_max,
                int(residual_next.abs().max().item()),
            )
            code.copy_(code_next.to(torch.int16))
            residual.copy_(residual_next.to(torch.int32))
            parameter.copy_(code_next.to(parameter.dtype) * self.weight_scale)


@torch.no_grad()
def quantize_q2_14(parameter: torch.Tensor) -> None:
    code = torch.clamp(
        torch.round(parameter.to(torch.float64) * float(1 << 14)),
        -(1 << 15),
        (1 << 15) - 1,
    )
    parameter.copy_(code.to(parameter.dtype) / float(1 << 14))


def class_balanced_episode_indices(
    targets,
    *,
    support_per_class: int,
    query_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Build disjoint, class-balanced support/query indices from one split."""
    by_class: dict[int, list[int]] = {class_id: [] for class_id in range(OUTPUT_DIM)}
    for index, label_value in enumerate(targets):
        label = int(label_value)
        if label in by_class:
            by_class[label].append(index)
    required = support_per_class + query_per_class
    if any(len(indices) < required for indices in by_class.values()):
        raise ValueError("dataset does not contain enough samples for every class")
    rng = random.Random(seed)
    for class_id in range(OUTPUT_DIM):
        rng.shuffle(by_class[class_id])
    support = [
        by_class[class_id][replica]
        for replica in range(support_per_class)
        for class_id in range(OUTPUT_DIM)
    ]
    query = [
        by_class[class_id][support_per_class + replica]
        for replica in range(query_per_class)
        for class_id in range(OUTPUT_DIM)
    ]
    if set(support) & set(query):
        raise RuntimeError("support/query construction produced overlapping indices")
    return support, query


def class_balanced_shuffled_indices(
    targets,
    *,
    samples_per_class: int,
    seed: int,
) -> list[int]:
    """Build one class-balanced index set with deterministic within-class shuffle."""
    support, _ = class_balanced_episode_indices(
        targets,
        support_per_class=samples_per_class,
        query_per_class=0,
        seed=seed,
    )
    return support


def build_h160_model(
    traces_root: Path,
    weights_path: Path,
    batch_size: int,
    device: torch.device,
):
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    from models.tp_mlp import TP_MLP  # pylint: disable=import-outside-toplevel

    packed = np.load(weights_path)
    required = {"W0", "W2", "S", "W0_scale", "W2_scale", "S_scale"}
    missing = required - set(packed.files)
    if missing:
        raise ValueError(f"H160 weight package missing arrays: {sorted(missing)}")
    if packed["W0"].shape != (IN_DIM, HIDDEN_DIM):
        raise ValueError(f"unexpected W0 shape: {packed['W0'].shape}")
    if packed["W2"].shape != (HIDDEN_DIM, OUTPUT_DIM):
        raise ValueError(f"unexpected W2 shape: {packed['W2'].shape}")
    if packed["S"].shape != (OUTPUT_DIM, HIDDEN_DIM):
        raise ValueError(f"unexpected S shape: {packed['S'].shape}")

    model = TP_MLP(
        hidden_layers=[HIDDEN_DIM],
        input_size=IN_DIM,
        output_size=OUTPUT_DIM,
        batch_size=batch_size,
        snn_params=_snn_params(),
    ).to(device)
    with torch.no_grad():
        model.network[0].fc.weight.copy_(torch.from_numpy(
            packed["W0"].astype(np.float32).T * float(packed["W0_scale"])
        ))
        model.network[1].fc.weight.copy_(torch.from_numpy(
            packed["W2"].astype(np.float32).T * float(packed["W2_scale"])
        ))
        model.target_propagator.fc.weight.copy_(torch.from_numpy(
            packed["S"].astype(np.float32).T * float(packed["S_scale"])
        ))
    return model


def state_cost(
    model,
    trainable_mask: str,
    optimizer_mode: str,
) -> dict[str, int]:
    w0_enabled, w2_enabled = TRAINABLE_MASKS[trainable_mask]
    mutable_parameters = (
        (model.network[0].fc.weight.numel() if w0_enabled else 0)
        + (model.network[1].fc.weight.numel() if w2_enabled else 0)
    )
    optimizer_state_multiplier = {
        "adam": 8,
        "sgd_momentum": 4,
        "direct_sign": 0,
        "fixed_sgd": 0,
        "int8_direct_residual": 0,
        "int8_magnitude_residual": 0,
    }[optimizer_mode]
    return {
        "mutable_parameters": mutable_parameters,
        "code_plus_residual_bytes": mutable_parameters * 4,
        "q2_14_code_plus_residual_bytes": mutable_parameters * 4,
        "fp32_weight_bytes": mutable_parameters * 4,
        "fp32_optimizer_state_bytes": mutable_parameters
        * optimizer_state_multiplier,
    }


def adapt_mask(
    candidate: Candidate,
    trainable_mask: str,
    *,
    traces_root: Path,
    weights_path: Path,
    adapt_batches: list[tuple[torch.Tensor, torch.Tensor]],
    test_batches: list[tuple[torch.Tensor, torch.Tensor]],
    baseline_clean: dict[str, Any],
    baseline_shifted: dict[str, Any],
    shift_dx: int,
    shift_dy: int,
    device: torch.device,
    vote_increment: int,
    vote_threshold: int,
    source_trace_bits: int,
    source_trace_format: str,
) -> dict[str, Any]:
    model = build_h160_model(
        traces_root, weights_path, int(adapt_batches[0][1].numel()), device
    )
    w0 = model.network[0].fc.weight
    w2 = model.network[1].fc.weight
    target_projector = model.target_propagator.fc.weight
    w0_enabled, w2_enabled = TRAINABLE_MASKS[trainable_mask]
    if candidate.mode in (
        "fixed_sgd",
        "int8_direct_residual",
        "int8_magnitude_residual",
    ):
        if trainable_mask not in ("off", "w2"):
            raise ValueError(
                f"{candidate.mode} FC evidence currently supports W2 only"
            )
    if candidate.mode == "fixed_sgd":
        quantize_q2_14(w2)
    w0_initial = w0.detach().clone()
    w2_initial = w2.detach().clone()
    target_projector_initial = target_projector.detach().clone()
    if candidate.mode == "fixed_sgd":
        optimizers = {
            "layer_1": FixedSgdResidualOptimizer(
                [w2], candidate.learning_rate
            )
        }
    elif candidate.mode == "int8_direct_residual":
        packed = np.load(weights_path)
        optimizers = {
            "layer_1": Int8DirectResidualOptimizer(
                [w2],
                weight_scale=float(packed["W2_scale"]),
                vote_increment=vote_increment,
                vote_threshold=vote_threshold,
            )
        }
    elif candidate.mode == "int8_magnitude_residual":
        packed = np.load(weights_path)
        optimizers = {
            "layer_1": Int8MagnitudeResidualOptimizer(
                [w2],
                weight_scale=float(packed["W2_scale"]),
                learning_rate=candidate.learning_rate,
                batch_count=int(adapt_batches[0][1].numel()),
                raw_gradient_fractional_bits=(
                    {
                        "rtl_u16_integer": 16,
                        "rtl_uq6_10": 26,
                        "q0_16_saturating": 32,
                    }[source_trace_format]
                ),
            )
        }
    else:
        optimizers = build_optimizers(model, candidate)
    selected_optimizer_names = {
        name
        for name, enabled in (("layer_0", w0_enabled), ("layer_1", w2_enabled))
        if enabled
    }
    if not selected_optimizer_names.issubset(optimizers):
        raise RuntimeError(
            f"optimizer keys {sorted(optimizers)} do not cover mask {trainable_mask}"
        )

    local_baseline_clean = baseline_clean
    local_baseline_shifted = baseline_shifted
    if candidate.mode in (
        "fixed_sgd",
        "int8_direct_residual",
        "int8_magnitude_residual",
    ):
        local_baseline_clean = evaluate(model, test_batches, shifted=False)
        local_baseline_shifted = evaluate(
            model,
            test_batches,
            shifted=True,
            shift_dx=shift_dx,
            shift_dy=shift_dy,
        )

    update_calls = 0
    w2_gradient_abs_max = 0.0
    w2_gradient_abs_sum = 0.0
    w2_gradient_nonzero = 0
    w2_quantized_gradient_nonzero = 0
    scheduler_candidate_tuples = 0
    scheduler_trace_nonzero_candidates = 0
    scheduler_error_nonzero_observations = 0
    scheduler_emitted_contributions = 0
    source_trace_observed_max_q016 = 0
    source_trace_saturated_observations = 0
    source_trace_float_abs_max = 0.0
    w2_gradient_contract_abs_max = 0.0
    w2_gradient_contract_sign_mismatches = 0
    started_at = time.perf_counter()
    for frames, labels in adapt_batches:
        frames = frames.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        shifted_frames = shift_nmnist_frames(frames, dx=shift_dx, dy=shift_dy)
        target = F.one_hot(labels, OUTPUT_DIM).to(dtype=torch.float32)
        model.reset_potential()
        rtl_trace_u16 = torch.zeros(
            (labels.numel(), HIDDEN_DIM), device=device, dtype=torch.int64
        )
        for timestep in range(shifted_frames.shape[1]):
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            model(shifted_frames[:, timestep, :], target)
            model.update()
            if w0.grad is None or w2.grad is None:
                raise RuntimeError("TP update did not generate W0 and W2 gradients")
            with torch.no_grad():
                source_trace_float_abs_max = max(
                    source_trace_float_abs_max,
                    float(model.trace[-1].detach().abs().max().item()),
                )
                if source_trace_format in ("rtl_u16_integer", "rtl_uq6_10"):
                    hidden_spike = torch.round(
                        model.hs[-1].detach()
                    ).to(torch.int64)
                    trace_fractional_bits = (
                        10 if source_trace_format == "rtl_uq6_10" else 0
                    )
                    rtl_trace_u16 = advance_rtl_u16_trace(
                        rtl_trace_u16,
                        hidden_spike,
                        fractional_bits=trace_fractional_bits,
                    )
                    trace_rounded_q016 = rtl_trace_u16
                else:
                    trace_rounded_q016 = torch.round(
                        model.trace[-1].detach() * 65536.0
                    )
                    trace_fractional_bits = 16
                source_trace_observed_max_q016 = max(
                    source_trace_observed_max_q016,
                    int(trace_rounded_q016.max().item()),
                )
                source_trace_saturated_observations += int(
                    torch.count_nonzero(
                        trace_rounded_q016 > ((1 << source_trace_bits) - 1)
                    ).item()
                )
                trace_q016 = torch.clamp(
                    trace_rounded_q016,
                    0,
                    (1 << source_trace_bits) - 1,
                ).to(torch.int32)
                error_q016 = torch.clamp(
                    torch.round(model.e.detach() * 65536.0),
                    -(1 << 17),
                    (1 << 17) - 1,
                ).to(torch.int32)
                trace_nonzero_per_sample = (trace_q016 != 0).sum(dim=1)
                error_nonzero_per_sample = (error_q016 != 0).sum(dim=1)
                scheduler_candidate_tuples += int(
                    trace_q016.shape[0] * trace_q016.shape[1] * error_q016.shape[1]
                )
                scheduler_trace_nonzero_candidates += int(
                    trace_nonzero_per_sample.sum().item() * error_q016.shape[1]
                )
                scheduler_error_nonzero_observations += int(
                    error_nonzero_per_sample.sum().item()
                )
                scheduler_emitted_contributions += int(
                    (trace_nonzero_per_sample * error_nonzero_per_sample).sum().item()
                )
                # CUDA has no int64 GEMM.  Each Q0.16 product and the B16 sum
                # are below 2^53, so FP64 GEMM followed by an integer cast is
                # exact for this production profile.
                raw_w2_gradient = torch.round(
                    error_q016.to(torch.float64).transpose(0, 1)
                    @ trace_q016.to(torch.float64)
                ).to(torch.int64)
                w2_quantized_gradient_nonzero += int(
                    torch.count_nonzero(raw_w2_gradient).item()
                )
                quantized_w2_gradient = raw_w2_gradient.to(torch.float64) / float(
                    trace_q016.shape[0]
                    * (1 << (16 + trace_fractional_bits))
                )
                w2_gradient_contract_abs_max = max(
                    w2_gradient_contract_abs_max,
                    float((
                        quantized_w2_gradient
                        - w2.grad.detach().to(torch.float64)
                    ).abs().max().item()),
                )
                quantized_sign = torch.sign(raw_w2_gradient)
                software_sign = torch.sign(w2.grad.detach())
                w2_gradient_contract_sign_mismatches += int(
                    torch.count_nonzero(quantized_sign != software_sign).item()
                )
            w2_gradient_abs_max = max(
                w2_gradient_abs_max, float(w2.grad.detach().abs().max().item())
            )
            w2_nonzero = w2.grad.detach() != 0
            if torch.any(w2_nonzero):
                w2_gradient_abs_sum += float(
                    w2.grad.detach().abs()[w2_nonzero].sum().item()
                )
                w2_gradient_nonzero += int(w2_nonzero.sum().item())
            for name in selected_optimizer_names:
                if candidate.mode in (
                    "int8_direct_residual",
                    "int8_magnitude_residual",
                ):
                    optimizers[name].step(raw_w2_gradient)
                else:
                    optimizers[name].step()
            update_calls += int(bool(selected_optimizer_names))
            model.detach_membrane_states()
    elapsed = time.perf_counter() - started_at

    if not torch.equal(target_projector_initial, target_projector.detach()):
        raise RuntimeError("train_s=false adaptation changed target projector S")
    w0_delta = _delta_stats(w0, w0_initial)
    w2_delta = _delta_stats(w2, w2_initial)
    if not w0_enabled and w0_delta["nonzero"]:
        raise RuntimeError(f"disabled W0 changed under {trainable_mask}")
    if not w2_enabled and w2_delta["nonzero"]:
        raise RuntimeError(f"disabled W2 changed under {trainable_mask}")
    clean_after = evaluate(model, test_batches, shifted=False)
    shifted_after = evaluate(
        model,
        test_batches,
        shifted=True,
        shift_dx=shift_dx,
        shift_dy=shift_dy,
    )
    optimizer_diagnostics: dict[str, int] | None = None
    if candidate.mode == "fixed_sgd":
        fixed_optimizer = optimizers["layer_1"]
        optimizer_diagnostics = {
            "code_writes": fixed_optimizer.code_writes,
            "code_lsb_steps": fixed_optimizer.code_lsb_steps,
            "increment_abs_max": fixed_optimizer.increment_abs_max,
            "residual_abs_max": fixed_optimizer.residual_abs_max,
        }
    elif candidate.mode == "int8_direct_residual":
        int8_optimizer = optimizers["layer_1"]
        optimizer_diagnostics = {
            "vote_increment": int8_optimizer.vote_increment,
            "vote_threshold": int8_optimizer.vote_threshold,
            "code_writes": int8_optimizer.code_writes,
            "code_lsb_steps": int8_optimizer.code_lsb_steps,
            "residual_writes": int8_optimizer.residual_writes,
            "residual_abs_max": int8_optimizer.residual_abs_max,
        }
    elif candidate.mode == "int8_magnitude_residual":
        magnitude_optimizer = optimizers["layer_1"]
        optimizer_diagnostics = {
            "residual_fractional_bits": (
                magnitude_optimizer.residual_fractional_bits
            ),
            "raw_gradient_multiplier": (
                magnitude_optimizer.raw_gradient_multiplier
            ),
            "raw_gradient_shift": magnitude_optimizer.raw_gradient_shift,
            "code_writes": magnitude_optimizer.code_writes,
            "code_lsb_steps": magnitude_optimizer.code_lsb_steps,
            "residual_writes": magnitude_optimizer.residual_writes,
            "scaled_delta_abs_max": magnitude_optimizer.scaled_delta_abs_max,
            "residual_abs_max": magnitude_optimizer.residual_abs_max,
        }
    return {
        "trainable_mask": trainable_mask,
        "optimizer": candidate.name,
        "optimizer_update_calls": update_calls,
        "w0_delta": w0_delta,
        "w2_delta": w2_delta,
        "state_cost": state_cost(model, trainable_mask, candidate.mode),
        "adaptation_seconds_wall": elapsed,
        "device": str(device),
        "baseline_clean": local_baseline_clean,
        "baseline_shifted": local_baseline_shifted,
        "w2_gradient_abs_max": w2_gradient_abs_max,
        "w2_gradient_abs_mean_nonzero": (
            w2_gradient_abs_sum / w2_gradient_nonzero
            if w2_gradient_nonzero
            else 0.0
        ),
        "w2_gradient_nonzero_observations": w2_gradient_nonzero,
        "w2_quantized_gradient_nonzero_observations": (
            w2_quantized_gradient_nonzero
        ),
        "w2_gradient_contract_abs_max": w2_gradient_contract_abs_max,
        "w2_gradient_contract_sign_mismatches": (
            w2_gradient_contract_sign_mismatches
        ),
        "scheduler_q016_diagnostics": {
            "candidate_tuples": scheduler_candidate_tuples,
            "trace_nonzero_candidates": scheduler_trace_nonzero_candidates,
            "error_nonzero_observations": scheduler_error_nonzero_observations,
            "emitted_contributions": scheduler_emitted_contributions,
            "source_trace_bits": source_trace_bits,
            "source_trace_format": source_trace_format,
            "source_trace_fractional_bits": (
                {
                    "rtl_u16_integer": 0,
                    "rtl_uq6_10": 10,
                    "q0_16_saturating": 16,
                }[source_trace_format]
            ),
            "source_trace_float_abs_max": source_trace_float_abs_max,
            "source_trace_observed_max_q016": source_trace_observed_max_q016,
            "source_trace_saturated_observations": (
                source_trace_saturated_observations
            ),
            "no_stall_cycle_model": (
                "2*Ncandidate + 2*Ntrace_nonzero + Nemitted"
            ),
            "no_stall_cycles": (
                2 * scheduler_candidate_tuples
                + 2 * scheduler_trace_nonzero_candidates
                + scheduler_emitted_contributions
            ),
        },
        "optimizer_diagnostics": optimizer_diagnostics,
        "source_trace_bits": source_trace_bits,
        "clean_after": clean_after,
        "shifted_after": shifted_after,
        "shifted_gain_points": (
            shifted_after["accuracy_percent"]
            - local_baseline_shifted["accuracy_percent"]
        ),
        "clean_retention_points": (
            clean_after["accuracy_percent"]
            - local_baseline_clean["accuracy_percent"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-samples-per-class", type=int, default=16)
    parser.add_argument("--test-samples-per-class", type=int, default=10)
    parser.add_argument(
        "--all-query-samples",
        action="store_true",
        help=(
            "Evaluate every sample in the selected query split. This is "
            "intended for a locked report and cannot be combined with a "
            "training-split query."
        ),
    )
    parser.add_argument("--shift-x", type=int, default=2)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=None,
        help=(
            "Enable a shuffled class-balanced episode. With --query-split train, "
            "support and query are disjoint subsets of the training split."
        ),
    )
    parser.add_argument(
        "--query-split",
        choices=("train", "test"),
        default="test",
        help="Dataset split used for query evaluation.",
    )
    parser.add_argument(
        "--trainable-masks",
        nargs="+",
        choices=tuple(TRAINABLE_MASKS),
        default=list(TRAINABLE_MASKS),
    )
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
    parser.add_argument("--learning-rate", type=float, default=2.0**-14)
    parser.add_argument("--vote-increment", type=int, default=1)
    parser.add_argument("--vote-threshold", type=int, default=1)
    parser.add_argument(
        "--source-trace-bits",
        type=int,
        choices=(16, 17),
        default=16,
        help="Unsigned FC W2 source-trace storage width.",
    )
    parser.add_argument(
        "--source-trace-format",
        choices=SOURCE_TRACE_FORMATS,
        default="rtl_u16_integer",
        help=(
            "rtl_u16_integer reproduces the current physical H40 recurrence; "
            "rtl_uq6_10 evaluates the same-width precision-corrected candidate; "
            "q0_16_saturating preserves the historical software conversion."
        ),
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. auto selects CUDA when available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    effective_seed = 38472 if args.episode_seed is None else args.episode_seed
    torch.manual_seed(effective_seed)
    torch.use_deterministic_algorithms(True)
    if args.batch_size < OUTPUT_DIM:
        raise ValueError("batch_size must include at least one sample per class")
    if args.all_query_samples and args.query_split != "test":
        raise ValueError("--all-query-samples requires --query-split test")
    if str(args.traces_root) not in sys.path:
        sys.path.insert(0, str(args.traces_root))
    import tonic  # pylint: disable=import-outside-toplevel

    train_dataset = tonic.datasets.NMNIST(save_to=str(args.data_root), train=True)
    test_dataset = tonic.datasets.NMNIST(save_to=str(args.data_root), train=False)
    if args.episode_seed is None:
        if args.query_split != "test":
            raise ValueError("--query-split train requires --episode-seed")
        adapt_indices = interleaved_class_indices(
            train_dataset.targets,
            samples_per_class=args.adapt_samples_per_class,
            start_replica=2,
        )
        query_indices = (
            list(range(len(test_dataset)))
            if args.all_query_samples
            else interleaved_class_indices(
                test_dataset.targets,
                samples_per_class=args.test_samples_per_class,
            )
        )
    elif args.query_split == "train":
        adapt_indices, query_indices = class_balanced_episode_indices(
            train_dataset.targets,
            support_per_class=args.adapt_samples_per_class,
            query_per_class=args.test_samples_per_class,
            seed=args.episode_seed,
        )
    else:
        adapt_indices = class_balanced_shuffled_indices(
            train_dataset.targets,
            samples_per_class=args.adapt_samples_per_class,
            seed=args.episode_seed,
        )
        query_indices = (
            list(range(len(test_dataset)))
            if args.all_query_samples
            else class_balanced_shuffled_indices(
                test_dataset.targets,
                samples_per_class=args.test_samples_per_class,
                seed=args.episode_seed ^ 0x5A17,
            )
        )
    adapt_batches = load_dataset_batches(
        traces_root=args.traces_root,
        data_root=args.data_root,
        train=True,
        indices=adapt_indices,
        batch_size=args.batch_size,
    )
    query_batches = load_dataset_batches(
        traces_root=args.traces_root,
        data_root=args.data_root,
        train=args.query_split == "train",
        indices=query_indices,
        batch_size=args.batch_size,
    )
    baseline_model = build_h160_model(
        args.traces_root, args.weights, args.batch_size, device
    )
    baseline_clean = evaluate(baseline_model, query_batches, shifted=False)
    baseline_shifted = evaluate(
        baseline_model, query_batches,
        shifted=True,
        shift_dx=args.shift_x,
        shift_dy=args.shift_y,
    )
    candidate_name = (
        f"int8_direct_residual_i{args.vote_increment}_t{args.vote_threshold}"
        if args.optimizer == "int8_direct_residual"
        else f"{args.optimizer}_{args.learning_rate:g}"
    )
    candidate = Candidate(
        candidate_name,
        args.optimizer,
        args.learning_rate,
        args.momentum,
    )
    masks = [
        adapt_mask(
            candidate,
            trainable_mask,
            traces_root=args.traces_root,
            weights_path=args.weights,
            adapt_batches=adapt_batches,
            test_batches=query_batches,
            baseline_clean=baseline_clean,
            baseline_shifted=baseline_shifted,
            shift_dx=args.shift_x,
            shift_dy=args.shift_y,
            device=device,
            vote_increment=args.vote_increment,
            vote_threshold=args.vote_threshold,
            source_trace_bits=args.source_trace_bits,
            source_trace_format=args.source_trace_format,
        )
        for trainable_mask in args.trainable_masks
    ]
    claim_boundary = (
        "Deterministic H160 mask screen from the same quantized W0/W2/S "
        "package used by RTL. It is a bounded software adaptation experiment, "
        "not fixed-point transaction, statistical-significance, or RTL evidence."
    )
    if args.optimizer == "int8_direct_residual":
        claim_boundary = (
            "Deterministic H160 W2 screen using the production INT8 code scale "
            "and the RTL direct-residual vote normalization. Source traces use "
            f"the {args.source_trace_format} contract, so this is bounded numeric "
            "evidence rather than an RTL transaction trace or final dataset claim."
        )
    elif args.optimizer == "int8_magnitude_residual":
        claim_boundary = (
            "Deterministic H160 W2 screen using the production INT8 code scale "
            "and a Q0.15 sub-LSB magnitude residual. Gradient generation remains "
            "the software TP model; the candidate must still be lowered to and "
            "verified against the FC RTL scaler before becoming hardware evidence."
        )
    result = {
        "status": "pass",
        "claim_boundary": claim_boundary,
        "network": "2312-160-10",
        "shift": {
            "dx_pixels": args.shift_x,
            "dy_pixels": args.shift_y,
            "zero_fill": True,
        },
        "episode_seed": args.episode_seed,
        "query_split": args.query_split,
        "device": str(device),
        "vote_increment": args.vote_increment,
        "vote_threshold": args.vote_threshold,
        "source_trace_bits": args.source_trace_bits,
        "source_trace_format": args.source_trace_format,
        "support_query_overlap": (
            len(set(adapt_indices) & set(query_indices))
            if args.query_split == "train"
            else 0
        ),
        "support_query_disjoint_by_split": args.query_split == "test",
        "all_query_samples": args.all_query_samples,
        "support_samples": len(adapt_indices),
        "query_samples": len(query_indices),
        "baseline_clean": baseline_clean,
        "baseline_shifted": baseline_shifted,
        "masks": masks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "FC_TP_H160_MASK_ABLATION_PASS "
        + " ".join(
            f"{item['trainable_mask']}="
            f"{item['shifted_after']['accuracy_percent']:.2f}%"
            for item in masks
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
