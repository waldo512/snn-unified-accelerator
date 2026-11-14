#!/usr/bin/env python3
"""Screen FC TP update candidates on one real N-MNIST B16 prefix.

This is a numerical and state-cost smoke test, not an accuracy benchmark.  Each
candidate starts from the same H400 checkpoint and sees the same first eight
dataset timesteps.  The TP model still generates W0/W2 gradients; only the
parameter update rule changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fc_tp_h400_b16_reference import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_TRACES_ROOT,
    HIDDEN_DIM,
    IN_DIM,
    OUTPUT_DIM,
    _load_flat,
    select_mixed_label_indices,
)


DEFAULT_OUTPUT = (
    ROOT / "results" / "learning_reference" / "fc_tp_h400_b16_candidate_smoke.json"
)
TRAINABLE_PARAMETERS = HIDDEN_DIM * IN_DIM + OUTPUT_DIM * HIDDEN_DIM


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    learning_rate: float
    momentum: float = 0.0
    output_learning_rate: float | None = None


CANDIDATES = (
    Candidate("adam_ref_lr1e-4", "adam", 1e-4),
    Candidate("sgd_momentum_lr1e-3", "sgd_momentum", 1e-3, 0.9),
    Candidate("sgd_momentum_lr1e-2", "sgd_momentum", 1e-2, 0.9),
    Candidate("direct_sign_step_2m14", "direct_sign", 2.0**-14),
    Candidate("direct_sign_step_2m12", "direct_sign", 2.0**-12),
)


class DirectSignOptimizer:
    """Fixed-magnitude sign update with no persistent optimizer state."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter], step_size: float):
        self.parameters = list(parameters)
        self.step_size = float(step_size)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.add_(parameter.grad.sign(), alpha=-self.step_size)


def _snn_params() -> dict[str, Any]:
    return {
        "l_vth": 1.0,
        "l_leak_m": 0.98,
        "l_leak_t": 0.98,
        "l_rec": False,
        "l_rst_type": "soft",
        "l_out_leak_m": 1.0,
        "train_s": False,
        "surrogate_type": "1",
        "surrogate_scale": 1.0,
        "norm_type": "none",
    }


def load_batch(
    traces_root: Path,
    data_root: Path,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    sys.path.insert(0, str(traces_root))
    import tonic  # pylint: disable=import-outside-toplevel
    import tonic.transforms as tonic_transforms  # pylint: disable=import-outside-toplevel
    from torch.utils.data import DataLoader, Subset  # pylint: disable=import-outside-toplevel

    transform = tonic_transforms.Compose([
        tonic_transforms.ToFrame(
            sensor_size=tonic.datasets.NMNIST.sensor_size,
            time_window=1000,
        ),
        lambda frames: frames.reshape(frames.shape[0], -1),
    ])
    dataset = tonic.datasets.NMNIST(
        save_to=str(data_root),
        train=True,
        first_saccade_only=True,
        transform=transform,
    )
    sample_indices = select_mixed_label_indices(dataset.targets, batch_size=batch_size)
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=True),
        num_workers=0,
    )
    frames, labels = next(iter(loader))
    return frames.to(dtype=torch.float32), labels.to(dtype=torch.long), sample_indices


def build_model(traces_root: Path, checkpoint_root: Path, batch_size: int):
    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    from models.tp_mlp import TP_MLP  # pylint: disable=import-outside-toplevel

    model = TP_MLP(
        hidden_layers=[HIDDEN_DIM],
        input_size=IN_DIM,
        output_size=OUTPUT_DIM,
        batch_size=batch_size,
        snn_params=_snn_params(),
    ).cpu()
    with torch.no_grad():
        model.network[0].fc.weight.copy_(
            _load_flat(
                checkpoint_root / "TP_NMNIST_network_0_fc_weight.pt",
                (HIDDEN_DIM, IN_DIM),
            )
        )
        model.network[1].fc.weight.copy_(
            _load_flat(
                checkpoint_root / "TP_NMNIST_network_1_fc_weight.pt",
                (OUTPUT_DIM, HIDDEN_DIM),
            )
        )
        model.target_propagator.fc.weight.copy_(
            _load_flat(
                checkpoint_root / "TP_NMNIST_target_propagator_fc_weight.pt",
                (HIDDEN_DIM, OUTPUT_DIM),
            )
        )
    return model


def build_optimizers(model, candidate: Candidate) -> dict[str, Any]:
    optimizers: dict[str, Any] = {}
    for name, parameters in model.get_layerwise_params():
        parameter_list = list(parameters)
        learning_rate = (
            candidate.output_learning_rate
            if name == "layer_1" and candidate.output_learning_rate is not None
            else candidate.learning_rate
        )
        if candidate.mode == "adam":
            optimizer = torch.optim.Adam(parameter_list, lr=learning_rate)
        elif candidate.mode == "sgd_momentum":
            optimizer = torch.optim.SGD(
                parameter_list,
                lr=learning_rate,
                momentum=candidate.momentum,
            )
        elif candidate.mode == "direct_sign":
            optimizer = DirectSignOptimizer(parameter_list, learning_rate)
        else:
            raise ValueError(f"unsupported candidate mode: {candidate.mode}")
        optimizers[name] = optimizer
    return optimizers


def inference_logits(model, frames: torch.Tensor) -> torch.Tensor:
    model.reset_potential()
    logits = None
    with torch.no_grad():
        for timestep in range(frames.shape[1]):
            logits = model.inference(frames[:, timestep, :]).detach().clone()
    if logits is None:
        raise ValueError("empty timestep sequence")
    return logits


def _delta_stats(after: torch.Tensor, before: torch.Tensor) -> dict[str, Any]:
    delta = after.detach() - before.detach()
    return {
        "nonzero": int(torch.count_nonzero(delta).item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "max_abs": float(delta.abs().max().item()),
        "finite": bool(torch.isfinite(delta).all().item()),
    }


def _state_cost(candidate: Candidate) -> dict[str, int]:
    weight_bytes = TRAINABLE_PARAMETERS * 4
    if candidate.mode == "adam":
        optimizer_state_bytes = TRAINABLE_PARAMETERS * 8
    elif candidate.mode == "sgd_momentum":
        optimizer_state_bytes = TRAINABLE_PARAMETERS * 4
    else:
        optimizer_state_bytes = 0
    return {
        "fp32_weight_bytes": weight_bytes,
        "fp32_optimizer_state_bytes": optimizer_state_bytes,
        "fp32_total_persistent_bytes": weight_bytes + optimizer_state_bytes,
    }


def run_candidate(
    candidate: Candidate,
    *,
    traces_root: Path,
    checkpoint_root: Path,
    frames: torch.Tensor,
    labels: torch.Tensor,
    train_timesteps: int,
    baseline_full_logits: torch.Tensor,
) -> dict[str, Any]:
    model = build_model(traces_root, checkpoint_root, batch_size=int(labels.numel()))
    target = F.one_hot(labels, OUTPUT_DIM).to(dtype=torch.float32)
    w0 = model.network[0].fc.weight
    w2 = model.network[1].fc.weight
    target_projector = model.target_propagator.fc.weight
    w0_initial = w0.detach().clone()
    w2_initial = w2.detach().clone()
    target_projector_initial = target_projector.detach().clone()
    optimizers = build_optimizers(model, candidate)

    gradient_history: list[dict[str, int]] = []
    started_at = time.perf_counter()
    model.reset_potential()
    for timestep in range(train_timesteps):
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        model(frames[:, timestep, :], target)
        model.update()
        if w0.grad is None or w2.grad is None:
            raise RuntimeError("TP update did not generate W0 and W2 gradients")
        gradient_history.append({
            "timestep": timestep,
            "w0_nonzero": int(torch.count_nonzero(w0.grad).item()),
            "w2_nonzero": int(torch.count_nonzero(w2.grad).item()),
        })
        for optimizer in optimizers.values():
            optimizer.step()
        model.detach_membrane_states()
    elapsed_seconds = time.perf_counter() - started_at

    if not torch.equal(target_projector_initial, target_projector.detach()):
        raise RuntimeError("train_s=false candidate changed target projector S")
    if not torch.isfinite(w0).all() or not torch.isfinite(w2).all():
        raise RuntimeError(f"{candidate.name} produced non-finite weights")

    prefix_logits = inference_logits(model, frames[:, :train_timesteps, :])
    full_logits = inference_logits(model, frames)
    full_predictions = full_logits.argmax(dim=1)
    full_logits_delta = full_logits - baseline_full_logits
    return {
        "name": candidate.name,
        "mode": candidate.mode,
        "learning_rate_or_step": candidate.learning_rate,
        "output_learning_rate_or_step": (
            candidate.output_learning_rate
            if candidate.output_learning_rate is not None
            else candidate.learning_rate
        ),
        "momentum": candidate.momentum,
        "train_timesteps": train_timesteps,
        "gradient_history": gradient_history,
        "w0_delta": _delta_stats(w0, w0_initial),
        "w2_delta": _delta_stats(w2, w2_initial),
        "target_projector_changed": 0,
        "prefix_predictions": prefix_logits.argmax(dim=1).tolist(),
        "full_predictions": full_predictions.tolist(),
        "full_batch_correct": int((full_predictions == labels).sum().item()),
        "full_batch_accuracy_percent": float(
            100.0 * (full_predictions == labels).to(dtype=torch.float32).mean().item()
        ),
        "prefix_logits_l2": float(torch.linalg.vector_norm(prefix_logits).item()),
        "full_logits_l2": float(torch.linalg.vector_norm(full_logits).item()),
        "full_logits_delta_l2_vs_baseline": float(
            torch.linalg.vector_norm(full_logits_delta).item()
        ),
        "full_logits_delta_max_abs_vs_baseline": float(
            full_logits_delta.abs().max().item()
        ),
        "elapsed_seconds": elapsed_seconds,
        "state_cost": _state_cost(candidate),
    }


def run_screen(
    *,
    traces_root: Path,
    checkpoint_root: Path,
    data_root: Path,
    output: Path,
    batch_size: int = 16,
    train_timesteps: int = 8,
) -> dict[str, Any]:
    torch.manual_seed(38472)
    torch.use_deterministic_algorithms(True)
    frames, labels, sample_indices = load_batch(traces_root, data_root, batch_size)
    if train_timesteps <= 0 or train_timesteps > frames.shape[1]:
        raise ValueError(
            f"train_timesteps must be in 1..{frames.shape[1]}, got {train_timesteps}"
        )

    baseline_model = build_model(traces_root, checkpoint_root, batch_size)
    baseline_logits = inference_logits(baseline_model, frames)
    baseline_predictions = baseline_logits.argmax(dim=1)
    candidates = [
        run_candidate(
            candidate,
            traces_root=traces_root,
            checkpoint_root=checkpoint_root,
            frames=frames,
            labels=labels,
            train_timesteps=train_timesteps,
            baseline_full_logits=baseline_logits,
        )
        for candidate in CANDIDATES
    ]
    for result in candidates:
        prediction = torch.tensor(result["full_predictions"], dtype=torch.long)
        result["full_predictions_changed_vs_baseline"] = int(
            torch.count_nonzero(prediction - baseline_predictions).item()
        )

    summary = {
        "status": "pass",
        "claim_boundary": (
            "Real N-MNIST H400/B16 TP-gradient smoke over a short prefix. "
            "It compares update stability and FP32 state lower bounds only; same-batch "
            "accuracy is not a generalization or online-adaptation claim. Direct-sign "
            "uses FP32 weights and is not yet a fixed-point RTL model."
        ),
        "source": {
            "checkpoint": str(checkpoint_root),
            "batch_size": batch_size,
            "sample_indices": sample_indices,
            "labels": labels.tolist(),
            "padded_timesteps": int(frames.shape[1]),
            "trained_prefix_timesteps": train_timesteps,
            "trainable_parameters": TRAINABLE_PARAMETERS,
        },
        "baseline": {
            "full_predictions": baseline_predictions.tolist(),
            "full_batch_correct": int((baseline_predictions == labels).sum().item()),
            "full_batch_accuracy_percent": float(
                100.0
                * (baseline_predictions == labels).to(dtype=torch.float32).mean().item()
            ),
            "full_logits_l2": float(torch.linalg.vector_norm(baseline_logits).item()),
        },
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-timesteps", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_screen(
        traces_root=args.traces_root,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        output=args.output,
        batch_size=args.batch_size,
        train_timesteps=args.train_timesteps,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
