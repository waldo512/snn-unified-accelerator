#!/usr/bin/env python3
"""Build a reproducible FC TP first-active-update reference from N-MNIST.

The source training command uses T=1 to request an optimizer update at every
dataset timestep.  This script follows that schedule from timestep zero and
captures the first timestep where both W0 and W2 gradients are nonzero for a
mixed-label B16 batch.  Optimizer state starts at zero because the supplied H400
checkpoint contains model parameters only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES_ROOT = Path(r"D:\Desktop\traces_propagation-main")
DEFAULT_CHECKPOINT_ROOT = Path(
    r"D:\Desktop\大创项目备份\SNN_Accelerator_50MHz"
    r"\15_s3_h400_pytorch_training_10k_e5"
    r"\saved_weights_best_h400_10k_e5"
)
DEFAULT_DATA_ROOT = Path(r"D:\datasets")
DEFAULT_OUTPUT_NPZ = (
    ROOT / "results" / "learning_reference" / "fc_tp_h400_b16_first_active_reference.npz"
)
DEFAULT_OUTPUT_JSON = (
    ROOT / "results" / "learning_reference" / "fc_tp_h400_b16_first_active_reference.json"
)

IN_DIM = 2312
HIDDEN_DIM = 400
OUTPUT_DIM = 10
LEARNING_RATE = 1e-4


def _load_flat(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    expected = int(np.prod(shape))
    if tensor.numel() != expected:
        raise ValueError(f"{path.name}: expected {expected} values, found {tensor.numel()}")
    return tensor.to(dtype=torch.float32).reshape(shape)


def select_mixed_label_indices(targets: list[int], batch_size: int = 16) -> list[int]:
    """Choose a deterministic mixed-label prefix without relying on shuffle RNG."""
    if batch_size < OUTPUT_DIM:
        raise ValueError(f"batch_size must be at least {OUTPUT_DIM}")
    by_class: dict[int, list[int]] = {class_id: [] for class_id in range(OUTPUT_DIM)}
    needed_per_class = 1 + (batch_size - OUTPUT_DIM + OUTPUT_DIM - 1) // OUTPUT_DIM
    for index, label_value in enumerate(targets):
        label = int(label_value)
        if label in by_class and len(by_class[label]) < needed_per_class:
            by_class[label].append(index)
        if all(len(indices) >= needed_per_class for indices in by_class.values()):
            break
    if any(not indices for indices in by_class.values()):
        raise ValueError("dataset does not contain every N-MNIST class")

    indices = [by_class[class_id][0] for class_id in range(OUTPUT_DIM)]
    remaining = batch_size - len(indices)
    replica = 1
    while remaining:
        for class_id in range(OUTPUT_DIM):
            if remaining == 0:
                break
            if len(by_class[class_id]) <= replica:
                raise ValueError(f"class {class_id} lacks sample replica {replica}")
            indices.append(by_class[class_id][replica])
            remaining -= 1
        replica += 1
    return indices


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    return {
        "shape": list(detached.shape),
        "nonzero": int(torch.count_nonzero(detached).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "l2": float(torch.linalg.vector_norm(detached).item()),
    }


def _optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    state = optimizer.state[parameter]
    step_value = state["step"]
    step = int(step_value.item()) if torch.is_tensor(step_value) else int(step_value)
    return state["exp_avg"].detach().cpu(), state["exp_avg_sq"].detach().cpu(), step


def _optimizer_state_or_zero(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    state = optimizer.state.get(parameter, {})
    if "exp_avg" not in state:
        return torch.zeros_like(parameter, device="cpu"), torch.zeros_like(parameter, device="cpu"), 0
    return _optimizer_state(optimizer, parameter)


def build_reference(
    *,
    traces_root: Path,
    checkpoint_root: Path,
    data_root: Path,
    output_npz: Path,
    output_json: Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    if not traces_root.is_dir():
        raise FileNotFoundError(f"missing traces source: {traces_root}")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"missing H400 checkpoint: {checkpoint_root}")
    if not (data_root / "NMNIST").is_dir():
        raise FileNotFoundError(f"missing N-MNIST dataset under {data_root}")

    sys.path.insert(0, str(traces_root))
    import tonic  # pylint: disable=import-outside-toplevel
    import tonic.transforms as tonic_transforms  # pylint: disable=import-outside-toplevel
    from torch.utils.data import DataLoader, Subset  # pylint: disable=import-outside-toplevel
    from models.tp_mlp import TP_MLP  # pylint: disable=import-outside-toplevel

    torch.manual_seed(38472)
    torch.use_deterministic_algorithms(True)

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
    labels = labels.to(dtype=torch.long)
    target_one_hot = F.one_hot(labels, OUTPUT_DIM).to(dtype=torch.float32)

    snn_params = {
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
    model = TP_MLP(
        hidden_layers=[HIDDEN_DIM],
        input_size=IN_DIM,
        output_size=OUTPUT_DIM,
        batch_size=batch_size,
        snn_params=snn_params,
    ).cpu()

    w0_path = checkpoint_root / "TP_NMNIST_network_0_fc_weight.pt"
    w2_path = checkpoint_root / "TP_NMNIST_network_1_fc_weight.pt"
    s_path = checkpoint_root / "TP_NMNIST_target_propagator_fc_weight.pt"
    with torch.no_grad():
        model.network[0].fc.weight.copy_(_load_flat(w0_path, (HIDDEN_DIM, IN_DIM)))
        model.network[1].fc.weight.copy_(_load_flat(w2_path, (OUTPUT_DIM, HIDDEN_DIM)))
        model.target_propagator.fc.weight.copy_(
            _load_flat(s_path, (HIDDEN_DIM, OUTPUT_DIM))
        )

    w0_parameter = model.network[0].fc.weight
    w2_parameter = model.network[1].fc.weight
    s_parameter = model.target_propagator.fc.weight
    s_before = s_parameter.detach().clone()

    optimizers = {
        name: torch.optim.Adam(parameters, lr=LEARNING_RATE)
        for name, parameters in model.get_layerwise_params()
    }
    model.reset_potential()
    captured: dict[str, Any] | None = None
    screen_windows: list[dict[str, Any]] = []
    gradient_history: list[dict[str, int]] = []
    for timestep in range(frames.shape[1]):
        for optimizer in optimizers.values():
            optimizer.zero_grad()

        input_timestep = frames[:, timestep, :].to(dtype=torch.float32)
        logits_before, target_logits = model(input_timestep, target_one_hot)
        model.update()
        if w0_parameter.grad is None or w2_parameter.grad is None:
            raise RuntimeError("TP update did not produce both W0 and W2 gradients")
        if s_parameter.grad is not None:
            raise RuntimeError("train_s=false unexpectedly produced a target-projector gradient")

        w0_gradient = w0_parameter.grad.detach().clone()
        w2_gradient = w2_parameter.grad.detach().clone()
        w0_nonzero = int(torch.count_nonzero(w0_gradient).item())
        w2_nonzero = int(torch.count_nonzero(w2_gradient).item())
        gradient_history.append({
            "timestep": timestep,
            "w0_gradient_nonzero": w0_nonzero,
            "w2_gradient_nonzero": w2_nonzero,
        })
        _, tp_target_distribution_step, tp_student_distribution_step = model.loss(
            model.trace[1], model.trace_target[1], model.trace_target[0]
        )
        hidden_membrane_post_step = model.network[0].membrane_potential.detach().clone()
        hidden_membrane_pre_step = hidden_membrane_post_step + model.hs[1].detach()
        hidden_surrogate_step = 1.0 / (
            1.0 + (torch.pi * (hidden_membrane_pre_step - 1.0)).pow(2)
        )
        screen_windows.append({
            "timestep": timestep,
            "input_timestep": input_timestep.detach().clone(),
            "hidden_trace": model.trace[1].detach().clone(),
            "target_input_trace": model.trace_target[0].detach().clone(),
            "target_hidden_trace": model.trace_target[1].detach().clone(),
            "hidden_surrogate": hidden_surrogate_step,
            "logits": logits_before.detach().clone(),
            "w0_gradient": w0_gradient,
            "w2_gradient": w2_gradient,
        })
        is_active = (
            w0_nonzero > 0
            and w2_nonzero > 0
        )
        if is_active:
            _, tp_target_distribution, tp_student_distribution = model.loss(
                model.trace[1], model.trace_target[1], model.trace_target[0]
            )
            hidden_membrane_post = model.network[0].membrane_potential.detach().clone()
            hidden_membrane_pre = hidden_membrane_post + model.hs[1].detach()
            hidden_surrogate = 1.0 / (
                1.0 + (torch.pi * (hidden_membrane_pre - 1.0)).pow(2)
            )
            w0_m_before, w0_v_before, w0_step_before = _optimizer_state_or_zero(
                optimizers["layer_0"], w0_parameter
            )
            w2_m_before, w2_v_before, w2_step_before = _optimizer_state_or_zero(
                optimizers["layer_1"], w2_parameter
            )
            captured = {
                "timestep": timestep,
                "input_timestep": input_timestep.detach().clone(),
                "logits_before": logits_before.detach().clone(),
                "target_logits": target_logits.detach().clone(),
                "hidden_spike": model.hs[1].detach().clone(),
                "target_hidden_spike": model.ts[1].detach().clone(),
                "input_trace": model.trace[0].detach().clone(),
                "hidden_trace": model.trace[1].detach().clone(),
                "hidden_membrane_pre": hidden_membrane_pre,
                "hidden_membrane_post": hidden_membrane_post,
                "hidden_surrogate": hidden_surrogate,
                "target_input_trace": model.trace_target[0].detach().clone(),
                "target_hidden_trace": model.trace_target[1].detach().clone(),
                "tp_target_distribution": tp_target_distribution.detach().clone(),
                "tp_student_distribution": tp_student_distribution.detach().clone(),
                "w0_before": w0_parameter.detach().clone(),
                "w0_gradient": w0_gradient,
                "w0_m_before": w0_m_before,
                "w0_v_before": w0_v_before,
                "w0_step_before": w0_step_before,
                "w2_before": w2_parameter.detach().clone(),
                "w2_gradient": w2_gradient,
                "w2_m_before": w2_m_before,
                "w2_v_before": w2_v_before,
                "w2_step_before": w2_step_before,
            }

        for optimizer in optimizers.values():
            optimizer.step()

        if captured is not None:
            break
        model.detach_membrane_states()

    if captured is None:
        raise RuntimeError("no timestep produced simultaneous nonzero W0 and W2 gradients")

    capture_timestep = int(captured["timestep"])
    input_timestep = captured["input_timestep"]
    logits_before = captured["logits_before"]
    target_logits = captured["target_logits"]
    hidden_spike = captured["hidden_spike"]
    target_hidden_spike = captured["target_hidden_spike"]
    input_trace = captured["input_trace"]
    hidden_trace = captured["hidden_trace"]
    hidden_membrane_pre = captured["hidden_membrane_pre"]
    hidden_membrane_post = captured["hidden_membrane_post"]
    hidden_surrogate = captured["hidden_surrogate"]
    target_input_trace = captured["target_input_trace"]
    target_hidden_trace = captured["target_hidden_trace"]
    tp_target_distribution = captured["tp_target_distribution"]
    tp_student_distribution = captured["tp_student_distribution"]
    w0_before = captured["w0_before"]
    w0_gradient = captured["w0_gradient"]
    w0_m_before = captured["w0_m_before"]
    w0_v_before = captured["w0_v_before"]
    w0_step_before = int(captured["w0_step_before"])
    w2_before = captured["w2_before"]
    w2_gradient = captured["w2_gradient"]
    w2_m_before = captured["w2_m_before"]
    w2_v_before = captured["w2_v_before"]
    w2_step_before = int(captured["w2_step_before"])
    w0_after = w0_parameter.detach().clone()
    w2_after = w2_parameter.detach().clone()
    s_after = s_parameter.detach().clone()
    if not torch.equal(s_before, s_after):
        raise RuntimeError("train_s=false changed target-projector S")

    w0_m, w0_v, w0_step = _optimizer_state(optimizers["layer_0"], w0_parameter)
    w2_m, w2_v, w2_step = _optimizer_state(optimizers["layer_1"], w2_parameter)
    expected_step = capture_timestep + 1
    if w0_step != expected_step or w2_step != expected_step:
        raise RuntimeError(
            f"expected Adam step {expected_step}, got W0={w0_step}, W2={w2_step}"
        )
    if w0_step_before != capture_timestep or w2_step_before != capture_timestep:
        raise RuntimeError(
            "optimizer history does not match the preceding dataset timesteps: "
            f"W0={w0_step_before}, W2={w2_step_before}, timestep={capture_timestep}"
        )

    model.reset_potential()
    with torch.no_grad():
        logits_after = None
        for timestep in range(capture_timestep + 1):
            logits_after = model.inference(
                frames[:, timestep, :].to(dtype=torch.float32)
            ).detach().clone()
        assert logits_after is not None

    arrays = {
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "labels": labels.numpy(),
        "input_timestep": input_timestep.numpy(),
        "target_one_hot": target_one_hot.numpy(),
        "input_trace": input_trace.numpy(),
        "hidden_spike": hidden_spike.numpy(),
        "hidden_trace": hidden_trace.numpy(),
        "hidden_membrane_pre": hidden_membrane_pre.numpy(),
        "hidden_membrane_post": hidden_membrane_post.numpy(),
        "hidden_surrogate": hidden_surrogate.numpy(),
        "target_input_trace": target_input_trace.numpy(),
        "target_hidden_spike": target_hidden_spike.numpy(),
        "target_hidden_trace": target_hidden_trace.numpy(),
        "tp_target_distribution": tp_target_distribution.numpy(),
        "tp_student_distribution": tp_student_distribution.numpy(),
        "logits_before": logits_before.detach().numpy(),
        "target_logits": target_logits.detach().numpy(),
        "w0_before": w0_before.numpy(),
        "w0_gradient": w0_gradient.numpy(),
        "w0_adam_m_before": w0_m_before.numpy(),
        "w0_adam_v_before": w0_v_before.numpy(),
        "w0_adam_m_after": w0_m.numpy(),
        "w0_adam_v_after": w0_v.numpy(),
        "w0_after": w0_after.numpy(),
        "w2_before": w2_before.numpy(),
        "w2_gradient": w2_gradient.numpy(),
        "w2_adam_m_before": w2_m_before.numpy(),
        "w2_adam_v_before": w2_v_before.numpy(),
        "w2_adam_m_after": w2_m.numpy(),
        "w2_adam_v_after": w2_v.numpy(),
        "w2_after": w2_after.numpy(),
        "s_fixed": s_before.numpy(),
        "logits_after": logits_after.numpy(),
        "screen_timesteps": np.asarray(
            [window["timestep"] for window in screen_windows], dtype=np.int64
        ),
        "screen_input_timestep": torch.stack(
            [window["input_timestep"] for window in screen_windows]
        ).numpy(),
        "screen_hidden_trace": torch.stack(
            [window["hidden_trace"] for window in screen_windows]
        ).numpy(),
        "screen_target_input_trace": torch.stack(
            [window["target_input_trace"] for window in screen_windows]
        ).numpy(),
        "screen_target_hidden_trace": torch.stack(
            [window["target_hidden_trace"] for window in screen_windows]
        ).numpy(),
        "screen_hidden_surrogate": torch.stack(
            [window["hidden_surrogate"] for window in screen_windows]
        ).numpy(),
        "screen_logits": torch.stack(
            [window["logits"] for window in screen_windows]
        ).numpy(),
        "screen_w0_gradient": torch.stack(
            [window["w0_gradient"] for window in screen_windows]
        ).numpy(),
        "screen_w2_gradient": torch.stack(
            [window["w2_gradient"] for window in screen_windows]
        ).numpy(),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)

    summary = {
        "status": "pass",
        "claim_boundary": (
            "Paper-source FP32 TP first active update for a deterministic mixed-label B16 batch. "
            "Execution starts at dataset timestep zero and includes all preceding online Adam steps; "
            "some earlier timesteps can update W0 before W2 becomes active. "
            "Adam m/v/step initially start at zero because the supplied checkpoint contains no optimizer state. "
            "This is a software reference, not fixed-point RTL equivalence or continued checkpoint training."
        ),
        "source_configuration": {
            "dataset": "N-MNIST train",
            "checkpoint": str(checkpoint_root),
            "hidden_dim": HIDDEN_DIM,
            "batch": batch_size,
            "captured_dataset_timestep": capture_timestep,
            "training_T_argument_semantics": "T=1 means optimizer step at every dataset timestep",
            "optimizer": "separate torch.optim.Adam per layer",
            "optimizer_state_initialization": "fresh zero m/v and step=0",
            "learning_rate": LEARNING_RATE,
            "train_s": False,
        },
        "batch": {
            "indices": sample_indices,
            "labels": [int(label) for label in labels.tolist()],
            "padded_dataset_timesteps": int(frames.shape[1]),
            "captured_input_nonzero": int(torch.count_nonzero(input_timestep).item()),
        },
        "gradient_history_through_capture": gradient_history,
        "softmax_screen_window_count": len(screen_windows),
        "observations": {
            "hidden_spike": tensor_stats(hidden_spike),
            "target_hidden_spike": tensor_stats(target_hidden_spike),
            "hidden_trace": tensor_stats(hidden_trace),
            "hidden_membrane_pre": tensor_stats(hidden_membrane_pre),
            "hidden_surrogate": tensor_stats(hidden_surrogate),
            "target_hidden_trace": tensor_stats(target_hidden_trace),
            "logits_before": tensor_stats(logits_before),
            "logits_after": tensor_stats(logits_after),
        },
        "updates": {
            "w0_gradient": tensor_stats(w0_gradient),
            "w0_delta": tensor_stats(w0_after - w0_before),
            "w0_adam_step_before": w0_step_before,
            "w0_adam_step_after": w0_step,
            "w2_gradient": tensor_stats(w2_gradient),
            "w2_delta": tensor_stats(w2_after - w2_before),
            "w2_adam_step_before": w2_step_before,
            "w2_adam_step_after": w2_step,
            "target_projector_gradient": None,
            "target_projector_changed_values": int(torch.count_nonzero(s_after - s_before).item()),
        },
        "artifacts": {"npz": str(output_npz), "summary": str(output_json)},
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, default=DEFAULT_TRACES_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_reference(
        traces_root=args.traces_root,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        output_npz=args.output_npz,
        output_json=args.output_json,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
