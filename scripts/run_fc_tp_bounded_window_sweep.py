#!/usr/bin/env python3
"""Sweep bounded direct-sign adaptation windows on shifted N-MNIST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fc_tp_h400_b16_reference import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_TRACES_ROOT,
)
from scripts.run_fc_tp_supervised_shift_adaptation import (  # noqa: E402
    adapt_candidate,
    evaluate,
    interleaved_class_indices,
    load_dataset_batches,
)
from scripts.screen_fc_tp_update_candidates import (  # noqa: E402
    Candidate,
    build_model,
)


DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "learning_reference"
    / "fc_tp_h400_bounded_window_sweep.json"
)


def parse_int_list(value: str, *, positive: bool) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("list must contain at least one integer")
    if positive and any(item <= 0 for item in values):
        raise ValueError("window sizes must be positive")
    return values


def compact_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in ("correct", "total", "accuracy_percent", "mean_batch_logits_l2")
    }


def run_sweep(
    *,
    traces_root: Path,
    checkpoint_root: Path,
    data_root: Path,
    output: Path,
    batch_size: int,
    test_samples_per_class: int,
    window_samples_per_class: tuple[int, ...],
    step_exponents: tuple[int, ...],
) -> dict[str, Any]:
    torch.manual_seed(38472)
    torch.use_deterministic_algorithms(True)

    if str(traces_root) not in sys.path:
        sys.path.insert(0, str(traces_root))
    import tonic  # pylint: disable=import-outside-toplevel

    train_dataset = tonic.datasets.NMNIST(save_to=str(data_root), train=True)
    test_dataset = tonic.datasets.NMNIST(save_to=str(data_root), train=False)
    test_indices = interleaved_class_indices(
        test_dataset.targets,
        samples_per_class=test_samples_per_class,
    )
    test_batches = load_dataset_batches(
        traces_root=traces_root,
        data_root=data_root,
        train=False,
        indices=test_indices,
        batch_size=batch_size,
    )
    baseline_model = build_model(traces_root, checkpoint_root, batch_size=batch_size)
    baseline_clean = evaluate(baseline_model, test_batches, shifted=False)
    baseline_shifted = evaluate(baseline_model, test_batches, shifted=True)

    results: list[dict[str, Any]] = []
    for samples_per_class in window_samples_per_class:
        adapt_indices = interleaved_class_indices(
            train_dataset.targets,
            samples_per_class=samples_per_class,
            start_replica=2,
        )
        adapt_batches = load_dataset_batches(
            traces_root=traces_root,
            data_root=data_root,
            train=True,
            indices=adapt_indices,
            batch_size=batch_size,
        )
        for exponent in step_exponents:
            candidate = Candidate(
                f"direct_sign_step_2m{abs(exponent)}",
                "direct_sign",
                2.0**exponent,
            )
            result = adapt_candidate(
                candidate,
                traces_root=traces_root,
                checkpoint_root=checkpoint_root,
                adapt_batches=adapt_batches,
                test_batches=test_batches,
                baseline_clean=baseline_clean,
                baseline_shifted=baseline_shifted,
            )
            result["window_samples_per_class"] = samples_per_class
            result["step_exponent"] = exponent
            result["clean_after"] = compact_evaluation(result["clean_after"])
            result["shifted_after"] = compact_evaluation(result["shifted_after"])
            results.append(result)

    summary = {
        "status": "pass",
        "claim_boundary": (
            "Deterministic FP32 supervised N-MNIST right-2 shift sweep for "
            "selecting a bounded direct-sign adaptation schedule. It is not "
            "fixed-point RTL or statistical-significance evidence."
        ),
        "source": {
            "batch_size": batch_size,
            "test_samples_per_class": test_samples_per_class,
            "held_out_test_samples": len(test_indices),
            "window_samples_per_class": list(window_samples_per_class),
            "step_exponents": list(step_exponents),
        },
        "baseline_clean": compact_evaluation(baseline_clean),
        "baseline_shifted": compact_evaluation(baseline_shifted),
        "results": results,
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
    parser.add_argument("--test-samples-per-class", type=int, default=100)
    parser.add_argument("--window-samples-per-class", default="16,32,64,100")
    parser.add_argument("--step-exponents", default="-15,-16,-17")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sweep(
        traces_root=args.traces_root,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        output=args.output,
        batch_size=args.batch_size,
        test_samples_per_class=args.test_samples_per_class,
        window_samples_per_class=parse_int_list(
            args.window_samples_per_class, positive=True
        ),
        step_exponents=parse_int_list(args.step_exponents, positive=False),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "runs": len(result["results"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
