from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_HANDOFF = Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
ROOT = Path(__file__).resolve().parents[1]
WH1_COUNT = 450 * 256
WOUT_COUNT = 256 * 20
CODE_MIN = -(1 << 15)
CODE_MAX = (1 << 15) - 1
WH1_INCREMENT = 25
WH1_THRESHOLD = 488
WOUT_INCREMENT = 31
WOUT_THRESHOLD = 65


def read_signed_hex(path: Path, bits: int, expected: int) -> list[int]:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    if len(lines) != expected:
        raise RuntimeError(f"{path.name}: expected {expected} values, got {len(lines)}")
    sign = 1 << (bits - 1)
    modulus = 1 << bits
    values: list[int] = []
    for line in lines:
        raw = int(line, 16)
        values.append(raw - modulus if raw & sign else raw)
    return values


def write_hex(path: Path, values: list[int], bits: int) -> None:
    mask = (1 << bits) - 1
    digits = bits // 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{value & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def direct_update(
    code: int,
    residual: int,
    gradient: int,
    increment: int,
    threshold: int,
) -> tuple[int, int, bool]:
    if gradient == 0:
        return code, residual, False

    direction = -1 if gradient > 0 else 1
    normalized = residual + direction * increment
    code_delta = 0
    if normalized >= threshold:
        normalized -= threshold
        code_delta += 1
        if normalized >= threshold:
            normalized -= threshold
            code_delta += 1
    elif normalized <= -threshold:
        normalized += threshold
        code_delta -= 1
        if normalized <= -threshold:
            normalized += threshold
            code_delta -= 1

    candidate = code + code_delta
    if candidate > CODE_MAX:
        return CODE_MAX, 0, True
    if candidate < CODE_MIN:
        return CODE_MIN, 0, True
    return candidate, normalized, False


def build_tensor(
    initial: list[int],
    gradients: list[int],
    increment: int,
    threshold: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    expected_code: list[int] = []
    expected_residual: list[int] = []
    nonzero = 0
    code_changes = 0
    positive_gradients = 0
    negative_gradients = 0
    saturations = 0

    for code, gradient in zip(initial, gradients, strict=True):
        updated_code, updated_residual, saturated = direct_update(
            code, 0, gradient, increment, threshold
        )
        expected_code.append(updated_code)
        expected_residual.append(updated_residual)
        if gradient != 0:
            nonzero += 1
        if gradient > 0:
            positive_gradients += 1
        elif gradient < 0:
            negative_gradients += 1
        if updated_code != code:
            code_changes += 1
        if saturated:
            saturations += 1

    return expected_code, expected_residual, {
        "elements": len(initial),
        "nonzero_gradients": nonzero,
        "zero_gradients": len(initial) - nonzero,
        "positive_gradients": positive_gradients,
        "negative_gradients": negative_gradients,
        "code_changes": code_changes,
        "saturations": saturations,
        "increment": increment,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full SHD M2 direct-residual parameter-store fixture from "
            "the already verified V25/V26 gradients and real V28 Q2.14 weights."
        )
    )
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    args = parser.parse_args()

    data_dir = ROOT / "tb" / "data"
    vector_dir = args.handoff / "tp_exp" / "vectors" / "m2_tp"
    wh1_initial = read_signed_hex(
        vector_dir / "V28_wh_q14_initial.mem", 16, WH1_COUNT
    )
    wout_initial = read_signed_hex(
        vector_dir / "V28_wo_q14_initial.mem", 16, WOUT_COUNT
    )
    wh1_gradient = read_signed_hex(
        data_dir / "srnn_m2_v25_wh1_gradient_q40.mem", 64, WH1_COUNT
    )
    wout_gradient = read_signed_hex(
        data_dir / "srnn_m2_v26_wout_gradient_q40.mem", 64, WOUT_COUNT
    )

    wh1_code, wh1_residual, wh1_summary = build_tensor(
        wh1_initial,
        wh1_gradient,
        increment=WH1_INCREMENT,
        threshold=WH1_THRESHOLD,
    )
    wout_code, wout_residual, wout_summary = build_tensor(
        wout_initial,
        wout_gradient,
        increment=WOUT_INCREMENT,
        threshold=WOUT_THRESHOLD,
    )

    write_hex(data_dir / "srnn_m2_v28_wh1_initial_q14.mem", wh1_initial, 16)
    write_hex(data_dir / "srnn_m2_v28_wout_initial_q14.mem", wout_initial, 16)
    write_hex(data_dir / "srnn_m2_v25_wh1_direct_code_expected_q14.mem", wh1_code, 16)
    write_hex(
        data_dir / "srnn_m2_v25_wh1_direct_residual_expected.mem",
        wh1_residual,
        16,
    )
    write_hex(data_dir / "srnn_m2_v26_wout_direct_code_expected_q14.mem", wout_code, 16)
    write_hex(
        data_dir / "srnn_m2_v26_wout_direct_residual_expected.mem",
        wout_residual,
        16,
    )

    summary = {
        "profile": "SHD_M2_B64_single_window_direct_residual",
        "code_format": "signed Q2.14",
        "gradient_format": "signed Q40",
        "initial_residual": 0,
        "code_min": CODE_MIN,
        "code_max": CODE_MAX,
        "wh1": wh1_summary,
        "wout": wout_summary,
        "total_elements": WH1_COUNT + WOUT_COUNT,
        "total_nonzero_gradients": (
            wh1_summary["nonzero_gradients"] + wout_summary["nonzero_gradients"]
        ),
        "total_zero_gradients": (
            wh1_summary["zero_gradients"] + wout_summary["zero_gradients"]
        ),
    }
    result_path = (
        ROOT
        / "results"
        / "learning_reference"
        / "srnn_m2_b64_direct_backend_fixture_summary.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")

    print(
        "SRNN_M2_DIRECT_BACKEND_FIXTURE_PASS "
        f"elements={summary['total_elements']} "
        f"updates={summary['total_nonzero_gradients']} "
        f"zeros={summary['total_zero_gradients']} "
        f"code_changes={wh1_summary['code_changes'] + wout_summary['code_changes']} "
        f"saturations={wh1_summary['saturations'] + wout_summary['saturations']}"
    )


if __name__ == "__main__":
    main()
