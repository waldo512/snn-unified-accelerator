from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tb" / "data"
RESULT_DIR = ROOT / "results" / "learning_reference"
ELEMENTS = 256 * 20
INCREMENT = 31
THRESHOLD = 65
CODE_MIN = -(1 << 15)
CODE_MAX = (1 << 15) - 1


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
    path.write_text(
        "".join(f"{value & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def direct_update(code: int, residual: int, gradient: int) -> tuple[int, int, bool]:
    if gradient == 0:
        return code, residual, False

    direction = -1 if gradient > 0 else 1
    normalized = residual + direction * INCREMENT
    code_delta = 0
    if normalized >= THRESHOLD:
        normalized -= THRESHOLD
        code_delta += 1
        if normalized >= THRESHOLD:
            normalized -= THRESHOLD
            code_delta += 1
    elif normalized <= -THRESHOLD:
        normalized += THRESHOLD
        code_delta -= 1
        if normalized <= -THRESHOLD:
            normalized += THRESHOLD
            code_delta -= 1

    candidate = code + code_delta
    if candidate > CODE_MAX:
        return CODE_MAX, 0, True
    if candidate < CODE_MIN:
        return CODE_MIN, 0, True
    return candidate, normalized, False


def apply_window(
    codes: list[int], residuals: list[int], gradients: list[int]
) -> tuple[list[int], list[int], int, int]:
    next_codes: list[int] = []
    next_residuals: list[int] = []
    changes = 0
    saturations = 0
    for code, residual, gradient in zip(codes, residuals, gradients, strict=True):
        updated_code, updated_residual, saturated = direct_update(
            code, residual, gradient
        )
        next_codes.append(updated_code)
        next_residuals.append(updated_residual)
        changes += updated_code != code
        saturations += saturated
    return next_codes, next_residuals, changes, saturations


def main() -> None:
    initial = read_signed_hex(
        DATA_DIR / "srnn_m2_v28_wout_initial_q14.mem", 16, ELEMENTS
    )
    gradients = read_signed_hex(
        DATA_DIR / "srnn_m2_v26_wout_gradient_q40.mem", 64, ELEMENTS
    )
    expected_window1_code = read_signed_hex(
        DATA_DIR / "srnn_m2_v26_wout_direct_code_expected_q14.mem", 16, ELEMENTS
    )
    expected_window1_residual = read_signed_hex(
        DATA_DIR / "srnn_m2_v26_wout_direct_residual_expected.mem", 16, ELEMENTS
    )

    zero_residuals = [0] * ELEMENTS
    window1_code, window1_residual, changes1, saturations1 = apply_window(
        initial, zero_residuals, gradients
    )
    if window1_code != expected_window1_code or window1_residual != expected_window1_residual:
        raise RuntimeError("existing one-window Wout fixture disagrees with reference update")

    window2_code, window2_residual, changes2, saturations2 = apply_window(
        window1_code, window1_residual, gradients
    )
    write_hex(
        DATA_DIR / "srnn_m2_v26_wout_two_window_code_expected_q14.mem",
        window2_code,
        16,
    )
    write_hex(
        DATA_DIR / "srnn_m2_v26_wout_two_window_residual_expected.mem",
        window2_residual,
        16,
    )

    nonzero = sum(gradient != 0 for gradient in gradients)
    summary = {
        "profile": "SHD_M2_B64_T1_Wout_only_two_windows",
        "batch": 64,
        "hidden": 256,
        "classes": 20,
        "elements": ELEMENTS,
        "contributions_per_window": 64 * ELEMENTS,
        "nonzero_gradients_per_window": nonzero,
        "zero_gradients_per_window": ELEMENTS - nonzero,
        "increment": INCREMENT,
        "threshold": THRESHOLD,
        "window1_code_changes": changes1,
        "window2_code_changes": changes2,
        "window1_saturations": saturations1,
        "window2_saturations": saturations2,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "srnn_m2_wout_two_window_fixture_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(
        "SRNN_M2_WOUT_TWO_WINDOW_FIXTURE_PASS "
        f"elements={ELEMENTS} nonzero={nonzero} zero={ELEMENTS - nonzero} "
        f"code_changes={changes1}/{changes2} saturations={saturations1}/{saturations2}"
    )


if __name__ == "__main__":
    main()
