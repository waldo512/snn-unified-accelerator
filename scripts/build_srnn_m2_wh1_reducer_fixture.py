from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_HANDOFF = Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
ROOT = Path(__file__).resolve().parents[1]
BATCH = 64
HIDDEN = 256
BACKBONE = 450


def read_signed_hex(path: Path, bits: int, expected: int) -> list[int]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    if len(lines) != expected:
        raise RuntimeError(f"{path.name}: expected {expected} values, got {len(lines)}")
    sign = 1 << (bits - 1)
    modulus = 1 << bits
    values: list[int] = []
    for line in lines:
        raw = int(line, 16)
        values.append(raw - modulus if raw & sign else raw)
    return values


def round_shift_even(value: int, shift: int) -> int:
    negative = value < 0
    magnitude = -value if negative else value
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return -quotient if negative else quotient


def round_div_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (
        remainder * 2 == denominator and quotient & 1
    ):
        quotient += 1
    return quotient


def surrogate_factor_q24(membrane_q14: int) -> int:
    delta = membrane_q14 - 8192
    term_q28 = round_shift_even(delta * delta * 10_349_030, 20)
    denominator_q28 = (1 << 28) + term_q28
    return min(round_div_even(1 << 52, denominator_q28), 1 << 24)


def write_hex(path: Path, values: list[int], bits: int) -> None:
    mask = (1 << bits) - 1
    digits = bits // 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{value & mask:0{digits}x}\n" for value in values),
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact SRNN M2 W_h1 dpre/backbone streams and verify that "
            "their outer-product reduction reproduces the formal V25 gradient."
        )
    )
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    args = parser.parse_args()

    vector_dir = args.handoff / "tp_exp" / "vectors" / "m2_tp"
    error = read_signed_hex(vector_dir / "V25_error_q40.mem", 64, BATCH * BATCH)
    input_trace = read_signed_hex(
        vector_dir / "V25_input_trace_q14.mem", 32, BATCH * HIDDEN
    )
    target_trace = read_signed_hex(
        vector_dir / "V25_target_trace_q14.mem", 32, BATCH * HIDDEN
    )
    membrane = read_signed_hex(
        vector_dir / "V25_membrane_pre_q14.mem", 32, 2 * BATCH * HIDDEN
    )
    backbone = read_signed_hex(
        vector_dir / "V25_backbone_q14.mem", 32, 2 * BATCH * BACKBONE
    )
    expected = read_signed_hex(
        vector_dir / "V25_wh1_gradient_q40.mem", 64, HIDDEN * BACKBONE
    )

    dpre: list[int] = []
    surrogate_factor: list[int] = []
    for path in range(2):
        for sample in range(BATCH):
            for hidden in range(HIDDEN):
                accumulator = 0
                for column in range(BATCH):
                    error_index = (
                        sample * BATCH + column
                        if path == 0
                        else column * BATCH + sample
                    )
                    trace = (
                        target_trace[column * HIDDEN + hidden]
                        if path == 0
                        else input_trace[column * HIDDEN + hidden]
                    )
                    accumulator += error[error_index] * trace
                incoming_q40 = round_shift_even(accumulator, 14)
                membrane_index = (
                    path * BATCH * HIDDEN + sample * HIDDEN + hidden
                )
                factor_q24 = surrogate_factor_q24(membrane[membrane_index])
                surrogate_factor.append(factor_q24)
                dpre.append(round_shift_even(incoming_q40 * factor_q24, 24))

    for hidden in range(HIDDEN):
        for backbone_neuron in range(BACKBONE):
            accumulator = 0
            for path in range(2):
                for sample in range(BATCH):
                    dpre_index = path * BATCH * HIDDEN + sample * HIDDEN + hidden
                    backbone_index = (
                        path * BATCH * BACKBONE
                        + sample * BACKBONE
                        + backbone_neuron
                    )
                    accumulator += dpre[dpre_index] * backbone[backbone_index]
            address = hidden * BACKBONE + backbone_neuron
            actual = round_shift_even(accumulator, 14)
            if actual != expected[address]:
                raise RuntimeError(
                    f"W_h1 reference mismatch at {address}: "
                    f"expected {expected[address]}, got {actual}"
                )

    output_dir = ROOT / "tb" / "data"
    write_hex(output_dir / "srnn_m2_v25_error_q40.mem", error, 64)
    write_hex(output_dir / "srnn_m2_v25_input_trace_q14.mem", input_trace, 32)
    write_hex(output_dir / "srnn_m2_v25_target_trace_q14.mem", target_trace, 32)
    write_hex(output_dir / "srnn_m2_v25_wh1_dpre_q40.mem", dpre, 64)
    write_hex(
        output_dir / "srnn_m2_v25_wh1_surrogate_uq24.mem",
        surrogate_factor,
        32,
    )
    write_hex(output_dir / "srnn_m2_v25_backbone_q14.mem", backbone, 32)
    print(
        "SRNN_M2_WH1_REDUCER_FIXTURE_PASS "
        f"dpre={len(dpre)} surrogate={len(surrogate_factor)} "
        f"backbone={len(backbone)} gradients={len(expected)}"
    )


if __name__ == "__main__":
    main()
