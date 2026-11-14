from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_HANDOFF = Path(r"D:\Desktop\SHD_M2_WH1_TP_ONLINE_HANDOFF_20260829")
ROOT = Path(__file__).resolve().parents[1]


def copy_checked(
    source: Path, destination: Path, expected_lines: int, hex_digits: int = 16
) -> None:
    lines = [line.strip().lower() for line in source.read_text().splitlines()]
    if len(lines) != expected_lines:
        raise RuntimeError(
            f"{source.name}: expected {expected_lines} values, got {len(lines)}"
        )
    if any(
        len(line) != hex_digits
        or any(c not in "0123456789abcdef" for c in line)
        for line in lines
    ):
        raise RuntimeError(
            f"{source.name}: expected one {hex_digits * 4}-bit hexadecimal value per line"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the formal SHD M2 Q40 gradient streams used by the unified adapter test."
    )
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    args = parser.parse_args()

    vector_dir = args.handoff / "tp_exp" / "vectors" / "m2_tp"
    output_dir = ROOT / "tb" / "data"
    copy_checked(
        vector_dir / "V25_wh1_gradient_q40.mem",
        output_dir / "srnn_m2_v25_wh1_gradient_q40.mem",
        115_200,
    )
    copy_checked(
        vector_dir / "V26_wout_gradient_q40.mem",
        output_dir / "srnn_m2_v26_wout_gradient_q40.mem",
        5_120,
    )
    copy_checked(
        vector_dir / "V26_ce_error_q40.mem",
        output_dir / "srnn_m2_v26_ce_error_q40.mem",
        1_280,
    )
    copy_checked(
        vector_dir / "V26_ce_gradient_trace_q14.mem",
        output_dir / "srnn_m2_v26_ce_gradient_trace_q14.mem",
        16_384,
        hex_digits=8,
    )
    print(
        "SRNN_M2_GRADIENT_STREAM_FIXTURE_PASS "
        "wh1=115200 wout=5120 ce_error=1280 hidden_trace=16384"
    )


if __name__ == "__main__":
    main()
