#!/usr/bin/env python3
"""Build a bounded fixed-point Conv8 TP gradient fixture.

The arithmetic order matches the retained SCNN handoff leaf, but the data
layout is emitted independently for direct differential testing and for the
new single-working-set tile replay engine.
"""

from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tb" / "data"
BATCH = 2
IN_CHANNELS = 2
OUT_CHANNELS = 2
H = 4
POOL_H = 2
K = 3
FRAC = 8


def qmul(a: int, b: int) -> int:
    return (a * b) >> FRAC


def trunc_div(value: int, divisor: int) -> int:
    return (1 if value >= 0 else -1) * (abs(value) // divisor)


def sat16(value: int) -> int:
    return max(-32768, min(32767, value))


def emit(name: str, values: list[int]) -> None:
    text = "\n".join(f"{value & 0xffff:04x}" for value in values) + "\n"
    (OUT / name).write_text(text, encoding="ascii")


def exp_score(delta: int) -> int:
    if delta >= 0:
        return 1 << FRAC
    if delta <= -(8 << FRAC):
        return 0
    magnitude = -delta
    denominator = (1 << FRAC) + magnitude + (
        ((magnitude * magnitude) >> FRAC) >> 1
    )
    return (1 << (2 * FRAC)) // denominator


def softmax_q8(scores: list[int]) -> list[int]:
    maximum = max(scores)
    exponentials = [exp_score(score - maximum) for score in scores]
    total = sum(exponentials)
    return [(value << FRAC) // total for value in exponentials]


def build_trace_modulation(
    student_trace: list[int],
    target_trace: list[int],
    prev_target_trace: list[int],
) -> list[int]:
    trace_size = OUT_CHANNELS * POOL_H * POOL_H
    prev_size = IN_CHANNELS * H * H
    z_scores: list[list[int]] = []
    y_scores: list[list[int]] = []
    for sample in range(BATCH):
        z_row: list[int] = []
        y_row: list[int] = []
        for pair in range(BATCH):
            dot = sum(
                qmul(
                    student_trace[sample * trace_size + index],
                    target_trace[pair * trace_size + index],
                )
                for index in range(trace_size)
            )
            z_row.append(trunc_div(dot, POOL_H * POOL_H))

            root_sum = 0
            for spatial in range(H * H):
                distance = sum(
                    ((
                        prev_target_trace[sample * prev_size + channel * H * H + spatial]
                        - prev_target_trace[pair * prev_size + channel * H * H + spatial]
                    ) ** 2) >> FRAC
                    for channel in range(IN_CHANNELS)
                )
                root_sum += math.isqrt(distance << FRAC)
            y_row.append(-((root_sum + H * H - 1) // (H * H)))
        z_scores.append(z_row)
        y_scores.append(y_row)

    modulation: list[int] = []
    for row in range(BATCH):
        student_probability = softmax_q8(z_scores[row])
        target_probability = softmax_q8(y_scores[row])
        modulation.extend(
            student - target
            for student, target in zip(student_probability, target_probability)
        )
    return modulation


def build_gradients(
    modulation: list[int],
    student_input: list[int],
    target_input: list[int],
    student_trace: list[int],
    target_trace: list[int],
    student_surrogate: list[int],
    target_surrogate: list[int],
) -> tuple[list[int], list[int]]:
    input_size = IN_CHANNELS * H * H
    local_size = OUT_CHANNELS * H * H
    trace_size = OUT_CHANNELS * POOL_H * POOL_H

    student_post = [0] * (BATCH * local_size)
    target_post = [0] * (BATCH * local_size)
    for sample in range(BATCH):
        for local in range(local_size):
            oc = local // (H * H)
            pos = local % (H * H)
            row = pos // H
            col = pos % H
            pool = oc * POOL_H * POOL_H + (row // 2) * POOL_H + col // 2

            factor_s = sum(
                qmul(modulation[sample * BATCH + pair],
                     target_trace[pair * trace_size + pool])
                for pair in range(BATCH)
            )
            post_s = qmul(factor_s,
                          student_surrogate[sample * local_size + local])
            student_post[sample * local_size + local] = sat16(
                trunc_div(trunc_div(post_s, POOL_H * POOL_H), 4)
            )

            factor_t = sum(
                qmul(modulation[pair * BATCH + sample],
                     student_trace[pair * trace_size + pool])
                for pair in range(BATCH)
            )
            post_t = qmul(factor_t,
                          target_surrogate[sample * local_size + local])
            target_post[sample * local_size + local] = sat16(
                trunc_div(trunc_div(post_t, POOL_H * POOL_H), 4)
            )

    gradients: list[int] = []
    signs: list[int] = []
    for oc in range(OUT_CHANNELS):
        for ic in range(IN_CHANNELS):
            for kr in range(K):
                for kc in range(K):
                    total = 0
                    for sample in range(BATCH):
                        for out_row in range(H):
                            for out_col in range(H):
                                in_row = out_row + kr - 1
                                in_col = out_col + kc - 1
                                if not (0 <= in_row < H and 0 <= in_col < H):
                                    continue
                                local = oc * H * H + out_row * H + out_col
                                source = ic * H * H + in_row * H + in_col
                                total += qmul(
                                    student_post[sample * local_size + local],
                                    student_input[sample * input_size + source],
                                )
                                total += qmul(
                                    target_post[sample * local_size + local],
                                    target_input[sample * input_size + source],
                                )
                    gradient = sat16(trunc_div(total, BATCH))
                    gradients.append(gradient)
                    signs.append(1 if gradient > 0 else -1 if gradient < 0 else 0)
    return gradients, signs


def main() -> int:
    input_size = IN_CHANNELS * H * H
    local_size = OUT_CHANNELS * H * H
    trace_size = OUT_CHANNELS * POOL_H * POOL_H

    student_input = [
        256 if ((i * 3 + b * 5) % 7) in (0, 1) else 0
        for b in range(BATCH) for i in range(input_size)
    ]
    target_input = [
        256 if ((i * 5 + b * 2 + 1) % 11) in (0, 3, 4) else 0
        for b in range(BATCH) for i in range(input_size)
    ]
    student_trace = [
        32 + ((i * 37 + b * 19) % 321)
        for b in range(BATCH) for i in range(trace_size)
    ]
    target_trace = [
        24 + ((i * 29 + b * 43) % 287)
        for b in range(BATCH) for i in range(trace_size)
    ]
    student_surrogate = [
        48 + ((i * 17 + b * 23) % 209)
        for b in range(BATCH) for i in range(local_size)
    ]
    target_surrogate = [
        40 + ((i * 13 + b * 31) % 193)
        for b in range(BATCH) for i in range(local_size)
    ]
    # Row-major M[b_student, b_target], signed Q8.8.
    modulation = [192, -96, 64, -160]

    student_post = [0] * (BATCH * local_size)
    target_post = [0] * (BATCH * local_size)
    for sample in range(BATCH):
        for local in range(local_size):
            oc = local // (H * H)
            pos = local % (H * H)
            row = pos // H
            col = pos % H
            pool = oc * POOL_H * POOL_H + (row // 2) * POOL_H + col // 2

            factor_s = sum(
                qmul(modulation[sample * BATCH + pair],
                     target_trace[pair * trace_size + pool])
                for pair in range(BATCH)
            )
            post_s = qmul(factor_s,
                          student_surrogate[sample * local_size + local])
            post_s = trunc_div(trunc_div(post_s, POOL_H * POOL_H), 4)
            student_post[sample * local_size + local] = sat16(post_s)

            factor_t = sum(
                qmul(modulation[pair * BATCH + sample],
                     student_trace[pair * trace_size + pool])
                for pair in range(BATCH)
            )
            post_t = qmul(factor_t,
                          target_surrogate[sample * local_size + local])
            post_t = trunc_div(trunc_div(post_t, POOL_H * POOL_H), 4)
            target_post[sample * local_size + local] = sat16(post_t)

    gradients: list[int] = []
    signs: list[int] = []
    for oc in range(OUT_CHANNELS):
        for ic in range(IN_CHANNELS):
            for kr in range(K):
                for kc in range(K):
                    total = 0
                    for sample in range(BATCH):
                        for out_row in range(H):
                            for out_col in range(H):
                                in_row = out_row + kr - 1
                                in_col = out_col + kc - 1
                                if not (0 <= in_row < H and 0 <= in_col < H):
                                    continue
                                local = oc * H * H + out_row * H + out_col
                                source = ic * H * H + in_row * H + in_col
                                total += qmul(
                                    student_post[sample * local_size + local],
                                    student_input[sample * input_size + source],
                                )
                                total += qmul(
                                    target_post[sample * local_size + local],
                                    target_input[sample * input_size + source],
                                )
                    gradient = sat16(trunc_div(total, BATCH))
                    gradients.append(gradient)
                    signs.append(1 if gradient > 0 else -1 if gradient < 0 else 0)

    if len(gradients) != OUT_CHANNELS * IN_CHANNELS * K * K:
        raise RuntimeError("gradient count mismatch")
    if not ({-1, 0, 1} <= set(signs)):
        raise RuntimeError(f"fixture does not cover all gradient signs: {set(signs)}")

    generated_modulation = build_trace_modulation(
        student_trace, target_trace, target_input
    )
    generated_gradients, generated_signs = build_gradients(
        generated_modulation,
        student_input,
        target_input,
        student_trace,
        target_trace,
        student_surrogate,
        target_surrogate,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    emit("scnn_conv8_tile_student_input_q8_8.mem", student_input)
    emit("scnn_conv8_tile_target_input_q8_8.mem", target_input)
    emit("scnn_conv8_tile_student_trace_q8_8.mem", student_trace)
    emit("scnn_conv8_tile_target_trace_q8_8.mem", target_trace)
    emit("scnn_conv8_tile_student_surrogate_q8_8.mem", student_surrogate)
    emit("scnn_conv8_tile_target_surrogate_q8_8.mem", target_surrogate)
    emit("scnn_conv8_tile_modulation_q8_8.mem", modulation)
    emit("scnn_conv8_tile_gradient_q8_8.mem", gradients)
    emit("scnn_conv8_tile_gradient_sign.mem", signs)
    emit("scnn_conv8_tile_generated_modulation_q8_8.mem", generated_modulation)
    emit("scnn_conv8_tile_generated_gradient_q8_8.mem", generated_gradients)
    emit("scnn_conv8_tile_generated_gradient_sign.mem", generated_signs)
    print(
        "SCNN_CONV8_TILE_FIXED_POINT_GOLDEN_PASS "
        f"weights={len(gradients)} positive={signs.count(1)} "
        f"negative={signs.count(-1)} zero={signs.count(0)} "
        f"generated_modulation={generated_modulation} "
        f"generated_positive={generated_signs.count(1)} "
        f"generated_negative={generated_signs.count(-1)} "
        f"generated_zero={generated_signs.count(0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
