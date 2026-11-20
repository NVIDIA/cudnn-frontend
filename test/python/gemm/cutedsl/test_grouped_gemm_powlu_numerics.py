# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture-independent PowLU numerical-boundary checks."""

import math
import struct

import pytest

pytestmark = pytest.mark.L0

FP32_MIN_NORMAL = 2.0**-126


def _bf16_value(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits << 16))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", value))[0]


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value) if value < 750.0 else 0.0
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value) if value > -750.0 else 0.0
    return exp_pos / (1.0 + exp_pos)


def _powlu_value_and_derivative(gate: float, power_m: float = 2.5, positive_floor: float = 0.0) -> tuple[float, float]:
    sigmoid = _sigmoid(gate)
    sigmoid_neg = _sigmoid(-gate)
    if gate <= 0.0:
        return gate * sigmoid, sigmoid * (1.0 + gate * sigmoid_neg)

    safe_gate = max(gate, positive_floor)
    sqrt_gate = math.sqrt(safe_gate)
    denominator = sqrt_gate + 1.0
    exponent = power_m / denominator
    log_gate = math.log(safe_gate)
    power = math.exp(exponent * log_gate)
    value = power * sigmoid
    exponent_grad = -power_m / (2.0 * sqrt_gate * denominator * denominator)
    derivative = math.exp((exponent - 1.0) * log_gate) * exponent * sigmoid + value * exponent_grad * log_gate + value * sigmoid_neg
    return value, derivative


def test_powlu_m25_is_finite_for_every_finite_bfloat16_gate():
    values = []
    derivatives = []
    for bits in range(0x10000):
        if bits & 0x7F80 == 0x7F80:
            continue
        value, derivative = _powlu_value_and_derivative(_bf16_value(bits))
        assert math.isfinite(value)
        assert math.isfinite(derivative)
        values.append(value)
        derivatives.append(derivative)

    assert -0.279 < min(values) <= 0.0
    assert 4.02 < max(values) < 4.03
    assert -0.101 < min(derivatives) < 0.0
    assert 1.12 < max(derivatives) < 1.13


def test_powlu_fp32_min_normal_floor_is_lossless_for_m25_bfloat16_subnormals():
    for bits in range(1, 0x80):
        gate = _bf16_value(bits)
        reference = _powlu_value_and_derivative(gate)
        clamped = _powlu_value_and_derivative(gate, positive_floor=FP32_MIN_NORMAL)
        assert tuple(map(_f32_bits, reference)) == tuple(map(_f32_bits, clamped))


def test_powlu_m2_positive_bfloat16_subnormal_boundary():
    for bits in range(1, 0x81):
        gate = _bf16_value(bits)
        value, derivative = _powlu_value_and_derivative(gate, power_m=2.0)
        assert _f32_bits(value) == _f32_bits(0.0)
        assert _f32_bits(derivative) == _f32_bits(gate)


def test_powlu_coarse_floor_changes_the_smallest_bfloat16_subnormal():
    gate = _bf16_value(1)
    reference = _powlu_value_and_derivative(gate)
    clamped = _powlu_value_and_derivative(gate, positive_floor=1.0e-12)
    assert tuple(map(_f32_bits, reference)) != tuple(map(_f32_bits, clamped))
