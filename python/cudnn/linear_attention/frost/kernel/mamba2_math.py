# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared FP32 scalar functions for the standalone SSD kernels."""

import cutlass.cute as cute


@cute.jit
def softplus(value):
    # Direct log(1+exp(x)) rounds to zero in the negative tail. A short
    # log1p series has relative truncation error <4e-10 for x < -4.
    result = value
    if value < -4.0:
        e = cute.math.exp(value, fastmath=True)
        result = e * (1.0 + e * (-0.5 + e * (1.0 / 3.0 + e * (-0.25 + e * 0.2))))
    elif value <= 20.0:
        result = cute.math.log(1.0 + cute.math.exp(value, fastmath=True), fastmath=True)
    return result
