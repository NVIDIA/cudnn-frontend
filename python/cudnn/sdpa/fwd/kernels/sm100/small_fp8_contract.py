# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared graph domain for exact small-head per-tensor FP8 implementations."""

from cudnn import data_type


def in_domain(facts):
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and (facts.d_qk, facts.d_v) in ((64, 64), (128, 128))
        and not (facts.invalid or facts.is_backward)
        and facts.is_fp8
        and not facts.is_mxfp8
        and facts.dtype in (data_type.FP8_E4M3, data_type.FP8_E5M2)
        and facts.uniform_dtype
    )
