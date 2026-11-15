# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Graph domain of the small-head half implementation migration."""

from cudnn import data_type


def in_domain(facts):
    mixed_kv = (facts.d_qk, facts.d_v) == (128, 128) and facts.kv_dtype in (data_type.FP8_E4M3, data_type.FP8_E5M2) and facts.dtype_o == facts.dtype
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and (facts.d_qk, facts.d_v) in ((32, 32), (64, 64), (128, 128))
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and (facts.uniform_dtype or mixed_kv)
    )
