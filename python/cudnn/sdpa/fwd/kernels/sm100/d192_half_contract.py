# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Half graph domain covered by the D192/V128 geometry."""

from cudnn import data_type
from .half_flavor_contract import padded_shape


def in_domain(facts):
    return (
        facts.device_cc in ((10, 0), (10, 3))
        and padded_shape(facts) == (192, 128)
        and not (facts.invalid or facts.is_backward or facts.is_fp8 or facts.is_mxfp8)
        and facts.dtype in (data_type.HALF, data_type.BFLOAT16)
        and facts.uniform_dtype
    )
