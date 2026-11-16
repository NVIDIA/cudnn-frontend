# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for the fixed SM100 D192/D256 two-by-two adapter."""

from .kernels.sm100.d256_f16_2x2.host_sm100 import SdpaBwdDslSm100D256

__all__ = ["SdpaBwdDslSm100D256"]
