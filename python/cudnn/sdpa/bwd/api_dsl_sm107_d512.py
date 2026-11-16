# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compatibility import for the fixed SM107 D512 backward adapter."""

from .kernels.sm107.d512_f16_2x2.host import SdpaBwdDslSm107D512
from .kernels.sm107.d512_f16_2x2.contract import KERNEL_FILE as _SM107_STAGE2_FILE_2X2, STAGE2_TAG as _SM107_STAGE2_TAG_2X2

__all__ = ["SdpaBwdDslSm107D512", "_SM107_STAGE2_FILE_2X2", "_SM107_STAGE2_TAG_2X2"]
