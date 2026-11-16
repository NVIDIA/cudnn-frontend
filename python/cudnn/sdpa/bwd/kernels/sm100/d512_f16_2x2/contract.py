# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity of the two-by-two stage-2 implementation; no runtime imports."""

from ..d512_half_contract import LEGACY_ROW, TWO_BY_TWO as NAME

KERNEL_FILE = "sm100/d512_f16_2x2/kernel.py"
STAGE2_TAG = NAME + "_stage2"

__all__ = ["KERNEL_FILE", "LEGACY_ROW", "NAME", "STAGE2_TAG"]
