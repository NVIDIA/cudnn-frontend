# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 fused half backward identity; no runtime imports."""

from ..half_contract import FUSED as NAME, LEGACY_ROW

KERNEL_FILE = "sm120/f16_fused/kernel.py"
__all__ = ["KERNEL_FILE", "LEGACY_ROW", "NAME"]
