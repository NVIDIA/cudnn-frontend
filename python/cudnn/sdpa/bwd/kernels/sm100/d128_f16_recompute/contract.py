# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity of the single-launch D64/D128 half backward implementation."""

from cudnn.sdpa.bwd.d128_selection import LEGACY_ROW, RECOMPUTE_OWNER as NAME

KERNEL_FILE = "sm100/d128_f16_recompute/kernel.py"

__all__ = ["KERNEL_FILE", "LEGACY_ROW", "NAME"]
