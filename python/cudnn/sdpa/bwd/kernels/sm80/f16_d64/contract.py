# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 D64 half backward identity; no runtime imports."""

from ..half_contract import D64 as NAME, LEGACY_ROW

KERNEL_FILE = "sm80/f16_d64/kernel.py"
__all__ = ["KERNEL_FILE", "LEGACY_ROW", "NAME"]
