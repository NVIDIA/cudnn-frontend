# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 two-subgroup half backward identity; no runtime imports."""

from ..half_contract import LEGACY_ROW, TWO_SUBGROUP as NAME

KERNEL_FILE = "sm80/f16_two_subgroup/kernel.py"
__all__ = ["KERNEL_FILE", "LEGACY_ROW", "NAME"]
