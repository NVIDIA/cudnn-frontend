# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Vocabulary shared by kernel templates and data-only configs."""

MASK_NONE = 0
MASK_PADDED = 1 << 0
MASK_CAUSAL = 1 << 1
MASK_SWA = 1 << 2

SCHED_NATURAL = 0
SCHED_LPT = 1
SCHED_LPT_L2 = 2

DTYPE_E4M3 = 0
DTYPE_E5M2 = 1
DTYPE_BF16 = 2
DTYPE_FP16 = 3
