# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 fused half backward identity; no runtime imports."""

NAME = "sdpa_bwd_sm120_f16_fused"
KERNEL_FILE = "sm120/f16_fused/kernel.py"
LEGACY_ROW = "sdpa_bwd_sm120"
# The row's other bodies, still served by the row (both run the SM80 two-subgroup body compiled for SM120).
WIDE = "sdpa_bwd_sm120_f16_wide"
PACKED = "sdpa_bwd_sm120_f16_packed"


def select_implementation(facts):
    """The body the SM120 half row runs for a declaration; ``lower_dsl_bwd`` dispatches on it."""
    if facts.d_qk > 256:
        return WIDE
    if facts.thd:
        return PACKED
    return NAME
