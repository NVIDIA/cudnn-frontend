# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which SM120 half backward body a declaration runs; no runtime imports."""

LEGACY_ROW = "sdpa_bwd_sm120"
FUSED = "sdpa_bwd_sm120_f16_fused"
# The SM80 two-subgroup body compiled for SM120: THD graphs and D_QK > 256.
TWO_SUBGROUP = "sdpa_bwd_sm120_f16_two_subgroup"


def select_implementation(facts):
    """The body the SM120 half row runs for a declaration; lower_dsl_bwd dispatches on it."""
    return TWO_SUBGROUP if facts.thd or facts.d_qk > 256 else FUSED
