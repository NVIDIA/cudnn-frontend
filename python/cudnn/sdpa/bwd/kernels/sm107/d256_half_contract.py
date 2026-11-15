# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legal fixed bodies of the SM107 D256 half graph row."""

from .d256_f16_4x1.contract import NAME as NATIVE
from ..sm100.d256_f16_2x2.contract import SM107_D256 as TWO_BY_TWO

LEGACY_ROW = "sdpa_bwd_sm107"


def select_implementation(facts):
    """Keep the native body first; the two-by-two diagnostic body stays opt-in."""
    return (NATIVE,) if facts.thd else (NATIVE, TWO_BY_TWO)
