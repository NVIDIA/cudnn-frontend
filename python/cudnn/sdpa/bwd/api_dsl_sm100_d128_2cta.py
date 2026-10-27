# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Direct-adapter entry for the d128 backward on the cta_group::2 datapath (grouped THD only, bring-up / probe route).

The shipping adapter carries the typed per-S heuristic field ``SdpaBwdDslSm100D128.cta_pair`` ("auto" = the default: the pair body at
S >= ``CTA_PAIR_MIN_SEQ`` on the measured part, else the one-CTA body | "off" | "on").  This subclass is that field pinned to ``"on"``: the
pair body at ANY S, every pair predicate declining BY NAME -- what a probe driver imports to validate and time the pair body on its own
(``tmem_map`` keeps its own default here: the first body unless the driver sets it).
"""

from .api_dsl_sm100_d128 import SdpaBwdDslSm100D128


class SdpaBwdDslSm100D128Cta2(SdpaBwdDslSm100D128):
    """The grouped THD d128 backward routed to the cta_group::2 pair body (``accumulation_grouped_thd_2cta``) at any S."""

    cta_pair: str = "on"
