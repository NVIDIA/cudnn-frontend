# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Direct-adapter entry for the d128 backward on the cta_group::2 datapath (grouped THD and its dense compact-BSHD twin; bring-up / probe route).

The shipping adapter carries the typed per-S heuristic field ``SdpaBwdDslSm100D128.cta_pair`` ("auto" = the default: the pair body at
S >= ``CTA_PAIR_MIN_SEQ[cc]`` on cc 10.0 / 10.3 / 10.7 (the per-S threshold was measured on GB200 and is carried per part), else the one-CTA
body | "off" | "on").  This
subclass is that field pinned to ``"on"``: the
pair body at ANY S, every pair predicate declining BY NAME -- what a probe driver imports to validate and time the pair body on its own.
"""

from .api_dsl_sm100_d128 import SdpaBwdDslSm100D128


class SdpaBwdDslSm100D128Cta2(SdpaBwdDslSm100D128):
    """The grouped d128 backward routed to the cta_group::2 pair body at any S: ``accumulation_grouped_thd_2cta`` on a packed declaration,
    ``accumulation_grouped_dense_2cta`` on a dense compact-BSHD one (the same five launches; ``d128_grouped_2cta_contract``)."""

    cta_pair: str = "on"
