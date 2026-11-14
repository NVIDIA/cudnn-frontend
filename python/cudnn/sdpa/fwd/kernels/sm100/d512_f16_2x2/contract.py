# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The wide half 2x2 implementation's identity and local contract."""

NAME = "sdpa_fwd_sm100_d512_2x2"
TEMPLATE_NAME = "prefill_d512_f16_2x2"
KERNEL_FILE = "sm100/d512_f16_2x2/kernel.py"
QUERY_TILE = 64
CTA_MMA = 2


def cluster_m(d_qk):
    return 2 if d_qk == 576 else 4


def mismatch(facts, knobs):
    if (facts.d_qk, facts.d_v) not in ((512, 512), (576, 512)):
        return "the 2x2 implementation serves exact D512/V512 or D576/V512"
    if facts.d_qk == 512 and facts.has_paged_kv:
        return "D512 2x2 does not serve paged KV; use the role-split implementation"
    if knobs is not None:
        if facts.d_qk == 512 and (knobs.split_kv or 1) > 1:
            return "D512 2x2 split-KV is unqualified; use the role-split implementation"
        if knobs.tile_m not in (None, QUERY_TILE):
            return "wide half 2x2 TILE_M must be 64"
        if facts.d_qk == 512 and knobs.pack_gqa and 64 % (facts.h_q // facts.h_kv):
            return "D512 2x2 packs whole GQA groups into its 64-row tile"
    return None
