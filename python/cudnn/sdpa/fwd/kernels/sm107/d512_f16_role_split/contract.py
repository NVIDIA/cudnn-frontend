# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The SM107 d512_f16_role_split implementation's local identity and admission."""

NAME = "sdpa_fwd_sm107_d512_role_split"
TEMPLATE_NAME = "prefill_d512_f16"
KERNEL_FILE = "sm107/d512_f16_role_split/kernel.py"
QUERY_TILE = 128
CTA_MMA = 2


def mismatch(facts, knobs):
    if (facts.d_qk, facts.d_v) != (512, 512):
        return "SM107 role-split serves exact D512/V512"
    if facts.has_paged_kv or not facts.uniform_dtype:
        return "SM107 role-split requires nonpaged, same-dtype Q/K/V/O"
    if knobs is not None and (knobs.pack_gqa or (knobs.split_kv or 1) > 1):
        return "SM107 packed/split wide-half plans require the 2x2 implementation"
    if knobs is not None and knobs.tile_m not in (None, QUERY_TILE):
        return f"this SM107 implementation requires TILE_M={QUERY_TILE}"
    return None
