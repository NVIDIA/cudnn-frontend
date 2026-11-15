# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Local identity and admission for the native SM107 d192_f16_multi_q body."""

NAME = "sdpa_fwd_sm107_d192_multi_q"
TEMPLATE_NAME = "prefill_d192_f16"
KERNEL_FILE = "sm107/d192_f16_multi_q/kernel.py"


def mismatch(facts, knobs):
    if (facts.d_qk, facts.d_v) != (192, 128):
        return "native D192 multi-Q requires exact D192/V128"
    if facts.has_paged_kv:
        return "paged D192 uses the native small-half multi-Q body"
    if not facts.uniform_dtype:
        return "native D192 multi-Q requires same-dtype Q/K/V/O"
    if knobs is not None and (knobs.cga not in (None, 2) or knobs.tile_m not in (None, 128)):
        return "native D192 multi-Q requires TILE_M=128 and TILE_CGA_M=2"
    return None
