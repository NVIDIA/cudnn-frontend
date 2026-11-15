# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Local identity and declaration boundaries of the ordinary work item."""

NAME = "sdpa_fwd_sm107_d128_multi_q"
TEMPLATE_NAME = "prefill_d128_f16"
KERNEL_FILE = "sm107/d128_f16_multi_q/kernel.py"


def mismatch(facts, knobs):
    if facts.d_qk == 192 and not facts.has_paged_kv:
        return "native small-half D192 requires paged KV; nonpaged D192 has another body"
    if knobs is None:
        return None
    if knobs.tile_m not in (None, 128) or knobs.warp_spec_cfg not in (None, 0):
        return "ordinary multi-Q requires TILE_M=128 and the two-query-group topology"
    cga = knobs.cga if knobs.cga is not None else (1 if facts.d_qk <= 64 else 2)
    if facts.d_qk in (32, 192) and cga != (1 if facts.d_qk == 32 else 2):
        return "ordinary D32 requires one CTA; paged D192 requires two CTAs"
    if cga not in (1, 2):
        return "ordinary multi-Q requires one or two CTAs"
    if facts.uniform_dtype and facts.d_qk in (64, 128) and cga == 1:
        if (knobs.split_kv or 1) > 1 and (facts.thd or not facts.has_paged_kv):
            return "this split geometry uses the shared SM100 single-Q body"
        if facts.d_qk == 128 and not facts.thd and not facts.has_paged_kv:
            return "dense D128 CTA1 uses the shared SM100 single-Q body"
    return None
