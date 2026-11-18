# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Local identity and admission for the native SM107 d256_f16_lookahead body."""

from ..half_flavor_contract import padded_shape

NAME = "sdpa_fwd_sm107_d256_lookahead"
TEMPLATE_NAME = "prefill_d256_f16"
KERNEL_FILE = "sm107/d256_f16_lookahead/kernel.py"


def mismatch(facts, knobs):
    shape = padded_shape(facts)
    if shape == (128, 128):
        if not facts.thd or not facts.uniform_dtype or facts.has_bias or facts.has_kv_mask or facts.has_epilogue_gate:
            return "native one-Q D128 requires half THD without bias, KV mask, mixed KV or gate"
        if knobs is not None and (knobs.cga not in (None, 1) or knobs.pack_gqa or (knobs.split_kv or 1) > 1 or knobs.warp_spec_cfg not in (None, 1)):
            return "native one-Q D128 requires CTA1, unsplit/unpacked and WARP_SPEC_CFG=1"
    elif shape == (256, 256):
        if facts.has_paged_kv and facts.uniform_dtype:
            return "same-dtype paged D256 uses the shared SM100 lookahead body"
        if facts.thd and not facts.has_paged_kv and knobs is not None and (knobs.split_kv or 1) > 1:
            return "nonpaged THD D256 split uses the shared SM100 lookahead body"
        if knobs is not None and (knobs.cga not in (None, 2) or knobs.warp_spec_cfg not in (None, 0)):
            return "native D256 lookahead requires CTA2 and its ordinary warp topology"
    else:
        return "native lookahead requires a D128/V128 or D256/V256 envelope"
    if knobs is not None and (knobs.tile_m not in (None, 128) or knobs.tile_n not in (None, 128)):
        return "native lookahead requires a 128x128 query/key tile"
    return None
