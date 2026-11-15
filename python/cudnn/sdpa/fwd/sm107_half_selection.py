# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Half body-selection policy shared by recommendation and placement.

The registry resolves the historical capability domain into scalar geometry
facts before calling this policy. This module never imports that registry or
the candidate generator; fixed hosts do not call it.
"""

from dataclasses import replace

from .config_sm100 import supports_paged_prefill_cga1


def select_sm107_half_geometry(facts, knobs, *, flavor, packed_split, thd_decode, decode_tile):
    """Name the existing small/mid-width Rubin body before fixed lowering.

    Logical head width does not identify a body: native small tiles may use
    a covering flavor, while split/paged layouts reuse two SM100 bodies.
    Preserve the legacy loader's precedence, including the native D128
    one-query topology before the shared single-Q cases. Only this central
    selection consults the ordinary/batch-pair crossover.
    """
    from . import config_sm107 as config
    from .kernels.sm107 import mid_half_contract as mid, small_half_contract as small
    from .kernels.sm107 import shared_half_contract as shared

    d = flavor[0]
    split = knobs.split_kv or 1
    group = facts.h_q // facts.h_kv if knobs.pack_gqa else 1
    if d == 128 and knobs.warp_spec_cfg == 1:
        return mid.explicit_geometry(mid.LOOKAHEAD, facts, knobs)

    cga = knobs.cga if knobs.cga is not None else (1 if d <= 64 else 2)
    tile_n = knobs.tile_n
    if d <= 128:
        if tile_n is None:
            tile_n = (
                128
                if d == 64 and packed_split
                else config.sm107_tile_n_pick(
                    d,
                    facts.s_kv,
                    has_bias=facts.has_bias,
                    has_kv_mask=facts.has_kv_mask,
                    padded=small.kv_padded(facts),
                    domain=dict(config.SM107_TILE_NS_BY_D_SHAPE).get(flavor, (128,)),
                )
            )
        pair = facts.has_bias and config.sm107_pair_item_active(
            facts.bias_dim[0],
            facts.b,
            facts.h_q,
            facts.s_q,
            facts.device_sm_count or 0,
            1,
            s_kv=facts.s_kv,
            d_flavor=d,
            tile_n=tile_n,
        )
        if facts.has_bias and knobs.cga is None:
            wgs = config.sm107_softmax_wgs_pick(
                d_flavor=d,
                s_kv=facts.s_kv,
                tile_n=tile_n,
                pair=pair,
                has_bias=True,
                has_kv_mask=facts.has_kv_mask,
                padded=small.kv_padded(facts),
                cga=1,
                b_flat=facts.b,
            )
            cga = config.sm107_bias_auto_cga(d, facts.s_kv, config.SM107_SMALL_D_CGAS.get(flavor, (2,)), pair=pair, tiles_q=wgs) or 1
    else:
        tile_n = 128 if tile_n is None else tile_n
        pair = False
    resolved = replace(knobs, cga=cga, tile_n=tile_n)

    def single_q():
        return shared.Geometry(shared.SINGLE_Q, d)

    if d == 192 and facts.has_paged_kv:
        return small.explicit_geometry(small.MULTI_Q, facts, resolved, d_flavor=d)
    # Uniform D64 uses the shared body for either THD split cache, dense
    # split, and short dense unsplit Q. N64 and native mixed-KV stay native.
    short_q = (
        not (facts.thd or facts.has_paged_kv or facts.has_bias or facts.has_kv_mask or facts.attn_scale_prefolded) and split == 1 and facts.s_q * group <= 128
    )
    if d == 64 and facts.uniform_dtype and cga == 1 and tile_n == 128:
        if (split > 1 and (not facts.has_paged_kv or facts.thd)) or short_q:
            return single_q()
    if (
        d == 128
        and cga == 1
        and thd_decode
        and not supports_paged_prefill_cga1(
            flavor, device_cc=facts.device_cc, thd=facts.thd, paged=facts.has_paged_kv, fp8=False, split_kv=split, max_q=facts.s_q
        )
    ):
        return single_q()
    if d == 256 and decode_tile:
        return mid.explicit_geometry(mid.SWAP_AB, facts, replace(resolved, tile_m=decode_tile))
    if cga == 1 and facts.thd and ((d == 192 and not facts.has_paged_kv) or (d == 128 and split > 1)):
        return single_q()
    if d == 256:
        if (facts.thd and not facts.has_paged_kv and split > 1) or (facts.has_paged_kv and facts.uniform_dtype):
            return shared.Geometry(shared.LOOKAHEAD, d)
        return mid.explicit_geometry(mid.LOOKAHEAD, facts, resolved)
    if d == 128 and cga == 1 and not facts.thd and not facts.has_paged_kv:
        return single_q()
    if d == 192:
        return mid.explicit_geometry(mid.D192, facts, resolved)
    name = small.BATCH_PAIR if pair and cga == 1 else small.MULTI_Q
    return small.explicit_geometry(name, facts, resolved, d_flavor=d)
