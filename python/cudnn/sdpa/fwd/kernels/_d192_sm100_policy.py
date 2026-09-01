# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host-side compile policy shared by the SM100 D192/D128 kernels.

The kernel modules are specialized by ``TemplateParams`` when they are loaded,
so shape-dependent choices must be resolved before ``load_template()`` runs.
Keeping that policy beside the D192 kernels avoids spreading D192 tuning rules
through the generic SDPA adapter without adding runtime branches to the kernel.
"""

from dataclasses import dataclass, fields, replace

from cudnn.frost.tile_dsl.constants import DTYPE_E4M3, DTYPE_E5M2, SCHED_LPT_L2, SCHED_NATURAL
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

_D192_D128 = (192, 128)


@dataclass(frozen=True)
class D192TemplateParams(TemplateParams):
    """D192-only additions to the shared SM100 module-cache key."""

    lpt_l2_size_mib: int = 0


def with_d192_lpt_l2_budget(cfg, params: TemplateParams):
    """Apply the D192 scheduler's optional L2 grouping budget to its CFG."""

    l2_size_mib = int(getattr(params, "lpt_l2_size_mib", 0))
    return replace(cfg, L2_SIZE_MIB=l2_size_mib) if l2_size_mib else cfg


def apply_d192_template_policy(
    params: TemplateParams,
    *,
    flavor: tuple[int, int],
    pertensor: bool,
    batch_size: int,
    h_q: int,
    pack_gqa_ratio: int,
    s_q: int,
    s_kv: int,
) -> TemplateParams:
    """Return the D192 specialization of otherwise semantic template params."""

    if flavor != _D192_D128:
        return params

    params = D192TemplateParams(**{field.name: getattr(params, field.name) for field in fields(TemplateParams)})

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    mxfp8 = fp8 and not pertensor
    thd = params.thd_varlen
    window_left = params.window_left
    window_right = params.window_right
    groups = batch_size * h_q // pack_gqa_ratio

    lpt_head_group = 8 if fp8 and not thd and groups % 8 == 0 else 1
    lpt_q_tiles = (s_q * pack_gqa_ratio + 511) // 512 if fp8 and not thd else 0

    lpt_l2_size_mib = 0
    lpt_l2_8k = params.sched_policy == SCHED_LPT_L2 and not thd and params.split_kv == 1 and s_q == 8192 and s_kv == 8192
    if lpt_l2_8k and pertensor and params.dtype_qkv == DTYPE_E4M3 and groups % 24 != 0 and groups % 16 == 0:
        # At 8K, 60 MiB groups 24 one-byte K/V heads; 40 MiB groups 16 and
        # avoids a short final group for these grids.
        lpt_l2_size_mib = 40
    elif lpt_l2_8k and not fp8 and groups % 12 != 0 and groups % 8 == 0:
        # Half inputs double the per-head K/V footprint: 60 MiB groups 12
        # heads, while 40 MiB groups 8 and avoids a short final group.
        lpt_l2_size_mib = 40

    template_window_right = window_right
    if fp8 and pertensor and window_left is None and window_right is None and not params.seq_kv_lens_present:
        # CUTLASS DSL 4.7 does not finish lowering the large-shape FP8
        # MASK_NONE x32 path. This bound removes no valid K and selects the
        # equivalent masked-interior lowering.
        template_window_right = s_kv

    square_br_as_tl = (
        params.split_kv == 1
        and not thd
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and window_left is None
        and window_right == 0
        and params.bottom_right
        and s_q == s_kv
        and 4096 < s_kv <= 8192
    )
    template_bottom_right = False if square_br_as_tl else params.bottom_right

    mx_dense_mid_causal_cga1 = (
        mxfp8
        and params.split_kv == 1
        and not thd
        and window_left is None
        and window_right == 0
        and not template_bottom_right
        and 4096 < s_kv <= 8192
        and (params.dtype_qkv == DTYPE_E5M2 or s_q >= 4096)
    )
    sched_policy = SCHED_NATURAL if mx_dense_mid_causal_cga1 else params.sched_policy

    # Per-tensor D192 favors independent CTAs for dense sliding windows and
    # E5M2 no-mask. Keep cga2's KV reuse for causal, THD, and split-KV.
    pt_cga1 = (
        pertensor
        and params.split_kv == 1
        and not thd
        and (
            window_left is not None
            or (params.dtype_qkv == DTYPE_E5M2 and window_right is None)
            or (
                params.dtype_qkv == DTYPE_E4M3
                and params.dtype_o in (DTYPE_E4M3, DTYPE_E5M2)
                and window_left is None
                and window_right == 0
                and not template_bottom_right
            )
        )
    )

    # D192 MX cga1 trades two-CTA cooperation for twice as many independent
    # assignments and half the KV stage depth. Small packed SWA tiles retain
    # cga2's reuse; masked dense and sufficiently long packed tiles use cga1.
    mx_cga1 = False
    if mxfp8 and params.split_kv == 1:
        masked = window_right is not None
        sliding = window_left is not None
        if thd:
            if params.dtype_qkv == DTYPE_E5M2 and not masked:
                mx_cga1 = True
            elif masked and sliding:
                min_s_kv = 4096 if params.dtype_qkv == DTYPE_E4M3 else 2048
                mx_cga1 = s_kv >= min_s_kv
            elif masked:
                mx_cga1 = s_kv >= 2048
        elif masked:
            mx_cga1 = params.dtype_qkv == DTYPE_E4M3 or sliding or s_kv <= 4096
    mx_cga1 = mx_cga1 or mx_dense_mid_causal_cga1

    return replace(
        params,
        window_right=template_window_right,
        bottom_right=template_bottom_right,
        sched_policy=sched_policy,
        lpt_head_group=lpt_head_group,
        lpt_q_tiles=lpt_q_tiles,
        lpt_l2_size_mib=lpt_l2_size_mib,
        cta_mma=1 if pt_cga1 or mx_cga1 else 2,
    )
