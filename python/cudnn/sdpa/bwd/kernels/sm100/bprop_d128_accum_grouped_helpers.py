# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense/THD grouped d128 chains with native partials and a final FP32 fold."""

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.thd import THD_SETUP_THREADS
from ..bprop_chain_common import dkv_reduce_host
from ..bprop_thd_reduce import dkv_reduce_thd_origins_host
from .bprop_d128_accum_grouped_f16 import main_dense_host, main_thd_host
from .bprop_d128_accum_thd_f16 import _tensor_map
from .bprop_d128_accum_thd_helpers import _origin_row, setup_thd_d128_kernel, delta_zero_thd_d128_kernel, cast_dq_thd_d128_kernel
from .bprop_d128_helpers import THREADS, ROWS_PER_CTA, delta_zero_host, cast_dq_host
from ...config_sm100_d128_accum import CFG


@cute.jit
def host_dense(q, k, v, o, do, lse, dq, dk, dv, dq_accum, delta, dk_part, dv_part, scale, causal: cutlass.Constexpr[bool], stream):
    """Four native launches; uniquely owned partials are folded in FP32."""
    delta_zero_host(o, do, delta, dq_accum, stream)
    main_dense_host(q, k, v, do, lse, delta, dq_accum, dk_part, dv_part, scale, causal, stream)
    cast_dq_host(dq_accum, dq, q.element_type, stream)
    dkv_reduce_host(dk_part, dv_part, dk, dv, CFG.D, CFG.D, q.shape[2] // k.shape[2], q.element_type, False, stream)


@cute.jit
def host_thd(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    dk_part,
    dv_part,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
):
    """Five native launches, with compact partials and independent final origins."""
    setup_thd_d128_kernel(
        _tensor_map(q, CFG.HEAD_BOX),
        _tensor_map(k, CFG.HEAD_BOX),
        _tensor_map(v, CFG.HEAD_BOX),
        _tensor_map(do, CFG.HEAD_BOX),
        _tensor_map(dq_accum, CFG.DQ_BOX),
        _tensor_map(dk_part, CFG.HEAD_BOX),
        _tensor_map(dv_part, CFG.HEAD_BOX),
        q,
        k,
        v,
        do,
        dq_accum,
        dk_part,
        dv_part,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
        partial_dkv=True,
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    delta_zero_thd_d128_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
        grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
    )
    main_thd_host(
        maps,
        lse,
        delta,
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        q.shape[2] // k.shape[2],
        stream,
    )
    cast_dq_thd_d128_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
        grid=(q_grid, batch, q.shape[2]), block=(THREADS, 1, 1), stream=stream
    )
    dkv_reduce_thd_origins_host(
        dk_part,
        dv_part,
        dk,
        dv,
        meta,
        origins,
        cutlass.Int32(batch),
        max_skv,
        _origin_row(origin_specs, 6),
        _origin_row(origin_specs, 7),
        CFG.D,
        q.shape[2] // k.shape[2],
        q.element_type,
        stream,
    )
