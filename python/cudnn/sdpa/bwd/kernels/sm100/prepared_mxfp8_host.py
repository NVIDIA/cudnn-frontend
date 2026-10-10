# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One pointer entry for the existing MXFP8 SF-repack and backward chain."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached
from cudnn.sdpa.bwd.kernels.sm120.prepared_host import _scratch, _view


@cute.jit
def host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    do_ptr: cute.Pointer,
    stats_ptr: cute.Pointer,
    dq_ptr: cute.Pointer,
    dk_ptr: cute.Pointer,
    dv_ptr: cute.Pointer,
    qt_ptr: cute.Pointer,
    kt_ptr: cute.Pointer,
    dot_ptr: cute.Pointer,
    do_half_ptr: cute.Pointer,
    sfq_ptr: cute.Pointer,
    sfqt_ptr: cute.Pointer,
    sfk_ptr: cute.Pointer,
    sfkt_ptr: cute.Pointer,
    sfv_ptr: cute.Pointer,
    sfdo_ptr: cute.Pointer,
    sfdot_ptr: cute.Pointer,
    workspace: cute.Pointer,
    scale: cutlass.Float32,
    repacks: cutlass.Constexpr,
    dq_kernel: cutlass.Constexpr,
    dkdv_kernel: cutlass.Constexpr,
    problem: cutlass.Constexpr,
    geometry: cutlass.Constexpr,
    regions: cutlass.Constexpr,
    causal: cutlass.Constexpr,
    stream: driver.CUstream,
):
    qg, kg, lg = geometry
    q, k, v = _view(q_ptr, qg), _view(k_ptr, kg), _view(v_ptr, kg)
    o, do = _view(o_ptr, qg), _view(do_ptr, qg)
    lse = _view(stats_ptr, lg)
    dq, dk, dv = _view(dq_ptr, qg), _view(dk_ptr, kg), _view(dv_ptr, kg)
    qt, kt, dot = _view(qt_ptr, qg), _view(kt_ptr, kg), _view(dot_ptr, qg)
    do_half = _view(do_half_ptr, qg)
    sf_ptrs = (sfq_ptr, sfqt_ptr, sfk_ptr, sfkt_ptr, sfv_ptr, sfdo_ptr, sfdot_ptr)
    scales = []
    for i in cutlass.range_constexpr(len(repacks)):
        repack, source = repacks[i]
        src = _view(sf_ptrs[source], ((repack.src_bytes,), (1,)))
        dst = _scratch(workspace, regions[i + 1], cutlass.Int8)
        repack(src, dst, stream)
        scales.append(_scratch(workspace, regions[i + 1], cutlass.Float8E8M0FNU))
    ws = _scratch(workspace, regions[0], cutlass.Uint8)
    right = cutlass.Int32(0) if cutlass.const_expr(causal) else None
    # Preserve the existing prologue: dQ writes sum(O*dO) and scaled LSE, then
    # dK/dV consumes those same workspace slots without running it again.
    dq_kernel(
        problem,
        q,
        k,
        kt,
        v,
        o,
        scales[0],
        scales[1],
        scales[2],
        scales[3],
        scales[4],
        dq,
        dk,
        dv,
        do,
        do_half,
        lse,
        None,
        None,
        scale,
        None,
        right,
        ws,
        stream,
        False,
    )
    dkdv_kernel(
        problem,
        q,
        qt,
        k,
        v,
        o,
        scales[5],
        scales[6],
        scales[7],
        scales[8],
        scales[9],
        scales[10],
        dk,
        dv,
        do,
        dot,
        do_half,
        lse,
        None,
        None,
        scale,
        None,
        right,
        ws,
        stream,
        True,
    )


def compile_host(repacks, dq_kernel, dkdv_kernel, problem, geometry, regions, causal, dtype, sm, cache_key):
    """Declare pointers only; all tensor layouts are constructed inside host IR."""

    def pointer(t):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16)

    fp8 = cutlass.Float8E4M3FN
    types = (fp8, fp8, fp8, dtype, fp8, cutlass.Float32, dtype, dtype, dtype, fp8, fp8, fp8, dtype) + (cutlass.Int8,) * 7
    return compile_cached(
        host,
        *(pointer(t) for t in types),
        pointer(cutlass.Uint8),
        cutlass.Float32(1),
        repacks,
        dq_kernel,
        dkdv_kernel,
        problem,
        geometry,
        regions,
        causal,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        # O2 is part of this kernel's validated contract: O3 spills the dQ MMA.
        options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch sm_{sm}a",
        cache_key=cache_key,
        symbol="frost_sdpa_bwd_sm100_mxfp8_prepared",
    )
