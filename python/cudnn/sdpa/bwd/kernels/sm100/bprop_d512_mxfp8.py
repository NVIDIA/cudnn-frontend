# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D512 block-scale backward with canonical F8_128x4 scales and fixed owners.

QK/dOV contract all512 features; each CTA writes128 gradient columns. Scores
are consumed into probability registers before dOV reuses their TMEM columns.
The released TMEM stores the scale factors; no global SF repack is needed.
"""

from typing import Optional
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
from cudnn.frost.tile_dsl.handles import GmemTileTma
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, amax_to_ue8m0_rp
from cudnn.frost.tile_dsl.tma import tma_load_tile
from cudnn.sdpa.bwd.kernels.sm100.bprop_d512_fp8 import _join, _output
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dkdv_fp8 import (
    FP8,
    HALF_DTYPES,
    LOG2E,
    _bpe,
    _mma_desc,
    _operand,
    _pack_fp8,
    _scale_pairs,
    _sf_copy,
    _sf_map,
    _sf_ptr,
    _sf_tile,
    _store_chunks,
    _tensor_map,
    _tmem_load,
    _view,
)

M = N = C = 128
D = 512
A, B, P, DS = 0, 65536, 131072, 147456
SFA, SFB, SFDS, SFP = 163840, 165888, 167936, 168448
BAR = 168960
HANDLE = BAR + 16
SMEM = BAR + 1024
TS, TG, TV = 0, 128, 256
TSA, TSB, TSC, TSD = 384, 400, 416, 420
SPEC = globals().get("FROST_TEMPLATE_PARAMS", (512, 512, False, True, 0))
WINDOW = int(SPEC[4])


@cute.jit
def _load(
    slab,
    bar,
    phase,
    adesc,
    bdesc,
    sfa,
    sfb,
    arow,
    brow,
    ahead,
    bhead,
    batch,
    cols: cutlass.Constexpr[int],
    col,
    ha: cutlass.Constexpr[int],
    hb: cutlass.Constexpr[int],
    trans: cutlass.Constexpr[bool],
):
    sfbytes = 512 if cutlass.const_expr(trans) else 2048
    if cute.arch.warp_idx() == 0:
        bar.arrive(n_bytes=2 * M * cols + 2 * sfbytes, pred=nvvm.elect_sync())
        tma_load_tile(_operand(slab, A, FP8, M, cols), GmemTileTma(adesc)(col, arow, ahead, batch), bar.smem_ptr)
        tma_load_tile(_operand(slab, B, FP8, M, cols), GmemTileTma(bdesc)(col, brow, bhead, batch), bar.smem_ptr)
        if cutlass.const_expr(trans):
            tma_load_tile(_sf_tile(slab, SFA, 512), GmemTileTma(sfa)(0, 0, arow // M, batch * ha + ahead, col // C), bar.smem_ptr)
            tma_load_tile(_sf_tile(slab, SFB, 512), GmemTileTma(sfb)(0, 0, brow // M, batch * hb + bhead, col // C), bar.smem_ptr)
        else:
            tma_load_tile(_sf_tile(slab, SFA, 2048), GmemTileTma(sfa)(0, 0, arow // M, ahead, batch), bar.smem_ptr)
            tma_load_tile(_sf_tile(slab, SFB, 2048), GmemTileTma(sfb)(0, 0, brow // M, bhead, batch), bar.smem_ptr)
    _join(bar, phase)
    nvvm.barrier_cta_sync()


@cute.jit
def _product(slab, done, phase, tmem):
    if cute.arch.warp_idx() == 0:
        _sf_copy(tmem, TSA, _sf_tile(slab, SFA, 2048).desc(), 4)
        _sf_copy(tmem, TSB, _sf_tile(slab, SFB, 2048).desc(), 4)
        mma_ss(
            _mma_desc(M, N, D, mx=True),
            _operand(slab, A, FP8, M, D).desc(),
            _operand(slab, B, FP8, N, D).desc(),
            nvvm.make_tmem_ptr(tmem + TS, cutlass.Float32),
            tmem_sf_a=_sf_ptr(tmem, TSA),
            tmem_sf_b=_sf_ptr(tmem, TSB),
        )
        if nvvm.elect_sync():
            done.arrive(cta_group=1)
    _join(done, phase)
    nvvm.barrier_cta_sync()


@cute.jit
def _probability(
    tmem, lse, batch, head, owner, other, sq, skv, scale, causal: cutlass.Constexpr[bool], window: cutlass.Constexpr[int], dq: cutlass.Constexpr[bool]
):
    tid = cute.arch.thread_idx()[0]
    row, group = tid % M, tid // M
    scores = _tmem_load(tmem + TS + group * 64, 64)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    probs = []
    for i in cutlass.range_constexpr(64):
        q = owner * M + row if cutlass.const_expr(dq) else other * N + group * 64 + i
        k = other * N + group * 64 + i if cutlass.const_expr(dq) else owner * M + row
        valid = (q < sq) & (k < skv)
        if cutlass.const_expr(causal):
            valid = valid & (k <= q)
        if cutlass.const_expr(window > 0):
            valid = valid & (k + window > q)
        stat = cutlass.Float32(lse[batch, head, cute.math.min(q, sq - 1)])
        val = cutlass.Float32(0.0)
        if valid:
            val = cute.math.exp2((scores[i] * scale - stat) * cutlass.Float32(LOG2E), fastmath=True) * cutlass.Float32(256.0)
        probs.append(val)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()
    return cutlass.Vector.from_elements(tuple(probs), cutlass.Float32)


@cute.jit
def _derivative(slab, tmem, probs, delta, batch, head, owner, other, sq, scale, dq: cutlass.Constexpr[bool]):
    tid = cute.arch.thread_idx()[0]
    row, group = tid % M, tid // M
    dp = _tmem_load(tmem + TS + group * 64, 64)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    ds = []
    for i in cutlass.range_constexpr(64):
        q = owner * M + row if cutlass.const_expr(dq) else other * N + group * 64 + i
        dstat = cutlass.Float32(delta[batch, head, cute.math.min(q, sq - 1)])
        ds.append((dp[i] - dstat) * (scale * cutlass.Float32(1.0 / 256.0)) * probs[i])
    e0, inv0 = amax_to_ue8m0_rp(abs_max_tree(ds[:32]))
    e1, inv1 = amax_to_ue8m0_rp(abs_max_tree(ds[32:]))
    words = _pack_fp8(_scale_pairs(ds[:32], inv0) + _scale_pairs(ds[32:], inv1))
    _store_chunks(slab, DS, row, words, group * 4, 4)
    sf = cutlass.Int32(e0) | (cutlass.Int32(e1) << 8)
    _view(slab, SFDS + (row % 32) * 16 + (row // 32) * 4 + group * 2, cutlass.Uint16, 1, alignment=2).subview(0).store(cutlass.Uint16(sf))
    if cutlass.const_expr(not dq):
        _store_chunks(slab, P, row, _pack_fp8([probs[i] for i in range(64)]), group * 4, 4)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()


@cute.jit
def _grad(slab, done, phase, tmem, accumulate, dq: cutlass.Constexpr[bool]):
    if cute.arch.warp_idx() == 0:
        _sf_copy(tmem, TSA, _sf_tile(slab, SFDS, 512).desc(), 1)
        _sf_copy(tmem, TSB, _sf_tile(slab, SFA, 512).desc(), 1)
        desc = _mma_desc(M, C, N, btranspose=True, mx=True)
        mma_ss(
            desc,
            _operand(slab, DS, FP8, M, N).desc(),
            _operand(slab, A, FP8, N, C, transpose=True).desc(),
            nvvm.make_tmem_ptr(tmem + TG, cutlass.Float32),
            accumulate=accumulate,
            tmem_sf_a=_sf_ptr(tmem, TSA),
            tmem_sf_b=_sf_ptr(tmem, TSB),
        )
        if cutlass.const_expr(not dq):
            _sf_copy(tmem, TSC, _sf_tile(slab, SFP, 512).desc(), 1)
            _sf_copy(tmem, TSD, _sf_tile(slab, SFB, 512).desc(), 1)
            mma_ss(
                desc,
                _operand(slab, P, FP8, M, N).desc(),
                _operand(slab, B, FP8, N, C, transpose=True).desc(),
                nvvm.make_tmem_ptr(tmem + TV, cutlass.Float32),
                accumulate=accumulate,
                tmem_sf_a=_sf_ptr(tmem, TSC),
                tmem_sf_b=_sf_ptr(tmem, TSD),
            )
        if nvvm.elect_sync():
            done.arrive(cta_group=1)
    _join(done, phase)
    nvvm.barrier_cta_sync()


@cute.kernel
def kernel(
    qmap: cutlass.GridConstant[tmap.TensorMap],
    kmap: cutlass.GridConstant[tmap.TensorMap],
    vmap: cutlass.GridConstant[tmap.TensorMap],
    domap: cutlass.GridConstant[tmap.TensorMap],
    dgmap: cutlass.GridConstant[tmap.TensorMap],
    dvmap: cutlass.GridConstant[tmap.TensorMap],
    lse: cute.Tensor,
    delta: cute.Tensor,
    qtmap: cutlass.GridConstant[tmap.TensorMap],
    ktmap: cutlass.GridConstant[tmap.TensorMap],
    dotmap: cutlass.GridConstant[tmap.TensorMap],
    sfq: cutlass.GridConstant[tmap.TensorMap],
    sfk: cutlass.GridConstant[tmap.TensorMap],
    sfv: cutlass.GridConstant[tmap.TensorMap],
    sfdo: cutlass.GridConstant[tmap.TensorMap],
    sfqt: cutlass.GridConstant[tmap.TensorMap],
    sfkt: cutlass.GridConstant[tmap.TensorMap],
    sfdot: cutlass.GridConstant[tmap.TensorMap],
    hq_count: cutlass.Constexpr[int],
    hk_count: cutlass.Constexpr[int],
    sq: cutlass.Int32,
    skv: cutlass.Int32,
    scale: cutlass.Float32,
    dtype: cutlass.Constexpr,
    ratio: cutlass.Constexpr[int],
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
):
    x, head, batch = cute.arch.block_idx()
    owner, chunk = x // 4, x % 4
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    slab = cutlass.Array(cutlass.Uint8, SMEM, alignment=1024, space=cutlass.AddressSpace.smem)
    full = MBarrier(_view(slab, BAR, cutlass.Int64, 1), stages=1, init_count=1, producer=Producer.TMA_LOAD, try_wait=True)
    done = MBarrier(_view(slab, BAR + 8, cutlass.Int64, 1), stages=1, init_count=1, producer=Producer.MMA_COMMIT, try_wait=True)
    slot = _view(slab, HANDLE, cutlass.Int32, 1, alignment=4)
    if warp == 0:
        nvvm.tcgen05_alloc(slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if nvvm.elect_sync():
            full.init()
            done.init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    tmem = slot.load()
    tid = cute.arch.thread_idx()[0]
    if tid < 128:
        _view(slab, SFP, cutlass.Int32, 128).subview(tid).store(cutlass.Int32(0x7F7F7F7F))
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.barrier_cta_sync()
    phase, math_phase, live = cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)
    lo = cutlass.Int32(0)
    hi = (skv + N - 1) // N if cutlass.const_expr(dq) else (sq + N - 1) // N
    if cutlass.const_expr(dq):
        if cutlass.const_expr(window > 0):
            lo = cute.math.max(cutlass.Int32(0), (owner * M - window + 1) // N)
        if cutlass.const_expr(causal):
            hi = cute.math.min(hi, owner + 1)
    else:
        if cutlass.const_expr(causal):
            lo = owner
        if cutlass.const_expr(window > 0):
            hi = cute.math.min(hi, ((owner + 1) * M + window - 1 + N - 1) // N)
    nheads = 1 if cutlass.const_expr(dq) else ratio
    for rh in cutlass.range(nheads, unroll=1):
        hq = head if cutlass.const_expr(dq) else head * ratio + rh
        hk = head // ratio if cutlass.const_expr(dq) else head
        for other in cutlass.range(lo, hi, unroll=1):
            if cutlass.const_expr(dq):
                _load(slab, full, phase, qmap, kmap, sfq, sfk, owner * M, other * N, hq, hk, batch, D, 0, hq_count, hk_count, False)
            else:
                _load(slab, full, phase, kmap, qmap, sfk, sfq, owner * M, other * N, hk, hq, batch, D, 0, hk_count, hq_count, False)
            phase = phase ^ 1
            _product(slab, done, math_phase, tmem)
            math_phase = math_phase ^ 1
            probs = _probability(tmem, lse, batch, hq, owner, other, sq, skv, scale, causal, window, dq)
            if cutlass.const_expr(dq):
                _load(slab, full, phase, domap, vmap, sfdo, sfv, owner * M, other * N, hq, hk, batch, D, 0, hq_count, hk_count, False)
            else:
                _load(slab, full, phase, vmap, domap, sfv, sfdo, owner * M, other * N, hk, hq, batch, D, 0, hk_count, hq_count, False)
            phase = phase ^ 1
            _product(slab, done, math_phase, tmem)
            math_phase = math_phase ^ 1
            _derivative(slab, tmem, probs, delta, batch, hq, owner, other, sq, scale, dq)
            if cutlass.const_expr(dq):
                _load(slab, full, phase, ktmap, ktmap, sfkt, sfkt, other * N, other * N, hk, hk, batch, C, chunk * C, hk_count, hk_count, True)
            else:
                _load(slab, full, phase, qtmap, dotmap, sfqt, sfdot, other * N, other * N, hq, hq, batch, C, chunk * C, hq_count, hq_count, True)
            phase = phase ^ 1
            _grad(slab, done, math_phase, tmem, live != 0, dq)
            math_phase = math_phase ^ 1
            live = cutlass.Int32(1)
    one = cutlass.Float32(1.0)
    _output(slab, tmem, dgmap, owner, head, batch, chunk, TG, one, one, live != 0, dtype, None)
    if cutlass.const_expr(not dq):
        _output(slab, tmem, dvmap, owner, head, batch, chunk, TV, cutlass.Float32(1.0 / 256.0), one, live != 0, dtype, None)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem, cutlass.Float32), cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dg: cute.Tensor,
    dv: cute.Tensor,
    qt: cute.Tensor,
    kt: cute.Tensor,
    dot: cute.Tensor,
    sf_q: cute.Tensor,
    sf_k: cute.Tensor,
    sf_v: cute.Tensor,
    sf_do: cute.Tensor,
    sf_qt: cute.Tensor,
    sf_kt: cute.Tensor,
    sf_dot: cute.Tensor,
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    dtype = dg.element_type
    if cutlass.const_expr(dtype not in HALF_DTYPES):
        raise ValueError("D512 MXFP8 backward gradients must be FP16 or BF16")
    if cutlass.const_expr(any(t.element_type != FP8 for t in (q, k, v, do, qt, kt, dot)) or any(t.shape[3] != D for t in (q, k, v, do, dg, dv))):
        raise ValueError("D512 MXFP8 backward requires full D512 E4M3 inputs and D512 gradients")
    ratio = q.shape[2] // k.shape[2]
    n = q.shape[1] if cutlass.const_expr(dq) else k.shape[1]
    h = q.shape[2] if cutlass.const_expr(dq) else k.shape[2]
    kernel(
        _tensor_map(q, M, 128),
        _tensor_map(k, M, 128),
        _tensor_map(v, M, 128),
        _tensor_map(do, M, 128),
        _tensor_map(dg, M, 128 // _bpe(dtype)),
        _tensor_map(dv, M, 128 // _bpe(dtype)),
        lse,
        delta,
        _tensor_map(qt, M, 128),
        _tensor_map(kt, M, 128),
        _tensor_map(dot, M, 128),
        _sf_map(sf_q),
        _sf_map(sf_k),
        _sf_map(sf_v),
        _sf_map(sf_do),
        _sf_map(sf_qt),
        _sf_map(sf_kt),
        _sf_map(sf_dot),
        q.shape[2],
        k.shape[2],
        q.shape[1],
        k.shape[1],
        scale,
        dtype,
        ratio,
        causal,
        window,
        dq,
    ).launch(grid=((n + M - 1) // M * 4, h, q.shape[0]), block=(256, 1, 1), stream=stream, min_blocks_per_mp=1)


@cute.jit
def dkdv_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dk: Optional[cute.Tensor],
    scale_dv: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
    qt: Optional[cute.Tensor] = None,
    dot: Optional[cute.Tensor] = None,
    sf_q: Optional[cute.Tensor] = None,
    sf_qt: Optional[cute.Tensor] = None,
    sf_k: Optional[cute.Tensor] = None,
    sf_v: Optional[cute.Tensor] = None,
    sf_do: Optional[cute.Tensor] = None,
    sf_dot: Optional[cute.Tensor] = None,
):
    host(q, k, v, do, lse, delta, dk, dv, qt, k, dot, sf_q, sf_k, sf_v, sf_do, sf_qt, sf_k, sf_dot, scale, causal, WINDOW, False, stream)


@cute.jit
def dq_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    descale_q: Optional[cute.Tensor],
    descale_k: Optional[cute.Tensor],
    descale_v: Optional[cute.Tensor],
    descale_s: Optional[cute.Tensor],
    scale_s: Optional[cute.Tensor],
    descale_do: Optional[cute.Tensor],
    descale_dp: Optional[cute.Tensor],
    scale_dq: Optional[cute.Tensor],
    scale_dp: Optional[cute.Tensor],
    amax_dq: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
    kt: Optional[cute.Tensor] = None,
    sf_q: Optional[cute.Tensor] = None,
    sf_do: Optional[cute.Tensor] = None,
    sf_k: Optional[cute.Tensor] = None,
    sf_kt: Optional[cute.Tensor] = None,
    sf_v: Optional[cute.Tensor] = None,
):
    host(q, k, v, do, lse, delta, dq, dq, q, kt, do, sf_q, sf_k, sf_v, sf_do, sf_q, sf_kt, sf_do, scale, causal, WINDOW, True, stream)
