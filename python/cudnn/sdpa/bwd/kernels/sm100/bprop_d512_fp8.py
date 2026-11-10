# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D512 FP8 backward with tiled gradient ownership and no score workspace.

Each query/key owner writes 128 gradient columns. QK and dOV retain their
complete 512-column contraction. The two 64-KiB input slabs are reused only
after the preceding MMA has completed; probabilities and dS stay in shared
memory. Key owners fold GQA heads in a fixed order.
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
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, fmax_f32, fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dkdv_fp8 import (
    FP8,
    HALF_DTYPES,
    LOG2E,
    _bpe,
    _fold_amax,
    _mma_desc,
    _operand,
    _pack_fp8,
    _scalar,
    _scale_pairs,
    _store_chunks,
    _tensor_map,
    _tmem_load,
    _view,
)

M = N = C = 128
D = 512
A = 0
B = 65536
P = 131072
DS = 147456
BAR = 163840
HANDLE = BAR + 16
SMEM = BAR + 1024
TS = 0
TDP = 128
TG = 256
TV = 384
SPEC = globals().get("FROST_TEMPLATE_PARAMS", (512, 512, False, False, 0))
WINDOW = int(SPEC[4])


@cute.jit
def _join(bar, phase):
    bar.wait(phase, spin=True)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _load_pair(slab, bar, phase, adesc, bdesc, arow, brow, ahead, bhead, batch, cols: cutlass.Constexpr[int], col):
    if cute.arch.warp_idx() == 0:
        bar.arrive(n_bytes=2 * M * cols, pred=nvvm.elect_sync())
        tma_load_tile(_operand(slab, A, FP8, M, cols), GmemTileTma(adesc)(col, arow, ahead, batch), bar.smem_ptr)
        tma_load_tile(_operand(slab, B, FP8, M, cols), GmemTileTma(bdesc)(col, brow, bhead, batch), bar.smem_ptr)
    _join(bar, phase)
    nvvm.barrier_cta_sync()


@cute.jit
def _product(slab, done, phase, tmem, dst: cutlass.Constexpr[int], cols: cutlass.Constexpr[int]):
    if cute.arch.warp_idx() == 0:
        mma_ss(
            _mma_desc(M, N, cols, mx=False),
            _operand(slab, A, FP8, M, cols).desc(),
            _operand(slab, B, FP8, N, cols).desc(),
            nvvm.make_tmem_ptr(tmem + dst, cutlass.Float32),
        )
        if nvvm.elect_sync():
            done.arrive(cta_group=1)
    _join(done, phase)
    nvvm.barrier_cta_sync()


@cute.jit
def _softmax(
    slab,
    tmem,
    lse,
    delta,
    batch,
    head,
    owner,
    other,
    sq,
    skv,
    s_scale,
    p_scale,
    dp_scale,
    delta_scale,
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
):
    tid = cute.arch.thread_idx()[0]
    row, group = tid % M, tid // M
    scores = _tmem_load(tmem + TS + group * 64, 64)
    dp = _tmem_load(tmem + TDP + group * 64, 64)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    probs, derivs = [], []
    for i in cutlass.range_constexpr(64):
        q = owner * M + row if cutlass.const_expr(dq) else other * N + group * 64 + i
        k = other * N + group * 64 + i if cutlass.const_expr(dq) else owner * M + row
        valid = (q < sq) & (k < skv)
        if cutlass.const_expr(causal):
            valid = valid & (k <= q)
        if cutlass.const_expr(window > 0):
            valid = valid & (k + window > q)
        q_safe = cute.math.min(q, sq - 1)
        stat = cutlass.Float32(lse[batch, head, q_safe])
        dstat = cutlass.Float32(delta[batch, head, q_safe])
        val = cutlass.Float32(0.0)
        if valid:
            val = cute.math.exp2((scores[i] * s_scale - stat) * cutlass.Float32(LOG2E), fastmath=True) * p_scale
        probs.append(val)
        derivs.append((dp[i] * dp_scale - dstat * delta_scale) * val)
    _store_chunks(slab, DS, row, _pack_fp8(derivs), group * 4, 4)
    if cutlass.const_expr(not dq):
        _store_chunks(slab, P, row, _pack_fp8(probs), group * 4, 4)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()
    return abs_max_tree(derivs)


@cute.jit
def _grad(slab, done, phase, tmem, accumulate, dq: cutlass.Constexpr[bool]):
    if cute.arch.warp_idx() == 0:
        desc = _mma_desc(M, C, N, btranspose=True, mx=False)
        mma_ss(
            desc,
            _operand(slab, DS, FP8, M, N).desc(),
            _operand(slab, A, FP8, N, C, transpose=True).desc(),
            nvvm.make_tmem_ptr(tmem + TG, cutlass.Float32),
            accumulate=accumulate,
        )
        if cutlass.const_expr(not dq):
            mma_ss(
                desc,
                _operand(slab, P, FP8, M, N).desc(),
                _operand(slab, B, FP8, N, C, transpose=True).desc(),
                nvvm.make_tmem_ptr(tmem + TV, cutlass.Float32),
                accumulate=accumulate,
            )
        if nvvm.elect_sync():
            done.arrive(cta_group=1)
    _join(done, phase)
    nvvm.barrier_cta_sync()


@cute.jit
def _output(
    slab, tmem, desc, owner, head, batch, chunk, plane: cutlass.Constexpr[int], factor, out_scale, live, dtype: cutlass.Constexpr, amax: Optional[cute.Tensor]
):
    tid = cute.arch.thread_idx()[0]
    row, group = tid % M, tid // M
    vals = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(64)), cutlass.Float32)
    if live:
        vals = _tmem_load(tmem + plane + group * 64, 64)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    scaled = _scale_pairs([vals[i] for i in range(64)], factor)
    if cutlass.const_expr(amax is not None):
        _fold_amax(abs_max_tree(scaled), amax)
    if cutlass.const_expr(dtype == FP8):
        _store_chunks(slab, A, row, _pack_fp8(_scale_pairs(scaled, out_scale)), group * 4, 4)
    else:
        words = cutlass.Vector.from_elements(tuple(fp32_to_fp16(scaled[2 * i], scaled[2 * i + 1], dtype=dtype) for i in range(32)), cutlass.Int32)
        _store_chunks(slab, A + group * M * 128, row, words, 0, 8)
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()
    if cute.arch.warp_idx() == 0:
        tma_store_tile(_operand(slab, A, dtype, M, C), GmemTileTma(desc)(chunk * C, owner * M, head, batch))
        tma_store_commit()
        tma_store_wait(0)
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
    scales: tuple,
    amax_g: Optional[cute.Tensor],
    amax_v: Optional[cute.Tensor],
    amax_p: Optional[cute.Tensor],
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
    descale_q, descale_k, descale_v, descale_s, scale_s, descale_do, descale_dp, scale_g, scale_v, scale_dp = tuple(_scalar(s) for s in scales)
    s_scale = scale * descale_q * descale_k
    delta_scale = scale * descale_s * scale_dp
    dp_scale = delta_scale * descale_v * descale_do
    am = opaque_f32_zero()
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
                _load_pair(slab, full, phase, qmap, kmap, owner * M, other * N, hq, hk, batch, D, 0)
            else:
                _load_pair(slab, full, phase, kmap, qmap, owner * M, other * N, hk, hq, batch, D, 0)
            phase = phase ^ 1
            _product(slab, done, math_phase, tmem, TS, D)
            math_phase = math_phase ^ 1
            if cutlass.const_expr(dq):
                _load_pair(slab, full, phase, domap, vmap, owner * M, other * N, hq, hk, batch, D, 0)
            else:
                _load_pair(slab, full, phase, vmap, domap, owner * M, other * N, hk, hq, batch, D, 0)
            phase = phase ^ 1
            _product(slab, done, math_phase, tmem, TDP, D)
            math_phase = math_phase ^ 1
            am = fmax_f32(am, _softmax(slab, tmem, lse, delta, batch, hq, owner, other, sq, skv, s_scale, scale_s, dp_scale, delta_scale, causal, window, dq))
            if cutlass.const_expr(dq):
                _load_pair(slab, full, phase, kmap, kmap, other * N, other * N, hk, hk, batch, C, chunk * C)
            else:
                _load_pair(slab, full, phase, qmap, domap, other * N, other * N, hq, hq, batch, C, chunk * C)
            phase = phase ^ 1
            _grad(slab, done, math_phase, tmem, live != 0, dq)
            math_phase = math_phase ^ 1
            live = cutlass.Int32(1)
    factor = descale_dp * (descale_k if cutlass.const_expr(dq) else descale_q)
    _output(slab, tmem, dgmap, owner, head, batch, chunk, TG, factor, scale_g, live != 0, dtype, amax_g)
    if cutlass.const_expr(not dq):
        _output(slab, tmem, dvmap, owner, head, batch, chunk, TV, descale_s * descale_do, scale_v, live != 0, dtype, amax_v)
    if cutlass.const_expr(amax_p is not None):
        _fold_amax(am / scale_dp, amax_p)
    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
    nvvm.barrier_cta_sync()
    if warp == 0:
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem, cutlass.Float32), cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)


kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    scales: tuple,
    amax_g: Optional[cute.Tensor],
    amax_v: Optional[cute.Tensor],
    amax_p: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    window: cutlass.Constexpr[int],
    dq: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    dtype = dg.element_type
    if cutlass.const_expr(dtype not in HALF_DTYPES and dtype != FP8):
        raise ValueError("D512 FP8 backward gradients must be FP8 E4M3, FP16, or BF16")
    if cutlass.const_expr(any(t.element_type != FP8 for t in (q, k, v, do)) or any(t.shape[3] != D for t in (q, k, v, do, dg, dv))):
        raise ValueError("D512 FP8 backward requires full D512 E4M3 inputs and D512 gradients")
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
        scales,
        amax_g,
        amax_v,
        amax_p,
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
    descale_q: cute.Tensor,
    descale_k: cute.Tensor,
    descale_v: cute.Tensor,
    descale_s: cute.Tensor,
    scale_s: cute.Tensor,
    descale_do: cute.Tensor,
    descale_dp: cute.Tensor,
    scale_dk: cute.Tensor,
    scale_dv: cute.Tensor,
    scale_dp: cute.Tensor,
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
):
    scales = (descale_q, descale_k, descale_v, descale_s, scale_s, descale_do, descale_dp, scale_dk, scale_dv, scale_dp)
    host(q, k, v, do, lse, delta, dk, dv, scales, amax_dk, amax_dv, amax_dp, scale, causal, WINDOW, False, stream)


@cute.jit
def dq_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    descale_q: cute.Tensor,
    descale_k: cute.Tensor,
    descale_v: cute.Tensor,
    descale_s: cute.Tensor,
    scale_s: cute.Tensor,
    descale_do: cute.Tensor,
    descale_dp: cute.Tensor,
    scale_dq: cute.Tensor,
    scale_dp: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    scale: cutlass.Float32,
    causal: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    ratio: cutlass.Constexpr[int] = 1,
):
    scales = (descale_q, descale_k, descale_v, descale_s, scale_s, descale_do, descale_dp, scale_dq, scale_dq, scale_dp)
    host(q, k, v, do, lse, delta, dq, dq, scales, amax_dq, None, amax_dp, scale, causal, WINDOW, True, stream)
