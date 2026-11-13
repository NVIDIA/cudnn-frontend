# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused SSA epilogue for GVR's exact top-k output: one CTA per row takes
the K (unordered, exact) selected column ids, gathers their scores, sorts
ascending by column id (bitmap rank-scatter -- winner ids are unique, so a
sorted position is the popcount prefix of its bit; the same technique the
filtered select kernel uses, ~13 barriers vs bitonic's 66), applies the SSA
clamp(end-1, 0) padding, and computes predict = softmax over the selected
scores. Replaces the torch sort+gather+softmax post-process (measured 3.5 ms
at 256K/K=2048) with a single kernel.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from ..cute_dsl_filter_topk.filtered_top_k_varlen_util import (
    _SSA_LOG2E,
    _ssa_block_reduce_max,
    _ssa_block_reduce_sum,
)
from ..cute_dsl_filter_topk.filtered_top_k_varlen_util import atomic_add as _smem_atomic_add

I32 = cutlass.Int32
F32 = cutlass.Float32
SEL_THR = 512
BMAP_WORDS = 8192  # 32KB bitmap -> row length <= 256K


@cute.kernel
def _gvr_sort_predict_kernel(
    mScore: cute.Tensor,  # [rows, ncols] fp32 (materialized S); unused if have_val
    mSel: cute.Tensor,  # [rows, top_k] i32 GVR exact GLOBAL ids (unordered)
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mOutIdx: cute.Tensor,
    mOutProb: cute.Tensor,
    top_k: cutlass.Constexpr,
    ncols: cutlass.Constexpr,
    mVal: cute.Tensor = None,  # [rows, top_k] fp32: the selected scores, slot-
    have_val: cutlass.Constexpr = False,  # aligned with mSel. When given, the
    # kernel skips the mScore[r, gid] gather -- for the compact-select path
    # where S is never materialized and the values ride alongside the ids.
    mCandIdx: cute.Tensor = None,  # [rows, ncols] i32: global col of each cand.
    remap: cutlass.Constexpr = False,  # when set, mSel holds candval POSITIONS
    # (-1 = pad); the kernel gathers gid = mCandIdx[r, pos] on-device (no host
    # remap). Pairs with have_val (the compact-select reuse path).
    mCounts: cute.Tensor = None,  # [rows] i32: per-row RAW #collected (cursor)
    mFlags: cute.Tensor = None,  # [rows+1] i32: count, then flagged row ids
    mFallbackPred: cute.Tensor = None,  # bool predicates for conditional repair
    cert: cutlass.Constexpr = False,  # on-device cursor certificate: flag rows
    # longer than top_k that collected < top_k, or that overflowed cert_cap.
    cert_cap: cutlass.Constexpr = 0,  # candidate-buffer capacity (tie_cap)
):
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    E = cutlass.const_expr(top_k // SEL_THR)

    smem = utils.SmemAllocator()
    sIdx = cute.make_tensor(smem.allocate_array(I32, top_k, byte_alignment=128), cute.make_layout((top_k,)))
    sVal = cute.make_tensor(smem.allocate_array(F32, top_k, byte_alignment=128), cute.make_layout((top_k,)))
    sBm = cute.make_tensor(smem.allocate_array(I32, BMAP_WORDS, byte_alignment=128), cute.make_layout((BMAP_WORDS,)))
    sScan = cute.make_tensor(smem.allocate_array(I32, SEL_THR, byte_alignment=128), cute.make_layout((SEL_THR,)))
    sRed = cute.make_tensor(smem.allocate_array(F32, 32, byte_alignment=128), cute.make_layout((32,)))

    row_start = I32(mStarts[r])
    row_end = I32(mEnds[r])
    length = row_end - row_start
    _padv = row_end - I32(1)
    if _padv < 0:
        _padv = I32(0)

    # ---- on-device cursor certificate (mirror of the native single_list
    # check): a row is INEXACT iff it is LONGER than top_k yet collected fewer
    # than top_k (seed too tight), or it overflowed the candidate capacity.
    # Compact bad row ids behind the atomic count so the fallback can gather
    # and repair only those rows. ----
    if cutlass.const_expr(cert):
        if tidx == 0:
            cnt = I32(mCounts[r])
            if (length > I32(top_k) and cnt < I32(top_k)) or cnt > I32(cert_cap):
                pos = cute.arch.atomic_add(mFlags.iterator, I32(1), scope="gpu")
                mFlags[pos + I32(1)] = I32(r)
                # Arm only the first node. Each completed 128-row repair chunk
                # decides on-device whether the next nested node is needed.
                mFallbackPred[0] = True

    # ---- load GVR ids, clamp invalid/out-of-[start,end) to the pad, gather
    # the row's score for each; stash in registers ----
    wIdx = cute.make_rmem_tensor(cute.make_layout((E,)), I32)
    wVal = cute.make_rmem_tensor(cute.make_layout((E,)), F32)
    wWin = cute.make_rmem_tensor(cute.make_layout((E,)), I32)  # 1 = real winner
    for sl in cutlass.range_constexpr(E):
        i = I32(sl * SEL_THR) + tidx
        gid = I32(mSel[r, i])
        if cutlass.const_expr(remap):
            # mSel holds a candval POSITION (-1 = pad); gather the global col.
            if gid >= 0:
                gid = I32(mCandIdx[r, gid])
            else:
                gid = I32(-1)
        valid = gid >= row_start and gid < row_end and i < length
        # A slot is a REAL winner iff its id is valid; the bitmap/scatter must
        # gate on this (NOT i<length): the compact path has < top_k winners even
        # when length > top_k (only `count` candidates), so pad slots at i<length
        # would otherwise collide on _padv's bit and corrupt the ranks.
        wWin[sl] = I32(0)
        if valid:
            wWin[sl] = I32(1)
        cid = gid
        if not valid:
            cid = _padv
        wIdx[sl] = cid
        v = -cutlass.Float32.inf
        if valid:
            if cutlass.const_expr(have_val):
                # value rides alongside the id (compact-select: no dense S)
                v = F32(mVal[r, i])
            else:
                v = F32(mScore[r, cid])
        wVal[sl] = v

    # ---- sort winners ascending by global id + pad. Rows up to 256K (=
    # 32*BMAP_WORDS) use the bitmap rank-scatter (unique ids -> popcount-prefix
    # rank, ~13 barriers). LONGER rows (512K/1M) would overflow the bitmap's
    # 256K span, so fall back to a bitonic sort of the K winners -- independent
    # of row length (mirrors the native compact_select). ----
    use_bmap = length <= I32(32 * BMAP_WORDS) and length > I32(0)
    WPT = cutlass.const_expr(BMAP_WORDS // SEL_THR)
    if use_bmap:
        for w in cutlass.range_constexpr(WPT):
            sBm[I32(w * SEL_THR) + tidx] = 0
        cute.arch.barrier()
        for sl in cutlass.range_constexpr(E):
            if wWin[sl] != I32(0):
                loc = wIdx[sl] - row_start
                _smem_atomic_add(sBm.iterator + (loc >> 5), I32(1) << (loc & I32(31)))
        cute.arch.barrier()
        pc = I32(0)
        for w in cutlass.range_constexpr(WPT):
            pc = pc + I32(cute.arch.popc(sBm[I32(tidx * WPT) + w]))
        sScan[tidx] = pc
        cute.arch.barrier()
        off = 1
        for _st in cutlass.range_constexpr(9):
            vprev = I32(0)
            if tidx >= I32(off):
                vprev = sScan[tidx - I32(off)]
            cute.arch.barrier()
            if tidx >= I32(off):
                sScan[tidx] = sScan[tidx] + vprev
            cute.arch.barrier()
            off = off * 2
        for sl in cutlass.range_constexpr(E):
            if wWin[sl] != I32(0):
                loc = wIdx[sl] - row_start
                wq = loc >> 5
                tq = wq // I32(WPT)
                base = I32(0)
                if tq > 0:
                    base = sScan[tq - 1]
                wj = tq * I32(WPT)
                while wj < wq:
                    base = base + I32(cute.arch.popc(sBm[wj]))
                    wj = wj + 1
                mask = (I32(1) << (loc & I32(31))) - I32(1)
                rank = base + I32(cute.arch.popc(sBm[wq] & mask))
                sIdx[rank] = wIdx[sl]
                sVal[rank] = wVal[sl]
        # tail pad: #winners = total popcount (compact path may have < top_k)
        total = sScan[I32(SEL_THR - 1)]
        if total > I32(top_k):
            total = I32(top_k)
        for sl in cutlass.range_constexpr(E):
            i = I32(sl * SEL_THR) + tidx
            if i >= total:
                sIdx[i] = _padv
                sVal[i] = -cutlass.Float32.inf
        cute.arch.barrier()
    if use_bmap == False:  # noqa: E712 (DSL runtime bool)
        # bitonic fallback (row longer than the bitmap): winners (gvr's
        # unordered output) go straight into the winner buffer, pads to
        # (idx=end-1, val=-inf), then a bitonic sort over top_k keyed
        # (idx asc, val desc) so a real winner precedes a same-index pad.
        for sl in cutlass.range_constexpr(E):
            i = I32(sl * SEL_THR) + tidx
            if wWin[sl] != I32(0):
                sIdx[i] = wIdx[sl]
                sVal[i] = wVal[sl]
            else:
                sIdx[i] = _padv
                sVal[i] = -cutlass.Float32.inf
        cute.arch.barrier()
        for kk in cutlass.range_constexpr(int(top_k).bit_length() - 1):
            kbit = cutlass.const_expr(2 << kk)
            for jj in cutlass.range_constexpr(kk + 1):
                jbit = cutlass.const_expr((2 << kk) >> (jj + 1))
                cute.arch.barrier()
                for sl in cutlass.range_constexpr(E):
                    i = I32(sl * SEL_THR) + tidx
                    p = i ^ I32(jbit)
                    if p > i:
                        ia = sIdx[i]
                        ib = sIdx[p]
                        va = sVal[i]
                        vb = sVal[p]
                        gt = ia > ib or (ia == ib and va < vb)
                        lt = ia < ib or (ia == ib and va > vb)
                        do_swap = I32(0)
                        if (i & I32(kbit)) == 0:
                            if gt:
                                do_swap = I32(1)
                        else:
                            if lt:
                                do_swap = I32(1)
                        if do_swap == 1:
                            sIdx[i] = ib
                            sIdx[p] = ia
                            sVal[i] = vb
                            sVal[p] = va
        cute.arch.barrier()

    # ---- predict softmax over the selected scores (pads -> 0) ----
    lm = -cutlass.Float32.inf
    for sl in cutlass.range_constexpr(E):
        x = F32(sVal[I32(sl * SEL_THR) + tidx])
        if x > lm:
            lm = x
    m = _ssa_block_reduce_max(lm, sRed, tidx, SEL_THR // 32)
    cute.arch.barrier()
    ls = cutlass.Float32(0.0)
    for sl in cutlass.range_constexpr(E):
        x = F32(sVal[I32(sl * SEL_THR) + tidx])
        if x > -cutlass.Float32.inf:
            ls = ls + cute.exp2((x - m) * F32(_SSA_LOG2E), fastmath=True)
    s = _ssa_block_reduce_sum(ls, sRed, tidx, SEL_THR // 32)
    inv = cute.rcp(s)
    for sl in cutlass.range_constexpr(E):
        i = I32(sl * SEL_THR) + tidx
        x = F32(sVal[i])
        mOutIdx[r, i] = sIdx[i]
        if x > -cutlass.Float32.inf:
            mOutProb[r, i] = cute.exp2((x - m) * F32(_SSA_LOG2E), fastmath=True) * inv
        else:
            mOutProb[r, i] = cutlass.Float32(0.0)


class _GvrSortPredict:
    def __init__(self, rows, ncols, top_k, have_val=False, remap=False, cert=False, cert_cap=0):
        self.rows = rows
        self.ncols = ncols
        self.top_k = top_k
        self.have_val = have_val
        self.remap = remap
        self.cert = cert
        self.cert_cap = cert_cap

    @cute.jit
    def __call__(
        self, score, sel, starts, ends, oidx, oprob, val=None, candidx=None, counts=None, flags=None, fallback_pred=None, stream: cuda.CUstream = None
    ):
        _gvr_sort_predict_kernel(
            score,
            sel,
            starts,
            ends,
            oidx,
            oprob,
            top_k=self.top_k,
            ncols=self.ncols,
            mVal=val,
            have_val=self.have_val,
            mCandIdx=candidx,
            remap=self.remap,
            mCounts=counts,
            mFlags=flags,
            cert=self.cert,
            mFallbackPred=fallback_pred,
            cert_cap=self.cert_cap,
        ).launch(
            grid=(self.rows, 1, 1),
            block=(SEL_THR, 1, 1),
            stream=stream,
        )


_compiled = {}


def gvr_sort_predict(score, sel, starts, ends, out_idx, out_prob, values=None, candidx=None, counts=None, flags=None, fallback_pred=None, cert_cap=0):
    """Ascending-by-global-id rank-scatter + predict. Modes (additive):
    - ``values`` given: use them directly, skip the ``score`` gather (compact
      path, S never materialized). Else gather from the dense ``score`` row.
    - ``candidx`` given (remap): ``sel`` holds candval POSITIONS; the kernel
      gathers the global col id = ``candidx[r, pos]`` on-device (no host
      gather). Pairs with ``values``.
    - ``counts``+``flags`` given (cert): on-device cursor certificate."""
    rows, ncols = score.shape
    top_k = sel.shape[1]
    have_val = values is not None
    remap = candidx is not None
    cert = counts is not None and flags is not None
    have_fallback_pred = fallback_pred is not None
    key = (rows, ncols, top_k, have_val, remap, cert, have_fallback_pred, int(cert_cap))
    fn = _compiled.get(key)
    if fn is None:

        def _c(t, a=16):
            return from_dlpack(t, assumed_align=a, enable_tvm_ffi=True)

        args = [_c(score), _c(sel), _c(starts, 4), _c(ends, 4), _c(out_idx), _c(out_prob)]
        if have_val:
            args.append(_c(values))
        if remap:
            args.append(_c(candidx))
        if cert:
            args += [_c(counts, 4), _c(flags, 4)]
        if have_fallback_pred:
            args.append(_c(fallback_pred, 1))
        fn = cute.compile(
            _GvrSortPredict(rows, ncols, top_k, have_val=have_val, remap=remap, cert=cert, cert_cap=int(cert_cap)),
            *args,
            stream=make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compiled[key] = fn
    # runtime args must match the compile-time `args` order exactly (positional)
    runtime = [score, sel, starts, ends, out_idx, out_prob]
    if have_val:
        runtime.append(values)
    if remap:
        runtime.append(candidx)
    if cert:
        runtime += [counts, flags]
    if have_fallback_pred:
        runtime.append(fallback_pred)
    fn(*runtime)
