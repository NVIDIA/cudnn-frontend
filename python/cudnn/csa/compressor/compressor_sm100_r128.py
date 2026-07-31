# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated ratio=128 CSA/HCA ``Compressor`` forward + backward kernels.

The generic ``compressor_sm100.py`` kernels keep the ENTIRE pooling window in per-thread
registers (a thread owns one ``(output block, head-dim vec-group)`` and materializes all
``win = coff * ratio`` window positions), which is optimal at the production
``ratio = 4`` (48 registers, 0 spill) but hits the 255-register cap from ``ratio = 32``
and spills kilobytes of local memory per thread at ``ratio = 128`` for both ``coff``
(measured: 2.5 KB st + 2.5 KB ld at ``coff = 1``, ~33 KB at ``coff = 2``).

The forward kernel removes the window residency entirely with a **chunked softmax**
whose per-chunk accumulation form is selected per ``nb_total`` schedule bucket
(online-rescale by default, two-phase where it measured faster; see
``_fwd_schedule_r128``):

  - The softmax reduction axis (window position ``k``) is independent per head-dim
    column, so each column only needs ``(m, den, acc)`` (max, exp-sum, weighted sum) —
    3 fp32 registers per dim regardless of the window length. The default buckets
    accumulate the triple **online** (classic running max with a predicated rescale;
    each operand is loaded once). The **two-phase** buckets first compute the exact
    chunk max (order-independent fp32 max, no exp), then accumulate ``den``/``acc``
    against that fixed max in ascending ``k`` — the eager two-pass softmax form per
    chunk, re-reading score/ape once from the L1/L2-resident chunk lines (kv is read
    once in both forms).
  - Lanes stay along head-dim columns (``vec`` adjacent dims per lane), so every
    window-position step is one fully coalesced 128-byte line per warp per operand —
    the layout is ``[token, coff * d]`` row-major, and splitting the window across
    lanes instead would make each lane read the same column of 32 different tokens
    (32 uncoalesced sectors per step).
  - The window is split across ``tchunks`` (``threadIdx.y``) chunk-rows of the CTA;
    each chunk-thread streams ``win / tchunks`` positions serially, and the
    ``tchunks`` partial triples per ``(block, dim)`` are merged once per output row
    through a small smem buffer (``3 * tchunks * threads_x * vec`` fp32) in a fixed
    serial order. ``tchunks = 1`` degenerates to a pure streaming kernel: no smem,
    no barrier.

Work decomposition: one CTA per output block row (``gridDim.x = nb_total``),
``gridDim.y`` spans head-dim column groups of ``threads_x * vec`` dims, CTA shape
``(threads_x, tchunks)``. Everything else (THD segment scan, static-capacity padding
rows with first-in-segment semantics, the ``coff == 2`` overlap window with the invalid
previous-block half for each segment's first block, fp32-only window math with a single
final bf16 rounding, pinned ``mul.rn``/``fma.rn``) matches ``compressor_sm100.py``.

Numerics: identical fp32 dataflow (bf16 kv/score loads widened to fp32, fp32 APE add,
fp32 exp / sums, one final bf16 rounding), but the forward reduction ORDER differs from
the ratio=4 kernel: per-chunk accumulation (online-rescale or exact-chunk-max two-phase,
schedule-selected) + fixed chunk merge instead of a two-pass serial pass over the whole
window. The result is a few fp32 ulps from the whole-window two-pass value, run-to-run
bitwise deterministic (fixed chunk boundaries and merge order, no atomics), and within
the r128 tolerance contract against the fp32-intermediate eager reference — same
values within final-bf16 rounding at the gate tolerances (absolute thresholds
calibrated on the gate's documented input distribution), the reference's NaN/Inf
propagation where its fp32 intermediates overflow, and fp64-oracle parity where the
fp32 intermediates stay finite (gate-checked; reproduce via
``benchmark/csa/gate_csa_compressor_r128.py``). ``exp`` is
``cute.math.exp`` exactly as in the ratio=4 kernel, EXCEPT in schedule buckets that
adopted the ``fastexp`` field (see ``_exp_fast``): those use the tolerance-contract
ex2.approx path, gated per bucket.

The backward kernel (``_compressor_bwd_r128_kernel``) stages each row's window into
shared memory chunk-parallel, accumulates per-chunk partial ``den`` / ``S`` sums
inside the e-pass (the approved ratio=128 deterministic tolerance contract), merges
them per column in a FIXED chunk order, and stores gradients with a hoisted ``1/den``
multiply — no serial sweeps, no per-element division. dKV/dScore are deterministic
(fixed orders, no atomics) and match the fp32-intermediate eager autograd within the
r128 tolerance contract (same values within the gate tolerances — absolute thresholds
calibrated on the gate's documented input distribution — the eager reference's NaN/Inf
propagation where its fp32 intermediates overflow, and fp64-oracle parity where the
fp32 intermediates stay finite), NOT bitwise. ``dAPE`` keeps the
ratio=4 contract (one fp32 atomic per ``(k, dim)`` per CTA into a caller-zeroed
buffer, amortized over ``rows_per_cta`` rows; not run-to-run deterministic), and the
kernel-side zero-writes to never-consumed ``dKV``/``dScore`` slots keep the ratio=4
ownership rules verbatim, parallelized across the CTA (the zero classes are up to 127
tokens at ratio=128).

This file intentionally does not touch the ratio=4 path; launch machinery mirrors
``compressor_sm100.py`` and reuses its cached fast-launcher infrastructure.
"""

from __future__ import annotations

import threading

import torch
import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
import cutlass.cute.arch as cute_arch
import cutlass.cute.math as cute_math

from .compressor_sm100 import (
    _EXT,
    _FastCache,
    _NEG_INF,
    _bf16_ptr,
    _f32_ptr,
    _ffma_rn,
    _fmul_rn,
    _i32_ptr,
    _raw_stream,
)

# One (block row, dim vec-group) chunk-softmax partial: chunk max / exp-sum / weighted
# sum. Merged across the CTA's tchunks chunk-rows through smem.

# Fast exp (the ``fastexp`` schedule field; tolerance-contract buckets only):
# ``exp(x) = 2^(x * log2e)`` through ``ex2.approx.ftz.f32`` (MUFU.EX2). The log2e
# multiply is split hi + lo and recombined with one fma, which removes the
# REPRESENTATION error of rounding log2(e) to a single fp32 constant; the rounding of
# the hi product itself and MUFU.EX2's approximation error remain. This is an
# empirically gated approximation, NOT a bounded-error exp: the API places no range
# restriction on ``score + APE``, so no a-priori accuracy bound is claimed — every
# bucket that enables the field must pass the r128 contract gate (tolerance vs fp32
# eager + fp64-oracle parity, ``benchmark/csa/gate_csa_compressor_r128.py``) on top of
# a measured win. Three issued instructions (FMUL + FFMA + MUFU.EX2) replace the
# ~8-instruction full-range expf sequence — measured ~25-30% of the window loop's
# issued instructions at the instruction-bound buckets (B200 SASS audit).
# exp(-inf) == +0 and NaN -> NaN are preserved by ex2.approx; ftz flushes only
# sub-2^-126 results, which softmax consumes as exact zeros (den >= exp(0) = 1 always
# survives).
_L2E_HI = 1.4426950216293335  # fp32(log2(e))
_L2E_LO = 1.925963033500011e-08  # fp32(log2(e) - _L2E_HI); residual ~4e-16


# ``cute.math.exp2``'s ``approx``/``ftz`` keywords are newer than the
# ``nvidia-cutlass-dsl>=4.5.0`` floor this project's dependency spec resolves to, so the
# same instruction is requested through ``fastmath=True``, which both versions accept.
# Both spellings lower to ``ex2.approx.ftz.f32`` -- the form the tolerance contract is
# calibrated on: on 4.6.1 the 16-kernel ``reg_probe_csa_compressor_r128.py`` PTX is
# byte-identical either way (1104 ``ex2.approx.ftz.f32``, same registers, no spills), and
# on 4.5.0 the same 1104 instructions are emitted.


def _exp_fast(x):
    y = _ffma_rn(x, cutlass.Float32(_L2E_LO), _fmul_rn(x, cutlass.Float32(_L2E_HI)))
    return cute_math.exp2(y, fastmath=True)


@cute.kernel
def _compressor_fwd_r128_kernel(
    mKV: cute.Tensor,  # flat [T * W] bf16, W = coff * d
    mScore: cute.Tensor,  # flat [T * W] bf16
    mAPE: cute.Tensor,  # flat [ratio * W] fp32
    mCu: cute.Tensor,  # [n_seq + 1] int32 (token cu_seqlens)
    mCuComp: cute.Tensor,  # [n_seq + 1] int32 (block cu_seqlens)
    mOut: cute.Tensor,  # flat [nb_total * d] bf16
    n_seq: cutlass.Int32,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    tchunks: cutlass.Constexpr,
    threads_x: cutlass.Constexpr,
    twophase: cutlass.Constexpr,
    fastexp: cutlass.Constexpr,
):
    """Forward: one CTA per output row; window chunked over threadIdx.y, chunk softmax.

    Thread ``(tidx, tidy)`` owns head dims ``[col * vec, col * vec + vec)`` (with
    ``col = bidy * threads_x + tidx``) and window positions
    ``[tidy * C, (tidy + 1) * C)`` (``C = win / tchunks``). Loads are vec-wide and
    coalesced along dims. For ``coff == 2`` a chunk never straddles the half-window
    boundary (``ratio % C == 0`` is enforced by the schedule), so the previous-block
    half (invalid for each segment's first block, ``bis == 0``) is skipped as a whole
    chunk: its partial stays empty (``den == 0``) and the merge ignores it.
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    ncol: cutlass.Constexpr = d // vec
    win: cutlass.Constexpr = 2 * ratio if coff == 2 else ratio
    C: cutlass.Constexpr = win // tchunks
    W: cutlass.Constexpr = coff * d
    col = bidy * threads_x + tidx
    bb = bidx  # one output row per CTA

    smem = cutlass.utils.SmemAllocator()
    # Partial-merge buffer, [tchunks][threads_x][vec] per quantity; unused (0 B) when
    # tchunks == 1 (the allocation below is skipped at trace time).
    if cutlass.const_expr(tchunks > 1):
        npart: cutlass.Constexpr = tchunks * threads_x * vec
        sM = smem.allocate_tensor(cutlass.Float32, cute.make_layout(npart), 16)
        sD = smem.allocate_tensor(cutlass.Float32, cute.make_layout(npart), 16)
        sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout(npart), 16)

    if col < ncol:
        cvec = col * vec

        # THD segment scan (identical to compressor_sm100): rows beyond the true
        # compressed count are static-capacity padding and gather the window from
        # token 0 with first-in-segment semantics, like the eager code.
        nb_valid = mCuComp[n_seq]
        seq_idx = cutlass.Int32(0)
        bis = cutlass.Int32(0)
        if bb < nb_valid:
            bis = cutlass.Int32(bb)
            for sg in cutlass.range(n_seq):
                cs = mCuComp[sg]
                ce = mCuComp[sg + 1]
                if bb >= cs:
                    if bb < ce:
                        seq_idx = sg
                        bis = bb - cs
        tok0 = mCu[seq_idx] + bis * ratio

        # This thread's window chunk [k0, k0 + C). Both coff forms share one loop
        # body: token row tok0 - ratio + k (== tok0 + k - ratio for the own half),
        # only the projection column / APE row / validity differ per half.
        k0 = tidy * C
        run_chunk = cutlass.Boolean(True)
        tok_row0 = tok0 + k0
        colbase = cvec
        ape_row0 = cutlass.Int32(k0)
        if cutlass.const_expr(coff == 2):
            tok_row0 = tok0 - ratio + k0
            if k0 < ratio:
                # Previous block's half-window: first-half projection columns, APE
                # row k; invalid (contributes nothing) for the segment's first block
                # and for static-capacity padding rows (bis == 0 in both cases).
                if bis == 0:
                    run_chunk = cutlass.Boolean(False)
            else:
                colbase = d + cvec
                ape_row0 = cutlass.Int32(k0 - ratio)

        # Chunk softmax over [k0, k0 + C), in one of two compile-time forms:
        #
        # - twophase (large d=128 contexts): phase A computes the exact chunk max
        #   (order-independent fp32 max, no exp), phase B accumulates den/acc against
        #   that fixed max in ascending k. Versus the online-rescale form this deletes
        #   the predicated rescale path (a second exp sequence + two multiplies that
        #   issued EVERY iteration for ~5-in-128 taken updates — ~40% of the loop's
        #   instructions, B200 SASS audit) at the cost of
        #   re-reading score/ape once per chunk. Measured win only where the kernel
        #   is issue-bound rather than latency/traffic-bound (131k-token d=128:
        #   c1 23.1 -> 18.5 us, c2 64.3 -> 49.6 us); the extra reload LOSES at small
        #   grids and at d=512 (DRAM ~40% there), so it is a schedule field, not a
        #   global change. Per chunk the arithmetic is exactly the eager two-pass
        #   softmax form (exp(s - max) with the true max).
        #
        # - online (all other buckets): classic running (m, den, acc) with a
        #   predicated rescale on running-max updates; loads each operand once.
        #
        # Both forms produce a (m, den, acc) partial with identical merge semantics;
        # both are run-to-run bitwise deterministic.
        fr_m = cute.make_rmem_tensor((vec,), cutlass.Float32)
        fr_d = cute.make_rmem_tensor((vec,), cutlass.Float32)
        fr_a = cute.make_rmem_tensor((vec,), cutlass.Float32)
        for j in cutlass.range_constexpr(vec):
            fr_m[j] = cutlass.Float32(_NEG_INF)
            fr_d[j] = cutlass.Float32(0.0)
            fr_a[j] = cutlass.Float32(0.0)

        if run_chunk:
            if cutlass.const_expr(twophase):
                for kk in cutlass.range(C, unroll=4):
                    off = cute.assume((tok_row0 + kk) * W + colbase, divby=vec)
                    aoff = cute.assume((ape_row0 + kk) * W + colbase, divby=vec)
                    fr_s = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    fr_p = cute.make_rmem_tensor((vec,), cutlass.Float32)
                    gS = cute.make_tensor(mScore.iterator + off, cute.make_layout(vec))
                    gA = cute.make_tensor(mAPE.iterator + aoff, cute.make_layout(vec))
                    cute.autovec_copy(gS, fr_s)
                    cute.autovec_copy(gA, fr_p)
                    for j in cutlass.range_constexpr(vec):
                        s = cutlass.Float32(fr_s[j]) + cutlass.Float32(fr_p[j])
                        if s > cutlass.Float32(fr_m[j]):
                            fr_m[j] = s
                for kk in cutlass.range(C, unroll=4):
                    off = cute.assume((tok_row0 + kk) * W + colbase, divby=vec)
                    aoff = cute.assume((ape_row0 + kk) * W + colbase, divby=vec)
                    fr_s = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    fr_k = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    fr_p = cute.make_rmem_tensor((vec,), cutlass.Float32)
                    gS = cute.make_tensor(mScore.iterator + off, cute.make_layout(vec))
                    gK = cute.make_tensor(mKV.iterator + off, cute.make_layout(vec))
                    gA = cute.make_tensor(mAPE.iterator + aoff, cute.make_layout(vec))
                    cute.autovec_copy(gS, fr_s)
                    cute.autovec_copy(gK, fr_k)
                    cute.autovec_copy(gA, fr_p)
                    for j in cutlass.range_constexpr(vec):
                        s = cutlass.Float32(fr_s[j]) + cutlass.Float32(fr_p[j])
                        u = cutlass.Float32(fr_k[j])
                        if cutlass.const_expr(fastexp):
                            e = _exp_fast(s - cutlass.Float32(fr_m[j]))
                        else:
                            e = cute_math.exp(s - cutlass.Float32(fr_m[j]))
                        fr_d[j] = cutlass.Float32(fr_d[j]) + e
                        fr_a[j] = _ffma_rn(u, e, fr_a[j])
            else:
                for kk in cutlass.range(C, unroll=4):
                    off = cute.assume((tok_row0 + kk) * W + colbase, divby=vec)
                    aoff = cute.assume((ape_row0 + kk) * W + colbase, divby=vec)
                    fr_s = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    fr_k = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    fr_p = cute.make_rmem_tensor((vec,), cutlass.Float32)
                    gS = cute.make_tensor(mScore.iterator + off, cute.make_layout(vec))
                    gK = cute.make_tensor(mKV.iterator + off, cute.make_layout(vec))
                    gA = cute.make_tensor(mAPE.iterator + aoff, cute.make_layout(vec))
                    cute.autovec_copy(gS, fr_s)
                    cute.autovec_copy(gK, fr_k)
                    cute.autovec_copy(gA, fr_p)
                    for j in cutlass.range_constexpr(vec):
                        s = cutlass.Float32(fr_s[j]) + cutlass.Float32(fr_p[j])
                        u = cutlass.Float32(fr_k[j])
                        m_old = cutlass.Float32(fr_m[j])
                        if s > m_old:
                            # New running max: rescale den/acc by exp(m_old - s); the
                            # first valid position rescales by exp(-inf) == 0 exactly.
                            if cutlass.const_expr(fastexp):
                                scale = _exp_fast(m_old - s)
                            else:
                                scale = cute_math.exp(m_old - s)
                            fr_d[j] = _fmul_rn(fr_d[j], scale)
                            fr_a[j] = _fmul_rn(fr_a[j], scale)
                            fr_m[j] = s
                        if cutlass.const_expr(fastexp):
                            e = _exp_fast(s - cutlass.Float32(fr_m[j]))
                        else:
                            e = cute_math.exp(s - cutlass.Float32(fr_m[j]))
                        fr_d[j] = cutlass.Float32(fr_d[j]) + e
                        fr_a[j] = _ffma_rn(u, e, fr_a[j])

        if cutlass.const_expr(tchunks == 1):
            fr_o = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
            for j in cutlass.range_constexpr(vec):
                fr_o[j] = cutlass.BFloat16(cutlass.Float32(fr_a[j]) / cutlass.Float32(fr_d[j]))
            ooff = cute.assume(bb * d + cvec, divby=vec)
            gO = cute.make_tensor(mOut.iterator + ooff, cute.make_layout(vec))
            cute.autovec_copy(fr_o, gO)
        else:
            base = (tidy * threads_x + tidx) * vec
            for j in cutlass.range_constexpr(vec):
                sM[base + j] = cutlass.Float32(fr_m[j])
                sD[base + j] = cutlass.Float32(fr_d[j])
                sA[base + j] = cutlass.Float32(fr_a[j])

    if cutlass.const_expr(tchunks > 1):
        cute.arch.barrier()
        if col < ncol:
            if tidy == 0:
                # Fixed-order serial merge of the tchunks partials per dim. Empty
                # partials (den == 0: the skipped previous-block half) are ignored;
                # every row has at least one valid position (the own half), so the
                # merged den is >= 1 and the final division is safe.
                cvec = col * vec
                fr_o = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                for j in cutlass.range_constexpr(vec):
                    m = cutlass.Float32(sM[tidx * vec + j])
                    den = cutlass.Float32(sD[tidx * vec + j])
                    acc = cutlass.Float32(sA[tidx * vec + j])
                    for t in cutlass.range_constexpr(1, tchunks):
                        slot = (t * threads_x + tidx) * vec + j
                        d2 = cutlass.Float32(sD[slot])
                        if d2 > 0:
                            m2 = cutlass.Float32(sM[slot])
                            a2 = cutlass.Float32(sA[slot])
                            mn = m
                            if m2 > mn:
                                mn = m2
                            # exp(-inf - mn) == 0 handles a still-empty running
                            # partial; both-empty never reaches here (d2 > 0).
                            s1 = cute_math.exp(m - mn)
                            s2 = cute_math.exp(m2 - mn)
                            den = _ffma_rn(den, s1, _fmul_rn(d2, s2))
                            acc = _ffma_rn(acc, s1, _fmul_rn(a2, s2))
                            m = mn
                    fr_o[j] = cutlass.BFloat16(acc / den)
                ooff = cute.assume(bb * d + cvec, divby=vec)
                gO = cute.make_tensor(mOut.iterator + ooff, cute.make_layout(vec))
                cute.autovec_copy(fr_o, gO)


@cute.jit
def _compressor_fwd_r128_launch(
    kv_ptr: cute.Pointer,
    score_ptr: cute.Pointer,
    ape_ptr: cute.Pointer,
    cu_ptr: cute.Pointer,
    cuc_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    stream: cuda_driver.CUstream,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    tchunks: cutlass.Constexpr,
    threads_x: cutlass.Constexpr,
    twophase: cutlass.Constexpr,
    fastexp: cutlass.Constexpr,
):
    """JIT entry point that wraps raw pointers into tensors and launches forward."""
    lay = cute.make_layout(_EXT)
    mKV = cute.make_tensor(kv_ptr, lay)
    mScore = cute.make_tensor(score_ptr, lay)
    mAPE = cute.make_tensor(ape_ptr, lay)
    mCu = cute.make_tensor(cu_ptr, lay)
    mCuComp = cute.make_tensor(cuc_ptr, lay)
    mOut = cute.make_tensor(out_ptr, lay)
    ncol = d // vec
    gy = (ncol + threads_x - 1) // threads_x
    _compressor_fwd_r128_kernel(
        mKV,
        mScore,
        mAPE,
        mCu,
        mCuComp,
        mOut,
        n_seq,
        ratio,
        d,
        coff,
        vec,
        tchunks,
        threads_x,
        twophase,
        fastexp,
    ).launch(grid=(nb_total, gy, 1), block=(threads_x, tchunks, 1), stream=stream)


_COMPILED = {}
_COMPILE_LOCK = threading.Lock()
_FAST = _FastCache()

# nb_total-bucketed schedule tables (B200-measured; see _fwd_schedule_r128 docstring).
# Key (coff, d) -> (vec, tchunks, twophase, fastexp); threads_x is always derived
# (one warp per chunk-row). Configs absent from a table use the default schedule in
# that bucket.
_SMALL_NB_MAX = 128
_LARGE_NB_MIN = 1024
# 3rd field: two-phase chunk softmax (exact chunk max, then fixed-max accumulation)
# instead of the online-rescale loop — wins only where the kernel is issue-bound
# AND the fast exp alone does not already clear the issue bottleneck (after the
# fastexp lever only the c1d128 large bucket keeps it; at c2d128-131k the
# online+fastexp form measured 33.7 us vs two-phase 49.6 / two-phase+fastexp 57.8).
# 4th field: fast exp (ex2.approx path, see _exp_fast) — the remaining instruction
# diet; deterministic but tolerance-contract (not the ratio=4 expf bit pattern), so
# it is enabled per bucket by measured win (nsys pure-kernel, B200; the inline
# numbers) + the contract gate.
_SMALL_SCHEDULES = {
    (1, 128): (2, 8, False, False),
    (2, 128): (2, 16, False, True),  # 8.21 -> 6.42 us (1x8192)
    (2, 512): (2, 8, False, True),  # 14.42 -> 10.91 us (1x8192)
}
_LARGE_SCHEDULES = {
    (1, 128): (4, 4, True, True),  # 18.58 -> 15.07 us (1x131072)
    (1, 512): (4, 4, False, True),  # 85.41 (v2 exact) -> 77.55 us (1x131072)
    (2, 512): (4, 4, False, True),  # 152.35 -> 117.71 us (1x131072)
}
# Configs whose DEFAULT bucket also runs the fast exp (measured win across the whole
# band: c2d128 3x8192 12.19 -> 8.99, 1x32768 14.43 -> 10.91, 1x65536 25.56 -> 16.39
# us; c2d128 >= 1024 rows intentionally falls through to the same schedule,
# measured 49.57 (two-phase) -> 33.67 us at 1x131072).
_DEFAULT_FASTEXP = {(2, 128)}


def _fwd_schedule_r128(ratio, d, coff, nb_total=None):
    """Launch schedule ``(vec, tchunks, threads_x, twophase, fastexp)`` for the
    ratio=128 forward.

    ``vec = 2`` (32-bit paired bf16 accesses) for even ``d`` — except the widest rows
    (``coff * d >= 1024``), where ``vec = 4`` halves the issued loads and measured
    ~1.4-1.9x faster at the large shapes (d=512/coff=2) — scalar ``vec = 1`` for odd
    ``d``. ``threads_x = 32`` keeps one warp per chunk-row (a warp's ``32 * vec``
    adjacent dims are one or two full 128-byte lines) and doubles the CTA count at
    ``d = 128`` versus 64-wide CTAs. ``tchunks`` window chunks give ``tchunks``-way
    window parallelism per row at a ~KB smem merge cost. ``coff == 2`` requires
    ``tchunks >= 2`` so a chunk never straddles the half-window boundary.

    The schedule is additionally **bucketed by ``nb_total`` (output rows)** — the JIT
    cache key contains the schedule, so each bucket compiles once per config:

    - **small packs** (``nb_total <= _SMALL_NB_MAX``): a 64-row pack launches only
      ``nb * gy`` CTAs (128 for d=128) on a 148-SM B200 — most SMs hold a single
      4-warp CTA and global-load latency is unhidden (measured achieved occupancy
      6%, eligible warps 0.08/cycle). Doubling/quadrupling ``tchunks`` puts 2-4x the
      warps on the same rows: measured on B200 (nsys pure-kernel)
      c1d128 1x8192 7.75 -> ~5 us, c2d128 1x8192 10.5 -> ~7.3 us, c2d512 1x8192
      18.9 -> ~13.3 us (with ``vec = 2``; at 64 rows occupancy beats load width).
      c1d512 keeps the default (its ``gy = 8`` grid already fills the machine).
    - **large contexts** (``nb_total >= _LARGE_NB_MIN``): c1d128 switches to
      ``vec = 4`` / ``gy = 1`` — 1024 CTAs at 131k tokens = 0.58 waves (the default's
      2048 CTAs = 1.153 waves leave a 272-CTA tail wave that costs ~10%) — plus the
      two-phase softmax and fast exp; c2d512 keeps its default geometry and adds the
      fast exp. c2d128 intentionally has NO large entry: its default schedule (with
      the fast exp, ``_DEFAULT_FASTEXP``) measured faster than every two-phase
      variant at 131k.
    - The boundaries are measured at 64 rows (win) and 256 rows (default wins); the
      64..128-row band applies the small schedule on the occupancy argument (256
      CTAs still leave SMs at <=2 of 12 resident CTAs); 129..1023 rows use the
      default; >= 1024 rows use the large table where it exists.

    The ``twophase`` and ``fastexp`` fields select the per-bucket arithmetic form
    (see the kernel docstring and ``_exp_fast``); ``fastexp`` buckets are the
    tolerance-contract instruction diet, adopted strictly per measured win + gate.
    Every schedule field is compile-time (part of the JIT cache key). All entries
    and boundaries are B200-measured (nsys pure-kernel A/B per bucket) and gated on
    the r128 numerics contract (``benchmark/csa/gate_csa_compressor_r128.py``).
    """
    W = coff * d
    if W >= 1024 and d % 4 == 0:
        vec = 4
    else:
        vec = 2 if d % 2 == 0 else 1
    tchunks = 8 if coff == 2 and vec == 2 else 4
    twophase = False
    fastexp = (coff, d) in _DEFAULT_FASTEXP
    if nb_total is not None:
        if nb_total <= _SMALL_NB_MAX and (coff, d) in _SMALL_SCHEDULES:
            vec, tchunks, twophase, fastexp = _SMALL_SCHEDULES[(coff, d)]
        elif nb_total >= _LARGE_NB_MIN and (coff, d) in _LARGE_SCHEDULES:
            vec, tchunks, twophase, fastexp = _LARGE_SCHEDULES[(coff, d)]
    if d % vec != 0:
        raise ValueError(f"vec={vec} must divide head_dim ({d})")
    ncol = d // vec
    threads_x = 32 if ncol >= 32 else ncol
    win = 2 * ratio if coff == 2 else ratio
    if win % tchunks != 0:
        raise ValueError(f"tchunks={tchunks} must divide the window ({win})")
    if coff == 2 and ratio % (win // tchunks) != 0:
        raise ValueError(f"coff=2 chunks must not straddle the half-window boundary (ratio={ratio}, chunk={win // tchunks})")
    if threads_x * tchunks > 1024:
        raise ValueError(f"CTA too large: {threads_x} x {tchunks}")
    return vec, tchunks, threads_x, twophase, fastexp


def _compile_fwd_r128(key, args, ratio, d, coff, schedule):
    """JIT-compile the forward launch entry for ``key`` (capture-guarded)."""
    with _COMPILE_LOCK:
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"CSA compressor (r128): first call for config {key} happened under "
                    "CUDA graph capture (JIT compilation is not capture-safe); compile() "
                    "or run one eager step for this configuration before capturing."
                )
            fn = cute.compile(_compressor_fwd_r128_launch, *args, ratio, d, coff, *schedule)
            _COMPILED[key] = fn
    return fn


def precompile_fwd_r128(ratio, d, coff, device, nb_total=None):
    """Ensure the ratio=128 forward kernel(s) for this configuration are JIT-compiled.

    With ``nb_total`` given, compiles exactly the schedule bucket that shape will use;
    without it, compiles every bucket this config can select at runtime (small /
    default / large), so a subsequent CUDA-graph capture cannot hit a cold bucket.
    """
    if nb_total is not None:
        candidates = [nb_total]
    else:
        candidates = [1, _SMALL_NB_MAX + 1, _LARGE_NB_MIN]
    schedules = {_fwd_schedule_r128(ratio, d, coff, nb) for nb in candidates}
    with torch.cuda.device(device):
        scratch_bf16 = torch.zeros(16, device=device, dtype=torch.bfloat16)
        scratch_f32 = torch.zeros(16, device=device, dtype=torch.float32)
        scratch_i32 = torch.zeros(16, device=device, dtype=torch.int32)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
        args = (
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _f32_ptr(scratch_f32),
            _i32_ptr(scratch_i32),
            _i32_ptr(scratch_i32),
            _bf16_ptr(scratch_bf16),
            cutlass.Int32(0),
            cutlass.Int32(1),
            stream,
        )
        for schedule in schedules:
            key = ("r128fwd", ratio, d, coff, schedule, device.index)
            if key in _COMPILED:
                continue
            _compile_fwd_r128(key, args, ratio, d, coff, schedule)


def run_fwd_r128(kv, score, ape, cu_i, cuc_i, out, nb_total, ratio, d, coff, stream_handle=None):
    """Launch the ratio=128 forward kernel (cached fast path -> slow path -> JIT).

    Same contract as ``compressor_sm100.run_fwd``: flat contiguous bf16 kv/score/out,
    fp32 ape, int32 cu_seqlens/cu_seqlens_comp, launch anchored in ``kv``'s device.
    ``nb_total == 0`` launches nothing. The launch schedule is bucketed by ``nb_total``
    (see ``_fwd_schedule_r128``); each bucket JIT-compiles once per config.
    """
    if nb_total == 0:
        return
    dev = kv.device.index
    schedule = _fwd_schedule_r128(ratio, d, coff, nb_total)
    key = ("r128fwd", ratio, d, coff, schedule, dev)
    if stream_handle is None:
        stream_handle = _raw_stream(dev)
    with torch.cuda.device(dev):
        launcher = _FAST.get(key)
        if launcher is not None:
            slots = launcher.slots
            slots[0].value = kv.data_ptr()
            slots[1].value = score.data_ptr()
            slots[2].value = ape.data_ptr()
            slots[3].value = cu_i.data_ptr()
            slots[4].value = cuc_i.data_ptr()
            slots[5].value = out.data_ptr()
            slots[6].value = nb_total
            slots[7].value = cu_i.numel() - 1
            slots[8].value = stream_handle
            launcher.launch()
            return
        stream = cuda_driver.CUstream(stream_handle)
        args = (
            _bf16_ptr(kv),
            _bf16_ptr(score),
            _f32_ptr(ape),
            _i32_ptr(cu_i),
            _i32_ptr(cuc_i),
            _bf16_ptr(out),
            cutlass.Int32(nb_total),
            cutlass.Int32(cu_i.numel() - 1),
            stream,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            fn = _compile_fwd_r128(key, args, ratio, d, coff, schedule)
        fn(*args)
        _FAST.put(key, fn, args)


# =============================================================================
# Backward
# =============================================================================


@cute.kernel
def _compressor_bwd_r128_kernel(
    mKV: cute.Tensor,  # flat [T * W] bf16, W = coff * d
    mScore: cute.Tensor,  # flat [T * W] bf16
    mAPE: cute.Tensor,  # flat [ratio * W] fp32
    mCu: cute.Tensor,  # [n_seq + 1] int32
    mCuComp: cute.Tensor,  # [n_seq + 1] int32
    mGO: cute.Tensor,  # flat [nb_total * d] bf16
    mGKV: cute.Tensor,  # flat [T * W] bf16 (fully written; may be uninitialized)
    mGS: cute.Tensor,  # flat [T * W] bf16 (fully written; may be uninitialized)
    mGAPE: cute.Tensor,  # flat [ratio * W] fp32 (zero-initialized)
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    total_tokens: cutlass.Int32,
    rows_per_cta: cutlass.Int32,  # runtime: one compiled kernel serves every row count
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    tchunks: cutlass.Constexpr,
    threads_x: cutlass.Constexpr,
    fastexp: cutlass.Constexpr,
):
    """Backward: staged-smem chunk-parallel phases with fused reductions.

    Phases per output row — stage -> e-pass+partials -> fixed-order merge -> store,
    4 barriers/row (CTA per ``rows_per_cta`` row group, CTA shape
    ``(threads_x, tchunks)``, ``gridDim.y`` spanning head-dim column groups). This
    reduction structure is the approved ratio=128 deterministic tolerance contract.

    1. **Stage (chunk-parallel):** as in the forward, chunk-row ``tidy`` owns window
       positions ``[tidy * C, (tidy + 1) * C)`` (``C = win / tchunks``); it loads its
       score/APE/kv slices coalesced (lanes on head-dim columns, ``vec`` dims per
       lane), stages ``s_k = f32(score) + ape`` into an fp32 smem tile and ``kv_k``
       into a bf16 smem tile, and tracks its chunk max. The invalid previous-block
       half of a segment's first block (``coff == 2``, ``bis == 0``) is staged as the
       constant ``(-inf, 0)`` pair, so the reductions need no validity special-casing.
    2. **e-pass + partials (chunk-parallel):** every chunk-row merges the ``tchunks``
       staged maxes for its columns (order-independent), overwrites its slots with
       ``e_k = exp(s_k - mx)``, and accumulates the chunk-partial sums
       ``den_c = sum e_k`` and ``S'_c = sum fma(dp_k, e_k)`` (``dp_k = go * kv_k``)
       into one smem slot per (chunk, column). Invalid slots contribute exactly 0.
    3. **Merge (one lane per column):** ``den`` and ``S'`` are reduced over the
       ``tchunks`` partials in a FIXED chunk order (deterministic), then the lane
       publishes ``1/den`` and ``S = S'/den`` — two IEEE divisions per COLUMN instead
       of one per element.
    4. **Store (chunk-parallel):** ``p_k = e_k * (1/den)`` (multiply, not divide),
       ``dkv_k = bf16(go * p_k)``, ``ds_k = bf16(fma.rn(p_k, -S, mul.rn(dp_k, p_k)))``.
       ``dAPE`` is accumulated into ``C * vec`` per-thread registers across the CTA's
       ``rows_per_cta`` rows and reduced with one fp32 atomic per ``(k, dim)`` per CTA
       at the end.

    The den/S reduction ORDER (per-chunk then fixed merge) and the reciprocal
    rounding differ from the eager serial scan — dKV/dScore are deterministic but
    tolerance-vs-eager, gate-checked against the forward-style tolerances and an
    fp64 oracle. Measured 1.10-1.47x on B200 across the shipped envelope over a
    bitwise-pinned reduction structure (serial ascending-``k`` ``den``/``S`` sweeps
    + a per-element ``div.rn`` p-pass, the ratio=4 op order).

    Kernel-side zero-writes keep the ratio=4 ownership verbatim, parallelized across
    the CTA (each class is up to 127 tokens at ratio=128, or 128 rows for the
    ``coff == 2`` first-half class):

      - per-segment tail tokens (``seqlen % ratio``, both halves) and, for
        ``coff == 2``, the first-half columns of the segment's LAST block's own tokens
        — written by that last block's CTA, tokens strided over ``tidy``;
      - all tokens of segments with zero output blocks (``seqlen < ratio``) — written
        by the ``bidx == 0`` CTA column, tokens strided over ``tidy``;
      - tokens beyond ``cu_seqlens[-1]`` (static token-capacity padding) — strided
        over ``(bidx, tidy)`` grid rows.

    Rows in ``[cu_seqlens_comp[-1], nb_total)`` are static-capacity padding; their
    incoming gradients are ignored, as in the ratio=4 backward.
    """
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    gdimx, _, _ = cute.arch.grid_dim()
    ncol: cutlass.Constexpr = d // vec  # vec-group count per output row
    win: cutlass.Constexpr = 2 * ratio if coff == 2 else ratio
    C: cutlass.Constexpr = win // tchunks
    W: cutlass.Constexpr = coff * d
    cols_pc: cutlass.Constexpr = threads_x * vec  # head-dim columns per CTA
    col = bidy * threads_x + tidx
    cvec = col * vec
    ZERO_BF16 = cutlass.BFloat16(0.0)

    smem = cutlass.utils.SmemAllocator()
    # [win][cols_pc] fp32 tile: holds s_k after stage, e_k after the e-pass (``p_k``
    # is formed at store time as ``e_k * (1/den)``). [win][cols_pc] bf16 tile: staged
    # kv. Small buffers for the chunk maxes and the published per-column den / S.
    sE = smem.allocate_tensor(cutlass.Float32, cute.make_layout(win * cols_pc), 16)
    sKV = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(win * cols_pc), 16)
    sMax = smem.allocate_tensor(cutlass.Float32, cute.make_layout(tchunks * cols_pc), 16)
    sDen = smem.allocate_tensor(cutlass.Float32, cute.make_layout(cols_pc), 16)
    sS = smem.allocate_tensor(cutlass.Float32, cute.make_layout(cols_pc), 16)
    # Per-chunk partial sums of den and S' = sum_k fma(dp_k, e_k) (the tolerance
    # contract's fused reductions), merged in fixed chunk order by one lane per
    # column — no serial sweeps, and the per-element p-pass division becomes
    # p = e * (1/den) at store time.
    sDenP = smem.allocate_tensor(cutlass.Float32, cute.make_layout(tchunks * cols_pc), 16)
    sSP = smem.allocate_tensor(cutlass.Float32, cute.make_layout(tchunks * cols_pc), 16)

    nb_valid = mCuComp[n_seq]

    fr_z = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
    for j in cutlass.range_constexpr(vec):
        fr_z[j] = ZERO_BF16

    # --- zero sweeps for never-consumed slots with no owning block row ---
    if col < ncol:
        # Segments with zero output blocks (seqlen < ratio, up to 127 tokens each):
        # owned by the bidx == 0 CTA column, tokens strided over the chunk-rows.
        if bidx == 0:
            for sg in cutlass.range(n_seq):
                if mCuComp[sg + 1] == mCuComp[sg]:
                    t0 = mCu[sg]
                    cnt = mCu[sg + 1] - t0
                    for i in cutlass.range((cnt - tidy + tchunks - 1) // tchunks):
                        t = t0 + tidy + i * tchunks
                        off = cute.assume(t * W + cvec, divby=vec)
                        cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + off, cute.make_layout(vec)))
                        cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + off, cute.make_layout(vec)))
                        if cutlass.const_expr(coff == 2):
                            off2 = cute.assume(t * W + d + cvec, divby=vec)
                            cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + off2, cute.make_layout(vec)))
                            cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + off2, cute.make_layout(vec)))

        # Static token-capacity padding of the gradient buffers: strided over
        # (bidx, tidy) grid rows. The quotient/remainder split keeps every
        # intermediate within int32 for any count < 2**31, as in the ratio=4 kernel.
        pad0 = mCu[n_seq]
        pad_count = total_tokens - pad0
        gr = bidx * tchunks + tidy
        nrows = gdimx * tchunks
        if gr < pad_count:
            my_count = pad_count // nrows
            if gr < pad_count % nrows:
                my_count = my_count + 1
            for i in cutlass.range(my_count):
                t = pad0 + gr + i * nrows
                off = cute.assume(t * W + cvec, divby=vec)
                cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + off, cute.make_layout(vec)))
                cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + off, cute.make_layout(vec)))
                if cutlass.const_expr(coff == 2):
                    off2 = cute.assume(t * W + d + cvec, divby=vec)
                    cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + off2, cute.make_layout(vec)))
                    cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + off2, cute.make_layout(vec)))

    # Per-thread chunk mapping, row-independent parts. A chunk never straddles the
    # coff == 2 half-window boundary (ratio % C == 0 enforced by the schedule), so
    # projection column base and APE row base are per-thread constants.
    k0 = tidy * C
    colbase = cutlass.Int32(cvec)
    ape_row0 = cutlass.Int32(k0)
    if cutlass.const_expr(coff == 2):
        if k0 >= ratio:
            colbase = cutlass.Int32(d + cvec)
            ape_row0 = cutlass.Int32(k0 - ratio)

    # dAPE accumulator: C * vec fp32 registers (<= 64 under the schedule defaults;
    # ptxas-verified spill-free), accumulated across the CTA's rows_per_cta rows,
    # one atomic per slot at the end. ``cutlass.range(..., unroll_full=True)``, not
    # ``range_constexpr``: same unrolled register zero-init with identical ptxas
    # register/spill counts and a slightly smaller prologue (the constexpr form
    # emitted one mov per slot and trips the DSL's slow-compile warning at the
    # 64-iteration schedules).
    fr_dape = cute.make_rmem_tensor((C * vec,), cutlass.Float32)
    for q in cutlass.range(C * vec, unroll_full=True):
        fr_dape[q] = cutlass.Float32(0.0)

    for rr in cutlass.range(rows_per_cta):
        bb = bidx * rows_per_cta + rr
        if bb < nb_valid:
            # Per-segment boundary scan (n_seq is small), as in the ratio=4 kernel.
            seq_idx = cutlass.Int32(0)
            bis = cutlass.Int32(bb)
            for sg in cutlass.range(n_seq):
                cs = mCuComp[sg]
                ce = mCuComp[sg + 1]
                if bb >= cs:
                    if bb < ce:
                        seq_idx = sg
                        bis = bb - cs
            tok0 = mCu[seq_idx] + bis * ratio
            is_last = bb + 1 == mCuComp[seq_idx + 1]

            # Both coff forms share one chunk loop body: token row tok0 - ratio + k
            # (== tok0 + k - ratio for the own half); only column base / APE row /
            # validity differ per half (all per-thread constants above).
            tok_row0 = tok0 + k0
            run_chunk = cutlass.Boolean(True)
            if cutlass.const_expr(coff == 2):
                tok_row0 = tok0 - ratio + k0
                if k0 < ratio:
                    if bis == 0:
                        run_chunk = cutlass.Boolean(False)

            # ---- phase 1: chunk-parallel stage (s, kv -> smem tiles, chunk max) ----
            if col < ncol:
                sbase = k0 * cols_pc + tidx * vec
                fr_m = cute.make_rmem_tensor((vec,), cutlass.Float32)
                for j in cutlass.range_constexpr(vec):
                    fr_m[j] = cutlass.Float32(_NEG_INF)
                if run_chunk:
                    for kk in cutlass.range(C, unroll=4):
                        off = cute.assume((tok_row0 + kk) * W + colbase, divby=vec)
                        aoff = cute.assume((ape_row0 + kk) * W + colbase, divby=vec)
                        fr_s = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                        fr_k = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                        fr_a = cute.make_rmem_tensor((vec,), cutlass.Float32)
                        cute.autovec_copy(cute.make_tensor(mScore.iterator + off, cute.make_layout(vec)), fr_s)
                        cute.autovec_copy(cute.make_tensor(mKV.iterator + off, cute.make_layout(vec)), fr_k)
                        cute.autovec_copy(cute.make_tensor(mAPE.iterator + aoff, cute.make_layout(vec)), fr_a)
                        for j in cutlass.range_constexpr(vec):
                            s = cutlass.Float32(fr_s[j]) + cutlass.Float32(fr_a[j])
                            sE[sbase + kk * cols_pc + j] = s
                            sKV[sbase + kk * cols_pc + j] = fr_k[j]
                            if s > fr_m[j]:
                                fr_m[j] = s
                else:
                    # Invalid previous-block half: the constant (-inf, 0) pair, exactly
                    # the values the ratio=4 kernel feeds its serial window.
                    for kk in cutlass.range(C, unroll=4):
                        for j in cutlass.range_constexpr(vec):
                            sE[sbase + kk * cols_pc + j] = cutlass.Float32(_NEG_INF)
                            sKV[sbase + kk * cols_pc + j] = ZERO_BF16
                for j in cutlass.range_constexpr(vec):
                    sMax[tidy * cols_pc + tidx * vec + j] = cutlass.Float32(fr_m[j])
            cute.arch.barrier()

            # ---- phase 2: chunk-parallel e-pass (mx merge + exp + den/S' partials) ----
            # This pass ALSO accumulates the chunk's partial den / S' sums
            # (registers, then one smem slot per (chunk, column)).
            if col < ncol:
                sbase = k0 * cols_pc + tidx * vec
                fr_go2 = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                gooff2 = cute.assume(bb * d + cvec, divby=vec)
                cute.autovec_copy(cute.make_tensor(mGO.iterator + gooff2, cute.make_layout(vec)), fr_go2)
                for j in cutlass.range_constexpr(vec):
                    # Chunk-max merge: max is order-independent, so the chunked max
                    # equals the ratio=4 kernel's serial scan bitwise. Redundant per
                    # chunk-row (cheap smem broadcasts), which keeps the barrier count
                    # down. mx is finite: the own half always has valid positions.
                    mx = cutlass.Float32(sMax[tidx * vec + j])
                    for t in cutlass.range_constexpr(1, tchunks):
                        v = cutlass.Float32(sMax[t * cols_pc + tidx * vec + j])
                        if v > mx:
                            mx = v
                    # Invalid slots become exp(-inf - mx) == 0 exactly, the value the
                    # ratio=4 kernel's serial window feeds its den sum.
                    go2 = cutlass.Float32(fr_go2[j])
                    den_p = cutlass.Float32(0.0)
                    sp_p = cutlass.Float32(0.0)
                    for kk in cutlass.range(C, unroll=8):
                        slot = sbase + kk * cols_pc + j
                        if cutlass.const_expr(fastexp):
                            e = _exp_fast(cutlass.Float32(sE[slot]) - mx)
                        else:
                            e = cute_math.exp(cutlass.Float32(sE[slot]) - mx)
                        sE[slot] = e
                        den_p = den_p + e
                        dp = go2 * cutlass.Float32(sKV[slot])
                        sp_p = _ffma_rn(dp, e, sp_p)
                    sDenP[tidy * cols_pc + tidx * vec + j] = den_p
                    sSP[tidy * cols_pc + tidx * vec + j] = sp_p
            cute.arch.barrier()

            # ---- phase 3: fused merge (tolerance contract): den and S' merged in
            # fixed chunk order by one lane per column; publish 1/den (the store
            # phase multiplies) and S = S'/den. Deterministic: fixed order, no
            # atomics; the reduction ORDER differs from the eager serial scan, and
            # the per-element division becomes a reciprocal multiply — both inside
            # the approved tolerance contract (gate-checked). ----
            if tidy < vec:
                c = tidy * threads_x + tidx
                if bidy * cols_pc + c < d:
                    den = cutlass.Float32(0.0)
                    sp = cutlass.Float32(0.0)
                    for t in cutlass.range_constexpr(tchunks):
                        den = den + cutlass.Float32(sDenP[t * cols_pc + c])
                        sp = sp + cutlass.Float32(sSP[t * cols_pc + c])
                    # den >= 1 (the max element contributes exp(0)); both divisions
                    # are exact IEEE div.rn, once per column instead of per element.
                    sDen[c] = cutlass.Float32(1.0) / den
                    sS[c] = sp / den
            cute.arch.barrier()

            # ---- phase 4: chunk-parallel gradient stores + dAPE accumulation ----
            if col < ncol:
                sbase = k0 * cols_pc + tidx * vec
                fr_go = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                gooff = cute.assume(bb * d + cvec, divby=vec)
                cute.autovec_copy(cute.make_tensor(mGO.iterator + gooff, cute.make_layout(vec)), fr_go)
                if run_chunk:
                    for kk in cutlass.range_constexpr(C):
                        off = cute.assume((tok_row0 + kk) * W + colbase, divby=vec)
                        fr_gkv = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                        fr_gs = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                        for j in cutlass.range_constexpr(vec):
                            # sE holds e_k; sDen holds 1/den — the p-pass division
                            # became one multiply per element (tolerance contract).
                            p = _fmul_rn(cutlass.Float32(sE[sbase + kk * cols_pc + j]), cutlass.Float32(sDen[tidx * vec + j]))
                            go = cutlass.Float32(fr_go[j])
                            dp = go * cutlass.Float32(sKV[sbase + kk * cols_pc + j])
                            ds = _ffma_rn(p, -cutlass.Float32(sS[tidx * vec + j]), _fmul_rn(dp, p))
                            fr_gkv[j] = cutlass.BFloat16(go * p)
                            fr_gs[j] = cutlass.BFloat16(ds)
                            fr_dape[kk * vec + j] = cutlass.Float32(fr_dape[kk * vec + j]) + ds
                        cute.autovec_copy(fr_gkv, cute.make_tensor(mGKV.iterator + off, cute.make_layout(vec)))
                        cute.autovec_copy(fr_gs, cute.make_tensor(mGS.iterator + off, cute.make_layout(vec)))

                # The segment's last block zeroes the never-consumed slots it uniquely
                # owns: (a) for coff == 2 the first-half columns of its own tokens
                # (no next block consumes them), (b) the segment's tail tokens
                # (seqlen % ratio, both halves). Tokens strided over the chunk-rows.
                if is_last:
                    if cutlass.const_expr(coff == 2):
                        for i in cutlass.range((ratio - tidy + tchunks - 1) // tchunks):
                            t = tok0 + tidy + i * tchunks
                            offz = cute.assume(t * W + cvec, divby=vec)
                            cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + offz, cute.make_layout(vec)))
                            cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + offz, cute.make_layout(vec)))
                    tail0 = tok0 + ratio
                    cnt = mCu[seq_idx + 1] - tail0
                    for i in cutlass.range((cnt - tidy + tchunks - 1) // tchunks):
                        t = tail0 + tidy + i * tchunks
                        offz = cute.assume(t * W + cvec, divby=vec)
                        cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + offz, cute.make_layout(vec)))
                        cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + offz, cute.make_layout(vec)))
                        if cutlass.const_expr(coff == 2):
                            offz2 = cute.assume(t * W + d + cvec, divby=vec)
                            cute.autovec_copy(fr_z, cute.make_tensor(mGKV.iterator + offz2, cute.make_layout(vec)))
                            cute.autovec_copy(fr_z, cute.make_tensor(mGS.iterator + offz2, cute.make_layout(vec)))
            # Tiles are reused by the next row's stage phase.
            cute.arch.barrier()

    # One fp32 atomic per owned (k, dim) per CTA (amortized over rows_per_cta rows).
    # Chunks that never ran (invalid halves, padding-row CTAs) accumulated 0.0.
    if col < ncol:
        for kk in cutlass.range_constexpr(C):
            for j in cutlass.range_constexpr(vec):
                cute_arch.atomic_add(mGAPE.iterator + ((ape_row0 + kk) * W + colbase + j), cutlass.Float32(fr_dape[kk * vec + j]))


@cute.jit
def _compressor_bwd_r128_launch(
    kv_ptr: cute.Pointer,
    score_ptr: cute.Pointer,
    ape_ptr: cute.Pointer,
    cu_ptr: cute.Pointer,
    cuc_ptr: cute.Pointer,
    go_ptr: cute.Pointer,
    gkv_ptr: cute.Pointer,
    gs_ptr: cute.Pointer,
    gape_ptr: cute.Pointer,
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    total_tokens: cutlass.Int32,
    rows_per_cta: cutlass.Int32,
    stream: cuda_driver.CUstream,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    tchunks: cutlass.Constexpr,
    threads_x: cutlass.Constexpr,
    fastexp: cutlass.Constexpr,
):
    """JIT entry point that wraps raw pointers into tensors and launches backward."""
    lay = cute.make_layout(_EXT)
    mKV = cute.make_tensor(kv_ptr, lay)
    mScore = cute.make_tensor(score_ptr, lay)
    mAPE = cute.make_tensor(ape_ptr, lay)
    mCu = cute.make_tensor(cu_ptr, lay)
    mCuComp = cute.make_tensor(cuc_ptr, lay)
    mGO = cute.make_tensor(go_ptr, lay)
    mGKV = cute.make_tensor(gkv_ptr, lay)
    mGS = cute.make_tensor(gs_ptr, lay)
    mGAPE = cute.make_tensor(gape_ptr, lay)
    ncol = d // vec
    gx = (nb_total + rows_per_cta - 1) // rows_per_cta
    gy = (ncol + threads_x - 1) // threads_x
    _compressor_bwd_r128_kernel(
        mKV,
        mScore,
        mAPE,
        mCu,
        mCuComp,
        mGO,
        mGKV,
        mGS,
        mGAPE,
        nb_total,
        n_seq,
        total_tokens,
        rows_per_cta,
        ratio,
        d,
        coff,
        vec,
        tchunks,
        threads_x,
        fastexp,
    ).launch(grid=(gx, gy, 1), block=(threads_x, tchunks, 1), stream=stream)


# Small-pack backward bucket (mirrors the forward's nb_total bucketing; the JIT key
# contains the schedule so each bucket compiles once per config). At d=128 a small
# pack launches only nb * 2 CTAs at vec=2; vec=1 halves the columns per CTA and
# doubles gridDim.y, which fills the underoccupied machine (clock-safe interleaved
# A/B, B200: c1d128 1x8192 13.5 -> 11.5 us, 3x8192 21.8 -> 19.7,
# c2d128 1x8192 17.3 -> 13.8) and LOSES from 256 rows on (1x32768: 0.85x/0.65x) —
# boundary measured at 192 (win) / 256 (loss). The vec=1 buckets keep the exact exp:
# fastexp measured 0.905x (c1) / 1.003x (c2) on top of them.
_BWD_SMALL_NB_MAX = 192
# (coff, d) -> (vec, tchunks, fastexp); threads_x is always derived (one warp per
# chunk-row).
_BWD_SMALL_SCHEDULES = {
    (1, 128): (1, 8, False),
    (2, 128): (1, 8, False),
}

_SM_COUNT_CACHE = {}


def _sm_count(dev):
    """SM count for device index ``dev`` (cached; used by the rows_per_cta pick)."""
    n = _SM_COUNT_CACHE.get(dev)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _SM_COUNT_CACHE[dev] = n
    return n


def _bwd_schedule_r128(ratio, d, coff, nb_total=None):
    """Launch schedule ``(vec, tchunks, threads_x, fastexp)`` for the ratio=128
    backward.

    ``vec = 2`` (32-bit paired bf16 accesses) for even ``d``, scalar ``vec = 1`` for
    odd ``d`` — no ``vec = 4`` variant: it doubles the smem tiles and registers
    (ptxas 186-195 regs) and measured 0.42-0.63x from the residency collapse.
    **Small packs** (``nb_total <= _BWD_SMALL_NB_MAX``, d=128) switch to ``vec = 1``
    (see ``_BWD_SMALL_SCHEDULES``). ``threads_x = 32`` keeps one warp per chunk-row.
    ``tchunks`` trades chunk-parallel width against resident CTAs per SM (more CTAs
    overlap other CTAs' phases): measured optimum ``tchunks = 8`` across the
    envelope except ``coff = 1`` at ``d >= 512``, where the 8x wider column grid
    already fills the machine and ``tchunks = 4`` (fewer threads, one more CTA/SM)
    wins ~25% at long context. ``coff == 2`` additionally requires chunks not to
    straddle the half-window boundary. ``fastexp`` (see ``_exp_fast``) measured a
    uniform +3-8% everywhere EXCEPT the vec=1 small buckets — the default is True
    outside them.
    """
    W = coff * d
    vec = 2 if d % 2 == 0 else 1
    tchunks = 4 if coff == 1 and d >= 512 else 8
    fastexp = True
    if nb_total is not None:
        if nb_total <= _BWD_SMALL_NB_MAX and (coff, d) in _BWD_SMALL_SCHEDULES:
            vec, tchunks, fastexp = _BWD_SMALL_SCHEDULES[(coff, d)]
    if d % vec != 0:
        raise ValueError(f"vec={vec} must divide head_dim ({d})")
    ncol = d // vec
    threads_x = 32 if ncol >= 32 else ncol
    win = 2 * ratio if coff == 2 else ratio
    if win % tchunks != 0:
        raise ValueError(f"tchunks={tchunks} must divide the window ({win})")
    if coff == 2 and ratio % (win // tchunks) != 0:
        raise ValueError(f"coff=2 chunks must not straddle the half-window boundary (ratio={ratio}, chunk={win // tchunks})")
    if threads_x * tchunks > 1024:
        raise ValueError(f"CTA too large: {threads_x} x {tchunks}")
    # smem: fp32 s/e tile + bf16 kv tile + max merge + den/S' partials + den/S
    # publish, 16 B aligned.
    smem_bytes = win * threads_x * vec * 6 + 3 * tchunks * threads_x * vec * 4 + 2 * threads_x * vec * 4 + 64
    if smem_bytes > 227 * 1024:
        raise ValueError(f"backward smem tile too large ({smem_bytes} B) for W={W}")
    return vec, tchunks, threads_x, fastexp


def _bwd_rows_per_cta(nb_total, ratio, d, coff, dev):
    """Runtime ``rows_per_cta`` for the backward launch.

    The measured optimum is the smallest R whose grid fits ONE resident wave
    (``ctas_per_sm * sm_count``): the per-CTA row loop is a serial pipeline, so a
    grid a few percent over one wave costs a whole extra wave (measured cliffs:
    1.16 waves = 1.5x one wave at c2/d512/65536-token). Larger R additionally
    amortizes the dAPE atomics and the per-row segment scan, which is why R grows
    with the pack instead of capping at one row. ``ctas_per_sm`` is the schedule's
    measured residency (register/smem-bound): 2 for the coff=2 schedule (T=8,
    ~101 KB smem), 3 for coff=1 T=8, 4 for the coff=1 T=4 (d >= 512) schedule.
    Capped at 16 (beyond one wave per SM the pipeline depth stops paying).
    """
    vec, tchunks, threads_x = _bwd_schedule_r128(ratio, d, coff, nb_total)[:3]
    gy = (d // vec + threads_x - 1) // threads_x
    ctas_per_sm = 2 if coff == 2 else (4 if tchunks <= 4 else 3)
    slots = ctas_per_sm * _sm_count(dev)
    return max(1, min(16, -((nb_total * gy) // -slots)))


def _compile_bwd_r128(key, args, ratio, d, coff, schedule):
    """JIT-compile the backward launch entry for ``key`` (capture-guarded)."""
    with _COMPILE_LOCK:
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"CSA compressor (r128): first call for config {key} happened under "
                    "CUDA graph capture (JIT compilation is not capture-safe); compile() "
                    "or run one eager step for this configuration before capturing."
                )
            fn = cute.compile(_compressor_bwd_r128_launch, *args, ratio, d, coff, *schedule)
            _COMPILED[key] = fn
    return fn


def precompile_bwd_r128(ratio, d, coff, device, nb_total=None):
    """Ensure the ratio=128 backward kernel(s) for this configuration are JIT-compiled.

    With ``nb_total`` given, compiles exactly the schedule bucket that shape will use;
    without it, compiles every bucket this config can select at runtime (small /
    default), so a subsequent CUDA-graph capture cannot hit a cold bucket.
    """
    if nb_total is not None:
        candidates = [nb_total]
    else:
        candidates = [1, _BWD_SMALL_NB_MAX + 1]
    schedules = {_bwd_schedule_r128(ratio, d, coff, nb) for nb in candidates}
    with torch.cuda.device(device):
        scratch_bf16 = torch.zeros(16, device=device, dtype=torch.bfloat16)
        scratch_f32 = torch.zeros(16, device=device, dtype=torch.float32)
        scratch_i32 = torch.zeros(16, device=device, dtype=torch.int32)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
        args = (
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _f32_ptr(scratch_f32),
            _i32_ptr(scratch_i32),
            _i32_ptr(scratch_i32),
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _f32_ptr(scratch_f32),
            cutlass.Int32(0),
            cutlass.Int32(1),
            cutlass.Int32(0),
            cutlass.Int32(1),
            stream,
        )
        for schedule in schedules:
            key = ("r128bwd", ratio, d, coff, schedule, device.index)
            if key in _COMPILED:
                continue
            _compile_bwd_r128(key, args, ratio, d, coff, schedule)


def run_bwd_r128(kv, score, ape, cu_i, cuc_i, go, gkv, gs, gape, nb_total, ratio, d, coff, stream_handle=None):
    """Launch the ratio=128 backward kernel (cached fast path -> slow path -> JIT).

    Same contract as ``compressor_sm100.run_bwd``: recompute-from-inputs (flat
    contiguous bf16 kv/score/grad_out, fp32 ape, int32 cu_seqlens/cu_seqlens_comp),
    ``gkv``/``gs`` fully written (may be uninitialized), ``gape`` zero-initialized
    (fp32 atomics), launch anchored in ``kv``'s device. ``nb_total == 0`` launches
    nothing.
    """
    if nb_total == 0:
        return
    dev = kv.device.index
    schedule = _bwd_schedule_r128(ratio, d, coff, nb_total)
    key = ("r128bwd", ratio, d, coff, schedule, dev)
    total_tokens = kv.numel() // (coff * d)
    rows = _bwd_rows_per_cta(nb_total, ratio, d, coff, dev)
    if stream_handle is None:
        stream_handle = _raw_stream(dev)
    with torch.cuda.device(dev):
        launcher = _FAST.get(key)
        if launcher is not None:
            slots = launcher.slots
            slots[0].value = kv.data_ptr()
            slots[1].value = score.data_ptr()
            slots[2].value = ape.data_ptr()
            slots[3].value = cu_i.data_ptr()
            slots[4].value = cuc_i.data_ptr()
            slots[5].value = go.data_ptr()
            slots[6].value = gkv.data_ptr()
            slots[7].value = gs.data_ptr()
            slots[8].value = gape.data_ptr()
            slots[9].value = nb_total
            slots[10].value = cu_i.numel() - 1
            slots[11].value = total_tokens
            slots[12].value = rows
            slots[13].value = stream_handle
            launcher.launch()
            return
        stream = cuda_driver.CUstream(stream_handle)
        args = (
            _bf16_ptr(kv),
            _bf16_ptr(score),
            _f32_ptr(ape),
            _i32_ptr(cu_i),
            _i32_ptr(cuc_i),
            _bf16_ptr(go),
            _bf16_ptr(gkv),
            _bf16_ptr(gs),
            _f32_ptr(gape),
            cutlass.Int32(nb_total),
            cutlass.Int32(cu_i.numel() - 1),
            cutlass.Int32(total_tokens),
            cutlass.Int32(rows),
            stream,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            fn = _compile_bwd_r128(key, args, ratio, d, coff, schedule)
        fn(*args)
        _FAST.put(key, fn, args)
