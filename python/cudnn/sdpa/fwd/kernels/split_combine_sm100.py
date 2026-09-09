# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-split combine (reduction) pass for the SM100 DSL prefill SDPA kernels.

Each split writes ``O_s`` (normalized by its own running sum) and ``lse_s`` into
a split-major workspace at batch coord ``b + s*B``; this pass reduces over ``s``.

A split whose range came out empty ends with total_sum == 0, which the epilogue
turns into ``O := 0 / lse := -inf`` — the identity here, so empty splits need no
special case.

One block per (q_row, head, batch); the block's threads stride over d_v.
"""

from typing import Callable, Optional, Tuple

from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

WARP_SIZE = 32
DEFAULT_THREADS = 128

NEG_INF = float("-inf")


@cute.kernel
def _combine_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] — split-major partial O
    lse_partial: cute.Tensor,  # [S*B, H, S_q]   — split-major partial LSE
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    # 1-element fp32 amax of the RECOMBINED O.  The per-split epilogues cannot
    # compute this: each sees only its own partial, and O is a convex
    # combination of those, so a max over partials over-reports (~2.9x at 8
    # splits).  The FP8 kernels therefore skip their in-kernel amax write when
    # SPLIT_KV > 1 and leave it to this pass.  None-specialized off otherwise.
    amax_o: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    threads: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    q_row = cute.arch.block_idx()[0]
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    op = cutlass.make_array_view(o_partial)
    lp = cutlass.make_array_view(lse_partial)
    oo = cutlass.make_array_view(o_out)

    # --- pass 1: M = max_s lse_s, then den = sum_s exp(lse_s - M) ---
    # The split count is normally <= 8.  Assign one split to each lane and
    # exchange the block-uniform LSE-derived values within the warp instead of
    # making every lane reload and exponentiate every split.  Keep the original
    # loop for unusually large split counts.
    neg_inf = cutlass.Float32(NEG_INF)
    lane = tidx & cutlass.Int32(WARP_SIZE - 1)
    # Every lane in a warp must execute the shuffles below.  Envelope head
    # dimensions only require a multiple of 8, so retain the scalar fallback
    # when the final warp is partial.
    warp_combine = (n_splits <= cutlass.Int32(WARP_SIZE)) & ((d_v & cutlass.Int32(WARP_SIZE - 1)) == cutlass.Int32(0))
    lse_lane = neg_inf
    m = neg_inf
    if warp_combine:
        if lane < n_splits:
            lse_row = lp[batch + lane * n_batch, head, :]
            lse_lane = cutlass.Float32(lse_row[q_row])
        for s in cutlass.range(0, n_splits, 1, unroll=1):
            m = cute.math.max(m, cutlass.Float32(cute.arch.shuffle_sync(lse_lane, s)))
    else:
        for s in cutlass.range(0, n_splits, 1, unroll=1):
            lse_row = lp[batch + s * n_batch, head, :]
            m = cute.math.max(m, cutlass.Float32(lse_row[q_row]))

    # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather
    # than exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
    all_dead = m == cutlass.Float32(NEG_INF)
    m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))

    # Same reasoning as pass 2: skip dead splits rather than trusting a fastmath
    # exp(-inf) to be exactly 0.
    weight_lane = cutlass.Float32(0.0)
    live_lane = cutlass.Int32(0)
    den = cutlass.Float32(0.0)
    if warp_combine:
        if lse_lane > neg_inf:
            weight_lane = cute.math.exp(lse_lane - m_safe, fastmath=True)
            live_lane = cutlass.Int32(1)
        for s in cutlass.range(0, n_splits, 1, unroll=1):
            den = den + cutlass.Float32(cute.arch.shuffle_sync(weight_lane, s))
    else:
        for s in cutlass.range(0, n_splits, 1, unroll=1):
            lse_row = lp[batch + s * n_batch, head, :]
            lse_s = cutlass.Float32(lse_row[q_row])
            if lse_s > neg_inf:
                den = den + cute.math.exp(lse_s - m_safe, fastmath=True)

    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_den.ir_value()))

    # --- pass 2: O = sum_s w_s O_s / den, accumulated in fp32 ---
    #
    # A dead split (empty KV range) carries lse_s = -inf, so its weight is
    # exp(-inf) == 0 and it should contribute nothing.  Relying on the ARITHMETIC
    # to erase it is not safe: 0 * x is NaN for a non-finite x, and under
    # fastmath the weight itself is only approximately zero.  Skip such splits
    # outright -- they are the identity element of this reduction by
    # construction, so branching is exact where multiplying is not.  (Observed:
    # d512 with 5 KV tiles over 8 splits produced NaN in the recombined O
    # without this guard, even though every partial slot held a clean
    # -inf / 0.)
    amax_local = cutlass.Float32(0.0)
    if warp_combine:
        for d0 in cutlass.range(tidx, d_v, threads, unroll=1):
            acc = cutlass.Float32(0.0)
            for s in cutlass.range(0, n_splits, 1, unroll=1):
                live_s = cutlass.Int32(cute.arch.shuffle_sync(live_lane, s))
                if live_s != cutlass.Int32(0):
                    w = cutlass.Float32(cute.arch.shuffle_sync(weight_lane, s))
                    o_row = op[batch + s * n_batch, q_row, head, :]
                    acc = acc + w * cutlass.Float32(o_row[d0])
            out_row = oo[batch, q_row, head, :]
            o_val = acc * inv_den
            out_row[d0] = o_val.to(o_out.element_type)
            if cutlass.const_expr(amax_o is not None):
                # amax over the fp32 PRE-CAST value, matching what the
                # single-pass epilogue reports.
                amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
    else:
        for d0 in cutlass.range(tidx, d_v, threads, unroll=1):
            acc = cutlass.Float32(0.0)
            for s in cutlass.range(0, n_splits, 1, unroll=1):
                lse_row = lp[batch + s * n_batch, head, :]
                lse_s = cutlass.Float32(lse_row[q_row])
                if lse_s > neg_inf:
                    w = cute.math.exp(lse_s - m_safe, fastmath=True)
                    o_row = op[batch + s * n_batch, q_row, head, :]
                    acc = acc + w * cutlass.Float32(o_row[d0])
            out_row = oo[batch, q_row, head, :]
            o_val = acc * inv_den
            out_row[d0] = o_val.to(o_out.element_type)
            if cutlass.const_expr(amax_o is not None):
                # amax over the fp32 PRE-CAST value, matching what the
                # single-pass epilogue reports.
                amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))

    # Reduce each warp before publishing its maximum.  The value is
    # non-negative, so its fp32 bit pattern orders the same as the float and an
    # integer atomicMax is exact -- the same trick the kernels' own epilogues
    # use.  One atomic per warp avoids having every lane contend on the same
    # global scalar (one atomic for a 32-thread block, four for the default).
    if cutlass.const_expr(amax_o is not None):
        _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
        amax_warp = cute.arch.warp_redux_sync(amax_local, kind="fmax")
        if (tidx & cutlass.Int32(31)) == cutlass.Int32(0):
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_warp.bitcast(cutlass.Int32))

    # --- the recombined LSE (only when the caller asked for Stats) ---
    if cutlass.const_expr(lse_out is not None):
        if tidx == cutlass.Int32(0):
            lo = cutlass.make_array_view(lse_out)
            lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
            lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(NEG_INF).ir_value(), lse_val.ir_value()))
            lo[batch, head, q_row] = lse_val


_combine_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    amax_o: Optional[cute.Tensor],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    threads: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, H, SQ, D = problem_size
    _combine_kernel(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        amax_o,
        cutlass.Int32(B),
        n_splits,
        cutlass.Int32(D),
        threads,
    ).launch(
        grid=(SQ, H, B),
        block=[threads, 1, 1],
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int,
    h: int,
    sq: int,
    d_v: int,
    splits: int,
    dtype_o: str = "f16",
    has_lse: bool = False,
    lse_stride: Optional[tuple[int, int, int]] = None,
    has_amax: bool = False,
    threads: int = DEFAULT_THREADS,
) -> Callable:
    """Compile the combine pass for one concrete (B, H, S_q, d_v, splits) shape.

    ``splits`` is baked into the workspace EXTENTS (batch dim ``splits*b``) but
    passed to the kernel as a runtime count, so the split axis is a dynamic loop
    — the pass is bandwidth-bound, so unrolling it buys nothing.  ``has_lse``
    controls whether the recombined LSE is written at all; with ``False`` the
    store is None-specialized out of the traced code.  ``has_amax`` does the
    same for the FP8-family amax of the recombined O.  ``lse_stride`` describes
    the caller-visible LSE output; the per-split LSE input workspace remains
    compact regardless of that final layout.  ``threads`` is compile-time so
    architectures with a large row grid can amortize the row-uniform reduction
    in one warp without changing the runtime callable signature.
    """
    if threads not in (WARP_SIZE, DEFAULT_THREADS):
        raise ValueError(f"split combine threads must be {WARP_SIZE} or {DEFAULT_THREADS}; got {threads}")
    elem = {"f16": cutlass.Float16, "bf16": cutlass.BFloat16}[dtype_o]

    fake_o_partial = cute.runtime.make_fake_compact_tensor(elem, (splits * b, sq, h, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fake_lse_partial = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (splits * b, h, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_o_out = cute.runtime.make_fake_compact_tensor(elem, (b, sq, h, d_v), stride_order=(3, 2, 1, 0), assumed_align=16)
    fake_lse_out = (
        (
            cute.runtime.make_fake_tensor(cutlass.Float32, (b, h, sq), lse_stride, assumed_align=4)
            if lse_stride is not None
            else cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, h, sq), stride_order=(2, 1, 0), assumed_align=16)
        )
        if has_lse
        else None
    )
    fake_amax_o = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4) if has_amax else None

    return cute.compile(
        _host,
        fake_o_partial,
        fake_lse_partial,
        fake_o_out,
        fake_lse_out,
        fake_amax_o,
        (b, h, sq, d_v),
        cutlass.Int32(0),
        threads,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
