# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""KV-split combine (reduction) pass for the SM100 DSL prefill SDPA kernels.

Each split writes ``O_s`` (normalized by its own running sum) and ``lse_s`` into
a split-major workspace at batch coord ``b + s*B``; this pass reduces over ``s``.

A split whose range came out empty ends with total_sum == 0, which the epilogue
turns into ``O := 0 / lse := -inf`` — the identity here, so empty splits need no
special case.

Two kernels serve the lean fp32-partial path.  The LEAN kernel (``_combine_kernel``):
one warp per (q_row, head, batch), four rows per block, a lane owning four
columns of every split, the split axis walked in groups of ``G_T`` loads per
memory round trip.  The DECODE-GRID kernel (``_combine_decode_kernel``): one block
per (q_row, head, batch, ``DECODE_CHUNK_COLS``-column chunk of d_v) -- four
times the blocks at d 256 -- with the split axis spread over the block's eight
16-lane chains (two per warp; chain ``c`` owns splits ``16 c .. 16 c + 15`` of
every round of 128), every chain issuing all of its loads before any math, each
chain keeping an online-softmax (max, sum, four columns) in registers, the eight
chains meeting ONCE in a 2 KiB SMEM exchange behind the kernel's single CTA
barrier (every one of the 128 threads arrives exactly once; nothing exits or
diverges around it) and warp 0 merging them in chain order and storing the
row.  So a thread's dependent-load chain is one DRAM round trip for up to 128
splits where the lean kernel pays one per 16 splits per pass, and a decode step
(B 1 x 24 Q heads = 24 lean blocks on 148 .. 212 SMs) fills 96 SMs instead of
24.  Which kernel runs is decided per launch in ``_launch_combine`` from the
runtime geometry: the decode-grid kernel when the lean kernel's grid holds at
most ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks (and the lean path's
own geometry holds: fp32 partials, ``d_v % 4 == 0``, a 16-byte aligned slab),
the lean kernel otherwise -- a prefill grid never changes kernel; the
quantized and packed entries keep their own kernels.  ``compile_ptr(decode_grid=)``
forces one form (``DECODE_GRID_ON`` / ``DECODE_GRID_OFF``) for tests and as a kill
switch.  Order statement: every reduction of both kernels is in a FIXED order
(no atomics on O / LSE), so a launch is deterministic; for ``n_splits <= 16``
only chain 0 of the decode-grid kernel holds splits and its arithmetic is the
lean kernel's own sequence, so the two kernels' outputs are BITWISE identical;
past 16 splits the decode-grid kernel associates the sum differently (eight
chain partials merged with ``exp(m_c - m)`` factors) and the two agree to fp32
rounding.  The SMEM exchange (fp32 words: ``red_m[8]``, ``red_den[8]``, then
``red_acc[8 chains][16 column groups][4]``) is lane-written and lane-read at the
same per-thread 16-byte stride -- 16 threads x 16 B = two full 128-byte bank
cycles per chain, no descriptor reads it, so it carries no swizzle.

Gate in the combine (``compile_ptr(gate=True)``, the dense half entry only): a
split launch of a kernel that has no epilogue-gate seams -- the d256 decode
tile -- carries the graph's ``mul(O_v, sigmoid(G))`` tail HERE.  The gate is
applied to the fp32 MERGED value, before the single cast to the output dtype,
with the arithmetic the gated prefill kernels use in their epilogue
(``_common_blackwell.gate_epilogue_pairs``): ``h = acc * (inv_den / 2)`` and
``O = h * tanh(g / 2) + h`` (one FMUL2, one MUFU.TANH per element, one FFMA2),
the dead-row SELECT per element AFTER the fma (``sdpa-invariants.md`` section 2:
never a multiply by zero, a gate value is never trusted on a dead row).  So the
rounding convention is FROST's: ONE rounding, of ``O32 * sigmoid(G)``, exactly
what the fused epilogue does -- a gated split plan and a gated unsplit plan of
the same graph differ only by the summation order of the attention itself.  The
unfused references round the merged O to the output dtype BEFORE the fp32 gate
and round again after it (vLLM's split merge, and the gated attention block's
torch reference, which gates the output-dtype O): two roundings, so they differ
from this pass by up to one output ulp plus the approximate tanh.  LSE / Stats
never see the gate (the partial LSEs are gate-free and the final one is their
log-sum-exp).  The gate is read element-wise through the caller's BSHD strides
(no layout constraint of its own; the engine rows keep the zero-copy rule the
fused kernels need so one G declaration serves both paths).
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from typing import Callable, Optional, Tuple

from functools import lru_cache
import hashlib
from pathlib import Path

from cudnn.sdpa.fwd.kernels._quantized import _unscale_amax_kernel
from cudnn.sdpa.fwd.heuristics import DECODE_COMBINE_MAX_LEAN_BLOCKS as _DECODE_COMBINE_MAX_LEAN_BLOCKS

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cudnn.frost.tile_dsl.pointwise import ffma2, fmul2, opaque_f32_zero
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v2
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

# This helper is imported normally, outside the parameterized template loader.
# Supply its source identity so template_key can persist its pointer entry points;
# the cache environment manifest also covers all transitive Python sources.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

# One WARP per (q_row, head, batch) and ROWS_PER_BLOCK rows per block: each lane
# owns four consecutive columns (a 16-byte load per split, so a warp reads a
# row's 128 columns as one 512-byte coalesced request), and the wider
# d256/d512 flavors loop the warp over 128-column chunks.  The previous
# one-block-per-row form (128 lanes, one scalar load each, split loop rolled)
# was DRAM-latency bound: measured 15.5 us (bf16) / 28.7 us (fp8) for
# 985 x 9 rows x 4 splits on B300 against ~18 MB of traffic.
THREADS = 128
ROWS_PER_BLOCK = THREADS // 32
# Splits whose LSE and partial-O rows are loaded up front, before any math.
G_S = 4
# Splits past G_S are walked in GROUPS of G_T: every group's LSE and partial-O
# rows are loaded before any of its math, so a warp pays one memory round trip
# per group instead of one per split.  The old ``unroll=1`` walk issued one
# dependent global load per split (twice: the max pass and the accumulate
# pass, per 128-column chunk), and a launch with fewer rows than the machine
# holds at once had nothing to hide the chain behind: the d256 decode tile at
# batch 1 (QH warps in all) read 40 us for 64 splits where 2 splits cost 3.3 us
# -- ~0.3 us, one DRAM round trip, per split per pass.  The arithmetic is
# unchanged: the same splits are accumulated in the same order (a slot past
# ``n_splits`` re-reads the last split and contributes nothing, as the head
# group's slots do), so the pass is bitwise what it was for every split count.
G_T = 16

# The DECODE-GRID form of the lean fp32-partial path (``_combine_decode_kernel``):
# one block per (q_row, head, batch, DECODE_CHUNK_COLS-column chunk of d_v) -- 4x
# the blocks of the lean kernel at d 256 -- and the split axis spread over the
# block's DECODE_CHAINS chains of DECODE_LANES_PER_CHAIN lanes (two half-warps per
# warp; a lane owns four consecutive columns, one 16-byte load per split).  Every
# chain issues the LSE and partial-O loads of its DECODE_SPLITS_PER_CHAIN slots
# before any math, so a lane's dependent-load chain is ONE memory round trip for
# up to DECODE_SPLITS_PER_ROUND splits (the lean kernel pays one per G_T splits
# per pass); the chains' running (max, sum, acc) meet once in SMEM and are merged
# in a FIXED order (chain 0 .. 7) by warp 0, which also stores the row.  The
# selection between the two kernels is the lean kernel's grid size against
# heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS (``_launch_combine``): a decode step's
# tens of blocks take this form, a prefill's thousands keep the lean kernel.
DECODE_THREADS = 128
DECODE_CHUNK_COLS = 64
DECODE_LANES_PER_CHAIN = DECODE_CHUNK_COLS // 4
DECODE_CHAINS = DECODE_THREADS // DECODE_LANES_PER_CHAIN
DECODE_SPLITS_PER_CHAIN = G_T
DECODE_SPLITS_PER_ROUND = DECODE_CHAINS * DECODE_SPLITS_PER_CHAIN
# The SMEM exchange: red_m[chain], red_den[chain], then red_acc[chain][lane][4].
DECODE_RED_DEN_OFF = DECODE_CHAINS
DECODE_RED_ACC_OFF = 2 * DECODE_CHAINS
DECODE_RED_WORDS = DECODE_RED_ACC_OFF + DECODE_CHAINS * DECODE_LANES_PER_CHAIN * 4
# ``compile_ptr(decode_grid=...)``: which form a pointer entry launches.
DECODE_GRID_AUTO = 0  # the grid rule (the adapters' entry)
DECODE_GRID_ON = 1  # always the decode-grid kernel where its geometry holds (tests; an explicit decode flag)
DECODE_GRID_OFF = 2  # always the lean kernel (tests: the bitwise twin; a kill switch)


def lean_grid_blocks(b: int, h: int, s_q: int) -> int:
    """The lean kernel's grid in blocks: ``ceil(S_q / ROWS_PER_BLOCK) x H_q x B`` (the quantity the selection rule reads)."""
    return -(-int(s_q) // ROWS_PER_BLOCK) * int(h) * int(b)


def decode_grid_selected(b: int, h: int, s_q: int, d_v: int) -> bool:
    """The host-side twin of ``_launch_combine``'s selection: whether a dense half launch with fp32 partials
    (16-byte aligned, the lean path's own precondition) of this geometry takes the decode-grid kernel --
    ``d_v % 4 == 0`` and the lean kernel's grid at most ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks.
    Quantized entries (an amax or an output scale) and half partials always take the lean kernel."""
    return int(d_v) % 4 == 0 and lean_grid_blocks(b, h, s_q) <= _DECODE_COMBINE_MAX_LEAN_BLOCKS


NEG_INF = float("-inf")

# The FP8 entries are reachable only as the OUTPUT type, where this pass
# performs the single cast.  "f32" is a PARTIAL-only type: the split epilogue can
# write its fp32 accumulator straight to global (no SMEM staging), so the
# reduction reads exactly what the mainloop produced with no intermediate
# rounding.
_ELEM = {
    "f32": cutlass.Float32,
    "f16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
    "e4m3": cutlass.Float8E4M3FN,
    "e5m2": cutlass.Float8E5M2,
}


def _ld_global_f32(addr):
    """32-bit global load of an fp32 at ``addr``."""
    r = nvvm.inline_ptx(
        "ld.global.f32 $0, [$1];",
        write_only_types=[cutlass.Float32],
        read_only_args=[addr],
    )
    return cutlass.Float32(r[0] if isinstance(r, (tuple, list)) else r)


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
    # 1-element fp32 output scale, present only when O is stored quantized.
    # The split kernels leave it out of their epilogue and write UNSCALED
    # partials; this pass applies it at the single cast, so the FP8 rounding
    # happens once, on the recombined value.
    scale_o: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    s_q: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],  # write the FINAL LSE in base 2 (stats_use_log2)
    # Ragged final rows (the decode tile's RAGGED_Q leg): (B+1,) int32 / int64 ragged
    # offsets of Q, O and Stats in ELEMENTS and their elements-per-token
    # divisors.  The partials stay dense; only the final O / LSE rows move to
    # ``offset[b] / div + q_row`` on the packed [1, T, ...] outputs, and a row
    # at or past this sequence's Q length (offset[b+1] - offset[b]) / div --
    # every row of a zero-length sequence -- is not written at all.  None
    # (None-specialized) keeps the dense (batch, q_row) placement.
    ragged_q: Optional[cute.Tensor] = None,
    ragged_o: Optional[cute.Tensor] = None,
    ragged_lse: Optional[cute.Tensor] = None,
    ragged_q_div: cutlass.Int64 = 1,
    ragged_o_div: cutlass.Int64 = 1,
    ragged_lse_div: cutlass.Int64 = 1,
    # Packed token capacities of the final O and Stats buffers (ragged only): a
    # row whose ragged placement lands at or past them is not written, so a
    # short buffer or a bad offset can never store outside the caller's bytes.
    ragged_o_cap: cutlass.Int32 = 0,
    ragged_lse_cap: cutlass.Int32 = 0,
    # The gate G of the graph's ``mul(O_v, sigmoid(G))`` tail, [B, S_q, H, D] in
    # the caller's strides (the dense placement only), applied to the fp32 merged
    # value before the single cast -- see the module docstring.  None
    # (None-specialized) is the plain reduction; its traced code is unchanged.
    gate: Optional[cute.Tensor] = None,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(32)
    q_row = cute.arch.block_idx()[0] * cutlass.Int32(ROWS_PER_BLOCK) + tidx // cutlass.Int32(32)
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    # A warp past S_q (last block only) has no row; there is no block-level
    # barrier below, so it simply falls through. Everything inside is
    # warp-uniform except the per-lane column guard.
    if q_row < s_q:
        op = cutlass.make_array_view(o_partial)
        lp = cutlass.make_array_view(lse_partial)
        oo = cutlass.make_array_view(o_out)

        # Where the recombined row lands: dense (batch, q_row), or the ragged
        # placement on the packed outputs (batch coord 0).
        o_batch = batch
        o_tok = q_row
        lse_batch = batch
        lse_tok = q_row
        row_live = cutlass.Int32(1) == cutlass.Int32(1)
        lse_live = row_live
        if cutlass.const_expr(ragged_q is not None):
            # Offsets are int32 or int64 elements; divide in 64 bits (a large packed
            # buffer's element offset can exceed 2^31) and keep the token index in 32.
            # Bounds are checked in 64 bits BEFORE narrowing: a negative or wrapped
            # offset must fail the capacity test, not alias a valid token after the cast.
            rq = cutlass.make_array_view(ragged_q)
            q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
            q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
            in_seq = cutlass.Int64(q_row) < q_len64
            o_batch = cutlass.Int32(0)
            o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
            row_live = in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
            o_tok = cutlass.Int32(o_tok64)
            lse_live = row_live
            if cutlass.const_expr(ragged_lse is not None):
                lse_batch = cutlass.Int32(0)
                lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
                lse_live = in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
                lse_tok = cutlass.Int32(lse_tok64)

        neg_inf = cutlass.Float32(NEG_INF)
        zero = cutlass.Float32(0.0)
        # Pre-declared: both paths assign them and the tail below reads them after the join.
        amax_local = cutlass.Float32(0.0)
        m_safe = zero
        den = zero
        all_dead = zero > zero
        o_base = o_partial.iterator.toint()
        # fp32 partials with 16-byte rows: the lean path.  Lane l owns columns 4l..4l+3 of
        # every split; each (b, h, q_row) base offset is computed once and the per-split
        # address is a single 64-bit add (the split axis is the leading, batch-stacked mode).
        lean = zero > zero
        if cutlass.const_expr(o_partial.element_type == cutlass.Float32):
            lean = ((d_v & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
        if lean:
            lse_base = lse_partial.iterator.toint() + cutlass.Int64(cute.crd2idx((batch, head, q_row), lse_partial.layout)) * 4
            lse_step = cutlass.Int64(lse_partial.stride[0]) * cutlass.Int64(n_batch) * 4
            o_step = cutlass.Int64(o_partial.stride[0]) * cutlass.Int64(n_batch) * 4
            s_last = n_splits - cutlass.Int32(1)
            out_base = o_out.iterator.toint()
            st_ok = zero > zero
            if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
                _st_strides = cutlass.Int64(o_out.stride[0]) | cutlass.Int64(o_out.stride[1]) | cutlass.Int64(o_out.stride[2])
                st_ok = (
                    ((out_base & cutlass.Int64(7)) == cutlass.Int64(0))
                    & (cutlass.Int64(o_out.stride[3]) == cutlass.Int64(1))
                    & ((_st_strides & cutlass.Int64(3)) == cutlass.Int64(0))
                )
            q_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(scale_o is not None):
                q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
            # Gate in the combine: the 0.5 handed to the packed multiply must be opaque (a constant
            # float into inline_ptx ICEs libNVVM).  Hoisted once per row, like the fused kernels do
            # per tile; traced only on the gated entry, so the plain entries' code is unchanged.
            if cutlass.const_expr(gate is not None):
                gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
            for cbase in cutlass.range(0, d_v, 128, unroll=1):
                d0 = cbase + lane * cutlass.Int32(4)
                if d0 < d_v:
                    o_row_base = o_base + cutlass.Int64(cute.crd2idx((batch, q_row, head, d0), o_partial.layout)) * 4
                    # Every load of the row goes out first -- the LSE of the first G_S splits and their partial-O
                    # rows (a clamped split index re-reads the last split for slots past n_splits; those get
                    # weight 0) -- so a warp pays ONE memory round trip instead of one per pass.
                    lse_v = []
                    o_v = []
                    for s in cutlass.range_constexpr(G_S):
                        s_eff64 = cutlass.Int64(cute.math.min(cutlass.Int32(s), s_last))
                        lse_v.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
                        o_v.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
                    m = neg_inf
                    for s in cutlass.range_constexpr(G_S):
                        m = cute.math.max(m, lse_v[s])
                    # Splits past G_S extend the max before any weight is formed: G_T LSEs per
                    # round trip (a clamped slot re-reads the last split; max is idempotent).
                    if n_splits > cutlass.Int32(G_S):
                        for s0 in cutlass.range(G_S, n_splits, G_T, unroll=1):
                            lse_t = []
                            for j in cutlass.range_constexpr(G_T):
                                s_eff64 = cutlass.Int64(cute.math.min(s0 + cutlass.Int32(j), s_last))
                                lse_t.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
                            for j in cutlass.range_constexpr(G_T):
                                m = cute.math.max(m, lse_t[j])
                    # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather than
                    # exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
                    all_dead = m == neg_inf
                    m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m.ir_value()))
                    acc0 = zero
                    acc1 = zero
                    acc2 = zero
                    acc3 = zero
                    den = zero
                    for s in cutlass.range_constexpr(G_S):
                        lse_s = lse_v[s]
                        v = o_v[s]
                        live = (lse_s > neg_inf) & (cutlass.Int32(s) < n_splits)
                        e = cute.math.exp(lse_s - m_safe, fastmath=True)
                        w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
                        v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
                        v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
                        v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
                        v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
                        den = den + w
                        acc0 = acc0 + w * v0
                        acc1 = acc1 + w * v1
                        acc2 = acc2 + w * v2
                        acc3 = acc3 + w * v3
                    if n_splits > cutlass.Int32(G_S):
                        # G_T splits per round trip: every LSE and partial-O row of the group
                        # is loaded first, then the group is accumulated in split order.  A
                        # slot past n_splits (the last group only) re-reads the last split
                        # and is SELECTED out of the sums, so den / acc hold exactly what the
                        # one-split-per-step walk left in them.
                        for s0 in cutlass.range(G_S, n_splits, G_T, unroll=1):
                            lse_t = []
                            o_t = []
                            for j in cutlass.range_constexpr(G_T):
                                s_eff64 = cutlass.Int64(cute.math.min(s0 + cutlass.Int32(j), s_last))
                                lse_t.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
                                o_t.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
                            for j in cutlass.range_constexpr(G_T):
                                lse_s = lse_t[j]
                                v = o_t[j]
                                in_range = (s0 + cutlass.Int32(j)) < n_splits
                                live = (lse_s > neg_inf) & in_range
                                e = cute.math.exp(lse_s - m_safe, fastmath=True)
                                w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
                                v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
                                v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
                                v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
                                v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
                                den = cutlass.Float32(arith.select(in_range.ir_value(), (den + w).ir_value(), den.ir_value()))
                                acc0 = cutlass.Float32(arith.select(in_range.ir_value(), (acc0 + w * v0).ir_value(), acc0.ir_value()))
                                acc1 = cutlass.Float32(arith.select(in_range.ir_value(), (acc1 + w * v1).ir_value(), acc1.ir_value()))
                                acc2 = cutlass.Float32(arith.select(in_range.ir_value(), (acc2 + w * v2).ir_value(), acc2.ir_value()))
                                acc3 = cutlass.Float32(arith.select(in_range.ir_value(), (acc3 + w * v3).ir_value(), acc3.ir_value()))
                    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
                    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))
                    outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
                    if cutlass.const_expr(gate is not None):
                        # The gated prefill kernels' epilogue arithmetic, verbatim
                        # (_common_blackwell.gate_epilogue_pairs): per pair one
                        # FMUL2 (g / 2), two MUFU.TANH, one FFMA2 (h * t + h), on the
                        # fp32 merged value; then the dead-row SELECT per element,
                        # AFTER the fma -- a dead row is exactly 0 whatever G holds
                        # (a NaN gate times the zero h would be NaN; a select is not).
                        # sigmoid's 0.5 folds into the scale (``h = acc * (inv_den / 2)``:
                        # exact, a power-of-two scaling of the plain output).
                        inv_den_half = inv_den * cutlass.Float32(0.5)
                        gv = cutlass.make_array_view(gate)
                        g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                        g1 = zero
                        g2 = zero
                        g3 = zero
                        if d0 + cutlass.Int32(1) < d_v:
                            g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                        if d0 + cutlass.Int32(2) < d_v:
                            g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                        if d0 + cutlass.Int32(3) < d_v:
                            g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
                        h0 = acc0 * inv_den_half
                        h1 = acc1 * inv_den_half
                        h2 = acc2 * inv_den_half
                        h3 = acc3 * inv_den_half
                        s0, s1 = fmul2(g0, g1, gate_half, gate_half)
                        s2, s3 = fmul2(g2, g3, gate_half, gate_half)
                        t0 = cute.math.tanh(s0, approx=True)
                        t1 = cute.math.tanh(s1, approx=True)
                        t2 = cute.math.tanh(s2, approx=True)
                        t3 = cute.math.tanh(s3, approx=True)
                        y0, y1 = ffma2(t0, t1, h0, h1, h0, h1)
                        y2, y3 = ffma2(t2, t3, h2, h3, h2, h3)
                        outs = (
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
                        )
                    vals = []
                    for i in cutlass.range_constexpr(4):
                        o_val = outs[i]
                        if cutlass.const_expr(amax_o is not None):
                            # Measured before scale_o, so this is already the pre-quant
                            # amax and needs no post-hoc divide.
                            amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
                        if cutlass.const_expr(scale_o is not None):
                            o_val = o_val * q_scale
                        vals.append(o_val)
                    if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
                        if st_ok & row_live & (d0 + cutlass.Int32(3) < d_v):
                            _cvt = "bf16x2" if cutlass.const_expr(o_out.element_type == cutlass.BFloat16) else "f16x2"
                            p01, p23 = nvvm.inline_ptx(
                                f"cvt.rn.{_cvt}.f32 $0, $3, $2;\ncvt.rn.{_cvt}.f32 $1, $5, $4;",
                                write_only_types=[cutlass.Int32, cutlass.Int32],
                                read_only_args=[vals[0], vals[1], vals[2], vals[3]],
                            )
                            o_idx = cute.crd2idx((o_batch, o_tok, head, d0), o_out.layout)
                            st_global_v2(out_base + cutlass.Int64(o_idx) * 2, (p01, p23), cutlass.Int32)
                        else:
                            for i in cutlass.range_constexpr(4):
                                if row_live & (d0 + cutlass.Int32(i) < d_v):
                                    oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
                    else:
                        for i in cutlass.range_constexpr(4):
                            # Index all modes: ArrayView's row slice is a pointer and drops
                            # the final mode's stride, so a subsequent [d0] would assume
                            # contiguous D.
                            if row_live & (d0 + cutlass.Int32(i) < d_v):
                                oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        else:
            # --- pass 1: M = max_s lse_s, then den = sum_s exp(lse_s - M) ---
            # Every lane redundantly walks the (very short) split axis; the values
            # are warp-uniform and hit L1, which is cheaper than staging them.
            m = cutlass.Float32(NEG_INF)
            for s in cutlass.range(0, n_splits, 1, unroll=4):
                lse_row = lp[batch + s * n_batch, head, :]
                m = cute.math.max(m, cutlass.Float32(lse_row[q_row]))

            # All splits dead (every row fully masked): emit O := 0 / lse := -inf
            # rather than exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
            all_dead = m == cutlass.Float32(NEG_INF)
            m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))

            # A dead split (empty KV range) carries lse_s = -inf and must contribute
            # nothing.  Its weight is dropped with a SELECT, not by trusting the
            # arithmetic: 0 * x is NaN for a non-finite x and a fastmath exp(-inf)
            # is only approximately zero.  (Observed: d512 with 5 KV tiles over 8
            # splits produced NaN without the guard.)
            neg_inf = cutlass.Float32(NEG_INF)
            zero = cutlass.Float32(0.0)
            den = cutlass.Float32(0.0)
            for s in cutlass.range(0, n_splits, 1, unroll=4):
                lse_row = lp[batch + s * n_batch, head, :]
                lse_s = cutlass.Float32(lse_row[q_row])
                live = lse_s > neg_inf
                e = cute.math.exp(lse_s - m_safe, fastmath=True)
                den = den + cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))

            inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
            inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))

            # --- pass 2: O = sum_s w_s O_s / den, accumulated in fp32 ---
            # The partial slab is compact [S*B, S_q, H, d_v]; lane l owns four
            # contiguous columns of every split (one 16-byte load for fp32 partials
            # when all four are inside the row); the split loop is branch-free and
            # unrolled so the loads of several splits are in flight together.
            o_base = o_partial.iterator.toint()
            # The 16-byte loads need d_v % 4 == 0 (row starts stay 16-byte aligned)
            # and a 16-byte aligned slab base: the pointer entries only promise the
            # element alignment and take a runtime d_v, so decide at runtime (one
            # warp-uniform test); otherwise every lane takes the element path.
            vec_ok = ((d_v & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
            amax_local = cutlass.Float32(0.0)
            q_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(scale_o is not None):
                q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
            # Gate in the combine: the 0.5 handed to the packed multiply must be opaque (a constant
            # float into inline_ptx ICEs libNVVM).  Hoisted once per row, like the fused kernels do
            # per tile; traced only on the gated entry, so the plain entries' code is unchanged.
            if cutlass.const_expr(gate is not None):
                gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
            for cbase in cutlass.range(0, d_v, 128, unroll=1):
                d0 = cbase + lane * cutlass.Int32(4)
                if d0 < d_v:
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for s in cutlass.range(0, n_splits, 1, unroll=4):
                        lse_row = lp[batch + s * n_batch, head, :]
                        lse_s = cutlass.Float32(lse_row[q_row])
                        live = lse_s > neg_inf
                        e = cute.math.exp(lse_s - m_safe, fastmath=True)
                        w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
                        e0 = zero
                        e1 = zero
                        e2 = zero
                        e3 = zero
                        # Element loads through the view, each bounded by the row, serve
                        # the half partials (flavors whose split epilogue keeps the staged
                        # TMA-store O path, e.g. d512) and any d_v / base that cannot take
                        # aligned 16-byte loads (legal through the runtime-shape pointer
                        # entries); aligned fp32 partials take one 16-byte load per split.
                        if cutlass.const_expr(o_partial.element_type == cutlass.Float32):
                            if vec_ok:
                                idx = cute.crd2idx((batch + s * n_batch, q_row, head, d0), o_partial.layout)
                                v = ld_global_v4(o_base + cutlass.Int64(idx) * 4, cutlass.Float32)
                                e0 = cutlass.Float32(v[0])
                                e1 = cutlass.Float32(v[1])
                                e2 = cutlass.Float32(v[2])
                                e3 = cutlass.Float32(v[3])
                            else:
                                o_row = op[batch + s * n_batch, q_row, head, :]
                                e0 = cutlass.Float32(o_row[d0])
                                if d0 + cutlass.Int32(1) < d_v:
                                    e1 = cutlass.Float32(o_row[d0 + cutlass.Int32(1)])
                                if d0 + cutlass.Int32(2) < d_v:
                                    e2 = cutlass.Float32(o_row[d0 + cutlass.Int32(2)])
                                if d0 + cutlass.Int32(3) < d_v:
                                    e3 = cutlass.Float32(o_row[d0 + cutlass.Int32(3)])
                        else:
                            o_row = op[batch + s * n_batch, q_row, head, :]
                            e0 = cutlass.Float32(o_row[d0])
                            if d0 + cutlass.Int32(1) < d_v:
                                e1 = cutlass.Float32(o_row[d0 + cutlass.Int32(1)])
                            if d0 + cutlass.Int32(2) < d_v:
                                e2 = cutlass.Float32(o_row[d0 + cutlass.Int32(2)])
                            if d0 + cutlass.Int32(3) < d_v:
                                e3 = cutlass.Float32(o_row[d0 + cutlass.Int32(3)])
                        # a dead slot may hold anything, including non-finite values
                        v0 = cutlass.Float32(arith.select(live.ir_value(), e0.ir_value(), zero.ir_value()))
                        v1 = cutlass.Float32(arith.select(live.ir_value(), e1.ir_value(), zero.ir_value()))
                        v2 = cutlass.Float32(arith.select(live.ir_value(), e2.ir_value(), zero.ir_value()))
                        v3 = cutlass.Float32(arith.select(live.ir_value(), e3.ir_value(), zero.ir_value()))
                        acc0 = acc0 + w * v0
                        acc1 = acc1 + w * v1
                        acc2 = acc2 + w * v2
                        acc3 = acc3 + w * v3
                    outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
                    if cutlass.const_expr(gate is not None):
                        # The gated prefill kernels' epilogue arithmetic, verbatim
                        # (_common_blackwell.gate_epilogue_pairs): per pair one
                        # FMUL2 (g / 2), two MUFU.TANH, one FFMA2 (h * t + h), on the
                        # fp32 merged value; then the dead-row SELECT per element,
                        # AFTER the fma -- a dead row is exactly 0 whatever G holds
                        # (a NaN gate times the zero h would be NaN; a select is not).
                        # sigmoid's 0.5 folds into the scale (``h = acc * (inv_den / 2)``:
                        # exact, a power-of-two scaling of the plain output).
                        inv_den_half = inv_den * cutlass.Float32(0.5)
                        gv = cutlass.make_array_view(gate)
                        g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                        g1 = zero
                        g2 = zero
                        g3 = zero
                        if d0 + cutlass.Int32(1) < d_v:
                            g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                        if d0 + cutlass.Int32(2) < d_v:
                            g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                        if d0 + cutlass.Int32(3) < d_v:
                            g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
                        h0 = acc0 * inv_den_half
                        h1 = acc1 * inv_den_half
                        h2 = acc2 * inv_den_half
                        h3 = acc3 * inv_den_half
                        s0, s1 = fmul2(g0, g1, gate_half, gate_half)
                        s2, s3 = fmul2(g2, g3, gate_half, gate_half)
                        t0 = cute.math.tanh(s0, approx=True)
                        t1 = cute.math.tanh(s1, approx=True)
                        t2 = cute.math.tanh(s2, approx=True)
                        t3 = cute.math.tanh(s3, approx=True)
                        y0, y1 = ffma2(t0, t1, h0, h1, h0, h1)
                        y2, y3 = ffma2(t2, t3, h2, h3, h2, h3)
                        outs = (
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                            cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
                        )
                    for i in cutlass.range_constexpr(4):
                        o_val = outs[i]
                        if cutlass.const_expr(amax_o is not None):
                            # Measured before scale_o, so this is already the pre-quant
                            # amax and needs no post-hoc divide.
                            amax_local = cute.math.max(amax_local, cute.math.max(o_val, -o_val))
                        if cutlass.const_expr(scale_o is not None):
                            o_val = o_val * q_scale
                        # Index all modes: ArrayView's row slice is a pointer and drops
                        # the final mode's stride, so a subsequent [d0] would assume
                        # contiguous D.
                        if row_live & (d0 + cutlass.Int32(i) < d_v):
                            oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = o_val.to(o_out.element_type)

        # amax: reduce across the warp, then ONE atomic per warp.  The value is
        # non-negative, so its fp32 bit pattern orders the same as the float and
        # an integer atomicMax is exact -- the same trick the kernels' own
        # epilogues use.
        if cutlass.const_expr(amax_o is not None):
            for off in cutlass.range_constexpr(5):
                amax_local = cute.math.max(amax_local, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_local, 16 >> off, 31, kind=nvvm.Shfl.BFLY)))
            if lane == cutlass.Int32(0):
                _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))

        # --- the recombined LSE (only when the caller asked for Stats) ---
        if cutlass.const_expr(lse_out is not None):
            if (lane == cutlass.Int32(0)) & lse_live:
                lo = cutlass.make_array_view(lse_out)
                lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
                lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(NEG_INF).ir_value(), lse_val.ir_value()))
                # Base-2 Stats (stats_use_log2): the partials stay natural (the
                # merge above needs them); only the final value converts.
                if cutlass.const_expr(stats_log2):
                    lse_val = lse_val * cutlass.Float32(1.4426950408889634)
                lo[lse_batch, head, lse_tok] = lse_val


_combine_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _combine_decode_kernel(
    o_partial: cute.Tensor,  # [S*B, S_q, H, D] fp32 split-major partial O (the lean path's precondition: fp32, 16-byte rows)
    lse_partial: cute.Tensor,  # [S*B, H, S_q] fp32 split-major partial LSE (natural log)
    o_out: cute.Tensor,  # [B, S_q, H, D]
    lse_out: Optional[cute.Tensor],  # [B, H, S_q] or None (None-specialized)
    amax_o: Optional[cute.Tensor],  # see _combine_kernel; None on every entry that takes this kernel today
    scale_o: Optional[cute.Tensor],  # see _combine_kernel; None on every entry that takes this kernel today
    n_batch: cutlass.Int32,
    n_splits: cutlass.Int32,
    d_v: cutlass.Int32,
    n_chunks: cutlass.Int32,  # ceil(d_v / DECODE_CHUNK_COLS): blockIdx.x = q_row * n_chunks + chunk
    stats_log2: cutlass.Constexpr[bool],
    ragged_q: Optional[cute.Tensor],
    ragged_o: Optional[cute.Tensor],
    ragged_lse: Optional[cute.Tensor],
    ragged_q_div: cutlass.Int64,
    ragged_o_div: cutlass.Int64,
    ragged_lse_div: cutlass.Int64,
    ragged_o_cap: cutlass.Int32,
    ragged_lse_cap: cutlass.Int32,
    gate: Optional[cute.Tensor],
) -> None:
    """The decode-grid form of the lean fp32-partial combine (module constants ``DECODE_*``; the layout
    + barrier / SMEM notes are in the module docstring).  One block per (q_row, head, batch,
    64-column chunk); chain ``c = tidx // 16`` reduces splits ``r * 128 + 16 c .. + 15`` of every round
    ``r`` with its own running max (the online-softmax merge across rounds; a single round for
    ``n_splits <= 128``), lane ``g = tidx % 16`` owns columns ``chunk * 64 + 4 g .. + 3``; the eight
    chains' (max, sum, acc) meet once in SMEM behind one CTA barrier and warp 0 merges them in chain
    order and stores the row.  The same arithmetic as ``_combine_kernel`` per split (``w = SELECT(live,
    exp(lse - m_safe), 0)``, ``v = SELECT(live, v, 0)``, ``den += w``, ``acc += w * v`` in split order
    from +0): for ``n_splits <= 16`` only chain 0 holds splits and the output is bitwise the lean
    kernel's; past 16 the sum is associated differently (fp32-rounding-level differences)."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    chain = tidx // cutlass.Int32(DECODE_LANES_PER_CHAIN)
    g = tidx % cutlass.Int32(DECODE_LANES_PER_CHAIN)
    bx = cute.arch.block_idx()[0]
    q_row = bx // n_chunks
    chunk = bx - q_row * n_chunks
    head = cute.arch.block_idx()[1]
    batch = cute.arch.block_idx()[2]

    op = cutlass.make_array_view(o_partial)
    oo = cutlass.make_array_view(o_out)

    # Where the recombined row lands: dense (batch, q_row), or the ragged placement on the packed
    # outputs (batch coord 0) -- _combine_kernel's placement, verbatim.
    o_batch = batch
    o_tok = q_row
    lse_batch = batch
    lse_tok = q_row
    row_live = cutlass.Int32(1) == cutlass.Int32(1)
    lse_live = row_live
    if cutlass.const_expr(ragged_q is not None):
        rq = cutlass.make_array_view(ragged_q)
        q_base = cutlass.Int64(rq[batch]) // cutlass.Int64(ragged_q_div)
        q_len64 = cutlass.Int64(rq[batch + cutlass.Int32(1)]) // cutlass.Int64(ragged_q_div) - q_base
        in_seq = cutlass.Int64(q_row) < q_len64
        o_batch = cutlass.Int32(0)
        o_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_o)[batch]) // cutlass.Int64(ragged_o_div) + cutlass.Int64(q_row)
        row_live = in_seq & (o_tok64 >= cutlass.Int64(0)) & (o_tok64 < cutlass.Int64(ragged_o_cap))
        o_tok = cutlass.Int32(o_tok64)
        lse_live = row_live
        if cutlass.const_expr(ragged_lse is not None):
            lse_batch = cutlass.Int32(0)
            lse_tok64 = cutlass.Int64(cutlass.make_array_view(ragged_lse)[batch]) // cutlass.Int64(ragged_lse_div) + cutlass.Int64(q_row)
            lse_live = in_seq & (lse_tok64 >= cutlass.Int64(0)) & (lse_tok64 < cutlass.Int64(ragged_lse_cap))
            lse_tok = cutlass.Int32(lse_tok64)

    neg_inf = cutlass.Float32(NEG_INF)
    zero = cutlass.Float32(0.0)
    # A thread's four columns.  A thread whose columns start past the row (the last chunk of a d_v that
    # is not a multiple of 64) LOADS a valid row start instead (d_v - 4: d_v % 4 == 0 and the slab is
    # 16-byte aligned on this path) and never stores: it still runs the chain so the chain's den
    # stays uniform and it still reaches the barrier.
    d0 = chunk * cutlass.Int32(DECODE_CHUNK_COLS) + g * cutlass.Int32(4)
    col_ok = d0 < d_v
    d_ld = cutlass.Int32(arith.select(col_ok.ir_value(), d0.ir_value(), (d_v - cutlass.Int32(4)).ir_value()))
    o_base = o_partial.iterator.toint()
    lse_base = lse_partial.iterator.toint() + cutlass.Int64(cute.crd2idx((batch, head, q_row), lse_partial.layout)) * 4
    lse_step = cutlass.Int64(lse_partial.stride[0]) * cutlass.Int64(n_batch) * 4
    o_step = cutlass.Int64(o_partial.stride[0]) * cutlass.Int64(n_batch) * 4
    o_row_base = o_base + cutlass.Int64(cute.crd2idx((batch, q_row, head, d_ld), o_partial.layout)) * 4
    s_last = n_splits - cutlass.Int32(1)
    chain_base = chain * cutlass.Int32(DECODE_SPLITS_PER_CHAIN)

    # The chain's running state over its slots: the max over its IN-RANGE splits, the sum of the
    # weights and the four weighted columns -- all +0 / -inf until a live split arrives.
    m_c = neg_inf
    den_c = zero
    acc0 = zero
    acc1 = zero
    acc2 = zero
    acc3 = zero
    for r0 in cutlass.range(0, n_splits, DECODE_SPLITS_PER_ROUND, unroll=1):
        s_base = r0 + chain_base
        # Every load of the round goes out first: the 16 LSEs and the 16 partial-O rows of the chain's
        # slots (a slot past n_splits re-reads the last split; its weight AND its payload are SELECTED to
        # zero below, and it is masked out of the max so an empty chain stays -inf).
        lse_v = []
        o_v = []
        for j in cutlass.range_constexpr(DECODE_SPLITS_PER_CHAIN):
            s_eff64 = cutlass.Int64(cute.math.min(s_base + cutlass.Int32(j), s_last))
            lse_v.append(_ld_global_f32(lse_base + s_eff64 * lse_step))
            o_v.append(ld_global_v4(o_row_base + s_eff64 * o_step, cutlass.Float32))
        m_r = neg_inf
        for j in cutlass.range_constexpr(DECODE_SPLITS_PER_CHAIN):
            in_range = (s_base + cutlass.Int32(j)) < n_splits
            m_r = cute.math.max(m_r, cutlass.Float32(arith.select(in_range.ir_value(), lse_v[j].ir_value(), neg_inf.ir_value())))
        m_new = cute.math.max(m_c, m_r)
        # m_safe only feeds the exponentials: a chain with no live split so far keeps exp(lse - 0),
        # which the live SELECT below zeroes anyway.
        m_safe_c = cutlass.Float32(arith.select((m_new == neg_inf).ir_value(), zero.ir_value(), m_new.ir_value()))
        if r0 > cutlass.Int32(0):
            # A later round (n_splits > 128 only): the online-softmax rescale of what the chain holds
            # onto its new max.  Never on the first round, so the first round's arithmetic is the lean
            # kernel's own sequence (no multiply ahead of the first add).
            alpha = cutlass.Float32(arith.select((m_c > neg_inf).ir_value(), cute.math.exp(m_c - m_safe_c, fastmath=True).ir_value(), zero.ir_value()))
            den_c = den_c * alpha
            acc0 = acc0 * alpha
            acc1 = acc1 * alpha
            acc2 = acc2 * alpha
            acc3 = acc3 * alpha
        m_c = m_new
        for j in cutlass.range_constexpr(DECODE_SPLITS_PER_CHAIN):
            lse_s = lse_v[j]
            v = o_v[j]
            live = (lse_s > neg_inf) & ((s_base + cutlass.Int32(j)) < n_splits)
            e = cute.math.exp(lse_s - m_safe_c, fastmath=True)
            w = cutlass.Float32(arith.select(live.ir_value(), e.ir_value(), zero.ir_value()))
            v0 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[0]).ir_value(), zero.ir_value()))
            v1 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[1]).ir_value(), zero.ir_value()))
            v2 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[2]).ir_value(), zero.ir_value()))
            v3 = cutlass.Float32(arith.select(live.ir_value(), cutlass.Float32(v[3]).ir_value(), zero.ir_value()))
            den_c = den_c + w
            acc0 = acc0 + w * v0
            acc1 = acc1 + w * v1
            acc2 = acc2 + w * v2
            acc3 = acc3 + w * v3

    # The SMEM exchange (the module's SMEM table): red_m[c] / red_den[c] by lane 0 of each chain, the
    # four columns by every thread at its own (c, g) slot; the per-thread mapping spreads the banks (16 threads x
    # 16 B = two full 128-B cycles per chain), no descriptor reads it.
    red = cutlass.Array(cutlass.Float32, DECODE_RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    red_ptr = Pointer(red.data_ptr(), dtype=cutlass.Float32)
    if g == cutlass.Int32(0):
        (red_ptr + chain).store(m_c)
        (red_ptr + (chain + cutlass.Int32(DECODE_RED_DEN_OFF))).store(den_c)
    acc_off = cutlass.Int32(DECODE_RED_ACC_OFF) + (chain * cutlass.Int32(DECODE_LANES_PER_CHAIN) + g) * cutlass.Int32(4)
    (red_ptr + acc_off).store(acc0)
    (red_ptr + (acc_off + cutlass.Int32(1))).store(acc1)
    (red_ptr + (acc_off + cutlass.Int32(2))).store(acc2)
    (red_ptr + (acc_off + cutlass.Int32(3))).store(acc3)
    # The ONE barrier of this kernel: every one of the 128 threads arrives exactly once (nothing above
    # exits or diverges around it); the merge below reads what the chains published.
    nvvm.barrier_cta_sync()

    if warp == cutlass.Int32(0):
        # Warp 0 merges the eight chains in chain order and stores the row: lanes 0..15 own the chunk's
        # sixteen column groups; lanes 16..31 run the same arithmetic on a duplicate column group (they
        # keep the warp converged for the amax shuffles) and store nothing.
        lo = lane < cutlass.Int32(DECODE_LANES_PER_CHAIN)
        gm = lane % cutlass.Int32(DECODE_LANES_PER_CHAIN)
        d0 = chunk * cutlass.Int32(DECODE_CHUNK_COLS) + gm * cutlass.Int32(4)
        m = neg_inf
        for c in cutlass.range_constexpr(DECODE_CHAINS):
            m = cute.math.max(m, cutlass.Float32((red_ptr + cutlass.Int32(c)).load()))
        # All splits dead (every row fully masked): emit O := 0 / lse := -inf rather than
        # exp(-inf - -inf) == NaN.  m_safe only feeds the exponentials.
        all_dead = m == neg_inf
        m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), m.ir_value()))
        den = zero
        acc0 = zero
        acc1 = zero
        acc2 = zero
        acc3 = zero
        for c in cutlass.range_constexpr(DECODE_CHAINS):
            m_cc = cutlass.Float32((red_ptr + cutlass.Int32(c)).load())
            den_cc = cutlass.Float32((red_ptr + cutlass.Int32(c + DECODE_RED_DEN_OFF)).load())
            # A chain with no live split (m_cc == -inf) contributes exactly 0 by SELECT; the chain that
            # holds the max contributes with the factor exp(0) == 1 exactly.
            f = cutlass.Float32(arith.select((m_cc > neg_inf).ir_value(), cute.math.exp(m_cc - m_safe, fastmath=True).ir_value(), zero.ir_value()))
            off = cutlass.Int32(DECODE_RED_ACC_OFF + c * DECODE_LANES_PER_CHAIN * 4) + gm * cutlass.Int32(4)
            a0 = cutlass.Float32((red_ptr + off).load())
            a1 = cutlass.Float32((red_ptr + (off + cutlass.Int32(1))).load())
            a2 = cutlass.Float32((red_ptr + (off + cutlass.Int32(2))).load())
            a3 = cutlass.Float32((red_ptr + (off + cutlass.Int32(3))).load())
            den = den + f * den_cc
            acc0 = acc0 + f * a0
            acc1 = acc1 + f * a1
            acc2 = acc2 + f * a2
            acc3 = acc3 + f * a3
        inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
        inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), inv_den.ir_value()))
        outs = (acc0 * inv_den, acc1 * inv_den, acc2 * inv_den, acc3 * inv_den)
        if cutlass.const_expr(gate is not None):
            # _combine_kernel's gate-in-combine arithmetic, verbatim (the gated prefill kernels'
            # epilogue: FMUL2, MUFU.TANH, FFMA2 on the fp32 merged value, the dead-row SELECT AFTER
            # the fma); the opaque 0.5 is hoisted here, once per row, like there.
            gate_half = opaque_f32_zero() + cutlass.Float32(0.5)
            inv_den_half = inv_den * cutlass.Float32(0.5)
            gv = cutlass.make_array_view(gate)
            g0 = zero
            g1 = zero
            g2 = zero
            g3 = zero
            if lo & col_ok:
                g0 = cutlass.Float32(gv[batch, q_row, head, d0])
                if d0 + cutlass.Int32(1) < d_v:
                    g1 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(1)])
                if d0 + cutlass.Int32(2) < d_v:
                    g2 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(2)])
                if d0 + cutlass.Int32(3) < d_v:
                    g3 = cutlass.Float32(gv[batch, q_row, head, d0 + cutlass.Int32(3)])
            hg0 = acc0 * inv_den_half
            hg1 = acc1 * inv_den_half
            hg2 = acc2 * inv_den_half
            hg3 = acc3 * inv_den_half
            s0, s1 = fmul2(g0, g1, gate_half, gate_half)
            s2, s3 = fmul2(g2, g3, gate_half, gate_half)
            t0 = cute.math.tanh(s0, approx=True)
            t1 = cute.math.tanh(s1, approx=True)
            t2 = cute.math.tanh(s2, approx=True)
            t3 = cute.math.tanh(s3, approx=True)
            y0, y1 = ffma2(t0, t1, hg0, hg1, hg0, hg1)
            y2, y3 = ffma2(t2, t3, hg2, hg3, hg2, hg3)
            outs = (
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y0.ir_value())),
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y1.ir_value())),
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y2.ir_value())),
                cutlass.Float32(arith.select(all_dead.ir_value(), zero.ir_value(), y3.ir_value())),
            )
        q_scale = cutlass.Float32(1.0)
        if cutlass.const_expr(scale_o is not None):
            q_scale = cutlass.Float32(cutlass.make_array_view(scale_o)[0])
        amax_local = zero
        vals = []
        for i in cutlass.range_constexpr(4):
            o_val = outs[i]
            if cutlass.const_expr(amax_o is not None):
                # Measured before scale_o (the pre-quant amax); a duplicate or past-the-row column
                # group folds nothing.
                fold = lo & (d0 + cutlass.Int32(i) < d_v)
                amax_local = cutlass.Float32(
                    arith.select(fold.ir_value(), cute.math.max(amax_local, cute.math.max(o_val, -o_val)).ir_value(), amax_local.ir_value())
                )
            if cutlass.const_expr(scale_o is not None):
                o_val = o_val * q_scale
            vals.append(o_val)
        store_row = row_live & lo & col_ok
        if cutlass.const_expr(o_out.element_type in (cutlass.Float16, cutlass.BFloat16)):
            out_base = o_out.iterator.toint()
            _st_strides = cutlass.Int64(o_out.stride[0]) | cutlass.Int64(o_out.stride[1]) | cutlass.Int64(o_out.stride[2])
            st_ok = (
                ((out_base & cutlass.Int64(7)) == cutlass.Int64(0))
                & (cutlass.Int64(o_out.stride[3]) == cutlass.Int64(1))
                & ((_st_strides & cutlass.Int64(3)) == cutlass.Int64(0))
            )
            if st_ok & store_row & (d0 + cutlass.Int32(3) < d_v):
                _cvt = "bf16x2" if cutlass.const_expr(o_out.element_type == cutlass.BFloat16) else "f16x2"
                p01, p23 = nvvm.inline_ptx(
                    f"cvt.rn.{_cvt}.f32 $0, $3, $2;\ncvt.rn.{_cvt}.f32 $1, $5, $4;",
                    write_only_types=[cutlass.Int32, cutlass.Int32],
                    read_only_args=[vals[0], vals[1], vals[2], vals[3]],
                )
                o_idx = cute.crd2idx((o_batch, o_tok, head, d0), o_out.layout)
                st_global_v2(out_base + cutlass.Int64(o_idx) * 2, (p01, p23), cutlass.Int32)
            else:
                for i in cutlass.range_constexpr(4):
                    if store_row & (d0 + cutlass.Int32(i) < d_v):
                        oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        else:
            for i in cutlass.range_constexpr(4):
                # Index all modes: ArrayView's row slice is a pointer and drops the final mode's stride.
                if store_row & (d0 + cutlass.Int32(i) < d_v):
                    oo[o_batch, o_tok, head, d0 + cutlass.Int32(i)] = vals[i].to(o_out.element_type)
        if cutlass.const_expr(amax_o is not None):
            # ONE atomic per block over the chunk's columns (an integer max on the non-negative fp32
            # bit pattern is exact and order-independent); all 32 lanes run the shuffles.
            for off_s in cutlass.range_constexpr(5):
                amax_local = cute.math.max(amax_local, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_local, 16 >> off_s, 31, kind=nvvm.Shfl.BFLY)))
            if lane == cutlass.Int32(0):
                _amax_ptr = Pointer(amax_o.iterator.raw_ptr(), dtype=cutlass.Int32)
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_ptr, amax_local.bitcast(cutlass.Int32))
        # The recombined LSE (when the caller asked for Stats): the chunk-0 block of the row writes it.
        if cutlass.const_expr(lse_out is not None):
            if (lane == cutlass.Int32(0)) & (chunk == cutlass.Int32(0)) & lse_live:
                lo_view = cutlass.make_array_view(lse_out)
                lse_val = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
                lse_val = cutlass.Float32(arith.select(all_dead.ir_value(), neg_inf.ir_value(), lse_val.ir_value()))
                if cutlass.const_expr(stats_log2):
                    lse_val = lse_val * cutlass.Float32(1.4426950408889634)
                lo_view[lse_batch, head, lse_tok] = lse_val


_combine_decode_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _combine_packed_kernel(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    total_q: cute.Tensor,
    n_splits: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],
    sinks: Optional[cute.Tensor] = None,
) -> None:
    """Four packed rows per CTA; each warp reuses split weights across D.

    Packed partials have no per-sequence padding or batch coordinate. The
    device total guards their unwritten tail before any partial is read.
    Final O/Stats retain the caller's strides and partial Stats stay in ln.
    """
    thread = cute.arch.thread_idx()[0]
    lane = thread % 32
    row = cute.arch.block_idx()[0] * 4 + thread // 32
    head = cute.arch.block_idx()[1]
    if (row >= o_partial.shape[1]) | (row >= cutlass.make_array_view(total_q)[0]):
        # Warp-uniform, with no CTA barriers in this kernel.
        nvvm.exit()
    op = cutlass.make_array_view(o_partial)
    lp = cutlass.make_array_view(lse_partial)
    oo = cutlass.make_array_view(o_out)
    neg_inf = cutlass.Float32(NEG_INF)
    # Sink-free partials carry only real keys. Add the per-head virtual key
    # exactly once, in the same stable normalizer as the real split weights.
    sink = neg_inf
    if cutlass.const_expr(sinks is not None):
        sink = cutlass.Float32(cutlass.make_array_view(sinks)[head])
    parallel = n_splits <= 32
    m = sink
    m_safe = cutlass.Float32(0.0)
    all_dead = m == neg_inf
    den = cutlass.Float32(0.0)
    lane_weight = cutlass.Float32(0.0)
    if parallel:
        lane_lse = neg_inf
        if lane < n_splits:
            lane_lse = cutlass.Float32(lp[lane, head, row])
        m = lane_lse
        for bit in cutlass.range_constexpr(5):
            m = cute.math.max(m, cute.arch.shuffle_sync_bfly(m, 1 << bit))
        if cutlass.const_expr(sinks is not None):
            m = cute.math.max(m, sink)
        all_dead = m == neg_inf
        m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))
        if lane_lse > neg_inf:
            lane_weight = cute.math.exp(lane_lse - m_safe, fastmath=True)
        den = lane_weight
        for bit in cutlass.range_constexpr(5):
            den = den + cute.arch.shuffle_sync_bfly(den, 1 << bit)
    else:
        # Preserve arbitrary runtime split counts for internal prepared users.
        for split in cutlass.range(0, n_splits, 1, unroll=1):
            m = cute.math.max(m, cutlass.Float32(lp[split, head, row]))
        all_dead = m == neg_inf
        m_safe = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), m.ir_value()))
        for split in cutlass.range(0, n_splits, 1, unroll=1):
            partial_lse = cutlass.Float32(lp[split, head, row])
            if partial_lse > neg_inf:
                den = den + cute.math.exp(partial_lse - m_safe, fastmath=True)
    if cutlass.const_expr(sinks is not None):
        if sink > neg_inf:
            den = den + cute.math.exp(sink - m_safe, fastmath=True)
    inv_den = cutlass.Float32(1.0) / cute.math.max(den, cutlass.Float32(1e-30))
    inv_den = cutlass.Float32(arith.select(all_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_den.ir_value()))
    for d_base in cutlass.range(0, o_partial.shape[3], 128, unroll=1):
        acc = cute.make_rmem_tensor((4,), cutlass.Float32)
        acc.fill(0.0)
        for split in cutlass.range(0, n_splits, 1, unroll=1):
            weight = cutlass.Float32(0.0)
            if parallel:
                weight = cute.arch.shuffle_sync(lane_weight, split)
            else:
                partial_lse = cutlass.Float32(lp[split, head, row])
                if partial_lse > neg_inf:
                    weight = cute.math.exp(partial_lse - m_safe, fastmath=True)
            # Dead splits may have non-finite payloads: skip their loads;
            # multiplication by zero would not remove a NaN contribution.
            if weight > cutlass.Float32(0.0):
                for slot in cutlass.range_constexpr(4):
                    d = d_base + lane + slot * 32
                    if d < o_partial.shape[3]:
                        acc[slot] = acc[slot] + weight * cutlass.Float32(op[split, row, head, d])
        for slot in cutlass.range_constexpr(4):
            d = d_base + lane + slot * 32
            if d < o_partial.shape[3]:
                oo[0, row, head, d] = (acc[slot] * inv_den).to(o_out.element_type)
    if cutlass.const_expr(lse_out is not None):
        if lane == 0:
            value = m_safe + cute.math.log(cute.math.max(den, cutlass.Float32(1e-30)), fastmath=True)
            value = cutlass.Float32(arith.select(all_dead.ir_value(), neg_inf.ir_value(), value.ir_value()))
            if cutlass.const_expr(stats_log2):
                value = value * cutlass.Float32(1.4426950408889634)
            cutlass.make_array_view(lse_out)[0, head, row] = value


_combine_packed_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_combine(
    o_partial: cute.Tensor,
    lse_partial: cute.Tensor,
    o_out: cute.Tensor,
    lse_out: Optional[cute.Tensor],
    amax_o: Optional[cute.Tensor],
    scale_o: Optional[cute.Tensor],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    stats_log2: cutlass.Constexpr[bool],
    ragged_q: Optional[cute.Tensor],
    ragged_o: Optional[cute.Tensor],
    ragged_lse: Optional[cute.Tensor],
    ragged_divs: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    gate: Optional[cute.Tensor],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """One launch for every ABI: dense placement (ragged tensors None) or the
    ragged-Q leg's placement at the offsets, bounded by the (O, Stats) packed
    token capacities; ``gate`` (dense placement only) is the gated entry's G.

    ``decode_grid`` picks the kernel: ``DECODE_GRID_AUTO`` launches the decode-grid
    kernel when the lean kernel's grid holds at most
    ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks and the lean path's
    geometry holds (fp32 partials, ``d_v % 4 == 0``, a 16-byte aligned slab; a
    runtime test, like the lean path's own), else the lean kernel; ``DECODE_GRID_ON``
    drops the grid bound; ``DECODE_GRID_OFF`` always launches the lean kernel.
    Quantized entries (an amax or an output scale) stay on the lean kernel."""
    B, H, SQ, D = problem_size
    # The lean kernel's own geometry test: lane l reads columns 4l..4l+3 of the compact fp32 partial
    # rows as one 16-byte load when d_v % 4 == 0 and the slab is 16-byte aligned; any other d_v / base
    # (legal through this runtime-shape entry) takes bounded element loads and stores instead.
    use_decode = cutlass.Int32(0) == cutlass.Int32(1)
    if cutlass.const_expr(decode_grid != DECODE_GRID_OFF and o_partial.element_type == cutlass.Float32 and amax_o is None and scale_o is None):
        d32 = cutlass.Int32(D)
        o_base = o_partial.iterator.toint()
        lean_ok = ((d32 & cutlass.Int32(3)) == cutlass.Int32(0)) & ((o_base & cutlass.Int64(15)) == cutlass.Int64(0))
        if cutlass.const_expr(decode_grid == DECODE_GRID_ON):
            use_decode = lean_ok
        else:
            lean_blocks = ((cutlass.Int32(SQ) + cutlass.Int32(ROWS_PER_BLOCK - 1)) // cutlass.Int32(ROWS_PER_BLOCK)) * cutlass.Int32(H) * cutlass.Int32(B)
            use_decode = lean_ok & (lean_blocks <= cutlass.Int32(_DECODE_COMBINE_MAX_LEAN_BLOCKS))
    if use_decode:
        n_chunks = (cutlass.Int32(D) + cutlass.Int32(DECODE_CHUNK_COLS - 1)) // cutlass.Int32(DECODE_CHUNK_COLS)
        _combine_decode_kernel(
            o_partial,
            lse_partial,
            o_out,
            lse_out,
            amax_o,
            scale_o,
            cutlass.Int32(B),
            n_splits,
            cutlass.Int32(D),
            n_chunks,
            stats_log2,
            ragged_q,
            ragged_o,
            ragged_lse,
            cutlass.Int64(ragged_divs[0]),
            cutlass.Int64(ragged_divs[1]),
            cutlass.Int64(ragged_divs[2]),
            cutlass.Int32(ragged_caps[0]),
            cutlass.Int32(ragged_caps[1]),
            gate,
        ).launch(
            grid=(cutlass.Int32(SQ) * n_chunks, H, B),
            block=[DECODE_THREADS, 1, 1],
            stream=stream,
        )
    else:
        _combine_kernel(
            o_partial,
            lse_partial,
            o_out,
            lse_out,
            amax_o,
            scale_o,
            cutlass.Int32(B),
            n_splits,
            cutlass.Int32(D),
            cutlass.Int32(SQ),
            stats_log2,
            ragged_q,
            ragged_o,
            ragged_lse,
            cutlass.Int64(ragged_divs[0]),
            cutlass.Int64(ragged_divs[1]),
            cutlass.Int64(ragged_divs[2]),
            cutlass.Int32(ragged_caps[0]),
            cutlass.Int32(ragged_caps[1]),
            gate,
        ).launch(
            grid=((SQ + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, H, B),
            block=[THREADS, 1, 1],
            stream=stream,
        )


@cute.jit
def _host_ptr(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind prepared f16/bf16 pointers to the same reduction as tensor callers.

    Partial workspaces are compact; final O and Stats use the actual caller
    strides. Widen before computing compact strides, including split batches.
    This is the dense placement's positional ABI (unchanged); the ragged-Q
    decode leg compiles :func:`_host_ptr_ragged` instead.
    """
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    _launch_combine(
        o_partial, lse_partial, o_out, lse_out, None, None, problem_size, n_splits, stats_log2, None, None, None, (1, 1, 1), (0, 0), None, decode_grid, stream
    )


@cute.jit
def _host_ptr_gated(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    gate_ptr: cute.Pointer,
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """:func:`_host_ptr` plus the gate: ``gate_ptr`` / ``gate_strides`` describe G
    as a ``[B, S_q, H, D_v]`` view in BSHD stride order (like ``o_strides``), read
    element-wise -- the dense placement's gate-in-combine ABI (a split launch of
    a kernel without epilogue-gate seams, the d256 decode tile).  The dense
    entry's positional ABI stays exactly what its callers pass."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    B, H, SQ, D = problem_size
    gate = cute.make_tensor(gate_ptr, cute.make_layout((B, SQ, H, D), stride=gate_strides))
    _launch_combine(
        o_partial, lse_partial, o_out, lse_out, None, None, problem_size, n_splits, stats_log2, None, None, None, (1, 1, 1), (0, 0), gate, decode_grid, stream
    )


@cute.jit
def _host_ptr_packed(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    total_q_ptr: cute.Pointer,
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
    sinks_ptr: Optional[cute.Pointer] = None,
) -> None:
    """Combine compact [split, packed token, head, D] partials (B must be 1)."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    total_q = cute.make_tensor(total_q_ptr, cute.make_layout((1,), stride=(1,)))
    sinks = cute.make_tensor(sinks_ptr, cute.make_layout((problem_size[1],), stride=(1,))) if cutlass.const_expr(sinks_ptr is not None) else None
    _combine_packed_kernel(o_partial, lse_partial, o_out, lse_out, total_q, n_splits, stats_log2, sinks).launch(
        grid=((problem_size[2] + 3) // 4, problem_size[1], 1), block=[THREADS, 1, 1], stream=stream
    )


@cute.jit
def _host_ptr_quantized(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    amax_o_ptr: Optional[cute.Pointer],
    scale_o_ptr: Optional[cute.Pointer],
    has_scale_o: cutlass.Constexpr[bool],
    stats_log2: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Reduce FP8-input partials, then scale/cast and report the recombined Amax.

    Quantized outputs keep unscaled partials and apply scale_o here. Half outputs
    retain their existing scaled partials; normalize their requested Amax after
    reduction. The preceding split-attention kernel initializes Amax, ordered
    before this reduction on the same stream. MXFP8 has no scalar output
    scale: its None-specialized pointer also removes Amax unscaling.
    """
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    amax_o = None
    if cutlass.const_expr(amax_o_ptr is not None):
        amax_o = cute.make_tensor(amax_o_ptr, cute.make_layout((1,), stride=(1,)))
    scale_o = None
    if cutlass.const_expr(has_scale_o):
        scale_o = cute.make_tensor(scale_o_ptr, cute.make_layout((1,), stride=(1,)))
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        amax_o,
        scale_o,
        problem_size,
        n_splits,
        stats_log2,
        None,
        None,
        None,
        (1, 1, 1),
        (0, 0),
        None,
        DECODE_GRID_OFF,
        stream,
    )
    if cutlass.const_expr(amax_o_ptr is not None and not has_scale_o and scale_o_ptr is not None):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@cute.jit
def _host_ptr_ragged(
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    o_out_ptr: cute.Pointer,
    lse_out_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int],
    n_splits: cutlass.Int32,
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_q_ptr: cute.Pointer,
    ragged_o_ptr: cute.Pointer,
    ragged_lse_ptr: Optional[cute.Pointer],
    ragged_divs: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    ragged_caps: Tuple[cutlass.Int32, cutlass.Int32],
    stats_log2: cutlass.Constexpr[bool],
    decode_grid: cutlass.Constexpr[int],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """The ragged-Q decode leg's pointer ABI: :func:`_host_ptr` plus the (B+1,)
    ragged offsets of Q / O / Stats (int32 or int64 elements), their
    elements-per-token divisors and the (O, Stats) packed token capacities; the
    final O / Stats are the packed outputs addressed at ``offset[b] / div +
    q_row`` with batch coord 0 (``o_strides[0]`` / ``lse_strides[0]`` are never
    stepped), bounded by the capacities."""
    o_partial, lse_partial, o_out, lse_out = _ptr_operands(
        o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides
    )
    B = problem_size[0]
    n_off = cutlass.Int32(B) + cutlass.Int32(1)
    ragged_q = cute.make_tensor(ragged_q_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_o = cute.make_tensor(ragged_o_ptr, cute.make_layout((n_off,), stride=(1,)))
    ragged_lse = None
    if cutlass.const_expr(ragged_lse_ptr is not None):
        ragged_lse = cute.make_tensor(ragged_lse_ptr, cute.make_layout((n_off,), stride=(1,)))
    _launch_combine(
        o_partial,
        lse_partial,
        o_out,
        lse_out,
        None,
        None,
        problem_size,
        n_splits,
        stats_log2,
        ragged_q,
        ragged_o,
        ragged_lse,
        ragged_divs,
        ragged_caps,
        None,
        decode_grid,
        stream,
    )


@cute.jit
def _ptr_operands(o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides):
    """The compact split-major partial slabs and the strided final O / Stats views of a pointer entry."""
    B, H, SQ, D = problem_size
    b64, h64, sq64, d64 = cutlass.Int64(B), cutlass.Int64(H), cutlass.Int64(SQ), cutlass.Int64(D)
    split_batches = b64 * cutlass.Int64(n_splits)
    o_partial = cute.make_tensor(o_partial_ptr, cute.make_layout((split_batches, SQ, H, D), stride=(sq64 * h64 * d64, h64 * d64, d64, 1)))
    lse_partial = cute.make_tensor(lse_partial_ptr, cute.make_layout((split_batches, H, SQ), stride=(h64 * sq64, sq64, 1)))
    o_out = cute.make_tensor(o_out_ptr, cute.make_layout((B, SQ, H, D), stride=o_strides))
    lse_out = None
    if cutlass.const_expr(lse_out_ptr is not None):
        lse_out = cute.make_tensor(lse_out_ptr, cute.make_layout((B, H, SQ), stride=lse_strides))
    return o_partial, lse_partial, o_out, lse_out


@lru_cache(maxsize=None)
def compile_ptr(
    dtype_o: str = "f16",
    dtype_partial: str = "f32",
    has_lse: bool = False,
    stats_log2: bool = False,
    ragged: bool = False,
    ragged_i64: bool = False,
    quantized: bool = False,
    has_amax: bool = False,
    has_scale_o: bool = False,
    has_scale_o_input: bool = True,
    packed: bool = False,
    has_sink: bool = False,
    gate: bool = False,
    dtype_gate: Optional[str] = None,
    decode_grid: int = DECODE_GRID_AUTO,
) -> Callable:
    """Compile a shape-generic pointer entry for prepared split execution.

    The runtime arguments are partial-O/partial-LSE/O/optional-LSE pointers,
    ``(B, H, S_q, D_v)``, split count, O strides in BSHD order, LSE strides
    in BHS order, the three ragged-offset pointers (``ragged``: (B+1,) Q / O /
    Stats offsets, int32 or -- ``ragged_i64`` -- int64; None-specialized off
    otherwise) and their elements-per-token divisors. Every stride is Int64,
    including singleton dimensions. Packed partials use a row-per-warp reduction
    and append the device-total-Q pointer; other entries share the existing
    reduction and launch. The quantized dense entry appends Amax/scale
    pointers; half and ragged entries keep their existing positional ABI.
    ``has_scale_o_input=False`` removes the scalar input and Amax unscale
    launch for MXFP8. Per-tensor FP8 retains both by default.
    ``gate=True`` (dense half launches only; ``dtype_gate`` "f16" / "bf16" names
    G's dtype) is the gate-in-combine entry: the dense ABI plus G's pointer and
    its four BSHD strides appended -- see the module docstring for the
    numerics it fixes.
    ``decode_grid`` (appended; ``DECODE_GRID_AUTO`` by default) selects the
    kernel FORM of the half / gated / ragged entries, a compile-time
    specialization that leaves every positional ABI unchanged: AUTO = the
    decode-grid kernel when the lean kernel's grid holds at most
    ``heuristics.DECODE_COMBINE_MAX_LEAN_BLOCKS`` blocks (the grid rule, decided
    per launch from the runtime geometry), else the lean kernel; ``DECODE_GRID_ON``
    / ``DECODE_GRID_OFF`` force one form (tests, an explicit decode flag, a kill
    switch).  The quantized and packed entries ignore it (always their own
    kernel).
    """
    if dtype_o not in ("f16", "bf16", "e4m3", "e5m2") or dtype_partial not in ("f16", "bf16", "f32"):
        raise ValueError("prepared split combine requires half/FP8 O and f16/bf16/f32 partials")
    if (has_amax or has_scale_o or dtype_o in ("e4m3", "e5m2")) and not quantized:
        raise ValueError("FP8 outputs and scalar operands require the quantized pointer entry")
    if has_scale_o and not has_scale_o_input:
        raise ValueError("scaled combine requires a scale_o input")
    if quantized and ragged:
        raise ValueError("the quantized pointer entry serves dense split launches")
    if packed and (ragged or quantized):
        raise ValueError("packed THD partials require the half pointer entry without ragged final placement")
    if has_sink and not packed:
        raise ValueError("sink-aware combine requires packed half partials")
    if ragged_i64 and not ragged:
        raise ValueError("ragged_i64 is a ragged specialization")
    if gate and (ragged or quantized or packed):
        raise ValueError("the gate-in-combine entry serves dense half split launches (no ragged placement, no quantized O, no packed partials)")
    if gate and dtype_gate not in ("f16", "bf16"):
        raise ValueError("the gate-in-combine entry needs dtype_gate 'f16' or 'bf16' (the gate G is a half tensor of O's shape)")
    if dtype_gate is not None and not gate:
        raise ValueError("dtype_gate is a gate=True specialization")
    if decode_grid not in (DECODE_GRID_AUTO, DECODE_GRID_ON, DECODE_GRID_OFF):
        raise ValueError("decode_grid must be DECODE_GRID_AUTO, DECODE_GRID_ON or DECODE_GRID_OFF")
    _cache_key = _template_key(globals(), locals(), "compile_ptr")
    gmem = cute.AddressSpace.gmem

    def P(dtype):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=dtype.width // 8)

    common = (
        P(_ELEM[dtype_partial]),
        P(cutlass.Float32),
        P(_ELEM[dtype_o]),
        P(cutlass.Float32) if has_lse else None,
        (0, 0, 0, 0),
        cutlass.Int32(0),
        (cutlass.Int64(0),) * 4,
        (cutlass.Int64(0),) * 3,
    )
    # The kernel-form selector is a trailing Constexpr of the half / gated / ragged entries (not a
    # runtime slot: the positional frame the native binder passes is unchanged).
    form = (int(decode_grid),)
    if packed:
        entry, extra, form = _host_ptr_packed, (P(cutlass.Int32),), ()
    elif quantized:
        entry, extra, form = (
            _host_ptr_quantized,
            (P(cutlass.Float32) if has_amax else None, P(cutlass.Float32) if has_scale_o_input else None, bool(has_scale_o)),
            (),
        )
    elif ragged:
        # The ragged-Q leg's entry appends the offsets / divisors / capacities; the
        # dense entry's positional ABI stays exactly what its callers pass.
        off_t = cutlass.Int64 if ragged_i64 else cutlass.Int32
        entry, extra = _host_ptr_ragged, (P(off_t), P(off_t), P(off_t) if has_lse else None, (cutlass.Int64(1),) * 3, (cutlass.Int32(0),) * 2)
    elif gate:
        # The gate-in-combine entry appends G's pointer and BSHD strides; the
        # dense entry's positional ABI stays exactly what its callers pass.
        entry, extra = _host_ptr_gated, (P(_ELEM[dtype_gate]), (cutlass.Int64(0),) * 4)
    else:
        entry, extra = _host_ptr, ()
    return _compile_cached(
        entry,
        *common,
        *extra,
        bool(stats_log2),
        *form,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_combine_ptr",
        **({"sinks_ptr": P(cutlass.Float32)} if has_sink else {}),
    )
