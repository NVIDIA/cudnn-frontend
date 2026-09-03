# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KF-integrated tensor-core mainloop for the QSA cell of the SM100
GQA-substrate sparse-attention forward kernel (PR4).

Source: KF campaign ``kkn1aah8y53ed4pwr3x78wvbyw``, round-7 winner
``qsa_r7_s8192_explicit_reduce4_guard`` (measured ~99% HBM peak at
seqlen=32768 on B200, bandwidth-bound-optimal per that campaign's own
analysis). Byte-identical vendored copy of the winner's ``kernel.py``:
``sm100_gqa/kf_qsa/qsa_kernel.py``. This module does **not** import that
file unmodified -- it re-expresses the same ``qsa_tc_kernel`` mainloop
(smem staging, cp.async double-buffered gather, cooperative
``warp.MmaF16BF16Op`` QK/PV, quad-warp online softmax) with two real
kernel-body changes required to match the frozen contract
(``python/cudnn/sparse_attention/fwd/api.py``), not just a thin
envelope gate around the un-adapted source:

1. **Storage-native ids, no bitmask fold.** The vendored kernel's
   ``qsa_kernel.py:133,138`` does ``idx_mask = NE - 1; sIdx[e] =
   gIdx[pos, e] & idx_mask`` -- a real computation on every staged index,
   not a harness artifact (grep-verified: this line exists in the kernel
   body, not the KF campaign's ``definition.json`` reference). Two
   problems this creates for the frozen contract: (a) ``&`` only equals
   ``mod NE`` when ``NE`` is a power of two -- the campaign's own harness
   only ever exercised power-of-two ``NE`` (seqlens 8192/32768/131072, all
   ``/4`` powers of two) so this was never caught; (b) it destroys the
   ``-1`` sentinel (``-1 & (NE-1) == NE-1`` in two's complement -- an
   invalid slot silently aliases to the last valid entry). This module
   removes the fold entirely (``sIdx[e] = gIdx[pos, e]``, a raw
   storage-native pass-through) and instead resolves each entry's
   validity explicitly (see point 2).
2. **Real ``-1``-sentinel / dead-row / tail-clamp handling**, entirely
   absent from the vendored kernel (its finalize block unconditionally
   computes ``lse = row_max*ln2 + log(row_sum)`` and normalizes by
   ``1/row_sum``, both wrong/undefined once an invalid entry is folded
   in and both wrong for an all-invalid row). This module:
   * clamps the *addressing* use of each entry to a safe in-range dummy
     (``max(min(entry, NE-1), 0)``) so an out-of-range or ``-1`` entry
     never drives an OOB ``cp.async`` gather;
   * separately tracks each entry/token's *validity*
     (``entry >= 0 and entry*g + toff < kv_bound`` -- the same tail-clamp
     semantics ``_common_sm100.resolve_entry_window`` implements for the
     scalar kernel) and biases invalid columns' QK scores by a large-but-
     finite negative constant (``_MASK_VAL = -1e9``) before the online
     softmax reduction, applied once per tile in shared memory
     (``sS``) right after the cooperative QK score bounce-copy, gated on
     ``lane == 0`` of the warp that owns those columns (no cross-warp
     race: each warp's ``ENT_PER_WARP*G`` columns are exclusively its
     own, matching the existing gather-address partition);
   * detects an entirely-dead row (every one of the fixed 512 entries
     invalid -- ``topk_length == 0``, or all ``-1``) via
     ``row_max[r] <= _DEAD_THRESH`` at the epilogue (real valid QK
     scores for normalized bf16 inputs are always ``>> _DEAD_THRESH``,
     so this reuses ``row_max`` itself rather than a separate flag) and
     overrides that row's output to ``lse = -inf`` / ``out = 0`` per the
     frozen contract, instead of the vendored kernel's
     always-divide-by-``row_sum`` finalize.
   Why a finite (not literal ``-inf``) mask/init value: ``_MASK_VAL``
   (``-1e9``) is finite, so ``exp2(old_max - new_max)`` and
   ``exp2(row - new_max)`` never hit an ``inf - inf`` NaN even when a
   whole tile (or the whole row) is masked -- the online-softmax
   correction factor cleanly underflows to exactly ``0.0`` once a real
   valid score arrives (``exp2(-1e9 - 50) == 0.0`` exactly, not an
   epsilon-large residual), so any transient mass a masked-only tile
   picks up before a real entry appears is exactly canceled, not just
   approximately. ``_INIT_VAL`` (``-1e30``, the vendored kernel's
   original init constant) stays strictly below ``_MASK_VAL`` so the
   two are distinguishable at the epilogue's dead-row check.

**Index scope (G=1): shared across all 24 Q heads, not per-KV-head.**
The vendored kernel already implements this natively -- ``sIdx`` is
staged once per CTA from ``gIdx[pos, e]`` with no KV-head axis at all
(the grid's ``kv_head`` dimension only selects which half of K/V's
``HKV=2`` storage a CTA reads via ``kvoff = kvh * D``; the *index* list
itself is identical for both KV heads and all 12 Q heads sharing each).
This matches the frozen contract's ``G == 1`` index-scope path in
``api.py`` (``topk_idxs`` is ``(T_q, topk)``, no group axis) exactly --
no adapter reshaping needed on the index tensor itself, only the usual
THD/BSHD leading-dim flatten every kernel in this package already does.
One real consequence: ``api.py``'s ``in_gqa_envelope`` predicate
currently *requires* ``self.group_scope == h_kv`` (see that module's
``check_support``) -- i.e. today's generic dispatch only ever reaches
the *other* (``G == H_kv``) kernels in this package. A ``G == 1`` cell
like this one is therefore **not reachable through
``cudnn.sparse_attention.sparse_attention_forward_wrapper``'s automatic
routing yet** (``api.py`` is out of this subtask's target files); it is
reachable only by importing ``dispatch.py`` directly and opting in
explicitly (``try_kf_qsa=True``). Widening ``api.py``'s envelope
predicate to admit a ``G == 1`` GQA-substrate cell is a real, separate
follow-up this round surfaces rather than silently working around.

**Scope narrowing kept from the vendored geometry** (real preconditions
of the mainloop itself, not adaptation shortcuts): ``D_k == D_v == 256``,
``H_q == 24``, ``H_kv == 2``, ``topk == 512`` (the smem/register tiling
is sized for exactly this shape -- ``MPAD=16`` pads 12 real Q heads per
KV-head group, ``NUM_TILES = topk // TILE_E`` is a Python-level
constant), ``index_granularity == 4``, BF16, single contiguous KV range
per row (no ``cu_seqlens``-driven ragged *KV* split -- THD only supplies
a flat global id space, matching every other kernel in this package).

**Round-1 KF-integration status (filled in by this round's Verify;
mirrors ``dispatch.py``'s existing honesty-note convention)**: see
``dispatch.py``'s module docstring for the outcome of compiling,
launching (hard timeout), oracle-correctness, and determinism checks
against ``test/python/sparse_attention/sparse_attention_reference.py``.
Until all four are independently confirmed, this module is reachable
only via ``try_kf_qsa=True`` on ``dispatch.py``'s
``sparse_attention_forward_wrapper`` -- never the default path.
"""

from __future__ import annotations

import math
import operator
from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.nvgpu import warp
from cutlass.cute.typing import BFloat16, Float32, Int32

# --- fixed KF-winner geometry (see module docstring: real mainloop preconditions) ---
HQ = 24
HKV = 2
D = 256
TOPK = 512
G = 4
HPK = HQ // HKV  # 12
MPAD = 16  # MMA M tile (12 heads + 4 pad)
TILE_E = 8
TILE_T = TILE_E * G  # 32 tokens per tile
NUM_TILES = TOPK // TILE_E  # 64
NBUF = 2
SPAD = 8
KV_ELEMS = TILE_T * D
V_ELEMS = TILE_T * D
NWARPS = 4
ENT_PER_WARP = TILE_E // NWARPS  # 2 entries owned by each warp per tile
NTHREADS = NWARPS * 32  # 128
QOVEC = 16
QITER = (MPAD + NWARPS * 2 - 1) // (NWARPS * 2)
KVD = HKV * D  # 512
SCALE = D**-0.5
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)

# Dead-row / mask constants (see module docstring: why finite, not -inf).
_INIT_VAL = -1.0e30  # row_max init sentinel (never reached by a real score, and strictly < _MASK_VAL)
_MASK_VAL = -1.0e9  # QK-score bias applied to invalid (padded / -1 / tail-clamped) columns
_DEAD_THRESH = -1.0e8  # row_max <= this at epilogue => row never saw a valid entry


def convert_layout_acc_mn(acc_layout):
    l = cute.make_layout(acc_layout.shape)
    shape = ((l.shape[0][1], l.shape[1]), (l.shape[0][0], *l.shape[0][2:], l.shape[2]), *l.shape[3:])
    stride = ((l.stride[0][1], l.stride[1]), (l.stride[0][0], *l.stride[0][2:], l.stride[2]), *l.stride[3:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def make_acc_tensor_mn_view(acc):
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


@cute.jit
def convert_layout_acc_frgA(acc_layout):
    l = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        ((l.shape[0], l.shape[2][0]), l.shape[1], l.shape[2][1]),
        stride=((l.stride[0], l.stride[2][0]), l.stride[1], l.stride[2][1]),
    )


@cute.jit
def warp_reduce(val, op, width=32):
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def warp_reduce4_max(val):
    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1))
    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=2))
    return val


@cute.jit
def warp_reduce4_sum(val):
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=2)
    return val


def transpose_view(a):
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def make_swizzle_layout():
    dslice = D // NWARPS
    atom = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3),
        0,
        cute.make_ordered_layout((8, dslice), order=(1, 0)),
    )
    return cute.tile_to_shape(atom, (TILE_T, D), order=(1, 0))


# === Device kernel ===
#
# Same mainloop shape as the vendored ``qsa_tc_kernel`` (see module
# docstring) with the fold removed and validity/dead-row handling added.
# ``IS_BSHD``/``USE_EXPLICIT`` are compile-time; ``kv_bound``/``s_q`` are
# runtime scalars (``kv_bound`` drives both the tail-clamp and -- unlike
# the vendored kernel, which derived ``NE`` from *Q*'s leading dim -- the
# entry-validity range, since entries index into KV space).
@cute.kernel
def qsa_tc_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gIdx: cute.Tensor,  # (T_q, TOPK) int32 -- G==1, no per-KV-head axis
    gLen: Optional[cute.Tensor],  # (T_q,) int32, or None (-> TOPK)
    gO: cute.Tensor,
    gLse: cute.Tensor,
    kv_bound: Int32,  # T_kv (THD) or S_kv (BSHD)
    s_q: Int32,  # rows per batch (BSHD) or T_q (THD)
    USE_EXPLICIT: cutlass.Constexpr[bool],
    IS_BSHD: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    pos, kvh, batch = cute.arch.block_idx()
    warp_id = tidx // 32
    lane = tidx % 32

    t_q = pos + batch * s_q
    kv_base = Int32(0)
    if const_expr(IS_BSHD):
        kv_base = batch * kv_bound

    from cutlass.utils import SmemAllocator as _SmemAllocator

    smem = _SmemAllocator()
    sIdx = smem.allocate_tensor(Int32, cute.make_layout((TOPK,)), byte_alignment=16)
    sKf = smem.allocate_tensor(BFloat16, cute.make_layout((NBUF * KV_ELEMS,)), byte_alignment=16)
    sVf = smem.allocate_tensor(BFloat16, cute.make_layout((NBUF * V_ELEMS,)), byte_alignment=16)
    sw_layout = make_swizzle_layout()
    sSf = smem.allocate_tensor(BFloat16, cute.make_layout((MPAD * (TILE_T + SPAD),)), byte_alignment=16)
    sQ = cute.make_tensor(sKf.iterator + V_ELEMS, cute.make_layout((MPAD, D), stride=(D, 1)))
    sS = cute.make_tensor(sSf.iterator, cute.make_layout((MPAD, TILE_T), stride=(TILE_T + SPAD, 1)))
    gKf = cute.make_tensor(gK.iterator, cute.make_layout((gK.shape[0] * KVD,)))
    gVf = cute.make_tensor(gV.iterator, cute.make_layout((gV.shape[0] * KVD,)))
    kvoff = kvh * D

    n_entries = Int32(TOPK)
    if const_expr(gLen is not None):
        n_entries = Int32(gLen[t_q])

    # ---- stage index list: raw storage-native pass-through, NO bitmask fold ----
    for it in cutlass.range_constexpr((TOPK + NTHREADS - 1) // NTHREADS):
        e = tidx + it * NTHREADS
        sIdx[e] = gIdx[t_q, e]
    # stage Q (12 real heads -> rows 0..11; pad rows 12..15 = 0)
    for it in cutlass.range_constexpr(QITER):
        r = it * (NWARPS * 2) + warp_id * 2 + lane // 16
        dd = (lane % 16) * QOVEC
        if r < Int32(HPK):
            head = kvh * HPK + r
            vec = (gQ.iterator.raw_ptr() + (t_q * HQ + head) * D + dd).load(32, count=QOVEC)
            (sQ.iterator.raw_ptr() + r * D + dd).store(vec, alignment=32)
        else:
            for e in cutlass.range_constexpr(QOVEC):
                sQ[r, dd + e] = BFloat16(0.0)
    cute.arch.sync_threads()

    op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
    tiled_mma_qk = cute.make_tiled_mma(op, (1, NWARPS, 1), permutation_mnk=(16, NWARPS * 8, 16))
    tiled_mma_full = cute.make_tiled_mma(op, (1, 1, 1), permutation_mnk=(16, 16, 16))
    tiled_mma_pv = cute.make_tiled_mma(op, (1, NWARPS, 1), permutation_mnk=(16, NWARPS * 8, 16))
    thr_qk = tiled_mma_qk.get_slice(tidx)
    thr_full = tiled_mma_full.get_slice(lane)
    thr_pv = tiled_mma_pv.get_slice(tidx)

    tSrQ = thr_qk.make_fragment_A(thr_qk.partition_A(sQ))
    cute.autovec_copy(thr_qk.partition_A(sQ), tSrQ)
    # ``sQ`` is smem-aliased onto K ring-buffer slot #1 (see its definition
    # above): slot #1 is first *written* by this loop's tile-0 "prefetch
    # next" (tile 1) cp.async gather below, several dozen instructions after
    # the read above with no intervening barrier. Different warps reach that
    # point at different rates -- a fast warp's cp.async writes into slot #1
    # can be issued while a slower warp is still mid-``autovec_copy`` reading
    # Q out of the very same bytes, corrupting a few of that call's Q values
    # non-deterministically (confirmed by repeated-call bisection: the
    # mismatch reproduces with the reduction path swapped out, so it is not
    # the warp_reduce4 quad-shuffle; it disappears once this barrier is
    # added). This sync_threads() closes that race by making every thread
    # finish reading ``sQ`` into ``tSrQ`` before any thread may start
    # overwriting its backing memory.
    cute.arch.sync_threads()

    acc_O = cute.make_rmem_tensor(thr_pv.partition_shape_C((MPAD, D)), Float32)
    acc_O.fill(0.0)
    acc_O_mn = make_acc_tensor_mn_view(acc_O)
    num_rows = cutlass.const_expr(cute.size(acc_O_mn, mode=[0]))

    row_max = [Float32(_INIT_VAL) for _ in range(num_rows)]
    row_sum = [Float32(0.0) for _ in range(num_rows)]

    qs = Float32(SCALE * LOG2E)

    dd = lane * 8
    lane_low = lane & Int32(7)
    c_base = (warp_id << 11) + ((lane >> 3) << 9)

    # NOTE (both gather blocks below, kept inlined -- not factored into a
    # helper -- to match the vendored kernel's own inline-twice structure
    # rather than risk an unproven nested-closure-inside-@cute.kernel
    # tracing pattern): address math clamps each entry into [0, NE) so an
    # invalid (-1 or tail-clamped) entry never drives an OOB cp.async -- the
    # *validity* decision (used to bias the QK score, not gate the gather)
    # is applied separately in the QK stage below, mirroring
    # ``_common_sm100.resolve_entry_window``'s "safe dummy window" pattern
    # for the scalar kernel.
    ne0 = cute.math.max(kv_bound // Int32(G), Int32(1))
    for el in cutlass.range_constexpr(ENT_PER_WARP):
        eidx = warp_id * ENT_PER_WARP + el
        raw_ent = sIdx[eidx]
        safe_ent = cute.math.max(cute.math.min(raw_ent, ne0 - Int32(1)), Int32(0))
        for toff in cutlass.range_constexpr(G):
            row_in = Int32(el * G + toff)
            c = c_base + (row_in << 6) + ((lane_low ^ row_in) << 3)
            src = (kv_base + safe_ent * G + toff) * KVD + kvoff + dd
            cute.arch.cp_async_shared_global(sKf.iterator + c, gKf.iterator + src, 16, "cg")
            cute.arch.cp_async_shared_global(sVf.iterator + c, gVf.iterator + src, 16, "cg")
    cute.arch.cp_async_commit_group()

    for tile in cutlass.range(NUM_TILES):
        cur = (tile % NBUF) * V_ELEMS
        if tile + 1 < Int32(NUM_TILES):
            nxt = ((tile + 1) % NBUF) * KV_ELEMS
            ebase_n = (tile + 1) * TILE_E
            ne_n = cute.math.max(kv_bound // Int32(G), Int32(1))
            for el in cutlass.range_constexpr(ENT_PER_WARP):
                eidx = warp_id * ENT_PER_WARP + el
                raw_ent = sIdx[ebase_n + eidx]
                safe_ent = cute.math.max(cute.math.min(raw_ent, ne_n - Int32(1)), Int32(0))
                for toff in cutlass.range_constexpr(G):
                    row_in = Int32(el * G + toff)
                    c = c_base + (row_in << 6) + ((lane_low ^ row_in) << 3)
                    src = (kv_base + safe_ent * G + toff) * KVD + kvoff + dd
                    cute.arch.cp_async_shared_global(sKf.iterator + nxt + c, gKf.iterator + src, 16, "cg")
                    cute.arch.cp_async_shared_global(sVf.iterator + nxt + c, gVf.iterator + src, 16, "cg")
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        sK = cute.make_tensor(sKf.iterator + cur, sw_layout)
        sV_full = cute.make_tensor(sVf.iterator + cur, sw_layout)
        sVt = transpose_view(sV_full)

        # ---- GEMM1: S = Q @ K^T ----
        tSrK = thr_qk.make_fragment_B(thr_qk.partition_B(sK))
        cute.autovec_copy(thr_qk.partition_B(sK), tSrK)
        acc_S_c = cute.make_rmem_tensor(thr_qk.partition_shape_C((MPAD, TILE_T)), Float32)
        acc_S_c.fill(0.0)
        for kb in cutlass.range_constexpr(cute.size(tSrK.shape[2])):
            cute.gemm(tiled_mma_qk, acc_S_c, tSrQ[None, None, kb], tSrK[None, None, kb], acc_S_c)
        acc_S_c.store(acc_S_c.load() * qs)
        acc_S_c_bf = cute.make_fragment_like(acc_S_c, BFloat16)
        acc_S_c_bf.store(acc_S_c.load().to(BFloat16))
        cute.autovec_copy(acc_S_c_bf, thr_qk.partition_C(sS))
        cute.arch.sync_threads()

        # ---- validity mask (real contract addition, absent from the
        # vendored kernel): bias each invalid column's score to
        # ``_MASK_VAL`` directly in the smem score-exchange buffer. Each
        # warp owns exactly the ``ENT_PER_WARP*G`` columns it gathered
        # above (col = warp_id*ENT_PER_WARP*G + el*G + toff), so this is
        # race-free without extra synchronization beyond the sync above;
        # ``lane == 0`` alone does the (redundant across lanes) writes to
        # avoid 32x duplicated smem traffic. ----
        ne_m = cute.math.max(kv_bound // Int32(G), Int32(1))
        ebase_m = Int32(tile * TILE_E)
        if lane == Int32(0):
            for el in cutlass.range_constexpr(ENT_PER_WARP):
                eidx = warp_id * ENT_PER_WARP + el
                ent = sIdx[ebase_m + eidx]
                entry_present = (ent >= Int32(0)) & (ent < ne_m) & (Int32(ebase_m + eidx) < n_entries)
                for toff in cutlass.range_constexpr(G):
                    col = warp_id * ENT_PER_WARP * G + el * G + toff
                    tok = ent * Int32(G) + Int32(toff)
                    valid = entry_present & (tok < kv_bound)
                    if not valid:
                        for rr in cutlass.range_constexpr(MPAD):
                            sS[rr, col] = BFloat16(_MASK_VAL)
        cute.arch.sync_threads()

        acc_S_bf = cute.make_rmem_tensor(thr_full.partition_shape_C((MPAD, TILE_T)), BFloat16)
        cute.autovec_copy(thr_full.partition_C(sS), acc_S_bf)
        acc_S = cute.make_rmem_tensor(thr_full.partition_shape_C((MPAD, TILE_T)), Float32)
        acc_S.store(acc_S_bf.load().to(Float32))

        # ---- online softmax over N (TILE_T), quad (width=4) reduction ----
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        for r in cutlass.range_constexpr(num_rows):
            row = acc_S_mn[r, None].load()
            m_local = row.reduce(cute.ReductionOp.MAX, Float32(_INIT_VAL), 0)
            if cutlass.const_expr(USE_EXPLICIT):
                m_cur = warp_reduce4_max(m_local)
            else:
                m_cur = warp_reduce(m_local, cute.arch.fmax, width=4)
            m_old = row_max[r]
            m_new = cute.arch.fmax(m_old, m_cur)
            row_max[r] = m_new
            corr = cute.math.exp2(m_old - m_new, fastmath=True)
            p = cute.math.exp2(row - m_new, fastmath=True)
            psum_local = p.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            row_sum[r] = row_sum[r] * corr + psum_local
            acc_S_mn[r, None].store(p)
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * corr)

        # ---- GEMM2: O += P @ V ----
        rP = cute.make_fragment_like(acc_S, BFloat16)
        rP.store(acc_S.load().to(BFloat16))
        tOrP = cute.make_tensor(rP.iterator, convert_layout_acc_frgA(rP.layout))
        tOrVt = thr_pv.make_fragment_B(thr_pv.partition_B(sVt))
        cute.autovec_copy(thr_pv.partition_B(sVt), tOrVt)
        for kb in cutlass.range_constexpr(cute.size(tOrVt.shape[2])):
            cute.gemm(tiled_mma_pv, acc_O, tOrP[None, None, kb], tOrVt[None, None, kb], acc_O)

        # ---- WAR-hazard fix (round-1 determinism root cause): the
        # NBUF=2 double-buffer ring reuses this iteration's `cur` slot as
        # `nxt` starting at the *next* iteration's top-of-loop cp.async
        # prefetch (nxt(tile+1) == cur(tile)), which is an asynchronous
        # copy-engine write with no ordering guarantee relative to this
        # iteration's *synchronous* shared-memory reads above (tSrK's
        # QK read and, in particular, tOrVt's PV read just above, which
        # lands late in the iteration, close to the loop-back point).
        # wait_group(1)+sync_threads at the TOP of the loop only orders
        # the *previous* cp.async group's writes before *this* iteration's
        # reads -- it says nothing about ordering *this* iteration's reads
        # before the *next* iteration's writes into the same buffer, and
        # different warps can race ahead at different speeds (e.g. a warp
        # that finishes its PV read quickly can already reach the next
        # iteration's cp.async issue while a slower warp is still reading
        # the old data), a genuine cross-warp WAR hazard. Root cause
        # confirmed by isolation: LSE (which only ever depends on K/sS,
        # read early in the iteration via GEMM1/softmax, well clear of the
        # reused-buffer window) reproduces bit-exact across repeated
        # identical calls, while `out` (which alone depends on this late
        # V read) does not -- see this module's docstring and dispatch.py
        # for the full account. This one barrier closes that window
        # without touching the wait_group threshold, so the K/V prefetch
        # for the *next* buffer still overlaps with this iteration's QK
        # GEMM/softmax as before -- only the reused-slot's write is now
        # ordered after every warp's reads of it.
        cute.arch.sync_threads()

    # ---- finalize (real contract addition: dead-row override) ----
    for r in cutlass.range_constexpr(num_rows):
        if cutlass.const_expr(USE_EXPLICIT):
            row_sum[r] = warp_reduce4_sum(row_sum[r])
        else:
            row_sum[r] = warp_reduce(row_sum[r], operator.add, width=4)
    for r in cutlass.range_constexpr(num_rows):
        is_dead = row_max[r] <= Float32(_DEAD_THRESH)
        inv = Float32(0.0)
        if not is_dead:
            inv = cute.arch.rcp_approx(row_sum[r])
        acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)

    sO = cute.make_tensor(sKf.iterator, cute.make_layout((MPAD, D), stride=(D, 1)))
    cute.arch.sync_threads()
    rO = cute.make_fragment_like(acc_O, BFloat16)
    rO.store(acc_O.load().to(BFloat16))
    cute.autovec_copy(rO, thr_pv.partition_C(sO))
    cute.arch.sync_threads()
    OCHUNK = HPK * D // QOVEC
    for it in cutlass.range_constexpr((OCHUNK + NTHREADS - 1) // NTHREADS):
        chunk = tidx + it * NTHREADS
        if chunk < Int32(OCHUNK):
            r = chunk // Int32(D // QOVEC)
            dd2 = (chunk % Int32(D // QOVEC)) * QOVEC
            head = kvh * HPK + r
            vec = (sO.iterator.raw_ptr() + r * D + dd2).load(32, count=QOVEC)
            (gO.iterator.raw_ptr() + (t_q * HQ + head) * D + dd2).store(vec, alignment=32)

    if warp_id == Int32(0):
        for r in cutlass.range_constexpr(num_rows):
            realrow = (lane // 4) + 8 * r
            if (lane % 4 == Int32(0)) and (realrow < Int32(HPK)):
                is_dead = row_max[r] <= Float32(_DEAD_THRESH)
                lse_val = Float32(float("-inf"))
                if not is_dead:
                    lse_val = row_max[r] * Float32(LN2) + cute.math.log(row_sum[r])
                gLse[t_q, kvh * HPK + realrow] = lse_val


@cute.jit
def qsa_launch(
    gQ,
    gK,
    gV,
    gIdx,
    gLen,
    gO,
    gLse,
    kv_bound: Int32,
    s_q: Int32,
    rows_per_batch: Int32,
    n_batch: Int32,
    USE_EXPLICIT: cutlass.Constexpr[bool],
    IS_BSHD: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
):
    qsa_tc_kernel(gQ, gK, gV, gIdx, gLen, gO, gLse, kv_bound, s_q, USE_EXPLICIT, IS_BSHD).launch(
        grid=(rows_per_batch, HKV, n_batch),
        block=(NTHREADS, 1, 1),
        stream=stream,
        preferred_smem_carveout=100,
    )


def _gpu_arch_flag(device: torch.device) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("gqa_prefill_bf16_sm100_kf_qsa compilation requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(f"gqa_prefill_bf16_sm100_kf_qsa requires an SM100-family GPU, found SM{major}{minor}")
    return {0: "sm_100a", 3: "sm_103a", 7: "sm_100f"}.get(minor, "sm_100a")


@lru_cache(maxsize=None)
def _compile(is_bshd: bool, has_topk_length: bool, use_explicit: bool, arch: str):
    fake_q = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=1), HQ, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_k = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=1), HKV, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_v = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=1), HKV, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_idx = cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(divisibility=1), TOPK), stride_order=(1, 0), assumed_align=4)
    fake_len = cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4) if has_topk_length else None
    fake_out = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=1), HQ, D), stride_order=(2, 1, 0), assumed_align=16)
    fake_lse = cute.runtime.make_fake_compact_tensor(Float32, (cute.sym_int(divisibility=1), HQ), stride_order=(1, 0), assumed_align=4)
    return cute.compile(
        qsa_launch,
        fake_q,
        fake_k,
        fake_v,
        fake_idx,
        fake_len,
        fake_out,
        fake_lse,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        use_explicit,
        is_bshd,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch} --opt-level 2",
    )


def fast_path_eligible(*, d_k: int, d_v: int, h_q: int, h_kv: int, index_granularity: int, topk: int) -> bool:
    """Cheap, side-effect-free structural check for this module's fixed
    KF-winner geometry (see module docstring)."""
    return d_k == D and d_v == D and h_q == HQ and h_kv == HKV and int(index_granularity) == G and int(topk) == TOPK


def _flatten_leading(t, keep_trailing: int):
    if t is None:
        return None
    lead = t.shape[: t.ndim - keep_trailing]
    trail = t.shape[t.ndim - keep_trailing :]
    return t.reshape((math.prod(lead),) + trail) if len(lead) > 1 else t


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    index_granularity: int = 4,
    softmax_scale: Optional[float] = None,
    stream=None,
) -> dict:
    """KF-integrated QSA cell: GQA 24Q/2KV, D_k==D_v==256, granularity=4,
    topk=512, ``G == 1`` (index set shared across all heads -- see module
    docstring for why ``api.py``'s generic routing does not reach this cell
    yet). Matches the ``fwd/api.py`` GQA-substrate call contract shape-wise;
    raises rather than silently mis-serving anything outside this kernel's
    fixed geometry (``ValueError``/``NotImplementedError`` -- callers that
    want a safe probe should use ``dispatch.py``'s ``_try_kf_qsa_fast_path``,
    which turns those into a ``None`` fallback signal).
    """
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"gqa_prefill_bf16_sm100_kf_qsa is BF16-only, got Q/K/V dtypes {q.dtype}/{k.dtype}/{v.dtype}")
    if attn_sink is not None:
        raise NotImplementedError("gqa_prefill_bf16_sm100_kf_qsa: the vendored KF QSA mainloop has no sink term")
    if softmax_scale is not None and abs(float(softmax_scale) - SCALE) > 1e-9:
        raise NotImplementedError(f"gqa_prefill_bf16_sm100_kf_qsa hardcodes softmax_scale={SCALE} (D**-0.5); got {softmax_scale}")

    is_thd = q.ndim == 3
    if is_thd and cu_seqlens_q is None:
        raise ValueError("THD (3-D) Q requires cu_seqlens_q")
    if is_thd and cu_seqlens_q is not None and int(cu_seqlens_q.numel()) != 2:
        raise NotImplementedError("gqa_prefill_bf16_sm100_kf_qsa: packed multi-sequence THD (>1 segment) is not supported")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")

    with torch.cuda.device(device):
        arch = _gpu_arch_flag(device)

        if is_thd:
            t_q, h_q, d_k = q.shape
            t_kv, h_kv, d_k_kv = k.shape
            _, _, d_v = v.shape
            rows_per_batch, n_batch = t_q, 1
            kv_bound = t_kv
            s_q = t_q
            q_flat, k_flat, v_flat = q, k, v
            idx_flat = topk_idxs
            len_flat = topk_length
        else:
            b, s_q_, h_q, d_k = q.shape
            _, s_kv, h_kv, d_k_kv = k.shape
            _, _, _, d_v = v.shape
            rows_per_batch, n_batch = s_q_, b
            kv_bound = s_kv
            s_q = s_q_
            q_flat = _flatten_leading(q, 2)
            k_flat = _flatten_leading(k, 2)
            v_flat = _flatten_leading(v, 2)
            idx_flat = _flatten_leading(topk_idxs, 2 if topk_idxs.ndim == q.ndim else 1)
            len_flat = _flatten_leading(topk_length, 1)

        if d_k_kv != d_k:
            raise ValueError(f"K head dim ({d_k_kv}) must match Q ({d_k})")
        topk = idx_flat.shape[-1]
        if not fast_path_eligible(d_k=int(d_k), d_v=int(d_v), h_q=int(h_q), h_kv=int(h_kv), index_granularity=int(index_granularity), topk=int(topk)):
            raise NotImplementedError(
                f"gqa_prefill_bf16_sm100_kf_qsa serves exactly D_k=D_v={D}, H_q={HQ}, H_kv={HKV}, "
                f"index_granularity={G}, topk={TOPK}; got D_k={d_k} D_v={d_v} H_q={h_q} H_kv={h_kv} "
                f"index_granularity={index_granularity} topk={topk}"
            )

        # G == 1 index scope: accept either (T_q, topk) directly, or a
        # (T_q, 1, topk) explicit-unit-group-axis form; a genuine per-KV-head
        # (T_q, H_kv, topk) tensor is out of this cell's scope (that is the
        # sibling G==H_kv kernels' contract, not this one's -- see module
        # docstring's G==1 discussion).
        if idx_flat.ndim == 3:
            if idx_flat.shape[1] != 1:
                raise NotImplementedError(
                    f"gqa_prefill_bf16_sm100_kf_qsa is a G==1 (shared-across-all-heads) kernel; " f"got a group axis of size {idx_flat.shape[1]} (!= 1)"
                )
            idx_flat = idx_flat.reshape(idx_flat.shape[0], idx_flat.shape[2])
        elif idx_flat.ndim != 2:
            raise ValueError(f"topk_idxs must be (T_q, topk) or (T_q, 1, topk) for this G==1 kernel, got shape {topk_idxs.shape}")
        if len_flat is not None and len_flat.ndim == 2:
            if len_flat.shape[1] != 1:
                raise NotImplementedError("gqa_prefill_bf16_sm100_kf_qsa is G==1; topk_length must have no (or unit) group axis")
            len_flat = len_flat.reshape(len_flat.shape[0])

        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()
        idx_flat = idx_flat.contiguous().to(torch.int32)
        if len_flat is not None:
            len_flat = len_flat.contiguous().to(torch.int32)

        total_q = rows_per_batch * n_batch
        out = torch.empty((total_q, h_q, d_v), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, h_q), dtype=torch.float32, device=device)

        use_explicit = int(kv_bound) == 8192
        compiled = _compile(not is_thd, len_flat is not None, use_explicit, arch)

        cu_stream = stream if stream is not None else _cuda_current_stream(device)
        compiled(
            q_flat,
            k_flat,
            v_flat,
            idx_flat,
            len_flat,
            out,
            lse,
            Int32(int(kv_bound)),
            Int32(int(s_q)),
            Int32(int(rows_per_batch)),
            Int32(int(n_batch)),
            cu_stream,
        )

    if is_thd:
        return {"out": out, "lse": lse}
    return {"out": out.reshape(b, s_q_, h_q, d_v), "lse": lse.reshape(b, s_q_, h_q)}


def _cuda_current_stream(device: torch.device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
