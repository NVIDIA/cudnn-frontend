# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tcgen05 (Blackwell warp-specialized) variant of indexer_aux_grad_precompute.

Supports rectangular Q/K, an explicit loss denominator, and a runtime FP32
loss coefficient. Sparse-GEMM backward uses the v4 deep-pipeline spmm (1 CTA/SM,
160KB smem [Q|K0|K1|S_hi|S_lo], 512 threads in 4 warp roles, TMEM
accumulators, hi||lo parallel scatter, K double-buffer, direct vec4-atomic
TMEM flush). Selected by backend._kernel_module for auxiliary-gradient precomputation.
"""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warp
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.utils import SmemAllocator

BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32


D = 128
# Default for direct kernel entry points; run() derives topk from the input
# tensors and passes it as a constexpr through the complete kernel family.
TOPK = int(os.environ.get("SSA_TOPK", "512"))
# Scatter probe window, in stream positions per row per pass; must be a
# multiple of the 4 lanes that share a row.  Swept 4/8/16/32 at
# 8Kx8K: 16 is the minimum at both TOPK=512 (0.2229/0.2201/0.2175/0.2214
# ms) and TOPK=2048 (0.4039/0.3769/0.3585/0.3671 ms) -- deeper windows
# issue more independent loads per round trip, past 16 the wasted
# overshoot reads win.
SCAT_WIN = int(os.environ.get("SSA_SPMM_SCAT_WIN", "16"))
SCAT_SUB = SCAT_WIN // 4
# The probe window base is snapped down to a SCAT_ALIGN multiple so each
# lane's SCAT_SUB-element slice is naturally aligned.  SCAT_SUB is the
# minimum legal value and wastes at most SCAT_SUB-1 already-consumed
# positions in the first window of a tile; larger values waste more (a
# fully-stale window re-runs the loop) for no addressing benefit.
SCAT_ALIGN = int(os.environ.get("SSA_SPMM_SCAT_ALIGN", str(SCAT_SUB)))
SCAT_MASK = ~(SCAT_ALIGN - 1)
SCAT_AL_I32 = min(16, SCAT_SUB * 4)
SCAT_AL_BF16 = min(16, SCAT_SUB * 2)
# dq_acc / dk_acc scratch layout used by the TILE path.  TMEM hands one
# accumulator ROW to each lane, so with a plain [N, D] accumulator the 32
# lanes of one vec4 RED land on 32 rows that are D*4 = 512 B apart: ncu
# measures exactly 32.00 sectors per RED request, with only 16 of every
# 32 B sector written, and RED is the kernel's largest single L2 consumer
# (lts__t_sectors_srcunit_tex_op_red 19.5% of peak elapsed at TOPK=512,
# against 13.2% for all L2 traffic).  Grouping D into ACC_VEC-wide planes --
# element (n, d) lives at (d // ACC_VEC) * N * ACC_VEC + n * ACC_VEC
# + d % ACC_VEC -- puts a lane's vec4 at n * ACC_VEC inside its plane, so a
# warp writes 32 * 16 = 512 B contiguous: 16 FULL sectors instead of 32 half
# sectors, at an unchanged instruction count.  The gather path keeps the
# plain [N, D] layout (a warp there shares one row and is already
# coalesced), so _cast_kernel takes the layout as a constexpr.
ACC_VEC = 4
BM = 128
BN = 128
NTHR = 512
# Column-split count, i.e. the CTA granularity.  It sets both the wave
# quantisation of the launch (SPLIT * nq CTAs over the SM count) and how
# many times the per-CTA prologue (Q staging, cursor binary search) and
# the dQ flush are paid.  Swept 3..8 against the transposed dispatch grid
# at 8192x8192: 5 is the best uniform choice, 4 wins only at K=512 and gives
# up nearly all of the K=2048 gain, and 8 or more pays for more prologues than
# the extra waves save.  5 with the OLD grid shape is a regression at every K
# (+2.3 / +4.4 / +3.9 percent), so the two changes are not separable.
SPLIT = 5
# The rectangular path scans the whole Tk domain, so it gives EVERY CTA the
# same ntile.  Neither scheduling change below applies to it: with equal-cost
# CTAs there is no imbalance for a list-scheduling order to remove, and the
# makespan is ceil(waves) * per-CTA cost, so a coarser split only rounds up
# harder.  Measured at Tq=8192, Tk=16384, carrying the causal settings into
# it costs +9.9 / +10.0 percent at K=512 / 2048.  The rectangular path
# therefore keeps the grid shape it already had.
#
# Its split is still worth tuning, for the other reason a launch loses time:
# wave quantisation.  ncu at Tq=8192, Tk=32768, K=2048 measures
# sm__cycles_active .max / .avg = 1.182 on the rectangular path, and
# 3 / 2.56 = 1.172 -- with equal-cost CTAs that ratio IS the partial last
# wave, not work imbalance.  Makespan goes as
# ceil(nq * S / nSM) * ceil(ntile / S), and at nq = Tq/BM = 64 on 150 SMs
# S = 6 gives 384 CTAs = 2.56 waves (a 44 percent empty third wave) while
# S = 7 gives 448 = 2.99 waves.  Swept 5/6/7/8/9/14 at Tq=8192 over
# Tk = 16K/32K/64K x K = 512/1024/2048, all candidates interleaved in one
# process: 7 is fastest at all nine cells, -6.07 to -10.15 percent against 6,
# and the measured ranking (7 < 9 < 14 < 6 < 8 < 5 at Tk=32K, K=2048)
# is the one the model predicts once the per-CTA prologue is charged.
#
# This is tuned for nq = 64 on a 150-SM part.  The model's argmin moves with
# nq and the SM count (it is 6, not 7, whenever nq * 6 is already a whole
# multiple of nSM), so a materially different Tq or GPU wants a re-sweep; the
# penalty for being one step off is ~10 percent, not the ~3x that the
# tile/gather choice above costs.
SPLIT_RECT = 7
# Number of q-tiles one rectangular CTA services against each K tile.  The dK
# accumulator is reset and flushed once per (CTA, k-tile), so the FP32 dK
# atomic traffic is (nq / QPAIR) * Tk * D * 4 bytes -- 2.15 GB per launch at
# Tq=8192, Tk=64K, QPAIR=1.  Folding two q-tiles into one CTA lets both of
# their S^T @ Q products accumulate into the SAME TMEM plane, so the plane is
# written out once instead of twice, and the traffic halves.  It costs one
# extra 32 KB Q buffer in smem (pool 5 -> 6 tiles = 196,608 B) and one extra
# BM x D FP32 dQ accumulator in TMEM (256 -> 512 columns, the whole array,
# which is free at the 1 CTA/SM this kernel already runs at).
QPAIR = 2
# QPAIR halves the q dimension of the rectangular grid, so the column split
# doubles to keep the same CTA count and therefore the same wave quantisation
# that SPLIT_RECT was tuned for (nq/2 * 14 = 448 CTAs at Tq=8192, identical to
# 64 * 7).  dQ is flushed once per CTA per q-tile, so doubling the split
# doubles a 29 MB stream -- negligible against the 1.07 GB it saves on dK.
SPLIT_RECT_QP = SPLIT_RECT * QPAIR
# Tile-vs-gather crossover for the rectangular path, expressed as a Tk/TOPK
# ratio.  The tile kernel scans the whole Tk domain, so its work is
# proportional to Tq * Tk; the gather kernel walks the selected set, so its
# work is proportional to Tq * TOPK.  The crossover therefore sits at a fixed
# RATIO, not at a fixed Tk.  Re-measured at Tq=8192 after the gather
# pipeline optimizations (full aux pipeline, six interleaved rounds of 100
# CUDA-event samples): gather is 27-34 percent slower at Tk/TOPK=32 but
# 2-7 percent faster at 48 and 23-28 percent faster at 64.  Use 48, the
# conservative measured boundary on the gather side of the crossover.
GATHER_TK_PER_TOPK = 48
# Precision invariant: dQ/dK reductions always target FP32 accumulation
# buffers.  BF16 is an input/output storage type only, never an accumulator.
# Keep per-warp staging bounded at 512 edges for arbitrary topk values.
# Split only topk=1024 across CTAs: its former unsplit 66,560-byte staging
# buffer capped residency, while eight 128-edge pieces use 9,216 bytes each.
# With FP32 dK accumulation this is 7.99% faster on the full auxiliary pipeline
# at (Tq, Tk) = (8192, 65536).  topk=512/2048 already use <=512-edge stages,
# where splitting measured below the 5% gate and therefore compiles out.
GATHER_STAGE_MAX = 512
GATHER_SPLIT_TOPK = 1024
# After key-major dK removes the dominant atomic sector stream, K-row reads
# limit the dQ/count stage.  A maximum-size CTA groups 32 unsplit query rows:
# at K=512/Tk=32K this raises measured K-row L1 hits from 3.31% to 16.39%,
# cuts L1-to-L2 load sectors 13.5%, and moves SM SOL from 39.59% to 54.83%.
# K=1024 already splits each row eight ways; keep its measured 8-row launch.
GATHER_ROW_COHORT = 32
PAD = 136
EPS = 1.0e-10
SCORE_SCALE = 0.08838834764831845

_compiled = None

# Dev-only IKET phase timing (no-op unless SPMM_IKET=1; compile with
# CUTE_DSL_COMPILER_OPT=iket and run under `run-iket ... profile`).
_SPMM_IKET = os.environ.get("SPMM_IKET", "0") == "1"
if _SPMM_IKET:
    import cutlass.cute.experimental.iket as _iket


def _ik_b(name):
    if cutlass.const_expr(_SPMM_IKET):
        return _iket.range_start(name)
    return None


def _ik_e(tok):
    if cutlass.const_expr(_SPMM_IKET):
        _iket.range_end(tok)


@cute.kernel
def _zero_acc_kernel(acc: cute.Tensor, n: cutlass.Int32):
    """Zero one FP32 accumulation buffer independently of Q/K row counts."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    linear = bidx * 256 + tidx
    if linear < n:
        (acc.iterator + linear).store(cutlass.Float32(0.0))


# ---------------------------------------------------------------- kernel A
@cute.kernel
def _dz_kernel(
    valid_lengths: cute.Tensor,
    predict: cute.Tensor,
    teacher: cute.Tensor,
    loss_denominator: cute.Tensor,
    loss_coeff: cutlass.Float32,
    dzw: cute.Tensor,
    klpart: cute.Tensor,
    topk: cutlass.Constexpr = TOPK,
):
    tidx, _, _ = cute.arch.thread_idx()
    query, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warpi = cute.arch.warp_idx()

    allocator = SmemAllocator()
    s_red = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), byte_alignment=16)
    if tidx < 32:
        s_red[tidx] = cutlass.Float32(0.0)
    cute.arch.barrier()

    tq = klpart.shape[0]
    valid = (valid_lengths.iterator + query).load()
    coef = loss_coeff / (loss_denominator.iterator + 0).load()
    base = query * topk

    _R = cutlass.const_expr((int(topk) + 127) // 128)
    p4 = cute.make_rmem_tensor(cute.make_layout((_R,)), cutlass.Float32)
    a4 = cute.make_rmem_tensor(cute.make_layout((_R,)), cutlass.Float32)
    p4.fill(0.0)
    a4.fill(0.0)
    pu = cutlass.Float32(0.0)
    pk = cutlass.Float32(0.0)
    for r in cutlass.range_constexpr(_R):
        slot = tidx + r * 128
        if slot < valid:
            p = (predict.iterator + base + slot).load()
            tg = (teacher.iterator + base + slot).load()
            al = tg * p / (p + EPS)
            p4[r] = p
            a4[r] = al
            pu = pu + al
            pk = pk + tg * (cute.log(tg + EPS) - cute.log(p + EPS))

    wu = cute.arch.warp_reduction_sum(pu)
    wk = cute.arch.warp_reduction_sum(pk)
    if lane == 0:
        s_red[warpi] = wu
        s_red[8 + warpi] = wk
    cute.arch.barrier()

    bu = s_red[lane % 8] if lane < 4 else cutlass.Float32(0.0)
    bk = s_red[8 + (lane % 8)] if lane < 4 else cutlass.Float32(0.0)
    bu = cute.arch.warp_reduction_sum(bu)
    bk = cute.arch.warp_reduction_sum(bk)

    if tidx == 0:
        (klpart.iterator + query).store(bk * coef)

    scale = coef * SCORE_SCALE
    for r in cutlass.range_constexpr(_R):
        slot = tidx + r * 128
        if cutlass.const_expr(int(topk) % 128 != 0):
            if slot < topk:
                val = cutlass.Float32(0.0)
                if slot < valid:
                    val = (p4[r] * bu - a4[r]) * scale
                val_hi = val.to(cutlass.BFloat16)
                (dzw.iterator + base + slot).store(val_hi)
                (dzw.iterator + tq * topk + base + slot).store((val - val_hi.to(cutlass.Float32)).to(cutlass.BFloat16))
        else:
            val = cutlass.Float32(0.0)
            if slot < valid:
                val = (p4[r] * bu - a4[r]) * scale
            val_hi = val.to(cutlass.BFloat16)
            (dzw.iterator + base + slot).store(val_hi)
            (dzw.iterator + tq * topk + base + slot).store((val - val_hi.to(cutlass.Float32)).to(cutlass.BFloat16))


# ------------------------------------------------------------ kernel B-alt
@cute.kernel
def _spmm_gather_kernel(  # noqa: C901
    q_idx: cute.Tensor,
    k_idx: cute.Tensor,
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    dzw: cute.Tensor,
    dq_acc: cute.Tensor,
    dk_acc: cute.Tensor,
    topk: cutlass.Constexpr = TOPK,
):
    """Gather-form sparse backward for LOW-DENSITY tiers (Tk >> TOPK).

    The tile-scatter spmm's work is proportional to Tq x Tk (dense tiles)
    while the nonzeros are fixed at Tq x TOPK: at Tk = 256K each 128x128
    tile holds ~32 nonzeros (0.2% density) and 99.8% of the MMA work
    multiplies zeros. This kernel computes the contraction by definition
    instead, with work proportional to Tq x TOPK x D and independent of Tk:

        dQ[r]        = sum_s dz[r, s] * K[ids[r, s]]     (register acc)
        dK[ids[r,s]] += dz[r, s] * Q[r]                  (vec4 gpu atomics)

    One warp per query row (8 rows per 256-thread CTA); ids/dz are staged to
    smem once per bounded chunk; lane j owns columns [4j, 4j+4). dz is
    reconstructed as fp32 hi+lo in ONE pass -- the hi/lo split exists only to
    feed bf16 tensor cores, which this path does not use, so numerics are exact
    fp32 per term. Both dQ and dK accumulate into FP32 buffers for every shape;
    output conversion happens only after all atomic accumulation completes.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warpi = cute.arch.warp_idx()
    tq = valid_lengths.shape[0]

    smem = utils.SmemAllocator()
    # Keep shared-memory use bounded for arbitrary top-k.  The production
    # topk=1024 specialization additionally divides the edge range over eight
    # CTAs; that split was beneficial only at exactly 1024 in the measured
    # sweep, so every other top-k retains a single row owner.
    gather_split = cutlass.const_expr(8 if int(topk) == GATHER_SPLIT_TOPK else 1)
    gather_chunk = cutlass.const_expr((int(topk) + gather_split - 1) // gather_split)
    gather_stage = cutlass.const_expr(min(GATHER_STAGE_MAX, gather_chunk))
    s_ids = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8, gather_stage), stride=(gather_stage, 1)), byte_alignment=16)
    s_dz = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8, gather_stage), stride=(gather_stage, 1)), byte_alignment=16)

    row = bidx * 8 + warpi
    if row < tq:
        vlen = (valid_lengths.iterator + row).load()
        rbase = row * topk

        # ---- Q row: lane j holds columns [4j, 4j+4) as fp32 ----
        # (loaded once; the chunk loop below re-fills the staging ring)
        qr = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        qptr = cute.make_tensor(
            cute.make_ptr(
                cutlass.BFloat16, q_idx.iterator.toint() + cutlass.Int64(row) * (D * 2) + cutlass.Int64(lane) * 8, cute.AddressSpace.gmem, assumed_align=8
            ),
            cute.make_layout((4,)),
        )
        for j in cutlass.range_constexpr(4):
            qr[j] = qptr[j].to(cutlass.Float32)

        acc = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        acc.fill(0.0)
        dkv = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)

        # ---- walk the selected set (ascending ids -> forward locality) ----
        eb = cutlass.Int32(0)
        elim = vlen
        if cutlass.const_expr(gather_split > 1):
            _, sidx, _ = cute.arch.block_idx()
            eb = sidx * gather_chunk
            elim = eb + gather_chunk
            if elim > vlen:
                elim = vlen
        while eb < elim:
            ne = elim - eb
            if ne > gather_stage:
                ne = gather_stage

            # Cooperatively stage one bounded chunk.  Every element is still
            # consumed in its original order, so this changes no arithmetic.
            i = lane
            while i < ne:
                src = eb + i
                s_ids[warpi, i] = (selected_indices.iterator + rbase + src).load()
                hi = (dzw.iterator + rbase + src).load().to(cutlass.Float32)
                lo = (dzw.iterator + tq * topk + rbase + src).load().to(cutlass.Float32)
                s_dz[warpi, i] = hi + lo
                i = i + 32
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()

            e = cutlass.Int32(0)
            while e < ne:
                kid = s_ids[warpi, e]
                dz = s_dz[warpi, e]
                kptr = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        k_idx.iterator.toint() + cutlass.Int64(kid) * (D * 2) + cutlass.Int64(lane) * 8,
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    ),
                    cute.make_layout((4,)),
                )
                for j in cutlass.range_constexpr(4):
                    kv = kptr[j].to(cutlass.Float32)
                    acc[j] = acc[j] + dz * kv
                    dkv[j] = dz * qr[j]
                cute.arch.atomic_add(dk_acc.iterator + kid * D + lane * 4, dkv.load(), scope="gpu")
                e = e + 1
            eb = eb + gather_stage
            if eb < elim:
                cute.arch.sync_warp()

        # ---- dQ: one vec4 atomic request per lane.  A single warp owns the
        # row, so this has no inter-warp contention and is negligible beside
        # the top-k dK requests. ----
        dqp = dq_acc.iterator + row * D + lane * 4
        cute.arch.atomic_add(dqp, acc.load(), scope="gpu")


# ----------------------------------------------- kernel B-alt key-major dK
@cute.kernel
def _zero_i32_kernel(dst: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 256 + tidx
    if i < n:
        (dst.iterator + i).store(cutlass.Int32(0))


@cute.kernel
def _spmm_gather_dq_count_kernel(
    k_idx: cute.Tensor,
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    dzw: cute.Tensor,
    dq_acc: cute.Tensor,
    dq_idx: cute.Tensor,
    key_counts: cute.Tensor,
    topk: cutlass.Constexpr = TOPK,
):
    """Compute dQ edge-major and histogram edges by destination key.

    The definition-form gather used to issue one 128-float FP32 dK RED per
    edge.  Those REDs are 66--71% of its measured L1-to-L2 sectors.  The
    key-major path below replaces them with one scalar count plus a later
    segmented FP32 reduction and one final vector store per key.
    """
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warpi = cute.arch.warp_idx()
    tq = valid_lengths.shape[0]

    gather_split = cutlass.const_expr(8 if int(topk) == GATHER_SPLIT_TOPK else 1)
    gather_chunk = cutlass.const_expr((int(topk) + gather_split - 1) // gather_split)
    row_cohort = cutlass.const_expr(8 if int(topk) == GATHER_SPLIT_TOPK else GATHER_ROW_COHORT)
    gather_stage = cutlass.const_expr(min(GATHER_STAGE_MAX, gather_chunk, 4096 // row_cohort))
    smem = utils.SmemAllocator()
    s_ids = smem.allocate_tensor(cutlass.Int32, cute.make_layout((row_cohort, gather_stage), stride=(gather_stage, 1)), byte_alignment=16)
    s_dz = smem.allocate_tensor(cutlass.Float32, cute.make_layout((row_cohort, gather_stage), stride=(gather_stage, 1)), byte_alignment=16)

    row = bidx * row_cohort + warpi
    if row < tq:
        vlen = (valid_lengths.iterator + row).load()
        rbase = row * topk
        acc = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        acc.fill(0.0)

        eb = cutlass.Int32(0)
        elim = vlen
        if cutlass.const_expr(gather_split > 1):
            _, sidx, _ = cute.arch.block_idx()
            eb = sidx * gather_chunk
            elim = eb + gather_chunk
            if elim > vlen:
                elim = vlen
        while eb < elim:
            ne = elim - eb
            if ne > gather_stage:
                ne = gather_stage
            i = lane
            while i < ne:
                src = eb + i
                kid_s = (selected_indices.iterator + rbase + src).load()
                s_ids[warpi, i] = kid_s
                # Count while the warp stages 32 independent edges instead
                # of serializing the same atomics through lane 0 below.
                cute.arch.atomic_add(
                    key_counts.iterator + kid_s,
                    cutlass.Int32(1),
                    scope="gpu",
                )
                hi = (dzw.iterator + rbase + src).load().to(cutlass.Float32)
                lo = (dzw.iterator + tq * topk + rbase + src).load().to(cutlass.Float32)
                s_dz[warpi, i] = hi + lo
                i = i + 32
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()

            e = cutlass.Int32(0)
            while e < ne:
                kid = s_ids[warpi, e]
                dz = s_dz[warpi, e]
                kptr = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        k_idx.iterator.toint() + cutlass.Int64(kid) * (D * 2) + cutlass.Int64(lane) * 8,
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    ),
                    cute.make_layout((4,)),
                )
                for j in cutlass.range_constexpr(4):
                    acc[j] = acc[j] + dz * kptr[j].to(cutlass.Float32)
                e = e + 1
            eb = eb + gather_stage
            if eb < elim:
                cute.arch.sync_warp()

        if cutlass.const_expr(gather_split > 1):
            cute.arch.atomic_add(
                dq_acc.iterator + row * D + lane * 4,
                acc.load(),
                scope="gpu",
            )
        else:
            out = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
            for j in cutlass.range_constexpr(4):
                out[j] = acc[j].to(cutlass.BFloat16)
            (dq_idx.iterator + row * D + lane * 4).store(out.load())


@cute.kernel
def _spmm_scatter_edges_kernel(
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    dzw: cute.Tensor,
    key_cursor: cute.Tensor,
    edge_rows: cute.Tensor,
    edge_dz: cute.Tensor,
    topk: cutlass.Constexpr = TOPK,
):
    """Materialize compact FP32 `(row, dz)` records in key-major order."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tq = valid_lengths.shape[0]
    edge = bidx * 256 + tidx
    total = tq * topk
    if edge < total:
        row = edge // topk
        slot = edge - row * topk
        valid = (valid_lengths.iterator + row).load()
        if slot < valid:
            kid = (selected_indices.iterator + edge).load()
            pos = cute.arch.atomic_add(key_cursor.iterator + kid, cutlass.Int32(1), scope="gpu")
            hi = (dzw.iterator + edge).load().to(cutlass.Float32)
            lo = (dzw.iterator + total + edge).load().to(cutlass.Float32)
            (edge_rows.iterator + pos).store(row.to(edge_rows.element_type))
            (edge_dz.iterator + pos).store(hi + lo)


@cute.kernel
def _spmm_key_reduce_kernel(
    q_idx: cute.Tensor,
    key_offsets: cute.Tensor,
    edge_rows: cute.Tensor,
    edge_dz: cute.Tensor,
    dk_idx: cute.Tensor,
):
    """One warp/key FP32 segmented reduction; one final dK vector store."""
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warpi = cute.arch.warp_idx()
    tk = dk_idx.shape[0]
    key = bidx * 8 + warpi
    if key < tk:
        begin = (key_offsets.iterator + key).load()
        end = (key_offsets.iterator + key + 1).load()
        acc = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        acc.fill(0.0)
        pos = begin
        while pos < end:
            # Each lane loads one (row, dz) record; the warp then walks the 32
            # broadcast records in order (same sum order as one at a time) with
            # the q-row loads unrolled, so a hot key's segment -- up to every
            # query row of the call -- is not one dependent load chain.
            r_l = cutlass.Int32(0)
            z_l = cutlass.Float32(0.0)
            if pos + lane < end:
                r_l = (edge_rows.iterator + pos + lane).load()
                z_l = (edge_dz.iterator + pos + lane).load()
            for e in cutlass.range_constexpr(32):
                row = cute.arch.shuffle_sync(r_l, e)
                dz = cute.arch.shuffle_sync(z_l, e)
                if pos + e < end:
                    qptr = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            q_idx.iterator.toint() + cutlass.Int64(row) * (D * 2) + cutlass.Int64(lane) * 8,
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        ),
                        cute.make_layout((4,)),
                    )
                    for j in cutlass.range_constexpr(4):
                        acc[j] = acc[j] + dz * qptr[j].to(cutlass.Float32)
            pos = pos + 32
        out = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
        for j in cutlass.range_constexpr(4):
            out[j] = acc[j].to(cutlass.BFloat16)
        (dk_idx.iterator + key * D + lane * 4).store(out.load())


# ---------------------------------------------------------------- kernel B
@cute.kernel
def _spmm_kernel_tcgen05(
    q_idx: cute.Tensor,
    k_idx: cute.Tensor,
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    dzw: cute.Tensor,
    dq_acc: cute.Tensor,
    dk_acc: cute.Tensor,
    mma1: cute.TiledMma,
    mma2: cute.TiledMma,
    lS_a1: cute.ComposedLayout,
    lK_b1: cute.ComposedLayout,
    lS_a2: cute.ComposedLayout,
    lQ_b2: cute.ComposedLayout,
    lFlatS: cute.ComposedLayout,
    lFlatQ: cute.ComposedLayout,
    rectangular: cutlass.Constexpr,
    topk: cutlass.Constexpr = TOPK,
):
    tidx, _, _ = cute.arch.thread_idx()
    # The block scheduler dispatches CTAs in x-fastest linear order, so on the
    # causal path the grid layout IS the list-scheduling order.  Per-CTA work
    # there is min(SPLIT, qt+1) tiles, i.e. 1 to ~11, on a launch of only ~2.1
    # waves, so the order decides the makespan: with SPLIT as the FAST
    # dimension the dispatch sequence is globally descending in work -- all
    # SPLIT pieces of the heaviest qt, then the next, down to the one-tile
    # rows -- which is the LPT order.  With qt fast (the previous shape) that
    # descending sweep restarted SPLIT times, so a 10-tile CTA was still being
    # issued in the last sub-sweep with no residency left to absorb it; ncu
    # measured sm__cycles_active .max / .avg = 1.22 there against 1.10 here.
    # The data-derived rectangular k bound below makes q groups unequal too.
    # At TOPK <= 1024, use the same split-fast/LPT order: it wins 5.6-7.0% at
    # the short rectangular cells and never regresses in the production sweep.
    # TOPK=2048 keeps qt fast for K-tile L2 reuse; its denser scatter dilutes
    # the tail win to 0.2-2.8%, below the retention gate.
    #
    # On the rectangular path a CTA services QPAIR consecutive q-tiles against
    # each K tile so that both of their dK contributions land in one TMEM
    # plane and are flushed once; qt below is then the FIRST q-tile of the
    # group.  The causal path keeps one q-tile per CTA (QP = 1): pairing there
    # would force the lower q-tile to scan the upper one's extra K tiles, on
    # which its S is entirely zero.
    QP = cutlass.const_expr(QPAIR if rectangular else 1)
    if cutlass.const_expr(rectangular):
        if cutlass.const_expr(int(topk) <= 1024):
            sp, qt, _ = cute.arch.block_idx()
            _, nqt, _ = cute.arch.grid_dim()
        else:
            qt, sp, _ = cute.arch.block_idx()
            nqt, _, _ = cute.arch.grid_dim()
    else:
        sp, qt, _ = cute.arch.block_idx()
        _, nqt, _ = cute.arch.grid_dim()
    qt = ((nqt - 1) - qt) * QP
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    tq = valid_lengths.shape[0]
    tk = k_idx.shape[0]

    smem = utils.SmemAllocator()
    # 160KB (QP=1) / 192KB (QP=2) pool, 1 CTA/SM:
    # [Q_0 32KB]..[Q_{QP-1}][K stage0 32KB][K stage1 32KB][S_hi][S_lo]
    # The S pair is reused by each q-tile in turn -- only Q has to stay
    # resident, because dK accumulates across the whole q group.
    pool = smem.allocate_array(BF16, (4 + QP) * BM * D, byte_alignment=1024)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=16)
    # per-warp scratch for the CTA-wide k-extent reduction below; the causal
    # path does not run that reduction, so it does not pay the 128 B either
    if cutlass.const_expr(rectangular):
        s_kend = smem.allocate_tensor(I32, cute.make_layout((32,)), byte_alignment=16)
    pK0 = pool + QP * BM * D
    pK1 = pool + (QP + 1) * BM * D
    pShi = pool + (QP + 2) * BM * D
    pSlo = pool + (QP + 3) * BM * D

    sQg = [cute.make_tensor(cute.recast_ptr(pool + g * BM * D, lQ_b2.inner), lQ_b2.outer) for g in range(QP)]
    sK0 = cute.make_tensor(cute.recast_ptr(pK0, lK_b1.inner), lK_b1.outer)
    sK1 = cute.make_tensor(cute.recast_ptr(pK1, lK_b1.inner), lK_b1.outer)
    sShi = cute.make_tensor(cute.recast_ptr(pShi, lS_a1.inner), lS_a1.outer)
    sShi_t = cute.make_tensor(cute.recast_ptr(pShi, lS_a2.inner), lS_a2.outer)
    sSlo = cute.make_tensor(cute.recast_ptr(pSlo, lS_a1.inner), lS_a1.outer)
    sSlo_t = cute.make_tensor(cute.recast_ptr(pSlo, lS_a2.inner), lS_a2.outer)
    sShi_f = cute.make_tensor(cute.recast_ptr(pShi, lFlatS.inner), lFlatS.outer)
    sSlo_f = cute.make_tensor(cute.recast_ptr(pSlo, lFlatS.inner), lFlatS.outer)
    sQ_fg = [cute.make_tensor(cute.recast_ptr(pool + g * BM * D, lFlatQ.inner), lFlatQ.outer) for g in range(QP)]
    sK0_f = cute.make_tensor(cute.recast_ptr(pK0, lFlatQ.inner), lFlatQ.outer)
    sK1_f = cute.make_tensor(cute.recast_ptr(pK1, lFlatQ.inner), lFlatQ.outer)
    # Raw 128-bit views clear an entire K stage before a predicated partial-
    # tile load. The swizzle is a bijection over these same bytes.
    sK0_zero = cute.make_tensor(
        cute.recast_ptr(pK0, dtype=cutlass.Int64),
        cute.make_layout((BN * D * 2 // 16, 2), stride=(2, 1)),
    )
    sK1_zero = cute.make_tensor(
        cute.recast_ptr(pK1, dtype=cutlass.Int64),
        cute.make_layout((BN * D * 2 // 16, 2), stride=(2, 1)),
    )
    # 128-bit zeroing view spanning BOTH S buffers (64KB)
    sZero = cute.make_tensor(cute.recast_ptr(pShi, dtype=cutlass.Int64), cute.make_layout((BM * BN * 4 // 16, 2), stride=(2, 1)))

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
    cute.arch.mbarrier_init_fence()

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHR)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    # QP dQ planes plus one shared dK plane, 128 columns each.  The allocation
    # must be a power of two, so QP=2 takes all 512 columns; at 1 CTA/SM that
    # costs nothing.
    tmem.allocate(256 if QP == 1 else 512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tQ_f = mma1.make_fragment_C(mma1.partition_shape_C((BM, D)))
    tAccQg = [cute.make_tensor(tmem_ptr + g * 128, tQ_f.layout) for g in range(QP)]
    tK_f = mma2.make_fragment_C(mma2.partition_shape_C((BN, D)))
    tAccK = cute.make_tensor(tmem_ptr + QP * 128, tK_f.layout)

    fS1_hi = mma1.make_fragment_A(sShi)
    fS1_lo = mma1.make_fragment_A(sSlo)
    fK1_0 = mma1.make_fragment_B(sK0)
    fK1_1 = mma1.make_fragment_B(sK1)
    fS2_hi = mma2.make_fragment_A(sShi_t)
    fS2_lo = mma2.make_fragment_A(sSlo_t)
    fQ2g = [mma2.make_fragment_B(sQg[g]) for g in range(QP)]
    nk1 = cutlass.const_expr(cute.size(fS1_hi, mode=[2]))
    nk2 = cutlass.const_expr(cute.size(fS2_hi, mode=[2]))

    tQ2 = tAccQg[0][(None, None), 0, 0]
    tK2 = tAccK[(None, None), 0, 0]
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), F32)
    cp_q = tcgen05.make_tmem_copy(ld32, tQ2)
    cp_k = tcgen05.make_tmem_copy(ld32, tK2)
    # tmem-copy tv layouts span 128 lanes; the TMEM row window follows
    # warp_id % 4, so every 4-warp group gets a coherent slice.
    ltid = tidx % 128
    thr_q = cp_q.get_slice(ltid)
    thr_k = cp_k.get_slice(ltid)
    tr_Qg = [thr_q.partition_S(tAccQg[g][(None, None), 0, 0]) for g in range(QP)]
    tr_K = thr_k.partition_S(tK2)
    # identity tensors give each TMEM fragment element its (row, col) in the
    # BM x D tile, so the flush can atomic straight to gmem (no smem staging)
    idC = cute.make_identity_tensor((BM, D))
    tr_cQ = thr_q.partition_D(idC)
    tr_cK = thr_k.partition_D(idC)
    nslq = cutlass.const_expr(cute.size(tr_Qg[0], mode=[2]))
    nslk = cutlass.const_expr(cute.size(tr_K, mode=[2]))
    rQ = cute.make_rmem_tensor(tr_cQ[None, None, 0].shape, F32)
    rK = cute.make_rmem_tensor(tr_cK[None, None, 0].shape, F32)
    rv4 = cute.make_rmem_tensor(cute.make_layout((4,)), F32)
    z2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Int64)
    z2.fill(0)
    # scatter probe staging: a lane holds SCAT_SUB CONSECUTIVE stream
    # positions, so one wide load replaces SCAT_SUB half-sector loads.
    r_iv = cute.make_rmem_tensor(cute.make_layout((SCAT_SUB,)), I32)
    r_hi = cute.make_rmem_tensor(cute.make_layout((SCAT_SUB,)), BF16)
    r_lo = cute.make_rmem_tensor(cute.make_layout((SCAT_SUB,)), BF16)

    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 16)), cute.make_layout((1, 8)))
    half = lane // 16
    thr_g2s = tiled_g2s.get_slice(lane % 16)
    qbase_i = q_idx.iterator.toint()
    kbase_i = k_idx.iterator.toint()
    nwarp = cutlass.const_expr(NTHR // 32)

    t0g = [(qt + g) * BM for g in range(QP)]
    qhi = t0g[0] + BM
    if qhi > tq:
        qhi = tq
    # CP1 preserves physical causality (qhi bounds the selected ids); the
    # rectangular path scans the whole Tk domain (selected ids are GLOBAL).
    ntile = (qhi + BN - 1) // BN
    if cutlass.const_expr(rectangular):
        ntile = (tk + BN - 1) // BN
    # ---- data-derived k extent.  `selected_indices` is ascending inside each
    # row's valid prefix -- the cursor binary search below relies on it -- so
    # entry vlen-1 is that row's LARGEST key id.  The max over the CTA's rows
    # bounds every k tile that can hold a nonzero of S; past it S is
    # identically zero and the tile body is pure waste: 64 KB of smem
    # zeroing, four MMA passes and a full BN x D fp32 dK flush of ZEROS
    # (adding zero to an already-zeroed dk_acc).  The scatter is the one
    # phase that already costs nothing there, because every row's cursor is
    # spent, which is why the waste never showed up as extra sectors.
    #
    # The bound is read off the data, so it is sound for ANY selection
    # distribution -- it can never skip a tile that holds a nonzero.  It only
    # PAYS when the routing has prefix structure, which causal/varlen
    # attention always has: row r attends [0, ends[r]).  On the rectangular
    # path ends[r] = (Tk - Tq) + r + 1, so q group g can only reach
    # (Tk - Tq)/BN + 2g + 2 of the ntile k tiles it currently scans.
    #
    # Shrinking ntile here rather than breaking out of the loop keeps every
    # column split even, so the SPLIT_RECT_QP CTAs of one q group stay
    # balanced against each other and only the ACROSS-group cost changes.
    #
    # Rectangular only: the causal path already bounds ntile by qhi, which
    # physical causality makes exact, so it would pay the reduction for
    # nothing.  Under const_expr the causal path traces to exactly HEAD.
    if cutlass.const_expr(rectangular):
        row_q = tidx >> 2
        kend = I32(0)
        if tidx % 4 == 0:
            for g in cutlass.range_constexpr(QP):
                rq = t0g[g] + row_q
                if rq < tq:
                    vq = (valid_lengths.iterator + rq).load()
                    if vq > 0:
                        iq = (selected_indices.iterator + rq * topk + vq - 1).load()
                        if iq + 1 > kend:
                            kend = iq + 1
        kend = cute.arch.warp_reduction_max(kend)
        if lane == 0:
            s_kend[warp_idx] = kend
        cute.arch.barrier()
        kall = I32(0)
        for w in cutlass.range_constexpr(NTHR // 32):
            if s_kend[w] > kall:
                kall = s_kend[w]
        ktile = (kall + BN - 1) // BN
        if ktile < ntile:
            ntile = ktile
    nsp = SPLIT
    if cutlass.const_expr(rectangular):
        nsp = SPLIT_RECT_QP
    grp = nsp
    if ntile < nsp:
        grp = ntile

    if sp < grp:
        jb = (sp * ntile) // grp
        je = ((sp + 1) * ntile) // grp

        _ik_p = _ik_b("prolog")
        # ---- stage Q tile + K(jb) into stage 0 (whole CTA; 16 warps).
        # A partial last row block (t0g[g] + BM > tq) pre-zeroes the Q
        # region first: rows past tq then hold zeros, so their (all-zero)
        # S^T rows multiply clean values -- stale smem could be NaN/Inf.
        # Zeroing through a linear view is swizzle-safe only for the full
        # region (the swizzle is a bijection of the same bytes). ----
        for g in cutlass.range_constexpr(QP):
            if t0g[g] + BM > tq:
                zQ = cute.make_tensor(cute.recast_ptr(pool + g * BM * D, dtype=cutlass.Int64), cute.make_layout((BM * D * 2 // 16, 2), stride=(2, 1)))
                for zi in cutlass.range_constexpr(BM * D * 2 // 16 // NTHR):
                    cute.autovec_copy(z2, zQ[zi * NTHR + tidx, None])
                cute.arch.barrier()
            for i in cutlass.range_constexpr(BM // (nwarp * 2)):
                r = i * (nwarp * 2) + warp_idx * 2 + half
                if t0g[g] + r < tq:
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(BF16, qbase_i + (t0g[g] + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D))
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sQ_fg[g], (1, D), (r, 0))),
                    )
        # Packed THD permits a partial final K tile. Clear it before issuing
        # predicated copies so MMA cannot consume stale shared-memory rows.
        if (jb + 1) * BN > tk:
            for zi in cutlass.range_constexpr(BN * D * 2 // 16 // NTHR):
                cute.autovec_copy(z2, sK0_zero[zi * NTHR + tidx, None])
            cute.arch.barrier()
        for i in cutlass.range_constexpr(BN // (nwarp * 2)):
            r = i * (nwarp * 2) + warp_idx * 2 + half
            if jb * BN + r < tk:
                cute.copy(
                    tiled_g2s,
                    thr_g2s.partition_S(
                        cute.make_tensor(
                            cute.make_ptr(BF16, kbase_i + (jb * BN + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D))
                        )
                    ),
                    thr_g2s.partition_D(cute.local_tile(sK0_f, (1, D), (r, 0))),
                )
        cute.arch.cp_async_commit_group()

        # ---- initial cursor: 4 lanes share one row.  Lanes 4r..4r+3 of a
        # warp walk row r's index stream together, so all 512 threads carry
        # row state and the whole CTA can scatter (v4 gave the scatter only
        # 256 of 512 threads, which is what left half the CTA on a barrier).
        row_l = tidx >> 2
        sub = tidx & 3
        nsh = (lane >> 2) * 4
        wg = warp_idx >> 2
        # Every q-tile in the group carries its own stream cursor.  Only `cur`
        # is loop-carried through the k-tile loop, so it is held in named
        # scalars (cur0 / cur1) rather than a list -- the DSL tracks
        # loop-carried values by name.
        vlen_g, rbase_g, sel_g, dzh_g, dzl_g = [], [], [], [], []
        cur0 = cutlass.Int32(0)
        cur1 = cutlass.Int32(0)
        for g in cutlass.range_constexpr(QP):
            vlen = cutlass.Int32(0)
            rbase = cutlass.Int32(0)
            if t0g[g] + row_l < tq:
                vlen = (valid_lengths.iterator + t0g[g] + row_l).load()
                rbase = (t0g[g] + row_l) * topk
            # byte bases for this lane's row; the probe adds only a small
            # int32 element offset on top, so no per-probe 64-bit math.
            sel_row = selected_indices.iterator.toint() + cutlass.Int64(rbase) * 4
            dzh_row = dzw.iterator.toint() + cutlass.Int64(rbase) * 2
            dzl_row = dzh_row + cutlass.Int64(tq) * topk * 2
            cur = cutlass.Int32(0)
            if jb > 0:
                tgt = jb * BN
                lo = cutlass.Int32(0)
                hi = vlen
                bsearch_steps = cutlass.const_expr((int(topk) + 1).bit_length())
                for _ in cutlass.range_constexpr(bsearch_steps):
                    mid = (lo + hi) >> 1
                    if lo < hi:
                        iv = (selected_indices.iterator + rbase + mid).load()
                        if iv < tgt:
                            lo = mid + 1
                        else:
                            hi = mid
                cur = lo
            vlen_g.append(vlen)
            rbase_g.append(rbase)
            sel_g.append(sel_row)
            dzh_g.append(dzh_row)
            dzl_g.append(dzl_row)
            if g == 0:
                cur0 = cur
            else:
                cur1 = cur

        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        _ik_e(_ik_p)

        ph = cutlass.Int32(0)
        st = cutlass.Int32(0)
        jt = jb
        while jt < je:
            j0 = jt * BN
            jhi = j0 + BN

            # ---- phase A: every warp does a slice of every job ----------
            # The scatter is the only K-proportional job here; zero / K
            # prefetch / previous-tile dK flush are fixed per tile.  v4 pinned
            # them to separate warp roles, so at TOPK=2048 half the CTA sat on
            # a barrier (ncu: barrier 40.9% of stall cycles vs 30.3% at 512).
            # Now the CTA runs the jobs in sequence at full width, and the
            # scatter uses 4 lanes per row, which takes the index/dz gather
            # from 26.4 to 9.6 sectors per request.
            #
            # `qg` walks the CTA's q-tiles.  range_constexpr unrolls
            # at trace time, so the three jobs that belong to the
            # K TILE rather than to the (q-tile, k-tile) pair -- the
            # K(jt+1) prefetch, the previous tile's dK flush and the
            # K(jt) wait -- are simply guarded on qg == 0, and at
            # QP = 1 the body traces to exactly what it was before.
            for qg in cutlass.range_constexpr(QP):
                _ik_z = _ik_b("zero")
                for i in cutlass.range_constexpr(BM * BN * 4 // 16 // NTHR):
                    cute.autovec_copy(z2, sZero[i * NTHR + tidx, None])
                _ik_e(_ik_z)
                if qg == 0:
                    _ik_k = _ik_b("kissue")
                    if tidx >= 384:
                        if jt + 1 < je:
                            if j0 + 2 * BN > tk:
                                if st == 0:
                                    for i in cutlass.range_constexpr(BN * D * 2 // 16 // 128):
                                        cute.autovec_copy(
                                            z2,
                                            sK1_zero[i * 128 + (tidx - 384), None],
                                        )
                                else:
                                    for i in cutlass.range_constexpr(BN * D * 2 // 16 // 128):
                                        cute.autovec_copy(
                                            z2,
                                            sK0_zero[i * 128 + (tidx - 384), None],
                                        )
                                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                            if st == 0:
                                for i in cutlass.range_constexpr(BN // (4 * 2)):
                                    r = i * 8 + (warp_idx - 12) * 2 + half
                                    if (j0 + BN) + r < tk:
                                        cute.copy(
                                            tiled_g2s,
                                            thr_g2s.partition_S(
                                                cute.make_tensor(
                                                    cute.make_ptr(BF16, kbase_i + ((j0 + BN) + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16),
                                                    cute.make_layout((1, D)),
                                                )
                                            ),
                                            thr_g2s.partition_D(cute.local_tile(sK1_f, (1, D), (r, 0))),
                                        )
                            else:
                                for i in cutlass.range_constexpr(BN // (4 * 2)):
                                    r = i * 8 + (warp_idx - 12) * 2 + half
                                    if (j0 + BN) + r < tk:
                                        cute.copy(
                                            tiled_g2s,
                                            thr_g2s.partition_S(
                                                cute.make_tensor(
                                                    cute.make_ptr(BF16, kbase_i + ((j0 + BN) + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16),
                                                    cute.make_layout((1, D)),
                                                )
                                            ),
                                            thr_g2s.partition_D(cute.local_tile(sK0_f, (1, D), (r, 0))),
                                        )
                            cute.arch.cp_async_commit_group()
                    _ik_e(_ik_k)
                cute.arch.barrier()
                cur = cur0 if qg == 0 else cur1
                vlen = vlen_g[qg]
                rbase = rbase_g[qg]
                sel_row = sel_g[qg]
                dzh_row = dzh_g[qg]
                dzl_row = dzl_g[qg]
                # ---- scatter: a row's window is SCAT_WIN stream positions and
                # lanes 4r..4r+3 split it, but each lane now owns SCAT_SUB
                # CONSECUTIVE positions instead of a stride-4 sub-step.  The
                # window base is snapped down to a SCAT_ALIGN multiple, so a
                # lane's slice is aligned and the whole probe is ONE wide load per
                # stream (LDG.128 ids + LDG.64 hi + LDG.64 lo) covering 8 rows x
                # one aligned 64B chunk -- v4's stride-4 form issued SCAT_SUB
                # loads per stream, each using half a sector and hitting L1 on
                # the sibling half.  Same bytes, ~4x fewer requests.
                #
                # Snapping down replays up to SCAT_ALIGN-1 already-consumed
                # positions of the FIRST window of each tile.  Those have
                # iv < j0 (a previous tile drained them), so they are counted --
                # keeping cur = base + n exact -- but predicated out of the
                # store.  Later windows replay nothing: the loop only continues
                # when n == SCAT_WIN, and cur = base + SCAT_WIN is then already
                # SCAT_ALIGN-aligned.  Rounding harder (SCAT_ALIGN = SCAT_WIN)
                # measures worse -- it halves the new positions the first window
                # covers, which costs an extra loop pass per tile.
                # The in-tile count is still the popcount of the row's ballot
                # nibble, exact because the ids are ascending (hits are a
                # prefix); summing the SCAT_SUB ballots covers the same
                # SCAT_WIN positions regardless of how lanes are assigned. ----
                _ik_s = _ik_b("scat")
                cur_run = cur
                more = cutlass.Int32(1)
                if cur_run >= vlen:
                    more = cutlass.Int32(0)
                while cute.arch.vote_any_sync(more > 0):
                    # a row whose window was fully consumed is done; without
                    # this the clamp below could re-present the final aligned
                    # window of a full-length row forever.
                    if cur_run >= vlen:
                        more = cutlass.Int32(0)
                    if cutlass.const_expr(int(topk) >= SCAT_WIN and int(topk) % SCAT_ALIGN == 0):
                        ab = cur_run & SCAT_MASK
                        # Keep the SCAT_WIN-wide read inside the row.  Clamping
                        # only moves ab further below cur_run, which the invariant
                        # above already tolerates, and the guard above stops the
                        # clamped window from repeating.
                        if ab > topk - SCAT_WIN:
                            ab = topk - SCAT_WIN
                        lb = ab + SCAT_SUB * sub
                        cute.autovec_copy(
                            cute.make_tensor(
                                cute.make_ptr(I32, sel_row + lb * 4, cute.AddressSpace.gmem, assumed_align=SCAT_AL_I32), cute.make_layout((SCAT_SUB,))
                            ),
                            r_iv,
                        )
                        cute.autovec_copy(
                            cute.make_tensor(
                                cute.make_ptr(BF16, dzh_row + lb * 2, cute.AddressSpace.gmem, assumed_align=SCAT_AL_BF16), cute.make_layout((SCAT_SUB,))
                            ),
                            r_hi,
                        )
                        cute.autovec_copy(
                            cute.make_tensor(
                                cute.make_ptr(BF16, dzl_row + lb * 2, cute.AddressSpace.gmem, assumed_align=SCAT_AL_BF16), cute.make_layout((SCAT_SUB,))
                            ),
                            r_lo,
                        )
                        n = cutlass.Int32(0)
                        for e in cutlass.range_constexpr(SCAT_SUB):
                            pos = lb + e
                            iv = r_iv[e]
                            hit = cutlass.Boolean(False)
                            if more > 0:
                                if pos < vlen:
                                    if iv < jhi:
                                        hit = cutlass.Boolean(True)
                            if hit:
                                if iv >= j0:
                                    sShi_f[row_l, iv - j0] = r_hi[e]
                                    sSlo_f[row_l, iv - j0] = r_lo[e]
                            m = cute.arch.vote_ballot_sync(hit)
                            n = n + cute.arch.popc((m >> nsh) & 0xF)
                        cur_run = ab + n
                        if n < SCAT_WIN:
                            more = cutlass.Int32(0)
                    else:
                        # The wide path relies on a full aligned window.  Retain
                        # four-lane cooperative scatter for smaller or unaligned
                        # arbitrary top-k values, using scalar guarded probes.
                        n = cutlass.Int32(0)
                        for e in cutlass.range_constexpr(SCAT_SUB):
                            pos = cur_run + e * 4 + sub
                            cpos = pos
                            if cpos >= vlen:
                                cpos = cutlass.Int32(0)
                            iv = (selected_indices.iterator + rbase + cpos).load()
                            hi = (dzw.iterator + rbase + cpos).load()
                            lo = (dzw.iterator + tq * topk + rbase + cpos).load()
                            hit = cutlass.Boolean(False)
                            if more > 0:
                                if pos < vlen:
                                    if iv < jhi:
                                        hit = cutlass.Boolean(True)
                            if hit:
                                if iv >= j0:
                                    sShi_f[row_l, iv - j0] = hi
                                    sSlo_f[row_l, iv - j0] = lo
                            m = cute.arch.vote_ballot_sync(hit)
                            n = n + cute.arch.popc((m >> nsh) & 0xF)
                        cur_run = cur_run + n
                        if n < SCAT_WIN:
                            more = cutlass.Int32(0)
                cur = cur_run
                _ik_e(_ik_s)
                if qg == 0:
                    # ---- flush previous tile's dK; the four warp groups take
                    # disjoint TMEM slices, so the flush runs at 4x v4's width ----
                    _ik_f = _ik_b("flush")
                    if jt > jb:
                        jprev = (jt - 1) * BN
                        for sl in cutlass.range_constexpr(nslk):
                            if wg == sl % 4:
                                cute.copy(cp_k, tr_K[None, None, sl], rK)
                                cute.arch.fence_view_async_tmem_load()
                                for g in cutlass.range_constexpr(cute.size(rK.shape) // 4):
                                    crd = tr_cK[None, None, sl][4 * g]
                                    for q in cutlass.range_constexpr(4):
                                        rv4[q] = rK[4 * g + q]
                                    if jprev + crd[0] < tk:
                                        cute.arch.atomic_add(
                                            dk_acc.iterator + (crd[1] // ACC_VEC) * (tk * ACC_VEC) + (jprev + crd[0]) * ACC_VEC, rv4.load(), scope="gpu"
                                        )
                    _ik_e(_ik_f)
                    if tidx >= 384:
                        # steady state: complete K(jt) (committed last iteration) while
                        # keeping the newest prefetch K(jt+1) in flight; on the last
                        # tile nothing was committed, so drain fully.
                        _ik_w = _ik_b("kwait")
                        if jt + 1 < je:
                            cute.arch.cp_async_wait_group(1)
                        else:
                            cute.arch.cp_async_wait_group(0)
                        _ik_e(_ik_w)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()
                # ---- MMA: hi + lo back-to-back (K resident), ONE commit/wait ----
                _ik_m = _ik_b("mma_issue")
                if warp_idx == 0:
                    if st == 0:
                        mma1.set(tcgen05.Field.ACCUMULATE, jt != jb)
                        for kb in cutlass.range_constexpr(nk1):
                            cute.gemm(mma1, tAccQg[qg], fS1_hi[None, None, kb, 0], fK1_0[None, None, kb, 0], tAccQg[qg])
                            mma1.set(tcgen05.Field.ACCUMULATE, True)
                        for kb in cutlass.range_constexpr(nk1):
                            cute.gemm(mma1, tAccQg[qg], fS1_lo[None, None, kb, 0], fK1_0[None, None, kb, 0], tAccQg[qg])
                    else:
                        mma1.set(tcgen05.Field.ACCUMULATE, jt != jb)
                        for kb in cutlass.range_constexpr(nk1):
                            cute.gemm(mma1, tAccQg[qg], fS1_hi[None, None, kb, 0], fK1_1[None, None, kb, 0], tAccQg[qg])
                            mma1.set(tcgen05.Field.ACCUMULATE, True)
                        for kb in cutlass.range_constexpr(nk1):
                            cute.gemm(mma1, tAccQg[qg], fS1_lo[None, None, kb, 0], fK1_1[None, None, kb, 0], tAccQg[qg])
                    # dK accumulates ACROSS the q group: the first q-tile of
                    # the group starts the plane, the rest add into it,
                    # and it is flushed once per k-tile, above.
                    mma2.set(tcgen05.Field.ACCUMULATE, qg != 0)
                    for kb in cutlass.range_constexpr(nk2):
                        cute.gemm(mma2, tAccK, fS2_hi[None, None, kb, 0], fQ2g[qg][None, None, kb, 0], tAccK)
                        mma2.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nk2):
                        cute.gemm(mma2, tAccK, fS2_lo[None, None, kb, 0], fQ2g[qg][None, None, kb, 0], tAccK)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 0)
                _ik_e(_ik_m)
                _ik_mw = _ik_b("mwait")
                cute.arch.mbarrier_wait(mbar + 0, ph)
                _ik_e(_ik_mw)
                ph = ph ^ 1
                if qg == 0:
                    cur0 = cur
                else:
                    cur1 = cur
            st = st ^ 1
            jt = jt + 1

        # ---- final flushes (last dK tile, then dQ), split across the four
        # warp groups the same way as the steady-state flush ----
        _ik_ep = _ik_b("epilog")
        jprev = (je - 1) * BN
        for sl in cutlass.range_constexpr(nslk):
            if wg == sl % 4:
                cute.copy(cp_k, tr_K[None, None, sl], rK)
                cute.arch.fence_view_async_tmem_load()
                for g in cutlass.range_constexpr(cute.size(rK.shape) // 4):
                    crd = tr_cK[None, None, sl][4 * g]
                    for q in cutlass.range_constexpr(4):
                        rv4[q] = rK[4 * g + q]
                    if jprev + crd[0] < tk:
                        cute.arch.atomic_add(dk_acc.iterator + (crd[1] // ACC_VEC) * (tk * ACC_VEC) + (jprev + crd[0]) * ACC_VEC, rv4.load(), scope="gpu")
        for qg in cutlass.range_constexpr(QP):
            # A trailing q-tile of the group can lie entirely past tq when nq
            # is not a multiple of QPAIR; its plane is zero and its rows do
            # not exist, so it is simply not written.  The group's first tile
            # always exists, and `or` short-circuits in Python, so at QP = 1
            # this guard costs no branch at all.
            if cutlass.const_expr(qg == 0) or t0g[qg] < tq:
                for sl in cutlass.range_constexpr(nslq):
                    if wg == sl % 4:
                        cute.copy(cp_q, tr_Qg[qg][None, None, sl], rQ)
                        cute.arch.fence_view_async_tmem_load()
                        for g in cutlass.range_constexpr(cute.size(rQ.shape) // 4):
                            crd = tr_cQ[None, None, sl][4 * g]
                            for q in cutlass.range_constexpr(4):
                                rv4[q] = rQ[4 * g + q]
                            if t0g[qg] + crd[0] < tq:
                                cute.arch.atomic_add(
                                    dq_acc.iterator + (crd[1] // ACC_VEC) * (tq * ACC_VEC) + (t0g[qg] + crd[0]) * ACC_VEC, rv4.load(), scope="gpu"
                                )
        _ik_e(_ik_ep)

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.kernel
def _cast_kernel(
    dq_acc: cute.Tensor,
    dk_acc: cute.Tensor,
    klpart: cute.Tensor,
    dq_idx: cute.Tensor,
    dk_idx: cute.Tensor,
    kl_loss: cute.Tensor,
    blocked: cutlass.Constexpr,
    direct_dq: cutlass.Constexpr,
    direct_dk: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warpi = cute.arch.warp_idx()
    tq = klpart.shape[0]
    tk = dk_idx.shape[0]
    q_total = tq * D
    k_total = tk * D

    allocator = SmemAllocator()
    s_red = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), byte_alignment=16)

    linear = bidx * 256 + tidx
    # `linear` walks the OUTPUT in row-major order, so the bf16 stores stay
    # perfectly coalesced either way; only the fp32 accumulator read changes
    # shape.  Under the blocked layout a warp's 32 consecutive d values split
    # into 8 aligned 16 B chunks one plane apart -- 8 sectors per 128 B read
    # instead of 4, on a buffer the spmm just left resident in L2.
    row = linear // D
    dcol = linear % D
    off = linear
    if cutlass.const_expr(blocked):
        off = (dcol // ACC_VEC) * (tq * ACC_VEC) + row * ACC_VEC + dcol % ACC_VEC
    if cutlass.const_expr(not direct_dq):
        if linear < q_total:
            (dq_idx.iterator + linear).store((dq_acc.iterator + off).load().to(cutlass.BFloat16))
    if cutlass.const_expr(blocked):
        off = (dcol // ACC_VEC) * (tk * ACC_VEC) + row * ACC_VEC + dcol % ACC_VEC
    if cutlass.const_expr(not direct_dk):
        if linear < k_total:
            (dk_idx.iterator + linear).store((dk_acc.iterator + off).load().to(cutlass.BFloat16))

    if bidx == 0:
        acc = cutlass.Float32(0.0)
        i = tidx
        while i < tq:
            acc = acc + (klpart.iterator + i).load()
            i = i + 256
        wsum = cute.arch.warp_reduction_sum(acc)
        if lane == 0:
            s_red[warpi] = wsum
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(8):
                tot = tot + s_red[w]
            (kl_loss.iterator + 0).store(tot)


@cute.jit
def _launch(
    q_idx: cute.Tensor,
    k_idx: cute.Tensor,
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    predict: cute.Tensor,
    teacher: cute.Tensor,
    loss_denominator: cute.Tensor,
    loss_coeff: cutlass.Float32,
    kl_loss: cute.Tensor,
    dq_idx: cute.Tensor,
    dk_idx: cute.Tensor,
    dzw: cute.Tensor,
    klpart: cute.Tensor,
    dq_acc: cute.Tensor,
    dk_acc: cute.Tensor,
    rectangular: cutlass.Constexpr,
    use_gather: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    tq = valid_lengths.shape[0]
    tk = k_idx.shape[0]
    _zero_acc_kernel(dq_acc, tq * D).launch(grid=((tq * D + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)
    _zero_acc_kernel(dk_acc, tk * D).launch(grid=((tk * D + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)
    _dz_kernel(valid_lengths, predict, teacher, loss_denominator, loss_coeff, dzw, klpart, topk=topk).launch(grid=(tq, 1, 1), block=(128, 1, 1), stream=stream)
    # GEMM1: accQ[BM, D] = S[BM, BN(K)] @ K'  (A K-major, B MN-major)
    mma1 = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.MN, F32, tcgen05.CtaGroup.ONE, (BM, D))
    # GEMM2: accK[BN, D] = S^T[BN, BM(K)] @ Q'  (A MN-major, B MN-major)
    mma2 = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.MN, F32, tcgen05.CtaGroup.ONE, (BN, D))
    lS_a1 = sm100_utils.make_smem_layout_a(mma1, (BM, D, BN), BF16, 1)
    lK_b1 = sm100_utils.make_smem_layout_b(mma1, (BM, D, BN), BF16, 1)
    lS_a2 = sm100_utils.make_smem_layout_a(mma2, (BN, D, BM), BF16, 1)
    lQ_b2 = sm100_utils.make_smem_layout_b(mma2, (BN, D, BM), BF16, 1)
    a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    lFlatS = cute.tile_to_shape(a128, (BM, BN), order=(0, 1))
    lFlatQ = cute.tile_to_shape(a128, (BM, D), order=(0, 1))

    if cutlass.const_expr(use_gather):
        gather_split = cutlass.const_expr(8 if int(topk) == GATHER_SPLIT_TOPK else 1)
        _spmm_gather_kernel(
            q_idx,
            k_idx,
            selected_indices,
            valid_lengths,
            dzw,
            dq_acc,
            dk_acc,
            topk=topk,
        ).launch(
            grid=((tq + 7) // 8, gather_split, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    else:
        nq = (tq + BM - 1) // BM
        # SPLIT fast makes the reversed-q dispatch order descending in work.
        gd = (SPLIT, nq, 1)
        if cutlass.const_expr(rectangular):
            if cutlass.const_expr(int(topk) <= 1024):
                gd = (SPLIT_RECT_QP, (nq + QPAIR - 1) // QPAIR, 1)
            else:
                gd = ((nq + QPAIR - 1) // QPAIR, SPLIT_RECT_QP, 1)
        _spmm_kernel_tcgen05(
            q_idx,
            k_idx,
            selected_indices,
            valid_lengths,
            dzw,
            dq_acc,
            dk_acc,
            mma1,
            mma2,
            lS_a1,
            lK_b1,
            lS_a2,
            lQ_b2,
            lFlatS,
            lFlatQ,
            rectangular,
            topk=topk,
        ).launch(
            grid=gd,
            block=(NTHR, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
    nb = (tq * D + 255) // 256
    nbk = (tk * D + 255) // 256
    if nbk > nb:
        nb = nbk
    _cast_kernel(
        dq_acc,
        dk_acc,
        klpart,
        dq_idx,
        dk_idx,
        kl_loss,
        not use_gather,
        False,
        False,
    ).launch(grid=(nb, 1, 1), block=(256, 1, 1), stream=stream)


@cute.jit
def _launch_gather_pre(
    k_idx: cute.Tensor,
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    predict: cute.Tensor,
    teacher: cute.Tensor,
    loss_denominator: cute.Tensor,
    loss_coeff: cutlass.Float32,
    dzw: cute.Tensor,
    klpart: cute.Tensor,
    dq_acc: cute.Tensor,
    dq_idx: cute.Tensor,
    key_counts: cute.Tensor,
    topk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Produce dz/dQ and the destination histogram for key-major dK."""
    tq = valid_lengths.shape[0]
    tk = k_idx.shape[0]
    gather_split = cutlass.const_expr(8 if int(topk) == GATHER_SPLIT_TOPK else 1)
    if cutlass.const_expr(gather_split > 1):
        _zero_acc_kernel(dq_acc, tq * D).launch(
            grid=((tq * D + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
    _zero_i32_kernel(key_counts, tk).launch(grid=((tk + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)
    _dz_kernel(valid_lengths, predict, teacher, loss_denominator, loss_coeff, dzw, klpart, topk=topk).launch(grid=(tq, 1, 1), block=(128, 1, 1), stream=stream)
    row_cohort = cutlass.const_expr(8 if int(topk) == GATHER_SPLIT_TOPK else GATHER_ROW_COHORT)
    _spmm_gather_dq_count_kernel(
        k_idx,
        selected_indices,
        valid_lengths,
        dzw,
        dq_acc,
        dq_idx,
        key_counts,
        topk=topk,
    ).launch(
        grid=((tq + row_cohort - 1) // row_cohort, gather_split, 1),
        block=(row_cohort * 32, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_gather_post(
    q_idx: cute.Tensor,
    selected_indices: cute.Tensor,
    valid_lengths: cute.Tensor,
    dzw: cute.Tensor,
    key_cursor: cute.Tensor,
    key_offsets: cute.Tensor,
    edge_rows: cute.Tensor,
    edge_dz: cute.Tensor,
    klpart: cute.Tensor,
    dq_acc: cute.Tensor,
    dk_acc: cute.Tensor,
    dq_idx: cute.Tensor,
    dk_idx: cute.Tensor,
    kl_loss: cute.Tensor,
    topk: cutlass.Constexpr,
    defer_dk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Materialize key-major edges, reduce dK, and cast final outputs."""
    tq = valid_lengths.shape[0]
    tk = dk_idx.shape[0]
    total = tq * topk
    _spmm_scatter_edges_kernel(
        selected_indices,
        valid_lengths,
        dzw,
        key_cursor,
        edge_rows,
        edge_dz,
        topk=topk,
    ).launch(grid=((total + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)
    if cutlass.const_expr(not defer_dk):
        _spmm_key_reduce_kernel(
            q_idx,
            key_offsets,
            edge_rows,
            edge_dz,
            dk_idx,
        ).launch(grid=((tk + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)
    direct_dq = cutlass.const_expr(int(topk) != GATHER_SPLIT_TOPK)
    nb = 1
    if cutlass.const_expr(not direct_dq):
        nb = (tq * D + 255) // 256
    _cast_kernel(
        dq_acc,
        dk_acc,
        klpart,
        dq_idx,
        dk_idx,
        kl_loss,
        False,
        direct_dq,
        True,
    ).launch(grid=(nb, 1, 1), block=(256, 1, 1), stream=stream)


def _as_cute(tensor, alignment):
    return from_dlpack(tensor, assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic()


@torch.no_grad()
def run(
    q_idx,
    k_idx,
    selected_indices,
    valid_lengths,
    predict,
    teacher,
    kl_loss,
    unit_dq_idx,
    unit_dk_idx,
    loss_denominator=None,
    *,
    loss_coeff: float = 0.1,
    rectangular: bool = False,
    spmm_impl: str = None,
    defer_dk: bool = False,
):
    global _compiled
    loss_coeff = float(loss_coeff)
    tq = q_idx.shape[0]
    tk = k_idx.shape[0]
    # spmm_impl: "auto" (default) / "tile" (the v4 warp-specialized dense-
    # tile kernel) / "gather" (definition-form, work independent of Tk).
    # Density = topk/Tk decides: the tile path wins while tiles are dense
    # (8K: ~6%), the gather path when they are nearly empty -- see
    # GATHER_TK_PER_TOPK for the measured crossover.
    if spmm_impl is None:
        spmm_impl = os.environ.get("SSA_SPMM_IMPL", "auto")
    if spmm_impl not in ("auto", "tile", "gather"):
        raise ValueError(f"unknown spmm_impl: {spmm_impl}")
    if tq <= 0 or selected_indices.numel() % tq != 0:
        raise ValueError("SSA aux selected_indices must contain full query rows")
    topk = selected_indices.numel() // tq
    if topk <= 0:
        raise ValueError(f"SSA aux top_k must be positive, got {topk}")
    tile_supported = topk % 128 == 0
    if spmm_impl == "tile" and not tile_supported:
        raise ValueError(f"SSA aux tile spmm requires top_k divisible by 128, got {topk}; " "use gather or auto for arbitrary top_k")
    if spmm_impl == "auto":
        if not tile_supported:
            spmm_impl = "gather"
        else:
            dense = tk < GATHER_TK_PER_TOPK * topk
            spmm_impl = "tile" if (dense or not rectangular) else "gather"
    use_gather = spmm_impl == "gather"
    dev = q_idx.device
    if (
        selected_indices.shape[0] != tq
        or valid_lengths.shape[0] != tq
        or predict.shape[0] != tq
        or teacher.shape[0] != tq
        or unit_dq_idx.shape[0] != tq
        or (unit_dk_idx is not None and unit_dk_idx.shape[0] != tk)
    ):
        raise ValueError("SSA aux Q/K indexer gradient shapes must match Tq/Tk")
    if unit_dk_idx is None and not defer_dk:
        raise ValueError("SSA aux needs unit_dk_idx unless the key gradient is deferred")
    if defer_dk and tq > 32768:
        raise ValueError("Deferred dK stores int16 query rows; Tq must be at most 32768")
    q = q_idx.view(tq, D)
    k = k_idx.view(tk, D)
    indices = selected_indices.view(tq, topk)
    lengths = valid_lengths.view(tq)
    p = predict.view(tq, topk)
    target = teacher.view(tq, topk)
    dq = unit_dq_idx.view(tq, D)
    # The deferred path never writes the unit key gradient (the reduce is
    # compiled out and the cast stores dQ only), so it may be omitted there.
    dk = unit_dk_idx.view(tk, D) if unit_dk_idx is not None else torch.empty((1, D), device=dev, dtype=torch.bfloat16)
    kl = kl_loss.view(1)
    # Direct SBHD component tests historically supplied no denominator.  Its
    # exact legacy value is the number of query rows; production THD/CP calls
    # always pass a resolved real-token/global-CP denominator explicitly.
    if loss_denominator is None:
        loss_denominator = torch.tensor([float(tq)], device=dev, dtype=torch.float32)
    denominator = loss_denominator.view(1)

    # Scratch for this call only: every plane is zeroed or fully rewritten by
    # the kernels below, all on this stream.  The deferred key-major records are
    # returned to the caller, which keeps them until the key gradient is finished.
    if defer_dk and not use_gather:
        raise RuntimeError("Deferred dK requires the key-major gather path")
    split_dq = use_gather and topk == GATHER_SPLIT_TOPK
    dzw = torch.empty((2, tq, topk), device=dev, dtype=torch.bfloat16)
    klpart = torch.empty((tq,), device=dev, dtype=torch.float32)
    dq_acc = torch.empty((tq, D) if (not use_gather or split_dq) else (1, D), device=dev, dtype=torch.float32)
    dk_acc = torch.empty((tk, D) if not use_gather else (1, D), device=dev, dtype=torch.float32)

    if _compiled is None:
        _compiled = {}
    # Tensor extents, strides, and loss_coeff are dynamic. Keep exact packed
    # lengths and the coefficient out of the JIT key to reuse specializations.
    compile_key = (dev.index, rectangular, use_gather, topk)

    if use_gather:
        # Globally invert the sparse edge list by destination key.  At the
        # production gather cells this workspace costs 8 bytes per edge but
        # removes the measured 16 full dK RED sectors per edge; a final warp
        # writes each FP32 dK row once after reducing all of its source rows.
        total = tq * topk
        key_counts = torch.empty((tk,), device=dev, dtype=torch.int32)
        key_cursor = torch.empty((tk,), device=dev, dtype=torch.int32)
        key_offsets = torch.empty((tk + 1,), device=dev, dtype=torch.int32)
        # Query rows of the deferred records fit int16 (Tq <= 32768 above).
        edge_rows = torch.empty((total,), device=dev, dtype=torch.int16 if defer_dk else torch.int32)
        edge_dz = torch.empty((total,), device=dev, dtype=torch.float32)

        pre_key = ("gather_pre",) + compile_key
        if pre_key not in _compiled:
            pre_args = (
                _as_cute(k, 16),
                _as_cute(indices, 16),
                _as_cute(lengths, 4),
                _as_cute(p, 16),
                _as_cute(target, 16),
                _as_cute(denominator, 4),
                cutlass.Float32(loss_coeff),
                _as_cute(dzw, 16),
                _as_cute(klpart, 4),
                _as_cute(dq_acc, 16),
                _as_cute(dq, 16),
                _as_cute(key_counts, 16),
            )
            _compiled[pre_key] = cute.compile(
                _launch_gather_pre, *pre_args, topk, stream=make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi"
            )
        _compiled[pre_key](
            k,
            indices,
            lengths,
            p,
            target,
            denominator,
            loss_coeff,
            dzw,
            klpart,
            dq_acc,
            dq,
            key_counts,
        )

        key_offsets[0].zero_()
        torch.cumsum(key_counts, dim=0, out=key_offsets[1:])
        key_cursor.copy_(key_offsets[:-1])

        post_key = ("gather_post", defer_dk) + compile_key
        if post_key not in _compiled:
            post_args = (
                _as_cute(q, 16),
                _as_cute(indices, 16),
                _as_cute(lengths, 4),
                _as_cute(dzw, 16),
                _as_cute(key_cursor, 16),
                _as_cute(key_offsets, 16),
                _as_cute(edge_rows, 16),
                _as_cute(edge_dz, 16),
                _as_cute(klpart, 4),
                _as_cute(dq_acc, 16),
                _as_cute(dk_acc, 16),
                _as_cute(dq, 16),
                _as_cute(dk, 16),
                _as_cute(kl, 4),
            )
            _compiled[post_key] = cute.compile(
                _launch_gather_post, *post_args, topk, defer_dk, stream=make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi"
            )
        _compiled[post_key](
            q,
            indices,
            lengths,
            dzw,
            key_cursor,
            key_offsets,
            edge_rows,
            edge_dz,
            klpart,
            dq_acc,
            dk_acc,
            dq,
            dk,
            kl,
        )
    else:
        if compile_key not in _compiled:
            args = (
                _as_cute(q, 16),
                _as_cute(k, 16),
                _as_cute(indices, 16),
                _as_cute(lengths, 4),
                _as_cute(p, 16),
                _as_cute(target, 16),
                _as_cute(denominator, 4),
                cutlass.Float32(loss_coeff),
                _as_cute(kl, 4),
                _as_cute(dq, 16),
                _as_cute(dk, 16),
                _as_cute(dzw, 16),
                _as_cute(klpart, 4),
                _as_cute(dq_acc, 16),
                _as_cute(dk_acc, 16),
            )
            _compiled[compile_key] = cute.compile(
                _launch, *args, rectangular, use_gather, topk, stream=make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi"
            )
        _compiled[compile_key](
            q,
            k,
            indices,
            lengths,
            p,
            target,
            denominator,
            loss_coeff,
            kl,
            dq,
            dk,
            dzw,
            klpart,
            dq_acc,
            dk_acc,
        )

    if defer_dk:
        return DeferredKeyGradient(q, key_offsets, edge_rows, edge_dz)


@cute.kernel
def _spmm_key_reduce_scaled_kernel(
    q_idx: cute.Tensor,
    key_offsets: cute.Tensor,
    edge_rows: cute.Tensor,
    edge_dz: cute.Tensor,
    dk_idx: cute.Tensor,
    grad_scale: cute.Tensor,
):
    """One warp/key FP32 segmented reduction; one final dK vector store."""
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warpi = cute.arch.warp_idx()
    tk = dk_idx.shape[0]
    key = bidx * 8 + warpi
    if key < tk:
        begin = (key_offsets.iterator + key).load()
        end = (key_offsets.iterator + key + 1).load()
        acc = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Float32)
        acc.fill(0.0)
        pos = begin
        while pos < end:
            # As _spmm_key_reduce_kernel: one (row, dz) record per lane, then
            # the warp walks the 32 broadcast records in order with the q-row
            # loads unrolled (same sum order). Rows are int16 on this path.
            r_l = cutlass.Int32(0)
            z_l = cutlass.Float32(0.0)
            if pos + lane < end:
                r_l = (edge_rows.iterator + pos + lane).load().to(cutlass.Int32)
                z_l = (edge_dz.iterator + pos + lane).load()
            for e in cutlass.range_constexpr(32):
                row = cute.arch.shuffle_sync(r_l, e)
                dz = cute.arch.shuffle_sync(z_l, e)
                if pos + e < end:
                    qptr = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            q_idx.iterator.toint() + cutlass.Int64(row) * (D * 2) + cutlass.Int64(lane) * 8,
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        ),
                        cute.make_layout((4,)),
                    )
                    for j in cutlass.range_constexpr(4):
                        acc[j] = acc[j] + dz * qptr[j].to(cutlass.Float32)
            pos = pos + 32
        out = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.BFloat16)
        for j in cutlass.range_constexpr(4):
            rounded = acc[j].to(cutlass.BFloat16).to(cutlass.Float32)
            scale = grad_scale[0].to(cutlass.BFloat16).to(cutlass.Float32)
            out[j] = (rounded * scale).to(cutlass.BFloat16)
        (dk_idx.iterator + key * D + lane * 4).store(out.load())


@cute.jit
def _launch_scaled(q, offsets, rows, dz, dk, scale, stream: cuda.CUstream):
    _spmm_key_reduce_scaled_kernel(q, offsets, rows, dz, dk, scale).launch(grid=((dk.shape[0] + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)


_compiled_scaled = {}


class DeferredKeyGradient:
    """Key-major edge records of one forward; ``finish`` reduces them into dK.

    ``tensors`` are what an autograd caller saves for backward (``from_saved``
    rebuilds the object there), so autograd owns their lifetime.
    """

    def __init__(self, q, offsets, rows, dz):
        self.q = q
        self.offsets = offsets
        self.rows = rows
        self.dz = dz

    @property
    def tensors(self):
        return (self.q, self.offsets, self.rows, self.dz)

    @classmethod
    def from_saved(cls, q, offsets, rows, dz):
        return cls(q, offsets, rows, dz)

    @torch.no_grad()
    def finish(self, scale):
        # Every backward gets its own result, including retain_graph reuse.
        device = self.q.device
        dk = torch.empty((self.offsets.numel() - 1, D), device=device, dtype=torch.bfloat16)
        scale = scale.reshape(1)
        assert scale.dtype == torch.float32 and scale.device == device
        args = (self.q, self.offsets, self.rows, self.dz, dk, scale)
        key = (device.index, self.rows.dtype)
        if key not in _compiled_scaled:
            _compiled_scaled[key] = cute.compile(
                _launch_scaled,
                *(_as_cute(t, 4 if i in (1, 2, 5) else 16) for i, t in enumerate(args)),
                stream=make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        _compiled_scaled[key](*args)
        return dk
