# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index scores and exact top-k selection for packed or rectangular rows.

Selected key IDs are global and sorted. Probabilities are normalized over
the selected keys. Scores use a triangular or dense FP32 workspace."""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32

D = 128
TOPK = 512
BLK = 128
THREADS_SCORE = 128
SCALE = 0.08838834764831845


@cute.kernel
def _score_kernel_tcgen05(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mS: cute.Tensor,
    S_mma: cute.TiledMma,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lFlat: cute.ComposedLayout,
    lFlatK: cute.ComposedLayout,
    tma_c: cute.CopyAtom = None,
    mC_tma: cute.Tensor = None,
    S_mma_h: cute.TiledMma = None,
    tri_rows: cutlass.Constexpr = False,
    rect: cutlass.Constexpr = False,
    n_tile: cutlass.Constexpr = BLK,
):
    # tri_rows: mS is the flat triangular ROW-CONTIGUOUS workspace the
    # vendored DKG top-k consumes: global row r = bi*BLK+rr starts at element
    # tri(bi)*BLK*BLK + rr*(bi+1)*BLK and is fully linear (CP1 causal square).
    # rect: mS is a plain row-major dense (q_pad, k_pad) fp32 workspace (CP /
    # Tq != Tk path; DKG consumes it as a standard 2D input). Launched on a
    # 2D grid (k_nblk, q_nblk); k_pad is derived from grid_dim.
    # neither: legacy block-compressed (nblocks, BLK, BLK) layout.
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()

    if cutlass.const_expr(rect):
        bjx, biy, _ = cute.arch.block_idx()
        nkb, nqb, _ = cute.arch.grid_dim()
        # band-of-8 rasterization: hardware launches blockIdx.x fastest, so a
        # resident wave natively sits inside ONE query row and re-streams the
        # whole K matrix per row (NCU @256K: 4.3GB DRAM read for a 64MB K,
        # 22% L2 hit). Remap so consecutive CTAs cover 8 query rows of the
        # same K column band. Measured: +6% at 128K, neutral elsewhere (the
        # kernel is latency-bound at 3.1TB/s of an ~8TB/s peak, so traffic
        # cuts pay little today; kept for the bandwidth headroom).
        linear = biy * nkb + bjx
        bi = biy
        bj = bjx
        if nqb % 8 == 0:
            band = linear // (8 * nkb)
            rem = linear % (8 * nkb)
            bi = band * 8 + rem % 8
            bj = rem // 8
        blk = bi * nkb + bj
    else:
        blk, _, _ = cute.arch.block_idx()
        # ---- triangular unravel: blk -> (bi, bj), bj <= bi ----
        lo = I32(0)
        hi = I32(1023)  # upper bound on bi; plain binary search
        while lo < hi:
            mid = (lo + hi + I32(1)) // 2
            if (mid * (mid + I32(1))) // 2 <= blk:
                lo = mid
            else:
                hi = mid - I32(1)
        bi = lo
        bj = blk - (bi * (bi + I32(1))) // 2

    smem = utils.SmemAllocator()
    # one raw 64KB region: Q tile (32KB) + K tile (32KB); after the MMA
    # completes it is reused as the fp32 staging buffer for the coalesced
    # epilogue store (the MMA has consumed Q/K by then)
    pQK = smem.allocate_array(BF16, BLK * D + n_tile * D, byte_alignment=1024)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=16)

    sQ = cute.make_tensor(cute.recast_ptr(pQK, lQs.inner), lQs.outer)
    sK = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lKs.inner), lKs.outer)
    # flat (BLK, D) views over the same swizzled memory for the g2s copies
    sQ2 = cute.make_tensor(cute.recast_ptr(pQK, lFlat.inner), lFlat.outer)
    sK2 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lFlatK.inner), lFlatK.outer)
    # plain fp32 epilogue staging views over the whole region
    sOut = cute.make_tensor(cute.recast_ptr(pQK, dtype=F32), cute.make_layout((BLK, BLK), stride=(BLK, 1)))
    sOutV = cute.make_tensor(cute.recast_ptr(pQK, dtype=F32), cute.make_layout((BLK * BLK // 4, 4), stride=(4, 1)))

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
    cute.arch.mbarrier_init_fence()

    # ---- TMEM ----
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS_SCORE)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(n_tile)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = S_mma.make_fragment_C(S_mma.partition_shape_C((BLK, n_tile)))
    tS = cute.make_tensor(tmem_ptr, tS_f.layout)

    fQ = S_mma.make_fragment_A(sQ)
    fK = S_mma.make_fragment_B(sK)

    # ---- stage Q (row block bi) and K (row block bj) via cp.async 64b ----
    qbase = mQ.iterator.toint()
    kbase = mK.iterator.toint()
    # cp.async only supports 128-bit copies: 16 lanes x 8 bf16 = one 128-elem
    # row per half-warp; the two halves of each warp stage different rows.
    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 16)), cute.make_layout((1, 8)))
    half = lane // 16
    thr_g2s = tiled_g2s.get_slice(lane % 16)

    nwarp = cutlass.const_expr(THREADS_SCORE // 32)
    for i in cutlass.range_constexpr(BLK // (nwarp * 2)):
        r = i * (nwarp * 2) + warp_idx * 2 + half
        cute.copy(
            tiled_g2s,
            thr_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, qbase + (bi * BLK + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
            ),
            thr_g2s.partition_D(cute.local_tile(sQ2, (1, D), (r, 0))),
        )
    for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
        r = i * (nwarp * 2) + warp_idx * 2 + half
        cute.copy(
            tiled_g2s,
            thr_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, kbase + (bj * n_tile + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
            ),
            thr_g2s.partition_D(cute.local_tile(sK2, (1, D), (r, 0))),
        )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()

    # ---- UMMA: S = Q @ K^T, fp32 accumulate in TMEM ----
    nk = cutlass.const_expr(cute.size(fQ, mode=[2]))
    if warp_idx == 0:
        S_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nk):
            cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK[None, None, kb, 0], tS)
            S_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + 0)

    cute.arch.mbarrier_wait(mbar + 0, I32(0))

    # ---- epilogue: TMEM -> registers -> smem staging -> coalesced store ----
    # The tmem tiled-copy hands each thread one output row; storing rows
    # straight to gmem is stride-512B (half-empty sectors), so stage through
    # the now-dead Q/K smem region and stream out with flat float4 stores.
    tS2 = tS[(None, None), 0, 0]
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), F32)
    cp_s = tcgen05.make_tmem_copy(ld32, tS2)
    thr_s = cp_s.get_slice(tidx)
    tr_S = thr_s.partition_S(tS2)
    rv4 = cute.make_rmem_tensor(cute.make_layout((4,)), F32)
    if cutlass.const_expr(rect):
        # rect: n_tile == 256. Stage each 128x128 fp32 half (the TMEM column
        # halves have exactly a 128-wide accumulator's layout at col offsets
        # 0/128 -- the spmm dual-accumulator trick), then TMA-bulk-store it.
        # The DKG dense_gemm sweep on this exact problem shape measured the
        # store path as THE bottleneck: 128x256+TMA 308us vs 1371us without
        # (N=32K); plain vec4 stores from smem cost us 464us.
        tSh_f = S_mma_h.make_fragment_C(S_mma_h.partition_shape_C((BLK, BLK)))
        cp_h = tcgen05.make_tmem_copy(ld32, cute.make_tensor(tmem_ptr, tSh_f.layout)[(None, None), 0, 0])
        thr_h = cp_h.get_slice(tidx)
        tr_sO = thr_h.partition_D(sOut)
        nslh = cutlass.const_expr(cute.size(tr_sO, mode=[2]))
        rSh = cute.make_rmem_tensor(tr_sO[None, None, 0].shape, F32)
        gC_epi = cute.flat_divide(mC_tma, (BLK, BLK))
        bSG_s, bSG_g = cpasync.tma_partition(
            tma_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sOut, 0, 2),
            cute.group_modes(gC_epi, 0, 2),
        )
        c_group = pipeline.CooperativeGroup(pipeline.Agent.ThreadBlock)
        c_pipe = pipeline.PipelineTmaStore.create(num_stages=1, producer_group=c_group)
        for h in cutlass.range_constexpr(2):
            tS_h = cute.make_tensor(tmem_ptr + h * BLK, tSh_f.layout)[(None, None), 0, 0]
            tr_Sh = thr_h.partition_S(tS_h)
            for sl in cutlass.range_constexpr(nslh):
                cute.copy(cp_h, tr_Sh[None, None, sl], rSh)
                tr_sO[None, None, sl].store(rSh.load() * F32(SCALE))
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                cute.copy(tma_c, bSG_s[(None,)], bSG_g[(None, bi, bj * 2 + h)])
                c_pipe.producer_commit()
                c_pipe.producer_acquire()
            cute.arch.barrier()
        c_pipe.producer_tail()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)
        return

    tr_sO = thr_s.partition_D(sOut)
    nsl = cutlass.const_expr(cute.size(tr_S, mode=[2]))
    rS = cute.make_rmem_tensor(tr_sO[None, None, 0].shape, F32)
    for sl in cutlass.range_constexpr(nsl):
        cute.copy(cp_s, tr_S[None, None, sl], rS)
        tr_sO[None, None, sl].store(rS.load() * F32(SCALE))
    cute.arch.fence_view_async_tmem_load()
    cute.arch.barrier()

    if cutlass.const_expr(tri_rows):
        # tile (bi, bj) lands at rows base(bi) + rr*(bi+1)*BLK, col bj*BLK.
        # Consecutive threads cover consecutive float4s of one row-slice
        # (32 threads x 16B = 512B contiguous) -> fully coalesced.
        row_stride = (bi + I32(1)) * I32(BLK)
        base_elems = (bi * (bi + I32(1)) // 2) * I32(BLK * BLK) + bj * I32(BLK)
        for i in cutlass.range_constexpr(BLK * BLK // 4 // THREADS_SCORE):
            idx = i * THREADS_SCORE + tidx
            rr = idx // (BLK // 4)
            c4 = idx % (BLK // 4)
            off = base_elems + rr * row_stride + c4 * I32(4)
            gp = cute.make_tensor(
                cute.make_ptr(F32, mS.iterator.toint() + cutlass.Int64(off) * 4, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((4,))
            )
            cute.autovec_copy(sOutV[idx, None], rv4)
            cute.autovec_copy(rv4, gp)
    else:
        gV = cute.make_tensor(
            cute.make_ptr(F32, mS.iterator.toint() + blk * (BLK * BLK * 4), cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout((BLK * BLK // 4, 4), stride=(4, 1)),
        )
        for i in cutlass.range_constexpr(BLK * BLK // 4 // THREADS_SCORE):
            idx = i * THREADS_SCORE + tidx
            cute.autovec_copy(sOutV[idx, None], rv4)
            cute.autovec_copy(rv4, gV[idx, None])

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


class SimplifiedSparseScoreDkg:
    """Score-only launcher feeding the vendored DKG top-k: triangular
    row-contiguous workspace for the CP1 causal square, plain row-major dense
    for the rectangular CP path."""

    def __init__(self, query_pad: int, key_pad: int, triangular: bool):
        self.q_nblk = query_pad // BLK
        # rect MMA tiles are 128x256 -> key_pad is 256-aligned by run()
        self.k_nblk = key_pad // BLK if triangular else key_pad // 256
        self.triangular = triangular

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        s: cute.Tensor,
        stream: cuda.CUstream,
    ):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs = sm100_utils.make_smem_layout_b(S_mma, (BLK, BLK, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))

        if cutlass.const_expr(self.triangular):
            nblocks = self.q_nblk * (self.q_nblk + 1) // 2
            _score_kernel_tcgen05(
                q,
                k,
                s,
                S_mma,
                lQs,
                lKs,
                lFlat,
                lFlat,
                tri_rows=True,
            ).launch(
                grid=(nblocks, 1, 1),
                block=(THREADS_SCORE, 1, 1),
                stream=stream,
            )
        else:
            # rect: 128x256 MMA tiles + TMA-store epilogue (128x128 fp32
            # halves) -- the winning config from the DKG dense_gemm sweep on
            # this problem shape (2737us vs our 4753us at Tk=256K).
            NT = 256
            S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
            lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
            lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
            epi_layout = cute.make_ordered_layout((BLK, BLK), order=(1, 0))
            tma_c, mC_tma = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), s, epi_layout, (BLK, BLK))
            _score_kernel_tcgen05(
                q,
                k,
                mC_tma,
                S_mma_w,
                lQs,
                lKs_w,
                lFlat,
                lFlatK,
                tma_c=tma_c,
                mC_tma=mC_tma,
                S_mma_h=S_mma,
                rect=True,
                n_tile=NT,
            ).launch(
                grid=(self.k_nblk, self.q_nblk, 1),
                block=(THREADS_SCORE, 1, 1),
                stream=stream,
            )


_state = {}
# fusion="auto" takes the filtered route for rectangular rows at this many keys.
_FILTERED_MIN_KEYS = 131072


def reserve(max_query_len, max_key_len, topk, *, triangular=False):
    """Prepare run()'s persistent filtered state for these maximum extents.

    Only the filtered route keeps state between calls; for extents that
    select another route this does nothing.
    """
    fusion = os.environ.get("SSA_INDEXER_FUSION", "auto")
    if fusion == "auto":
        fusion = "filtered" if not triangular and max_key_len >= _FILTERED_MIN_KEYS else "off"
    if fusion == "filtered":
        from .indexer_forward_topk_fused_tcgen05 import reserve_fused_filtered

        reserve_fused_filtered(max_query_len, max_key_len, topk)


@torch.no_grad()
def run(
    q_idx,
    k_idx,
    topk_indices,
    predict,
    key_starts=None,
    key_ends=None,
    *,
    triangular: bool = True,
    score_impl: str = None,
    fusion: str = None,
    topk_impl: str = None,
    topk_overflow_policy: str = "REREAD_ALWAYS",
):
    """Rectangular SSA indexer: tcgen05 score + DKG top-k.

    Same contract as the baseline module: local Q rows against globally
    addressed K with explicit per-row ``[start, end)`` bounds; selected IDs
    are global coordinates into ``k_idx``, ascending, padded with
    ``clamp(end-1, 0)``.
    """
    from .cute_dsl_filter_topk import cute_dsl_topk_prefill_wrapper

    # fusion knob (env SSA_INDEXER_FUSION overrides when the
    # caller does not pass the kwarg):
    #   "off"  -- explicit dense-score comparison path; kernels below.
    #   "filtered" -- seeded single pass: ONE full GEMM, no S, no full
    #             histogram; sample-pass threshold seed + cursor-count exact
    #             verification. The measured-best fusion scheme.
    #   "auto" (default) -- use filtered compact select for rectangular
    #             Tk >= 128K at every supported TopK (512, 1024, 2048),
    #             avoiding the full score matrix at long contexts.
    #             "off" materializes S and is not the bounded-memory path.
    if fusion is None:
        fusion = os.environ.get("SSA_INDEXER_FUSION", "auto")
    if fusion in ("hist", "full"):
        raise ValueError(f"SSA fusion={fusion!r} is an unqualified experimental route; " "use 'auto', 'off', or 'filtered'")
    if fusion not in ("auto", "off", "filtered"):
        raise ValueError(f"unknown fusion: {fusion}")
    if topk_overflow_policy != "REREAD_ALWAYS":
        raise ValueError("SSA DKG top-k requires topk_overflow_policy='REREAD_ALWAYS'; " "the conditional REREAD overflow branch is not exact")
    if fusion == "auto":
        fusion = "filtered" if not triangular and k_idx.shape[0] >= _FILTERED_MIN_KEYS else "off"
    if fusion == "filtered":
        from .indexer_forward_topk_fused_tcgen05 import run_fused_filtered

        return run_fused_filtered(q_idx, k_idx, topk_indices, predict, key_starts, key_ends)

    # score_impl knob: "native" (default; the hand-written tcgen05 kernel) or
    # "dense_gemm" (vendored DKG DenseGemmKernel at the sweep-winning config
    # 128x256 + TMA store; rect only). Env SSA_SCORE_IMPL overrides when the
    # caller does not pass the kwarg (the caller does not).
    if score_impl is None:
        score_impl = os.environ.get("SSA_SCORE_IMPL", "auto")
    if score_impl not in ("auto", "native", "dense_gemm"):
        raise ValueError(f"unknown score_impl: {score_impl}")

    query_len = q_idx.shape[0]
    key_len = k_idx.shape[0]
    if key_starts is None or key_ends is None:
        if key_starts is not None or key_ends is not None:
            raise ValueError("SSA indexer requires both key_starts and key_ends")
        if query_len != key_len:
            raise ValueError("rectangular SSA dispatch requires explicit key bounds")
        key_starts = torch.zeros((query_len,), device=q_idx.device, dtype=torch.int32)
        key_ends = torch.arange(1, query_len + 1, device=q_idx.device, dtype=torch.int32).clamp_max(key_len)
    if key_starts.shape != (query_len,) or key_ends.shape != (query_len,):
        raise ValueError(
            "SSA indexer expects one contiguous [start,end) pair per query row; "
            f"got starts={tuple(key_starts.shape)}, ends={tuple(key_ends.shape)}, "
            f"query_len={query_len}"
        )
    if key_starts.dtype != torch.int32 or key_ends.dtype != torch.int32:
        raise TypeError("SSA indexer key_starts/key_ends must be int32")
    if triangular and query_len != key_len:
        raise ValueError("triangular workspace requires query_len == key_len")

    query_pad = ((query_len + BLK - 1) // BLK) * BLK
    # rect MMA tiles are 128x256: pad K (and the dense workspace width) to
    # 256; the pad columns are never read (bounds end <= key_len).
    _kalign = BLK if triangular else 256
    key_pad = ((key_len + _kalign - 1) // _kalign) * _kalign
    q = q_idx.view(query_len, D)
    k = k_idx.view(key_len, D)
    if query_pad != query_len:
        qp = torch.zeros((query_pad, D), dtype=q.dtype, device=q.device)
        qp[:query_len].copy_(q)
        q = qp
    if key_pad != key_len:
        kp = torch.zeros((key_pad, D), dtype=k.dtype, device=k.device)
        kp[:key_len].copy_(k)
        k = kp

    topk = topk_indices.numel() // query_len
    if topk not in (512, 1024, 2048):
        raise ValueError(f"SSA indexer top_k must be 512/1024/2048, got {topk}")
    out_idx = topk_indices.view(query_len, topk)
    out_prob = predict.view(query_len, topk)
    starts = key_starts.contiguous()
    ends = key_ends.contiguous()

    # auto: the vendored DenseGemmKernel wins at the kernel level from 64K up
    # (1.15-1.48x) but its TMA-store L2 footprint slows the downstream top-k
    # at exactly the 64K L2-margin tier (pipeline -27%, reproduced twice);
    # from 128K the top-k is DRAM-bound anyway and the gain is net (+14%/+5%
    # pipeline at 128K/256K). Dispatch dense_gemm only at >= 128K columns.
    if score_impl == "auto":
        score_impl = "dense_gemm" if (not triangular and key_len >= 131072) else "native"
    use_dg = score_impl == "dense_gemm" and not triangular
    # Score tensors are block-padded below.  Exact packed Tq/Tk can vary while
    # the executable shapes remain identical, so cache by those executable
    # shapes rather than recompiling for every document-boundary layout.
    dev = q.device
    stream_id = torch.cuda.current_stream(dev).cuda_stream
    # The materialized score workspace is mutable scratch.  Separate streams
    # may execute concurrently, so sharing one shape-keyed workspace would
    # race even though calls on one stream remain ordered.
    cache_key = (dev.index, stream_id, query_pad, key_pad, triangular, use_dg)
    st = _state.get(cache_key)
    if st is None:
        if triangular:
            nblk = query_pad // BLK
            ws = torch.empty((nblk * (nblk + 1) // 2) * BLK * BLK, dtype=torch.float32, device=dev)
        else:
            ws = torch.empty((query_pad, key_pad), dtype=torch.float32, device=dev)
        if use_dg:
            import cuda.bindings.driver as _cuda

            from .dense_gemm_sm100 import DenseGemmKernel

            gemm = DenseGemmKernel(F32, False, (128, 256), (1, 1), True)
            mnkl = (query_pad, key_pad, D, 1)
            if not gemm.can_implement(mnkl, BF16, F32, "k", "k", "n"):
                raise ValueError(f"dense_gemm cannot implement {mnkl}")
            a_c = from_dlpack(q.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
            b_c = from_dlpack(k.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
            c_c = from_dlpack(ws.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
            stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            score_compiled = cute.compile(gemm, a_c, b_c, c_c, stream, lambda x: x * SCALE)
        else:
            op = SimplifiedSparseScoreDkg(query_pad, key_pad, triangular)
            score_compiled = cute.compile(
                op,
                from_dlpack(q, assumed_align=16, enable_tvm_ffi=True),
                from_dlpack(k, assumed_align=16, enable_tvm_ffi=True),
                from_dlpack(ws, assumed_align=16, enable_tvm_ffi=True),
                stream=make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        st = dict(ws=ws, score=score_compiled)
        _state[cache_key] = st

    if use_dg:
        # from_dlpack wrapping costs ~50us each on the host and would delay
        # the launch; memoize by (data_ptr, stream) -- training loops reuse
        # allocator addresses, so this hits nearly always.
        import cuda.bindings.driver as _cuda

        raw_stream = torch.cuda.current_stream().cuda_stream
        sig = (q.data_ptr(), k.data_ptr(), raw_stream)
        if st.get("dg_sig") != sig:
            st["dg_args"] = (
                from_dlpack(q.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1),
                from_dlpack(k.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1),
                from_dlpack(st["ws"].unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1),
                _cuda.CUstream(raw_stream),
            )
            st["dg_sig"] = sig
        a_c, b_c, c_c, stream = st["dg_args"]
        st["score"](a_c, b_c, c_c, stream)
    else:
        st["score"](q, k, st["ws"])
    if triangular:
        topk_input = st["ws"]
    else:
        # standard 2D row-major input; slice off the padded query rows so the
        # DKG grid matches query_len exactly (the slice stays compact)
        topk_input = st["ws"][:query_len]
    # topk_impl knob: "dkg" (vendored DKG filtered top-k), "gvr" (vendored GVR
    # self-sampling top-k -- exact, fully on-device via gvr_sort_predict), or
    # "auto" (default) remains on DKG. The direct GVR kernel shifts each row's
    # fp32 base by key_starts[r] before float4 loads, so arbitrary packed-THD
    # starts can violate its 16-byte alignment requirement. There is no
    # synchronization-free host predicate that can prove every device start
    # aligned; fail closed instead of risking a context-poisoning launch.
    # Filtered fusion may still explicitly qualify GVR on its compact,
    # allocator-aligned candidate matrix via SSA_FILTERED_SELECT=gvr.
    if topk_impl is None:
        topk_impl = os.environ.get("SSA_TOPK_IMPL", "auto")
    if topk_impl not in ("dkg", "gvr", "auto"):
        raise ValueError(f"unknown topk_impl: {topk_impl}")
    if topk_impl == "auto":
        topk_impl = "dkg"
    if topk_impl == "gvr":
        raise ValueError(
            "SSA direct GVR top-k is unsupported for the arbitrary packed-THD " "key-start contract; select topk_impl='dkg' or use filtered fusion"
        )
    cute_dsl_topk_prefill_wrapper(
        topk_input,
        starts,
        ends,
        topk,
        overflow_policy=topk_overflow_policy,
        return_val=False,
        in_kernel_sort=True,
        ssa_epilogue=True,
        ssa_tri_input=triangular,
        logical_num_cols=key_pad,
        subtract_row_start=False,
        out_indices=out_idx,
        out_probs=out_prob,
        # The float-compare classifier cannot place an exact +0.0 at the
        # signed-zero bin straddle: it drops that element and leaves one
        # output slot unwritten. Integer-key classification is exact.
        boundary_classify=False,
    )
