"""Lean SM100 indexer-forward score kernel — H=64 / D=128 specialization.

Persistent tcgen05 CuTe DSL kernel producing dense lightning-indexer scores

    out[i, j] = sm_scale * sum_h relu( dot(q[i, h, :], kv[j, :]) ) * w[i, h]

for ``j`` in the per-row visibility window ``[ks[i], ke[i])`` (both clamped
to ``S_k``). ``sm_scale`` is a runtime scalar applied to the fp32
head-reduced score (same placement as the legacy kernel: post head-reduce,
pre causal mask); the ``sm_scale == 1.0`` variant compiles the multiply
out entirely (``sm_scale=None`` — an absent optional folds out of the
parameter layout at trace time), keeping the production instruction
stream identical to the scale-free schedule. ``w`` may be BF16
or FP32; BF16 weights are up-converted to FP32 in the staging copy (exact),
so the register math is identical for both ingest dtypes. Positions inside
a swept 128-column KV tile but outside the row's window are written
``-inf``; columns in tiles the kernel never sweeps are left untouched —
callers that rely on ``-inf`` there must pre-fill the output
(``indexer_forward_lean_wrapper`` does). ``S_k`` may be any positive size:
the KV TMA descriptor carries the true extent, so partial trailing tiles
are zero-filled by the TMA hardware and the fp32 stores are bounds-guarded
(``col < S_k``); when ``S_k`` is a multiple of the 128-row KV tile the
wrapper compiles with a fully static K extent instead.

Schedule (the lean fast path for the ``qhead_per_kv_head == 64`` case):

  * swapAB UMMA: A = one dense 128-row KV tile (M = 128, TMA), B = the
    tile's TQ=4 tokens x H=64 packed (token, head) query rows (N = 256,
    TMA, loaded once per tile), K = head_dim 128; fp32 accumulation in
    TMEM, 2 x 256-column slot ring (512 TMEM columns).
  * Static reversed-LPT persistent grid: ``min(sm_count, num_tiles)`` CTAs;
    CTA ``b`` handles linear tile ids ``b, b+G, b+2G, ...`` mapped in
    reverse so block 0 takes the largest causal KV window (LPT balance on
    the triangular work distribution). No dynamic tile scheduler.
  * 12 warps / 384 threads: warp 0 = TMA load (Q double-buffered with
    next-block prefetch, KV ``kv_stage``-deep), warp 1 = UMMA, warps 2-3
    idle; two epilogue warpgroups (warps 4-11) each own one fixed TMEM
    slot and drain alternating KV tiles.
  * Raw-mbarrier rolling split-LDTM drain: one full/empty mbarrier pair +
    1-bit phase per epilogue warpgroup; each 64-column token chunk is one
    Ld32x32b x64 LDTM, and every LDTM after the first is issued behind the
    previous chunk's FMA reduction; the TMEM slot is released after the
    last fence, before the last reduction, so the UMMA warp refills it
    while the epilogue finishes math out of registers.
  * fp32 relu-weight head-sum in registers (packed f32x2 FMA, fixed
    reduction order — deterministic run-to-run, no atomics).

Numerics: bf16 tensor-core products with fp32 accumulation and an fp32
relu/weight/head-sum epilogue. Register budget: 128 x 24 (load/MMA
warpgroup) + 256 x 216 (epilogue warpgroups) = 58368 <= 64K. Shared
memory: 2 x 64 KB sQ + 3 x 32 KB sK + 2 KB sW = 226 KB <= the 227 KB CTA
limit.

Shape support is validated by ``IndexerForwardLean.check_support()`` in
``api_lean.py``; the asserts here are compile-time backstops only.
"""

import cuda.bindings.driver as cuda

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import BFloat16, Float32


class IndexerForwardSm100Lean:
    """Persistent swapAB dense-TMA score kernel, one 4-token tile per trip."""

    def __init__(self, num_heads: int, head_dim: int, sm_count: int):
        # check_support() gates dispatch; these are compile-time backstops.
        assert num_heads == 64, "lean schedule is specialized for H=64"
        assert head_dim == 128, "lean schedule is specialized for D=128"
        assert sm_count > 0
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sm_count = sm_count
        self.io_dtype = BFloat16

        self.tq = 4  # query tokens per tile
        self.n_block = 128  # KV rows per MMA tile (M)
        self.n_cols = self.tq * num_heads  # MMA N = 256 packed query rows
        self.k_block = head_dim  # single K chunk
        self.kv_stage = 3  # KV TMA pipeline depth
        self.q_stage = 2  # Q double-buffer w/ next-block prefetch
        # (M, N, K) = (kv block, tq*heads, head_dim)
        self.mma_tiler = (self.n_block, self.n_cols, self.k_block)

        self.load_warp_id = 0
        self.mma_warp_id = 1
        # 12 warps: WG0 = load + MMA + 2 idle; WG1/WG2 = epilogue.
        self.epi_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.num_warps = 12
        self.threads_per_cta = 32 * self.num_warps

        SM100_TMEM_COLS = 512
        self.num_tmem_slots = SM100_TMEM_COLS // self.n_cols  # 2
        self.tmem_alloc_cols = SM100_TMEM_COLS

        # one Ld32x32b x64 LDTM per 64-column token chunk
        self.epi_rep = 64

        # register redistribution across 384 threads:
        # 128 x 24 + 256 x 216 = 58368 <= 64K regs.
        self.num_regs_wg0 = 24
        self.num_regs_epi = 216

        # participants: mma warp + 8 epilogue warps
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (1 + len(self.epi_warp_ids)))
        self.epi_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * len(self.epi_warp_ids))

    # -----------------------------------------------------------------
    # host side
    # -----------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (S*H, D) bf16 (q.view(S*H, D), row-major)
        mKV: cute.Tensor,  # (SKV, D) bf16
        mW: cute.Tensor,  # (S, H) fp32 or bf16
        mKS: cute.Tensor,  # (S,) int32
        mKE: cute.Tensor,  # (S,) int32
        mOut: cute.Tensor,  # (S, SKV) fp32
        sm_scale: Optional[Float32],  # None -> the multiply is compiled out
        stream: cuda.CUstream,
    ):
        num_tiles = cute.size(mQ.shape[0]) // self.n_cols  # S // TQ
        mQ_v = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (self.n_cols, self.head_dim, num_tiles),
                stride=(self.head_dim, 1, self.n_cols * self.head_dim),
            ),
        )

        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            cta_group,
            self.mma_tiler[:2],
        )
        # epilogue-only tiled mma (no MMA issued with it): the drain runs in
        # per-token 64-column chunks ((128, 64) fragment/copy layouts).
        tiled_mma_epi = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            cta_group,
            (self.n_block, self.num_heads),
        )

        sK_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.io_dtype, self.kv_stage)
        sQ_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.io_dtype, self.q_stage)

        cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mKV,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_v,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        self.tma_copy_q_bytes = self.n_cols * self.head_dim * (self.io_dtype.width // 8)
        self.tma_copy_k_bytes = self.n_block * self.k_block * (self.io_dtype.width // 8)

        num_ctas = cutlass.min(Int32(self.sm_count), num_tiles)
        self.kernel(
            tiled_mma,
            tiled_mma_epi,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_Q,
            tma_tensor_Q,
            mW,
            mKS,
            mKE,
            mOut,
            sm_scale,
            sQ_layout,
            sK_layout,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    # -----------------------------------------------------------------
    # device side
    # -----------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_epi: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        mW: cute.Tensor,  # (S, H) fp32 or bf16
        mKS: cute.Tensor,  # (S,) int32
        mKE: cute.Tensor,  # (S,) int32
        mOut: cute.Tensor,  # (S, SKV) fp32
        sm_scale: Optional[Float32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
    ):
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        skv = Int32(cute.size(mOut.shape[1]))
        # persistent schedule: CTA b handles linear ids b, b+G, b+2G, ...
        # mapped in *reverse* (largest KV window first, LPT balance).
        num_tiles = Int32(cute.size(mW.shape[0])) // self.tq
        trips = cute.ceil_div(num_tiles - Int32(bidx), gdim)

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.kv_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.num_tmem_slots]
            tmem_holding_buf: cutlass.Int32
            sW: cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.tq * self.num_heads], 16]
            sQ: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.io_dtype, cute.cosize(sK_layout)], 1024]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ---- pipelines ----
        pipe_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        )
        pipe_K = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.K_mbar_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_k_bytes,
            defer_sync=True,
        )
        pipe_S = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            num_stages=self.num_tmem_slots,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epi_warp_ids[0],
        )
        # raw handle to the pipe_S mbarrier ring for the epilogue's minimal
        # consumer state machine (hoisted here: the SharedStorage python
        # object cannot cross the dynamic warpgroup branch)
        s_mbar_base = storage.S_mbar_ptr.data_ptr().align(min_align=8)

        pipeline.pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sW = storage.sW.get_tensor(cute.make_layout((2 * self.tq * self.num_heads,)))

        pipeline.pipeline_init_wait()

        # accumulator reference layouts
        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C(cute.select(self.mma_tiler, mode=[0, 1]))
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_tmem_slots))
        # each slot drains in TQ 64-column token chunks
        epi_chunks = self.num_tmem_slots * self.tq
        acc_shape_epi = tiled_mma_epi.partition_shape_C((self.n_block, self.num_heads))
        acc_fake_epi = tiled_mma_epi.make_fragment_C(cute.append(acc_shape_epi, epi_chunks))

        wg_idx = tidx // 128

        # =============================================================
        # Warpgroup 0: load warp + MMA warp (+ 2 idle warps)
        # =============================================================
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)

            if warp_idx == self.load_warp_id:
                # --- partitions (loop-invariant) ---
                gQ = cute.local_tile(
                    tma_tensor_Q,
                    cute.select(self.mma_tiler, mode=[1, 2]),
                    (None, None, None),
                )
                tSgQ = thr_mma.partition_B(gQ)
                tQsQ, tQgQ = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tSgQ, 0, 3),
                )
                gK = cute.local_tile(
                    tma_tensor_K,
                    cute.select(self.mma_tiler, mode=[0, 2]),
                    (None, None),
                )
                tSgK = thr_mma.partition_A(gK)
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stage)
                k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
                # Q of the first block
                tile0 = num_tiles - Int32(1) - Int32(bidx)
                pipe_Q.producer_acquire(q_prod)
                cute.copy(
                    tma_atom_Q,
                    tQgQ[(None, 0, 0, tile0)],
                    tQsQ[(None, q_prod.index)],
                    tma_bar_ptr=pipe_Q.producer_get_barrier(q_prod),
                )
                q_prod.advance()
                for w in cutlass.range(trips):
                    linear = Int32(bidx) + w * gdim
                    # prefetch the NEXT block's Q ahead of this block's
                    # KV tiles (so MMA never stalls on Q at a block boundary)
                    nxt = linear + gdim
                    if nxt < num_tiles:
                        ntile = num_tiles - Int32(1) - nxt
                        pipe_Q.producer_acquire(q_prod)
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[(None, 0, 0, ntile)],
                            tQsQ[(None, q_prod.index)],
                            tma_bar_ptr=pipe_Q.producer_get_barrier(q_prod),
                        )
                        q_prod.advance()

                    tile_idx = num_tiles - Int32(1) - linear
                    q0 = tile_idx * self.tq
                    ks_min = skv
                    ke_max = Int32(0)
                    for r in cutlass.range_constexpr(self.tq):
                        ks_min = cutlass.min(ks_min, cutlass.min(Int32(mKS[q0 + r]), skv))
                        ke_max = cutlass.max(ke_max, cutlass.min(Int32(mKE[q0 + r]), skv))
                    k_tile0 = ks_min // self.n_block
                    span = cutlass.max(ke_max - k_tile0 * self.n_block, Int32(0))
                    n_iters = cute.arch.make_warp_uniform(cute.ceil_div(span, self.n_block))
                    k_tile0 = cute.arch.make_warp_uniform(k_tile0)

                    for i in cutlass.range(n_iters):
                        pipe_K.producer_acquire(k_prod)
                        cute.copy(
                            tma_atom_K,
                            tKgK[(None, k_tile0 + i, 0)],
                            tKsK[(None, k_prod.index)],
                            tma_bar_ptr=pipe_K.producer_get_barrier(k_prod),
                        )
                        k_prod.advance()

            elif warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                tmem_base = tmem.retrieve_ptr(Float32)
                tAcc = cute.make_tensor(tmem_base, acc_fake.layout)

                tSrK = tiled_mma.make_fragment_A(sK)
                tSrQ = tiled_mma.make_fragment_B(sQ)

                q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.q_stage)
                k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)
                s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tmem_slots)

                for w in cutlass.range(trips):
                    linear = Int32(bidx) + w * gdim
                    tile_idx = num_tiles - Int32(1) - linear
                    q0 = tile_idx * self.tq
                    ks_min = skv
                    ke_max = Int32(0)
                    for r in cutlass.range_constexpr(self.tq):
                        ks_min = cutlass.min(ks_min, cutlass.min(Int32(mKS[q0 + r]), skv))
                        ke_max = cutlass.max(ke_max, cutlass.min(Int32(mKE[q0 + r]), skv))
                    k_tile0 = ks_min // self.n_block
                    span = cutlass.max(ke_max - k_tile0 * self.n_block, Int32(0))
                    n_iters = cute.arch.make_warp_uniform(cute.ceil_div(span, self.n_block))

                    pipe_Q.consumer_wait(q_cons)
                    for i in cutlass.range(n_iters):
                        pipe_S.producer_acquire(s_prod)
                        acc = tAcc[(None, None, None, s_prod.index)]
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        pipe_K.consumer_wait(k_cons)
                        for kb in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tSrK[(None, None, kb, k_cons.index)],
                                tSrQ[(None, None, kb, q_cons.index)],
                                acc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        pipe_K.consumer_release(k_cons)
                        k_cons.advance()
                        pipe_S.producer_commit(s_prod)
                        s_prod.advance()
                    pipe_Q.consumer_release(q_cons)
                    q_cons.advance()

        # =============================================================
        # Warpgroups 1-2: epilogue — each owns one TMEM slot, drains
        # alternating KV tiles (per-warpgroup UMMA ping-pong)
        # =============================================================
        else:
            cute.arch.setmaxregister_increase(self.num_regs_epi)
            if warp_idx == self.epi_warp_ids[0]:
                tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_base = tmem.retrieve_ptr(Float32)
            tAccE = cute.make_tensor(tmem_base, acc_fake_epi.layout)

            tidx_wg = tidx % 128
            epi_wg = wg_idx - 1  # 0 or 1 == owned TMEM slot

            thr_mma_epi = tiled_mma_epi.get_slice(tidx_wg)
            cS = cute.make_identity_tensor((self.n_block, self.num_heads))
            tAcc0 = tAccE[(None, None, None, 0)]
            # Compile-time layout probe: a probe tiled-copy built once out
            # here derives this thread's KV row within the tile, the
            # per-chunk register fragment shape, and the per-chunk TMEM
            # source partitions (tTR_srcs below) — all of which feed the
            # persistent loop. The tiled-copy actually issuing the LDTMs
            # is re-created INSIDE the tile loop so no !cute.tiled_copy
            # value is live across a dynamic back-edge (DSL 4.5.x
            # make_tmem_copy loop-carry limitation); the probe itself is
            # trace-time only and costs nothing at runtime.
            epi_rep = self.epi_rep
            probe_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(epi_rep)),
                Float32,
            )
            thr_probe = tcgen05.make_tmem_copy(probe_atom, tAcc0).get_slice(tidx_wg)
            tScS = thr_probe.partition_D(thr_mma_epi.partition_C(cS))
            # this thread's KV row within the tile (== TMEM lane)
            row = cute.get(tScS[0], mode=[0])

            tSrS_shape = thr_probe.partition_D(cute.make_identity_tensor(tAcc0.shape)).shape
            # rolling split-LDTM lookahead: two live fragments so every LDTM
            # after the first is in flight behind the previous chunk's FMA
            # reduction. Chunk computations are independent -> the reduction
            # order (and hence the result) matches a serial per-chunk drain
            # bit for bit.
            tSrS_a = cute.make_rmem_tensor(tSrS_shape, Float32)
            tSrS_b = cute.make_rmem_tensor(tSrS_shape, Float32)
            rW = cute.make_rmem_tensor((self.num_heads,), Float32)

            mW_flat = cute.make_tensor(
                mW.iterator,
                cute.make_layout((cute.size(mW.shape[0]) * self.num_heads,)),
            )
            # WG-private weight staging buffer (1KB apart -> align holds)
            sW_wg = (sW.iterator + epi_wg * (self.tq * self.num_heads)).align(16)

            # Raw-mbarrier drain state: each epilogue WG owns exactly ONE
            # of the two 256-column TMEM slots, so the generic pipeline
            # consumer state machine (phase XORs, index SELs, advances) is
            # replaced by one fixed full/empty mbarrier pair and a 1-bit
            # phase. Semantically identical to consumer_wait(full,
            # phase0-start) / arrive(empty): same mbarriers, same counts
            # (128 arrives per stage = this WG). Hoisted per-chunk TMEM
            # partitions (tq of them, fixed per WG) remove the per-tile
            # dynamic TMEM layout arithmetic. Plain tensors cross the
            # dynamic back-edge fine (the 4.5.x ICE is tiled-copy-only).
            tTR_srcs = []
            for cc in cutlass.range_constexpr(self.tq):
                tTR_srcs.append(
                    thr_probe.partition_S(
                        tAccE[
                            (
                                None,
                                None,
                                None,
                                Int32(epi_wg) * self.tq + cc,
                            )
                        ]
                    )
                )
            full_bar = s_mbar_base + Int32(epi_wg)
            empty_bar = s_mbar_base + (Int32(epi_wg) + self.num_tmem_slots)
            phase = Int32(0)

            ks_q = cute.make_rmem_tensor((self.tq,), Int32)
            ke_q = cute.make_rmem_tensor((self.tq,), Int32)

            g_par = Int32(0)  # parity of the global KV-tile counter

            for w in cutlass.range(trips):
                linear = Int32(bidx) + w * gdim
                tile_idx = num_tiles - Int32(1) - linear
                q0 = tile_idx * self.tq
                ks_min = skv
                ke_max = Int32(0)
                for r in cutlass.range_constexpr(self.tq):
                    ks_min = cutlass.min(ks_min, cutlass.min(Int32(mKS[q0 + r]), skv))
                    ke_max = cutlass.max(ke_max, cutlass.min(Int32(mKE[q0 + r]), skv))
                k_tile0 = ks_min // self.n_block
                span = cutlass.max(ke_max - k_tile0 * self.n_block, Int32(0))
                n_iters = cute.arch.make_warp_uniform(cute.ceil_div(span, self.n_block))
                k_tile0 = cute.arch.make_warp_uniform(k_tile0)

                # stage this block's weights into the WG-private buffer.
                # pre-barrier = WAR guard (a lagging warp of THIS WG may
                # still read the previous block's weights); post = RAW.
                # BF16 weights are up-converted here (exact), so the fp32
                # register math below is dtype-independent.
                cute.arch.barrier(barrier_id=4 + epi_wg, number_of_threads=128)
                wi = tidx_wg
                while wi < self.tq * self.num_heads:
                    sW[epi_wg * (self.tq * self.num_heads) + wi] = Float32(mW_flat[q0 * self.num_heads + wi])
                    wi += 128
                cute.arch.barrier(barrier_id=4 + epi_wg, number_of_threads=128)

                # per-row visibility windows for the -inf mask on
                # out-of-window positions inside swept tiles
                for r in cutlass.range_constexpr(self.tq):
                    ks_q[r] = Int32(mKS[q0 + r])
                    ke_q[r] = Int32(mKE[q0 + r])

                # this WG's KV tiles: local i with (g_par+i) % 2 == epi_wg
                i0 = (g_par + Int32(epi_wg)) % Int32(2)
                n_my = cutlass.max((n_iters - i0 + Int32(1)) // Int32(2), Int32(0))
                for t in cutlass.range(n_my):
                    i = i0 + 2 * t
                    col = (k_tile0 + i) * self.n_block + row
                    # wait this WG's fixed slot (1-bit phase)
                    cute.arch.mbarrier_wait(full_bar, phase)
                    _atomV = cute.make_copy_atom(
                        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(epi_rep)),
                        Float32,
                    )
                    _t2rV = tcgen05.make_tmem_copy(_atomV, tAcc0)
                    # rolling split-LDTM drain: ld c0, then per chunk
                    # {fence; issue the NEXT chunk's LDTM (or release the
                    # slot after the last fence); reduce the current chunk}
                    # — every LDTM after the first flies behind the previous
                    # chunk's FMA chain, and the UMMA warp refills the slot
                    # during the last reduction.
                    cute.copy(_t2rV, tTR_srcs[0], tSrS_a)
                    for qg in cutlass.range_constexpr(self.tq):
                        tSrSv = tSrS_a if qg % 2 == 0 else tSrS_b
                        nSrSv = tSrS_b if qg % 2 == 0 else tSrS_a
                        cute.arch.fence_view_async_tmem_load()
                        if const_expr(qg == self.tq - 1):
                            cute.arch.mbarrier_arrive(empty_bar)
                        else:
                            cute.copy(
                                _t2rV,
                                tTR_srcs[qg + 1],
                                nSrSv,
                            )
                        sW_g = cute.make_tensor(
                            sW_wg + qg * self.num_heads,
                            cute.make_layout((self.num_heads,)),
                        )
                        cute.autovec_copy(sW_g, rW)
                        acc2 = (Float32(0.0), Float32(0.0))
                        for j in cutlass.range_constexpr(0, self.num_heads, 2):
                            v0 = cute.arch.fmax(tSrSv[j], Float32(0.0))
                            v1 = cute.arch.fmax(tSrSv[j + 1], Float32(0.0))
                            acc2 = cute.arch.fma_packed_f32x2(
                                (v0, v1),
                                (rW[j], rW[j + 1]),
                                acc2,
                                rnd="rn",
                            )
                        score = acc2[0] + acc2[1]
                        if const_expr(sm_scale is not None):
                            # sm_scale on the fp32 head-reduced score
                            # (legacy placement: post-reduce, pre-mask)
                            score = score * sm_scale
                        # -inf on out-of-window positions inside the swept
                        # tile (in-window values are untouched by the mask)
                        ksq = Int32(ks_q[qg])
                        keq = Int32(ke_q[qg])
                        masked = Float32(float("-inf"))
                        if (col >= ksq) and (col < keq):
                            masked = score
                        score = masked
                        if col < skv:
                            mOut[q0 + qg, col] = score
                    phase = phase ^ Int32(1)
                g_par = (g_par + n_iters) % Int32(2)

            # all epilogue TMEM reads done before dealloc
            self.epi_sync_barrier.arrive_and_wait()
            if warp_idx == self.epi_warp_ids[0]:
                cute.arch.dealloc_tmem(tmem_base, self.tmem_alloc_cols)
