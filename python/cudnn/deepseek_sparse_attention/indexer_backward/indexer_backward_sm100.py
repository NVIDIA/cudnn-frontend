"""
Indexer Backward — SM100 CuTe-DSL, 3-kernel design.

Three kernels launched sequentially on the same stream:

  Kernel 1 (CuTe DSL): score_grad — compute sum_grad and grad_signal from
      AttnScore and IdxScore, overwrite both Score tensors in-place.
      Unsupported inputs trigger an exception before this stage launches.
  Kernel 2 (CuTe DSL): kernel_gemm — warp-specialized GEMM kernel (below).
      dK is accumulated in float32 via atomicAdd for correctness/perf.
  Kernel 3 (PyTorch):  dk_convert — cast dK from float32 to output dtype
      (same as dQ, dW).

Kernel 2 — Warp specialization (16 warps, 512 threads):
  Warp 0:      Load (Q via TMA, weights)
  Warp 1:      MMA  (3-stage sK pipeline, 2-stage TMEM S/dK: GEMM1 runs 1 block ahead)
  Warps 2-3:   Idle
  Warps 4-7:   Compute warpgroup
               (per-block sGradSignal load, TMEM readback S → dS → dW, dQ TMA store)
  Warps 8-11:  K loading warpgroup (sparse cp.async gather, 3-stage sK)
  Warps 12-15: Reduce warpgroup (TMEM readback dK → atomicAdd to f32 gmem, 2-stage)

TopkIdxs are pre-loaded into SMEM cooperatively by all 512 threads before warp dispatch.
K/dK are flattened in ``__call__`` to a 2D ``(B*S_k, D)`` view so the kernel
indexes them by **global flat KV ids**. ``topk_indices_global=True`` (default,
matches the public fwd convention): ``mTopkIdx`` already carries
``b * seqlen_k + local`` and is loaded directly. ``topk_indices_global=False``:
ids are local-per-batch; the kernel adds ``batch_idx * S_k_per_batch`` to
convert (const_expr-branched). (THD will reuse the same flat-id contract:
``cu_seqlens_k[b] + local`` indexes the ``(T_k, D)`` packed buffer.)
grad_signal (precomputed by kernel 1) is loaded per topk-block by the compute warpgroup.

SMEM (kernel 2): sGradSignal[block_I] replaces the former sAttnScore/sIdxScore/sScratch,
  freeing ~3.5 KB for a larger sTopkIdxs buffer.
TMEM: S0/dK0 @0, dQ @128, S1/dK1 @256 (384/512 cols).

Barriers for kernel 2:
  mbar[0-1]:  S_full_0/1     (MMA commits after GEMM1  → Compute waits)
  mbar[2-3]:  dS_ready_0/1   (Compute arrives after dS  → MMA waits)
  mbar[4-5]:  dK_full_0/1    (MMA commits after GEMM2  → Reduce waits)
  mbar[6-7]:  dK_empty_0/1   (Reduce arrives            → MMA waits)
  mbar[8-10]: K_loaded_0/1/2  (K-load arrives            → MMA waits, 3-stage)
  mbar[11-13]:K_consumed_0/1/2(MMA commits after GEMM3  → K-load waits, 3-stage)
  mbar[14]:   W_loaded        (Load arrives              → Compute waits)

Each warp/warpgroup has its own independent loop, communicating via barriers.
No CLC persistent scheduling (simple grid = batch × seqlen).
"""

from __future__ import annotations

import math
from functools import partial
import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.utils.blackwell_helpers import (
    make_trivial_tiled_mma as _make_trivial_tiled_mma,
    make_smem_layout_a as _make_smem_layout_a,
    make_smem_layout_b as _make_smem_layout_b,
    make_smem_layout_epi as _make_smem_layout_epi,
)
from cutlass.utils.layout import LayoutEnum

import cutlass.utils.blackwell_helpers as sm100_utils_basic

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
)

mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")


# Barrier indices for kernel_gemm — per-stage barriers for S_FULL, DS_READY, K_LOADED
# to avoid phase-wrap when producer runs 2 blocks ahead of consumer.
# sK uses 3-stage pipeline (Opt-7): K_LOADED and K_CONSUMED are per-stage (×3).
MBAR_S_FULL_0 = 0
MBAR_S_FULL_1 = 1
MBAR_DS_READY_0 = 2
MBAR_DS_READY_1 = 3
MBAR_DK_FULL_0 = 4
MBAR_DK_FULL_1 = 5
MBAR_DK_EMPTY_0 = 6
MBAR_DK_EMPTY_1 = 7
MBAR_K_LOADED_0 = 8
MBAR_K_LOADED_1 = 9
MBAR_K_LOADED_2 = 10
MBAR_K_CONSUMED_0 = 11
MBAR_K_CONSUMED_1 = 12
MBAR_K_CONSUMED_2 = 13
MBAR_W_LOADED = 14
MBAR_DQ_DONE = 15
NUM_BARRIERS = 16

CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)

_score_grad_cute_cache: dict = {}


class IndexerBackwardSm100:
    arch = 100
    WARP_SIZE = 32
    WARPGROUP_SIZE = 128
    NUM_WARPS = 16
    THREADS_PER_CTA = 512

    # Warp assignments
    load_warp_id = 0
    mma_warp_id = 1
    # Warps 2-3: idle
    compute_warp_id = (4, 5, 6, 7)
    k_load_warp_id = (8, 9, 10, 11)
    reduce_warp_id = (12, 13, 14, 15)

    def __init__(self, head_dim, heads=64, block_I=128, topk=512, topk_indices_global: bool = True):
        self.head_dim = head_dim
        self.heads = heads
        self.block_I = block_I
        self.topk = topk
        # When True (default, matches the public fwd convention), mTopkIdx
        # carries global KV ids (``b * seqlen_k + local``); the kernel uses
        # them as flat ids into the (B*S_k, D) K/dK view directly. When
        # False, mTopkIdx carries local-per-batch ids and the kernel adds
        # ``batch_idx * S_k_per_batch`` to convert. Const_expr-branched.
        self.topk_indices_global = topk_indices_global
        assert heads >= 64
        assert topk % block_I == 0
        self.num_topk_blocks = topk // block_I

        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.heads_padded = int(math.ceil(heads / 8) * 8)

        # GEMM tilers (M, N, K) — cute.gemm, SMEM operands, TMEM acc
        # GEMM1: S[H,TileN] = Q[H,D] @ K[TileN,D].  A=Q K-major, B=K K-major
        self.gemm1_tiler = (self.heads_padded, self.block_I, self.head_dim_padded)
        # GEMM2 (SwapAB): dK[TileN,D] = dS[TileN,H] @ Q[D,H].  A=dS MN-major, B=Q MN-major
        self.gemm2_tiler = (self.block_I, self.head_dim_padded, self.heads_padded)
        # GEMM3: dQ[H,D] += dS[H,TileN] @ K^T[D,TileN].  A=dS K-major, B=K^T MN-major
        self.gemm3_tiler = (self.heads_padded, self.head_dim_padded, self.block_I)

        self.acc_dtype = Float32

        # TMEM layout (2-stage S/dK for software pipeline):
        #   S0/dK0(fp32): offset 0,   128 cols (even blocks)
        #   dQ(fp32):     offset 128, 128 cols (accumulated across iterations)
        #   S1/dK1(fp32): offset 256, 128 cols (odd blocks)
        #   Total: 384 <= 512
        self.tmem_s0_offset = 0
        self.tmem_dq_offset = 128
        self.tmem_s1_offset = 256
        self.tmem_alloc_cols = 512

        # Register budgets — must sum to 512 per thread (65536 regs / 128 threads per WG)
        # Compute needs 128+ (tSrS=64 + dw_accum=64), Reduce needs 128+ (tDKrDK=128)
        self.num_regs_wg0 = 40
        self.num_regs_compute = 200
        self.num_regs_reduce = 200
        self.num_regs_kload = 32

        self.buffer_align_bytes = 1024

        # TMA config
        self.cluster_shape = (1, 1, 1, 1)
        self.Q_mbar_size = 2  # PipelineTmaUmma with 1 stage
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.WARPGROUP_SIZE,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.WARP_SIZE + 2 * self.WARPGROUP_SIZE,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mW: cute.Tensor,
        mK: cute.Tensor,
        mdQ: cute.Tensor,
        mdW: cute.Tensor,
        mdK_f32: cute.Tensor,
        mGradSignal: cute.Tensor,
        mTopkIdx: cute.Tensor,
        sm_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type

        # Q/W/dQ/dW/GradSignal/TopkIdx: transpose (bs, seqlen, ...) → (seqlen, ..., bs)
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 3, 0]))
        mW = cute.make_tensor(mW.iterator, cute.select(mW.layout, mode=[1, 2, 0]))
        mdQ = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 2, 3, 0]))
        mdW = cute.make_tensor(mdW.iterator, cute.select(mdW.layout, mode=[1, 2, 0]))
        mGradSignal = cute.make_tensor(mGradSignal.iterator, cute.select(mGradSignal.layout, mode=[1, 2, 0]))
        mTopkIdx = cute.make_tensor(mTopkIdx.iterator, cute.select(mTopkIdx.layout, mode=[1, 2, 0]))

        # K/dK: flatten (B, S_k, D) → (B*S_k, D). topk ids are global flat KV
        # positions (b*S_k + local), so they index the flat view directly with
        # no per-batch offset round-trip. Free reshape: BSHD is contiguous so
        # stride collapses to (D, 1).
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (cute.size(mK.shape[0]) * cute.size(mK.shape[1]), cute.size(mK.shape[2])),
                stride=(cute.size(mK.shape[2]), 1),
            ),
        )
        mdK_f32 = cute.make_tensor(
            mdK_f32.iterator,
            cute.make_layout(
                (cute.size(mdK_f32.shape[0]) * cute.size(mdK_f32.shape[1]), cute.size(mdK_f32.shape[2])),
                stride=(cute.size(mdK_f32.shape[2]), 1),
            ),
        )

        cta_group = tcgen05.CtaGroup.ONE

        # All GEMMs: SS path (A & B from SMEM, accumulator in TMEM)
        tmma1 = _make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.gemm1_tiler[:2],
        )
        tmma2 = _make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.gemm2_tiler[:2],
        )
        tmma3 = _make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.gemm3_tiler[:2],
        )

        # SMEM layouts — primary views
        # sK/sKt: 3-stage pipeline (Opt-7) for hiding K-load scatter latency.
        # sdS: 2-stage pipeline (tied to TMEM S/dK 2-stage).
        sQ_layout = _make_smem_layout_a(tmma1, self.gemm1_tiler, self.q_dtype, 1)
        sK_layout = _make_smem_layout_b(tmma1, self.gemm1_tiler, self.k_dtype, 3)
        sdS_layout = _make_smem_layout_a(tmma3, self.gemm3_tiler, self.q_dtype, 2)
        # Epilogue-style store layout for stmatrix writes to sdS (same physical SMEM).
        # COL_MAJOR (M-major) + square tile → physically compatible with A-operand layout.
        sdS_store_layout = _make_smem_layout_epi(
            self.q_dtype,
            LayoutEnum.COL_MAJOR,
            (self.heads_padded, self.block_I),
            2,
        )
        # SwapAB GEMM2: A=dS (2-stage, from sdS SMEM), B=Q (1-stage, from sQ SMEM)
        sdS_g2a_layout = _make_smem_layout_a(tmma2, self.gemm2_tiler, self.q_dtype, 2)
        sKt_layout = _make_smem_layout_b(tmma3, self.gemm3_tiler, self.k_dtype, 3)
        sQ_g2b_layout = _make_smem_layout_b(tmma2, self.gemm2_tiler, self.q_dtype, 1)

        # --- TMA atoms ---
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        cluster_layout_vmnk = cute.make_layout(self.cluster_shape)

        # TMA Q load (A-operand for GEMM1: Q[H, D] K-major)
        # make_tiled_tma_atom_A tiles first 2 tensor modes with (M, K)
        # So reorder mQ to (heads, dim, seqlen, batch) → mode 0=H=64, mode 1=D=128
        mQ_tma = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0, 3]))
        Q_smem_layout_tma = cute.select(sQ_layout, mode=[0, 1, 2])
        tma_atom_Q, mQ_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ_tma,
            Q_smem_layout_tma,
            self.gemm1_tiler,
            tmma1,
            cluster_layout_vmnk.shape,
        )
        self.tma_copy_Q_bytes = cute.size_in_bytes(self.q_dtype, Q_smem_layout_tma)

        # Epilogue SMEM layout for dQ store (bf16, row-major = D contiguous)
        sdQ_epi_layout = _make_smem_layout_epi(
            self.q_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.heads_padded, self.head_dim_padded),
            1,
        )

        # TMA dQ store — reorder mdQ so tiled dims (heads, dim) come first
        mdQ_tma = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 2, 0, 3]))
        sdQ_epi_smem_layout = cute.select(sdQ_epi_layout, mode=[0, 1])
        tma_atom_dQ, mdQ_tma = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ_tma,
            sdQ_epi_smem_layout,
            (self.heads_padded, self.head_dim_padded),
        )

        seqlen = cute.size(mQ.shape[0])
        batch_size = cute.size(mQ.shape[3]) if cute.rank(mQ.shape) > 3 else 1

        self.kernel_gemm(
            mQ_tma,
            mW,
            mK,
            mdQ_tma,
            mdW,
            mdK_f32,
            mGradSignal,
            mTopkIdx,
            sm_scale,
            tmma1,
            tmma2,
            tmma3,
            sQ_layout,
            sdS_g2a_layout,
            sK_layout,
            sKt_layout,
            sdS_layout,
            sQ_g2b_layout,
            sdS_store_layout,
            tma_atom_Q,
            tma_atom_dQ,
            sdQ_epi_layout,
            seqlen,
            batch_size,
        ).launch(
            grid=(batch_size, seqlen, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel_gemm(
        self,
        mQ,
        mW,
        mK,
        mdQ,
        mdW,
        mdK_f32,
        mGradSignal,
        mTopkIdx,
        sm_scale: Float32 | float,
        tmma1,
        tmma2,
        tmma3,
        sQ_layout,
        sdS_g2a_layout,
        sK_layout,
        sKt_layout,
        sdS_layout,
        sQ_g2b_layout,
        sdS_store_layout,
        tma_atom_Q,
        tma_atom_dQ,
        sdQ_epi_layout,
        seqlen: Int32,
        batch_size: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        batch_idx = cute.arch.block_idx()[0]
        seq_idx = cute.arch.block_idx()[1]
        seqlen_k = cute.size(mK.shape[0])

        # TMA descriptor prefetch (load warp only)
        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dQ)

        # SMEM allocation
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sdS_size = cute.cosize(sdS_layout)

        # Compute sTopkIdxs capacity from remaining SMEM (dsa-next pattern)
        def _align_up(x, a):
            return (x + a - 1) // a * a

        _elem_bytes = self.q_dtype.width // 8
        _tma_align = self.buffer_align_bytes  # 1024
        _non_tma_align = 128

        _offset = 0
        _offset += self.Q_mbar_size * 8  # Q_mbar: Int64 × Q_mbar_size
        _offset += NUM_BARRIERS * 8  # mbar: Int64 × NUM_BARRIERS
        _offset += 4  # tmem_holding_buf: Int32
        _offset = _align_up(_offset, _tma_align)
        _offset += int(sQ_size) * _elem_bytes  # sQ
        _offset = _align_up(_offset, _tma_align)
        _offset += int(sK_size) * _elem_bytes  # sK
        _offset = _align_up(_offset, _tma_align)
        _offset += int(sdS_size) * _elem_bytes  # sdS
        _offset = _align_up(_offset, _non_tma_align)
        _offset += self.topk * 4  # sGradSignal: Float32 × topk
        _offset = _align_up(_offset, _non_tma_align)
        # sTopkIdxs goes here — compute remaining space
        _topk_idx_offset = _offset
        # Account for sW that comes after sTopkIdxs,
        # with worst-case alignment padding.
        _tail = 0
        _tail += _non_tma_align  # worst-case align padding before sW
        _tail += self.heads * _elem_bytes  # sW
        _max_smem_bytes = 227 * 1024
        smem_topk_capacity = (_max_smem_bytes - _topk_idx_offset - _tail) // 4

        @cute.struct
        class SharedStorage:
            Q_mbar: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            mbar: cute.struct.MemRange[cutlass.Int64, NUM_BARRIERS]
            tmem_holding_buf: Int32
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]
            sdS: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sdS_size], self.buffer_align_bytes]
            sGradSignal: cute.struct.Align[cute.struct.MemRange[Float32, self.topk], 128]
            sTopkIdxs: cute.struct.Align[cute.struct.MemRange[Int32, smem_topk_capacity], 128]
            sW: cute.struct.Align[cute.struct.MemRange[self.q_dtype, self.heads], 128]

        assert SharedStorage.size_in_bytes() <= _max_smem_bytes, (
            f"SharedStorage ({SharedStorage.size_in_bytes()} bytes) exceeds {_max_smem_bytes} bytes (227KB), " f"smem_topk_capacity={smem_topk_capacity}"
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        Q_mbar_ptr = storage.Q_mbar.data_ptr()
        mbar = storage.mbar.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        # Swizzled SMEM tensors
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
        sdS_store = storage.sdS.get_tensor(sdS_store_layout.outer, swizzle=sdS_store_layout.inner)
        # Recast views for transposed / SwapAB operands
        # SwapAB GEMM2: A=dS (recast sdS as A-operand), B=Q (recast sQ as B-operand)
        sdS_g2a = cute.make_tensor(cute.recast_ptr(sdS.iterator, sdS_g2a_layout.inner), sdS_g2a_layout.outer)
        sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sQ_g2b = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQ_g2b_layout.inner), sQ_g2b_layout.outer)

        sGradSignal = storage.sGradSignal.get_tensor(cute.make_layout((self.topk,), stride=(1,)))
        sTopkIdxs = storage.sTopkIdxs.get_tensor(cute.make_layout((smem_topk_capacity,), stride=(1,)))
        sW = storage.sW.get_tensor(cute.make_layout((self.heads,), stride=(1,)))

        # dQ epilogue SMEM — reuses sK physical memory (safe: dQ store happens after all iterations)
        sdQ_epi = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sdQ_epi_layout.inner),
            sdQ_epi_layout.outer,
        )

        # --- Q TMA load: partition (dsa-next pattern) ---
        Q_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cute.make_layout(self.cluster_shape),
        )
        Q_producer, Q_consumer = Q_pipeline.make_participants()

        # local_tile: tile dims use None (keep all tiles), L dims use runtime index
        gQ = cute.local_tile(
            mQ,
            cute.select(self.gemm1_tiler, mode=[0, 2]),
            (None, None, seq_idx, batch_idx),
        )
        # partition_A → tma_partition (dsa-next pattern)
        gemm1_thr_mma = tmma1.get_slice(0)
        tAgQ = gemm1_thr_mma.partition_A(gQ)
        tQsQ, tQgQ_mkl = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tAgQ, 0, 3),
        )

        # --- dQ TMA store: partition (dsa-next pattern) ---
        dQ_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARPGROUP_SIZE,
            ),
        )

        # local_tile: concrete tile indices (only 1 tile each), runtime L indices
        gdQ = cute.local_tile(
            mdQ,
            (self.heads_padded, self.head_dim_padded),
            (0, 0, seq_idx, batch_idx),
        )
        sdQ_epi_slice = sdQ_epi[None, None, 0]
        tdQsdQ, tdQgdQ_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_epi_slice, 0, 2),
            cute.group_modes(gdQ, 0, 2),
        )

        # Init all barriers (warp 0)
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DS_READY_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DS_READY_1, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_EMPTY_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DK_EMPTY_1, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_2, 1)
            cute.arch.mbarrier_init(mbar + MBAR_W_LOADED, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DQ_DONE, 1)
        cute.arch.sync_threads()

        # Pre-load topk indices into SMEM cooperatively (all 512 threads).
        # Load up to smem_topk_capacity; reads beyond that fall back to global memory.
        # K/dK are flattened to (B*S_k, D) above, so consumers index by global
        # flat KV ids. ``topk_indices_global=True`` (default): ``mTopkIdx`` already
        # carries global ids (``b * seqlen_k + local``); load directly.
        # ``topk_indices_global=False``: ids are local-per-batch; add
        # ``batch_idx * S_k_per_batch`` to convert. Invalid (-1) entries stay
        # negative (skipped in the local→global add) and are rejected by the
        # ``>= 0`` bounds check at consumers.
        batch_offset_l2g = Int32(0) if const_expr(self.topk_indices_global) else batch_idx * (seqlen_k // batch_size)
        _load_bound = const_expr(min(self.topk, smem_topk_capacity))
        TOPK_PER_THREAD = const_expr((_load_bound + self.THREADS_PER_CTA - 1) // self.THREADS_PER_CTA)
        for ii in cutlass.range_constexpr(TOPK_PER_THREAD):
            pos = ii * self.THREADS_PER_CTA + tidx
            if pos < _load_bound:
                raw_id = Int32(mTopkIdx[seq_idx, pos, batch_idx])
                if const_expr(self.topk_indices_global):
                    sTopkIdxs[pos] = raw_id
                else:
                    sTopkIdxs[pos] = raw_id + batch_offset_l2g if raw_id >= 0 else raw_id
        cute.arch.sync_threads()

        # Pre-compute accumulator shapes/layouts from tmma before dispatch,
        # so branches that don't run _mma_warp never touch the tmma objects
        # (avoids MLIR SSA domination issues from tmma.set() inside _mma_warp).
        s_acc_shape = tmma1.partition_shape_C(self.gemm1_tiler[:2])
        s_acc_layout = tmma1.make_fragment_C(s_acc_shape).layout
        dq_acc_shape = tmma3.partition_shape_C(self.gemm3_tiler[:2])
        dq_acc_layout = tmma3.make_fragment_C(dq_acc_shape).layout
        dk_acc_shape = tmma2.partition_shape_C(self.gemm2_tiler[:2])
        dk_acc_layout = tmma2.make_fragment_C(dk_acc_shape).layout

        # =============================================================
        # Warp dispatch — setmaxnreg rebalances registers across WGs.
        # =============================================================
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)
            self._load_warp(
                mW,
                mGradSignal,
                sW,
                sGradSignal,
                tma_atom_Q,
                tQsQ,
                tQgQ_mkl,
                Q_producer,
                seq_idx,
                batch_idx,
                tidx,
                mbar,
            )

        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS_0, tStS_1, tDqDq, tDkDk_0, tDkDk_1 = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            self._mma_warp(
                sQ,
                sdS_g2a,
                sK,
                sKt,
                sdS,
                sQ_g2b,
                tmma1,
                tmma2,
                tmma3,
                tStS_0,
                tStS_1,
                tDqDq,
                tDkDk_0,
                tDkDk_1,
                Q_consumer,
                mbar,
            )

        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS_0, tStS_1, tDqDq, tDkDk_0, tDkDk_1 = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            self._compute_warpgroup(
                mdW,
                sGradSignal,
                sW,
                sdS_store,
                sdS,
                sdQ_epi_slice,
                s_acc_shape,
                dq_acc_shape,
                tStS_0,
                tStS_1,
                tDqDq,
                tma_atom_dQ,
                tdQsdQ,
                tdQgdQ_mkl,
                dQ_store_pipeline,
                sm_scale,
                seq_idx,
                batch_idx,
                tidx,
                warp_idx,
                mbar,
            )
            if warp_idx == self.compute_warp_id[0]:
                cute.arch.dealloc_tmem(tmem_ptr_base, self.tmem_alloc_cols)

        elif warp_idx in self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_kload)
            self._k_load_warpgroup(
                mK,
                sK,
                sTopkIdxs,
                mTopkIdx,
                seq_idx,
                batch_idx,
                seqlen_k,
                batch_size,
                tidx,
                mbar,
            )

        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS_0, tStS_1, tDqDq, tDkDk_0, tDkDk_1 = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            self._reduce_warpgroup(
                mdK_f32,
                sTopkIdxs,
                mTopkIdx,
                dk_acc_shape,
                tDkDk_0,
                tDkDk_1,
                sm_scale,
                seq_idx,
                batch_idx,
                seqlen_k,
                batch_size,
                tidx,
                mbar,
            )

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)

    # =========================================================================
    # Warp 0: Load warp
    # =========================================================================
    @cute.jit
    def _load_warp(
        self,
        mW,
        mGradSignal,
        sW,
        sGradSignal,
        tma_atom_Q,
        tQsQ,
        tQgQ_mkl,
        Q_producer,
        seq_idx,
        batch_idx,
        tidx,
        mbar,
    ):
        """Load warp: TMA Q load once, loads W and grad_signal once."""
        lane_id = tidx % self.WARP_SIZE

        # Load grad_signal[topk] to sGradSignal
        GS_PER_THREAD = const_expr((self.topk + self.WARP_SIZE - 1) // self.WARP_SIZE)
        for si in cutlass.range_constexpr(GS_PER_THREAD):
            pos = si * self.WARP_SIZE + lane_id
            if pos < self.topk:
                sGradSignal[pos] = mGradSignal[seq_idx, pos, batch_idx]

        # Load W[heads] to sW
        W_PER_THREAD = const_expr((self.heads + self.WARP_SIZE - 1) // self.WARP_SIZE)
        for wi in cutlass.range_constexpr(W_PER_THREAD):
            idx = wi * self.WARP_SIZE + lane_id
            if idx < self.heads:
                sW[idx] = mW[seq_idx, idx, batch_idx]

        cute.arch.fence_view_async_shared()
        # Signal W + grad_signal loaded for compute warpgroup
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar + MBAR_W_LOADED)

        # --- TMA Q load (dsa-next pattern: cute.copy with pre-partitioned tensors) ---
        Q_producer.reset()
        handle_Q = Q_producer.acquire_and_advance()
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[None, 0, 0],  # global: all atom data, RestM=0, RestK=0
            tQsQ[None, 0],  # SMEM: all atom data, stage=0
            tma_bar_ptr=handle_Q.barrier,
        )

    # =========================================================================
    # Warp 1: MMA warp (3-stage sK pipeline, 2-stage TMEM S/dK)
    # =========================================================================
    @cute.jit
    def _mma_warp(
        self,
        sQ,
        sdS_g2a,
        sK,
        sKt,
        sdS,
        sQ_g2b,
        tmma1,
        tmma2,
        tmma3,
        tStS_0,
        tStS_1,
        tDqDq,
        tDkDk_0,
        tDkDk_1,
        Q_consumer,
        mbar,
    ):
        """MMA warp: 3-stage sK pipeline (Opt-7), 2-stage TMEM S/dK.

        Structure: Prologue(Fill[0]) → Main(Fill[bi]+Drain[bi-1]) → Epilogue(Drain[last])
        GEMM1(S) runs 1 block ahead, hiding Compute latency behind the next GEMM1.

        sK uses 3-stage pipeline (bi%3) to hide K-load scatter gather latency.
        TMEM S/dK accumulators remain 2-stage (bi%2).
        K_CONSUMED is per-sK-stage (3 barriers) so K-load can run 3 blocks ahead.
        """
        Q_consumer.reset()
        Q_consumer.wait_and_advance()

        # --- A/B fragments from SMEM ---
        # sK/sKt: 3-stage (stage dim = last dim), sdS: 2-stage, sQ: 1-stage
        tSrQ = tmma1.make_fragment_A(sQ)
        tSrK = tmma1.make_fragment_B(sK)  # 3-stage
        tDKrA_g2 = tmma2.make_fragment_A(sdS_g2a)  # SwapAB: A=dS, 2-stage
        tDKrB_g2 = tmma2.make_fragment_B(sQ_g2b)  # SwapAB: B=Q, 1-stage
        tDQrDS = tmma3.make_fragment_A(sdS)  # 2-stage
        tDQrKt = tmma3.make_fragment_B(sKt)  # 3-stage

        dk_empty_0_phase = Int32(0)
        dk_empty_1_phase = Int32(0)
        ds_ready_0_phase = Int32(0)
        ds_ready_1_phase = Int32(0)
        k_loaded_0_phase = Int32(0)
        k_loaded_1_phase = Int32(0)
        k_loaded_2_phase = Int32(0)
        is_first_dq = True

        # =============================================================
        # Prologue: Fill block 0 (sK stage 0, TMEM stage 0)
        # =============================================================
        cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_0, k_loaded_0_phase)
        k_loaded_0_phase ^= 1

        tmma1.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
            cute.gemm(tmma1, tStS_0, tSrQ[None, None, k_block, 0], tSrK[None, None, k_block, 0], tStS_0)
            tmma1.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + MBAR_S_FULL_0)

        # =============================================================
        # Main loop: bi = 1 .. num_topk_blocks-1
        #   Fill[bi]:    GEMM1(S) → TMEM_S[bi%2], reads sK[bi%3]
        #   Drain[bi-1]: wait dS → GEMM2(dK) → GEMM3(dQ) using sKt[(bi-1)%3]
        # =============================================================
        for bi_offset in cutlass.range_constexpr(self.num_topk_blocks - 1):
            bi = bi_offset + 1

            # ------ Fill[bi]: GEMM1 for current block ------
            # DK_EMPTY: wait for TMEM slot reuse (2-stage, bi%2)
            if bi >= 2:
                if bi % 2 == 0:
                    cute.arch.mbarrier_wait(mbar + MBAR_DK_EMPTY_0, dk_empty_0_phase)
                    dk_empty_0_phase ^= 1
                else:
                    cute.arch.mbarrier_wait(mbar + MBAR_DK_EMPTY_1, dk_empty_1_phase)
                    dk_empty_1_phase ^= 1

            # K_LOADED: wait for sK data (3-stage, bi%3)
            if bi % 3 == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_0, k_loaded_0_phase)
                k_loaded_0_phase ^= 1
            elif bi % 3 == 1:
                cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_1, k_loaded_1_phase)
                k_loaded_1_phase ^= 1
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_2, k_loaded_2_phase)
                k_loaded_2_phase ^= 1

            # GEMM1: tStS[bi%2] = Q @ sK[bi%3]
            tmma1.set(tcgen05.Field.ACCUMULATE, False)
            if bi % 2 == 0:
                for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                    cute.gemm(tmma1, tStS_0, tSrQ[None, None, k_block, 0], tSrK[None, None, k_block, bi % 3], tStS_0)
                    tmma1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_S_FULL_0)
            else:
                for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                    cute.gemm(tmma1, tStS_1, tSrQ[None, None, k_block, 0], tSrK[None, None, k_block, bi % 3], tStS_1)
                    tmma1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_S_FULL_1)

            # ------ Drain[bi-1]: GEMM2(dK) + GEMM3(dQ) for previous block ------
            if (bi - 1) % 2 == 0:
                # Prev TMEM stage 0
                cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_0, ds_ready_0_phase)
                ds_ready_0_phase ^= 1

                tmma2.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                    cute.gemm(tmma2, tDkDk_0, tDKrA_g2[None, None, k_block, 0], tDKrB_g2[None, None, k_block, 0], tDkDk_0)
                    tmma2.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_DK_FULL_0)

                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                is_first_dq = False
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll=4):
                    cute.gemm(tmma3, tDqDq, tDQrDS[None, None, k_block, 0], tDQrKt[None, None, k_block, (bi - 1) % 3], tDqDq)
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)
                # K_CONSUMED: release sK stage (bi-1)%3
                if (bi - 1) % 3 == 0:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MBAR_K_CONSUMED_0)
                elif (bi - 1) % 3 == 1:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MBAR_K_CONSUMED_1)
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MBAR_K_CONSUMED_2)
            else:
                # Prev TMEM stage 1
                cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_1, ds_ready_1_phase)
                ds_ready_1_phase ^= 1

                tmma2.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                    cute.gemm(tmma2, tDkDk_1, tDKrA_g2[None, None, k_block, 1], tDKrB_g2[None, None, k_block, 0], tDkDk_1)
                    tmma2.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_DK_FULL_1)

                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                is_first_dq = False
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll=4):
                    cute.gemm(tmma3, tDqDq, tDQrDS[None, None, k_block, 1], tDQrKt[None, None, k_block, (bi - 1) % 3], tDqDq)
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)
                # K_CONSUMED: release sK stage (bi-1)%3
                if (bi - 1) % 3 == 0:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MBAR_K_CONSUMED_0)
                elif (bi - 1) % 3 == 1:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MBAR_K_CONSUMED_1)
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + MBAR_K_CONSUMED_2)

        # =============================================================
        # Epilogue: Drain last block
        # TMEM stage: (num_topk_blocks-1)%2, sK stage: (num_topk_blocks-1)%3
        # =============================================================
        LAST_TMEM_STAGE = const_expr((self.num_topk_blocks - 1) % 2)
        LAST_SK_STAGE = const_expr((self.num_topk_blocks - 1) % 3)
        if LAST_TMEM_STAGE == 0:
            cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_0, ds_ready_0_phase)
            tmma2.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                cute.gemm(tmma2, tDkDk_0, tDKrA_g2[None, None, k_block, 0], tDKrB_g2[None, None, k_block, 0], tDkDk_0)
                tmma2.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DK_FULL_0)
            tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
            for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll=4):
                cute.gemm(tmma3, tDqDq, tDQrDS[None, None, k_block, 0], tDQrKt[None, None, k_block, LAST_SK_STAGE], tDqDq)
                tmma3.set(tcgen05.Field.ACCUMULATE, True)
            if LAST_SK_STAGE == 0:
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_K_CONSUMED_0)
            elif LAST_SK_STAGE == 1:
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_K_CONSUMED_1)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_K_CONSUMED_2)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DQ_DONE)
        else:
            cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_1, ds_ready_1_phase)
            tmma2.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                cute.gemm(tmma2, tDkDk_1, tDKrA_g2[None, None, k_block, 1], tDKrB_g2[None, None, k_block, 0], tDkDk_1)
                tmma2.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DK_FULL_1)
            tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
            for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll=4):
                cute.gemm(tmma3, tDqDq, tDQrDS[None, None, k_block, 1], tDQrKt[None, None, k_block, LAST_SK_STAGE], tDqDq)
                tmma3.set(tcgen05.Field.ACCUMULATE, True)
            if LAST_SK_STAGE == 0:
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_K_CONSUMED_0)
            elif LAST_SK_STAGE == 1:
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_K_CONSUMED_1)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_K_CONSUMED_2)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DQ_DONE)

    # =========================================================================
    # Warps 4-7: Compute/Epilogue warpgroup (2-stage S readback + sdS write)
    # =========================================================================
    @cute.jit
    def _compute_warpgroup(
        self,
        mdW,
        sGradSignal,
        sW,
        sdS_store,
        sdS,
        sdQ_epi_slice,
        s_acc_shape,
        dq_acc_shape,
        tStS_0,
        tStS_1,
        tDqDq,
        tma_atom_dQ,
        tdQsdQ,
        tdQgdQ_mkl,
        dQ_store_pipeline,
        sm_scale: Float32 | float,
        seq_idx,
        batch_idx,
        tidx,
        warp_idx,
        mbar,
    ):
        """Compute/Epilogue warpgroup: TMEM readback S → register dS → stmatrix sdS, dQ/dW output.

        Optimizations (per dsa-next pattern):
          - TMEM reduced to 1-mode for flat register layout
          - 2D identity for scalar (h, n) coordinates: mode=[0] → h, mode=[1] → n
          - sW pre-loaded into bf16 register array (scalar h enables rW[h])
          - Paired f32x2 packed ops (fma_packed_f32x2 for dW accumulation)
          - stmatrix bulk store (8 warp-level instructions vs 32+ scalar STS per thread)
          - Compute and store phases fully separated
        """
        wg_tidx = tidx % self.WARPGROUP_SIZE
        warp_id_in_wg = wg_tidx // self.WARP_SIZE
        lane_id = wg_tidx % self.WARP_SIZE
        compute_warp0 = Int32(self.compute_warp_id[0])

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )

        # --- TMEM readback (keep original partitioning for coordinate fidelity) ---
        tiled_tmem_load_s_0 = tcgen05.make_tmem_copy(tmem_load_atom, tStS_0)
        thr_tmem_load_s_0 = tiled_tmem_load_s_0.get_slice(wg_tidx)
        tStS_t2r_0 = thr_tmem_load_s_0.partition_S(tStS_0)

        tiled_tmem_load_s_1 = tcgen05.make_tmem_copy(tmem_load_atom, tStS_1)
        thr_tmem_load_s_1 = tiled_tmem_load_s_1.get_slice(wg_tidx)
        tStS_t2r_1 = thr_tmem_load_s_1.partition_S(tStS_1)

        # Logical GEMM views for direct dS writes (stage 0/1).
        sdS_gemm_view_0 = cute.composition(
            sdS[None, None, None, 0],
            cute.make_layout((self.heads_padded, self.block_I)),
        )
        sdS_gemm_view_1 = cute.composition(
            sdS[None, None, None, 1],
            cute.make_layout((self.heads_padded, self.block_I)),
        )

        # Coordinate map matched to the same TMEM load partition used by tSrS.
        cS = cute.make_identity_tensor(s_acc_shape)
        tCcS = thr_tmem_load_s_0.partition_D(cS)
        tSrS_shape = tCcS.shape

        # --- TMEM readback (dQ — original 3-mode, NOT reduced) ---
        tiled_tmem_load_dq = tcgen05.make_tmem_copy(tmem_load_atom, tDqDq)
        thr_tmem_load_dq = tiled_tmem_load_dq.get_slice(wg_tidx)
        tDqDq_t2r = thr_tmem_load_dq.partition_S(tDqDq)
        cDQ = cute.make_identity_tensor(dq_acc_shape)
        tCcDQ = thr_tmem_load_dq.partition_D(cDQ)
        tDQrDQ_shape = tCcDQ.shape

        # ---- Wait for W loaded by load warp ----
        cute.arch.mbarrier_wait(mbar + MBAR_W_LOADED, Int32(0))

        # ---- Per topk-block iteration (2-stage S/dS) ----
        s_full_0_phase = Int32(0)
        s_full_1_phase = Int32(0)

        dw_accum = cute.make_rmem_tensor(tSrS_shape, Float32)
        for ei in cutlass.range_constexpr(cute.size(dw_accum)):
            dw_accum[ei] = Float32(0.0)

        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        for bi in cutlass.range(0, self.num_topk_blocks):
            i_st = bi * self.block_I

            # Wait for S ready from MMA (per-stage barrier)
            if bi % 2 == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_S_FULL_0, s_full_0_phase)
                s_full_0_phase ^= 1
                cute.copy(tiled_tmem_load_s_0, tStS_t2r_0, tSrS)
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_S_FULL_1, s_full_1_phase)
                s_full_1_phase ^= 1
                cute.copy(tiled_tmem_load_s_1, tStS_t2r_1, tSrS)

            # Phase 1: Compute dS (→ tSrS), accumulate dW — paired f32x2.
            for ei in cutlass.range(0, cute.size(tSrS), 2):
                h0 = cute.get(tCcS[ei], mode=[0, 0])
                n0 = cute.get(tCcS[ei], mode=[0, 1])
                h1 = cute.get(tCcS[ei + 1], mode=[0, 0])
                n1 = cute.get(tCcS[ei + 1], mode=[0, 1])

                tSrS[ei], tSrS[ei + 1] = mul_packed_f32x2(
                    (tSrS[ei], tSrS[ei + 1]),
                    (Float32(sm_scale), Float32(sm_scale)),
                )
                s0 = tSrS[ei]
                s1 = tSrS[ei + 1]

                w0 = Float32(sW[h0])
                w1 = Float32(sW[h1])
                gs0 = sGradSignal[i_st + n0]
                gs1 = sGradSignal[i_st + n1]

                s_pos_0 = s0 > Float32(0.0)
                s_pos_1 = s1 > Float32(0.0)
                relu_s0 = s0 if s_pos_0 else Float32(0.0)
                relu_s1 = s1 if s_pos_1 else Float32(0.0)

                dw_accum[ei], dw_accum[ei + 1] = fma_packed_f32x2(
                    (gs0, gs1),
                    (relu_s0, relu_s1),
                    (dw_accum[ei], dw_accum[ei + 1]),
                )

                tSrS[ei] = gs0 * w0 if s_pos_0 else Float32(0.0)
                tSrS[ei + 1] = gs1 * w1 if s_pos_1 else Float32(0.0)

            cute.arch.fence_view_async_tmem_load()

            # Phase 2: Convert dS f32→bf16, write to sdS via coordinate mapping.
            tSrS_f16 = cute.make_rmem_tensor(tSrS.shape, self.q_dtype)
            for ei in cutlass.range_constexpr(cute.size(tSrS)):
                tSrS_f16[ei] = self.q_dtype(tSrS[ei])

            if bi % 2 == 0:
                for ei in cutlass.range_constexpr(cute.size(tSrS_f16)):
                    h = cute.get(tCcS[ei], mode=[0, 0])
                    n = cute.get(tCcS[ei], mode=[0, 1])
                    sdS_gemm_view_0[h, n] = tSrS_f16[ei]
            else:
                for ei in cutlass.range_constexpr(cute.size(tSrS_f16)):
                    h = cute.get(tCcS[ei], mode=[0, 0])
                    n = cute.get(tCcS[ei], mode=[0, 1])
                    sdS_gemm_view_1[h, n] = tSrS_f16[ei]

            cute.arch.fence_proxy("async.shared", space="cta")

            if bi % 2 == 0:
                cute.arch.mbarrier_arrive(mbar + MBAR_DS_READY_0)
            else:
                cute.arch.mbarrier_arrive(mbar + MBAR_DS_READY_1)

        # ---- Step 3: After all iterations — dQ via TMA store, dW via warp reduction ----

        # Wait for MMA warp to finish the final GEMM3 (dQ accumulation).
        cute.arch.mbarrier_wait(mbar + MBAR_DQ_DONE, Int32(0))

        tDQrDQ = cute.make_rmem_tensor(tDQrDQ_shape, Float32)
        cute.copy(tiled_tmem_load_dq, tDqDq_t2r, tDQrDQ)

        tDQrDQ_bf16 = cute.make_rmem_tensor(tDQrDQ.shape, self.q_dtype)
        for ei in cutlass.range_constexpr(cute.size(tDQrDQ)):
            tDQrDQ_bf16[ei] = self.q_dtype(tDQrDQ[ei] * Float32(sm_scale))

        cute.arch.fence_view_async_tmem_load()

        # dQ staging via coordinate writes.
        sdQ_gemm_view = cute.composition(
            sdQ_epi_slice,
            cute.make_layout((self.heads_padded, self.head_dim_padded)),
        )
        for ei in cutlass.range_constexpr(cute.size(tDQrDQ_bf16)):
            h = cute.get(tCcDQ[ei], mode=[0, 0])
            d = cute.get(tCcDQ[ei], mode=[0, 1])
            sdQ_gemm_view[h, d] = tDQrDQ_bf16[ei]

        self.compute_sync_barrier.arrive_and_wait()
        cute.arch.fence_proxy("async.shared", space="cta")
        self.compute_sync_barrier.arrive_and_wait()

        if warp_idx == compute_warp0:
            dQ_store_pipeline.producer_acquire()
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)
            dQ_store_pipeline.producer_commit()

        HEADS_PER_WARP = const_expr(self.heads_padded // 4)
        warp_base_h = warp_id_in_wg * Int32(HEADS_PER_WARP)
        for h_local in cutlass.range_constexpr(HEADS_PER_WARP):
            h = warp_base_h + h_local
            my_partial = Float32(0.0)
            for ei in cutlass.range_constexpr(cute.size(dw_accum)):
                if cute.get(tCcS[ei], mode=[0, 0]) == h:
                    my_partial = my_partial + dw_accum[ei]
            total = cute.arch.warp_reduction_sum(my_partial)
            if lane_id == 0:
                mdW[seq_idx, h, batch_idx] = self.q_dtype(total)

    # =========================================================================
    # Warps 12-15: Reduce warpgroup (dK T2R readback + 2-wide atomic_add, 2-stage)
    # =========================================================================
    @cute.jit
    def _reduce_warpgroup(
        self,
        mdK_f32,
        sTopkIdxs,
        mTopkIdx,
        dk_acc_shape,
        tDkDk_0,
        tDkDk_1,
        sm_scale: Float32 | float,
        seq_idx,
        batch_idx,
        seqlen_k,
        batch_size,
        tidx,
        mbar,
    ):
        """Reduce warpgroup: TMEM readback dK → 2-wide atomic_add to global f32 memory."""
        wg_tidx = tidx % self.WARPGROUP_SIZE

        tmem_load_atom_dk = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )

        tiled_tmem_load_dk_0 = tcgen05.make_tmem_copy(tmem_load_atom_dk, tDkDk_0)
        thr_tmem_load_dk_0 = tiled_tmem_load_dk_0.get_slice(wg_tidx)
        tDkDk_t2r_0 = thr_tmem_load_dk_0.partition_S(tDkDk_0)
        cDK = cute.make_identity_tensor(dk_acc_shape)
        tCcDK = thr_tmem_load_dk_0.partition_D(cDK)
        tDKrDK_shape = tCcDK.shape

        tiled_tmem_load_dk_1 = tcgen05.make_tmem_copy(tmem_load_atom_dk, tDkDk_1)
        thr_tmem_load_dk_1 = tiled_tmem_load_dk_1.get_slice(wg_tidx)
        tDkDk_t2r_1 = thr_tmem_load_dk_1.partition_S(tDkDk_1)

        dk_full_0_phase = Int32(0)
        dk_full_1_phase = Int32(0)
        for bi in cutlass.range(0, self.num_topk_blocks):
            tDKrDK = cute.make_rmem_tensor(tDKrDK_shape, Float32)
            if bi % 2 == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_DK_FULL_0, dk_full_0_phase)
                dk_full_0_phase ^= 1
                cute.copy(tiled_tmem_load_dk_0, tDkDk_t2r_0, tDKrDK)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(mbar + MBAR_DK_EMPTY_0)
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_DK_FULL_1, dk_full_1_phase)
                dk_full_1_phase ^= 1
                cute.copy(tiled_tmem_load_dk_1, tDkDk_t2r_1, tDKrDK)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(mbar + MBAR_DK_EMPTY_1)

            # dK reduction: 2-wide atomic_add per consecutive (n, d/d+1) pair.
            # mdK_f32 is the flat (B*S_k, D) view; topk_idx (global) indexes
            # directly. SMEM-cached ids are already global (preload converted
            # local→global when topk_indices_global=False); gmem fallback
            # mirrors that conversion via const_expr branch.
            batch_offset_l2g = Int32(0) if const_expr(self.topk_indices_global) else batch_idx * (seqlen_k // batch_size)
            for pair in cutlass.range_constexpr(cute.size(tDKrDK) // 2):
                ei = pair * 2
                n = cute.get(tCcDK[ei], mode=[0, 0])
                d = cute.get(tCcDK[ei], mode=[0, 1])
                idx_pos = bi * self.block_I + n
                topk_idx = Int32(0)
                if idx_pos < cute.size(sTopkIdxs.layout):
                    topk_idx = Int32(sTopkIdxs[idx_pos])
                else:
                    raw_id = Int32(mTopkIdx[seq_idx, idx_pos, batch_idx])
                    if const_expr(self.topk_indices_global):
                        topk_idx = raw_id
                    else:
                        topk_idx = raw_id + batch_offset_l2g if raw_id >= 0 else raw_id
                if topk_idx >= 0 and topk_idx < seqlen_k:
                    dk_row = mdK_f32[topk_idx, None]
                    dk_pairs = cute.flat_divide(dk_row, (2,))
                    rdK_pair = cute.make_rmem_tensor((2,), Float32)
                    rdK_pair[0] = tDKrDK[ei] * Float32(sm_scale)
                    rdK_pair[1] = tDKrDK[ei + 1] * Float32(sm_scale)
                    cute.arch.atomic_add(
                        dk_pairs[None, d // 2].iterator.llvm_ptr,
                        rdK_pair.load(),
                    )

    # =========================================================================
    # Warps 8-11: K loading warpgroup (3-stage sK, Opt-7)
    # =========================================================================
    @cute.jit
    def _k_load_warpgroup(
        self,
        mK,
        sK,
        sTopkIdxs,
        mTopkIdx,
        seq_idx,
        batch_idx,
        seqlen_k,
        batch_size,
        tidx,
        mbar,
    ):
        """K loading warpgroup: sparse cp.async gather into 3-stage sK.

        3-stage pipeline allows K-load to run 3 blocks ahead of MMA,
        hiding the scatter gather latency (~5300 clk per block).
        """
        wg_tidx = tidx % self.WARPGROUP_SIZE

        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.k_dtype,
            num_bits_per_copy=128,
        )
        async_thr_copy = cute.make_tiled_copy_tv(
            async_copy_atom,
            cute.make_layout((1,)),
            cute.make_layout((8,)),
        ).get_slice(0)

        GROUP_SIZE = const_expr(8)
        NUM_GROUPS = const_expr(self.WARPGROUP_SIZE // 8)
        ROWS_PER_GROUP = const_expr(self.block_I // NUM_GROUPS)
        idx_in_group = wg_tidx % GROUP_SIZE
        group_idx_local = wg_tidx // GROUP_SIZE

        # 3-stage sK slices.
        sK_slice_0 = cute.composition(
            sK[None, None, None, 0],
            cute.make_layout((self.block_I, self.head_dim_padded)),
        )
        sK_slice_1 = cute.composition(
            sK[None, None, None, 1],
            cute.make_layout((self.block_I, self.head_dim_padded)),
        )
        sK_slice_2 = cute.composition(
            sK[None, None, None, 2],
            cute.make_layout((self.block_I, self.head_dim_padded)),
        )
        # mK is the flat (B*S_k, D) view; topk_idx (global) indexes directly.
        # gmem fallback mirrors the SMEM preload's local→global conversion when
        # topk_indices_global=False; const_expr-folded to a no-op for default.
        batch_offset_l2g = Int32(0) if const_expr(self.topk_indices_global) else batch_idx * (seqlen_k // batch_size)

        k_consumed_0_phase_kload = Int32(0)
        k_consumed_1_phase_kload = Int32(0)
        k_consumed_2_phase_kload = Int32(0)

        for bi in cutlass.range_constexpr(self.num_topk_blocks):
            # Back-pressure: wait for MMA to finish using sK[bi%3].
            # K_CONSUMED[s] fires after GEMM3 (last read of sKt[s]).
            # bi=0,1,2: sK stages fresh, no wait. bi>=3: must wait.
            if bi >= 3:
                if bi % 3 == 0:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_0, k_consumed_0_phase_kload)
                    k_consumed_0_phase_kload ^= 1
                elif bi % 3 == 1:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_1, k_consumed_1_phase_kload)
                    k_consumed_1_phase_kload ^= 1
                else:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_2, k_consumed_2_phase_kload)
                    k_consumed_2_phase_kload ^= 1

            sK_slice = sK_slice_0 if bi % 3 == 0 else (sK_slice_1 if bi % 3 == 1 else sK_slice_2)

            for r in cutlass.range_constexpr(ROWS_PER_GROUP):
                row = r * NUM_GROUPS + group_idx_local
                idx_pos = bi * self.block_I + row
                topk_idx = Int32(0)
                if idx_pos < cute.size(sTopkIdxs.layout):
                    topk_idx = Int32(sTopkIdxs[idx_pos])
                else:
                    raw_id = Int32(mTopkIdx[seq_idx, idx_pos, batch_idx])
                    if const_expr(self.topk_indices_global):
                        topk_idx = raw_id
                    else:
                        topk_idx = raw_id + batch_offset_l2g if raw_id >= 0 else raw_id
                if topk_idx >= 0 and topk_idx < seqlen_k:
                    gK_raw = mK[topk_idx, None]
                    gK = cute.make_tensor(
                        cute.make_ptr(self.k_dtype, gK_raw.iterator.llvm_ptr, cute.AddressSpace.gmem, assumed_align=16),
                        gK_raw.layout,
                    )
                    gChunks = cute.flat_divide(gK, (8,))
                    sRow = sK_slice[row, None]
                    sChunks = cute.flat_divide(sRow, (8,))
                    for tile in cutlass.range_constexpr(self.head_dim_padded // 64):
                        chunk_idx = tile * 8 + idx_in_group
                        tSg = async_thr_copy.partition_S(gChunks[None, chunk_idx])
                        tSs = async_thr_copy.partition_D(sChunks[None, chunk_idx])
                        cute.copy(async_copy_atom, tSg, tSs)
                else:
                    sRow = sK_slice[row, None]
                    sChunks = cute.flat_divide(sRow, (8,))
                    for tile in cutlass.range_constexpr(self.head_dim_padded // 64):
                        chunk_idx = tile * 8 + idx_in_group
                        sChunks[None, chunk_idx].fill(0)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()

            # Signal K loaded (3-stage barrier)
            if bi % 3 == 0:
                cute.arch.mbarrier_arrive(mbar + MBAR_K_LOADED_0)
            elif bi % 3 == 1:
                cute.arch.mbarrier_arrive(mbar + MBAR_K_LOADED_1)
            else:
                cute.arch.mbarrier_arrive(mbar + MBAR_K_LOADED_2)

    @cute.jit
    def get_tmem_tensor(self, s_acc_layout, dq_acc_layout, dk_acc_layout, tmem_ptr_base: cute.Pointer):
        tStS_0 = cute.make_tensor(tmem_ptr_base + self.tmem_s0_offset, s_acc_layout)
        tStS_1 = cute.make_tensor(tmem_ptr_base + self.tmem_s1_offset, s_acc_layout)
        tDqDq = cute.make_tensor(tmem_ptr_base + self.tmem_dq_offset, dq_acc_layout)
        tDkDk_0 = cute.make_tensor(tmem_ptr_base + self.tmem_s0_offset, dk_acc_layout)
        tDkDk_1 = cute.make_tensor(tmem_ptr_base + self.tmem_s1_offset, dk_acc_layout)
        return tStS_0, tStS_1, tDqDq, tDkDk_0, tDkDk_1


# =============================================================================
# Factory
# =============================================================================
_compile_cache: dict = {}


def indexer_backward_sm100(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale=1.0,
    block_I=128,
    topk_indices_global: bool = True,
):
    # ``grad_scale`` is intentionally **not** an argument: it's a host scalar
    # that the kernels consume only as a multiplicative factor, so it's
    # passed at ``_run`` call time (and forwarded to ``score_grad`` as a
    # runtime ``Float32``). Keeping it out of this factory's signature +
    # cache key avoids spurious recompiles when the caller changes the
    # runtime loss scaling for the same tensor shape.
    #
    # ``topk_indices_global`` selects the topk-id contract:
    #   True  (default): mTopkIdx carries global flat ids — load directly.
    #   False (legacy):  mTopkIdx carries local-per-batch ids — kernel adds
    #                    ``batch_idx * S_k_per_batch`` to convert to global
    #                    flat for the (B*S_k, D) K/dK view.
    # Const_expr-branched in the kernel so it's part of the cache key.
    # THD packed varlen is supported at the wrapper level by treating the
    # packed tensors as a single B=1 BSHD batch (sparse path's topk indices
    # already encode per-batch validity, so no kernel-side cu_seqlens are
    # needed). See ``_indexer_backward_sparse_thd`` in csrc/bwd/__init__.py.
    key = (batch, seqlen, seqlen_k, heads, dim, topk, sm_scale, block_I, topk_indices_global)
    if key not in _compile_cache:
        _compile_cache[key] = _build_cute_dsl_kernel(batch, seqlen, seqlen_k, heads, dim, topk, sm_scale, block_I, topk_indices_global=topk_indices_global)
    return _compile_cache[key]


class ScoreGradSm100:
    """CuTe DSL kernel for in-place score_grad precompute."""

    THREADS_PER_CTA = 128
    WARP_SIZE = 32
    NUM_WARPS = THREADS_PER_CTA // WARP_SIZE

    def __init__(self, topk: int):
        self.topk = topk

    @cute.jit
    def __call__(
        self,
        mAttnScore: cute.Tensor,
        mIndexScore: cute.Tensor,
        mGradLoss: cute.Tensor,
        grad_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        # (b, s, t) -> (s, t, b): topk dim contiguous for per-CTA strided loops.
        mAttnScore = cute.make_tensor(mAttnScore.iterator, cute.select(mAttnScore.layout, mode=[1, 2, 0]))
        mIndexScore = cute.make_tensor(mIndexScore.iterator, cute.select(mIndexScore.layout, mode=[1, 2, 0]))

        seqlen = cute.size(mAttnScore.shape[0])
        batch_size = cute.size(mAttnScore.shape[2]) if cute.rank(mAttnScore.shape) > 2 else 1
        self.kernel_score_grad(mAttnScore, mIndexScore, mGradLoss, grad_scale).launch(
            grid=(batch_size, seqlen, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel_score_grad(self, mAttnScore, mIndexScore, mGradLoss, grad_scale: Float32 | float):
        tidx = cute.arch.thread_idx()[0]
        batch_idx = cute.arch.block_idx()[0]
        seq_idx = cute.arch.block_idx()[1]
        # grad_scale is a compile/runtime scalar (loss_coeff / (b*sq));
        # grad_loss lives in a shape-(1,) f32 GPU tensor (from autograd).
        # Fold them together once per CTA — the compiler will hoist.
        grad_scale_f32 = Float32(grad_scale) * Float32(mGradLoss[0])

        @cute.struct
        class SharedStorage:
            thread_sums: cute.struct.Align[cute.struct.MemRange[Float32, self.THREADS_PER_CTA], 128]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        thread_sums = storage.thread_sums.get_tensor(cute.make_layout((self.THREADS_PER_CTA,), stride=(1,)))

        local_sum = Float32(0.0)
        for pos in cutlass.range(tidx, self.topk, self.THREADS_PER_CTA):
            target = Float32(mAttnScore[seq_idx, pos, batch_idx])
            predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
            target_eff = cute.arch.fmax(target, Float32(CLIP_PROB_MIN))
            log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
            local_sum += -target_eff * log_clip_mask * grad_scale_f32

        thread_sums[tidx] = local_sum
        cute.arch.sync_threads()

        if tidx == 0:
            block_sum = Float32(0.0)
            for i in cutlass.range_constexpr(self.THREADS_PER_CTA):
                block_sum += thread_sums[i]
            thread_sums[0] = block_sum
        cute.arch.sync_threads()

        sum_grad = thread_sums[0]
        for pos in cutlass.range(tidx, self.topk, self.THREADS_PER_CTA):
            target = Float32(mAttnScore[seq_idx, pos, batch_idx])
            predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
            target_eff = cute.arch.fmax(target, Float32(CLIP_PROB_MIN))
            log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
            g_i = -target_eff * log_clip_mask * grad_scale_f32
            mAttnScore[seq_idx, pos, batch_idx] = g_i - predict * sum_grad
            mIndexScore[seq_idx, pos, batch_idx] = sum_grad


def _score_grad_inplace_cute(AttnScore, IndexScore, GradLoss, grad_scale, current_stream=None):
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

    # Kernel reads ``mGradLoss[0]`` so it must be at least 1-D. ``to_cute_tensor``
    # defaults ``leading_dim = ndim - 1`` which collapses to -1 for a 0-D scalar
    # and trips cute's layout validator. The public wrapper reshapes upstream;
    # this guard keeps direct factory callers (benchmarks, tests) safe too.
    if GradLoss.ndim == 0:
        GradLoss = GradLoss.reshape(1)

    _, _, topk = AttnScore.shape
    compile_key = (topk,)
    s = _resolve_stream(current_stream)
    if compile_key not in _score_grad_cute_cache:
        kernel_obj = ScoreGradSm100(topk=topk)
        _score_grad_cute_cache[compile_key] = cute.compile(
            kernel_obj,
            to_cute_tensor(AttnScore),
            to_cute_tensor(IndexScore),
            to_cute_tensor(GradLoss),
            cutlass.Float32(float(grad_scale)),
            s,
            options=compile_options("--opt-level 3"),
        )

    _score_grad_cute_cache[compile_key](
        AttnScore,
        IndexScore,
        GradLoss,
        cutlass.Float32(float(grad_scale)),
        s,
    )


def _score_grad_inplace(AttnScore, IndexScore, GradLoss, grad_scale, block_I=128, current_stream=None):
    """Kernel 1: Compute clipped-log KL grad_signal from target/predict.

    Results overwrite the two Score tensors in-place:
      AttnScore ← grad_signal   (per topk element)
      IndexScore ← sum_grad     (broadcast scalar per (batch, seqlen))

    grad_scale: Python float (loss_coeff / (b*sq)), passed as a runtime
                ``Float32`` arg — not in the kernel cache key.
    GradLoss:   shape-(1,) f32 GPU tensor from autograd; read once per CTA.
    """
    # Match kl_div(log_target=True) with input/target clipped to [-100, 0]:
    #   input  = clip(log_predict)
    #   target = clip(log_target)
    # dL/dlog_predict = -exp(target) * I(log_predict >= -100)
    # dL/dlogits = g - predict * sum(g)
    can_use_cute = (
        AttnScore.is_cuda
        and IndexScore.is_cuda
        and AttnScore.dtype == torch.float32
        and IndexScore.dtype == torch.float32
        and AttnScore.is_contiguous()
        and IndexScore.is_contiguous()
        and AttnScore.ndim == 3
        and AttnScore.shape == IndexScore.shape
    )
    if not can_use_cute:
        raise NotImplementedError("score_grad_inplace requires contiguous fp32 CUDA tensors with matching " "3D shapes; the torch fallback was removed")
    _score_grad_inplace_cute(AttnScore, IndexScore, GradLoss, grad_scale, current_stream=current_stream)


def _build_cute_dsl_kernel(batch, seqlen, seqlen_k, heads, dim, topk, sm_scale, block_I, topk_indices_global: bool = True):
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("Requires SM100+")
    kernel_obj = IndexerBackwardSm100(
        head_dim=dim,
        heads=heads,
        block_I=block_I,
        topk=topk,
        topk_indices_global=topk_indices_global,
    )

    compiled_holder = [None]

    def _ensure_compiled(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, AttnScore, TopkIndices, current_stream=None):
        """Lazy-compile the GEMM kernel (kernel 2)."""
        if compiled_holder[0] is None:
            s = _resolve_stream(current_stream)
            cute_args = [to_cute_tensor(t) for t in [IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, AttnScore, TopkIndices]]
            compiled_holder[0] = cute.compile(
                kernel_obj,
                *cute_args,
                cutlass.Float32(sm_scale),
                s,
                options=compile_options("--opt-level 3"),
            )

    def _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, TopkIndices, current_stream=None):
        """Run only kernel 2 (GEMM). Caller must have run kernel 1 and zeroed dIndexK_f32."""
        s = _resolve_stream(current_stream)
        _ensure_compiled(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, TopkIndices, current_stream=current_stream)
        with torch.cuda.nvtx.range("indexer_backward_dsl_gemm"):
            compiled_holder[0](
                IndexQ,
                Weights,
                IndexK,
                dIndexQ,
                dWeights,
                dIndexK_f32,
                GradSignal,
                TopkIndices,
                cutlass.Float32(sm_scale),
                s,
            )

    def _run(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK, AttnScore, IndexScore, TopkIndices, GradLoss, grad_scale, current_stream=None):
        # ``grad_scale`` is a host scalar (Python float / 0-D fp32 tensor)
        # multiplied into ``score_grad`` as a runtime ``Float32`` arg —
        # changing it across calls does **not** trigger recompilation.
        score_grad = partial(_score_grad_inplace, block_I=block_I)

        # Kernel 1: Compute grad_signal from scores (CuTe DSL only).
        score_grad(AttnScore, IndexScore, GradLoss, grad_scale, current_stream=current_stream)

        if dIndexK.dtype == torch.float32:
            # Caller provided a pre-zeroed f32 buffer; write directly (no extra
            # alloc + cast). This matches the SM90 _run fast path.
            _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK, AttnScore, TopkIndices, current_stream=current_stream)
        else:
            # Need a separate f32 buffer for atomicAdd, then cast back to output dtype.
            with _torch_stream_context(current_stream):
                dIndexK_f32 = torch.zeros_like(dIndexK, dtype=torch.float32)
            _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, AttnScore, TopkIndices, current_stream=current_stream)
            with _torch_stream_context(current_stream):
                dIndexK.copy_(dIndexK_f32)

    _run.score_grad = partial(_score_grad_inplace, block_I=block_I)
    _run.gemm_only = _run_gemm_only

    return _run
