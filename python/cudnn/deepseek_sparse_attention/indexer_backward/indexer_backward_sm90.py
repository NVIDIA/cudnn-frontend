# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Indexer Backward — SM90 CuTe-DSL, 3-kernel design (Hopper).

Three kernels launched sequentially on the same stream:

  Kernel 1 (CuTe DSL): score_grad — compute sum_grad and grad_signal from
      AttnScore and IdxScore, overwrite AttnScore with grad_signal in-place.
      Unsupported inputs trigger an exception before this stage launches.
  Kernel 2 (CuTe DSL): kernel_gemm — 3-warpgroup GEMM kernel (below).
      Ordinary dK is accumulated in float32 via cp.reduce.async.bulk.
      Deterministic execution uses stream-ordered bounded waves, one CTA-private
      FP32 dK plane per possible producer, and a fixed-order fold.
  Finalization (same stream): deterministic mode runs a fixed-order CuTe DSL
      dK plane reducer; ordinary BF16 dK uses a PyTorch copy/cast, while
      ordinary FP32 dK needs no finalizer.

Kernel 2 — 3-Warpgroup architecture (12 warps, 384 threads):
  WG0 (warps 0-3): Compute A — first half of N/M splits
  WG1 (warps 4-7): Compute B — second half of N/M splits
    Both cooperatively: Load Q/grad_signal/weights from GMEM to SMEM
    Per topk-block: GEMM1(S_half) -> fused dS/dW -> GEMM2(dK_half) -> GEMM3(dQ_half)
    Epilogue: output dQ_half, ordinary dW/dK via reductions; deterministic dW
    keeps the private-slot fold for small top-k and uses a fixed lane-quad then
    warp-order fold for top-k >= 512, while dK targets the CTA's private plane.
  WG2 (warps 8-11): K-load warpgroup
    Per topk-block: loads indices + sparse K via cp.async gather (3-stage)

GEMM splitting across 2 compute WGs:
  GEMM1 S[H, I]:    split N=I  -> each WG computes S[:, I/2]
  GEMM2 dK[D, I]:   split M=D  -> each WG computes dK[D/2, :]
  GEMM3 dQ[H, D]:   split N=D  -> each WG computes dQ[:, D/2]

Barriers:
  mbar[0-2]: K_LOADED_0/1/2      (K-load -> both Compute WGs, per-stage, 3-stage)
  mbar[3-5]: K_CONSUMED_0/1/2    (both Compute WGs -> K-load, arrival count=2)
  mbar[6-8]: INDICES_READY_0/1/2 (K-load internal)
  mbar[9]:   Q_TMA               (TMA completion)
  CTA barrier(3): compute_sync (256 threads: WG0+WG1)
  CTA barrier(4): wg_sched_0  (ping-pong: WG0 syncs on this, WG1 arrives)
  CTA barrier(5): wg_sched_1  (ping-pong: WG1 syncs on this, WG0 arrives)

SMEM (kernel 2): sGradSignal[topk] holds precomputed grad_signal from kernel 1.
Ordinary grid: (seqlen, batch, 1). Deterministic launches bounded
(min(remaining_rows, dK planes), 1, 1) waves. Each CTA handles one query row.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cudnn._cutlass_compat import LayoutEnum, SmemAllocator

from cudnn.api_base import WorkspaceCarver, ws_align
from cudnn.frost.buffers import memset_zero_async
from cudnn.deepseek_sparse_attention.utils import copy as copy_ops
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
)
from cudnn.deepseek_sparse_attention.utils.seqlen import seqlen_info as _seqlen_info
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from cudnn.deepseek_sparse_attention.utils.sm90.mma import (
    gemm,
    gemm_zero_init,
    make_smem_layout,
)
from cudnn.deepseek_sparse_attention.utils.sm90.primitives import transpose_view

from .deterministic_dk_sm100 import (
    DEFAULT_DK_PLANES,
    deterministic_dk_planes_bytes,
    deterministic_dk_workspace_bytes,
    reduce_deterministic_dk,
    validate_deterministic_dk_plane_policy,
)

MBAR_K_LOADED_0 = 0
MBAR_K_LOADED_1 = 1
MBAR_K_LOADED_2 = 2
MBAR_K_CONSUMED_0 = 3
MBAR_K_CONSUMED_1 = 4
MBAR_K_CONSUMED_2 = 5
MBAR_INDICES_READY_0 = 6
MBAR_INDICES_READY_1 = 7
MBAR_INDICES_READY_2 = 8
MBAR_Q_TMA = 9
NUM_BARRIERS = 10
NUM_K_STAGES = 3

EPS = 1e-10
CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)

_score_grad_cute_cache: dict = {}


class IndexerBackwardSm90:
    arch = 90
    WARP_SIZE = 32
    WARPGROUP_SIZE = 128
    NUM_WARPS = 12
    THREADS_PER_CTA = 384
    NUM_COMPUTE_WGS = 2
    TOTAL_COMPUTE_THREADS = 256
    COMPUTE_WG_A = 0  # warps 0-3
    COMPUTE_WG_B = 1  # warps 4-7
    KLOAD_WG = 2  # warps 8-11

    def __init__(
        self,
        head_dim,
        heads=64,
        block_I=128,
        topk=512,
        is_dense=False,
        topk_indices_global: bool = True,
        ratio: int = 1,
        deterministic: bool = False,
        deterministic_dk_planes: int = DEFAULT_DK_PLANES,
    ):
        """Initialize the SM90 warp-specialized Indexer backward kernel."""
        self.head_dim = head_dim
        self.heads = heads
        self.block_I = block_I
        self.topk = topk
        self.is_dense = is_dense
        assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
        self.ratio = ratio
        self.deterministic = bool(deterministic)
        self.deterministic_dk_planes = validate_deterministic_dk_plane_policy(self.deterministic, deterministic_dk_planes)
        assert not self.deterministic or not self.is_dense, "deterministic SM90 IndexerBackward serves sparse mode only"
        # Public forward/topk returns global KV ids by default. Sparse mode
        # decodes them to local per-batch ids before indexing K/dK views.
        self.topk_indices_global = topk_indices_global
        assert heads >= 64
        assert is_dense or topk % block_I == 0
        self.num_topk_blocks = (topk + block_I - 1) // block_I
        self.grad_signal_smem_size = self.block_I if is_dense else self.topk

        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.heads_padded = int(math.ceil(heads / 8) * 8)

        self.half_block_I = self.block_I // 2
        self.half_head_dim = self.head_dim_padded // 2

        # Full GEMM tilers (M, N, K) — reference
        self.gemm1_tiler = (self.heads_padded, self.block_I, self.head_dim_padded)
        self.gemm2_tiler = (self.head_dim_padded, self.block_I, self.heads_padded)
        self.gemm3_tiler = (self.heads_padded, self.head_dim_padded, self.block_I)

        # Per-WG half tilers (M, N) for TiledMMA
        self.gemm1_tiler_half = (self.heads_padded, self.half_block_I)
        self.gemm2_tiler_half = (64, self.block_I)
        self.gemm3_tiler_half = (self.heads_padded, self.half_head_dim)

        self.acc_dtype = Float32

        self.dw_elems_per_thread = self.heads_padded * self.block_I // self.WARPGROUP_SIZE
        self.dw_smem_size = self.WARPGROUP_SIZE * self.dw_elems_per_thread
        # ``topk`` already specializes the generated loop nest and compile
        # cache. Keep the lower-overhead private-slot fold for small top-k;
        # amortize the fixed quad shuffle/fold for larger loop nests without
        # introducing another compile-key axis.
        self.use_deterministic_dw_quad = self.deterministic and self.topk >= 512
        self.dw_deterministic_slots = self.TOTAL_COMPUTE_THREADS * 2 if self.deterministic and not self.use_deterministic_dw_quad else 1
        self.dw_deterministic_warp_partials = (self.TOTAL_COMPUTE_THREADS // self.WARP_SIZE) * self.heads if self.deterministic else 1

        self.dk_staging_stride_n = self.head_dim_padded + 4  # 132: 16B-aligned + bank-conflict-free
        self.dk_staging_elems = self.dk_staging_stride_n * self.block_I

        self.num_regs_compute = 232
        self.num_regs_kload = 40

        self.buffer_align_bytes = 1024

        self.COMPUTE_SYNC_BARRIER_ID = 3

        # Ping-pong scheduler barriers (WG0 syncs on 4, WG1 syncs on 5)
        self.SCHED_BARRIER_WG0 = 4
        self.SCHED_BARRIER_WG1 = 5

        # dK staging-buffer lifetime barriers.  Each compute WG owns a
        # disjoint D/2 half, so they can synchronize independently.
        self.DK_STAGING_BARRIER_WG0 = 6
        self.DK_STAGING_BARRIER_WG1 = 7

    @cute.jit
    def _dense_num_k_blocks(self, q_token, seqlen_q, seqlen_k, q_causal_offset):
        """Return dense K blocks that can contain valid ratio-causal columns."""
        ratio = Int32(self.ratio)
        col_limit = (q_causal_offset + q_token + Int32(1)) // ratio
        col_limit = col_limit if col_limit > Int32(0) else Int32(0)
        col_limit = col_limit if col_limit < seqlen_k else seqlen_k
        return cute.ceil_div(col_limit, self.block_I)

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
        mTopkIdx: cute.Tensor = None,
        sm_scale: Float32 | float = 1.0,
        stream: cuda.CUstream = None,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        max_seqlen_q: Int32 = None,
        max_seqlen_k: Int32 = None,
        mQCausalOffsets=None,
        row_base: Int32 = 0,
    ):
        is_varlen = const_expr(self.is_dense and mCuSeqlensQ is not None)
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type

        # THD packed dense mode keeps tensors in their natural packed layout:
        #   Q/dQ: (T_q,H,D), K/dK: (T_k,D), W/dW/GradSignal: (T_q,*).
        if const_expr(not is_varlen):
            # BSHD / sparse mode: (bs, seqlen, ...) -> (seqlen, ..., bs)
            mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 3, 0]))
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 2, 0]))
            mW = cute.make_tensor(mW.iterator, cute.select(mW.layout, mode=[1, 2, 0]))
            mdQ = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 2, 3, 0]))
            mdW = cute.make_tensor(mdW.iterator, cute.select(mdW.layout, mode=[1, 2, 0]))
            mdK_f32 = cute.make_tensor(mdK_f32.iterator, cute.select(mdK_f32.layout, mode=[1, 2, 0]))
            mGradSignal = cute.make_tensor(mGradSignal.iterator, cute.select(mGradSignal.layout, mode=[1, 2, 0]))
        if const_expr(not self.is_dense):
            mTopkIdx = cute.make_tensor(mTopkIdx.iterator, cute.select(mTopkIdx.layout, mode=[1, 2, 0]))

        # GEMM1: S[H, I/2] per WG.  A=K-major(SS), B=K-major.  tiler N halved.
        # A=K-major lets us use TMA-loaded sQ directly (D contiguous),
        # eliminating the sQ→sQ_mn SMEM copy.
        tmma1 = sm90_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=self.gemm1_tiler_half,
        )
        # GEMM2: dK[D/2, I] per WG.  both MN-major.  tiler M=64 = D/2.
        tmma2 = sm90_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=self.gemm2_tiler_half,
        )
        # GEMM3: dQ[H, D/2] per WG.  A=K-major(RS), B=MN-major.  tiler N halved.
        tmma3 = sm90_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=self.gemm3_tiler_half,
        )

        # Primary SMEM layouts (K-major = ROW_MAJOR: last dim contiguous)
        sQ_layout = make_smem_layout(
            self.q_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.heads_padded, self.head_dim_padded),
        )
        sK_layout = make_smem_layout(
            self.k_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.block_I, self.head_dim_padded),
            stage=NUM_K_STAGES,
        )
        sdS_layout = make_smem_layout(
            self.q_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.heads_padded, self.block_I),
        )
        # TMA Q load: BSHD (seqlen, heads, dim, batch) -> (heads, dim, seqlen, batch);
        # THD (total_q, heads, dim) -> (heads, dim, total_q).
        if const_expr(is_varlen):
            mQ_tma = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0]))
        else:
            mQ_tma = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0, 3]))
        tma_atom_Q, mQ_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ_tma,
            sQ_layout,
            (self.heads_padded, self.head_dim_padded),
        )
        self.tma_copy_Q_bytes = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1]))

        # TMA dQ store: reuse sQ_layout (same bf16 swizzled layout)
        if const_expr(is_varlen):
            mdQ_tma = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 2, 0]))
        else:
            mdQ_tma = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 2, 0, 3]))
        tma_atom_dQ, mdQ_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdQ_tma,
            sQ_layout,
            (self.heads_padded, self.head_dim_padded),
        )

        # Dense mode: create TMA atom for K loading (sequential, no scatter-gather)
        if const_expr(self.is_dense):
            sK_single_layout = make_smem_layout(
                self.k_dtype,
                LayoutEnum.ROW_MAJOR,
                (self.block_I, self.head_dim_padded),
            )
            tma_atom_K, mK_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mK,
                sK_single_layout,
                (self.block_I, self.head_dim_padded),
            )
            self.tma_copy_K_bytes = cute.size_in_bytes(self.k_dtype, cute.select(sK_single_layout, mode=[0, 1]))
            mK_for_kernel = mK_tma
        else:
            tma_atom_K = None
            mK_for_kernel = mK

        if const_expr(is_varlen):
            seqlen = Int32(max_seqlen_q)
            seqlen_k = Int32(max_seqlen_k)
            batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1
        else:
            seqlen = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            batch_size = cute.size(mQ.shape[3]) if cute.rank(mQ.shape) > 3 else 1

        if const_expr(self.deterministic):
            wave_rows = cutlass.min(seqlen * batch_size - row_base, self.deterministic_dk_planes)
            launch_grid = (wave_rows, 1, 1)
        else:
            launch_grid = (seqlen, batch_size, 1)
        self.kernel(
            mQ_tma,
            mW,
            mK_for_kernel,
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
            tma_atom_Q,
            tma_atom_dQ,
            sQ_layout,
            sK_layout,
            sdS_layout,
            seqlen,
            seqlen_k,
            batch_size,
            tma_atom_K,
            mCuSeqlensQ,
            mCuSeqlensK,
            mQCausalOffsets,
            row_base,
        ).launch(
            grid=launch_grid,
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        mW,
        mK,
        mKScalar,
        mdQ,
        mdW,
        mdK_f32,
        mGradSignal,
        mTopkIdx,
        sm_scale: Float32 | float,
        tmma1,
        tmma2,
        tmma3,
        tma_atom_Q,
        tma_atom_dQ,
        sQ_layout,
        sK_layout,
        sdS_layout,
        seqlen: Int32,
        seqlen_k_static: Int32,
        batch_size: Int32,
        tma_atom_K: cute.CopyAtom = None,
        mCuSeqlensQ=None,
        mCuSeqlensK=None,
        mQCausalOffsets=None,
        row_base: Int32 = 0,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = tidx // self.WARPGROUP_SIZE
        # ---- SMEM allocation ----
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sdS_size = cute.cosize(sdS_layout)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, NUM_BARRIERS]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]
            sdS: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sdS_size], self.buffer_align_bytes]
            sGradSignal: cute.struct.Align[cute.struct.MemRange[Float32, self.grad_signal_smem_size], 128]
            sIndices0: cute.struct.Align[cute.struct.MemRange[Int32, self.block_I], 128]
            sIndices1: cute.struct.Align[cute.struct.MemRange[Int32, self.block_I], 128]
            sIndices2: cute.struct.Align[cute.struct.MemRange[Int32, self.block_I], 128]
            sW: cute.struct.Align[cute.struct.MemRange[self.q_dtype, self.heads], 128]
            sDwPartial: cute.struct.Align[cute.struct.MemRange[Float32, self.heads_padded], 128]
            sDwContrib: cute.struct.Align[cute.struct.MemRange[Float32, self.dw_deterministic_slots], 128]
            sDwHead: cute.struct.Align[cute.struct.MemRange[Int32, self.dw_deterministic_slots], 128]
            sDwWarpPartial: cute.struct.Align[cute.struct.MemRange[Float32, self.dw_deterministic_warp_partials], 128]
            sdK_staging: cute.struct.Align[cute.struct.MemRange[Float32, self.dk_staging_elems], 128]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar = storage.mbar.data_ptr()

        # Swizzled SMEM tensors
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)

        # Transposed views via composition (swap first two dims, preserve swizzle)
        sQt = transpose_view(sQ)
        sKt = transpose_view(sK)
        sdSt = transpose_view(sdS)

        sGradSignal = storage.sGradSignal.get_tensor(cute.make_layout((self.grad_signal_smem_size,), stride=(1,)))
        sIndices0 = storage.sIndices0.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        sIndices1 = storage.sIndices1.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        sIndices2 = storage.sIndices2.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        sW = storage.sW.get_tensor(cute.make_layout((self.heads,), stride=(1,)))

        sDwPartial = storage.sDwPartial.get_tensor(cute.make_layout((self.heads_padded,), stride=(1,)))
        sDwContrib = storage.sDwContrib.get_tensor(cute.make_layout((self.dw_deterministic_slots,), stride=(1,)))
        sDwHead = storage.sDwHead.get_tensor(cute.make_layout((self.dw_deterministic_slots,), stride=(1,)))
        sDwWarpPartial = storage.sDwWarpPartial.get_tensor(cute.make_layout((self.dw_deterministic_warp_partials,), stride=(1,)))

        sdK_staging = storage.sdK_staging.get_tensor(cute.make_layout((128, self.block_I), stride=(1, self.dk_staging_stride_n)))

        # Descriptor state is immutable throughout the CTA.
        cpasync.prefetch_descriptor(tma_atom_Q)
        cpasync.prefetch_descriptor(tma_atom_dQ)
        if const_expr(self.is_dense):
            cpasync.prefetch_descriptor(tma_atom_K)

        # Set the register split once for the CTA's fixed warp roles.
        if warp_group_idx == self.COMPUTE_WG_A:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
        elif warp_group_idx == self.COMPUTE_WG_B:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_kload)

        if const_expr(self.deterministic):
            # Host-side bounded waves are submitted in one stream. Within a
            # wave CTA i owns plane i and row row_base+i; later waves cannot
            # reuse that plane until this launch completes.
            producer_idx = cute.arch.block_idx()[0]
            mdK_plane = mdK_f32[None, None, producer_idx]
            flat_row_idx = row_base + producer_idx
            batch_idx = flat_row_idx // seqlen
            seq_idx = flat_row_idx - batch_idx * seqlen
            self._run_row(
                mQ,
                mW,
                mK,
                mKScalar,
                mdQ,
                mdW,
                mdK_plane,
                mGradSignal,
                mTopkIdx,
                sm_scale,
                tmma1,
                tmma2,
                tmma3,
                tma_atom_Q,
                tma_atom_dQ,
                sQ_layout,
                sQ,
                sQt,
                sK,
                sKt,
                sdS,
                sdSt,
                sGradSignal,
                sIndices0,
                sIndices1,
                sIndices2,
                sW,
                sDwPartial,
                sDwContrib,
                sDwHead,
                sDwWarpPartial,
                sdK_staging,
                mbar,
                seqlen,
                seqlen_k_static,
                batch_size,
                tma_atom_K,
                mCuSeqlensQ,
                mCuSeqlensK,
                mQCausalOffsets,
                seq_idx,
                batch_idx,
                tidx,
                warp_idx,
                warp_group_idx,
            )
        else:
            self._run_row(
                mQ,
                mW,
                mK,
                mKScalar,
                mdQ,
                mdW,
                mdK_f32,
                mGradSignal,
                mTopkIdx,
                sm_scale,
                tmma1,
                tmma2,
                tmma3,
                tma_atom_Q,
                tma_atom_dQ,
                sQ_layout,
                sQ,
                sQt,
                sK,
                sKt,
                sdS,
                sdSt,
                sGradSignal,
                sIndices0,
                sIndices1,
                sIndices2,
                sW,
                sDwPartial,
                sDwContrib,
                sDwHead,
                sDwWarpPartial,
                sdK_staging,
                mbar,
                seqlen,
                seqlen_k_static,
                batch_size,
                tma_atom_K,
                mCuSeqlensQ,
                mCuSeqlensK,
                mQCausalOffsets,
                cute.arch.block_idx()[0],
                cute.arch.block_idx()[1],
                tidx,
                warp_idx,
                warp_group_idx,
            )

    @cute.jit
    def _run_row(
        self,
        mQ,
        mW,
        mK,
        mKScalar,
        mdQ,
        mdW,
        mdK_f32,
        mGradSignal,
        mTopkIdx,
        sm_scale,
        tmma1,
        tmma2,
        tmma3,
        tma_atom_Q,
        tma_atom_dQ,
        sQ_layout,
        sQ,
        sQt,
        sK,
        sKt,
        sdS,
        sdSt,
        sGradSignal,
        sIndices0,
        sIndices1,
        sIndices2,
        sW,
        sDwPartial,
        sDwContrib,
        sDwHead,
        sDwWarpPartial,
        sdK_staging,
        mbar,
        seqlen,
        seqlen_k_static,
        batch_size,
        tma_atom_K,
        mCuSeqlensQ,
        mCuSeqlensK,
        mQCausalOffsets,
        seq_idx,
        batch_idx,
        tidx,
        warp_idx,
        warp_group_idx,
    ):
        """Run one fully drained query row in its ordinary or bounded-wave CTA."""
        is_varlen = const_expr(self.is_dense and mCuSeqlensQ is not None)
        q_offset, k_offset, seqlen_q_b, seqlen_k_b = _seqlen_info(
            mCuSeqlensQ,
            mCuSeqlensK,
            Int32(batch_idx),
            seqlen,
            seqlen_k_static,
        )
        q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]

        if const_expr(is_varlen):
            mQ_b = cute.domain_offset((Int32(0), Int32(0), q_offset), mQ)
            mdQ_b = cute.domain_offset((Int32(0), Int32(0), q_offset), mdQ)
            mK_b = cute.domain_offset((k_offset, Int32(0)), mK)
            mK_scalar_b = cute.domain_offset((k_offset, Int32(0)), mKScalar)
            mdK_b = cute.domain_offset((k_offset, Int32(0)), mdK_f32)
            mW_b = cute.domain_offset((q_offset, Int32(0)), mW)
            mdW_b = cute.domain_offset((q_offset, Int32(0)), mdW)
            mGradSignal_b = cute.domain_offset((q_offset, Int32(0)), mGradSignal)
            mTopkIdx_b = mTopkIdx
        else:
            mQ_b = mQ[None, None, None, batch_idx]
            mdQ_b = mdQ[None, None, None, batch_idx]
            mK_b = mK[None, None, batch_idx]
            mK_scalar_b = mKScalar[None, None, batch_idx]
            if const_expr(self.deterministic):
                mdK_b = cute.domain_offset((batch_idx * seqlen_k_static, Int32(0)), mdK_f32)
            else:
                mdK_b = mdK_f32[None, None, batch_idx]
            mW_b = mW[None, None, batch_idx]
            mdW_b = mdW[None, None, batch_idx]
            mGradSignal_b = mGradSignal[None, None, batch_idx]
            if const_expr(not self.is_dense):
                mTopkIdx_b = mTopkIdx[None, None, batch_idx]
            else:
                mTopkIdx_b = mTopkIdx

        gQ = cute.local_tile(mQ_b, (self.heads_padded, self.head_dim_padded), (0, 0, seq_idx))
        load_Q, _, _ = copy_ops.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
        gdQ = cute.local_tile(mdQ_b, (self.heads_padded, self.head_dim_padded), (0, 0, seq_idx))
        store_dQ, _, _ = copy_ops.tma_get_copy_fn(tma_atom_dQ, 0, cute.make_layout(1), sQ, gdQ, single_stage=True)

        if warp_idx == 0:
            if const_expr(self.is_dense):
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, 1)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, 1)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, 1)
            else:
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_0, 2)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_1, 2)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_2, 2)
            if const_expr(not self.is_dense):
                cute.arch.mbarrier_init(mbar + MBAR_INDICES_READY_0, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_INDICES_READY_1, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_INDICES_READY_2, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_Q_TMA, 1)
        cute.arch.sync_threads()

        # THD launches a rectangular grid over max_seqlen_q; CTAs past the
        # batch-local length must do no work.
        if seq_idx < seqlen_q_b:
            if warp_group_idx == self.COMPUTE_WG_A:
                self._compute_warpgroup(
                    mQ_b,
                    mW_b,
                    mK_b,
                    mdQ_b,
                    mdW_b,
                    mdK_b,
                    mGradSignal_b,
                    sQ,
                    sQt,
                    sK,
                    sKt,
                    sdS,
                    sdSt,
                    sGradSignal,
                    sIndices0,
                    sIndices1,
                    sIndices2,
                    sW,
                    sDwPartial,
                    sDwContrib,
                    sDwHead,
                    sDwWarpPartial,
                    sdK_staging,
                    tmma1,
                    tmma2,
                    tmma3,
                    load_Q,
                    store_dQ,
                    sm_scale,
                    seq_idx,
                    batch_idx,
                    seqlen_q_b,
                    seqlen_k_b,
                    q_causal_offset,
                    tidx,
                    warp_idx,
                    mbar,
                    0,
                )
            if warp_group_idx == self.COMPUTE_WG_B:
                self._compute_warpgroup(
                    mQ_b,
                    mW_b,
                    mK_b,
                    mdQ_b,
                    mdW_b,
                    mdK_b,
                    mGradSignal_b,
                    sQ,
                    sQt,
                    sK,
                    sKt,
                    sdS,
                    sdSt,
                    sGradSignal,
                    sIndices0,
                    sIndices1,
                    sIndices2,
                    sW,
                    sDwPartial,
                    sDwContrib,
                    sDwHead,
                    sDwWarpPartial,
                    sdK_staging,
                    tmma1,
                    tmma2,
                    tmma3,
                    load_Q,
                    store_dQ,
                    sm_scale,
                    seq_idx,
                    batch_idx,
                    seqlen_q_b,
                    seqlen_k_b,
                    q_causal_offset,
                    tidx,
                    warp_idx,
                    mbar,
                    1,
                )
            if warp_group_idx == self.KLOAD_WG:
                if const_expr(self.is_dense):
                    self._k_load_warpgroup_dense_inline(
                        mK_b,
                        mK_scalar_b,
                        sK,
                        batch_idx,
                        seq_idx,
                        seqlen_q_b,
                        seqlen_k_b,
                        q_causal_offset,
                        tidx,
                        mbar,
                        tma_atom_K,
                    )
                else:
                    self._k_load_warpgroup(
                        mK_b,
                        mTopkIdx_b,
                        sK,
                        sIndices0,
                        sIndices1,
                        sIndices2,
                        batch_idx,
                        seq_idx,
                        seqlen_k_b,
                        tidx,
                        mbar,
                    )

    # =========================================================================
    # WG0/WG1: Cooperative Compute warpgroups
    # =========================================================================
    @cute.jit
    def _compute_warpgroup(
        self,
        mQ,
        mW,
        mK,
        mdQ,
        mdW,
        mdK_f32,
        mGradSignal,
        sQ,
        sQt,
        sK,
        sKt,
        sdS,
        sdSt,
        sGradSignal,
        sIndices0,
        sIndices1,
        sIndices2,
        sW,
        sDwPartial,
        sDwContrib,
        sDwHead,
        sDwWarpPartial,
        sdK_staging,
        tmma1,
        tmma2,
        tmma3,
        load_Q,
        store_dQ,
        sm_scale: Float32 | float,
        seq_idx,
        batch_idx,
        seqlen_q,
        seqlen_k,
        q_causal_offset,
        tidx,
        warp_idx,
        mbar,
        compute_wg_idx,
    ):
        """Run one compute warpgroup and its staged dK reduction."""
        wg_tidx = tidx % self.WARPGROUP_SIZE
        dk_staging_barrier_id = self.DK_STAGING_BARRIER_WG0 if compute_wg_idx == 0 else self.DK_STAGING_BARRIER_WG1

        # ---- Step 0: Load Q (TMA), grad_signal, weights — 256 threads cooperate ----
        if const_expr(self.use_deterministic_dw_quad):
            # Eight compute warps publish one partial per head. Initialize all
            # 8x64 slots here and reuse _load_q_and_scores' final compute-group
            # barrier instead of adding a deterministic-only synchronization.
            for partial_idx in cutlass.range_constexpr(2):
                sDwWarpPartial[partial_idx * self.TOTAL_COMPUTE_THREADS + tidx] = Float32(0.0)
        self._load_q_and_scores(
            mW,
            mGradSignal,
            sGradSignal,
            sW,
            load_Q,
            seq_idx,
            batch_idx,
            tidx,
            warp_idx,
            mbar,
        )

        # ---- Step 1: Setup per-WG SMEM half-views and GEMM partitions ----

        # GEMM1: split N=I → each WG gets half of sK in I dimension (3-stage)
        sK_s0_full = cute.composition(sK[None, None, 0], cute.make_layout((self.block_I, self.head_dim_padded)))
        sK_s1_full = cute.composition(sK[None, None, 1], cute.make_layout((self.block_I, self.head_dim_padded)))
        sK_s2_full = cute.composition(sK[None, None, 2], cute.make_layout((self.block_I, self.head_dim_padded)))
        sK_s0_half = cute.local_tile(sK_s0_full, (self.half_block_I, self.head_dim_padded), (compute_wg_idx, 0))
        sK_s1_half = cute.local_tile(sK_s1_full, (self.half_block_I, self.head_dim_padded), (compute_wg_idx, 0))
        sK_s2_half = cute.local_tile(sK_s2_full, (self.half_block_I, self.head_dim_padded), (compute_wg_idx, 0))

        thr_mma1 = tmma1.get_slice(wg_tidx)
        tSrQ = thr_mma1.make_fragment_A(thr_mma1.partition_A(sQ))
        tSrK_s0 = thr_mma1.make_fragment_B(thr_mma1.partition_B(sK_s0_half))
        tSrK_s1 = thr_mma1.make_fragment_B(thr_mma1.partition_B(sK_s1_half))
        tSrK_s2 = thr_mma1.make_fragment_B(thr_mma1.partition_B(sK_s2_half))

        s_acc_shape = tmma1.partition_shape_C(self.gemm1_tiler_half)
        cS_half = cute.make_identity_tensor(self.gemm1_tiler_half)
        tCcS_half = thr_mma1.partition_C(cS_half)

        # GEMM2: split M=D → each WG gets half of sQt in D dimension
        sQt_half = cute.local_tile(sQt, (64, self.heads_padded), (compute_wg_idx, 0))

        thr_mma2 = tmma2.get_slice(wg_tidx)
        tDKrQt_half = thr_mma2.make_fragment_A(thr_mma2.partition_A(sQt_half))
        tDKrDSt = thr_mma2.make_fragment_B(thr_mma2.partition_B(sdSt))
        dk_acc_shape = tmma2.partition_shape_C(self.gemm2_tiler_half)

        # GEMM3: split N=D → each WG gets half of sKt in D dimension (3-stage)
        sKt_s0_full = cute.composition(sKt[None, None, 0], cute.make_layout((self.head_dim_padded, self.block_I)))
        sKt_s1_full = cute.composition(sKt[None, None, 1], cute.make_layout((self.head_dim_padded, self.block_I)))
        sKt_s2_full = cute.composition(sKt[None, None, 2], cute.make_layout((self.head_dim_padded, self.block_I)))
        sKt_s0_half = cute.local_tile(sKt_s0_full, (self.half_head_dim, self.block_I), (compute_wg_idx, 0))
        sKt_s1_half = cute.local_tile(sKt_s1_full, (self.half_head_dim, self.block_I), (compute_wg_idx, 0))
        sKt_s2_half = cute.local_tile(sKt_s2_full, (self.half_head_dim, self.block_I), (compute_wg_idx, 0))

        thr_mma3 = tmma3.get_slice(wg_tidx)
        tDQrDS = thr_mma3.make_fragment_A(thr_mma3.partition_A(sdS))
        tDQrKt_s0 = thr_mma3.make_fragment_B(thr_mma3.partition_B(sKt_s0_half))
        tDQrKt_s1 = thr_mma3.make_fragment_B(thr_mma3.partition_B(sKt_s1_half))
        tDQrKt_s2 = thr_mma3.make_fragment_B(thr_mma3.partition_B(sKt_s2_half))
        dq_acc_shape = tmma3.partition_shape_C(self.gemm3_tiler_half)
        cDQ_half = cute.make_identity_tensor(self.gemm3_tiler_half)
        tCcDQ_half = thr_mma3.partition_C(cDQ_half)

        sdS_view = cute.composition(sdS, cute.make_layout((self.heads_padded, self.block_I)))

        # Persistent dQ accumulator (halved: each WG owns D/2)
        acc_dQ = cute.make_rmem_tensor(dq_acc_shape, Float32)
        for ei in cutlass.range(0, cute.size(acc_dQ), unroll=32):
            acc_dQ[ei] = Float32(0.0)

        # dW: 2-register per-thread accumulation
        my_first_h = cute.get(tCcS_half[0], mode=[0])
        my_second_h = my_first_h
        for ei in cutlass.range(0, cute.size(s_acc_shape), unroll=32):
            h_check = cute.get(tCcS_half[ei], mode=[0])
            if h_check != my_first_h:
                my_second_h = h_check
        dw_h0 = Float32(0.0)
        dw_h1 = Float32(0.0)

        # Preload per-head weight values into registers (each thread touches only 2 h-values)
        w_reg_h0 = Float32(sW[my_first_h]) if my_first_h < self.heads else Float32(0.0)
        w_reg_h1 = Float32(sW[my_second_h]) if my_second_h < self.heads else Float32(0.0)

        # Ordinary execution merges dW through CTA-local atomics. Deterministic
        # execution uses either private slots or lane-quad warp partials below,
        # so neither variant needs this shared accumulator initialization.
        DW_PER_THREAD = const_expr((self.heads_padded + self.TOTAL_COMPUTE_THREADS - 1) // self.TOTAL_COMPUTE_THREADS)
        if const_expr(not self.deterministic):
            for di in cutlass.range_constexpr(DW_PER_THREAD):
                dw_init_idx = di * self.TOTAL_COMPUTE_THREADS + tidx
                if dw_init_idx < self.heads_padded:
                    sDwPartial[dw_init_idx] = Float32(0.0)
            cute.arch.fence_view_async_shared()
            nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

        n_offset = compute_wg_idx * self.half_block_I

        # shared-mem store dK staging: partition sdK_staging using GEMM2's MMA layout
        sdK_staging_half = cute.local_tile(sdK_staging, (64, self.block_I), (compute_wg_idx, 0))
        tCsDK_staging = thr_mma2.partition_C(sdK_staging_half)

        # Fused pass sdS write via stmatrix (r2s bulk copy, replaces scalar shared-mem stores)
        sdS_half = cute.local_tile(sdS_view, (self.heads_padded, self.half_block_I), (0, compute_wg_idx))
        stmatrix_atom_ds = cute.make_copy_atom(warp.StMatrix8x8x16bOp(), self.q_dtype)
        tiled_r2s_ds = cute.make_tiled_copy_C(stmatrix_atom_ds, tmma1)
        thr_r2s_ds = tiled_r2s_ds.get_slice(wg_tidx)
        tRdDS = thr_r2s_ds.partition_D(sdS_half)

        # Pass 4: 2D block view of sGradSignal for partition-based access (sparse only)
        if const_expr(not self.is_dense):
            sGS_per_block = cute.make_tensor(
                sGradSignal.iterator,
                cute.make_layout(
                    (self.block_I, self.num_topk_blocks),
                    stride=(1, self.block_I),
                ),
            )

        k_loaded_0_phase = Int32(0)
        k_loaded_1_phase = Int32(0)
        k_loaded_2_phase = Int32(0)

        # Ping-pong init: WG1 pre-arrives on WG0's scheduler barrier,
        # giving WG0 the head start for the first sync.
        if compute_wg_idx == 1:
            nvvm.barrier_cta_arrive(self.SCHED_BARRIER_WG0, self.TOTAL_COMPUTE_THREADS)

        if const_expr(self.is_dense):
            num_topk_blocks_cur = self._dense_num_k_blocks(seq_idx, seqlen_q, seqlen_k, q_causal_offset)
        else:
            num_topk_blocks_cur = Int32(self.num_topk_blocks)

        # ---- Step 3: Main loop over topk blocks ----
        for bi in cutlass.range(0, num_topk_blocks_cur, unroll=1):
            i_st = bi * self.block_I
            stage = bi % NUM_K_STAGES

            # Dense: load grad_signal for this block into sGradSignal (block_I floats)
            if const_expr(self.is_dense):
                DENSE_GS_PER_THREAD = const_expr((self.block_I + self.TOTAL_COMPUTE_THREADS - 1) // self.TOTAL_COMPUTE_THREADS)
                for gi in cutlass.range_constexpr(DENSE_GS_PER_THREAD):
                    pos = gi * self.TOTAL_COMPUTE_THREADS + tidx
                    if pos < self.block_I:
                        k_pos = i_st + pos
                        if k_pos < seqlen_k:
                            sGradSignal[pos] = mGradSignal[seq_idx, k_pos]
                        else:
                            sGradSignal[pos] = Float32(0.0)
                cute.arch.fence_view_async_shared()
                nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

            # Wait for K data from K-load WG
            if stage == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_0, k_loaded_0_phase)
                k_loaded_0_phase ^= 1
            elif stage == 1:
                cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_1, k_loaded_1_phase)
                k_loaded_1_phase ^= 1
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_K_LOADED_2, k_loaded_2_phase)
                k_loaded_2_phase ^= 1

            # ----- GEMM1: S[H, I/2] = Q[H, D] x K_half[I/2, D] -----
            # Ping-pong: wait for turn → issue WGMMA → signal other WG → wait result
            if compute_wg_idx == 0:
                nvvm.barrier_cta_sync(self.SCHED_BARRIER_WG0, thread_count=self.TOTAL_COMPUTE_THREADS)
            else:
                nvvm.barrier_cta_sync(self.SCHED_BARRIER_WG1, thread_count=self.TOTAL_COMPUTE_THREADS)

            acc_S = cute.make_rmem_tensor(s_acc_shape, Float32)
            if stage == 0:
                gemm(tmma1, acc_S, tSrQ, tSrK_s0, zero_init=True, wg_wait=-1)
            elif stage == 1:
                gemm(tmma1, acc_S, tSrQ, tSrK_s1, zero_init=True, wg_wait=-1)
            else:
                gemm(tmma1, acc_S, tSrQ, tSrK_s2, zero_init=True, wg_wait=-1)

            if compute_wg_idx == 0:
                nvvm.barrier_cta_arrive(self.SCHED_BARRIER_WG1, self.TOTAL_COMPUTE_THREADS)
            else:
                nvvm.barrier_cta_arrive(self.SCHED_BARRIER_WG0, self.TOTAL_COMPUTE_THREADS)

            warpgroup.wait_group(0)

            # ----- Fused pass: compute dS → registers, then stmatrix → sdS -----
            # Pass 4: broadcast partition of sGradSignal — eliminates n-coordinate cute.get
            if const_expr(self.is_dense):
                sGS_cur_block = sGradSignal
            else:
                sGS_cur_block = sGS_per_block[None, bi]
            sGS_half_chunks = cute.flat_divide(sGS_cur_block, (self.half_block_I,))
            sGS_broadcast = cute.make_tensor(
                sGS_half_chunks[None, compute_wg_idx].iterator,
                cute.make_layout(
                    (self.heads_padded, self.half_block_I),
                    stride=(0, 1),
                ),
            )
            tCsGS = thr_mma1.partition_C(sGS_broadcast)

            acc_dS = cute.make_rmem_tensor(s_acc_shape, self.q_dtype)
            for ei in cutlass.range(0, cute.size(acc_S), unroll=32):
                h = cute.get(tCcS_half[ei], mode=[0])
                s_val = acc_S[ei] * Float32(sm_scale)

                grad_signal = Float32(tCsGS[ei])
                w_val = w_reg_h0 if h == my_first_h else w_reg_h1
                is_pos = s_val > Float32(0.0)
                ds_val = (grad_signal * w_val) if is_pos else Float32(0.0)

                acc_dS[ei] = self.q_dtype(ds_val)

                dw_val = (grad_signal * s_val) if is_pos else Float32(0.0)
                if h == my_first_h:
                    dw_h0 = dw_h0 + dw_val
                else:
                    dw_h1 = dw_h1 + dw_val

            # Bulk write dS to SMEM via stmatrix (replaces per-element scalar shared-mem stores)
            tRsDS = tiled_r2s_ds.retile(acc_dS)
            cute.copy(tiled_r2s_ds, tRsDS, tRdDS)
            cute.arch.fence_view_async_shared()
            # Both WGs must finish writing their half of sdS before GEMM2/3 read full sdS
            nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

            # ----- GEMM2 + GEMM3 (ping-pong, pipeline depth 2) -----
            # Ping-pong: wait for turn
            if compute_wg_idx == 0:
                nvvm.barrier_cta_sync(self.SCHED_BARRIER_WG0, thread_count=self.TOTAL_COMPUTE_THREADS)
            else:
                nvvm.barrier_cta_sync(self.SCHED_BARRIER_WG1, thread_count=self.TOTAL_COMPUTE_THREADS)

            # GEMM2: dK[D/2, I] = Qt_half[D/2, H] x dSt[I, H]
            acc_dK = cute.make_rmem_tensor(dk_acc_shape, Float32)
            gemm(tmma2, acc_dK, tDKrQt_half, tDKrDSt, zero_init=True, wg_wait=-1)

            # GEMM3: dQ[H, D/2] += dS[H, I] x Kt_half[D/2, I]
            if stage == 0:
                gemm(tmma3, acc_dQ, tDQrDS, tDQrKt_s0, zero_init=False, wg_wait=-1)
            elif stage == 1:
                gemm(tmma3, acc_dQ, tDQrDS, tDQrKt_s1, zero_init=False, wg_wait=-1)
            else:
                gemm(tmma3, acc_dQ, tDQrDS, tDQrKt_s2, zero_init=False, wg_wait=-1)

            # Signal other WG: it can start its GEMM2+3 while we do shared-mem stores / memory
            if compute_wg_idx == 0:
                nvvm.barrier_cta_arrive(self.SCHED_BARRIER_WG1, self.TOTAL_COMPUTE_THREADS)
            else:
                nvvm.barrier_cta_arrive(self.SCHED_BARRIER_WG0, self.TOTAL_COMPUTE_THREADS)

            warpgroup.wait_group(1)  # GEMM2 done (acc_dK ready to write to shared mem)

            # Deferred DMA wait: previous iteration's bulk reduce must finish
            # reading sdK_staging before we overwrite it (both WGs do their own reduce)
            cute.arch.cp_async_bulk_wait_group(0, read=True)

            # The bulk-group wait is thread-local, while the MMA accumulator is
            # scattered across all threads in the two compute WGs.  Make sure
            # every thread has finished its previous staging-buffer read before
            # any thread starts overwriting fragments owned by a different row.
            nvvm.barrier_cta_sync(dk_staging_barrier_id, thread_count=self.WARPGROUP_SIZE)

            # Stage acc_dK × sm_scale to SMEM via partition (no cute.get needed)
            for ei in cutlass.range(0, cute.size(acc_dK), unroll=64):
                tCsDK_staging[ei] = acc_dK[ei] * Float32(sm_scale)

            warpgroup.wait_group(0)  # GEMM3 done — sK no longer needed

            cute.arch.fence_view_async_shared()

            # A row's 64 FP32 values are produced by multiple threads, while
            # the row-reduce launchers issue one bulk reduce per row. The proxy
            # fence publishes each producer's writes; this execution barrier
            # makes every launcher wait for all producers in the warpgroup.
            nvvm.barrier_cta_sync(dk_staging_barrier_id, thread_count=self.WARPGROUP_SIZE)

            # Per-WG dK bulk reduce: each WG handles its D/2 half independently
            my_ni = wg_tidx
            if my_ni < self.block_I:
                if const_expr(self.is_dense):
                    topk_idx = i_st + my_ni
                else:
                    topk_idx = Int32(sIndices0[my_ni]) if stage == 0 else (Int32(sIndices1[my_ni]) if stage == 1 else Int32(sIndices2[my_ni]))
                if topk_idx >= 0 and topk_idx < seqlen_k:
                    gdK_row = mdK_f32[topk_idx, None]
                    gdK_half = cute.local_tile(gdK_row, (self.half_head_dim,), (compute_wg_idx,))
                    copy_ops.cpasync_reduce_bulk_add_f32(
                        sdK_staging_half[None, my_ni].iterator,
                        gdK_half.iterator,
                        self.half_head_dim * 4,
                    )
            cute.arch.cp_async_bulk_commit_group()

            # Signal K_CONSUMED: sK can be reused by K-load WG
            if wg_tidx == 0:
                if stage == 0:
                    cute.arch.mbarrier_arrive(mbar + MBAR_K_CONSUMED_0)
                elif stage == 1:
                    cute.arch.mbarrier_arrive(mbar + MBAR_K_CONSUMED_1)
                else:
                    cute.arch.mbarrier_arrive(mbar + MBAR_K_CONSUMED_2)

        # Wait for final iteration's bulk reduce DMA (both WGs do their own reduce)
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

        # ---- Step 4: Epilogue — write dQ and dW to GMEM ----

        # Stage acc_dQ × sm_scale as bf16 to sQ (reuse sQ as TMA S2G epilogue buffer;
        # Q data no longer needed). Each WG writes its D-half.
        sQ_half = cute.local_tile(sQ, (self.heads_padded, self.half_head_dim), (0, compute_wg_idx))
        tCsDQ = thr_mma3.partition_C(sQ_half)

        for ei in cutlass.range(0, cute.size(acc_dQ), unroll=32):
            tCsDQ[ei] = self.q_dtype(acc_dQ[ei] * Float32(sm_scale))

        if const_expr(self.deterministic):
            if const_expr(self.use_deterministic_dw_quad):
                # Hopper WGMMA's H64 C layout maps
                # (T128,V64)->(M64,N128) with thread modes
                # (_4,_8,_4):(128,1,16): every four consecutive lanes own the
                # same two M/head coordinates (eight rows apart). Fold each
                # head in ascending lane order, matching the private-slot
                # association. Keep the distinct-head guard as a tripwire for
                # a future layout that exposes only one head per lane.
                # All lanes execute every shuffle; only each quad leader
                # publishes.
                compute_warp_idx = tidx // self.WARP_SIZE
                lane_idx = tidx % self.WARP_SIZE
                quad_base_lane = (lane_idx // 4) * 4
                dw_warp_h0 = Float32(0.0)
                dw_warp_h1 = Float32(0.0)
                for source_offset in cutlass.range_constexpr(4):
                    source_lane = quad_base_lane + source_offset
                    dw_warp_h0 += cute.arch.shuffle_sync(dw_h0, source_lane)
                    dw_warp_h1 += cute.arch.shuffle_sync(dw_h1, source_lane)
                if lane_idx % 4 == 0:
                    sDwWarpPartial[compute_warp_idx * self.heads + my_first_h] = dw_warp_h0
                    if my_second_h != my_first_h:
                        sDwWarpPartial[compute_warp_idx * self.heads + my_second_h] = dw_warp_h1
            else:
                # Small top-k keeps the lower-overhead private-slot fold.
                slot = tidx * 2
                sDwContrib[slot] = dw_h0
                sDwHead[slot] = Int32(my_first_h)
                sDwContrib[slot + 1] = dw_h1
                sDwHead[slot + 1] = Int32(my_second_h)
        else:
            # dW ordinary fast path: merge the two-register accumulators via
            # shared-memory atomics inside this CTA.
            dw_part = cute.flat_divide(sDwPartial, (1,))
            if my_first_h < self.heads:
                cute.arch.atomic_add(dw_part[None, my_first_h].iterator.llvm_ptr, dw_h0)
            if my_second_h != my_first_h and my_second_h < self.heads:
                cute.arch.atomic_add(dw_part[None, my_second_h].iterator.llvm_ptr, dw_h1)

        cute.arch.fence_view_async_shared()
        nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

        if const_expr(self.deterministic and not self.use_deterministic_dw_quad):
            # Fold each warp's private slots in increasing numeric order. The
            # second fixed eight-warp fold below preserves the original small
            # top-k association.
            compute_warp_idx = tidx // self.WARP_SIZE
            lane_idx = tidx % self.WARP_SIZE
            slots_per_warp = self.WARP_SIZE * 2
            for head_pass in cutlass.range_constexpr(2):
                dw_partial_head = lane_idx + head_pass * self.WARP_SIZE
                dw_warp_sum = Float32(0.0)
                for local_slot_idx in cutlass.range(0, slots_per_warp, unroll=1):
                    contribution_idx = compute_warp_idx * slots_per_warp + local_slot_idx
                    if sDwHead[contribution_idx] == dw_partial_head:
                        dw_warp_sum += sDwContrib[contribution_idx]
                sDwWarpPartial[compute_warp_idx * self.heads + dw_partial_head] = dw_warp_sum
            cute.arch.fence_view_async_shared()
            nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

        # dQ: TMA S2G bulk copy (1 instruction replaces per-element scalar stores)
        if warp_idx == 0:
            with cute.arch.elect_one():
                store_dQ()
                cute.arch.cp_async_bulk_commit_group()
        if compute_wg_idx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        # Only WG0 writes dW to GMEM.
        if compute_wg_idx == 0:
            for di in cutlass.range_constexpr(DW_PER_THREAD):
                dw_output_idx = di * self.TOTAL_COMPUTE_THREADS + tidx
                if dw_output_idx < self.heads:
                    if const_expr(self.deterministic):
                        dw_sum = Float32(0.0)
                        for partial_warp_idx in cutlass.range_constexpr(self.TOTAL_COMPUTE_THREADS // self.WARP_SIZE):
                            dw_sum += sDwWarpPartial[partial_warp_idx * self.heads + dw_output_idx]
                        mdW[seq_idx, dw_output_idx] = self.q_dtype(dw_sum)
                    else:
                        mdW[seq_idx, dw_output_idx] = self.q_dtype(sDwPartial[dw_output_idx])

    # =========================================================================
    # Q (TMA) / grad_signal / weights cooperative loading (both compute WGs)
    # =========================================================================
    @cute.jit
    def _load_q_and_scores(
        self,
        mW,
        mGradSignal,
        sGradSignal,
        sW,
        load_Q,
        seq_idx,
        batch_idx,
        tidx,
        warp_idx,
        mbar,
    ):
        # Q via TMA: warp 0 of WG0 issues the copy
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + MBAR_Q_TMA, self.tma_copy_Q_bytes)
            load_Q(tma_bar_ptr=mbar + MBAR_Q_TMA)

        # GradSignal: sparse loads all upfront; dense loads per-block in main loop
        if const_expr(not self.is_dense):
            GS_PER_THREAD = const_expr((self.topk + self.TOTAL_COMPUTE_THREADS - 1) // self.TOTAL_COMPUTE_THREADS)
            for si in cutlass.range_constexpr(GS_PER_THREAD):
                pos = si * self.TOTAL_COMPUTE_THREADS + tidx
                if pos < self.topk:
                    sGradSignal[pos] = mGradSignal[seq_idx, pos]

        W_PER_THREAD = const_expr((self.heads + self.TOTAL_COMPUTE_THREADS - 1) // self.TOTAL_COMPUTE_THREADS)
        for wi in cutlass.range_constexpr(W_PER_THREAD):
            idx = wi * self.TOTAL_COMPUTE_THREADS + tidx
            if idx < self.heads:
                sW[idx] = mW[seq_idx, idx]

        cute.arch.fence_view_async_shared()
        nvvm.barrier_cta_sync(self.COMPUTE_SYNC_BARRIER_ID, thread_count=self.TOTAL_COMPUTE_THREADS)

        # Wait for Q TMA to complete
        cute.arch.mbarrier_wait(mbar + MBAR_Q_TMA, Int32(0))

    # =========================================================================
    # WG2: K-load warpgroup (3-stage sK + triple-buffered indices)
    # =========================================================================
    @cute.jit
    def _k_load_warpgroup(
        self,
        mK,
        mTopkIdx,
        sK,
        sIndices0,
        sIndices1,
        sIndices2,
        batch_idx,
        seq_idx,
        seqlen_k,
        tidx,
        mbar,
    ):
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

        sK_slice_0 = cute.composition(
            sK[None, None, 0],
            cute.make_layout((self.block_I, self.head_dim_padded)),
        )
        sK_slice_1 = cute.composition(
            sK[None, None, 1],
            cute.make_layout((self.block_I, self.head_dim_padded)),
        )
        sK_slice_2 = cute.composition(
            sK[None, None, 2],
            cute.make_layout((self.block_I, self.head_dim_padded)),
        )
        k_consumed_0_phase = Int32(0)
        k_consumed_1_phase = Int32(0)
        k_consumed_2_phase = Int32(0)
        indices_0_phase = Int32(0)
        indices_1_phase = Int32(0)
        indices_2_phase = Int32(0)

        for bi in cutlass.range_constexpr(self.num_topk_blocks):
            stage = bi % NUM_K_STAGES

            # Back-pressure: wait for compute WG to finish with sK[stage]
            if bi >= NUM_K_STAGES:
                if stage == 0:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_0, k_consumed_0_phase)
                    k_consumed_0_phase ^= 1
                elif stage == 1:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_1, k_consumed_1_phase)
                    k_consumed_1_phase ^= 1
                else:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_2, k_consumed_2_phase)
                    k_consumed_2_phase ^= 1

            # Load indices for this block
            i_st = bi * self.block_I
            sIndices = sIndices0 if stage == 0 else (sIndices1 if stage == 1 else sIndices2)

            batch_offset = batch_idx * seqlen_k if const_expr(self.topk_indices_global) else Int32(0)
            IDX_PER_THREAD = const_expr((self.block_I + self.WARPGROUP_SIZE - 1) // self.WARPGROUP_SIZE)
            for ii in cutlass.range_constexpr(IDX_PER_THREAD):
                pos = ii * self.WARPGROUP_SIZE + wg_tidx
                if pos < self.block_I:
                    sIndices[pos] = Int32(mTopkIdx[seq_idx, i_st + pos]) - batch_offset

            cute.arch.fence_view_async_shared()
            if stage == 0:
                cute.arch.mbarrier_arrive(mbar + MBAR_INDICES_READY_0)
            elif stage == 1:
                cute.arch.mbarrier_arrive(mbar + MBAR_INDICES_READY_1)
            else:
                cute.arch.mbarrier_arrive(mbar + MBAR_INDICES_READY_2)

            # Self-wait to ensure indices are visible to all threads in WG
            if stage == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_INDICES_READY_0, indices_0_phase)
                indices_0_phase ^= 1
            elif stage == 1:
                cute.arch.mbarrier_wait(mbar + MBAR_INDICES_READY_1, indices_1_phase)
                indices_1_phase ^= 1
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_INDICES_READY_2, indices_2_phase)
                indices_2_phase ^= 1

            sK_slice = sK_slice_0 if stage == 0 else (sK_slice_1 if stage == 1 else sK_slice_2)

            # Sparse gather: each group of 8 threads loads one K row
            for r in cutlass.range_constexpr(ROWS_PER_GROUP):
                row = r * NUM_GROUPS + group_idx_local
                topk_idx = Int32(sIndices[row])
                if topk_idx >= 0 and topk_idx < seqlen_k:
                    gK_raw = mK[topk_idx, None]
                    gK = cute.make_tensor(
                        cute.make_ptr(
                            self.k_dtype,
                            gK_raw.iterator.llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
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

            if stage == 0:
                cute.arch.mbarrier_arrive(mbar + MBAR_K_LOADED_0)
            elif stage == 1:
                cute.arch.mbarrier_arrive(mbar + MBAR_K_LOADED_1)
            else:
                cute.arch.mbarrier_arrive(mbar + MBAR_K_LOADED_2)

    @cute.jit
    def _k_load_warpgroup_dense_inline(
        self,
        mK,
        mKScalar,
        sK,
        batch_idx,
        seq_idx,
        seqlen_q,
        seqlen_k,
        q_causal_offset,
        tidx,
        mbar,
        tma_atom_K,
    ):
        """Dense mode K loading via TMA (sequential blocks, no scatter-gather)."""
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        wg_tidx = tidx % self.WARPGROUP_SIZE

        k_consumed_0_phase = Int32(0)
        k_consumed_1_phase = Int32(0)
        k_consumed_2_phase = Int32(0)

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

        num_k_blocks = self._dense_num_k_blocks(seq_idx, seqlen_q, seqlen_k, q_causal_offset)
        for bi in cutlass.range(0, num_k_blocks, unroll=1):
            stage = bi % NUM_K_STAGES
            n_block = bi

            if bi >= NUM_K_STAGES:
                if stage == 0:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_0, k_consumed_0_phase)
                    k_consumed_0_phase ^= 1
                elif stage == 1:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_1, k_consumed_1_phase)
                    k_consumed_1_phase ^= 1
                else:
                    cute.arch.mbarrier_wait(mbar + MBAR_K_CONSUMED_2, k_consumed_2_phase)
                    k_consumed_2_phase ^= 1

            # Runtime stage → select the SMEM K-stage view and K_LOADED barrier
            # by indexing/arithmetic; a constexpr-stage ternary doesn't lower in
            # this runtime loop. K_LOADED_{0,1,2} are consecutive barriers.
            sK_slice = cute.composition(
                sK[None, None, stage],
                cute.make_layout((self.block_I, self.head_dim_padded)),
            )
            mbar_k = mbar + MBAR_K_LOADED_0 + stage

            block_end = (n_block + 1) * self.block_I
            if block_end <= seqlen_k:
                gK_tile = cute.local_tile(
                    mK,
                    (self.block_I, self.head_dim_padded),
                    (n_block, 0),
                )
                load_fn, _, _ = copy_ops.tma_get_copy_fn(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    gK_tile,
                    sK_slice,
                    single_stage=True,
                )

                if warp_idx_in_wg == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_k,
                            self.tma_copy_K_bytes,
                        )
                    load_fn(tma_bar_ptr=mbar_k)
            else:
                for r in cutlass.range_constexpr(ROWS_PER_GROUP):
                    row = r * NUM_GROUPS + group_idx_local
                    k_pos = n_block * self.block_I + row
                    if k_pos < seqlen_k:
                        gK_raw = mKScalar[k_pos, None]
                        gK = cute.make_tensor(
                            cute.make_ptr(
                                self.k_dtype,
                                gK_raw.iterator.llvm_ptr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
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
                if warp_idx_in_wg == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_k)


# =============================================================================
# Factory
# =============================================================================
# compile_key -> compiled GEMM kernel. Single-layer cache (matches the forward
# ``_interface.py`` and ``_score_grad_cute_cache``): the key holds only params
# that change the generated code, and entries are filled lazily on first
# execute (``_ensure_compiled``) because ``cute.compile`` needs real tensors.
_compile_cache: dict = {}


def indexer_backward_sm90(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale=1.0,
    block_I=128,
    topk_indices_global: bool = True,
    deterministic: bool = False,
    deterministic_dk_planes: int = DEFAULT_DK_PLANES,
):
    """Build the shape-polymorphic SM90 sparse IndexerBackward launcher.

    ``deterministic=True`` selects stream-ordered bounded waves and a fixed
    FP32 dK plane count from ``{96, 128, 148}``; the caller supplies the exact
    workspace through the returned launcher's ``workspace=`` argument. Batch
    and sequence extents remain runtime-only and do not specialize compilation.
    """
    # ``grad_scale`` is intentionally **not** an argument: it's a host scalar
    # passed to ``_run`` at call time (forwarded into the score-grad kernel as a
    # runtime ``Float32``). Keeping it out of the cache key avoids spurious
    # recompiles.
    #
    # ``batch``/``seqlen``/``seqlen_k`` stay in the signature (the ``api.py``
    # call site passes them positionally) but are **not** keyed: in the kernel
    # they are runtime-only (dynamic tensor extents + ``Int32`` grid/args, with
    # ``seqlen``/``seqlen_k`` forwarded as ``cutlass.Int32`` at call time), so a
    # single compiled kernel serves every shape. Keying them would force
    # recompiles under varlen / changing batch.
    #
    # Score input is probability-domain predict, as on SM100.
    score_input_is_log = False
    deterministic_dk_planes = validate_deterministic_dk_plane_policy(deterministic, deterministic_dk_planes)
    return _build_cute_dsl_kernel(
        batch,
        seqlen,
        seqlen_k,
        heads,
        dim,
        topk,
        sm_scale,
        block_I,
        score_input_is_log=score_input_is_log,
        topk_indices_global=topk_indices_global,
        deterministic=deterministic,
        deterministic_dk_planes=deterministic_dk_planes,
    )


class ScoreGradSm90:
    """CuTe DSL kernel for in-place score_grad precompute (SM90)."""

    THREADS_PER_CTA = 128

    def __init__(self, topk: int, index_is_log: bool):
        self.topk = topk
        self.index_is_log = index_is_log

    @cute.jit
    def __call__(
        self,
        mAttnScore: cute.Tensor,
        mIndexScore: cute.Tensor,
        mGradLoss: cute.Tensor,
        grad_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        # (b, s, t) -> (s, t, b): contiguous topk traversal per CTA.
        mAttnScore = cute.make_tensor(mAttnScore.iterator, cute.select(mAttnScore.layout, mode=[1, 2, 0]))
        mIndexScore = cute.make_tensor(mIndexScore.iterator, cute.select(mIndexScore.layout, mode=[1, 2, 0]))

        seqlen = cute.size(mAttnScore.shape[0])
        batch_size = cute.size(mAttnScore.shape[2]) if cute.rank(mAttnScore.shape) > 2 else 1
        self.kernel_score_grad(mAttnScore, mIndexScore, mGradLoss, grad_scale).launch(
            grid=(seqlen, batch_size, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel_score_grad(self, mAttnScore, mIndexScore, mGradLoss, grad_scale: Float32 | float):
        tidx = cute.arch.thread_idx()[0]
        seq_idx = cute.arch.block_idx()[0]
        batch_idx = cute.arch.block_idx()[1]
        # grad_scale is a compile/runtime scalar (loss_coeff / (b*sq));
        # grad_loss lives in a shape-(1,) f32 GPU tensor (from autograd).
        # Fold them together once per CTA — the compiler will hoist.
        grad_scale_f32 = Float32(grad_scale) * Float32(mGradLoss[0])

        @cute.struct
        class SharedStorage:
            thread_sums: cute.struct.Align[cute.struct.MemRange[Float32, self.THREADS_PER_CTA], 128]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        thread_sums = storage.thread_sums.get_tensor(cute.make_layout((self.THREADS_PER_CTA,), stride=(1,)))

        TOPK_PER_THREAD = const_expr((self.topk + self.THREADS_PER_CTA - 1) // self.THREADS_PER_CTA)
        local_sum = Float32(0.0)
        for ii in cutlass.range_constexpr(TOPK_PER_THREAD):
            pos = ii * self.THREADS_PER_CTA + tidx
            if pos < self.topk:
                target = Float32(mAttnScore[seq_idx, pos, batch_idx])
                target_eff = cute.arch.fmax(target, Float32(CLIP_PROB_MIN))
                if const_expr(self.index_is_log):
                    log_predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
                    log_clip_mask = Float32(1.0) if log_predict >= Float32(CLIP_LOG_MIN) else Float32(0.0)
                else:
                    predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
                    log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
                local_sum += -target_eff * log_clip_mask * grad_scale_f32

        thread_sums[tidx] = local_sum
        cute.arch.sync_threads()

        if tidx == 0:
            block_sum = Float32(0.0)
            for i in cutlass.range(self.THREADS_PER_CTA, unroll_full=True):
                block_sum += thread_sums[i]
            thread_sums[0] = block_sum
        cute.arch.sync_threads()

        sum_grad = thread_sums[0]
        for ii in cutlass.range_constexpr(TOPK_PER_THREAD):
            pos = ii * self.THREADS_PER_CTA + tidx
            if pos < self.topk:
                target = Float32(mAttnScore[seq_idx, pos, batch_idx])
                target_eff = cute.arch.fmax(target, Float32(CLIP_PROB_MIN))
                if const_expr(self.index_is_log):
                    log_predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
                    predict = cute.math.exp(log_predict, fastmath=True)
                    log_clip_mask = Float32(1.0) if log_predict >= Float32(CLIP_LOG_MIN) else Float32(0.0)
                else:
                    predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
                    log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
                g_i = -target_eff * log_clip_mask * grad_scale_f32
                mAttnScore[seq_idx, pos, batch_idx] = g_i - predict * sum_grad


def _score_grad_inplace_cute(
    AttnScore,
    IndexScore,
    GradLoss,
    grad_scale,
    index_is_log: bool = False,
    current_stream=None,
):
    # Kernel reads ``mGradLoss[0]`` so it must be at least 1-D; defend direct
    # factory callers passing a 0-D scalar (the public wrapper reshapes already).
    if GradLoss.ndim == 0:
        GradLoss = GradLoss.reshape(1)

    _, _, topk = AttnScore.shape
    compile_key = (topk, bool(index_is_log))
    s = _resolve_stream(current_stream)
    if compile_key not in _score_grad_cute_cache:
        kernel_obj = ScoreGradSm90(
            topk=topk,
            index_is_log=bool(index_is_log),
        )
        _score_grad_cute_cache[compile_key] = cute.compile(
            kernel_obj,
            to_cute_tensor(AttnScore),
            to_cute_tensor(IndexScore),
            to_cute_tensor(GradLoss),
            cutlass.Float32(float(grad_scale)),
            s,
            options=compile_options(),
        )

    _score_grad_cute_cache[compile_key](
        AttnScore,
        IndexScore,
        GradLoss,
        cutlass.Float32(float(grad_scale)),
        s,
    )


def _score_grad_inplace(
    AttnScore,
    IndexScore,
    GradLoss,
    grad_scale,
    index_is_log: bool = False,
    current_stream=None,
):
    """Kernel 1: Compute clipped-log KL grad_signal from target/predict.

    Results overwrite AttnScore with grad_signal in-place:
      AttnScore ← grad_signal   (per topk element)

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
    _score_grad_inplace_cute(
        AttnScore,
        IndexScore,
        GradLoss,
        grad_scale,
        index_is_log=index_is_log,
        current_stream=current_stream,
    )


def _build_cute_dsl_kernel(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale,
    block_I,
    score_input_is_log=True,
    topk_indices_global: bool = True,
    deterministic: bool = False,
    deterministic_dk_planes: int = DEFAULT_DK_PLANES,
):
    cap = torch.cuda.get_device_capability()[0]
    if cap < 9:
        raise RuntimeError(f"Requires SM90+ (got SM{cap}0)")
    if cap >= 10:
        raise RuntimeError("Use SM100 kernel for Blackwell")

    deterministic_dk_planes = validate_deterministic_dk_plane_policy(deterministic, deterministic_dk_planes)
    kernel_obj = IndexerBackwardSm90(
        head_dim=dim,
        heads=heads,
        block_I=block_I,
        topk=topk,
        topk_indices_global=topk_indices_global,
        deterministic=deterministic,
        deterministic_dk_planes=deterministic_dk_planes,
    )

    # Only params that change the generated code (mirrors forward _interface.py).
    # ``seqlen``/``seqlen_k``/``sm_scale`` are NOT keyed: they're forwarded as
    # runtime ``cutlass.Int32``/``Float32`` args below (seqlen* unused on the
    # sparse path).
    compile_key = (heads, dim, topk, block_I, score_input_is_log, topk_indices_global, deterministic, deterministic_dk_planes)

    def _ensure_compiled(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK_f32,
        GradSignal,
        TopkIndices,
        current_stream=None,
    ):
        """Lazy-compile the GEMM kernel (kernel 2) on first execute (needs real tensors)."""
        s = _resolve_stream(current_stream)
        if compile_key not in _compile_cache:
            cute_args = [
                to_cute_tensor(t)
                for t in [
                    IndexQ,
                    Weights,
                    IndexK,
                    dIndexQ,
                    dWeights,
                    dIndexK_f32,
                    GradSignal,
                    TopkIndices,
                ]
            ]

            # Pass dummy Int32 values for max_seqlen_q/k: the kernel's __call__
            # signature declares these as ``Int32 = None`` and the JIT cannot
            # cast ``None`` to ``Int32`` (raises DSLRuntimeError). They are
            # only consumed when is_varlen=True (dense + cu_seqlens), so the
            # values here are unused but must be valid Int32.
            _compile_cache[compile_key] = cute.compile(
                kernel_obj,
                *cute_args,
                cutlass.Float32(sm_scale),
                s,
                None,
                None,
                cutlass.Int32(seqlen),
                cutlass.Int32(seqlen_k),
                None,
                cutlass.Int32(0),
                options=compile_options(),
            )

    def _run_gemm_only(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK_f32,
        GradSignal,
        TopkIndices,
        current_stream=None,
    ):
        """Run only kernel 2 (GEMM). Caller must have run kernel 1 and zeroed dIndexK_f32."""
        s = _resolve_stream(current_stream)
        _ensure_compiled(
            IndexQ,
            Weights,
            IndexK,
            dIndexQ,
            dWeights,
            dIndexK_f32,
            GradSignal,
            TopkIndices,
            current_stream=current_stream,
        )
        total_rows = int(IndexQ.shape[0] * IndexQ.shape[1])
        row_bases = range(0, total_rows, deterministic_dk_planes) if deterministic else (0,)
        for row_base in row_bases:
            _compile_cache[compile_key](
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
                None,
                None,
                cutlass.Int32(seqlen),
                cutlass.Int32(seqlen_k),
                None,
                cutlass.Int32(row_base),
            )

    def _run(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        IndexScore,
        TopkIndices,
        GradLoss,
        grad_scale,
        current_stream=None,
        workspace=None,
    ):
        # ``grad_scale`` is a host scalar (Python float) forwarded as a
        # runtime ``Float32`` arg to the score-grad kernel; changing it
        # across calls does not trigger recompilation.
        s = _resolve_stream(current_stream)
        # Carve every caller-owned target before kernel 1 mutates AttnScore.
        # Deterministic execution always accumulates into fixed CTA-private
        # FP32 planes and folds them once in numeric plane order.
        if deterministic:
            planes_bytes = deterministic_dk_planes_bytes(batch, seqlen_k, dim, deterministic_dk_planes)
            carver = WorkspaceCarver(
                workspace,
                deterministic_dk_workspace_bytes(batch, seqlen_k, dim, deterministic_dk_planes),
                "IndexerBackward deterministic (sm90)",
            )
            dIndexK_f32 = carver.take(planes_bytes // 4, torch.float32).view(deterministic_dk_planes, batch * seqlen_k, dim)
        elif dIndexK.dtype == torch.float32:
            dIndexK_f32 = dIndexK
        else:
            carver = WorkspaceCarver(workspace, ws_align(dIndexK.numel() * 4), "IndexerBackward (sm90)")
            dIndexK_f32 = carver.take(dIndexK.numel(), torch.float32).view(dIndexK.shape)
        memset_zero_async(dIndexK_f32.data_ptr(), dIndexK_f32.numel() * 4, s)

        # Kernel 1: Compute grad_signal from scores (CuTe DSL only).
        _score_grad_inplace(
            AttnScore,
            IndexScore,
            GradLoss,
            grad_scale,
            index_is_log=score_input_is_log,
            current_stream=current_stream,
        )
        _run_gemm_only(
            IndexQ,
            Weights,
            IndexK,
            dIndexQ,
            dWeights,
            dIndexK_f32,
            AttnScore,
            TopkIndices,
            current_stream=current_stream,
        )
        if deterministic:
            reduce_deterministic_dk(dIndexK_f32, dIndexK, deterministic_dk_planes, current_stream)
        elif dIndexK_f32 is not dIndexK:
            with _torch_stream_context(current_stream):
                dIndexK.copy_(dIndexK_f32)

    _run.score_grad = partial(_score_grad_inplace, index_is_log=score_input_is_log)
    _run.gemm_only = _run_gemm_only
    _run.score_input_is_log = score_input_is_log

    return _run
