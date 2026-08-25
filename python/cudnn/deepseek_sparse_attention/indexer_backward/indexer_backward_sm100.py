# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Indexer Backward — SM100 CuTe-DSL, 3-kernel design.

Three kernels launched sequentially on the same stream:

  Kernel 1 (CuTe DSL): score_grad — compute sum_grad and grad_signal from
      AttnScore and IdxScore, overwrite AttnScore with grad_signal.
      Unsupported inputs trigger an exception before this stage launches.
  Kernel 2 (CuTe DSL): kernel_gemm — warp-specialized GEMM kernel (below).
      dK is accumulated in float32; the optimized path stages padded rows in
      SMEM and issues 512-byte cp.reduce.async.bulk FP32 additions.
  Kernel 3 (PyTorch):  dk_convert — cast dK from float32 to output dtype
      (same as dQ, dW).

Kernel 2 — Warp specialization (16 warps, 512 threads):
  Warp 0:      Load (Q via TMA, weights, grad_signal)
  Warp 1:      MMA  (3-stage sK pipeline, 2-stage TMEM S/dK: GEMM1 runs 1 block ahead)
  Warps 2-3:   Idle
  Warps 4-7:   Compute warpgroup
               (per-block sGradSignal load, TMEM readback S → dS → dW, dQ TMA store)
  Warps 8-11:  K loading warpgroup (TMA Gather4 for global IDs; manual
               cp.async fallback for local IDs, 3-stage sK)
  Warps 12-15: Reduce warpgroup (wide TMEM readback → padded ping-pong SMEM
               → cp.reduce.async.bulk to f32 gmem, 2-stage)

TopkIdxs are pre-loaded into SMEM cooperatively by all 512 threads before warp dispatch.
K/dK are flattened in ``__call__`` to a 2D ``(B*S_k, D)`` view so the kernel
indexes them by **global flat KV ids**. ``topk_indices_global=True`` (default,
matches the public fwd convention): ``mTopkIdx`` already carries
``b * seqlen_k + local`` and is loaded directly. ``topk_indices_global=False``:
ids are local-per-batch; the kernel adds ``batch_idx * S_k_per_batch`` to
convert (const_expr-branched). (THD will reuse the same flat-id contract:
``cu_seqlens_k[b] + local`` indexes the ``(T_k, D)`` packed buffer.)
grad_signal (precomputed by kernel 1) is loaded per topk-block by the compute warpgroup.

SMEM (kernel 2): full-row grad_signal/top-k staging plus a four-warp padded
  FP32 ping-pong buffer for bulk dK reduction.
TMEM: S0/dK0 @0, dQ @128, S1/dK1 @256 (384/512 cols).

Barriers for kernel 2:
  mbar[0-1]:  S_full_0/1     (MMA commits after GEMM1  → Compute waits)
  mbar[2-3]:  dS_ready_0/1   (Compute arrives after dS  → MMA waits)
  mbar[4-5]:  dK_full_0/1    (MMA commits after GEMM2  → Reduce waits)
  mbar[6-7]:  dK_empty_0/1   (Reduce arrives            → MMA waits)
  mbar[8-10]: K_loaded_0/1/2  (K-load arrives            → MMA waits, 3-stage)
  mbar[11-13]:K_consumed_0/1/2(MMA commits after GEMM3  → K-load waits, 3-stage)
  mbar[14]:   W_loaded        (Load arrives              → Compute waits)
  mbar[15]:   dQ_done         (MMA commits after GEMM3    → Compute waits)
  mbar[16]:   reduce_done     (Reduce arrives after T2R  → TMEM owner waits)
  mbar[17-18]:dS_half_0/1    (TopK=512 first-half dS publication)

Each warp/warpgroup has its own independent loop, communicating via barriers.
TopK=512 uses a grid-stride persistent CTA path capped at one CTA per SM; it
retains TMEM across rows and consolidates per-row barrier initialization.
TopK=1024/2048 retain the one-query-per-CTA 2-D grid.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, cast
import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute import atom as cute_atom
from cutlass.cute import core as cute_core
from cutlass import Float32, Int32, const_expr
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.cpasync.copy import (
    CopyBulkTensorTileG2SNonExecTrait,
)
from cutlass.cute.nvgpu.cpasync.helpers import TmaInfo
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
from cudnn.deepseek_sparse_attention.utils.copy import cpasync_reduce_bulk_add_f32
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
)

mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")


@dsl_user_op
def _tcgen05_fence_after_thread_sync(*, loc=None, ip=None):
    """Order subsequent tcgen05 operations after an inter-thread wait."""
    nvvm.tcgen05_fence(
        nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tcgen05_fence_before_thread_sync(*, loc=None, ip=None):
    """Order prior tcgen05 operations before an inter-thread signal."""
    nvvm.tcgen05_fence(
        nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC,
        loc=loc,
        ip=ip,
    )


# Barrier indices for kernel_gemm — per-stage barriers for S_FULL, DS_READY, K_LOADED
# to avoid phase-wrap when producer runs 2 blocks ahead of consumer.
# sK uses a 3-stage pipeline: K_LOADED and K_CONSUMED are per-stage (×3).
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
MBAR_REDUCE_DONE = 16
MBAR_DS_HALF_0 = 17
MBAR_DS_HALF_1 = 18
MBAR_ROW_FREE_0 = 19
MBAR_ROW_FREE_1 = 20
MBAR_DQ_FREE_0 = 21
MBAR_DQ_FREE_1 = 22
MBAR_GW_READY_0 = 23
MBAR_GW_READY_1 = 24
NUM_BARRIERS = 25

CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)

_score_grad_cute_cache: dict = {}


@dsl_user_op
def _load_global_i32x4(gmem_ptr, *, loc=None, ip=None):
    """Load four contiguous int32 values with one 16-byte global load."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal(
            [T.i32(), T.i32(), T.i32(), T.i32()],
        ),
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.global.v4.u32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _make_tiled_tma_gather4_atom(
    gmem_tensor,
    gmem_coord_tensor,
    smem_layout,
    mma_tiler_mnk,
    tiled_mma,
    *,
    loc=None,
    ip=None,
):
    """Build the SM100 2-D ``tile::gather4`` atom missing from DSL 4.6.1.

    The 4.6.1 wheel already ships the Gather4 MLIR operation and lowering, but
    its public Python helper/export is absent.  This is the minimal equivalent
    of ``make_tiled_tma_atom(..., gmem_coord_tensor=...)`` documented by that
    same wheel; it deliberately reuses the standard executable TMA-load trait.
    """
    smem_rank = cute_core.rank(smem_layout)
    assert smem_rank == 3 or smem_rank == 4

    stored_smem_layout = smem_layout
    if smem_rank == 4:
        smem_layout = cute_core.select(smem_layout, mode=[0, 1, 2])

    # Match make_tiled_tma_atom_B: the B operand's MMA N/K projection is
    # generally hierarchical and is not equivalent to a plain (N, K) tile.
    ident = cute_core.make_identity_layout(gmem_tensor.shape, loc=loc, ip=ip)
    mma_tiler_nk = (mma_tiler_mnk[1], *mma_tiler_mnk[2:])
    g_tile = cute_core.composition(
        ident,
        mma_tiler_nk,
        loc=loc,
        ip=ip,
    )
    cta_v_map = tiled_mma._thrfrg_B(g_tile)
    cta_v_map = cute_core.get(cta_v_map, mode=[1])
    cta_v_map = cute_core.dice(
        cta_v_map,
        (1, (1,) * cute_core.rank(g_tile)),
    )

    smem_for_ir = smem_layout
    if isinstance(smem_for_ir, cute_core._ComposedLayout):
        smem_for_ir = smem_for_ir.value

    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    res = _cute_nvgpu_ir.atom_make_non_exec_2d_gather4_tma_load(
        cast(Any, gmem_tensor).value,
        gmem_coord_tensor.layout,
        smem_for_ir,
        cta_v_map,
        _cute_nvgpu_ir.GatherScatterTmaLoadEnum.sm_100,
        num_multicast=1,
        loc=loc,
        ip=ip,
    )
    return TmaInfo(
        cute_atom.CopyAtom(op, CopyBulkTensorTileG2SNonExecTrait(res[0])),
        res[1],
        stored_smem_layout,
    )


@dsl_user_op
def _tma_gather4_k_rows(
    tma_atom,
    smem_ptr,
    column,
    row0,
    row1,
    row2,
    row3,
    transaction_barrier,
    *,
    loc=None,
    ip=None,
):
    """Issue one lane-level 4-row x 128-byte SM100 Gather4 transaction."""
    desc_ptr_type = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        cute.AddressSpace.generic,
        64,
    )
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(
        tma_atom._trait.value,
        loc=loc,
        ip=ip,
    )
    desc_ptr = _cute_nvgpu_ir.get_tma_desc_addr(
        desc_ptr_type,
        exec_atom,
        loc=loc,
        ip=ip,
    )
    desc_ptr_i64 = desc_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    barrier_ptr_i32 = transaction_barrier.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_ptr_i32,
            desc_ptr_i64,
            Int32(column).ir_value(),
            Int32(row0).ir_value(),
            Int32(row1).ir_value(),
            Int32(row2).ir_value(),
            Int32(row3).ir_value(),
            barrier_ptr_i32,
        ],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4" ".mbarrier::complete_tx::bytes" " [$0], [$1, {$2, $3, $4, $5, $6}], [$7];",
        "r,l,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class IndexerBackwardSm100:
    arch = 100
    WARP_SIZE = 32
    WARPGROUP_SIZE = 128
    NUM_WARPS = 16
    THREADS_PER_CTA = 512

    # dK bulk-reduce staging: four reducer warps, two ping-pong buffers per
    # warp, eight 512-byte rows per buffer, and 32-byte row padding.
    DK_STAGE_ROW_FLOATS = 136
    DK_STAGE_ROWS = 8
    DK_STAGE_BUFFERS = 2
    DK_STAGE_ELEMENTS = 4 * DK_STAGE_BUFFERS * DK_STAGE_ROWS * DK_STAGE_ROW_FLOATS

    # Warp assignments
    load_warp_id = 0
    mma_warp_id = 1
    # Warps 2-3: idle
    compute_warp_id = (4, 5, 6, 7)
    k_load_warp_id = (8, 9, 10, 11)
    reduce_warp_id = (12, 13, 14, 15)

    def __init__(
        self,
        head_dim,
        heads=64,
        block_I=128,
        topk=512,
        total_seqlen_k: int | None = None,
        total_rows: int = 1,
        persistent_grid_size: int = 1,
        topk_indices_global: bool = True,
    ):
        self.head_dim = head_dim
        self.heads = heads
        self.block_I = block_I
        self.topk = topk
        self.total_seqlen_k = total_seqlen_k
        self.total_rows = total_rows
        self.persistent_grid_size = persistent_grid_size
        # When True (default, matches the public fwd convention), mTopkIdx
        # carries global KV ids (``b * seqlen_k + local``); the kernel uses
        # them as flat ids into the (B*S_k, D) K/dK view directly. When
        # False, mTopkIdx carries local-per-batch ids and the kernel adds
        # ``batch_idx * S_k_per_batch`` to convert. Const_expr-branched.
        self.topk_indices_global = topk_indices_global
        assert heads >= 64
        assert topk > 0
        assert topk % block_I == 0
        self.num_topk_blocks = topk // block_I

        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.heads_padded = int(math.ceil(heads / 8) * 8)
        # Half-dS publication is specialized for the production TopK=512,
        # H64 x I128 fragment map. Larger TopK values retain the full-dS drain
        # order because the additional barrier traffic regresses them.
        self.use_ds_half = topk == 512 and self.heads_padded == 64 and block_I == 128
        self.use_tma_gather = topk_indices_global
        self.use_persistent = topk == 512 and self.heads_padded == 64 and self.head_dim_padded == 128 and block_I == 128
        # The cross-row K role is specialized around SM100 Gather4, whose
        # coordinate tensor consumes the public global-id contract. Local-id
        # inputs use the fully drained persistent-row fallback.
        self.use_cross_row_persistent = self.use_persistent and topk_indices_global

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
        self.tmem_dq_p1_offset = 384
        self.tmem_alloc_cols = 512

        # Register budgets — must sum to 512 per thread (65536 regs / 128 threads per WG)
        # Compute needs 128+ (tSrS=64 + dw_accum=64), Reduce needs 128+ (tDKrDK=128)
        self.num_regs_wg0 = 40
        # The dS compute warpgroup owns the largest register working set;
        # all four allocations exactly consume the per-CTA SM100 budget.
        self.num_regs_compute = 224
        self.num_regs_reduce = 200
        self.num_regs_kload = 48

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
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.gemm1_tiler[:2],
        )
        tmma2 = _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.gemm2_tiler[:2],
        )
        tmma3 = _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.gemm3_tiler[:2],
        )

        # SMEM layouts — primary views
        # sK/sKt: 3-stage pipeline for hiding K-load scatter latency.
        # sdS: 2-stage pipeline (tied to TMEM S/dK 2-stage).
        sQ_layout = _make_smem_layout_a(tmma1, self.gemm1_tiler, self.q_dtype, 1)
        sK_layout = _make_smem_layout_b(tmma1, self.gemm1_tiler, self.k_dtype, 3)
        sdS_layout = _make_smem_layout_a(tmma3, self.gemm3_tiler, self.q_dtype, 2)
        # Epilogue-style store layout for stmatrix writes to sdS (same physical SMEM).
        # dS is logical [H, I].  GEMM3 consumes it as a K-major A operand, so
        # its physical storage is row-major in that logical view.  GEMM2 sees
        # the same bytes as the transposed [I, H] MN-major A operand.
        sdS_store_layout = _make_smem_layout_epi(
            self.q_dtype,
            LayoutEnum.ROW_MAJOR,
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

        # Hardware sparse gather. The index-coordinate tensor has the
        # same logical 2-D shape as K, but its D mode is broadcast (stride 0):
        # each group of four row coordinates supplies one Gather4 instruction.
        # The coordinate tensor is descriptor metadata; the issuing lanes pass
        # row IDs explicitly to the Gather4 instruction.
        gI_desc = cute.make_tensor(
            mTopkIdx.iterator,
            cute.make_layout(
                (self.total_seqlen_k, self.head_dim_padded),
                stride=(1, 0),
            ),
        )
        mK_gather = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (self.total_seqlen_k, self.head_dim_padded),
                stride=(self.head_dim_padded, 1),
            ),
        )
        K_smem_layout_tma = cute.select(sK_layout, mode=[0, 1, 2])
        tma_atom_K_gather, _ = _make_tiled_tma_gather4_atom(
            mK_gather,
            gI_desc,
            K_smem_layout_tma,
            self.gemm1_tiler,
            tmma1,
        )

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

        grid_rows = min(self.total_rows, self.persistent_grid_size) if self.use_persistent else self.total_rows
        launch_grid = (grid_rows, 1, 1) if self.use_persistent else (seqlen, batch_size, 1)
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
            tma_atom_K_gather,
            tma_atom_dQ,
            sdQ_epi_layout,
            seqlen,
            batch_size,
        ).launch(
            grid=launch_grid,
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
        tma_atom_K_gather,
        tma_atom_dQ,
        sdQ_epi_layout,
        seqlen: Int32,
        batch_size: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if const_expr(self.use_persistent):
            flat_row_idx = cute.arch.block_idx()[0]
            batch_idx = flat_row_idx // seqlen
            seq_idx = flat_row_idx - batch_idx * seqlen
        else:
            seq_idx = cute.arch.block_idx()[0]
            batch_idx = cute.arch.block_idx()[1]
            flat_row_idx = batch_idx * seqlen + seq_idx
        seqlen_k = cute.size(mK.shape[0])

        # TMA descriptor prefetch (load warp only)
        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dQ)
        if const_expr(self.use_tma_gather) and warp_idx == self.k_load_warp_id[0]:
            cpasync.prefetch_descriptor(tma_atom_K_gather)

        # SMEM allocation
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sdS_size = cute.cosize(sdS_layout)
        sdQ_epi_size = cute.cosize(sdQ_epi_layout)
        _row_operand_stages = 2 if self.use_cross_row_persistent else 1
        _sQ_storage_size = int(sQ_size) * _row_operand_stages
        _grad_storage_size = self.topk * _row_operand_stages
        _weight_storage_size = self.heads * _row_operand_stages
        _dq_epi_storage_size = int(sdQ_epi_size) if self.use_cross_row_persistent else 1

        # Cross-row persistence double-buffers top-k IDs; all other paths need
        # exactly one complete row. SharedStorage's size assertion below is the
        # single source of truth for legal specializations.
        smem_topk_capacity = 2 * self.topk if self.use_cross_row_persistent else self.topk
        _dk_stage_elements = self.DK_STAGE_ELEMENTS
        _max_smem_bytes = 227 * 1024

        @cute.struct
        class SharedStorage:
            Q_mbar: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            mbar: cute.struct.MemRange[cutlass.Int64, NUM_BARRIERS]
            tmem_holding_buf: Int32
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, _sQ_storage_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]
            sdS: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sdS_size], self.buffer_align_bytes]
            sGradSignal: cute.struct.Align[cute.struct.MemRange[Float32, _grad_storage_size], 128]
            sTopkIdxs: cute.struct.Align[cute.struct.MemRange[Int32, smem_topk_capacity], 128]
            sW: cute.struct.Align[cute.struct.MemRange[self.q_dtype, _weight_storage_size], 128]
            sdKStage: cute.struct.Align[cute.struct.MemRange[Float32, _dk_stage_elements], 128]
            sdQEpilogue: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, _dq_epi_storage_size],
                self.buffer_align_bytes,
            ]

        assert SharedStorage.size_in_bytes() <= _max_smem_bytes, (
            f"SharedStorage ({SharedStorage.size_in_bytes()} bytes) exceeds {_max_smem_bytes} bytes (227KB), " f"smem_topk_capacity={smem_topk_capacity}"
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        Q_mbar_ptr = storage.Q_mbar.data_ptr()
        mbar = storage.mbar.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        # Swizzled SMEM tensors
        sK_raw_ptr = storage.sK.data_ptr()
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
        sdKStage = storage.sdKStage.get_tensor(cute.make_layout((_dk_stage_elements,), stride=(1,)))

        if const_expr(self.use_cross_row_persistent):
            # K is live across row boundaries, so unlike the serial path dQ
            # cannot alias its storage.  This dedicated tile is reused only by
            # the compute warpgroup, with the previous TMA store drained just
            # before the next overwrite.
            sdQ_epi_cross = storage.sdQEpilogue.get_tensor(
                sdQ_epi_layout.outer,
                swizzle=sdQ_epi_layout.inner,
            )
            self._run_persistent_cross_row(
                mQ,
                mW,
                mK,
                mdQ,
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
                sK,
                sKt,
                sdS,
                sQ_g2b_layout,
                sdS_store,
                sK_raw_ptr,
                storage.sQ.data_ptr(),
                storage.sGradSignal.data_ptr(),
                storage.sTopkIdxs.data_ptr(),
                storage.sW.data_ptr(),
                sdKStage,
                sdQ_epi_cross,
                Q_mbar_ptr,
                mbar,
                tmem,
                tma_atom_Q,
                tma_atom_K_gather,
                tma_atom_dQ,
                seqlen,
                batch_size,
                seqlen_k,
                flat_row_idx,
                tidx,
                warp_idx,
            )
            return

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
            defer_sync=self.use_persistent,
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

        # Initialize each barrier exactly once.
        if tidx == 0:
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DS_READY_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DS_READY_1, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_EMPTY_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DK_EMPTY_1, self.WARPGROUP_SIZE)
            if const_expr(self.use_tma_gather):
                # Transaction barrier: one explicit arrive-and-expect plus the
                # 32 KiB Gather4 completion for each K stage.
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, 1)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, 1)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, 1)
            else:
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_2, 1)
            cute.arch.mbarrier_init(mbar + MBAR_W_LOADED, self.WARP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DQ_DONE, 1)
            cute.arch.mbarrier_init(
                mbar + MBAR_REDUCE_DONE,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_HALF_0,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_HALF_1,
                self.WARPGROUP_SIZE,
            )
        if const_expr(self.use_persistent):
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Pre-load the complete top-k row into SMEM cooperatively.
        # K/dK are flattened to (B*S_k, D) above, so consumers index by global
        # flat KV ids. ``topk_indices_global=True`` (default): ``mTopkIdx`` already
        # carries global ids (``b * seqlen_k + local``); load directly.
        # ``topk_indices_global=False``: ids are local-per-batch. Only ids in
        # ``[0, S_k_per_batch)`` are converted to global flat ids; negative or
        # positive-OOB entries are normalized to -1 so they cannot alias a row
        # in a neighboring batch after adding the batch offset.
        seqlen_k_per_batch = seqlen_k // batch_size
        batch_offset_l2g = Int32(0) if const_expr(self.topk_indices_global) else batch_idx * seqlen_k_per_batch
        TOPK_PER_THREAD = const_expr((self.topk + self.THREADS_PER_CTA - 1) // self.THREADS_PER_CTA)
        for ii in cutlass.range_constexpr(TOPK_PER_THREAD):
            pos = ii * self.THREADS_PER_CTA + tidx
            if pos < self.topk:
                raw_id = Int32(mTopkIdx[seq_idx, pos, batch_idx])
                if const_expr(self.topk_indices_global):
                    sTopkIdxs[pos] = raw_id
                else:
                    sTopkIdxs[pos] = raw_id + batch_offset_l2g if raw_id >= 0 and raw_id < seqlen_k_per_batch else Int32(-1)
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

        # tcgen05 ``set(ACCUMULATE, ...)`` mutates the Python Atom wrapper.
        # Keep pristine wrappers for the staged persistent loop so its MMA
        # operands are rooted in values that dominate the loop region.
        if const_expr(self.use_persistent):
            tmma1_persistent = tmma1.__new_from_mlir_values__(
                tmma1.__extract_mlir_values__(),
            )
            tmma2_persistent = tmma2.__new_from_mlir_values__(
                tmma2.__extract_mlir_values__(),
            )
            tmma3_persistent = tmma3.__new_from_mlir_values__(
                tmma3.__extract_mlir_values__(),
            )

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
                tDqDq,
                tDkDk_0,
                tDkDk_1,
                Q_consumer,
                mbar,
                Int32(0),
                Int32(0),
                Int32(0),
                mbar,
                Int32(0),
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
                Int32(0),
                mbar + MBAR_W_LOADED,
                Int32(0),
                mbar,
            )
            if warp_idx == self.compute_warp_id[0]:
                cute.arch.mbarrier_wait(mbar + MBAR_REDUCE_DONE, Int32(0))
                _tcgen05_fence_after_thread_sync()
                if const_expr(not self.use_persistent):
                    cute.arch.dealloc_tmem(tmem_ptr_base, self.tmem_alloc_cols)
                if const_expr(self.use_persistent):
                    dQ_store_pipeline.producer_tail()

        elif warp_idx in self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_kload)
            self._k_load_warpgroup(
                mK,
                sK,
                sK_raw_ptr,
                sTopkIdxs,
                mTopkIdx,
                tma_atom_K_gather,
                seq_idx,
                batch_idx,
                seqlen_k,
                tidx,
                mbar,
                Int32(0),
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
                sdKStage,
                dk_acc_shape,
                tDkDk_0,
                tDkDk_1,
                sm_scale,
                seqlen_k,
                tidx,
                mbar,
            )

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)

        if const_expr(self.use_persistent):
            # The local-id fallback fully drains each row before reusing the
            # single operand set; TMEM remains allocated across rows.
            cute.arch.sync_threads()
            row_stride = cute.arch.grid_dim()[0]
            for next_flat_row in cutlass.range(
                flat_row_idx + row_stride,
                self.total_rows,
                row_stride,
            ):
                next_batch_idx = next_flat_row // seqlen
                next_seq_idx = next_flat_row - next_batch_idx * seqlen
                self._run_persistent_row_serial(
                    mQ,
                    mW,
                    mK,
                    mdQ,
                    mdW,
                    mdK_f32,
                    mGradSignal,
                    mTopkIdx,
                    sm_scale,
                    tmma1_persistent,
                    tmma2_persistent,
                    tmma3_persistent,
                    sQ,
                    sdS_g2a,
                    sK,
                    sKt,
                    sdS,
                    sQ_g2b,
                    sdS_store,
                    sK_raw_ptr,
                    sGradSignal,
                    sTopkIdxs,
                    sW,
                    sdKStage,
                    sdQ_epi,
                    Q_mbar_ptr,
                    mbar,
                    tmem,
                    tma_atom_Q,
                    tma_atom_K_gather,
                    tma_atom_dQ,
                    s_acc_shape,
                    s_acc_layout,
                    dq_acc_shape,
                    dq_acc_layout,
                    dk_acc_shape,
                    dk_acc_layout,
                    seqlen_k,
                    batch_size,
                    next_seq_idx,
                    next_batch_idx,
                    tidx,
                    warp_idx,
                )
                cute.arch.sync_threads()
            if const_expr(self.use_persistent):
                if warp_idx == self.compute_warp_id[0]:
                    persistent_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                    cute.arch.dealloc_tmem(
                        persistent_tmem_ptr,
                        self.tmem_alloc_cols,
                    )

    @cute.jit
    def _issue_cross_row_q(
        self,
        tmma1,
        gQ,
        sQ,
        tma_atom_Q,
        q_ready_barrier,
    ):
        """Issue one Q TMA using statically aligned SMEM/barrier operands."""
        tAgQ = tmma1.get_slice(0).partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tAgQ, 0, 3),
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                q_ready_barrier,
                self.tma_copy_Q_bytes,
            )
        # CopyBulkTensorTileG2SOp inserts its own elect_one.  Nesting it under
        # an explicit election is documented by the DSL as a deadlock.
        cute.copy(
            tma_atom_Q,
            tQgQ[None, 0, 0],
            tQsQ[None, 0],
            tma_bar_ptr=q_ready_barrier,
        )

    @cute.jit
    def _run_persistent_cross_row(
        self,
        mQ,
        mW,
        mK,
        mdQ,
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
        sK,
        sKt,
        sdS,
        sQ_g2b_layout,
        sdS_store,
        sK_raw_ptr,
        sQ_storage_ptr,
        sGrad_storage_ptr,
        sTopk_storage_ptr,
        sW_storage_ptr,
        sdKStage,
        sdQ_epi,
        Q_mbar_ptr,
        mbar,
        tmem,
        tma_atom_Q,
        tma_atom_K_gather,
        tma_atom_dQ,
        seqlen,
        batch_size,
        seqlen_k,
        first_flat_row,
        tidx,
        warp_idx,
    ):
        """TopK=512 persistent CTA with dominance-safe role-local row loops.

        Only scalar row/stage/phase values are loop carried.  In particular,
        the mutable tcgen05 MMA wrappers are cloned inside ``_mma_warp`` and
        never escape warp 1's dynamic branch.
        """
        s_acc_shape = tmma1.partition_shape_C(self.gemm1_tiler[:2])
        s_acc_layout = tmma1.make_fragment_C(s_acc_shape).layout
        dq_acc_shape = tmma3.partition_shape_C(self.gemm3_tiler[:2])
        dq_acc_layout = tmma3.make_fragment_C(dq_acc_shape).layout
        dk_acc_shape = tmma2.partition_shape_C(self.gemm2_tiler[:2])
        dk_acc_layout = tmma2.make_fragment_C(dk_acc_shape).layout
        sdS_g2a = cute.make_tensor(
            cute.recast_ptr(sdS.iterator, sdS_g2a_layout.inner),
            sdS_g2a_layout.outer,
        )

        if tidx == 0:
            cute.arch.mbarrier_init(Q_mbar_ptr, 1)
            cute.arch.mbarrier_init(Q_mbar_ptr + 1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_1, 1)
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_READY_0,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_READY_1,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_1, 1)
            cute.arch.mbarrier_init(
                mbar + MBAR_DK_EMPTY_0,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DK_EMPTY_1,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_2, 1)
            cute.arch.mbarrier_init(mbar + MBAR_W_LOADED, self.WARP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DQ_DONE, 1)
            cute.arch.mbarrier_init(
                mbar + MBAR_REDUCE_DONE,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_HALF_0,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_HALF_1,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_ROW_FREE_0,
                3 * self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_ROW_FREE_1,
                3 * self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DQ_FREE_0,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DQ_FREE_1,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(mbar + MBAR_GW_READY_0, self.WARP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_GW_READY_1, self.WARP_SIZE)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        row_stride = cute.arch.grid_dim()[0]
        sQ_size = const_expr(int(cute.cosize(sQ_layout)))
        sQ_p0 = cute.make_tensor(
            cute.recast_ptr(sQ_storage_ptr, sQ_layout.inner),
            sQ_layout.outer,
        )
        sQ_p1 = cute.make_tensor(
            cute.recast_ptr(
                sQ_storage_ptr + sQ_size,
                sQ_layout.inner,
            ),
            sQ_layout.outer,
        )

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)
            lane_id = tidx % self.WARP_SIZE
            for row in cutlass.range(
                first_flat_row,
                self.total_rows,
                row_stride,
            ):
                it = (row - first_flat_row) // row_stride
                parity = it & 1
                batch_idx = row // seqlen
                seq_idx = row - batch_idx * seqlen
                epoch = it // 2
                gQ = cute.local_tile(
                    mQ,
                    cute.select(self.gemm1_tiler, mode=[0, 2]),
                    (None, None, seq_idx, batch_idx),
                )
                # Q's last reader is the MMA warp.  Its DQ_DONE commit is
                # ordered after every Q-consuming GEMM for row it-2, so Q can
                # be refilled before the slower compute/K-load/reduce roles
                # collectively release the rest of this operand parity.
                if it >= 2:
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_DQ_DONE,
                        Int32((it - 2) & 1),
                    )
                if parity == 0:
                    self._issue_cross_row_q(
                        tmma1,
                        gQ,
                        sQ_p0,
                        tma_atom_Q,
                        Q_mbar_ptr,
                    )
                else:
                    self._issue_cross_row_q(
                        tmma1,
                        gQ,
                        sQ_p1,
                        tma_atom_Q,
                        Q_mbar_ptr + 1,
                    )
                # The remaining row operands share one parity buffer and are
                # overwritten only after all three consumer warpgroups retire.
                if it >= 2:
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_ROW_FREE_0 + parity,
                        Int32((epoch - 1) & 1),
                    )
                sGrad_row = cute.make_tensor(
                    sGrad_storage_ptr + parity * self.topk,
                    cute.make_layout((self.topk,), stride=(1,)),
                )
                sTopk_row = cute.make_tensor(
                    sTopk_storage_ptr + parity * self.topk,
                    cute.make_layout((self.topk,), stride=(1,)),
                )
                sW_row = cute.make_tensor(
                    sW_storage_ptr + parity * self.heads,
                    cute.make_layout((self.heads,), stride=(1,)),
                )
                seqlen_k_per_batch = seqlen_k // batch_size
                batch_offset_l2g = Int32(0) if const_expr(self.topk_indices_global) else batch_idx * seqlen_k_per_batch
                for step in cutlass.range_constexpr(
                    (self.topk + self.WARP_SIZE - 1) // self.WARP_SIZE,
                ):
                    pos = step * self.WARP_SIZE + lane_id
                    if pos < self.topk:
                        raw_id = Int32(mTopkIdx[seq_idx, pos, batch_idx])
                        if const_expr(self.topk_indices_global):
                            sTopk_row[pos] = raw_id
                        else:
                            sTopk_row[pos] = raw_id + batch_offset_l2g if raw_id >= 0 and raw_id < seqlen_k_per_batch else Int32(-1)
                        sGrad_row[pos] = mGradSignal[
                            seq_idx,
                            pos,
                            batch_idx,
                        ]
                for step in cutlass.range_constexpr(
                    (self.heads + self.WARP_SIZE - 1) // self.WARP_SIZE,
                ):
                    h = step * self.WARP_SIZE + lane_id
                    if h < self.heads:
                        sW_row[h] = mW[seq_idx, h, batch_idx]
                cute.arch.fence_view_async_shared()
                # Every load-warp lane publishes its own SMEM writes.
                cute.arch.mbarrier_arrive(
                    mbar + MBAR_GW_READY_0 + parity,
                )
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS_0, tStS_1, tDqDq_0, tDkDk_0, tDkDk_1 = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            tDqDq_1 = cute.make_tensor(
                tmem_ptr_base + self.tmem_dq_p1_offset,
                dq_acc_layout,
            )
            for row in cutlass.range(
                first_flat_row,
                self.total_rows,
                row_stride,
            ):
                it = (row - first_flat_row) // row_stride
                parity = it & 1
                epoch = it // 2
                if parity == 0:
                    sQ_g2b_row = cute.make_tensor(
                        cute.recast_ptr(
                            sQ_p0.iterator,
                            sQ_g2b_layout.inner,
                        ),
                        sQ_g2b_layout.outer,
                    )
                    self._mma_warp(
                        sQ_p0,
                        sdS_g2a,
                        sK,
                        sKt,
                        sdS,
                        sQ_g2b_row,
                        tmma1,
                        tmma2,
                        tmma3,
                        tStS_0,
                        tStS_1,
                        tDqDq_0,
                        tDqDq_1,
                        tDkDk_0,
                        tDkDk_1,
                        Q_mbar_ptr,
                        mbar,
                        Int32(it & 1),
                        it,
                        Int32(epoch & 1),
                        mbar + MBAR_DQ_FREE_0,
                        Int32((epoch - 1) & 1),
                    )
                else:
                    sQ_g2b_row = cute.make_tensor(
                        cute.recast_ptr(
                            sQ_p1.iterator,
                            sQ_g2b_layout.inner,
                        ),
                        sQ_g2b_layout.outer,
                    )
                    self._mma_warp(
                        sQ_p1,
                        sdS_g2a,
                        sK,
                        sKt,
                        sdS,
                        sQ_g2b_row,
                        tmma1,
                        tmma2,
                        tmma3,
                        tStS_0,
                        tStS_1,
                        tDqDq_0,
                        tDqDq_1,
                        tDkDk_0,
                        tDkDk_1,
                        Q_mbar_ptr + 1,
                        mbar,
                        Int32(it & 1),
                        it,
                        Int32(epoch & 1),
                        mbar + MBAR_DQ_FREE_1,
                        Int32((epoch - 1) & 1),
                    )
        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS_0, tStS_1, tDqDq_0, _, _ = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            tDqDq_1 = cute.make_tensor(
                tmem_ptr_base + self.tmem_dq_p1_offset,
                dq_acc_layout,
            )
            dQ_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.WARPGROUP_SIZE,
                ),
            )
            sdQ_epi_slice = sdQ_epi[None, None, 0]
            for row in cutlass.range(
                first_flat_row,
                self.total_rows,
                row_stride,
            ):
                it = (row - first_flat_row) // row_stride
                parity = it & 1
                epoch = it // 2
                batch_idx = row // seqlen
                seq_idx = row - batch_idx * seqlen
                sGrad_row = cute.make_tensor(
                    sGrad_storage_ptr + parity * self.topk,
                    cute.make_layout((self.topk,), stride=(1,)),
                )
                sW_row = cute.make_tensor(
                    sW_storage_ptr + parity * self.heads,
                    cute.make_layout((self.heads,), stride=(1,)),
                )
                gdQ = cute.local_tile(
                    mdQ,
                    (self.heads_padded, self.head_dim_padded),
                    (0, 0, seq_idx, batch_idx),
                )
                tdQsdQ, tdQgdQ = cpasync.tma_partition(
                    tma_atom_dQ,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sdQ_epi_slice, 0, 2),
                    cute.group_modes(gdQ, 0, 2),
                )
                self._compute_warpgroup(
                    mdW,
                    sGrad_row,
                    sW_row,
                    sdS_store,
                    sdS,
                    sdQ_epi_slice,
                    s_acc_shape,
                    dq_acc_shape,
                    tStS_0,
                    tStS_1,
                    tDqDq_0,
                    tDqDq_1,
                    tma_atom_dQ,
                    tdQsdQ,
                    tdQgdQ,
                    dQ_store_pipeline,
                    sm_scale,
                    seq_idx,
                    batch_idx,
                    tidx,
                    warp_idx,
                    mbar,
                    Int32(it & 1),
                    mbar + MBAR_GW_READY_0 + parity,
                    Int32(epoch & 1),
                    mbar + MBAR_DQ_FREE_0 + parity,
                )
                cute.arch.mbarrier_arrive(
                    mbar + MBAR_ROW_FREE_0 + parity,
                )

            last_it = (self.total_rows - 1 - first_flat_row) // row_stride
            last_parity = last_it & 1
            cute.arch.mbarrier_wait(
                mbar + MBAR_ROW_FREE_0 + last_parity,
                Int32((last_it // 2) & 1),
            )
            if warp_idx == self.compute_warp_id[0]:
                dQ_store_pipeline.producer_tail()
                _tcgen05_fence_after_thread_sync()
                cute.arch.dealloc_tmem(tmem_ptr_base, self.tmem_alloc_cols)

        elif warp_idx in self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_kload)
            for row in cutlass.range(
                first_flat_row,
                self.total_rows,
                row_stride,
            ):
                it = (row - first_flat_row) // row_stride
                parity = it & 1
                batch_idx = row // seqlen
                seq_idx = row - batch_idx * seqlen
                sTopk_row = cute.make_tensor(
                    sTopk_storage_ptr + parity * self.topk,
                    cute.make_layout((self.topk,), stride=(1,)),
                )
                self._k_load_warpgroup(
                    mK,
                    sK,
                    sK_raw_ptr,
                    sTopk_row,
                    mTopkIdx,
                    tma_atom_K_gather,
                    seq_idx,
                    batch_idx,
                    seqlen_k,
                    tidx,
                    mbar,
                    Int32(it),
                )
                cute.arch.mbarrier_arrive(
                    mbar + MBAR_ROW_FREE_0 + parity,
                )

        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            _, _, _, tDkDk_0, tDkDk_1 = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            for row in cutlass.range(
                first_flat_row,
                self.total_rows,
                row_stride,
            ):
                it = (row - first_flat_row) // row_stride
                parity = it & 1
                batch_idx = row // seqlen
                seq_idx = row - batch_idx * seqlen
                sTopk_row = cute.make_tensor(
                    sTopk_storage_ptr + parity * self.topk,
                    cute.make_layout((self.topk,), stride=(1,)),
                )
                self._reduce_warpgroup(
                    mdK_f32,
                    sTopk_row,
                    sdKStage,
                    dk_acc_shape,
                    tDkDk_0,
                    tDkDk_1,
                    sm_scale,
                    seqlen_k,
                    tidx,
                    mbar,
                )
                cute.arch.mbarrier_arrive(
                    mbar + MBAR_ROW_FREE_0 + parity,
                )
            lane_id = tidx % self.WARP_SIZE
            if lane_id < 8:
                cute.arch.cp_async_bulk_wait_group(0)

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_wg0)

    @cute.jit
    def _run_persistent_row_serial(
        self,
        mQ,
        mW,
        mK,
        mdQ,
        mdW,
        mdK_f32,
        mGradSignal,
        mTopkIdx,
        sm_scale,
        tmma1,
        tmma2,
        tmma3,
        sQ,
        sdS_g2a,
        sK,
        sKt,
        sdS,
        sQ_g2b,
        sdS_store,
        sK_raw_ptr,
        sGradSignal,
        sTopkIdxs,
        sW,
        sdKStage,
        sdQ_epi,
        Q_mbar_ptr,
        mbar,
        tmem,
        tma_atom_Q,
        tma_atom_K_gather,
        tma_atom_dQ,
        s_acc_shape,
        s_acc_layout,
        dq_acc_shape,
        dq_acc_layout,
        dk_acc_shape,
        dk_acc_layout,
        seqlen_k,
        batch_size,
        seq_idx,
        batch_idx,
        tidx,
        warp_idx,
    ):
        """Run one fully drained row in a persistent CTA."""
        Q_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cute.make_layout(self.cluster_shape),
            defer_sync=self.use_persistent,
        )
        Q_producer, Q_consumer = Q_pipeline.make_participants()
        gQ = cute.local_tile(
            mQ,
            cute.select(self.gemm1_tiler, mode=[0, 2]),
            (None, None, seq_idx, batch_idx),
        )
        gemm1_thr_mma = tmma1.get_slice(0)
        tAgQ = gemm1_thr_mma.partition_A(gQ)
        tQsQ, tQgQ_mkl = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tAgQ, 0, 3),
        )

        dQ_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARPGROUP_SIZE,
            ),
        )
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

        if tidx == 0:
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_S_FULL_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DS_READY_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DS_READY_1, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_FULL_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_DK_EMPTY_0, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DK_EMPTY_1, self.WARPGROUP_SIZE)
            if const_expr(self.use_tma_gather):
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, 1)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, 1)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, 1)
            else:
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_0, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_1, self.WARPGROUP_SIZE)
                cute.arch.mbarrier_init(mbar + MBAR_K_LOADED_2, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_K_CONSUMED_2, 1)
            cute.arch.mbarrier_init(mbar + MBAR_W_LOADED, self.WARP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_DQ_DONE, 1)
            cute.arch.mbarrier_init(
                mbar + MBAR_REDUCE_DONE,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_HALF_0,
                self.WARPGROUP_SIZE,
            )
            cute.arch.mbarrier_init(
                mbar + MBAR_DS_HALF_1,
                self.WARPGROUP_SIZE,
            )
        if const_expr(self.use_persistent):
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        seqlen_k_per_batch = seqlen_k // batch_size
        batch_offset_l2g = Int32(0) if const_expr(self.topk_indices_global) else batch_idx * seqlen_k_per_batch
        topk_per_thread = const_expr((self.topk + self.THREADS_PER_CTA - 1) // self.THREADS_PER_CTA)
        for ii in cutlass.range_constexpr(topk_per_thread):
            pos = ii * self.THREADS_PER_CTA + tidx
            if pos < self.topk:
                raw_id = Int32(mTopkIdx[seq_idx, pos, batch_idx])
                if const_expr(self.topk_indices_global):
                    sTopkIdxs[pos] = raw_id
                else:
                    sTopkIdxs[pos] = raw_id + batch_offset_l2g if raw_id >= 0 and raw_id < seqlen_k_per_batch else Int32(-1)
        cute.arch.sync_threads()

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
                tDqDq,
                tDkDk_0,
                tDkDk_1,
                Q_consumer,
                mbar,
                Int32(0),
                Int32(0),
                Int32(0),
                mbar,
                Int32(0),
            )
        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                if const_expr(not self.use_persistent):
                    tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS_0, tStS_1, tDqDq, _, _ = self.get_tmem_tensor(
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
                Int32(0),
                mbar + MBAR_W_LOADED,
                Int32(0),
                mbar,
            )
            if warp_idx == self.compute_warp_id[0]:
                cute.arch.mbarrier_wait(mbar + MBAR_REDUCE_DONE, Int32(0))
                _tcgen05_fence_after_thread_sync()
                if const_expr(not self.use_persistent):
                    cute.arch.dealloc_tmem(tmem_ptr_base, self.tmem_alloc_cols)
                dQ_store_pipeline.producer_tail()
        elif warp_idx in self.k_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_kload)
            self._k_load_warpgroup(
                mK,
                sK,
                sK_raw_ptr,
                sTopkIdxs,
                mTopkIdx,
                tma_atom_K_gather,
                seq_idx,
                batch_idx,
                seqlen_k,
                tidx,
                mbar,
                Int32(0),
            )
        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            _, _, _, tDkDk_0, tDkDk_1 = self.get_tmem_tensor(
                s_acc_layout,
                dq_acc_layout,
                dk_acc_layout,
                tmem_ptr_base,
            )
            self._reduce_warpgroup(
                mdK_f32,
                sTopkIdxs,
                sdKStage,
                dk_acc_shape,
                tDkDk_0,
                tDkDk_1,
                sm_scale,
                seqlen_k,
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
        # Signal W + grad_signal loaded for compute warpgroup.
        # All 32 lanes arrive (count = WARP_SIZE): mbarrier.arrive has release
        # semantics for the *executing thread* only, so a single elected
        # arrival would not order the other 31 lanes' sW/sGradSignal stores
        # before the consumer's mbarrier_wait (racecheck flags exactly those
        # 31 lanes). Whole-warp arrival closes the happens-before chain.
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
    def _gemm_dq_parity(
        self,
        tmma3,
        tDqDq,
        tDqDq_p1,
        frag_a,
        frag_b,
        row_iteration,
    ):
        """Issue GEMM3 through one of two statically aligned TMEM views."""
        if const_expr(self.use_cross_row_persistent):
            if (row_iteration & 1) == 0:
                cute.gemm(tmma3, tDqDq, frag_a, frag_b, tDqDq)
            else:
                cute.gemm(tmma3, tDqDq_p1, frag_a, frag_b, tDqDq_p1)
        else:
            cute.gemm(tmma3, tDqDq, frag_a, frag_b, tDqDq)

    @cute.jit
    def _wait_rotating_k_loaded(self, mbar, row_iteration, block_index):
        global_block = row_iteration * self.num_topk_blocks + block_index
        stage = global_block % 3
        cute.arch.mbarrier_wait(
            mbar + MBAR_K_LOADED_0 + stage,
            Int32((global_block // 3) & 1),
        )
        _tcgen05_fence_after_thread_sync()

    @cute.jit
    def _commit_k_consumed(self, mbar, stage):
        with cute.arch.elect_one():
            tcgen05.commit(mbar + MBAR_K_CONSUMED_0 + stage)

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
        tDqDq_p1,
        tDkDk_0,
        tDkDk_1,
        Q_consumer_or_barrier,
        mbar,
        persistent_row_phase,
        row_iteration,
        q_ready_phase,
        dq_free_barrier,
        dq_free_phase,
    ):
        """MMA warp: 3-stage sK pipeline, 2-stage TMEM S/dK.

        Structure: Prologue(Fill[0]) → Main(Fill[bi]+Drain[bi-1]) → Epilogue(Drain[last])
        GEMM1(S) runs 1 block ahead, hiding Compute latency behind the next GEMM1.

        sK uses 3-stage pipeline (bi%3) to hide K-load scatter gather latency.
        TMEM S/dK accumulators remain 2-stage (bi%2).
        K_CONSUMED is per-sK-stage (3 barriers) so K-load can run 3 blocks ahead.
        """
        # ``Atom.set`` replaces the Python wrapper's internal MLIR SSA value.
        # Never mutate the kernel-argument wrappers here: this helper is
        # emitted inside the runtime warp-role branch, so leaking a value
        # defined in that branch into a later branch/loop violates SSA
        # dominance.  Role-local wrappers keep every atom_set_value and use in
        # the same control-flow region.  This is also what lets the continuous
        # persistent path call this helper from a dynamic row loop safely.
        tmma1 = tmma1.__new_from_mlir_values__(tmma1.__extract_mlir_values__())
        tmma2 = tmma2.__new_from_mlir_values__(tmma2.__extract_mlir_values__())
        tmma3 = tmma3.__new_from_mlir_values__(tmma3.__extract_mlir_values__())

        if const_expr(self.use_cross_row_persistent):
            cute.arch.mbarrier_wait(
                Q_consumer_or_barrier,
                Int32(q_ready_phase),
            )
        else:
            Q_consumer_or_barrier.reset()
            Q_consumer_or_barrier.wait_and_advance()

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
        ds_half_0_phase = Int32(0)
        ds_half_1_phase = Int32(0)
        k_loaded_0_phase = Int32(0)
        k_loaded_1_phase = Int32(persistent_row_phase)
        k_loaded_2_phase = Int32(persistent_row_phase)
        is_first_dq = True

        # The serial path reinitialized these barriers every row.  In the
        # continuous path each TMEM stage completes twice per TopK=512 row:
        # consume the second completion from row it-1 before its first reuse,
        # then let the original in-row phase-0 waits cover blocks 2 and 3.
        if const_expr(self.use_cross_row_persistent):
            if row_iteration >= 1:
                cute.arch.mbarrier_wait(
                    mbar + MBAR_DK_EMPTY_0,
                    Int32(1),
                )
                _tcgen05_fence_after_thread_sync()

        # =============================================================
        # Prologue: Fill block 0 (sK stage 0, TMEM stage 0)
        # =============================================================
        if const_expr(self.use_cross_row_persistent):
            k_stage_0 = row_iteration % 3
            self._wait_rotating_k_loaded(mbar, row_iteration, 0)
        else:
            k_stage_0 = const_expr(0)
            cute.arch.mbarrier_wait(
                mbar + MBAR_K_LOADED_0,
                k_loaded_0_phase,
            )
            k_loaded_0_phase ^= 1
            _tcgen05_fence_after_thread_sync()

        tmma1.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
            cute.gemm(
                tmma1,
                tStS_0,
                tSrQ[None, None, k_block, 0],
                tSrK[None, None, k_block, k_stage_0],
                tStS_0,
            )
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
            if const_expr(self.use_cross_row_persistent):
                fill_k_stage = (row_iteration + bi) % 3
                drain_k_stage = (row_iteration + bi - 1) % 3
            else:
                fill_k_stage = const_expr(bi % 3)
                drain_k_stage = const_expr((bi - 1) % 3)

            if const_expr(self.use_cross_row_persistent and bi == 1):
                if row_iteration >= 1:
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_DK_EMPTY_1,
                        Int32(1),
                    )
                    _tcgen05_fence_after_thread_sync()

            # ------ Fill[bi]: GEMM1 for current block ------
            # DK_EMPTY: wait for TMEM slot reuse (2-stage, bi%2)
            if bi >= 2:
                if bi % 2 == 0:
                    cute.arch.mbarrier_wait(mbar + MBAR_DK_EMPTY_0, dk_empty_0_phase)
                    dk_empty_0_phase ^= 1
                else:
                    cute.arch.mbarrier_wait(mbar + MBAR_DK_EMPTY_1, dk_empty_1_phase)
                    dk_empty_1_phase ^= 1

            # K_LOADED rotates continuously across persistent rows.
            if const_expr(self.use_cross_row_persistent):
                self._wait_rotating_k_loaded(mbar, row_iteration, bi)
            else:
                if bi % 3 == 0:
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_K_LOADED_0,
                        k_loaded_0_phase,
                    )
                    k_loaded_0_phase ^= 1
                elif bi % 3 == 1:
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_K_LOADED_1,
                        k_loaded_1_phase,
                    )
                    k_loaded_1_phase ^= 1
                else:
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_K_LOADED_2,
                        k_loaded_2_phase,
                    )
                    k_loaded_2_phase ^= 1
                # Order the next tcgen05 MMA after both the K-loaded wait and,
                # when reusing a TMEM stage, the earlier DK_EMPTY wait.
                _tcgen05_fence_after_thread_sync()

            # GEMM1: tStS[bi%2] = Q @ sK[bi%3]
            tmma1.set(tcgen05.Field.ACCUMULATE, False)
            if bi % 2 == 0:
                for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                    cute.gemm(tmma1, tStS_0, tSrQ[None, None, k_block, 0], tSrK[None, None, k_block, fill_k_stage], tStS_0)
                    tmma1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_S_FULL_0)
            else:
                for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                    cute.gemm(tmma1, tStS_1, tSrQ[None, None, k_block, 0], tSrK[None, None, k_block, fill_k_stage], tStS_1)
                    tmma1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_S_FULL_1)

            # ------ Drain[bi-1]: GEMM2(dK) + GEMM3(dQ) for previous block ------
            if const_expr(self.use_cross_row_persistent and bi == 1):
                # dQ has two TMEM parities.  Release happens immediately after
                # the compute warpgroup's T2R, before its SMEM/TMA epilogue.
                if row_iteration >= 2:
                    cute.arch.mbarrier_wait(
                        dq_free_barrier,
                        Int32(dq_free_phase),
                    )
                    _tcgen05_fence_after_thread_sync()
            if (bi - 1) % 2 == 0:
                # Prev TMEM stage 0
                if const_expr(self.use_ds_half):
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_DS_HALF_0,
                        ds_half_0_phase,
                    )
                    ds_half_0_phase ^= 1
                    _tcgen05_fence_after_thread_sync()
                    tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                    is_first_dq = False
                    for k_block in cutlass.range(
                        0,
                        cute.size(tDQrDS, mode=[2]) // 2,
                        unroll=4,
                    ):
                        self._gemm_dq_parity(
                            tmma3,
                            tDqDq,
                            tDqDq_p1,
                            tDQrDS[None, None, k_block, 0],
                            tDQrKt[None, None, k_block, drain_k_stage],
                            row_iteration,
                        )
                        tmma3.set(tcgen05.Field.ACCUMULATE, True)

                cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_0, ds_ready_0_phase)
                ds_ready_0_phase ^= 1
                _tcgen05_fence_after_thread_sync()

                tmma2.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                    cute.gemm(tmma2, tDkDk_0, tDKrA_g2[None, None, k_block, 0], tDKrB_g2[None, None, k_block, 0], tDkDk_0)
                    tmma2.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_DK_FULL_0)

                if const_expr(not self.use_ds_half):
                    tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                    is_first_dq = False
                DQ_K_BEGIN = const_expr(cute.size(tDQrDS, mode=[2]) // 2 if self.use_ds_half else 0)
                for k_block in cutlass.range(
                    DQ_K_BEGIN,
                    cute.size(tDQrDS, mode=[2]),
                    unroll=4,
                ):
                    self._gemm_dq_parity(
                        tmma3,
                        tDqDq,
                        tDqDq_p1,
                        tDQrDS[None, None, k_block, 0],
                        tDQrKt[None, None, k_block, drain_k_stage],
                        row_iteration,
                    )
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)
                # Release the continuously rotating sparse-K stage.
                self._commit_k_consumed(mbar, drain_k_stage)
            else:
                # Prev TMEM stage 1
                if const_expr(self.use_ds_half):
                    cute.arch.mbarrier_wait(
                        mbar + MBAR_DS_HALF_1,
                        ds_half_1_phase,
                    )
                    ds_half_1_phase ^= 1
                    _tcgen05_fence_after_thread_sync()
                    tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                    is_first_dq = False
                    for k_block in cutlass.range(
                        0,
                        cute.size(tDQrDS, mode=[2]) // 2,
                        unroll=4,
                    ):
                        self._gemm_dq_parity(
                            tmma3,
                            tDqDq,
                            tDqDq_p1,
                            tDQrDS[None, None, k_block, 1],
                            tDQrKt[None, None, k_block, drain_k_stage],
                            row_iteration,
                        )
                        tmma3.set(tcgen05.Field.ACCUMULATE, True)

                cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_1, ds_ready_1_phase)
                ds_ready_1_phase ^= 1
                _tcgen05_fence_after_thread_sync()

                tmma2.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                    cute.gemm(tmma2, tDkDk_1, tDKrA_g2[None, None, k_block, 1], tDKrB_g2[None, None, k_block, 0], tDkDk_1)
                    tmma2.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_DK_FULL_1)

                if const_expr(not self.use_ds_half):
                    tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                    is_first_dq = False
                DQ_K_BEGIN = const_expr(cute.size(tDQrDS, mode=[2]) // 2 if self.use_ds_half else 0)
                for k_block in cutlass.range(
                    DQ_K_BEGIN,
                    cute.size(tDQrDS, mode=[2]),
                    unroll=4,
                ):
                    self._gemm_dq_parity(
                        tmma3,
                        tDqDq,
                        tDqDq_p1,
                        tDQrDS[None, None, k_block, 1],
                        tDQrKt[None, None, k_block, drain_k_stage],
                        row_iteration,
                    )
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)
                # Release the continuously rotating sparse-K stage.
                self._commit_k_consumed(mbar, drain_k_stage)

        # =============================================================
        # Epilogue: Drain last block
        # TMEM stage: (num_topk_blocks-1)%2, sK stage: (num_topk_blocks-1)%3
        # =============================================================
        LAST_TMEM_STAGE = const_expr((self.num_topk_blocks - 1) % 2)
        if const_expr(self.use_cross_row_persistent):
            last_sk_stage = (row_iteration + self.num_topk_blocks - 1) % 3
        else:
            last_sk_stage = const_expr((self.num_topk_blocks - 1) % 3)
        if LAST_TMEM_STAGE == 0:
            if const_expr(self.use_ds_half):
                cute.arch.mbarrier_wait(
                    mbar + MBAR_DS_HALF_0,
                    ds_half_0_phase,
                )
                _tcgen05_fence_after_thread_sync()
                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                is_first_dq = False
                for k_block in cutlass.range(
                    0,
                    cute.size(tDQrDS, mode=[2]) // 2,
                    unroll=4,
                ):
                    self._gemm_dq_parity(
                        tmma3,
                        tDqDq,
                        tDqDq_p1,
                        tDQrDS[None, None, k_block, 0],
                        tDQrKt[None, None, k_block, last_sk_stage],
                        row_iteration,
                    )
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_0, ds_ready_0_phase)
            _tcgen05_fence_after_thread_sync()
            tmma2.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                cute.gemm(tmma2, tDkDk_0, tDKrA_g2[None, None, k_block, 0], tDKrB_g2[None, None, k_block, 0], tDkDk_0)
                tmma2.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DK_FULL_0)
            if const_expr(not self.use_ds_half):
                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
            DQ_K_BEGIN = const_expr(cute.size(tDQrDS, mode=[2]) // 2 if self.use_ds_half else 0)
            for k_block in cutlass.range(
                DQ_K_BEGIN,
                cute.size(tDQrDS, mode=[2]),
                unroll=4,
            ):
                self._gemm_dq_parity(
                    tmma3,
                    tDqDq,
                    tDqDq_p1,
                    tDQrDS[None, None, k_block, 0],
                    tDQrKt[None, None, k_block, last_sk_stage],
                    row_iteration,
                )
                tmma3.set(tcgen05.Field.ACCUMULATE, True)
            self._commit_k_consumed(mbar, last_sk_stage)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DQ_DONE)
        else:
            if const_expr(self.use_ds_half):
                cute.arch.mbarrier_wait(
                    mbar + MBAR_DS_HALF_1,
                    ds_half_1_phase,
                )
                _tcgen05_fence_after_thread_sync()
                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
                is_first_dq = False
                for k_block in cutlass.range(
                    0,
                    cute.size(tDQrDS, mode=[2]) // 2,
                    unroll=4,
                ):
                    self._gemm_dq_parity(
                        tmma3,
                        tDqDq,
                        tDqDq_p1,
                        tDQrDS[None, None, k_block, 1],
                        tDQrKt[None, None, k_block, last_sk_stage],
                        row_iteration,
                    )
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.mbarrier_wait(mbar + MBAR_DS_READY_1, ds_ready_1_phase)
            _tcgen05_fence_after_thread_sync()
            tmma2.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                cute.gemm(tmma2, tDkDk_1, tDKrA_g2[None, None, k_block, 1], tDKrB_g2[None, None, k_block, 0], tDkDk_1)
                tmma2.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_DK_FULL_1)
            if const_expr(not self.use_ds_half):
                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq)
            DQ_K_BEGIN = const_expr(cute.size(tDQrDS, mode=[2]) // 2 if self.use_ds_half else 0)
            for k_block in cutlass.range(
                DQ_K_BEGIN,
                cute.size(tDQrDS, mode=[2]),
                unroll=4,
            ):
                self._gemm_dq_parity(
                    tmma3,
                    tDqDq,
                    tDqDq_p1,
                    tDQrDS[None, None, k_block, 1],
                    tDQrKt[None, None, k_block, last_sk_stage],
                    row_iteration,
                )
                tmma3.set(tcgen05.Field.ACCUMULATE, True)
            self._commit_k_consumed(mbar, last_sk_stage)
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
        tDqDq_p1,
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
        persistent_row_phase,
        gw_ready_barrier,
        gw_ready_phase,
        dq_free_barrier,
    ):
        """Compute/Epilogue warpgroup: TMEM readback S → register dS → stmatrix sdS, dQ/dW output.

        Optimizations (per dsa-next pattern):
          - TMEM reduced to 1-mode for flat register layout
          - 2D identity for scalar (h, n) coordinates: mode=[0] → h, mode=[1] → n
          - sW pre-loaded into bf16 register array (scalar h enables rW[h])
          - Paired f32x2 packed ops (fma_packed_f32x2 for dW accumulation)
          - stmatrix bulk store (8 warp-level instructions vs 32+ scalar shared-mem stores per thread)
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

        # The STMatrix epilogue below uses the logical [H, I] ROW_MAJOR view
        # required by GEMM3's K-major A operand.  The same bytes are GEMM2's
        # transposed [I, H] MN-major A operand.  Other legal factory shapes
        # retain the original coordinate store because this ownership mapping
        # is specialized for the production H64 x I128 tile.
        use_stmatrix_ds = const_expr(
            self.heads_padded == 64 and self.block_I == 128,
        )
        if const_expr(use_stmatrix_ds):
            # --- TMEM readback (canonical 2D epilogue view) ---
            tStS_epi_0 = tStS_0[((None, None), 0, 0)]
            tiled_tmem_load_s_0 = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_epi_0,
            )
            thr_tmem_load_s_0 = tiled_tmem_load_s_0.get_slice(wg_tidx)
            tStS_t2r_0 = thr_tmem_load_s_0.partition_S(tStS_epi_0)

            tStS_epi_1 = tStS_1[((None, None), 0, 0)]
            tiled_tmem_load_s_1 = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_epi_1,
            )
            thr_tmem_load_s_1 = tiled_tmem_load_s_1.get_slice(wg_tidx)
            tStS_t2r_1 = thr_tmem_load_s_1.partition_S(tStS_epi_1)

            # Derive R2S ownership from the TMEM-load ownership.  This selects
            # STSM and preserves each register element's logical coordinate.
            smem_store_atom = sm100_utils_basic.get_smem_store_op(
                LayoutEnum.ROW_MAJOR,
                self.q_dtype,
                self.acc_dtype,
                tiled_tmem_load_s_0,
            )
            tiled_smem_store = cute.make_tiled_copy_D(
                smem_store_atom,
                tiled_tmem_load_s_0,
            )
            thr_smem_store = tiled_smem_store.get_slice(wg_tidx)
            tRS_sdS = thr_smem_store.partition_D(sdS_store)

            cS = cute.make_identity_tensor(
                (self.heads_padded, self.block_I),
            )
            tCcS = thr_tmem_load_s_0.partition_D(cS)
        else:
            # General-shape fallback: preserve the MMA accumulator's complete
            # coordinate hierarchy and write through its native sdS view.
            tiled_tmem_load_s_0 = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_0,
            )
            thr_tmem_load_s_0 = tiled_tmem_load_s_0.get_slice(wg_tidx)
            tStS_t2r_0 = thr_tmem_load_s_0.partition_S(tStS_0)

            tiled_tmem_load_s_1 = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_1,
            )
            thr_tmem_load_s_1 = tiled_tmem_load_s_1.get_slice(wg_tidx)
            tStS_t2r_1 = thr_tmem_load_s_1.partition_S(tStS_1)

            sdS_gemm_view_0 = cute.composition(
                sdS[None, None, None, 0],
                cute.make_layout((self.heads_padded, self.block_I)),
            )
            sdS_gemm_view_1 = cute.composition(
                sdS[None, None, None, 1],
                cute.make_layout((self.heads_padded, self.block_I)),
            )
            cS = cute.make_identity_tensor(s_acc_shape)
            tCcS = thr_tmem_load_s_0.partition_D(cS)
        tSrS_shape = tCcS.shape

        # Use the canonical 2-D accumulator view when CuTe can lower dQ
        # staging to STMatrix; other legal shapes use coordinate stores.
        use_stmatrix_dq = const_expr(self.heads_padded == 64 and self.head_dim_padded == 128)
        if const_expr(use_stmatrix_dq):
            tDqDq_load_view = tDqDq[((None, None), 0, 0)]
            if const_expr(self.use_cross_row_persistent):
                tDqDq_p1_load_view = tDqDq_p1[((None, None), 0, 0)]
        else:
            tDqDq_load_view = tDqDq
            if const_expr(self.use_cross_row_persistent):
                tDqDq_p1_load_view = tDqDq_p1
        tiled_tmem_load_dq = tcgen05.make_tmem_copy(
            tmem_load_atom,
            tDqDq_load_view,
        )
        thr_tmem_load_dq = tiled_tmem_load_dq.get_slice(wg_tidx)
        tDqDq_t2r = thr_tmem_load_dq.partition_S(tDqDq_load_view)
        if const_expr(self.use_cross_row_persistent):
            tiled_tmem_load_dq_p1 = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tDqDq_p1_load_view,
            )
            thr_tmem_load_dq_p1 = tiled_tmem_load_dq_p1.get_slice(wg_tidx)
            tDqDq_p1_t2r = thr_tmem_load_dq_p1.partition_S(
                tDqDq_p1_load_view,
            )
        if const_expr(use_stmatrix_dq):
            cDQ = cute.make_identity_tensor(
                (self.heads_padded, self.head_dim_padded),
            )
        else:
            cDQ = cute.make_identity_tensor(dq_acc_shape)
        tCcDQ = thr_tmem_load_dq.partition_D(cDQ)
        tDQrDQ_shape = tCcDQ.shape
        if const_expr(use_stmatrix_dq):
            smem_store_atom_dq = sm100_utils_basic.get_smem_store_op(
                LayoutEnum.ROW_MAJOR,
                self.q_dtype,
                self.acc_dtype,
                tiled_tmem_load_dq,
            )
            tiled_smem_store_dq = cute.make_tiled_copy_D(
                smem_store_atom_dq,
                tiled_tmem_load_dq,
            )
            thr_smem_store_dq = tiled_smem_store_dq.get_slice(wg_tidx)
            tRDQ_sdQ = thr_smem_store_dq.partition_D(sdQ_epi_slice)

        # ---- Wait for W loaded by load warp ----
        cute.arch.mbarrier_wait(
            gw_ready_barrier,
            Int32(gw_ready_phase),
        )

        # ---- Per topk-block iteration (2-stage S/dS) ----
        s_full_0_phase = Int32(0)
        s_full_1_phase = Int32(0)

        dw_accum = cute.make_rmem_tensor(tSrS_shape, Float32)
        for ei in cutlass.range_constexpr(cute.size(dw_accum)):
            dw_accum[ei] = Float32(0.0)

        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        # For the production 64x128 accumulator and 16dp256b8x ownership,
        # every thread visits exactly two heads. Hoist those two conversions
        # instead of reloading BF16 weights for every score pair.
        if const_expr(use_stmatrix_ds):
            h_base = warp_id_in_wg * 16 + lane_id // 4
            weight_lo = Float32(sW[h_base])
            weight_hi = Float32(sW[h_base + 8])

        for bi in cutlass.range(0, self.num_topk_blocks):
            i_st = bi * self.block_I

            # Wait for S ready from MMA (per-stage barrier)
            if bi % 2 == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_S_FULL_0, s_full_0_phase)
                s_full_0_phase ^= 1
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_S_FULL_1, s_full_1_phase)
                s_full_1_phase ^= 1
            _tcgen05_fence_after_thread_sync()
            if bi % 2 == 0:
                cute.copy(tiled_tmem_load_s_0, tStS_t2r_0, tSrS)
            else:
                cute.copy(tiled_tmem_load_s_1, tStS_t2r_1, tSrS)

            # Phase 1: Compute dS (→ tSrS), accumulate dW — paired f32x2.
            for ei in cutlass.range(0, cute.size(tSrS), 2):
                if const_expr(use_stmatrix_ds):
                    h0 = cute.get(tCcS[ei], mode=[0])
                    n0 = cute.get(tCcS[ei], mode=[1])
                    h1 = cute.get(tCcS[ei + 1], mode=[0])
                    n1 = cute.get(tCcS[ei + 1], mode=[1])
                else:
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

                if const_expr(use_stmatrix_ds):
                    # Pairs alternate between the thread's low/high head;
                    # both elements in one pair share that head.
                    pair_weight = weight_lo if (ei // 2) % 2 == 0 else weight_hi
                    w0 = pair_weight
                    w1 = pair_weight
                else:
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

                ds0, ds1 = mul_packed_f32x2(
                    (gs0, gs1),
                    (w0, w1),
                )
                tSrS[ei] = ds0 if s_pos_0 else Float32(0.0)
                tSrS[ei + 1] = ds1 if s_pos_1 else Float32(0.0)

            cute.arch.fence_view_async_tmem_load()

            # Phase 2: Convert dS f32→bf16, then use STSM on the production
            # tile or the native-layout coordinate fallback on other shapes.
            tSrS_f16 = cute.make_rmem_tensor(tSrS.shape, self.q_dtype)
            for ei in cutlass.range_constexpr(cute.size(tSrS)):
                tSrS_f16[ei] = self.q_dtype(tSrS[ei])

            if const_expr(use_stmatrix_ds):
                tRS_rdS = tiled_smem_store.retile(tSrS_f16)
                if const_expr(self.use_ds_half):
                    # The final retiled-copy mode is the two 64-column
                    # repetitions. Publish columns 0..63 first so MMA can
                    # issue the first half of dQ while this warpgroup stores
                    # columns 64..127; DS_READY still covers the full tile.
                    if bi % 2 == 0:
                        cute.copy(
                            tiled_smem_store,
                            tRS_rdS[(None, None, 0)],
                            tRS_sdS[(None, None, 0, 0)],
                        )
                    else:
                        cute.copy(
                            tiled_smem_store,
                            tRS_rdS[(None, None, 0)],
                            tRS_sdS[(None, None, 0, 1)],
                        )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    _tcgen05_fence_before_thread_sync()
                    if bi % 2 == 0:
                        cute.arch.mbarrier_arrive(mbar + MBAR_DS_HALF_0)
                    else:
                        cute.arch.mbarrier_arrive(mbar + MBAR_DS_HALF_1)

                    if bi % 2 == 0:
                        cute.copy(
                            tiled_smem_store,
                            tRS_rdS[(None, None, 1)],
                            tRS_sdS[(None, None, 1, 0)],
                        )
                    else:
                        cute.copy(
                            tiled_smem_store,
                            tRS_rdS[(None, None, 1)],
                            tRS_sdS[(None, None, 1, 1)],
                        )
                else:
                    if bi % 2 == 0:
                        cute.copy(
                            tiled_smem_store,
                            tRS_rdS,
                            tRS_sdS[(None, None, None, 0)],
                        )
                    else:
                        cute.copy(
                            tiled_smem_store,
                            tRS_rdS,
                            tRS_sdS[(None, None, None, 1)],
                        )
            else:
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
            _tcgen05_fence_before_thread_sync()

            if bi % 2 == 0:
                cute.arch.mbarrier_arrive(mbar + MBAR_DS_READY_0)
            else:
                cute.arch.mbarrier_arrive(mbar + MBAR_DS_READY_1)

        # ---- Step 3: After all iterations — dQ via TMA store, dW via warp reduction ----

        # Wait for MMA warp to finish the final GEMM3 (dQ accumulation).
        cute.arch.mbarrier_wait(
            mbar + MBAR_DQ_DONE,
            Int32(persistent_row_phase),
        )
        _tcgen05_fence_after_thread_sync()

        tDQrDQ = cute.make_rmem_tensor(tDQrDQ_shape, Float32)
        if const_expr(self.use_cross_row_persistent):
            if (persistent_row_phase & 1) == 0:
                cute.copy(tiled_tmem_load_dq, tDqDq_t2r, tDQrDQ)
            else:
                cute.copy(
                    tiled_tmem_load_dq_p1,
                    tDqDq_p1_t2r,
                    tDQrDQ,
                )
        else:
            cute.copy(tiled_tmem_load_dq, tDqDq_t2r, tDQrDQ)

        tDQrDQ_bf16 = cute.make_rmem_tensor(tDQrDQ.shape, self.q_dtype)
        for ei in cutlass.range(0, cute.size(tDQrDQ), 2):
            scaled0, scaled1 = mul_packed_f32x2(
                (tDQrDQ[ei], tDQrDQ[ei + 1]),
                (Float32(sm_scale), Float32(sm_scale)),
            )
            tDQrDQ_bf16[ei] = self.q_dtype(scaled0)
            tDQrDQ_bf16[ei + 1] = self.q_dtype(scaled1)

        cute.arch.fence_view_async_tmem_load()
        _tcgen05_fence_before_thread_sync()

        if const_expr(self.use_cross_row_persistent):
            # MMA may start accumulating row it+2 into this TMEM parity as
            # soon as every compute thread has completed its dQ T2R.
            cute.arch.mbarrier_arrive(dq_free_barrier)

            # The dedicated dQ SMEM tile is single-buffered.  Drain row it-1's
            # TMA store before any thread overwrites it, then rendezvous the
            # whole compute warpgroup.
            if warp_idx == compute_warp0:
                dQ_store_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

        if const_expr(use_stmatrix_dq):
            tRDQ_rdQ = tiled_smem_store_dq.retile(tDQrDQ_bf16)
            cute.copy(tiled_smem_store_dq, tRDQ_rdQ, tRDQ_sdQ)
        else:
            # General-shape dQ staging via coordinate writes.
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
            if const_expr(not self.use_cross_row_persistent):
                dQ_store_pipeline.producer_acquire()
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)
            dQ_store_pipeline.producer_commit()

        if const_expr(use_stmatrix_ds):
            # 16dp256b8x gives each thread two heads, selected by pair bit 0;
            # the four lanes sharing lane//4 cover disjoint columns for those
            # same heads.  Reduce only that 4-lane subgroup instead of doing
            # 16 full-warp reductions and repeatedly scanning all 64 values.
            sum_low = Float32(0.0)
            sum_high = Float32(0.0)
            for ei in cutlass.range_constexpr(cute.size(dw_accum)):
                if (ei // 2) % 2 == 0:
                    sum_low = sum_low + dw_accum[ei]
                else:
                    sum_high = sum_high + dw_accum[ei]
            sum_low = cute.arch.warp_reduction_sum(
                sum_low,
                threads_in_group=4,
            )
            sum_high = cute.arch.warp_reduction_sum(
                sum_high,
                threads_in_group=4,
            )
            if lane_id % 4 == 0:
                h0 = warp_id_in_wg * 16 + lane_id // 4
                mdW[seq_idx, h0, batch_idx] = self.q_dtype(sum_low)
                mdW[seq_idx, h0 + 8, batch_idx] = self.q_dtype(sum_high)
        else:
            HEADS_PER_WARP = const_expr(self.heads_padded // 4)
            warp_base_h = warp_id_in_wg * Int32(HEADS_PER_WARP)
            for h_local in cutlass.range_constexpr(HEADS_PER_WARP):
                h = warp_base_h + h_local
                my_partial = Float32(0.0)
                for ei in cutlass.range_constexpr(cute.size(dw_accum)):
                    if const_expr(use_stmatrix_ds):
                        elem_h = cute.get(tCcS[ei], mode=[0])
                    else:
                        elem_h = cute.get(tCcS[ei], mode=[0, 0])
                    if elem_h == h:
                        my_partial = my_partial + dw_accum[ei]
                total = cute.arch.warp_reduction_sum(my_partial)
                if lane_id == 0:
                    mdW[seq_idx, h, batch_idx] = self.q_dtype(total)

    # =========================================================================
    # Warps 12-15: Reduce warpgroup (wide dK T2R + bulk FP32 reduce, 2-stage)
    # =========================================================================
    @cute.jit
    def _reduce_warpgroup(
        self,
        mdK_f32,
        sTopkIdxs,
        sdKStage,
        dk_acc_shape,
        tDkDk_0,
        tDkDk_1,
        sm_scale: Float32 | float,
        seqlen_k,
        tidx,
        mbar,
    ):
        """Reduce dK through padded SMEM and bulk FP32 global additions."""
        wg_tidx = tidx % self.WARPGROUP_SIZE
        lane_id = wg_tidx % self.WARP_SIZE
        warp_in_wg = wg_tidx // self.WARP_SIZE

        tmem_load_atom_dk = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(16)),
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
                _tcgen05_fence_after_thread_sync()
                cute.copy(tiled_tmem_load_dk_0, tDkDk_t2r_0, tDKrDK)
                cute.arch.fence_view_async_tmem_load()
                _tcgen05_fence_before_thread_sync()
                if bi == self.num_topk_blocks - 1:
                    cute.arch.mbarrier_arrive(mbar + MBAR_REDUCE_DONE)
                cute.arch.mbarrier_arrive(mbar + MBAR_DK_EMPTY_0)
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_DK_FULL_1, dk_full_1_phase)
                dk_full_1_phase ^= 1
                _tcgen05_fence_after_thread_sync()
                cute.copy(tiled_tmem_load_dk_1, tDkDk_t2r_1, tDKrDK)
                cute.arch.fence_view_async_tmem_load()
                _tcgen05_fence_before_thread_sync()
                if bi == self.num_topk_blocks - 1:
                    cute.arch.mbarrier_arrive(mbar + MBAR_REDUCE_DONE)
                cute.arch.mbarrier_arrive(mbar + MBAR_DK_EMPTY_1)

            # The 16dp256b16x map materializes eight rows per warp into one
            # padded SMEM ping-pong buffer; lanes 0..7 issue one 512-byte FP32
            # bulk reduction per row.
            stage_row = lane_id // 4
            WARP_STAGE_FLOATS = const_expr(self.DK_STAGE_BUFFERS * self.DK_STAGE_ROWS * self.DK_STAGE_ROW_FLOATS)
            BUFFER_FLOATS = const_expr(self.DK_STAGE_ROWS * self.DK_STAGE_ROW_FLOATS)
            for pass_idx in cutlass.range_constexpr(4):
                if lane_id < 8:
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                cute.arch.sync_warp()

                buffer_base = warp_in_wg * WARP_STAGE_FLOATS + (pass_idx % 2) * BUFFER_FLOATS
                for b6 in cutlass.range_constexpr(2):
                    for col_group in cutlass.range_constexpr(8):
                        ei = (pass_idx // 2) * 64 + b6 * 32 + col_group * 4 + (pass_idx % 2) * 2
                        col = (lane_id % 4) * 2 + col_group * 8 + b6 * 64
                        dst = buffer_base + stage_row * self.DK_STAGE_ROW_FLOATS + col
                        scaled0, scaled1 = mul_packed_f32x2(
                            (tDKrDK[ei], tDKrDK[ei + 1]),
                            (Float32(sm_scale), Float32(sm_scale)),
                        )
                        sdKStage[dst] = scaled0
                        sdKStage[dst + 1] = scaled1

                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane_id < 8:
                    n = warp_in_wg * 32 + (pass_idx // 2) * 16 + (pass_idx % 2) * 8 + lane_id
                    topk_idx = Int32(sTopkIdxs[bi * self.block_I + n])
                    if topk_idx >= 0 and topk_idx < seqlen_k:
                        cpasync_reduce_bulk_add_f32(
                            sdKStage.iterator + buffer_base + lane_id * self.DK_STAGE_ROW_FLOATS,
                            mdK_f32[topk_idx, None].iterator,
                            self.head_dim_padded * 4,
                        )
                    # Advance every issuing lane's group sequence even for an
                    # invalid sparse row, matching the bulk pipeline contract.
                    cute.arch.cp_async_bulk_commit_group()

        if const_expr(not self.use_cross_row_persistent) and lane_id < 8:
            # Full completion (not just shared-read completion) before the
            # following stream-ordered cast can consume dK_f32.
            cute.arch.cp_async_bulk_wait_group(0)

    # =========================================================================
    # Warps 8-11: K loading warpgroup (3-stage sK)
    # =========================================================================
    @cute.jit
    def _k_load_warpgroup(
        self,
        mK,
        sK,
        sK_raw_ptr,
        sTopkIdxs,
        mTopkIdx,
        tma_atom_K_gather,
        seq_idx,
        batch_idx,
        seqlen_k,
        tidx,
        mbar,
        persistent_row_phase,
    ):
        """K loading warpgroup: sparse cp.async gather into 3-stage sK.

        3-stage pipeline allows K-load to run 3 blocks ahead of MMA,
        hiding the scatter gather latency (~5300 clk per block).
        """
        wg_tidx = tidx % self.WARPGROUP_SIZE

        if const_expr(self.use_tma_gather):
            if const_expr(self.use_cross_row_persistent):
                # Treat the four K blocks of every row as one flattened
                # producer stream.  Rotating by global block number avoids
                # the 0,1,2,0 -> 0,1,2,0 row-boundary restart and lets the
                # gather role remain three blocks ahead across rows.
                topk_row_ptr = mTopkIdx[seq_idx, None, batch_idx].iterator
                K_STAGE_BYTES = const_expr(self.block_I * self.head_dim_padded * self.k_dtype.width // 8)
                warp_in_wg = wg_tidx // self.WARP_SIZE
                lane_in_warp = wg_tidx % self.WARP_SIZE
                for bi in cutlass.range_constexpr(self.num_topk_blocks):
                    global_block = persistent_row_phase * self.num_topk_blocks + bi
                    stage = global_block % 3
                    stage_occurrence = global_block // 3
                    if global_block >= 3:
                        cute.arch.mbarrier_wait(
                            mbar + MBAR_K_CONSUMED_0 + stage,
                            Int32((stage_occurrence - 1) & 1),
                        )

                    if wg_tidx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + MBAR_K_LOADED_0 + stage,
                            K_STAGE_BYTES,
                        )
                    if lane_in_warp < 8:
                        n = warp_in_wg * 32 + lane_in_warp * 4
                        idx_pos = bi * self.block_I + n
                        row0, row1, row2, row3 = _load_global_i32x4(
                            topk_row_ptr + idx_pos,
                        )
                        row0 = row0 if row0 >= 0 and row0 < seqlen_k else Int32(-1)
                        row1 = row1 if row1 >= 0 and row1 < seqlen_k else Int32(-1)
                        row2 = row2 if row2 >= 0 and row2 < seqlen_k else Int32(-1)
                        row3 = row3 if row3 >= 0 and row3 < seqlen_k else Int32(-1)
                        for stripe in cutlass.range_constexpr(2):
                            dst_offset = stage * self.block_I * self.head_dim_padded + stripe * self.block_I * 64 + n * 64
                            _tma_gather4_k_rows(
                                tma_atom_K_gather,
                                sK_raw_ptr + dst_offset,
                                stripe * 64,
                                row0,
                                row1,
                                row2,
                                row3,
                                mbar + MBAR_K_LOADED_0 + stage,
                            )
                return

            k_consumed_0_phase_tma = Int32(0)
            k_consumed_1_phase_tma = Int32(persistent_row_phase & 1)
            k_consumed_2_phase_tma = Int32(persistent_row_phase & 1)
            K_STAGE_BYTES = const_expr(self.block_I * self.head_dim_padded * self.k_dtype.width // 8)
            for bi in cutlass.range_constexpr(self.num_topk_blocks):
                if bi >= 3:
                    if bi % 3 == 0:
                        cute.arch.mbarrier_wait(
                            mbar + MBAR_K_CONSUMED_0,
                            k_consumed_0_phase_tma,
                        )
                        k_consumed_0_phase_tma ^= 1
                    elif bi % 3 == 1:
                        cute.arch.mbarrier_wait(
                            mbar + MBAR_K_CONSUMED_1,
                            k_consumed_1_phase_tma,
                        )
                        k_consumed_1_phase_tma ^= 1
                    else:
                        cute.arch.mbarrier_wait(
                            mbar + MBAR_K_CONSUMED_2,
                            k_consumed_2_phase_tma,
                        )
                        k_consumed_2_phase_tma ^= 1

                stage = const_expr(bi % 3)
                if wg_tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + MBAR_K_LOADED_0 + stage,
                        K_STAGE_BYTES,
                    )
                # One Gather4 atom is 4 rows x 64 bf16 columns. Lanes 0..7 of
                # each warp issue two stripes, matching the hardware layout.
                warp_in_wg = wg_tidx // self.WARP_SIZE
                lane_in_warp = wg_tidx % self.WARP_SIZE
                if lane_in_warp < 8:
                    n = warp_in_wg * 32 + lane_in_warp * 4
                    idx_pos = bi * self.block_I + n
                    row0 = Int32(sTopkIdxs[idx_pos])
                    row1 = Int32(sTopkIdxs[idx_pos + 1])
                    row2 = Int32(sTopkIdxs[idx_pos + 2])
                    row3 = Int32(sTopkIdxs[idx_pos + 3])
                    row0 = row0 if row0 >= 0 and row0 < seqlen_k else Int32(-1)
                    row1 = row1 if row1 >= 0 and row1 < seqlen_k else Int32(-1)
                    row2 = row2 if row2 >= 0 and row2 < seqlen_k else Int32(-1)
                    row3 = row3 if row3 >= 0 and row3 < seqlen_k else Int32(-1)
                    for stripe in cutlass.range_constexpr(2):
                        # Raw (unswizzled) stage address. The descriptor's
                        # 128B swizzle maps this base to sK's UMMA layout.
                        dst_offset = stage * self.block_I * self.head_dim_padded + stripe * self.block_I * 64 + n * 64
                        _tma_gather4_k_rows(
                            tma_atom_K_gather,
                            sK_raw_ptr + dst_offset,
                            stripe * 64,
                            row0,
                            row1,
                            row2,
                            row3,
                            mbar + MBAR_K_LOADED_0 + stage,
                        )
            return

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
        # mK is the flat (B*S_k, D) view; SMEM already holds global IDs.

        k_consumed_0_phase_kload = Int32(0)
        k_consumed_1_phase_kload = Int32(persistent_row_phase)
        k_consumed_2_phase_kload = Int32(persistent_row_phase)

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
                topk_idx = Int32(sTopkIdxs[idx_pos])
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
# compile_key -> compiled GEMM kernel. Single-layer cache, same pattern as the
# forward ``_interface.py`` and ``_score_grad_cute_cache`` below: the key holds
# only params that change the generated code, and entries are filled lazily on
# first execute (``_ensure_compiled``) because ``cute.compile`` needs the real
# tensors that the shape-only factory call doesn't have yet.
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
    # ``batch``/``seqlen``/``seqlen_k`` specialize the persistent-row schedule
    # and Gather4 descriptor extent, so they are part of the GEMM compile key.
    # ``grad_scale`` remains runtime-only (forwarded into ``score_grad`` as a
    # ``Float32`` at call time), so changing loss scaling does not recompile.
    #
    # ``topk_indices_global`` selects the topk-id contract:
    #   True  (default): mTopkIdx carries global flat ids — load directly.
    #   False (legacy):  mTopkIdx carries local-per-batch ids — kernel adds
    #                    ``batch_idx * S_k_per_batch`` to convert to global
    #                    flat for the (B*S_k, D) K/dK view.
    # Const_expr-branched in the kernel, so it **is** part of the compile key.
    # THD packed varlen is supported at the wrapper level by treating the
    # packed tensors as a single B=1 BSHD batch (sparse path's topk indices
    # already encode per-batch validity, so no kernel-side cu_seqlens are
    # needed). See ``_indexer_backward_sparse_thd`` in csrc/bwd/__init__.py.
    return _build_cute_dsl_kernel(
        batch,
        seqlen,
        seqlen_k,
        heads,
        dim,
        topk,
        sm_scale,
        block_I,
        topk_indices_global=topk_indices_global,
    )


class ScoreGradSm100:
    """CuTe DSL kernel for in-place score_grad precompute."""

    THREADS_PER_CTA = 256
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
            # One partial per warp. The former implementation staged all 128
            # thread partials and reduced them serially in thread 0.
            warp_sums: cute.struct.Align[cute.struct.MemRange[Float32, self.NUM_WARPS], 128]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_sums = storage.warp_sums.get_tensor(cute.make_layout((self.NUM_WARPS,), stride=(1,)))

        local_sum = Float32(0.0)
        for pos in cutlass.range(tidx, self.topk, self.THREADS_PER_CTA):
            target = Float32(mAttnScore[seq_idx, pos, batch_idx])
            predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
            target_eff = cute.arch.fmax(target, Float32(CLIP_PROB_MIN))
            log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
            local_sum += -target_eff * log_clip_mask * grad_scale_f32

        warp_idx = tidx // self.WARP_SIZE
        warp_sum = cute.arch.warp_reduction_sum(local_sum)
        with cute.arch.elect_one():
            warp_sums[warp_idx] = warp_sum
        cute.arch.sync_threads()

        sum_grad = Float32(0.0)
        for warp in cutlass.range_constexpr(self.NUM_WARPS):
            sum_grad += warp_sums[warp]
        for pos in cutlass.range(tidx, self.topk, self.THREADS_PER_CTA):
            target = Float32(mAttnScore[seq_idx, pos, batch_idx])
            predict = Float32(mIndexScore[seq_idx, pos, batch_idx])
            target_eff = cute.arch.fmax(target, Float32(CLIP_PROB_MIN))
            log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
            g_i = -target_eff * log_clip_mask * grad_scale_f32
            mAttnScore[seq_idx, pos, batch_idx] = g_i - predict * sum_grad


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
            options=compile_options("--opt-level 2"),
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

    Results overwrite AttnScore in-place with grad_signal. IndexScore remains
    unchanged after being read as the predict input.

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


def _build_cute_dsl_kernel(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale,
    block_I,
    topk_indices_global: bool = True,
):
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("Requires SM100+")
    persistent_grid_size = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    kernel_obj = IndexerBackwardSm100(
        head_dim=dim,
        heads=heads,
        block_I=block_I,
        topk=topk,
        total_seqlen_k=batch * seqlen_k,
        total_rows=batch * seqlen,
        persistent_grid_size=persistent_grid_size,
        topk_indices_global=topk_indices_global,
    )

    compile_key = (
        batch,
        seqlen,
        seqlen_k,
        heads,
        dim,
        topk,
        block_I,
        topk_indices_global,
    )

    def _ensure_compiled(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, AttnScore, TopkIndices, current_stream=None):
        """Lazy-compile the GEMM kernel (kernel 2) on first execute (needs real tensors)."""
        if compile_key not in _compile_cache:
            s = _resolve_stream(current_stream)
            cute_args = [to_cute_tensor(t) for t in [IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, AttnScore, TopkIndices]]
            _compile_cache[compile_key] = cute.compile(
                kernel_obj,
                *cute_args,
                cutlass.Float32(sm_scale),
                s,
                options=compile_options("--opt-level 2"),
            )

    def _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, TopkIndices, current_stream=None):
        """Run only kernel 2 (GEMM). Caller must have run kernel 1 and zeroed dIndexK_f32."""
        s = _resolve_stream(current_stream)
        _ensure_compiled(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, TopkIndices, current_stream=current_stream)
        with torch.cuda.nvtx.range("indexer_backward_dsl_gemm"):
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
            )

    def _run(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK, AttnScore, IndexScore, TopkIndices, GradLoss, grad_scale, current_stream=None):
        # ``grad_scale`` is a host scalar (Python float / 0-D fp32 tensor)
        # multiplied into ``score_grad`` as a runtime ``Float32`` arg —
        # changing it across calls does **not** trigger recompilation.
        score_grad = partial(_score_grad_inplace, block_I=block_I)

        # Kernel 1: Compute grad_signal from scores (CuTe DSL only).
        score_grad(AttnScore, IndexScore, GradLoss, grad_scale, current_stream=current_stream)

        if dIndexK.dtype == torch.float32:
            # fp32 output: the dK epilogue atomic-adds into this buffer, so it
            # must start zeroed. Zero it internally on the selected stream
            # (cheap; removes the fragile caller pre-zero contract) rather than
            # trusting the caller. This zero-init is a promised stage of the
            # execute pipeline (see the IndexerBackward docstring) and mirrors
            # the SM90 backend and the DenseIndexerBackward fp32 paths, which
            # zero their fp32 dK buffer the same way.
            with _torch_stream_context(current_stream):
                dIndexK.zero_()
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
