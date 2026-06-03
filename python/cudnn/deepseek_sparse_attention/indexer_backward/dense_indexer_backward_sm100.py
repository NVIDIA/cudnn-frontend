"""
Indexer Backward — Dense Mode SM100 CuTe-DSL, 3-kernel design.

Three kernels launched sequentially on the same stream:

  Kernel 1 (CuTe DSL): ScoreGradDense — scalar elementwise + reduction kernel.
      From raw score + denom, normalize → softmax backprop reduction → compute
      grad_signal, in-place overwrite PredictRaw buffer.
      Grid = (B, max_seqlen_q, 1), Block = (128, 1, 1), 4 warps.

  Kernel 2 (CuTe DSL): DenseIndexerBackward2QGemmSm100 — warp-specialized GEMM kernel.
      Three GEMMs (S, dK, dQ) with elementwise dS/dW computation.
      Two Q tokens per CTA (K-reuse): 6 GEMMs / K block.
      Reads pre-computed grad_signal from GMEM via Load warp (2-stage matching K).
      dK accumulated in float32 via cp.reduce.async.bulk.
      Grid = (B, ceil(max_seqlen_q/2), 1), Block = (384, 1, 1), 12 warps.

  Kernel 3 (PyTorch): dk_convert — cast dK from float32 to output dtype.

Kernel 2 — Warp specialization (12 warps, 384 threads):
  Warp 0:      Load (TMA Q×2 + TMA K 2-stage + per-block grad_signal + W)
  Warp 1:      MMA  (Phase A q0 + Phase B q1, 6 GEMMs / K block)
  Warps 2-3:   Idle (register donors)
  Warps 4-7:   Compute warpgroup
               T2R S → dS/dW (with grad_signal) → coord write sdS → dQ/dW epilogue
  Warps 8-11:  Reduce warpgroup
               TMEM T2R dK → sdK_reduce (SMEM) → cp.reduce.async.bulk to dK_f32

BSHD vs THD (explicit branches, no implicit packing):

  cu_seqlens_q/k = None  →  BSHD mode:
    Q  : (B, S_q, H, D)         K  : (B, S_k, D)
    W  : (B, S_q, H)            dW : (B, S_q, H)
    dQ : (B, S_q, H, D)         dK : (B, S_k, D)
    GradSignal / Score : (B, S_q, S_k)
    LSE / L1Norm       : (B, S_q)

  cu_seqlens_q/k present  →  THD packed mode (no batch dim):
    Q  : (T_q, H, D)            K  : (T_k, D)
    W  : (T_q, H)               dW : (T_q, H)
    dQ : (T_q, H, D)            dK : (T_k, D)
    GradSignal / Score : (T_q, max_seqlen_k)   [second dim is batch-local k]
    LSE / L1Norm       : (T_q,)

Bottom-right ratio causal:
    q_start_b  = seqlen_k_b * ratio - seqlen_q_b
    col_limit  = min(seqlen_k_b, (q_start_b + q_local + 1) // ratio)
  When seqlen_q_b == seqlen_k_b * ratio, q_start_b == 0 (legacy behavior).
  Constraint per batch: seqlen_q_b <= seqlen_k_b * ratio.
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

from cudnn.deepseek_sparse_attention.utils.copy import cpasync_reduce_bulk_add_f32
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream as _resolve_stream
from cudnn.deepseek_sparse_attention.utils.seqlen import seqlen_info as _seqlen_info

mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")

DENOM_EPS = 1e-10
CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)


# =====================================================================
# Per-batch seqlen / offset helper
# =====================================================================
# Returns (q_offset, k_offset, seqlen_q_b, seqlen_k_b) for `batch_idx`.
#   BSHD: cu_seqlens are None; offsets are 0; lengths are static (S_q / S_k).
#   THD : reads cu_seqlens; offsets and lengths are dynamic per batch.
# Branches at compile time on `mCuSeqlensQ is None` (constexpr).

# =====================================================================
# Kernel 1: ScoreGradDense — standalone score gradient kernel
# =====================================================================


class ScoreGradDense:
    """Kernel 1: Normalize raw scores → clipped-log KL backprop → grad_signal.

    Pure scalar elementwise + reduction kernel. 128 threads, 4 warps, no warp
    specialization, no TC/MMA.

    Two-phase algorithm:
      Phase 1: 128 threads stride-128 over S_k, compute g per position
               (dL/dlog_predict under clipped log), accumulate local_sum.
               Cross-warp reduce → sum_grad.
      Phase 2: Re-traverse S_k, recompute predict/g (L1 hit), write
               grad_signal = g - predict * sum_grad in-place to PredictRaw.

    SMEM: only 16 bytes (4 × fp32 cross-warp reduction scratch).
    """

    WARP_SIZE = 32
    THREADS_PER_CTA = 128

    def __init__(
        self,
        max_seqlen_q: int,
        max_seqlen_k: int,
        ratio: int = 1,
    ):
        assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
        # ``grad_scale`` is intentionally **not** stored on the object —
        # baking it as ``self.grad_scale`` would freeze it into the
        # compiled kernel (every change → recompile). Keep it as a runtime
        # ``Float32`` arg on ``__call__`` so the same compiled kernel can
        # be reused when the runtime loss scaling changes for the same
        # tensor shape.
        # For BSHD: max_seqlen_q == S_q, max_seqlen_k == S_k.
        # For THD : max_seqlen_q / k cap the grid + row-stride respectively.
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.ratio = ratio

    @cute.jit
    def __call__(self, mPredictRaw, mTargetRaw, mLSE, mDenomTarget, mCuSeqlensQ, mCuSeqlensK, grad_scale: Float32, stream):
        # BSHD: mCuSeqlensQ/K = None; tensors carry batch dim.
        #   PredictRaw/TargetRaw: (B, S_q, S_k)   LSE/Denom: (B, S_q)
        # THD: mCuSeqlensQ/K provided; tensors are packed.
        #   PredictRaw/TargetRaw: (T_q, max_seqlen_k)   LSE/Denom: (T_q,)
        is_varlen = const_expr(mCuSeqlensQ is not None)

        if const_expr(is_varlen):
            # THD: keep packed layout (T_q, max_K). Batch from cu_seqlens.
            batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1
        else:
            # BSHD: transpose (B, S_q, S_k) -> (S_q, S_k, B) and (B, S_q) -> (S_q, B)
            mPredictRaw = cute.make_tensor(
                mPredictRaw.iterator,
                cute.select(mPredictRaw.layout, mode=[1, 2, 0]),
            )
            mTargetRaw = cute.make_tensor(
                mTargetRaw.iterator,
                cute.select(mTargetRaw.layout, mode=[1, 2, 0]),
            )
            mLSE = cute.make_tensor(
                mLSE.iterator,
                cute.select(mLSE.layout, mode=[1, 0]),
            )
            mDenomTarget = cute.make_tensor(
                mDenomTarget.iterator,
                cute.select(mDenomTarget.layout, mode=[1, 0]),
            )
            batch_size = cute.size(mPredictRaw.shape[2])

        # Row stride along K dim — equals S_k (BSHD) or max_seqlen_k (THD).
        seqlen_k_pad = cute.size(mPredictRaw.shape[1])

        self.kernel_score_grad(
            mPredictRaw,
            mTargetRaw,
            mLSE,
            mDenomTarget,
            mCuSeqlensQ,
            mCuSeqlensK,
            grad_scale,
            seqlen_k_pad,
            Int32(self.max_seqlen_q),
            Int32(self.max_seqlen_k),
        ).launch(
            grid=(batch_size, self.max_seqlen_q, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_score_grad(
        self,
        mPredictRaw,
        mTargetRaw,
        mLSE,
        mDenomTarget,
        mCuSeqlensQ,
        mCuSeqlensK,
        grad_scale: Float32,
        seqlen_k_pad: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        batch_idx = cute.arch.block_idx()[0]
        seq_local = cute.arch.block_idx()[1]
        warp_id = tidx // self.WARP_SIZE
        lane_id = tidx % self.WARP_SIZE

        is_varlen = const_expr(mCuSeqlensQ is not None)

        q_offset, _k_offset, seqlen_q_b, seqlen_k_b = _seqlen_info(
            mCuSeqlensQ,
            mCuSeqlensK,
            Int32(batch_idx),
            seqlen_q_static,
            seqlen_k_static,
        )

        # Out-of-range CTAs (seq_local >= seqlen_q_b in this batch): skip work.
        # CuTe DSL forbids early `return`, so wrap the body in an `if` block.
        if seq_local < seqlen_q_b:
            # SMEM: 4 fp32 for cross-warp reduction scratch
            smem = cutlass.utils.SmemAllocator()

            @cute.struct
            class SharedStorage:
                sReduceScratch: cute.struct.Align[cute.struct.MemRange[Float32, 4], 128]

            storage = smem.allocate(SharedStorage)
            sReduceScratch = storage.sReduceScratch.get_tensor(cute.make_layout((4,), stride=(1,)))

            # Per-batch row views of the score / denom tensors.
            #   BSHD: (S_q, S_k, B) -> (S_q, S_k) by slicing batch_idx; LSE/Denom (S_q, B) -> (S_q,).
            #   THD : (T_q, max_K) -> domain-offset by q_offset along T_q; LSE/Denom (T_q,) similarly.
            if const_expr(is_varlen):
                mPredict_b = cute.domain_offset((q_offset, Int32(0)), mPredictRaw)
                mTarget_b = cute.domain_offset((q_offset, Int32(0)), mTargetRaw)
                mLSE_b = cute.domain_offset((q_offset,), mLSE)
                mDenom_b = cute.domain_offset((q_offset,), mDenomTarget)
            else:
                mPredict_b = mPredictRaw[None, None, batch_idx]
                mTarget_b = mTargetRaw[None, None, batch_idx]
                mLSE_b = mLSE[None, batch_idx]
                mDenom_b = mDenomTarget[None, batch_idx]

            # Scalar loads (shared across all threads in CTA)
            lse_val = mLSE_b[seq_local]
            denom_val = mDenom_b[seq_local]

            LOG2E = Float32(1.4426950408889634)

            # Bottom-right ratio causal:
            #   q_start_b  = seqlen_k_b * ratio - seqlen_q_b
            #   col_limit  = min(seqlen_k_b, (q_start_b + q_local + 1) // ratio)
            # Equals legacy ((q_local+1)//ratio) when seqlen_q_b == seqlen_k_b * ratio.
            ratio = Int32(self.ratio)
            q_start_b = seqlen_k_b * ratio - seqlen_q_b
            col_limit_raw = (q_start_b + Int32(seq_local) + Int32(1)) // ratio
            col_limit = col_limit_raw if col_limit_raw < seqlen_k_b else seqlen_k_b

            # --- Phase 1: Accumulate sum_grad ---
            local_sum = Float32(0.0)
            pos = tidx
            while pos < seqlen_k_pad:
                if pos < col_limit:
                    pr = mPredict_b[seq_local, pos]
                    tr = mTarget_b[seq_local, pos]

                    predict = cute.math.exp2((pr - lse_val) * LOG2E, fastmath=True)
                    target = tr / (denom_val + Float32(DENOM_EPS))
                    target_eff = target if target >= Float32(CLIP_PROB_MIN) else Float32(CLIP_PROB_MIN)
                    log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
                    g = -target_eff * log_clip_mask * grad_scale

                    local_sum = local_sum + g
                pos = pos + Int32(128)

            # Cross-warp reduction (4 warps)
            warp_sum = cute.arch.warp_reduction_sum(local_sum)
            with cute.arch.elect_one():
                sReduceScratch[warp_id] = warp_sum
            cute.arch.sync_threads()
            sum_grad = sReduceScratch[0] + sReduceScratch[1] + sReduceScratch[2] + sReduceScratch[3]

            # --- Phase 2: Write grad_signal in-place to PredictRaw ---
            # Padding columns (pos >= seqlen_k_b in THD, or pos >= col_limit) get 0.
            pos = tidx
            while pos < seqlen_k_pad:
                if pos < col_limit:
                    pr = mPredict_b[seq_local, pos]
                    tr = mTarget_b[seq_local, pos]

                    predict = cute.math.exp2((pr - lse_val) * LOG2E, fastmath=True)
                    target = tr / (denom_val + Float32(DENOM_EPS))
                    target_eff = target if target >= Float32(CLIP_PROB_MIN) else Float32(CLIP_PROB_MIN)
                    log_clip_mask = Float32(1.0) if predict >= Float32(CLIP_PROB_MIN) else Float32(0.0)
                    g = -target_eff * log_clip_mask * grad_scale

                    grad_signal = g - predict * sum_grad
                    mPredict_b[seq_local, pos] = grad_signal
                else:
                    # masked / padding: grad_signal must be 0 — downstream
                    # GEMM multiplies by this directly and must not contaminate dQ/dK/dW.
                    mPredict_b[seq_local, pos] = Float32(0.0)
                pos = pos + Int32(128)


# =====================================================================
# Kernel 2: DenseIndexerBackward2QGemmSm100 — 2Q Token K-Reuse variant
# =====================================================================

# Barrier indices
MBAR_2Q_S_FULL = 0  # MMA → Compute (phase-flipped 2×/block)
MBAR_2Q_DS_READY = 1  # Compute → MMA (phase-flipped 2×/block)
MBAR_2Q_DK_FULL = 2  # MMA → Reduce (1×/block, after Phase B GEMM2)
MBAR_2Q_DK_EMPTY = 3  # Reduce → MMA (1×/block)
MBAR_2Q_GS_LOADED_0 = 4  # Load → Compute (2-stage K pipeline)
MBAR_2Q_GS_LOADED_1 = 5
MBAR_2Q_W_LOADED = 6
NUM_2Q_BARRIERS = 7


class DenseIndexerBackward2QGemmSm100:
    """Kernel 2: Two Q-tokens per CTA, K-reuse optimization.

    Each CTA handles 2 Q tokens simultaneously, loading K data once for both.
    6 GEMMs per K block (2 phases x 3 GEMMs).

    TMEM layout:
      Offset 0:    S      (128 cols) — single buffer, Phase A/B alternate
      Offset 128:  dK     (128 cols) — single buffer, Phase A clear + Phase B accumulate
      Offset 256:  dQ_q0  (128 cols) — Q token 0 persistent accumulator
      Offset 384:  dQ_q1  (128 cols) — Q token 1 persistent accumulator

    Barriers: 7 custom.
    Grid: (batch, ceil(seqlen_q/2), 1).
    """

    arch = 100
    WARP_SIZE = 32
    WARPGROUP_SIZE = 128
    NUM_WARPS = 12
    THREADS_PER_CTA = 384

    # Warp assignments (12 warps total)
    load_warp_id = 0
    mma_warp_id = 1
    # Warps 2-3: idle (register donors)
    compute_warp_id = (4, 5, 6, 7)
    reduce_warp_id = (8, 9, 10, 11)

    def __init__(self, head_dim, heads=64, block_I=128):
        self.head_dim = head_dim
        self.heads = heads
        self.block_I = block_I
        assert heads >= 64

        self.head_dim_padded = int(math.ceil(head_dim / 16) * 16)
        self.heads_padded = int(math.ceil(heads / 8) * 8)

        # GEMM tilers (M, N, K) — same as 1Q
        self.gemm1_tiler = (self.heads_padded, self.block_I, self.head_dim_padded)
        self.gemm2_tiler = (self.block_I, self.head_dim_padded, self.heads_padded)
        self.gemm3_tiler = (self.heads_padded, self.head_dim_padded, self.block_I)

        self.acc_dtype = Float32

        # TMEM layout (2Q: S and dK CANNOT share)
        self.tmem_s_offset = 0  # S (single-buffered)
        self.tmem_dk_offset = 128  # dK (single-buffered, Phase A clear + Phase B accumulate)
        self.tmem_dq0_offset = 256  # dQ_q0 persistent accumulator
        self.tmem_dq1_offset = 384  # dQ_q1 persistent accumulator
        self.tmem_alloc_cols = 512

        # Register budgets
        self.num_regs_wg0 = 40
        self.num_regs_compute = 256
        self.num_regs_reduce = 200

        self.buffer_align_bytes = 1024

        # TMA config
        self.cluster_shape = (1, 1, 1, 1)
        self.Q_mbar_size = 4  # 2 Q tokens x 2 barriers each (producer + consumer)
        self.K_mbar_size = 4  # 2-stage: 2 producer + 2 consumer
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
        mCuSeqlensQ,
        mCuSeqlensK,
        sm_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        stream: cuda.CUstream,
    ):
        # Two explicit modes (no implicit packing):
        #   BSHD (mCuSeqlensQ/K = None):
        #     Q  (B, S_q, H, D)   K  (B, S_k, D)   W (B, S_q, H)
        #     dQ (B, S_q, H, D)   dK (B, S_k, D)   dW (B, S_q, H)
        #     GradSignal (B, S_q, S_k)
        #   THD (cu_seqlens given):
        #     Q  (T_q, H, D)      K  (T_k, D)      W (T_q, H)
        #     dQ (T_q, H, D)      dK (T_k, D)      dW (T_q, H)
        #     GradSignal (T_q, max_seqlen_k)
        is_varlen = const_expr(mCuSeqlensQ is not None)

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type

        # Layout transposes — sequence dim leading for TMA / per-batch views:
        #   BSHD Q  (B,S_q,H,D)  -> (H, D, S_q, B)     mode [2,3,1,0]
        #   THD  Q  (T_q,H,D)    -> (H, D, T_q)        mode [1,2,0]
        #   BSHD K  (B,S_k,D)    -> (S_k, D, B)        mode [1,2,0]
        #   THD  K  (T_k,D)      -> kept as is
        #   BSHD W/GS  (B,*,*)   -> (*, *, B)          mode [1,2,0]
        #   THD  W/GS  (*, *)    -> kept as is
        if const_expr(is_varlen):
            mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 2, 0]))
            mdQ = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 2, 0]))
            # K, W, dW, mdK_f32, GradSignal stay as is.
        else:
            mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[2, 3, 1, 0]))
            mdQ = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[2, 3, 1, 0]))
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 2, 0]))
            mdK_f32 = cute.make_tensor(mdK_f32.iterator, cute.select(mdK_f32.layout, mode=[1, 2, 0]))
            mW = cute.make_tensor(mW.iterator, cute.select(mW.layout, mode=[1, 2, 0]))
            mdW = cute.make_tensor(mdW.iterator, cute.select(mdW.layout, mode=[1, 2, 0]))
            mGradSignal = cute.make_tensor(
                mGradSignal.iterator,
                cute.select(mGradSignal.layout, mode=[1, 2, 0]),
            )

        cta_group = tcgen05.CtaGroup.ONE

        # All GEMMs: SS path
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

        # SMEM layouts
        # sQ: 2 separate single-stage regions (q0 and q1)
        sQ_q0_layout = _make_smem_layout_a(tmma1, self.gemm1_tiler, self.q_dtype, 1)
        sQ_q1_layout = _make_smem_layout_a(tmma1, self.gemm1_tiler, self.q_dtype, 1)
        sK_layout = _make_smem_layout_b(tmma1, self.gemm1_tiler, self.k_dtype, 2)  # 2-stage
        sdS_layout = _make_smem_layout_a(tmma3, self.gemm3_tiler, self.q_dtype, 2)  # 2-stage
        sdS_store_layout = _make_smem_layout_epi(
            self.q_dtype,
            LayoutEnum.COL_MAJOR,
            (self.heads_padded, self.block_I),
            2,
        )
        sdS_g2a_layout = _make_smem_layout_a(tmma2, self.gemm2_tiler, self.q_dtype, 2)
        sKt_layout = _make_smem_layout_b(tmma3, self.gemm3_tiler, self.k_dtype, 2)  # 2-stage
        sQ_q0_g2b_layout = _make_smem_layout_b(tmma2, self.gemm2_tiler, self.q_dtype, 1)
        sQ_q1_g2b_layout = _make_smem_layout_b(tmma2, self.gemm2_tiler, self.q_dtype, 1)

        # --- TMA atoms ---
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        cluster_layout_vmnk = cute.make_layout(self.cluster_shape)

        # TMA Q load — 2 separate TMA partitions for q0 and q1.
        # Descriptor is 4D (H,D,S_q,B) for BSHD, 3D (H,D,T_q) for THD.
        # The trailing 1-2 dims are runtime indices selected per CTA via local_tile.
        Q_q0_smem_layout_tma = cute.select(sQ_q0_layout, mode=[0, 1, 2])
        tma_atom_Q_q0, mQ_q0_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            Q_q0_smem_layout_tma,
            self.gemm1_tiler,
            tmma1,
            cluster_layout_vmnk.shape,
        )

        Q_q1_smem_layout_tma = cute.select(sQ_q1_layout, mode=[0, 1, 2])
        tma_atom_Q_q1, mQ_q1_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            Q_q1_smem_layout_tma,
            self.gemm1_tiler,
            tmma1,
            cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.q_dtype, Q_q0_smem_layout_tma)

        # TMA K load (2-stage). 3D (S_k,D,B) for BSHD, 2D (T_k,D) for THD.
        K_smem_layout_tma = cute.select(sK_layout, mode=[0, 1, 2])
        tma_atom_K, mK_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            K_smem_layout_tma,
            self.gemm1_tiler,
            tmma1,
            cluster_layout_vmnk.shape,
        )
        self.tma_copy_K_bytes = cute.size_in_bytes(self.k_dtype, K_smem_layout_tma)

        # dQ epilogue SMEM layout
        sdQ_epi_layout = _make_smem_layout_epi(
            self.q_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.heads_padded, self.head_dim_padded),
            1,
        )

        # TMA dQ store — 2 separate store descriptors for q0 and q1.
        # Same dim count as mQ: 4D for BSHD, 3D for THD.
        sdQ_epi_smem_layout = cute.select(sdQ_epi_layout, mode=[0, 1])
        tma_atom_dQ_q0, mdQ_q0_tma = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ,
            sdQ_epi_smem_layout,
            (self.heads_padded, self.head_dim_padded),
        )
        tma_atom_dQ_q1, mdQ_q1_tma = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ,
            sdQ_epi_smem_layout,
            (self.heads_padded, self.head_dim_padded),
        )

        if const_expr(is_varlen):
            batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1
        else:
            # mQ after transpose is (H, D, S_q, B); B is mode 3.
            batch_size = cute.size(mQ.shape[3])
        grid_q = (max_seqlen_q + 1) // 2  # ceil(max_seqlen_q / 2)

        self.kernel_gemm_dense_2q(
            mQ_q0_tma,
            mQ_q1_tma,
            mW,
            mK_tma,
            mdQ_q0_tma,
            mdQ_q1_tma,
            mdW,
            mdK_f32,
            mGradSignal,
            mCuSeqlensQ,
            mCuSeqlensK,
            sm_scale,
            Int32(max_seqlen_q),
            Int32(max_seqlen_k),
            tmma1,
            tmma2,
            tmma3,
            sQ_q0_layout,
            sQ_q1_layout,
            sdS_g2a_layout,
            sK_layout,
            sKt_layout,
            sdS_layout,
            sQ_q0_g2b_layout,
            sQ_q1_g2b_layout,
            sdS_store_layout,
            tma_atom_Q_q0,
            tma_atom_Q_q1,
            tma_atom_K,
            tma_atom_dQ_q0,
            tma_atom_dQ_q1,
            sdQ_epi_layout,
        ).launch(
            grid=(batch_size, grid_q, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel_gemm_dense_2q(
        self,
        mQ_q0,
        mQ_q1,
        mW,
        mK,
        mdQ_q0,
        mdQ_q1,
        mdW,
        mdK_f32,
        mGradSignal,
        mCuSeqlensQ,
        mCuSeqlensK,
        sm_scale: Float32 | float,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        tmma1,
        tmma2,
        tmma3,
        sQ_q0_layout,
        sQ_q1_layout,
        sdS_g2a_layout,
        sK_layout,
        sKt_layout,
        sdS_layout,
        sQ_q0_g2b_layout,
        sQ_q1_g2b_layout,
        sdS_store_layout,
        tma_atom_Q_q0,
        tma_atom_Q_q1,
        tma_atom_K,
        tma_atom_dQ_q0,
        tma_atom_dQ_q1,
        sdQ_epi_layout,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        batch_idx = cute.arch.block_idx()[0]
        seq_idx = cute.arch.block_idx()[1]

        is_varlen = const_expr(mCuSeqlensQ is not None)

        # Per-batch offsets / lengths.
        # BSHD: q_offset=k_offset=0; lengths static (S_q / S_k).
        # THD : both from cu_seqlens.
        q_offset, k_offset, seqlen_q_b, seqlen_k_b = _seqlen_info(
            mCuSeqlensQ,
            mCuSeqlensK,
            Int32(batch_idx),
            seqlen_q_static,
            seqlen_k_static,
        )

        # 2Q pair indices are batch-local. Per-batch tensor views are built
        # below so the rest of the kernel uses q0_local / q1_local uniformly
        # (no q_global needed — domain_offset / batch slice handles it).
        q0_local = Int32(seq_idx) * Int32(2)
        q1_local = q0_local + Int32(1)
        has_q0 = q0_local < seqlen_q_b
        has_q1 = q1_local < seqlen_q_b

        # Per-batch views — collapse BSHD batch dim and THD T-offset into a
        # single batch-local indexing scheme. After this, mQ_q0_b/mQ_q1_b have
        # shape (H, D, S_q_or_T_q) and mW_b/mdW_b/mGS_b/mdK_b have shape
        # (S_q_or_T_q, *).
        if const_expr(is_varlen):
            mQ_q0_b = cute.domain_offset((Int32(0), Int32(0), q_offset), mQ_q0)
            mQ_q1_b = cute.domain_offset((Int32(0), Int32(0), q_offset), mQ_q1)
            mdQ_q0_b = cute.domain_offset((Int32(0), Int32(0), q_offset), mdQ_q0)
            mdQ_q1_b = cute.domain_offset((Int32(0), Int32(0), q_offset), mdQ_q1)
            mK_b = cute.domain_offset((k_offset, Int32(0)), mK)
            mdK_b = cute.domain_offset((k_offset, Int32(0)), mdK_f32)
            mW_b = cute.domain_offset((q_offset, Int32(0)), mW)
            mdW_b = cute.domain_offset((q_offset, Int32(0)), mdW)
            mGS_b = cute.domain_offset((q_offset, Int32(0)), mGradSignal)
        else:
            mQ_q0_b = mQ_q0[None, None, None, batch_idx]
            mQ_q1_b = mQ_q1[None, None, None, batch_idx]
            mdQ_q0_b = mdQ_q0[None, None, None, batch_idx]
            mdQ_q1_b = mdQ_q1[None, None, None, batch_idx]
            mK_b = mK[None, None, batch_idx]
            mdK_b = mdK_f32[None, None, batch_idx]
            mW_b = mW[None, None, batch_idx]
            mdW_b = mdW[None, None, batch_idx]
            mGS_b = mGradSignal[None, None, batch_idx]

        num_kv_blocks = (seqlen_k_b + self.block_I - 1) // self.block_I

        # TMA descriptor prefetch (load warp only)
        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q_q0)
            cpasync.prefetch_descriptor(tma_atom_Q_q1)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_dQ_q0)
            cpasync.prefetch_descriptor(tma_atom_dQ_q1)

        # SMEM allocation
        sQ_q0_size = cute.cosize(sQ_q0_layout)
        sQ_q1_size = cute.cosize(sQ_q1_layout)
        sK_size = cute.cosize(sK_layout)
        sdS_size = cute.cosize(sdS_layout)

        _elem_bytes = self.q_dtype.width // 8

        @cute.struct
        class SharedStorage:
            Q_q0_mbar: cute.struct.MemRange[cutlass.Int64, 2]  # 1-stage TMA: producer+consumer
            Q_q1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            K_mbar: cute.struct.MemRange[cutlass.Int64, self.K_mbar_size]
            mbar: cute.struct.MemRange[cutlass.Int64, NUM_2Q_BARRIERS]
            tmem_holding_buf: Int32
            sQ_q0: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_q0_size], self.buffer_align_bytes]
            sQ_q1: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sQ_q1_size], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, sK_size], self.buffer_align_bytes]
            sdS: cute.struct.Align[cute.struct.MemRange[self.q_dtype, sdS_size], self.buffer_align_bytes]
            sGradSignal_q0_0: cute.struct.Align[cute.struct.MemRange[Float32, self.block_I], 128]
            sGradSignal_q0_1: cute.struct.Align[cute.struct.MemRange[Float32, self.block_I], 128]
            sGradSignal_q1_0: cute.struct.Align[cute.struct.MemRange[Float32, self.block_I], 128]
            sGradSignal_q1_1: cute.struct.Align[cute.struct.MemRange[Float32, self.block_I], 128]
            sW: cute.struct.Align[cute.struct.MemRange[self.q_dtype, self.heads * 2], 128]
            sdK_reduce: cute.struct.Align[cute.struct.MemRange[Float32, self.block_I * self.head_dim_padded], 128]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        Q_q0_mbar_ptr = storage.Q_q0_mbar.data_ptr()
        Q_q1_mbar_ptr = storage.Q_q1_mbar.data_ptr()
        K_mbar_ptr = storage.K_mbar.data_ptr()
        mbar = storage.mbar.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        # Swizzled SMEM tensors
        sQ_q0 = storage.sQ_q0.get_tensor(sQ_q0_layout.outer, swizzle=sQ_q0_layout.inner)
        sQ_q1 = storage.sQ_q1.get_tensor(sQ_q1_layout.outer, swizzle=sQ_q1_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
        sdS_store = storage.sdS.get_tensor(sdS_store_layout.outer, swizzle=sdS_store_layout.inner)
        # Recast views for transposed / SwapAB operands
        sdS_g2a = cute.make_tensor(cute.recast_ptr(sdS.iterator, sdS_g2a_layout.inner), sdS_g2a_layout.outer)
        sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sQ_q0_g2b = cute.make_tensor(cute.recast_ptr(sQ_q0.iterator, sQ_q0_g2b_layout.inner), sQ_q0_g2b_layout.outer)
        sQ_q1_g2b = cute.make_tensor(cute.recast_ptr(sQ_q1.iterator, sQ_q1_g2b_layout.inner), sQ_q1_g2b_layout.outer)

        # GradSignal: 2-stage for each Q token (matches 2-stage K pipeline)
        sGradSignal_q0_0 = storage.sGradSignal_q0_0.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        sGradSignal_q0_1 = storage.sGradSignal_q0_1.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        sGradSignal_q1_0 = storage.sGradSignal_q1_0.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        sGradSignal_q1_1 = storage.sGradSignal_q1_1.get_tensor(cute.make_layout((self.block_I,), stride=(1,)))
        # sW: holds W for both Q tokens (first heads = q0, second heads = q1)
        sW_full = storage.sW.get_tensor(cute.make_layout((self.heads * 2,), stride=(1,)))

        # sdK_reduce: row-major (block_I, head_dim) fp32 for bulk reduce
        sdK_reduce = storage.sdK_reduce.get_tensor(
            cute.make_layout((self.block_I, self.head_dim_padded), stride=(self.head_dim_padded, 1)),
        )

        # dQ epilogue SMEM — reuses sK physical memory
        sdQ_epi = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sdQ_epi_layout.inner),
            sdQ_epi_layout.outer,
        )

        # --- Q TMA load partitions (q0 and q1) ---
        Q_q0_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=Q_q0_mbar_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cute.make_layout(self.cluster_shape),
        )
        Q_q0_producer, Q_q0_consumer = Q_q0_pipeline.make_participants()

        Q_q1_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=Q_q1_mbar_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_Q_bytes,
            cta_layout_vmnk=cute.make_layout(self.cluster_shape),
        )
        Q_q1_producer, Q_q1_consumer = Q_q1_pipeline.make_participants()

        gemm1_thr_mma = tmma1.get_slice(0)

        # Q TMA partitions — mQ_q*_b is per-batch view (H, D, S_q_or_T_q);
        # batch is already absorbed via slice (BSHD) or domain_offset (THD).
        gQ_q0 = cute.local_tile(
            mQ_q0_b,
            cute.select(self.gemm1_tiler, mode=[0, 2]),
            (None, None, q0_local),
        )
        tAgQ_q0 = gemm1_thr_mma.partition_A(gQ_q0)
        tQsQ_q0, tQgQ_q0_mkl = cpasync.tma_partition(
            tma_atom_Q_q0,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_q0, 0, 3),
            cute.group_modes(tAgQ_q0, 0, 3),
        )

        gQ_q1 = cute.local_tile(
            mQ_q1_b,
            cute.select(self.gemm1_tiler, mode=[0, 2]),
            (None, None, q1_local),
        )
        tAgQ_q1 = gemm1_thr_mma.partition_A(gQ_q1)
        tQsQ_q1, tQgQ_q1_mkl = cpasync.tma_partition(
            tma_atom_Q_q1,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_q1, 0, 3),
            cute.group_modes(tAgQ_q1, 0, 3),
        )

        # --- K TMA load partition (2-stage) ---
        K_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=K_mbar_ptr,
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_K_bytes,
            cta_layout_vmnk=cute.make_layout(self.cluster_shape),
        )
        K_producer, K_consumer = K_pipeline.make_participants()

        # mK_b is the per-batch view: 2D (S_k, D) for both BSHD and THD.
        # (BSHD: batch-sliced from the 3D global mK; THD: domain_offset by
        # k_offset.) bi=0 maps to this batch's first K block.
        # (TMA only requires the box element address to be 128B-aligned;
        # head_dim*sizeof(elem) >= 128B in our configs, so any row-aligned
        # offset works.)
        gK = cute.local_tile(
            mK_b,
            cute.select(self.gemm1_tiler, mode=[1, 2]),
            (None, None),
        )
        tBgK = gemm1_thr_mma.partition_B(gK)
        tKsK, tKgK_mkl = cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tBgK, 0, 3),
        )

        # --- dQ TMA store partitions (q0 and q1) ---
        dQ_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARPGROUP_SIZE,
            ),
        )

        # mdQ_q*_b is per-batch view (H, D, S_q_or_T_q); pick the q_local slot.
        gdQ_q0 = cute.local_tile(
            mdQ_q0_b,
            (self.heads_padded, self.head_dim_padded),
            (0, 0, q0_local),
        )
        sdQ_epi_slice = sdQ_epi[None, None, 0]
        tdQsdQ_q0, tdQgdQ_q0_mkl = cpasync.tma_partition(
            tma_atom_dQ_q0,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_epi_slice, 0, 2),
            cute.group_modes(gdQ_q0, 0, 2),
        )

        gdQ_q1 = cute.local_tile(
            mdQ_q1_b,
            (self.heads_padded, self.head_dim_padded),
            (0, 0, q1_local),
        )
        tdQsdQ_q1, tdQgdQ_q1_mkl = cpasync.tma_partition(
            tma_atom_dQ_q1,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_epi_slice, 0, 2),
            cute.group_modes(gdQ_q1, 0, 2),
        )

        # Init all custom barriers (warp 0)
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar + MBAR_2Q_S_FULL, 1)
            cute.arch.mbarrier_init(mbar + MBAR_2Q_DS_READY, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_2Q_DK_FULL, 1)
            cute.arch.mbarrier_init(mbar + MBAR_2Q_DK_EMPTY, self.WARPGROUP_SIZE)
            cute.arch.mbarrier_init(mbar + MBAR_2Q_GS_LOADED_0, 1)
            cute.arch.mbarrier_init(mbar + MBAR_2Q_GS_LOADED_1, 1)
            cute.arch.mbarrier_init(mbar + MBAR_2Q_W_LOADED, 1)
        cute.arch.sync_threads()

        # Pre-compute accumulator shapes/layouts
        s_acc_shape = tmma1.partition_shape_C(self.gemm1_tiler[:2])
        s_acc_layout = tmma1.make_fragment_C(s_acc_shape).layout
        dq_acc_shape = tmma3.partition_shape_C(self.gemm3_tiler[:2])
        dq_acc_layout = tmma3.make_fragment_C(dq_acc_shape).layout
        dk_acc_shape = tmma2.partition_shape_C(self.gemm2_tiler[:2])
        dk_acc_layout = tmma2.make_fragment_C(dk_acc_shape).layout

        # =============================================================
        # Warp dispatch — entire dispatch is gated on has_q0 because
        # CTAs whose 2Q pair is past seqlen_q_b have no work. CuTe DSL
        # forbids early `return`, so we wrap instead.
        # =============================================================
        if has_q0:
            if warp_idx == self.load_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_wg0)
                self._load_warp_2q(
                    mW_b,
                    mGS_b,
                    sW_full,
                    sGradSignal_q0_0,
                    sGradSignal_q0_1,
                    sGradSignal_q1_0,
                    sGradSignal_q1_1,
                    tma_atom_Q_q0,
                    tQsQ_q0,
                    tQgQ_q0_mkl,
                    tma_atom_Q_q1,
                    tQsQ_q1,
                    tQgQ_q1_mkl,
                    tma_atom_K,
                    tKsK,
                    tKgK_mkl,
                    Q_q0_producer,
                    Q_q1_producer,
                    K_producer,
                    q0_local,
                    q1_local,
                    has_q1,
                    seqlen_k_b,
                    tidx,
                    mbar,
                    num_kv_blocks,
                )

            elif warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_wg0)
                tmem.wait_for_alloc()
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
                tStS, tDkDk, tDqDq_0, tDqDq_1 = self.get_tmem_tensor_2q(
                    s_acc_layout,
                    dq_acc_layout,
                    dk_acc_layout,
                    tmem_ptr_base,
                )
                self._mma_warp_2q(
                    sQ_q0,
                    sQ_q1,
                    sdS_g2a,
                    sK,
                    sKt,
                    sdS,
                    sQ_q0_g2b,
                    sQ_q1_g2b,
                    tmma1,
                    tmma2,
                    tmma3,
                    tStS,
                    tDkDk,
                    tDqDq_0,
                    tDqDq_1,
                    Q_q0_consumer,
                    Q_q1_consumer,
                    K_consumer,
                    has_q1,
                    mbar,
                    num_kv_blocks,
                )

            elif warp_idx in self.compute_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_compute)
                if warp_idx == self.compute_warp_id[0]:
                    tmem.allocate(self.tmem_alloc_cols)
                tmem.wait_for_alloc()
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
                tStS, tDkDk, tDqDq_0, tDqDq_1 = self.get_tmem_tensor_2q(
                    s_acc_layout,
                    dq_acc_layout,
                    dk_acc_layout,
                    tmem_ptr_base,
                )
                self._compute_warpgroup_2q(
                    mdW_b,
                    sGradSignal_q0_0,
                    sGradSignal_q0_1,
                    sGradSignal_q1_0,
                    sGradSignal_q1_1,
                    sW_full,
                    sdS,
                    sdQ_epi_slice,
                    s_acc_shape,
                    dq_acc_shape,
                    tStS,
                    tDqDq_0,
                    tDqDq_1,
                    tma_atom_dQ_q0,
                    tdQsdQ_q0,
                    tdQgdQ_q0_mkl,
                    tma_atom_dQ_q1,
                    tdQsdQ_q1,
                    tdQgdQ_q1_mkl,
                    dQ_store_pipeline,
                    sm_scale,
                    q0_local,
                    q1_local,
                    has_q1,
                    tidx,
                    warp_idx,
                    mbar,
                    num_kv_blocks,
                )
                if warp_idx == self.compute_warp_id[0]:
                    cute.arch.dealloc_tmem(tmem_ptr_base, self.tmem_alloc_cols)

            elif warp_idx in self.reduce_warp_id:
                cute.arch.setmaxregister_increase(self.num_regs_reduce)
                tmem.wait_for_alloc()
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
                tStS, tDkDk, tDqDq_0, tDqDq_1 = self.get_tmem_tensor_2q(
                    s_acc_layout,
                    dq_acc_layout,
                    dk_acc_layout,
                    tmem_ptr_base,
                )
                self._reduce_warpgroup_2q(
                    mdK_b,
                    dk_acc_shape,
                    tDkDk,
                    sdK_reduce,
                    sm_scale,
                    seqlen_k_b,
                    tidx,
                    mbar,
                    num_kv_blocks,
                )

            else:
                # Idle warps 2-3
                cute.arch.setmaxregister_decrease(self.num_regs_wg0)

    # =========================================================================
    # Load warp (2Q): TMA Q_q0/Q_q1 + TMA K 2-stage + GradSignal per Q token + W
    # =========================================================================
    @cute.jit
    def _load_grad_signal_to_buf(
        self,
        mGS_b,
        sGS,
        q_local,
        seqlen_k_b,
        lane_id,
        bi,
    ):
        """Load one block of GradSignal[q_local, kv_pos] into sGradSignal.

        mGS_b is the per-batch view: 2D (S_q, S_k) for BSHD or (T_q, max_K)
        with T-offset already applied for THD. q_local is the batch-local
        Q token index.
        """
        GS_PER_THREAD = const_expr((self.block_I + self.WARP_SIZE - 1) // self.WARP_SIZE)
        for si in cutlass.range_constexpr(GS_PER_THREAD):
            pos = si * self.WARP_SIZE + lane_id
            if pos < self.block_I:
                kv_pos = bi * self.block_I + pos
                if kv_pos < seqlen_k_b:
                    sGS[pos] = mGS_b[q_local, kv_pos]
                else:
                    sGS[pos] = Float32(0.0)

    @cute.jit
    def _load_warp_2q(
        self,
        mW_b,
        mGS_b,
        sW_full,
        sGradSignal_q0_0,
        sGradSignal_q0_1,
        sGradSignal_q1_0,
        sGradSignal_q1_1,
        tma_atom_Q_q0,
        tQsQ_q0,
        tQgQ_q0_mkl,
        tma_atom_Q_q1,
        tQsQ_q1,
        tQgQ_q1_mkl,
        tma_atom_K,
        tKsK,
        tKgK_mkl,
        Q_q0_producer,
        Q_q1_producer,
        K_producer,
        q0_local,
        q1_local,
        has_q1,
        seqlen_k_b,
        tidx,
        mbar,
        num_kv_blocks,
    ):
        """Load warp: TMA Q_q0+Q_q1 once, TMA K per-block (2-stage), W for both Q tokens, GradSignal per-block (2-stage per Q token)."""
        lane_id = tidx % self.WARP_SIZE

        # Load W for q0: sW_full[0..heads-1]
        W_PER_THREAD = const_expr((self.heads + self.WARP_SIZE - 1) // self.WARP_SIZE)
        for wi in cutlass.range_constexpr(W_PER_THREAD):
            idx = wi * self.WARP_SIZE + lane_id
            if idx < self.heads:
                sW_full[idx] = mW_b[q0_local, idx]

        # Load W for q1: sW_full[heads..2*heads-1] (guarded)
        if has_q1:
            for wi in cutlass.range_constexpr(W_PER_THREAD):
                idx = wi * self.WARP_SIZE + lane_id
                if idx < self.heads:
                    sW_full[self.heads + idx] = mW_b[q1_local, idx]

        cute.arch.fence_view_async_shared()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar + MBAR_2Q_W_LOADED)

        # --- TMA Q load (q0) ---
        Q_q0_producer.reset()
        handle_Q_q0 = Q_q0_producer.acquire_and_advance()
        cute.copy(
            tma_atom_Q_q0,
            tQgQ_q0_mkl[None, 0, 0],
            tQsQ_q0[None, 0],
            tma_bar_ptr=handle_Q_q0.barrier,
        )

        # --- TMA Q load (q1, guarded) ---
        Q_q1_producer.reset()
        if has_q1:
            handle_Q_q1 = Q_q1_producer.acquire_and_advance()
            cute.copy(
                tma_atom_Q_q1,
                tQgQ_q1_mkl[None, 0, 0],
                tQsQ_q1[None, 0],
                tma_bar_ptr=handle_Q_q1.barrier,
            )

        # --- TMA K pipeline (2-stage) + GradSignal per-block ---
        K_producer.reset()

        # Pre-start TMA K[0], K[1] (2-stage prefetch)
        for bi in cutlass.range(0, 2):
            if bi < num_kv_blocks:
                handle_K = K_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_K,
                    tKgK_mkl[None, bi, 0],
                    tKsK[None, handle_K.index],
                    tma_bar_ptr=handle_K.barrier,
                )

        # Main loop: TMA K (from bi=2) + GradSignal for all blocks
        for bi in cutlass.range(0, num_kv_blocks):
            # TMA K for bi >= 2 (0, 1 pre-loaded above)
            if bi >= 2:
                handle_K = K_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_K,
                    tKgK_mkl[None, bi, 0],
                    tKsK[None, handle_K.index],
                    tma_bar_ptr=handle_K.barrier,
                )

            # GradSignal load → sGradSignal[bi%2] (2-stage, matching K pipeline)
            if bi % 2 == 0:
                self._load_grad_signal_to_buf(
                    mGS_b,
                    sGradSignal_q0_0,
                    q0_local,
                    seqlen_k_b,
                    lane_id,
                    bi,
                )
                if has_q1:
                    self._load_grad_signal_to_buf(
                        mGS_b,
                        sGradSignal_q1_0,
                        q1_local,
                        seqlen_k_b,
                        lane_id,
                        bi,
                    )
                cute.arch.fence_view_async_shared()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar + MBAR_2Q_GS_LOADED_0)
            else:
                self._load_grad_signal_to_buf(
                    mGS_b,
                    sGradSignal_q0_1,
                    q0_local,
                    seqlen_k_b,
                    lane_id,
                    bi,
                )
                if has_q1:
                    self._load_grad_signal_to_buf(
                        mGS_b,
                        sGradSignal_q1_1,
                        q1_local,
                        seqlen_k_b,
                        lane_id,
                        bi,
                    )
                cute.arch.fence_view_async_shared()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar + MBAR_2Q_GS_LOADED_1)

    # =========================================================================
    # MMA warp (2Q): Sequential loop, 6 GEMMs per K block
    # =========================================================================
    @cute.jit
    def _mma_warp_2q(
        self,
        sQ_q0,
        sQ_q1,
        sdS_g2a,
        sK,
        sKt,
        sdS,
        sQ_q0_g2b,
        sQ_q1_g2b,
        tmma1,
        tmma2,
        tmma3,
        tStS,
        tDkDk,
        tDqDq_0,
        tDqDq_1,
        Q_q0_consumer,
        Q_q1_consumer,
        K_consumer,
        has_q1,
        mbar,
        num_kv_blocks,
    ):
        """MMA warp (2Q): No Fill+Drain overlap, fully sequential per K block.

        Per K block:
          Phase A (Q token 0): GEMM1→wait dS→GEMM2(clear)→GEMM3(dQ_q0 acc)
          Phase B (Q token 1): GEMM1→wait dS→GEMM2(acc)→GEMM3(dQ_q1 acc)
          signal DK_FULL, release K
        """
        Q_q0_consumer.reset()
        Q_q0_consumer.wait_and_advance()
        Q_q1_consumer.reset()
        if has_q1:
            Q_q1_consumer.wait_and_advance()

        K_consumer.reset()

        # A/B fragments from SMEM
        tSrQ_q0 = tmma1.make_fragment_A(sQ_q0)
        tSrQ_q1 = tmma1.make_fragment_A(sQ_q1)
        tSrK = tmma1.make_fragment_B(sK)  # 2-stage
        tDKrA_g2 = tmma2.make_fragment_A(sdS_g2a)  # 2-stage
        tDKrB_g2_q0 = tmma2.make_fragment_B(sQ_q0_g2b)
        tDKrB_g2_q1 = tmma2.make_fragment_B(sQ_q1_g2b)
        tDQrDS = tmma3.make_fragment_A(sdS)  # 2-stage
        tDQrKt = tmma3.make_fragment_B(sKt)  # 2-stage

        dk_empty_phase = Int32(0)
        ds_ready_phase = Int32(0)
        s_full_phase = Int32(0)
        is_first_dq_q0 = True
        is_first_dq_q1 = True

        for bi in cutlass.range(0, num_kv_blocks):
            # Wait DK_EMPTY (single-buffered dK; skip first block)
            if bi > 0:
                cute.arch.mbarrier_wait(mbar + MBAR_2Q_DK_EMPTY, dk_empty_phase)
                dk_empty_phase ^= 1

            # Wait for K[bi] ready
            K_handle = K_consumer.wait_and_advance()
            k_stage = K_handle.index

            # ---- Phase A (Q token 0) ----
            # GEMM1: S = Q_q0 @ K[bi]
            tmma1.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ_q0, mode=[2]), unroll=4):
                cute.gemm(tmma1, tStS, tSrQ_q0[None, None, k_block, 0], tSrK[None, None, k_block, k_stage], tStS)
                tmma1.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_2Q_S_FULL)

            # Wait for dS from Compute (Phase A)
            cute.arch.mbarrier_wait(mbar + MBAR_2Q_DS_READY, ds_ready_phase)
            ds_ready_phase ^= 1

            # GEMM2: dK = dS_q0^T @ Q_q0 (ACCUMULATE=False, clear)
            tmma2.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                cute.gemm(tmma2, tDkDk, tDKrA_g2[None, None, k_block, bi % 2], tDKrB_g2_q0[None, None, k_block, 0], tDkDk)
                tmma2.set(tcgen05.Field.ACCUMULATE, True)

            # GEMM3: dQ_q0 += dS @ Kt[bi]
            tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq_q0)
            is_first_dq_q0 = False
            for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll=4):
                cute.gemm(tmma3, tDqDq_0, tDQrDS[None, None, k_block, bi % 2], tDQrKt[None, None, k_block, k_stage], tDqDq_0)
                tmma3.set(tcgen05.Field.ACCUMULATE, True)

            # ---- Phase B (Q token 1, guarded) ----
            if has_q1:
                # GEMM1: S = Q_q1 @ K[bi]
                tmma1.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tSrQ_q1, mode=[2]), unroll=4):
                    cute.gemm(tmma1, tStS, tSrQ_q1[None, None, k_block, 0], tSrK[None, None, k_block, k_stage], tStS)
                    tmma1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + MBAR_2Q_S_FULL)

                # Wait for dS from Compute (Phase B)
                cute.arch.mbarrier_wait(mbar + MBAR_2Q_DS_READY, ds_ready_phase)
                ds_ready_phase ^= 1

                # GEMM2: dK += dS_q1^T @ Q_q1 (ACCUMULATE=True!)
                tmma2.set(tcgen05.Field.ACCUMULATE, True)
                for k_block in cutlass.range(0, cute.size(tDKrA_g2, mode=[2]), unroll=4):
                    cute.gemm(tmma2, tDkDk, tDKrA_g2[None, None, k_block, bi % 2], tDKrB_g2_q1[None, None, k_block, 0], tDkDk)

                # GEMM3: dQ_q1 += dS @ Kt[bi]
                tmma3.set(tcgen05.Field.ACCUMULATE, not is_first_dq_q1)
                is_first_dq_q1 = False
                for k_block in cutlass.range(0, cute.size(tDQrDS, mode=[2]), unroll=4):
                    cute.gemm(tmma3, tDqDq_1, tDQrDS[None, None, k_block, bi % 2], tDQrKt[None, None, k_block, k_stage], tDqDq_1)
                    tmma3.set(tcgen05.Field.ACCUMULATE, True)

            # Signal DK_FULL (after both phases)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + MBAR_2Q_DK_FULL)

            # Release K stage
            K_handle.release()

    # =========================================================================
    # Helper: dS/dW computation for one block (same as 1Q)
    # =========================================================================
    @cute.jit
    def _compute_ds_dw_block(self, tSrS, dw_accum, sGS, sW, tCcS, sm_scale: Float32 | float):
        """Compute dS and accumulate dW for one block using grad_signal from sGS."""
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
            gs0 = sGS[n0]
            gs1 = sGS[n1]

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

    # =========================================================================
    # Compute warpgroup (2Q): Per K block Phase A + Phase B, then epilogue
    # =========================================================================
    @cute.jit
    def _compute_warpgroup_2q(
        self,
        mdW_b,
        sGradSignal_q0_0,
        sGradSignal_q0_1,
        sGradSignal_q1_0,
        sGradSignal_q1_1,
        sW_full,
        sdS,
        sdQ_epi_slice,
        s_acc_shape,
        dq_acc_shape,
        tStS,
        tDqDq_0,
        tDqDq_1,
        tma_atom_dQ_q0,
        tdQsdQ_q0,
        tdQgdQ_q0_mkl,
        tma_atom_dQ_q1,
        tdQsdQ_q1,
        tdQgdQ_q1_mkl,
        dQ_store_pipeline,
        sm_scale: Float32 | float,
        q0_local,
        q1_local,
        has_q1,
        tidx,
        warp_idx,
        mbar,
        num_kv_blocks,
    ):
        """Compute warpgroup (2Q): Phase A (dS_q0 + dW_q0) + Phase B (dS_q1 + dW_q1) per K block, then epilogue for both Q tokens."""
        wg_tidx = tidx % self.WARPGROUP_SIZE
        warp_id_in_wg = wg_tidx // self.WARP_SIZE
        lane_id = wg_tidx % self.WARP_SIZE
        compute_warp0 = Int32(self.compute_warp_id[0])

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )

        # --- TMEM readback (S, single-buffered) ---
        tiled_tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_tmem_load_s = tiled_tmem_load_s.get_slice(wg_tidx)
        tStS_t2r = thr_tmem_load_s.partition_S(tStS)

        # Logical GEMM views for direct dS writes (stage 0/1).
        # Writing through the same A-operand view that GEMM2/GEMM3 read from
        # guarantees layout compatibility (stmatrix epilogue layout differs
        # from the swizzled A-operand layout, so coord writes are required).
        sdS_gemm_view_0 = cute.composition(
            sdS[None, None, None, 0],
            cute.make_layout((self.heads_padded, self.block_I)),
        )
        sdS_gemm_view_1 = cute.composition(
            sdS[None, None, None, 1],
            cute.make_layout((self.heads_padded, self.block_I)),
        )

        cS = cute.make_identity_tensor(s_acc_shape)
        tCcS = thr_tmem_load_s.partition_D(cS)
        tSrS_shape = tCcS.shape

        # --- TMEM readback (dQ_q0 and dQ_q1) ---
        tiled_tmem_load_dq0 = tcgen05.make_tmem_copy(tmem_load_atom, tDqDq_0)
        thr_tmem_load_dq0 = tiled_tmem_load_dq0.get_slice(wg_tidx)
        tDqDq_0_t2r = thr_tmem_load_dq0.partition_S(tDqDq_0)
        cDQ = cute.make_identity_tensor(dq_acc_shape)
        tCcDQ = thr_tmem_load_dq0.partition_D(cDQ)
        tDQrDQ_shape = tCcDQ.shape

        tiled_tmem_load_dq1 = tcgen05.make_tmem_copy(tmem_load_atom, tDqDq_1)
        thr_tmem_load_dq1 = tiled_tmem_load_dq1.get_slice(wg_tidx)
        tDqDq_1_t2r = thr_tmem_load_dq1.partition_S(tDqDq_1)

        # Wait for W loaded
        cute.arch.mbarrier_wait(mbar + MBAR_2Q_W_LOADED, Int32(0))

        # Create sW views for q0 and q1 from sW_full
        sW_q0 = sW_full  # sW_full[0..heads-1] via direct indexing in _compute_ds_dw_block
        # For q1, we use offset heads in the sW_full buffer

        s_full_phase = Int32(0)
        gs_0_phase = Int32(0)
        gs_1_phase = Int32(0)

        # dW accumulators for q0 and q1
        dw_accum_q0 = cute.make_rmem_tensor(tSrS_shape, Float32)
        for ei in cutlass.range(cute.size(dw_accum_q0), unroll_full=True):
            dw_accum_q0[ei] = Float32(0.0)

        dw_accum_q1 = cute.make_rmem_tensor(tSrS_shape, Float32)
        for ei in cutlass.range(cute.size(dw_accum_q1), unroll_full=True):
            dw_accum_q1[ei] = Float32(0.0)

        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        for bi in cutlass.range(0, num_kv_blocks):
            # Wait for GradSignal loaded by Load warp (2-stage, bi%2)
            if bi % 2 == 0:
                cute.arch.mbarrier_wait(mbar + MBAR_2Q_GS_LOADED_0, gs_0_phase)
                gs_0_phase ^= 1
            else:
                cute.arch.mbarrier_wait(mbar + MBAR_2Q_GS_LOADED_1, gs_1_phase)
                gs_1_phase ^= 1

            # ---- Phase A (Q token 0) ----
            # Wait for S ready from MMA (Phase A)
            cute.arch.mbarrier_wait(mbar + MBAR_2Q_S_FULL, s_full_phase)
            s_full_phase ^= 1
            cute.copy(tiled_tmem_load_s, tStS_t2r, tSrS)

            # Compute dS_q0 + accumulate dW_q0
            if bi % 2 == 0:
                self._compute_ds_dw_block(
                    tSrS,
                    dw_accum_q0,
                    sGradSignal_q0_0,
                    sW_q0,
                    tCcS,
                    sm_scale,
                )
            else:
                self._compute_ds_dw_block(
                    tSrS,
                    dw_accum_q0,
                    sGradSignal_q0_1,
                    sW_q0,
                    tCcS,
                    sm_scale,
                )

            cute.arch.fence_view_async_tmem_load()

            # Convert dS f32→bf16, write to sdS via coordinate mapping (2-stage, bi%2).
            tSrS_f16 = cute.make_rmem_tensor(tSrS.shape, self.q_dtype)
            for ei in cutlass.range(cute.size(tSrS), unroll_full=True):
                tSrS_f16[ei] = self.q_dtype(tSrS[ei])

            if bi % 2 == 0:
                for ei in cutlass.range(cute.size(tSrS_f16), unroll_full=True):
                    h = cute.get(tCcS[ei], mode=[0, 0])
                    n = cute.get(tCcS[ei], mode=[0, 1])
                    sdS_gemm_view_0[h, n] = tSrS_f16[ei]
            else:
                for ei in cutlass.range(cute.size(tSrS_f16), unroll_full=True):
                    h = cute.get(tCcS[ei], mode=[0, 0])
                    n = cute.get(tCcS[ei], mode=[0, 1])
                    sdS_gemm_view_1[h, n] = tSrS_f16[ei]

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(mbar + MBAR_2Q_DS_READY)

            # ---- Phase B (Q token 1, guarded) ----
            if has_q1:
                # Wait for S ready from MMA (Phase B)
                cute.arch.mbarrier_wait(mbar + MBAR_2Q_S_FULL, s_full_phase)
                s_full_phase ^= 1
                cute.copy(tiled_tmem_load_s, tStS_t2r, tSrS)

                # Compute dS_q1 + accumulate dW_q1, using sW_full offset by heads
                if bi % 2 == 0:
                    self._compute_ds_dw_block_q1(
                        tSrS,
                        dw_accum_q1,
                        sGradSignal_q1_0,
                        sW_full,
                        tCcS,
                        sm_scale,
                    )
                else:
                    self._compute_ds_dw_block_q1(
                        tSrS,
                        dw_accum_q1,
                        sGradSignal_q1_1,
                        sW_full,
                        tCcS,
                        sm_scale,
                    )

                cute.arch.fence_view_async_tmem_load()

                # Convert dS f32→bf16, write to sdS via coordinate mapping (2-stage, bi%2).
                tSrS_f16_b = cute.make_rmem_tensor(tSrS.shape, self.q_dtype)
                for ei in cutlass.range(cute.size(tSrS), unroll_full=True):
                    tSrS_f16_b[ei] = self.q_dtype(tSrS[ei])

                if bi % 2 == 0:
                    for ei in cutlass.range(cute.size(tSrS_f16_b), unroll_full=True):
                        h = cute.get(tCcS[ei], mode=[0, 0])
                        n = cute.get(tCcS[ei], mode=[0, 1])
                        sdS_gemm_view_0[h, n] = tSrS_f16_b[ei]
                else:
                    for ei in cutlass.range(cute.size(tSrS_f16_b), unroll_full=True):
                        h = cute.get(tCcS[ei], mode=[0, 0])
                        n = cute.get(tCcS[ei], mode=[0, 1])
                        sdS_gemm_view_1[h, n] = tSrS_f16_b[ei]

                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(mbar + MBAR_2Q_DS_READY)

        # dQ staging via coordinate writes — same pattern as sdS.
        sdQ_gemm_view = cute.composition(
            sdQ_epi_slice,
            cute.make_layout((self.heads_padded, self.head_dim_padded)),
        )

        # ---- Epilogue: dQ for q0 via TMA store ----
        tDQrDQ_q0 = cute.make_rmem_tensor(tDQrDQ_shape, Float32)
        cute.copy(tiled_tmem_load_dq0, tDqDq_0_t2r, tDQrDQ_q0)

        tDQrDQ_q0_bf16 = cute.make_rmem_tensor(tDQrDQ_q0.shape, self.q_dtype)
        for ei in cutlass.range(cute.size(tDQrDQ_q0), unroll_full=True):
            tDQrDQ_q0_bf16[ei] = self.q_dtype(tDQrDQ_q0[ei] * Float32(sm_scale))

        cute.arch.fence_view_async_tmem_load()

        for ei in cutlass.range(cute.size(tDQrDQ_q0_bf16), unroll_full=True):
            h = cute.get(tCcDQ[ei], mode=[0, 0])
            d = cute.get(tCcDQ[ei], mode=[0, 1])
            sdQ_gemm_view[h, d] = tDQrDQ_q0_bf16[ei]

        self.compute_sync_barrier.arrive_and_wait()
        cute.arch.fence_proxy("async.shared", space="cta")
        self.compute_sync_barrier.arrive_and_wait()

        if warp_idx == compute_warp0:
            dQ_store_pipeline.producer_acquire()
            cute.copy(tma_atom_dQ_q0, tdQsdQ_q0, tdQgdQ_q0_mkl)
            dQ_store_pipeline.producer_commit()

        # ---- Epilogue: dQ for q1 via TMA store (guarded) ----
        if has_q1:
            # Wait for q0 TMA store to complete before reusing sdQ_epi SMEM
            dQ_store_pipeline.producer_acquire()

            tDQrDQ_q1 = cute.make_rmem_tensor(tDQrDQ_shape, Float32)
            cute.copy(tiled_tmem_load_dq1, tDqDq_1_t2r, tDQrDQ_q1)

            tDQrDQ_q1_bf16 = cute.make_rmem_tensor(tDQrDQ_q1.shape, self.q_dtype)
            for ei in cutlass.range(cute.size(tDQrDQ_q1), unroll_full=True):
                tDQrDQ_q1_bf16[ei] = self.q_dtype(tDQrDQ_q1[ei] * Float32(sm_scale))

            cute.arch.fence_view_async_tmem_load()

            for ei in cutlass.range(cute.size(tDQrDQ_q1_bf16), unroll_full=True):
                h = cute.get(tCcDQ[ei], mode=[0, 0])
                d = cute.get(tCcDQ[ei], mode=[0, 1])
                sdQ_gemm_view[h, d] = tDQrDQ_q1_bf16[ei]

            self.compute_sync_barrier.arrive_and_wait()
            cute.arch.fence_proxy("async.shared", space="cta")
            self.compute_sync_barrier.arrive_and_wait()

            if warp_idx == compute_warp0:
                cute.copy(tma_atom_dQ_q1, tdQsdQ_q1, tdQgdQ_q1_mkl)
                dQ_store_pipeline.producer_commit()

        # ---- Epilogue: dW for q0 via warp reduction ----
        # mdW_b is the per-batch view: (S_q, H) BSHD or (T_q, H) THD with T-offset
        # already applied. q*_local indexes within the batch.
        HEADS_PER_WARP = const_expr(self.heads_padded // 4)
        warp_base_h = warp_id_in_wg * Int32(HEADS_PER_WARP)
        for h_local in cutlass.range_constexpr(HEADS_PER_WARP):
            h = warp_base_h + h_local
            my_partial = Float32(0.0)
            for ei in cutlass.range(cute.size(dw_accum_q0), unroll_full=True):
                if cute.get(tCcS[ei], mode=[0, 0]) == h:
                    my_partial = my_partial + dw_accum_q0[ei]
            total = cute.arch.warp_reduction_sum(my_partial)
            if lane_id == 0:
                mdW_b[q0_local, h] = self.q_dtype(total)

        # ---- Epilogue: dW for q1 via warp reduction (guarded) ----
        if has_q1:
            for h_local in cutlass.range_constexpr(HEADS_PER_WARP):
                h = warp_base_h + h_local
                my_partial = Float32(0.0)
                for ei in cutlass.range(cute.size(dw_accum_q1), unroll_full=True):
                    if cute.get(tCcS[ei], mode=[0, 0]) == h:
                        my_partial = my_partial + dw_accum_q1[ei]
                total = cute.arch.warp_reduction_sum(my_partial)
                if lane_id == 0:
                    mdW_b[q1_local, h] = self.q_dtype(total)

    # =========================================================================
    # Helper: dS/dW for Q token 1 (reads W from sW_full[heads+h])
    # =========================================================================
    @cute.jit
    def _compute_ds_dw_block_q1(self, tSrS, dw_accum, sGS, sW_full, tCcS, sm_scale: Float32 | float):
        """Compute dS and accumulate dW for Q token 1, using W offset by heads in sW_full."""
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

            # Q1 W offset: heads + h
            w0 = Float32(sW_full[self.heads + h0])
            w1 = Float32(sW_full[self.heads + h1])
            gs0 = sGS[n0]
            gs1 = sGS[n1]

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

    # =========================================================================
    # Reduce warpgroup (2Q): Single-buffered dK
    # =========================================================================
    @cute.jit
    def _reduce_warpgroup_2q(
        self,
        mdK_b,
        dk_acc_shape,
        tDkDk,
        sdK_reduce,
        sm_scale: Float32 | float,
        seqlen_k_b,
        tidx,
        mbar,
        num_kv_blocks,
    ):
        """Reduce warpgroup (2Q): single-buffered dK (wait DK_FULL, T2R, scatter, bulk DMA, signal DK_EMPTY).

        mdK_b is the per-batch view: 2D (S_k, D) BSHD or (T_k, D) THD with
        k-offset already applied. Each block writes to row bi*block_I."""
        wg_tidx = tidx % self.WARPGROUP_SIZE

        tmem_load_atom_dk = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )

        tiled_tmem_load_dk = tcgen05.make_tmem_copy(tmem_load_atom_dk, tDkDk)
        thr_tmem_load_dk = tiled_tmem_load_dk.get_slice(wg_tidx)
        tDkDk_t2r = thr_tmem_load_dk.partition_S(tDkDk)
        cDK = cute.make_identity_tensor(dk_acc_shape)
        tCcDK = thr_tmem_load_dk.partition_D(cDK)
        tDKrDK_shape = tCcDK.shape

        dk_full_phase = Int32(0)
        for bi in cutlass.range(0, num_kv_blocks):
            # 1. Wait DK_FULL
            cute.arch.mbarrier_wait(mbar + MBAR_2Q_DK_FULL, dk_full_phase)
            dk_full_phase ^= 1

            # 2. T2R readback dK
            tDKrDK = cute.make_rmem_tensor(tDKrDK_shape, Float32)
            cute.copy(tiled_tmem_load_dk, tDkDk_t2r, tDKrDK)
            cute.arch.fence_view_async_tmem_load()

            # 3. Signal DK_EMPTY immediately after T2R (single-buffered)
            cute.arch.mbarrier_arrive(mbar + MBAR_2Q_DK_EMPTY)

            # 4. Wait for previous bulk reduce to finish, then signal TMA engine is free
            if bi > 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)

            # 5. Scatter-write: registers → sdK_reduce
            for pair in cutlass.range(cute.size(tDKrDK) // 2, unroll_full=True):
                ei = pair * 2
                n = cute.get(tCcDK[ei], mode=[0, 0])
                d = cute.get(tCcDK[ei], mode=[0, 1])
                kv_pos = bi * self.block_I + n
                if kv_pos < seqlen_k_b:
                    sdK_reduce[n, d] = tDKrDK[ei] * Float32(sm_scale)
                    sdK_reduce[n, d + 1] = tDKrDK[ei + 1] * Float32(sm_scale)

            cute.arch.fence_proxy("async.shared", space="cta")

            # 6. Single-thread bulk reduce DMA — only ONE thread in the
            # reduce warpgroup must issue cp.async.bulk; otherwise each
            # warp's elect_one fires a separate bulk DMA, causing 4×
            # over-accumulation in the global dK output.
            actual_rows = seqlen_k_b - bi * self.block_I
            if actual_rows > self.block_I:
                actual_rows = self.block_I
            store_bytes = actual_rows * self.head_dim_padded * 4

            if wg_tidx == 0:
                gdK_block = mdK_b[bi * self.block_I, None]
                cpasync_reduce_bulk_add_f32(
                    sdK_reduce.iterator,
                    gdK_block.iterator,
                    store_bytes,
                )
                cute.arch.cp_async_bulk_commit_group()

        # Wait for final bulk reduce
        if wg_tidx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def get_tmem_tensor_2q(self, s_acc_layout, dq_acc_layout, dk_acc_layout, tmem_ptr_base: cute.Pointer):
        tStS = cute.make_tensor(tmem_ptr_base + self.tmem_s_offset, s_acc_layout)
        tDkDk = cute.make_tensor(tmem_ptr_base + self.tmem_dk_offset, dk_acc_layout)
        tDqDq_0 = cute.make_tensor(tmem_ptr_base + self.tmem_dq0_offset, dq_acc_layout)
        tDqDq_1 = cute.make_tensor(tmem_ptr_base + self.tmem_dq1_offset, dq_acc_layout)
        return tStS, tDkDk, tDqDq_0, tDqDq_1


# =============================================================================
# Factory
# =============================================================================
_compile_cache: dict = {}


def dense_indexer_backward_sm100(
    batch,
    max_seqlen_q,
    max_seqlen_k,
    heads,
    dim,
    sm_scale=1.0,
    block_I=128,
    ratio=1,
    is_varlen=False,
):
    """Build / fetch a compiled SM100 dense backward gradient kernel.

    Two explicit modes (no implicit BSHD-to-THD packing):
      ``is_varlen=False`` (BSHD): max_seqlen_q == S_q, max_seqlen_k == S_k.
        Inputs are 4D / 3D / 2D tensors with leading batch dim.
      ``is_varlen=True``  (THD) : max_seqlen_q / k cap the kernel's per-CTA
        grid + row stride. Inputs are packed (T_q, ...) tensors plus
        cu_seqlens_q / k.

    ``ratio`` is the indexer compression ratio. Bottom-right causal mask is
    applied: kv_local < (seqlen_k_b * ratio - seqlen_q_b + q_local + 1) // ratio.
    Per batch we require ``seqlen_q_b <= seqlen_k_b * ratio``. ``ratio`` must
    be passed explicitly — auto-inferring from S_q / S_k is unsafe under THD.

    ``grad_scale`` is intentionally **not** an argument to this factory: it's
    a host scalar consumed only as a multiplicative factor inside
    ``ScoreGradDense``, threaded through as a runtime ``Float32`` arg at
    ``_run`` call time so changing it does not trigger recompilation.
    """
    assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
    key = (
        is_varlen,
        batch,
        max_seqlen_q,
        max_seqlen_k,
        heads,
        dim,
        sm_scale,
        block_I,
        ratio,
    )
    if key not in _compile_cache:
        _compile_cache[key] = _build_cute_dsl_kernel(
            batch,
            max_seqlen_q,
            max_seqlen_k,
            heads,
            dim,
            sm_scale,
            block_I,
            ratio,
            is_varlen,
        )
    return _compile_cache[key]


def _build_cute_dsl_kernel(batch, max_seqlen_q, max_seqlen_k, heads, dim, sm_scale, block_I, ratio, is_varlen):
    from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("Requires SM100+")

    # Kernel 1: ScoreGradDense — applies bottom-right ratio causal mask so
    # masked / padding columns produce grad_signal=0 (won't contaminate GEMM).
    score_grad_obj = ScoreGradDense(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        ratio=ratio,
    )

    # Kernel 2: 2Q GEMM kernel (reads pre-computed grad_signal, 2 Q tokens per CTA).
    # is_varlen branches at compile time on whether CuSeqlens* are None.
    gemm_obj = DenseIndexerBackward2QGemmSm100(head_dim=dim, heads=heads, block_I=block_I)

    compiled_score_grad = [None]
    compiled_gemm = [None]

    def _ensure_compiled_score_grad(IdxScoreRaw, IdxLSE, AttnScoreRaw, AttnL1Norm, CuSeqlensQ, CuSeqlensK, grad_scale, current_stream=None):
        """Lazy-compile kernel 1 (score gradient).

        For BSHD (is_varlen=False) CuSeqlensQ/K are None and the compiled
        kernel uses static seqlen from tensor shape. For THD they are int32
        cu_seqlens tensors of shape (B+1,).

        ``cutlass.Float32(...)`` is passed at compile to fill the
        ``grad_scale`` positional slot in ``ScoreGradDense.__call__`` —
        without it the next positional (``stream``) shifts up and CuTe
        rejects the call. Because ``grad_scale`` is not part of this
        factory's cache key, the compiled kernel is reused unchanged when
        the runtime ``grad_scale`` differs from the value seen at compile.
        """
        if compiled_score_grad[0] is None:
            s = _resolve_stream(current_stream)
            cuq_arg = to_cute_tensor(CuSeqlensQ) if CuSeqlensQ is not None else None
            cuk_arg = to_cute_tensor(CuSeqlensK) if CuSeqlensK is not None else None
            compiled_score_grad[0] = cute.compile(
                score_grad_obj,
                to_cute_tensor(IdxScoreRaw),
                to_cute_tensor(AttnScoreRaw),
                to_cute_tensor(IdxLSE),
                to_cute_tensor(AttnL1Norm),
                cuq_arg,
                cuk_arg,
                cutlass.Float32(float(grad_scale)),
                s,
                options=compile_options("--opt-level 3"),
            )

    def _ensure_compiled_gemm(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, CuSeqlensQ, CuSeqlensK, current_stream=None):
        """Lazy-compile kernel 2 (2Q GEMM).

        For BSHD (is_varlen=False) CuSeqlens* are None. For THD they are
        int32 cu_seqlens tensors of shape (B+1,).
        """
        if compiled_gemm[0] is None:
            s = _resolve_stream(current_stream)
            cuq_arg = to_cute_tensor(CuSeqlensQ) if CuSeqlensQ is not None else None
            cuk_arg = to_cute_tensor(CuSeqlensK) if CuSeqlensK is not None else None
            compiled_gemm[0] = cute.compile(
                gemm_obj,
                to_cute_tensor(IndexQ),
                to_cute_tensor(Weights),
                to_cute_tensor(IndexK),
                to_cute_tensor(dIndexQ),
                to_cute_tensor(dWeights),
                to_cute_tensor(dIndexK_f32),
                to_cute_tensor(GradSignal),
                cuq_arg,
                cuk_arg,
                cutlass.Float32(sm_scale),
                cutlass.Int32(max_seqlen_q),
                cutlass.Int32(max_seqlen_k),
                s,
                options=compile_options("--opt-level 3"),
            )

    def _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, CuSeqlensQ=None, CuSeqlensK=None, current_stream=None):
        """Run only kernel 2 (2Q GEMM). Caller must have run kernel 1 and zeroed dIndexK_f32."""
        # Match the compile-time is_varlen to avoid dispatching into a kernel
        # compiled for the other mode.
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled kernel must not receive cu_seqlens_q/k"
        s = _resolve_stream(current_stream)
        _ensure_compiled_gemm(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, CuSeqlensQ, CuSeqlensK, current_stream=current_stream)
        with torch.cuda.nvtx.range("indexer_backward_dsl_dense_gemm_2q"):
            compiled_gemm[0](
                IndexQ,
                Weights,
                IndexK,
                dIndexQ,
                dWeights,
                dIndexK_f32,
                GradSignal,
                CuSeqlensQ,
                CuSeqlensK,
                cutlass.Float32(sm_scale),
                cutlass.Int32(max_seqlen_q),
                cutlass.Int32(max_seqlen_k),
                s,
            )

    def _run(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK_f32,
        AttnScoreRaw,
        AttnL1Norm,
        IdxScoreRaw,
        IdxLSE,
        grad_scale,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        current_stream=None,
    ):
        """Full dense backward: kernel 1 (score grad) + kernel 2 (2Q GEMM).

        BSHD: pass tensors with batch dim and CuSeqlens*=None.
        THD : pass packed tensors and CuSeqlens* (B+1,) int32.
        Kernel 3 (dK f32 → output dtype) is handled by the caller.

        ``grad_scale`` is a host scalar (Python float, ``loss_coeff /
        (b*sq)``); supplied at call time as a runtime ``Float32`` arg to
        kernel 1 so changing it does not require recompilation.
        """
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled kernel must not receive cu_seqlens_q/k"
        s = _resolve_stream(current_stream)

        # Kernel 1 (CuTe DSL): in-place overwrites IdxScoreRaw with grad_signal.
        # Grid = (B, max_seqlen_q, 1); CTAs past per-batch seqlen exit early.
        _ensure_compiled_score_grad(IdxScoreRaw, IdxLSE, AttnScoreRaw, AttnL1Norm, CuSeqlensQ, CuSeqlensK, grad_scale, current_stream=current_stream)
        with torch.cuda.nvtx.range("indexer_backward_dsl_dense_score_grad"):
            compiled_score_grad[0](
                IdxScoreRaw,
                AttnScoreRaw,
                IdxLSE,
                AttnL1Norm,
                CuSeqlensQ,
                CuSeqlensK,
                cutlass.Float32(float(grad_scale)),
                s,
            )

        # Kernel 2 (CuTe DSL): 2Q three GEMMs — IdxScoreRaw now contains grad_signal.
        # Grid = (B, ceil(max_seqlen_q/2), 1).
        _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, IdxScoreRaw, CuSeqlensQ, CuSeqlensK, current_stream=current_stream)

    _run.gemm_only = _run_gemm_only
    _run.is_varlen = is_varlen
    _run.ratio = ratio
    return _run
