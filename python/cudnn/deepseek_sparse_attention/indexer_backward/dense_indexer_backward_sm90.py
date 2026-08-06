# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense Indexer Backward — SM90 CuTe-DSL factory."""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

from .indexer_backward_sm90 import (
    CLIP_LOG_MIN,
    CLIP_PROB_MIN,
    EPS,
    IndexerBackwardSm90,
)

_dense_compile_cache: dict = {}
_dense_score_grad_compile_cache: dict = {}


@cute.jit
def _dense_seqlen_info(
    mCuSeqlensQ,
    mCuSeqlensK,
    batch_idx: Int32,
    seqlen_q_static: Int32,
    seqlen_k_static: Int32,
):
    """Return batch-local offsets and lengths for BSHD or THD dense mode."""
    if const_expr(mCuSeqlensQ is None):
        return Int32(0), Int32(0), seqlen_q_static, seqlen_k_static
    else:
        q_offset = mCuSeqlensQ[batch_idx]
        seqlen_q_b = mCuSeqlensQ[batch_idx + Int32(1)] - q_offset
        k_offset = mCuSeqlensK[batch_idx]
        seqlen_k_b = mCuSeqlensK[batch_idx + Int32(1)] - k_offset
        return q_offset, k_offset, seqlen_q_b, seqlen_k_b


class ScoreGradDenseSm90:
    """Dense score-grad precompute: raw scores + denoms -> grad_signal."""

    WARP_SIZE = 32
    THREADS_PER_CTA = 128

    def __init__(self, ratio: int = 1, block_I: int = 128):
        assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
        self.ratio = ratio
        self.block_I = block_I

    @cute.jit
    def __call__(
        self,
        mIdxScoreRaw,
        mAttnScoreRaw,
        mIdxLSE,
        mAttnL1Norm,
        mCuSeqlensQ,
        mCuSeqlensK,
        mQCausalOffsets,
        mGradLoss,
        grad_scale: Float32,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        stream,
    ):
        # BSHD:
        #   IdxScoreRaw/AttnScoreRaw: (B, S_q, S_k), IdxLSE/AttnL1Norm: (B, S_q)
        # THD:
        #   IdxScoreRaw/AttnScoreRaw: (T_q, max_K), IdxLSE/AttnL1Norm: (T_q,)
        is_varlen = const_expr(mCuSeqlensQ is not None)

        if const_expr(is_varlen):
            batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1
        else:
            mIdxScoreRaw = cute.make_tensor(
                mIdxScoreRaw.iterator,
                cute.select(mIdxScoreRaw.layout, mode=[1, 2, 0]),
            )
            mAttnScoreRaw = cute.make_tensor(
                mAttnScoreRaw.iterator,
                cute.select(mAttnScoreRaw.layout, mode=[1, 2, 0]),
            )
            mIdxLSE = cute.make_tensor(
                mIdxLSE.iterator,
                cute.select(mIdxLSE.layout, mode=[1, 0]),
            )
            mAttnL1Norm = cute.make_tensor(
                mAttnL1Norm.iterator,
                cute.select(mAttnL1Norm.layout, mode=[1, 0]),
            )
            batch_size = cute.size(mIdxScoreRaw.shape[2])

        seqlen_k_pad = cute.size(mIdxScoreRaw.shape[1])

        self.kernel_score_grad(
            mIdxScoreRaw,
            mAttnScoreRaw,
            mIdxLSE,
            mAttnL1Norm,
            mCuSeqlensQ,
            mCuSeqlensK,
            mQCausalOffsets,
            mGradLoss,
            grad_scale,
            seqlen_k_pad,
            max_seqlen_q,
            max_seqlen_k,
        ).launch(
            grid=(max_seqlen_q, batch_size, 1),
            block=[self.THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_score_grad(
        self,
        mIdxScoreRaw,
        mAttnScoreRaw,
        mIdxLSE,
        mAttnL1Norm,
        mCuSeqlensQ,
        mCuSeqlensK,
        mQCausalOffsets,
        mGradLoss,
        grad_scale: Float32,
        seqlen_k_pad: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        seq_local = cute.arch.block_idx()[0]
        batch_idx = cute.arch.block_idx()[1]
        warp_id = tidx // self.WARP_SIZE

        is_varlen = const_expr(mCuSeqlensQ is not None)
        q_offset, _k_offset, seqlen_q_b, seqlen_k_b = _dense_seqlen_info(
            mCuSeqlensQ,
            mCuSeqlensK,
            Int32(batch_idx),
            seqlen_q_static,
            seqlen_k_static,
        )
        q_causal_offset_b = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
        grad_scale_f32 = Float32(grad_scale) * Float32(mGradLoss[0])

        if seq_local < seqlen_q_b:
            smem = cutlass.utils.SmemAllocator()

            @cute.struct
            class SharedStorage:
                sReduceScratch: cute.struct.Align[
                    cute.struct.MemRange[Float32, 4],
                    128,
                ]

            storage = smem.allocate(SharedStorage)
            sReduceScratch = storage.sReduceScratch.get_tensor(cute.make_layout((4,), stride=(1,)))

            if const_expr(is_varlen):
                mIdxScore_b = cute.domain_offset((q_offset, Int32(0)), mIdxScoreRaw)
                mAttnScore_b = cute.domain_offset((q_offset, Int32(0)), mAttnScoreRaw)
                mIdxLSE_b = cute.domain_offset((q_offset,), mIdxLSE)
                mAttnL1_b = cute.domain_offset((q_offset,), mAttnL1Norm)
            else:
                mIdxScore_b = mIdxScoreRaw[None, None, batch_idx]
                mAttnScore_b = mAttnScoreRaw[None, None, batch_idx]
                mIdxLSE_b = mIdxLSE[None, batch_idx]
                mAttnL1_b = mAttnL1Norm[None, batch_idx]

            idx_lse_val = mIdxLSE_b[seq_local]
            attn_l1_val = mAttnL1_b[seq_local]
            LOG2E = Float32(1.4426950408889634)

            ratio = Int32(self.ratio)
            col_limit_raw = (q_causal_offset_b + Int32(seq_local) + Int32(1)) // ratio
            col_limit = col_limit_raw if col_limit_raw < seqlen_k_b else seqlen_k_b
            col_limit = col_limit if col_limit > Int32(0) else Int32(0)

            local_sum = Float32(0.0)
            pos = tidx
            while pos < col_limit:
                idx_raw = mIdxScore_b[seq_local, pos]
                attn_raw = mAttnScore_b[seq_local, pos]
                score_minus_lse = idx_raw - idx_lse_val
                predict = cute.math.exp2(
                    score_minus_lse * LOG2E,
                    fastmath=True,
                )
                target = attn_raw / (attn_l1_val + Float32(EPS))
                target_eff = target if target >= Float32(CLIP_PROB_MIN) else Float32(CLIP_PROB_MIN)
                log_clip_mask = Float32(1.0) if score_minus_lse >= Float32(CLIP_LOG_MIN) else Float32(0.0)
                local_sum = local_sum + (-target_eff * log_clip_mask * grad_scale_f32)
                pos = pos + Int32(128)

            warp_sum = cute.arch.warp_reduction_sum(local_sum)
            with cute.arch.elect_one():
                sReduceScratch[warp_id] = warp_sum
            cute.arch.sync_threads()
            sum_grad = sReduceScratch[0] + sReduceScratch[1] + sReduceScratch[2] + sReduceScratch[3]

            block_i = Int32(self.block_I)
            zero_limit = ((col_limit + block_i - Int32(1)) // block_i) * block_i
            zero_limit = zero_limit if zero_limit < seqlen_k_pad else seqlen_k_pad

            pos = tidx
            while pos < zero_limit:
                if pos < col_limit:
                    idx_raw = mIdxScore_b[seq_local, pos]
                    attn_raw = mAttnScore_b[seq_local, pos]
                    score_minus_lse = idx_raw - idx_lse_val
                    predict = cute.math.exp2(
                        score_minus_lse * LOG2E,
                        fastmath=True,
                    )
                    target = attn_raw / (attn_l1_val + Float32(EPS))
                    target_eff = target if target >= Float32(CLIP_PROB_MIN) else Float32(CLIP_PROB_MIN)
                    log_clip_mask = Float32(1.0) if score_minus_lse >= Float32(CLIP_LOG_MIN) else Float32(0.0)
                    g = -target_eff * log_clip_mask * grad_scale_f32
                    mIdxScore_b[seq_local, pos] = g - predict * sum_grad
                else:
                    mIdxScore_b[seq_local, pos] = Float32(0.0)
                pos = pos + Int32(128)


def dense_indexer_backward_sm90(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    sm_scale=1.0,
    block_I=128,
    ratio=1,
    is_varlen=False,
    has_q_causal_offsets=False,
):
    """Factory for the dense indexer backward gradient kernel on SM90."""
    assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
    # batch/seqlen/seqlen_k are runtime (dynamic grid dims + Int32 args), so
    # they're not keyed. sm_scale is a runtime Float32 arg (not keyed).
    return _build_cute_dsl_dense_kernel(
        batch,
        seqlen,
        seqlen_k,
        heads,
        dim,
        sm_scale,
        block_I,
        ratio,
        is_varlen,
        has_q_causal_offsets,
    )


def _build_cute_dsl_dense_kernel(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    sm_scale,
    block_I,
    ratio,
    is_varlen,
    has_q_causal_offsets,
):
    cap = torch.cuda.get_device_capability()[0]
    if cap < 9:
        raise RuntimeError(f"Requires SM90+ (got SM{cap}0)")
    if cap >= 10:
        raise RuntimeError("Use SM100 kernel for Blackwell")

    topk = seqlen_k
    score_grad_obj = ScoreGradDenseSm90(ratio=ratio, block_I=block_I)
    kernel_obj = IndexerBackwardSm90(
        head_dim=dim,
        heads=heads,
        block_I=block_I,
        topk=topk,
        is_dense=True,
        ratio=ratio,
    )

    # Compile caches are keyed only by params that change generated code. In
    # dense mode, K-block traversal is runtime-sized by seqlen_k.
    score_grad_key = (is_varlen, ratio, block_I, bool(has_q_causal_offsets))
    gemm_key = (is_varlen, heads, dim, block_I, ratio, bool(has_q_causal_offsets))
    dummy_topk_holder = [None]

    def _get_dummy_topk(device, current_stream=None):
        if dummy_topk_holder[0] is None or dummy_topk_holder[0].device != device:
            with _torch_stream_context(current_stream):
                dummy_topk_holder[0] = torch.zeros(batch, seqlen, seqlen_k, device=device, dtype=torch.int32)
        return dummy_topk_holder[0]

    def _ensure_compiled(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK_f32,
        GradSignal,
        CuSeqlensQ,
        CuSeqlensK,
        QCausalOffsets,
        current_stream=None,
    ):
        s = _resolve_stream(current_stream)
        if gemm_key not in _dense_compile_cache:
            dummy_topk = _get_dummy_topk(IndexQ.device, current_stream=current_stream)
            cuq_arg = to_cute_tensor(CuSeqlensQ) if CuSeqlensQ is not None else None
            cuk_arg = to_cute_tensor(CuSeqlensK) if CuSeqlensK is not None else None
            q_offsets_arg = to_cute_tensor(QCausalOffsets) if QCausalOffsets is not None else None
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
                    dummy_topk,
                ]
            ]
            _dense_compile_cache[gemm_key] = cute.compile(
                kernel_obj,
                *cute_args,
                cutlass.Float32(sm_scale),
                s,
                cuq_arg,
                cuk_arg,
                cutlass.Int32(seqlen),
                cutlass.Int32(seqlen_k),
                q_offsets_arg,
                options=compile_options(),
            )

    def _ensure_compiled_score_grad(
        IdxScoreRaw,
        IdxLSE,
        AttnScoreRaw,
        AttnL1Norm,
        CuSeqlensQ,
        CuSeqlensK,
        QCausalOffsets,
        GradLoss,
        grad_scale,
        current_stream=None,
    ):
        if score_grad_key not in _dense_score_grad_compile_cache:
            s = _resolve_stream(current_stream)
            cuq_arg = to_cute_tensor(CuSeqlensQ) if CuSeqlensQ is not None else None
            cuk_arg = to_cute_tensor(CuSeqlensK) if CuSeqlensK is not None else None
            q_offsets_arg = to_cute_tensor(QCausalOffsets) if QCausalOffsets is not None else None
            _dense_score_grad_compile_cache[score_grad_key] = cute.compile(
                score_grad_obj,
                to_cute_tensor(IdxScoreRaw),
                to_cute_tensor(AttnScoreRaw),
                to_cute_tensor(IdxLSE),
                to_cute_tensor(AttnL1Norm),
                cuq_arg,
                cuk_arg,
                q_offsets_arg,
                to_cute_tensor(GradLoss, assumed_align=4, leading_dim=0),
                cutlass.Float32(float(grad_scale)),
                cutlass.Int32(seqlen),
                cutlass.Int32(seqlen_k),
                s,
                options=compile_options("--opt-level 3"),
            )

    def _run_score_grad_only(
        IdxScoreRaw,
        IdxLSE,
        AttnScoreRaw,
        AttnL1Norm,
        GradLoss,
        grad_scale,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        QCausalOffsets=None,
        current_stream=None,
    ):
        if GradLoss.ndim == 0:
            GradLoss = GradLoss.reshape(1)
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled score-grad kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled score-grad kernel must not receive cu_seqlens_q/k"
        if has_q_causal_offsets:
            assert QCausalOffsets is not None, "offset-compiled score-grad kernel requires q_causal_offsets at runtime"
        else:
            assert QCausalOffsets is None, "non-offset compiled score-grad kernel must not receive q_causal_offsets"
        s = _resolve_stream(current_stream)
        _ensure_compiled_score_grad(
            IdxScoreRaw,
            IdxLSE,
            AttnScoreRaw,
            AttnL1Norm,
            CuSeqlensQ,
            CuSeqlensK,
            QCausalOffsets,
            GradLoss,
            grad_scale,
            current_stream=current_stream,
        )
        _dense_score_grad_compile_cache[score_grad_key](
            IdxScoreRaw,
            AttnScoreRaw,
            IdxLSE,
            AttnL1Norm,
            CuSeqlensQ,
            CuSeqlensK,
            QCausalOffsets,
            GradLoss,
            cutlass.Float32(float(grad_scale)),
            cutlass.Int32(seqlen),
            cutlass.Int32(seqlen_k),
            s,
        )

    def _run_gemm_only(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK_f32,
        GradSignal,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        QCausalOffsets=None,
        current_stream=None,
    ):
        """Run fused dense Kernel 2. Caller must run score_grad first."""
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled kernel must not receive cu_seqlens_q/k"
        if has_q_causal_offsets:
            assert QCausalOffsets is not None, "offset-compiled kernel requires q_causal_offsets at runtime"
        else:
            assert QCausalOffsets is None, "non-offset compiled kernel must not receive q_causal_offsets"
        dummy_topk = _get_dummy_topk(IndexQ.device, current_stream=current_stream)
        s = _resolve_stream(current_stream)

        _ensure_compiled(
            IndexQ,
            Weights,
            IndexK,
            dIndexQ,
            dWeights,
            dIndexK_f32,
            GradSignal,
            CuSeqlensQ,
            CuSeqlensK,
            QCausalOffsets,
            current_stream=current_stream,
        )
        _dense_compile_cache[gemm_key](
            IndexQ,
            Weights,
            IndexK,
            dIndexQ,
            dWeights,
            dIndexK_f32,
            GradSignal,
            dummy_topk,
            cutlass.Float32(sm_scale),
            s,
            CuSeqlensQ,
            CuSeqlensK,
            cutlass.Int32(seqlen),
            cutlass.Int32(seqlen_k),
            QCausalOffsets,
        )

    def _run(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        attn_scores_raw,
        attn_l1norm,
        idx_scores_raw,
        idx_lse,
        GradLoss,
        grad_scale,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        QCausalOffsets=None,
        current_stream=None,
    ):
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled kernel must not receive cu_seqlens_q/k"
        _run_score_grad_only(
            idx_scores_raw,
            idx_lse,
            attn_scores_raw,
            attn_l1norm,
            GradLoss,
            grad_scale,
            CuSeqlensQ,
            CuSeqlensK,
            QCausalOffsets,
            current_stream=current_stream,
        )
        grad_signal = idx_scores_raw

        if dIndexK.dtype == torch.float32:
            _run_gemm_only(
                IndexQ,
                Weights,
                IndexK,
                dIndexQ,
                dWeights,
                dIndexK,
                grad_signal,
                CuSeqlensQ,
                CuSeqlensK,
                QCausalOffsets,
                current_stream=current_stream,
            )
        else:
            with _torch_stream_context(current_stream):
                dIndexK_f32 = torch.zeros_like(dIndexK, dtype=torch.float32)
            _run_gemm_only(
                IndexQ,
                Weights,
                IndexK,
                dIndexQ,
                dWeights,
                dIndexK_f32,
                grad_signal,
                CuSeqlensQ,
                CuSeqlensK,
                QCausalOffsets,
                current_stream=current_stream,
            )
            with _torch_stream_context(current_stream):
                dIndexK.copy_(dIndexK_f32)

    _run.score_grad = _run_score_grad_only
    _run.gemm_only = _run_gemm_only
    _run.ratio = ratio
    _run.is_varlen = is_varlen

    return _run
