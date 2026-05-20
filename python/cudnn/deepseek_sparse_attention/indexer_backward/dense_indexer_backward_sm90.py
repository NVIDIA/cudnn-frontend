"""Dense Indexer Backward — SM90 CuTe-DSL factory."""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

from .indexer_backward_sm90 import (
    CLIP_PROB_MIN,
    EPS,
    IndexerBackwardSm90,
)

_dense_compile_cache: dict = {}


def _bottom_right_valid_mask(seqlen_q: int, seqlen_k: int, ratio: int, device) -> torch.Tensor:
    q_start = seqlen_k * ratio - seqlen_q
    q_idx = torch.arange(seqlen_q, device=device)
    kv_idx = torch.arange(seqlen_k, device=device)
    col_limit = ((q_start + q_idx + 1) // ratio).clamp(max=seqlen_k)
    return kv_idx.unsqueeze(0) < col_limit.unsqueeze(1)


def _score_grad_dense_slice_inplace(attn_slice, idx_slice, grad_scale, ratio):
    seqlen_q, seqlen_k = idx_slice.shape
    valid_mask = _bottom_right_valid_mask(seqlen_q, seqlen_k, ratio, idx_slice.device)

    masked_idx_scores = idx_slice.masked_fill(~valid_mask, -1e9)
    index_score = torch.softmax(masked_idx_scores, dim=-1)
    attn_score_masked = attn_slice.masked_fill(~valid_mask, 0.0)
    attn_score = attn_score_masked / attn_score_masked.sum(dim=-1, keepdim=True).clamp(min=EPS)

    target_eff = attn_score.clamp(min=CLIP_PROB_MIN)
    log_clip_mask = ((index_score >= CLIP_PROB_MIN) & valid_mask).to(attn_slice.dtype)
    g = -target_eff * log_clip_mask * grad_scale
    sum_grad = g.sum(dim=-1, keepdim=True)
    grad_signal = (g - index_score * sum_grad).masked_fill(~valid_mask, 0.0)
    attn_slice.copy_(grad_signal)


def _score_grad_dense_inplace(
    attn_scores_raw,
    attn_l1norm,
    idx_scores_raw,
    idx_lse,
    grad_scale,
    ratio: int = 1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    """Compute dense grad_signal from raw scores and overwrite attn scores."""
    assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
    is_varlen = cu_seqlens_q is not None
    if is_varlen:
        assert cu_seqlens_k is not None
        cu_q = cu_seqlens_q.detach().cpu().tolist()
        cu_k = cu_seqlens_k.detach().cpu().tolist()
        for b in range(len(cu_q) - 1):
            qs, qe = cu_q[b], cu_q[b + 1]
            ks, ke = cu_k[b], cu_k[b + 1]
            sq_b = qe - qs
            sk_b = ke - ks
            if sq_b == 0 or sk_b == 0:
                continue
            _score_grad_dense_slice_inplace(
                attn_scores_raw[qs:qe, :sk_b],
                idx_scores_raw[qs:qe, :sk_b],
                grad_scale,
                ratio,
            )
            if sk_b < attn_scores_raw.shape[1]:
                attn_scores_raw[qs:qe, sk_b:].zero_()
    else:
        for b in range(idx_scores_raw.shape[0]):
            _score_grad_dense_slice_inplace(
                attn_scores_raw[b],
                idx_scores_raw[b],
                grad_scale,
                ratio,
            )


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
):
    """Factory for the dense indexer backward gradient kernel on SM90."""
    assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
    key = (
        is_varlen,
        batch,
        seqlen,
        seqlen_k,
        heads,
        dim,
        sm_scale,
        block_I,
        ratio,
    )
    if key not in _dense_compile_cache:
        _dense_compile_cache[key] = _build_cute_dsl_dense_kernel(
            batch,
            seqlen,
            seqlen_k,
            heads,
            dim,
            sm_scale,
            block_I,
            ratio,
            is_varlen,
        )
    return _dense_compile_cache[key]


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
):
    cap = torch.cuda.get_device_capability()[0]
    if cap < 9:
        raise RuntimeError(f"Requires SM90+ (got SM{cap}0)")
    if cap >= 10:
        raise RuntimeError("Use SM100 kernel for Blackwell")

    topk = seqlen_k
    kernel_obj = IndexerBackwardSm90(
        head_dim=dim,
        heads=heads,
        block_I=block_I,
        topk=topk,
        is_dense=True,
    )

    compiled_holder = [None]
    dummy_topk_holder = [None]

    def _get_dummy_topk(device, current_stream=None):
        if dummy_topk_holder[0] is None or dummy_topk_holder[0].device != device:
            with _torch_stream_context(current_stream):
                dummy_topk_holder[0] = torch.zeros(batch, seqlen, seqlen_k, device=device, dtype=torch.int32)
        return dummy_topk_holder[0]

    def _ensure_compiled(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, CuSeqlensQ, CuSeqlensK, current_stream=None):
        s = _resolve_stream(current_stream)
        if compiled_holder[0] is None:
            dummy_topk = _get_dummy_topk(IndexQ.device, current_stream=current_stream)
            cuq_arg = to_cute_tensor(CuSeqlensQ) if CuSeqlensQ is not None else None
            cuk_arg = to_cute_tensor(CuSeqlensK) if CuSeqlensK is not None else None
            cute_args = [to_cute_tensor(t) for t in [IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, dummy_topk]]
            compiled_holder[0] = cute.compile(
                kernel_obj,
                *cute_args,
                cutlass.Float32(sm_scale),
                s,
                cuq_arg,
                cuk_arg,
                cutlass.Int32(seqlen),
                cutlass.Int32(seqlen_k),
                options=compile_options(),
            )

    def _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, CuSeqlensQ=None, CuSeqlensK=None, current_stream=None):
        """Run fused dense Kernel 2. Caller must run score_grad first."""
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled kernel must not receive cu_seqlens_q/k"
        dummy_topk = _get_dummy_topk(IndexQ.device, current_stream=current_stream)
        s = _resolve_stream(current_stream)

        _ensure_compiled(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, GradSignal, CuSeqlensQ, CuSeqlensK, current_stream=current_stream)
        compiled_holder[0](
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
        grad_scale,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        current_stream=None,
    ):
        if is_varlen:
            assert CuSeqlensQ is not None and CuSeqlensK is not None, "THD-compiled kernel requires cu_seqlens_q/k at runtime"
        else:
            assert CuSeqlensQ is None and CuSeqlensK is None, "BSHD-compiled kernel must not receive cu_seqlens_q/k"
        # Keep the full SM90 dense pipeline aligned with /code/indexer: score-grad
        # is still torch-based and therefore runs on PyTorch's current stream.
        # Avoid mixing it with a non-current-stream GEMM until the score-grad
        # stage has a DSL implementation upstream.
        _score_grad_dense_inplace(
            attn_scores_raw,
            attn_l1norm,
            idx_scores_raw,
            idx_lse,
            grad_scale,
            ratio=ratio,
            cu_seqlens_q=CuSeqlensQ,
            cu_seqlens_k=CuSeqlensK,
        )
        grad_signal = attn_scores_raw

        if dIndexK.dtype == torch.float32:
            _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK, grad_signal, CuSeqlensQ, CuSeqlensK)
        else:
            dIndexK_f32 = torch.zeros_like(dIndexK, dtype=torch.float32)
            _run_gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK_f32, grad_signal, CuSeqlensQ, CuSeqlensK)
            dIndexK.copy_(dIndexK_f32)

    _run.score_grad = _score_grad_dense_inplace
    _run.gemm_only = _run_gemm_only
    _run.ratio = ratio
    _run.is_varlen = is_varlen

    return _run
