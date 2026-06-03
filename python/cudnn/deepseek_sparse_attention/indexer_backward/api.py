"""APIBase wrappers for sparse and dense indexer backward.

Combines backend-specific CuTe-DSL kernels:

* ScoreGradSm90 or ScoreGradSm100 -- in-place score-grad precompute (kernel 1).
* IndexerBackwardSm90 or IndexerBackwardSm100 -- warp-specialized backward (kernel 2).
* DenseIndexerBackward uses the dense full-KV score-grad + GEMM factories.

A pure-torch dtype cast for ``dIndexK`` (FP32 → output dtype) finishes the
pipeline (kernel 3).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import cuda.bindings.driver as cuda

from cudnn.api_base import APIBase, TupleDict
from cudnn.deepseek_sparse_attention.utils.runtime import (
    torch_stream_context as _torch_stream_context,
)

from .dense_indexer_backward_sm100 import dense_indexer_backward_sm100
from .dense_indexer_backward_sm90 import dense_indexer_backward_sm90
from .indexer_backward_sm100 import indexer_backward_sm100
from .indexer_backward_sm90 import indexer_backward_sm90


def _as_grad_loss_tensor(grad_loss: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    if torch.is_tensor(grad_loss):
        return grad_loss.detach().to(device=device, dtype=torch.float32, copy=False).reshape(1)
    return torch.tensor([float(grad_loss)], dtype=torch.float32, device=device)


def _contiguous_input(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _contiguous_mutable(tensor: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if tensor.is_contiguous():
        return tensor, None
    return tensor.contiguous(), tensor


def _contiguous_output(tensor: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if tensor.is_contiguous():
        return tensor, None
    return torch.empty_like(tensor, memory_format=torch.contiguous_format), tensor


def _copy_back_if_needed(tensor: torch.Tensor, original: Optional[torch.Tensor]) -> None:
    if original is not None:
        original.copy_(tensor)


def _max_from_cu_seqlens(cu_seqlens: torch.Tensor, name: str) -> int:
    if cu_seqlens is None:
        raise ValueError(f"{name} is required")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1 or not cu_seqlens.is_cuda:
        raise ValueError(f"{name} must be a 1D CUDA int32 tensor")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(lengths.max().item()) if lengths.numel() else 0


def _dense_shapes(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    attn_score: torch.Tensor,
    attn_l1norm: torch.Tensor,
    index_score: torch.Tensor,
    index_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
) -> tuple[bool, int, int, int, int, int, int, int]:
    is_thd_q = cu_seqlens_q is not None
    is_thd_k = cu_seqlens_k is not None
    if is_thd_q != is_thd_k:
        raise ValueError("DenseIndexerBackward THD mode requires both cu_seqlens_q and cu_seqlens_k")
    is_thd = is_thd_q

    if is_thd:
        if index_q.ndim != 3 or weights.ndim != 2 or index_k.ndim != 2:
            raise ValueError("THD dense indexer backward expects q=(T_q,H,D), w=(T_q,H), k=(T_k,D)")
        total_q, heads, head_dim = index_q.shape
        total_k, head_dim_k = index_k.shape
        if head_dim != head_dim_k:
            raise ValueError("index_q and index_k head dimensions must match")
        batch = int(cu_seqlens_q.shape[0]) - 1
        max_q = int(max_seqlen_q) if max_seqlen_q is not None else _max_from_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
        max_k = int(max_seqlen_k) if max_seqlen_k is not None else _max_from_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")
        expected_score_shape = (total_q, max_k)
        expected_denom_shape = (total_q,)
        if weights.shape != (total_q, heads):
            raise ValueError(f"weights shape mismatch: expected {(total_q, heads)}, got {tuple(weights.shape)}")
        if attn_score.shape != expected_score_shape or index_score.shape != expected_score_shape:
            raise ValueError("THD dense score tensors must have shape " f"{expected_score_shape}, got {tuple(attn_score.shape)} and {tuple(index_score.shape)}")
        if attn_l1norm.shape != expected_denom_shape or index_lse.shape != expected_denom_shape:
            raise ValueError("THD dense denom tensors must have shape " f"{expected_denom_shape}, got {tuple(attn_l1norm.shape)} and {tuple(index_lse.shape)}")
        return True, batch, total_q, total_k, heads, head_dim, max_q, max_k

    if index_q.ndim != 4 or weights.ndim != 3 or index_k.ndim != 3:
        raise ValueError("BSHD dense indexer backward expects q=(B,S_q,H,D), w=(B,S_q,H), k=(B,S_k,D)")
    batch, seqlen_q, heads, head_dim = index_q.shape
    batch_k, seqlen_k, head_dim_k = index_k.shape
    if batch != batch_k or head_dim != head_dim_k:
        raise ValueError("index_q and index_k batch/head dimensions must match")
    if weights.shape != (batch, seqlen_q, heads):
        raise ValueError(f"weights shape mismatch: expected {(batch, seqlen_q, heads)}, got {tuple(weights.shape)}")
    expected_score_shape = (batch, seqlen_q, seqlen_k)
    expected_denom_shape = (batch, seqlen_q)
    if attn_score.shape != expected_score_shape or index_score.shape != expected_score_shape:
        raise ValueError("BSHD dense score tensors must have shape " f"{expected_score_shape}, got {tuple(attn_score.shape)} and {tuple(index_score.shape)}")
    if attn_l1norm.shape != expected_denom_shape or index_lse.shape != expected_denom_shape:
        raise ValueError("BSHD dense denom tensors must have shape " f"{expected_denom_shape}, got {tuple(attn_l1norm.shape)} and {tuple(index_lse.shape)}")
    return False, batch, batch * seqlen_q, batch * seqlen_k, heads, head_dim, seqlen_q, seqlen_k


class IndexerBackward(APIBase):
    """End-to-end indexer backward (3 fused stages).

    Given the forward-computed ``AttnScore`` (target distribution) and
    ``IndexScore`` (predict distribution) along with the sparse top-K indices,
    produces the gradients ``d_index_q``, ``d_weights``, ``d_index_k``.

    ``grad_scale = loss_coeff / (B * S_q)`` is a runtime scalar passed at
    ``execute`` time alongside ``grad_loss`` — the kernel consumes both as
    multiplicative factors inside the score-grad precompute. Neither is
    part of the compile cache, so changing them across iterations reuses
    the same compiled kernel.
    """

    def __init__(
        self,
        sample_index_q: torch.Tensor,  # (B, S_q, H, D) BF16
        sample_weights: torch.Tensor,  # (B, S_q, H) BF16
        sample_index_k: torch.Tensor,  # (B, S_k, D) BF16
        sample_d_index_q: torch.Tensor,  # same shape/dtype as index_q
        sample_d_weights: torch.Tensor,  # same shape/dtype as weights
        sample_d_index_k: torch.Tensor,  # same shape/dtype as index_k
        sample_attn_score: torch.Tensor,  # (B, S_q, topk) FP32 — target
        sample_index_score: torch.Tensor,  # (B, S_q, topk) FP32 — predict
        sample_topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
        sm_scale: float = 1.0,
        block_I: int = 128,
    ):
        super().__init__()
        self.iq_desc = self._make_tensor_desc(sample_index_q, name="sample_index_q")
        self.w_desc = self._make_tensor_desc(sample_weights, name="sample_weights")
        self.ik_desc = self._make_tensor_desc(sample_index_k, name="sample_index_k")
        self.diq_desc = self._make_tensor_desc(sample_d_index_q, name="sample_d_index_q")
        self.dw_desc = self._make_tensor_desc(sample_d_weights, name="sample_d_weights")
        self.dik_desc = self._make_tensor_desc(sample_d_index_k, name="sample_d_index_k")
        self.attn_desc = self._make_tensor_desc(sample_attn_score, name="sample_attn_score")
        self.idx_score_desc = self._make_tensor_desc(sample_index_score, name="sample_index_score")
        self.topk_desc = self._make_tensor_desc(sample_topk_indices, name="sample_topk_indices")

        b, s_q, h, d = sample_index_q.shape
        s_k = sample_index_k.shape[1]
        topk = sample_topk_indices.shape[-1]
        self.batch = b
        self.seqlen = s_q
        self.seqlen_k = s_k
        self.heads = h
        self.head_dim = d
        self.topk = topk
        self.sm_scale = float(sm_scale)
        self.block_I = int(block_I)

    def check_support(self) -> bool:
        major, _ = torch.cuda.get_device_capability()
        self._runtime_error_if(
            major != 9 and major < 10,
            f"IndexerBackward requires SM90 or SM100+, found SM{major}",
        )
        self._check_dtype(self.iq_desc, torch.bfloat16, name="index_q")
        self._check_dtype(self.w_desc, torch.bfloat16, name="weights")
        self._check_dtype(self.ik_desc, torch.bfloat16, name="index_k")
        self._check_dtype(self.attn_desc, torch.float32, name="attn_score")
        self._check_dtype(self.idx_score_desc, torch.float32, name="index_score")
        self._check_dtype(self.topk_desc, torch.int32, name="topk_indices")
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        # indexer_backward_sm100 returns a closure whose GEMM stage lazy-
        # compiles on first call. ``grad_scale`` is supplied at execute()
        # time — passing it to the factory is a no-op now that it's not
        # part of the cache key, and it's been removed from the factory
        # signature on both backends.
        major, _ = torch.cuda.get_device_capability()
        kernel_factory = indexer_backward_sm90 if major == 9 else indexer_backward_sm100
        self._compiled_kernel = kernel_factory(
            self.batch,
            self.seqlen,
            self.seqlen_k,
            self.heads,
            self.head_dim,
            self.topk,
            sm_scale=self.sm_scale,
            block_I=self.block_I,
        )

    def execute(
        self,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        d_index_q: torch.Tensor,
        d_weights: torch.Tensor,
        d_index_k: torch.Tensor,
        attn_score: torch.Tensor,
        index_score: torch.Tensor,
        topk_indices: torch.Tensor,
        loss_coeff: float = 1.0,
        grad_loss: Union[float, torch.Tensor] = 1.0,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        # ``grad_scale`` is forwarded as a runtime ``Float32`` arg; the
        # compiled kernel is reused when loss_coeff changes for the same
        # cached tensor shape.
        grad_scale = float(loss_coeff) / (int(self.batch) * int(self.seqlen))
        grad_loss_tensor = _as_grad_loss_tensor(grad_loss, index_q.device)
        self._compiled_kernel(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            index_score,
            topk_indices,
            grad_loss_tensor,
            grad_scale,
            current_stream,
        )


class DenseIndexerBackward(APIBase):
    """Dense full-KV indexer backward.

    Consumes raw dense attention/indexer score tensors and their denominators,
    computes the dense KL score gradient, then runs the indexer GEMM backward
    to produce ``d_index_q``, ``d_weights``, and ``d_index_k``.
    """

    def __init__(
        self,
        sample_index_q: torch.Tensor,
        sample_weights: torch.Tensor,
        sample_index_k: torch.Tensor,
        sample_d_index_q: torch.Tensor,
        sample_d_weights: torch.Tensor,
        sample_d_index_k: torch.Tensor,
        sample_attn_score: torch.Tensor,
        sample_attn_l1norm: torch.Tensor,
        sample_index_score: torch.Tensor,
        sample_index_lse: torch.Tensor,
        sm_scale: float = 1.0,
        block_I: int = 128,
        ratio: int = 1,
        is_thd: bool = False,
        batch: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ):
        super().__init__()
        self.iq_desc = self._make_tensor_desc(sample_index_q, name="sample_index_q")
        self.w_desc = self._make_tensor_desc(sample_weights, name="sample_weights")
        self.ik_desc = self._make_tensor_desc(sample_index_k, name="sample_index_k")
        self.diq_desc = self._make_tensor_desc(sample_d_index_q, name="sample_d_index_q")
        self.dw_desc = self._make_tensor_desc(sample_d_weights, name="sample_d_weights")
        self.dik_desc = self._make_tensor_desc(sample_d_index_k, name="sample_d_index_k")
        self.attn_desc = self._make_tensor_desc(sample_attn_score, name="sample_attn_score")
        self.attn_denom_desc = self._make_tensor_desc(sample_attn_l1norm, name="sample_attn_l1norm")
        self.idx_score_desc = self._make_tensor_desc(sample_index_score, name="sample_index_score")
        self.idx_lse_desc = self._make_tensor_desc(sample_index_lse, name="sample_index_lse")

        self.sm_scale = float(sm_scale)
        self.block_I = int(block_I)
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)

        if self.is_thd:
            total_q, heads, head_dim = sample_index_q.shape
            total_k = sample_index_k.shape[0]
            if batch is None or max_seqlen_q is None or max_seqlen_k is None:
                raise ValueError("THD dense indexer backward requires batch and max_seqlen_q/k")
            self.batch = int(batch)
            self.normalization_tokens = int(total_q)
            self.total_k = int(total_k)
            self.max_seqlen_q = int(max_seqlen_q)
            self.max_seqlen_k = int(max_seqlen_k)
        else:
            b, seqlen_q, heads, head_dim = sample_index_q.shape
            self.batch = int(b)
            self.normalization_tokens = int(b * seqlen_q)
            self.total_k = int(b * sample_index_k.shape[1])
            self.max_seqlen_q = int(seqlen_q)
            self.max_seqlen_k = int(sample_index_k.shape[1])
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self._uses_current_stream_pipeline = False

    def check_support(self) -> bool:
        major, _ = torch.cuda.get_device_capability()
        self._runtime_error_if(
            major != 9 and major < 10,
            f"DenseIndexerBackward requires SM90 or SM100+, found SM{major}",
        )
        self._check_dtype(self.iq_desc, torch.bfloat16, name="index_q")
        self._check_dtype(self.w_desc, torch.bfloat16, name="weights")
        self._check_dtype(self.ik_desc, torch.bfloat16, name="index_k")
        self._check_dtype(self.diq_desc, torch.bfloat16, name="d_index_q")
        self._check_dtype(self.dw_desc, torch.bfloat16, name="d_weights")
        self._check_dtype(self.dik_desc, [torch.bfloat16, torch.float32], name="d_index_k")
        self._check_dtype(self.attn_desc, torch.float32, name="attn_score")
        self._check_dtype(self.attn_denom_desc, torch.float32, name="attn_l1norm")
        self._check_dtype(self.idx_score_desc, torch.float32, name="index_score")
        self._check_dtype(self.idx_lse_desc, torch.float32, name="index_lse")
        self._value_error_if(self.block_I <= 0, f"block_I must be positive, got {self.block_I}")
        self._value_error_if(self.ratio < 1, f"ratio must be >= 1, got {self.ratio}")
        self._value_error_if(self.heads < 64, f"DenseIndexerBackward requires heads >= 64, got {self.heads}")
        self._value_error_if(
            self.max_seqlen_q > self.max_seqlen_k * self.ratio,
            "DenseIndexerBackward requires S_q <= S_k * ratio for bottom-right causal alignment",
        )
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        major, _ = torch.cuda.get_device_capability()
        kernel_factory = dense_indexer_backward_sm90 if major == 9 else dense_indexer_backward_sm100
        self._uses_current_stream_pipeline = major == 9
        self._compiled_kernel = kernel_factory(
            self.batch,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.heads,
            self.head_dim,
            sm_scale=self.sm_scale,
            block_I=self.block_I,
            ratio=self.ratio,
            is_varlen=self.is_thd,
        )

    def execute(
        self,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        d_index_q: torch.Tensor,
        d_weights: torch.Tensor,
        d_index_k: torch.Tensor,
        attn_score: torch.Tensor,
        attn_l1norm: torch.Tensor,
        index_score: torch.Tensor,
        index_lse: torch.Tensor,
        loss_coeff: float = 1.0,
        grad_loss: Union[float, torch.Tensor] = 1.0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        backend_stream = None if self._uses_current_stream_pipeline else current_stream
        with _torch_stream_context(backend_stream):
            grad_loss_value = float(_as_grad_loss_tensor(grad_loss, index_q.device).item())
            grad_scale = float(loss_coeff) * grad_loss_value / max(int(self.normalization_tokens), 1)

            # Dense backward's dK path uses atomic/bulk reductions into fp32.
            d_index_k_target = d_index_k
            if d_index_k.dtype == torch.float32:
                d_index_k_f32 = d_index_k
                d_index_k_f32.zero_()
            else:
                d_index_k_f32 = torch.zeros_like(d_index_k, dtype=torch.float32)

        self._compiled_kernel(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k_f32,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            grad_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            backend_stream,
        )

        if d_index_k_f32 is not d_index_k_target:
            with _torch_stream_context(backend_stream):
                d_index_k_target.copy_(d_index_k_f32)


_cache_of_IndexerBackwardObjects: dict = {}


def indexer_backward_wrapper(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    attn_score: torch.Tensor,
    index_score: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    grad_loss: Union[float, torch.Tensor] = 1.0,
    block_I: int = 128,
    d_index_q: Optional[torch.Tensor] = None,
    d_weights: Optional[torch.Tensor] = None,
    d_index_k: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'d_index_q', 'd_weights', 'd_index_k'}``.

    ``attn_score`` and ``index_score`` are consumed in-place: the kernel
    overwrites ``attn_score`` with ``grad_signal`` and ``index_score`` with
    ``sum_grad`` during the score-grad precompute stage.

    Args:
        sm_scale: indexer softmax scale baked into the forward via the
            weights-scaling trick.
        loss_coeff: coefficient scaling the KL-divergence loss in the
            forward (``indexer_loss = loss_coeff * kl.mean()``).
        grad_loss: scalar gradient of the outer loss w.r.t. the KL term;
            typically ``1.0`` when the KL loss is summed into the total
            training loss with unit weight. Accepts a 0-D tensor or float.
    """
    if d_index_q is None:
        d_index_q = torch.empty_like(index_q)
    if d_weights is None:
        d_weights = torch.empty_like(weights)
    if d_index_k is None:
        d_index_k = torch.empty_like(index_k)

    b, s_q, h, d = index_q.shape
    s_k = index_k.shape[1]
    topk = topk_indices.shape[-1]

    # ``grad_scale = loss_coeff / (B * S_q)`` is a runtime arg (forwarded
    # into the score-grad kernel as a runtime ``Float32``), so it is not
    # part of the cache key — same compiled kernel reused across calls
    # with different loss_coeff for the same tensor shape. Shape
    # changes still get their own cache entries.
    key = (
        index_q.dtype,
        weights.dtype,
        index_k.dtype,
        b,
        s_q,
        s_k,
        h,
        d,
        topk,
        float(sm_scale),
        int(block_I),
    )
    obj = _cache_of_IndexerBackwardObjects.get(key)
    if obj is None:
        obj = IndexerBackward(
            sample_index_q=index_q,
            sample_weights=weights,
            sample_index_k=index_k,
            sample_d_index_q=d_index_q,
            sample_d_weights=d_weights,
            sample_d_index_k=d_index_k,
            sample_attn_score=attn_score,
            sample_index_score=index_score,
            sample_topk_indices=topk_indices,
            sm_scale=sm_scale,
            block_I=block_I,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_IndexerBackwardObjects[key] = obj

    obj.execute(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        loss_coeff=loss_coeff,
        grad_loss=grad_loss,
        current_stream=stream,
    )
    return TupleDict(d_index_q=d_index_q, d_weights=d_weights, d_index_k=d_index_k)


_cache_of_DenseIndexerBackwardObjects: dict = {}


def dense_indexer_backward_wrapper(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    attn_score: torch.Tensor,
    attn_l1norm: torch.Tensor,
    index_score: torch.Tensor,
    index_lse: torch.Tensor,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    grad_loss: Union[float, torch.Tensor] = 1.0,
    block_I: int = 128,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    d_index_q: Optional[torch.Tensor] = None,
    d_weights: Optional[torch.Tensor] = None,
    d_index_k: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Dense full-KV indexer backward. Returns ``{'d_index_q', 'd_weights', 'd_index_k'}``.

    ``attn_score`` and ``index_score`` are raw dense scores from
    ``dense_attn_score_recompute_wrapper`` and
    ``dense_indexer_score_recompute_wrapper`` respectively. They are consumed
    in-place by the score-gradient precompute stage.
    """
    major, _ = torch.cuda.get_device_capability()
    backend_stream = None if major == 9 else stream

    with _torch_stream_context(backend_stream):
        cu_seqlens_q = _contiguous_input(cu_seqlens_q) if cu_seqlens_q is not None else None
        cu_seqlens_k = _contiguous_input(cu_seqlens_k) if cu_seqlens_k is not None else None

        index_q_exec = _contiguous_input(index_q)
        weights_exec = _contiguous_input(weights)
        index_k_exec = _contiguous_input(index_k)
        attn_l1norm_exec = _contiguous_input(attn_l1norm)
        index_lse_exec = _contiguous_input(index_lse)
        attn_score_exec, attn_score_original = _contiguous_mutable(attn_score)
        index_score_exec, index_score_original = _contiguous_mutable(index_score)

        (
            is_thd,
            batch,
            total_q,
            total_k,
            heads,
            head_dim,
            max_q,
            max_k,
        ) = _dense_shapes(
            index_q_exec,
            weights_exec,
            index_k_exec,
            attn_score_exec,
            attn_l1norm_exec,
            index_score_exec,
            index_lse_exec,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )

        if d_index_q is None:
            d_index_q = torch.empty_like(index_q_exec)
        if d_weights is None:
            d_weights = torch.empty_like(weights_exec)
        if d_index_k is None:
            d_index_k = torch.empty_like(index_k_exec)

        d_index_q_result = d_index_q
        d_weights_result = d_weights
        d_index_k_result = d_index_k
        d_index_q_exec, d_index_q_original = _contiguous_output(d_index_q)
        d_weights_exec, d_weights_original = _contiguous_output(d_weights)
        d_index_k_exec, d_index_k_original = _contiguous_output(d_index_k)

    key = (
        index_q.dtype,
        weights.dtype,
        index_k.dtype,
        d_index_k.dtype,
        is_thd,
        batch,
        heads,
        head_dim,
        max_q,
        max_k,
        total_q if is_thd else None,
        total_k if is_thd else None,
        float(sm_scale),
        int(block_I),
        int(ratio),
    )
    obj = _cache_of_DenseIndexerBackwardObjects.get(key)
    if obj is None:
        obj = DenseIndexerBackward(
            sample_index_q=index_q_exec,
            sample_weights=weights_exec,
            sample_index_k=index_k_exec,
            sample_d_index_q=d_index_q_exec,
            sample_d_weights=d_weights_exec,
            sample_d_index_k=d_index_k_exec,
            sample_attn_score=attn_score_exec,
            sample_attn_l1norm=attn_l1norm_exec,
            sample_index_score=index_score_exec,
            sample_index_lse=index_lse_exec,
            sm_scale=sm_scale,
            block_I=block_I,
            ratio=ratio,
            is_thd=is_thd,
            batch=batch,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_DenseIndexerBackwardObjects[key] = obj

    obj.execute(
        index_q_exec,
        weights_exec,
        index_k_exec,
        d_index_q_exec,
        d_weights_exec,
        d_index_k_exec,
        attn_score_exec,
        attn_l1norm_exec,
        index_score_exec,
        index_lse_exec,
        loss_coeff=loss_coeff,
        grad_loss=grad_loss,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        current_stream=backend_stream,
    )
    with _torch_stream_context(backend_stream):
        _copy_back_if_needed(attn_score_exec, attn_score_original)
        _copy_back_if_needed(index_score_exec, index_score_original)
        _copy_back_if_needed(d_index_q_exec, d_index_q_original)
        _copy_back_if_needed(d_weights_exec, d_weights_original)
        _copy_back_if_needed(d_index_k_exec, d_index_k_original)

    return TupleDict(d_index_q=d_index_q_result, d_weights=d_weights_result, d_index_k=d_index_k_result)
