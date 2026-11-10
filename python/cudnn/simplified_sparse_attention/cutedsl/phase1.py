# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full-support SSA attention and indexer objective for Phase-1 warmup.

Q/indexer-Q may be CP-local while MQA K/V and indexer-K are gathered into
global packed order. Row-wise bounds describe document-local causal support.
Both the dense output and teacher/student KL use streaming FP32 accumulation,
without materializing a persistent quadratic teacher tensor.

The teacher average requires all query heads. A caller that shards heads must
reduce teacher mass across ranks before forming the global teacher.
"""

import importlib
import math
from typing import Optional, Union

import torch

from cudnn.frost.buffers import cutedsl_requirement_error

_KF_SUPPORTED_QUERY_HEADS = (16, 32, 96)
_KF_TEACHER_DIM = 256
_KF_INDEXER_DIM = 128
_KF_KERNEL = None
_FLASH_ATTN4_VARLEN = None


def _phase1_kf_kernel_module():
    """Check the DSL version before importing the selected CuTe kernel."""
    global _KF_KERNEL
    error = cutedsl_requirement_error("dense_indexer_kl", floor=(4, 8, 0))
    if error:
        raise ImportError(error)
    if _KF_KERNEL is None:
        _KF_KERNEL = importlib.import_module(f"{__package__}.kernels.phase1_dense_indexer_kl")
    return _KF_KERNEL


def _flash_attn4_varlen():
    """Import the optional dense teacher backend only when explicitly selected."""
    global _FLASH_ATTN4_VARLEN
    if _FLASH_ATTN4_VARLEN is None:
        try:
            from flash_attn.cute.interface import flash_attn_varlen_func
        except ImportError as error:
            raise ImportError(
                "SSA Phase-1 dense_attention_backend='flash-attn-4' requires " "the flash-attn-4 package with flash_attn.cute.interface"
            ) from error
        _FLASH_ATTN4_VARLEN = flash_attn_varlen_func
    return _FLASH_ATTN4_VARLEN


def _canonicalize_attention_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalize B=1 SBHD or packed THD MQA inputs."""
    if query.ndim == 4:
        if query.size(1) != 1:
            raise ValueError("SSA Phase 1 supports only batch size one for SBHD input")
        query = query[:, 0]
    elif query.ndim != 3:
        raise ValueError("SSA Phase-1 query must be [Tq,Hq,D] or [Tq,1,Hq,D]")

    compact = []
    for name, tensor in (("key", key), ("value", value)):
        if tensor.ndim == 4:
            if tensor.shape[1:3] != (1, 1):
                raise ValueError(f"SSA Phase 1 requires one MQA {name} head")
            tensor = tensor[:, 0, 0]
        elif tensor.ndim == 3 and tensor.size(1) == 1:
            tensor = tensor[:, 0]
        else:
            raise ValueError(f"SSA Phase-1 {name} must be [Tk,1,D] or [Tk,1,1,D]")
        compact.append(tensor)
    key, value = compact

    if query.size(-1) != key.size(-1):
        raise ValueError("SSA Phase-1 teacher Q/K head dimensions must match")
    if key.size(0) != value.size(0):
        raise ValueError("SSA Phase-1 K/V row counts must match")
    if key.device != query.device or value.device != query.device:
        raise ValueError("SSA Phase-1 attention inputs must be on one device")
    return query, key, value


def _canonicalize_row_metadata(
    query: torch.Tensor,
    key: torch.Tensor,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    query_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move and validate per-query legal-key metadata."""
    starts = key_starts.to(device=query.device, dtype=torch.int64).reshape(-1)
    ends = key_ends.to(device=query.device, dtype=torch.int64).reshape(-1)
    valid = query_valid.to(device=query.device, dtype=torch.bool).reshape(-1)
    if starts.shape != (query.size(0),) or ends.shape != (query.size(0),):
        raise ValueError("SSA Phase-1 key_starts/key_ends must have shape [Tq]")
    if valid.shape != (query.size(0),):
        raise ValueError("SSA Phase-1 query_valid must have shape [Tq]")
    if query.device.type == "cpu":
        if not bool(((starts >= 0) & (starts <= ends) & (ends <= key.size(0))).all()):
            raise ValueError("SSA Phase-1 received out-of-range key bounds")
        if not bool(((~valid) | (ends > starts)).all()):
            raise ValueError("every valid query row must have non-empty key support")
    return starts, ends, valid


@torch.no_grad()
def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    query_valid: torch.Tensor,
    softmax_scale: float,
    query_chunk_size: Optional[int] = None,
    key_chunk_size: Optional[int] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Compute local-query/global-MQA-KV dense attention with online softmax.

    This Phase-1-only path is deliberately detached: the main model is frozen,
    while the separate indexer KL autograd function owns all trainable gradients.
    The returned THD layout matches packed attention: ``[Tq,Hq,Dv]``.
    """
    query, key, value = _canonicalize_attention_inputs(query, key, value)
    if query.size(0) == 0 or key.size(0) == 0:
        raise ValueError("SSA Phase-1 requires nonempty Q and K/V streams")
    starts, ends, valid = _canonicalize_row_metadata(query, key, key_starts, key_ends, query_valid)
    query_chunk_size = min(query.size(0), query_chunk_size or 256)
    key_chunk_size = min(key.size(0), key_chunk_size or 1024)
    if query_chunk_size <= 0 or key_chunk_size <= 0:
        raise ValueError("SSA Phase-1 chunk sizes must be positive")

    accumulation_dtype = _accumulation_dtype(query, key, value)
    outputs = []
    teacher_lses = []
    for q_start in range(0, query.size(0), query_chunk_size):
        q_end = min(q_start + query_chunk_size, query.size(0))
        query_tile = query[q_start:q_end].to(dtype=accumulation_dtype)
        starts_tile = starts[q_start:q_end]
        ends_tile = ends[q_start:q_end]
        valid_tile = valid[q_start:q_end]
        row_max = torch.full(
            (query_tile.size(0), query_tile.size(1)),
            float("-inf"),
            dtype=accumulation_dtype,
            device=query.device,
        )
        row_sum = torch.zeros_like(row_max)
        row_output = torch.zeros((*row_max.shape, value.size(-1)), dtype=accumulation_dtype, device=query.device)
        for k_start in range(0, key.size(0), key_chunk_size):
            k_end = min(k_start + key_chunk_size, key.size(0))
            legal = _legal_mask(starts_tile, ends_tile, valid_tile, k_start, k_end)
            logits = torch.matmul(query_tile, key[k_start:k_end].to(dtype=accumulation_dtype).t()) * softmax_scale
            masked_logits = logits.masked_fill(~legal[:, None, :], float("-inf"))
            block_max = masked_logits.amax(dim=-1)
            new_max = torch.maximum(row_max, block_max)
            safe_new_max = torch.where(torch.isfinite(new_max), new_max, 0.0)
            old_scale = torch.where(torch.isfinite(row_max), torch.exp(row_max - safe_new_max), 0.0)
            weights = torch.exp(masked_logits - safe_new_max[:, :, None])
            row_output = row_output * old_scale[:, :, None] + torch.matmul(weights, value[k_start:k_end].to(dtype=accumulation_dtype))
            row_sum = row_sum * old_scale + weights.sum(dim=-1)
            row_max = new_max
        output_tile = row_output / row_sum.clamp_min(torch.finfo(accumulation_dtype).tiny)[:, :, None]
        output_tile.masked_fill_(~valid_tile[:, None, None], 0.0)
        outputs.append(output_tile)
        if return_lse:
            # ``row_max`` and ``row_sum`` are the online-softmax state from
            # the QK pass above, so this is the authoritative natural-log LSE
            # without another teacher matmul. Invalid physical rows use a
            # finite placeholder because query_valid excludes them downstream.
            lse_tile = row_max + torch.log(row_sum)
            lse_tile.masked_fill_(~valid_tile[:, None], 0.0)
            teacher_lses.append(lse_tile)

    output = torch.cat(outputs, dim=0).to(dtype=value.dtype)
    if return_lse:
        return output, torch.cat(teacher_lses, dim=0).contiguous()
    return output


@torch.no_grad()
def dense_attention_flash4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    query_valid: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen dense teacher through FlashAttention-4 with O(T) storage.

    The backend consumes packed ``[T,H,D]`` Q and MQA K/V, returns the dense
    attention output and the natural-log softmax LSE used by the fused Phase-1
    KL kernel, and never materializes the quadratic attention probabilities.
    Unequal Q/K sequence lengths use FlashAttention's bottom-right causal
    alignment.
    """
    query, key, value = _canonicalize_attention_inputs(query, key, value)
    if not query.is_cuda:
        raise NotImplementedError("SSA Phase-1 FlashAttention-4 requires CUDA tensors")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise NotImplementedError("SSA Phase-1 FlashAttention-4 requires BF16 Q/K/V")
    if query.size(-1) != _KF_TEACHER_DIM or value.size(-1) != _KF_TEACHER_DIM:
        raise NotImplementedError("SSA Phase-1 FlashAttention-4 requires Dqk=Dv=256")
    if query.size(1) not in _KF_SUPPORTED_QUERY_HEADS:
        raise NotImplementedError("SSA Phase-1 FlashAttention-4 requires " f"Hq in {_KF_SUPPORTED_QUERY_HEADS}")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("SSA Phase-1 FlashAttention-4 max sequence lengths must be positive")

    cu_q = cu_seqlens_q.to(device=query.device, dtype=torch.int32).contiguous()
    cu_k = cu_seqlens_k.to(device=query.device, dtype=torch.int32).contiguous()
    if cu_q.ndim != 1 or cu_k.ndim != 1 or cu_q.shape != cu_k.shape or cu_q.numel() < 2:
        raise ValueError("SSA Phase-1 FlashAttention-4 requires matching one-dimensional Q/K cu_seqlens")
    valid = query_valid.to(device=query.device, dtype=torch.bool).reshape(-1)
    if valid.shape != (query.size(0),):
        raise ValueError("SSA Phase-1 FlashAttention-4 query_valid must have shape [Tq]")
    result = _flash_attn4_varlen()(
        query.contiguous(),
        key[:, None, :].contiguous(),
        value[:, None, :].contiguous(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        softmax_scale=float(softmax_scale),
        causal=True,
        window_size=(-1, -1),
        return_lse=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("FlashAttention-4 did not return the required (output, LSE) pair")
    output, lse = result
    if output.shape != (query.size(0), query.size(1), value.size(-1)):
        raise RuntimeError(f"FlashAttention-4 returned unexpected output shape {tuple(output.shape)}")
    if lse.shape == (query.size(1), query.size(0)):
        lse = lse.transpose(0, 1)
    elif lse.shape != (query.size(0), query.size(1)):
        raise RuntimeError(f"FlashAttention-4 returned unexpected LSE shape {tuple(lse.shape)}")
    if lse.dtype != torch.float32:
        raise RuntimeError(f"FlashAttention-4 returned {lse.dtype} LSE; FP32 is required")

    # FlashAttention owns both returned tensors. Mask in place so physical THD
    # padding does not add another Hq*Tq*Dv allocation to the teacher path.
    output.masked_fill_(~valid[:, None, None], 0.0)
    lse.masked_fill_(~valid[:, None], 0.0)
    lse = lse.contiguous()
    return output, lse


def _canonicalize_inputs(
    query: torch.Tensor, key: torch.Tensor, q_idx: torch.Tensor, k_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalize B=1 SBHD or packed THD inputs to compact row-major views."""
    if query.ndim == 4:
        if query.size(1) != 1:
            raise ValueError("SSA Phase 1 supports only batch size one for SBHD input")
        query = query[:, 0]
    elif query.ndim != 3:
        raise ValueError("SSA Phase-1 query must be [Tq,Hq,D] or [Tq,1,Hq,D]")

    if key.ndim == 4:
        if key.shape[1:3] != (1, 1):
            raise ValueError("SSA Phase 1 requires one MQA key head")
        key = key[:, 0, 0]
    elif key.ndim == 3 and key.size(1) == 1:
        key = key[:, 0]
    else:
        raise ValueError("SSA Phase-1 key must be [Tk,1,D] or [Tk,1,1,D]")

    if q_idx.ndim == 4 and q_idx.shape[1:3] == (1, 1):
        q_idx = q_idx[:, 0, 0]
    elif q_idx.ndim == 3 and q_idx.size(1) == 1:
        q_idx = q_idx[:, 0]
    elif q_idx.ndim != 2:
        raise ValueError("SSA Phase-1 indexer Q must contain exactly one head")

    if k_idx.ndim == 4 and k_idx.shape[1:3] == (1, 1):
        k_idx = k_idx[:, 0, 0]
    elif k_idx.ndim == 3 and k_idx.size(1) == 1:
        k_idx = k_idx[:, 0]
    elif k_idx.ndim != 2:
        raise ValueError("SSA Phase-1 indexer K must contain exactly one head")

    if query.size(0) != q_idx.size(0):
        raise ValueError("SSA Phase-1 Q and indexer-Q row counts must match")
    if key.size(0) != k_idx.size(0):
        raise ValueError("SSA Phase-1 K and indexer-K row counts must match")
    if query.size(-1) != key.size(-1):
        raise ValueError("SSA Phase-1 teacher Q/K head dimensions must match")
    if q_idx.size(-1) != k_idx.size(-1):
        raise ValueError("SSA Phase-1 student Q/K head dimensions must match")
    tensors = (query, key, q_idx, k_idx)
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("SSA Phase-1 inputs must be on one device")
    return query, key, q_idx, k_idx


def _accumulation_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Use FP32 for mixed-precision training and retain FP64 test oracles."""
    return torch.float64 if any(tensor.dtype == torch.float64 for tensor in tensors) else torch.float32


def _legal_mask(starts: torch.Tensor, ends: torch.Tensor, query_valid: torch.Tensor, k_start: int, k_end: int) -> torch.Tensor:
    key_positions = torch.arange(k_start, k_end, device=starts.device, dtype=starts.dtype)
    return query_valid[:, None] & (key_positions >= starts[:, None]) & (key_positions < ends[:, None])


def _logits_block(
    query: torch.Tensor,
    key: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    query_valid: torch.Tensor,
    k_start: int,
    k_end: int,
    teacher_scale: float,
    student_scale: float,
    accumulation_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    legal = _legal_mask(starts, ends, query_valid, k_start, k_end)
    teacher_logits = torch.matmul(query.to(dtype=accumulation_dtype), key[k_start:k_end].to(dtype=accumulation_dtype).t()) * teacher_scale
    student_logits = (
        torch.matmul(
            q_idx.to(dtype=accumulation_dtype),
            k_idx[k_start:k_end].to(dtype=accumulation_dtype).t(),
        )
        * student_scale
    )
    teacher_logits.masked_fill_(~legal[:, None, :], float("-inf"))
    student_logits.masked_fill_(~legal, float("-inf"))
    return teacher_logits, student_logits, legal


def _softmax_log_normalizers(
    query: torch.Tensor,
    key: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    query_valid: torch.Tensor,
    key_chunk_size: int,
    teacher_scale: float,
    student_scale: float,
    accumulation_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_lse = torch.full((query.size(0), query.size(1)), float("-inf"), dtype=accumulation_dtype, device=query.device)
    student_lse = torch.full((query.size(0),), float("-inf"), dtype=accumulation_dtype, device=query.device)
    for k_start in range(0, key.size(0), key_chunk_size):
        k_end = min(k_start + key_chunk_size, key.size(0))
        teacher_logits, student_logits, _ = _logits_block(
            query,
            key,
            q_idx,
            k_idx,
            starts,
            ends,
            query_valid,
            k_start,
            k_end,
            teacher_scale,
            student_scale,
            accumulation_dtype,
        )
        teacher_lse = torch.logaddexp(teacher_lse, torch.logsumexp(teacher_logits, dim=-1))
        student_lse = torch.logaddexp(student_lse, torch.logsumexp(student_logits, dim=-1))
    return teacher_lse, student_lse


def _probabilities_block(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    legal: torch.Tensor,
    teacher_lse: torch.Tensor,
    student_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_shift = torch.where(
        legal[:, None, :],
        teacher_logits - teacher_lse[:, :, None],
        torch.zeros_like(teacher_logits),
    )
    teacher = torch.exp(teacher_shift) * legal[:, None, :]
    teacher = teacher.mean(dim=1)
    student_shift = torch.where(legal, student_logits - student_lse[:, None], torch.zeros_like(student_logits))
    student = torch.exp(student_shift) * legal
    return teacher, student, student_shift


def _dense_kl_forward_and_grads(
    query: torch.Tensor,
    key: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    query_valid: torch.Tensor,
    denominator: torch.Tensor,
    teacher_scale: float,
    student_scale: float,
    loss_coeff: float,
    query_chunk_size: int,
    key_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the KL and its indexer gradients in the same two streaming passes.

    The first pass computes the teacher and student normalizers.  The second
    pass already reconstructs both distributions for the scalar KL, so it also
    accumulates ``dQ_idx`` and ``dK_idx``.  Saving those O(T) gradients avoids
    repeating both quadratic passes from custom autograd backward.
    """
    accumulation_dtype = _accumulation_dtype(query, key, q_idx, k_idx)
    total_kl = torch.zeros((), dtype=accumulation_dtype, device=query.device)
    grad_q_idx = torch.zeros_like(q_idx, dtype=accumulation_dtype)
    grad_k_idx = torch.zeros_like(k_idx, dtype=accumulation_dtype)
    loss_scale = torch.as_tensor(loss_coeff, dtype=accumulation_dtype, device=query.device) / denominator.to(dtype=accumulation_dtype).reshape(())
    for q_start in range(0, query.size(0), query_chunk_size):
        q_end = min(q_start + query_chunk_size, query.size(0))
        query_tile = query[q_start:q_end]
        q_idx_tile = q_idx[q_start:q_end]
        starts_tile = starts[q_start:q_end]
        ends_tile = ends[q_start:q_end]
        valid_tile = query_valid[q_start:q_end]
        teacher_lse, student_lse = _softmax_log_normalizers(
            query_tile,
            key,
            q_idx_tile,
            k_idx,
            starts_tile,
            ends_tile,
            valid_tile,
            key_chunk_size,
            teacher_scale,
            student_scale,
            accumulation_dtype,
        )
        for k_start in range(0, key.size(0), key_chunk_size):
            k_end = min(k_start + key_chunk_size, key.size(0))
            teacher_logits, student_logits, legal = _logits_block(
                query_tile,
                key,
                q_idx_tile,
                k_idx,
                starts_tile,
                ends_tile,
                valid_tile,
                k_start,
                k_end,
                teacher_scale,
                student_scale,
                accumulation_dtype,
            )
            teacher, student, student_log_prob = _probabilities_block(teacher_logits, student_logits, legal, teacher_lse, student_lse)
            positive_teacher = legal & (teacher > 0)
            teacher_log_prob = torch.where(positive_teacher, torch.log(teacher), torch.zeros_like(teacher))
            kl = teacher * (teacher_log_prob - student_log_prob)
            total_kl = total_kl + torch.where(positive_teacher, kl, 0.0).sum()
            grad_scores = (student - teacher) * loss_scale
            grad_q_idx[q_start:q_end].add_(torch.matmul(grad_scores, k_idx[k_start:k_end].to(dtype=accumulation_dtype)) * student_scale)
            grad_k_idx[k_start:k_end].add_(torch.matmul(grad_scores.t(), q_idx_tile.to(dtype=accumulation_dtype)) * student_scale)
    # Preserve the original scalar-loss operation order; ``loss_scale`` above
    # is used only for the precomputed gradients.
    loss = total_kl * loss_coeff / denominator.to(dtype=accumulation_dtype).reshape(())
    return loss, grad_q_idx, grad_k_idx


class _SSAPhase1DenseIndexerKL(torch.autograd.Function):
    """Return precomputed full-support KL gradients only for indexer Q/K."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        q_idx,
        k_idx,
        starts,
        ends,
        query_valid,
        denominator,
        teacher_scale,
        student_scale,
        loss_coeff,
        query_chunk_size,
        key_chunk_size,
    ):
        with torch.no_grad():
            loss, grad_q_idx, grad_k_idx = _dense_kl_forward_and_grads(
                query,
                key,
                q_idx,
                k_idx,
                starts,
                ends,
                query_valid,
                denominator,
                teacher_scale,
                student_scale,
                loss_coeff,
                query_chunk_size,
                key_chunk_size,
            )
        ctx.save_for_backward(grad_q_idx, grad_k_idx)
        ctx.q_idx_dtype = q_idx.dtype
        ctx.k_idx_dtype = k_idx.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        grad_q_idx, grad_k_idx = ctx.saved_tensors
        need_q = ctx.needs_input_grad[2]
        need_k = ctx.needs_input_grad[3]
        if grad_loss is None or not (need_q or need_k):
            return (None,) * 13
        grad_loss = grad_loss.to(dtype=grad_q_idx.dtype)
        return (
            None,
            None,
            # PyTorch requires a custom Function gradient to match its input
            # dtype and will cast an FP32 return at this boundary back to BF16.
            # Accumulate above in FP32, then make that unavoidable linear-input
            # boundary explicit. Retaining FP32 through indexer WGRAD would
            # require fusing/manual accumulation in this Function.
            None if not need_q else (grad_q_idx * grad_loss).to(dtype=ctx.q_idx_dtype),
            None if not need_k else (grad_k_idx * grad_loss).to(dtype=ctx.k_idx_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _validate_phase1_kf_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    teacher_lse: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    query_valid: torch.Tensor,
    denominator: torch.Tensor,
    teacher_scale: float,
    student_scale: float,
) -> None:
    """Validate the exact exported Kernel Factory ABI."""
    query_length, query_heads, teacher_dim = query.shape
    key_length = key.size(0)
    if query_heads not in _KF_SUPPORTED_QUERY_HEADS or teacher_dim != _KF_TEACHER_DIM:
        raise NotImplementedError("SSA Phase-1 KF requires query [Tq,Hq,256] with " f"Hq in {_KF_SUPPORTED_QUERY_HEADS}, got {tuple(query.shape)}")
    if key.shape != (key_length, _KF_TEACHER_DIM):
        raise NotImplementedError(f"SSA Phase-1 KF requires canonical key [Tk,256], got {tuple(key.shape)}")
    if q_idx.shape != (query_length, _KF_INDEXER_DIM):
        raise NotImplementedError(f"SSA Phase-1 KF requires canonical q_idx [Tq,128], got {tuple(q_idx.shape)}")
    if k_idx.shape != (key_length, _KF_INDEXER_DIM):
        raise NotImplementedError(f"SSA Phase-1 KF requires canonical k_idx [Tk,128], got {tuple(k_idx.shape)}")
    if teacher_lse.shape != (query_length, query_heads):
        raise ValueError("SSA Phase-1 KF teacher_lse must be [Tq,Hq], got " f"{tuple(teacher_lse.shape)} for Tq={query_length}, Hq={query_heads}")
    if starts.shape != (query_length,) or ends.shape != (query_length,):
        raise ValueError("SSA Phase-1 KF key_starts/key_ends must have shape [Tq]")
    if query_valid.shape != (query_length,):
        raise ValueError("SSA Phase-1 KF query_valid must have shape [Tq]")
    if denominator.shape != ():
        raise ValueError("SSA Phase-1 KF loss_denominator must be a scalar")

    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, q_idx, k_idx)):
        raise NotImplementedError("SSA Phase-1 KF requires BF16 q/k/q_idx/k_idx")
    if teacher_lse.dtype != torch.float32:
        raise NotImplementedError("SSA Phase-1 KF requires FP32 teacher_lse")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise ValueError("SSA Phase-1 KF requires int32 key_starts/key_ends")
    if query_valid.dtype != torch.bool:
        raise ValueError("SSA Phase-1 KF requires bool query_valid")
    if denominator.dtype != torch.float32:
        raise ValueError("SSA Phase-1 KF requires an FP32 loss_denominator")

    tensors = (key, teacher_lse, q_idx, k_idx, starts, ends, query_valid, denominator)
    if not query.is_cuda or any(tensor.device != query.device for tensor in tensors):
        raise NotImplementedError("SSA Phase-1 KF requires all inputs on one CUDA device")
    if torch.cuda.get_device_capability(query.device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("SSA Phase-1 KF supports SM100 and SM103 only; got capability " f"{torch.cuda.get_device_capability(query.device)}")

    expected_teacher_scale = 1.0 / math.sqrt(_KF_TEACHER_DIM)
    expected_student_scale = 1.0 / math.sqrt(_KF_INDEXER_DIM)
    if not math.isclose(teacher_scale, expected_teacher_scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise NotImplementedError("SSA Phase-1 KF hard-codes teacher_softmax_scale=" f"{expected_teacher_scale}; got {teacher_scale}")
    if not math.isclose(student_scale, expected_student_scale, rel_tol=0.0, abs_tol=1.0e-12):
        raise NotImplementedError("SSA Phase-1 KF hard-codes student_softmax_scale=" f"{expected_student_scale}; got {student_scale}")

    int32_max = 2**31 - 1
    query_span = (query_length - 1) * query.stride(0) + query_heads * _KF_TEACHER_DIM
    key_span = (key_length - 1) * key.stride(0) + _KF_TEACHER_DIM
    query_layout_ok = (
        query.stride(2) == 1
        and query.stride(1) == _KF_TEACHER_DIM
        and query.stride(0) >= query_heads * _KF_TEACHER_DIM
        and query.stride(0) % 8 == 0
        and query_span <= int32_max
        and query.data_ptr() % 16 == 0
    )
    key_layout_ok = key.stride(1) == 1 and key.stride(0) >= _KF_TEACHER_DIM and key.stride(0) % 8 == 0 and key_span <= int32_max and key.data_ptr() % 16 == 0
    if not query_layout_ok or not key_layout_ok:
        raise NotImplementedError(
            "SSA Phase-1 KF requires pitched, compact-inner, 16-byte-aligned q/k rows; " f"got q stride={query.stride()}, k stride={key.stride()}"
        )
    if not q_idx.is_contiguous() or not k_idx.is_contiguous():
        raise ValueError("SSA Phase-1 KF requires compact q_idx/k_idx")
    if not teacher_lse.is_contiguous():
        raise ValueError("SSA Phase-1 KF requires contiguous teacher_lse")
    if not all(tensor.is_contiguous() for tensor in (starts, ends, query_valid, denominator)):
        raise ValueError("SSA Phase-1 KF requires contiguous row metadata and denominator")
    if q_idx.data_ptr() % 16 != 0 or k_idx.data_ptr() % 16 != 0:
        raise ValueError("SSA Phase-1 KF requires 16-byte-aligned q_idx/k_idx")


class _SSAPhase1KFIndexerKL(torch.autograd.Function):
    """Apply the coefficient to the KF unit loss and saved FP32 unit gradients."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        q_idx,
        k_idx,
        teacher_lse,
        starts,
        ends,
        query_valid,
        denominator,
        loss_coeff,
    ):
        unit_loss = torch.empty((), dtype=torch.float32, device=query.device)
        unit_grad_q = torch.empty(q_idx.shape, dtype=torch.float32, device=q_idx.device)
        unit_grad_k = torch.empty(k_idx.shape, dtype=torch.float32, device=k_idx.device)
        torch.cuda.nvtx.range_push("ssa.phase1.dense_indexer_kl.kf")
        try:
            # The teacher is frozen; this Function owns indexer autograd.
            _phase1_kf_kernel_module().run(
                query.detach(),
                key.unsqueeze(1).detach(),
                teacher_lse.detach(),
                q_idx.detach(),
                k_idx.detach(),
                starts.detach(),
                ends.detach(),
                query_valid.detach(),
                denominator.detach(),
                unit_loss,
                unit_grad_q,
                unit_grad_k,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        ctx.save_for_backward(unit_grad_q, unit_grad_k)
        ctx.q_idx_dtype = q_idx.dtype
        ctx.k_idx_dtype = k_idx.dtype
        ctx.loss_coeff = float(loss_coeff)
        return unit_loss * ctx.loss_coeff

    @staticmethod
    def backward(ctx, grad_loss):
        unit_grad_q, unit_grad_k = ctx.saved_tensors
        need_q = ctx.needs_input_grad[2]
        need_k = ctx.needs_input_grad[3]
        if grad_loss is None or not (need_q or need_k):
            return (None,) * 10
        scale = grad_loss.to(dtype=torch.float32) * ctx.loss_coeff
        return (
            None,
            None,
            None if not need_q else (unit_grad_q * scale).to(dtype=ctx.q_idx_dtype),
            None if not need_k else (unit_grad_k * scale).to(dtype=ctx.k_idx_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dense_indexer_kl(
    query: torch.Tensor,
    key: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    query_valid: torch.Tensor,
    loss_denominator: torch.Tensor,
    teacher_softmax_scale: float,
    student_softmax_scale: float,
    loss_coeff: float,
    query_chunk_size: Optional[int] = None,
    key_chunk_size: Optional[int] = None,
    teacher_lse: Optional[torch.Tensor] = None,
    use_kf: bool = False,
) -> torch.Tensor:
    """Evaluate the packed Phase-1 teacher/student KL without dense materialization.

    Args:
        query: Teacher query containing every head, [Tq,Hq,D] or [Tq,1,Hq,D].
        key: Globally ordered MQA teacher key, [Tk,1,D] or B=1 SBHD.
        q_idx: CP-local one-head indexer query.
        k_idx: Globally ordered one-head indexer key.
        key_starts: Inclusive global key bound for every local query row.
        key_ends: Exclusive global key bound for every local query row.
        query_valid: False for physical packed-padding query rows.
        loss_denominator: Global real-query count (or one in per-token-loss mode).
        teacher_softmax_scale: Main-attention QK scale.
        student_softmax_scale: Indexer QK scale.
        loss_coeff: Indexer KL coefficient.
        query_chunk_size: Maximum local query rows materialized at once.
        key_chunk_size: Maximum global key rows materialized at once.
        teacher_lse: Natural-log dense-teacher LSE from the paired attention
            call. Required only when ``use_kf`` is true.
        use_kf: Select the fused SM100/SM103 CuTe kernel. Requires BF16
            inputs, Hq in {16,32,96}, D=256, and indexer dimension 128.
            False selects the chunked PyTorch implementation.

    Returns:
        Scalar local KL contribution normalized by the global denominator.

    Autograd returns gradients only for indexer Q/K. The teacher and its
    LSE are detached. Both paths allocate scratch and precompute gradients
    during forward. External CUDA graph capture and higher-order gradients
    are unsupported. Atomic reductions in the fused path can vary by run.
    """
    if query.is_cuda:
        with torch.cuda.device(query.device):
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("dense_indexer_kl does not support external CUDA graph capture.")
    query, key, q_idx, k_idx = _canonicalize_inputs(query, key, q_idx, k_idx)
    query_length = query.size(0)
    key_length = key.size(0)
    if query_length == 0 or key_length == 0:
        raise ValueError("SSA Phase-1 requires nonempty Q and K streams")
    metadata_dtype = torch.int32 if use_kf else torch.int64
    starts = key_starts.to(device=query.device, dtype=metadata_dtype).reshape(-1).contiguous()
    ends = key_ends.to(device=query.device, dtype=metadata_dtype).reshape(-1).contiguous()
    valid = query_valid.to(device=query.device, dtype=torch.bool).reshape(-1).contiguous()
    if starts.shape != (query_length,) or ends.shape != (query_length,):
        raise ValueError("SSA Phase-1 key_starts/key_ends must have shape [Tq]")
    if valid.shape != (query_length,):
        raise ValueError("SSA Phase-1 query_valid must have shape [Tq]")
    denominator_dtype = torch.float32 if use_kf else loss_denominator.dtype
    denominator = loss_denominator.to(device=query.device, dtype=denominator_dtype).reshape(1)
    if query.device.type == "cpu":
        if not bool(((starts >= 0) & (starts <= ends) & (ends <= key_length)).all()):
            raise ValueError("SSA Phase-1 received out-of-range key bounds")
        if not bool(((~valid) | (ends > starts)).all()):
            raise ValueError("every valid query row must have non-empty key support")
        if not bool((denominator > 0).all()):
            raise ValueError("SSA Phase-1 loss denominator must be positive")
    query_chunk_size = min(query_length, 256 if query_chunk_size is None else query_chunk_size)
    key_chunk_size = min(key_length, 1024 if key_chunk_size is None else key_chunk_size)
    if query_chunk_size <= 0 or key_chunk_size <= 0:
        raise ValueError("SSA Phase-1 chunk sizes must be positive")
    if use_kf:
        if teacher_lse is None:
            raise ValueError("SSA Phase-1 KF requires teacher_lse returned by the paired dense attention")
        q_idx = q_idx.contiguous()
        k_idx = k_idx.contiguous()
        denominator = denominator.reshape(())
        _validate_phase1_kf_inputs(
            query,
            key,
            teacher_lse,
            q_idx,
            k_idx,
            starts,
            ends,
            valid,
            denominator,
            float(teacher_softmax_scale),
            float(student_softmax_scale),
        )
        with torch.cuda.device(query.device):
            return _SSAPhase1KFIndexerKL.apply(
                query,
                key,
                q_idx,
                k_idx,
                teacher_lse,
                starts,
                ends,
                valid,
                denominator,
                float(loss_coeff),
            )
    return _SSAPhase1DenseIndexerKL.apply(
        query,
        key,
        q_idx,
        k_idx,
        starts,
        ends,
        valid,
        denominator,
        float(teacher_softmax_scale),
        float(student_softmax_scale),
        float(loss_coeff),
        query_chunk_size,
        key_chunk_size,
    )
