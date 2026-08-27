# SPDX-License-Identifier: Apache-2.0

"""Allocation-free execution adapter for NVFP4 QAT attention backward."""

from __future__ import annotations

from functools import reduce
from operator import mul

import torch
import triton

from ._kernels import attention_backward_dkdv, attention_backward_dq, attention_backward_preprocess
from ._nvfp4 import fake_quantize_kv, fake_quantize_q


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _align_up(value: int, alignment: int = 16) -> int:
    return (value + alignment - 1) // alignment * alignment


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def _carve_tensor(workspace: torch.Tensor, offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    offset = _align_up(offset)
    nbytes = _numel(shape) * dtype.itemsize
    tensor = workspace.narrow(0, offset, nbytes).view(dtype).view(shape)
    return tensor, offset + nbytes


def compile_nvfp4_attention_qat_backward(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    q_strides: tuple[int, ...],
    k_strides: tuple[int, ...],
    *,
    softmax_scale: float,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    dq_num_stages: int,
    dkdv_num_stages: int,
) -> tuple[object, ...]:
    """Compile every Triton specialization without allocating or launching."""
    batch, heads, seqlen_q, head_dim = q_shape
    seqlen_kv = k_shape[2]
    fake_q_strides = _contiguous_strides(q_shape)
    fake_k_strides = _contiguous_strides(k_shape)
    quant_block = 32

    compiled = [
        fake_quantize_q.warmup(
            torch.bfloat16,
            torch.bfloat16,
            *q_strides,
            *fake_q_strides,
            heads,
            seqlen_q,
            block_m=quant_block,
            head_dim=head_dim,
            grid=(triton.cdiv(seqlen_q, quant_block), batch * heads),
            num_warps=4,
            num_stages=2,
        ),
        fake_quantize_kv.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            *k_strides,
            *fake_k_strides,
            heads,
            seqlen_kv,
            block_n=quant_block,
            head_dim=head_dim,
            grid=(triton.cdiv(seqlen_kv, quant_block), batch * heads),
            num_warps=4,
            num_stages=2,
        ),
        attention_backward_preprocess.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            seqlen_q,
            block_m=128,
            head_dim=head_dim,
            grid=(triton.cdiv(seqlen_q, 128), batch * heads),
            num_warps=4,
            num_stages=2,
        ),
    ]

    common_args = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        softmax_scale,
        torch.bfloat16,
    )
    stride_args = (
        q_strides[0],
        k_strides[0],
        q_strides[1],
        k_strides[1],
        q_strides[2],
        k_strides[2],
        q_strides[3],
        k_strides[3],
        heads,
        seqlen_q,
        seqlen_kv,
    )
    constants = {
        "block_m": block_m,
        "block_n": block_n,
        "head_dim": head_dim,
        "causal": is_causal,
        "num_warps": num_warps,
    }
    compiled.append(
        attention_backward_dq.warmup(
            *common_args,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            *stride_args,
            **constants,
            grid=(triton.cdiv(seqlen_q, block_m), batch * heads),
            num_stages=dq_num_stages,
        )
    )
    compiled.append(
        attention_backward_dkdv.warmup(
            *common_args,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            *stride_args,
            **constants,
            grid=(triton.cdiv(seqlen_kv, block_n), batch * heads),
            num_stages=dkdv_num_stages,
        )
    )
    return tuple(compiled)


def run_nvfp4_attention_qat_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    high_precision_o: torch.Tensor,
    grad_o: torch.Tensor,
    lse: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    workspace: torch.Tensor,
    *,
    softmax_scale: float,
    is_causal: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    dq_num_stages: int,
    dkdv_num_stages: int,
) -> None:
    """Launch fake quantization, delta preprocessing, dQ, and dK/dV."""
    batch, heads, seqlen_q, head_dim = q.shape
    seqlen_kv = k.shape[2]

    offset = 0
    fake_q, offset = _carve_tensor(workspace, offset, tuple(q.shape), q.dtype)
    fake_k, offset = _carve_tensor(workspace, offset, tuple(k.shape), k.dtype)
    fake_v, offset = _carve_tensor(workspace, offset, tuple(v.shape), v.dtype)
    delta, _ = _carve_tensor(workspace, offset, tuple(lse.shape), torch.float32)

    quant_block = 32
    q_grid = (triton.cdiv(seqlen_q, quant_block), batch * heads)
    kv_grid = (triton.cdiv(seqlen_kv, quant_block), batch * heads)
    fake_quantize_q[q_grid](
        q,
        fake_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        fake_q.stride(0),
        fake_q.stride(1),
        fake_q.stride(2),
        fake_q.stride(3),
        heads,
        seqlen_q,
        block_m=quant_block,
        head_dim=head_dim,
        num_warps=4,
        num_stages=2,
    )
    fake_quantize_kv[kv_grid](
        k,
        v,
        fake_k,
        fake_v,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        fake_k.stride(0),
        fake_k.stride(1),
        fake_k.stride(2),
        fake_k.stride(3),
        heads,
        seqlen_kv,
        block_n=quant_block,
        head_dim=head_dim,
        num_warps=4,
        num_stages=2,
    )

    preprocess_block = 128
    preprocess_grid = (triton.cdiv(seqlen_q, preprocess_block), batch * heads)
    attention_backward_preprocess[preprocess_grid](
        high_precision_o,
        grad_o,
        delta,
        seqlen_q,
        block_m=preprocess_block,
        head_dim=head_dim,
        num_warps=4,
        num_stages=2,
    )

    common_args = (
        fake_q,
        fake_k,
        fake_v,
        softmax_scale,
        grad_o,
    )
    common_strides = (
        q.stride(0),
        k.stride(0),
        q.stride(1),
        k.stride(1),
        q.stride(2),
        k.stride(2),
        q.stride(3),
        k.stride(3),
        heads,
        seqlen_q,
        seqlen_kv,
    )

    dq_grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    attention_backward_dq[dq_grid](
        *common_args,
        grad_q,
        lse,
        delta,
        *common_strides,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        causal=is_causal,
        num_warps=num_warps,
        num_stages=dq_num_stages,
    )

    dkdv_grid = (triton.cdiv(seqlen_kv, block_n), batch * heads)
    attention_backward_dkdv[dkdv_grid](
        *common_args,
        grad_k,
        grad_v,
        lse,
        delta,
        *common_strides,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        causal=is_causal,
        num_warps=num_warps,
        num_stages=dkdv_num_stages,
    )
