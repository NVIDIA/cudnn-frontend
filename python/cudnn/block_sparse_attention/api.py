# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public wrappers for the CuTe DSL block-sparse attention kernels."""

from __future__ import annotations

from typing import Optional

import torch

from cudnn.api_base import TupleDict


def _validate_layout(layout: str) -> None:
    if layout not in {"bhsd", "bshd"}:
        raise ValueError(f"layout must be 'bhsd' or 'bshd', got {layout!r}")


def _canonical_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str):
    _validate_layout(layout)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all be rank-4 tensors")
    if layout == "bhsd":
        b, h_q, s_q, d_qk = q.shape
        b_k, h_kv, s_k, d_k = k.shape
        b_v, h_v, s_v, d_v = v.shape
    else:
        b, s_q, h_q, d_qk = q.shape
        b_k, s_k, h_kv, d_k = k.shape
        b_v, s_v, h_v, d_v = v.shape
    if min(b, h_q, h_kv, s_q, s_k, d_qk, d_v) < 1:
        raise ValueError("q, k, and v dimensions must all be positive")
    if (b_k, b_v) != (b, b):
        raise ValueError("q, k, and v batch dimensions must match")
    if (h_v, s_v) != (h_kv, s_k):
        raise ValueError("k and v head and sequence dimensions must match")
    if d_k != d_qk:
        raise ValueError("q and k head dimensions must match")
    if h_q % h_kv != 0:
        raise ValueError("the number of query heads must be divisible by the number of KV heads")
    if q.dtype not in {torch.float16, torch.bfloat16} or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same float16 or bfloat16 dtype")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("q, k, and v must have a contiguous head dimension")
    if not all(t.is_cuda for t in (q, k, v)):
        raise RuntimeError("block sparse attention requires CUDA tensors")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("q, k, and v must be on the same CUDA device")
    return b, h_q, h_kv, s_q, s_k, d_qk, d_v


def _validate_sparse_metadata(
    q2k_block_index: torch.Tensor,
    q2k_block_nums: Optional[torch.Tensor],
    block_sizes: Optional[torch.Tensor],
    *,
    expected_prefix: tuple[int, int, int],
    num_kv_blocks: int,
    device: torch.device,
    allowed_block_size_ranks: tuple[int, ...],
) -> None:
    if q2k_block_index.ndim != 4 or q2k_block_index.dtype != torch.int32:
        raise ValueError("q2k_block_index must be a rank-4 int32 tensor")
    if tuple(q2k_block_index.shape[:3]) != expected_prefix:
        raise ValueError("q2k_block_index shape prefix must be " f"{expected_prefix}, got {tuple(q2k_block_index.shape[:3])}")
    if q2k_block_index.shape[-1] < 1:
        raise ValueError("q2k_block_index must have a non-empty KV-block capacity")
    if not q2k_block_index.is_cuda or q2k_block_index.device != device:
        raise ValueError("q2k_block_index must be on the same CUDA device as q")
    if q2k_block_nums is not None:
        if q2k_block_nums.ndim != 3 or q2k_block_nums.dtype != torch.int32:
            raise ValueError("q2k_block_nums must be a rank-3 int32 tensor")
        if tuple(q2k_block_nums.shape) != expected_prefix:
            raise ValueError(f"q2k_block_nums shape must be {expected_prefix}, got {tuple(q2k_block_nums.shape)}")
        if not q2k_block_nums.is_cuda or q2k_block_nums.device != device:
            raise ValueError("q2k_block_nums must be on the same CUDA device as q")
    if block_sizes is not None:
        if block_sizes.dtype != torch.int32:
            raise ValueError("block_sizes must have dtype int32")
        if block_sizes.ndim not in allowed_block_size_ranks:
            raise ValueError(f"block_sizes rank must be one of {allowed_block_size_ranks}, got {block_sizes.ndim}")
        expected_shapes = {
            1: (num_kv_blocks,),
            2: (expected_prefix[0], num_kv_blocks),
            3: (expected_prefix[0], expected_prefix[1], num_kv_blocks),
        }
        if tuple(block_sizes.shape) != expected_shapes[block_sizes.ndim]:
            raise ValueError(f"block_sizes shape must be {expected_shapes[block_sizes.ndim]}, " f"got {tuple(block_sizes.shape)}")
        if not block_sizes.is_cuda or block_sizes.device != device:
            raise ValueError("block_sizes must be on the same CUDA device as q")


def _device_arch(tensor: torch.Tensor) -> int:
    major, minor = torch.cuda.get_device_capability(tensor.device)
    return major * 10 + minor


def _validate_fixed_block_count(
    block_sparse_num: int,
    capacity: int,
    *,
    require_even: bool,
) -> None:
    minimum = 2 if require_even else 1
    if block_sparse_num < minimum or block_sparse_num > capacity:
        raise ValueError(f"block_sparse_num must be in [{minimum}, {capacity}], got {block_sparse_num}")
    if require_even and block_sparse_num % 2:
        raise ValueError("block_sparse_num must be even for this backend")


def _validate_backward_tensors(
    do_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    dq_tensor: Optional[torch.Tensor],
    dk_tensor: Optional[torch.Tensor],
    dv_tensor: Optional[torch.Tensor],
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
) -> None:
    if do_tensor.shape != o_tensor.shape or o_tensor.shape != q_tensor.shape:
        raise ValueError("do_tensor, o_tensor, and q_tensor must have identical shapes")
    if do_tensor.dtype != q_tensor.dtype or o_tensor.dtype != q_tensor.dtype:
        raise ValueError("do_tensor and o_tensor must have the same dtype as q_tensor")
    if lse_tensor.dtype != torch.float32:
        raise ValueError("lse_tensor must have dtype float32")
    tensors = (do_tensor, o_tensor, lse_tensor)
    if not all(t.is_cuda and t.device == q_tensor.device for t in tensors):
        raise ValueError("do_tensor, o_tensor, and lse_tensor must be on the same CUDA device as q")
    if do_tensor.stride(-1) != 1 or o_tensor.stride(-1) != 1:
        raise ValueError("do_tensor and o_tensor must have a contiguous head dimension")
    expected_outputs = (
        (dq_tensor, q_tensor, "dq_tensor"),
        (dk_tensor, k_tensor, "dk_tensor"),
        (dv_tensor, v_tensor, "dv_tensor"),
    )
    for output, reference, name in expected_outputs:
        if output is None:
            continue
        if output.shape != reference.shape or output.dtype != reference.dtype:
            raise ValueError(f"{name} must match the corresponding input shape and dtype")
        if not output.is_cuda or output.device != q_tensor.device or output.stride(-1) != 1:
            raise ValueError(f"{name} must be on the same CUDA device as q with a contiguous head dimension")


def block_sparse_attention_forward(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: Optional[int] = None,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    *,
    sparse_block_size: Optional[int] = None,
    allow_empty_block_nums: bool = False,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    layout: str = "bhsd",
    kv_splits: int | str = 1,
    use_clc: Optional[bool] = None,
) -> TupleDict:
    """Run non-causal block-sparse scaled dot-product attention.

    The sparse pattern is supplied as a list of KV block ids for every query
    block.  This wrapper only dispatches to Python CuTe DSL kernels; the former
    SM100 C++/AOT extension is intentionally not part of this package.

    Sparse metadata values are a caller contract; see the "Sparse metadata"
    section of ``docs/fe-oss-apis/bsa.md`` for the required value ranges.
    """

    batch, num_q_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim, value_dim = _canonical_shapes(q_tensor, k_tensor, v_tensor, layout)
    arch = _device_arch(q_tensor)
    arch_family = arch // 10
    if arch_family not in {9, 10, 11, 12}:
        raise RuntimeError(f"block sparse attention requires SM90-SM120, found SM{arch}")

    if sparse_block_size is None:
        sparse_block_size = 64 if arch_family in {9, 12} else 128
    if sparse_block_size not in {64, 128}:
        raise ValueError("sparse_block_size must be 64 or 128")
    if arch_family == 9:
        if isinstance(kv_splits, str) or not 1 <= int(kv_splits) <= 256:
            raise ValueError("SM90 kv_splits must be an integer in [1, 256]")

    if arch_family in {9, 12} and sparse_block_size != 64:
        raise NotImplementedError(f"SM{arch} only provides a blk64 forward path")
    if arch_family == 9:
        if head_dim not in {64, 96, 128} or value_dim not in {64, 96, 128}:
            raise NotImplementedError("SM90 forward supports QK and V dimensions 64, 96, or 128")
        if seqlen_q % 64:
            raise NotImplementedError("SM90 forward requires seqlen_q to be a multiple of 64")
    elif arch_family == 12:
        if head_dim != 128 or value_dim != 128:
            raise NotImplementedError("SM120 forward requires QK and V dimensions of 128")
    elif sparse_block_size == 64:
        if q_tensor.dtype != torch.bfloat16 or head_dim != 128 or value_dim != 128:
            raise NotImplementedError("SM100/SM110 blk64 forward requires BF16 and QK=V=128")
        if num_q_heads != num_kv_heads:
            raise NotImplementedError("SM100/SM110 blk64 forward supports MHA only")
    elif (head_dim, value_dim) not in {(64, 64), (96, 96), (128, 128)}:
        raise NotImplementedError("SM100/SM110 blk128 forward supports (QK, V) dimensions " f"(64, 64), (96, 96), or (128, 128); got ({head_dim}, {value_dim})")

    gqa_ratio = num_q_heads // num_kv_heads
    if pack_gqa is True and not (arch_family in {10, 11} and sparse_block_size == 128):
        raise NotImplementedError("pack_gqa is available only on the SM100/SM110 blk128 path")
    if pack_gqa is True and 128 % gqa_ratio:
        raise ValueError("pack_gqa=True requires the GQA ratio to divide 128")
    pack_gqa_effective = (
        arch_family in {10, 11} and sparse_block_size == 128 and (gqa_ratio > 1 if pack_gqa is None else bool(pack_gqa)) and 128 % gqa_ratio == 0
    )
    if pack_gqa_effective:
        metadata_heads = num_kv_heads
        metadata_q_blocks = (seqlen_q * gqa_ratio + sparse_block_size - 1) // sparse_block_size
    else:
        metadata_heads = num_q_heads
        metadata_q_blocks = (seqlen_q + sparse_block_size - 1) // sparse_block_size
    expected_prefix = (batch, metadata_heads, metadata_q_blocks)
    num_kv_blocks = (seqlen_k + sparse_block_size - 1) // sparse_block_size
    allowed_block_size_ranks = (1, 2, 3) if arch_family in {9, 12} else (1,)
    _validate_sparse_metadata(
        q2k_block_index,
        q2k_block_nums,
        block_sizes,
        expected_prefix=expected_prefix,
        num_kv_blocks=num_kv_blocks,
        device=q_tensor.device,
        allowed_block_size_ranks=allowed_block_size_ranks,
    )

    if block_sparse_num is None:
        block_sparse_num = int(q2k_block_index.shape[-1])
    if q2k_block_nums is None:
        require_even = arch_family in {10, 11} and sparse_block_size == 128
        _validate_fixed_block_count(
            block_sparse_num,
            q2k_block_index.shape[-1],
            require_even=require_even,
        )

    with torch.cuda.device(q_tensor.device):
        from . import _interface

        if sparse_block_size == 64 and arch_family in {10, 11}:
            out, lse = _interface.bsa_attn_fwd_blk64_cutedsl(
                q_tensor,
                k_tensor,
                v_tensor,
                q2k_block_index,
                block_sizes,
                q2k_block_nums=q2k_block_nums,
                softmax_scale=softmax_scale,
                layout=layout,
                block_sparse_num=block_sparse_num,
                allow_empty_block_nums=allow_empty_block_nums,
                use_clc=use_clc,
                kv_splits=kv_splits,
            )
        else:
            if kv_splits != 1 and arch_family != 9:
                raise NotImplementedError("kv_splits is currently exposed only by the SM90 and " "SM100/SM110 blk64 CuTe DSL paths")
            if use_clc is not None:
                raise NotImplementedError("use_clc is currently exposed only by the SM100/SM110 blk64 CuTe DSL path")
            out, lse = _interface.bsa_attn_fwd(
                q_tensor,
                k_tensor,
                v_tensor,
                q2k_block_index,
                block_sparse_num,
                block_sizes,
                q2k_block_nums=q2k_block_nums,
                allow_empty_block_nums=allow_empty_block_nums,
                softmax_scale=softmax_scale,
                pack_gqa=pack_gqa,
                return_lse=True,
                layout=layout,
                kv_splits=kv_splits,
            )

    return TupleDict(o_tensor=out, lse_tensor=lse)


def block_sparse_attention_backward(
    do_tensor: torch.Tensor,
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: Optional[int] = None,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    *,
    softmax_scale: Optional[float] = None,
    dq_tensor: Optional[torch.Tensor] = None,
    dk_tensor: Optional[torch.Tensor] = None,
    dv_tensor: Optional[torch.Tensor] = None,
    bucket_size_blocks: Optional[int] = None,
    sparse_block_size: Optional[int] = None,
    layout: str = "bhsd",
) -> TupleDict:
    """Compute explicit dQ, dK, and dV for block-sparse attention.

    Sparse metadata values are a caller contract; see the "Sparse metadata"
    section of ``docs/fe-oss-apis/bsa.md`` for the required value ranges.
    """

    batch, num_q_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim, value_dim = _canonical_shapes(q_tensor, k_tensor, v_tensor, layout)
    arch = _device_arch(q_tensor)
    arch_family = arch // 10
    if arch_family not in {9, 10, 11}:
        raise RuntimeError(f"block sparse attention backward requires SM90-SM110, found SM{arch}")
    if q_tensor.dtype != torch.bfloat16:
        raise NotImplementedError("block sparse attention backward requires BF16")
    if num_q_heads != num_kv_heads:
        raise NotImplementedError("block sparse attention backward supports MHA only")
    if head_dim != value_dim:
        raise NotImplementedError("block sparse attention backward requires equal QK and V dimensions")

    if sparse_block_size is None:
        sparse_block_size = 64 if arch_family == 9 else 128
    if sparse_block_size not in {64, 128}:
        raise ValueError("sparse_block_size must be 64 or 128")
    if arch_family == 9 and sparse_block_size != 64:
        raise NotImplementedError("SM90 backward only provides a blk64 path")
    if sparse_block_size == 64 and head_dim != 128:
        raise NotImplementedError("blk64 backward requires head_dim=128")
    if sparse_block_size == 128 and head_dim not in {64, 128}:
        raise NotImplementedError("SM100/SM110 blk128 backward requires head_dim=64 or 128")
    if sparse_block_size == 128 and block_sizes is not None:
        raise NotImplementedError("SM100/SM110 blk128 backward does not yet support block_sizes; " "use full physical KV blocks and pass block_sizes=None")

    expected_prefix = (
        batch,
        num_q_heads,
        (seqlen_q + sparse_block_size - 1) // sparse_block_size,
    )
    _validate_sparse_metadata(
        q2k_block_index,
        q2k_block_nums,
        block_sizes,
        expected_prefix=expected_prefix,
        num_kv_blocks=(seqlen_k + sparse_block_size - 1) // sparse_block_size,
        device=q_tensor.device,
        allowed_block_size_ranks=(1, 2),
    )
    if block_sparse_num is None:
        block_sparse_num = int(q2k_block_index.shape[-1])
    if q2k_block_nums is None:
        _validate_fixed_block_count(
            block_sparse_num,
            q2k_block_index.shape[-1],
            require_even=sparse_block_size == 128,
        )

    expected_lse_shape = (batch, num_q_heads, seqlen_q)
    if tuple(lse_tensor.shape) != expected_lse_shape:
        raise ValueError(f"lse_tensor shape must be {expected_lse_shape}, got {tuple(lse_tensor.shape)}")
    _validate_backward_tensors(
        do_tensor,
        q_tensor,
        o_tensor,
        lse_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        k_tensor,
        v_tensor,
    )

    with torch.cuda.device(q_tensor.device):
        from . import _interface

        dq, dk, dv = _interface.bsa_attn_bwd(
            do_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            q2k_block_index,
            block_sparse_num,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
            dq=dq_tensor,
            dk=dk_tensor,
            dv=dv_tensor,
            bucket_size_blocks=bucket_size_blocks,
            sparse_block_size=sparse_block_size,
            layout=layout,
        )
    return TupleDict(dq_tensor=dq, dk_tensor=dk, dv_tensor=dv)


__all__ = [
    "block_sparse_attention_forward",
    "block_sparse_attention_backward",
]
