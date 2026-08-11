# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public wrappers for the CuTe DSL block-sparse attention kernels."""

from __future__ import annotations

from typing import Optional

import cutlass

from cudnn.api_base import TupleDict
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import detect_framework, get_device, get_shape, get_strides, is_torch_tensor


def _validate_layout(layout: str) -> None:
    if layout not in {"bhsd", "bshd"}:
        raise ValueError(f"layout must be 'bhsd' or 'bshd', got {layout!r}")


def _is_cuda(tensor) -> bool:
    return get_device(tensor).type == "cuda"


def _same_device(a, b) -> bool:
    return get_device(a) == get_device(b)


def _validate_framework(framework: str, name: str = "block sparse attention") -> None:
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for {name}; pass torch tensors or JAX arrays")


def _canonical_shapes(q, k, v, layout: str):
    _validate_layout(layout)
    q_shape, k_shape, v_shape = get_shape(q), get_shape(k), get_shape(v)
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError("q, k, and v must all be rank-4 tensors")
    if layout == "bhsd":
        b, h_q, s_q, d_qk = q_shape
        b_k, h_kv, s_k, d_k = k_shape
        b_v, h_v, s_v, d_v = v_shape
    else:
        b, s_q, h_q, d_qk = q_shape
        b_k, s_k, h_kv, d_k = k_shape
        b_v, s_v, h_v, d_v = v_shape
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
    q_dtype = _convert_to_cutlass_data_type(q.dtype)
    if q_dtype not in (cutlass.Float16, cutlass.BFloat16) or any(_convert_to_cutlass_data_type(t.dtype) != q_dtype for t in (k, v)):
        raise ValueError("q, k, and v must have the same float16 or bfloat16 dtype")
    if any(get_strides(t)[-1] != 1 for t in (q, k, v)):
        raise ValueError("q, k, and v must have a contiguous head dimension")
    if not all(_is_cuda(t) for t in (q, k, v)):
        raise RuntimeError("block sparse attention requires CUDA tensors")
    if any(not _same_device(t, q) for t in (k, v)):
        raise ValueError("q, k, and v must be on the same CUDA device")
    return b, h_q, h_kv, s_q, s_k, d_qk, d_v


def _validate_sparse_metadata(
    q2k_block_index,
    q2k_block_nums,
    block_sizes,
    *,
    expected_prefix: tuple[int, int, int],
    num_kv_blocks: int,
    reference,
    allowed_block_size_ranks: tuple[int, ...],
) -> None:
    index_shape = get_shape(q2k_block_index)
    if len(index_shape) != 4 or _convert_to_cutlass_data_type(q2k_block_index.dtype) != cutlass.Int32:
        raise ValueError("q2k_block_index must be a rank-4 int32 tensor")
    if index_shape[:3] != expected_prefix:
        raise ValueError("q2k_block_index shape prefix must be " f"{expected_prefix}, got {index_shape[:3]}")
    if index_shape[-1] < 1:
        raise ValueError("q2k_block_index must have a non-empty KV-block capacity")
    if not _is_cuda(q2k_block_index) or not _same_device(q2k_block_index, reference):
        raise ValueError("q2k_block_index must be on the same CUDA device as q")
    if q2k_block_nums is not None:
        nums_shape = get_shape(q2k_block_nums)
        if len(nums_shape) != 3 or _convert_to_cutlass_data_type(q2k_block_nums.dtype) != cutlass.Int32:
            raise ValueError("q2k_block_nums must be a rank-3 int32 tensor")
        if nums_shape != expected_prefix:
            raise ValueError(f"q2k_block_nums shape must be {expected_prefix}, got {nums_shape}")
        if not _is_cuda(q2k_block_nums) or not _same_device(q2k_block_nums, reference):
            raise ValueError("q2k_block_nums must be on the same CUDA device as q")
    if block_sizes is not None:
        sizes_shape = get_shape(block_sizes)
        if _convert_to_cutlass_data_type(block_sizes.dtype) != cutlass.Int32:
            raise ValueError("block_sizes must have dtype int32")
        if len(sizes_shape) not in allowed_block_size_ranks:
            raise ValueError(f"block_sizes rank must be one of {allowed_block_size_ranks}, got {len(sizes_shape)}")
        expected_shapes = {
            1: (num_kv_blocks,),
            2: (expected_prefix[0], num_kv_blocks),
            3: (expected_prefix[0], expected_prefix[1], num_kv_blocks),
        }
        if sizes_shape != expected_shapes[len(sizes_shape)]:
            raise ValueError(f"block_sizes shape must be {expected_shapes[len(sizes_shape)]}, " f"got {sizes_shape}")
        if not _is_cuda(block_sizes) or not _same_device(block_sizes, reference):
            raise ValueError("block_sizes must be on the same CUDA device as q")


def _device_context(tensor):
    """torch: bind the CUDA device of ``tensor`` for the call; JAX: no-op (single-context)."""
    if is_torch_tensor(tensor):
        import torch

        return torch.cuda.device(tensor.device)
    import contextlib

    return contextlib.nullcontext()


def _device_arch(tensor) -> int:
    if is_torch_tensor(tensor):
        import torch

        major, minor = torch.cuda.get_device_capability(tensor.device)
    else:
        from cudnn.tensor_adapter import get_compute_capability

        major, minor = get_compute_capability()
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
    do_tensor,
    q_tensor,
    o_tensor,
    lse_tensor,
    dq_tensor,
    dk_tensor,
    dv_tensor,
    k_tensor,
    v_tensor,
) -> None:
    if get_shape(do_tensor) != get_shape(o_tensor) or get_shape(o_tensor) != get_shape(q_tensor):
        raise ValueError("do_tensor, o_tensor, and q_tensor must have identical shapes")
    q_dtype = _convert_to_cutlass_data_type(q_tensor.dtype)
    if _convert_to_cutlass_data_type(do_tensor.dtype) != q_dtype or _convert_to_cutlass_data_type(o_tensor.dtype) != q_dtype:
        raise ValueError("do_tensor and o_tensor must have the same dtype as q_tensor")
    if _convert_to_cutlass_data_type(lse_tensor.dtype) != cutlass.Float32:
        raise ValueError("lse_tensor must have dtype float32")
    tensors = (do_tensor, o_tensor, lse_tensor)
    if not all(_is_cuda(t) and _same_device(t, q_tensor) for t in tensors):
        raise ValueError("do_tensor, o_tensor, and lse_tensor must be on the same CUDA device as q")
    if get_strides(do_tensor)[-1] != 1 or get_strides(o_tensor)[-1] != 1:
        raise ValueError("do_tensor and o_tensor must have a contiguous head dimension")
    expected_outputs = (
        (dq_tensor, q_tensor, "dq_tensor"),
        (dk_tensor, k_tensor, "dk_tensor"),
        (dv_tensor, v_tensor, "dv_tensor"),
    )
    for output, reference, name in expected_outputs:
        if output is None:
            continue
        if get_shape(output) != get_shape(reference) or _convert_to_cutlass_data_type(output.dtype) != _convert_to_cutlass_data_type(reference.dtype):
            raise ValueError(f"{name} must match the corresponding input shape and dtype")
        if not _is_cuda(output) or not _same_device(output, q_tensor) or get_strides(output)[-1] != 1:
            raise ValueError(f"{name} must be on the same CUDA device as q with a contiguous head dimension")


def block_sparse_attention_forward(
    q_tensor,
    k_tensor,
    v_tensor,
    q2k_block_index,
    block_sparse_num: Optional[int] = None,
    block_sizes=None,
    q2k_block_nums=None,
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

    Tensor parameters are type-erased: torch tensors and JAX arrays are both
    accepted (torch is imported only for torch tensors). JAX arrays are
    supported on the SM100/SM110 blk128 path only; see the "JAX support"
    section of ``docs/fe-oss-apis/bsa.md``.
    """

    framework = detect_framework(q_tensor)
    _validate_framework(framework, "block_sparse_attention_forward")
    batch, num_q_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim, value_dim = _canonical_shapes(q_tensor, k_tensor, v_tensor, layout)
    arch = _device_arch(q_tensor)
    arch_family = arch // 10
    if arch_family not in {9, 10, 11, 12}:
        raise RuntimeError(f"block sparse attention requires SM90-SM120, found SM{arch}")

    if sparse_block_size is None:
        sparse_block_size = 64 if arch_family in {9, 12} else 128
    if sparse_block_size not in {64, 128}:
        raise ValueError("sparse_block_size must be 64 or 128")
    if framework == "jax" and not (arch_family in {10, 11} and sparse_block_size == 128):
        raise ValueError(
            "JAX arrays are supported only on the SM100/SM110 blk128 forward path "
            f"(got SM{arch}, sparse_block_size={sparse_block_size}); pass torch tensors for the other backends"
        )
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
        if _convert_to_cutlass_data_type(q_tensor.dtype) != cutlass.BFloat16 or head_dim != 128 or value_dim != 128:
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
        reference=q_tensor,
        allowed_block_size_ranks=allowed_block_size_ranks,
    )

    if block_sparse_num is None:
        block_sparse_num = int(get_shape(q2k_block_index)[-1])
    if q2k_block_nums is None:
        require_even = arch_family in {10, 11} and sparse_block_size == 128
        _validate_fixed_block_count(
            block_sparse_num,
            get_shape(q2k_block_index)[-1],
            require_even=require_even,
        )

    with _device_context(q_tensor):
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
    do_tensor,
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    lse_tensor,
    q2k_block_index,
    block_sparse_num: Optional[int] = None,
    block_sizes=None,
    q2k_block_nums=None,
    *,
    softmax_scale: Optional[float] = None,
    dq_tensor=None,
    dk_tensor=None,
    dv_tensor=None,
    bucket_size_blocks: Optional[int] = None,
    sparse_block_size: Optional[int] = None,
    layout: str = "bhsd",
) -> TupleDict:
    """Compute explicit dQ, dK, and dV for block-sparse attention.

    Sparse metadata values are a caller contract; see the "Sparse metadata"
    section of ``docs/fe-oss-apis/bsa.md`` for the required value ranges.

    Tensor parameters are type-erased: torch tensors and JAX arrays are both
    accepted (torch is imported only for torch tensors). JAX arrays are
    supported on the SM100/SM110 blk128 path only; see the "JAX support"
    section of ``docs/fe-oss-apis/bsa.md``.
    """

    framework = detect_framework(q_tensor)
    _validate_framework(framework, "block_sparse_attention_backward")
    batch, num_q_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim, value_dim = _canonical_shapes(q_tensor, k_tensor, v_tensor, layout)
    arch = _device_arch(q_tensor)
    arch_family = arch // 10
    if arch_family not in {9, 10, 11}:
        raise RuntimeError(f"block sparse attention backward requires SM90-SM110, found SM{arch}")
    if _convert_to_cutlass_data_type(q_tensor.dtype) != cutlass.BFloat16:
        raise NotImplementedError("block sparse attention backward requires BF16")
    if num_q_heads != num_kv_heads:
        raise NotImplementedError("block sparse attention backward supports MHA only")
    if head_dim != value_dim:
        raise NotImplementedError("block sparse attention backward requires equal QK and V dimensions")

    if sparse_block_size is None:
        sparse_block_size = 64 if arch_family == 9 else 128
    if sparse_block_size not in {64, 128}:
        raise ValueError("sparse_block_size must be 64 or 128")
    if framework == "jax" and not (arch_family in {10, 11} and sparse_block_size == 128):
        raise ValueError(
            "JAX arrays are supported only on the SM100/SM110 blk128 backward path "
            f"(got SM{arch}, sparse_block_size={sparse_block_size}); pass torch tensors for the other backends"
        )
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
        reference=q_tensor,
        allowed_block_size_ranks=(1, 2),
    )
    if block_sparse_num is None:
        block_sparse_num = int(get_shape(q2k_block_index)[-1])
    if q2k_block_nums is None:
        _validate_fixed_block_count(
            block_sparse_num,
            get_shape(q2k_block_index)[-1],
            require_even=sparse_block_size == 128,
        )

    expected_lse_shape = (batch, num_q_heads, seqlen_q)
    if get_shape(lse_tensor) != expected_lse_shape:
        raise ValueError(f"lse_tensor shape must be {expected_lse_shape}, got {get_shape(lse_tensor)}")
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

    with _device_context(q_tensor):
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
