#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Float16, BFloat16

from ._kernels.hstu_fwd import HSTUAttentionForwardSm100
from ._kernels.hstu_bwd import HSTUAttentionBackwardSm100


def _normalize_scaling_seqlen(
    scaling_seqlen: Optional[float],
    max_seqlen_q: int,
) -> float:
    """Resolve the public HSTU sequence-normalization factor."""
    value = float(max_seqlen_q if scaling_seqlen is None else scaling_seqlen)
    if value <= 0.0:
        raise ValueError(f"scaling_seqlen must be positive, got {value}")
    return value


def _mark_dynamic_tensor(
    tensor: torch.Tensor,
    leading_dim: int,
    *,
    compact: bool = False,
    stride_order=(2, 3, 0, 4, 1),
):
    if tensor.data_ptr() % 16 != 0:
        raise ValueError("HSTU CuTe tensor storage must be 16-byte aligned")
    cute_tensor = from_dlpack(
        tensor.detach(), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=leading_dim)
    if compact:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=1,
            stride_order=stride_order,
            divisibility=64,
        )
    return cute_tensor


def _mark_optional_tensor(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return _mark_dynamic_tensor(tensor, tensor.ndim - 1)


def _is_head_major_compact(t: torch.Tensor) -> bool:
    if t.dim() != 3:
        return False
    total_tokens, _, head_dim = t.shape
    return t.stride() == (head_dim, total_tokens * head_dim, 1)


def _as_bwd_compact_layout(t: torch.Tensor) -> torch.Tensor:
    if _is_head_major_compact(t):
        head_major = t.permute(1, 0, 2)
    else:
        head_major = t.permute(1, 0, 2).clone(memory_format=torch.contiguous_format)
    return head_major.permute(1, 2, 0).unsqueeze(3).unsqueeze(2)


def _empty_bwd_compact_layout_like(t: torch.Tensor) -> torch.Tensor:
    total_tokens, num_heads, head_dim = t.shape
    head_major = torch.empty(
        (num_heads, total_tokens, head_dim),
        dtype=t.dtype,
        device=t.device,
    )
    return head_major.permute(1, 2, 0).unsqueeze(3).unsqueeze(2)


def _as_bwd_original_qkv_layout(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 2, 1).unsqueeze(2).unsqueeze(4)


def _supports_bwd_original_qkv_layout(t: torch.Tensor) -> bool:
    if t.dim() != 3 or t.stride(2) != 1:
        return False
    if t.data_ptr() % 16 != 0:
        return False
    dimensions = sorted(
        (int(stride), int(size))
        for size, stride in zip(t.shape, t.stride())
        if size > 1
    )
    covered_span = 1
    for stride, size in dimensions:
        if stride < covered_span:
            return False
        covered_span += (size - 1) * stride
    # The backward kernel uses 128-bit global copy/TMA paths. For bf16/fp16,
    # token and head offsets must stay 8-element aligned.
    return t.stride(0) % 8 == 0 and t.stride(1) % 8 == 0


def hstu_varlen_fwd_100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: torch.Tensor,
    paged_kv: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    page_indptrs: Optional[torch.Tensor] = None,
    scaling_seqlen: Optional[float] = None,
    *,
    out: Optional[torch.Tensor] = None,
    _compile_only: bool = False,
):
    scaling_seqlen = _normalize_scaling_seqlen(scaling_seqlen, max_seqlen_q)
    q_dtype = q.dtype
    assert (
        q_dtype == torch.bfloat16 or q_dtype == torch.float16
    ), "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"

    head_dim = q.shape[2]
    head_dim_v = v.shape[2]
    assert head_dim == head_dim_v, "head_dim and head_dim_v must be equal"
    assert head_dim in (64, 128, 256), "Only support head_dim 64, 128 and 256"

    kBlockM = 128
    kBlockN = 128
    window_size_left = (
        max_seqlen_k
        if window_size_left < 0 or window_size_left > max_seqlen_k
        else window_size_left
    )
    window_size_right = (
        max_seqlen_k
        if window_size_right < 0 or window_size_right > max_seqlen_k
        else window_size_right
    )
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (
        window_size_left < max_seqlen_k or window_size_right < max_seqlen_k
    ) and not is_causal
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0
    is_paged = paged_kv is not None
    if is_paged:
        assert (
            is_causal
        ), "Paged KV is True, but causal mask is False, this is not supported."
        assert not is_local, (
            "Paged KV is True, but local mask is True, this is not supported."
        )
        assert (
            not is_arbitrary
        ), "Paged KV is True, but arbitrary mask is True, this is not supported."
        assert (
            page_ids is not None and page_indptrs is not None
        ), "Paged KV is True, but page metadata is missing."
        assert (
            paged_kv.dim() == 5 and paged_kv.shape[0] > 0 and paged_kv.shape[2] == 128
        ), "Only accept a non-empty 5-D paged KV table with page_size=128"

    # Keep the public output in the standard contiguous (T, H, D) layout so
    # downstream callers can flatten it with view() without an extra copy.
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    else:
        if out.shape != q.shape:
            raise ValueError(
                f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}"
            )
        if out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must have the same dtype and device as q")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    compile_key = (
        q.device,
        q_dtype,
        head_dim,
        kBlockM,
        kBlockN,
        is_causal,
        is_local,
        is_arbitrary,
        is_paged,
        func_num,
    )

    if _compile_only and compile_key in hstu_varlen_fwd_100.compile_cache:
        return out, None

    # The forward kernel only needs a contiguous last dim (q/k/v are passed via
    # mark_layout_dynamic(leading_dim=ndim-1)); full contiguity is not required.
    # When the (T,H,D) inputs already have a unit-stride last dim and 128-bit
    # aligned token/head strides, feed them in their original layout and skip the
    # contiguous copy. Non-aligned execution inputs use a real D2D clone; a
    # compile-only miss only needs matching empty layout samples.
    needs_contiguous_inputs = not (
        _supports_bwd_original_qkv_layout(q)
        and _supports_bwd_original_qkv_layout(k)
        and _supports_bwd_original_qkv_layout(v)
    )
    if needs_contiguous_inputs:
        if _compile_only:
            q, k, v = [
                torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                for tensor in (q, k, v)
            ]
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)
            v = v.clone(memory_format=torch.contiguous_format)

    paged_kv_flat = None
    if is_paged:
        paged_kv_flat = paged_kv.view(-1, paged_kv.shape[-2], paged_kv.shape[-1])

    if compile_key not in hstu_varlen_fwd_100.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            _mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (q, k, v, out)
        ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
            _mark_dynamic_tensor(tensor, tensor.ndim - 1)
            for tensor in (cu_seqlens_q, cu_seqlens_k)
        ]
        func_tensor = _mark_optional_tensor(func)
        paged_kv_tensor, page_ids_tensor, page_indptrs_tensor = [
            _mark_optional_tensor(tensor)
            for tensor in (paged_kv_flat, page_ids, page_indptrs)
        ]
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        hstu_fwd_sm100 = HSTUAttentionForwardSm100(
            head_dim=head_dim,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            is_paged=is_paged,
            func_num=func_num,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
        )
        with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
            hstu_varlen_fwd_100.compile_cache[compile_key] = cute.compile(
                hstu_fwd_sm100,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                max_seqlen_q,
                max_seqlen_k,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                alpha,
                scaling_seqlen,
                compile_stream,
                window_size_left,
                window_size_right,
                func_tensor,
                paged_kv_tensor,
                page_ids_tensor,
                page_indptrs_tensor,
                options="--enable-tvm-ffi",
            )

    if _compile_only:
        return out, None

    with torch.cuda.nvtx.range("hstu_varlen_fwd_kernel"):
        compiled_fwd = hstu_varlen_fwd_100.compile_cache[compile_key]
        compiled_fwd(
            q,
            k,
            v,
            out,
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q,
            cu_seqlens_k,
            alpha,
            scaling_seqlen,
            window_size_left,
            window_size_right,
            func,
            paged_kv_flat,
            page_ids,
            page_indptrs,
        )

    return out, None


hstu_varlen_fwd_100.compile_cache = {}


def hstu_varlen_bwd_100(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: torch.Tensor,
    deterministic: bool,
    scaling_seqlen: Optional[float] = None,
    *,
    _compile_only: bool = False,
):
    scaling_seqlen = _normalize_scaling_seqlen(scaling_seqlen, max_seqlen_q)
    if deterministic:
        raise NotImplementedError("deterministic HSTU backward is not supported")
    # asserts
    q_dtype = q.dtype
    assert (
        q_dtype == torch.bfloat16 or q_dtype == torch.float16
    ), "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"
    assert do.dtype == q_dtype, "do and q must have the same dtype"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must have dtype int32"
    assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must have dtype int32"

    batch_size = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    total_k = k.shape[0]
    num_heads_k = k.shape[1]

    assert head_dim == 64 or head_dim == 128, "Only support head_dim 64 and 128"
    assert (
        num_heads == num_heads_k
    ), "Number of heads in key/value and query must be equal"

    kBlockM = 128
    kBlockN = 128
    window_size_left = (
        max_seqlen_k
        if window_size_left < 0 or window_size_left > max_seqlen_k
        else window_size_left
    )
    window_size_right = (
        max_seqlen_k
        if window_size_right < 0 or window_size_right > max_seqlen_k
        else window_size_right
    )
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (
        window_size_left < max_seqlen_k or window_size_right < max_seqlen_k
    ) and not is_causal
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0

    q_orig, k_orig, v_orig = q, k, v
    dq_orig, dk_orig, dv_orig = dq, dk, dv
    use_original_qkv_layout = (
        _supports_bwd_original_qkv_layout(q)
        and _supports_bwd_original_qkv_layout(k)
        and _supports_bwd_original_qkv_layout(v)
    )
    use_original_do_layout = _supports_bwd_original_qkv_layout(do)
    no_preallocated_grads = dq is None and dk is None and dv is None
    preallocated_original_grads = (
        dq is not None
        and dk is not None
        and dv is not None
        and _supports_bwd_original_qkv_layout(dq)
        and _supports_bwd_original_qkv_layout(dk)
        and _supports_bwd_original_qkv_layout(dv)
    )
    use_original_grad_layout = use_original_qkv_layout and (
        no_preallocated_grads or preallocated_original_grads
    )
    compile_key = (
        q_orig.device,
        q_dtype,
        head_dim,
        kBlockM,
        kBlockN,
        use_original_qkv_layout,
        use_original_do_layout,
        use_original_grad_layout,
        is_causal,
        is_local,
        is_arbitrary,
        func_num,
    )
    if _compile_only and compile_key in hstu_varlen_bwd_100.compile_cache:
        if no_preallocated_grads:
            dq_orig, dk_orig, dv_orig = [
                torch.empty_like(tensor, memory_format=torch.preserve_format)
                for tensor in (q_orig, k_orig, v_orig)
            ]
        return dq_orig, dk_orig, dv_orig

    if use_original_qkv_layout:
        q = _as_bwd_original_qkv_layout(q)
        k = _as_bwd_original_qkv_layout(k)
        v = _as_bwd_original_qkv_layout(v)
    elif _compile_only:
        q, k, v = [_empty_bwd_compact_layout_like(tensor) for tensor in (q, k, v)]
    else:
        q = _as_bwd_compact_layout(q)
        k = _as_bwd_compact_layout(k)
        v = _as_bwd_compact_layout(v)

    # `do` shares the same original-layout fast path as q/k/v: when its layout is
    # 128-bit aligned, hand it to the kernel directly (a permute/view) instead of
    # cloning into the head-major compact layout. The kernel reads do as a K-major
    # TMA-B operand and adapts to the strides, exactly as it already does for q.
    # A non-head-major `do` (the common autograd grad layout) otherwise pays a real
    # ~90us (D128) / ~34us (D64) clone per backward call.
    if use_original_do_layout:
        do = _as_bwd_original_qkv_layout(do)
    elif _compile_only:
        do = _empty_bwd_compact_layout_like(do)
    else:
        do = _as_bwd_compact_layout(do)

    if use_original_grad_layout:
        if no_preallocated_grads:
            dq_orig, dk_orig, dv_orig = [
                torch.empty_strided(
                    tensor.shape,
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                for tensor in (q_orig, k_orig, v_orig)
            ]
        dq, dk, dv = [
            _as_bwd_original_qkv_layout(tensor)
            for tensor in (dq_orig, dk_orig, dv_orig)
        ]
    elif use_original_qkv_layout:
        dq = _empty_bwd_compact_layout_like(q_orig)
        dk = _empty_bwd_compact_layout_like(k_orig)
        dv = _empty_bwd_compact_layout_like(v_orig)
    else:
        dq = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
        dk = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
        dv = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=v.device)

    workspace_seqlen = (max_seqlen_q + 7) // 8 * 8
    workspace_head_dim = (head_dim + 7) // 8 * 8
    # Use torch.empty on GPU + .zero_() instead of torch.zeros(...).cuda()
    # torch.zeros().cuda() creates a CPU zero tensor then copies to GPU (~34ms for 128MB)
    # torch.empty().zero_() allocates on GPU directly and zeros via GPU kernel (~0.1ms)
    workspace_torch = torch.empty(
        (batch_size, num_heads, workspace_seqlen, workspace_head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    if not _compile_only:
        workspace_torch.zero_()
    problem_shape = (
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        Int32(head_dim),
        ((Int32(1), Int32(num_heads)), Int32(batch_size)),
    )
    if compile_key not in hstu_varlen_bwd_100.compile_cache:
        q_tensor, k_tensor, v_tensor = [
            _mark_dynamic_tensor(
                tensor,
                1,
                compact=not use_original_qkv_layout,
            )
            for tensor in (q, k, v)
        ]
        do_tensor = _mark_dynamic_tensor(
            do,
            1,
            compact=not use_original_do_layout,
        )
        dq_tensor, dk_tensor, dv_tensor = [
            _mark_dynamic_tensor(
                tensor,
                1,
                compact=True,
                stride_order=(
                    (0, 2, 3, 4, 1) if use_original_grad_layout else (2, 3, 0, 4, 1)
                ),
            )
            for tensor in (dq, dk, dv)
        ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
            _mark_dynamic_tensor(tensor, tensor.ndim - 1)
            for tensor in (cu_seqlens_q, cu_seqlens_k)
        ]
        func_tensor = _mark_optional_tensor(func)
        workspace = _mark_dynamic_tensor(
            workspace_torch,
            workspace_torch.ndim - 1,
        )
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        hstu_bwd_sm100 = HSTUAttentionBackwardSm100(
            element_dtype=Float16 if q_dtype == torch.float16 else BFloat16,
            head_dim=head_dim,
            kBlockM=kBlockM,
            kBlockN=kBlockN,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
        )
        with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
            hstu_varlen_bwd_100.compile_cache[compile_key] = cute.compile(
                hstu_bwd_sm100,
                problem_shape,
                q_tensor,
                k_tensor,
                v_tensor,
                dq_tensor,
                dk_tensor,
                dv_tensor,
                do_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                Int32(window_size_left),
                Int32(window_size_right),
                func_tensor,
                alpha,
                scaling_seqlen,
                workspace,
                compile_stream,
                options="--enable-tvm-ffi",
            )

    if _compile_only:
        return dq_orig, dk_orig, dv_orig

    with torch.cuda.nvtx.range("hstu_varlen_bwd_kernel"):
        compiled_bwd = hstu_varlen_bwd_100.compile_cache[compile_key]
        compiled_bwd(
            problem_shape,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            do,
            cu_seqlens_q,
            cu_seqlens_k,
            Int32(window_size_left),
            Int32(window_size_right),
            func,
            alpha,
            scaling_seqlen,
            workspace_torch,
        )

    if use_original_grad_layout:
        return dq_orig, dk_orig, dv_orig

    dq = dq.squeeze(4).squeeze(2).permute(0, 2, 1)
    dk = dk.squeeze(4).squeeze(2).permute(0, 2, 1)
    dv = dv.squeeze(4).squeeze(2).permute(0, 2, 1)

    if dq_orig is not None:
        dq_orig.copy_(dq)
    if dk_orig is not None:
        dk_orig.copy_(dk)
    if dv_orig is not None:
        dv_orig.copy_(dv)

    return dq, dk, dv


hstu_varlen_bwd_100.compile_cache = {}
