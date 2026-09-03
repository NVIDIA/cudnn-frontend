#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Float16, BFloat16

from ._kernels.hstu_fwd import HSTUAttentionForwardSm100
from ._kernels.hstu_bwd import HSTUAttentionBackwardSm100
from ._kernels.block_sparse_builder import (
    build_hstu_k2q_block_sparse,
    build_hstu_q2k_block_sparse,
)
from ._kernels.block_sparsity import HSTUBlockSparseTensors


def _cutlass_dsl_version() -> tuple[int, int, int]:
    """Return the installed CUTLASS DSL version without another dependency."""
    version = getattr(cutlass, "__version__", None)
    try:
        parts = str(version).split(".")
        parsed = []
        for index in range(3):
            digits = ""
            for character in parts[index]:
                if not character.isdigit():
                    break
                digits += character
            if not digits:
                raise ValueError
            parsed.append(int(digits))
        return tuple(parsed)
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse CUTLASS DSL version {version!r}") from exc


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
    cute_tensor = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
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


def _mark_block_sparse_tensors(tensors):
    if tensors is None:
        return None
    return HSTUBlockSparseTensors(*(_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in tensors[:6]))


def _runtime_block_sparse_tensors(tensors):
    if tensors is None:
        return None
    return tuple(tensors[:6])


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
    total_tokens, num_heads, head_dim = t.shape
    stride_token, stride_head, stride_dim = t.stride()
    return t.as_strided(
        (total_tokens, head_dim, 1, num_heads, 1),
        (
            stride_token,
            stride_dim,
            stride_head * num_heads,
            stride_head,
            stride_dim,
        ),
    )


def _supports_bwd_original_qkv_layout(t: torch.Tensor) -> bool:
    if t.dim() != 3 or t.stride(2) != 1:
        return False
    if t.data_ptr() % 16 != 0:
        return False
    dimensions = sorted((int(stride), int(size)) for size, stride in zip(t.shape, t.stride()) if size > 1)
    covered_span = 1
    for stride, size in dimensions:
        if stride < covered_span:
            return False
        covered_span += (size - 1) * stride
    # The backward kernel uses 128-bit global copy/TMA paths. For bf16/fp16,
    # token and head offsets must stay 8-element aligned.
    return t.stride(0) % 8 == 0 and t.stride(1) % 8 == 0


def _supports_bwd_direct_grad_layout(t: torch.Tensor) -> bool:
    """Return whether the fused epilogue can write directly to ``t``.

    The D64/D128 epilogue uses 128-bit stores, so the unit-stride head
    dimension and the token/head offsets must remain 16-byte aligned.
    The kernel restores those dynamic-stride divisibility assumptions before
    constructing its output views.
    """
    # Keep zero strides out of this dynamic-layout cache variant. DLPack/CuTe
    # represents broadcast strides as static zero, so mixing them with a
    # previously compiled nonzero-stride descriptor would not be type-safe.
    return _supports_bwd_original_qkv_layout(t) and t.stride(0) != 0 and t.stride(1) != 0


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
    assert q_dtype == torch.bfloat16 or q_dtype == torch.float16, "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"

    head_dim = q.shape[2]
    head_dim_v = v.shape[2]
    assert head_dim == head_dim_v, "head_dim and head_dim_v must be equal"
    assert head_dim in (64, 128, 256), "Only support head_dim 64, 128 and 256"

    kBlockM = 128
    kBlockN = 128
    window_size_left = max_seqlen_k if window_size_left < 0 or window_size_left > max_seqlen_k else window_size_left
    window_size_right = max_seqlen_k if window_size_right < 0 or window_size_right > max_seqlen_k else window_size_right
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (window_size_left < max_seqlen_k or window_size_right < max_seqlen_k) and not is_causal
    is_arbitrary = func is not None
    use_auto_block_metadata = is_arbitrary
    func_num = func.shape[-2] if func is not None else 0
    is_paged = paged_kv is not None
    use_2cta_instrs = (
        torch.cuda.get_device_capability(q.device) == (10, 7)
        and head_dim == 128
        and is_causal
        and not is_local
        and not is_arbitrary
        and not is_paged
        and q.shape[1] == k.shape[1] == v.shape[1]
    )
    # The 4.5.x PipelineAsync descriptor path is nondeterministic when CUDA
    # contexts contend for an SM. Keep CLC scheduling, but use its direct
    # work-coordinate path until the corrected 4.6 pipeline implementation.
    use_clc_descriptor = _cutlass_dsl_version() >= (4, 6, 0)
    if is_paged:
        assert is_causal, "Paged KV is True, but causal mask is False, this is not supported."
        assert not is_local, "Paged KV is True, but local mask is True, this is not supported."
        assert not is_arbitrary, "Paged KV is True, but arbitrary mask is True, this is not supported."
        assert page_ids is not None and page_indptrs is not None, "Paged KV is True, but page metadata is missing."
        assert paged_kv.dim() == 5 and paged_kv.shape[0] > 0 and paged_kv.shape[2] == 128, "Only accept a non-empty 5-D paged KV table with page_size=128"

    # Keep the public output in the standard contiguous (T, H, D) layout so
    # downstream callers can flatten it with view() without an extra copy.
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    else:
        if out.shape != q.shape:
            raise ValueError(f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}")
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
        use_auto_block_metadata,
        use_2cta_instrs,
        use_clc_descriptor,
    )

    block_sparse_tensors = None
    if use_auto_block_metadata:
        q2k_block_size = (
            kBlockM if head_dim == 256 else 2 * kBlockM,
            kBlockN,
        )
        with torch.cuda.nvtx.range("hstu_q2k_block_sparse_builder"):
            block_sparse_tensors = build_hstu_q2k_block_sparse(
                func,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size=q2k_block_size,
                compile_only=_compile_only,
            )

    if _compile_only and compile_key in hstu_varlen_fwd_100.compile_cache:
        return out, None

    # The forward kernel only needs a contiguous last dim (q/k/v are passed via
    # mark_layout_dynamic(leading_dim=ndim-1)); full contiguity is not required.
    # When the (T,H,D) inputs already have a unit-stride last dim and 128-bit
    # aligned token/head strides, feed them in their original layout and skip the
    # contiguous copy. Non-aligned execution inputs use a real D2D clone; a
    # compile-only miss only needs matching empty layout samples.
    needs_contiguous_inputs = not (_supports_bwd_original_qkv_layout(q) and _supports_bwd_original_qkv_layout(k) and _supports_bwd_original_qkv_layout(v))
    if needs_contiguous_inputs:
        if _compile_only:
            q, k, v = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for tensor in (q, k, v)]
        else:
            q = q.clone(memory_format=torch.contiguous_format)
            k = k.clone(memory_format=torch.contiguous_format)
            v = v.clone(memory_format=torch.contiguous_format)

    paged_kv_flat = None
    if is_paged:
        paged_kv_flat = paged_kv.view(-1, paged_kv.shape[-2], paged_kv.shape[-1])

    if compile_key not in hstu_varlen_fwd_100.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (q, k, v, out)]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (cu_seqlens_q, cu_seqlens_k)]
        func_tensor = _mark_optional_tensor(func)
        paged_kv_tensor, page_ids_tensor, page_indptrs_tensor = [_mark_optional_tensor(tensor) for tensor in (paged_kv_flat, page_ids, page_indptrs)]
        block_sparse_cute = _mark_block_sparse_tensors(block_sparse_tensors)
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
            use_auto_block_metadata=use_auto_block_metadata,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_descriptor=use_clc_descriptor,
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
                block_sparse_cute,
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
            _runtime_block_sparse_tensors(block_sparse_tensors),
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
    assert q_dtype == torch.bfloat16 or q_dtype == torch.float16, "Only support bf16 and fp16"
    assert k.dtype == q_dtype, "k and q must have the same dtype"
    assert v.dtype == q_dtype, "v and q must have the same dtype"
    assert do.dtype == q_dtype, "do and q must have the same dtype"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must have dtype int32"
    assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must have dtype int32"

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_heads_k = k.shape[1]

    assert head_dim in (64, 128, 256), "Only support head_dim 64, 128 and 256"
    assert num_heads == num_heads_k, "Number of heads in key/value and query must be equal"
    assert k.shape[2] == head_dim, "k and q must have the same head_dim"
    assert v.shape[2] == head_dim, "v and q must have the same head_dim"
    assert do.shape == q.shape, "do and q must have the same shape"

    m_block_size = 128
    n_block_size = 128
    window_size_left = max_seqlen_k if window_size_left < 0 or window_size_left > max_seqlen_k else window_size_left
    window_size_right = max_seqlen_k if window_size_right < 0 or window_size_right > max_seqlen_k else window_size_right
    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_local = (window_size_left < max_seqlen_k or window_size_right < max_seqlen_k) and not is_causal
    is_arbitrary = func is not None
    func_num = func.shape[-2] if func is not None else 0
    use_2cta_instrs = head_dim == 128 and not is_arbitrary
    if head_dim == 256:
        # The fused one-CTA kernel's live TMEM ranges exceed the SM100
        # 512-column capacity at D=256. Use the dedicated two-kernel path:
        # dQ first, followed by dK/dV.
        from ._kernels.hstu_bwd_256_cute import hstu_varlen_bwd_256_cute

        return hstu_varlen_bwd_256_cute(
            do,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dq,
            dk,
            dv,
            window_size_left,
            window_size_right,
            alpha,
            scaling_seqlen,
            func=func,
            _compile_only=_compile_only,
        )

    use_auto_block_metadata = is_arbitrary
    block_sparse_tensors = None
    if use_auto_block_metadata:
        # Build on every execution so in-place func updates, including CUDA
        # Graph replay updates, are visible to the consumer.  The private K2Q
        # layout is fixed by the fused D64/D128 backward tile contract.
        block_sparse_tensors = build_hstu_k2q_block_sparse(
            func,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size=(m_block_size, n_block_size),
            compile_only=_compile_only,
        )

    q_orig, k_orig, v_orig = q, k, v
    dq_orig, dk_orig, dv_orig = dq, dk, dv
    use_original_qkv_layout = _supports_bwd_original_qkv_layout(q) and _supports_bwd_original_qkv_layout(k) and _supports_bwd_original_qkv_layout(v)
    use_original_do_layout = _supports_bwd_original_qkv_layout(do)
    no_preallocated_grads = dq is None and dk is None and dv is None
    implicit_direct_grads = no_preallocated_grads and all(_supports_bwd_direct_grad_layout(tensor) for tensor in (q_orig, k_orig, v_orig))
    preallocated_direct_grads = (
        dq is not None
        and dk is not None
        and dv is not None
        and _supports_bwd_direct_grad_layout(dq)
        and _supports_bwd_direct_grad_layout(dk)
        and _supports_bwd_direct_grad_layout(dv)
    )
    # Gradient stores are independent of whether q/k/v use their original
    # layouts or compact read-only staging buffers.
    use_original_grad_layout = implicit_direct_grads or preallocated_direct_grads
    compile_key = (
        q_orig.device,
        q_dtype,
        head_dim,
        m_block_size,
        n_block_size,
        use_original_qkv_layout,
        use_original_do_layout,
        use_original_grad_layout,
        is_causal,
        is_local,
        is_arbitrary,
        func_num,
        use_auto_block_metadata,
        use_2cta_instrs,
    )
    if _compile_only and compile_key in hstu_varlen_bwd_100.compile_cache:
        if no_preallocated_grads:
            dq_orig, dk_orig, dv_orig = [torch.empty_like(tensor, memory_format=torch.preserve_format) for tensor in (q_orig, k_orig, v_orig)]
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

    # Preserve an aligned dO layout and avoid a compact staging copy.
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
        dq, dk, dv = [_as_bwd_original_qkv_layout(tensor) for tensor in (dq_orig, dk_orig, dv_orig)]
    elif use_original_qkv_layout:
        dq = _empty_bwd_compact_layout_like(q_orig)
        dk = _empty_bwd_compact_layout_like(k_orig)
        dv = _empty_bwd_compact_layout_like(v_orig)
    else:
        dq = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
        dk = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
        dv = torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device=v.device)

    workspace_head_dim = (head_dim + 7) // 8 * 8
    # Allocate and initialize the accumulation workspace directly on the GPU.
    workspace_padding_rows = batch_size * m_block_size if use_2cta_instrs else 0
    workspace_torch = torch.empty(
        (
            num_heads,
            q.shape[0] + workspace_padding_rows,
            workspace_head_dim,
        ),
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
                compact=not use_original_grad_layout,
            )
            for tensor in (dq, dk, dv)
        ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [_mark_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (cu_seqlens_q, cu_seqlens_k)]
        func_tensor = _mark_optional_tensor(func)
        workspace = _mark_dynamic_tensor(
            workspace_torch,
            workspace_torch.ndim - 1,
        )
        block_sparse_cute = _mark_block_sparse_tensors(block_sparse_tensors)
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        hstu_bwd_sm100 = HSTUAttentionBackwardSm100(
            element_dtype=Float16 if q_dtype == torch.float16 else BFloat16,
            head_dim=head_dim,
            tile_m=m_block_size,
            tile_n=n_block_size,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            use_auto_block_metadata=use_auto_block_metadata,
            use_2cta_instrs=use_2cta_instrs,
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
                block_sparse_cute,
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
            _runtime_block_sparse_tensors(block_sparse_tensors),
        )

    if use_original_grad_layout:
        return dq_orig, dk_orig, dv_orig

    dq = dq.squeeze(4).squeeze(2).permute(0, 2, 1)
    dk = dk.squeeze(4).squeeze(2).permute(0, 2, 1)
    dv = dv.squeeze(4).squeeze(2).permute(0, 2, 1)

    if dq_orig is not None:
        dq_orig.copy_(dq)
        dq = dq_orig
    if dk_orig is not None:
        dk_orig.copy_(dk)
        dk = dk_orig
    if dv_orig is not None:
        dv_orig.copy_(dv)
        dv = dv_orig

    return dq, dk, dv


hstu_varlen_bwd_100.compile_cache = {}
