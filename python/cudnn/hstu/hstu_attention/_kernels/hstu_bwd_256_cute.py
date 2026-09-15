# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated two-kernel CuTe DSL backward for HSTU head dimension 256.

The implementation uses separate DQ and DK/DV kernels. The score transform is
HSTU SiLU rather than softmax.
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

from .hstu_bwd_256_cute_dkdv import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from .hstu_bwd_256_cute_dq import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from .block_sparse_builder import build_hstu_d256_bwd_block_sparse
from .block_sparsity import HSTUBlockSparseTensors


def _as_bshkrd_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
) -> cute.Tensor:
    """View a packed (T,H,D) tensor as (1,T,H_k,H_r,D)."""
    assert cutlass.const_expr(cute.rank(tensor.layout) == 3)
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, tensor.shape[0], h_k, h_r, tensor.shape[2]),
            stride=(
                0,
                tensor.stride[0],
                tensor.stride[1] * h_r,
                tensor.stride[1],
                tensor.stride[2],
            ),
        ),
    )


class HSTUAttentionBackwardSm100D256:
    """Launch the dedicated DQ kernel followed by the dedicated DK/DV kernel."""

    def __init__(
        self,
        *,
        is_causal: bool,
        is_arbitrary: bool,
        func_num: int,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        skip_residual_mask: bool,
        use_auto_block_metadata: bool = False,
    ):
        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            cutlass.Float32,
            (128, 128, 256),
            is_causal,
            window_size_left,
            window_size_right,
            skip_residual_mask=skip_residual_mask,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            use_auto_block_metadata=use_auto_block_metadata,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            cutlass.Float32,
            (128, 64, 256),
            is_causal,
            window_size_left,
            window_size_right,
            skip_residual_mask=skip_residual_mask,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            use_auto_block_metadata=use_auto_block_metadata,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        func: cute.Tensor | None,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        alpha: cutlass.Float32,
        normalization_scale: cutlass.Float32,
        q2k_block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        k2q_block_sparse_tensors: Optional[HSTUBlockSparseTensors],
        stream: cuda.CUstream,
    ):
        assert cutlass.const_expr(cute.rank(q.layout) == 3)
        assert cutlass.const_expr(cute.rank(k.layout) == 3)
        h_q = q.shape[1]
        h_k = k.shape[1]
        h_r = h_q // h_k
        q = _as_bshkrd_tensor(q, h_k, h_r)
        k = _as_bshkrd_tensor(k, h_k, 1)
        v = _as_bshkrd_tensor(v, h_k, 1)
        do = _as_bshkrd_tensor(do, h_k, h_r)
        dq = _as_bshkrd_tensor(dq, h_k, h_r)
        dk = _as_bshkrd_tensor(dk, h_k, 1)
        dv = _as_bshkrd_tensor(dv, h_k, 1)
        self.dq_kernel(
            q,
            k,
            v,
            dq,
            do,
            cu_seqlens_q,
            cu_seqlens_k,
            func,
            max_seqlen_q,
            alpha,
            alpha * normalization_scale,
            q2k_block_sparse_tensors,
            stream,
        )
        self.dkdv_kernel(
            q,
            k,
            v,
            dk,
            dv,
            do,
            cu_seqlens_q,
            cu_seqlens_k,
            func,
            max_seqlen_q,
            max_seqlen_k,
            alpha,
            normalization_scale,
            k2q_block_sparse_tensors,
            stream,
        )


def _dynamic_tensor(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)


def _dynamic_optional_tensor(
    tensor: Optional[torch.Tensor],
) -> Optional[cute.Tensor]:
    return None if tensor is None else _dynamic_tensor(tensor, tensor.ndim - 1)


def _dynamic_block_sparse_tensors(
    tensors,
) -> Optional[HSTUBlockSparseTensors]:
    if tensors is None:
        return None
    return HSTUBlockSparseTensors(*(_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in tensors[:6]))


def _runtime_block_sparse_tensors(tensors):
    if tensors is None:
        return None
    return tuple(tensors[:6])


def _copy_to_optional_output(
    work: torch.Tensor,
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    if output is None:
        return work
    assert output.shape == work.shape
    assert output.dtype == work.dtype
    assert output.device == work.device
    if output.data_ptr() == work.data_ptr():
        return output
    output.copy_(work)
    return output


def _native_output_buffer(
    output: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if (
        output is not None
        and output.shape == reference.shape
        and output.dtype == reference.dtype
        and output.device == reference.device
        and output.is_contiguous()
    ):
        return output
    return torch.empty_like(reference)


def _compact_input_buffer(
    tensor: torch.Tensor,
    *,
    compile_only: bool,
) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor
    if compile_only:
        return torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    return tensor.contiguous()


def hstu_varlen_bwd_256_cute(
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
    scaling_seqlen: float,
    *,
    func: Optional[torch.Tensor] = None,
    _compile_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile and run the native CuTe DSL D=256 path.

    Native predicates cover unmasked, causal, local, and arbitrary-interval
    attention.
    """
    assert q.ndim == k.ndim == v.ndim == do.ndim == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda and do.is_cuda
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert q.dtype == k.dtype == v.dtype == do.dtype
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == do.shape[-1] == 256
    assert q.shape[1] == k.shape[1] == v.shape[1] == do.shape[1]
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_q.numel() == cu_seqlens_k.numel()
    assert max_seqlen_q > 0 and max_seqlen_k > 0
    assert scaling_seqlen > 0.0

    is_causal = window_size_left == max_seqlen_k and window_size_right == 0
    is_unmasked = window_size_left == max_seqlen_k and window_size_right == max_seqlen_k
    is_local = not (is_causal or is_unmasked)
    is_arbitrary = func is not None
    assert not is_arbitrary or not (is_causal or is_local)
    func_num = 0
    if is_arbitrary:
        assert func is not None
        assert func.ndim == 3 and func.shape[0] == 1
        assert func.device == q.device and func.dtype == torch.int32
        assert func.shape[-1] >= q.shape[0]
        func_num = func.shape[-2]
        assert func_num > 0 and func_num % 2 == 1

    use_auto_block_metadata = is_arbitrary
    q2k_block_sparse_tensors = None
    k2q_block_sparse_tensors = None
    if use_auto_block_metadata:
        (
            q2k_block_sparse_tensors,
            k2q_block_sparse_tensors,
        ) = build_hstu_d256_bwd_block_sparse(
            func,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            compile_only=_compile_only,
        )

    batch_size = cu_seqlens_q.numel() - 1
    # Full tiles may skip predicates only when no semantic mask remains.
    skip_residual_mask = (
        is_unmasked
        and not is_arbitrary
        and q.shape[0] == batch_size * max_seqlen_q
        and k.shape[0] == batch_size * max_seqlen_k
        and max_seqlen_q % 128 == 0
        and max_seqlen_k % 128 == 0
    )
    compile_key = (
        q.device,
        q.dtype,
        q.shape[1],
        is_causal,
        is_local,
        is_arbitrary,
        func_num,
        window_size_left if is_local else None,
        window_size_right if is_local else None,
        skip_residual_mask,
        use_auto_block_metadata,
    )
    if _compile_only and compile_key in hstu_varlen_bwd_256_cute.compile_cache:
        return tuple(
            output if output is not None else torch.empty_like(reference, memory_format=torch.preserve_format)
            for output, reference in ((dq, q), (dk, k), (dv, v))
        )

    # The dedicated kernels require fully compact TMA operands. Preserve the
    # public layout contract by materializing compact inputs when necessary.
    q_work = _compact_input_buffer(q, compile_only=_compile_only)
    k_work = _compact_input_buffer(k, compile_only=_compile_only)
    v_work = _compact_input_buffer(v, compile_only=_compile_only)
    do_work = _compact_input_buffer(do, compile_only=_compile_only)
    dq_work = _native_output_buffer(dq, q_work)
    dk_work = _native_output_buffer(dk, k_work)
    dv_work = _native_output_buffer(dv, v_work)
    normalization_scale = 1.0 / scaling_seqlen
    if compile_key not in hstu_varlen_bwd_256_cute.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor = [_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (q_work, k_work, v_work, do_work)]
        dq_tensor, dk_tensor, dv_tensor = [_dynamic_tensor(tensor, tensor.ndim - 1) for tensor in (dq_work, dk_work, dv_work)]
        cu_q_tensor, cu_k_tensor = [_dynamic_tensor(tensor, 0) for tensor in (cu_seqlens_q, cu_seqlens_k)]
        func_tensor = _dynamic_optional_tensor(func)
        q2k_block_sparse_cute = _dynamic_block_sparse_tensors(q2k_block_sparse_tensors)
        k2q_block_sparse_cute = _dynamic_block_sparse_tensors(k2q_block_sparse_tensors)
        compile_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = HSTUAttentionBackwardSm100D256(
            is_causal=is_causal,
            is_arbitrary=is_arbitrary,
            func_num=func_num,
            window_size_left=window_size_left if is_local else None,
            window_size_right=window_size_right if is_local else None,
            skip_residual_mask=skip_residual_mask,
            use_auto_block_metadata=use_auto_block_metadata,
        )
        hstu_varlen_bwd_256_cute.compile_cache[compile_key] = cute.compile(
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            cu_q_tensor,
            cu_k_tensor,
            func_tensor,
            Int32(max_seqlen_q),
            Int32(max_seqlen_k),
            alpha,
            normalization_scale,
            q2k_block_sparse_cute,
            k2q_block_sparse_cute,
            compile_stream,
            options="--enable-tvm-ffi",
        )

    if _compile_only:
        return tuple(output if output is not None else work for output, work in ((dq, dq_work), (dk, dk_work), (dv, dv_work)))

    compiled = hstu_varlen_bwd_256_cute.compile_cache[compile_key]
    compiled(
        q_work,
        k_work,
        v_work,
        do_work,
        dq_work,
        dk_work,
        dv_work,
        cu_seqlens_q,
        cu_seqlens_k,
        func,
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        alpha,
        normalization_scale,
        _runtime_block_sparse_tensors(q2k_block_sparse_tensors),
        _runtime_block_sparse_tensors(k2q_block_sparse_tensors),
    )
    return (
        _copy_to_optional_output(dq_work, dq),
        _copy_to_optional_output(dk_work, dk),
        _copy_to_optional_output(dv_work, dv),
    )


hstu_varlen_bwd_256_cute.compile_cache = {}
