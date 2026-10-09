# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral public API for SM100 unfused BF16 grouped GEMM."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from cuda.bindings import driver as cuda

import cutlass

from cudnn.api_base import APIBase, TupleDict
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import (
    canonicalize_unit_dim_strides,
    cuda_is_available,
    detect_framework,
    get_compute_capability,
    get_data_ptr,
    get_device,
    get_shape,
    get_strides,
)

from ._bf16_api import GroupedGemmBf16API, _validate_pointer_tensor
from ..moe_utils import MoEWeightMode

__all__ = ["GroupedGemmBf16API"]


class GroupedGemmSm100(APIBase):
    """Public lifecycle facade with deferred BF16 descriptor initialization."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_b: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        generate_c: bool = False,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
    ) -> None:
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        if acc_dtype is None:
            self._pending_init_kwargs["acc_dtype"] = cutlass.Float32
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            self._implementation = GroupedGemmBf16API(**self._pending_init_kwargs)
            self._kernel = self._implementation._kernel
        supported = self._implementation.check_support()
        self._is_supported = self._implementation._is_supported
        if supported:
            self._pending_init_kwargs = None
        return supported

    def compile(self) -> None:
        if self._implementation is None:
            self.check_support()
        if self._is_supported:
            self._implementation._is_supported = True
        self._implementation.compile()
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        self._implementation.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
            bias_tensor=bias_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel


_cache_of_GroupedGemmSm100Objects: dict[tuple, GroupedGemmSm100] = {}

# Operand-metadata key -> the wrapper's derived result. Holds one entry per distinct
# (operand metadata, config) the process sees, the same growth as the op cache above.
_wrapper_memo: dict = {}


def _stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
    strides = get_strides(tensor)
    shape = get_shape(tensor)
    return tuple(
        index
        for index, _ in sorted(
            enumerate(strides),
            key=lambda item: (item[1], shape[item[0]]),
        )
    )


def _operand_meta(tensor: Optional[torch.Tensor]) -> Optional[tuple]:
    """Everything the wrapper's derivation reads off an operand, and nothing else.

    Deliberately not the object's identity: a tensor's address is recycled by CPython
    as soon as it is freed, so an id-keyed memo answers for tensors it never saw.
    Reading these five values costs ~0.3 us and cannot alias a differently-shaped
    operand. Data pointers are excluded because they vary per call and nothing derived
    here depends on them -- their alignment is re-checked inside execute().
    """
    if tensor is None:
        return None
    device = get_device(tensor)
    return (get_shape(tensor), get_strides(tensor), tensor.dtype, device.type, device.index)


def _allocate_output(framework: str, shape: tuple, stride: tuple, dtype, device) -> Any:
    from cudnn.tensor_adapter import framework_dtype

    if framework == "torch":
        import torch

        return torch.empty_strided(shape, stride, dtype=framework_dtype(dtype, "torch"), device=device)
    import jax
    import jax.numpy as jnp

    # n-major C-contiguous; the extent-1 batch dim's stride is unobservable.
    # The kernel writes into this buffer on the launch stream; materialize it first.
    return jax.block_until_ready(jnp.empty(shape, dtype=framework_dtype(dtype, "jax"), device=device))


def _tensor_signature(tensor: Optional[torch.Tensor], *, dynamic_m: bool = False) -> tuple:
    if tensor is None:
        return (None, None, None, None)
    device = get_device(tensor)
    shape = (None, *get_shape(tensor)[1:]) if dynamic_m else get_shape(tensor)
    return (
        shape,
        _stride_order(tensor),
        _convert_to_cutlass_data_type(tensor.dtype),
        (device.type, device.index),
    )


def _validate_output(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    tensor_shape = get_shape(tensor)
    tensor_stride = canonicalize_unit_dim_strides(tensor_shape, get_strides(tensor))
    expected_stride = canonicalize_unit_dim_strides(shape, stride)
    if tensor_shape != shape or tensor_stride != expected_stride or _convert_to_cutlass_data_type(tensor.dtype) != dtype or get_device(tensor) != device:
        raise ValueError(
            f"{name} must have shape {shape}, stride {stride}, dtype {dtype}, "
            f"device {device}; got shape {tensor_shape}, stride "
            f"{tensor_stride}, dtype {tensor.dtype}, device {get_device(tensor)}"
        )


def _normalize_call(
    a_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    b_tensor: Optional[torch.Tensor],
    b_ptrs: Optional[torch.Tensor],
    n: Optional[int],
    b_dtype: Optional[torch.dtype],
    prob_tensor: Optional[torch.Tensor],
    c_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    m_aligned: int,
    framework: str,
) -> tuple[bool, int, int, int]:
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either b_tensor or b_ptrs, not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either b_tensor or b_ptrs")
    if framework == "jax" and is_dense:
        raise ValueError(
            "Dense weight mode (b_tensor) is not expressible as JAX arrays "
            "(the expert-outermost strided B layout has no row-major equivalent); "
            "use discrete mode (b_ptrs) with per-expert weight pointers"
        )

    if _convert_to_cutlass_data_type(a_tensor.dtype) is not cutlass.BFloat16:
        raise ValueError(f"a_tensor must have dtype torch.bfloat16, got {a_tensor.dtype}")
    if len(get_shape(a_tensor)) != 3 or get_shape(a_tensor)[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {get_shape(a_tensor)}")
    if prob_tensor is None:
        raise ValueError("prob_tensor is required")
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")
    if c_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
        raise ValueError(f"c_dtype must be BF16, FP16, or FP32, got {c_dtype}")
    if d_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
        raise ValueError(f"d_dtype must be BF16, FP16, or FP32, got {d_dtype}")
    if m_aligned != 256:
        raise ValueError(f"m_aligned must be 256, got {m_aligned}")
    if not cuda_is_available():
        raise RuntimeError("CUDA is not available")
    major, minor = get_compute_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        raise RuntimeError(f"GroupedGemmSm100 requires SM100+, found SM{compute_capability}")

    tensor_m, k, _ = get_shape(a_tensor)
    if tensor_m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {tensor_m}")
    if get_shape(prob_tensor) != (tensor_m, 1, 1):
        raise ValueError(f"prob_tensor must have shape {(tensor_m, 1, 1)}, got " f"{get_shape(prob_tensor)}")
    if _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"prob_tensor must have dtype torch.float32, got {prob_tensor.dtype}")
    if is_dense:
        if n is not None:
            raise ValueError("Dense mode forbids n")
        if b_dtype is not None:
            raise ValueError("Dense mode forbids b_dtype")
        if _convert_to_cutlass_data_type(b_tensor.dtype) is not cutlass.BFloat16:
            raise ValueError(f"b_tensor must have dtype torch.bfloat16, got {b_tensor.dtype}")
        if len(get_shape(b_tensor)) != 3:
            raise ValueError(f"b_tensor must have shape (n, k, experts), got {get_shape(b_tensor)}")
        n, b_k, experts = get_shape(b_tensor)
        if b_k != k:
            raise ValueError(f"b_tensor K dimension ({b_k}) must match a_tensor ({k})")
    else:
        experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        if get_device(b_ptrs) != get_device(a_tensor):
            raise ValueError(f"b_ptrs must be on the same device as a_tensor " f"({get_device(a_tensor)}), got {get_device(b_ptrs)}")
        if get_data_ptr(b_ptrs) % 8 != 0:
            raise ValueError("b_ptrs data pointer must be 8-byte aligned")
        offsets_shape = get_shape(padded_offsets)
        if len(offsets_shape) == 1 and experts != offsets_shape[0]:
            raise ValueError(f"b_ptrs length mismatch: expected {offsets_shape[0]}, " f"got {experts}")
        if n is None or b_dtype is None:
            raise ValueError("Discrete mode requires n and b_dtype")
        if b_dtype is not cutlass.BFloat16:
            raise ValueError(f"b_dtype must be torch.bfloat16 for the BF16 backend, got {b_dtype}")
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

    return is_dense, tensor_m, n, experts


def grouped_gemm_wrapper_sm100(
    a_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    prob_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    c_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    c_tensor: Optional[torch.Tensor] = None,
    d_tensor: Optional[torch.Tensor] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    vector_f32: bool = False,
    m_aligned: int = 256,
    generate_c: bool = False,
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    # Hot-loop memo. Everything between here and op.execute() is derivation -- resolving
    # dtypes, deriving (m, n, experts), and rebuilding the op cache key -- and it is a
    # pure function of the operands' metadata plus the scalar config, both of which are
    # in the key below. So a hit skips the derivation and reuses its result. It does not
    # skip any check: op.execute() still validates every operand, including the data
    # pointers the key deliberately omits. Only the allocate-the-outputs case is memoized;
    # caller-supplied c/d need _validate_output and fall through.
    _memo_key = None
    if c_tensor is None and d_tensor is None:
        _memo_key = (
            type(a_tensor),
            _operand_meta(a_tensor),
            _operand_meta(padded_offsets),
            _operand_meta(alpha_tensor),
            _operand_meta(prob_tensor),
            _operand_meta(b_tensor),
            _operand_meta(b_ptrs),
            _operand_meta(bias_tensor),
            n,
            b_dtype,
            b_major,
            acc_dtype,
            c_dtype,
            d_dtype,
            cd_major,
            tuple(mma_tiler_mn),
            None if cluster_shape_mn is None else tuple(cluster_shape_mn),
            vector_f32,
            m_aligned,
            generate_c,
            use_dynamic_sched,
            os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"),
        )
        memo = _wrapper_memo.get(_memo_key)
        if memo is not None:
            op, framework, expected_shape, expected_stride, memo_c_dtype, memo_d_dtype, is_dense = memo
            internal_c = _allocate_output(framework, expected_shape, expected_stride, memo_c_dtype, a_tensor.device)
            d_out = _allocate_output(framework, expected_shape, expected_stride, memo_d_dtype, a_tensor.device)
            op.execute(
                a_tensor=a_tensor,
                c_tensor=internal_c,
                d_tensor=d_out,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                b_tensor=b_tensor if is_dense else None,
                b_ptrs=None if is_dense else b_ptrs,
                bias_tensor=bias_tensor,
                prob_tensor=prob_tensor,
                current_stream=current_stream,
            )
            return TupleDict(d_tensor=d_out, c_tensor=internal_c if generate_c else None)

    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_wrapper_sm100; pass torch tensors or JAX arrays")
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    c_dtype = _convert_to_cutlass_data_type(c_dtype) if c_dtype is not None else cutlass.BFloat16
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None
    if framework == "jax" and bias_tensor is not None:
        raise ValueError(
            "bias_tensor is not expressible as a JAX array (its (n, experts) column-major layout has no row-major equivalent); " "omit bias for JAX inputs"
        )
    is_dense, tensor_m, n_out, expert_cnt = _normalize_call(
        a_tensor,
        padded_offsets,
        b_tensor,
        b_ptrs,
        n,
        b_dtype,
        prob_tensor,
        c_dtype,
        d_dtype,
        cd_major,
        m_aligned,
        framework,
    )
    expected_shape = (tensor_m, n_out, 1)
    expected_stride = (n_out, 1, tensor_m * n_out)

    def _allocate(dtype):
        return _allocate_output(framework, expected_shape, expected_stride, dtype, a_tensor.device)

    if c_tensor is None:
        internal_c = _allocate(c_dtype)
    else:
        _validate_output(
            c_tensor,
            name="c_tensor",
            shape=expected_shape,
            stride=expected_stride,
            dtype=c_dtype,
            device=get_device(a_tensor),
        )
        internal_c = c_tensor
    if d_tensor is None:
        d_tensor = _allocate(d_dtype)
    else:
        _validate_output(
            d_tensor,
            name="d_tensor",
            shape=expected_shape,
            stride=expected_stride,
            dtype=d_dtype,
            device=get_device(a_tensor),
        )

    overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    workspace_bytes = (128 * expert_cnt if not is_dense else 0) + (4 if use_dynamic_sched else 0)
    a_device = get_device(a_tensor)
    workspace_signature = (
        (max(workspace_bytes, 1),),
        (a_device.type, a_device.index),
    )
    cache_key = (
        "bf16",
        "dense" if is_dense else "discrete",
        *_tensor_signature(a_tensor, dynamic_m=True),
        *_tensor_signature(b_tensor),
        *_tensor_signature(b_ptrs),
        *_tensor_signature(internal_c, dynamic_m=True),
        *_tensor_signature(d_tensor, dynamic_m=True),
        *_tensor_signature(padded_offsets),
        *_tensor_signature(alpha_tensor),
        *_tensor_signature(bias_tensor),
        *_tensor_signature(prob_tensor, dynamic_m=True),
        n_out,
        expert_cnt,
        b_dtype,
        b_major,
        acc_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn) if cluster_shape_mn is not None else None,
        vector_f32,
        m_aligned,
        generate_c,
        use_dynamic_sched,
        overlap_margin,
        workspace_signature,
    )

    op = _cache_of_GroupedGemmSm100Objects.get(cache_key)
    if op is None:
        op = GroupedGemmSm100(
            sample_a=a_tensor,
            sample_c=internal_c,
            sample_d=d_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_b=b_tensor if is_dense else None,
            sample_bias=bias_tensor,
            sample_prob=prob_tensor,
            num_experts=None if is_dense else expert_cnt,
            b_shape=None if is_dense else (n_out, a_tensor.shape[1]),
            b_dtype=None if is_dense else b_dtype,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            generate_c=generate_c,
            b_major=b_major,
            use_dynamic_sched=use_dynamic_sched,
        )
        assert op.check_support(), "Unsupported configuration"
        op.compile()
        _cache_of_GroupedGemmSm100Objects[cache_key] = op

    if _memo_key is not None:
        _wrapper_memo[_memo_key] = (op, framework, expected_shape, expected_stride, c_dtype, d_dtype, is_dense)

    op.execute(
        a_tensor=a_tensor,
        c_tensor=internal_c,
        d_tensor=d_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        b_tensor=b_tensor if is_dense else None,
        b_ptrs=None if is_dense else b_ptrs,
        bias_tensor=bias_tensor,
        prob_tensor=prob_tensor,
        current_stream=current_stream,
    )
    return TupleDict(
        d_tensor=d_tensor,
        c_tensor=internal_c if generate_c else None,
    )


__all__ = [
    "GroupedGemmSm100",
    "grouped_gemm_wrapper_sm100",
]
