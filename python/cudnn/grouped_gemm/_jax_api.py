# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared JAX metadata and lowering helpers for contiguous grouped GEMMs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from cutlass.jax import TensorSpec

from .._cute_compiler import compile_options_for_target
from .._dense_gemm import (
    block_scale_shape as _canonical_block_scale_shape,
    data_type_bits,
)
from .._jax import JaxApiBase, JaxTensorDesc, TupleDict
from .._jax.gemm import BLOCK_SCALE_MODE, gemm_a_mode, gemm_b_mode, gemm_output_mode
from .._jax.layout import normalize_mode, to_public_axes

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)
MAX_EXPERTS = 1024
FP8_SF_VEC_SIZE = 32
TWO_CTA_MMA_TILER_M = 256
FIX_PAD_SIZE = 256
HADAMARD_SIZE = 16
SF_VEC_SIZES = (16, 32)

_NO_DEFAULT = object()


def as_dtype(value: Any) -> Any:
    """Return a normalized JAX dtype without retaining an array value."""

    if not isinstance(value, type) and hasattr(value, "dtype"):
        value = value.dtype
    return jnp.dtype(value)


def is_fp4_dtype(value: Any) -> bool:
    """Return whether ``value`` is JAX's native logical FP4 dtype."""

    return as_dtype(value) == jnp.dtype(jnp.float4_e2m1fn)


def is_fp8_dtype(value: Any) -> bool:
    """Return whether ``value`` is one of the supported JAX FP8 data dtypes."""

    return as_dtype(value) in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }


def is_low_precision_output_dtype(value: Any) -> bool:
    """Return whether an output uses a native FP4 or FP8 dtype."""

    return is_fp4_dtype(value) or is_fp8_dtype(value)


def require_dtype(
    value: Any,
    valid_dtypes: Iterable[Any],
    *,
    name: str = "dtype",
    default: Any = _NO_DEFAULT,
) -> Any:
    """Normalize and validate a JAX dtype argument."""

    if value is None:
        if default is _NO_DEFAULT:
            raise ValueError(f"{name} must not be None")
        value = default
    dtype = as_dtype(value)
    valid = tuple(as_dtype(item) for item in valid_dtypes)
    if dtype not in valid:
        supported = ", ".join(item.name for item in valid)
        raise ValueError(f"{name} must be one of {{{supported}}}, got {dtype}")
    return dtype


def require_array(
    value: Any,
    *,
    name: str,
    rank: int | Iterable[int] | None = None,
    shape: Sequence[int] | None = None,
    dtype: Any | Iterable[Any] | None = None,
) -> tuple[int, ...]:
    """Validate abstract array metadata and return its public shape."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must have shape and dtype metadata")
    metadata_shape = tuple(value.shape)
    # Sample descriptors store canonical kernel axes. Exact shape checks in
    # this adapter describe the public JAX value, so compare against the
    # descriptor's inverse-mapped array shape when one is available.
    actual_shape = (
        tuple(value.array_shape)
        if shape is not None and hasattr(value, "array_shape")
        else metadata_shape
    )
    if rank is not None:
        ranks = (rank,) if isinstance(rank, int) else tuple(rank)
        if len(actual_shape) not in ranks:
            raise ValueError(
                f"{name} must have rank in {ranks}, got shape {actual_shape}"
            )
    if shape is not None and actual_shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {actual_shape}")
    if dtype is not None:
        if (
            isinstance(dtype, Iterable)
            and not isinstance(dtype, (str, bytes, type))
            and not hasattr(dtype, "dtype")
        ):
            valid_dtypes = tuple(dtype)
        else:
            valid_dtypes = (dtype,)
        require_dtype(value, valid_dtypes, name=f"{name}.dtype")
    return actual_shape


def require_layout(name: str, value: str, supported: tuple[str, ...]) -> str:
    """Validate an explicit public axis-order string."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if value not in supported:
        choices = ", ".join(repr(item) for item in supported)
        raise ValueError(f"{name} must be one of ({choices}), got {value!r}")
    return value


def _row_major_spec(rank: int, mode: tuple[int, ...]) -> TensorSpec:
    return TensorSpec(layout=tuple(reversed(range(rank))), mode=mode)


def gemm_a_tensor_spec(layout: str) -> TensorSpec:
    return _row_major_spec(3, gemm_a_mode(layout))


def gemm_b_tensor_spec(layout: str) -> TensorSpec:
    return _row_major_spec(3, gemm_b_mode(layout))


def gemm_c_tensor_spec(layout: str, *, name: str = "c_layout") -> TensorSpec:
    return _row_major_spec(
        3, gemm_output_mode(require_layout(name, layout, ("LMN", "LNM")), name=name)
    )


def block_scale_tensor_spec() -> TensorSpec:
    """Describe public row-major ``L,tiles,tiles,32,4,4`` scale arrays."""

    return _row_major_spec(6, BLOCK_SCALE_MODE)


def probability_tensor_spec() -> TensorSpec:
    """Map public row-major ``(L, 1, M)`` to canonical ``(M, 1, L)``."""

    return _row_major_spec(3, (2, 1, 0))


def grouped_bias_tensor_spec() -> TensorSpec:
    """Map public row-major ``(L, N)`` to canonical ``(N, L)``."""

    return _row_major_spec(2, (1, 0))


def grouped_workspace_tensor_spec() -> TensorSpec:
    return TensorSpec(ptr_assumed_align=128)


@dataclass(frozen=True)
class _SampleDesc(JaxTensorDesc):
    """JAX descriptor retaining the native CUTLASS lowering specification."""

    tensor_spec: Any = None
    jax_mode: tuple[int, ...] = field(default_factory=tuple)

    @property
    def array_shape(self) -> tuple[int, ...]:
        return to_public_axes(self.shape, self.jax_mode)


def _make_desc(
    value: Any,
    *,
    tensor_spec: TensorSpec | None,
    name: str,
    init_value: bool | int | float | None = None,
) -> _SampleDesc:
    public_shape = require_array(value, name=name)
    mode = normalize_mode(
        len(public_shape), None if tensor_spec is None else tensor_spec.mode
    )
    desc = JaxApiBase._to_tensor_desc(
        value,
        name,
        mode=mode,
        init_value=init_value,
    )
    return _SampleDesc(
        dtype=desc.dtype,
        shape=desc.shape,
        stride=desc.stride,
        stride_order=desc.stride_order,
        name=desc.name,
        init_value=desc.init_value,
        tensor_spec=tensor_spec,
        jax_mode=mode,
    )


def make_buffer_desc(
    name: str,
    shape: Sequence[int],
    dtype: Any,
    *,
    tensor_spec: TensorSpec | None = None,
    init_value: bool | int | float | None = None,
) -> _SampleDesc:
    """Describe an inferred JAX output or workspace buffer."""

    if not name:
        raise ValueError("buffer descriptor name must not be empty")
    return _make_desc(
        jax.ShapeDtypeStruct(tuple(shape), dtype),
        tensor_spec=tensor_spec,
        name=name,
        init_value=init_value,
    )


def as_gemm_tensor_desc(name: str, value: Any, tensor_spec: TensorSpec) -> _SampleDesc:
    if isinstance(value, _SampleDesc):
        if value.jax_mode != normalize_mode(value.ndim, tensor_spec.mode):
            raise ValueError(
                f"{name} descriptor mode does not match the requested layout"
            )
        return value
    return _make_desc(value, tensor_spec=tensor_spec, name=name)


class ApiBaseJax(JaxApiBase):
    """Sample-signature-bound grouped GEMM callable using ``JaxApiBase``."""

    def __init__(self) -> None:
        self._is_supported = False

    def check_support(self) -> bool:
        if not self._is_supported:
            self._check_support()
            self._is_supported = True
        return True

    def _check_support(self) -> None:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.check_support()
        return self._call_impl(*args, **kwargs)

    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def make_tensor_desc(
        self, value: Any, *, tensor_spec: TensorSpec | None = None, name: str = ""
    ) -> _SampleDesc:
        return _make_desc(value, tensor_spec=tensor_spec, name=name or "value")

    def make_optional_tensor_desc(
        self,
        value: Any | None,
        *,
        tensor_spec: TensorSpec | None = None,
        name: str = "",
    ) -> _SampleDesc | None:
        return (
            None
            if value is None
            else self.make_tensor_desc(value, tensor_spec=tensor_spec, name=name)
        )

    @staticmethod
    def as_optional_dtype(value: Any | None) -> Any | None:
        return None if value is None else as_dtype(value)

    @staticmethod
    def freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
        # JAX captures the current configuration while tracing. Keep ordinary
        # Python state so advanced users may reconfigure an un-jitted class
        # instance and let the surrounding JIT establish its own cache key.
        return dict(values)

    def check_tensor_signatures(
        self,
        expected: Mapping[str, _SampleDesc | None],
        values: Mapping[str, Any],
    ) -> None:
        for name, expected_desc in expected.items():
            value = values[name]
            if value is None or expected_desc is None:
                if value is not None or expected_desc is not None:
                    raise ValueError(
                        f"{name} presence does not match the sample signature"
                    )
                continue
            actual_shape = require_array(value, name=name)
            if actual_shape != expected_desc.array_shape:
                raise ValueError(
                    f"{name} tensor shape mismatch: expected {expected_desc.array_shape}, got {actual_shape}"
                )
            if as_dtype(value) != as_dtype(expected_desc.dtype):
                raise ValueError(
                    f"{name} tensor dtype mismatch: expected {expected_desc.dtype}, got {value.dtype}"
                )


class _GroupedKernelCaller(JaxApiBase):
    def check_support(self) -> bool:
        return True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("_GroupedKernelCaller is an internal lowering helper")


_CALLER = _GroupedKernelCaller()


def call_cutedsl(
    fn: Any,
    inputs: Sequence[Any],
    *,
    outputs: Sequence[JaxTensorDesc],
    output_seeds: Sequence[Any | None] | None = None,
    workspaces: Sequence[JaxTensorDesc] = (),
    input_specs: Sequence[TensorSpec | None] | None = None,
    static_args: Mapping[str, Any] | None = None,
    allow_cuda_graph: bool = True,
    use_static_tensors: bool = True,
) -> tuple[Any, ...]:
    """Lower a grouped kernel with canonical stream/input/output/workspace order."""

    static = dict(static_args or {})
    cluster_shape = static.get("cluster_shape_mn")
    margin = int(static.pop("cluster_overlap_margin", 0))

    output_specs = tuple(outputs)
    workspace_specs = tuple(workspaces)
    if output_seeds is None:
        supplied_output_seeds = (None,) * len(output_specs)
    else:
        supplied_output_seeds = tuple(output_seeds)
        if len(supplied_output_seeds) != len(output_specs):
            raise ValueError(
                f"Expected {len(output_specs)} output seeds, "
                f"got {len(supplied_output_seeds)}"
            )

    if any(0 in desc.shape for desc in output_specs):
        results = []
        for desc, seed in zip(output_specs, supplied_output_seeds):
            spec = getattr(desc, "tensor_spec", None)
            mode = None if spec is None else spec.mode
            if seed is not None:
                if desc.init_value is not None:
                    raise ValueError(
                        f"{desc.name} cannot have both an explicit seed and init_value"
                    )
                _CALLER._check_tensor_signature(seed, desc, mode=mode)
                results.append(seed)
                continue
            metadata = _CALLER._materialize_tensor_desc(
                desc,
                mode=mode,
            )
            if desc.init_value is None:
                results.append(jnp.empty(metadata.shape, dtype=metadata.dtype))
            else:
                results.append(
                    jnp.full(
                        metadata.shape,
                        desc.init_value,
                        dtype=metadata.dtype,
                    )
                )
        return tuple(results)

    if cluster_shape is not None:
        static["max_active_clusters"] = _CALLER._get_max_active_clusters(
            int(cluster_shape[0]) * int(cluster_shape[1]),
            overlap_margin=margin,
        )

    def launch(stream: Any, *args: Any) -> None:
        fn(stream, *args, **static)

    compute_capability = _CALLER._resolve_compute_capability(
        None,
        SUPPORTED_COMPUTE_CAPABILITIES,
        "grouped GEMM",
    )
    return _CALLER._call_kernel(
        tuple(inputs),
        launch=launch,
        output_descs=output_specs,
        output_seeds=supplied_output_seeds,
        workspace_descs=workspace_specs,
        input_spec=None if input_specs is None else tuple(input_specs),
        output_spec=tuple(getattr(desc, "tensor_spec", None) for desc in output_specs),
        workspace_spec=tuple(
            getattr(desc, "tensor_spec", None) for desc in workspace_specs
        ),
        allow_cuda_graph=allow_cuda_graph,
        compile_options=compile_options_for_target(compute_capability),
        use_static_tensors=use_static_tensors,
    )


def block_scale_shape(
    rows: int, k: int, batch: int, sf_vec_size: int
) -> tuple[int, ...]:
    """Return the public row-major scale shape used by grouped JAX APIs."""

    return to_public_axes(
        _canonical_block_scale_shape(rows, k, batch, sf_vec_size),
        BLOCK_SCALE_MODE,
    )


def require_grouped_gemm_inputs(
    a_tensor: Any,
    b_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    *,
    max_experts: int = MAX_EXPERTS,
    valid_ab_dtypes: Iterable[Any] | None = None,
) -> tuple[int, int, int, int, Any]:
    """Validate canonical grouped GEMM matrix metadata."""

    if valid_ab_dtypes is None:
        valid_ab_dtypes = (
            jnp.float4_e2m1fn,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
        )
    a_shape = require_array(a_tensor, name="a_tensor", rank=3, dtype=valid_ab_dtypes)
    b_shape = require_array(b_tensor, name="b_tensor", rank=3, dtype=as_dtype(a_tensor))
    m, k, a_batch = a_shape
    n, b_k, experts = b_shape
    if a_batch != 1 or b_k != k:
        raise ValueError(
            f"Grouped GEMM expects A=(M,K,1) and B=(N,K,L), got {a_shape} and {b_shape}"
        )
    if m < 0 or any(value <= 0 for value in (n, k, experts)):
        raise ValueError(
            "Grouped GEMM requires M >= 0 and positive N, K, and L; "
            f"got M={m}, N={n}, K={k}, L={experts}"
        )
    if experts > max_experts:
        raise ValueError(
            f"The number of experts must be at most {max_experts}, got {experts}"
        )
    require_array(
        padded_offsets, name="padded_offsets", shape=(experts,), dtype=jnp.int32
    )
    require_array(
        alpha_tensor, name="alpha_tensor", shape=(experts,), dtype=jnp.float32
    )
    return m, n, k, experts, as_dtype(a_tensor)


def require_grouped_block_scales(
    sfa_tensor: Any,
    sfb_tensor: Any,
    *,
    m: int,
    n: int,
    k: int,
    experts: int,
    sf_vec_size: int,
    valid_dtypes: Iterable[Any],
) -> Any:
    require_array(
        sfa_tensor,
        name="sfa_tensor",
        shape=block_scale_shape(m, k, 1, sf_vec_size),
        dtype=valid_dtypes,
    )
    sf_dtype = as_dtype(sfa_tensor)
    require_array(
        sfb_tensor,
        name="sfb_tensor",
        shape=block_scale_shape(n, k, experts, sf_vec_size),
        dtype=sf_dtype,
    )
    return sf_dtype


def require_grouped_fp8_scales(
    sfa_tensor: Any,
    sfb_tensor: Any,
    *,
    m: int,
    n: int,
    k: int,
    experts: int,
    sf_vec_size: int,
) -> Any:
    return require_grouped_block_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        experts=experts,
        sf_vec_size=sf_vec_size,
        valid_dtypes=(jnp.float8_e8m0fnu,),
    )


def require_grouped_input_scales(
    sfa_tensor: Any,
    sfb_tensor: Any,
    *,
    m: int,
    n: int,
    k: int,
    experts: int,
    sf_vec_size: int,
    ab_dtype: Any,
) -> Any:
    """Validate MXFP8, MXFP4, or NVFP4 scale metadata.

    Native JAX FP4 arrays use logical element shapes.  This helper deliberately
    does not accept raw ``uint8`` payloads or emulate Torch's packed-storage
    reinterpretation.
    """

    if is_fp4_dtype(ab_dtype):
        if sf_vec_size not in SF_VEC_SIZES:
            raise ValueError(
                f"FP4 grouped GEMM requires sf_vec_size in {SF_VEC_SIZES}, got {sf_vec_size}"
            )
        sf_dtype = require_grouped_block_scales(
            sfa_tensor,
            sfb_tensor,
            m=m,
            n=n,
            k=k,
            experts=experts,
            sf_vec_size=sf_vec_size,
            valid_dtypes=(jnp.float8_e8m0fnu, jnp.float8_e4m3fn),
        )
        if sf_dtype == jnp.dtype(jnp.float8_e4m3fn) and sf_vec_size != 16:
            raise ValueError("float8_e4m3fn scales require sf_vec_size=16")
        return sf_dtype

    if is_fp8_dtype(ab_dtype):
        if sf_vec_size != FP8_SF_VEC_SIZE:
            raise ValueError(
                f"FP8 grouped GEMM requires sf_vec_size={FP8_SF_VEC_SIZE}, got {sf_vec_size}"
            )
        return require_grouped_fp8_scales(
            sfa_tensor,
            sfb_tensor,
            m=m,
            n=n,
            k=k,
            experts=experts,
            sf_vec_size=sf_vec_size,
        )

    raise ValueError(f"Unsupported grouped GEMM input dtype {as_dtype(ab_dtype)}")


def require_grouped_vector(
    name: str, tensor: Any, *, length: int, dtype: Any = None
) -> Any:
    require_array(
        tensor,
        name=name,
        shape=(length,),
        dtype=jnp.float32 if dtype is None else dtype,
    )
    return as_dtype(tensor)


def require_grouped_probability(name: str, tensor: Any, *, m: int) -> None:
    require_array(tensor, name=name, shape=(1, 1, m), dtype=jnp.float32)


def require_16_byte_extent(name: str, elements: int, dtype: Any) -> None:
    from .._jax.datatypes import jax_to_cudnn_dtype

    require_contiguous_alignment(
        name, elements, data_type_bits(jax_to_cudnn_dtype(dtype))
    )


def require_contiguous_alignment(name: str, elements: int, element_bits: int) -> None:
    if elements * element_bits % 128:
        raise ValueError(f"{name} contiguous extent must span a multiple of 16 bytes")


def require_grouped_mma_tiler(
    value: Sequence[int],
    *,
    allowed_m: tuple[int, ...] = (64, 128, 256),
    allowed_n: tuple[int, ...] = (128, 256),
) -> tuple[int, int]:
    value = tuple(value)
    if len(value) != 2 or value[0] not in allowed_m or value[1] not in allowed_n:
        raise ValueError(f"Unsupported grouped GEMM mma_tiler_mn {value}")
    return value


def require_grouped_cluster_shape(
    value: Sequence[int], *, mma_tiler_mn: tuple[int, int]
) -> tuple[int, int]:
    value = tuple(value)
    if len(value) != 2 or any(item <= 0 or item & (item - 1) for item in value):
        raise ValueError(
            f"cluster_shape_mn entries must be positive powers of two, got {value}"
        )
    if any(item > 4 for item in value) or value[0] * value[1] > 16:
        raise ValueError(f"cluster_shape_mn product must be at most 16, got {value}")
    cta_group = 2 if mma_tiler_mn[0] == TWO_CTA_MMA_TILER_M else 1
    if value[0] % cta_group:
        raise ValueError(
            f"cluster_shape_mn[0] must be divisible by {cta_group}, got {value[0]}"
        )
    cluster_m = value[0] // cta_group * mma_tiler_mn[0]
    if cluster_m not in (128, 256):
        raise ValueError(
            f"Grouped GEMM cluster M tile must be 128 or 256, got {cluster_m}"
        )
    return value


def dense_workspace_bytes(use_dynamic_sched: bool) -> int:
    return 4 if use_dynamic_sched else 0


def grouped_wgrad_workspace_bytes(expert_cnt: int, input_order: str) -> int:
    from .moe_utils import MoEWeightMode, WGradInputOrder, WgradSfTensormapConstructor

    return WgradSfTensormapConstructor.get_workspace_size(
        WGradInputOrder(input_order),
        MoEWeightMode.DENSE,
        expert_cnt,
    )


def normalize_wgrad_input_order(input_order: Any) -> Any:
    from .moe_utils import WGradInputOrder

    if isinstance(input_order, WGradInputOrder):
        return input_order
    try:
        return WGradInputOrder(input_order)
    except ValueError:
        choices = ", ".join(order.value for order in WGradInputOrder)
        raise ValueError(
            f"input_order must be one of {{{choices}}}, got {input_order!r}"
        ) from None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


__all__ = [
    "ApiBaseJax",
    "FIX_PAD_SIZE",
    "FP8_SF_VEC_SIZE",
    "HADAMARD_SIZE",
    "MAX_EXPERTS",
    "SF_VEC_SIZES",
    "TWO_CTA_MMA_TILER_M",
    "TupleDict",
    "as_dtype",
    "as_gemm_tensor_desc",
    "block_scale_shape",
    "block_scale_tensor_spec",
    "call_cutedsl",
    "ceil_div",
    "dense_workspace_bytes",
    "gemm_a_tensor_spec",
    "gemm_b_tensor_spec",
    "gemm_c_tensor_spec",
    "grouped_bias_tensor_spec",
    "grouped_wgrad_workspace_bytes",
    "grouped_workspace_tensor_spec",
    "is_fp4_dtype",
    "is_fp8_dtype",
    "is_low_precision_output_dtype",
    "make_buffer_desc",
    "normalize_wgrad_input_order",
    "probability_tensor_spec",
    "require_16_byte_extent",
    "require_array",
    "require_contiguous_alignment",
    "require_dtype",
    "require_grouped_block_scales",
    "require_grouped_cluster_shape",
    "require_grouped_fp8_scales",
    "require_grouped_gemm_inputs",
    "require_grouped_input_scales",
    "require_grouped_mma_tiler",
    "require_grouped_probability",
    "require_grouped_vector",
    "require_layout",
]
