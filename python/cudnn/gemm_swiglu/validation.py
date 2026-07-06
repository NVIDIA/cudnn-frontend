# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation for block-scaled GEMM + SwiGLU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..api_base import TensorDesc, canonical_dtype_name
from ..gemm_validation import (
    block_scale_shape,
    require_block_scale_shapes,
    require_contiguous_alignment,
    require_gemm_shapes,
    require_shape,
)

_AB_DTYPES = frozenset({"float4_e2m1fn", "float8_e4m3fn", "float8_e5m2"})
_SCALE_DTYPES = frozenset({"float8_e4m3fn", "float8_e8m0fnu"})
_OUTPUT_DTYPES = frozenset(
    {
        "bfloat16",
        "float16",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
    }
)


@dataclass(frozen=True)
class QuantizedGemmSwigluPlan:
    """Validated logical signature for block-scaled GEMM + SwiGLU."""

    m: int
    n: int
    k: int
    batch: int
    output_n: int
    ab_dtype_name: str
    scale_dtype_name: str
    ab12_dtype_name: str
    c_dtype_name: str
    a_major: str
    b_major: str
    output_major: str
    generate_amax: bool
    generate_sfc: bool

    @property
    def ab12_shape(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.batch)

    @property
    def c_shape(self) -> tuple[int, int, int]:
        return (self.m, self.output_n, self.batch)

    @property
    def requires_amax(self) -> bool:
        return self.ab_dtype_name == "float4_e2m1fn" and self.c_dtype_name == "bfloat16"


def _logical_dtype_name(desc: TensorDesc) -> str:
    if getattr(desc, "packing", "native") == "fp4x2":
        return "float4_e2m1fn"
    return desc.dtype_name


def _logical_element_bits(desc: TensorDesc) -> int:
    if _logical_dtype_name(desc) == "float4_e2m1fn":
        return 4
    if desc.element_bits is None:
        raise ValueError(f"Cannot determine the element width of {desc.name or desc.dtype_name}")
    return desc.element_bits


def _require_supported_dtype(name: str, actual: str, supported: frozenset[str]) -> None:
    if actual not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"{name} dtype must be one of {{{choices}}}, got {actual}")


def _require_same_storage_dtype(name: str, lhs: TensorDesc, rhs: TensorDesc) -> None:
    lhs_storage = getattr(lhs, "storage_dtype_name", lhs.dtype_name)
    rhs_storage = getattr(rhs, "storage_dtype_name", rhs.dtype_name)
    lhs_packing = getattr(lhs, "packing", "native")
    rhs_packing = getattr(rhs, "packing", "native")
    if (lhs_storage, lhs_packing) != (rhs_storage, rhs_packing):
        lhs_label = lhs_storage if lhs_packing == "native" else f"{lhs_storage}/{lhs_packing}"
        rhs_label = rhs_storage if rhs_packing == "native" else f"{rhs_storage}/{rhs_packing}"
        raise ValueError(f"{name} dtypes must match, got {lhs_label} and {rhs_label}")


def _require_compact_major(
    name: str,
    desc: TensorDesc,
    layouts: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> str:
    for major, (stride, stride_order) in layouts.items():
        if desc.stride == stride and desc.stride_order == stride_order:
            return major
    expected = [stride for stride, _ in layouts.values()]
    raise ValueError(f"{name} tensor must use one of the compact strides {expected}, got " f"stride {desc.stride} with order {desc.stride_order}")


def _require_singleton(name: str, desc: TensorDesc) -> None:
    if tuple(desc.shape) not in {(1,), (1, 1, 1)}:
        raise ValueError(f"{name} must have shape (1,) or (1, 1, 1), got {tuple(desc.shape)}")


def validate_quantized_gemm_swiglu(
    a: TensorDesc,
    b: TensorDesc,
    ab12: TensorDesc,
    c: TensorDesc,
    *,
    sfa: TensorDesc | None,
    sfb: TensorDesc | None,
    amax: TensorDesc | None,
    sfc: TensorDesc | None,
    norm_const: TensorDesc | None,
    acc_dtype: Any,
    output_n: int,
    sf_vec_size: int,
    supported_sf_vec_sizes: Sequence[int],
    mma_tiler_mn: Sequence[int],
) -> QuantizedGemmSwigluPlan:
    """Validate the block-scaled contract shared by Torch and JAX adapters."""

    if sfa is None or sfb is None:
        raise ValueError("sfa_tensor and sfb_tensor are required for quantized GEMM + SwiGLU")

    m, n, k, batch = require_gemm_shapes(a.shape, b.shape)
    require_shape("AB12", ab12.shape, (m, n, batch))
    require_shape("C", c.shape, (m, output_n, batch))

    _require_same_storage_dtype("a_tensor and b_tensor", a, b)
    ab_dtype_name = _logical_dtype_name(a)
    _require_supported_dtype("a_tensor and b_tensor", ab_dtype_name, _AB_DTYPES)
    if canonical_dtype_name(acc_dtype) != "float32":
        raise ValueError(f"Accumulator dtype must be float32, got {canonical_dtype_name(acc_dtype)}")

    supported_sf_vec_sizes = tuple(supported_sf_vec_sizes)
    if sf_vec_size not in supported_sf_vec_sizes:
        raise ValueError(f"sf_vec_size must be one of {supported_sf_vec_sizes}, got {sf_vec_size}")

    _require_same_storage_dtype("sfa_tensor and sfb_tensor", sfa, sfb)
    scale_dtype_name = _logical_dtype_name(sfa)
    _require_supported_dtype("sfa_tensor and sfb_tensor", scale_dtype_name, _SCALE_DTYPES)
    require_block_scale_shapes(
        sfa.shape,
        sfb.shape,
        m=m,
        n=n,
        k=k,
        batch=batch,
        sf_vec_size=sf_vec_size,
    )

    ab12_dtype_name = _logical_dtype_name(ab12)
    c_dtype_name = _logical_dtype_name(c)
    _require_supported_dtype("AB12", ab12_dtype_name, _OUTPUT_DTYPES)
    _require_supported_dtype("C", c_dtype_name, _OUTPUT_DTYPES)

    if sfc is not None:
        _require_same_storage_dtype("sfa_tensor and sfc_tensor", sfa, sfc)
        require_shape(
            "sfc_tensor",
            sfc.shape,
            block_scale_shape(m, output_n, batch, sf_vec_size),
        )
    if norm_const is not None:
        require_shape("norm_const_tensor", norm_const.shape, (1,))
    if amax is not None:
        _require_singleton("amax_tensor", amax)
        if _logical_dtype_name(amax) != "float32":
            raise ValueError(f"amax_tensor dtype must be float32, got {_logical_dtype_name(amax)}")

    is_fp4 = ab_dtype_name == "float4_e2m1fn"
    is_fp8 = ab_dtype_name.startswith("float8_")
    c_is_fp8 = c_dtype_name.startswith("float8_")
    ab12_is_fp8 = ab12_dtype_name.startswith("float8_")

    if is_fp4 and c_is_fp8:
        raise ValueError("FP4 A and B are not compatible with an FP8 C dtype")
    if c_is_fp8 and (sfc is None or norm_const is None):
        raise ValueError("sfc_tensor and norm_const_tensor are required when C is FP8")
    if is_fp4 and c_dtype_name == "bfloat16" and amax is None:
        raise ValueError("amax_tensor is required when A and B are FP4 and C is bfloat16")
    if c_dtype_name == "float32" and ab12_dtype_name == "float32":
        raise NotImplementedError("float32 C with float32 AB12 is disabled because of a kernel bug")

    if is_fp8 and not (scale_dtype_name == "float8_e8m0fnu" and sf_vec_size == 32):
        raise ValueError("FP8 A and B require float8_e8m0fnu scale factors and sf_vec_size=32")
    if is_fp4 and scale_dtype_name == "float8_e4m3fn" and sf_vec_size == 32:
        raise ValueError("FP4 A and B do not support float8_e4m3fn scale factors with sf_vec_size=32")
    if is_fp8 and (c_is_fp8 or ab12_is_fp8 or ab12_dtype_name == "float32"):
        raise ValueError("MXFP8 requires float16 or bfloat16 AB12 and a non-FP8 C dtype")

    a_major = _require_compact_major(
        "A",
        a,
        {
            "m": ((1, m, m * k), (0, 1, 2)),
            "k": ((k, 1, m * k), (1, 0, 2)),
        },
    )
    b_major = _require_compact_major(
        "B",
        b,
        {
            "n": ((1, n, n * k), (0, 1, 2)),
            "k": ((k, 1, n * k), (1, 0, 2)),
        },
    )
    ab12_major = _require_compact_major(
        "AB12",
        ab12,
        {
            "m": ((1, m, m * n), (0, 1, 2)),
            "n": ((n, 1, m * n), (1, 0, 2)),
        },
    )
    c_major = _require_compact_major(
        "C",
        c,
        {
            "m": ((1, m, m * output_n), (0, 1, 2)),
            "n": ((output_n, 1, m * output_n), (1, 0, 2)),
        },
    )
    if ab12_major != c_major:
        raise ValueError(f"AB12 and C must use the same major mode, got {ab12_major} and {c_major}")
    if is_fp4 and (a_major, b_major, ab12_major) != ("k", "k", "n"):
        raise ValueError("FP4 requires k-major A and B and n-major outputs, got " f"{a_major}-major, {b_major}-major, and {ab12_major}-major")

    mma_m, mma_n = tuple(mma_tiler_mn)
    if m % mma_m or n % mma_n:
        raise ValueError(f"M and N must be divisible by mma_tiler_mn {tuple(mma_tiler_mn)}, got M={m}, N={n}")

    require_contiguous_alignment("A", m if a_major == "m" else k, _logical_element_bits(a))
    require_contiguous_alignment("B", n if b_major == "n" else k, _logical_element_bits(b))
    require_contiguous_alignment("AB12", m if ab12_major == "m" else n, _logical_element_bits(ab12))
    require_contiguous_alignment("C", m if c_major == "m" else output_n, _logical_element_bits(c))

    return QuantizedGemmSwigluPlan(
        m=m,
        n=n,
        k=k,
        batch=batch,
        output_n=output_n,
        ab_dtype_name=ab_dtype_name,
        scale_dtype_name=scale_dtype_name,
        ab12_dtype_name=ab12_dtype_name,
        c_dtype_name=c_dtype_name,
        a_major=a_major,
        b_major=b_major,
        output_major=ab12_major,
        generate_amax=amax is not None,
        generate_sfc=sfc is not None and norm_const is not None,
    )


__all__ = ["QuantizedGemmSwigluPlan", "validate_quantized_gemm_swiglu"]
