# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation for dense block-scaled GEMM + amax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..api_base import TensorDesc, canonical_dtype_name
from ..gemm_validation import (
    require_block_scale_shapes,
    require_contiguous_alignment,
    require_gemm_shapes,
    require_shape,
)

_AB_DTYPES = frozenset({"float4_e2m1fn", "float8_e4m3fn", "float8_e5m2"})
_SCALE_DTYPES = frozenset({"float8_e4m3fn", "float8_e8m0fnu"})
_C_DTYPES = frozenset(
    {
        "bfloat16",
        "float16",
        "float32",
        "float4_e2m1fn",
        "float8_e4m3fn",
        "float8_e5m2",
    }
)
_WIDE_C_DTYPES = frozenset({"bfloat16", "float16", "float32"})


@dataclass(frozen=True)
class GemmAmaxPlan:
    """Validated logical signature and layout choices for GEMM + amax."""

    m: int
    n: int
    k: int
    batch: int
    ab_dtype_name: str
    scale_dtype_name: str
    c_dtype_name: str
    a_major: str
    b_major: str
    c_major: str

    @property
    def c_shape(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.batch)

    @property
    def amax_shape(self) -> tuple[int, int, int]:
        return (1, 1, 1)


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


def validate_gemm_amax(
    a: TensorDesc,
    b: TensorDesc,
    sfa: TensorDesc,
    sfb: TensorDesc,
    c: TensorDesc,
    amax: TensorDesc,
    *,
    acc_dtype: Any,
    sf_vec_size: int,
    supported_sf_vec_sizes: Sequence[int],
    mma_tiler_mn: Sequence[int],
) -> GemmAmaxPlan:
    """Validate the kernel-domain contract shared by Torch and JAX.

    Framework adapters remain responsible for narrowing their public dtype
    surface, selecting the target device, and converting these descriptors to
    their execution-specific tensor metadata.
    """

    m, n, k, batch = require_gemm_shapes(a.shape, b.shape)
    _require_same_storage_dtype("a_tensor and b_tensor", a, b)
    ab_dtype_name = _logical_dtype_name(a)
    _require_supported_dtype("a_tensor and b_tensor", ab_dtype_name, _AB_DTYPES)

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
        sfa_name="sfa_tensor",
        sfb_name="sfb_tensor",
    )

    c_dtype_name = _logical_dtype_name(c)
    _require_supported_dtype("C", c_dtype_name, _C_DTYPES)
    require_shape("C", c.shape, (m, n, batch))
    require_shape("amax", amax.shape, (1, 1, 1))
    if _logical_dtype_name(amax) != "float32":
        raise ValueError(f"amax dtype must be float32, got {_logical_dtype_name(amax)}")
    if canonical_dtype_name(acc_dtype) != "float32":
        raise ValueError(f"Accumulator dtype must be float32, got {canonical_dtype_name(acc_dtype)}")

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
    c_major = _require_compact_major(
        "C",
        c,
        {
            "m": ((1, m, m * n), (0, 1, 2)),
            "n": ((n, 1, m * n), (1, 0, 2)),
        },
    )

    if ab_dtype_name == "float4_e2m1fn" and (a_major, b_major) != ("k", "k"):
        raise ValueError("Float4 A and B tensors require k-major layouts, got " f"{a_major}-major and {b_major}-major")
    if c_dtype_name == "float4_e2m1fn" and c_major != "n":
        raise ValueError(f"Float4 C tensors require n-major layout, got {c_major}-major")
    if c_dtype_name == "float4_e2m1fn" and ab_dtype_name != "float4_e2m1fn":
        raise ValueError("Float4 C requires Float4 A and B, got " f"C={c_dtype_name}, A/B={ab_dtype_name}")
    if c_dtype_name.startswith("float8_") and ab_dtype_name.startswith("float8_"):
        raise NotImplementedError("FP8 A/B with FP8 C is unsupported because it fails to launch")

    if scale_dtype_name == "float8_e4m3fn" and sf_vec_size == 32:
        raise ValueError("float8_e4m3fn scale factors do not support sf_vec_size=32")
    if ab_dtype_name.startswith("float8_") and sf_vec_size == 16:
        raise ValueError("FP8 A and B tensors do not support sf_vec_size=16")

    mma_tiler_mn = tuple(mma_tiler_mn)
    if ab_dtype_name == "float4_e2m1fn" and mma_tiler_mn[1] == 256 and k <= 128:
        raise ValueError(f"mma_tiler_mn (X, 256) requires K > 128 for Float4, got {k}")
    if mma_tiler_mn == (128, 256) and sf_vec_size == 16 and c_dtype_name in _WIDE_C_DTYPES:
        raise NotImplementedError("mma_tiler_mn (128, 256), sf_vec_size=16, and a 16/32-bit C dtype " "are unsupported because the kernel fails to launch")

    require_contiguous_alignment("A", m if a_major == "m" else k, _logical_element_bits(a))
    require_contiguous_alignment("B", n if b_major == "n" else k, _logical_element_bits(b))
    require_contiguous_alignment("C", m if c_major == "m" else n, _logical_element_bits(c))

    return GemmAmaxPlan(
        m=m,
        n=n,
        k=k,
        batch=batch,
        ab_dtype_name=ab_dtype_name,
        scale_dtype_name=scale_dtype_name,
        c_dtype_name=c_dtype_name,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
    )


__all__ = ["GemmAmaxPlan", "validate_gemm_amax"]
