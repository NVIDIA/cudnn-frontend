# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation helpers for dense GEMM operations."""

from __future__ import annotations

from operator import index
from typing import Iterable

from .. import data_type
from ..common.tensor_desc import TensorDesc

_STANDARD_MMA_M = (128, 256)
_STANDARD_MMA_N = tuple(range(32, 257, 32))
BLOCK_SCALE_STRIDE_ORDER = (3, 1, 0, 4, 2, 5)

_DATA_TYPE_BITS_BY_NAME = {
    "BOOLEAN": 8,
    "INT4": 4,
    "INT8": 8,
    "UINT8": 8,
    "HALF": 16,
    "BFLOAT16": 16,
    "FLOAT": 32,
    "DOUBLE": 64,
    "INT32": 32,
    "INT64": 64,
    "FP4_E2M1": 4,
    "FP8_E4M3": 8,
    "FP8_E5M2": 8,
    "FP8_E8M0": 8,
    "FAST_FLOAT_FOR_FP8": 32,
}


def _require_tensor_desc(tensor: TensorDesc, label: str) -> None:
    if not isinstance(tensor, TensorDesc):
        raise TypeError(f"{label} must be a TensorDesc, got {type(tensor).__name__}")


def _integer_pair(value: Iterable[int], label: str) -> tuple[int, int]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{label} must contain two integers, got {value!r}") from error

    if len(values) != 2:
        raise ValueError(f"{label} must contain two integers, got {values}")

    normalized = []
    for entry in values:
        if isinstance(entry, bool):
            raise TypeError(f"{label} must contain integers, got {values}")
        try:
            normalized.append(index(entry))
        except TypeError as error:
            raise TypeError(f"{label} must contain integers, got {values}") from error
    return normalized[0], normalized[1]


def _format_values(values: Iterable[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def require_gemm_inputs(a: TensorDesc, b: TensorDesc) -> tuple[int, int, int, int]:
    """Validate canonical ``A[M,K,L]`` and ``B[N,K,L]`` descriptors.

    Returns ``(M, N, K, L)``. Layout and dtype validation are deliberately
    separate because their supported combinations differ between kernels.
    """

    _require_tensor_desc(a, "A")
    _require_tensor_desc(b, "B")
    if a.ndim != 3:
        raise ValueError(f"A must have rank 3, got shape {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"B must have rank 3, got shape {b.shape}")

    m, k, l = a.shape
    n, b_k, b_l = b.shape
    for axis, extent in (("M", m), ("N", n), ("K", k), ("L", l)):
        if extent <= 0:
            raise ValueError(f"{axis} must be positive, got {extent}")

    if (b_k, b_l) != (k, l):
        raise ValueError(f"B shape mismatch: expected (N, {k}, {l}), got {b.shape}")
    return m, n, k, l


def block_scale_shape(
    rows: int,
    k: int,
    batch: int,
    sf_vec_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Return the canonical packed scale-factor shape for a dense GEMM."""

    if isinstance(sf_vec_size, bool):
        raise TypeError(f"sf_vec_size must be an integer, got {sf_vec_size!r}")
    try:
        sf_vec_size = index(sf_vec_size)
    except TypeError as error:
        raise TypeError(f"sf_vec_size must be an integer, got {sf_vec_size!r}") from error
    if sf_vec_size <= 0:
        raise ValueError(f"sf_vec_size must be positive, got {sf_vec_size}")
    row_tiles = (rows + 127) // 128
    scale_k = (k + sf_vec_size - 1) // sf_vec_size
    k_tiles = (scale_k + 3) // 4
    return (32, 4, row_tiles, 4, k_tiles, batch)


def require_block_scale_layout(tensor: TensorDesc, label: str) -> None:
    """Require the packed scale-factor layout used by dense SM100 GEMMs."""

    _require_tensor_desc(tensor, label)
    if not tensor.is_compact(BLOCK_SCALE_STRIDE_ORDER):
        raise ValueError(
            f"{label} must use the packed block-scale layout with stride order "
            f"{BLOCK_SCALE_STRIDE_ORDER}, got stride {tensor.stride} and "
            f"stride order {tensor.stride_order}"
        )


def require_tensor_shape(
    tensor: TensorDesc,
    expected: tuple[int, ...],
    *,
    label: str | None = None,
) -> None:
    """Require a descriptor to have an exact logical shape."""

    tensor_label = label or getattr(tensor, "name", "") or "Tensor"
    _require_tensor_desc(tensor, tensor_label)
    expected = tuple(expected)
    if tensor.shape != expected:
        raise ValueError(f"{tensor_label} must have shape {expected}, got {tensor.shape}")


def require_compact_major(
    tensor: TensorDesc,
    mode0_label: str,
    mode1_label: str,
) -> str:
    """Resolve the compact major mode of a canonical rank-three GEMM tensor.

    Canonical GEMM tensors may make mode 0 or mode 1 contiguous. The batch
    mode (mode 2) must remain outermost. The returned label is supplied by the
    caller, for example ``("m", "k")`` for ``A[M,K,L]``.
    """

    tensor_label = getattr(tensor, "name", "") or "Tensor"
    _require_tensor_desc(tensor, tensor_label)
    if tensor.ndim != 3:
        raise ValueError(f"{tensor_label} must have rank 3, got shape {tensor.shape}")

    labels_by_order = {
        (0, 1, 2): mode0_label,
        (1, 0, 2): mode1_label,
    }
    major = labels_by_order.get(tensor.stride_order)
    if major is None or not tensor.is_compact(tensor.stride_order):
        raise ValueError(f"{tensor_label} must be compact {mode0_label}-major or " f"{mode1_label}-major, got shape {tensor.shape} and stride {tensor.stride}")
    return major


def data_type_bits(dtype: data_type) -> int:
    """Return the storage width of a canonical cuDNN data type."""

    if not isinstance(dtype, data_type):
        raise TypeError(f"dtype must be a cudnn.data_type, got {type(dtype).__name__}")

    for name, bits in _DATA_TYPE_BITS_BY_NAME.items():
        member = getattr(data_type, name, None)
        if member is not None and dtype == member:
            return bits
    raise ValueError(f"Unsupported cuDNN data type {dtype}")


def require_16_byte_alignment(tensor: TensorDesc) -> None:
    """Require the tensor's contiguous extent to cover whole 16-byte units.

    This is the logical-extent requirement used by dense GEMM TMA operands;
    framework adapters remain responsible for the base-pointer alignment.
    ``require_compact_major`` should be called first for GEMM matrices.
    """

    tensor_label = getattr(tensor, "name", "") or "Tensor"
    _require_tensor_desc(tensor, tensor_label)
    if tensor.ndim == 0:
        raise ValueError(f"{tensor_label} must have at least one dimension")

    contiguous_dimension = tensor.stride_order[0]
    bits = data_type_bits(tensor.cudnn_dtype)
    extent = tensor.shape[contiguous_dimension]
    if extent * bits % 128 != 0:
        required_multiple = 128 // bits
        raise ValueError(f"{tensor_label} contiguous extent must be a multiple of " f"{required_multiple} elements for 16-byte alignment, got {extent}")


def require_mma_tiler(
    mma_tiler_mn: Iterable[int],
    *,
    allowed_m: Iterable[int] = _STANDARD_MMA_M,
    allowed_n: Iterable[int] = _STANDARD_MMA_N,
) -> tuple[int, int]:
    """Validate and normalize an ``(M, N)`` MMA tile."""

    mma_m, mma_n = _integer_pair(mma_tiler_mn, "mma_tiler_mn")
    allowed_m = tuple(allowed_m)
    allowed_n = tuple(allowed_n)
    if mma_m not in allowed_m:
        raise ValueError(f"mma_tiler_mn[0] must be in {_format_values(allowed_m)}, got {mma_m}")
    if mma_n not in allowed_n:
        raise ValueError(f"mma_tiler_mn[1] must be in {_format_values(allowed_n)}, got {mma_n}")
    return mma_m, mma_n


def require_cluster_shape(
    cluster_shape_mn: Iterable[int],
    *,
    cta_group_size: int = 1,
    max_cluster_size: int = 16,
) -> tuple[int, int]:
    """Validate and normalize an ``(M, N)`` Blackwell cluster shape."""

    cluster_m, cluster_n = _integer_pair(cluster_shape_mn, "cluster_shape_mn")
    if isinstance(cta_group_size, bool) or not isinstance(cta_group_size, int) or cta_group_size <= 0:
        raise ValueError(f"cta_group_size must be a positive integer, got {cta_group_size!r}")
    if isinstance(max_cluster_size, bool) or not isinstance(max_cluster_size, int) or max_cluster_size <= 0:
        raise ValueError(f"max_cluster_size must be a positive integer, got {max_cluster_size!r}")

    if cluster_m <= 0 or cluster_n <= 0 or cluster_m & (cluster_m - 1) or cluster_n & (cluster_n - 1):
        raise ValueError("cluster_shape_mn entries must be positive powers of two, " f"got {(cluster_m, cluster_n)}")
    if cluster_m * cluster_n > max_cluster_size:
        raise ValueError(f"cluster_shape_mn product must not exceed {max_cluster_size}, " f"got {(cluster_m, cluster_n)}")
    if cluster_m % cta_group_size != 0:
        raise ValueError(f"cluster_shape_mn[0] must be divisible by CTA group size " f"{cta_group_size}, got {cluster_m}")
    return cluster_m, cluster_n


__all__ = [
    "BLOCK_SCALE_STRIDE_ORDER",
    "block_scale_shape",
    "data_type_bits",
    "require_16_byte_alignment",
    "require_block_scale_layout",
    "require_cluster_shape",
    "require_compact_major",
    "require_gemm_inputs",
    "require_mma_tiler",
    "require_tensor_shape",
]
