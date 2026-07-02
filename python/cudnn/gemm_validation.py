# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation rules shared by dense GEMM frontends."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Optional


def ceil_div(value: int, divisor: int) -> int:
    """Return ``ceil(value / divisor)`` for positive integers."""

    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (value + divisor - 1) // divisor


def require_shape(name: str, actual: Sequence[int], expected: Sequence[int]) -> tuple[int, ...]:
    """Require an exact logical tensor shape and return it as a tuple."""

    actual = tuple(actual)
    expected = tuple(expected)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")
    return actual


def require_gemm_shapes(
    a_shape: Sequence[int],
    b_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    """Validate dense GEMM input shapes and return ``M, N, K, L``."""

    a_shape = tuple(a_shape)
    b_shape = tuple(b_shape)
    if len(a_shape) != 3:
        raise ValueError(f"a_tensor must have rank 3, got shape {a_shape}")
    if len(b_shape) != 3:
        raise ValueError(f"b_tensor must have rank 3, got shape {b_shape}")

    m, k, batch = a_shape
    n, b_k, b_batch = b_shape
    if (b_k, b_batch) != (k, batch):
        raise ValueError("a_tensor and b_tensor must have matching K and L dimensions, " f"got {a_shape} and {b_shape}")

    dimensions = {"M": m, "N": n, "K": k, "L": batch}
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("GEMM dimensions must be positive, got " + ", ".join(nonpositive))
    return m, n, k, batch


def block_scale_shape(
    rows: int,
    k: int,
    batch: int,
    sf_vec_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Return the six-dimensional native block-scale shape."""

    if sf_vec_size <= 0:
        raise ValueError(f"sf_vec_size must be positive, got {sf_vec_size}")
    scale_k_tiles = ceil_div(ceil_div(k, sf_vec_size), 4)
    return (32, 4, ceil_div(rows, 128), 4, scale_k_tiles, batch)


def require_block_scale_shapes(
    sfa_shape: Sequence[int],
    sfb_shape: Sequence[int],
    *,
    m: int,
    n: int,
    k: int,
    batch: int,
    sf_vec_size: int,
    sfa_name: str = "sfa_tensor",
    sfb_name: str = "sfb_tensor",
) -> None:
    """Validate the A and B block-scale tensor shapes."""

    require_shape(sfa_name, sfa_shape, block_scale_shape(m, k, batch, sf_vec_size))
    require_shape(sfb_name, sfb_shape, block_scale_shape(n, k, batch, sf_vec_size))


def _format_values(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def require_mma_tiler(
    mma_tiler_mn: Sequence[int],
    *,
    allowed_m: Collection[int],
    allowed_n: Collection[int],
) -> tuple[int, int]:
    """Validate an MMA tile against explicit M and N domains."""

    mma_tiler_mn = tuple(mma_tiler_mn)
    allowed_m = tuple(sorted(allowed_m))
    allowed_n = tuple(sorted(allowed_n))
    if len(mma_tiler_mn) != 2 or mma_tiler_mn[0] not in allowed_m or mma_tiler_mn[1] not in allowed_n:
        raise ValueError("mma_tiler_mn must have M in " f"{_format_values(allowed_m)} and N in {_format_values(allowed_n)}, " f"got {mma_tiler_mn}")
    return mma_tiler_mn


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""

    return value > 0 and value & (value - 1) == 0


def require_cluster_shape(
    cluster_shape_mn: Sequence[int],
    *,
    mma_m: int,
    two_cta_mma_m: int,
    max_ctas: int,
    max_dimension: Optional[int] = None,
) -> tuple[int, int]:
    """Validate a cluster shape and its 2-CTA MMA requirement."""

    cluster_shape_mn = tuple(cluster_shape_mn)
    if len(cluster_shape_mn) != 2:
        raise ValueError(f"cluster_shape_mn must have two dimensions, got {cluster_shape_mn}")

    cluster_m, cluster_n = cluster_shape_mn
    dimensions_valid = is_power_of_two(cluster_m) and is_power_of_two(cluster_n)
    if max_dimension is not None:
        dimensions_valid = dimensions_valid and cluster_m <= max_dimension and cluster_n <= max_dimension
    if not dimensions_valid or cluster_m * cluster_n > max_ctas:
        dimension_limit = "" if max_dimension is None else f", each at most {max_dimension}"
        raise ValueError(
            "cluster_shape_mn dimensions must be positive powers of two" f"{dimension_limit} with product at most {max_ctas}, got {cluster_shape_mn}"
        )
    if mma_m == two_cta_mma_m and cluster_m % 2:
        raise ValueError("cluster_shape_mn[0] must be divisible by 2 with a " f"{two_cta_mma_m}-wide M tile")
    return cluster_shape_mn


def require_contiguous_alignment(
    name: str,
    elements: int,
    element_bits: int,
    *,
    alignment_bytes: int = 16,
) -> None:
    """Require a contiguous tensor extent to meet a byte alignment."""

    if element_bits <= 0:
        raise ValueError(f"element_bits must be positive, got {element_bits}")
    if alignment_bytes <= 0:
        raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")
    if elements * element_bits % (alignment_bytes * 8):
        raise ValueError(f"{name}'s contiguous extent must be {alignment_bytes}-byte aligned, " f"got {elements} elements at {element_bits} bits each")


def require_full_mma_rows(
    m: int,
    mma_m: int,
    *,
    cta_group_size: int = 1,
    reason: str = "",
) -> None:
    """Require M to contain complete per-CTA rows for an MMA tile."""

    if cta_group_size <= 0 or mma_m % cta_group_size:
        raise ValueError(f"cta_group_size must divide mma_m, got {cta_group_size} and {mma_m}")
    rows_per_cta = mma_m // cta_group_size
    if m % rows_per_cta:
        suffix = "" if not reason else f" because {reason}"
        raise ValueError(f"M must be divisible by {rows_per_cta} (CTA_M for TILE_M={mma_m})" f"{suffix}, got {m}")


def resolve_max_active_clusters(max_active_clusters: int, overlap_margin: int) -> int:
    """Apply the configured cluster overlap margin and validate the result."""

    resolved = max_active_clusters - overlap_margin
    if resolved <= 0:
        raise ValueError("max_active_clusters must be positive after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
    return resolved
