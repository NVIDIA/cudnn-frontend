# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral validation for RMSNorm + RHT + amax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cudnn.api_base import TensorDesc

DEFAULT_NUM_THREADS_BY_N = {
    2048: 128,
    4096: 256,
    7168: 128,
    8192: 512,
    16384: 1024,
    32768: 512,
}
RPC_CANDIDATES = (2, 4, 8)
TARGET_MIN_CTAS = 148


@dataclass(frozen=True)
class RmsNormRhtAmaxPlan:
    """Validated, framework-independent operation metadata."""

    m: int
    n: int
    num_threads: int
    rows_per_cta: int
    output_shape: tuple[int, int]
    amax_shape: tuple[int]


def best_num_threads(n: int) -> Optional[int]:
    for num_threads in (1024, 512, 256, 128, 64):
        if n % num_threads != 0:
            continue
        ept = n // num_threads
        if ept >= 8 and ept % 8 == 0:
            return num_threads
    return None


def pick_rows_per_cta(m: int) -> int:
    for rows_per_cta in reversed(RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        num_ctas = m // rows_per_cta
        if num_ctas >= TARGET_MIN_CTAS:
            return rows_per_cta
    return RPC_CANDIDATES[0]


def resolve_launch_config(
    m: int,
    n: int,
    *,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
) -> Tuple[int, int]:
    """Validate dimensions and resolve target-independent launch parameters."""

    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if n % 16 != 0:
        raise ValueError(f"N must be divisible by 16 for the Hadamard block size, got {n}")

    resolved_num_threads = num_threads
    if resolved_num_threads is None:
        resolved_num_threads = DEFAULT_NUM_THREADS_BY_N.get(n, best_num_threads(n))
    if resolved_num_threads is None:
        raise ValueError(f"No valid num_threads found for N={n}")
    if resolved_num_threads <= 0:
        raise ValueError(f"num_threads must be positive, got {resolved_num_threads}")
    if resolved_num_threads % 32 != 0:
        raise ValueError(f"num_threads must be warp-aligned, got {resolved_num_threads}")
    if resolved_num_threads > 1024:
        raise ValueError("num_threads must not exceed the CUDA block size limit, " f"got {resolved_num_threads}")

    resolved_rows_per_cta = rows_per_cta
    if resolved_rows_per_cta is None:
        resolved_rows_per_cta = pick_rows_per_cta(m)
    if resolved_rows_per_cta <= 0:
        raise ValueError(f"rows_per_cta must be positive, got {resolved_rows_per_cta}")
    if m % resolved_rows_per_cta != 0:
        raise ValueError("M must be divisible by rows_per_cta, " f"got M={m}, rows_per_cta={resolved_rows_per_cta}")
    if n % resolved_num_threads != 0:
        raise ValueError(f"N={n} must be divisible by num_threads={resolved_num_threads}")

    ept = n // resolved_num_threads
    if ept < 8 or ept % 8 != 0:
        raise ValueError(f"EPT={ept} must be >= 8 and divisible by 8")

    return resolved_num_threads, resolved_rows_per_cta


def _require_rank(tensor: TensorDesc, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tensor.shape}")


def _require_shape(tensor: TensorDesc, shape: tuple[int, ...], name: str) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tensor.shape}")


def _require_dtype(tensor: TensorDesc, dtype_name: str, name: str) -> None:
    if tensor.dtype_name != dtype_name:
        raise ValueError(f"{name} must have dtype {dtype_name}, got {tensor.dtype_name}")


def validate_rmsnorm_rht_amax(
    x: TensorDesc,
    weight: TensorDesc,
    *,
    output: Optional[TensorDesc] = None,
    amax: Optional[TensorDesc] = None,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
) -> RmsNormRhtAmaxPlan:
    """Validate logical tensor metadata and infer the operation outputs.

    Physical layout and device capability remain adapter responsibilities:
    Torch validates observed strides and its CUDA device, while JAX declares a
    compact custom-call layout with ``cutlass.jax.TensorSpec``.
    """

    _require_rank(x, 2, "X")
    _require_rank(weight, 1, "W")
    _require_dtype(x, "bfloat16", "X")
    _require_dtype(weight, "bfloat16", "W")

    m, n = x.shape
    _require_shape(weight, (n,), "W")
    resolved_num_threads, resolved_rows_per_cta = resolve_launch_config(
        m,
        n,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )

    output_shape = (m, n)
    amax_shape = (m // resolved_rows_per_cta,)
    if output is not None:
        _require_shape(output, output_shape, "O")
        _require_dtype(output, "bfloat16", "O")
    if amax is not None:
        _require_shape(amax, amax_shape, "Amax")
        _require_dtype(amax, "float32", "Amax")

    return RmsNormRhtAmaxPlan(
        m=m,
        n=n,
        num_threads=resolved_num_threads,
        rows_per_cta=resolved_rows_per_cta,
        output_shape=output_shape,
        amax_shape=amax_shape,
    )


__all__ = [
    "DEFAULT_NUM_THREADS_BY_N",
    "RPC_CANDIDATES",
    "RmsNormRhtAmaxPlan",
    "TARGET_MIN_CTAS",
    "best_num_threads",
    "pick_rows_per_cta",
    "resolve_launch_config",
    "validate_rmsnorm_rht_amax",
]
