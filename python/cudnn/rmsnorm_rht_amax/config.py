# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral launch configuration for RMSNorm + RHT + amax."""

from typing import Optional, Tuple

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


__all__ = [
    "DEFAULT_NUM_THREADS_BY_N",
    "RPC_CANDIDATES",
    "TARGET_MIN_CTAS",
    "best_num_threads",
    "pick_rows_per_cta",
    "resolve_launch_config",
]
