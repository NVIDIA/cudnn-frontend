# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral SM100 configuration for score-recompute kernels."""

from __future__ import annotations

from dataclasses import dataclass
import math

_SM100_SMEM_BYTES = 228 * 1024


@dataclass(frozen=True)
class SparseScoreKernelConfig:
    """Static launch configuration for ``SparseScoreRecomputeSm100``."""

    m_block_size: int
    n_block_size: int
    k_block_size: int | None
    kv_stage: int
    have_topk_length: bool
    topk_in_smem: bool


@dataclass(frozen=True)
class DenseScoreKernelConfig:
    """Static launch configuration for ``DenseScoreRecomputeSm100``."""

    m_block_size: int
    n_block_size: int
    k_block_size: int | None
    kv_stage: int


def dispatch_sparse_attn_tile_params(
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    *,
    compact: bool,
) -> tuple[int, int, int | None]:
    """Select the tuned SM100 sparse-attention tile."""

    m = qhead_per_kv_head
    if compact:
        if head_dim == 512 and qhead_per_kv_head == 64:
            return m, 64, None
        if head_dim == 512 and qhead_per_kv_head == 128:
            return m, 128, 128
        if head_dim == 576 and qhead_per_kv_head == 64:
            return m, 64, None
        if head_dim == 576 and qhead_per_kv_head == 128:
            return m, 64, 192
        return m, 64, None

    if head_dim == 512 and qhead_per_kv_head == 64:
        return m, 128, 256
    if head_dim == 512 and qhead_per_kv_head == 128 and topk <= 512:
        return m, 128, 128
    if head_dim == 512 and qhead_per_kv_head == 128:
        return m, 128, 256
    if head_dim == 576 and qhead_per_kv_head in (64, 128):
        return m, 128, 192
    return m, 64, None


def resolve_sparse_score_smem_config(
    *,
    score_type: str,
    head_dim: int,
    m_block_size: int,
    n_block_size: int,
    k_block_size: int | None,
    topk: int,
) -> tuple[int, bool]:
    """Return ``(kv_stage, topk_in_smem)`` for an explicit tile choice."""

    if score_type not in ("indexer", "attention"):
        raise ValueError(f"score_type must be 'indexer' or 'attention', got {score_type!r}")
    if head_dim <= 0:
        raise ValueError(f"head dimension must be positive, got {head_dim}")
    if m_block_size <= 0 or n_block_size <= 0:
        raise ValueError(f"m_block_size and n_block_size must be positive, got {m_block_size} and {n_block_size}")
    if topk <= 0 or topk % n_block_size:
        raise ValueError(f"topk ({topk}) must be positive and a multiple of n_block_size ({n_block_size})")

    head_dim_padded = math.ceil(head_dim / 16) * 16
    k_block_size_eff = head_dim_padded if k_block_size is None else k_block_size
    if k_block_size_eff <= 0:
        raise ValueError(f"k_block_size must be positive, got {k_block_size_eff}")
    if head_dim_padded % k_block_size_eff:
        raise ValueError(f"padded head dimension ({head_dim_padded}) must be divisible by k_block_size ({k_block_size_eff})")
    if k_block_size_eff % 64:
        raise ValueError(f"k_block_size must be a multiple of 64, got {k_block_size_eff}")

    per_head_element_bytes = 2 if score_type == "indexer" else 4
    k_bytes_per_stage = n_block_size * k_block_size_eff * 2
    q_bytes = m_block_size * head_dim_padded * 2
    topk_bytes = topk * 2 * 4
    per_head_bytes = m_block_size * 2 * per_head_element_bytes
    fixed_overhead = per_head_bytes + 2048

    topk_in_smem = True
    smem_overhead = topk_bytes + fixed_overhead
    kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - q_bytes - smem_overhead) // k_bytes_per_stage))
    total_smem = q_bytes + k_bytes_per_stage * kv_stage + smem_overhead
    if total_smem > _SM100_SMEM_BYTES:
        topk_in_smem = False
        smem_overhead = fixed_overhead
        kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - q_bytes - smem_overhead) // k_bytes_per_stage))
        total_smem = q_bytes + k_bytes_per_stage * kv_stage + smem_overhead
        if total_smem > _SM100_SMEM_BYTES:
            raise ValueError(
                "SM100 shared-memory requirement exceeds 228 KiB even with top-K indices in global memory: "
                f"head_dim={head_dim}, m_block_size={m_block_size}, "
                f"n_block_size={n_block_size}, k_block_size={k_block_size_eff}"
            )

    return kv_stage, topk_in_smem


def resolve_sparse_score_kernel_config(
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    have_topk_length: bool,
) -> SparseScoreKernelConfig:
    """Resolve the SM100 sparse tile and shared-memory configuration."""

    if score_type not in ("indexer", "attention"):
        raise ValueError(f"score_type must be 'indexer' or 'attention', got {score_type!r}")
    if head_dim <= 0 or head_dim % 64:
        raise ValueError(f"head dimension must be a positive multiple of 64, got {head_dim}")
    if qhead_per_kv_head not in (32, 64, 128):
        raise ValueError(f"qhead_per_kv_head must be 32, 64, or 128 for the SM100 sparse score kernel, got {qhead_per_kv_head}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    if score_type == "indexer":
        m_block_size, n_block_size, k_block_size = qhead_per_kv_head, 128, None
    else:
        m_block_size, n_block_size, k_block_size = dispatch_sparse_attn_tile_params(
            head_dim,
            qhead_per_kv_head,
            topk,
            compact=have_topk_length,
        )

    if topk % n_block_size:
        raise ValueError(f"topk ({topk}) must be a multiple of the selected n_block_size ({n_block_size})")

    kv_stage, topk_in_smem = resolve_sparse_score_smem_config(
        score_type=score_type,
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
        topk=topk,
    )
    return SparseScoreKernelConfig(
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
        kv_stage=kv_stage,
        have_topk_length=bool(have_topk_length),
        topk_in_smem=topk_in_smem,
    )


def _select_dense_k_block_size(
    head_dim_padded: int,
    m_block_size: int,
    n_block_size: int,
    per_head_element_bytes: int,
) -> int:
    q_bytes = m_block_size * head_dim_padded * 2
    per_head_bytes = m_block_size * 2 * per_head_element_bytes
    available_k_bytes = _SM100_SMEM_BYTES - q_bytes - per_head_bytes - 2048
    max_k_block_size = max(64, available_k_bytes // (n_block_size * 2))
    max_k_block_size = (max_k_block_size // 64) * 64

    for candidate in range(min(head_dim_padded, max_k_block_size), 63, -64):
        if head_dim_padded % candidate == 0:
            return candidate
    raise ValueError(
        "No SM100 dense-score K tile divides the padded head dimension: "
        f"head_dim_padded={head_dim_padded}, m_block_size={m_block_size}, n_block_size={n_block_size}"
    )


def _dense_m_block_size(qhead_per_kv_head: int, head_dim: int, per_head_element_bytes: int) -> int:
    head_dim_padded = math.ceil(head_dim / 16) * 16
    n_block_size = 128
    m_block_size = qhead_per_kv_head * 2
    q_bytes = m_block_size * head_dim_padded * 2
    minimum_k_bytes = n_block_size * 64 * 2
    per_head_bytes = m_block_size * 2 * per_head_element_bytes
    if q_bytes + minimum_k_bytes + per_head_bytes + 4096 <= _SM100_SMEM_BYTES:
        return m_block_size
    return qhead_per_kv_head


def resolve_dense_score_smem_config(
    *,
    score_type: str,
    head_dim: int,
    m_block_size: int,
    n_block_size: int,
    k_block_size: int | None,
) -> tuple[int, int]:
    """Return the effective K tile and stage count for a dense tile choice."""

    if score_type not in ("indexer", "attention"):
        raise ValueError(f"score_type must be 'indexer' or 'attention', got {score_type!r}")
    if head_dim <= 0 or m_block_size <= 0 or n_block_size <= 0:
        raise ValueError(f"head_dim, m_block_size, and n_block_size must be positive, got {head_dim}, {m_block_size}, and {n_block_size}")
    head_dim_padded = math.ceil(head_dim / 16) * 16
    per_head_element_bytes = 2 if score_type == "indexer" else 4
    effective_k_block_size = k_block_size
    if effective_k_block_size is None:
        effective_k_block_size = _select_dense_k_block_size(
            head_dim_padded,
            m_block_size,
            n_block_size,
            per_head_element_bytes,
        )
    if effective_k_block_size <= 0 or effective_k_block_size % 64:
        raise ValueError(f"k_block_size must be a positive multiple of 64, got {effective_k_block_size}")
    if head_dim_padded % effective_k_block_size:
        raise ValueError(f"padded head dimension ({head_dim_padded}) must be divisible by k_block_size ({effective_k_block_size})")

    k_bytes_per_stage = n_block_size * effective_k_block_size * 2
    q_bytes = m_block_size * head_dim_padded * 2
    per_head_bytes = m_block_size * 2 * per_head_element_bytes
    kv_stage = min(4, max(1, (_SM100_SMEM_BYTES - q_bytes - per_head_bytes - 2048) // k_bytes_per_stage))
    return effective_k_block_size, kv_stage


def resolve_dense_score_kernel_config(*, score_type: str, head_dim: int, qhead_per_kv_head: int) -> DenseScoreKernelConfig:
    """Resolve the tuned SM100 dense-score tile."""

    if score_type not in ("indexer", "attention"):
        raise ValueError(f"score_type must be 'indexer' or 'attention', got {score_type!r}")
    if head_dim <= 0 or head_dim % 64:
        raise ValueError(f"head dimension must be a positive multiple of 64, got {head_dim}")
    if qhead_per_kv_head not in (32, 64, 128):
        raise ValueError(f"qhead_per_kv_head must be 32, 64, or 128 for the SM100 dense score kernel, got {qhead_per_kv_head}")

    per_head_element_bytes = 2 if score_type == "indexer" else 4
    m_block_size = _dense_m_block_size(qhead_per_kv_head, head_dim, per_head_element_bytes)
    n_block_size = 128
    head_dim_padded = math.ceil(head_dim / 16) * 16

    if score_type == "indexer" and head_dim == 128:
        k_block_size = head_dim_padded
    elif score_type == "attention" and head_dim in (512, 576):
        k_block_size = 64
    else:
        k_block_size = _select_dense_k_block_size(
            head_dim_padded,
            m_block_size,
            n_block_size,
            per_head_element_bytes,
        )

    k_block_size, kv_stage = resolve_dense_score_smem_config(
        score_type=score_type,
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
    )
    return DenseScoreKernelConfig(
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=None if k_block_size == head_dim_padded else k_block_size,
        kv_stage=kv_stage,
    )


__all__ = [
    "DenseScoreKernelConfig",
    "SparseScoreKernelConfig",
    "dispatch_sparse_attn_tile_params",
    "resolve_dense_score_kernel_config",
    "resolve_dense_score_smem_config",
    "resolve_sparse_score_kernel_config",
    "resolve_sparse_score_smem_config",
]
