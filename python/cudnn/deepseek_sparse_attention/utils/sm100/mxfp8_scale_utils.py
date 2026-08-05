# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side MXFP8 scale packing helpers for the SM100 Indexer DSL path."""

from __future__ import annotations

import math

import torch

from cudnn.deepseek_sparse_attention.utils.runtime import ceil_div as _ceil_div

_ATOM_MN = 128
_ATOM_K = 4
_DATA_PATHS = 32
_DATA_PATH_STRIDE = 16
_ATOM_ELEMS = 512


def _cu_seqlens_to_list(
    cu_seqlens: torch.Tensor,
    *,
    total: int,
    name: str,
) -> list[int]:
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    values = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    if not values:
        raise ValueError(f"{name} must have at least one element")
    if values[0] != 0:
        raise ValueError(f"{name}[0] must be 0")
    if values[-1] != total:
        raise ValueError(f"{name}[-1] must equal the scale tensor sequence length")
    if any(cur < prev for prev, cur in zip(values, values[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return values


def make_scale_cu_seqlens_padded(
    cu_seqlens: torch.Tensor,
    token_alignment: int,
) -> torch.Tensor:
    """Return per-sequence padded token prefixes for a compact THD tensor.

    ``cu_seqlens`` remains the prefix for compact Q/K data.  The returned
    ``(B+1,)`` int32 tensor is the prefix for the independently padded MXFP8
    scale storage, where each sequence length is minimally rounded up to
    ``token_alignment``. Callers may instead pass the THD scale packers any span
    that covers the logical sequence and is a multiple of ``token_alignment``.
    The operation stays on the input device.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be 1D with at least two elements")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32")
    if token_alignment <= 0:
        raise ValueError("token_alignment must be positive")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    padded_lengths = ((lengths + int(token_alignment) - 1) // int(token_alignment)) * int(token_alignment)
    out = torch.zeros_like(cu_seqlens)
    torch.cumsum(padded_lengths, dim=0, out=out[1:])
    return out


def _validate_scale_cu_seqlens_padded(
    cu_seqlens: list[int],
    cu_seqlens_scale_padded: torch.Tensor,
    *,
    token_alignment: int,
    name: str,
) -> list[int]:
    if cu_seqlens_scale_padded.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if cu_seqlens_scale_padded.dtype != torch.int32:
        raise ValueError(f"{name} must be int32")
    values = [int(v) for v in cu_seqlens_scale_padded.detach().cpu().tolist()]
    if len(values) != len(cu_seqlens):
        raise ValueError(f"{name} must have the same length as cu_seqlens")
    if not values or values[0] != 0:
        raise ValueError(f"{name}[0] must be 0")
    if any(cur < prev for prev, cur in zip(values, values[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    for b, (logical0, logical1, padded0, padded1) in enumerate(zip(cu_seqlens, cu_seqlens[1:], values, values[1:])):
        logical_len = logical1 - logical0
        padded_len = padded1 - padded0
        if padded_len < logical_len or padded_len % token_alignment != 0:
            raise ValueError(f"{name} batch {b} span must be at least {logical_len} and a " f"multiple of {token_alignment}, got {padded_len}")
    return values


def _blockscaled_indices(
    mn: int,
    sf_groups: int,
    l_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return flat indices for NVIDIA's packed blockscaled SF layout."""
    mn_tiles = _ceil_div(mn, _ATOM_MN)
    sf_tiles = _ceil_div(sf_groups, _ATOM_K)
    sf_per_l = _ATOM_ELEMS * mn_tiles * sf_tiles

    m, sf = torch.meshgrid(
        torch.arange(mn, device=device, dtype=torch.int64),
        torch.arange(sf_groups, device=device, dtype=torch.int64),
        indexing="ij",
    )
    base = (
        (m // _ATOM_MN) * (_ATOM_ELEMS * sf_tiles)
        + (sf // _ATOM_K) * _ATOM_ELEMS
        + _DATA_PATH_STRIDE * (m % _DATA_PATHS)
        + _ATOM_K * ((m % _ATOM_MN) // _DATA_PATHS)
        + (sf % _ATOM_K)
    )
    l_offsets = torch.arange(l_size, device=device, dtype=torch.int64)[:, None, None]
    return base.unsqueeze(0) + l_offsets * sf_per_l


def pack_blockscaled_scale_mkl(scale_mkl: torch.Tensor) -> torch.Tensor:
    """Pack logical ``(MN, sf_groups, L)`` scales into Blackwell SF layout.

    The returned tensor has physical shape ``(L, padded_MN, padded_sf_groups)``.
    ``sf_groups`` is ``ceil_div(head_dim, 32)`` for MXFP8.
    """
    if scale_mkl.ndim != 3:
        raise ValueError(f"scale_mkl must have shape (MN, sf_groups, L), got {tuple(scale_mkl.shape)}")
    mn, sf_groups, l_size = scale_mkl.shape
    m_padded = _ceil_div(mn, _ATOM_MN) * _ATOM_MN
    sf_padded = _ceil_div(sf_groups, _ATOM_K) * _ATOM_K
    packed = torch.zeros(
        (l_size, m_padded, sf_padded),
        device=scale_mkl.device,
        dtype=scale_mkl.dtype,
    )
    idx = _blockscaled_indices(mn, sf_groups, l_size, device=scale_mkl.device)
    packed.reshape(-1)[idx.reshape(-1)] = scale_mkl.permute(2, 0, 1).reshape(-1)
    return packed.contiguous()


def unpack_blockscaled_scale_mkl(
    scale_packed: torch.Tensor,
    mn: int,
    sf_groups: int,
) -> torch.Tensor:
    """Unpack a packed scale tensor back to logical ``(MN, sf_groups, L)``."""
    if scale_packed.ndim != 3:
        raise ValueError("scale_packed must have shape (L, padded_MN, padded_sf_groups), " f"got {tuple(scale_packed.shape)}")
    l_size = scale_packed.shape[0]
    idx = _blockscaled_indices(mn, sf_groups, l_size, device=scale_packed.device)
    return scale_packed.reshape(-1)[idx].permute(1, 2, 0).contiguous()


def logical_q_scale_to_mkl(
    q_scale: torch.Tensor,
    qhead_per_kv_head: int,
) -> torch.Tensor:
    """Convert BSHG Q scales to packed-Q logical ``(MN, sf_groups, L)`` order.

    Row order matches the kernel PackGQA order:
    ``packed_m = q_token * qhead_per_kv_head + h_local``.
    The L dimension is flattened as ``batch * n_heads_kv + kv_head``.
    """
    if q_scale.ndim != 4:
        raise ValueError(f"q_scale must have shape (B, S_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    bs, seqlen_q, n_heads_q, sf_groups = q_scale.shape
    if n_heads_q % qhead_per_kv_head != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by qhead_per_kv_head " f"({qhead_per_kv_head})")
    n_heads_kv = n_heads_q // qhead_per_kv_head
    q_view = q_scale.view(bs, seqlen_q, n_heads_kv, qhead_per_kv_head, sf_groups)
    return q_view.permute(1, 3, 4, 0, 2).reshape(seqlen_q * qhead_per_kv_head, sf_groups, bs * n_heads_kv).contiguous()


def logical_k_scale_to_mkl(k_scale: torch.Tensor) -> torch.Tensor:
    """Convert BSHG K scales to logical ``(MN, sf_groups, L)`` order."""
    if k_scale.ndim != 4:
        raise ValueError(f"k_scale must have shape (B, S_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    bs, seqlen_k, n_heads_kv, sf_groups = k_scale.shape
    return k_scale.permute(1, 3, 0, 2).reshape(seqlen_k, sf_groups, bs * n_heads_kv).contiguous()


def logical_q_scale_to_mkl_thd(
    q_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_scale_padded: torch.Tensor,
    qhead_per_kv_head: int,
) -> torch.Tensor:
    """Convert compact THG Q scales to padded packed-Q ``(MN, groups, L)``.

    Sequences are concatenated along ``MN`` using
    ``cu_seqlens_q_scale_padded``. Each sequence's token span must cover the
    logical sequence and, after multiplying by ``qhead_per_kv_head``, produce
    a multiple of 128 packed-MN rows. The caller-provided span is preserved
    exactly. ``L`` contains only KV heads, not the batch dimension.
    """
    if q_scale.ndim != 3:
        raise ValueError(f"q_scale must have shape (total_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    total_q, n_heads_q, sf_groups = q_scale.shape
    if n_heads_q % qhead_per_kv_head != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by qhead_per_kv_head " f"({qhead_per_kv_head})")
    cu_q = _cu_seqlens_to_list(cu_seqlens_q, total=total_q, name="cu_seqlens_q")
    n_heads_kv = n_heads_q // qhead_per_kv_head
    token_alignment = 128 // math.gcd(128, qhead_per_kv_head)
    cu_q_scale = _validate_scale_cu_seqlens_padded(
        cu_q,
        cu_seqlens_q_scale_padded,
        token_alignment=token_alignment,
        name="cu_seqlens_q_scale_padded",
    )
    bs = len(cu_q) - 1
    out = torch.zeros(
        cu_q_scale[-1] * qhead_per_kv_head,
        sf_groups,
        n_heads_kv,
        device=q_scale.device,
        dtype=q_scale.dtype,
    )
    for b in range(bs):
        q0 = cu_q[b]
        q1 = cu_q[b + 1]
        q_len = q1 - q0
        if q_len == 0:
            continue
        q_view = q_scale[q0:q1].view(q_len, n_heads_kv, qhead_per_kv_head, sf_groups)
        q_scale0 = cu_q_scale[b] * qhead_per_kv_head
        out[q_scale0 : q_scale0 + q_len * qhead_per_kv_head] = q_view.permute(0, 2, 3, 1).reshape(q_len * qhead_per_kv_head, sf_groups, n_heads_kv)
    return out.contiguous()


def logical_k_scale_to_mkl_thd(
    k_scale: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k_scale_padded: torch.Tensor,
) -> torch.Tensor:
    """Convert compact THG K scales to padded logical ``(MN, groups, L)``."""
    if k_scale.ndim != 3:
        raise ValueError(f"k_scale must have shape (total_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    total_k, n_heads_kv, sf_groups = k_scale.shape
    cu_k = _cu_seqlens_to_list(cu_seqlens_k, total=total_k, name="cu_seqlens_k")
    cu_k_scale = _validate_scale_cu_seqlens_padded(
        cu_k,
        cu_seqlens_k_scale_padded,
        token_alignment=128,
        name="cu_seqlens_k_scale_padded",
    )
    bs = len(cu_k) - 1
    out = torch.zeros(
        cu_k_scale[-1],
        sf_groups,
        n_heads_kv,
        device=k_scale.device,
        dtype=k_scale.dtype,
    )
    for b in range(bs):
        k0 = cu_k[b]
        k1 = cu_k[b + 1]
        k_len = k1 - k0
        if k_len == 0:
            continue
        k_scale0 = cu_k_scale[b]
        out[k_scale0 : k_scale0 + k_len] = k_scale[k0:k1].permute(0, 2, 1)
    return out.contiguous()


def pack_q_scale_bshd(q_scale: torch.Tensor, qhead_per_kv_head: int = 64) -> torch.Tensor:
    """Pack logical BSHD Q scales for the Indexer MXFP8 kernel."""
    return pack_blockscaled_scale_mkl(logical_q_scale_to_mkl(q_scale, qhead_per_kv_head))


def pack_k_scale_bshd(k_scale: torch.Tensor) -> torch.Tensor:
    """Pack logical BSHD K scales for the Indexer MXFP8 kernel."""
    return pack_blockscaled_scale_mkl(logical_k_scale_to_mkl(k_scale))


def pack_q_scale_thd(
    q_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_scale_padded: torch.Tensor,
    qhead_per_kv_head: int = 64,
) -> torch.Tensor:
    """Pack logical THD Q scales for the Indexer MXFP8 kernel."""
    return pack_blockscaled_scale_mkl(
        logical_q_scale_to_mkl_thd(
            q_scale,
            cu_seqlens_q,
            cu_seqlens_q_scale_padded,
            qhead_per_kv_head,
        )
    )


def pack_k_scale_thd(
    k_scale: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k_scale_padded: torch.Tensor,
) -> torch.Tensor:
    """Pack logical THD K scales for the Indexer MXFP8 kernel."""
    return pack_blockscaled_scale_mkl(
        logical_k_scale_to_mkl_thd(
            k_scale,
            cu_seqlens_k,
            cu_seqlens_k_scale_padded,
        )
    )
