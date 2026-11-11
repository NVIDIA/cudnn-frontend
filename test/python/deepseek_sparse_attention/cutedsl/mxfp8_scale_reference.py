# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference implementation of the Blackwell MXFP8 scale layout."""

import math

import torch

_ATOM_MN = 128
_ATOM_K = 4
_DATA_PATHS = 32
_DATA_PATH_STRIDE = 16
_ATOM_ELEMS = 512


def _ceil_div(value: int, divisor: int) -> int:
    """Return ``value`` divided by ``divisor``, rounded up."""
    return (value + divisor - 1) // divisor


def _cu_seqlens_to_list(cu_seqlens: torch.Tensor, *, total: int, name: str) -> list[int]:
    """Validate a cumulative-length tensor and return its host values."""
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    values = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if not values or values[0] != 0:
        raise ValueError(f"{name}[0] must be 0")
    if values[-1] != total:
        raise ValueError(f"{name}[-1] must equal the scale tensor sequence length")
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return values


def _validate_padded_prefix(
    cu_seqlens: list[int],
    padded: torch.Tensor,
    *,
    token_alignment: int,
    name: str,
) -> list[int]:
    """Validate a per-sequence padded cumulative-length prefix."""
    values = [int(value) for value in padded.detach().cpu().tolist()]
    if padded.ndim != 1 or padded.dtype != torch.int32 or len(values) != len(cu_seqlens):
        raise ValueError(f"{name} must be a matching 1D int32 prefix")
    if not values or values[0] != 0:
        raise ValueError(f"{name}[0] must be 0")
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    for logical0, logical1, padded0, padded1 in zip(cu_seqlens, cu_seqlens[1:], values, values[1:]):
        logical_length = logical1 - logical0
        padded_length = padded1 - padded0
        if padded_length < logical_length or padded_length % token_alignment:
            raise ValueError(f"{name} spans must cover each sequence and respect alignment")
    return values


def _blockscaled_indices(mn: int, sf_groups: int, l_size: int, *, device: torch.device) -> torch.Tensor:
    """Return flat physical offsets for logical ``(L, MN, groups)`` scales."""
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
    """Pack logical ``(MN, groups, L)`` scales with PyTorch operations."""
    if scale_mkl.ndim != 3:
        raise ValueError("scale_mkl must have shape (MN, sf_groups, L)")
    mn, sf_groups, l_size = scale_mkl.shape
    shape = (
        l_size,
        _ceil_div(mn, _ATOM_MN) * _ATOM_MN,
        _ceil_div(sf_groups, _ATOM_K) * _ATOM_K,
    )
    packed = torch.zeros(shape, device=scale_mkl.device, dtype=scale_mkl.dtype)
    indices = _blockscaled_indices(mn, sf_groups, l_size, device=scale_mkl.device).reshape(-1)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and scale_mkl.dtype == e8m0_dtype:
        source = scale_mkl.view(torch.uint8).permute(2, 0, 1).reshape(-1)
        packed.view(torch.uint8).reshape(-1)[indices] = source
    else:
        source = scale_mkl.permute(2, 0, 1).reshape(-1)
        packed.reshape(-1)[indices] = source
    return packed


def unpack_blockscaled_scale_mkl(scale_packed: torch.Tensor, mn: int, sf_groups: int) -> torch.Tensor:
    """Unpack physical scales to logical ``(MN, groups, L)`` order."""
    indices = _blockscaled_indices(mn, sf_groups, scale_packed.shape[0], device=scale_packed.device)
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and scale_packed.dtype == e8m0_dtype:
        unpacked = scale_packed.view(torch.uint8).reshape(-1)[indices].permute(1, 2, 0).contiguous()
        return unpacked.view(e8m0_dtype)
    return scale_packed.reshape(-1)[indices].permute(1, 2, 0).contiguous()


def logical_q_scale_to_mkl(q_scale: torch.Tensor, qhead_per_kv_head: int) -> torch.Tensor:
    """Convert logical BSHD query scales to ``(MN, groups, L)`` order."""
    bs, seqlen_q, n_heads_q, sf_groups = q_scale.shape
    n_heads_kv = n_heads_q // qhead_per_kv_head
    q_view = q_scale.view(bs, seqlen_q, n_heads_kv, qhead_per_kv_head, sf_groups)
    return q_view.permute(1, 3, 4, 0, 2).reshape(seqlen_q * qhead_per_kv_head, sf_groups, bs * n_heads_kv).contiguous()


def logical_k_scale_to_mkl(k_scale: torch.Tensor) -> torch.Tensor:
    """Convert logical BSHD key scales to ``(MN, groups, L)`` order."""
    bs, seqlen_k, n_heads_kv, sf_groups = k_scale.shape
    return k_scale.permute(1, 3, 0, 2).reshape(seqlen_k, sf_groups, bs * n_heads_kv).contiguous()


def logical_q_scale_to_mkl_thd(
    q_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    padded: torch.Tensor,
    qhead_per_kv_head: int,
) -> torch.Tensor:
    """Convert logical THD query scales to padded ``(MN, groups, L)`` order."""
    total_q, n_heads_q, sf_groups = q_scale.shape
    n_heads_kv = n_heads_q // qhead_per_kv_head
    cu_q = _cu_seqlens_to_list(cu_seqlens_q, total=total_q, name="cu_seqlens_q")
    alignment = 128 // math.gcd(128, qhead_per_kv_head)
    cu_padded = _validate_padded_prefix(cu_q, padded, token_alignment=alignment, name="cu_seqlens_q_scale_padded")
    out = torch.zeros(
        cu_padded[-1] * qhead_per_kv_head,
        sf_groups,
        n_heads_kv,
        device=q_scale.device,
        dtype=q_scale.dtype,
    )
    for batch in range(len(cu_q) - 1):
        q0, q1 = cu_q[batch : batch + 2]
        q_length = q1 - q0
        if q_length:
            q_view = q_scale[q0:q1].view(q_length, n_heads_kv, qhead_per_kv_head, sf_groups)
            start = cu_padded[batch] * qhead_per_kv_head
            out[start : start + q_length * qhead_per_kv_head] = q_view.permute(0, 2, 3, 1).reshape(q_length * qhead_per_kv_head, sf_groups, n_heads_kv)
    return out


def logical_k_scale_to_mkl_thd(k_scale: torch.Tensor, cu_seqlens_k: torch.Tensor, padded: torch.Tensor) -> torch.Tensor:
    """Convert logical THD key scales to padded ``(MN, groups, L)`` order."""
    total_k, n_heads_kv, sf_groups = k_scale.shape
    cu_k = _cu_seqlens_to_list(cu_seqlens_k, total=total_k, name="cu_seqlens_k")
    cu_padded = _validate_padded_prefix(cu_k, padded, token_alignment=128, name="cu_seqlens_k_scale_padded")
    out = torch.zeros(cu_padded[-1], sf_groups, n_heads_kv, device=k_scale.device, dtype=k_scale.dtype)
    for batch in range(len(cu_k) - 1):
        k0, k1 = cu_k[batch : batch + 2]
        if k1 > k0:
            out[cu_padded[batch] : cu_padded[batch] + k1 - k0] = k_scale[k0:k1].permute(0, 2, 1)
    return out


def pack_q_scale_bshd(q_scale: torch.Tensor, qhead_per_kv_head: int = 64) -> torch.Tensor:
    """Pack logical BSHD query scales with the PyTorch oracle."""
    return pack_blockscaled_scale_mkl(logical_q_scale_to_mkl(q_scale, qhead_per_kv_head))


def pack_k_scale_bshd(k_scale: torch.Tensor) -> torch.Tensor:
    """Pack logical BSHD key scales with the PyTorch oracle."""
    return pack_blockscaled_scale_mkl(logical_k_scale_to_mkl(k_scale))


def pack_qk_scale_bshd(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    qhead_per_kv_head: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack logical BSHD query and key scales with the PyTorch oracle."""
    return pack_q_scale_bshd(q_scale, qhead_per_kv_head), pack_k_scale_bshd(k_scale)


def pack_q_scale_thd(
    q_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    padded: torch.Tensor,
    qhead_per_kv_head: int = 64,
) -> torch.Tensor:
    """Pack logical THD query scales with explicit padded offsets."""
    return pack_blockscaled_scale_mkl(logical_q_scale_to_mkl_thd(q_scale, cu_seqlens_q, padded, qhead_per_kv_head))


def pack_k_scale_thd(k_scale: torch.Tensor, cu_seqlens_k: torch.Tensor, padded: torch.Tensor) -> torch.Tensor:
    """Pack logical THD key scales with explicit padded offsets."""
    return pack_blockscaled_scale_mkl(logical_k_scale_to_mkl_thd(k_scale, cu_seqlens_k, padded))
