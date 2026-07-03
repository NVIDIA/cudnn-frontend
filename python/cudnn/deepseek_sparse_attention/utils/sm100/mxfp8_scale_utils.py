"""Host-side MXFP8 scale packing helpers for the SM100 Indexer DSL path."""

from __future__ import annotations

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
    qhead_per_kv_head: int,
    max_seqlen_q: int | None = None,
) -> torch.Tensor:
    """Convert THG Q scales to padded packed-Q ``(MN, sf_groups, L)`` order.

    ``L`` is flattened as ``batch * n_heads_kv + kv_head``. Within each
    varlen batch, ``MN`` uses the same PackGQA row order as BSHD:
    ``packed_m = q_token_local * qhead_per_kv_head + h_local``. Rows beyond
    the batch-local query length are zero padding.
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
    bs = len(cu_q) - 1
    lengths = [cu_q[b + 1] - cu_q[b] for b in range(bs)]
    inferred_max = max(lengths) if lengths else 0
    if max_seqlen_q is None:
        max_seqlen_q = inferred_max
    if max_seqlen_q < inferred_max:
        raise ValueError("max_seqlen_q must be at least the largest q sequence length")
    out = torch.zeros(
        max_seqlen_q * qhead_per_kv_head,
        sf_groups,
        bs * n_heads_kv,
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
        out[
            : q_len * qhead_per_kv_head,
            :,
            b * n_heads_kv : (b + 1) * n_heads_kv,
        ] = q_view.permute(
            0, 2, 3, 1
        ).reshape(q_len * qhead_per_kv_head, sf_groups, n_heads_kv)
    return out.contiguous()


def logical_k_scale_to_mkl_thd(
    k_scale: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Convert THG K scales to padded logical ``(MN, sf_groups, L)`` order."""
    if k_scale.ndim != 3:
        raise ValueError(f"k_scale must have shape (total_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    total_k, n_heads_kv, sf_groups = k_scale.shape
    cu_k = _cu_seqlens_to_list(cu_seqlens_k, total=total_k, name="cu_seqlens_k")
    bs = len(cu_k) - 1
    lengths = [cu_k[b + 1] - cu_k[b] for b in range(bs)]
    inferred_max = max(lengths) if lengths else 0
    if max_seqlen_k is None:
        max_seqlen_k = inferred_max
    if max_seqlen_k < inferred_max:
        raise ValueError("max_seqlen_k must be at least the largest k sequence length")
    out = torch.zeros(
        max_seqlen_k,
        sf_groups,
        bs * n_heads_kv,
        device=k_scale.device,
        dtype=k_scale.dtype,
    )
    for b in range(bs):
        k0 = cu_k[b]
        k1 = cu_k[b + 1]
        k_len = k1 - k0
        if k_len == 0:
            continue
        out[:k_len, :, b * n_heads_kv : (b + 1) * n_heads_kv] = k_scale[k0:k1].permute(0, 2, 1)
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
    qhead_per_kv_head: int = 64,
    max_seqlen_q: int | None = None,
) -> torch.Tensor:
    """Pack logical THD Q scales for the Indexer MXFP8 kernel."""
    return pack_blockscaled_scale_mkl(
        logical_q_scale_to_mkl_thd(
            q_scale,
            cu_seqlens_q,
            qhead_per_kv_head,
            max_seqlen_q=max_seqlen_q,
        )
    )


def pack_k_scale_thd(
    k_scale: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Pack logical THD K scales for the Indexer MXFP8 kernel."""
    return pack_blockscaled_scale_mkl(
        logical_k_scale_to_mkl_thd(
            k_scale,
            cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )
    )
