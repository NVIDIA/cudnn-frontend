# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-only MXFP8 scale packing helpers for the SM100 Indexer DSL path.

Runtime packing requires contiguous one-byte CUDA inputs, a supported GPU, and
the CuTe DSL version required by the DSA kernels. PyTorch reference packing is
kept in the test suite and is deliberately not a runtime fallback.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from cudnn._torch_stream import device_context, record_streams
from cudnn.deepseek_sparse_attention.utils.runtime import ceil_div as _ceil_div
from cudnn.deepseek_sparse_attention.utils.runtime import (
    resolve_stream,
    torch_stream_context,
)

_ATOM_MN = 128
_ATOM_K = 4


def make_scale_cu_seqlens_padded(cu_seqlens: torch.Tensor, token_alignment: int) -> torch.Tensor:
    """Return per-sequence padded token prefixes for a compact THD tensor.

    ``cu_seqlens`` remains the prefix for compact Q/K data. The returned
    ``(B+1,)`` int32 tensor is the prefix for independently padded MXFP8 scale
    storage, where each sequence length is minimally rounded up to
    ``token_alignment``. Callers may instead pass the THD scale packers any
    span that covers the logical sequence and is a multiple of
    ``token_alignment``. The operation stays on the input device.
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


def minimal_padded_token_capacity(total_tokens: int, num_sequences: int, token_alignment: int) -> int:
    """Return a host-only capacity for minimally padded THD scale prefixes.

    If ``cu_seqlens_scale_padded`` was produced by
    :func:`make_scale_cu_seqlens_padded`, its final value is at most this
    capacity. The formula depends only on declared tensor shapes, so callers
    can allocate reusable ``out`` storage without reading a device prefix:
    ``floor((T + (alignment-1)*min(T,B)) / alignment) * alignment``. The
    ``min(T,B)`` term excludes necessarily empty sequences from the padding
    envelope. Custom prefixes with padding beyond the minimum must size
    ``out`` from their own host-side envelope instead.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be nonnegative")
    if num_sequences < 0:
        raise ValueError("num_sequences must be nonnegative")
    if num_sequences == 0 and total_tokens != 0:
        raise ValueError("num_sequences must be positive when total_tokens is nonzero")
    if token_alignment <= 0:
        raise ValueError("token_alignment must be positive")
    # At most one nonempty sequence can be created by each token. Empty
    # sequences contribute no padding, so using num_sequences directly would
    # overestimate the envelope when total_tokens < num_sequences.
    active_sequences = min(int(total_tokens), int(num_sequences))
    return ((int(total_tokens) + (int(token_alignment) - 1) * active_sequences) // int(token_alignment)) * int(token_alignment)


def _validate_thd_cuda_prefixes(
    scale: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_scale_padded: torch.Tensor,
) -> None:
    """Validate THD prefix metadata without inspecting device values."""
    for prefix, name in (
        (cu_seqlens, "cu_seqlens"),
        (cu_seqlens_scale_padded, "cu_seqlens_scale_padded"),
    ):
        if prefix.ndim != 1 or prefix.numel() < 2:
            raise ValueError(f"{name} must be 1D with at least two elements")
        if prefix.dtype != torch.int32:
            raise ValueError(f"{name} must be int32")
        if not prefix.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if prefix.device != scale.device:
            raise ValueError(f"{name} must be on {scale.device}")
    if cu_seqlens.shape != cu_seqlens_scale_padded.shape:
        raise ValueError("cu_seqlens and cu_seqlens_scale_padded must have the same shape")


def _packed_scale_shape(mn: int, sf_groups: int, l_size: int) -> tuple[int, int, int]:
    """Return the physical ``(L, padded_MN, padded_groups)`` shape."""
    return (
        l_size,
        _ceil_div(mn, _ATOM_MN) * _ATOM_MN,
        _ceil_div(sf_groups, _ATOM_K) * _ATOM_K,
    )


def _shares_nonempty_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two nonempty tensors share the same backing storage."""
    return bool(lhs.numel() and rhs.numel() and lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr())


def _validate_pack_out_storage(out: torch.Tensor, src: torch.Tensor) -> None:
    """Validate common dtype, device, layout, and aliasing requirements."""
    if out.dtype != src.dtype or out.device != src.device:
        raise ValueError(f"out must have dtype {src.dtype} on {src.device}, got {out.dtype} on {out.device}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if _shares_nonempty_storage(out, src):
        raise ValueError("out must not alias the source scale tensor")


def _validate_pack_out(out: torch.Tensor, src: torch.Tensor, shape: tuple[int, int, int]) -> None:
    """Validate exact-shape caller-owned packed storage."""
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}, got {tuple(out.shape)}")
    _validate_pack_out_storage(out, src)


def _validate_thd_pack_out(out: torch.Tensor, src: torch.Tensor, *, l_size: int, sf_groups: int) -> None:
    """Validate caller-owned THD storage."""
    expected_tail = _ceil_div(sf_groups, _ATOM_K) * _ATOM_K
    if out.ndim != 3 or out.shape[0] != l_size or out.shape[1] % _ATOM_MN != 0 or out.shape[2] != expected_tail:
        raise ValueError(f"out must have shape ({l_size}, multiple_of_128, {expected_tail}), got {tuple(out.shape)}")
    _validate_pack_out_storage(out, src)


@lru_cache(maxsize=None)
def _cuda_packer_requirement_error(device_index: int) -> str | None:
    """Return why the CUDA packer is unavailable on ``device_index``."""
    from cudnn.frost.buffers import (
        CUTEDSL_MIN_VERSION,
        cutedsl_arch_requirement_error,
        cutedsl_requirement_error,
        cutedsl_state,
    )

    installed, _ = cutedsl_state()
    if not installed:
        floor = ".".join(str(component) for component in CUTEDSL_MIN_VERSION)
        return f"SM100 DSA MXFP8 scale packing requires nvidia-cutlass-dsl >= {floor}"
    requirement_error = cutedsl_requirement_error("SM100 DSA MXFP8 scale packing")
    if requirement_error:
        return requirement_error

    from cudnn.deepseek_sparse_attention.utils.compiler import _ARCH_MAP
    from cudnn.deepseek_sparse_attention.utils.runtime import device_capability

    capability = device_capability(device_index)
    if capability not in _ARCH_MAP:
        return f"SM100 DSA MXFP8 scale packing does not support compute capability {capability[0]}.{capability[1]}"
    arch_error = cutedsl_arch_requirement_error(capability)
    if arch_error:
        return arch_error
    return None


def _require_cuda_packer(scale: torch.Tensor, *, name: str) -> None:
    """Reject inputs that cannot use the runtime CUDA packer."""
    if not scale.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if scale.element_size() != 1:
        raise TypeError(f"{name} must have one-byte elements")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    requirement_error = _cuda_packer_requirement_error(scale.device.index)
    if requirement_error:
        raise RuntimeError(requirement_error)


def _allocate_pack_out(src: torch.Tensor, shape: tuple[int, int, int], out: torch.Tensor | None, stream) -> torch.Tensor:
    """Return validated caller storage or allocate an exact CUDA tensor."""
    if out is not None:
        _validate_pack_out(out, src, shape)
        return out
    with torch_stream_context(stream, src.device):
        return torch.empty(shape, device=src.device, dtype=src.dtype)


@lru_cache(maxsize=1)
def _cuda_pack_launchers():
    """Return gated CUDA launchers without repeating import/version work."""
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement_error = cutedsl_requirement_error("SM100 DSA MXFP8 scale packing")
    if requirement_error:
        raise RuntimeError(requirement_error)
    from cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_pack import (
        _launch_pack,
        _launch_qk_bshd_pack,
    )

    return _launch_pack, _launch_qk_bshd_pack


def _launch_cuda_packer(
    scale: torch.Tensor,
    packed: torch.Tensor,
    *,
    mn: int,
    sf_groups: int,
    l_size: int,
    source_layout: str,
    stream=None,
    seq_len: int = 0,
    n_heads_kv: int = 0,
    qhead_per_kv_head: int = 1,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_scale_padded: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch one validated raw-byte pack and return ``packed``."""
    launch_pack, _ = _cuda_pack_launchers()
    with device_context(scale.device):
        record_streams((scale, packed, cu_seqlens, cu_seqlens_scale_padded), stream, scale.device)
        launch_pack(
            scale.view(torch.uint8).reshape(-1),
            packed.view(torch.uint8).reshape(-1),
            mn=mn,
            sf_groups=sf_groups,
            l_size=l_size,
            source_layout=source_layout,
            stream=resolve_stream(stream),
            seq_len=seq_len,
            n_heads_kv=n_heads_kv,
            qhead_per_kv_head=qhead_per_kv_head,
            cu_seqlens=cu_seqlens,
            cu_seqlens_scale_padded=cu_seqlens_scale_padded,
        )
    return packed


def _launch_cuda_qk_bshd_packer(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    *,
    q_mn: int,
    k_mn: int,
    sf_groups: int,
    l_size: int,
    q_seq_len: int,
    k_seq_len: int,
    n_heads_kv: int,
    qhead_per_kv_head: int,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused BSHD Q/K packer and return both outputs."""
    _, launch_qk_bshd_pack = _cuda_pack_launchers()
    with device_context(q_scale.device):
        record_streams((q_scale, k_scale, q_packed, k_packed), stream, q_scale.device)
        launch_qk_bshd_pack(
            q_scale.view(torch.uint8).reshape(-1),
            k_scale.view(torch.uint8).reshape(-1),
            q_packed.view(torch.uint8).reshape(-1),
            k_packed.view(torch.uint8).reshape(-1),
            q_mn=q_mn,
            k_mn=k_mn,
            sf_groups=sf_groups,
            l_size=l_size,
            q_seq_len=q_seq_len,
            k_seq_len=k_seq_len,
            n_heads_kv=n_heads_kv,
            qhead_per_kv_head=qhead_per_kv_head,
            stream=resolve_stream(stream),
        )
    return q_packed, k_packed


def pack_blockscaled_scale_mkl(scale_mkl: torch.Tensor, *, out: torch.Tensor | None = None, stream=None) -> torch.Tensor:
    """Pack contiguous one-byte CUDA ``(MN, sf_groups, L)`` scales.

    The result has shape ``(L, ceil128(MN), ceil4(sf_groups))``. ``out`` may
    provide contiguous exact-shape storage of the same dtype and device and
    must not alias ``scale_mkl``. With ``out``, the operation allocates nothing
    and is CUDA-graph capturable after warm-up. ``stream`` is an optional CUDA
    driver stream. Unsupported tensors or environments raise instead of
    falling back to PyTorch.
    """
    if scale_mkl.ndim != 3:
        raise ValueError(f"scale_mkl must have shape (MN, sf_groups, L), got {tuple(scale_mkl.shape)}")
    _require_cuda_packer(scale_mkl, name="scale_mkl")
    mn, sf_groups, l_size = scale_mkl.shape
    packed = _allocate_pack_out(scale_mkl, _packed_scale_shape(mn, sf_groups, l_size), out, stream)
    return _launch_cuda_packer(
        scale_mkl,
        packed,
        mn=mn,
        sf_groups=sf_groups,
        l_size=l_size,
        source_layout="mkl",
        stream=stream,
    )


def pack_q_scale_bshd(
    q_scale: torch.Tensor,
    qhead_per_kv_head: int = 64,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack contiguous one-byte CUDA ``(B,S_q,H_q,G)`` Q scales.

    The result has shape ``(B*H_kv, ceil128(S_q*q), ceil4(G))``, where
    ``q=qhead_per_kv_head``. ``out`` may provide contiguous exact-shape storage
    of the same dtype and device and must not alias ``q_scale``. The
    caller-owned path allocates nothing and is CUDA-graph capturable after
    warm-up. ``stream`` is an optional CUDA driver stream. Unsupported tensors
    or environments raise instead of falling back to PyTorch.
    """
    if q_scale.ndim != 4:
        raise ValueError(f"q_scale must have shape (B, S_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    bs, seqlen_q, n_heads_q, sf_groups = q_scale.shape
    if n_heads_q % qhead_per_kv_head != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by qhead_per_kv_head ({qhead_per_kv_head})")
    _require_cuda_packer(q_scale, name="q_scale")
    n_heads_kv = n_heads_q // qhead_per_kv_head
    mn = seqlen_q * qhead_per_kv_head
    packed = _allocate_pack_out(q_scale, _packed_scale_shape(mn, sf_groups, bs * n_heads_kv), out, stream)
    return _launch_cuda_packer(
        q_scale,
        packed,
        mn=mn,
        sf_groups=sf_groups,
        l_size=bs * n_heads_kv,
        source_layout="q_bshd",
        stream=stream,
        seq_len=seqlen_q,
        n_heads_kv=n_heads_kv,
        qhead_per_kv_head=qhead_per_kv_head,
    )


def pack_k_scale_bshd(k_scale: torch.Tensor, *, out: torch.Tensor | None = None, stream=None) -> torch.Tensor:
    """Pack contiguous one-byte CUDA ``(B,S_k,H_kv,G)`` K scales.

    The result has shape ``(B*H_kv, ceil128(S_k), ceil4(G))``. ``out`` may
    provide contiguous exact-shape storage of the same dtype and device and
    must not alias ``k_scale``. The caller-owned path allocates nothing and is
    CUDA-graph capturable after warm-up. ``stream`` is an optional CUDA driver
    stream. Unsupported tensors or environments raise instead of falling back
    to PyTorch.
    """
    if k_scale.ndim != 4:
        raise ValueError(f"k_scale must have shape (B, S_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    bs, seqlen_k, n_heads_kv, sf_groups = k_scale.shape
    _require_cuda_packer(k_scale, name="k_scale")
    packed = _allocate_pack_out(k_scale, _packed_scale_shape(seqlen_k, sf_groups, bs * n_heads_kv), out, stream)
    return _launch_cuda_packer(
        k_scale,
        packed,
        mn=seqlen_k,
        sf_groups=sf_groups,
        l_size=bs * n_heads_kv,
        source_layout="k_bshd",
        stream=stream,
        seq_len=seqlen_k,
        n_heads_kv=n_heads_kv,
    )


def pack_qk_scale_bshd(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    qhead_per_kv_head: int = 64,
    *,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack contiguous one-byte CUDA BSHD Q and K scales in one launch.

    Q and K must share batch size, scale groups, dtype, and device, with
    ``H_q == H_kv * qhead_per_kv_head``. ``q_out`` and ``k_out`` may provide
    contiguous exact-shape storage of the same dtype and device. Outputs must
    not alias either input or each other. Supplying both makes the operation
    allocation-free and CUDA-graph capturable after warm-up. Unsupported
    tensors or environments raise instead of falling back to PyTorch.
    """
    if q_scale.ndim != 4:
        raise ValueError(f"q_scale must have shape (B, S_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if k_scale.ndim != 4:
        raise ValueError(f"k_scale must have shape (B, S_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    bs, seqlen_q, n_heads_q, sf_groups = q_scale.shape
    bs_k, seqlen_k, n_heads_kv, k_sf_groups = k_scale.shape
    if bs != bs_k or sf_groups != k_sf_groups:
        raise ValueError("q_scale and k_scale must have matching batch size and sf_groups")
    if n_heads_q != n_heads_kv * qhead_per_kv_head:
        raise ValueError(f"n_heads_q ({n_heads_q}) must equal n_heads_kv * qhead_per_kv_head ({n_heads_kv} * {qhead_per_kv_head})")
    if q_scale.dtype != k_scale.dtype or q_scale.device != k_scale.device:
        raise ValueError("q_scale and k_scale must have the same dtype and device")
    _require_cuda_packer(q_scale, name="q_scale")
    _require_cuda_packer(k_scale, name="k_scale")

    q_mn = seqlen_q * qhead_per_kv_head
    l_size = bs * n_heads_kv
    q_shape = _packed_scale_shape(q_mn, sf_groups, l_size)
    k_shape = _packed_scale_shape(seqlen_k, sf_groups, l_size)
    if q_out is not None:
        _validate_pack_out(q_out, q_scale, q_shape)
    if k_out is not None:
        _validate_pack_out(k_out, k_scale, k_shape)
    for packed, name in ((q_out, "q_out"), (k_out, "k_out")):
        if packed is not None and (_shares_nonempty_storage(packed, q_scale) or _shares_nonempty_storage(packed, k_scale)):
            raise ValueError(f"{name} must not alias either source scale tensor")
    if q_out is not None and k_out is not None and _shares_nonempty_storage(q_out, k_out):
        raise ValueError("q_out and k_out must not alias each other")

    q_packed = q_out if q_out is not None else _allocate_pack_out(q_scale, q_shape, None, stream)
    k_packed = k_out if k_out is not None else _allocate_pack_out(k_scale, k_shape, None, stream)
    return _launch_cuda_qk_bshd_packer(
        q_scale,
        k_scale,
        q_packed,
        k_packed,
        q_mn=q_mn,
        k_mn=seqlen_k,
        sf_groups=sf_groups,
        l_size=l_size,
        q_seq_len=seqlen_q,
        k_seq_len=seqlen_k,
        n_heads_kv=n_heads_kv,
        qhead_per_kv_head=qhead_per_kv_head,
        stream=stream,
    )


def pack_q_scale_thd(
    q_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_q_scale_padded: torch.Tensor,
    qhead_per_kv_head: int = 64,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack CUDA ``(total_q,H_q,G)`` THD Q scales into required ``out``.

    Caller-owned ``out`` keeps the path allocation-free and avoids reading
    device prefixes on the host. It has shape
    ``(H_kv, M_capacity, ceil4(G))``, where ``M_capacity`` is a multiple of 128
    covering the padded Q extent times ``qhead_per_kv_head``. It must be
    contiguous, match the input dtype and device, and not alias ``q_scale``.
    Prefix tensors must be contiguous int32 CUDA tensors on the input device
    with matching shapes. Their values are a caller contract. The operation is
    CUDA-graph capturable after warm-up; ``stream`` is an optional CUDA driver
    stream. Unsupported inputs raise instead of falling back to PyTorch.
    """
    if q_scale.ndim != 3:
        raise ValueError(f"q_scale must have shape (total_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    _, n_heads_q, sf_groups = q_scale.shape
    if n_heads_q % qhead_per_kv_head != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by qhead_per_kv_head ({qhead_per_kv_head})")
    if out is None:
        raise ValueError("out is required for THD MXFP8 scale packing")
    _require_cuda_packer(q_scale, name="q_scale")
    n_heads_kv = n_heads_q // qhead_per_kv_head
    _validate_thd_cuda_prefixes(q_scale, cu_seqlens_q, cu_seqlens_q_scale_padded)
    _validate_thd_pack_out(out, q_scale, l_size=n_heads_kv, sf_groups=sf_groups)
    return _launch_cuda_packer(
        q_scale,
        out,
        mn=out.shape[1],
        sf_groups=sf_groups,
        l_size=n_heads_kv,
        source_layout="q_thd",
        stream=stream,
        qhead_per_kv_head=qhead_per_kv_head,
        cu_seqlens=cu_seqlens_q,
        cu_seqlens_scale_padded=cu_seqlens_q_scale_padded,
    )


def pack_k_scale_thd(
    k_scale: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k_scale_padded: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack CUDA ``(total_k,H_kv,G)`` THD K scales into required ``out``.

    Caller-owned ``out`` keeps the path allocation-free and avoids reading
    device prefixes on the host. It has shape
    ``(H_kv, M_capacity, ceil4(G))``, where ``M_capacity`` is a multiple of 128
    covering the padded K extent. It must be contiguous, match the input dtype
    and device, and not alias ``k_scale``. Prefix tensors must be contiguous
    int32 CUDA tensors on the input device with matching shapes. Their values
    are a caller contract. The operation is CUDA-graph capturable after warm-up;
    ``stream`` is an optional CUDA driver stream. Unsupported inputs raise
    instead of falling back to PyTorch.
    """
    if k_scale.ndim != 3:
        raise ValueError(f"k_scale must have shape (total_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    if out is None:
        raise ValueError("out is required for THD MXFP8 scale packing")
    _require_cuda_packer(k_scale, name="k_scale")
    _, n_heads_kv, sf_groups = k_scale.shape
    _validate_thd_cuda_prefixes(k_scale, cu_seqlens_k, cu_seqlens_k_scale_padded)
    _validate_thd_pack_out(out, k_scale, l_size=n_heads_kv, sf_groups=sf_groups)
    return _launch_cuda_packer(
        k_scale,
        out,
        mn=out.shape[1],
        sf_groups=sf_groups,
        l_size=n_heads_kv,
        source_layout="k_thd",
        stream=stream,
        cu_seqlens=cu_seqlens_k,
        cu_seqlens_scale_padded=cu_seqlens_k_scale_padded,
    )
