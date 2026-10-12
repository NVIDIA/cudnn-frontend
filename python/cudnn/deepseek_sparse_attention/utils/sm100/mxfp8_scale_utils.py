# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 scale packing helpers for the SM100 Indexer DSL path.

CUDA E8M0 inputs use a shape-generic single-launch packer when the GPU and
CuTe DSL version support it. Other environments, CPU tensors, non-byte dtypes
and strided inputs retain the PyTorch reference implementation.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch

from cudnn._torch_stream import device_context, record_streams
from cudnn.deepseek_sparse_attention.utils.runtime import ceil_div as _ceil_div, resolve_stream, torch_stream_context

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
    """Copy a logical prefix to the host and validate its exact extent."""
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


def minimal_padded_token_capacity(
    total_tokens: int,
    num_sequences: int,
    token_alignment: int,
) -> int:
    """Return a host-only capacity for minimally padded THD scale prefixes.

    If ``cu_seqlens_scale_padded`` was produced by
    :func:`make_scale_cu_seqlens_padded`, its final value is at most this
    capacity.  The formula depends only on declared tensor shapes, so callers
    can allocate reusable ``out`` storage without reading a device prefix:
    ``floor((T + (alignment-1)*min(T,B)) / alignment) * alignment``.  The
    ``min(T,B)`` term excludes necessarily empty sequences from the padding
    envelope.
    Custom prefixes with padding beyond the minimum must size ``out`` from
    their own host-side envelope instead.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be nonnegative")
    if num_sequences < 0:
        raise ValueError("num_sequences must be nonnegative")
    if num_sequences == 0 and total_tokens != 0:
        raise ValueError("num_sequences must be positive when total_tokens is nonzero")
    if token_alignment <= 0:
        raise ValueError("token_alignment must be positive")
    # At most one nonempty sequence can be created by each token.  Empty
    # sequences contribute no padding, so using num_sequences directly would
    # overestimate the envelope when total_tokens < num_sequences.
    active_sequences = min(int(total_tokens), int(num_sequences))
    return ((int(total_tokens) + (int(token_alignment) - 1) * active_sequences) // int(token_alignment)) * int(token_alignment)


def _validate_scale_cu_seqlens_padded(
    cu_seqlens: list[int],
    cu_seqlens_scale_padded: torch.Tensor,
    *,
    token_alignment: int,
    name: str,
) -> list[int]:
    """Copy a padded prefix to the host and validate every sequence span."""
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


def _packed_scale_shape(mn: int, sf_groups: int, l_size: int) -> tuple[int, int, int]:
    """Return the physical ``(L, padded_MN, padded_groups)`` shape."""
    return (
        l_size,
        _ceil_div(mn, _ATOM_MN) * _ATOM_MN,
        _ceil_div(sf_groups, _ATOM_K) * _ATOM_K,
    )


def _validate_pack_out(
    out: torch.Tensor,
    src: torch.Tensor,
    shape: tuple[int, int, int],
) -> None:
    """Validate exact-shape caller-owned packed storage."""
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}, got {tuple(out.shape)}")
    _validate_pack_out_storage(out, src)


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


def _validate_thd_pack_out(
    out: torch.Tensor,
    src: torch.Tensor,
    *,
    l_size: int,
    sf_groups: int,
    min_rows: int | None = None,
) -> None:
    """Validate caller-owned THD storage and an optional live-row extent."""
    expected_tail = _ceil_div(sf_groups, _ATOM_K) * _ATOM_K
    if out.ndim != 3 or out.shape[0] != l_size or out.shape[1] % _ATOM_MN != 0 or out.shape[2] != expected_tail:
        raise ValueError(f"out must have shape ({l_size}, multiple_of_128, {expected_tail}), got {tuple(out.shape)}")
    _validate_pack_out_storage(out, src)
    if min_rows is not None and out.shape[1] < min_rows:
        raise ValueError(f"out capacity {tuple(out.shape)} cannot hold packed extent ({l_size}, {min_rows}, {expected_tail})")


def _allocate_pack_out(
    src: torch.Tensor,
    shape: tuple[int, int, int],
    out: torch.Tensor | None,
    stream,
) -> torch.Tensor:
    """Return validated caller storage or allocate an exact packed tensor."""
    if out is not None:
        _validate_pack_out(out, src, shape)
        return out
    if src.is_cuda:
        from cudnn.deepseek_sparse_attention.utils.runtime import torch_stream_context

        with torch_stream_context(stream, src.device):
            return torch.empty(shape, device=src.device, dtype=src.dtype)
    return torch.empty(shape, device=src.device, dtype=src.dtype)


@lru_cache(maxsize=None)
def _cuda_packer_supports_device(device_index: int) -> bool:
    """Keep the reference path on the package's older DSL floor and other GPUs."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        return False
    from cudnn.deepseek_sparse_attention.utils.compiler import _ARCH_MAP
    from cudnn.deepseek_sparse_attention.utils.runtime import device_capability

    return device_capability(device_index) in _ARCH_MAP


def _can_use_cuda_packer(scale: torch.Tensor) -> bool:
    """Return whether both the tensor and environment support direct packing."""
    return scale.is_cuda and scale.element_size() == 1 and scale.is_contiguous() and _cuda_packer_supports_device(scale.device.index)


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
    src_bytes = scale.view(torch.uint8).reshape(-1)
    dst_bytes = packed.view(torch.uint8).reshape(-1)
    with device_context(scale.device):
        record_streams((scale, packed, cu_seqlens, cu_seqlens_scale_padded), stream, scale.device)
        launch_stream = resolve_stream(stream)
        launch_pack(
            src_bytes,
            dst_bytes,
            mn=mn,
            sf_groups=sf_groups,
            l_size=l_size,
            source_layout=source_layout,
            stream=launch_stream,
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
    """Launch the fused BSHD Q/K packer and return both output tensors."""
    _, launch_qk_bshd_pack = _cuda_pack_launchers()
    q_src_bytes = q_scale.view(torch.uint8).reshape(-1)
    k_src_bytes = k_scale.view(torch.uint8).reshape(-1)
    q_dst_bytes = q_packed.view(torch.uint8).reshape(-1)
    k_dst_bytes = k_packed.view(torch.uint8).reshape(-1)
    with device_context(q_scale.device):
        record_streams((q_scale, k_scale, q_packed, k_packed), stream, q_scale.device)
        launch_stream = resolve_stream(stream)
        launch_qk_bshd_pack(
            q_src_bytes,
            k_src_bytes,
            q_dst_bytes,
            k_dst_bytes,
            q_mn=q_mn,
            k_mn=k_mn,
            sf_groups=sf_groups,
            l_size=l_size,
            q_seq_len=q_seq_len,
            k_seq_len=k_seq_len,
            n_heads_kv=n_heads_kv,
            qhead_per_kv_head=qhead_per_kv_head,
            stream=launch_stream,
        )
    return q_packed, k_packed


def pack_blockscaled_scale_mkl(
    scale_mkl: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack logical ``(MN, sf_groups, L)`` scales into Blackwell SF layout.

    The returned tensor has physical shape ``(L, padded_MN, padded_sf_groups)``.
    ``sf_groups`` is ``ceil_div(head_dim, 32)`` for MXFP8.  For a contiguous
    one-byte CUDA input, packing and padding use one asynchronous CuTe DSL
    launch on ``stream``; ``out`` may provide caller-owned storage and makes
    the operation allocation-free.  ``out`` must be contiguous, have the
    returned shape/dtype/device, and must not alias ``scale_mkl``.  The return
    value is ``out`` when it is supplied.  ``stream`` is an optional CUDA
    driver stream and defaults to PyTorch's current stream on the input device.
    The caller-owned CUDA path is graph-capturable after its first, compiling
    call; input and output allocator lifetimes are recorded on the launch
    stream.  CPU, non-byte, and unsupported GPU/DSL environments use the
    allocation-compatible PyTorch reference path.
    """
    if scale_mkl.ndim != 3:
        raise ValueError(f"scale_mkl must have shape (MN, sf_groups, L), got {tuple(scale_mkl.shape)}")
    mn, sf_groups, l_size = scale_mkl.shape
    shape = _packed_scale_shape(mn, sf_groups, l_size)
    packed = _allocate_pack_out(scale_mkl, shape, out, stream)
    if _can_use_cuda_packer(scale_mkl):
        return _launch_cuda_packer(
            scale_mkl,
            packed,
            mn=mn,
            sf_groups=sf_groups,
            l_size=l_size,
            source_layout="mkl",
            stream=stream,
        )
    with torch_stream_context(stream, scale_mkl.device):
        record_streams((scale_mkl, packed), stream, scale_mkl.device)
        packed.zero_()
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


def pack_q_scale_bshd(
    q_scale: torch.Tensor,
    qhead_per_kv_head: int = 64,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack logical ``(B,S_q,H_q,G)`` Q scales for the Indexer MXFP8 kernel.

    Contiguous one-byte CUDA inputs are read directly in BSHD order and packed
    in one asynchronous launch, without an intermediate MKL tensor.  ``out``
    is an optional caller-owned contiguous ``(B*H_kv, ceil128(S_q*q),
    ceil4(G))`` tensor of the same dtype/device, where
    ``q=qhead_per_kv_head``.  It must not alias ``q_scale``.  The return value
    aliases ``out`` when supplied.  ``stream`` is an optional CUDA driver
    stream and otherwise means PyTorch's current stream on the input device.
    The caller-owned path performs no allocation, records buffer lifetimes on
    the launch stream, and is CUDA-graph capturable after the first JIT compile.
    CPU, non-byte, and unsupported GPU/DSL environments use the PyTorch
    reference implementation.
    """
    if q_scale.ndim != 4:
        raise ValueError(f"q_scale must have shape (B, S_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    bs, seqlen_q, n_heads_q, sf_groups = q_scale.shape
    if n_heads_q % qhead_per_kv_head != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by qhead_per_kv_head ({qhead_per_kv_head})")
    n_heads_kv = n_heads_q // qhead_per_kv_head
    mn = seqlen_q * qhead_per_kv_head
    shape = _packed_scale_shape(mn, sf_groups, bs * n_heads_kv)
    packed = _allocate_pack_out(q_scale, shape, out, stream)
    if _can_use_cuda_packer(q_scale):
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
    with torch_stream_context(stream, q_scale.device):
        record_streams((q_scale, packed), stream, q_scale.device)
        return pack_blockscaled_scale_mkl(logical_q_scale_to_mkl(q_scale, qhead_per_kv_head), out=packed, stream=stream)


def pack_k_scale_bshd(
    k_scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack logical ``(B,S_k,H_kv,G)`` K scales for the Indexer MXFP8 kernel.

    Contiguous one-byte CUDA inputs are packed directly in one asynchronous
    launch.  ``out`` is an optional caller-owned contiguous
    ``(B*H_kv, ceil128(S_k), ceil4(G))`` tensor of the same dtype/device; it
    must not alias ``k_scale``.  The return value aliases ``out`` when supplied.
    ``stream`` is an optional CUDA driver stream and otherwise means PyTorch's
    current stream on the input device.  The caller-owned path performs no
    allocation, records buffer lifetimes on the launch stream, and is
    CUDA-graph capturable after the first JIT compile.  CPU, non-byte, and unsupported GPU/DSL environments
    use the PyTorch reference implementation.
    """
    if k_scale.ndim != 4:
        raise ValueError(f"k_scale must have shape (B, S_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    bs, seqlen_k, n_heads_kv, sf_groups = k_scale.shape
    shape = _packed_scale_shape(seqlen_k, sf_groups, bs * n_heads_kv)
    packed = _allocate_pack_out(k_scale, shape, out, stream)
    if _can_use_cuda_packer(k_scale):
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
    with torch_stream_context(stream, k_scale.device):
        record_streams((k_scale, packed), stream, k_scale.device)
        return pack_blockscaled_scale_mkl(logical_k_scale_to_mkl(k_scale), out=packed, stream=stream)


def pack_qk_scale_bshd(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    qhead_per_kv_head: int = 64,
    *,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack logical BSHD Q and K scales together for the MXFP8 indexer.

    ``q_scale`` has shape ``(B,S_q,H_q,G)`` and ``k_scale`` has shape
    ``(B,S_k,H_kv,G)``.  They must share batch size, scale groups, dtype,
    device, and ``H_q == H_kv * qhead_per_kv_head``.  Contiguous one-byte CUDA
    inputs are packed in one asynchronous launch.  The returned tensors have
    shapes ``(B*H_kv, ceil128(S_q*q), ceil4(G))`` and
    ``(B*H_kv, ceil128(S_k), ceil4(G))``, where
    ``q=qhead_per_kv_head``.

    ``q_out`` and ``k_out`` optionally provide caller-owned contiguous storage
    of those shapes, dtype, and device; neither may alias either input.  When
    both are supplied the CUDA path allocates nothing, records all four buffer
    lifetimes on ``stream``, and is CUDA-graph capturable after its first JIT
    compile.  ``stream`` is an optional CUDA driver stream and otherwise means
    PyTorch's current stream on the input device.  CPU, non-byte, strided, or unsupported GPU/DSL
    inputs use the two reference packers and preserve the same return contract.
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
        raise ValueError(f"n_heads_q ({n_heads_q}) must equal n_heads_kv * qhead_per_kv_head " f"({n_heads_kv} * {qhead_per_kv_head})")
    if q_scale.dtype != k_scale.dtype or q_scale.device != k_scale.device:
        raise ValueError("q_scale and k_scale must have the same dtype and device")

    q_mn = seqlen_q * qhead_per_kv_head
    l_size = bs * n_heads_kv
    q_shape = _packed_scale_shape(q_mn, sf_groups, l_size)
    k_shape = _packed_scale_shape(seqlen_k, sf_groups, l_size)
    q_packed = _allocate_pack_out(q_scale, q_shape, q_out, stream)
    k_packed = _allocate_pack_out(k_scale, k_shape, k_out, stream)
    for packed, name in ((q_packed, "q_out"), (k_packed, "k_out")):
        if _shares_nonempty_storage(packed, q_scale) or _shares_nonempty_storage(packed, k_scale):
            raise ValueError(f"{name} must not alias either source scale tensor")
    if _shares_nonempty_storage(q_packed, k_packed):
        raise ValueError("q_out and k_out must not alias each other")

    if _can_use_cuda_packer(q_scale) and _can_use_cuda_packer(k_scale):
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

    return (
        pack_q_scale_bshd(q_scale, qhead_per_kv_head, out=q_packed, stream=stream),
        pack_k_scale_bshd(k_scale, out=k_packed, stream=stream),
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
    """Pack logical ``(total_q,H_q,G)`` THD Q scales for the MXFP8 indexer.

    For a contiguous one-byte CUDA input, supplying caller-owned ``out``
    enables the fully asynchronous path: one kernel reads both device prefixes,
    copies live scale bytes, and writes all per-sequence and capacity padding as
    ``0x00``.  No prefix value is read on the host.  ``out`` must be
    contiguous, have the input dtype/device, and have shape
    ``(H_kv, M_capacity, ceil4(G))`` with ``M_capacity`` a multiple of
    128 large enough for ``cu_seqlens_q_scale_padded[-1] *
    qhead_per_kv_head``.  It must not alias ``q_scale``.  Any reference
    fallback (CPU, non-byte, strided input, or unsupported GPU/DSL) reads both prefix tensors on the
    host, allocates an exact-size temporary, then copies it into ``out`` and
    zero-fills unused capacity; that path cannot run during CUDA Graph capture.
    Without ``out`` the reference path returns its exact live extent.  The
    return value aliases ``out`` when supplied.
    ``stream`` is an optional CUDA driver stream and otherwise means PyTorch's
    current stream on the input device.  The direct CUDA caller-owned path
    records all buffer lifetimes on that stream and is CUDA-graph capturable
    after the first JIT compile.  On that path both prefix tensors are
    contiguous CUDA int32 arrays of
    length ``B+1`` on the input device.  ``cu_seqlens_q`` starts at zero, ends
    at ``total_q``, and is nondecreasing; the padded prefix starts at zero, is
    nondecreasing, and each span covers its logical sequence with the required
    Q-token alignment.  Those device values are a caller contract and are not
    read back for host validation on the asynchronous path.
    """
    if q_scale.ndim != 3:
        raise ValueError(f"q_scale must have shape (total_q, H_q, sf_groups), got {tuple(q_scale.shape)}")
    if qhead_per_kv_head <= 0:
        raise ValueError("qhead_per_kv_head must be positive")
    _total_q, n_heads_q, sf_groups = q_scale.shape
    if n_heads_q % qhead_per_kv_head != 0:
        raise ValueError(f"n_heads_q ({n_heads_q}) must be divisible by qhead_per_kv_head ({qhead_per_kv_head})")
    n_heads_kv = n_heads_q // qhead_per_kv_head
    if out is not None and _can_use_cuda_packer(q_scale):
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
    with torch_stream_context(stream, q_scale.device):
        record_streams((q_scale, out, cu_seqlens_q, cu_seqlens_q_scale_padded), stream, q_scale.device)
        if out is not None:
            _validate_thd_pack_out(out, q_scale, l_size=n_heads_kv, sf_groups=sf_groups)
        packed = pack_blockscaled_scale_mkl(
            logical_q_scale_to_mkl_thd(
                q_scale,
                cu_seqlens_q,
                cu_seqlens_q_scale_padded,
                qhead_per_kv_head,
            ),
            stream=stream,
        )
        if out is None:
            return packed
        _validate_thd_pack_out(out, q_scale, l_size=n_heads_kv, sf_groups=sf_groups, min_rows=packed.shape[1])
        record_streams((packed,), stream, q_scale.device)
        out[:, : packed.shape[1]].copy_(packed)
        out[:, packed.shape[1] :].zero_()
        return out


def pack_k_scale_thd(
    k_scale: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k_scale_padded: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Pack logical ``(total_k,H_kv,G)`` THD K scales for the MXFP8 indexer.

    For a contiguous one-byte CUDA input, caller-owned ``out`` makes packing
    one asynchronous launch with no allocation or device-to-host read.  ``out``
    must be contiguous, have the input dtype/device, and have shape
    ``(H_kv, M_capacity, ceil4(G))`` with a multiple-of-128 capacity covering
    ``cu_seqlens_k_scale_padded[-1]``; it must not alias ``k_scale``.  Any
    reference fallback (CPU, non-byte, strided input, or unsupported GPU/DSL) reads both prefix
    tensors on the host, allocates an exact-size temporary, then copies it into
    ``out`` and zero-fills unused capacity; that path cannot run during CUDA
    Graph capture.  Without ``out`` the reference path returns its exact live
    extent.  The return value aliases ``out`` when supplied.
    ``stream`` is an optional CUDA driver stream and otherwise means PyTorch's
    current stream on the input device.  The direct CUDA caller-owned path
    records all buffer lifetimes on that stream and is CUDA-graph capturable
    after the first JIT compile.  On that path both prefix tensors are
    contiguous CUDA int32 arrays of
    length ``B+1`` on the input device.  ``cu_seqlens_k`` starts at zero, ends
    at ``total_k``, and is nondecreasing; the padded prefix starts at zero, is
    nondecreasing, and each span covers its logical sequence with 128-token
    alignment.  Those device values are a caller contract and are not read
    back for host validation on the asynchronous path.
    """
    if k_scale.ndim != 3:
        raise ValueError(f"k_scale must have shape (total_k, H_kv, sf_groups), got {tuple(k_scale.shape)}")
    _total_k, n_heads_kv, sf_groups = k_scale.shape
    if out is not None and _can_use_cuda_packer(k_scale):
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
    with torch_stream_context(stream, k_scale.device):
        record_streams((k_scale, out, cu_seqlens_k, cu_seqlens_k_scale_padded), stream, k_scale.device)
        if out is not None:
            _validate_thd_pack_out(out, k_scale, l_size=n_heads_kv, sf_groups=sf_groups)
        packed = pack_blockscaled_scale_mkl(
            logical_k_scale_to_mkl_thd(
                k_scale,
                cu_seqlens_k,
                cu_seqlens_k_scale_padded,
            ),
            stream=stream,
        )
        if out is None:
            return packed
        _validate_thd_pack_out(out, k_scale, l_size=n_heads_kv, sf_groups=sf_groups, min_rows=packed.shape[1])
        record_streams((packed,), stream, k_scale.device)
        out[:, : packed.shape[1]].copy_(packed)
        out[:, packed.shape[1] :].zero_()
        return out
