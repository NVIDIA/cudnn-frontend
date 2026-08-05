# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Torch views into the MegaMoE kernel's persistent workspace pools.

The kernel lays out its local/shared workspaces from host-visible region
tables (``kernel._local_offsets`` / ``_local_region_specs``; see
``megamoe_kernel_mxfp8._build_local_region_specs``).  Only the counter prefix
before ``l1_token_buffer`` (local) / ``src_token_topk_idx`` (shared) is
zeroed between launches, so every data region — the dispatched-token pool,
``token_src_metadata``, ``fc1_output``, the ``combine_quant`` staging — is
readable after a forward until the next launch overwrites it.  The backward
megakernel (see BWD_DESIGN.md) builds on exactly these pools; this module
gives the host-side (probe / v0 backward) access to the same bytes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

_CUTE_TO_TORCH = {
    "Uint8": torch.uint8,
    "Int32": torch.int32,
    "Int64": torch.int64,
    "Float32": torch.float32,
    "Float16": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float8E4M3FN": torch.float8_e4m3fn,
    "Float8E5M2": torch.float8_e5m2,
    "Float8E8M0FNU": getattr(torch, "float8_e8m0fnu", torch.uint8),
}


def _torch_dtype(cute_dtype) -> torch.dtype:
    name = getattr(cute_dtype, "__name__", None) or type(cute_dtype).__name__
    if name not in _CUTE_TO_TORCH:
        raise KeyError(f"no torch dtype mapping for cute dtype {name!r}")
    return _CUTE_TO_TORCH[name]


def _region_views(
    workspace: torch.Tensor, specs, offsets
) -> Dict[str, torch.Tensor]:
    views: Dict[str, torch.Tensor] = {}
    for spec in specs:
        dt = _torch_dtype(spec.cute_dtype)
        off = offsets[spec.name]
        raw = workspace[off : off + spec.nbytes]
        views[spec.name] = raw.view(dt).view(*spec.shape)
    return views


def local_pool_views(fwd) -> Dict[str, torch.Tensor]:
    """Views into ``fwd.local_workspace`` (compile must have happened)."""
    k = fwd._kernel
    return _region_views(fwd.local_workspace, k._local_region_specs, k._local_offsets)


def shared_pool_views(fwd) -> Dict[str, torch.Tensor]:
    """Views into this rank's ``fwd.shared_workspace`` slice."""
    k = fwd._kernel
    return _region_views(
        fwd.shared_workspace, k._shared_region_specs, k._shared_offsets
    )


def decode_token_src_metadata(
    meta_bytes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rows, 8) uint8 -> (src_rank, src_token, src_topk, reduce_n, consumed).

    Wire format (src/token_comm.py::TokenSrcMetadata): one little-endian i64,
    lo32 = src_token, hi32 = (src_rank << 16) | (consumed << 12)
    | (reduce_n << 8) | src_topk.
    """
    v = meta_bytes.contiguous().view(torch.int64).view(-1)
    lo = (v & 0xFFFFFFFF).to(torch.int32)
    hi = (v >> 32).to(torch.int64)
    src_rank = ((hi >> 16) & 0xFFFF).to(torch.int32)
    src_topk = (hi & 0xFF).to(torch.int32)
    reduce_n = ((hi >> 8) & 0xF).to(torch.int32)
    consumed = ((hi >> 12) & 0x1).to(torch.int32)
    return src_rank, lo, src_topk, reduce_n, consumed


def expert_slot_offsets(
    arrivals_per_expert, pad: int = 128
) -> Tuple[list, list]:
    """Pool/fc1_c row layout: expert e's arrivals start at ``doff[e]``,
    slots padded to ``pad`` rows (mega_runner.py generate_c offsets)."""
    doff, cursor = [], 0
    for n in arrivals_per_expert:
        doff.append(cursor)
        cursor += -(-int(n) // pad) * pad
    return doff, [int(n) for n in arrivals_per_expert]
