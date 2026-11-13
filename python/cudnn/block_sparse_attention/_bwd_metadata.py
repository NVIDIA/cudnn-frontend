# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt metadata of the SM90 blk64 backward (``block_sparse_attention_prepare_backward``).

The backward's metadata kernels (the bucketed K-to-Q CSR build and, on the paired-KV path, the pair bitsets, matching,
refinement and partitioned CSR) depend on the sparse pattern and the sequence lengths only, never on Q/K/V/dO. A
:class:`BsaBackwardMetadata` holds their outputs so a caller that runs several backward passes over one mask (every
layer of a training step, or the backward of the forward that just ran) launches them once, outside the backward's
critical path, and the backward itself runs the preprocess, the main kernel and the postprocess only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


def _tensor_identity(t: Optional[Tensor]) -> Optional[tuple]:
    """Storage address, shape, strides, dtype and device of a metadata tensor, plus its in-place version counter."""
    if t is None:
        return None
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t._version)


def metadata_source(q2k_block_index: Tensor, block_sparse_num: Optional[int], block_sizes: Optional[Tensor], q2k_block_nums: Optional[Tensor]) -> tuple:
    """Identity of the metadata arguments a prepared backward was built from.

    The prepared object remembers it so a backward call that passes it with other metadata, or with the same tensors
    after an in-place change, is rejected instead of silently walking the wrong pattern. ``block_sparse_num`` only
    matters with fixed counts (``q2k_block_nums=None``).
    """
    if q2k_block_nums is not None:
        block_sparse_num = None
    elif block_sparse_num is None:
        block_sparse_num = int(q2k_block_index.shape[-1])
    return (_tensor_identity(q2k_block_index), block_sparse_num, _tensor_identity(block_sizes), _tensor_identity(q2k_block_nums))


def source_mismatch(built: tuple, current: tuple) -> Optional[str]:
    """None when ``current`` names the metadata ``built`` was made from, else what differs ("different" or "modified")."""
    for name, a, b in zip(("q2k_block_index", "block_sparse_num", "block_sizes", "q2k_block_nums"), built, current):
        if isinstance(a, tuple) and isinstance(b, tuple):
            if a[:5] != b[:5]:
                return f"a different {name} tensor (storage, shape, strides, dtype or device)"
            if a[5] != b[5]:
                return f"{name} modified in place since the metadata was prepared (version counter {a[5]} -> {b[5]})"
        elif a != b:
            return f"a different {name} ({a!r} vs {b!r})"
    return None


@dataclass(eq=False)
class BsaBackwardMetadata:
    """Backward metadata of one sparse pattern for ``block_sparse_attention_backward(..., metadata=...)`` (SM90 blk64).

    Built by :func:`block_sparse_attention_prepare_backward` from ``q2k_block_index`` / ``q2k_block_nums`` /
    ``block_sizes`` and the sequence lengths only, so it serves any number of backward calls over the same metadata
    tensors. The object keeps those tensors alive and remembers their storage, shape, strides and in-place version
    counters; a backward call with other metadata, or after an in-place change of the same tensors, is rejected.
    """

    device: torch.device
    batch_size: int
    num_heads: int
    seqlen_q: int
    seqlen_k: int
    bucket_size_blocks: int
    num_q_groups: int
    use_paired_kv: bool  # the paired-KV kernel (True) or the regular blk64 kernel (False) runs over this metadata
    direct_kv: bool  # regular path with one Q group and no KV tail: dK/dV are stored directly, without the FP32 workspace
    has_kv_tail: bool
    _offsets: Tensor  # bucketed K2Q CSR offsets (batch, heads, num_q_groups, num_kv_blocks + 1); the partition kernel's order when paired
    _indices: Tensor  # bucketed K2Q CSR Q-block ids (batch, heads, max_edges)
    _block_sizes: Optional[Tensor]  # (batch, num_kv_blocks) int32 view the SM90 kernels take, or None for full blocks
    _q2k_block_index: Tensor  # the scatter kernel's input (contiguous when paired)
    _pair_nums: Optional[Tensor]  # per-row counts the pairing reads (the caller's, or a full count for fixed-count lists)
    _pair_tensors: Optional[Tuple[Tensor, ...]]  # (bits, duplicates, flags, residuals, tasks) when paired
    _pair_sync: Optional[Tensor]  # (batch, heads, 2, num_kv_blocks) int32 zeros when paired: the per-call dK/dV arrival bookkeeping template
    _source: tuple  # metadata_source(...) of the arguments this was built from
    _keep_alive: tuple  # the caller's metadata tensors, so their storage cannot be recycled behind _source
    _pairing_ready: bool = False  # the pairing kernels have run (always True for an object the public API returns)
    sparse_block_size: int = 64
    arch: int = 90

    def __repr__(self) -> str:
        path = "paired" if self.use_paired_kv else ("direct_kv" if self.direct_kv else "regular")
        return (
            f"BsaBackwardMetadata(batch_size={self.batch_size}, num_heads={self.num_heads}, seqlen_q={self.seqlen_q}, "
            f"seqlen_k={self.seqlen_k}, bucket_size_blocks={self.bucket_size_blocks}, num_q_groups={self.num_q_groups}, path={path}, device={self.device})"
        )
