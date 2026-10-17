# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SM100-family contract helpers of the indexer forward (dense and compressed-logits paths).

One place for the three rules every entry point of ``indexer_forward`` applies, so the APIBase class, the dense
interface and the compressed-logits orchestration cannot drift apart:

* which ``qhead_per_kv_head`` groups the unified score kernel serves per precision,
* how many query tokens one MMA tile packs for a head group (``m_block_size = qhead_per_kv_head * q_tokens``),
* the geometry sanity check between a query length and the compressed key length it is scored against.
"""

from __future__ import annotations

# Head groups the BF16 unified score kernel packs along the MMA N axis. The epilogue head-reduce walks
# ``qhead_per_kv_head`` TMEM columns per query token in pairs, so a group must be even and the N tile
# (``qhead_per_kv_head * q_tokens_per_tile``) a multiple of 16. The small groups (4, 8, 16) serve
# block-sparse selectors with a few indexer heads per token; 32 and 64 are the original targets.
SUPPORTED_QHEAD_PER_KV_HEAD_BF16 = (4, 8, 16, 32, 64)
# The MXFP8 kernel keeps its 128-row tile for every group (its scale atom is 128 packed rows, so the
# scale packing ``packed_m = token * group + head`` does not depend on the group): a small group packs
# ``128 // group`` tokens per tile, split between the two epilogue warpgroups. Same set as BF16; a
# plain-e4m3 scorer is the MXFP8 kernel with all-ones E8M0 scale blobs.
SUPPORTED_QHEAD_PER_KV_HEAD_MXFP8 = (4, 8, 16, 32, 64)

# Query tokens packed per MMA tile at head_dim 128, keyed by head group. The 32 / 64 entries are the
# measured SM100 picks of each path and stay as they were (the dense path packs 4 tokens x 32 heads, the
# compressed-logits path 2); the small groups pack 8 tokens so the tile keeps at least 32 N columns
# (4 heads x 8 tokens = 32) -- the per-group reduce and the compact store are per token, so the tile
# width changes the schedule only, never a score.
_DENSE_Q_TOKENS_PER_TILE = {64: 2, 32: 4, 16: 8, 8: 8, 4: 8}
_COMPRESSED_Q_TOKENS_PER_TILE = {64: 2, 32: 2, 16: 8, 8: 8, 4: 8}


def supported_qhead_per_kv_head(precision: str) -> tuple[int, ...]:
    """Head groups the SM100 unified indexer-score kernel serves for ``precision``."""
    return SUPPORTED_QHEAD_PER_KV_HEAD_MXFP8 if precision == "mxfp8" else SUPPORTED_QHEAD_PER_KV_HEAD_BF16


def validate_qhead_per_kv_head(qhead_per_kv_head: int, precision: str) -> None:
    """Typed decline for a head group the kernel does not pack (``ValueError`` naming the supported set)."""
    supported = supported_qhead_per_kv_head(precision)
    if qhead_per_kv_head not in supported:
        raise ValueError(f"precision={precision!r} indexer requires qhead_per_kv_head in {supported}, got {qhead_per_kv_head}")


def max_q_tokens_per_tile(qhead_per_kv_head: int, head_dim: int, *, compressed: bool) -> int:
    """Query tokens one BF16 MMA tile packs for ``qhead_per_kv_head`` (the table above at head_dim 128, else 2)."""
    if head_dim != 128:
        return 2
    table = _COMPRESSED_Q_TOKENS_PER_TILE if compressed else _DENSE_Q_TOKENS_PER_TILE
    return table.get(qhead_per_kv_head, 2)


def resolve_m_block_size(m_block_size: int, qhead_per_kv_head: int, head_dim: int, *, compressed: bool, path: str) -> int:
    """Resolve the packed-M tile width of the BF16 kernel.

    The default ``m_block_size=128`` shrinks to ``qhead_per_kv_head * max_q_tokens`` for a head group that
    would otherwise pack more query tokens than the kernel's epilogue carries; an explicit width past that
    cap is a ``ValueError``.
    """
    cap = max_q_tokens_per_tile(qhead_per_kv_head, head_dim, compressed=compressed)
    if m_block_size // qhead_per_kv_head > cap:
        if m_block_size == 128:
            return qhead_per_kv_head * cap
        raise ValueError(f"{path} supports at most {cap} q tokens per tile for qhead_per_kv_head={qhead_per_kv_head}; " f"got m_block_size={m_block_size}")
    return m_block_size


def check_q_covered_by_k(seqlen_q: int, seqlen_k: int, ratio: int, *, what_q: str = "seqlen_q", what_k: str = "seqlen_k") -> None:
    """Geometry check: the longest query row must not see more compressed keys than ``seqlen_k`` holds.

    Under the top-left ratio-causal mask row ``r`` sees ``floor((r + 1) / ratio)`` compressed keys, so the
    last row of ``seqlen_q`` rows sees ``floor(seqlen_q / ratio)``; that count fits ``seqlen_k`` keys exactly
    when ``seqlen_q <= seqlen_k * ratio + (ratio - 1)``. The ``ratio - 1`` trailing tokens are the ones that
    have not completed a compressed block yet (a 4:1 compression of 2051 tokens holds 512 keys): they score
    against the complete blocks only and are a legitimate geometry, not a mismatch.
    """
    if seqlen_q > seqlen_k * ratio + (ratio - 1):
        raise ValueError(
            f"{what_q} ({seqlen_q}) must be <= {what_k} * ratio + (ratio - 1) ({seqlen_k * ratio + ratio - 1}): "
            f"its last row would see more compressed keys than {what_k} holds"
        )
