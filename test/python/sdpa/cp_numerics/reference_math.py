# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R0: an independent FP64 mathematical oracle for chunked attention merging.

This module is deliberately *not* part of the code under test. It contains three
separate implementations that are used for three different purposes:

``full_attention_fp64``
    materialises the whole ``[Sq, Skv]`` score matrix and normalises it in one
    shot. This is the **expected** value for every R0/R1 comparison. It never
    looks at chunk partials and never runs a merge.
``chunk_partials_fp64`` / ``chunk_partial_with_mask_fp64``
    normalise one key/value chunk on its own, producing ``(O_i, L_i)`` pairs.
    These are the **inputs** handed to the merge under test. The chunked path
    shares no code with ``full_attention_fp64`` beyond the mask predicate, so a
    bug in the monolithic path cannot silently cancel a bug in the chunked path.
``analytic_merge_fp64``
    the closed-form merge from the execution plan::

        L = log(sum_i exp(L_i))
        O = sum_i exp(L_i - L) * O_i

    It exists **only** to cross-check ``full_attention_fp64`` against
    ``chunk_partials_fp64`` at FP64 (a "does the fixture make sense" check). It
    is *not* the expected value for the merged result under test, and R1 must not
    be graded against it: the plan explicitly forbids substituting a
    ``logsumexp`` plus reduction for the target implementation's per-step order.

Semantics that are fixed here (and recorded in every fixture header):

* LSE is the **natural** logarithm of the sum of exponentials (``lse_base =
  "e"``). The plan forbids mixing natural-log LSE with log2 LSE.
* Masking is **top-left causal**: a query at global position ``p`` may attend to
  a key at global position ``j`` iff ``j <= p``. ``bottom_right`` alignment is
  not implemented here and is not claimed.
* An **empty chunk** for a query row (no unmasked key in that chunk) yields
  ``L_i = -inf`` and ``O_i = 0``.
* A **fully masked row** (no unmasked key in any chunk) yields ``L = -inf`` and
  ``O = 0``. The reference therefore never emits NaN for an all-masked row; a
  target implementation that does emit NaN is reported as a *behavioural
  difference*, never silently patched to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

#: ``lse_base`` value this module produces and expects.
LSE_BASE = "e"

#: Sentinel used for "this row has no unmasked key".
EMPTY_LSE: float = float("-inf")


@dataclass(frozen=True)
class AttentionResult:
    """Result of one attention computation.

    ``out`` is ``[B, Sq, H, D]`` and ``lse`` is ``[B, H, Sq]``; both are FP64.
    """

    out: torch.Tensor
    lse: torch.Tensor

    def __post_init__(self) -> None:
        if self.out.dtype != torch.float64 or self.lse.dtype != torch.float64:
            raise TypeError("R0 oracle is FP64-only")
        if self.out.dim() != 4 or self.lse.dim() != 3:
            raise ValueError("out must be [B, Sq, H, D] and lse [B, H, Sq]")
        if (self.out.shape[0], self.out.shape[1], self.out.shape[2]) != (
            self.lse.shape[0],
            self.lse.shape[2],
            self.lse.shape[1],
        ):
            raise ValueError(f"shape mismatch: out {tuple(self.out.shape)} vs lse {tuple(self.lse.shape)}")


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------


def top_left_causal_keep_mask(
    q_len: int,
    kv_len: int,
    *,
    q_offset: int = 0,
    kv_offset: int = 0,
    causal: bool,
    valid_kv_len: Optional[int] = None,
) -> torch.Tensor:
    """Boolean ``[q_len, kv_len]`` mask; ``True`` means "may attend".

    ``q_offset`` / ``kv_offset`` are the global positions of index 0 of each
    axis, so a chunked computation can reuse the same global predicate.
    ``valid_kv_len`` excludes a ragged tail (padding) from the valid key set.
    """
    if q_len < 0 or kv_len < 0:
        raise ValueError("lengths must be non-negative")
    q_pos = torch.arange(q_offset, q_offset + q_len, dtype=torch.int64).unsqueeze(1)
    kv_pos = torch.arange(kv_offset, kv_offset + kv_len, dtype=torch.int64).unsqueeze(0)
    keep = torch.ones((q_len, kv_len), dtype=torch.bool)
    if causal:
        keep &= kv_pos <= q_pos
    if valid_kv_len is not None:
        keep &= kv_pos < valid_kv_len
    return keep


def causal_keep_mask_from_positions(
    q_global_positions: torch.Tensor,
    kv_global_positions: torch.Tensor,
    *,
    causal: bool,
    valid_kv_len: Optional[int] = None,
) -> torch.Tensor:
    """Mask from explicit global positions, for axes that concatenate disjoint chunks."""
    q_pos = q_global_positions.to(torch.int64).unsqueeze(1)
    kv_pos = kv_global_positions.to(torch.int64).unsqueeze(0)
    keep = torch.ones((q_pos.shape[0], kv_pos.shape[1]), dtype=torch.bool)
    if causal:
        keep &= kv_pos <= q_pos
    if valid_kv_len is not None:
        keep &= kv_pos < valid_kv_len
    return keep


# ---------------------------------------------------------------------------
# path A: monolithic explicit attention (the expected values)
# ---------------------------------------------------------------------------


def full_attention_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    q_offset: int = 0,
    kv_offset: int = 0,
    valid_kv_len: Optional[int] = None,
) -> AttentionResult:
    """Monolithic FP64 attention over the complete key/value axis.

    Shapes: ``q`` ``[B, Sq, H, D]``, ``k``/``v`` ``[B, Skv, H, D]``. The score
    matrix is materialised in full, which is exactly why this path is
    independent of any chunking or merge logic.
    """
    q64 = q.detach().to(torch.float64)
    k64 = k.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    if q64.dim() != 4 or k64.dim() != 4 or v64.dim() != 4:
        raise ValueError("expected [B, S, H, D] tensors")
    if k64.shape != v64.shape:
        raise ValueError("k and v must have the same shape")
    if q64.shape[0] != k64.shape[0] or q64.shape[2] != k64.shape[2] or q64.shape[3] != k64.shape[3]:
        raise ValueError("q/k/v batch, head and head-dim must match")

    scores = torch.einsum("bqhd,bkhd->bhqk", q64, k64) * float(scale)  # [B, H, Sq, Skv]
    keep = top_left_causal_keep_mask(
        scores.shape[-2],
        scores.shape[-1],
        q_offset=q_offset,
        kv_offset=kv_offset,
        causal=causal,
        valid_kv_len=valid_kv_len,
    )
    return _normalise(scores, v64, keep)


def full_attention_with_mask_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    keep_mask: torch.Tensor,
) -> AttentionResult:
    """Monolithic FP64 attention with an explicit ``[Sq, Skv]`` keep mask.

    Used when the fixture deliberately removes a chunk's contribution (the
    empty-chunk probes): the expected value must then be recomputed over the
    *retained* key set, not taken from the untouched full attention.
    """
    q64 = q.detach().to(torch.float64)
    k64 = k.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    if keep_mask.shape != (q64.shape[1], k64.shape[1]):
        raise ValueError(f"keep_mask {tuple(keep_mask.shape)} does not match {(q64.shape[1], k64.shape[1])}")
    scores = torch.einsum("bqhd,bkhd->bhqk", q64, k64) * float(scale)
    return _normalise(scores, v64, keep_mask.to(torch.bool))


def _normalise(scores: torch.Tensor, v64: torch.Tensor, keep: torch.Tensor) -> AttentionResult:
    """Shared FP64 softmax normalisation; ``scores`` is pre-masked-aggregation."""
    scores = scores.masked_fill(~keep, EMPTY_LSE)
    row_max = scores.amax(dim=-1)  # -inf for a fully masked row
    finite_row_max = torch.isfinite(row_max)
    safe_max = torch.where(finite_row_max, row_max, torch.zeros_like(row_max))
    p = torch.exp(scores - safe_max.unsqueeze(-1))
    denom = p.sum(dim=-1)  # 0 for a fully masked row
    has_mass = denom > 0
    lse = torch.where(has_mass, torch.log(denom) + safe_max, torch.full_like(denom, EMPTY_LSE))
    out = torch.einsum("bhqk,bkhd->bqhd", p, v64)
    safe_denom = torch.where(has_mass, denom, torch.ones_like(denom))
    # [B, H, Sq] -> [B, Sq, H, 1] to broadcast against the [B, Sq, H, D] output
    keep_out = has_mass.movedim(1, 2).unsqueeze(-1)
    out = torch.where(keep_out, out / safe_denom.movedim(1, 2).unsqueeze(-1), torch.zeros_like(out))
    # ``movedim`` produces a permuted view, and TensorIterator can hand back an
    # output that inherits that layout. Consumers (and TE's own ``out.view``)
    # assume a dense kernel-output layout, so normalise it here.
    return AttentionResult(out=out.contiguous(), lse=lse.contiguous())


# ---------------------------------------------------------------------------
# path B: chunked partials (the inputs to the merge under test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPartial:
    """One ``(O_i, L_i)`` pair: ``out`` ``[B, Sq, H, D]``, ``lse`` ``[B, H, Sq]``."""

    chunk_index: int
    kv_begin: int
    kv_end: int
    out: torch.Tensor
    lse: torch.Tensor


def chunk_partial_with_mask_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    keep_mask: torch.Tensor,
    chunk_index: int = 0,
    kv_begin: int = 0,
    kv_end: Optional[int] = None,
) -> ChunkPartial:
    """Normalise one key/value slice with an explicit ``[Sq, Skv]`` keep mask.

    This is the general entry point: it can express a KV axis built from several
    disjoint global chunks (TE's diagonal / all sections concatenate the source
    rank's two chunks), which a single ``kv_offset`` cannot.
    """
    q64 = q.detach().to(torch.float64)
    k64 = k.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    if keep_mask.shape != (q64.shape[1], k64.shape[1]):
        raise ValueError(f"keep_mask {tuple(keep_mask.shape)} does not match q/kv lengths {(q64.shape[1], k64.shape[1])}")
    scores = torch.einsum("bqhd,bkhd->bhqk", q64, k64) * float(scale)
    result = _normalise(scores, v64, keep_mask.to(torch.bool))
    return ChunkPartial(
        chunk_index=chunk_index,
        kv_begin=kv_begin,
        kv_end=k64.shape[1] if kv_end is None else kv_end,
        out=result.out,
        lse=result.lse,
    )


def chunk_partials_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    chunk_bounds: Sequence[Tuple[int, int]],
    q_offset: int = 0,
    valid_kv_len: Optional[int] = None,
) -> List[ChunkPartial]:
    """Normalise each contiguous key/value chunk independently, in FP64.

    Each chunk gets its **own** row max, so ``L_i`` is a per-chunk log-sum-exp
    and ``O_i`` is already softmax-normalised inside the chunk. A chunk with no
    unmasked key for a row contributes ``L_i = -inf`` and ``O_i = 0``.
    """
    q64 = q.detach().to(torch.float64)
    k64 = k.detach().to(torch.float64)
    v64 = v.detach().to(torch.float64)
    partials: List[ChunkPartial] = []
    for index, (begin, end) in enumerate(chunk_bounds):
        if end < begin:
            raise ValueError(f"chunk {index}: end {end} < begin {begin}")
        keep = top_left_causal_keep_mask(
            q64.shape[1],
            end - begin,
            q_offset=q_offset,
            kv_offset=begin,
            causal=causal,
            valid_kv_len=valid_kv_len,
        )
        partials.append(
            chunk_partial_with_mask_fp64(
                q64,
                k64[:, begin:end],
                v64[:, begin:end],
                scale=scale,
                keep_mask=keep,
                chunk_index=index,
                kv_begin=begin,
                kv_end=end,
            )
        )
    return partials


# ---------------------------------------------------------------------------
# path C: analytic merge (cross-check only -- NEVER the expected value for R1)
# ---------------------------------------------------------------------------


def analytic_merge_fp64(partials: Sequence[ChunkPartial]) -> AttentionResult:
    """Closed-form merge in FP64. Cross-check helper only; see the module docstring.

    Rows with no mass in any chunk get ``L = -inf`` and ``O = 0`` by explicit
    policy rather than by letting ``exp(-inf - -inf)`` produce NaN.
    """
    if not partials:
        raise ValueError("no partials to merge")
    lses = torch.stack([p.lse for p in partials], dim=0)  # [N, B, H, Sq]
    outs = torch.stack([p.out for p in partials], dim=0)  # [N, B, Sq, H, D]
    finite = torch.isfinite(lses)
    any_finite = finite.any(dim=0)
    # ``logsumexp`` must see the real ``-inf``: substituting 0 for an empty
    # chunk's L would add exp(0) to the sum and corrupt every row that has one.
    global_lse = torch.logsumexp(lses, dim=0)
    global_lse = torch.where(any_finite, global_lse, torch.full_like(global_lse, EMPTY_LSE))
    weights = torch.exp(lses - global_lse.unsqueeze(0))  # [N, B, H, Sq]
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    out = (weights.movedim(2, 3).unsqueeze(-1) * outs).sum(dim=0)
    return AttentionResult(out=out.contiguous(), lse=global_lse.contiguous())


# ---------------------------------------------------------------------------
# global token bookkeeping used by the CP schedule
# ---------------------------------------------------------------------------


def rank_chunk_pair(rank: int, cp_size: int) -> Tuple[int, int]:
    """TE's dual-chunk load-balanced assignment: rank ``r`` owns ``(r, 2P-1-r)``.

    Verified against the pinned TE revision:
    ``transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py``
    ``get_batch_on_this_cp_rank`` (``total_slices = 2 * cp_size``, first slice
    ``cp_rank``, second slice ``total_slices - cp_rank - 1``).
    """
    if not 0 <= rank < cp_size:
        raise ValueError(f"rank {rank} out of range for cp_size {cp_size}")
    return rank, 2 * cp_size - 1 - rank


def chunk_bounds_from_splits(splits: Sequence[int]) -> List[Tuple[int, int]]:
    """Turn chunk lengths into ``(begin, end)`` bounds."""
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    for size in splits:
        if size < 0:
            raise ValueError("chunk sizes must be non-negative")
        bounds.append((cursor, cursor + size))
        cursor += size
    return bounds


def even_chunk_bounds(total: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Equal chunking; requires exact divisibility."""
    if n_chunks <= 0 or total % n_chunks != 0:
        raise ValueError(f"{total} tokens do not divide evenly into {n_chunks} chunks")
    return chunk_bounds_from_splits([total // n_chunks] * n_chunks)
