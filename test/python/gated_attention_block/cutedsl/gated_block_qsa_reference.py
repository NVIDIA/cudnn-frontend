# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch oracle for the gated attention block under Qwen Sparse Attention (QSA).

QSA is the dense gated attention of ``gated_block_reference.py`` -- project, QK-RMSNorm, partial RoPE, SDPA,
``* sigmoid(GATE)``, out-project -- with the SDPA's key set restricted per query to (a) the keys of the query's
SELECTED complete ``block_size``-token blocks and (b) the OPEN tail block of its visible range, always, both under the
causal mask and the per-sequence length.  The selection comes from an indexer (``qsa_indexer_reference``) or from the
caller as a block-id list; the block itself is given the list.  For ``<= top_k * block_size + block_size - 1``
visible tokens (2051 at the default geometry) every complete block is selected and the function IS the dense one.

This module follows the Qwen4Exp modeling code of HF transformers (``transformers/models/qwen4_exp/
modeling_qwen4_exp.py``, Apache-2.0, "Copyright 2026 The Qwen Team and The HuggingFace Inc. team"); it is
re-derived in the house style, not copied, and every mirrored formula carries the ``file:line`` of that file
(``modeling:`` below) as it stood on 2026-10-07.  No ``transformers`` import: the cross-check against the package is
an optional, skip-if-absent test.

What is here, and what each mirrors:

``RefQsaSpec``
    The declaration-time QSA attributes, kept independent of the library's own spec on purpose (the
    ``RefGeometry`` rule: a reference that imports the thing it validates can agree with a bug).

``qsa_visible_mask``
    The visible-key set of every query row -- the selection rule of ``modeling:712-756`` (complete blocks of the
    visible range, the tail appended unconditionally at ``:754-755``) ANDed with the causal mask (``:847-852``) --
    spelled as the COUNT RULE the kernel applies on device: ``n_sel_default = min(top_k, floor((p + 1) / block_size))``,
    ``count = clamp(block_lens[r], 0, n_sel_default)`` when ``block_lens`` is given, a ``-1`` at an index below the
    count contributes no key, entries at or beyond the count are never read.  ``pos0`` switches the tail start to the
    STEP-0 position for the decode mode's shared list (every token from the step-0 tail start to the row's own
    position stays visible).  Idempotent on duplicate ids.

``qsa_indexer_reference``
    The indexer of ``modeling:694-756``: project ``[T, (index_heads + 1) * index_head_dim]``, RMSNorm + RoPE the
    queries at the token position, mean-pool each complete block's raw keys in fp32, RMSNorm + RoPE the pooled key at
    the BLOCK START position, ``score = sum_h relu(q_h . kbar_b) / sqrt(index_head_dim)``, top-``min(top_k, n_blocks)``
    with the smaller block id winning a tie (``torch.topk``'s order is unspecified; the kernel's ``tie_break = 1``).
    ``hf_rounding=True`` reproduces HF's dtype flow (the pooled key cast to the key dtype before the norm, the norm
    outputs cast back, the rotations in the model dtype); the default keeps everything after the projection in
    ``acc_dtype``.

``gated_attention_block_qsa_reference`` / ``gated_attention_block_qsa_reference_packed``
    The dense fp32 oracle with ``qsa_visible_mask`` in place of the causal mask, chunked over query tiles; stages
    (1)-(3), (5), (6) are the dense oracle's own functions.  Returns every intermediate (``RefQsaOutputs``) so a miss
    localizes.  The packed form runs it per sequence of a THD packing (block ids are per-sequence block numbers).

``make_qsa_inputs``
    ``make_inputs`` plus the fifth (INDEX) band of ``W_qkvg`` when declared and a block-id list from one of three
    sources: ``"full"`` (the first ``min(top_k, floor((p + 1) / block_size))`` blocks -- the identity below the bound),
    ``"indexer"`` (``qsa_indexer_reference`` on random weights) or ``"synthetic"`` (a random subset of the complete
    blocks).  The dense draws (``h``, the four dense bands, ``w_o``, the tables) are bitwise the dense block's.

Conventions shared with the dense oracle: norm weights are the DIRECT multipliers the block consumes (a zero-centred
HF weight ``w`` is handed over as ``1 + w`` by the loader, ``modeling:160-163``); ``cos`` / ``sin`` are the NeoX
``[B, S, rope_dim]`` tables with duplicated halves; ``block_ids`` are int32 ``[T, top_k]`` (dense: ``[B, S, top_k]``),
``-1`` padded, block ``b`` = KV tokens ``[b * block_size, (b + 1) * block_size)`` of the query's OWN sequence.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gated_block_reference import (  # noqa: E402
    RefGeometry,
    RefOutputs,
    _packed_rows,
    apply_partial_rope,
    make_inputs,
    qk_norm_rope_reference,
    sequence_slices,
    split_qkvg,
)

# ---------------------------------------------------------------------------
# The declaration-time attributes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefQsaSpec:
    """Mirror of the block's QSA declaration attributes, kept independent on purpose.

    Model provenance (Qwen3.8-Flash-Next: ``indexer_budget`` 2048, ``indexer_compress_ratio`` 4, a 4-head x 128
    MQA indexer, ``indexer_kv_heads`` pinned to 1 by the HF validator, ``configuration_qwen4_exp.py:220-221``) lives
    here only as the defaults; the fields are named by op geometry.  Every field changes the FUNCTION (which keys a
    query sees, who computes the selection, how many projection columns exist), none is a performance knob.
    """

    block_size: int = 4  # tokens per selectable block (indexer_compress_ratio)
    top_k: int = 512  # blocks per query (indexer_budget // block_size)
    index_source: str = "caller"  # "caller": the block is given block_ids; "indexer": the block runs the indexer
    index_band: bool = False  # W_qkvg carries a fifth band of (index_heads + index_kv_heads) * index_head_dim columns
    index_heads: int = 4  # indexer_n_heads (the MQA scorer heads)
    index_kv_heads: int = 1  # indexer_kv_heads
    index_head_dim: int = 128  # indexer_head_dim
    index_norm_eps: float = 1e-6  # the indexer's q / k RMSNorm epsilon (HF: config.rms_norm_eps, modeling:682-683)

    @property
    def token_budget(self) -> int:
        """``indexer_budget``: the selected-token budget of a query, tail excluded."""
        return self.top_k * self.block_size

    @property
    def identity_bound(self) -> int:
        """The largest visible count for which the selection is the identity: ``floor(n / block_size) <= top_k``
        for every ``n <= top_k * block_size + block_size - 1`` (``modeling:716``, the selection width)."""
        return self.top_k * self.block_size + self.block_size - 1

    @property
    def index_band_width(self) -> int:
        """Columns of the INDEX band: the indexer's q heads first, the raw-key head last (``modeling:677-681``)."""
        return (self.index_heads + self.index_kv_heads) * self.index_head_dim

    def validate(self, geom: Optional[RefGeometry] = None) -> None:
        """The declaration rows of the block's own validator, mirrored (``ValueError`` names the failure's look)."""
        if self.block_size != 4:
            raise ValueError(f"block_size must be 4 (one gather of four rows per block), got {self.block_size}")
        if not (4 <= self.top_k <= 512) or self.top_k % 4 != 0:
            raise ValueError(f"top_k must be a multiple of 4 in [4, 512] (a 16-byte id row), got {self.top_k}")
        if self.index_source not in ("caller", "indexer"):
            raise ValueError(f"index_source must be 'caller' or 'indexer', got {self.index_source!r}")
        if self.index_kv_heads != 1:
            raise ValueError(f"index_kv_heads must be 1 (one shared raw-key head), got {self.index_kv_heads}")
        if self.index_heads <= 0 or self.index_head_dim <= 0:
            raise ValueError("index_heads and index_head_dim must be positive")
        if self.index_band_width % 64 != 0:
            raise ValueError(f"the INDEX band must be a multiple of 64 columns, got {self.index_band_width}")
        if self.index_norm_eps <= 0.0:
            raise ValueError("index_norm_eps must be positive")
        if geom is not None and geom.rope_dim > self.index_head_dim:
            raise ValueError(f"rope_dim {geom.rope_dim} must fit the index head ({self.index_head_dim}): the indexer rotates with the attention's tables")


@dataclass(frozen=True)
class RefQsaGeometry(RefGeometry):
    """``RefGeometry`` plus the QSA declaration, appended LAST (``None`` = a dense block).

    ``n_qkvg`` grows by the INDEX band iff ``qsa.index_band``; the four dense bands and their offsets are untouched, so
    ``split_qkvg`` over the first ``n_qkvg_dense`` columns is the dense splitter.  Until ``RefGeometry`` carries the
    field itself this subclass is the mirror of ``GatedAttentionBlockGeometry(qsa=...)``.
    """

    qsa: Optional[RefQsaSpec] = None

    @property
    def n_qkvg_dense(self) -> int:
        return (2 * self.h_q + 2 * self.h_kv) * self.d_head

    @property
    def n_qkvg(self) -> int:  # type: ignore[override]
        band = self.qsa.index_band_width if (self.qsa is not None and self.qsa.index_band) else 0
        return self.n_qkvg_dense + band

    @property
    def index_offset(self) -> Optional[int]:
        """Column where the INDEX band starts, ``None`` without the band."""
        return self.n_qkvg_dense if (self.qsa is not None and self.qsa.index_band) else None


def dense_geometry(geom: RefGeometry) -> RefGeometry:
    """The same geometry without the QSA declaration (a plain ``RefGeometry``), for ``make_inputs`` and the dense oracle."""
    return RefGeometry(**{f.name: getattr(geom, f.name) for f in dataclasses.fields(RefGeometry)})


# Qwen3.8-Flash-Next's QSA layer at TP 1: d_model 2560, 24 query heads over 2 KV heads, head dim 256 with the leading
# 64 rotating (rope_theta 1e7), a 4 x 128 indexer under a 2048-token budget in blocks of 4.
GEOMETRY_FLASH_NEXT = RefQsaGeometry(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64, rope_base=1e7, qsa=RefQsaSpec())

# A shrunk shape with the same structure, for oracle-speed runs.
GEOMETRY_QSA_SMALL = RefQsaGeometry(d_model=256, h_q=4, h_kv=2, d_head=64, rope_dim=16, qsa=RefQsaSpec())


# ---------------------------------------------------------------------------
# The visible-key set
# ---------------------------------------------------------------------------


def _rows_long(x, rows: int, device, name: str) -> torch.Tensor:
    """A per-row int64 vector from an int, a 0-d / 1-d tensor or a list (broadcast to ``rows``)."""
    t = torch.as_tensor(x, device=device).to(torch.long)
    if t.dim() == 0:
        return t.expand(rows).clone()
    t = t.reshape(-1)
    if t.numel() == 1 and rows != 1:
        return t.expand(rows).clone()
    if t.numel() != rows:
        raise ValueError(f"{name} must have one entry per row ({rows}), got {t.numel()}")
    return t


def qsa_visible_mask(
    block_ids: torch.Tensor,
    block_lens,
    positions,
    kv_lens,
    S_kv: int,
    block_size: int,
    *,
    top_k: Optional[int] = None,
    pos0=None,
) -> torch.Tensor:
    """``[rows, S_kv]`` bool, True where a key is VISIBLE to the row, for ``rows`` query rows of ONE sequence each.

    Per row ``r`` at position ``p`` (0-based within its sequence) with sequence length ``L`` (``kv_lens``), the visible
    count is ``n_vis = min(p + 1, L)`` and, with ``bs = block_size``,

    * ``n_sel_default = min(top_k, floor((anchor + 1) / bs))`` -- the number of complete blocks the row may select,
      derived from the position (the kernel reads no count from the host); ``anchor = p``, or ``pos0`` when given;
    * ``count = clamp(block_lens[r], 0, n_sel_default)`` when ``block_lens`` is given, else ``n_sel_default`` (and
      never more than the list width): the entries READ;
    * ``ids(r) = { block_ids[r, i] : i < count, block_ids[r, i] >= 0 }`` -- a ``-1`` below the count contributes no
      key, an entry at or beyond the count is never read, valid-looking or not;
    * ``V(r) = { t < n_vis : floor(t / bs) in ids(r) }  UNION  { t : bs * floor(n_anchor / bs) <= t < n_vis }`` with
      ``n_anchor = min(anchor + 1, L)`` -- the listed complete blocks inside the causal range plus the open tail
      (``modeling:754-755`` appends it unconditionally; ``:847-852`` ANDs the causal mask).

    A listed block at or past ``floor(n_vis / bs)`` contributes nothing (masked key by key); duplicate ids are
    idempotent; ``n_vis <= 0`` (``L == 0`` or ``p < 0``) is a dead row (all False).  ``pos0`` (per row, usually one
    value per sequence) is the STEP-0 position of the decode mode's shared list: the count comes from it and the tail
    runs from the step-0 tail start to the row's own position, so the block completed between ``pos0`` and ``p``
    (which the step-0 list cannot hold) stays visible, the row's own token included -- pass ``pos0=None`` for the
    per-row-tail reading (which hides that block at ``p = bs * k + bs - 1``).  ``top_k`` defaults to the list width.
    """
    if block_ids.dim() != 2:
        raise ValueError(f"block_ids must be [rows, top_k], got {tuple(block_ids.shape)}")
    rows, width = block_ids.shape
    dev = block_ids.device
    bs = int(block_size)
    if bs <= 0:
        raise ValueError(f"block_size must be positive, got {bs}")
    s_kv = int(S_kv)
    cap = width if top_k is None else int(top_k)

    p = _rows_long(positions, rows, dev, "positions")
    kv = torch.full((rows,), s_kv, dtype=torch.long, device=dev) if kv_lens is None else _rows_long(kv_lens, rows, dev, "kv_lens")
    anchor = p if pos0 is None else _rows_long(pos0, rows, dev, "pos0")

    n_vis = torch.minimum(p + 1, kv).clamp_min(0)
    n_anchor = torch.minimum(anchor + 1, kv).clamp_min(0)
    n_sel_default = torch.minimum(torch.div(anchor + 1, bs, rounding_mode="floor"), torch.full_like(p, cap)).clamp_min(0)
    count = n_sel_default if block_lens is None else torch.minimum(_rows_long(block_lens, rows, dev, "block_lens").clamp_min(0), n_sel_default)
    count = torch.minimum(count, torch.full_like(count, width))

    ids = block_ids.to(torch.long)
    n_blocks_total = -(-s_kv // bs)  # the block axis of the KV tensor; ids past it go to the sentinel column
    read = torch.arange(width, device=dev)[None, :] < count[:, None]
    live = read & (ids >= 0) & (ids < n_blocks_total)
    target = torch.where(live, ids, torch.full_like(ids, n_blocks_total))
    member = torch.zeros(rows, n_blocks_total + 1, dtype=torch.bool, device=dev)
    if width:
        member.scatter_(1, target, True)
    member = member[:, :n_blocks_total]

    t = torch.arange(s_kv, device=dev)
    listed = member[:, torch.div(t, bs, rounding_mode="floor")]  # [rows, S_kv]
    tail_start = torch.div(n_anchor, bs, rounding_mode="floor") * bs
    visible = (t[None, :] < n_vis[:, None]) & (listed | (t[None, :] >= tail_start[:, None]))
    return visible


def block_ids_contract_violations(
    block_ids: torch.Tensor,
    positions,
    kv_lens,
    *,
    block_size: int,
    block_lens=None,
    top_k: Optional[int] = None,
) -> List[str]:
    """TEST-ONLY detector of the block-id contract (the library never reads a list on the host).

    Checks every row: a valid prefix then ``-1`` only; every valid id ``< floor(n_vis / block_size)`` (a complete block of
    the row's visible range); no duplicate ids; ``block_lens`` (when given) equal to the prefix length and at most
    ``min(top_k, floor((p + 1) / block_size))``.  Returns one line per violated rule (empty = clean); the rows that
    violate are counted, the first offender named.
    """
    if block_ids.dim() != 2:
        raise ValueError(f"block_ids must be [rows, top_k], got {tuple(block_ids.shape)}")
    rows, width = block_ids.shape
    dev = block_ids.device
    bs = int(block_size)
    ids = block_ids.to(torch.long)
    p = _rows_long(positions, rows, dev, "positions")
    kv = _rows_long(kv_lens, rows, dev, "kv_lens")
    n_vis = torch.minimum(p + 1, kv).clamp_min(0)
    n_complete = torch.div(n_vis, bs, rounding_mode="floor")
    cap = width if top_k is None else int(top_k)
    out: List[str] = []

    def _report(name: str, bad_rows: torch.Tensor) -> None:
        n = int(bad_rows.sum())
        if n:
            first = int(bad_rows.nonzero()[0])
            out.append(f"{name}: {n} row(s), first at row {first} (ids {ids[first].tolist()[: min(width, 12)]}...)")

    valid = ids >= 0
    prefix_len = valid.to(torch.long).sum(dim=1)
    prefix_form = valid == (torch.arange(width, device=dev)[None, :] < prefix_len[:, None])
    _report("not a valid prefix followed by -1", ~prefix_form.all(dim=1))
    _report("an id at or past floor(n_vis / block_size)", (valid & (ids >= n_complete[:, None])).any(dim=1))
    _report("more valid ids than min(top_k, complete blocks)", prefix_len > torch.minimum(n_complete, torch.full_like(n_complete, cap)))
    if width > 1:
        srt = torch.sort(torch.where(valid, ids, torch.full_like(ids, -1)), dim=1).values
        dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any(dim=1)
        _report("duplicate ids in a row", dup)
    if block_lens is not None:
        bl = _rows_long(block_lens, rows, dev, "block_lens")
        _report("block_lens != the valid prefix length", bl != prefix_len)
    return out


# ---------------------------------------------------------------------------
# Block-id list builders (test-side; the kernels never derive a list)
# ---------------------------------------------------------------------------


def full_block_ids(positions: torch.Tensor, top_k: int, block_size: int, *, kv_lens=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The first ``min(top_k, floor(n_vis / block_size))`` complete blocks of every row, in order, ``-1`` padded ->
    ``(block_ids int32 [..., top_k], block_lens int32 [...])``.  Below the identity bound this is EVERY complete block
    (the selection is the identity); past it the LAST complete blocks are the omitted ones (at ``S = 2052`` row 2051
    has 513 complete blocks and loses block 512 -- the one holding its own token)."""
    p = positions.to(torch.long)
    n_vis = p + 1 if kv_lens is None else torch.minimum(p + 1, torch.as_tensor(kv_lens, device=p.device).to(torch.long).expand_as(p))
    n = torch.minimum(torch.div(n_vis.clamp_min(0), int(block_size), rounding_mode="floor"), torch.full_like(p, int(top_k)))
    j = torch.arange(int(top_k), device=p.device)
    ids = torch.where(j < n[..., None], j.expand(*p.shape, int(top_k)), torch.full((*p.shape, int(top_k)), -1, dtype=torch.long, device=p.device))
    return ids.to(torch.int32), n.to(torch.int32)


def random_block_ids(
    positions: torch.Tensor, top_k: int, block_size: int, *, generator: torch.Generator, kv_lens=None, shuffle: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A uniformly random subset of ``min(top_k, floor(n_vis / block_size))`` of the row's complete blocks, in random
    order (``shuffle=False``: ascending), ``-1`` padded -> ``(block_ids int32, block_lens int32)``.  Below the identity
    bound the SET is the full one (the order random); past it a random ``top_k``-subset.  Vectorised: one random key
    per (row, block), masked and sorted."""
    p = positions.to(torch.long)
    dev = p.device
    n_vis = p + 1 if kv_lens is None else torch.minimum(p + 1, torch.as_tensor(kv_lens, device=dev).to(torch.long).expand_as(p))
    n_complete = torch.div(n_vis.clamp_min(0), int(block_size), rounding_mode="floor")
    n = torch.minimum(n_complete, torch.full_like(p, int(top_k)))
    flat_p = p.reshape(-1)
    n_blocks_max = int(n_complete.max()) if flat_p.numel() else 0
    keys = torch.rand(flat_p.numel(), max(n_blocks_max, 1), generator=generator, device=dev)
    avail = torch.arange(max(n_blocks_max, 1), device=dev)[None, :] < n_complete.reshape(-1)[:, None]
    keys = keys.masked_fill(~avail, 2.0)  # unavailable blocks sort last
    order = torch.argsort(keys, dim=1)[:, : int(top_k)]
    if not shuffle:
        order = torch.sort(
            torch.where(torch.arange(order.shape[1], device=dev)[None, :] < n.reshape(-1)[:, None], order, torch.full_like(order, 1 << 40)), dim=1
        ).values
    take = torch.arange(order.shape[1], device=dev)[None, :] < n.reshape(-1)[:, None]
    ids = torch.full((flat_p.numel(), int(top_k)), -1, dtype=torch.long, device=dev)
    ids[:, : order.shape[1]] = torch.where(take, order, torch.full_like(order, -1))
    return ids.reshape(*p.shape, int(top_k)).to(torch.int32), n.to(torch.int32)


# ---------------------------------------------------------------------------
# The indexer (selection) reference
# ---------------------------------------------------------------------------


def _rms_norm_direct(x: torch.Tensor, w: torch.Tensor, eps: float, acc_dtype: torch.dtype) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * w`` in ``acc_dtype`` (``modeling:155-163``; ``w`` is the direct multiplier,
    HF's ``(1 + weight)`` already folded by the caller)."""
    x32 = x.to(acc_dtype)
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.to(acc_dtype)


def qsa_indexer_reference(
    h: torch.Tensor,
    w_i: torch.Tensor,
    w_iq_norm: torch.Tensor,
    w_ik_norm: torch.Tensor,
    cos_blk: Optional[torch.Tensor],
    sin_blk: Optional[torch.Tensor],
    cos_tok: torch.Tensor,
    sin_tok: torch.Tensor,
    spec: RefQsaSpec,
    geom: RefGeometry,
    *,
    seq_lens=None,
    hf_rounding: bool = False,
    acc_dtype: torch.dtype = torch.float32,
    q_chunk: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The QSA indexer of ``modeling:694-756`` -> ``(scores, block_ids, block_lens, k_raw, k_compressed)``.

    * ``w_i [(index_heads + index_kv_heads) * index_head_dim, d_model]`` is the ``index_qk_proj`` weight, q heads first
      then the raw-key head (``modeling:677-681``, the split at ``:699-703``); ``h [B, S, d_model]``.
    * ``cos_tok`` / ``sin_tok`` ``[B, S, rope_dim]`` are the attention's own tables (the indexer rotates its leading
      ``rope_dim`` dims with them, ``:706``, ``:627-662``); ``cos_blk`` / ``sin_blk`` ``[B, n_blocks, rope_dim]`` are
      those tables at the BLOCK START positions ``block_size * j`` (``:737-742``), ``None`` = taken from ``cos_tok``.
    * per sequence ``b`` of length ``L_b`` (``seq_lens``, else ``S``): ``kbar_j = RoPE(RMSNorm(mean_fp32(k_raw[4j..4j+3])),
      pos 4j)`` for the ``floor(L_b / 4)`` complete blocks (``:729-742``); query ``t`` scores the blocks below
      ``floor(min(t + 1, L_b) / 4)`` (``:718-722``) as ``sum_h relu(q_h . kbar_j) / sqrt(index_head_dim)``
      (``:744-746``) and keeps the top ``min(top_k, n_complete)`` (``:749``), ties to the smaller block id.
    * ``hf_rounding=True``: the pooled key is cast to the key dtype BEFORE its norm (``:735``), both norms return the
      model dtype (``:164``), the rotations run in it; the scores are fp32 either way (``:744-745`` casts both sides).
      ``False`` (default): everything after the projection stays in ``acc_dtype``.
    * norm weights are direct multipliers; ``getattr(geom, "norm_weight_offset", 0.0)`` is added so the oracle and
      the loader read one geometry fact.

    Returns ``scores [B, S, n_blocks_max]`` fp32 (``-inf`` at and past a row's block-causal limit), ``block_ids
    [B, S, top_k]`` int32 (``-1`` padded, by score descending), ``block_lens [B, S]`` int32, ``k_raw [B, S,
    index_head_dim]`` in ``h.dtype`` (what HF caches, pre-norm, un-rotated), ``k_compressed [B, n_blocks_max,
    index_head_dim]`` (zero rows past a sequence's blocks).
    """
    if h.dim() != 3:
        raise ValueError(f"h must be [B, S, d_model], got {tuple(h.shape)}")
    b, s, d_model = h.shape
    hq_i, hk_i, d_i, bs = spec.index_heads, spec.index_kv_heads, spec.index_head_dim, spec.block_size
    if hk_i != 1:
        raise ValueError(f"index_kv_heads must be 1 (configuration_qwen4_exp.py:220-221), got {hk_i}")
    if w_i.shape != (spec.index_band_width, d_model):
        raise ValueError(f"w_i must be [{spec.index_band_width}, {d_model}], got {tuple(w_i.shape)}")
    rope_dim = int(cos_tok.shape[-1])
    if rope_dim > d_i:
        raise ValueError(f"rope_dim {rope_dim} exceeds index_head_dim {d_i} (configuration_qwen4_exp.py:226-229)")
    if w_iq_norm.shape != (d_i,) or w_ik_norm.shape != (d_i,):
        raise ValueError(f"the indexer norm weights must be [{d_i}]")
    dev, out_dtype, eps, top_k = h.device, h.dtype, float(spec.index_norm_eps), int(spec.top_k)
    offset = float(getattr(geom, "norm_weight_offset", 0.0))
    wq = w_iq_norm.to(acc_dtype) + offset
    wk = w_ik_norm.to(acc_dtype) + offset

    # modeling:699-705 -- one projection, the q heads then the raw-key head; a GEMM output rounded once to the io dtype.
    qk = (h.to(acc_dtype) @ w_i.to(acc_dtype).t()).to(out_dtype)
    q_i = qk[..., : hq_i * d_i].reshape(b, s, hq_i, d_i)
    k_raw = qk[..., hq_i * d_i :].reshape(b, s, d_i)

    # modeling:706 -- q_layernorm then RoPE at the token position (partial: the leading rope_dim dims).
    q = _rms_norm_direct(q_i, wq, eps, acc_dtype)
    if hf_rounding:
        q = q.to(out_dtype)
    q = apply_partial_rope(q, cos_tok.to(q.dtype), sin_tok.to(q.dtype), rope_dim)

    n_blocks_max = s // bs
    if cos_blk is None or sin_blk is None:
        cos_blk, sin_blk = cos_tok[:, 0::bs][:, :n_blocks_max], sin_tok[:, 0::bs][:, :n_blocks_max]
    kbar_dtype = out_dtype if hf_rounding else acc_dtype
    k_compressed = torch.zeros(b, n_blocks_max, d_i, dtype=kbar_dtype, device=dev)
    scores = torch.full((b, s, n_blocks_max), float("-inf"), dtype=torch.float32, device=dev)
    block_ids = torch.full((b, s, top_k), -1, dtype=torch.int32, device=dev)
    block_lens = torch.zeros(b, s, dtype=torch.int32, device=dev)
    inv_sqrt_d = 1.0 / math.sqrt(d_i)

    for bi in range(b):
        length = s if seq_lens is None else int(seq_lens[bi])
        nb = max(0, min(length, s)) // bs
        if nb > 0:
            # modeling:729-742 -- fp32 mean of the four raw keys (cast to the key dtype under hf_rounding), k_layernorm,
            # RoPE at the block START position.
            pooled = k_raw[bi, : nb * bs].reshape(nb, bs, d_i).float().mean(dim=1)
            if hf_rounding:
                pooled = pooled.to(out_dtype)
            kb = _rms_norm_direct(pooled, wk, eps, acc_dtype)
            if hf_rounding:
                kb = kb.to(out_dtype)
            kb = apply_partial_rope(kb[None, :, None, :], cos_blk[bi : bi + 1, :nb].to(kb.dtype), sin_blk[bi : bi + 1, :nb].to(kb.dtype), rope_dim)[0, :, 0]
            k_compressed[bi, :nb] = kb.to(kbar_dtype)
        for lo in range(0, s, q_chunk):
            hi = min(lo + q_chunk, s)
            pos = torch.arange(lo, hi, device=dev)
            n_complete = torch.div(torch.minimum(pos + 1, torch.full_like(pos, length)).clamp_min(0), bs, rounding_mode="floor")
            k_row = torch.minimum(n_complete, torch.full_like(n_complete, top_k))
            block_lens[bi, lo:hi] = k_row.to(torch.int32)
            if nb == 0:
                continue
            # modeling:744-746 -- fp32 scores, relu, head sum, 1/sqrt(d); the block-causal limit of :718-722.
            sc = torch.einsum("thd,jd->thj", q[bi, lo:hi].float(), k_compressed[bi, :nb].float())
            sc = torch.relu(sc).sum(dim=1) * inv_sqrt_d
            limit = torch.arange(nb, device=dev)[None, :] < n_complete[:, None]
            sc = sc.masked_fill(~limit, float("-inf"))
            scores[bi, lo:hi, :nb] = sc
            # modeling:749 -- topk(min(block_topk, num_complete_blocks)); a stable descending sort makes the smaller block
            # id win a tie (torch.topk leaves the order unspecified).
            order = torch.sort(sc, dim=1, descending=True, stable=True).indices[:, :top_k]
            take = torch.arange(order.shape[1], device=dev)[None, :] < k_row[:, None]
            block_ids[bi, lo:hi, : order.shape[1]] = torch.where(take, order, torch.full_like(order, -1)).to(torch.int32)
    return scores, block_ids, block_lens, k_raw, k_compressed


# ---------------------------------------------------------------------------
# The block oracle
# ---------------------------------------------------------------------------


@dataclass
class RefQsaOutputs(RefOutputs):
    """``RefOutputs`` plus what the QSA block adds."""

    index_q_raw: Optional[torch.Tensor] = None  # [B, S, index_heads, index_head_dim] the band's indexer queries, pre-norm; None without the band
    index_k_raw: Optional[torch.Tensor] = None  # [B, S, index_head_dim] the band's raw indexer key, pre-norm, un-rotated (what HF caches)
    n_visible: Optional[torch.Tensor] = None  # [B, S] int64: |V(r)| per query row (0 = a dead row: O = 0, LSE = -inf)


def _resolve_spec(geom: RefGeometry, spec: Optional[RefQsaSpec], block_ids: torch.Tensor) -> RefQsaSpec:
    if spec is not None:
        return spec
    g_spec = getattr(geom, "qsa", None)
    if g_spec is not None:
        return g_spec
    return RefQsaSpec(top_k=int(block_ids.shape[-1]))


def split_qkvg_qsa(proj: torch.Tensor, geom: RefGeometry, spec: RefQsaSpec):
    """``[B, S, N] -> (Q, GATE, K, V, INDEX)``: the dense splitter over the four dense bands, the INDEX band (or
    ``None``) after them -- ``[B, S, (index_heads + 1) * index_head_dim]``, q heads first, the raw-key head last."""
    n_dense = (2 * geom.h_q + 2 * geom.h_kv) * geom.d_head
    q, gate, k, v = split_qkvg(proj[..., :n_dense], geom)
    band = None
    if spec.index_band:
        if proj.shape[-1] != n_dense + spec.index_band_width:
            raise ValueError(f"proj must carry {n_dense} dense + {spec.index_band_width} INDEX columns, got {proj.shape[-1]}")
        band = proj[..., n_dense : n_dense + spec.index_band_width]
    elif proj.shape[-1] != n_dense:
        raise ValueError(f"proj must have {n_dense} columns, got {proj.shape[-1]}")
    return q, gate, k, v, band


def gated_attention_block_qsa_reference(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geom: RefGeometry,
    *,
    block_ids: torch.Tensor,
    block_lens: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    q_chunk: int = 512,
    acc_dtype: torch.dtype = torch.float32,
    spec: Optional[RefQsaSpec] = None,
) -> RefQsaOutputs:
    """FP32 oracle of the gated attention block under QSA, chunked over query tiles.

    The dense oracle (``gated_attention_block_reference``) with ``qsa_visible_mask`` in place of the causal mask: the
    projection, QK-RMSNorm + partial RoPE, the gate and the out projection are the dense oracle's own functions in the
    dense oracle's order, so with the FULL list the two agree to the bit.  ``block_ids`` is ``[B, S, top_k]`` or
    ``[B * S, top_k]`` (int32 by contract, any integer dtype accepted), ``block_lens`` ``[B, S]`` / ``[B * S]`` or
    ``None``; row ``s`` of batch ``b`` is at position ``s`` and sees ``min(s + 1, seq_lens[b])`` tokens.  A row with no
    visible key (``seq_lens[b] == 0``; an empty list with an empty tail) is SELECTED to ``O = 0``, ``LSE = -inf`` --
    never a floored denominator, never residue times a sigmoid; ``out[b] == 0`` exactly for a dead entry.

    The QSA declaration is ``geom.qsa`` (``RefQsaGeometry``), else ``spec`` (appended keyword), else the defaults with
    ``top_k = block_ids.shape[-1]``.  QSA composes with the causal mask only: ``geom.is_causal`` must hold and the
    window fields must be unbounded (``causal_bottom_right`` is a no-op for the block's self-attention and accepted).
    ``acc_dtype`` as in the dense oracle (``torch.float64`` for an unrounded chain).
    """
    spec = _resolve_spec(geom, spec, block_ids)
    b, s, d_model = h.shape
    n_dense = (2 * geom.h_q + 2 * geom.h_kv) * geom.d_head
    n_total = n_dense + (spec.index_band_width if spec.index_band else 0)
    if d_model != geom.d_model:
        raise ValueError(f"h last dim {d_model} != geometry d_model {geom.d_model}")
    if w_qkvg.shape != (n_total, geom.d_model):
        raise ValueError(f"w_qkvg must be [{n_total}, {geom.d_model}] ({n_dense} dense + the INDEX band), got {tuple(w_qkvg.shape)}")
    if w_o.shape != (geom.d_model, geom.h_q * geom.d_head):
        raise ValueError(f"w_o must be [{geom.d_model}, {geom.h_q * geom.d_head}], got {tuple(w_o.shape)}")
    if not geom.is_causal:
        raise ValueError("QSA composes with the causal mask (modeling:847-852); geom.is_causal must be True")
    if int(geom.window_left) >= 0 or int(geom.window_right) >= 0:
        raise ValueError("QSA has no sliding-window form; window_left / window_right must be -1")
    if block_ids.shape[-1] == 0 and spec.top_k != 0:
        raise ValueError("block_ids has no columns")
    ids = block_ids.reshape(b, s, -1)
    lens = None if block_lens is None else block_lens.reshape(b, s)

    dev = h.device
    out_dtype = h.dtype
    d = geom.d_head
    rep = geom.h_q // geom.h_kv

    # (1) fused QKV+GATE(+INDEX) projection, fp32 accumulate, one rounding.
    proj = (h.to(acc_dtype) @ w_qkvg.to(acc_dtype).t()).to(out_dtype)
    q_pre, gate, k_pre, v, band = split_qkvg_qsa(proj, geom, spec)
    index_q_raw = index_k_raw = None
    if band is not None:
        index_q_raw = band[..., : spec.index_heads * spec.index_head_dim].reshape(b, s, spec.index_heads, spec.index_head_dim)
        index_k_raw = band[..., spec.index_heads * spec.index_head_dim :].reshape(b, s, spec.index_head_dim)

    # (2)+(3) QK-RMSNorm then partial RoPE -- V is NOT normed; the dense oracle's function, one final rounding.
    q, rstd_q = qk_norm_rope_reference(q_pre, w_q_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=acc_dtype)
    k, rstd_k = qk_norm_rope_reference(k_pre, w_k_norm, cos, sin, geom.rope_dim, geom.qk_norm_eps, qk_norm=geom.qk_norm, acc_dtype=acc_dtype)

    # (4) SDPA over the QSA-visible keys, chunked over q tiles, fp32.
    o = torch.zeros(b, s, geom.h_q, d, device=dev, dtype=acc_dtype)
    lse = torch.full((b, geom.h_q, s), float("-inf"), device=dev, dtype=acc_dtype)
    n_visible = torch.zeros(b, s, device=dev, dtype=torch.long)

    k_b = k.to(acc_dtype).repeat_interleave(rep, dim=2)  # [B, S, H_q, D]
    v_b = v.to(acc_dtype).repeat_interleave(rep, dim=2)

    for bi in range(b):
        kv_len = s if seq_lens is None else int(seq_lens[bi])  # a host read: test-side only
        for lo in range(0, s, q_chunk):
            hi = min(lo + q_chunk, s)
            qc = q[bi, lo:hi].to(acc_dtype).transpose(0, 1)  # [H_q, s_q, D]
            kc = k_b[bi].transpose(0, 1)  # [H_q, S, D]
            vc = v_b[bi].transpose(0, 1)
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * geom.scale  # [H_q, s_q, S]

            allowed = qsa_visible_mask(
                ids[bi, lo:hi],
                None if lens is None else lens[bi, lo:hi],
                torch.arange(lo, hi, device=dev),
                kv_len,
                s,
                spec.block_size,
                top_k=spec.top_k,
            )
            n_visible[bi, lo:hi] = allowed.sum(dim=-1)
            scores = scores.masked_fill(~allowed[None], float("-inf"))

            row_max = scores.amax(dim=-1)  # [H_q, s_q]
            dead = torch.isinf(row_max) & (row_max < 0)  # no visible key at all
            safe_max = torch.where(dead, torch.zeros_like(row_max), row_max)
            p = torch.exp(scores - safe_max[..., None])
            p = torch.where(allowed[None], p, torch.zeros_like(p))
            denom = p.sum(dim=-1)  # [H_q, s_q]

            # SELECT, not a multiply by zero: residue can be a NaN bit pattern, and NaN * 0 is NaN.
            o_chunk = torch.matmul(p, vc) / torch.where(dead, torch.ones_like(denom), denom)[..., None]
            o_chunk = torch.where(dead[..., None], torch.zeros_like(o_chunk), o_chunk)
            o[bi, lo:hi] = o_chunk.transpose(0, 1)

            lse_chunk = safe_max + torch.log(denom)
            lse[bi, :, lo:hi] = torch.where(dead, torch.full_like(lse_chunk, float("-inf")), lse_chunk)

    o = o.to(out_dtype)

    # (5) gate -- AFTER the dead-row substitution above, never before (modeling:889-890: after attention, before o_proj).
    o_gated = (o.to(acc_dtype) * torch.sigmoid(gate.to(acc_dtype))).to(out_dtype)

    # (6) out projection.
    o_flat = o_gated.reshape(b, s, geom.h_q * d)
    out = (o_flat.to(acc_dtype) @ w_o.to(acc_dtype).t()).to(out_dtype)

    return RefQsaOutputs(
        out=out,
        q_pre=q_pre,
        k_pre=k_pre,
        gate=gate,
        v=v,
        q=q,
        k=k,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        o=o,
        o_gated=o_gated,
        lse=lse,
        index_q_raw=index_q_raw,
        index_k_raw=index_k_raw,
        n_visible=n_visible,
    )


def gated_attention_block_qsa_reference_packed(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geom: RefGeometry,
    lens,
    *,
    block_ids: torch.Tensor,
    block_lens: Optional[torch.Tensor] = None,
    q_chunk: int = 512,
    acc_dtype: torch.dtype = torch.float32,
    spec: Optional[RefQsaSpec] = None,
) -> list:
    """The QSA oracle per sequence of a THD packing: ``gated_attention_block_qsa_reference`` on ``h[:, lo:hi]`` with
    that sequence's own ``cos`` / ``sin`` rows and its rows of ``block_ids`` / ``block_lens`` (block ids are the
    sequence's OWN block numbers: block ``j`` = its tokens ``[4j, 4j + 4)``).  One ``RefQsaOutputs`` per sequence
    (``[1, len_i, ...]`` tensors) or ``None`` for an empty one.  ``h``, ``cos``, ``sin`` may be rank-2 ``[T, .]`` or
    rank-3 ``[1, T, .]``; ``block_ids`` ``[T, top_k]`` or ``[1, T, top_k]``, ``block_lens`` ``[T]`` or ``[1, T]``.
    Pair it with ``gated_block_reference.compare_packed``."""
    h3, cos3, sin3 = _packed_rows(h), _packed_rows(cos), _packed_rows(sin)
    ids3 = block_ids.unsqueeze(0) if block_ids.dim() == 2 else block_ids
    lens3 = None if block_lens is None else (block_lens.unsqueeze(0) if block_lens.dim() == 1 else block_lens)
    refs = []
    for lo, hi in sequence_slices(lens):
        if hi == lo:
            refs.append(None)
            continue
        refs.append(
            gated_attention_block_qsa_reference(
                h3[:, lo:hi],
                w_qkvg,
                w_q_norm,
                w_k_norm,
                cos3[:, lo:hi],
                sin3[:, lo:hi],
                w_o,
                geom,
                block_ids=ids3[:, lo:hi],
                block_lens=None if lens3 is None else lens3[:, lo:hi],
                q_chunk=q_chunk,
                acc_dtype=acc_dtype,
                spec=spec,
            )
        )
    return refs


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

INDEX_SOURCES = ("full", "indexer", "synthetic")


def make_qsa_inputs(
    geom: RefGeometry,
    batch: int,
    seq_len: int,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    index_source: str = "full",
    seq_lens: Optional[torch.Tensor] = None,
) -> dict:
    """``make_inputs`` for a QSA block: the dense draws are BITWISE the dense block's (``h``, the four dense bands of
    ``w_qkvg``, the norm weights, the tables, ``w_o`` -- same generator, same order), the INDEX band rows (when
    ``geom.qsa.index_band``) are appended from a second generator, and ``block_ids`` ``[B, S, top_k]`` int32 /
    ``block_lens`` ``[B, S]`` int32 come from ``index_source``:

    * ``"full"``: ``full_block_ids`` -- every complete block below the identity bound (the dense function), the first
      ``top_k`` past it;
    * ``"indexer"``: ``qsa_indexer_reference`` on the band rows (or a separate random ``w_i`` without the band) with
      unit norm weights; ``w_i`` / ``w_iq_norm`` / ``w_ik_norm`` / ``index_scores`` are added to the dict;
    * ``"synthetic"``: ``random_block_ids`` -- a random subset in random order (the full SET below the bound).

    ``seq_lens`` (``[B]`` int32, appended) bounds each sequence's complete blocks for the list and is returned as is.
    """
    spec = getattr(geom, "qsa", None) or RefQsaSpec()
    if index_source not in INDEX_SOURCES:
        raise ValueError(f"index_source must be one of {INDEX_SOURCES}, got {index_source!r}")
    inp = make_inputs(dense_geometry(geom), batch, seq_len, device=device, dtype=dtype, seed=seed)
    g2 = torch.Generator(device=device).manual_seed(int(seed) + 0x9E37)

    def randn(*shape, std=0.02):
        return (torch.randn(*shape, generator=g2, device=device, dtype=torch.float32) * std).to(dtype)

    w_i = None
    if spec.index_band:
        w_i = randn(spec.index_band_width, geom.d_model)
        inp["w_qkvg"] = torch.cat([inp["w_qkvg"], w_i], dim=0)

    positions = torch.arange(seq_len, device=device).expand(batch, seq_len)
    kv_lens = None if seq_lens is None else seq_lens.to(device=device, dtype=torch.long)[:, None]
    if index_source == "full":
        ids, lens = full_block_ids(positions, spec.top_k, spec.block_size, kv_lens=kv_lens)
    elif index_source == "synthetic":
        ids, lens = random_block_ids(positions, spec.top_k, spec.block_size, generator=g2, kv_lens=kv_lens)
    else:
        if w_i is None:
            w_i = randn(spec.index_band_width, geom.d_model)
        w_iq_norm = torch.ones(spec.index_head_dim, device=device, dtype=dtype)
        w_ik_norm = torch.ones(spec.index_head_dim, device=device, dtype=dtype)
        scores, ids, lens, _k_raw, _k_c = qsa_indexer_reference(
            inp["h"], w_i, w_iq_norm, w_ik_norm, None, None, inp["cos"], inp["sin"], spec, geom, seq_lens=seq_lens
        )
        inp.update(w_i=w_i, w_iq_norm=w_iq_norm, w_ik_norm=w_ik_norm, index_scores=scores)
    inp["block_ids"] = ids.contiguous()
    inp["block_lens"] = lens.contiguous()
    inp["seq_lens"] = seq_lens
    return inp
