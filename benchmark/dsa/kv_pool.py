# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV pools and top-k index lists for the DSA benchmark suite.

The sparse core kernels gather flat ``int32`` row ids out of one ``(rows, D)``
KV pool. This module builds the pool (flat or paged through a block table),
the per-query id lists the kernels consume and the host-side causal counts
the indexer FLOP model needs. Torch only; nothing here imports ``cudnn``.

Id lists are generated in row chunks so no ``(rows x kv_len)`` tensor is ever
live, with ``torch.argsort`` over each row's visible range only. The chunk
shrinks automatically for long pools (a 128k-token request keeps the scratch
at a few hundred MiB). The kernels' own buffers are a separate matter: the
mxfp8 indexer disables microbatching and its candidate buffer grows as
``S * (S + 1) / 2`` fp32 at ratio 1.

Positions follow the FE indexer kernels' convention. A request's pool holds
``kv_len`` tokens including the ``q_tokens`` being produced, so query row
``j`` sits at position ``kv_len - q_tokens + j`` in generation and at ``j``
in context (``q_seqlen == kv_seqlen``, offsets zero). :func:`make_topk_ids`
draws causal lists (tokens ``<= pos``) by default; with ``causal=False``
every row draws from the whole request (the core context / bprop rows: every
query gathers the full top-k, the per-token upper bound). With index
granularity ``g`` a row selects complete ``g``-token blocks and the block
list is expanded to token ids; a causal row with fewer than one complete
block gets ``topk_length = 0`` (the kernels clamp).

The per-token byte footprints (:func:`record_bytes`, :func:`index_key_bytes`)
are defined torch-free in :mod:`support` and re-exported here.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import torch

from .support import index_key_bytes, record_bytes

__all__ = [
    "PagedPool",
    "causal_visible_counts",
    "index_key_bytes",
    "make_paged_pool",
    "make_topk_ids",
    "record_bytes",
    "row_positions",
]

# Upper bound on the elements of one chunk's (chunk_rows x max_visible) random score tensor:
# 32 Mi fp32 (128 MiB) plus the int64 argsort it produces.
_CHUNK_ELEMS = 1 << 25


class PagedPool(NamedTuple):
    """A paged KV pool: ``pool (num_pages * page_size, D)``, ``block_table (num_requests, max_pages) int32``
    and ``flat_index(req, tok)`` mapping a request-local token to its pool row."""

    pool: torch.Tensor
    block_table: torch.Tensor
    flat_index: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_paged_pool(
    num_requests: int,
    kv_len: int,
    page_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device,
    slot_order: str = "randperm",
    spare_pages: int = 4,
    generator: Optional[torch.Generator] = None,
) -> PagedPool:
    """Build a paged KV pool and the block table that addresses it.

    Every request owns ``max_pages = ceil(kv_len / page_size)`` pages drawn
    from ``num_requests * max_pages + spare_pages`` physical pages. With
    ``slot_order="randperm"`` the pages are assigned in random order so the
    gathers scatter across the pool as a serving allocator would place them;
    ``"contiguous"`` assigns them in request order (pool row == flat row when
    ``kv_len % page_size == 0``). Returns a :class:`PagedPool`; its
    ``flat_index(req, tok)`` takes ints or (broadcastable) integer tensors and
    returns ``block_table[req, tok // page_size] * page_size + tok %
    page_size``. The pool values are ``0.1 * randn``.
    """
    if page_size <= 0 or kv_len <= 0 or num_requests <= 0:
        raise ValueError(f"num_requests, kv_len and page_size must be positive, got {num_requests}, {kv_len}, {page_size}")
    max_pages = -(-kv_len // page_size)
    num_pages = num_requests * max_pages + int(spare_pages)
    if slot_order == "randperm":
        slots = torch.randperm(num_pages, device=device, generator=generator)[: num_requests * max_pages]
    elif slot_order == "contiguous":
        slots = torch.arange(num_requests * max_pages, device=device)
    else:
        raise ValueError(f"slot_order must be randperm or contiguous, got {slot_order!r}")
    block_table = slots.view(num_requests, max_pages).to(torch.int32).contiguous()
    pool = torch.randn(num_pages * page_size, head_dim, dtype=dtype, device=device, generator=generator).mul_(0.1)
    table64 = block_table.to(torch.int64)

    def flat_index(req, tok):
        req = torch.as_tensor(req, device=device, dtype=torch.int64)
        tok = torch.as_tensor(tok, device=device, dtype=torch.int64)
        return table64[req, tok // page_size] * page_size + tok % page_size

    return PagedPool(pool, block_table, flat_index)


def row_positions(q_tokens: int, kv_len: int, phase: str) -> torch.Tensor:
    """Causal position of each of a request's ``q_tokens`` query rows (CPU int64).

    ``context``: row ``j`` is token ``j`` of a ``q_seqlen == kv_seqlen``
    prefill (the indexer kernels with zero offsets). ``generation``: the rows
    are the last ``q_tokens`` positions of the ``kv_len``-token cache, so row
    ``j`` sits at ``kv_len - q_tokens + j`` (the ``q_causal_offsets`` the
    worker passes are ``kv_len - q_tokens``).
    """
    j = torch.arange(int(q_tokens), dtype=torch.int64)
    if phase == "context":
        return j
    if phase == "generation":
        return int(kv_len) - int(q_tokens) + j
    raise ValueError(f"phase must be context or generation, got {phase!r}")


def causal_visible_counts(q_tokens: int, kv_len: int, phase: str, ratio: int = 1) -> torch.Tensor:
    """Index keys each query row of one request may score (CPU int64, ``(q_tokens,)``).

    Follows the FE indexer kernels' completed-group rule, ``k_local <
    (offset + q_local + 1) // ratio``: ``n_vis(pos) = (pos + 1) // ratio``
    (floor), clamped to the pool length ``kv_len // ratio`` (the worker's K
    pool holds the complete groups only). The DSv4.1 reference model also
    admits the open group (ceil), a difference of at most one key per row.
    Used for the indexer FLOP count ``2 * H_idx * 128 * sum(n_vis)``.
    """
    ratio = int(ratio)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    pool_len = int(kv_len) // ratio
    pos = row_positions(q_tokens, kv_len, phase)
    return torch.clamp((pos + 1) // ratio, 0, pool_len)


def make_topk_ids(
    num_requests: int,
    q_tokens: int,
    kv_len: int,
    topk: int,
    *,
    phase: str,
    causal: bool = True,
    index_granularity: int = 1,
    block_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    device="cuda",
    generator: Optional[torch.Generator] = None,
    chunk_rows: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-query gather lists into a ``num_requests x kv_len`` pool.

    Returns ``(ids, topk_length)``: ``ids`` is ``int32 (num_requests *
    q_tokens, topk)`` with row ``req * q_tokens + j`` holding unique ids inside
    request ``req``'s own range, and ``topk_length`` is ``int32 (rows,)``. A
    ``causal`` row at position ``pos`` (see :func:`row_positions`) may only
    pick tokens ``<= pos``; when fewer than ``topk`` are visible the row holds
    all of them first, is padded with ``-1`` and ``topk_length`` is the valid
    prefix. With ``causal=False`` every row draws from all ``kv_len`` tokens
    of its request, so ``topk_length == topk`` whenever ``kv_len >= topk``.
    With ``index_granularity`` ``g`` the row picks ``topk // g`` complete
    ``g``-token blocks ``<= pos`` and each block expands to ``g`` consecutive
    token ids. Flat pools map token ``t`` of request ``r`` to ``r * kv_len +
    t``; with ``block_table`` and ``page_size`` (from :func:`make_paged_pool`)
    to ``block_table[r, t // P] * P + t % P``. Generated ``chunk_rows`` rows at
    a time, the chunk shrunk so one chunk's scratch stays under
    :data:`_CHUNK_ELEMS` elements.
    """
    g = int(index_granularity)
    if g <= 0 or topk % g:
        raise ValueError(f"topk ({topk}) must be a positive multiple of index_granularity ({g})")
    if (block_table is None) != (page_size is None):
        raise ValueError("block_table and page_size go together")
    if phase == "context" and q_tokens > kv_len:
        raise ValueError(f"context rows need q_tokens ({q_tokens}) <= kv_len ({kv_len})")
    rows = int(num_requests) * int(q_tokens)
    topk_sel = int(topk) // g
    ids = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    topk_length = torch.empty(rows, dtype=torch.int32, device=device)
    if rows == 0:
        return ids, topk_length

    # Visible complete blocks per row of one request, tiled over requests.
    if causal:
        n_vis_req = torch.clamp((row_positions(q_tokens, kv_len, phase) + 1) // g, 0, int(kv_len) // g)
    else:
        n_vis_req = torch.full((int(q_tokens),), int(kv_len) // g, dtype=torch.int64)
    n_vis_all = n_vis_req.repeat(int(num_requests)).to(device)
    req_all = torch.arange(rows, device=device, dtype=torch.int64) // int(q_tokens)
    topk_length.copy_((torch.minimum(n_vis_all, torch.tensor(topk_sel, device=device)) * g).to(torch.int32))
    expand = torch.arange(g, device=device, dtype=torch.int64)
    if block_table is not None:
        table64 = block_table.to(device=device, dtype=torch.int64)
        page = int(page_size)

    start = 0
    while start < rows:
        max_vis = int(n_vis_all[start : start + chunk_rows].max())
        n = max(1, min(chunk_rows, rows - start, _CHUNK_ELEMS // max(1, max_vis)))
        n_vis = n_vis_all[start : start + n]
        width = int(n_vis.max())
        if width:
            scores = torch.rand(n, width, device=device, generator=generator)
            col = torch.arange(width, device=device)
            scores.masked_fill_(col.view(1, width) >= n_vis.view(n, 1), 2.0)  # invisible blocks sort last
            take = min(width, topk_sel)
            blocks = scores.argsort(dim=-1)[:, :take]
            blocks.masked_fill_(col[:take].view(1, take) >= n_vis.view(n, 1), -1)
            tok = torch.where(blocks.unsqueeze(-1) >= 0, blocks.unsqueeze(-1) * g + expand, -1).view(n, take * g)
            valid = tok >= 0
            req = req_all[start : start + n].view(n, 1)
            if block_table is None:
                flat = req * int(kv_len) + tok
            else:
                tok0 = torch.clamp(tok, min=0)
                flat = table64[req, tok0 // page] * page + tok0 % page
            ids[start : start + n, : take * g] = torch.where(valid, flat, -1).to(torch.int32)
        start += n
    return ids, topk_length
