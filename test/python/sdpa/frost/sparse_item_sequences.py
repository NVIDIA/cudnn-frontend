# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Work lists with a PRESCRIBED tile-count sequence per resident CTA for the Rubin index-list sparse d256 forward.

The sparse core (``kernels/sm107/sparse_d256_f16.py``) is persistent: a CTA runs its own grid position and then every work
item the cluster-launch-control scheduler hands it, and EVERY ring barrier's slot and parity must come from a pipeline state
carried across those items in every role.  A per-item restart of the parity arithmetic is invisible on a one-item-per-CTA
grid and fails exactly on an odd / even MIX of tile counts between consecutive items: with ``n_tiles = 1`` twice in a row the
``s_empty`` ring is a phase behind (a hang), ``3, 1, 2`` reads a stale score slot, two items whose tile counts sum to
``2 x n_tiles % 3 != 0`` desynchronise the three-deep K/V ring.  This module builds inputs on which those mixes are the
TYPICAL per-CTA sequence, so the kernel's outputs can be compared with the oracle, bitwise across launches and processes.

How a sequence is forced without a scheduler hook.  Items are handed out in the grid's linear order (query token fastest,
then KV head, then batch) to ``R`` resident CTAs (one per SM: the kernel's shared-memory footprint allows no second CTA), so
the items of one wave ``[w R, (w + 1) R)`` of the token axis land on distinct CTAs and consecutive waves are consecutive items
of the same CTA while every item of a wave costs the same.  The tile count of an item is set through ``block_lens`` (the
kernel clamps it on device to the position's default), so wave ``w`` gets the count that yields ``seq[w % len(seq)]`` tiles:
``count = min(32 n - has_open, top_k)`` (17 tiles need the full 512-entry list plus an open tail block; with no tail the
largest count is 16 tiles).  Timing jitter can shift one CTA by an item against the wave grid, so the sequence is repeated
(``reps``) and every adjacent pair of the sequence occurs for every CTA; the construction is a STRONG stress of the carried
states, not a guarantee of one exact per-CTA order.

The open tail block's phase (``pos % 4`` in the kernel's natural form, where ``n_vis = pos + 1``) is held constant across a
sequence through the per-batch KV length: for every query at ``pos >= L - 1`` the visible range is ``n_vis = L``, so
``L % 4`` is the tail phase of every designed item at once (``phase = 0``: no open block; ``1 .. 3``: an open block of that
many keys).  ``phase = None`` is the natural form (no KV length): the four residues interleave inside every wave.

Dead items.  A ``0`` in the sequence is a wave with ``block_lens = 0``: with no open block (``phase = 0``, or the natural
positions ``pos % 4 == 3``) the item is DEAD -- the kernel runs ONE clamped tile of ``-1`` rows and must land ``O = 0`` /
``LSE = -inf`` exactly -- and with an open block it is the smallest live item (the tail keys only).  ``dead_batches`` appends
batch entries with KV length 0, whose every item is dead (a contiguous run of dead items in the linear order).

Residue.  The tokens of the LAST filler wave (and any ``nan_waves``) carry NaN query rows: the item that runs them leaves NaN
in both score slots of the tensor-memory accumulator and in the output accumulator, so the next item of that CTA starts on
NaN residue.  A correct kernel overwrites both (the first BMM of every item does not accumulate) and SELECTs a dead row's
output; any NaN on a row whose query is finite is a carried-state defect, not a numerics one.

Everything the kernel derives on device (``n_tiles``, ``count``, ``has_open``, ``dead``) is re-derived here on the host by the
same formula so a cell can state what it exercised (``expected_tiles``, ``expected_dead``).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

D = 256
BLOCK_SIZE = 4
BLOCKS_PER_TILE = 32
TOP_K_MAX = 512
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORACLE_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "gated_attention_block", "cutedsl")

# The sequences the carried-state design names: a hang shape, a stale-slot shape, the ring-wrap mixes, the maximal item next
# to the minimal one, and the dead-item interleave.
CARRIED_STATE_SEQUENCES: Tuple[Tuple[int, ...], ...] = ((1, 1, 1), (3, 1, 2), (17, 1), (2, 3, 4), (1, 17, 1), (0, 3, 0, 1))


def tiles_of(count: int, has_open: int) -> int:
    """The kernel's clamp: ``max(1, ceil((count + has_open) / 32))``."""
    return max(1, -(-(count + has_open) // BLOCKS_PER_TILE))


def count_for_tiles(n_tiles: int, has_open: int, top_k: int = TOP_K_MAX) -> int:
    """The smallest list count whose item runs ``n_tiles`` tiles under ``has_open`` (0 for a dead / tail-only item).  17 tiles
    exist only with the full 512-entry list AND an open block; the count is capped at ``top_k`` so a 17 asked for without a
    tail yields the 16 the kernel will run (``tiles_of`` says which)."""
    if n_tiles <= 0:
        return 0
    return min(BLOCKS_PER_TILE * n_tiles - has_open, top_k)


def expected_item_bounds(pos: torch.Tensor, block_lens: torch.Tensor, kv_len: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host twin of the kernel's one bounds helper -> ``(n_tiles, dead)`` per item (``kv_len`` = the sequence's visible length)."""
    p = pos.to(torch.long)
    n_vis = torch.minimum(p + 1, kv_len.to(torch.long)).clamp_min(0)
    n_sel_default = torch.minimum(torch.div(p + 1, BLOCK_SIZE, rounding_mode="floor"), torch.full_like(p, top_k)).clamp_min(0)
    count = torch.minimum(block_lens.to(torch.long).clamp_min(0), n_sel_default)
    has_open = (n_vis % BLOCK_SIZE != 0).to(torch.long)
    listed = count + has_open
    dead = (kv_len.to(torch.long) <= 0) | (listed == 0)
    n_tiles = torch.clamp(-(-listed // BLOCKS_PER_TILE), min=1)
    return n_tiles, dead


@dataclass
class ItemSequenceWorkList:
    """One cell's operands plus what the host expects of them."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_ids: torch.Tensor
    block_lens: torch.Tensor
    seq_kv_lens: Optional[torch.Tensor]  # int32 [B] or None (the natural form)
    kv_lens: List[int]  # per batch, the length the oracle sees (S_kv when the natural form)
    top_k: int
    phase: Optional[int]
    seq: Tuple[int, ...]
    resident_ctas: int
    designed_start: int  # the first designed token; [0, designed_start) are filler items
    waves: List[int]  # the tile-count target of every designed wave
    expected_tiles: torch.Tensor  # int64 [B, S]
    expected_dead: torch.Tensor  # bool [B, S]
    nan_tokens: torch.Tensor  # bool [S]: query rows poisoned with NaN (every head, every batch)

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        B, S, H, _ = self.q.shape
        return B, S, H, self.k.shape[2], self.k.shape[1]

    def describe(self) -> Dict[str, object]:
        B, S, H, KH, SKV = self.shape
        hist = torch.bincount(self.expected_tiles[0, self.designed_start :].cpu(), minlength=18).tolist()
        return dict(
            seq=list(self.seq),
            phase="natural" if self.phase is None else int(self.phase),
            B=B,
            S=S,
            S_kv=SKV,
            H=H,
            KH=KH,
            top_k=self.top_k,
            resident_ctas=self.resident_ctas,
            designed_start=self.designed_start,
            waves=list(self.waves),
            items=B * S * KH,
            designed_tiles_hist={n: c for n, c in enumerate(hist) if c},
            dead_rows=int(self.expected_dead.sum()),
            nan_tokens=int(self.nan_tokens.sum()),
            kv_lens=list(self.kv_lens),
        )


def build_item_sequence_work_list(
    *,
    seq: Sequence[int],
    phase: Optional[int],
    resident_ctas: int,
    reps: int,
    H: int,
    KH: int,
    dtype: torch.dtype,
    device,
    top_k: int = TOP_K_MAX,
    dead_batches: int = 0,
    nan_filler_wave: bool = True,
    nan_waves: Sequence[int] = (),
    seed: int = 0,
) -> ItemSequenceWorkList:
    """Build the cell.  ``seq`` = the per-CTA tile-count sequence (0 = a dead / tail-only wave); ``phase`` = the tail phase
    (``None`` natural, else ``L % 4``); ``resident_ctas`` = the wave width (the device's SM count); ``reps`` = how many times the
    sequence repeats (``len(seq) * reps >= 3`` designed items per CTA); ``dead_batches`` appends batches of KV length 0;
    ``nan_waves`` = designed-wave indices (negative from the end) whose query rows are NaN besides the last filler wave."""
    seq = tuple(int(n) for n in seq)
    if not seq or any(n < 0 or n > 17 for n in seq):
        raise ValueError(f"seq must hold tile counts in 0..17, got {seq}")
    if phase is not None and phase not in (0, 1, 2, 3):
        raise ValueError(f"phase must be None or 0..3, got {phase}")
    if resident_ctas <= 0 or reps <= 0 or len(seq) * reps < 3:
        raise ValueError("need resident_ctas > 0 and len(seq) * reps >= 3 consecutive designed items per CTA")
    if H % KH != 0 or not 1 <= H // KH <= 16:
        raise ValueError(f"H / KH must be a query-head group in 1..16, got {H}/{KH}")
    R = int(resident_ctas)
    waves = [seq[w % len(seq)] for w in range(len(seq) * reps)]
    W = len(waves)
    g = torch.Generator(device=device).manual_seed(seed)

    # The block space: enough complete blocks for the largest count (sized at has_open = 0, the larger count), at least one tile.
    k0 = max(BLOCKS_PER_TILE, max(count_for_tiles(n, 0, top_k) for n in waves))
    if phase is None:
        T0 = BLOCK_SIZE * k0 - 1  # the first token whose default count admits k0 complete blocks
        S = T0 + W * R
        SKV = S  # natural form: every key up to the position exists
        L0 = SKV
    else:
        L0 = BLOCK_SIZE * k0 + int(phase)  # n_vis = L0 for every designed token -> the tail phase is L0 % 4
        T0 = L0 - 1
        S = T0 + W * R
        SKV = L0
    B = 1 + int(dead_batches)
    kv_lens = [L0] + [0] * int(dead_batches)

    pos = torch.arange(S, device=device)
    # Per-token tail flag of the designed region (constant under a phase, natural otherwise) and the counts per wave.
    n_vis = torch.minimum(pos + 1, torch.tensor(L0, device=device)).clamp_min(0)
    h_tok = (n_vis % BLOCK_SIZE != 0).to(torch.long)
    block_lens0 = torch.zeros(S, dtype=torch.long, device=device)  # fillers: count 0 (one tail-only or dead tile)
    for w, n in enumerate(waves):
        lo, hi = T0 + w * R, T0 + (w + 1) * R
        if n > 0:
            block_lens0[lo:hi] = torch.tensor([count_for_tiles(n, int(h), top_k) for h in h_tok[lo:hi].tolist()], device=device)
    # Lists: a random permutation of the token's complete blocks (the oracle's set rule accepts any order), -1 past the count.
    n_complete = torch.minimum(torch.div(n_vis, BLOCK_SIZE, rounding_mode="floor"), torch.full_like(pos, k0))
    keys = torch.rand(S, k0, generator=g, device=device)
    keys = keys.masked_fill(torch.arange(k0, device=device)[None, :] >= n_complete[:, None], 2.0)
    order = torch.argsort(keys, dim=1)[:, :top_k]
    width = order.shape[1]
    ids0 = torch.full((S, top_k), -1, dtype=torch.long, device=device)
    count0 = torch.minimum(block_lens0, torch.minimum(n_complete, torch.full_like(pos, top_k)))
    ids0[:, :width] = torch.where(torch.arange(width, device=device)[None, :] < count0[:, None], order, torch.full_like(order, -1))

    block_ids = torch.full((B, S, top_k), -1, dtype=torch.int32, device=device)
    block_ids[0] = ids0.to(torch.int32)
    block_lens = torch.zeros(B, S, dtype=torch.int32, device=device)
    block_lens[0] = block_lens0.to(torch.int32)
    # The natural form passes no KV lengths -- unless a dead batch exists: its length 0 is a device fact the kernel can only
    # learn from the tensor (batch 0's entry is then the full extent, so its visible range stays pos + 1).
    seq_kv_lens = None if (phase is None and dead_batches == 0) else torch.tensor(kv_lens, dtype=torch.int32, device=device)

    q = torch.randn(B, S, H, D, device=device, dtype=torch.float32, generator=g)
    k = torch.randn(B, SKV, KH, D, device=device, dtype=torch.float32, generator=g)
    v = torch.randn(B, SKV, KH, D, device=device, dtype=torch.float32, generator=g)
    nan_tokens = torch.zeros(S, dtype=torch.bool, device=device)
    if nan_filler_wave:
        nan_tokens[max(0, T0 - R) : T0] = True
    for w in nan_waves:
        w = int(w) % W
        nan_tokens[T0 + w * R : T0 + (w + 1) * R] = True
    q[:, nan_tokens] = float("nan")

    kv_len_t = torch.tensor(kv_lens, device=device, dtype=torch.long)[:, None].expand(B, S)
    expected_tiles, expected_dead = expected_item_bounds(pos[None, :].expand(B, S), block_lens.to(torch.long), kv_len_t, top_k)
    return ItemSequenceWorkList(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v.to(dtype),
        block_ids=block_ids.contiguous(),
        block_lens=block_lens.contiguous(),
        seq_kv_lens=seq_kv_lens,
        kv_lens=kv_lens,
        top_k=top_k,
        phase=phase,
        seq=seq,
        resident_ctas=R,
        designed_start=T0,
        waves=waves,
        expected_tiles=expected_tiles,
        expected_dead=expected_dead,
        nan_tokens=nan_tokens,
    )


def sentinel_for(dtype: torch.dtype) -> float:
    """A finite value no attention output reaches (fp16 cannot hold 1.5e30)."""
    return 1.5e30 if dtype == torch.bfloat16 else 6.0e4


def poisoned_outputs(wl: ItemSequenceWorkList) -> Tuple[torch.Tensor, torch.Tensor]:
    B, S, H, _, _ = wl.shape
    sent = sentinel_for(wl.q.dtype)
    o = torch.full((B, S, H, D), sent, device=wl.q.device, dtype=wl.q.dtype)
    lse = torch.full((B, H, S), sent, device=wl.q.device, dtype=torch.float32)
    return o, lse


def run_sparse_forward(wl: ItemSequenceWorkList, o: torch.Tensor, lse: torch.Tensor, scale: float, stream=None) -> None:
    """One launch through the adapter (typed declines first), synchronised."""
    from cudnn.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

    a = SparseGqaFwdDslSm107(
        q=wl.q, k=wl.k, v=wl.v, o=o, lse=lse, block_ids=wl.block_ids, block_lens=wl.block_lens, seq_kv_lens=wl.seq_kv_lens, top_k=wl.top_k, scale=scale
    )
    assert a.check_support()
    a.compile()  # plan time: the adapter never compiles on the execute path (the declared block_lens presence selects the variant)
    a.execute(stream=stream)
    torch.cuda.synchronize()


def reference_outputs(wl: ItemSequenceWorkList, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 over the half-rounded operands; the oracle's visible set per row; natural-log LSE; a dead row is O = 0 / LSE = -inf;
    a NaN query row is NaN (it is excluded from the comparison by ``evaluate``)."""
    if _ORACLE_DIR not in sys.path:
        sys.path.insert(0, _ORACLE_DIR)
    import gated_block_qsa_reference as oracle

    B, S, H, KH, SKV = wl.shape
    G = H // KH
    dev = wl.q.device
    ref_o = torch.zeros(B, S, H, D, device=dev, dtype=torch.float32)
    ref_lse = torch.full((B, H, S), float("-inf"), device=dev, dtype=torch.float32)
    for b in range(B):
        L = int(wl.kv_lens[b])
        allowed = oracle.qsa_visible_mask(wl.block_ids[b], wl.block_lens[b], torch.arange(S, device=dev), L, SKV, BLOCK_SIZE, top_k=wl.top_k)
        for h in range(H):
            kk, vv = wl.k[b, :, h // G].float(), wl.v[b, :, h // G].float()
            sc = (wl.q[b, :, h].float() @ kk.t()) * scale
            sc = sc.masked_fill(~allowed, float("-inf"))
            rmax = sc.amax(dim=-1)
            dead = torch.isinf(rmax) & (rmax < 0)
            safe = torch.where(dead, torch.zeros_like(rmax), rmax)
            p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
            den = p.sum(dim=-1)
            oc = (p @ vv) / torch.where(dead, torch.ones_like(den), den)[:, None]
            ref_o[b, :, h] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
            ref_lse[b, h] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
    return ref_o, ref_lse


def checked_digest(wl: ItemSequenceWorkList, o: torch.Tensor, lse: torch.Tensor) -> str:
    """sha256 of the rows the comparison covers (every row whose query is finite, plus the dead rows): the cross-process
    bitwise pin.  NaN-query live rows are left out -- their values are not part of the contract."""
    B, S, H, _, _ = wl.shape
    nan_rows = wl.nan_tokens[None, :, None].expand(B, S, H) & ~wl.expected_dead[:, :, None].expand(B, S, H)
    keep = ~nan_rows
    h = hashlib.sha256()
    h.update(o[keep].contiguous().view(torch.int16).cpu().numpy().tobytes())
    h.update(lse.permute(0, 2, 1)[keep].contiguous().view(torch.int32).cpu().numpy().tobytes())
    return h.hexdigest()


def evaluate(wl: ItemSequenceWorkList, o: torch.Tensor, lse: torch.Tensor, ref_o: torch.Tensor, ref_lse: torch.Tensor, atol: float) -> Dict[str, object]:
    """The cell's verdicts: sentinel survivors (never written), NaN on a finite-query row, dead rows exact, the oracle distance
    on the live finite-query rows.  Returns metrics + ``failures`` (empty = PASS); the caller asserts or maps to an exit code."""
    B, S, H, _, _ = wl.shape
    sent = sentinel_for(wl.q.dtype)
    of = o.float()
    dead = wl.expected_dead[:, :, None].expand(B, S, H)  # [B, S, H]
    ref_dead = torch.isinf(ref_lse).permute(0, 2, 1)  # the oracle's own dead rows, [B, S, H]
    nan_q = wl.nan_tokens[None, :, None].expand(B, S, H)
    lse_bsh = lse.permute(0, 2, 1)
    ref_lse_bsh = ref_lse.permute(0, 2, 1)
    failures: List[str] = []

    sent_o_rows = int((of == sent).any(dim=-1).sum())
    sent_lse = int((lse == sent).sum())
    if sent_o_rows or sent_lse:
        failures.append(f"sentinel survived on {sent_o_rows} O rows / {sent_lse} LSE cells (never written)")

    # The host's dead set and the oracle's must agree (a construction check, not a kernel check).
    if not torch.equal(dead, ref_dead):
        failures.append(f"host dead set ({int(dead.sum())}) != oracle dead set ({int(ref_dead.sum())})")

    live_finite = ~dead & ~nan_q
    nan_o_rows = int((~torch.isfinite(of)).any(dim=-1)[live_finite].sum())
    nan_lse = int((~torch.isfinite(lse_bsh))[live_finite].sum())
    if nan_o_rows or nan_lse:
        failures.append(f"non-finite output on {nan_o_rows} O rows / {nan_lse} LSE cells whose query is finite (residue travelled)")

    dead_o_ok = bool((of[dead] == 0).all()) if dead.any() else True
    dead_lse_ok = bool((torch.isinf(lse_bsh[dead]) & (lse_bsh[dead] < 0)).all()) if dead.any() else True
    if not (dead_o_ok and dead_lse_ok):
        failures.append(f"dead rows not exact: O == 0 {dead_o_ok}, LSE == -inf {dead_lse_ok}")

    d_o = (of - ref_o).abs()
    d_o = torch.where(live_finite[..., None], d_o, torch.zeros_like(d_o))
    d_lse = torch.where(live_finite, (lse_bsh - ref_lse_bsh).abs(), torch.zeros_like(lse_bsh))
    max_o = float(d_o.max()) if live_finite.any() else 0.0
    max_lse = float(d_lse.max()) if live_finite.any() else 0.0
    if not (max_o <= atol and max_lse <= atol):
        failures.append(f"outside the budget: max|dO| {max_o:.5f} max|dLSE| {max_lse:.6f} > {atol}")

    return dict(
        failures=failures,
        max_abs_o=max_o,
        max_abs_lse=max_lse,
        sentinel_o_rows=sent_o_rows,
        sentinel_lse=sent_lse,
        nonfinite_o_rows=nan_o_rows,
        nonfinite_lse=nan_lse,
        dead_rows=int(dead.sum()),
        dead_o_is_zero=dead_o_ok,
        dead_lse_is_neg_inf=dead_lse_ok,
        live_rows_checked=int(live_finite.sum()),
        nan_query_rows=int((nan_q & ~dead).sum()),
        digest=checked_digest(wl, o, lse),
    )


def default_scale() -> float:
    return 1.0 / math.sqrt(D)
