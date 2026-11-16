# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tail launch of a wave-split batch (``choose_tail``: the W full waves run uncut as the prefix, the R leftover tiles as a
standalone chain or d_v-split launch) and its device-side admission.  The first kernel of the prefix launch that reads the item
count runs :func:`gate_body` in the CTA that reads it: the sequence lengths of ``cu_seqlens`` in, two int32 words out::

    gate[0] tail_enable   1 = the prefix runs the first prefix_tiles tiles and the tail launch the rest, 0 = the prefix runs the whole batch
    gate[1] item limit    the prefix prologue's item count: prefix_tiles when the tail runs, n_tiles when it does not, 0 when the chain runs
    gate[2] chain_enable  1 = the length-rule chain group runs the whole batch (a plan with ``chain_pieces``), 0 = it runs nothing"""

import math
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from .piece_chain import (
    CHAIN_MAX_PIECES,
    CHAIN_MIN_MEAN_TOKENS,
    CHAIN_MIN_PIECES,
    CHAIN_MIN_UNITS_PER_PIECE,
    CHAIN_PIECE_TOKENS,
    DV_SPLIT_TILES,
    piece_count,
)

GATE_WORDS = 3
TAIL_RAGGED_PCT = 125
TAIL_MAX_WAVES = 4


class TailChoice(NamedTuple):
    """The tail scheme of a batch above one wave of (sequence, head) tiles: the prefix runs the first ``prefix_tiles`` tiles
    (``W`` full waves, tile order ``b * HO + h``) and the ``tiles`` leftover tiles run as a standalone chain over the
    sequences from ``batch_base`` on, the heads below ``first_head_lo`` of that first sequence belonging to the prefix.
    ``dv_split``: the tail runs as the d_v split (every leftover tile as ``DV_SPLIT_TILES`` CTAs, ``pieces`` 0) instead of
    the chain."""

    prefix_tiles: int
    tiles: int
    batch_base: int
    first_head_lo: int
    num_seqs: int
    pieces: int
    unit_chunks: int
    dv_split: bool


def choose_tail(*, num_seqs, heads_out, num_sm, total_tokens, b_t, cadence_tokens, expand_num, dim_v=None):
    """The tail scheme of a plan without ``batch_invariant`` whose tiles exceed one wave, or None: the ``W = tiles // num_sm``
    full waves (at most ``TAIL_MAX_WAVES``) run uncut as the prefix and the ``R = tiles mod num_sm`` leftover tiles are a
    standalone call."""
    num_seqs, heads_out, num_sm = int(num_seqs), int(heads_out), int(num_sm)
    tiles = num_seqs * heads_out
    full_waves = tiles // num_sm
    leftover = tiles - full_waves * num_sm
    if full_waves < 1 or full_waves > TAIL_MAX_WAVES or leftover == 0:
        return None
    b_t = int(b_t)
    expand_num = max(1, int(expand_num))
    cadence_chunks = max(1, int(cadence_tokens) // b_t)
    unit_chunks = math.lcm(expand_num, cadence_chunks)
    prefix_tiles = full_waves * num_sm
    batch_base = prefix_tiles // heads_out
    first_head_lo = prefix_tiles - batch_base * heads_out
    tail_seqs = num_seqs - batch_base
    mean_tokens = int(total_tokens) / max(1, num_seqs)
    tail_chunks = -(-int(mean_tokens * tail_seqs * expand_num) // b_t)
    cap = max(CHAIN_MAX_PIECES, int(total_tokens) // (num_seqs * CHAIN_PIECE_TOKENS))
    pieces = min(num_sm // leftover, cap, tail_chunks // (tail_seqs * CHAIN_MIN_UNITS_PER_PIECE * unit_chunks))
    if pieces >= CHAIN_MIN_PIECES and mean_tokens >= CHAIN_MIN_MEAN_TOKENS:
        return TailChoice(prefix_tiles, leftover, batch_base, first_head_lo, tail_seqs, pieces, unit_chunks, False)
    if dim_v is None or int(dim_v) != 128 or pieces != DV_SPLIT_TILES:
        return None
    return TailChoice(prefix_tiles, leftover, batch_base, first_head_lo, tail_seqs, 0, unit_chunks, True)


@cute.jit
def gate_body(
    n_threads: cutlass.Constexpr[int],
    tidx,
    heads_out,
    num_sms: cutlass.Constexpr[int],
    mCuSeqlens: cute.Tensor,
    mGate: cute.Tensor,
    chain_pieces: cutlass.Constexpr[int] = 0,
    tail: cutlass.Constexpr[bool] = True,
):
    """The gate of a prefix launch over its ``cu_seqlens`` and the int32 ``[GATE_WORDS]`` words, called by every thread of one
    CTA of ``n_threads`` threads before that CTA reads word 1 (CTA barriers inside; the words are written when it returns):
    ``n_tiles`` every (sequence, head) tile of the batch at ``heads_out`` heads (a prefix launch runs one tile per head),
    ``prefix_tiles`` its full waves of ``num_sms`` tiles (``n_tiles`` without ``tail``).  ``chain_pieces``: the chain runs when
    the sequences the length rule cuts at that many slots fill at most half the SMs with their tiles and the cut halves the
    longest item."""
    num_seqs = cutlass.Int32(mCuSeqlens.shape[0]) - cutlass.Int32(1)
    n_tiles = num_seqs * heads_out
    prefix_tiles = (n_tiles // cutlass.Int32(num_sms)) * cutlass.Int32(num_sms) if cutlass.const_expr(tail) else n_tiles
    sLongest = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=8)
    sShortest = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=8)
    sCutItem = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=8)
    sCutSeqs = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if tidx == cutlass.Int32(0):
        sLongest[0] = cutlass.Int32(0)
        sShortest[0] = cutlass.Int32(0x7FFFFFFF)
        sCutItem[0] = cutlass.Int32(0)
        sCutSeqs[0] = cutlass.Int32(0)
    nvvm.barrier_cta_sync()
    longest = cutlass.Int32(0)
    shortest = cutlass.Int32(0x7FFFFFFF)
    cut_item = cutlass.Int32(0)
    cut_seqs = cutlass.Int32(0)
    b = tidx
    while b < num_seqs:
        n = cutlass.Int32(mCuSeqlens[b + cutlass.Int32(1)]) - cutlass.Int32(mCuSeqlens[b])
        longest = n if n > longest else longest
        shortest = n if (n > cutlass.Int32(0)) and (n < shortest) else shortest
        if cutlass.const_expr(chain_pieces > 0):
            count = piece_count(cutlass.Int32(chain_pieces), 1, 1, True, n, cutlass.Int32(1), CHAIN_MIN_PIECES)
            item = (n + count - cutlass.Int32(1)) // count
            cut_item = item if item > cut_item else cut_item
            cut_seqs = cut_seqs + (cutlass.Int32(1) if count > cutlass.Int32(1) else cutlass.Int32(0))
        b = b + cutlass.Int32(n_threads)
    nvvm.atomicrmw("max", sLongest.data_ptr(0), longest, space=nvvm.SharedSpace.shared_cta)
    nvvm.atomicrmw("min", sShortest.data_ptr(0), shortest, space=nvvm.SharedSpace.shared_cta)
    if cutlass.const_expr(chain_pieces > 0):
        nvvm.atomicrmw("max", sCutItem.data_ptr(0), cut_item, space=nvvm.SharedSpace.shared_cta)
        nvvm.atomicrmw("add", sCutSeqs.data_ptr(0), cut_seqs, space=nvvm.SharedSpace.shared_cta)
    nvvm.barrier_cta_sync()
    if tidx == cutlass.Int32(0):
        ragged = sLongest[0] * cutlass.Int32(100) > sShortest[0] * cutlass.Int32(TAIL_RAGGED_PCT)
        tail_enable = cutlass.Int32(0) if ragged else cutlass.Int32(1)
        items = n_tiles if ragged else prefix_tiles
        if cutlass.const_expr(chain_pieces > 0):
            cut_tiles = sCutSeqs[0] * heads_out * cutlass.Int32(2)
            chain = (cut_tiles > cutlass.Int32(0)) and (cut_tiles <= cutlass.Int32(num_sms)) and (sCutItem[0] * cutlass.Int32(2) <= sLongest[0])
            tail_enable = cutlass.Int32(0) if chain else tail_enable
            items = cutlass.Int32(0) if chain else items
            mGate[2] = cutlass.Int32(1) if chain else cutlass.Int32(0)
        mGate[0] = tail_enable
        mGate[1] = items
    nvvm.barrier_cta_sync()
