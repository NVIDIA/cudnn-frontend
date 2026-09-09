# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared helpers for the FROST SM100 SDPA-backward kernels.

Today this is the persistent-scheduler tile decode. It is a deliberate copy of
the ``SCHED_NATURAL`` / ``SPLIT_PIPELINE == 1`` arm of
``cudnn.sdpa.fwd.kernels._common_sm100.make_sdpa_helpers`` rather than an import
of it: that factory also carries split-KV, pack-GQA and THD, none of which the
backward has, and cross-pass imports between kernel trees are the coupling the
FROST engine contract warns about. The arithmetic below is identical, so the two
must stay in step -- if the forward's natural decode changes, change this too.

The LPT / LPT_L2 policies are NOT copied; they need ``lpt_tile_coords`` and its
L2-residency model. ``config_sm100._validate_cfg_d512`` rejects them, so the
missing arms are unreachable rather than silently wrong.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.thd import thd_decode_unit


def make_bwd_decode(CFG):
    """Return ``(decode_initial, decode_payload)`` for a role-split cluster.

    Both map a cluster coordinate to the FULL per-tile context::

        (q_block, head, batch, q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv)

    ``q_block`` is in units of TILE_M rows and already carries ``cta_in_pair``.
    The two sub-groups of a cluster deliberately land on the SAME q rows: sg1's
    CTA k must hold the row-block sg0's CTA k holds, because the fp32 S ship is
    lane-to-lane across ``cross_sg_peer = cta_id ^ CTA_MMA``. That falls out of
    keying the row on ``cta_in_pair`` (0/1 within a pair) rather than on the
    cluster-wide CTA id.

    **The per-sequence values ride WITH the decode rather than beside it.**
    Under THD a tile change also changes ``cu_q[b]``, ``cu_k[b]``,
    ``row_off[b]`` and both lengths, and every warp role needs some subset of
    them.  Returning them from the same call is what makes "forgot to refresh
    after advancing the tile" impossible -- that omission would not fault, it
    would address the previous sequence.  Dense returns
    ``(0, 0, 0, scalar_q, scalar_kv)`` for the trailing five and every use folds
    away at trace time.

    Under ``CFG.THD_VARLEN`` both arms decode the SAME linear unit id -- from
    ``blockIdx.x`` for the first tile, from the persistent scheduler's payload
    word afterwards -- so there is one THD body. A unit is
    ``TILE_M * CTA_MMA`` q rows of one head of one sequence, matching what the
    setup launch counted into ``live``.
    """
    _thd = int(getattr(CFG, "THD_VARLEN", 0))
    _CGA_TILE_M = CFG.TILE_M * CFG.CTA_MMA

    @cute.jit
    def _coords(meta_t, batch, n_batch, scalar_seqlen_q, scalar_seqlen_kv):
        """``(q_tok, kv_tok, ws_row, seqlen_q, seqlen_kv)`` for one sequence.

        ``q_tok`` / ``kv_tok`` are ``cu_q[b]`` / ``cu_k[b]``, added to the
        SEQUENCE coordinate of the packed Q/dO and K/V descriptors (whose batch
        coordinate is then always 0 -- a packed tensor has one batch).
        ``ws_row`` is ``row_off[b]``, added to the row coordinate of the blocked
        S/dS workspace.  The two lengths drive the per-cell mask and the kv loop
        bounds, replacing the scalars a dense launch threads.
        """
        if cutlass.const_expr(_thd):
            meta = cutlass.make_array_view(meta_t)
            cu_q0 = n_batch
            cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
            row0 = cutlass.Int32(4) * n_batch + cutlass.Int32(4)
            q_tok = cutlass.Int32(meta[cu_q0 + batch])
            kv_tok = cutlass.Int32(meta[cu_k0 + batch])
            # A DEAD unit (the grid is occupancy-sized, not work-sized) decodes
            # batch == n_batch.  Every read below then stays IN BOUNDS of the
            # metadata buffer and yields a NEGATIVE length: cu_q[n_batch] is the
            # packed total and the word after it is cu_k[0] == 0.  Each role's
            # kv range collapses to empty instead of faulting.
            s_q = cutlass.Int32(meta[cu_q0 + batch + cutlass.Int32(1)]) - q_tok
            s_kv = cutlass.Int32(meta[cu_k0 + batch + cutlass.Int32(1)]) - kv_tok
            return q_tok, kv_tok, cutlass.Int32(meta[row0 + batch]), s_q, s_kv
        return cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), scalar_seqlen_q, scalar_seqlen_kv

    @cute.jit
    def _thd_decode(linear_cta, meta_t, n_batch, n_qh, cta_in_pair):
        u = linear_cta // cutlass.Int32(CFG.CGA_M)
        meta = cutlass.make_array_view(meta_t)
        q_tile_idx, batch, head = thd_decode_unit(meta, n_batch, u, n_qh, cutlass.Int32(_CGA_TILE_M), False)
        return q_tile_idx * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair, head, batch

    @cute.jit
    def decode_initial(bidx, bidy, bidz, cta_in_pair, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv):
        """From blockIdx, for the first tile."""
        if cutlass.const_expr(_thd):
            q_block, head, batch = _thd_decode(bidx, meta_t, n_batch, n_qh, cta_in_pair)
        else:
            q_block = (bidx // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
            head, batch = bidy, bidz
        q_tok, kv_tok, ws_row, s_q, s_kv = _coords(meta_t, batch, n_batch, seqlen_q, seqlen_kv)
        return q_block, head, batch, q_tok, kv_tok, ws_row, s_q, s_kv

    @cute.jit
    def decode_payload(t0, t1, cta_in_pair, meta_t, n_batch, n_qh, seqlen_q, seqlen_kv):
        """From the scheduler payload: dense (CLC) ``t0`` is the cancelled
        cluster's blockIdx.x and ``t1`` packs head in the low 16 bits and batch
        in the high 16; THD (persistent claim) puts ``unit * CGA_M`` in ``t0``
        and leaves ``t1`` unused."""
        if cutlass.const_expr(_thd):
            q_block, head, batch = _thd_decode(t0, meta_t, n_batch, n_qh, cta_in_pair)
        else:
            q_block = (t0 // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
            head = t1 & cutlass.Int32(0xFFFF)
            batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
        q_tok, kv_tok, ws_row, s_q, s_kv = _coords(meta_t, batch, n_batch, seqlen_q, seqlen_kv)
        return q_block, head, batch, q_tok, kv_tok, ws_row, s_q, s_kv

    return decode_initial, decode_payload


__all__ = ["make_bwd_decode"]
