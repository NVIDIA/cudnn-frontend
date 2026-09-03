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


def make_bwd_decode(CFG):
    """Return ``(decode_initial, decode_payload)`` for a role-split cluster.

    Both map a cluster coordinate to ``(q_block, head, batch)`` where
    ``q_block`` is in units of TILE_M rows and already carries ``cta_in_pair``.

    The two sub-groups of a cluster deliberately land on the SAME q rows: sg1's
    CTA k must hold the row-block sg0's CTA k holds, because the fp32 S ship is
    lane-to-lane across ``cross_sg_peer = cta_id ^ CTA_MMA``. That falls out of
    keying the row on ``cta_in_pair`` (0/1 within a pair) rather than on the
    cluster-wide CTA id.
    """

    @cute.jit
    def decode_initial(bidx, bidy, bidz, cta_in_pair):
        """From blockIdx, for the first tile."""
        q_block = (bidx // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
        return q_block, bidy, bidz

    @cute.jit
    def decode_payload(t0, t1, cta_in_pair):
        """From a CLC response: ``t0`` is the cancelled cluster's blockIdx.x,
        ``t1`` packs head in the low 16 bits and batch in the high 16."""
        q_block = (t0 // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
        head = t1 & cutlass.Int32(0xFFFF)
        batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
        return q_block, head, batch

    return decode_initial, decode_payload


__all__ = ["make_bwd_decode"]
