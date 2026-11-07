# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The context a FROST LA summary leaves for the apply of the same tokens: one byte buffer with the workspace's
lifecycle, carved by the summary's geometry on both ends."""

from __future__ import annotations

from cudnn.frost.workspace import WorkspaceLayout, carve_plan

from .piece_chain import choose_pieces


class Context:
    """One context's carve, with the summary's cut ``(pieces, unit_chunks)`` and which groups it holds."""

    __slots__ = ("size", "names", "carve", "pieces", "unit_chunks", "has_rows", "has_tables")

    def __init__(self, owner, layout, regions, pieces, unit_chunks, has_rows, has_tables):
        self.size = layout.size
        self.names = [name for name, off, dt, shape in regions]
        self.carve = carve_plan(owner, [(off, dt, shape) for name, off, dt, shape in regions]) if regions else None
        self.pieces = pieces
        self.unit_chunks = unit_chunks
        self.has_rows = has_rows
        self.has_tables = has_tables

    def regions(self, context):
        """The regions over the caller's ``context`` buffer (a validated :class:`~cudnn.frost.workspace.Workspace`)."""
        return dict(zip(self.names, context.carve(self.carve))) if self.names else {}


def backward_context(
    owner,
    *,
    full,
    total,
    num_householder,
    h_q,
    h_k,
    h_o,
    d_k,
    d_v,
    b_t,
    io_name,
    num_sm,
    cadence_tokens,
    batch_invariant,
    use_qk_l2norm,
    tinv_rows=None,
):
    """The backward summary's context over ``total`` k rows of one local sequence: the normalized q / k rows and inverse
    norms (``full`` families, where the backward normalizes the same rows), and for a chained summary the chunk factors
    (``full``) and the per-piece transitions and state gradients."""
    expand_num = max(1, int(num_householder))
    q_rows = int(total) // expand_num
    pieces, unit_chunks = choose_pieces(
        num_seqs=1,
        heads_out=int(h_o),
        num_sm=int(num_sm),
        total_tokens=q_rows,
        b_t=int(b_t),
        cadence_tokens=int(cadence_tokens or 0),
        batch_invariant=bool(batch_invariant),
        compose_tail=True,
        expand_num=expand_num,
        reverse=True,
    )
    has_rows = bool(full and use_qk_l2norm and (expand_num == 1 or int(d_v) == 64))
    has_tables = pieces > 0
    layout = WorkspaceLayout()
    regions = []
    if has_rows:
        regions.append(("q_n", layout.add(q_rows * h_q * d_k * 2), io_name, (q_rows, h_q, d_k)))
        regions.append(("k_n", layout.add(total * h_k * d_k * 2), io_name, (total, h_k, d_k)))
        regions.append(("inv_q", layout.add(q_rows * h_q * 4), "float32", (q_rows, h_q)))
        regions.append(("inv_k", layout.add(total * h_k * 4), "float32", (total, h_k)))
    if full and has_tables:
        rows = tinv_rows(q_rows, pieces, expand_num, b_t)
        regions.append(("tinv", layout.add(rows * h_o * b_t * b_t * 2), io_name, (rows, h_o, b_t, b_t)))
    if has_tables:
        regions.append(("state_m", layout.add(pieces * h_o * d_k * d_k * 4), "float32", (pieces, h_o, d_k, d_k)))
        regions.append(("state_g", layout.add(pieces * h_o * d_v * d_k * 4), "float32", (pieces, h_o, d_v, d_k)))
    return Context(owner, layout, regions, pieces, unit_chunks, has_rows, has_tables)


def forward_context(owner, *, total, h_k, d_k, io_name, use_qk_l2norm):
    """The forward summary's context over ``total`` k rows: the normalized k rows under ``use_qk_l2norm``."""
    layout = WorkspaceLayout()
    regions = []
    if use_qk_l2norm:
        regions.append(("k_n", layout.add(total * h_k * d_k * 2), io_name, (total, h_k, d_k)))
    return Context(owner, layout, regions, 0, 0, bool(use_qk_l2norm), False)
