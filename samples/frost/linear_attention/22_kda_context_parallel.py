# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 22: KDA (Kimi Delta Attention) context parallelism.

One packed stream is cut into ``cp_size`` equal slabs standing in for the ranks.  Every slab summarizes its boundary
sequence with ``kimi_delta_attention_summary`` on a two-entry device view of ``cu_seqlens`` (the last local sequence from a zero
state), the summaries are stacked as the all-gather would stack them, and each slab runs ``kimi_delta_attention`` with the slice of
preceding ranks as ``preceding_state_update`` / ``preceding_transition``: the op folds them on the device into a
one-row composed state the first local sequence starts from and returns it for the backward.  The backward mirrors it with ``kimi_delta_attention_summary_bwd``
on the first local sequence (a zero outgoing-state gradient) and ``torch.ops.cudnn.kimi_delta_attention_bwd`` from the composed state
with the following ranks' summaries folded into the one-row gradient state the last local sequence starts from.  The rank bounds come from
the rank-bound rule on the global sequence boundaries; the slabs' outputs and gradients are checked against the unsplit op.  The backward prepare also returns its intermediates (``output_intermediates``), which the apply takes back and
skips the passes that would recompute them (a multi-sequence slab takes the per-token rows and leaves the per-piece tables).
"""

from __future__ import annotations

import torch

from cudnn.linear_attention import ops as la_ops


def _rms_ratio(out, ref):
    out, ref = out.detach().double(), ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _rank_bounds(bounds, slab_len, rank):
    """The ``(preceding_rank_start, following_rank_stop)`` of one rank: the rank holding the first token of this rank's first
    sequence, and one past the rank holding the last token of its last sequence."""
    start, stop = rank * slab_len, (rank + 1) * slab_len
    first_seq = max(i for i in range(len(bounds) - 1) if bounds[i] <= start)
    last_seq = max(i for i in range(len(bounds) - 1) if bounds[i] < stop)
    return bounds[first_seq] // slab_len, (bounds[last_seq + 1] - 1) // slab_len + 1


def _local_cu(bounds, slab_len, rank, device):
    start, stop = rank * slab_len, (rank + 1) * slab_len
    return torch.tensor([0, *[b - start for b in bounds if start < b < stop], slab_len], dtype=torch.int32, device=device)


def main(seq_lens=(700, 1700, 100, 500, 1096), cp_size: int = 4, H: int = 2, D: int = 128) -> None:
    torch.manual_seed(0)
    device = "cuda"
    total = sum(seq_lens)
    assert total % cp_size == 0
    slab_len = total // cp_size
    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    cu_whole = torch.tensor(bounds, dtype=torch.int32, device=device)
    q = torch.nn.functional.normalize(torch.randn(total, H, D, device=device), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(total, H, D, device=device), dim=-1).bfloat16()
    v = (torch.randn(total, H, D, device=device) * 0.5).bfloat16()
    gate = torch.empty(total, H, D, device=device).uniform_(0.9, 1.0).log()
    beta = torch.rand(total, H, device=device)
    leaves = dict(q=q, k=k, v=v, g=gate, beta=beta)
    rows = {name: 1 for name in leaves}
    scale = D**-0.5
    CHECKPOINT = 16

    do = (torch.randn(total, H, D, device=device) * 0.1).bfloat16()

    # ---- the unsplit stream under autograd: the reference the ranks must reproduce ------------------------------
    whole = {name: x.detach().clone().requires_grad_(True) for name, x in leaves.items()}
    t, cu = whole, cu_whole
    o_whole = la_ops.kimi_delta_attention(t["q"], t["k"], t["v"], t["g"], t["beta"], cu, scale)[0]
    grads_whole = dict(zip(whole, torch.autograd.grad([o_whole], list(whole.values()), [do])))

    # ---- per rank: the slab's tensors and local cu_seqlens; prepare the boundary summaries --------------------
    slabs, cus, dos = [], [], []
    for rank in range(cp_size):
        slabs.append({name: x[rank * slab_len * rows[name] : (rank + 1) * slab_len * rows[name]] for name, x in leaves.items()})
        cus.append(_local_cu(bounds, slab_len, rank, device))
        dos.append(do[rank * slab_len : (rank + 1) * slab_len])
    summaries = []
    for t, cu in zip(slabs, cus):
        h, m = la_ops.kimi_delta_attention_summary(t["k"], t["v"], t["g"], t["beta"], cu[-2:])  # the LAST local sequence, from a zero state
        summaries.append((h[0], m[0]))
    h_all = torch.stack([s[0] for s in summaries])  # what the all-gather hands every rank
    m_all = torch.stack([s[1] for s in summaries])

    # ---- forward apply: the preceding slice, folded by the op into the one-row composed state ----------------------
    outputs = []
    for rank, (t, cu) in enumerate(zip(slabs, cus)):
        preceding_rank_start, _ = _rank_bounds(bounds, slab_len, rank)
        h, m = h_all[preceding_rank_start:rank], m_all[preceding_rank_start:rank]
        o, _, checkpoints, composed = la_ops.kimi_delta_attention(
            t["q"], t["k"], t["v"], t["g"], t["beta"], cu, scale, preceding_state_update=h, preceding_transition=m, checkpoint_every_n_tokens=CHECKPOINT
        )
        outputs.append((o, composed, checkpoints))
    o_cp = torch.cat([out[0] for out in outputs])
    r_o = _rms_ratio(o_cp, o_whole)
    assert r_o < 2e-2, f"o rms ratio {r_o:.4g}"
    # the composed state of the last rank, against the fp64 fold of the gathered summaries
    preceding_rank_start, _ = _rank_bounds(bounds, slab_len, cp_size - 1)
    x = torch.zeros(*(H, D, D), dtype=torch.float64, device=device)
    for j in range(preceding_rank_start, cp_size - 1):
        h, m = h_all[j].double(), m_all[j].double()
        x = x @ m + h
    r_x = _rms_ratio(outputs[-1][1][0], x)
    assert r_x < 1e-5, f"composed state rms ratio {r_x:.4g}"

    # ---- backward: prepare on the first local sequence, gather, apply with the following slice ------------------
    summaries = []
    for t, cu, do in zip(slabs, cus, dos):
        g_, t_, intermediates = la_ops.kimi_delta_attention_summary_bwd(
            t["q"], t["k"], t["g"], t["beta"], do, cu[:2], scale=scale, output_transition=True, output_intermediates=True, checkpoint_every_n_tokens=CHECKPOINT
        )  # the FIRST local sequence
        summaries.append((g_[0], t_[0], intermediates))
    g_all = torch.stack([s[0] for s in summaries])
    t_all = torch.stack([s[1] for s in summaries])
    grads = {}
    for rank, (t, cu, do) in enumerate(zip(slabs, cus, dos)):
        _, following_rank_stop = _rank_bounds(bounds, slab_len, rank)
        composed, checkpoints = outputs[rank][1], outputs[rank][2]
        g_, t_ = g_all[rank + 1 : following_rank_stop], t_all[rank + 1 : following_rank_stop]
        for name, grad in zip(
            ("q", "k", "v", "g", "beta"),
            torch.ops.cudnn.kimi_delta_attention_bwd(
                do,
                t["q"],
                t["k"],
                t["v"],
                t["g"],
                t["beta"],
                cu,
                scale,
                composed_initial_state=composed if composed.numel() else None,
                following_state_grad=g_,
                following_transition=t_,
                state_checkpoints=checkpoints if checkpoints.numel() else None,
                checkpoint_every_n_tokens=CHECKPOINT,
                **summaries[rank][2],
            )[:5],
        ):
            if name is not None:
                grads.setdefault(name, []).append(grad)
    worst = 0.0
    for name, pieces in grads.items():
        r = _rms_ratio(torch.cat(pieces), grads_whole[name])
        assert r < 5e-2, f"d{name} rms ratio {r:.4g}"
        worst = max(worst, r)
    print(
        f"[22] PASS  kda context parallel      seq_lens={list(seq_lens)} cp_size={cp_size} H={H} (o rms {r_o:.2e}, composed rms {r_x:.2e}, worst grad rms {worst:.2e})"
    )


if __name__ == "__main__":
    main()
