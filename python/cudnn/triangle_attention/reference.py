# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp64 reference of triangle attention (AlphaFold 3 Algorithms 14 / 15: attention over one axis of the pair representation with a
pair bias and a key mask), cuEquivariance's tensor convention.

For ``q, k, v`` of shape ``(B, N, H, S, D)`` (``k`` / ``v`` may carry ``H_kv`` heads dividing ``H``), a pair bias ``(B|1, 1, H|1, S_q, S_kv)``
shared by the ``N`` rows of an outer batch entry, and ONE key-mask form -- a dense boolean ``mask`` ``(B, N, 1, 1, S_kv)`` (True = live)
or right-padded ``kv_lengths`` ``(B, N, 1, 1, 1)`` (keys ``j < kv_lengths[b, n]`` are live) -- with ``scale`` (None = ``1 / sqrt(D)``)::

    s[b, n, h, i, j] = scale * <q[b, n, h, i], k[b, n, h, j]> + bias[b, 0, h, i, j]        for a live key j, -inf otherwise
    max[b, n, h, i]  = max_j s                        lse = max + log( sum_j exp(s - max) )
    o[b, n, h, i]    = sum_j softmax_j(s) v[b, n, h, j]

Everything runs in fp64 on the tensors' device.  Masked-row contract (the FROST SDPA invariant for an empty key range): a row with
NO live key reads ``o = 0``, ``lse = max = -inf``, on both mask forms.  cuEquivariance's two routes read such a row differently
(``kv_lengths``: ``o = lse = max = 0``; dense mask: a finite ``-1e9`` is folded, so ``o`` = the uniform average of ``v`` and ``lse =
max = -1e9``) -- a parity test against it compares rows WITH keys only.  Returns ``(o, lse, max)`` as fp64 tensors shaped
``(B, N, H, S_q, D)``, ``(B, N, H, S_q)``, ``(B, N, H, S_q)``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def triangle_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    *,
    kv_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(o, lse, max)`` in fp64 for the formula in the module docstring (the bias enters at its own stored value: pass the bf16 /
    fp16 bias the kernel reads, not its fp32 source, to measure the kernel's arithmetic alone)."""
    if mask is not None and kv_lengths is not None:
        raise ValueError("pass either mask or kv_lengths, not both")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.ndim != 5:
            raise ValueError(f"{name} must be (B, N, H, S, D), got {tuple(t.shape)}")
    if bias.ndim != 5:
        raise ValueError(f"bias must be (B|1, 1, H|1, S_q, S_kv), got {tuple(bias.shape)}")
    b, n, h, s_q, d = q.shape
    h_kv, s_kv = k.shape[2], k.shape[3]
    if h % h_kv:
        raise ValueError(f"the KV head count must divide the Q head count, got H={h}, H_kv={h_kv}")
    group = h // h_kv
    qd = q.double()
    kd = k.double().repeat_interleave(group, dim=2) if group > 1 else k.double()
    vd = v.double().repeat_interleave(group, dim=2) if group > 1 else v.double()
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(d)
    scores = torch.einsum("bnhqd,bnhkd->bnhqk", qd, kd) * sc + bias.double()  # the bias broadcasts over N (and over H when declared 1)
    live = torch.ones((b, n, 1, 1, s_kv), dtype=torch.bool, device=q.device)
    if mask is not None:
        if tuple(mask.shape) != (b, n, 1, 1, s_kv) or mask.dtype != torch.bool:
            raise ValueError(f"mask must be a boolean (B, N, 1, 1, S_kv) = {(b, n, 1, 1, s_kv)} tensor, got {tuple(mask.shape)} {mask.dtype}")
        live = live & mask
    if kv_lengths is not None:
        if tuple(kv_lengths.shape) != (b, n, 1, 1, 1):
            raise ValueError(f"kv_lengths must be (B, N, 1, 1, 1) = {(b, n, 1, 1, 1)}, got {tuple(kv_lengths.shape)}")
        live = live & (torch.arange(s_kv, device=q.device).view(1, 1, 1, 1, s_kv) < kv_lengths.to(device=q.device, dtype=torch.int64))
    scores = scores.masked_fill(~live, float("-inf"))
    row_max = scores.amax(dim=-1)  # -inf on a row without a live key
    p = torch.exp(scores - row_max.unsqueeze(-1)).nan_to_num(0.0)  # exp(-inf - -inf) = nan on an empty row -> 0
    row_sum = p.sum(dim=-1)
    lse = row_max + torch.log(row_sum)  # -inf + -inf = -inf on an empty row
    o = torch.einsum("bnhqk,bnhkd->bnhqd", (p / row_sum.unsqueeze(-1)).nan_to_num(0.0), vd)
    return o, lse, row_max


__all__ = ["triangle_attention_reference"]
