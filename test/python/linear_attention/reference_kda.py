# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
fp64 recurrent reference for KDA (Kimi Delta Attention) linear attention.

KDA generalizes GDN's scalar per-token decay to a per-key-channel decay
vector ``alpha_t = exp(g_t) in (0, 1]^K``, applied as a diagonal matrix on
the K-axis of the recurrent state (``k``/``q`` already feature-mapped, ``q``
pre-scaled):

    S_t = (I - beta_t k_t^T k_t) Diag(alpha_t) S_{t-1} + beta_t k_t^T v_t
    o_t = q_t S_t

with scalar per-token write strength ``beta_t``. The decay is applied FIRST
(``S_dec = Diag(alpha_t) S_{t-1}``) and the delta-rule correction reads the
already-decayed state, so unlike GDN the order matters. Supports grouped
heads (every input's head count must divide ``HO = max(Hq, Hv)``; heads are
replicated onto the HO output heads), an optional initial state, and varlen
packed batches via ``cu_seqlens``.

All math runs in fp64 on the input device and is differentiable, so it
doubles as the gradient oracle for the bprop tests.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def rms_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """RMS(out - ref) / RMS(ref), the bf16-appropriate parity metric."""
    out = out.detach().double()
    ref = ref.detach().double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def _recurrent_dense(q, k, v, alpha, beta, S0):
    """Dense recurrence in [B, HV, T, *] layout, fp64. Returns (o, S_T).

    ``alpha`` is per-key-channel: [B, HV, T, K] (GDN's is scalar [B, HV, T])."""
    T = q.shape[2]
    S = S0
    outs = []
    for t in range(T):
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        at = alpha[:, :, t, :]  # [B, HV, K] per-channel decay
        bt = beta[:, :, t]
        S = at[..., None] * S  # decay first: Diag(alpha) S, per-K-row scaling
        kt_S = (kt.unsqueeze(-2) @ S).squeeze(-2)
        residual = vt - kt_S  # reads the already-decayed state (no scalar alpha here)
        S = S + bt[..., None, None] * (kt.unsqueeze(-1) @ residual.unsqueeze(-2))
        outs.append((q[:, :, t, :].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, dim=2), S


def kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KDA reference.

    Args:
        q, k: ``[B, T, Hq/Hk, K]``; v: ``[B, T, Hv, V]``; g: ``[B, T, Hg, K]``
            (log-space per-channel decay); beta: ``[B, T, Hb]`` (scalar per
            token/head). Head counts must divide ``HO = max(Hq, Hv)``.
        scale: applied to q; defaults to ``1/sqrt(K)``.
        initial_state: ``[B, HO, K, V]`` (or ``[N, HO, K, V]`` with cu_seqlens).
        cu_seqlens: packed varlen boundaries (requires B == 1).

    Returns:
        ``(o, final_state)`` in fp64: o ``[B, T, HO, V]``, final_state
        ``[B, HO, K, V]`` (``[N, HO, K, V]`` with cu_seqlens).
    """
    K = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    HO = max(q.shape[2], v.shape[2])

    def expand(x):
        r = HO // x.shape[2]
        return x.repeat_interleave(r, dim=2) if r > 1 else x

    qf = expand(q.double() * scale)
    kf = expand(k.double())
    vf = expand(v.double())
    alphaf = expand(g.double().exp())  # [B, T, HO, K]
    betaf = expand(beta.double())

    # [B, T, HO, *] -> [B, HO, T, *]
    qf = qf.permute(0, 2, 1, 3)
    kf = kf.permute(0, 2, 1, 3)
    vf = vf.permute(0, 2, 1, 3)
    alphaf = alphaf.permute(0, 2, 1, 3)  # keeps the trailing K axis
    betaf = betaf.permute(0, 2, 1)

    V = v.shape[-1]
    HV = HO

    if cu_seqlens is None:
        B = q.shape[0]
        if initial_state is None:
            S0 = torch.zeros(B, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            S0 = initial_state.double()
        o, S = _recurrent_dense(qf, kf, vf, alphaf, betaf, S0)
        return o.permute(0, 2, 1, 3), S

    assert q.shape[0] == 1, "cu_seqlens requires packed batch B == 1"
    bounds = cu_seqlens.tolist()
    outs, states = [], []
    for n in range(len(bounds) - 1):
        s, e = bounds[n], bounds[n + 1]
        if initial_state is None:
            S0 = torch.zeros(1, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            S0 = initial_state[n : n + 1].double()
        if e == s:
            states.append(S0)
            continue
        o_n, S_n = _recurrent_dense(qf[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e], alphaf[:, :, s:e], betaf[:, :, s:e], S0)
        outs.append(o_n)
        states.append(S_n)
    if outs:
        o = torch.cat(outs, dim=2).permute(0, 2, 1, 3)
    else:
        o = qf.new_zeros(1, 0, HV, V)
    return o, torch.cat(states, dim=0)
