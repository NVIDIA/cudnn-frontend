# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
fp64 recurrent reference for GDN-2 (Gated DeltaNet v2) linear attention.

GDN-2 generalizes GDN's scalar gates to three channel-wise gates: a per-key
decay ``alpha_t = exp(g_t) in (0, 1]^K``, a per-key erase gate ``beta_t in
R^K``, and a NEW per-value write gate ``w_t in R^V`` (``k``/``q`` already
feature-mapped, ``q`` pre-scaled):

    S_t = (I - k_t (beta_t . k_t)^T) Diag(alpha_t) S_{t-1} + k_t (w_t . v_t)^T
    o_t = q_t S_t

Applied in order (this exact ordering is what the kernel implements):

    S_dec = Diag(alpha_t) S_{t-1}          # per-K-channel decay first
    erase = (beta_t . k_t) S_dec           # read of the already-decayed state
    v_new = w_t . v_t - erase
    S_t   = S_dec + k_t^T v_new            # rank-1 write with RAW k
    o_t   = q_t S_t

Collapsing ``beta_t = w_t`` to a scalar recovers KDA; collapsing ``g_t`` too
recovers GDN v1. Supports grouped heads (every input's head count must divide
``HO = max(Hq, Hv)``), an optional initial state, and varlen packed batches
via ``cu_seqlens``.

All math runs in fp64 on the input device and is differentiable, so it
doubles as the gradient oracle for the bprop tests. The recurrent state is
kept K-major ``[N, HO, K, V]`` here; the kernel keeps it V-major
``[N, HO, V, K]`` (transpose at the boundary).
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


def _recurrent_dense(q, k, v, alpha, beta, w, S0):
    """Dense recurrence in [B, HV, T, *] layout, fp64. Returns (o, S_T).

    alpha, beta: per-key-channel [B, HV, T, K]; w: per-value-channel [B, HV, T, V]."""
    T = q.shape[2]
    S = S0
    outs = []
    for t in range(T):
        kt = k[:, :, t, :]  # [B, HV, K]
        vt = v[:, :, t, :]  # [B, HV, V]
        at = alpha[:, :, t, :]  # [B, HV, K]  (= exp(g_t))
        bt = beta[:, :, t, :]  # [B, HV, K]
        wt = w[:, :, t, :]  # [B, HV, V]
        S = at[..., None] * S  # per-K-channel decay first
        erase = ((bt * kt).unsqueeze(-2) @ S).squeeze(-2)  # (beta . k)^T S_dec: [B, HV, V]
        v_new = wt * vt - erase
        S = S + kt.unsqueeze(-1) @ v_new.unsqueeze(-2)  # k (x) v_new
        outs.append((q[:, :, t, :].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, dim=2), S


def gdn2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN-2 reference.

    Args:
        q, k: ``[B, T, Hq/Hk, K]``; v: ``[B, T, Hv, V]``; g, beta:
            ``[B, T, Hg/Hb, K]`` (g log-space per-channel decay, beta the
            per-key erase gate); w: ``[B, T, Hw, V]`` (per-value write gate).
            Head counts must divide ``HO = max(Hq, Hv)``.
        scale: applied to q; defaults to ``1/sqrt(K)``.
        initial_state: ``[B, HO, K, V]`` (or ``[N, HO, K, V]`` with cu_seqlens),
            K-major.
        cu_seqlens: packed varlen boundaries (requires B == 1).

    Returns:
        ``(o, final_state)`` in fp64: o ``[B, T, HO, V]``, final_state
        ``[B, HO, K, V]`` (``[N, HO, K, V]`` with cu_seqlens), K-major.
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
    betaf = expand(beta.double())  # [B, T, HO, K]
    wf = expand(w.double())  # [B, T, HO, V]

    # [B, T, HO, *] -> [B, HO, T, *]
    qf = qf.permute(0, 2, 1, 3)
    kf = kf.permute(0, 2, 1, 3)
    vf = vf.permute(0, 2, 1, 3)
    alphaf = alphaf.permute(0, 2, 1, 3)
    betaf = betaf.permute(0, 2, 1, 3)
    wf = wf.permute(0, 2, 1, 3)

    V = v.shape[-1]
    HV = HO

    if cu_seqlens is None:
        B = q.shape[0]
        if initial_state is None:
            S0 = torch.zeros(B, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            S0 = initial_state.double()
        o, S = _recurrent_dense(qf, kf, vf, alphaf, betaf, wf, S0)
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
        o_n, S_n = _recurrent_dense(qf[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e], alphaf[:, :, s:e], betaf[:, :, s:e], wf[:, :, s:e], S0)
        outs.append(o_n)
        states.append(S_n)
    if outs:
        o = torch.cat(outs, dim=2).permute(0, 2, 1, 3)
    else:
        o = qf.new_zeros(1, 0, HV, V)
    return o, torch.cat(states, dim=0)
