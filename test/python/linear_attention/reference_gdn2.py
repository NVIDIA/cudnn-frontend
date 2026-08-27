# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
fp64 recurrent reference for GDN-2 (Gated DeltaNet v2) linear attention.

GDN-2 generalizes GDN's scalar gates to three channel-wise gates: a per-key
decay ``alpha_t = exp(g_t) in (0, 1]^K``, a per-key erase gate ``beta_t in
R^K``, and a NEW per-value write gate ``w_t in R^V`` (``S_t`` is the
recurrent state; ``k``/``q`` already feature-mapped, ``q`` pre-scaled):

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
kept V-major ``[N, HO, V, K]`` here, matching the kernel ABI (VK, k
contiguous).
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


def recurrent_dense(q, k, v, alpha, beta, w, state0):
    """Dense recurrence in [B, HV, T, *] layout, fp64. Returns (o, final state).

    alpha, beta: per-key-channel [B, HV, T, K]; w: per-value-channel [B, HV, T, V]."""
    T = q.shape[2]
    state = state0
    outs = []
    for t in range(T):
        kt = k[:, :, t, :]  # [B, HV, K]
        vt = v[:, :, t, :]  # [B, HV, V]
        at = alpha[:, :, t, :]  # [B, HV, K]  (= exp(g_t))
        bt = beta[:, :, t, :]  # [B, HV, K]
        wt = w[:, :, t, :]  # [B, HV, V]
        state = at[..., None] * state  # per-K-channel decay first
        erase = ((bt * kt).unsqueeze(-2) @ state).squeeze(-2)  # (beta . k)^T on the decayed state: [B, HV, V]
        v_new = wt * vt - erase
        state = state + kt.unsqueeze(-1) @ v_new.unsqueeze(-2)  # k (x) v_new
        outs.append((q[:, :, t, :].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outs, dim=2), state


def beta_guard_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    io_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp64 mirror of the kernel beta guard (``frost/common/beta_guard.py``).

    ``k`` l2-normalized, ``alpha = exp(g)`` per token; all ``[B, T, HO, K]``.
    Returns ``(beta_eff, unsafe, fallback)``; the decision masks let tests
    assert the sensor actually fires and compare against other guard
    implementations."""
    weight = k * k
    n = weight.sum(-1)
    a = (beta * weight).sum(-1)
    nu = (beta * beta * weight).sum(-1)
    r2 = (n * nu - a * a).clamp_min(0.0)
    inv_c2 = alpha.amax(-1).pow(2)
    c2 = 1.0 / inv_c2
    r2_crit = ((c2 - 1.0) * (1.0 - (1.0 - a).pow(2) * inv_c2)).clamp_min(0.0)
    unsafe = (n > 1.0e-20) & (r2 > r2_crit)
    mu = a / n.clamp_min(1.0e-20)
    eta = ((1.0 - 1.0 / 32) * r2_crit / r2.clamp_min(1.0e-30)).sqrt().clamp(0.0, 1.0)
    projected = mu[..., None] + eta[..., None] * (beta - mu[..., None])
    candidate_q = torch.where(unsafe[..., None], projected, beta).to(io_dtype).double()
    a_q = (candidate_q * weight).sum(-1)
    nu_q = (candidate_q * candidate_q * weight).sum(-1)
    r2_q = (n * nu_q - a_q * a_q).clamp_min(0.0)
    r2_crit_q = ((c2 - 1.0) * (1.0 - (1.0 - a_q).pow(2) * inv_c2)).clamp_min(0.0)
    quant_tol = 4.0 * torch.finfo(io_dtype).eps * (n * nu_q + a_q * a_q)
    fallback = unsafe & (r2_q > r2_crit_q + quant_tol)
    mu_q = (a_q / n.clamp_min(1.0e-20)).to(io_dtype).double()
    beta_eff = torch.where(fallback[..., None], mu_q[..., None], candidate_q)
    return beta_eff, unsafe, fallback


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
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    use_beta_sigmoid: bool = False,
    beta_guard: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN-2 reference.

    Args:
        q, k: ``[B, T, Hq/Hk, K]``; v: ``[B, T, Hv, V]``; g, beta:
            ``[B, T, Hg/Hb, K]`` (g log-space per-channel decay, beta the
            per-key erase gate); w: ``[B, T, Hw, V]`` (per-value write gate).
            Head counts must divide ``HO = max(Hq, Hv)``.
        scale: applied to q; defaults to ``1/sqrt(K)``.
        initial_state: ``[B, HO, V, K]`` (or ``[N, HO, V, K]`` with cu_seqlens), V-major.
        cu_seqlens: packed varlen boundaries (requires B == 1).
        safe_gate: treat ``g`` as raw logits and use the log decay
            ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``
            (differentiable; a_log ``[Hg]``, dt_bias ``[Hg, K]``).
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        use_beta_sigmoid: treat ``beta`` as raw logits; apply ``sigmoid``.
        beta_guard: apply the erase-side beta safeguard (straight-through:
            the projection is detached, gradients flow as identity to
            ``beta`` and not at all to ``g``). Requires l2-normalized ``k``.

    Returns:
        ``(o, final_state)`` in fp64: o ``[B, T, HO, V]``, final_state
        ``[B, HO, V, K]`` (``[N, HO, V, K]`` with cu_seqlens), V-major.
    """
    K = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    HO = max(q.shape[2], v.shape[2])

    qf = q.double() * scale
    kf = k.double()
    vf = v.double()
    gf = g.double()
    if safe_gate:
        lb = -5.0 if gate_lower_bound is None else float(gate_lower_bound)
        gf = lb * torch.sigmoid(a_log.double().exp()[:, None] * (gf + dt_bias.double()))
    alphaf = gf.exp()  # [B, T, HO, K]
    betaf = beta.double()  # [B, T, HO, K]
    if use_beta_sigmoid:
        betaf = betaf.sigmoid()
    wf = w.double()  # [B, T, HO, V]
    # expand tensors for grouped heads (view, no copy), as in the sdpa references
    if q.shape[2] != HO:
        qf = qf.unsqueeze(3).expand(-1, -1, -1, HO // q.shape[2], -1).reshape(q.shape[0], q.shape[1], HO, -1)
    if k.shape[2] != HO:
        kf = kf.unsqueeze(3).expand(-1, -1, -1, HO // k.shape[2], -1).reshape(k.shape[0], k.shape[1], HO, -1)
    if v.shape[2] != HO:
        vf = vf.unsqueeze(3).expand(-1, -1, -1, HO // v.shape[2], -1).reshape(v.shape[0], v.shape[1], HO, -1)
    if g.shape[2] != HO:
        alphaf = alphaf.unsqueeze(3).expand(-1, -1, -1, HO // g.shape[2], -1).reshape(g.shape[0], g.shape[1], HO, -1)
    if beta.shape[2] != HO:
        betaf = betaf.unsqueeze(3).expand(-1, -1, -1, HO // beta.shape[2], -1).reshape(beta.shape[0], beta.shape[1], HO, -1)
    if w.shape[2] != HO:
        wf = wf.unsqueeze(3).expand(-1, -1, -1, HO // w.shape[2], -1).reshape(w.shape[0], w.shape[1], HO, -1)

    if beta_guard:
        beta_eff, _, _ = beta_guard_reference(kf.detach(), betaf.detach(), alphaf.detach(), beta.dtype)
        betaf = betaf + (beta_eff - betaf.detach())

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
            state0 = torch.zeros(B, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            state0 = initial_state.double().transpose(-1, -2).contiguous()
        o, state = recurrent_dense(qf, kf, vf, alphaf, betaf, wf, state0)
        return o.permute(0, 2, 1, 3), state.transpose(-1, -2).contiguous()

    assert q.shape[0] == 1, "cu_seqlens requires packed batch B == 1"
    bounds = cu_seqlens.tolist()
    outs, states = [], []
    for n in range(len(bounds) - 1):
        s, e = bounds[n], bounds[n + 1]
        if initial_state is None:
            state0 = torch.zeros(1, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            state0 = initial_state[n : n + 1].double().transpose(-1, -2).contiguous()
        if e == s:
            states.append(state0)
            continue
        o_n, state_n = recurrent_dense(qf[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e], alphaf[:, :, s:e], betaf[:, :, s:e], wf[:, :, s:e], state0)
        outs.append(o_n)
        states.append(state_n)
    if outs:
        o = torch.cat(outs, dim=2).permute(0, 2, 1, 3)
    else:
        o = qf.new_zeros(1, 0, HV, V)
    return o, torch.cat(states, dim=0).transpose(-1, -2).contiguous()
