# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
fp64 recurrent reference for GDN (Gated DeltaNet) linear attention.

Per-token recurrence (``k``/``q`` already feature-mapped, ``q`` pre-scaled):

    S_t = alpha_t (I - beta_t k_t^T k_t) S_{t-1} + beta_t k_t^T v_t
    o_t = q_t S_t

where ``S_t`` is the recurrent state, with scalar per-token decay
``alpha_t = exp(g_t)`` and write strength ``beta_t``. Supports grouped heads (every input's head count must divide
``HO = max(Hq, Hv)``; heads are replicated onto the HO output heads), an
optional initial state, and varlen packed batches via ``cu_seqlens``.

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


def recurrent_dense(q, k, v, alpha, beta, state0):
    """Dense recurrence in [B, HV, T, *] layout, fp64. Returns (o, final state)."""
    T = q.shape[2]
    state = state0
    outs = []
    for t in range(T):
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        at = alpha[:, :, t]
        bt = beta[:, :, t]
        kt_state = (kt.unsqueeze(-2) @ state).squeeze(-2)
        residual = vt - at[..., None] * kt_state
        state = at[..., None, None] * state + bt[..., None, None] * (kt.unsqueeze(-1) @ residual.unsqueeze(-2))
        outs.append((q[:, :, t, :].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outs, dim=2), state


def gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    safe_gate: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    use_beta_sigmoid: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN reference.

    Args:
        q, k: ``[B, T, Hq/Hk, K]``; v: ``[B, T, Hv, V]``; g, beta:
            ``[B, T, Hg/Hb]``. Head counts must divide ``HO = max(Hq, Hv)``.
        scale: applied to q; defaults to ``1/sqrt(K)``.
        initial_state: ``[B, HO, V, K]`` (or ``[N, HO, V, K]`` with cu_seqlens),
            V-major -- the kernel ABI.  Transposed on entry; the fp64
            recurrence below runs in the natural ``S[k, v]``.
        cu_seqlens: packed varlen boundaries (requires B == 1).
        safe_gate: treat ``g`` as raw logits and use the log decay
            ``-exp(a_log) * softplus(g + dt_bias)`` (differentiable; a_log /
            dt_bias are per-head ``[Hg]``).
        use_beta_sigmoid: treat ``beta`` as raw logits; apply ``sigmoid``.

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
        gf = -a_log.double().exp() * torch.nn.functional.softplus(gf + dt_bias.double())
    alphaf = gf.exp()
    betaf = beta.double()
    if use_beta_sigmoid:
        betaf = betaf.sigmoid()
    # expand tensors for grouped heads (view, no copy), as in the sdpa references
    if q.shape[2] != HO:
        qf = qf.unsqueeze(3).expand(-1, -1, -1, HO // q.shape[2], -1).reshape(q.shape[0], q.shape[1], HO, -1)
    if k.shape[2] != HO:
        kf = kf.unsqueeze(3).expand(-1, -1, -1, HO // k.shape[2], -1).reshape(k.shape[0], k.shape[1], HO, -1)
    if v.shape[2] != HO:
        vf = vf.unsqueeze(3).expand(-1, -1, -1, HO // v.shape[2], -1).reshape(v.shape[0], v.shape[1], HO, -1)
    if g.shape[2] != HO:
        alphaf = alphaf.unsqueeze(3).expand(-1, -1, -1, HO // g.shape[2]).reshape(g.shape[0], g.shape[1], HO)
    if beta.shape[2] != HO:
        betaf = betaf.unsqueeze(3).expand(-1, -1, -1, HO // beta.shape[2]).reshape(beta.shape[0], beta.shape[1], HO)
    HV = HO

    # [B, T, HV, *] -> [B, HV, T, *]
    qf = qf.permute(0, 2, 1, 3)
    kf = kf.permute(0, 2, 1, 3)
    vf = vf.permute(0, 2, 1, 3)
    alphaf = alphaf.permute(0, 2, 1)
    betaf = betaf.permute(0, 2, 1)

    V = v.shape[-1]

    if cu_seqlens is None:
        B = q.shape[0]
        if initial_state is None:
            state0 = torch.zeros(B, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            state0 = initial_state.double().transpose(-1, -2).contiguous()
        o, state = recurrent_dense(qf, kf, vf, alphaf, betaf, state0)
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
        o_n, state_n = recurrent_dense(qf[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e], alphaf[:, :, s:e], betaf[:, :, s:e], state0)
        outs.append(o_n)
        states.append(state_n)
    if outs:
        o = torch.cat(outs, dim=2).permute(0, 2, 1, 3)
    else:
        o = qf.new_zeros(1, 0, HV, V)
    return o, torch.cat(states, dim=0).transpose(-1, -2).contiguous()
