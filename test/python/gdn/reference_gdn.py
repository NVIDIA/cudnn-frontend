"""
fp64 recurrent reference for GDN (Gated DeltaNet) linear attention.

Per-token recurrence (``k``/``q`` already feature-mapped, ``q`` pre-scaled):

    S_t = alpha_t (I - beta_t k_t^T k_t) S_{t-1} + beta_t k_t^T v_t
    o_t = q_t S_t

with scalar per-token decay ``alpha_t = exp(g_t)`` and write strength
``beta_t``. Supports GVA (HV > H: q/k heads shared across HV/H groups),
an optional initial state, and varlen packed batches via ``cu_seqlens``.

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
    """Dense recurrence in [B, HV, T, *] layout, fp64. Returns (o, S_T)."""
    T = q.shape[2]
    S = S0
    outs = []
    for t in range(T):
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        at = alpha[:, :, t]
        bt = beta[:, :, t]
        kt_S = (kt.unsqueeze(-2) @ S).squeeze(-2)
        residual = vt - at[..., None] * kt_S
        S = at[..., None, None] * S + bt[..., None, None] * (kt.unsqueeze(-1) @ residual.unsqueeze(-2))
        outs.append((q[:, :, t, :].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(outs, dim=2), S


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN reference.

    Args:
        q, k: ``[B, T, H, K]``; v: ``[B, T, HV, V]``; g, beta: ``[B, T, HV]``.
        scale: applied to q; defaults to ``1/sqrt(K)``.
        initial_state: ``[B, HV, K, V]`` (or ``[N, HV, K, V]`` with cu_seqlens).
        cu_seqlens: packed varlen boundaries (requires B == 1).

    Returns:
        ``(o, final_state)`` in fp64: o ``[B, T, HV, V]``, final_state
        ``[B, HV, K, V]`` (``[N, HV, K, V]`` with cu_seqlens).
    """
    K = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    H, HV = q.shape[2], v.shape[2]
    qf = q.double() * scale
    kf = k.double()
    if HV != H:
        qf = qf.repeat_interleave(HV // H, dim=2)
        kf = kf.repeat_interleave(HV // H, dim=2)
    vf = v.double()
    alphaf = g.double().exp()
    betaf = beta.double()

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
        o_n, S_n = _recurrent_dense(
            qf[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e], alphaf[:, :, s:e], betaf[:, :, s:e], S0
        )
        outs.append(o_n)
        states.append(S_n)
    o = torch.cat(outs, dim=2).permute(0, 2, 1, 3)
    return o, torch.cat(states, dim=0)
