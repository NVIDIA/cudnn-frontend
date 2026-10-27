# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
fp64 recurrent reference for the Mamba-2 SSD layer.

Per-token recurrence (``q`` = C, ``k`` = B, ``v`` = x, ``g`` = the raw dt; ``q`` pre-scaled):

    dt_t = clamp(softplus(g_t + dt_bias), dt_min, dt_max)
    S_t  = alpha_t S_{t-1} + dt_t k_t^T v_t,    alpha_t = exp(A dt_t)
    o_t  = q_t S_t + D v_t        (* z_t sigmoid(z_t))

the gated delta rule without the erase term: one per-token scalar ``dt`` is both the decay exponent and the
write strength. Supports grouped heads (every input's head count must divide ``HO = max(Hq, Hv)``; heads are
replicated onto the HO output heads), an optional initial state, and varlen packed batches via ``cu_seqlens``.

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


def recurrent_dense(q, k, v, alpha, dt, state0):
    """Dense recurrence in [B, HV, T, *] layout, fp64. Returns (o, final state)."""
    T = q.shape[2]
    state = state0
    outs = []
    for t in range(T):
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        at = alpha[:, :, t]
        dt_t = dt[:, :, t]
        state = at[..., None, None] * state + dt_t[..., None, None] * (kt.unsqueeze(-1) @ vt.unsqueeze(-2))
        outs.append((q[:, :, t, :].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outs, dim=2), state


def mamba2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    A: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mamba-2 reference.

    Args:
        q, k: ``[B, T, Hq/Hk, K]`` (C and B); v: ``[B, T, Hv, V]`` (x); g:
            ``[B, T, Hg]`` the raw dt. Head counts must divide ``HO = max(Hq, Hv)``.
        A: ``[HO]`` negative; dt_bias: ``[HO]``; D: ``[HO]`` or ``[HO, V]`` the skip
            weight; z: like ``v``, the output gate ``z sigmoid(z)``.
        scale: applied to q; defaults to ``1/sqrt(K)`` (the suite's convention; the
            op itself has none and the test adapter pre-scales C).
        initial_state: ``[B, HO, V, K]`` (or ``[N, HO, V, K]`` with cu_seqlens),
            V-major -- the kernel ABI.  Transposed on entry; the fp64
            recurrence below runs in the natural ``S[k, v]``.
        cu_seqlens: packed varlen boundaries (requires B == 1).
        dt_softplus, dt_min, dt_max: the dt transform ``clamp(softplus(g + dt_bias))``.

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
    dtf = g.double()
    if dt_bias is not None:
        dtf = dtf + dt_bias.double()
    if dt_softplus:
        dtf = torch.nn.functional.softplus(dtf)
    dtf = dtf.clamp(float(dt_min), float(dt_max))
    alphaf = (A.double() * dtf).exp()
    # expand tensors for grouped heads (view, no copy), as in the sdpa references
    if q.shape[2] != HO:
        qf = qf.unsqueeze(3).expand(-1, -1, -1, HO // q.shape[2], -1).reshape(q.shape[0], q.shape[1], HO, -1)
    if k.shape[2] != HO:
        kf = kf.unsqueeze(3).expand(-1, -1, -1, HO // k.shape[2], -1).reshape(k.shape[0], k.shape[1], HO, -1)
    if v.shape[2] != HO:
        vf = vf.unsqueeze(3).expand(-1, -1, -1, HO // v.shape[2], -1).reshape(v.shape[0], v.shape[1], HO, -1)
    if g.shape[2] != HO:
        alphaf = alphaf.unsqueeze(3).expand(-1, -1, -1, HO // g.shape[2]).reshape(g.shape[0], g.shape[1], HO)
        dtf = dtf.unsqueeze(3).expand(-1, -1, -1, HO // g.shape[2]).reshape(g.shape[0], g.shape[1], HO)
    HV = HO
    skip = vf

    # [B, T, HV, *] -> [B, HV, T, *]
    qf = qf.permute(0, 2, 1, 3)
    kf = kf.permute(0, 2, 1, 3)
    vf = vf.permute(0, 2, 1, 3)
    alphaf = alphaf.permute(0, 2, 1)
    dtf = dtf.permute(0, 2, 1)

    V = v.shape[-1]

    if cu_seqlens is None:
        B = q.shape[0]
        if initial_state is None:
            state0 = torch.zeros(B, HV, K, V, dtype=torch.float64, device=q.device)
        else:
            state0 = initial_state.double().transpose(-1, -2).contiguous()
        o, state = recurrent_dense(qf, kf, vf, alphaf, dtf, state0)
        o = o.permute(0, 2, 1, 3)
        final_state = state.transpose(-1, -2).contiguous()
    else:
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
            o_n, state_n = recurrent_dense(qf[:, :, s:e], kf[:, :, s:e], vf[:, :, s:e], alphaf[:, :, s:e], dtf[:, :, s:e], state0)
            outs.append(o_n)
            states.append(state_n)
        if outs:
            o = torch.cat(outs, dim=2).permute(0, 2, 1, 3)
        else:
            o = qf.new_zeros(1, 0, HV, V)
        final_state = torch.cat(states, dim=0).transpose(-1, -2).contiguous()

    if D is not None:
        if D.dim() == 1:
            o = o + D.double()[:, None] * skip
        else:
            o = o + D.double() * skip
    if z is not None:
        o = o * torch.nn.functional.silu(z.double())
    return o, final_state
