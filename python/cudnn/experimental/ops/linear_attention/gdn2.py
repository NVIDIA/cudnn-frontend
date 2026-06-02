"""
PyTorch custom operator for Gated DeltaNet-2 (GDN-2) linear attention.

GDN-2 (NVlabs, https://github.com/NVlabs/GatedDeltaNet-2) decouples the
erase and write operations of the delta rule by replacing GDN's scalar
``beta`` with two independent channel-wise gates and adds a per-channel
decay. The per-token recurrence is

    S_t = (I - k_t (b_t \\odot k_t)^T) diag(d_t) S_{t-1}
          + k_t (w_t \\odot v_t)^T,
    o_t = q_t S_t,

where for each token ``t``:
- ``q_t, k_t \\in R^K``    are query/key features (no internal phi),
- ``v_t   \\in R^V``       is the value,
- ``b_t   \\in R^K``       is the channel-wise *erase* gate (key axis),
- ``w_t   \\in R^V``       is the channel-wise *write* gate (value axis),
- ``d_t   \\in R^K``       is the per-channel decay (``d_t = exp(g_t)``).

This generalises KDA: setting ``b_t = beta_t * 1_K`` and ``w_t = 1_V``
recovers the KDA recurrence; setting in addition ``d_t = alpha_t * 1_K``
recovers GDN.

Reference forward runs the per-token recurrence in fp32; backward runs
autograd through that same recurrence. A chunked / fused kernel can
replace the forward op body without changing the public API.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch

from ._common import (
    bthd_to_bhtd,
    bhtd_to_bthd,
    compute_dtype_for,
    maybe_zero_state,
)


# ---------------------------------------------------------------------------
# Reference math (fp32, ``[B, H, T, *]`` layout)
# ---------------------------------------------------------------------------


def _gdn2_forward_recurrent(q, k, v, alpha, b_gate, w_gate, initial_state):
    """Per-token GDN-2 recurrence in ``[B, H, T, *]`` layout. All fp32.

    Args:
        q, k:   ``[B, H, T, K]``
        v:      ``[B, H, T, V]``
        alpha:  ``[B, H, T, K]``  per-channel decay (= ``exp(g)``)
        b_gate: ``[B, H, T, K]``  erase gate
        w_gate: ``[B, H, T, V]``  write gate
        initial_state: ``[B, H, K, V]``
    """
    T = q.shape[2]
    S = initial_state
    out_steps = []
    for t in range(T):
        kt = k[:, :, t, :]                                # (B, H, K)
        vt = v[:, :, t, :]                                # (B, H, V)
        at = alpha[:, :, t, :]                            # (B, H, K)
        bt = b_gate[:, :, t, :]                           # (B, H, K)
        wt = w_gate[:, :, t, :]                           # (B, H, V)
        S = at[..., :, None] * S                          # per-channel decay on rows
        bk = bt * kt                                      # (B, H, K) = b_t \odot k_t
        bk_S = (bk.unsqueeze(-2) @ S).squeeze(-2)         # (B, H, V) = (b_t \odot k_t)^T S
        S = S - kt.unsqueeze(-1) @ bk_S.unsqueeze(-2)     # erase
        wv = wt * vt                                      # (B, H, V) = w_t \odot v_t
        S = S + kt.unsqueeze(-1) @ wv.unsqueeze(-2)       # write
        qt = q[:, :, t, :].unsqueeze(-2)                  # (B, H, 1, K)
        out_steps.append((qt @ S).squeeze(-2))
    return torch.stack(out_steps, dim=2), S


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_OP_NAMESPACE = "cudnn"
_OP_NAME = "gated_delta_net_v2"


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_fwd", mutates_args=())
def _gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float,
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN-2 forward (recurrent reference in fp32).

    ``chunk_size`` is accepted for API symmetry with the GDN/KDA ops; the
    current reference body runs the per-token recurrence directly.
    """
    del chunk_size
    B, T, H, K = q.shape
    V = v.shape[-1]
    orig_dtype = q.dtype
    device = q.device
    cdt = compute_dtype_for(orig_dtype)

    q_f = bthd_to_bhtd(q).to(cdt) * scale
    k_f = bthd_to_bhtd(k).to(cdt)
    v_f = bthd_to_bhtd(v).to(cdt)
    g_f = bthd_to_bhtd(g).to(cdt)
    alpha_f = torch.exp(g_f)
    b_f = bthd_to_bhtd(b).to(cdt)
    w_f = bthd_to_bhtd(w).to(cdt)

    S0 = (initial_state.to(cdt) if initial_state is not None
          else maybe_zero_state(None, B, H, K, V, cdt, device))

    o_bhtv, S_T = _gdn2_forward_recurrent(q_f, k_f, v_f, alpha_f, b_f, w_f, S0)
    o = bhtd_to_bthd(o_bhtv).to(orig_dtype)
    if output_final_state:
        final = S_T.to(orig_dtype)
    else:
        final = torch.empty(0, dtype=orig_dtype, device=device)
    return o, final


@_gdn2_fwd.register_fake
def _gdn2_fwd_fake(q, k, v, g, b, w, scale, chunk_size, initial_state=None, output_final_state=False):
    B, T, H, K = q.shape
    V = v.shape[-1]
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    if output_final_state:
        final = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    else:
        final = torch.empty(0, dtype=q.dtype, device=q.device)
    return o, final


def _gdn2_setup_context(ctx, inputs, output):
    q, k, v, g, b, w, scale, chunk_size, initial_state, output_final_state = inputs
    ctx.save_for_backward(q, k, v, g, b, w)
    ctx.scale = scale
    ctx.initial_state = initial_state


def _gdn2_backward(ctx, dO, dFinal):
    """Backward via autograd through the per-token recurrent reference."""
    del dFinal
    q, k, v, g, b, w = ctx.saved_tensors
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = ctx.scale

    cdt = compute_dtype_for(q.dtype)
    with torch.enable_grad():
        q_r = (bthd_to_bhtd(q).to(cdt) * scale).detach().requires_grad_(True)
        k_r = bthd_to_bhtd(k).to(cdt).detach().requires_grad_(True)
        v_r = bthd_to_bhtd(v).to(cdt).detach().requires_grad_(True)
        g_r = bthd_to_bhtd(g).to(cdt).detach().requires_grad_(True)
        b_r = bthd_to_bhtd(b).to(cdt).detach().requires_grad_(True)
        w_r = bthd_to_bhtd(w).to(cdt).detach().requires_grad_(True)
        S0 = (ctx.initial_state.to(cdt) if ctx.initial_state is not None
              else torch.zeros(B, H, K, V, dtype=cdt, device=q.device))

        alpha_r = torch.exp(g_r)
        out, _ = _gdn2_forward_recurrent(q_r, k_r, v_r, alpha_r, b_r, w_r, S0)
        dO_f = bthd_to_bhtd(dO.contiguous()).to(cdt)
        out.backward(dO_f)

    dq = bhtd_to_bthd(q_r.grad).to(q.dtype) * scale
    dk = bhtd_to_bthd(k_r.grad).to(k.dtype)
    dv = bhtd_to_bthd(v_r.grad).to(v.dtype)
    dg = bhtd_to_bthd(g_r.grad).to(g.dtype)
    db = bhtd_to_bthd(b_r.grad).to(b.dtype)
    dw = bhtd_to_bthd(w_r.grad).to(w.dtype)
    return dq, dk, dv, dg, db, dw, None, None, None, None


torch.library.register_autograd(
    f"{_OP_NAMESPACE}::{_OP_NAME}_fwd",
    _gdn2_backward,
    setup_context=_gdn2_setup_context,
)


def gated_delta_net_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Gated DeltaNet-2 (GDN-2) linear attention.

    Args:
        q: queries, ``[B, T, H, K]``.
        k: keys,    ``[B, T, H, K]``.
        v: values,  ``[B, T, H, V]``.
        g: log-space per-channel decay, ``[B, T, H, K]`` (``alpha = exp(g)``).
        b: channel-wise *erase* gate (key axis), ``[B, T, H, K]``. Typically
            produced by a sigmoid projection.
        w: channel-wise *write* gate (value axis), ``[B, T, H, V]``. Typically
            produced by a sigmoid projection.
        scale: attention scale applied to ``q`` before the recurrence. Defaults
            to ``1 / sqrt(K)``.
        initial_state: optional recurrent state ``[B, H, K, V]``.
        output_final_state: if ``True``, also return the post-last-token state.
        chunk_size: accepted for API symmetry with GDN/KDA. The current
            reference body runs the per-token recurrence directly.

    Returns:
        ``(o, final_state)`` — ``o`` is ``[B, T, H, V]``; ``final_state`` is
        ``[B, H, K, V]`` when ``output_final_state=True``, else empty.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.cudnn.gated_delta_net_v2_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        b.contiguous(),
        w.contiguous(),
        float(scale),
        int(chunk_size),
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=bool(output_final_state),
    )
