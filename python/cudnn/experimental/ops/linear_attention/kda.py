"""
PyTorch custom operator for Kimi Delta Attention (KDA) linear attention.

KDA replaces GDN's scalar per-token decay with a per-channel decay
``diag(alpha_t)``, giving

    S_t = (I - beta_t k_t^T k_t) diag(alpha_t) S_{t-1} + beta_t k_t^T v_t,
    o_t = q_t S_t,

where ``k_t = phi(K_t)``, ``q_t = phi(Q_t)``. ``alpha_t`` is per-channel
in ``(0, 1]``; ``beta_t`` is a per-token scalar. The feature map ``phi``
is **not** modeled here.

Reference forward uses the chunked formulation in
``LinearAttentionRef/reference/linear_attention_kda.py``; backward runs
autograd through the per-token recurrence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch

from ._common import (
    bthd_to_bhtd,
    bhtd_to_bthd,
    bth_to_bht,
    bht_to_bth,
    chunk_factors_channelwise,
    compute_dtype_for,
    maybe_zero_state,
    pad_to_multiple,
)


# ---------------------------------------------------------------------------
# Reference math (fp32, ``[B, H, T, *]`` layout)
# ---------------------------------------------------------------------------


def _kda_forward_recurrent(q, k, v, alpha, beta, initial_state):
    """Per-token KDA recurrence in ``[B, H, T, *]`` layout. All fp32."""
    T = q.shape[2]
    S = initial_state
    out_steps = []
    for t in range(T):
        kt = k[:, :, t, :]                                           # (B, H, K)
        vt = v[:, :, t, :]
        at = alpha[:, :, t, :]                                       # (B, H, K) per-channel
        bt = beta[:, :, t][..., None, None]                          # (B, H, 1, 1)
        S = at[..., :, None] * S                                     # row-c scaled by at[c]
        kt_S = (kt.unsqueeze(-2) @ S).squeeze(-2)
        residual = vt - kt_S
        S = S + bt * (kt.unsqueeze(-1) @ residual.unsqueeze(-2))
        qt = q[:, :, t, :].unsqueeze(-2)
        out_steps.append((qt @ S).squeeze(-2))
    return torch.stack(out_steps, dim=2), S


def _kda_forward_chunked(q, k, v, alpha, beta, initial_state, chunk_size):
    """Chunked KDA forward in ``[B, H, T, *]`` layout. All fp32."""
    B, H, T_orig, K = q.shape
    V = v.shape[-1]
    Bsize = chunk_size

    q, pad = pad_to_multiple(q, Bsize, dim=2)
    k, _ = pad_to_multiple(k, Bsize, dim=2)
    v, _ = pad_to_multiple(v, Bsize, dim=2)
    alpha, _ = pad_to_multiple(alpha, Bsize, dim=2, value=1.0)
    beta, _ = pad_to_multiple(beta, Bsize, dim=2)
    T = q.shape[2]
    Tc = T // Bsize

    q_c = q.unflatten(2, (Tc, Bsize))                                # (B, H, Tc, Bs, K)
    k_c = k.unflatten(2, (Tc, Bsize))
    v_c = v.unflatten(2, (Tc, Bsize))
    alpha_c = alpha.unflatten(2, (Tc, Bsize))                        # (B, H, Tc, Bs, K)
    beta_c = beta.unflatten(2, (Tc, Bsize))                          # (B, H, Tc, Bs)

    Lambda, Gamma, g = chunk_factors_channelwise(alpha_c)

    I_B = torch.eye(Bsize, dtype=q.dtype, device=q.device)
    L_strict = torch.tril(torch.ones(Bsize, Bsize, dtype=q.dtype, device=q.device), diagonal=-1)
    M_causal = torch.tril(torch.ones(Bsize, Bsize, dtype=q.dtype, device=q.device), diagonal=0)

    S = initial_state
    Os = []
    for t in range(Tc):
        pQ = q_c[:, :, t]                                            # (B, H, Bs, K)
        pK = k_c[:, :, t]
        Vt = v_c[:, :, t]
        bt = beta_c[:, :, t]                                         # (B, H, Bs)
        Lambda_t = Lambda[:, :, t]                                   # (B, H, Bs, K)
        Gamma_t = Gamma[:, :, t]                                     # (B, H, Bs, K)
        g_t = g[:, :, t]                                             # (B, H, K)
        diag_b = torch.diag_embed(bt)

        Kp = pK * Gamma_t
        Km = pK / Gamma_t
        Qp = pQ * Gamma_t

        kernel = Kp @ Km.transpose(-2, -1)
        A_mat = I_B + L_strict * (diag_b @ kernel)
        T_mat = torch.linalg.solve_triangular(A_mat, diag_b, upper=False)

        Y_t = Vt - Kp @ S
        tilde_V = T_mat @ Y_t
        H_t = (Qp @ Km.transpose(-2, -1)) * M_causal
        Os.append(H_t @ tilde_V + Qp @ S)
        S = g_t[..., :, None] * S + (pK * Lambda_t).transpose(-2, -1) @ tilde_V

    out = torch.cat(Os, dim=2)[:, :, :T_orig, :]
    return out, S


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_OP_NAMESPACE = "cudnn"
_OP_NAME = "kimi_delta_attention"


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_fwd", mutates_args=())
def _kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    orig_dtype = q.dtype
    device = q.device
    cdt = compute_dtype_for(orig_dtype)

    q_f = bthd_to_bhtd(q).to(cdt) * scale
    k_f = bthd_to_bhtd(k).to(cdt)
    v_f = bthd_to_bhtd(v).to(cdt)
    g_f = bthd_to_bhtd(g).to(cdt)                                     # per-channel log decay
    alpha_f = torch.exp(g_f)
    beta_f = bth_to_bht(beta).to(cdt)

    S0 = (initial_state.to(cdt) if initial_state is not None
          else maybe_zero_state(None, B, H, K, V, cdt, device))

    o_bhtv, S_T = _kda_forward_chunked(q_f, k_f, v_f, alpha_f, beta_f, S0, chunk_size)
    o = bhtd_to_bthd(o_bhtv).to(orig_dtype)
    if output_final_state:
        final = S_T.to(orig_dtype)
    else:
        final = torch.empty(0, dtype=orig_dtype, device=device)
    return o, final


@_kda_fwd.register_fake
def _kda_fwd_fake(q, k, v, g, beta, scale, chunk_size, initial_state=None, output_final_state=False):
    B, T, H, K = q.shape
    V = v.shape[-1]
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    if output_final_state:
        final = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    else:
        final = torch.empty(0, dtype=q.dtype, device=q.device)
    return o, final


def _kda_setup_context(ctx, inputs, output):
    q, k, v, g, beta, scale, chunk_size, initial_state, output_final_state = inputs
    ctx.save_for_backward(q, k, v, g, beta)
    ctx.scale = scale
    ctx.initial_state = initial_state


def _kda_backward(ctx, dO, dFinal):
    """Backward via autograd through the per-token recurrent reference."""
    del dFinal
    q, k, v, g, beta = ctx.saved_tensors
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = ctx.scale

    cdt = compute_dtype_for(q.dtype)
    with torch.enable_grad():
        q_r = (bthd_to_bhtd(q).to(cdt) * scale).detach().requires_grad_(True)
        k_r = bthd_to_bhtd(k).to(cdt).detach().requires_grad_(True)
        v_r = bthd_to_bhtd(v).to(cdt).detach().requires_grad_(True)
        g_r = bthd_to_bhtd(g).to(cdt).detach().requires_grad_(True)
        beta_r = bth_to_bht(beta).to(cdt).detach().requires_grad_(True)
        S0 = (ctx.initial_state.to(cdt) if ctx.initial_state is not None
              else torch.zeros(B, H, K, V, dtype=cdt, device=q.device))

        alpha_r = torch.exp(g_r)
        out, _ = _kda_forward_recurrent(q_r, k_r, v_r, alpha_r, beta_r, S0)
        dO_f = bthd_to_bhtd(dO.contiguous()).to(cdt)
        out.backward(dO_f)

    dq = bhtd_to_bthd(q_r.grad).to(q.dtype) * scale
    dk = bhtd_to_bthd(k_r.grad).to(k.dtype)
    dv = bhtd_to_bthd(v_r.grad).to(v.dtype)
    dg = bhtd_to_bthd(g_r.grad).to(g.dtype)
    dbeta = bht_to_bth(beta_r.grad).to(beta.dtype)
    return dq, dk, dv, dg, dbeta, None, None, None, None


torch.library.register_autograd(
    f"{_OP_NAMESPACE}::{_OP_NAME}_fwd",
    _kda_backward,
    setup_context=_kda_setup_context,
)


def kimi_delta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Kimi Delta Attention (KDA) linear attention.

    Args:
        q: queries, ``[B, T, H, K]``.
        k: keys,    ``[B, T, H, K]``.
        v: values,  ``[B, T, H, V]``.
        g: log-space *per-channel* decay, ``[B, T, H, K]`` (``alpha = exp(g)``).
        beta: per-token write strength, ``[B, T, H]``.
        scale: attention scale applied to ``q`` before the recurrence. Defaults
            to ``1 / sqrt(K)``.
        initial_state: optional recurrent state ``[B, H, K, V]``.
        output_final_state: if ``True``, also return the post-last-token state.
        chunk_size: chunk length used by the reference forward.

    Returns:
        ``(o, final_state)`` — ``o`` is ``[B, T, H, V]``; ``final_state`` is
        ``[B, H, K, V]`` when ``output_final_state=True``, else empty.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.cudnn.kimi_delta_attention_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        float(scale),
        int(chunk_size),
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=bool(output_final_state),
    )
