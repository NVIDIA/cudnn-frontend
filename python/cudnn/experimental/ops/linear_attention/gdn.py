"""
PyTorch custom operator for Gated DeltaNet (GDN) linear attention.

GDN combines a DeltaNet beta-gated update with a scalar Mamba-2-style decay,
yielding the per-token recurrence

    S_t = alpha_t (I - beta_t k_t^T k_t) S_{t-1} + beta_t k_t^T v_t,
    o_t = q_t S_t,

where ``k_t = phi(K_t)`` and ``q_t = phi(Q_t)``. ``alpha_t`` is a scalar
per-token decay in ``(0, 1]``; ``beta_t`` is a scalar per-token write
strength. The feature map ``phi`` is **not** modeled here — ``q`` and ``k``
are taken as already-mapped.

This is the reference implementation: forward runs the chunked formulation
from ``LinearAttentionRef/reference/linear_attention_gdn.py``; backward
runs autograd through the per-token recurrence. The op is registered
through ``torch.library.custom_op`` so it composes with autograd,
``torch.compile``, and DDP.
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
    chunk_factors_scalar,
    compute_dtype_for,
    maybe_zero_state,
    pad_to_multiple,
)


# ---------------------------------------------------------------------------
# Reference math (fp32, ``[B, H, T, *]`` layout)
# ---------------------------------------------------------------------------


def _gdn_forward_recurrent(q, k, v, alpha, beta, initial_state):
    """Per-token GDN recurrence in ``[B, H, T, *]`` layout. All fp32."""
    T = q.shape[2]
    S = initial_state
    out_steps = []
    for t in range(T):
        kt = k[:, :, t, :]                            # (B, H, K)
        vt = v[:, :, t, :]                            # (B, H, V)
        at = alpha[:, :, t][..., None, None]          # (B, H, 1, 1)
        at_s = alpha[:, :, t][..., None]              # (B, H, 1)
        bt = beta[:, :, t][..., None, None]           # (B, H, 1, 1)
        kt_S = (kt.unsqueeze(-2) @ S).squeeze(-2)     # (B, H, V)
        residual = vt - at_s * kt_S
        S = at * S + bt * (kt.unsqueeze(-1) @ residual.unsqueeze(-2))
        qt = q[:, :, t, :].unsqueeze(-2)              # (B, H, 1, K)
        out_steps.append((qt @ S).squeeze(-2))        # (B, H, V)
    return torch.stack(out_steps, dim=2), S


def _gdn_forward_chunked(q, k, v, alpha, beta, initial_state, chunk_size):
    """Chunked GDN forward in ``[B, H, T, *]`` layout. All fp32."""
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

    q_c = q.unflatten(2, (Tc, Bsize))                                       # (B, H, Tc, Bs, K)
    k_c = k.unflatten(2, (Tc, Bsize))
    v_c = v.unflatten(2, (Tc, Bsize))
    alpha_c = alpha.unflatten(2, (Tc, Bsize))                                # (B, H, Tc, Bs)
    beta_c = beta.unflatten(2, (Tc, Bsize))

    Lambda, Gamma, L_mat, g = chunk_factors_scalar(alpha_c)

    I_B = torch.eye(Bsize, dtype=q.dtype, device=q.device)
    L_strict = torch.tril(torch.ones(Bsize, Bsize, dtype=q.dtype, device=q.device), diagonal=-1)

    S = initial_state
    Os = []
    for t in range(Tc):
        pQ = q_c[:, :, t]                                                    # (B, H, Bs, K)
        pK = k_c[:, :, t]
        Vt = v_c[:, :, t]
        bt = beta_c[:, :, t]                                                 # (B, H, Bs)
        Lambda_t = Lambda[:, :, t].unsqueeze(-1)                             # (B, H, Bs, 1)
        Gamma_t = Gamma[:, :, t].unsqueeze(-1)
        L_t = L_mat[:, :, t]                                                 # (B, H, Bs, Bs)
        g_t = g[:, :, t][..., None, None]                                    # (B, H, 1, 1)
        diag_b = torch.diag_embed(bt)                                        # (B, H, Bs, Bs)

        kkT = pK @ pK.transpose(-2, -1)
        A_mat = I_B + L_strict * (diag_b @ (L_t * kkT))
        T_mat = torch.linalg.solve_triangular(A_mat, diag_b, upper=False)

        Y_t = Vt - (pK * Gamma_t) @ S
        tilde_V = T_mat @ Y_t
        H_t = (pQ @ pK.transpose(-2, -1)) * L_t
        Os.append(H_t @ tilde_V + (pQ * Gamma_t) @ S)
        S = g_t * S + (pK.transpose(-2, -1) * Lambda_t.transpose(-2, -1)) @ tilde_V

    out = torch.cat(Os, dim=2)[:, :, :T_orig, :]
    return out, S


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_OP_NAMESPACE = "cudnn"
_OP_NAME = "gated_delta_net"


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_fwd", mutates_args=())
def _gdn_fwd(
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
    """GDN forward (reference / chunked, fp32 inside).

    Returns ``(o, final_state)``; ``final_state`` is a zero-size tensor when
    ``output_final_state`` is ``False`` so the op has a static schema.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    orig_dtype = q.dtype
    device = q.device
    cdt = compute_dtype_for(orig_dtype)

    q_f = bthd_to_bhtd(q).to(cdt) * scale                                    # absorb scale into q
    k_f = bthd_to_bhtd(k).to(cdt)
    v_f = bthd_to_bhtd(v).to(cdt)
    g_f = bth_to_bht(g).to(cdt)                                              # log-decay
    alpha_f = torch.exp(g_f)
    beta_f = bth_to_bht(beta).to(cdt)

    S0 = (initial_state.to(cdt) if initial_state is not None
          else maybe_zero_state(None, B, H, K, V, cdt, device))

    o_bhtv, S_T = _gdn_forward_chunked(q_f, k_f, v_f, alpha_f, beta_f, S0, chunk_size)
    o = bhtd_to_bthd(o_bhtv).to(orig_dtype)
    if output_final_state:
        final = S_T.to(orig_dtype)
    else:
        final = torch.empty(0, dtype=orig_dtype, device=device)
    return o, final


@_gdn_fwd.register_fake
def _gdn_fwd_fake(q, k, v, g, beta, scale, chunk_size, initial_state=None, output_final_state=False):
    B, T, H, K = q.shape
    V = v.shape[-1]
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    if output_final_state:
        final = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)
    else:
        final = torch.empty(0, dtype=q.dtype, device=q.device)
    return o, final


def _gdn_setup_context(ctx, inputs, output):
    q, k, v, g, beta, scale, chunk_size, initial_state, output_final_state = inputs
    ctx.save_for_backward(q, k, v, g, beta)
    ctx.scale = scale
    ctx.initial_state = initial_state


def _gdn_backward(ctx, dO, dFinal):
    """Backward via autograd through the per-token recurrent reference.

    The body runs as a regular Python function (no custom-op dispatcher
    indirection), so autograd is live and ``out.backward(dO)`` traces
    correctly. A fused chunked backward can replace this without touching
    the public API.
    """
    del dFinal  # we do not propagate grad through the final-state output
    q, k, v, g, beta = ctx.saved_tensors
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = ctx.scale

    cdt = compute_dtype_for(q.dtype)
    with torch.enable_grad():
        q_r = (bthd_to_bhtd(q).to(cdt) * scale).detach().requires_grad_(True)
        k_r = bthd_to_bhtd(k).to(cdt).detach().requires_grad_(True)
        v_r = bthd_to_bhtd(v).to(cdt).detach().requires_grad_(True)
        g_r = bth_to_bht(g).to(cdt).detach().requires_grad_(True)
        beta_r = bth_to_bht(beta).to(cdt).detach().requires_grad_(True)
        S0 = (ctx.initial_state.to(cdt) if ctx.initial_state is not None
              else torch.zeros(B, H, K, V, dtype=cdt, device=q.device))

        alpha_r = torch.exp(g_r)
        out, _ = _gdn_forward_recurrent(q_r, k_r, v_r, alpha_r, beta_r, S0)
        dO_f = bthd_to_bhtd(dO.contiguous()).to(cdt)
        out.backward(dO_f)

    dq = bhtd_to_bthd(q_r.grad).to(q.dtype) * scale
    dk = bhtd_to_bthd(k_r.grad).to(k.dtype)
    dv = bhtd_to_bthd(v_r.grad).to(v.dtype)
    dg = bht_to_bth(g_r.grad).to(g.dtype)
    dbeta = bht_to_bth(beta_r.grad).to(beta.dtype)
    # Match the fwd signature: q, k, v, g, beta, scale, chunk_size, initial_state, output_final_state
    return dq, dk, dv, dg, dbeta, None, None, None, None


torch.library.register_autograd(
    f"{_OP_NAMESPACE}::{_OP_NAME}_fwd",
    _gdn_backward,
    setup_context=_gdn_setup_context,
)


def gated_delta_net(
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
    """Gated DeltaNet (GDN) linear attention.

    Args:
        q: queries, ``[B, T, H, K]``.
        k: keys,    ``[B, T, H, K]``.
        v: values,  ``[B, T, H, V]``.
        g: log-space scalar decay, ``[B, T, H]`` (so ``alpha = exp(g) in (0, 1]``).
        beta: per-token write strength, ``[B, T, H]``.
        scale: attention scale applied to ``q`` before the recurrence. Defaults
            to ``1 / sqrt(K)``.
        initial_state: optional recurrent state ``[B, H, K, V]`` (otherwise zero).
        output_final_state: if ``True``, return the state after the last token
            in addition to the per-token output.
        chunk_size: chunk length used by the reference forward.

    Returns:
        ``(o, final_state)`` where ``o`` is ``[B, T, H, V]``. ``final_state`` is
        ``[B, H, K, V]`` when ``output_final_state=True``, otherwise an
        empty tensor.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.cudnn.gated_delta_net_fwd(
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
