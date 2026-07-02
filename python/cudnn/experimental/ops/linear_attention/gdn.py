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

Forward and backward dispatch to the vendored cuTile chunked kernel
(``_gdn_chunk_cutile``), which needs the ``cuda.tile`` runtime. The op is
registered through ``torch.library.custom_op`` so it composes with autograd,
``torch.compile``, and DDP.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch


def _get_cutile_fns():
    # Lazy import: the kernel module pulls in ``cuda.tile``.
    from ._gdn_chunk_cutile import (
        chunk_gated_delta_rule_fwd as _ct_fwd,
        chunk_gated_delta_rule_bwd as _ct_bwd,
    )
    return _ct_fwd, _ct_bwd


_OP_NAMESPACE = "cudnn"
_OP_NAME = "gated_delta_net"

_BT = 64  # cuTile kernel chunk size


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_fwd", mutates_args=())
def _gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDN forward.

    Returns ``(o, final_state, g_cumsum, A)``; ``g_cumsum`` and ``A`` are
    intermediates saved for the backward. ``final_state`` is a zero-size
    tensor when ``output_final_state`` is ``False``.
    """
    _ct_fwd, _ = _get_cutile_fns()
    g_cumsum, o, A, final_state_t, _, _ = _ct_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        state_v_first=False,
    )
    if not output_final_state:
        final_state_t = torch.empty(0, dtype=q.dtype, device=q.device)
    return o.to(q.dtype), final_state_t.to(q.dtype), g_cumsum, A


@_gdn_fwd.register_fake
def _gdn_fwd_fake(q, k, v, g, beta, scale, initial_state=None, output_final_state=False):
    B, T, H, K = q.shape
    V = v.shape[-1]
    HV = beta.shape[2]
    o = q.new_empty(B, T, H, V)
    final = q.new_empty(B, H, K, V) if output_final_state else q.new_empty(0)
    g_cumsum = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    A = torch.empty(B, T, HV, _BT, dtype=torch.float32, device=q.device)
    return o, final, g_cumsum, A


def _gdn_setup_context(ctx, inputs, output):
    q, k, v, g, beta, scale, initial_state, output_final_state = inputs
    o, final_state, g_cumsum, A = output
    # save_for_backward cannot hold None; keep initial_state as an attribute.
    ctx.save_for_backward(q, k, v, g, beta, g_cumsum, A)
    ctx.initial_state = initial_state
    ctx.scale = scale


def _gdn_backward(ctx, dO, dFinal, d_g_cumsum_unused, d_A_unused):
    q, k, v, g_raw, beta, g_cumsum, A = ctx.saved_tensors
    initial_state = ctx.initial_state

    _, _ct_bwd = _get_cutile_fns()
    dht = dFinal if (dFinal is not None and dFinal.numel() > 0) else None
    dq, dk, dk2, dv, db, dg, dh0, _dA_log, _ddt_bias = _ct_bwd(
        q=q,
        k=k,
        v=v,
        g=g_cumsum,  # already cumsum'd in fwd
        beta=beta,
        A=A,
        scale=ctx.scale,
        initial_state=initial_state,
        do=dO.contiguous(),
        dht=dht,
        state_v_first=False,
    )
    dk.add_(dk2)
    dh0_out = dh0 if initial_state is not None else None
    # q, k, v, g, beta, scale, initial_state, output_final_state
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dg.to(g_raw.dtype),
        db.to(beta.dtype),
        None,
        dh0_out,
        None,
    )


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
):
    """Gated DeltaNet (GDN) linear attention.

    Args:
        q: queries, ``[B, T, H, K]``.
        k: keys,    ``[B, T, H, K]``.
        v: values,  ``[B, T, H, V]``.
        g: log-space scalar decay per token, ``[B, T, H]``
           (``alpha = exp(g) in (0, 1]``).
        beta: per-token write strength, ``[B, T, H]``.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state ``[B, H, K, V]`` (otherwise zero).
        output_final_state: if ``True``, also return the state after the last token.

    Returns:
        ``(o, final_state)`` where ``o`` is ``[B, T, H, V]``. ``final_state`` is
        ``[B, H, K, V]`` when ``output_final_state=True``, otherwise empty.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    o, final_state, _g_cumsum, _A = torch.ops.cudnn.gated_delta_net_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        float(scale),
        initial_state=initial_state.contiguous() if initial_state is not None else None,
        output_final_state=bool(output_final_state),
    )
    return o, final_state
