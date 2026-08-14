# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.

Maps the flash-linear-attention public signature onto cuDNN's native
``gated_delta_net`` (Blackwell/SM100) and falls back to the wrapped FLA function
for anything cuDNN does not serve, so results never change and never regress.

FLA layout is ``[B, T, H, ...]`` batch-first; ``g``/``beta`` are indexed by the
*value* heads ``HV`` and FLA's grouped-value attention has ``HV >= H``, so native
``HO = max(H, HV) = HV`` and the head mapping is a plain reshape + float cast.

FLA's ``GatedDeltaNet`` layer drives the kernel with fused knobs; the adapter
reproduces each transform in torch (so autograd flows to the raw inputs and the
``A_log``/``dt_bias`` parameters) and hands the native op the values it expects:

* ``use_gate_in_kernel``  -> ``g = -exp(A_log) * softplus(g + dt_bias)`` (per-token
  log decay; native accumulates it, like FLA's naive reference).
* ``use_beta_sigmoid_in_kernel`` -> ``beta = sigmoid(beta)`` (post-sigmoid write
  strength; ``allow_neg_eigval`` would scale by 2 and is not yet served).
* ``use_qk_l2norm_in_kernel`` -> the shim L2-normalizes q/k and clears the flag,
  since the native in-kernel flag is not served for these shapes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import cudnn
from cudnn.linear_attention.ops import gated_delta_net

try:
    # FLA ships an efficient fused L2-norm kernel; torch F.normalize (fwd+bwd) is
    # ~2.6x slower and would erase the kernel win on the layer's fused path.
    from fla.modules.l2norm import l2norm as _fla_l2norm
except Exception:  # pragma: no cover
    _fla_l2norm = None


def _l2norm(x):
    if _fla_l2norm is not None:
        return _fla_l2norm(x)
    return F.normalize(x, p=2.0, dim=-1)


# Native declines a graph it cannot serve with one of these; treat as a fallback.
_DECLINE = (cudnn.cudnnGraphNotSupportedError, NotImplementedError)

# Diagnostic: which path the last shimmed call took ("native" | "fallback:<reason>").
_LAST = {"path": None}


def last_path() -> str | None:
    """The route the most recent shimmed call took. For tests/telemetry only."""
    return _LAST["path"]


class _Decline(Exception):
    """Raised internally when a call cannot be adapted to the native op."""


def _to_native(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    use_qk_l2norm_in_kernel,
    use_beta_sigmoid_in_kernel,
    use_gate_in_kernel,
    A_log,
    dt_bias,
):
    if q.dim() != 4:
        raise _Decline("expected [B, T, H, K]")
    B, T, H, _ = q.shape
    HV = v.shape[2]
    HO = max(H, HV)

    if cu_seqlens is None:
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    else:
        if B != 1:
            raise _Decline("varlen requires B==1 (FLA contract)")
        cu = cu_seqlens.to(torch.int32)

    if use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    if use_gate_in_kernel:
        if A_log is None:
            raise _Decline("use_gate_in_kernel requires A_log")
        gg = g.float()
        if dt_bias is not None:
            gg = gg + dt_bias.float()
        g = -A_log.float().exp() * F.softplus(gg)
    g = g.float()
    beta = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta.float()

    def thd(t):
        return t.reshape(-1, *t.shape[2:])

    g2, beta2 = thd(g), thd(beta)
    if g2.shape[-1] != HO or beta2.shape[-1] != HO:
        raise _Decline("g/beta head count does not match HO=max(H,HV)")

    h0 = None if initial_state is None else initial_state.float().contiguous()
    o, fs = gated_delta_net(
        thd(q),
        thd(k),
        thd(v),
        g2,
        beta2,
        cu,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,  # the shim already normalized q/k
    )
    o = o.reshape(B, T, *o.shape[1:])  # native o is shaped like v (THD) -> [B,T,HV,V]
    return o, (fs if output_final_state else None)


def make_chunk_gated_delta_rule(real_fn):
    """Wrap FLA's ``chunk_gated_delta_rule`` with a cuDNN fast path + FLA fallback."""

    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        cp_context=None,
        **kwargs,
    ):
        A_log = kwargs.get("A_log")
        dt_bias = kwargs.get("dt_bias")
        use_gate_in_kernel = kwargs.get("use_gate_in_kernel", False)

        def fallback(reason):
            _LAST["path"] = f"fallback:{reason}"
            return real_fn(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                allow_neg_eigval=allow_neg_eigval,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                cp_context=cp_context,
                **kwargs,
            )

        # Variants the native op does not model -> incumbent.
        if allow_neg_eigval or cp_context is not None:
            return fallback("variant")
        # state_v_first only changes the recurrent-state layout, so it is a no-op
        # for a stateless (training) call; decline only when a state is exchanged.
        if state_v_first and (initial_state is not None or output_final_state):
            return fallback("state_v_first")
        if not (q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 10):
            return fallback("pre-Blackwell")
        try:
            out = _to_native(
                q,
                k,
                v,
                g,
                beta,
                scale,
                initial_state,
                output_final_state,
                cu_seqlens,
                use_qk_l2norm_in_kernel,
                use_beta_sigmoid_in_kernel,
                use_gate_in_kernel,
                A_log,
                dt_bias,
            )
        except (_Decline, *_DECLINE) as e:
            return fallback(type(e).__name__)
        _LAST["path"] = "native"
        return out

    chunk_gated_delta_rule.__wrapped__ = real_fn
    return chunk_gated_delta_rule
