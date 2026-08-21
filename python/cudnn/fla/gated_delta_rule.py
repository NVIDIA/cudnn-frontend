# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.

Maps the flash-linear-attention public signature onto cuDNN's native
``gated_delta_net`` (Blackwell/SM100) and falls back to the wrapped FLA function
for anything cuDNN does not serve, so results never change and never regress.

FLA layout is ``[B, T, H, ...]`` batch-first; ``g``/``beta`` are indexed by the
*value* heads ``HV`` and FLA's grouped-value attention has ``HV >= H``, so native
``HO = max(H, HV) = HV`` and the head mapping is a plain reshape + float cast.

FLA's ``GatedDeltaNet`` layer drives the kernel with fused knobs; the native op
now fuses each transform in-kernel (fwd+bwd), so the adapter forwards the raw
inputs and the fusion flags rather than reproducing the math in torch:

* ``use_gate_in_kernel`` -> ``safe_gate`` with ``a_log``/``dt_bias``; the kernel
  applies ``g = -exp(a_log) * softplus(g + dt_bias)`` (native ``safe_gate`` matches
  FLA's log decay exactly). FLA may omit ``dt_bias``; native requires it, so a
  zero bias is synthesized.
* ``use_beta_sigmoid_in_kernel`` -> forwarded; the kernel applies ``sigmoid(beta)``
  (raw beta stays io-dtype). ``allow_neg_eigval`` would scale by 2 and is not served.
* ``use_qk_l2norm_in_kernel`` -> forwarded; the kernel L2-normalizes q/k.
"""

from __future__ import annotations

import torch

import cudnn
from cudnn.linear_attention.ops import gated_delta_net

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

    # Fuse the gate in-kernel via safe_gate: pass raw g logits + a_log/dt_bias and
    # let the kernel compute -exp(a_log)*softplus(g+dt_bias). safe_gate requires
    # both; FLA may omit dt_bias, so synthesize a zero bias.
    a_log_t = dt_bias_t = None
    if use_gate_in_kernel:
        if A_log is None:
            raise _Decline("use_gate_in_kernel requires A_log")
        a_log_t = A_log.float().reshape(-1)
        dt_bias_t = dt_bias.float().reshape(-1) if dt_bias is not None else torch.zeros_like(a_log_t)
        if a_log_t.shape[0] != HO or dt_bias_t.shape[0] != HO:
            raise _Decline("a_log/dt_bias head count does not match HO=max(H,HV)")

    # g is always fp32 (raw logits under safe_gate, else FLA's precomputed log decay).
    # beta is io-dtype logits when the kernel applies the sigmoid, else fp32 post-activation.
    g = g.float()
    beta = beta.to(q.dtype) if use_beta_sigmoid_in_kernel else beta.float()

    def thd(t):
        # FLA's fused QKV short-conv returns one compact [B,T,Q+K+V]
        # allocation and splits it into strided q/k/v views.  The native GDN
        # kernels require compact THD inputs, so materialize only when needed;
        # contiguous() is a no-op for the usual already-compact inputs.
        return t.reshape(-1, *t.shape[2:]).contiguous()

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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=use_gate_in_kernel,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
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
        if not state_v_first and (initial_state is not None or output_final_state):
            return fallback("state_v_first=False")
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
