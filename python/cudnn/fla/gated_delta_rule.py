# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.

Maps the flash-linear-attention public signature onto cuDNN's native
``gated_delta_net`` (Blackwell/SM100) and falls back to the wrapped FLA function
for anything cuDNN does not serve, so results never change and never regress.

FLA layout is ``[B, T, H, ...]`` batch-first; ``g``/``beta`` are indexed by the
*value* heads ``HV`` and FLA's grouped-value attention has ``HV >= H``, so native
``HO = max(H, HV) = HV`` and the head mapping is a plain reshape. The recurrent
state is exchanged in FLA's layout: V-major ``[N, HV, V, K]`` (``state_v_first``)
passes straight through, the default K-major ``[N, HV, K, V]`` is transposed at the
boundary; the state rides fp32 and ``final_state`` comes back fp32, as in FLA.
Plans are pinned to the FROST engine so every forward cuDNN accepts also has a
backward (the cuTile engine's forward-only modes fall back to FLA).

FLA's ``GatedDeltaNet`` layer drives the kernel with fused knobs; the native op
fuses each transform in-kernel (fwd+bwd), so the adapter forwards the raw
inputs and the fusion flags rather than reproducing the math in torch:

* ``use_gate_in_kernel`` -> ``safe_gate`` with ``a_log``/``dt_bias``; the kernel
  applies ``g = -exp(a_log) * softplus(g + dt_bias)``. FLA may omit ``dt_bias``;
  it passes through as ``None`` (zero bias). FLA requires ``A_log``.
* ``use_beta_sigmoid_in_kernel`` -> forwarded; the kernel applies ``sigmoid(beta)``
  to raw logits in float32 or the io dtype (other dtypes are widened to float32).
* ``allow_neg_eigval`` -> forwarded; the kernel applies ``2 * sigmoid(beta)``.
* ``use_qk_l2norm_in_kernel`` -> forwarded; the kernel L2-normalizes q/k.

Declined (FLA runs): ``cp_context``, pre-Blackwell devices, and any graph the
FROST GDN engine does not serve (head dims outside {64, 128}, fp32 io).
"""

from __future__ import annotations

import torch

import cudnn
from cudnn.linear_attention.ops import gated_delta_net

from .state_layout import TransposeState

_DECLINE = (cudnn.cudnnGraphNotSupportedError, NotImplementedError)

_LAST = {"path": None}


def last_path() -> str | None:
    """The route the most recent shimmed call took ("native" | "fallback:<reason>"). For tests/telemetry only."""
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
    allow_neg_eigval,
    state_v_first,
    use_gate_in_kernel,
    A_log,
    dt_bias,
):
    if q.dim() != 4:
        raise _Decline("expected [B, T, H, K]")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise _Decline("allow_neg_eigval requires use_beta_sigmoid_in_kernel (FLA contract)")
    B, T, H, _ = q.shape
    HV = v.shape[2]
    HO = max(H, HV)

    if cu_seqlens is None:
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    else:
        if B != 1:
            raise _Decline("varlen requires B==1 (FLA contract)")
        cu = cu_seqlens.to(torch.int32)

    a_log_t = dt_bias_t = None
    if use_gate_in_kernel:
        if A_log is None:
            raise _Decline("use_gate_in_kernel requires A_log")
        a_log_t = A_log.float().reshape(-1)
        dt_bias_t = dt_bias.float().reshape(-1) if dt_bias is not None else None
        if a_log_t.shape[0] != HO or (dt_bias_t is not None and dt_bias_t.shape[0] != HO):
            raise _Decline("a_log/dt_bias head count does not match HO=max(H,HV)")

    if beta.dtype not in (q.dtype, torch.float32):
        beta = beta.float()
    if g.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        g = g.float()

    def thd(t):
        return t.reshape(-1, *t.shape[2:]).contiguous()

    g2, beta2 = thd(g), thd(beta)
    if g2.shape[-1] != HO or beta2.shape[-1] != HO:
        raise _Decline("g/beta head count does not match HO=max(H,HV)")

    h0 = initial_state
    if h0 is not None:
        h0 = h0.float()
        h0 = TransposeState.apply(h0) if not state_v_first else h0.contiguous()
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
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=use_gate_in_kernel,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        plan_name="gdn_frost",
    )
    o = o.reshape(B, T, *o.shape[1:])
    if not output_final_state:
        return o, None
    if not state_v_first:
        fs = TransposeState.apply(fs)
    return o, fs


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

        if cp_context is not None:
            return fallback("cp_context")
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
                allow_neg_eigval,
                state_v_first,
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
