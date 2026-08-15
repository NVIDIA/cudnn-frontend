# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for ``fla.ops.kda.chunk_kda`` (Kimi Delta Attention).

Maps FLA's ``chunk_kda`` onto cuDNN's native ``kimi_delta_attention`` (Blackwell/
SM100) and falls back to the wrapped FLA function for anything cuDNN does not serve.

KDA uses a **channel-wise** log decay ``g: [B,T,H,K]`` and a **scalar** write
strength ``beta: [B,T,H]``. cuDNN's KDA kernel L2-normalizes q/k in-kernel (fwd+bwd),
so that stays fused; its beta-sigmoid and safe-gate transforms are forward-only, so
for a trainable shim the gate and beta activations are reproduced in torch (autograd
flows to the raw inputs and the ``A_log``/``dt_bias`` parameters):

* ``use_gate_in_kernel``  -> ``g = -exp(A_log) * softplus(g + dt_bias)`` (per-channel)
* ``safe_gate``           -> ``g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
* ``use_beta_sigmoid_in_kernel`` -> ``beta = sigmoid(beta)``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import cudnn
from cudnn.linear_attention.ops import kimi_delta_attention

_DECLINE = (cudnn.cudnnGraphNotSupportedError, NotImplementedError)

_LAST = {"path": None}


def last_path() -> str | None:
    return _LAST["path"]


class _Decline(Exception):
    pass


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
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    safe_gate,
    lower_bound,
    A_log,
    dt_bias,
):
    if q.dim() != 4:
        raise _Decline("expected [B, T, H, K]")
    if q.dtype == torch.float16:
        raise _Decline("cuDNN KDA is unstable in fp16 (NaN); bf16 only")
    B, T, H, K = q.shape

    if cu_seqlens is None:
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    else:
        if B != 1:
            raise _Decline("varlen requires B==1 (FLA contract)")
        cu = cu_seqlens.to(torch.int32)

    # gate: cuDNN's in-kernel gate is forward-only -> reproduce in torch (channel-wise).
    # A_log/dt_bias describe the gate over the H key/query heads (matching g's [B,T,H,K]),
    # not the value heads; a mismatched element count means we cannot adapt -> decline.
    g = g.float()
    if safe_gate or use_gate_in_kernel:
        if A_log is None or dt_bias is None:
            raise _Decline("gate transform requires A_log and dt_bias")
        if A_log.numel() != H or dt_bias.numel() != H * K:
            raise _Decline("A_log/dt_bias do not match [H] / [H, K]")
        a = A_log.float().view(1, 1, H, 1)
        b = dt_bias.float().reshape(H, K)
        if safe_gate:
            # FLA owns the safe-gate lower_bound default; don't guess it here.
            if lower_bound is None:
                raise _Decline("safe_gate without explicit lower_bound")
            g = lower_bound * torch.sigmoid(a.exp() * (g + b))
        else:
            g = -a.exp() * F.softplus(g + b)

    beta = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta.float()

    def thd(t):
        return t.reshape(-1, *t.shape[2:])

    h0 = None if initial_state is None else initial_state.float().contiguous()
    o, fs = kimi_delta_attention(
        thd(q),
        thd(k),
        thd(v),
        thd(g),
        thd(beta),
        cu,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,  # native, fwd+bwd
        use_beta_sigmoid_in_kernel=False,  # done above
        safe_gate=False,  # done above
    )
    o = o.reshape(B, T, *o.shape[1:])
    return o, (fs if output_final_state else None)


def make_chunk_kda(real_fn):
    """Wrap FLA's ``chunk_kda`` with a cuDNN fast path + FLA fallback."""

    def chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=False,
        lower_bound=None,
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        cp_context=None,
        **kwargs,
    ):
        A_log = kwargs.get("A_log")
        dt_bias = kwargs.get("dt_bias")

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
                use_gate_in_kernel=use_gate_in_kernel,
                use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                allow_neg_eigval=allow_neg_eigval,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                disable_recompute=disable_recompute,
                return_intermediate_states=return_intermediate_states,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                cp_context=cp_context,
                **kwargs,
            )

        if allow_neg_eigval or cp_context is not None or return_intermediate_states:
            return fallback("variant")
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
                use_gate_in_kernel,
                use_beta_sigmoid_in_kernel,
                safe_gate,
                lower_bound,
                A_log,
                dt_bias,
            )
        except (_Decline, *_DECLINE) as e:
            return fallback(type(e).__name__)
        _LAST["path"] = "native"
        return out

    chunk_kda.__wrapped__ = real_fn
    return chunk_kda
