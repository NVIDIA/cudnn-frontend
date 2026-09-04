# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for ``fla.ops.kda.chunk_kda`` (Kimi Delta Attention).

Maps FLA's ``chunk_kda`` onto cuDNN's native ``kimi_delta_attention`` (Blackwell/
SM100) and falls back to the wrapped FLA function for anything cuDNN does not serve.

KDA uses a **channel-wise** log decay ``g: [B,T,HV,K]`` and a **scalar** write
strength ``beta: [B,T,HV]``; under grouped-value attention (``HV > H``) the gates,
``A_log`` (``[HV]``) and ``dt_bias`` (``[HV*K]``) live at the value heads, which is
native ``HO = max(H, HV)``. The native op fuses the gate, beta-sigmoid and q/k
L2-norm transforms (fwd+bwd), so the adapter forwards the raw inputs and flags:

* ``use_gate_in_kernel`` with ``lower_bound`` -> native ``safe_gate`` with
  ``a_log``/``dt_bias``/``gate_lower_bound``; the kernel applies
  ``g = lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``. FLA may omit ``A_log``
  (unit amplitude) or ``dt_bias`` (zero bias); the absent parameter passes
  through as ``None`` and the native op applies the same defaults. FLA's
  ``safe_gate`` flag only selects its own kernel path and is not a transform.
* ``use_gate_in_kernel`` without ``lower_bound`` -> ``g = -exp(A_log) * softplus(g +
  dt_bias)`` reproduced in torch; the native KDA op has no fused param for it.
* ``use_beta_sigmoid_in_kernel`` -> forwarded; the kernel applies ``sigmoid(beta)``
  to raw logits in float32 or the io dtype (other dtypes are widened to float32).
* ``allow_neg_eigval`` -> forwarded; the kernel applies ``2 * sigmoid(beta)``.
* ``use_qk_l2norm_in_kernel`` -> forwarded (native, fwd+bwd).
* ``return_intermediate_states`` -> the native per-64-token ``state_checkpoints``
  series, sliced to the valid rows and shaped like FLA's ``h``.

The recurrent state is exchanged in FLA's layout: V-major ``[N, HV, V, K]``
(``state_v_first``) passes straight through, the default K-major ``[N, HV, K, V]`` is
transposed at the boundary; ``final_state`` comes back fp32, as in FLA. Plans are
pinned to the FROST engine so every forward cuDNN accepts also has a backward (the
cuTile engine's forward-only modes fall back to FLA).

Declined (FLA runs): fp32 io (the kernels are bf16/fp16), ``cp_context``, a fused
``lower_bound`` outside FLA's safe range ``[-5, 0)``, pre-Blackwell devices, and any
graph the FROST KDA engine does not serve (head dims outside {64, 128}).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import cudnn
from cudnn.linear_attention.ops import kimi_delta_attention

from .state_layout import TransposeState

_DECLINE = (cudnn.cudnnGraphNotSupportedError, NotImplementedError)

FLA_CHUNK = 64

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
    allow_neg_eigval,
    safe_gate,
    lower_bound,
    state_v_first,
    return_intermediate_states,
    cu_seqlens_cpu,
    A_log,
    dt_bias,
):
    if q.dim() != 4:
        raise _Decline("expected [B, T, H, K]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise _Decline(f"cuDNN KDA io must be bf16 or fp16, got {q.dtype}")
    if return_intermediate_states and not torch.is_inference_mode_enabled():
        raise _Decline("return_intermediate_states requires inference mode (FLA contract)")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise _Decline("allow_neg_eigval requires use_beta_sigmoid_in_kernel (FLA contract)")
    if safe_gate and use_gate_in_kernel and lower_bound is None:
        raise _Decline("safe_gate with use_gate_in_kernel requires lower_bound (FLA contract)")
    B, T, H, K = q.shape
    HO = max(H, v.shape[2])

    if cu_seqlens is None:
        cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
    else:
        if B != 1:
            raise _Decline("varlen requires B==1 (FLA contract)")
        cu = cu_seqlens.to(torch.int32)

    native_safe_gate = False
    a_log_t = dt_bias_t = gate_lb = None
    if use_gate_in_kernel:
        if A_log is not None and A_log.numel() != HO:
            raise _Decline("A_log does not match [HO]")
        if dt_bias is not None and dt_bias.numel() != HO * K:
            raise _Decline("dt_bias does not match [HO, K]")
        if lower_bound is not None:
            if not (-5.0 <= float(lower_bound) < 0.0):
                raise _Decline("lower_bound outside FLA's safe range [-5, 0)")
            native_safe_gate = True
            a_log_t = None if A_log is None else A_log.float().reshape(HO)
            dt_bias_t = None if dt_bias is None else dt_bias.float().reshape(HO, K)
            gate_lb = float(lower_bound)
        else:
            if A_log is None:
                raise _Decline("use_gate_in_kernel requires A_log or lower_bound (FLA contract)")
            g = g.float()
            if dt_bias is not None:
                g = g + dt_bias.float().reshape(HO, K)
            g = -A_log.float().view(1, 1, HO, 1).exp() * F.softplus(g)

    if beta.dtype not in (q.dtype, torch.float32):
        beta = beta.float()
    if g.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        g = g.float()

    def thd(t):
        return t.reshape(-1, *t.shape[2:]).contiguous()

    h0 = initial_state
    if h0 is not None:
        h0 = h0.float()
        h0 = TransposeState.apply(h0) if not state_v_first else h0.contiguous()
    out = kimi_delta_attention(
        thd(q),
        thd(k),
        thd(v),
        thd(g),
        thd(beta),
        cu,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=native_safe_gate,
        gate_lower_bound=gate_lb,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        checkpoint_every_n_tokens=FLA_CHUNK if return_intermediate_states else 0,
        plan_name="kda_frost",
    )
    o = out[0].reshape(B, T, *out[0].shape[1:])
    fs = out[1] if output_final_state else None
    if fs is not None and not state_v_first:
        fs = TransposeState.apply(fs)
    if not return_intermediate_states:
        return o, fs
    if cu_seqlens is None:
        rows, lead = B * ((T + FLA_CHUNK - 1) // FLA_CHUNK), B
    else:
        lens = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()).diff()
        rows, lead = int(((lens + FLA_CHUNK - 1) // FLA_CHUNK).sum()), 1
    h = out[2][:rows].reshape(lead, rows // lead, HO, *out[2].shape[2:])
    if not state_v_first:
        h = h.transpose(-1, -2).contiguous()
    return o, fs, h


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
                use_gate_in_kernel,
                use_beta_sigmoid_in_kernel,
                allow_neg_eigval,
                safe_gate,
                lower_bound,
                state_v_first,
                return_intermediate_states,
                cu_seqlens_cpu,
                A_log,
                dt_bias,
            )
        except (_Decline, *_DECLINE) as e:
            return fallback(type(e).__name__)
        _LAST["path"] = "native"
        return out

    chunk_kda.__wrapped__ = real_fn
    return chunk_kda
