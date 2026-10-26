# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for ``fla.ops.kda.chunk_kda`` (Kimi Delta Attention).

Maps FLA's ``chunk_kda`` onto cuDNN's native ``kimi_delta_attention`` (Blackwell
SM100+ and Hopper SM90) and falls back to the wrapped FLA function for anything
cuDNN does not serve.

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
pinned per architecture: the FROST engine on SM100+, where every forward it accepts
also has a backward (the cuTile engine's forward-only modes fall back to FLA), and
the Hopper CUDA engine on SM90. The Hopper backward serves fewer configurations than
its forward (no fused q/k L2-norm, beta sigmoid or safe gate, no grouped value heads,
default scale only), so when gradients are required the shim checks the backward
graph against the pinned plan before running the forward, and the whole call falls
back to FLA if the backward would decline. The Hopper engine parallelizes over
(sequence, value head) pairs, so a call with fewer pairs than two thirds of the
SMs also falls back: there FLA's chunk-parallel kernel is faster.

Declined (FLA runs): fp32 io (the kernels are bf16/fp16), ``cp_context``, a fused
``lower_bound`` outside FLA's safe range ``[-5, 0)``, devices other than SM90 and
SM100+, and any graph the pinned engine does not serve -- on SM100+ head dims
outside {64, 128}; on SM90 anything but bf16 io with head dim 128,
``allow_neg_eigval``, ``return_intermediate_states`` (the Hopper kernel produces no
state series), a safe gate missing ``A_log`` or ``dt_bias``, and fewer
(sequence, value head) pairs than two thirds of the SMs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import cudnn
from cudnn.linear_attention.ops import kimi_delta_attention
from cudnn.linear_attention.ops.common import select_plan, torch_dtype_to_cudnn
from cudnn.linear_attention.ops.kda import build_bprop_graph

from .state_layout import TransposeState

_DECLINE = (cudnn.cudnnGraphNotSupportedError, NotImplementedError)

FLA_CHUNK = 64

_LAST = {"path": None}


def last_path() -> str | None:
    return _LAST["path"]


def _plan_for(device) -> str | None:
    """The KDA plan pinned on ``device``, or ``None`` where cuDNN has no KDA engine."""
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 10:
        return "kda_frost"
    if (major, minor) == (9, 0):
        return "kda_hopper_cuda"
    return None


# Plans that parallelize over (sequence, value head) pairs. Up to about one pair per SM
# kda_hopper_cuda's time stays flat (its chunk scan is latency-bound) while FLA's
# chunk-parallel kernel grows with the work, so FLA wins on few pairs. Measured on GH200
# (132 SMs, K=V=128, bf16, fused transforms, 8K tokens per sequence, H = 16/32/64): FLA
# time / cuDNN time was 0.78-0.88 at 48-64 pairs, 0.96 at 80, 1.09-1.15 at 96 and
# 1.34-1.72 at 128-160, independent of H. Below two thirds of the SMs the call goes to FLA.

_PAIR_PARALLEL = ("kda_hopper_cuda",)


def _underfilled(plan_name, q, v, cu_seqlens) -> bool:
    """Whether ``plan_name`` would leave SMs idle on this call."""
    if plan_name not in _PAIR_PARALLEL or q.dim() != 4 or v.dim() != 4:
        return False
    n_seqs = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    pairs = n_seqs * max(q.shape[2], v.shape[2])
    return 3 * pairs < 2 * torch.cuda.get_device_properties(q.device).multi_processor_count


# Pinned plans whose backward serves every graph their forward accepts; any other
# plan has its backward checked before the forward runs.
_BWD_COMPLETE = ("kda_frost",)

_BWD_CHECKED: dict = {}


def _check_bwd(plan_name, q, k, v, g, beta, cu, scale, h0, output_final_state, flags, a_log, dt_bias):
    """Raise the pinned plan's decline if it cannot serve this call's backward.

    The autograd backward reuses the forward's ``plan_name`` and has no fallback, so
    a backward decline found after the forward would fail the training step. The
    result is cached per graph signature, as the op caches its graphs: the decline's
    text only, raised as a fresh ``_Decline`` each call. A cached exception object
    would grow its traceback on every re-raise and keep each call's frames, and with
    them its input tensors, alive for the life of the process."""
    total, H, K = q.shape
    HK, (HV, V) = k.shape[1], v.shape[1:]
    use_qk_l2norm, use_beta_sigmoid, allow_neg_eigval, safe_gate, gate_lb = flags
    state_dtype = None if h0 is None else h0.dtype
    dstate_dtype = (state_dtype or torch.float32) if output_final_state else None
    key = (
        plan_name,
        q.device,
        total,
        cu.shape[0] - 1,
        H,
        HK,
        HV,
        K,
        V,
        q.dtype,
        g.dtype,
        beta.dtype,
        state_dtype,
        dstate_dtype,
        cu.dtype,
        scale,
        flags,
        None if a_log is None else a_log.dtype,
        None if dt_bias is None else dt_bias.dtype,
    )
    if key not in _BWD_CHECKED:
        try:
            graph, _ = build_bprop_graph(
                total,
                cu.shape[0] - 1,
                H,
                HK,
                HV,
                K,
                V,
                torch_dtype_to_cudnn(q.dtype),
                torch_dtype_to_cudnn(g.dtype),
                torch_dtype_to_cudnn(beta.dtype),
                None if state_dtype is None else torch_dtype_to_cudnn(state_dtype),
                None if dstate_dtype is None else torch_dtype_to_cudnn(dstate_dtype),
                torch_dtype_to_cudnn(cu.dtype),
                None,
                float(scale),
                bool(use_qk_l2norm),
                False,
                use_beta_sigmoid=bool(use_beta_sigmoid),
                allow_neg_eigval=bool(allow_neg_eigval),
                safe_gate=bool(safe_gate),
                gate_lower_bound=gate_lb,
                a_log_dtype=None if a_log is None else torch_dtype_to_cudnn(a_log.dtype),
                dt_bias_dtype=None if dt_bias is None else torch_dtype_to_cudnn(dt_bias.dtype),
            )
            select_plan(graph, plan_name)
            _BWD_CHECKED[key] = None
        except _DECLINE as e:
            _BWD_CHECKED[key] = f"{type(e).__name__}: {e}"
    if _BWD_CHECKED[key] is not None:
        raise _Decline(_BWD_CHECKED[key])


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
    plan_name,
    needs_grad,
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
    q, k, v, g, beta = (thd(t) for t in (q, k, v, g, beta))
    if needs_grad and plan_name not in _BWD_COMPLETE:
        flags = (use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, native_safe_gate, gate_lb)
        _check_bwd(plan_name, q, k, v, g, beta, cu, scale if scale is not None else K**-0.5, h0, output_final_state, flags, a_log_t, dt_bias_t)
    out = kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
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
        plan_name=plan_name,
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
        plan_name = _plan_for(q.device) if q.is_cuda else None
        if plan_name is None:
            return fallback("unsupported-device")
        if _underfilled(plan_name, q, v, cu_seqlens):
            return fallback("underfilled")
        needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state, A_log, dt_bias))
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
                plan_name,
                needs_grad,
            )
        except (_Decline, *_DECLINE) as e:
            return fallback(type(e).__name__)
        _LAST["path"] = "native"
        return out

    chunk_kda.__wrapped__ = real_fn
    return chunk_kda
