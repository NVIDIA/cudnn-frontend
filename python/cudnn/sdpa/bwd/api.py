# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple
import inspect
import logging
import math

from cuda.bindings import driver as cuda
import torch


from cudnn.api_base import APIBase, TupleDict

_logger = logging.getLogger(__name__)


_KERNEL_MOD = {}


def _stream_ctx(current_stream):
    """Context manager dispatching onto ``current_stream`` (a ``cuda.CUstream``
    or raw stream int); the kernels launch on torch's current stream, so an
    ExternalStream context routes them.  ``None`` keeps the current stream, and
    a raw handle equal to torch's current/default stream reuses that torch
    stream object rather than wrapping it: ``ExternalStream(0)`` breaks
    re-execution on some torch builds (NGC), where every launch after the
    compile run silently no-ops (all-zero outputs; caught by test_mhas_v2's
    determinism re-run).  Mirrors gemm/cutedsl/grouped/backend_utils.py."""
    import contextlib

    if current_stream is None:
        return contextlib.nullcontext()
    handle = int(current_stream)
    torch_current = torch.cuda.current_stream()
    if handle in (0, 1, 2) or handle == torch_current.cuda_stream:
        return contextlib.nullcontext()
    torch_default = torch.cuda.default_stream()
    if handle == torch_default.cuda_stream:
        return torch.cuda.stream(torch_default)
    return torch.cuda.stream(torch.cuda.ExternalStream(handle))


# The generic kernel now supports d_qk != d_v (split sub-groups) and d up to
# 256, so the envelope spans gptoss(64,64) / llama(128,128) / dsv3(192,128) /
# qwen(256,256).  Kernel constraint: d_qk >= d_v (the per-sub-group split).
# The flavor only sets the d-pad target; the kernel derives qo_stages / drop-sDQ
# from the (padded) d_qk for the A100 SMEM budget.
from ..fwd import config_sm80 as _fwd_config_sm80

_FLAVOR_DIMS = {
    name: (cfg.D_QK, cfg.D_V)
    for name, cfg in (
        ("gptoss", _fwd_config_sm80.GPTOSS_CFG),
        ("llama", _fwd_config_sm80.LLAMA_CFG),
        ("dsv3", _fwd_config_sm80.DSV3_CFG),
        ("qwen", _fwd_config_sm80.QWEN_CFG),
    )
}
_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")


def _load_kernel_module(key: str = "f16"):
    """Lazily import + cache an SM80 BPROP kernel module.

    ``"f16"`` is the GENERIC kernel (``bprop_f16_sm80``): fully parameterized
    on d_qk/d_v with the full feature set (masks / bias / dBias /
    sink / rope / THD / deterministic).  ``"d64"`` is the
    dedicated plain-dense d=64 MHA perf variant (~2x faster on A100); it
    supports NO features — its ``backward(**_ignored)`` silently swallows
    every feature kwarg, so callers must never rely on the signature filter
    and only select it through :func:`_d64_fast_path_eligible`.
    """
    if key not in _KERNEL_MOD:
        if key == "d64":
            from .kernels import bprop_d64_f16_sm80 as _mod
        else:
            from .kernels import bprop_f16_sm80 as _mod

        _KERNEL_MOD[key] = _mod
    return _KERNEL_MOD[key]


def _d64_fast_path_eligible(*, d_qk, d_v, h_q, h_kv, s_q, s_kv, mask_token, right_bound, causal_bottom_right, bw_kwargs) -> bool:
    """Whether the dedicated d=64 kernel can serve this call EXACTLY.

    The perf variant computes a plain dense MHA backward and nothing else;
    every condition here guards a feature it would silently ignore.
    """
    d64 = _load_kernel_module("d64")
    if (d_qk, d_v) != (64, 64) or h_q != h_kv:
        return False
    if s_q % d64.M_BLOCK != 0 or s_kv % d64.N_BLOCK != 0:
        return False
    if mask_token != "none" or right_bound != 0 or causal_bottom_right:
        return False
    for feature in ("seq_kv_lens", "seq_len_q", "bias", "sinks", "rope_freqs"):
        if bw_kwargs.get(feature) is not None:
            return False
    if bw_kwargs.get("deterministic"):
        return False
    return True


def _pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest BPROP flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)`` (fdqk >= d_qk and fdv >= d_v); the user's heads are padded
    up to the flavor dim.  The kernel supports d_qk != d_v but requires the
    (padded) d_qk >= d_v — the flavor list guarantees this (every flavor has
    fdqk >= fdv, and a d_qk < d_v case lands on an equal-d flavor after pad)."""
    for flavor in _SUPPORTED_FLAVORS:
        fdqk, fdv = _FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SUPPORTED_FLAVORS:
        fdqk, fdv = _FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(f"SM80 BPROP: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}); " f"supported: {_FLAVOR_DIMS}.")


def _pad_last_dim(t: torch.Tensor, new_last: int) -> torch.Tensor:
    """Zero-pad the trailing dim of an fp16/bf16 tensor up to ``new_last``."""
    old_last = t.shape[-1]
    if old_last == new_last:
        return t
    if old_last > new_last:
        raise ValueError(f"_pad_last_dim: tensor's last dim {old_last} exceeds target {new_last}")
    pad = torch.zeros((*t.shape[:-1], new_last - old_last), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1).contiguous()


def _bshd(t: torch.Tensor) -> torch.Tensor:
    """BHSD → BSHD (stride-only transpose; contiguous-ify only if needed)."""
    x = t.transpose(1, 2)
    return x if x.is_contiguous() else x.contiguous()


# ---------------------------------------------------------------------------
# APIBase subclass.
# ---------------------------------------------------------------------------
class SdpabwdSm80(APIBase):
    """SM80 (A100) SDPA backward.

    Mirrors the SM80 forward adapter.  Inputs are the forward activations (Q/K/V/O), the
    loss gradient dO, and the forward stats LSE.  Outputs dQ/dK/dV (+ dBias when
    an additive bias is present).
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_do: torch.Tensor,
        sample_lse: torch.Tensor,
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        scale_softmax: Optional[float] = None,
        causal_bottom_right: bool = False,
        has_seq_kv_lens: bool = False,
        has_bias: bool = False,
    ):
        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__ (bwd)")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.do_desc = self._make_tensor_desc(sample_do, name="dO")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="lse")

        self.is_causal = is_causal
        self.window_size_left, self.window_size_right = window_size
        self.scale_softmax = scale_softmax
        self.causal_bottom_right = bool(causal_bottom_right)
        self.has_seq_kv_lens = bool(has_seq_kv_lens)
        self.has_bias = bool(has_bias)

        # Filled by check_support().
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound: int = 0
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self._logger.debug("__init__ (bwd) completed")

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support (bwd)")

        _REQ = (3, 1, 2, 0)
        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc", "do_desc"]:
            d = getattr(self, desc_name)
            self._value_error_if(d.ndim != 4, f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}")
            _shape = d.shape
            _act = tuple(ax for ax in d.stride_order if _shape[ax] != 1)
            _exp = tuple(ax for ax in _REQ if _shape[ax] != 1)
            self._value_error_if(
                _act != _exp, f"{d.name} must have d,h,s,b stride order (3,1,2,0) " f"(size-1 dims wildcarded); got {d.stride_order} shape {_shape}"
            )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")
        self._check_tensor_shape(self.do_desc, (b, h_qo, s_qo, d_v), name="dO")

        for label, val in (("B", b), ("H_q", h_qo), ("H_kv", h_kv), ("S_q", s_qo), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(h_qo % h_kv != 0, f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA")

        # Kernel supports d_qk != d_v (split sub-groups) but requires d_qk >= d_v
        # (a d_qk < d_v case is padded up to an equal-d flavor by _pick_flavor).
        self._value_error_if(d_qk < d_v, f"SM80 BPROP requires D_QK >= D_V; got D_QK={d_qk}, D_V={d_v}")
        max_dqk = max(fdqk for fdqk, _ in _FLAVOR_DIMS.values())
        max_dv = max(fdv for _, fdv in _FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_dqk or d_v > max_dv,
            f"SM80 BPROP: head dim (D_QK={d_qk}, D_V={d_v}) exceeds supported " f"envelope (D_QK<={max_dqk}, D_V<={max_dv}); larger heads not yet ported.",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in [self.k_desc, self.v_desc, self.o_desc, self.do_desc]:
            self._check_dtype(desc, self.dtype, name=desc.name, extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16)")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
        self._value_error_if(not self.lse_desc.is_contiguous(), "LSE must be contiguous on SM80")

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM80 BPROP")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._value_error_if((major, minor) != (8, 0), f"SdpabwdSm80 requires SM80 (A100); found SM{major}{minor} on {device}")

        self.flavor = _pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _FLAVOR_DIMS[self.flavor]
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        # ---- mask token (same resolution as the forward adapter) ------
        swa_left = self.window_size_left
        swa_right = self.window_size_right
        self.right_bound = 0
        if self.is_causal:
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound = max(0, swa_right)
        elif swa_left >= 0:
            # A left window alone selects SWA; window_size_right is only
            # meaningful with is_causal=True.
            self._not_implemented_error_if(swa_right > 0, "SM80 BPROP: non-causal SWA with window_size_right > 0 unsupported")
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            # window_size=(-1, r) without is_causal: a bare right bound has no
            # diagonal to anchor to — reject rather than silently pick a mask
            # (mirrors the forward adapter and the THD path).
            self._not_implemented_error_if(
                swa_right >= 0,
                "SM80 BPROP: window_size_right without a left window or is_causal=True has no effect; pass is_causal=True or a left window",
            )
            self.mask_token = "none"
            self.swa_window_runtime = 0

        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or self.window_size_left >= 0),
            "SM80 BPROP: causal_bottom_right requires is_causal and/or a left window",
        )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self._is_supported = True
        self._logger.debug("check_support (bwd) completed")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """No-op — the kernel module owns its own per-shape ``lru_cache``;
        first ``execute()`` JITs and reuses thereafter."""
        self._logger.debug("Entering compile (bwd, no-op — kernel self-caches)")
        self._ensure_support_checked()
        self._compiled_kernel = True
        self._logger.debug("compile (bwd) completed")

    # ------------------------------------------------------------------
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        do_tensor: torch.Tensor,
        lse_tensor: torch.Tensor,
        dq_tensor: torch.Tensor,
        dk_tensor: torch.Tensor,
        dv_tensor: torch.Tensor,
        dbias_tensor: Optional[torch.Tensor] = None,
        dsink_tensor: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        seq_len_q: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> None:
        self._logger.debug("Entering execute (bwd)")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpabwdSm80 is not compiled")
        scale_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)

        kernel = _load_kernel_module()

        # BHSD → BSHD for the kernel.
        Q, K, V = _bshd(q_tensor), _bshd(k_tensor), _bshd(v_tensor)
        O, dO = _bshd(o_tensor), _bshd(do_tensor)

        pad_v = self.head_dim_v < self.flavor_d_v
        pad_qk = self.head_dim_qk < self.flavor_d_qk
        if pad_qk:
            Q = _pad_last_dim(Q, self.flavor_d_qk)
            K = _pad_last_dim(K, self.flavor_d_qk)
        if pad_v:
            V = _pad_last_dim(V, self.flavor_d_v)
            O = _pad_last_dim(O, self.flavor_d_v)
            dO = _pad_last_dim(dO, self.flavor_d_v)

        # Build the feature-kwarg superset; drop any the kernel doesn't accept.
        bw_kwargs = dict(
            scale=scale_val,
            mask=self.mask_token,
            swa_window=int(self.swa_window_runtime),
            right_bound=int(self.right_bound),
            causal_bottom_right=self.causal_bottom_right,
            seq_kv_lens=seq_kv_lens,
            seq_len_q=seq_len_q,
            bias=bias_tensor,
            sinks=sinks,
            rope_freqs=rope_freqs,
            deterministic=bool(deterministic),
        )
        # Route plain dense MHA d=64 calls to the dedicated perf kernel
        # (~2x faster on A100).  The gate must stay exhaustive: the d64
        # kernel's ``backward(**_ignored)`` silently swallows any feature
        # kwarg it does not implement, so an under-gated call would produce
        # wrong gradients rather than an error.
        if _d64_fast_path_eligible(
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            h_q=q_tensor.shape[1],
            h_kv=k_tensor.shape[1],
            s_q=q_tensor.shape[2],
            s_kv=k_tensor.shape[2],
            mask_token=self.mask_token,
            right_bound=int(self.right_bound),
            causal_bottom_right=self.causal_bottom_right,
            bw_kwargs=bw_kwargs,
        ):
            kernel = _load_kernel_module("d64")
            self._logger.debug("execute (bwd): routing to the dedicated d64 kernel")
        accepted = inspect.signature(kernel.backward).parameters
        bw_kwargs = {kk: vv for kk, vv in bw_kwargs.items() if kk in accepted}

        with _stream_ctx(current_stream):
            res = kernel.backward(Q, K, V, dO, O, lse_tensor, **bw_kwargs)
        dQ_k, dK_k, dV_k = res[0], res[1], res[2]
        # backward() appends optional grads in a FIXED order: dBias (if bias),
        # then dSink (if sinks).  Reconstruct positions from what we passed.
        _idx = 3
        dBias_k = None
        dSink_k = None
        if bias_tensor is not None:
            dBias_k = res[_idx]
            _idx += 1
        if sinks is not None:
            dSink_k = res[_idx]
            _idx += 1

        # Slice off any d-padding, transpose BSHD → BHSD, copy into user tensors.
        if pad_qk:
            dQ_k = dQ_k[..., : self.head_dim_qk]
            dK_k = dK_k[..., : self.head_dim_qk]
        if pad_v:
            dV_k = dV_k[..., : self.head_dim_v]
        dq_tensor.copy_(dQ_k.transpose(1, 2))
        dk_tensor.copy_(dK_k.transpose(1, 2))
        dv_tensor.copy_(dV_k.transpose(1, 2))
        if dbias_tensor is not None and dBias_k is not None:
            # dBias is head-major [., H, SQ, SKV] (like bias) — no transpose.
            dbias_tensor.copy_(dBias_k.to(dbias_tensor.dtype))
        if dsink_tensor is not None and dSink_k is not None:
            dsink_tensor.copy_(dSink_k.to(dsink_tensor.dtype))
        self._logger.debug("execute (bwd) completed")


# ---------------------------------------------------------------------------
# THD / varlen backward (mirrors fwd/api.py::_thd_forward).
# ---------------------------------------------------------------------------
def _thd_backward(q, k, v, o, do, lse, *, cu_q, cu_k, scale_softmax, is_causal, window_size, causal_bottom_right, sinks=None, deterministic=False):
    """THD / varlen backward: q/k/v/o/do are PACKED ``[1, T, H, D]`` (BSHD,
    B==1 — no transpose), ``lse`` is packed ``[1, H, T_q]`` (head-major,
    matching the kernel's THD LSE layout), and cu_q/cu_k are ``[n_seq+1]``
    cumulative seqlens.  Routes straight to the kernel's THD backward
    (over-provisioned grid; MHA-only), reusing the flavor-pick + d-pad.
    Returns packed ``[1, T, H, D]`` dQ/dK/dV (BHSD-equivalent for B==1)."""
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    flavor = _pick_flavor(d_qk, d_v)
    fdqk, fdv = _FLAVOR_DIMS[flavor]
    # Resolve the default scale from the USER's head dim before padding: the
    # kernel would otherwise derive 1/sqrt(D) from the padded flavor width
    # (e.g. 1/sqrt(128) for a d=96 llama-flavor call) — silently wrong
    # gradients.  Mirrors the forward THD path.
    if scale_softmax is None or scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    pad_qk = d_qk < fdqk
    pad_v = d_v < fdv
    if pad_qk:
        q = _pad_last_dim(q, fdqk)
        k = _pad_last_dim(k, fdqk)
    if pad_v:
        v = _pad_last_dim(v, fdv)
        o = _pad_last_dim(o, fdv)
        do = _pad_last_dim(do, fdv)
    # mask token from cuDNN's (is_causal, window_size=(left,right)).
    wl, wr = window_size
    if is_causal and wl >= 0:
        mask_token, swa = "causal_swa", wl
    elif is_causal:
        mask_token, swa = "causal", 0
    elif wl >= 0:
        mask_token, swa = "swa", wl
    else:
        mask_token, swa = "none", 0
    right_bound = wr if (is_causal and wr is not None and wr > 0) else 0
    kernel = _load_kernel_module()
    sinks_t = sinks.to(dtype=torch.float32, device=q.device).reshape(h_q).contiguous() if sinks is not None else None
    bw_kwargs = dict(
        scale=scale_softmax,
        mask=mask_token,
        swa_window=int(swa),
        right_bound=int(right_bound),
        causal_bottom_right=bool(causal_bottom_right),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        sinks=sinks_t,
        deterministic=bool(deterministic),
    )
    acc = inspect.signature(kernel.backward).parameters
    bw_kwargs = {kk: vv for kk, vv in bw_kwargs.items() if kk in acc}
    res = kernel.backward(q, k, v, do, o, lse, **bw_kwargs)
    dQ_k, dK_k, dV_k = res[0], res[1], res[2]
    _idx = 3
    dSink_k = None
    if sinks_t is not None:
        dSink_k = res[_idx]
        _idx += 1
    if pad_qk:
        dQ_k = dQ_k[..., :d_qk].contiguous()
        dK_k = dK_k[..., :d_qk].contiguous()
    if pad_v:
        dV_k = dV_k[..., :d_v].contiguous()
    out = TupleDict(dq_tensor=dQ_k, dk_tensor=dK_k, dv_tensor=dV_k)
    if dSink_k is not None:
        out["dsink_tensor"] = dSink_k
    return out


# ---------------------------------------------------------------------------
# Functional wrapper (mirrors the forward surface).
# ---------------------------------------------------------------------------
_cache_of_objects: dict = {}


def sdpa_bwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    o_tensor: torch.Tensor,
    do_tensor: torch.Tensor,
    lse_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    causal_bottom_right: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    rope_freqs: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> TupleDict:
    """SM80 (A100) SDPA backward.

    Returns ``TupleDict(dq_tensor=..., dk_tensor=..., dv_tensor=...
    [, dbias_tensor=...][, dsink_tensor=...])`` — BHSD grads; dBias
    head-major [., H, SQ, SKV] when ``bias_tensor`` is given; dSink (H,)
    fp32 when ``sinks`` is given (stable order: dq, dk, dv, dbias, dsink).
    ALiBi and block_mask are not supported (use the graph API, which routes
    them to the cuDNN backend); bias/dBias remain fully served.
    """
    # THD / varlen: q/k/v/o/dO are PACKED [1, T, H, D] (BSHD) + cu_seqlens;
    # lse is packed [1, H, T_q].  Dedicated path that skips the dense BHSD
    # transpose + dense grad alloc (mirrors fwd/api.py's THD branch).
    if cum_seqlen_q_tensor is not None:
        # Reject dense-only features up front: _thd_backward accepts only
        # sinks/deterministic, and silently computing gradients without a
        # requested feature is worse than an error.
        for label, present in (
            ("bias_tensor", bias_tensor is not None),
            ("rope_freqs", rope_freqs is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) backward does not support {label}; the dense path serves it")
        with _stream_ctx(current_stream):
            return _thd_backward(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                do_tensor,
                lse_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                sinks=sinks,
                deterministic=deterministic,
            )
    for nm, t in (("Q", q_tensor), ("V", v_tensor), ("O", o_tensor), ("dO", do_tensor)):
        if t.ndim != 4:
            raise ValueError(f"{nm} must be rank-4 BHSD; got {t.ndim}D")

    # Allocate grad outputs in cuDNN-FE BHSD-physical stride order (3,1,2,0):
    # contiguous (B, S, H, D) then transpose to a (B, H, S, D) view.
    b, h_q, s_q, d_qk = q_tensor.shape
    d_v = v_tensor.shape[-1]
    dq = torch.empty((b, s_q, h_q, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    # dK/dV take K/V leading shape (GQA: h_kv heads).
    h_kv, s_kv = k_tensor.shape[1], k_tensor.shape[2]
    dk = torch.empty((b, s_kv, h_kv, d_qk), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    dv = torch.empty((b, s_kv, h_kv, d_v), dtype=q_tensor.dtype, device=q_tensor.device).transpose(1, 2)
    # dBias: fp32, same shape as bias ([., H, SQ, SKV]).
    dbias = torch.zeros_like(bias_tensor, dtype=torch.float32) if bias_tensor is not None else None
    # dSink: fp32 [H] (sink-logit gradient).
    dsink = torch.zeros(h_q, dtype=torch.float32, device=q_tensor.device) if sinks is not None else None

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        q_tensor.dtype,
        is_causal,
        window_size,
        scale_softmax,
        causal_bottom_right,
        seq_kv_lens is not None,
        bias_tensor is not None,
        (bias_tensor.dtype if bias_tensor is not None else None),
        sinks is not None,
        rope_freqs is not None,
        q_tensor.device,
    )
    sdpa_bwd = _cache_of_objects.get(cache_key)
    if sdpa_bwd is None:
        _logger.debug("sdpa_bwd_wrapper_sm80: building new SdpabwdSm80")
        sdpa_bwd = SdpabwdSm80(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_do=do_tensor,
            sample_lse=lse_tensor,
            is_causal=is_causal,
            window_size=window_size,
            scale_softmax=scale_softmax,
            causal_bottom_right=causal_bottom_right,
            has_seq_kv_lens=seq_kv_lens is not None,
            has_bias=bias_tensor is not None,
        )
        assert sdpa_bwd.check_support(), "Unsupported configuration"
        sdpa_bwd.compile()
        _cache_of_objects[cache_key] = sdpa_bwd

    sdpa_bwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        do_tensor=do_tensor,
        lse_tensor=lse_tensor,
        dq_tensor=dq,
        dk_tensor=dk,
        dv_tensor=dv,
        dbias_tensor=dbias,
        dsink_tensor=dsink,
        scale_softmax=scale_softmax,
        current_stream=current_stream,
        seq_kv_lens=seq_kv_lens,
        seq_len_q=seq_len_q,
        bias_tensor=bias_tensor,
        sinks=sinks,
        rope_freqs=rope_freqs,
        deterministic=deterministic,
    )

    out = TupleDict(dq_tensor=dq, dk_tensor=dk, dv_tensor=dv)
    if dbias is not None:
        out["dbias_tensor"] = dbias
    if dsink is not None:
        out["dsink_tensor"] = dsink
    return out
