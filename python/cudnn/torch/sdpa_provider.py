# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "CUDNN" torch.nn.attention provider: torch.sdpa on the cuDNN Python API.

Routes PyTorch's cuDNN SDPA backend through the cudnn-frontend Python API
instead of the vendored C++ frontend. Overrides the CUDA dispatch-key kernels
of

    aten::_scaled_dot_product_cudnn_attention
    aten::_scaled_dot_product_cudnn_attention_backward

with Python implementations that call the cudnn-frontend Python API custom ops
(``torch.ops.cudnn.sdpa_fwd`` / ``sdpa_bwd`` from ``cudnn.sdpa.fwd.torch_op``).
The native Autograd wrapper of the aten op is untouched: it saves our forward's
outputs and routes grad through the (also overridden) aten backward, so vanilla

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)

transparently runs on the Python API after ``install()``.

Conveniently, aten's logsumexp convention for this op is (B, H, S, 1) float32
(keepdim) — bit-identical in layout to cuDNN's Stats tensor, so tensors cross
the boundary with no reshape or copy.

Hybrid fallback to the C++ worker ops (bit-exact with the shadowed native
kernel): attn_bias, dropout_p > 0, and the padded dense backward (per-batch
lengths). Dense and varlen backward both run on the python API. The forward runs on the python API
either way, so training still exercises the python fwd path.
"""

import math
from typing import Optional

import torch

# Importing this module registers torch.ops.cudnn.sdpa_fwd / sdpa_bwd.
import cudnn.sdpa.fwd.torch_op as _cudnn_ops  # noqa: F401

_lib: Optional[torch.library.Library] = None

# Observability for tests: how many aten calls the bridge served on the
# python API vs fell back (cpp = C++ worker ops; fa2 = flash varlen kernels).
calls = {"fwd": 0, "bwd": 0, "fwd_cpp": 0, "bwd_cpp": 0, "fwd_fa2": 0, "bwd_fa2": 0}


def _fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
):
    if attn_bias is not None or dropout_p != 0.0 or return_debug_mask:
        # Not wired in the python path yet — fall back to the C++ implementation
        # through the (un-shadowed) worker op. Bit-exact with the native kernel.
        calls["fwd_cpp"] += 1
        return torch.ops.aten._cudnn_attention_forward(
            query, key, value, attn_bias, None, None,
            query.size(-2), key.size(-2), compute_log_sumexp,
            dropout_p, is_causal, return_debug_mask, scale=scale,
        )  # fmt: skip

    calls["fwd"] += 1
    attn_scale = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))

    # Below-autograd call: runs the raw CUDA impl (graph-cached cuDNN execute).
    o, stats = torch.ops.cudnn.sdpa_fwd(query, key, value, attn_scale, is_causal=is_causal, return_lse=compute_log_sumexp)

    # aten contract: (output, logsumexp(B,H,S,1) f32, cum_seq_q, cum_seq_k,
    #                 max_q, max_k, philox_seed, philox_offset, debug_attn_mask)
    philox_seed = torch.zeros((), dtype=torch.long, device=query.device)
    philox_offset = torch.zeros((), dtype=torch.long, device=query.device)
    return (o, stats, None, None, query.size(-2), key.size(-2), philox_seed, philox_offset, None)


def _bwd(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    cum_seq_q: Optional[torch.Tensor],
    cum_seq_k: Optional[torch.Tensor],
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: Optional[float] = None,
):
    # Anything cudnn::sdpa_bwd does not serve goes to the C++ worker op
    # (bit-exact with the shadowed native kernel): attention bias, dropout,
    # and the padded dense path (per-batch lengths). Everything else runs on
    # the python API, closing the last C++ hop in a dense training step.
    if attn_bias is not None or dropout_p > 0.0 or cum_seq_q is not None:
        calls["bwd_cpp"] += 1
        return torch.ops.aten._cudnn_attention_backward(
            grad_out, query, key, value, out, logsumexp,
            philox_seed, philox_offset, attn_bias, cum_seq_q, cum_seq_k,
            max_q, max_k, dropout_p, is_causal, scale=scale,
        )  # fmt: skip

    calls["bwd"] += 1
    attn_scale = scale if scale is not None else query.shape[-1] ** -0.5
    # aten hands us logsumexp as (B, H, S) fp32 — exactly the layout the dense
    # backward wants, modulo the trailing 1 the descriptor declares.
    lse = logsumexp if logsumexp.dim() == 4 else logsumexp.unsqueeze(-1)
    return torch.ops.cudnn.sdpa_bwd(
        grad_out, query, key, value, out, lse, attn_scale,
        is_causal=is_causal,
        is_deterministic=torch.are_deterministic_algorithms_enabled(),
    )  # fmt: skip


def install() -> None:
    """Register the overrides (idempotent per process: last registration wins)."""
    global _lib
    if _lib is None:
        _lib = torch.library.Library("aten", "IMPL")
    _lib.impl("_scaled_dot_product_cudnn_attention", _fwd, "CUDA")
    _lib.impl("_scaled_dot_product_cudnn_attention_backward", _bwd, "CUDA")


# ---------------------------------------------------------------------------
# varlen_attn (THD) via the cuDNN python API
#
# torch.nn.attention.varlen.varlen_attn routes to flash kernels in 2.13 (its
# in-tree cuDNN branch is dead: `_should_use_cudnn` is hardcoded False, and
# its `_cudnn_attention_backward` call predates the 2.13 schema). We hook one
# level up instead: override the `torch_attn::_varlen_attn{,_backward}`
# custom ops at the CUDA key. Their autograd wiring is untouched; unlike the
# dead branch we also serve GQA and causal sliding windows.
# ---------------------------------------------------------------------------


def _norm_window(window_size):
    ws = list(window_size) if window_size is not None else [-1, -1]
    if len(ws) != 2:
        raise ValueError(f"window_size must have length 2, got {len(ws)}")
    return ws


def _fa_window_left_to_cudnn(w: int) -> int:
    """FA2 window_size=(w, 0) attends to [i-w, i] — w tokens back PLUS self.
    cuDNN's diagonal_band_left_bound=lb masks j <= i-lb, i.e. lb visible
    tokens including self. So lb = w + 1."""
    return w + 1 if w >= 0 else -1


def _fa_window_right_to_cudnn(is_causal: bool, w: int) -> tuple[bool, int]:
    """FA2 window_size=(_, r) attends up to column i+r; cuDNN's
    diagonal_band_right_bound=rb masks columns BEYOND i+rb, so rb = r with no
    offset (unlike the left bound, which is exclusive and needs the +1).

    r == 0 is exactly causal, and is folded into is_causal so the common case
    keeps hitting the same graph-cache entry it always did. r > 0 admits future
    columns, which contradicts is_causal -- the op rejects that pairing."""
    if w == 0:
        return True, -1
    if w > 0:
        return is_causal, w
    return is_causal, -1


def _varlen_supported(ws, seqused_k=None, block_table=None, num_splits=None) -> bool:
    """Configs the cudnn python varlen path serves today; everything else falls
    back to the flash kernels (exactly what the stock op body runs)."""
    if seqused_k is not None or block_table is not None:
        # Paged KV stays on flash by design (Phase-1 scope decision): the op's
        # stock body already serves it correctly.
        return False
    if num_splits is not None and num_splits != 1:
        return False
    # Both bounds are expressible now: window_left -> diagonal_band_left_bound,
    # window_right -> diagonal_band_right_bound. Asymmetric bands included.
    return True


def _varlen_fwd_flash(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale, ws, seqused_k, block_table, num_splits):
    calls["fwd_fa2"] += 1
    output, softmax_lse, _rng, _, _ = torch.ops.aten._flash_attention_forward(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, 0.0, is_causal,
        return_debug_mask=False, scale=scale,
        window_size_left=ws[0], window_size_right=ws[1],
        seqused_k=seqused_k, block_table=block_table, num_splits=num_splits,
    )  # fmt: skip
    rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    return output, softmax_lse, rng_state


def _varlen_fwd(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal=False, scale=None, window_size=None, enable_gqa=False, seqused_k=None, block_table=None, num_splits=None,):  # fmt: skip
    ws = _norm_window(window_size)
    if not _varlen_supported(ws, seqused_k, block_table, num_splits):
        return _varlen_fwd_flash(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, scale, ws, seqused_k, block_table, num_splits)
    is_causal, window_right = _fa_window_right_to_cudnn(is_causal, ws[1])

    calls["fwd"] += 1
    attn_scale = scale if scale is not None else query.shape[-1] ** -0.5
    o, stats = torch.ops.cudnn.sdpa_fwd(
        query, key, value, attn_scale,
        is_causal=is_causal, window_left=_fa_window_left_to_cudnn(ws[0]), window_right=window_right,
        cu_seqlens_q=cu_seq_q, cu_seqlens_kv=cu_seq_k,
        max_seqlen_q=max_q, max_seqlen_kv=max_k, return_lse=True,
    )  # fmt: skip
    lse = stats.squeeze(-1).transpose(0, 1).contiguous()  # (T,H,1) -> (H,T) flash convention
    rng_state = torch.zeros((2,), dtype=torch.uint64, device=query.device)
    return o, lse, rng_state


def _varlen_fwd_out(out, query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, is_causal=False, scale=None, window_size=None, enable_gqa=False, seqused_k=None, block_table=None, num_splits=None,):  # fmt: skip
    """torch_attn::_varlen_attn_out — same as fwd but writes into `out`; returns lse."""
    ws = _norm_window(window_size)
    if not _varlen_supported(ws, seqused_k, block_table, num_splits):
        calls["fwd_fa2"] += 1
        return torch.ops.aten._flash_attention_forward_no_dropout_inplace(
            out, query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, 0.0, is_causal,
            False, scale=scale, window_size_left=ws[0], window_size_right=ws[1],
            seqused_k=seqused_k, block_table=block_table, num_splits=num_splits,
        )  # fmt: skip
    o, lse, _rng = _varlen_fwd(
        query, key, value, cu_seq_q, cu_seq_k, max_q, max_k,
        is_causal=is_causal, scale=scale, window_size=window_size, enable_gqa=enable_gqa,
        seqused_k=seqused_k, block_table=block_table, num_splits=num_splits,
    )  # fmt: skip
    out.copy_(o)
    return lse


def _varlen_bwd(grad_out, query, key, value, out, lse, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, rng_state, scale=None, window_size=None,):  # fmt: skip
    ws = _norm_window(window_size)
    if not _varlen_supported(ws):
        calls["bwd_fa2"] += 1  # fwd for this config ran flash too (same predicate)
        unused = torch.empty(0, device=query.device)
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out, query, key, value, out, lse, cu_seq_q, cu_seq_k, max_q, max_k,
            0.0, is_causal, rng_state, unused, scale=scale,
            window_size_left=ws[0], window_size_right=ws[1],
        )  # fmt: skip
        return dq, dk, dv
    is_causal, window_right = _fa_window_right_to_cudnn(is_causal, ws[1])

    calls["bwd"] += 1
    attn_scale = scale if scale is not None else query.shape[-1] ** -0.5
    # (H, T) packed -> (B, H, max_q, 1) padded: the backend rejects ragged LSE
    # for bprop THD on SM8X/SM12X, so the bwd op takes the padded layout.
    # (H, T) -> (T, H) for the shared device-side repad. The naive
    # `for i in range(B): int(cu_seq_q[i])` loop that used to live here was
    # 2*B blocking D2H copies per backward call, before the kernel even
    # launched — an async-launch API turned synchronous, and un-capturable
    # (python/cudnn/AGENTS.md Rule 3).
    lse_padded = _cudnn_ops.thd_lse_to_padded(lse.transpose(0, 1), cu_seq_q, max_q)
    dq, dk, dv = torch.ops.cudnn.sdpa_bwd(
        grad_out, query, key, value, out, lse_padded, attn_scale,
        is_causal=is_causal, window_left=_fa_window_left_to_cudnn(ws[0]), window_right=window_right,
        cu_seqlens_q=cu_seq_q, cu_seqlens_kv=cu_seq_k,
        max_seqlen_q=max_q, max_seqlen_kv=max_k,
        is_deterministic=torch.are_deterministic_algorithms_enabled(),
    )  # fmt: skip
    return dq, dk, dv


# ---------------------------------------------------------------------------
# torch.nn.attention flash-impl registry integration (PyTorch 2.13+)
#
# The same mechanism FA3/FA4 use: activation registers python overrides of
# existing CUDA kernels; restore drops the Library handles to deregister.
#
#     import cudnn.torch  # registers "CUDNN" (no activation)
#     torch.nn.attention.activate_flash_attention_impl("CUDNN")
# ---------------------------------------------------------------------------


class _RegistryHandle:
    def __init__(self, *libs: torch.library.Library):
        self._libs = list(libs)

    def remove(self) -> None:
        for lib in self._libs:
            lib._destroy()
        self._libs = []


def _registry_register() -> _RegistryHandle:
    import cudnn.sdpa.fwd.torch_op  # noqa: F401 — registers cudnn::sdpa_fwd / sdpa_bwd

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_scaled_dot_product_cudnn_attention", _fwd, "CUDA")
    lib.impl("_scaled_dot_product_cudnn_attention_backward", _bwd, "CUDA")
    vlib = torch.library.Library("torch_attn", "IMPL")
    from torch.nn.attention import varlen as _varlen_mod  # noqa: F401 — ensure torch_attn ops are defined

    vlib.impl("_varlen_attn", _varlen_fwd, "CUDA")
    vlib.impl("_varlen_attn_out", _varlen_fwd_out, "CUDA")
    vlib.impl("_varlen_attn_backward", _varlen_bwd, "CUDA")
    return _RegistryHandle(lib, vlib)


def _register_with_torch() -> None:
    try:
        from torch.nn.attention import register_flash_attention_impl
    except ImportError:
        return  # torch < 2.13: use install() directly
    register_flash_attention_impl("CUDNN", register_fn=_registry_register)


_register_with_torch()


def served_plan_names() -> list:
    """Which execution plan served each cached graph (debug/reporting)."""
    names = []
    for graph, _ws in _cudnn_ops._graph_cache.values():
        try:
            names.append(graph.get_plan_name_at_index(graph._plan_index))
        except Exception as e:  # noqa: BLE001
            names.append(f"<unavailable: {e}>")
    return names
