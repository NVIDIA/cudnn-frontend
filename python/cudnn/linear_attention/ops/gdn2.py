# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch custom operator for Gated DeltaNet v2 (GDN-2) linear attention.

GDN-2 generalizes GDN's scalar gates to three channel-wise gates:

    S_t = (I - k_t (beta_t . k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t . v_t)^T,
    o_t = scale * q_t S_t,

with per-key-channel log decay ``g_t in R^K``, per-key erase gate
``beta_t in R^K`` (applied inside the erase read of the decayed state), and a
per-value write gate ``w_t in R^V`` (applied to the written value).

Layout follows the graph-API GDN-2 node: THD — token-packed
``[total_tokens, heads, dim]`` q/k/v, ``g``/``beta`` ``[total_tokens, HO, K]``,
``w`` ``[total_tokens, HO, V]``, plus ``cu_seqlens`` boundaries.

The op is a thin adapter over the graph API (the SDPA-op pattern): forward
executes a cached single-node ``GDN2`` pygraph. Engine selection happens at
graph planning time over the registered python engines — ``Gdn2FrostEngine``
is the only GDN-2 engine (SM100/SM103). The op is **forward only** (the
FROST GDN-2 backward kernel is a stub), so no autograd is registered;
differentiating through it raises. Registered through
``torch.library.custom_op`` so it composes with ``torch.compile``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

import cudnn

_OP_NAMESPACE = "cudnn"
_OP_NAME = "gated_delta_net_v2"

_TORCH_TO_CUDNN_DTYPE = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
}

# one graph per static configuration (shapes, dtypes, scale, flags, device)
_fwd_graph_cache: Dict[tuple, tuple] = {}
_ws_cache: Dict[int, "torch.Tensor"] = {}


def _graph_workspace(graph, device):
    """Caller-side workspace for a compiled graph (the explicit-workspace
    convention: query the plan's size, allocate, pass to execute)."""
    if not graph._is_built:
        # mirror execute()'s auto-build: plan via the router first (a bare
        # build() would lower GDN2 to the backend, which has no lowering)
        if not graph._planning_done:
            graph.create_execution_plans()
        if graph.selected_engine is None:
            graph.build()
        else:
            graph.build_plans()
    size = graph.get_workspace_size()
    ws = _ws_cache.get(id(graph))
    if ws is None or ws.numel() < size or ws.device != device:
        ws = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
        _ws_cache[id(graph)] = ws
    return ws


_handle_cache: Dict[int, int] = {}


def _graph_handle(device):
    """Per-device cuDNN handle carrying the caller's current stream
    (classic ``set_stream`` semantics)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    handle = _handle_cache.get(idx)
    if handle is None:
        with torch.cuda.device(idx):
            handle = cudnn.create_handle()
        _handle_cache[idx] = handle
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


_engines = None


def _gdn2_engines():
    """Engines pinned for this process, or None to let the manifest decide.

    The cuTile suite sets this to validate the cuTile engines specifically;
    under manifest routing the FROST engines would serve the shapes they
    claim. A registered engine also suppresses the manifest's own copy, so a
    pin is exact."""
    return _engines


def _cudnn_dtype(dtype: Optional[torch.dtype]):
    return _TORCH_TO_CUDNN_DTYPE[dtype] if dtype is not None else None


def _check_dtype(name, t, want) -> None:
    if t.dtype != want:
        raise TypeError(f"{_OP_NAME}: {name} must be {want} (kernel-native; callers convert), got {t.dtype}")


def _build_fwd_graph(total, N, H, HV, K, V, io_dtype, g_dtype, gate_dtype, state_dtype, scale, output_final_state, use_qk_l2norm):
    graph = cudnn.pygraph()
    for _engine in _gdn2_engines() or ():  # explicit pin (tests); None => the manifest decides
        graph.register_backend(_engine)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, H, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HV, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HV, K], data_type=gate_dtype, name="beta")
    w_t = graph.tensor([total, HV, V], data_type=gate_dtype, name="w")
    cu_t = graph.tensor([N + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    s0_t = None
    if state_dtype is not None:
        s0_t = graph.tensor([N, HV, K, V], data_type=state_dtype, name="initial_state")
    O_t, fs_t, _h_t = graph.gdn2(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        w=w_t,
        cu_seqlens=cu_t,
        initial_state=s0_t,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        name="gdn2",
    )
    return graph, dict(q=q_t, k=k_t, v=v_t, g=g_t, beta=beta_t, w=w_t, cu=cu_t, s0=s0_t, O=O_t, fs=fs_t)


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_fwd", mutates_args=())
def _gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN-2 forward via a cached single-node GDN2 pygraph (THD layout).

    Returns ``(o, final_state)``; ``final_state`` is a zero-size tensor when
    ``output_final_state`` is ``False``.
    """
    total, H, K = q.shape
    if k.shape[1] != H:
        raise ValueError(f"k must carry the same head count as q ({H}), got {k.shape[1]}")
    HV, V = v.shape[1], v.shape[2]
    N = cu_seqlens.shape[0] - 1
    device = q.device
    cu = cu_seqlens.to(torch.int32).contiguous()
    _check_dtype("g", g, torch.float32)
    _check_dtype("beta", beta, q.dtype)
    _check_dtype("w", w, q.dtype)
    if initial_state is not None:
        _check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    g32 = g.contiguous()
    beta_io = beta.contiguous()
    w_io = w.contiguous()
    s0 = initial_state.contiguous() if initial_state is not None else None

    key = (
        total,
        N,
        H,
        HV,
        K,
        V,
        q.dtype,
        bool(s0 is not None),
        float(scale),
        bool(output_final_state),
        bool(use_qk_l2norm_in_kernel),
        device,
    )
    if key not in _fwd_graph_cache:
        _fwd_graph_cache[key] = _build_fwd_graph(
            total,
            N,
            H,
            HV,
            K,
            V,
            _cudnn_dtype(q.dtype),
            cudnn.data_type.FLOAT,
            _cudnn_dtype(q.dtype),
            cudnn.data_type.FLOAT if s0 is not None else None,
            float(scale),
            bool(output_final_state),
            bool(use_qk_l2norm_in_kernel),
        )
    graph, t = _fwd_graph_cache[key]

    o = torch.empty(total, HV, V, dtype=q.dtype, device=device)
    variant_pack = {
        t["q"]: q.contiguous(),
        t["k"]: k.contiguous(),
        t["v"]: v.contiguous(),
        t["g"]: g32,
        t["beta"]: beta_io,
        t["w"]: w_io,
        t["cu"]: cu,
        t["O"]: o,
    }
    if s0 is not None:
        variant_pack[t["s0"]] = s0
    final_state = torch.empty(0, dtype=torch.float32, device=device)
    if output_final_state:
        final_state = torch.empty(N, HV, K, V, dtype=torch.float32, device=device)
        variant_pack[t["fs"]] = final_state
    graph.execute(variant_pack, workspace=_graph_workspace(graph, device), handle=_graph_handle(device))
    return o, final_state


@_gdn2_fwd.register_fake
def _gdn2_fwd_fake(q, k, v, g, beta, w, cu_seqlens, scale, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False):
    total, _H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    N = cu_seqlens.shape[0] - 1
    o = q.new_empty(total, HV, V)
    final = q.new_empty((N, HV, K, V) if output_final_state else (0,), dtype=torch.float32)
    return o, final


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gated_delta_net_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet v2 (GDN-2) linear attention — forward only.

    THD layout (matches the graph-API GDN-2 node):

        q, k: ``[total_tokens, H, K]``; v: ``[total_tokens, HV, V]``;
        g, beta: ``[total_tokens, HV, K]``; w: ``[total_tokens, HV, V]``;
        cu_seqlens: ``[N+1]`` int32;
        initial_state / final_state: ``[N, HV, K, V]``.

    A dense batch of N equal-length sequences is expressed as
    ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Args:
        g: per-key-channel log-space decay (``alpha = exp(g)``).
        beta: per-key erase gate.
        w: per-value write gate.
        cu_seqlens: ``[N+1]`` int32 sequence boundaries over the packed tokens.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state (otherwise zero).
        output_final_state: if ``True``, also return the per-sequence state
            after the last token.
        use_qk_l2norm_in_kernel: if ``True``, L2-normalize the q/k rows inside
            the kernel; if ``False``, pass q/k as given (the caller owns their
            conditioning).

    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v``. ``final_state`` is
        empty unless ``output_final_state=True``.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.cudnn.gated_delta_net_v2_fwd(
        q,
        k,
        v,
        g,
        beta,
        w,
        cu_seqlens,
        float(scale),
        initial_state=initial_state,
        output_final_state=bool(output_final_state),
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
    )
