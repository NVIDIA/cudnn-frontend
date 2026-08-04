# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch custom operator for Kimi Delta Attention (KDA) linear attention.

KDA is GDN with a per-key-channel decay: it replaces GDN's scalar per-token
decay ``alpha_t`` with a decay vector ``alpha_t in (0, 1]^K``, applied as a
diagonal matrix on the K-axis of the recurrent state:

    S_t = (I - beta_t k_t^T k_t) Diag(alpha_t) S_{t-1} + beta_t k_t^T v_t,
    o_t = q_t S_t,

with ``alpha_t = exp(g_t)`` (``g_t in R^K``, log-space per channel) and
scalar write strength ``beta_t``. The decay is applied first, so the
delta-rule correction reads the already-decayed state.

Layout follows the graph-API KDA node: THD — token-packed ``[total_tokens,
heads, dim]`` tensors plus ``cu_seqlens`` sequence boundaries. ``g`` is the
per-key-channel log decay ``[total_tokens, heads, dim]``; ``beta`` is scalar
``[total_tokens, heads]``.

The op is a thin adapter over the graph API (the SDPA-op pattern): forward
and backward execute cached single-node ``KDA`` / ``KDA_BWD`` pygraphs.
Engine selection happens at graph planning time over the registered python
engines: ``KdaFrostEngine`` (forward, default on SM100/SM103) with
``KdaCuTileEngine`` as the fallback — and the backward engine everywhere
(the FROST KDA backward kernel is a stub). Registered through
``torch.library.custom_op`` so it composes with autograd, ``torch.compile``,
and DDP.

The backward graph recomputes the forward's cheap intermediates (cumulative
gate, intra-chunk WY factors) — the ``KDA_BWD`` node contract keeps them off
the autograd wire.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

import cudnn

_OP_NAMESPACE = "cudnn"
_OP_NAME = "kimi_delta_attention"

_TORCH_TO_CUDNN_DTYPE = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
}

# one graph per static configuration (shapes, dtypes, scale, flags, device)
_fwd_graph_cache: Dict[tuple, tuple] = {}
_bwd_graph_cache: Dict[tuple, tuple] = {}
_ws_cache: Dict[int, "torch.Tensor"] = {}


def _graph_workspace(graph, device):
    """Caller-side workspace for a compiled graph (the explicit-workspace
    convention: query the plan's size, allocate, pass to execute)."""
    if not graph._is_built:
        # mirror execute()'s auto-build: plan via the router first (a bare
        # build() would lower KDA to the backend, which has no lowering)
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


def _kda_engines():
    """Engines pinned for this process, or None to let the manifest decide.

    The cuTile suite sets this to validate the cuTile engines specifically;
    under manifest routing the FROST engines would serve the shapes they
    claim. A registered engine also suppresses the manifest's own copy, so a
    pin is exact."""
    return _engines


def _cudnn_dtype(dtype: Optional[torch.dtype]):
    return _TORCH_TO_CUDNN_DTYPE[dtype] if dtype is not None else None


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _check_dtype(name, t, want) -> None:
    if t.dtype != want:
        raise TypeError(f"{_OP_NAME}: {name} must be {want} (kernel-native; callers convert), got {t.dtype}")


def _build_fwd_graph(total, N, H, HV, K, V, io_dtype, g_dtype, beta_dtype, state_dtype, scale, output_final_state, use_qk_l2norm):
    graph = cudnn.pygraph()
    for _engine in _kda_engines() or ():  # explicit pin (tests); None => the manifest decides
        graph.register_backend(_engine)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, H, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HV, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HV], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    s0_t = None
    if state_dtype is not None:
        s0_t = graph.tensor([N, HV, K, V], data_type=state_dtype, name="initial_state")
    O_t, fs_t, _h_t = graph.kda(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=s0_t,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        name="kda",
    )
    return graph, dict(q=q_t, k=k_t, v=v_t, g=g_t, beta=beta_t, cu=cu_t, s0=s0_t, O=O_t, fs=fs_t)


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_fwd", mutates_args=())
def _kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KDA forward via a cached single-node KDA pygraph (THD layout).

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
    _check_dtype("beta", beta, torch.float32)
    if initial_state is not None:
        _check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    g32 = g.contiguous()
    beta32 = beta.contiguous()
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
            cudnn.data_type.FLOAT,
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
        t["beta"]: beta32,
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


@_kda_fwd.register_fake
def _kda_fwd_fake(q, k, v, g, beta, cu_seqlens, scale, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False):
    total, _H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    N = cu_seqlens.shape[0] - 1
    o = q.new_empty(total, HV, V)
    final = q.new_empty((N, HV, K, V) if output_final_state else (0,), dtype=torch.float32)
    return o, final


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _build_bwd_graph(total, N, H, HV, K, V, io_dtype, g_dtype, beta_dtype, state_dtype, dht_dtype, scale, use_qk_l2norm):
    graph = cudnn.pygraph()
    for _engine in _kda_engines() or ():  # explicit pin (tests); None => the manifest decides
        graph.register_backend(_engine)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, H, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HV, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HV], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    dO_t = graph.tensor([total, HV, V], data_type=io_dtype, name="dO")
    s0_t = None
    if state_dtype is not None:
        s0_t = graph.tensor([N, HV, K, V], data_type=state_dtype, name="initial_state")
    dfs_t = None
    if dht_dtype is not None:
        dfs_t = graph.tensor([N, HV, K, V], data_type=dht_dtype, name="d_final_state")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dS0_t = graph.kda_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=dO_t,
        initial_state=s0_t,
        d_final_state=dfs_t,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        name="kda_bwd",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        dO=dO_t,
        s0=s0_t,
        dfs=dfs_t,
        dQ=dQ_t,
        dK=dK_t,
        dV=dV_t,
        dG=dG_t,
        dBeta=dBeta_t,
        dS0=dS0_t,
    )


@torch.library.custom_op(f"{_OP_NAMESPACE}::{_OP_NAME}_bwd", mutates_args=())
def _kda_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    d_final_state: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDA backward via a cached single-node KDA_BWD pygraph (THD layout).

    Returns ``(dq, dk, dv, dg, dbeta, d_initial_state)``; ``d_initial_state``
    is a zero-size tensor when ``initial_state`` is ``None``.
    """
    total, H, K = q.shape
    if k.shape[1] != H:
        raise ValueError(f"k must carry the same head count as q ({H}), got {k.shape[1]}")
    HV, V = v.shape[1], v.shape[2]
    N = cu_seqlens.shape[0] - 1
    device = q.device
    cu = cu_seqlens.to(torch.int32).contiguous()
    _check_dtype("g", g, torch.float32)
    _check_dtype("beta", beta, torch.float32)
    if initial_state is not None:
        _check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    if d_final_state is not None:
        _check_dtype("d_final_state", d_final_state, torch.float32)
    s0 = initial_state.contiguous() if initial_state is not None else None
    dht = d_final_state.contiguous() if d_final_state is not None else None

    key = (
        total,
        N,
        H,
        HV,
        K,
        V,
        q.dtype,
        g.dtype,
        beta.dtype,
        s0.dtype if s0 is not None else None,
        dht.dtype if dht is not None else None,
        float(scale),
        bool(use_qk_l2norm_in_kernel),
        device,
    )
    if key not in _bwd_graph_cache:
        _bwd_graph_cache[key] = _build_bwd_graph(
            total,
            N,
            H,
            HV,
            K,
            V,
            _cudnn_dtype(q.dtype),
            _cudnn_dtype(g.dtype),
            _cudnn_dtype(beta.dtype),
            _cudnn_dtype(s0.dtype) if s0 is not None else None,
            _cudnn_dtype(dht.dtype) if dht is not None else None,
            float(scale),
            bool(use_qk_l2norm_in_kernel),
        )
    graph, t = _bwd_graph_cache[key]

    dq = torch.empty(total, H, K, dtype=q.dtype, device=device)
    dk = torch.empty(total, H, K, dtype=k.dtype, device=device)
    dv = torch.empty(total, HV, V, dtype=v.dtype, device=device)
    dg = torch.empty(total, HV, K, dtype=g.dtype, device=device)
    dbeta = torch.empty(total, HV, dtype=beta.dtype, device=device)
    variant_pack = {
        t["q"]: q.contiguous(),
        t["k"]: k.contiguous(),
        t["v"]: v.contiguous(),
        t["g"]: g.contiguous(),
        t["beta"]: beta.contiguous(),
        t["cu"]: cu,
        t["dO"]: dO.contiguous(),
        t["dQ"]: dq,
        t["dK"]: dk,
        t["dV"]: dv,
        t["dG"]: dg,
        t["dBeta"]: dbeta,
    }
    dh032 = None
    if s0 is not None:
        variant_pack[t["s0"]] = s0
        dh032 = torch.empty_like(s0)
        variant_pack[t["dS0"]] = dh032
    if dht is not None:
        variant_pack[t["dfs"]] = dht
    graph.execute(variant_pack, workspace=_graph_workspace(graph, device), handle=_graph_handle(device))
    dh0 = dh032 if dh032 is not None else torch.empty(0, dtype=torch.float32, device=device)
    return dq, dk, dv, dg, dbeta, dh0


@_kda_bwd.register_fake
def _kda_bwd_fake(dO, q, k, v, g, beta, cu_seqlens, scale, initial_state=None, d_final_state=None, use_qk_l2norm_in_kernel=False):
    dh0 = torch.empty_like(initial_state) if initial_state is not None else q.new_empty(0, dtype=torch.float32)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        dh0,
    )


# ---------------------------------------------------------------------------
# Autograd registration
# ---------------------------------------------------------------------------


def _kda_setup_context(ctx, inputs, output):
    q, k, v, g, beta, cu_seqlens, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel = inputs
    # save_for_backward cannot hold None; keep initial_state as an attribute.
    ctx.save_for_backward(q, k, v, g, beta, cu_seqlens)
    ctx.initial_state = initial_state
    ctx.scale = scale
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel


def _kda_backward(ctx, dO, dFinal):
    q, k, v, g, beta, cu_seqlens = ctx.saved_tensors
    initial_state = ctx.initial_state

    dht = dFinal if (dFinal is not None and dFinal.numel() > 0) else None
    dq, dk, dv, dg, dbeta, dh0 = torch.ops.cudnn.kimi_delta_attention_bwd(
        dO.contiguous(),
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        ctx.scale,
        initial_state=initial_state,
        d_final_state=dht,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
    )
    # q, k, v, g, beta, cu_seqlens, scale, initial_state, output_final_state,
    # use_qk_l2norm_in_kernel
    return (
        dq,
        dk,
        dv,
        dg,
        dbeta,
        None,
        None,
        dh0 if initial_state is not None else None,
        None,
        None,
    )


torch.library.register_autograd(
    f"{_OP_NAMESPACE}::{_OP_NAME}_fwd",
    _kda_backward,
    setup_context=_kda_setup_context,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kimi_delta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Kimi Delta Attention (KDA) linear attention.

    THD layout (matches the graph-API KDA node):

        q, k: ``[total_tokens, H, K]``; v: ``[total_tokens, HV, V]``
        g: ``[total_tokens, HV, K]`` (per-key-channel log decay);
        beta: ``[total_tokens, HV]`` (scalar); cu_seqlens: ``[N+1]`` int32
        initial_state / final_state: ``[N, HV, K, V]``

    A dense batch of N equal-length sequences is expressed as
    ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Dtypes are kernel-native and strict (callers convert): ``g``, ``beta``
    and the states are float32; ``final_state``, ``dG``, ``dBeta`` and
    ``d_initial_state`` are returned in float32.

    Args:
        g: per-key-channel log-space decay (``alpha = exp(g) in (0, 1]^K``).
        beta: per-token scalar write strength.
        cu_seqlens: ``[N+1]`` int32 sequence boundaries over the packed tokens.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state (otherwise zero).
        output_final_state: if ``True``, also return the per-sequence state
            after the last token.
        use_qk_l2norm_in_kernel: if ``True``, L2-normalize the q/k rows inside
            the kernel (the KDA model's feature map); if ``False``, pass q/k
            as given (the caller owns their conditioning).

    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v``. ``final_state`` is
        empty unless ``output_final_state=True``.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.ops.cudnn.kimi_delta_attention_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        float(scale),
        initial_state=initial_state,
        output_final_state=bool(output_final_state),
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
    )
