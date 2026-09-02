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

The op is a thin adapter over the graph API: forward
and backward execute cached single-node ``KDA`` / ``KDA_BWD`` pygraphs.
Engine selection happens at graph planning time over the manifest's python
engines: ``KdaFrostEngine`` (default on SM100/SM103, forward and backward)
with ``KdaCuTileEngine`` as the fallback. Registered through
``torch.library.custom_op`` so it composes with autograd, ``torch.compile``,
and DDP.

Graph caching ensures cuDNN graphs are built once per unique configuration
and reused across calls.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import cudnn

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
    torch.int32: cudnn.data_type.INT32,
    torch.int64: cudnn.data_type.INT64,
}

fprop_cache: Dict[tuple, tuple] = {}
bprop_cache: Dict[tuple, tuple] = {}
cudnn_handles: Dict[int, int] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select_plan(graph, plan_name):
    """Pin one execution plan by name on a freshly built graph (create the
    plans, select by name, check support); ``None`` keeps default routing."""
    if plan_name is None:
        return
    graph.create_execution_plans()
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    matches = [i for i, n in enumerate(names) if n == plan_name or n.startswith(plan_name + "[")]
    if not matches:
        raise cudnn.cudnnGraphNotSupportedError(f"no {plan_name} plan for this graph (offered: {names})")
    graph.select_plan(matches[0])
    graph.check_support()


def graph_workspace(graph, device):
    """Caller-side workspace for a compiled graph."""
    if not graph._is_built:
        if not graph._planning_done:
            graph.create_execution_plans()
        if graph.selected_engine is None:
            graph.build()
        else:
            graph.build_plans()
    size = graph.get_workspace_size()
    workspace = getattr(graph, "la_ops_workspace", None)
    if workspace is None or workspace.numel() < size or workspace.device != device:
        workspace = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
        graph.la_ops_workspace = workspace
    return workspace


def get_handle(device):
    """Per-device cuDNN handle carrying the caller's current stream."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    handle = cudnn_handles.get(idx)
    if handle is None:
        with torch.cuda.device(idx):
            handle = cudnn.create_handle()
        cudnn_handles[idx] = handle
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


def torch_dtype_to_cudnn(dtype: torch.dtype):
    """Map a PyTorch dtype to a cuDNN data_type enum."""
    return TORCH_DTYPE_TO_CUDNN[dtype]


def check_dtype(name, t, want) -> None:
    if t.dtype != want:
        raise TypeError(f"kimi_delta_attention: {name} must be {want} (kernel-native; callers convert), got {t.dtype}")


def make_fprop_cache_key(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    io_dtype,
    k_dtype,
    v_dtype,
    k_shape,
    v_shape,
    cu_dtype,
    scale,
    output_final_state,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    safe_gate,
    gate_lower_bound,
    has_initial_state,
    checkpoint,
    device,
    plan_name,
):
    return (
        "fprop",
        total,
        N,
        H,
        HK,
        HV,
        K,
        V,
        io_dtype,
        k_dtype,
        v_dtype,
        k_shape,
        v_shape,
        cu_dtype,
        float(scale),
        bool(output_final_state),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(safe_gate),
        float(gate_lower_bound) if gate_lower_bound is not None else None,
        bool(has_initial_state),
        checkpoint,
        device,
        plan_name,
    )


def make_bprop_cache_key(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    io_dtype,
    k_dtype,
    v_dtype,
    do_dtype,
    k_shape,
    v_shape,
    cu_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    dstate_in_dtype,
    checkpoint_rows,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    safe_gate,
    gate_lower_bound,
    device,
    plan_name,
):
    return (
        "bprop",
        total,
        N,
        H,
        HK,
        HV,
        K,
        V,
        io_dtype,
        k_dtype,
        v_dtype,
        do_dtype,
        k_shape,
        v_shape,
        cu_dtype,
        g_dtype,
        beta_dtype,
        state_dtype,
        dstate_in_dtype,
        checkpoint_rows,
        float(scale),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(safe_gate),
        float(gate_lower_bound) if gate_lower_bound is not None else None,
        device,
        plan_name,
    )


# ---------------------------------------------------------------------------
# Forward graph builder
# ---------------------------------------------------------------------------


def build_fprop_graph(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    io_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    cu_dtype,
    scale,
    output_final_state,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    safe_gate,
    gate_lower_bound,
    checkpoint,
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    a_log_t = None
    dt_bias_t = None
    if safe_gate:
        a_log_t = graph.tensor([HO], data_type=cudnn.data_type.FLOAT, name="a_log")
        dt_bias_t = graph.tensor([HO, K], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    O_t, fs_t, state_checkpoints_t = graph.kda(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        checkpoint_every_n_tokens=checkpoint,
        name="kda",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        state0=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        O=O_t,
        fs=fs_t,
        state_checkpoints=state_checkpoints_t,
    )


# ---------------------------------------------------------------------------
# Forward custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::kimi_delta_attention_fwd", mutates_args=())
def kda_fwd(
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
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDA forward (internal): a cached single-node KDA pygraph, THD layout.

    Returns ``(o, final_state, state_checkpoints)``; ``final_state`` / ``state_checkpoints`` are zero-size
    tensors when ``output_final_state`` is ``False`` /
    ``checkpoint_every_n_tokens`` is ``0``.
    """
    total, H, K = q.shape
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV}); canonical GQA shares grouped k/v heads")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    cu = cu_seqlens
    check_dtype("g", g, torch.float32)
    if use_beta_sigmoid_in_kernel:
        check_dtype("beta", beta, q.dtype)
    else:
        check_dtype("beta", beta, torch.float32)
    if safe_gate:
        if a_log is None or dt_bias is None:
            raise ValueError("kimi_delta_attention: safe_gate requires a_log and dt_bias")
        check_dtype("a_log", a_log, torch.float32)
        check_dtype("dt_bias", dt_bias, torch.float32)
    elif a_log is not None or dt_bias is not None:
        raise ValueError("kimi_delta_attention: a_log/dt_bias require safe_gate=True")
    if initial_state is not None:
        check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    for tensor_name, tensor in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu_seqlens),
        ("initial_state", initial_state),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"kimi_delta_attention: {tensor_name} must be on q's device ({device}); got {tensor.device}")
    state0 = initial_state if initial_state is not None else None
    checkpoint = int(checkpoint_every_n_tokens)

    cache_key = make_fprop_cache_key(
        total,
        N,
        H,
        HK,
        HV,
        K,
        V,
        q.dtype,
        k.dtype,
        v.dtype,
        tuple(k.shape),
        tuple(v.shape),
        cu_seqlens.dtype,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        safe_gate,
        gate_lower_bound,
        state0 is not None,
        checkpoint,
        device,
        plan_name,
    )
    if cache_key not in fprop_cache:
        fprop_cache[cache_key] = build_fprop_graph(
            total,
            N,
            H,
            HK,
            HV,
            K,
            V,
            torch_dtype_to_cudnn(q.dtype),
            cudnn.data_type.FLOAT,
            torch_dtype_to_cudnn(beta.dtype),
            cudnn.data_type.FLOAT if state0 is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            float(scale),
            bool(output_final_state),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            bool(use_beta_sigmoid_in_kernel),
            bool(safe_gate),
            float(gate_lower_bound) if gate_lower_bound is not None else None,
            checkpoint,
        )
        select_plan(fprop_cache[cache_key][0], plan_name)

    graph, t = fprop_cache[cache_key]

    o = torch.empty(total, HO, V, dtype=q.dtype, device=device)
    variant_pack = {
        t["q"]: q,
        t["k"]: k,
        t["v"]: v,
        t["g"]: g,
        t["beta"]: beta,
        t["cu"]: cu,
        t["O"]: o,
    }
    if state0 is not None:
        variant_pack[t["state0"]] = state0
    if safe_gate:
        variant_pack[t["a_log"]] = a_log
        variant_pack[t["dt_bias"]] = dt_bias
    final_state = torch.empty(0, dtype=torch.float32, device=device)
    if output_final_state:
        final_state = torch.empty(N, HO, V, K, dtype=torch.float32, device=device)
        variant_pack[t["fs"]] = final_state
    state_checkpoints = torch.empty(0, dtype=q.dtype, device=device)
    if checkpoint > 0:
        total_checkpoints = max(total // checkpoint + N, 1)
        state_checkpoints = torch.empty(total_checkpoints, HO, V, K, dtype=q.dtype, device=device)
        variant_pack[t["state_checkpoints"]] = state_checkpoints
    graph.execute(variant_pack, workspace=graph_workspace(graph, device), handle=get_handle(device))
    return o, final_state, state_checkpoints


@kda_fwd.register_fake
def kda_fwd_fake(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    safe_gate=False,
    gate_lower_bound=None,
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    plan_name: Optional[str] = None,
):
    total, H, K = q.shape
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV}); canonical GQA shares grouped k/v heads")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    o = q.new_empty(total, HO, V)
    final = q.new_empty((N, HO, V, K) if output_final_state else (0,), dtype=torch.float32)
    if checkpoint_every_n_tokens > 0:
        total_checkpoints = max(total // int(checkpoint_every_n_tokens) + N, 1)
        state_checkpoints = q.new_empty(total_checkpoints, HO, V, K)
    else:
        state_checkpoints = q.new_empty(0)
    return o, final, state_checkpoints


# ---------------------------------------------------------------------------
# Backward graph builder
# ---------------------------------------------------------------------------


def build_bprop_graph(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    io_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    dstate_in_dtype,
    cu_dtype,
    checkpoint_rows,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid=False,
    safe_gate=False,
    gate_lower_bound=None,
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    dO_t = graph.tensor([total, HO, V], data_type=io_dtype, name="dO")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    dfs_t = None
    if dstate_in_dtype is not None:
        dfs_t = graph.tensor([N, HO, V, K], data_type=dstate_in_dtype, name="d_final_state")
    checkpoints_t = None
    if checkpoint_rows is not None:
        checkpoints_t = graph.tensor([checkpoint_rows, HO, V, K], data_type=io_dtype, name="state_checkpoints")
    a_log_t = None
    dt_bias_t = None
    if safe_gate:
        a_log_t = graph.tensor([HO], data_type=cudnn.data_type.FLOAT, name="a_log")
        dt_bias_t = graph.tensor([HO, K], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dstate0_t, dA_t, dDt_t = graph.kda_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=dO_t,
        state_checkpoints=checkpoints_t,
        initial_state=state0_t,
        d_final_state=dfs_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        safe_gate=safe_gate or None,
        gate_lower_bound=gate_lower_bound,
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
        state0=state0_t,
        dfs=dfs_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        dQ=dQ_t,
        dK=dK_t,
        dV=dV_t,
        dG=dG_t,
        dBeta=dBeta_t,
        dstate0=dstate0_t,
        d_a_log=dA_t,
        d_dt_bias=dDt_t,
        checkpoints=checkpoints_t,
    )


# ---------------------------------------------------------------------------
# Backward custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::kimi_delta_attention_bwd", mutates_args=())
def kda_bwd(
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
    state_checkpoints: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KDA backward (internal): a cached single-node KDA_BWD pygraph, THD layout.

    ``state_checkpoints`` is the forward's per-chunk state series (io dtype,
    chunk cadence); when given, the engine consumes it instead of running
    the checkpoint recompute pass. Returns ``(dq, dk, dv, dg, dbeta,
    d_initial_state, d_a_log, d_dt_bias)``; ``d_initial_state`` is a
    zero-size tensor when ``initial_state`` is ``None``, and ``d_a_log`` /
    ``d_dt_bias`` are zero-size tensors unless ``safe_gate``. With
    ``safe_gate``, ``g`` is the raw logits and ``dg`` is the raw-logit
    gradient; with ``use_beta_sigmoid_in_kernel``, ``beta`` is io-dtype
    logits and ``dbeta`` is the raw-logit gradient.
    """
    total, H, K = q.shape
    if 0 in dO.stride():
        dO = dO.contiguous()
    if d_final_state is not None and 0 in d_final_state.stride():
        d_final_state = d_final_state.contiguous()
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV}); canonical GQA shares grouped k/v heads")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    cu = cu_seqlens
    check_dtype("g", g, torch.float32)
    if use_beta_sigmoid_in_kernel:
        check_dtype("beta", beta, q.dtype)
    else:
        check_dtype("beta", beta, torch.float32)
    if safe_gate:
        if a_log is None or dt_bias is None:
            raise ValueError("kimi_delta_attention: safe_gate requires a_log and dt_bias")
        check_dtype("a_log", a_log, torch.float32)
        check_dtype("dt_bias", dt_bias, torch.float32)
    elif a_log is not None or dt_bias is not None:
        raise ValueError("kimi_delta_attention: a_log/dt_bias require safe_gate=True")
    if initial_state is not None:
        check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    if d_final_state is not None:
        check_dtype("d_final_state", d_final_state, torch.float32)
    if state_checkpoints is not None:
        check_dtype("state_checkpoints", state_checkpoints, q.dtype)
    for tensor_name, tensor in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu_seqlens),
        ("dO", dO),
        ("d_final_state", d_final_state),
        ("state_checkpoints", state_checkpoints),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"kimi_delta_attention: {tensor_name} must be on q's device ({device}); got {tensor.device}")
    state0 = initial_state if initial_state is not None else None
    dstate_in = d_final_state if d_final_state is not None else None

    cache_key = make_bprop_cache_key(
        total,
        N,
        H,
        HK,
        HV,
        K,
        V,
        q.dtype,
        k.dtype,
        v.dtype,
        dO.dtype,
        tuple(k.shape),
        tuple(v.shape),
        cu_seqlens.dtype,
        g.dtype,
        beta.dtype,
        state0.dtype if state0 is not None else None,
        dstate_in.dtype if dstate_in is not None else None,
        state_checkpoints.shape[0] if state_checkpoints is not None else None,
        scale,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        safe_gate,
        gate_lower_bound,
        device,
        plan_name,
    )
    if cache_key not in bprop_cache:
        bprop_cache[cache_key] = build_bprop_graph(
            total,
            N,
            H,
            HK,
            HV,
            K,
            V,
            torch_dtype_to_cudnn(q.dtype),
            torch_dtype_to_cudnn(g.dtype),
            torch_dtype_to_cudnn(beta.dtype),
            torch_dtype_to_cudnn(state0.dtype) if state0 is not None else None,
            torch_dtype_to_cudnn(dstate_in.dtype) if dstate_in is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            state_checkpoints.shape[0] if state_checkpoints is not None else None,
            float(scale),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            use_beta_sigmoid=bool(use_beta_sigmoid_in_kernel),
            safe_gate=bool(safe_gate),
            gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
        )
        select_plan(bprop_cache[cache_key][0], plan_name)

    graph, t = bprop_cache[cache_key]

    dq = torch.empty(total, H, K, dtype=q.dtype, device=device)
    dk = torch.empty(total, HK, K, dtype=q.dtype, device=device)
    dv = torch.empty(total, HV, V, dtype=q.dtype, device=device)
    dg = torch.empty(total, HO, K, dtype=g.dtype, device=device)
    dbeta = torch.empty(total, HO, dtype=beta.dtype, device=device)
    variant_pack = {
        t["q"]: q,
        t["k"]: k,
        t["v"]: v,
        t["g"]: g,
        t["beta"]: beta,
        t["cu"]: cu,
        t["dO"]: dO,
        t["dQ"]: dq,
        t["dK"]: dk,
        t["dV"]: dv,
        t["dG"]: dg,
        t["dBeta"]: dbeta,
    }
    dstate0 = None
    if state0 is not None:
        variant_pack[t["state0"]] = state0
        dstate0 = torch.empty_like(state0)
        variant_pack[t["dstate0"]] = dstate0
    if dstate_in is not None:
        variant_pack[t["dfs"]] = dstate_in
    if state_checkpoints is not None:
        variant_pack[t["checkpoints"]] = state_checkpoints
    d_a_log = torch.empty(0, dtype=torch.float32, device=device)
    d_dt_bias = torch.empty(0, dtype=torch.float32, device=device)
    if safe_gate:
        variant_pack[t["a_log"]] = a_log
        variant_pack[t["dt_bias"]] = dt_bias
        d_a_log = torch.empty(HO, dtype=torch.float32, device=device)
        d_dt_bias = torch.empty(HO, K, dtype=torch.float32, device=device)
        variant_pack[t["d_a_log"]] = d_a_log
        variant_pack[t["d_dt_bias"]] = d_dt_bias
    graph.execute(variant_pack, workspace=graph_workspace(graph, device), handle=get_handle(device))
    if dstate0 is None:
        dstate0 = torch.empty(0, dtype=torch.float32, device=device)
    return dq, dk, dv, dg, dbeta, dstate0, d_a_log, d_dt_bias


@kda_bwd.register_fake
def kda_bwd_fake(
    dO,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state=None,
    d_final_state=None,
    state_checkpoints=None,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    safe_gate=False,
    gate_lower_bound=None,
    a_log=None,
    dt_bias=None,
    plan_name=None,
):
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("kimi_delta_attention: safe_gate requires a_log and dt_bias")
    dstate0 = torch.empty_like(initial_state) if initial_state is not None else q.new_empty(0, dtype=torch.float32)
    d_a_log = torch.empty_like(a_log) if safe_gate else q.new_empty(0, dtype=torch.float32)
    d_dt_bias = torch.empty_like(dt_bias) if safe_gate else q.new_empty(0, dtype=torch.float32)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        dstate0,
        d_a_log,
        d_dt_bias,
    )


# ---------------------------------------------------------------------------
# Autograd registration
# ---------------------------------------------------------------------------


def kda_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        safe_gate,
        gate_lower_bound,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        plan_name,
    ) = inputs
    saved = [q, k, v, g, beta, cu_seqlens]
    ctx.checkpoint_reuse = checkpoint_every_n_tokens == 16 and output[2].numel() > 0
    if ctx.checkpoint_reuse:
        saved.append(output[2])
    if safe_gate:
        saved.extend([a_log, dt_bias])
    ctx.save_for_backward(*saved)
    ctx.initial_state = initial_state
    ctx.scale = scale
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.batch_invariant = batch_invariant
    ctx.plan_name = plan_name
    ctx.use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
    ctx.safe_gate = bool(safe_gate)
    ctx.gate_lower_bound = gate_lower_bound
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def kda_backward(ctx, dO, dFinal, dstate_checkpoints):
    a_log = dt_bias = None
    if ctx.safe_gate:
        a_log, dt_bias = ctx.saved_tensors[-2:]
    if ctx.checkpoint_reuse:
        q, k, v, g, beta, cu_seqlens, state_checkpoints = ctx.saved_tensors[:7]
    else:
        q, k, v, g, beta, cu_seqlens = ctx.saved_tensors[:6]
        state_checkpoints = None
    initial_state = ctx.initial_state

    if dO is None:
        dO = torch.zeros(q.shape[0], max(q.shape[1], v.shape[1]), v.shape[2], dtype=q.dtype, device=q.device)
    dstate_in = dFinal if (dFinal is not None and dFinal.numel() > 0) else None
    dq, dk, dv, dg, dbeta, dstate0, d_a_log, d_dt_bias = torch.ops.cudnn.kimi_delta_attention_bwd(
        dO,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        ctx.scale,
        initial_state=initial_state,
        d_final_state=dstate_in,
        state_checkpoints=state_checkpoints,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
        batch_invariant=ctx.batch_invariant,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
        safe_gate=ctx.safe_gate,
        gate_lower_bound=ctx.gate_lower_bound,
        a_log=a_log,
        dt_bias=dt_bias,
        plan_name=ctx.plan_name,
    )
    return (
        dq,
        dk,
        dv,
        dg,
        dbeta,
        None,
        None,
        dstate0 if initial_state is not None else None,
        None,
        None,
        None,
        None,
        None,
        None,
        d_a_log if ctx.safe_gate else None,
        d_dt_bias if ctx.safe_gate else None,
        None,
        None,
    )


torch.library.register_autograd(
    "cudnn::kimi_delta_attention_fwd",
    kda_backward,
    setup_context=kda_setup_context,
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
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
):
    """Kimi Delta Attention (KDA) linear attention.

    THD layout (matches the graph-API KDA node):

        q: ``[total_tokens, H, K]``; k: ``[total_tokens, HK, K]`` (HK = H, or
        HK = HV for canonical GQA: grouped K/V heads shared across query groups); v: ``[total_tokens, HV, V]``
        g: ``[total_tokens, HO, K]`` (per-key-channel log decay);
        beta: ``[total_tokens, HO]`` (scalar); cu_seqlens: ``[N+1]`` int32
        initial_state / final_state: ``[N, HO, V, K]``
        (``HO = max(H, HV)``: the gates, output, and state heads)

    A dense batch of N equal-length sequences is expressed as
    ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Dtypes are kernel-native and strict (callers convert): ``g`` and the
    states are float32; ``final_state``, ``dG`` and ``d_initial_state`` are
    returned in float32.  ``beta`` and ``dBeta`` are float32, or io dtype
    under ``use_beta_sigmoid_in_kernel``.

    Args:
        g: per-key-channel log-space decay (``alpha = exp(g) in (0, 1]^K``),
            or raw pre-activation logits when ``safe_gate=True``.
        beta: per-token scalar write strength (float32 post-sigmoid), or
            io-dtype logits when ``use_beta_sigmoid_in_kernel=True``.
        cu_seqlens: ``[N+1]`` int32 sequence boundaries over the packed tokens.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state (otherwise zero).
        output_final_state: if ``True``, also return the per-sequence state
            after the last token.
        use_qk_l2norm_in_kernel: if ``True``, L2-normalize the Q/K rows inside
            the kernel (the KDA model's feature map); if ``False``, pass Q/K
            as given (the caller owns their conditioning).
        batch_invariant: if ``True``, each sequence's results are bitwise
            independent of the batch composition (whole-sequence scheduling;
            disables split-K load balancing).
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` inside the kernel;
            the backward returns the raw-logit beta gradient.
        safe_gate: interpret ``g`` through the safe-gate transform
            ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``.
            Requires ``a_log`` and ``dt_bias``; the backward returns the
            raw-logit ``g`` gradient plus ``a_log`` / ``dt_bias`` gradients.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        a_log: ``[HO]`` float32 safe-gate per-head log-amplitude.
        dt_bias: ``[HO, K]`` float32 safe-gate channel bias.
        checkpoint_every_n_tokens: if ``> 0``, also return the per-chunk
            recurrent state series ``state_checkpoints`` (``[total_checkpoints, HO, V, K]`` io dtype,
            one entry per N tokens strictly before each sequence end; the
            FROST engine requires a positive multiple of the kernel chunk size, 16). The series is
            a non-differentiable dump.

        plan_name: optionally pin one execution plan by name (the plan
            API's ``get_plan_name_at_index`` names, e.g. ``kda_frost``); a
            graph offering no such plan raises ``cudnnGraphNotSupportedError``.
    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v``, or
        ``(o, final_state, state_checkpoints)`` when ``checkpoint_every_n_tokens > 0``.
        ``final_state`` is empty unless ``output_final_state=True``.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    o, final_state, state_checkpoints = torch.ops.cudnn.kimi_delta_attention_fwd(
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
        batch_invariant=bool(batch_invariant),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        safe_gate=bool(safe_gate),
        gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        plan_name=plan_name,
    )
    if checkpoint_every_n_tokens > 0:
        return o, final_state, state_checkpoints
    return o, final_state
