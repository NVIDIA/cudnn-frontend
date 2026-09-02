# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch custom operator for Gated DeltaNet (GDN) linear attention.

GDN combines a DeltaNet beta-gated update with a scalar Mamba-2-style decay,
yielding the per-token recurrence

    S_t = alpha_t (I - beta_t k_t^T k_t) S_{t-1} + beta_t k_t^T v_t,
    o_t = q_t S_t,

where ``S_t`` is the recurrent state, ``alpha_t`` a scalar per-token decay
in ``(0, 1]``, and ``beta_t`` a scalar per-token write strength.

Layout follows the graph-API GDN node: THD — token-packed ``[total_tokens,
heads, dim]`` tensors plus ``cu_seqlens`` sequence boundaries.

The op is a thin adapter over the graph API: forward
and backward execute cached single-node ``GDN`` / ``GDN_BWD`` pygraphs.
Engine selection happens at graph planning time over the manifest's python
engines: ``GdnFrostEngine`` (default on SM100/SM103) with ``GdnCuTileEngine``
as the fallback everywhere else. Registered through
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
    """Caller-side workspace for a compiled graph (grow-only, held on the
    graph object itself — same lifetime by construction, no id()-keyed
    side table)."""
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
        raise TypeError(f"gated_delta_net: {name} must be {want} (kernel-native; callers convert), got {t.dtype}")


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
    has_initial_state,
    has_d_final_state,
    checkpoint_rows,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    safe_gate,
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
        bool(has_initial_state),
        bool(has_d_final_state),
        checkpoint_rows,
        float(scale),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(safe_gate),
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
    checkpoint,
    use_beta_sigmoid=False,
    safe_gate=False,
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    a_log_t = None
    dt_bias_t = None
    if safe_gate:
        a_log_t = graph.tensor([HO], data_type=cudnn.data_type.FLOAT, name="a_log")
        dt_bias_t = graph.tensor([HO], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    O_t, fs_t, state_checkpoints_t = graph.gdn(
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
        use_beta_sigmoid=use_beta_sigmoid or None,
        safe_gate=safe_gate or None,
        checkpoint_every_n_tokens=checkpoint,
        name="gdn",
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


@torch.library.custom_op("cudnn::gated_delta_net_fwd", mutates_args=())
def gdn_fwd(
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
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDN forward (internal): a cached single-node GDN pygraph, THD layout.

    Returns ``(o, final_state, state_checkpoints)``; ``final_state`` / ``state_checkpoints`` are zero-size
    tensors when ``output_final_state`` is ``False`` /
    ``checkpoint_every_n_tokens`` is ``0``.
    """
    total, H, K = q.shape
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV}); canonical GQA shares grouped k/v heads")
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"gated_delta_net: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    cu = cu_seqlens
    check_dtype("g", g, torch.float32)
    if use_beta_sigmoid_in_kernel:
        check_dtype("beta", beta, q.dtype)
    else:
        check_dtype("beta", beta, torch.float32)
    if safe_gate:
        if a_log is None or dt_bias is None:
            raise ValueError("gated_delta_net: safe_gate requires a_log and dt_bias")
        check_dtype("a_log", a_log, torch.float32)
        check_dtype("dt_bias", dt_bias, torch.float32)
    elif a_log is not None or dt_bias is not None:
        raise ValueError("gated_delta_net: a_log/dt_bias require safe_gate=True")
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
            raise ValueError(f"gated_delta_net: {tensor_name} must be on q's device ({device}); got {tensor.device}")
    g32 = g
    beta32 = beta
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
            checkpoint,
            use_beta_sigmoid=bool(use_beta_sigmoid_in_kernel),
            safe_gate=bool(safe_gate),
        )
        select_plan(fprop_cache[cache_key][0], plan_name)

    graph, t = fprop_cache[cache_key]

    HO = max(H, HV)
    o = torch.empty(total, HO, V, dtype=q.dtype, device=device)
    variant_pack = {
        t["q"]: q,
        t["k"]: k,
        t["v"]: v,
        t["g"]: g32,
        t["beta"]: beta32,
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


@gdn_fwd.register_fake
def gdn_fwd_fake(
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
        raise ValueError(f"gated_delta_net: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
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
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO], data_type=g_dtype, name="g")
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
        dt_bias_t = graph.tensor([HO], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dstate0_t, dA_t, dDt_t = graph.gdn_bwd(
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
        name="gdn_bwd",
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


@torch.library.custom_op("cudnn::gated_delta_net_bwd", mutates_args=())
def gdn_bwd(
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
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDN backward (internal): a cached single-node GDN_BWD pygraph, THD layout.

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
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"gated_delta_net: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    cu = cu_seqlens
    check_dtype("g", g, torch.float32)
    if use_beta_sigmoid_in_kernel:
        check_dtype("beta", beta, q.dtype)
    else:
        check_dtype("beta", beta, torch.float32)
    if safe_gate:
        if a_log is None or dt_bias is None:
            raise ValueError("gated_delta_net: safe_gate requires a_log and dt_bias")
        check_dtype("a_log", a_log, torch.float32)
        check_dtype("dt_bias", dt_bias, torch.float32)
    elif a_log is not None or dt_bias is not None:
        raise ValueError("gated_delta_net: a_log/dt_bias require safe_gate=True")
    if initial_state is not None:
        check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
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
            raise ValueError(f"gated_delta_net: {tensor_name} must be on q's device ({device}); got {tensor.device}")
    g32 = g
    beta32 = beta
    state0 = initial_state if initial_state is not None else None
    if d_final_state is not None:
        check_dtype("d_final_state", d_final_state, torch.float32)
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
        state0 is not None,
        dstate_in is not None,
        state_checkpoints.shape[0] if state_checkpoints is not None else None,
        scale,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        safe_gate,
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
            cudnn.data_type.FLOAT,
            torch_dtype_to_cudnn(beta.dtype),
            cudnn.data_type.FLOAT if state0 is not None else None,
            cudnn.data_type.FLOAT if dstate_in is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            state_checkpoints.shape[0] if state_checkpoints is not None else None,
            float(scale),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            use_beta_sigmoid=bool(use_beta_sigmoid_in_kernel),
            safe_gate=bool(safe_gate),
        )
        select_plan(bprop_cache[cache_key][0], plan_name)

    graph, t = bprop_cache[cache_key]

    HO = max(H, HV)
    dq = torch.empty(total, H, K, dtype=q.dtype, device=device)
    dk = torch.empty(total, HK, K, dtype=q.dtype, device=device)
    dv = torch.empty(total, HV, V, dtype=q.dtype, device=device)
    dg32 = torch.empty(total, HO, dtype=torch.float32, device=device)
    dbeta = torch.empty(total, HO, dtype=beta.dtype, device=device)
    variant_pack = {
        t["q"]: q,
        t["k"]: k,
        t["v"]: v,
        t["g"]: g32,
        t["beta"]: beta32,
        t["cu"]: cu,
        t["dO"]: dO,
        t["dQ"]: dq,
        t["dK"]: dk,
        t["dV"]: dv,
        t["dG"]: dg32,
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
        d_dt_bias = torch.empty(HO, dtype=torch.float32, device=device)
        variant_pack[t["d_a_log"]] = d_a_log
        variant_pack[t["d_dt_bias"]] = d_dt_bias
    graph.execute(variant_pack, workspace=graph_workspace(graph, device), handle=get_handle(device))
    if dstate0 is None:
        dstate0 = torch.empty(0, dtype=torch.float32, device=device)
    return dq, dk, dv, dg32, dbeta, dstate0, d_a_log, d_dt_bias


@gdn_bwd.register_fake
def gdn_bwd_fake(
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
    a_log=None,
    dt_bias=None,
    plan_name=None,
):
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("gated_delta_net: safe_gate requires a_log and dt_bias")
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


def gdn_setup_context(ctx, inputs, output):
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
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        plan_name,
    ) = inputs
    saved = [q, k, v, g, beta, cu_seqlens]
    ctx.checkpoint_reuse = checkpoint_every_n_tokens == 64 and output[2].numel() > 0
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
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def gdn_backward(ctx, dO, dFinal, dstate_checkpoints):
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
    dq, dk, dv, dg, dbeta, dstate0, d_a_log, d_dt_bias = torch.ops.cudnn.gated_delta_net_bwd(
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
        d_a_log if ctx.safe_gate else None,
        d_dt_bias if ctx.safe_gate else None,
        None,
        None,
    )


torch.library.register_autograd(
    "cudnn::gated_delta_net_fwd",
    gdn_backward,
    setup_context=gdn_setup_context,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gated_delta_net(
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
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
):
    """Gated DeltaNet (GDN) linear attention.

    THD layout (matches the graph-API GDN node):

        q: ``[total_tokens, H, K]``; k: ``[total_tokens, HK, K]`` (HK = H, or
        HK = HV for canonical GQA: grouped K/V heads shared across query groups); v: ``[total_tokens, HV, V]``
        g, beta: ``[total_tokens, HO]`` with ``HO = max(H, HV)``;
        cu_seqlens: ``[N+1]`` int32; O and the states live at HO heads
        (initial_state / final_state: ``[N, HO, V, K]``)

    A dense batch of N equal-length sequences is expressed as
    ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Dtypes are kernel-native and strict (callers convert): ``g`` and the
    states are float32; ``final_state``, ``dG`` and ``d_initial_state`` are
    returned in float32.  ``beta`` and ``dBeta`` are float32, or io dtype
    under ``use_beta_sigmoid_in_kernel``.

    Args:
        g: log-space scalar decay per token (``alpha = exp(g) in (0, 1]``),
            or raw pre-activation logits when ``safe_gate=True``.
        beta: per-token write strength (float32), or io-dtype logits when
            ``use_beta_sigmoid_in_kernel=True``.
        cu_seqlens: ``[N+1]`` int32 sequence boundaries over the packed tokens.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state (otherwise zero).
        output_final_state: if ``True``, also return the per-sequence state
            after the last token.
        use_qk_l2norm_in_kernel: if ``True``, L2-normalize the Q/K rows inside
            the kernel. Engines that cannot honor it decline the graph.
        batch_invariant: if ``True``, each sequence's results are bitwise
            independent of the batch composition (whole-sequence scheduling;
            disables split-K load balancing).
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` inside the kernel.
        safe_gate: interpret ``g`` through the safe-gate transform
            ``-exp(a_log) * softplus(g + dt_bias)``. Requires ``a_log`` and
            ``dt_bias``.
        a_log: ``[HO]`` float32 safe-gate per-head log-amplitude.
        dt_bias: ``[HO]`` float32 safe-gate per-head bias.
        checkpoint_every_n_tokens: if ``> 0``, also return the per-chunk
            recurrent state series ``state_checkpoints`` (``[total_checkpoints, HO, V, K]`` io dtype,
            one entry per N tokens strictly before each sequence end; the
            FROST engine requires a positive multiple of the kernel chunk size, 64). The series is
            a non-differentiable dump.

        plan_name: optionally pin one execution plan by name (the plan
            API's ``get_plan_name_at_index`` names, e.g. ``gdn_frost``); a
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
    o, final_state, state_checkpoints = torch.ops.cudnn.gated_delta_net_fwd(
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
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        plan_name=plan_name,
    )
    if checkpoint_every_n_tokens > 0:
        return o, final_state, state_checkpoints
    return o, final_state
