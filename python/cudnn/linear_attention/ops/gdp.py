# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch custom operator for Gated DeltaProduct (GDP) linear attention.

GDP applies ``num_householder`` beta-gated Householder updates per token with
one scalar decay per token: the GDN recurrence on an expanded sub-token
timeline, with the gate acting on sub-token 0 and the readout on sub-token
``n - 1``.  k/v/beta arrive already expanded at ``total_tokens * n`` rows;
q/g and the outputs O/dQ/dG live at real-token rows.  The op is a thin
adapter over cached single-node ``GDP`` / ``GDP_BWD`` pygraphs, served by
``GdpFrostEngine`` on the unmodified GDN kernels.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import cudnn

from .gdn import check_dtype, get_handle, graph_workspace, select_plan, torch_dtype_to_cudnn

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


fprop_cache: Dict[tuple, tuple] = {}
bprop_cache: Dict[tuple, tuple] = {}


def make_fprop_cache_key(
    total,
    N,
    H,
    HK,
    HV,
    K,
    V,
    num_householder,
    io_dtype,
    k_shape,
    v_shape,
    cu_dtype,
    scale,
    output_final_state,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    has_initial_state,
    checkpoint,
    device,
    plan_name,
):
    return (
        "gdp_fprop",
        total,
        N,
        H,
        HK,
        HV,
        K,
        V,
        num_householder,
        io_dtype,
        k_shape,
        v_shape,
        cu_dtype,
        float(scale),
        bool(output_final_state),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
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
    num_householder,
    io_dtype,
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
    device,
    plan_name,
):
    return (
        "gdp_bprop",
        total,
        N,
        H,
        HK,
        HV,
        K,
        V,
        num_householder,
        io_dtype,
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
        device,
        plan_name,
    )


def check_gdp_rows(op_name, total, num_householder, k, v, beta, g, dO=None) -> None:
    expanded = total * num_householder
    for name, t, want in (("k", k, expanded), ("v", v, expanded), ("beta", beta, expanded), ("g", g, total), ("dO", dO, total)):
        if t is not None and t.shape[0] != want:
            raise ValueError(f"{op_name}: {name} must carry {want} rows (q rows {'* num_householder' if want == expanded else ''}), got {t.shape[0]}")


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
    num_householder,
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
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    expanded = total * num_householder
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([expanded, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([expanded, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO], data_type=g_dtype, name="g")
    beta_t = graph.tensor([expanded, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    O_t, fs_t, state_checkpoints_t = graph.gdp(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        num_householder=num_householder,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        checkpoint_every_n_tokens=checkpoint,
        name="gdp",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        state0=state0_t,
        O=O_t,
        fs=fs_t,
        state_checkpoints=state_checkpoints_t,
    )


# ---------------------------------------------------------------------------
# Forward custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::gated_delta_product_fwd", mutates_args=())
def gdp_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_householder: int,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDP forward (internal): a cached single-node GDP pygraph, THD layout.

    Returns ``(o, final_state, state_checkpoints)``; ``final_state`` /
    ``state_checkpoints`` are zero-size tensors when ``output_final_state``
    is ``False`` / ``checkpoint_every_n_tokens`` is ``0``.
    """
    total, H, K = q.shape
    n = int(num_householder)
    if n < 1:
        raise ValueError(f"gated_delta_product: num_householder must be a positive integer, got {num_householder}")
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV}); canonical GQA shares grouped k/v heads")
    check_gdp_rows("gated_delta_product", total, n, k, v, beta, g)
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"gated_delta_product: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    cu = cu_seqlens
    check_dtype("g", g, torch.float32)
    if use_beta_sigmoid_in_kernel:
        check_dtype("beta", beta, q.dtype)
    else:
        check_dtype("beta", beta, torch.float32)
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
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"gated_delta_product: {tensor_name} must be on q's device ({device}); got {tensor.device}")
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
        n,
        q.dtype,
        tuple(k.shape),
        tuple(v.shape),
        cu_seqlens.dtype,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
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
            n,
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
        )
        select_plan(fprop_cache[cache_key][0], plan_name)

    graph, t = fprop_cache[cache_key]

    HO = max(H, HV)
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
    final_state = torch.empty(0, dtype=torch.float32, device=device)
    if output_final_state:
        final_state = torch.empty(N, HO, V, K, dtype=torch.float32, device=device)
        variant_pack[t["fs"]] = final_state
    state_checkpoints = torch.empty(0, dtype=q.dtype, device=device)
    if checkpoint > 0:
        total_checkpoints = max(total * n // checkpoint + N, 1)
        state_checkpoints = torch.empty(total_checkpoints, HO, V, K, dtype=q.dtype, device=device)
        variant_pack[t["state_checkpoints"]] = state_checkpoints
    graph.execute(variant_pack, workspace=graph_workspace(graph, device), handle=get_handle(device))
    return o, final_state, state_checkpoints


@gdp_fwd.register_fake
def gdp_fwd_fake(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    num_householder,
    scale,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
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
        raise ValueError(f"gated_delta_product: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    o = q.new_empty(total, HO, V)
    final = q.new_empty((N, HO, V, K) if output_final_state else (0,), dtype=torch.float32)
    if checkpoint_every_n_tokens > 0:
        total_checkpoints = max(total * int(num_householder) // int(checkpoint_every_n_tokens) + N, 1)
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
    num_householder,
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
):
    graph = cudnn.pygraph()
    HO = max(H, HV)
    expanded = total * num_householder
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([expanded, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([expanded, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO], data_type=g_dtype, name="g")
    beta_t = graph.tensor([expanded, HO], data_type=beta_dtype, name="beta")
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
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dstate0_t = graph.gdp_bwd(
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
        num_householder=num_householder,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        name="gdp_bwd",
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
        dQ=dQ_t,
        dK=dK_t,
        dV=dV_t,
        dG=dG_t,
        dBeta=dBeta_t,
        dstate0=dstate0_t,
        checkpoints=checkpoints_t,
    )


# ---------------------------------------------------------------------------
# Backward custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::gated_delta_product_bwd", mutates_args=())
def gdp_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_householder: int,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    d_final_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GDP backward (internal): a cached single-node GDP_BWD pygraph, THD layout.

    Returns ``(dq, dk, dv, dg, dbeta, d_initial_state)``; ``d_initial_state``
    is a zero-size tensor when ``initial_state`` is ``None``. With
    ``use_beta_sigmoid_in_kernel``, ``beta`` is io-dtype logits and ``dbeta``
    is the raw-logit gradient.
    """
    total, H, K = q.shape
    n = int(num_householder)
    if 0 in dO.stride():
        dO = dO.contiguous()
    if d_final_state is not None and 0 in d_final_state.stride():
        d_final_state = d_final_state.contiguous()
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV}); canonical GQA shares grouped k/v heads")
    check_gdp_rows("gated_delta_product", total, n, k, v, beta, g, dO=dO)
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"gated_delta_product: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    cu = cu_seqlens
    check_dtype("g", g, torch.float32)
    if use_beta_sigmoid_in_kernel:
        check_dtype("beta", beta, q.dtype)
    else:
        check_dtype("beta", beta, torch.float32)
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
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"gated_delta_product: {tensor_name} must be on q's device ({device}); got {tensor.device}")
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
        n,
        q.dtype,
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
            n,
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
        )
        select_plan(bprop_cache[cache_key][0], plan_name)

    graph, t = bprop_cache[cache_key]

    HO = max(H, HV)
    expanded = total * n
    dq = torch.empty(total, H, K, dtype=q.dtype, device=device)
    dk = torch.empty(expanded, HK, K, dtype=q.dtype, device=device)
    dv = torch.empty(expanded, HV, V, dtype=q.dtype, device=device)
    dg32 = torch.empty(total, HO, dtype=torch.float32, device=device)
    dbeta = torch.empty(expanded, HO, dtype=beta.dtype, device=device)
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
    graph.execute(variant_pack, workspace=graph_workspace(graph, device), handle=get_handle(device))
    if dstate0 is None:
        dstate0 = torch.empty(0, dtype=torch.float32, device=device)
    return dq, dk, dv, dg32, dbeta, dstate0


@gdp_bwd.register_fake
def gdp_bwd_fake(
    dO,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    num_householder,
    scale,
    initial_state=None,
    d_final_state=None,
    state_checkpoints=None,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    plan_name=None,
):
    dstate0 = torch.empty_like(initial_state) if initial_state is not None else q.new_empty(0, dtype=torch.float32)
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        dstate0,
    )


# ---------------------------------------------------------------------------
# Autograd registration
# ---------------------------------------------------------------------------


def gdp_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        num_householder,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        checkpoint_every_n_tokens,
        plan_name,
    ) = inputs
    saved = [q, k, v, g, beta, cu_seqlens]
    ctx.checkpoint_reuse = checkpoint_every_n_tokens == 64 and output[2].numel() > 0
    if ctx.checkpoint_reuse:
        saved.append(output[2])
    ctx.save_for_backward(*saved)
    ctx.initial_state = initial_state
    ctx.num_householder = num_householder
    ctx.scale = scale
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.batch_invariant = batch_invariant
    ctx.plan_name = plan_name
    ctx.use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def gdp_backward(ctx, dO, dFinal, dstate_checkpoints):
    if ctx.checkpoint_reuse:
        q, k, v, g, beta, cu_seqlens, state_checkpoints = ctx.saved_tensors[:7]
    else:
        q, k, v, g, beta, cu_seqlens = ctx.saved_tensors[:6]
        state_checkpoints = None
    initial_state = ctx.initial_state

    if dO is None:
        dO = torch.zeros(q.shape[0], max(q.shape[1], v.shape[1]), v.shape[2], dtype=q.dtype, device=q.device)
    dstate_in = dFinal if (dFinal is not None and dFinal.numel() > 0) else None
    dq, dk, dv, dg, dbeta, dstate0 = torch.ops.cudnn.gated_delta_product_bwd(
        dO,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        ctx.num_householder,
        ctx.scale,
        initial_state=initial_state,
        d_final_state=dstate_in,
        state_checkpoints=state_checkpoints,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
        batch_invariant=ctx.batch_invariant,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
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
        None,
        dstate0 if initial_state is not None else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "cudnn::gated_delta_product_fwd",
    gdp_backward,
    setup_context=gdp_setup_context,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gated_delta_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_householder: int,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
):
    """Gated DeltaProduct (GDP) linear attention.

    GDP applies ``n = num_householder`` beta-gated Householder updates per
    token with one scalar decay per token (the decay acts before the token's
    updates; the readout follows the last one).  THD layout on two timelines:

        q: ``[total_tokens, H, K]``; g: ``[total_tokens, HO]`` fp32;
        k: ``[total_tokens * n, HK, K]``; v: ``[total_tokens * n, HV, V]``;
        beta: ``[total_tokens * n, HO]``;
        cu_seqlens: ``[N+1]`` int32 over the real tokens; O and the states
        live at HO = max(H, HV) heads (initial_state / final_state:
        ``[N, HO, V, K]``)

    ``num_householder == 1`` is exactly ``gated_delta_net``.  Dtypes are
    kernel-native and strict: ``g`` and the states are float32; ``beta`` is
    float32, or io-dtype logits under ``use_beta_sigmoid_in_kernel`` (the
    fused sigmoid maps to ``(0, 1)``; for negative-eigenvalue GDP pass fp32
    ``beta = 2 * sigmoid(x)`` with the fusion off).

    Args:
        g: log-space scalar decay per real token (``alpha = exp(g) in (0, 1]``).
        beta: per-Householder write strength on the expanded rows.
        num_householder: Householder updates per token (``n >= 1``).
        checkpoint_every_n_tokens: counts EXPANDED sub-tokens and must be a
            positive multiple of the kernel chunk size, 64. ``64`` lets the
            backward reuse the series; a multiple of ``lcm(64, n)`` puts every
            checkpoint on a real-token boundary.
        plan_name: optionally pin one execution plan by name (``gdp_frost``).

    Other arguments and returns match :func:`gated_delta_net`.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    o, final_state, state_checkpoints = torch.ops.cudnn.gated_delta_product_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        int(num_householder),
        float(scale),
        initial_state=initial_state,
        output_final_state=bool(output_final_state),
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        batch_invariant=bool(batch_invariant),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        plan_name=plan_name,
    )
    if checkpoint_every_n_tokens > 0:
        return o, final_state, state_checkpoints
    return o, final_state
