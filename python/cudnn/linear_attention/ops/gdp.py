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
    g_dtype,
    beta_dtype,
    state_dtype,
    k_shape,
    v_shape,
    cu_dtype,
    scale,
    output_final_state,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    a_log_dtype,
    dt_bias_dtype,
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
        g_dtype,
        beta_dtype,
        state_dtype,
        k_shape,
        v_shape,
        cu_dtype,
        float(scale),
        bool(output_final_state),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(safe_gate),
        a_log_dtype,
        dt_bias_dtype,
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
    g_dtype,
    beta_dtype,
    state_dtype,
    dstate_in_dtype,
    k_shape,
    v_shape,
    cu_dtype,
    checkpoint_rows,
    checkpoint_every_n_tokens,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    a_log_dtype,
    dt_bias_dtype,
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
        g_dtype,
        beta_dtype,
        state_dtype,
        dstate_in_dtype,
        k_shape,
        v_shape,
        cu_dtype,
        checkpoint_rows,
        int(checkpoint_every_n_tokens),
        float(scale),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(safe_gate),
        a_log_dtype,
        dt_bias_dtype,
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
    allow_neg_eigval=False,
    safe_gate=False,
    a_log_dtype=None,
    dt_bias_dtype=None,
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
    a_log_t = dt_bias_t = None
    if safe_gate:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
        dt_bias_t = graph.tensor([HO], data_type=dt_bias_dtype, name="dt_bias")
    O_t, fs_t, state_checkpoints_t = graph.gdp(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        num_householder=num_householder,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        allow_neg_eigval=allow_neg_eigval or None,
        safe_gate=safe_gate or None,
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
        a_log=a_log_t,
        dt_bias=dt_bias_t,
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
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
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
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("gated_delta_product: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("gated_delta_product: safe_gate requires a_log and dt_bias")
    if not safe_gate and (a_log is not None or dt_bias is not None):
        raise ValueError("gated_delta_product: a_log/dt_bias require safe_gate=True")
    if safe_gate:
        check_dtype("a_log", a_log, (torch.float32, torch.bfloat16, torch.float16))
        check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
    cu = cu_seqlens
    check_dtype("g", g, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("beta", beta, q.dtype if use_beta_sigmoid_in_kernel else (torch.float32, q.dtype))
    if initial_state is not None:
        check_dtype("initial_state", initial_state, (torch.float32, torch.bfloat16))
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
            raise ValueError(f"gated_delta_product: {tensor_name} must be on q's device ({device}); got {tensor.device}")
    state0 = initial_state if initial_state is not None else None
    state_out_dtype = state0.dtype if state0 is not None else torch.float32
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
        g.dtype,
        beta.dtype,
        state0.dtype if state0 is not None else None,
        tuple(k.shape),
        tuple(v.shape),
        cu_seqlens.dtype,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        a_log.dtype if safe_gate else None,
        dt_bias.dtype if safe_gate else None,
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
            torch_dtype_to_cudnn(g.dtype),
            torch_dtype_to_cudnn(beta.dtype),
            torch_dtype_to_cudnn(state0.dtype) if state0 is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            float(scale),
            bool(output_final_state),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            checkpoint,
            use_beta_sigmoid=bool(use_beta_sigmoid_in_kernel),
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            a_log_dtype=torch_dtype_to_cudnn(a_log.dtype) if safe_gate else None,
            dt_bias_dtype=torch_dtype_to_cudnn(dt_bias.dtype) if safe_gate else None,
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
    if safe_gate:
        variant_pack[t["a_log"]] = a_log
        variant_pack[t["dt_bias"]] = dt_bias
    final_state = torch.empty(0, dtype=state_out_dtype, device=device)
    if output_final_state:
        final_state = torch.empty(N, HO, V, K, dtype=state_out_dtype, device=device)
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
    allow_neg_eigval=False,
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
        raise ValueError(f"gated_delta_product: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("gated_delta_product: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    o = q.new_empty(total, HO, V)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    final = q.new_empty((N, HO, V, K) if output_final_state else (0,), dtype=state_dtype)
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
    allow_neg_eigval=False,
    safe_gate=False,
    a_log_dtype=None,
    dt_bias_dtype=None,
    checkpoint_every_n_tokens=0,
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
    a_log_t = dt_bias_t = None
    if safe_gate:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
        dt_bias_t = graph.tensor([HO], data_type=dt_bias_dtype, name="dt_bias")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dstate0_t, d_a_log_t, d_dt_bias_t = graph.gdp_bwd(
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
        num_householder=num_householder,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens or None,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        allow_neg_eigval=allow_neg_eigval or None,
        safe_gate=safe_gate or None,
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
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        d_a_log=d_a_log_t,
        d_dt_bias=d_dt_bias_t,
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
    checkpoint_every_n_tokens: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    check_dtype("dO", dO, q.dtype)
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
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("gated_delta_product: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if safe_gate:
        if a_log is None or dt_bias is None:
            raise ValueError("gated_delta_product: safe_gate requires a_log and dt_bias")
        check_dtype("a_log", a_log, (torch.float32, torch.bfloat16, torch.float16))
        check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
    elif a_log is not None or dt_bias is not None:
        raise ValueError("gated_delta_product: a_log/dt_bias require safe_gate=True")
    cu = cu_seqlens
    check_dtype("g", g, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("beta", beta, q.dtype if use_beta_sigmoid_in_kernel else (torch.float32, q.dtype))
    if initial_state is not None:
        check_dtype("initial_state", initial_state, (torch.float32, torch.bfloat16))
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    dstate_dtype = initial_state.dtype if initial_state is not None else torch.float32
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
        check_dtype("d_final_state", d_final_state, (torch.float32, torch.bfloat16))
        if d_final_state.dtype != dstate_dtype:
            raise TypeError(f"gated_delta_product: d_final_state must be {dstate_dtype} (one state dtype per kernel)")
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
        g.dtype,
        beta.dtype,
        state0.dtype if state0 is not None else None,
        dstate_in.dtype if dstate_in is not None else None,
        tuple(k.shape),
        tuple(v.shape),
        cu_seqlens.dtype,
        state_checkpoints.shape[0] if state_checkpoints is not None else None,
        checkpoint_every_n_tokens,
        scale,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        a_log.dtype if safe_gate else None,
        dt_bias.dtype if safe_gate else None,
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
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            a_log_dtype=torch_dtype_to_cudnn(a_log.dtype) if safe_gate else None,
            dt_bias_dtype=torch_dtype_to_cudnn(dt_bias.dtype) if safe_gate else None,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        )
        select_plan(bprop_cache[cache_key][0], plan_name)

    graph, t = bprop_cache[cache_key]

    HO = max(H, HV)
    expanded = total * n
    dq = torch.empty(total, H, K, dtype=q.dtype, device=device)
    dk = torch.empty(expanded, HK, K, dtype=q.dtype, device=device)
    dv = torch.empty(expanded, HV, V, dtype=q.dtype, device=device)
    dg = torch.empty(total, HO, dtype=g.dtype, device=device)
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
        d_a_log = torch.empty_like(a_log)
        d_dt_bias = torch.empty_like(dt_bias)
        variant_pack[t["a_log"]] = a_log
        variant_pack[t["dt_bias"]] = dt_bias
        variant_pack[t["d_a_log"]] = d_a_log
        variant_pack[t["d_dt_bias"]] = d_dt_bias
    graph.execute(variant_pack, workspace=graph_workspace(graph, device), handle=get_handle(device))
    if dstate0 is None:
        dstate0 = torch.empty(0, dtype=dstate_dtype, device=device)
    return dq, dk, dv, dg, dbeta, dstate0, d_a_log, d_dt_bias


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
    checkpoint_every_n_tokens=0,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    a_log=None,
    dt_bias=None,
    plan_name=None,
):
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
        allow_neg_eigval,
        safe_gate,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        plan_name,
    ) = inputs
    saved = [q, k, v, g, beta, cu_seqlens]
    ctx.checkpoint_reuse = checkpoint_every_n_tokens > 0 and checkpoint_every_n_tokens % 64 == 0 and output[2].numel() > 0
    ctx.checkpoint_every_n_tokens = checkpoint_every_n_tokens if ctx.checkpoint_reuse else 0
    if ctx.checkpoint_reuse:
        saved.append(output[2])
    ctx.safe_gate = bool(safe_gate)
    if ctx.safe_gate:
        saved.extend((a_log, dt_bias))
    ctx.save_for_backward(*saved)
    ctx.initial_state = initial_state
    ctx.num_householder = num_householder
    ctx.scale = scale
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.batch_invariant = batch_invariant
    ctx.plan_name = plan_name
    ctx.use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
    ctx.allow_neg_eigval = bool(allow_neg_eigval)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def gdp_backward(ctx, dO, dFinal, dstate_checkpoints):
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
    dq, dk, dv, dg, dbeta, dstate0, d_a_log, d_dt_bias = torch.ops.cudnn.gated_delta_product_bwd(
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
        checkpoint_every_n_tokens=ctx.checkpoint_every_n_tokens,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
        batch_invariant=ctx.batch_invariant,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=ctx.allow_neg_eigval,
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
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
):
    """Gated DeltaProduct (GDP) linear attention.

    GDP applies ``n = num_householder`` beta-gated Householder updates per
    token with one scalar decay per token (the decay acts before the token's
    updates; the readout follows the last one).  THD layout on two timelines:

        q: ``[total_tokens, H, K]``; g: ``[total_tokens, HO]``;
        k: ``[total_tokens * n, HK, K]``; v: ``[total_tokens * n, HV, V]``;
        beta: ``[total_tokens * n, HO]``;
        cu_seqlens: ``[N+1]`` int32 over the real tokens; O and the states
        live at HO = max(H, HV) heads (initial_state / final_state:
        ``[N, HO, V, K]``)

    ``num_householder == 1`` is exactly ``gated_delta_net``.  Dtypes are
    kernel-native and strict (callers convert).  ``g`` is float32, bfloat16
    or float16 and ``dG`` comes back in the same dtype.  ``beta`` and
    ``dBeta`` are float32 or the io dtype (io-dtype logits under
    ``use_beta_sigmoid_in_kernel``; the fused sigmoid maps to ``(0, 1)``, or
    to ``(0, 2)`` under ``allow_neg_eigval``).  ``initial_state`` is float32
    or bfloat16, and ``final_state`` / the state gradients come back in the
    same dtype.

    Args:
        g: log-space scalar decay per real token (``alpha = exp(g) in (0, 1]``).
        beta: per-Householder write strength on the expanded rows.
        num_householder: Householder updates per token (``n >= 1``).
        safe_gate: interpret ``g`` through the safe-gate transform
            ``-exp(a_log) * softplus(g + dt_bias)``. Applied in the expansion
            pass, because the transform has no finite logit that leaves the
            filler sub-token rows neutral. Requires ``a_log`` and ``dt_bias``.
        a_log: ``[HO]`` safe-gate per-head log-amplitude (float32, bfloat16 or
            float16; ``d_a_log`` comes back in the same dtype).
        dt_bias: ``[HO]`` safe-gate per-head bias (float32, bfloat16 or
            float16; ``d_dt_bias`` comes back in the same dtype).
        allow_neg_eigval: scale the fused beta sigmoid by 2, so the delta-rule
            operator ``I - beta k k^T`` can reach negative eigenvalues
            (a reflection rather than a projection). Requires
            ``use_beta_sigmoid_in_kernel``.
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
        allow_neg_eigval=bool(allow_neg_eigval),
        safe_gate=bool(safe_gate),
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        plan_name=plan_name,
    )
    if checkpoint_every_n_tokens > 0:
        return o, final_state, state_checkpoints
    return o, final_state
