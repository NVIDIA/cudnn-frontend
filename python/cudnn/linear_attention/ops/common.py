# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the linear-attention torch custom ops: the cuDNN handle and workspace plumbing, plan pinning,
the dtype map and the summary ops' cache key and autograd hooks."""

from typing import Dict

import torch
import cudnn

TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
    torch.int32: cudnn.data_type.INT32,
    torch.int64: cudnn.data_type.INT64,
}


cudnn_handles: Dict[int, int] = {}


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
    """Caller-side workspace for a compiled graph (grow-only, held on the graph object itself)."""
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


def make_summary_cache_key(
    op,
    total,
    N,
    HK,
    HV,
    HO,
    K,
    V,
    io_dtype,
    v_dtype,
    k_shape,
    v_shape,
    cu_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    output_transition,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    a_log_dtype,
    dt_bias_dtype,
    device,
    plan_name,
    gate_lower_bound=None,
    gate_domain="log",
    beta_guard=False,
    w_dtype=None,
    w_shape=None,
    num_householder=1,
    scale=None,
    d_final_state_dtype=None,
    do_dtype=None,
    do_shape=None,
    q_shape=None,
):
    return (
        op,
        total,
        N,
        HK,
        HV,
        HO,
        K,
        V,
        io_dtype,
        v_dtype,
        k_shape,
        v_shape,
        cu_dtype,
        g_dtype,
        beta_dtype,
        state_dtype,
        bool(output_transition),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(safe_gate),
        str(gate_domain),
        a_log_dtype,
        dt_bias_dtype,
        gate_lower_bound,
        bool(beta_guard),
        w_dtype,
        w_shape,
        int(num_householder),
        scale,
        d_final_state_dtype,
        do_dtype,
        do_shape,
        q_shape,
        device,
        plan_name,
    )


def summary_setup_context(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[0], output[1])


def summary_backward(ctx, d_output, d_transition):
    raise RuntimeError("summaries are non-differentiable")
