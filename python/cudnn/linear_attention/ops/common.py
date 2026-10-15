# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the linear-attention torch custom ops: per-call execution resources, plan pinning, the dtype map
and the summary ops' cache key and autograd hooks."""

import threading

import torch
import cudnn

TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
    torch.int32: cudnn.data_type.INT32,
    torch.int64: cudnn.data_type.INT64,
}


# cuDNN handles are not thread-safe. Each host thread keeps one per device and
# rebinds it to the launch stream before every execution.
_tls = threading.local()
_graph_lock_guard = threading.Lock()


def _graph_lock(graph):
    """Return the lock that protects one graph's build and lazy initialization."""
    lock = getattr(graph, "_la_ops_execution_lock", None)
    if lock is None:
        with _graph_lock_guard:
            lock = getattr(graph, "_la_ops_execution_lock", None)
            if lock is None:
                lock = threading.Lock()
                graph._la_ops_execution_lock = lock
    return lock


def _stream_handle(stream):
    """Return a raw CUDA stream handle from a torch stream or integer-like handle."""
    return int(stream.cuda_stream if hasattr(stream, "cuda_stream") else stream)


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


def graph_workspace(graph, device, stream=None):
    """Build ``graph`` if needed, then allocate caller-owned scratch on ``stream``."""
    stream = torch.cuda.current_stream(device) if stream is None else stream
    lock = _graph_lock(graph)
    if not graph._is_built:
        with lock:
            if not graph._is_built:
                if not graph._planning_done:
                    graph.create_execution_plans()
                if graph.selected_engine is None:
                    graph.build()
                else:
                    graph.build_plans()
    with torch.cuda.device(device), torch.cuda.stream(stream):
        return torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device=device)


def get_handle(device, stream=None):
    """This thread's cuDNN handle for ``device``, rebound to ``stream``."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    stream = torch.cuda.current_stream(device) if stream is None else stream
    handles = getattr(_tls, "handles", None)
    if handles is None:
        handles = _tls.handles = {}
    handle = handles.get(idx)
    if handle is None:
        with torch.cuda.device(idx):
            handle = cudnn.create_handle()
        handles[idx] = handle
    cudnn.set_stream(handle=handle, stream=_stream_handle(stream))
    return handle


def make_execution_resources(graph, device):
    """Return per-call scratch and this thread's handle for one launch stream.

    The caller owns the returned workspace only for this invocation. PyTorch's
    stream-aware allocator delays eager reuse and retains capture allocations in
    the CUDA graph's private memory pool.
    """
    stream = torch.cuda.current_stream(device)
    return graph_workspace(graph, device, stream), get_handle(device, stream)


def execute_graph(graph, variant_pack, workspace, handle):
    """Execute a cached graph with caller-owned scratch and a thread-private handle.

    The first call is serialized because FROST plans finish lazy compilation and
    initialize immutable launch records there. Steady-state launches stay lock-free.
    """
    if getattr(graph, "_la_ops_execution_ready", False):
        graph.execute(variant_pack, workspace=workspace, handle=handle)
        return
    lock = _graph_lock(graph)
    with lock:
        if not getattr(graph, "_la_ops_execution_ready", False):
            graph.execute(variant_pack, workspace=workspace, handle=handle)
            graph._la_ops_execution_ready = True
            return
    graph.execute(variant_pack, workspace=workspace, handle=handle)


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
    beta_shape=None,
):
    """Return the complete plan-cache identity for a linear-attention summary."""
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
        beta_shape,
    )


def summary_setup_context(ctx, inputs, output):
    """Mark summary outputs as non-differentiable for the torch autograd wrapper."""
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[0], output[1])


def summary_backward(ctx, d_output, d_transition):
    """Reject differentiation through summary-only custom ops."""
    raise RuntimeError("summaries are non-differentiable")
