# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the linear-attention torch custom ops: per-call execution resources, plan pinning, the dtype map
and the summary ops' cache key and autograd hooks."""

import threading

import torch
import cudnn

from ..graph_analyzer import BACKWARD_CONTEXT_GROUPS, BACKWARD_CONTEXT_PORTS, intermediate_dims  # noqa: F401  (re-exported to the family ops)

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


def check_composed_summaries(op, state_name, transition_name, state_update, transition, state_shape, transition_shape, device) -> int:
    """Validate one direction's composed context-parallel summaries, the summary ops' fp32 outputs over the preceding
    or following ranks stacked as ``[P, *state_shape]`` and ``[P, *transition_shape]``, and return the rank count ``P``;
    both ``None`` or a zero-length pair (a rank with no preceding or following ranks) is 0.  The graph declares both
    ports from ``P`` and the state geometry, so the buffers' own rank count, shapes and dtype are checked here; their
    layout is the plan's dense-layout check at execute."""
    if (state_update is None) != (transition is None):
        raise ValueError(f"{op}: {state_name} and {transition_name} come together")
    if state_update is None:
        return 0
    if state_update.dtype != torch.float32 or transition.dtype != torch.float32:
        raise TypeError(f"{op}: {state_name} and {transition_name} must be float32 (the summary ops' outputs); got {state_update.dtype} and {transition.dtype}")
    if state_update.shape[0] != transition.shape[0]:
        raise ValueError(
            f"{op}: {state_name} and {transition_name} must carry the same rank count; got {tuple(state_update.shape)} and {tuple(transition.shape)}"
        )
    if state_update.shape[0] == 0:
        return 0
    if tuple(state_update.shape[1:]) != tuple(state_shape):
        raise ValueError(f"{op}: {state_name} must be [P, {', '.join(str(d) for d in state_shape)}]; got {tuple(state_update.shape)}")
    if tuple(transition.shape[1:]) != tuple(transition_shape):
        raise ValueError(f"{op}: {transition_name} must be [P, {', '.join(str(d) for d in transition_shape)}]; got {tuple(transition.shape)}")
    for name, t in ((state_name, state_update), (transition_name, transition)):
        if t.device != device:
            raise ValueError(f"{op}: {name} must be on the op's device ({device}); got {t.device}")
    return int(state_update.shape[0])


def check_composed_state(op, composed, state_shape, device):
    """The forward's one-row composed boundary state handed to the backward: fp32 ``[1, *state_shape]`` on ``device``,
    None when absent or zero-size (a rank with no preceding summaries)."""
    if composed is None or composed.numel() == 0:
        return None
    if composed.dtype != torch.float32:
        raise TypeError(f"{op}: composed_initial_state must be float32; got {composed.dtype}")
    if tuple(composed.shape) != (1, *state_shape):
        raise ValueError(
            f"{op}: composed_initial_state must be [1, {', '.join(str(d) for d in state_shape)}] (the forward's one-row composed state); got {tuple(composed.shape)}"
        )
    if composed.device != device:
        raise ValueError(f"{op}: composed_initial_state must be on the op's device ({device}); got {composed.device}")
    return composed.contiguous()


def summary_setup_context(ctx, inputs, output):
    """Mark summary outputs as non-differentiable for the torch autograd wrapper."""
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*output)


def summary_backward(ctx, *grads):
    """Reject differentiation through summary-only custom ops."""
    raise RuntimeError("summaries are non-differentiable")


def backward_context_shapes(family, *, total, num_seqs, h_q, h_k, h_o, d_k, d_v, num_householder=1, use_qk_l2norm=False):
    """Dims per intermediate port of a family's backward summary for one geometry (``None`` = not emitted): the normalized
    q / k rows and their inverse norms for gdn / gdp under ``use_qk_l2norm`` where the backward normalizes the same rows
    (``num_householder == 1`` or ``d_v == 64``); for one local sequence the chunk factors (gdn / gdp) and the per-piece
    tables (every family but mamba2).  ``total`` is the k row count (the expanded timeline for gdp)."""
    full = family in ("gdn", "gdp")
    expand_num = max(1, int(num_householder))
    dims = dict(
        k_dim=[int(total), int(h_k), int(d_k)],
        q_dim=[int(total) // expand_num, int(h_q), int(d_k)],
        g_dim=[int(total), int(h_o)],
        do_dim=[int(total), int(h_o), int(d_v)],
    )
    out = dict.fromkeys(BACKWARD_CONTEXT_PORTS)
    if full and use_qk_l2norm and (expand_num == 1 or int(d_v) == 64):
        out.update({name: intermediate_dims(name, **dims) for name in ("q_normalized", "k_normalized", "q_inv_norm", "k_inv_norm")})
    if int(num_seqs) == 1 and family != "mamba2":
        out.update({name: intermediate_dims(name, **dims) for name in (("chunk_factors",) if full else ()) + ("piece_transitions", "piece_state_grads")})
    return out


def check_backward_context(op, intermediates, shapes, io_dtype, device):
    """Validate the backward summary's intermediates given to a backward op: ``intermediates`` maps the port names to tensors or
    None (a zero-size tensor is one the summary did not emit and counts as None); ``shapes`` is ``backward_context_shapes``
    for the backward's own geometry.  Each group comes together; a port this backward has no use for (the per-piece
    tables of a one-sequence summary on a multi-sequence slab, the normalized rows where the backward normalizes other
    rows) is left out; the rest must be dense, on ``device``, in the io dtype (rows and chunk factors) or fp32 (inverse
    norms and piece tables) and shaped as ``shapes`` says.  Returns the ports to bind."""
    present = {name: t for name, t in intermediates.items() if t is not None and t.numel() > 0}
    for group in BACKWARD_CONTEXT_GROUPS:
        got = [name for name in group if name in present]
        if got and len(got) != len(group):
            raise ValueError(f"{op}: {', '.join(group)} come together (the backward summary's intermediates); got {', '.join(got)}")
    bound = {}
    for name, t in present.items():
        want = shapes.get(name)
        if want is None:
            continue
        if tuple(t.shape) != tuple(want):
            raise ValueError(f"{op}: {name} must be {list(want)} (the summary's intermediates for these tokens); got {tuple(t.shape)}")
        want_dtype = io_dtype if name in ("q_normalized", "k_normalized", "chunk_factors") else torch.float32
        if t.dtype != want_dtype:
            raise TypeError(f"{op}: {name} must be {want_dtype} (the backward summary's output); got {t.dtype}")
        if t.device != device:
            raise ValueError(f"{op}: {name} must be on the op's device ({device}); got {t.device}")
        bound[name] = t.contiguous()
    return bound


def backward_context_outputs(shapes, io_dtype, device):
    """The summary op's intermediate outputs for ``shapes`` (``backward_context_shapes``): a dense tensor per emitted port, a
    zero-size one per port the summary does not emit."""
    out = {}
    for name, dims in shapes.items():
        dtype = io_dtype if name in ("q_normalized", "k_normalized", "chunk_factors") else torch.float32
        out[name] = torch.empty(dims if dims is not None else (0,), dtype=dtype, device=device)
    return out
