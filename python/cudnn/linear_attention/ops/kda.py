# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch custom operator for Kimi Delta Attention (KDA) linear attention.

KDA is GDN with a per-key-channel decay; GDN's scalar per-token decay
``alpha_t`` becomes a decay vector ``alpha_t in (0, 1]^K``, applied as a
diagonal matrix on the K-axis of the recurrent state:

    S_t = (I - beta_t k_t^T k_t) Diag(alpha_t) S_{t-1} + beta_t k_t^T v_t,
    o_t = q_t S_t,

with ``alpha_t = exp(g_t)`` (``g_t in R^K``, log-space per channel) and
scalar write strength ``beta_t``. The decay is applied first, so the
delta-rule correction reads the already-decayed state.

Layout follows the graph-API KDA node (THD), token-packed ``[total_tokens,
heads, dim]`` tensors plus ``cu_seqlens`` sequence boundaries. ``g`` is the
per-key-channel log decay ``[total_tokens, heads, dim]``; ``beta`` is scalar
``[total_tokens, heads]``.

The op is a thin adapter over the graph API; forward
and backward execute cached single-node ``KDA`` / ``KDA_BWD`` pygraphs.
Engine selection happens at graph planning time over the manifest's python
engines, ``KdaFrostEngine`` (default on SM100/SM103, forward and backward)
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

from .common import (
    BACKWARD_CONTEXT_PORTS,
    backward_context_outputs,
    backward_context_shapes,
    check_backward_context,
    check_composed_state,
    check_composed_summaries,
    execute_graph,
    make_execution_resources,
    make_summary_cache_key,
    select_plan,
    summary_backward,
    summary_setup_context,
    torch_dtype_to_cudnn,
)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


CHECKPOINT_CHUNK = 16

fprop_cache: Dict[tuple, tuple] = {}
bprop_cache: Dict[tuple, tuple] = {}
summary_cache: Dict[tuple, tuple] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_dtype(name, t, want) -> None:
    wants = want if isinstance(want, tuple) else (want,)
    if t.dtype not in wants:
        names = " or ".join(str(w) for w in wants)
        raise TypeError(f"kimi_delta_attention: {name} must be {names} (kernel-native; callers convert), got {t.dtype}")


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
    g_dtype,
    beta_dtype,
    state_dtype,
    scale,
    output_final_state,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    gate_lower_bound,
    gate_domain,
    a_log_dtype,
    dt_bias_dtype,
    checkpoint,
    checkpoint_dtype,
    device,
    plan_name,
    overwrite_initial_state,
    state_pool_rows=None,
    composed_rows=None,
):
    """Return the complete plan-cache identity for a KDA forward graph."""
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
        g_dtype,
        beta_dtype,
        state_dtype,
        float(scale),
        bool(output_final_state),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(safe_gate),
        float(gate_lower_bound) if gate_lower_bound is not None else None,
        str(gate_domain),
        a_log_dtype,
        dt_bias_dtype,
        checkpoint,
        checkpoint_dtype,
        device,
        plan_name,
        bool(overwrite_initial_state),
        state_pool_rows,
        composed_rows,
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
    checkpoint_dtype,
    checkpoint_every_n_tokens,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    gate_lower_bound,
    gate_domain,
    a_log_dtype,
    dt_bias_dtype,
    device,
    plan_name,
    composed_rows=None,
    composed_state=False,
):
    """Return the complete plan-cache identity for a KDA backward graph."""
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
        checkpoint_dtype,
        int(checkpoint_every_n_tokens),
        float(scale),
        bool(use_qk_l2norm),
        bool(batch_invariant),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        bool(safe_gate),
        float(gate_lower_bound) if gate_lower_bound is not None else None,
        str(gate_domain),
        a_log_dtype,
        dt_bias_dtype,
        device,
        plan_name,
        composed_rows,
        bool(composed_state),
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
    allow_neg_eigval,
    safe_gate,
    gate_lower_bound,
    gate_domain,
    checkpoint,
    checkpoint_dtype,
    a_log_dtype=None,
    dt_bias_dtype=None,
    overwrite_initial_state=False,
    state_indices_dtype=None,
    state_pool_rows=None,
    composed_rows=None,
):
    """Build a single-node KDA forward graph and its tensor placeholders."""
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
        state_rows = N if state_pool_rows is None else state_pool_rows
        state0_t = graph.tensor([state_rows, HO, V, K], data_type=state_dtype, name="initial_state")
    state_indices_t = None
    if state_indices_dtype is not None:
        state_indices_t = graph.tensor([N], data_type=state_indices_dtype, name="state_indices")
    pool_ports = {"state_indices": state_indices_t} if state_indices_t is not None else {}
    composed_ports = {}
    if composed_rows is not None:
        composed_ports = dict(
            preceding_state_update=graph.tensor([composed_rows, HO, V, K], data_type=cudnn.data_type.FLOAT, name="preceding_state_update"),
            preceding_transition=graph.tensor([composed_rows, HO, K, K], data_type=cudnn.data_type.FLOAT, name="preceding_transition"),
        )
    a_log_t = None
    dt_bias_t = None
    if a_log_dtype is not None:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([HO, K], data_type=dt_bias_dtype, name="dt_bias")
    O_t, fs_t, state_checkpoints_t, composed_t = graph.kda(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        **pool_ports,
        **composed_ports,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=None if gate_domain == "log" else gate_domain,
        overwrite_initial_state=overwrite_initial_state or None,
        checkpoint_every_n_tokens=checkpoint,
        name="kda",
    )
    if state_checkpoints_t is not None:
        state_checkpoints_t.set_data_type(cudnn.data_type.FLOAT if checkpoint_dtype == torch.float32 else io_dtype)
    if composed_t is not None:
        composed_t.set_data_type(cudnn.data_type.FLOAT)
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        state0=state0_t,
        state_indices=state_indices_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        O=O_t,
        fs=fs_t,
        state_checkpoints=state_checkpoints_t,
        composed_initial_state=composed_t,
        **composed_ports,
    )


# ---------------------------------------------------------------------------
# Forward custom op
# ---------------------------------------------------------------------------


def run_kda_fwd(
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
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_dtype: Optional[torch.dtype] = None,
    plan_name: Optional[str] = None,
    final_state_out: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    preceding_state_update: Optional[torch.Tensor] = None,
    preceding_transition: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal KDA forward over a cached single-node KDA pygraph in THD layout.

    Returns ``(o, final_state, state_checkpoints)``; ``final_state`` / ``state_checkpoints`` are zero-size
    tensors when ``output_final_state`` is ``False`` /
    ``checkpoint_every_n_tokens`` is ``0``.
    ``final_state_out`` binds the final_state output to a caller buffer (the in-place op passes ``initial_state``);
    ``state_indices`` (int32 ``[N]``) makes ``initial_state`` a pool whose row ``state_indices[i]`` seeds sequence ``i``
    and receives its final state.  ``preceding_state_update`` / ``preceding_transition`` (fp32 ``[P, HO, V, K]`` /
    ``[P, HO, K, K]``, the forward summaries of the ``P`` preceding context-parallel ranks in rank order) are folded on
    the device into the one-row ``composed_initial_state`` (fp32 ``[1, HO, V, K]``, the fourth output) that the first
    local sequence starts from; the caller's ``initial_state`` row 0, when given, is the state beyond the farthest rank
    and the other rows stay the other sequences' starts (zero without one).  Zero-size without summaries, as with an
    empty ``P = 0`` pair.
    """
    total, H, K = q.shape
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV})")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    cu = cu_seqlens
    check_dtype("g", g, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("beta", beta, (torch.float32, q.dtype))
    if safe_gate:
        if a_log is not None:
            check_dtype("a_log", a_log, (torch.float32, torch.bfloat16, torch.float16))
        if dt_bias is not None:
            check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
    elif a_log is not None or dt_bias is not None:
        raise ValueError("kimi_delta_attention: a_log/dt_bias require safe_gate=True")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention: gate_domain='linear' cannot combine with safe_gate=True")
    state_pool_rows = None
    if state_indices is not None:
        if initial_state is None:
            raise ValueError("kimi_delta_attention: state_indices selects rows of the initial_state pool, so initial_state is required")
        check_dtype("state_indices", state_indices, torch.int32)
        if state_indices.dim() != 1 or state_indices.shape[0] != N:
            raise ValueError(f"kimi_delta_attention: state_indices must be [N] = [{N}]; got {tuple(state_indices.shape)}")
        if int(checkpoint_every_n_tokens):
            raise ValueError("kimi_delta_attention: state_indices cannot be combined with checkpoint_every_n_tokens")
        if (
            tuple(initial_state.stride()[1:]) != (V * K, K, 1)
            or initial_state.stride(0) < max(H, HV) * V * K
            or initial_state.stride(0) % (16 // initial_state.element_size())
        ):
            raise ValueError("kimi_delta_attention: state_indices requires a pool whose rows are dense [HO, V, K] blocks on a 16-byte aligned slot stride")
        state_pool_rows = int(initial_state.shape[0])
    if initial_state is not None:
        check_dtype("initial_state", initial_state, (torch.float32, torch.bfloat16, torch.float16))
        if state_indices is None and initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    for tensor_name, tensor in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu_seqlens),
        ("initial_state", initial_state),
        ("state_indices", state_indices),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"kimi_delta_attention: {tensor_name} must be on q's device ({device}); got {tensor.device}")
    state0 = initial_state if initial_state is not None else None
    composed_rows = check_composed_summaries(
        "kimi_delta_attention",
        "preceding_state_update",
        "preceding_transition",
        preceding_state_update,
        preceding_transition,
        (max(H, HV), V, K),
        (max(H, HV), K, K),
        device,
    )
    composed = torch.empty(0, dtype=torch.float32, device=device)
    if composed_rows:
        if state_indices is not None or final_state_out is not None:
            raise ValueError("kimi_delta_attention: preceding summaries cannot combine with state_indices or overwrite_initial_state")
        composed = torch.empty(1, max(H, HV), V, K, dtype=torch.float32, device=device)
    state_out_dtype = state0.dtype if state0 is not None else torch.float32
    checkpoint = int(checkpoint_every_n_tokens)
    if checkpoint_dtype not in (None, torch.float32, q.dtype):
        raise ValueError(f"kimi_delta_attention: checkpoint_dtype must be None, torch.float32 or the io dtype ({q.dtype}), got {checkpoint_dtype}")
    if checkpoint_dtype is None:
        checkpoint_dtype = q.dtype

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
        g.dtype,
        beta.dtype,
        state0.dtype if state0 is not None else None,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_lower_bound,
        gate_domain,
        a_log.dtype if a_log is not None else None,
        dt_bias.dtype if dt_bias is not None else None,
        checkpoint,
        checkpoint_dtype,
        device,
        plan_name,
        final_state_out is not None,
        state_pool_rows,
        composed_rows=composed_rows or None,
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
            torch_dtype_to_cudnn(g.dtype),
            torch_dtype_to_cudnn(beta.dtype),
            torch_dtype_to_cudnn(state0.dtype) if state0 is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            float(scale),
            bool(output_final_state),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            bool(use_beta_sigmoid_in_kernel),
            bool(allow_neg_eigval),
            bool(safe_gate),
            float(gate_lower_bound) if gate_lower_bound is not None else None,
            str(gate_domain),
            checkpoint,
            checkpoint_dtype,
            a_log_dtype=torch_dtype_to_cudnn(a_log.dtype) if a_log is not None else None,
            dt_bias_dtype=torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            overwrite_initial_state=final_state_out is not None,
            state_indices_dtype=torch_dtype_to_cudnn(state_indices.dtype) if state_indices is not None else None,
            state_pool_rows=state_pool_rows,
            composed_rows=composed_rows or None,
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
    if state_indices is not None:
        variant_pack[t["state_indices"]] = state_indices
    if composed_rows:
        variant_pack[t["preceding_state_update"]] = preceding_state_update
        variant_pack[t["preceding_transition"]] = preceding_transition
        variant_pack[t["composed_initial_state"]] = composed
    if a_log is not None:
        variant_pack[t["a_log"]] = a_log
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
    state_out_dtype = state0.dtype if state0 is not None else torch.float32
    final_state = torch.empty(0, dtype=state_out_dtype, device=device)
    if final_state_out is not None:
        final_state = final_state_out
        variant_pack[t["fs"]] = final_state
    elif output_final_state:
        final_state = torch.empty(N, HO, V, K, dtype=state_out_dtype, device=device)
        variant_pack[t["fs"]] = final_state
    state_checkpoints = torch.empty(0, dtype=q.dtype, device=device)
    if checkpoint > 0:
        total_checkpoints = max(total // checkpoint + N, 1)
        state_checkpoints = torch.empty(total_checkpoints, HO, V, K, dtype=checkpoint_dtype, device=device)
        variant_pack[t["state_checkpoints"]] = state_checkpoints
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return o, final_state, state_checkpoints, composed


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
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_dtype: Optional[torch.dtype] = None,
    plan_name: Optional[str] = None,
    preceding_state_update: Optional[torch.Tensor] = None,
    preceding_transition: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional KDA forward: :func:`run_kda_fwd` with fresh outputs; the autograd formula registers below."""
    return run_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_dtype=checkpoint_dtype,
        plan_name=plan_name,
        preceding_state_update=preceding_state_update,
        preceding_transition=preceding_transition,
    )


@torch.library.custom_op("cudnn::kimi_delta_attention_fwd_overwrite_state", mutates_args=("initial_state",))
def kda_fwd_overwrite_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_dtype: Optional[torch.dtype] = None,
    plan_name: Optional[str] = None,
    state_indices: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KDA forward whose final state overwrites ``initial_state``, the rows ``state_indices`` names when that int32 ``[N]``
    table addresses a state pool.  The engine runs the chain or the uncut schedule and
    never the split-K cut, so the state a sequence reads at its first chunk is the one it writes at its last.  Inference
    only: a mutating op registers no autograd formula.  Returns ``(o, state_checkpoints)``."""
    o, _, state_checkpoints, _ = run_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        scale=scale,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_dtype=checkpoint_dtype,
        plan_name=plan_name,
        output_final_state=bool(output_final_state),
        final_state_out=initial_state if output_final_state else None,
        state_indices=state_indices,
    )
    return o, state_checkpoints


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
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    checkpoint_dtype=None,
    plan_name: Optional[str] = None,
    preceding_state_update=None,
    preceding_transition=None,
):
    """Return FakeTensor-compatible output metadata for KDA forward."""
    total, H, K = q.shape
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV})")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention: gate_domain='linear' cannot combine with safe_gate=True")
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    o = q.new_empty(total, HO, V)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    composed_rows = check_composed_summaries(
        "kimi_delta_attention", "preceding_state_update", "preceding_transition", preceding_state_update, preceding_transition, (HO, V, K), (HO, K, K), q.device
    )
    composed = q.new_empty((1, HO, V, K) if composed_rows else (0,), dtype=torch.float32)
    final = q.new_empty((N, HO, V, K) if output_final_state else (0,), dtype=state_dtype)
    if checkpoint_every_n_tokens > 0:
        total_checkpoints = max(total // int(checkpoint_every_n_tokens) + N, 1)
        state_checkpoints = q.new_empty(
            total_checkpoints,
            HO,
            V,
            K,
            dtype=torch.float32 if checkpoint_dtype == torch.float32 else q.dtype,
        )
    else:
        state_checkpoints = q.new_empty(0)
    return o, final, state_checkpoints, composed


@kda_fwd_overwrite_state.register_fake
def kda_fwd_overwrite_state_fake(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    checkpoint_dtype=None,
    plan_name=None,
    state_indices=None,
    output_final_state=True,
):
    """Return fake outputs for the in-place KDA forward custom op."""
    o, _, state_checkpoints, _ = kda_fwd_fake(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        scale=scale,
        initial_state=initial_state if state_indices is None else None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_dtype=checkpoint_dtype,
        plan_name=plan_name,
        output_final_state=bool(output_final_state),
    )
    return o, state_checkpoints


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
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log_dtype=None,
    dt_bias_dtype=None,
    checkpoint_every_n_tokens=0,
    checkpoint_dtype=None,
    composed_rows=None,
    has_composed_state=False,
    intermediates=None,
):
    """Build a single-node KDA backward graph and its tensor placeholders."""
    graph = cudnn.pygraph()
    HO = max(H, HV)
    q_t = graph.tensor([total, H, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, HV, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, HO, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    dO_t = graph.tensor([total, HO, V], data_type=io_dtype, name="dO")
    composed_state_t = graph.tensor([1, HO, V, K], data_type=cudnn.data_type.FLOAT, name="composed_initial_state") if has_composed_state else None
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, HO, V, K], data_type=state_dtype, name="initial_state")
    dfs_t = None
    if dstate_in_dtype is not None:
        dfs_t = graph.tensor([N, HO, V, K], data_type=dstate_in_dtype, name="d_final_state")
    checkpoints_t = None
    if checkpoint_rows is not None:
        checkpoints_t = graph.tensor([checkpoint_rows, HO, V, K], data_type=checkpoint_dtype, name="state_checkpoints")
    composed_ports = {}
    if composed_rows is not None:
        composed_ports = dict(
            following_state_grad=graph.tensor([composed_rows, HO, V, K], data_type=cudnn.data_type.FLOAT, name="following_state_grad"),
            following_transition=graph.tensor([composed_rows, HO, K, K], data_type=cudnn.data_type.FLOAT, name="following_transition"),
        )
    a_log_t = None
    dt_bias_t = None
    if a_log_dtype is not None:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([HO, K], data_type=dt_bias_dtype, name="dt_bias")
    intermediate_ports = {name: graph.tensor(list(dims), data_type=dtype, name=name) for name, (dims, dtype) in (intermediates or {}).items()}
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
        composed_initial_state=composed_state_t,
        **composed_ports,
        **intermediate_ports,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens or None,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid or None,
        allow_neg_eigval=allow_neg_eigval or None,
        safe_gate=safe_gate or None,
        gate_lower_bound=gate_lower_bound,
        gate_domain=None if gate_domain == "log" else gate_domain,
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
        composed_initial_state=composed_state_t,
        **composed_ports,
        **intermediate_ports,
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
    checkpoint_every_n_tokens: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    plan_name: Optional[str] = None,
    following_state_grad: Optional[torch.Tensor] = None,
    following_transition: Optional[torch.Tensor] = None,
    composed_initial_state: Optional[torch.Tensor] = None,
    q_normalized: Optional[torch.Tensor] = None,
    k_normalized: Optional[torch.Tensor] = None,
    q_inv_norm: Optional[torch.Tensor] = None,
    k_inv_norm: Optional[torch.Tensor] = None,
    chunk_factors: Optional[torch.Tensor] = None,
    piece_transitions: Optional[torch.Tensor] = None,
    piece_state_grads: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal KDA backward over a cached single-node KDA_BWD pygraph in THD layout.

    ``state_checkpoints`` is the forward's state series (io dtype or fp32 at the forward's cadence); when given, the
    engine reuses io-dtype rows at the chunk cadence and regenerates the dense series from any other rows instead of
    running the checkpoint recompute pass.  ``following_state_grad`` / ``following_transition`` (fp32 ``[P, HO, V, K]`` /
    ``[P, HO, K, K]``, the backward summaries of the ``P`` following context-parallel ranks in rank order) are folded on
    the device into a one-row gradient state that the last local sequence starts from (the caller's ``d_final_state``
    row, when given, is the gradient beyond the farthest rank; every other sequence starts from its ``d_final_state``
    row, or zero).  ``composed_initial_state`` (fp32 ``[1, HO, V, K]``, the forward's fourth output) is the first local
    sequence's initial state under context parallelism, beside the caller's optional ``initial_state``.
    ``q_normalized`` .. ``piece_state_grads`` are the backward summary's intermediates over the same tokens
    (``output_intermediates``): KDA takes the per-piece transitions and state gradients in place of its per-piece summary
    passes (the l2norm and chunk factor are in-kernel, so the other entries are never emitted); a zero-size tensor is
    one the summary did not emit, and each group comes together.  Returns ``(dq, dk, dv, dg, dbeta, d_initial_state,
    d_a_log, d_dt_bias)``.
    """
    total, H, K = q.shape
    if 0 in dO.stride():
        dO = dO.contiguous()
    check_dtype("dO", dO, q.dtype)
    if d_final_state is not None and 0 in d_final_state.stride():
        d_final_state = d_final_state.contiguous()
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV})")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    cu = cu_seqlens
    check_dtype("g", g, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("beta", beta, (torch.float32, q.dtype))
    if safe_gate:
        if a_log is not None:
            check_dtype("a_log", a_log, (torch.float32, torch.bfloat16, torch.float16))
        if dt_bias is not None:
            check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
    elif a_log is not None or dt_bias is not None:
        raise ValueError("kimi_delta_attention: a_log/dt_bias require safe_gate=True")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention: gate_domain='linear' cannot combine with safe_gate=True")
    if initial_state is not None:
        check_dtype("initial_state", initial_state, (torch.float32, torch.bfloat16, torch.float16))
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    dstate_dtype = initial_state.dtype if initial_state is not None else torch.float32
    if d_final_state is not None:
        check_dtype("d_final_state", d_final_state, (torch.float32, torch.bfloat16, torch.float16))
        if d_final_state.dtype != dstate_dtype:
            raise TypeError(f"kimi_delta_attention: d_final_state must be {dstate_dtype} (one state dtype per kernel)")
    if state_checkpoints is not None:
        check_dtype("state_checkpoints", state_checkpoints, (torch.float32, q.dtype))
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
    composed_rows = check_composed_summaries(
        "kimi_delta_attention",
        "following_state_grad",
        "following_transition",
        following_state_grad,
        following_transition,
        (max(H, HV), V, K),
        (max(H, HV), K, K),
        device,
    )
    composed_state = check_composed_state("kimi_delta_attention", composed_initial_state, (max(H, HV), V, K), device)
    intermediate_args = dict(
        q_normalized=q_normalized,
        k_normalized=k_normalized,
        q_inv_norm=q_inv_norm,
        k_inv_norm=k_inv_norm,
        chunk_factors=chunk_factors,
        piece_transitions=piece_transitions,
        piece_state_grads=piece_state_grads,
    )
    intermediates = {}
    if any(t is not None and t.numel() > 0 for t in intermediate_args.values()):
        intermediates = check_backward_context(
            "kimi_delta_attention",
            intermediate_args,
            backward_context_shapes(
                "kda",
                total=total,
                num_seqs=N,
                h_q=H,
                h_k=HK,
                h_o=max(H, HV),
                d_k=K,
                d_v=V,
                use_qk_l2norm=use_qk_l2norm_in_kernel,
            ),
            q.dtype,
            device,
        )

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
        state_checkpoints.dtype if state_checkpoints is not None else None,
        checkpoint_every_n_tokens,
        scale,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_lower_bound,
        gate_domain,
        a_log.dtype if a_log is not None else None,
        dt_bias.dtype if dt_bias is not None else None,
        device,
        plan_name,
        composed_rows=composed_rows or None,
        composed_state=composed_state is not None,
    ) + tuple(sorted(intermediates))
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
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
            gate_domain=str(gate_domain),
            a_log_dtype=torch_dtype_to_cudnn(a_log.dtype) if a_log is not None else None,
            dt_bias_dtype=torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            checkpoint_dtype=torch_dtype_to_cudnn(state_checkpoints.dtype) if state_checkpoints is not None else None,
            composed_rows=composed_rows or None,
            has_composed_state=composed_state is not None,
            intermediates={name: (tuple(tensor.shape), torch_dtype_to_cudnn(tensor.dtype)) for name, tensor in intermediates.items()},
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
        dstate0 = torch.empty(state0.shape, dtype=dstate_dtype, device=device)
        variant_pack[t["dstate0"]] = dstate0
    if dstate_in is not None:
        variant_pack[t["dfs"]] = dstate_in
    if composed_rows:
        variant_pack[t["following_state_grad"]] = following_state_grad
        variant_pack[t["following_transition"]] = following_transition
    if composed_state is not None:
        variant_pack[t["composed_initial_state"]] = composed_state
    if state_checkpoints is not None:
        variant_pack[t["checkpoints"]] = state_checkpoints
    for name, tensor in intermediates.items():
        variant_pack[t[name]] = tensor
    d_a_log = torch.empty(0, dtype=torch.float32, device=device)
    d_dt_bias = torch.empty(0, dtype=torch.float32, device=device)
    if a_log is not None:
        variant_pack[t["a_log"]] = a_log
        d_a_log = torch.empty_like(a_log)
        variant_pack[t["d_a_log"]] = d_a_log
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
        d_dt_bias = torch.empty_like(dt_bias)
        variant_pack[t["d_dt_bias"]] = d_dt_bias
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    if dstate0 is None:
        dstate0 = torch.empty(0, dtype=dstate_dtype, device=device)
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
    checkpoint_every_n_tokens=0,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    plan_name=None,
    following_state_grad=None,
    following_transition=None,
    composed_initial_state=None,
    q_normalized=None,
    k_normalized=None,
    q_inv_norm=None,
    k_inv_norm=None,
    chunk_factors=None,
    piece_transitions=None,
    piece_state_grads=None,
):
    """Return FakeTensor-compatible output metadata for KDA backward."""
    dstate_dtype = initial_state.dtype if initial_state is not None else torch.float32
    if d_final_state is not None and d_final_state.dtype != dstate_dtype:
        raise TypeError(f"kimi_delta_attention: d_final_state must be {dstate_dtype} (one state dtype per kernel)")
    dstate0 = initial_state.new_empty(initial_state.shape, dtype=dstate_dtype) if initial_state is not None else q.new_empty(0, dtype=dstate_dtype)
    d_a_log = torch.empty_like(a_log) if a_log is not None else q.new_empty(0, dtype=torch.float32)
    d_dt_bias = torch.empty_like(dt_bias) if dt_bias is not None else q.new_empty(0, dtype=torch.float32)
    return (
        q.new_empty(q.shape),
        q.new_empty(k.shape),
        q.new_empty(v.shape),
        g.new_empty(g.shape),
        beta.new_empty(beta.shape),
        dstate0,
        d_a_log,
        d_dt_bias,
    )


# ---------------------------------------------------------------------------
# Autograd registration
# ---------------------------------------------------------------------------


def kda_setup_context(ctx, inputs, output):
    """Save KDA forward inputs and static options needed by autograd."""
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
        allow_neg_eigval,
        safe_gate,
        gate_lower_bound,
        gate_domain,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        checkpoint_dtype,
        plan_name,
        preceding_state_update,
        preceding_transition,
    ) = inputs
    saved = [q, k, v, g, beta, cu_seqlens]
    ctx.checkpoint_reuse = checkpoint_every_n_tokens > 0 and checkpoint_every_n_tokens % CHECKPOINT_CHUNK == 0 and output[2].numel() > 0
    ctx.checkpoint_every_n_tokens = checkpoint_every_n_tokens if ctx.checkpoint_reuse else 0
    if ctx.checkpoint_reuse:
        saved.append(output[2])
    ctx.has_a_log = a_log is not None
    ctx.has_dt_bias = dt_bias is not None
    if ctx.has_a_log:
        saved.append(a_log)
    if ctx.has_dt_bias:
        saved.append(dt_bias)
    ctx.save_for_backward(*saved)
    # under context parallelism the backward starts from the composed state, and the caller's initial_state (the
    # state beyond the farthest preceding rank) gets no gradient
    ctx.composed = output[3].numel() > 0
    ctx.initial_state = initial_state
    ctx.composed_state = output[3] if ctx.composed else None
    ctx.scale = scale
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.batch_invariant = batch_invariant
    ctx.plan_name = plan_name
    ctx.use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
    ctx.allow_neg_eigval = bool(allow_neg_eigval)
    ctx.safe_gate = bool(safe_gate)
    ctx.gate_lower_bound = gate_lower_bound
    ctx.gate_domain = str(gate_domain)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2], output[3])


def kda_backward(ctx, dO, dFinal, dstate_checkpoints, d_composed_initial_state):
    """Apply the registered first-order KDA backward formula."""
    # Non-reentrant activation checkpointing allows one saved-tensor unpack.
    saved_tensors = ctx.saved_tensors
    if ctx.checkpoint_reuse:
        q, k, v, g, beta, cu_seqlens, state_checkpoints = saved_tensors[:7]
        gate_params = list(saved_tensors[7:])
    else:
        q, k, v, g, beta, cu_seqlens = saved_tensors[:6]
        state_checkpoints = None
        gate_params = list(saved_tensors[6:])
    a_log = gate_params.pop(0) if ctx.has_a_log else None
    dt_bias = gate_params.pop(0) if ctx.has_dt_bias else None
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
        composed_initial_state=ctx.composed_state,
        state_checkpoints=state_checkpoints,
        checkpoint_every_n_tokens=ctx.checkpoint_every_n_tokens,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
        batch_invariant=ctx.batch_invariant,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=ctx.allow_neg_eigval,
        safe_gate=ctx.safe_gate,
        gate_lower_bound=ctx.gate_lower_bound,
        gate_domain=ctx.gate_domain,
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
        dstate0 if initial_state is not None and not ctx.composed else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        d_a_log if ctx.has_a_log else None,
        d_dt_bias if ctx.has_dt_bias else None,
        None,
        None,
        None,
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
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_dtype: Optional[torch.dtype] = None,
    plan_name: Optional[str] = None,
    overwrite_initial_state: bool = False,
    state_indices: Optional[torch.Tensor] = None,
    preceding_state_update: Optional[torch.Tensor] = None,
    preceding_transition: Optional[torch.Tensor] = None,
):
    """Kimi Delta Attention (KDA) linear attention.

    THD layout (matches the graph-API KDA node):

        q: ``[total_tokens, H, K]``; k: ``[total_tokens, HK, K]`` (HK = H or
        HK = HV); v: ``[total_tokens, HV, V]``
        g: ``[total_tokens, HO, K]`` (per-key-channel log decay);
        beta: ``[total_tokens, HO]`` (scalar); cu_seqlens: ``[N+1]`` int32
        initial_state / final_state: ``[N, HO, V, K]``
        (``HO = max(H, HV)``, the gates, output, and state heads)

    A dense batch of N equal-length sequences is expressed as
    ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Dtypes are kernel-native and strict (callers convert).  ``g`` is float32,
    bfloat16 or float16 and ``dG`` comes back in the same dtype.  ``beta`` and
    ``dBeta`` are float32 or the io dtype (raw logits of either dtype under
    ``use_beta_sigmoid_in_kernel``).  ``initial_state`` is float32 or bfloat16
    and ``final_state`` and the state gradients (``d_final_state``,
    ``d_initial_state``) follow it.

    Args:
        g: per-key-channel decay: log-space by default (``alpha = exp(g) in
            (0, 1]^K``), the linear decay ``alpha`` itself when
            ``gate_domain="linear"``, or raw pre-activation logits when
            ``safe_gate=True``.
        beta: per-token scalar write strength (float32 or the io dtype,
            post-sigmoid), or float32/io-dtype logits when
            ``use_beta_sigmoid_in_kernel=True``.
        cu_seqlens: ``[N+1]`` int32/int64 sequence boundaries over the packed tokens.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state (otherwise zero); float32 or
            bfloat16, and ``final_state`` comes back in the same dtype.
        output_final_state: if ``True``, also return the per-sequence state
            after the last token.
        use_qk_l2norm_in_kernel: if ``True``, L2-normalize the Q/K rows inside
            the kernel (the KDA model's feature map); if ``False``, pass Q/K
            as given (the caller owns their conditioning).
        batch_invariant: if ``True``, each sequence's results are bitwise
            independent of the batch composition (whole-sequence scheduling;
            disables split-K load balancing).
        overwrite_initial_state: if ``True``, ``final_state`` is written
            into the ``initial_state`` buffer, which is required and
            returned as the final state; the engine runs the chain or the
            uncut schedule and never the split-K cut, so the aliasing is
            race-free.  Inference only: the in-place op has no autograd
            formula.
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` inside the kernel;
            the backward returns the raw-logit beta gradient.
        allow_neg_eigval: scale the fused beta sigmoid by 2, so the delta-rule
            operator ``I - beta k k^T`` can reach negative eigenvalues
            (a reflection rather than a projection). Requires
            ``use_beta_sigmoid_in_kernel``.
        safe_gate: interpret ``g`` through the safe-gate transform
            ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``.
            ``a_log`` and ``dt_bias`` are optional; an absent ``a_log`` is
            unit amplitude (``exp(a_log) = 1``), an absent ``dt_bias`` is zero
            bias, and an absent parameter gets no gradient. The backward
            returns the raw-logit ``g`` gradient plus the gradients of the
            parameters that were given. Passing either without
            ``safe_gate=True`` is an error.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        gate_domain: ``"log"`` (default): ``g`` holds ``ln(alpha)``;
            ``"linear"``: ``g`` holds ``alpha in (0, 1]^K`` (a 1e-10 floor is
            applied in-kernel) and ``dG`` comes back with respect to ``alpha``.
            Cannot combine with ``safe_gate``.
        a_log: ``[HO]`` safe-gate per-head log-amplitude (float32, bfloat16 or
            float16; ``d_a_log`` comes back in the same dtype), or ``None``
            for unit amplitude.
        dt_bias: ``[HO, K]`` safe-gate channel bias (float32, bfloat16 or
            float16; ``d_dt_bias`` comes back in the same dtype), or ``None``
            for zero bias.
        checkpoint_every_n_tokens: if ``> 0``, also return the per-chunk
            recurrent state series ``state_checkpoints`` (``[total_checkpoints, HO, V, K]``, dtype per ``checkpoint_dtype``,
            one entry per N tokens strictly before each sequence end; the
            FROST engine requires a positive multiple of the kernel chunk size, 16). The series is
            a non-differentiable dump.
        checkpoint_dtype: ``None`` stores io-dtype rows; ``torch.float32`` stores fp32 rows (the backward reuses io rows at the
            chunk cadence and regenerates the dense series from any other rows).

        plan_name: optionally pin one execution plan by name (the plan
            API's ``get_plan_name_at_index`` names, e.g. ``kda_frost``); a
            graph offering no such plan raises ``cudnnGraphNotSupportedError``.
        state_indices: optional ``[N]`` int32 table making ``initial_state`` a
            pool ``[N_pool, HO, V, K]``: sequence ``i`` reads row
            ``state_indices[i]`` and, under ``output_final_state``, writes its
            final state back to that row in place, so a serving stack keeps
            its paged state pool without a gather and scatter around the
            call.  Implies ``overwrite_initial_state``.  Forward only; cannot
            combine with ``checkpoint_every_n_tokens``.

        preceding_state_update: context parallelism: the ``final_state`` outputs of
            :func:`kimi_delta_attention_summary` over the preceding ranks, stacked in
            rank order as fp32 ``[P, HO, V, K]`` (``P = 0`` for the first rank).
            The op folds them, with ``preceding_transition``, on the device into
            a one-row composed state (fp32 ``[1, HO, V, K]``) that sequence 0 of
            this rank, the one crossing the rank boundary, starts from; the other
            sequences start from their ``initial_state`` rows (zero without one)
            and the caller's row 0 enters the fold as the state beyond the
            farthest rank.  The op returns that row and the backward takes it
            back as ``composed_initial_state``; ``initial_state`` itself then
            gets no gradient.
        preceding_transition: the matching ``transition`` outputs, fp32
            ``[P, HO, K, K]``.

    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v``, or
        ``(o, final_state, state_checkpoints)`` when ``checkpoint_every_n_tokens > 0``, or
        ``(o, final_state, state_checkpoints, composed_initial_state)`` when
        ``preceding_state_update`` is given (the composed state is fp32
        ``[N, HO, V, K]``, the forward's starting state; pass it as
        ``initial_state`` to ``torch.ops.cudnn.kimi_delta_attention_bwd``).
        ``final_state`` is empty unless ``output_final_state=True``; under
        ``overwrite_initial_state`` or ``state_indices`` it is the caller's buffer.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if state_indices is not None and initial_state is None:
        raise ValueError("kimi_delta_attention: state_indices selects rows of the initial_state pool, so initial_state is required")
    if overwrite_initial_state or state_indices is not None:
        if initial_state is None:
            raise ValueError("kimi_delta_attention: overwrite_initial_state requires initial_state")
        if preceding_state_update is not None or preceding_transition is not None:
            raise ValueError("kimi_delta_attention: preceding summaries cannot combine with overwrite_initial_state or state_indices")
        o, state_checkpoints = torch.ops.cudnn.kimi_delta_attention_fwd_overwrite_state(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            float(scale),
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
            batch_invariant=bool(batch_invariant),
            use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
            gate_domain=str(gate_domain),
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            checkpoint_dtype=checkpoint_dtype,
            plan_name=plan_name,
            state_indices=state_indices,
            output_final_state=bool(output_final_state) or bool(overwrite_initial_state),
        )
        final_state = initial_state if (output_final_state or overwrite_initial_state) else initial_state.new_empty(0)
    else:
        o, final_state, state_checkpoints, composed_initial_state = torch.ops.cudnn.kimi_delta_attention_fwd(
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
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
            gate_domain=str(gate_domain),
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            checkpoint_dtype=checkpoint_dtype,
            plan_name=plan_name,
            preceding_state_update=preceding_state_update,
            preceding_transition=preceding_transition,
        )
        if preceding_state_update is not None:
            return o, final_state, state_checkpoints, composed_initial_state
    if checkpoint_every_n_tokens > 0:
        return o, final_state, state_checkpoints
    return o, final_state


# ---------------------------------------------------------------------------
# KDA summary graph builder
# ---------------------------------------------------------------------------


def build_kda_summary_graph(
    total,
    N,
    HK,
    HV,
    HO,
    K,
    V,
    io_dtype,
    g_dtype,
    beta_dtype,
    state_dtype,
    cu_dtype,
    output_transition,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    gate_lower_bound,
    gate_domain,
    a_log_dtype=None,
    dt_bias_dtype=None,
):
    """Build a KDA forward-summary graph and its tensor placeholders."""
    graph = cudnn.pygraph()
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
    if a_log_dtype is not None:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([HO, K], data_type=dt_bias_dtype, name="dt_bias")
    fs_t, transition_t = graph.kda_summary(
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        output_transition=output_transition,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=None if gate_domain == "log" else gate_domain,
        name="kda_summary",
    )
    fs_t.set_data_type(cudnn.data_type.FLOAT)
    if transition_t is not None:
        transition_t.set_data_type(cudnn.data_type.FLOAT)
    return graph, dict(
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu=cu_t,
        state0=state0_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        fs=fs_t,
        transition=transition_t,
    )


# ---------------------------------------------------------------------------
# KDA summary custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::kimi_delta_attention_summary", mutates_args=())
def kda_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal KDA summary over a cached single-node KDA_SUMMARY pygraph.  Returns ``(final_state,
    transition)`` float32, ``transition`` zero-size unless ``output_transition``."""
    total, HK, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    HO = g.shape[1]
    if beta.shape[1] != HO:
        raise ValueError(f"kimi_delta_attention_summary: beta must carry HO = g's head count ({HO}); got {beta.shape[1]}")
    N = cu_seqlens.shape[0] - 1
    device = k.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention_summary: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention_summary: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    cu = cu_seqlens
    check_dtype("k", k, (torch.float16, torch.bfloat16))
    check_dtype("v", v, k.dtype)
    check_dtype("g", g, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("beta", beta, (torch.float32, k.dtype))
    if safe_gate:
        if a_log is not None:
            check_dtype("a_log", a_log, (torch.float32, torch.bfloat16, torch.float16))
        if dt_bias is not None:
            check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
    elif a_log is not None or dt_bias is not None:
        raise ValueError("kimi_delta_attention_summary: a_log/dt_bias require safe_gate=True")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention_summary: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention_summary: gate_domain='linear' cannot combine with safe_gate=True")
    if initial_state is not None:
        check_dtype("initial_state", initial_state, torch.float32)
        if initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    for tensor_name, tensor in (
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu_seqlens),
        ("initial_state", initial_state),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"kimi_delta_attention_summary: {tensor_name} must be on k's device ({device}); got {tensor.device}")
    state0 = initial_state if initial_state is not None else None

    cache_key = make_summary_cache_key(
        "kda_summary",
        total,
        N,
        HK,
        HV,
        HO,
        K,
        V,
        k.dtype,
        v.dtype,
        tuple(k.shape),
        tuple(v.shape),
        cu_seqlens.dtype,
        g.dtype,
        beta.dtype,
        state0.dtype if state0 is not None else None,
        output_transition,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        a_log.dtype if a_log is not None else None,
        dt_bias.dtype if dt_bias is not None else None,
        device,
        plan_name,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
    )
    if cache_key not in summary_cache:
        summary_cache[cache_key] = build_kda_summary_graph(
            total,
            N,
            HK,
            HV,
            HO,
            K,
            V,
            torch_dtype_to_cudnn(k.dtype),
            torch_dtype_to_cudnn(g.dtype),
            torch_dtype_to_cudnn(beta.dtype),
            torch_dtype_to_cudnn(state0.dtype) if state0 is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            bool(output_transition),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            bool(use_beta_sigmoid_in_kernel),
            bool(allow_neg_eigval),
            bool(safe_gate),
            float(gate_lower_bound) if gate_lower_bound is not None else None,
            str(gate_domain),
            a_log_dtype=torch_dtype_to_cudnn(a_log.dtype) if a_log is not None else None,
            dt_bias_dtype=torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
        )
        select_plan(summary_cache[cache_key][0], plan_name)

    graph, t = summary_cache[cache_key]

    final_state = torch.empty(N, HO, V, K, dtype=torch.float32, device=device)
    variant_pack = {
        t["k"]: k,
        t["v"]: v,
        t["g"]: g,
        t["beta"]: beta,
        t["cu"]: cu,
        t["fs"]: final_state,
    }
    if state0 is not None:
        variant_pack[t["state0"]] = state0
    if a_log is not None:
        variant_pack[t["a_log"]] = a_log
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
    transition = torch.empty(0, dtype=torch.float32, device=device)
    if output_transition:
        transition = torch.empty(N, HO, K, K, dtype=torch.float32, device=device)
        variant_pack[t["transition"]] = transition
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return final_state, transition


@kda_summary.register_fake
def kda_summary_fake(
    k,
    v,
    g,
    beta,
    cu_seqlens,
    initial_state=None,
    output_transition=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    """Return FakeTensor-compatible outputs for a KDA forward summary."""
    total, HK, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    HO = g.shape[1]
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention_summary: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention_summary: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention_summary: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention_summary: gate_domain='linear' cannot combine with safe_gate=True")
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    final_state = k.new_empty((N, HO, V, K), dtype=torch.float32)
    transition = k.new_empty((N, HO, K, K) if output_transition else (0,), dtype=torch.float32)
    return final_state, transition


# ---------------------------------------------------------------------------
# KDA summary-bwd graph builder
# ---------------------------------------------------------------------------


def build_kda_summary_bwd_graph(
    total,
    N,
    HQ,
    HK,
    HV,
    HO,
    K,
    V,
    io_dtype,
    g_dtype,
    beta_dtype,
    dstate_dtype,
    cu_dtype,
    scale,
    use_qk_l2norm,
    batch_invariant,
    use_beta_sigmoid,
    allow_neg_eigval,
    safe_gate,
    gate_lower_bound,
    gate_domain,
    a_log_dtype=None,
    dt_bias_dtype=None,
    output_transition=False,
    output_intermediates=False,
    checkpoint_every_n_tokens=0,
):
    """Build a KDA backward-summary graph and its tensor placeholders."""
    graph = cudnn.pygraph()
    q_t = graph.tensor([total, HQ, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, HK, K], data_type=io_dtype, name="k")
    g_t = graph.tensor([total, HO, K], data_type=g_dtype, name="g")
    beta_t = graph.tensor([total, HO], data_type=beta_dtype, name="beta")
    dO_t = graph.tensor([total, HV, V], data_type=io_dtype, name="dO")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    d_final_state_t = None
    if dstate_dtype is not None:
        d_final_state_t = graph.tensor([N, HO, V, K], data_type=dstate_dtype, name="d_final_state")
    a_log_t = None
    dt_bias_t = None
    if a_log_dtype is not None:
        a_log_t = graph.tensor([HO], data_type=a_log_dtype, name="a_log")
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([HO, K], data_type=dt_bias_dtype, name="dt_bias")
    d_initial_state_t, transition_t, *intermediate_t = graph.kda_summary_bwd(
        q=q_t,
        k=k_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=dO_t,
        d_final_state=d_final_state_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        output_transition=output_transition,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        batch_invariant=batch_invariant,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=None if gate_domain == "log" else gate_domain,
        output_intermediates=output_intermediates or None,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens or None,
        name="kda_summary_bwd",
    )
    d_initial_state_t.set_data_type(cudnn.data_type.FLOAT)
    if transition_t is not None:
        transition_t.set_data_type(cudnn.data_type.FLOAT)
    intermediates = {**dict.fromkeys(BACKWARD_CONTEXT_PORTS), **dict(zip(("piece_transitions", "piece_state_grads"), intermediate_t))}
    for name in ("q_inv_norm", "k_inv_norm", "piece_transitions", "piece_state_grads"):
        if intermediates[name] is not None:
            intermediates[name].set_data_type(cudnn.data_type.FLOAT)
    return graph, dict(
        q=q_t,
        k=k_t,
        g=g_t,
        beta=beta_t,
        dO=dO_t,
        cu=cu_t,
        d_final_state=d_final_state_t,
        a_log=a_log_t,
        dt_bias=dt_bias_t,
        d_initial_state=d_initial_state_t,
        transition=transition_t,
        **intermediates,
    )


# ---------------------------------------------------------------------------
# KDA summary-bwd custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("cudnn::kimi_delta_attention_summary_bwd", mutates_args=())
def kda_summary_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    d_final_state: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
    output_intermediates: bool = False,
    checkpoint_every_n_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal KDA backward summary over a cached single-node KDA_SUMMARY_BWD pygraph.  Returns
    ``d_initial_state`` float32 ``[N, HO, V, K]`` (``G`` with no ``d_final_state``, the full ``dh0`` otherwise),
    ``transition`` float32 ``[N, HO, K, K]`` in the backward's stored-domain orientation (``M_buf^T``), zero-size unless
    ``output_transition``, and the intermediates for the backward over the same tokens (``q_normalized``, ``k_normalized``,
    ``q_inv_norm``, ``k_inv_norm``, ``chunk_factors``, ``piece_transitions``, ``piece_state_grads``): the intermediates
    the backward would otherwise recompute, each zero-size unless ``output_intermediates`` emits it for this geometry
    (KDA emits the per-piece tables alone, for one sequence cut into the backward's pieces at the backward's
    ``checkpoint_every_n_tokens``; its l2norm and chunk factor are in-kernel)."""
    total, HQ, K = q.shape
    if output_intermediates and not output_transition:
        raise ValueError("kimi_delta_attention_summary_bwd: output_intermediates requires output_transition")
    HK = k.shape[1]
    HV, V = dO.shape[1], dO.shape[2]
    HO = g.shape[1]
    if beta.shape[1] != HO:
        raise ValueError(f"kimi_delta_attention_summary_bwd: beta must carry HO = g's head count ({HO}); got {beta.shape[1]}")
    N = cu_seqlens.shape[0] - 1
    device = dO.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention_summary_bwd: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention_summary_bwd: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    cu = cu_seqlens
    check_dtype("k", k, (torch.float16, torch.bfloat16))
    check_dtype("q", q, k.dtype)
    check_dtype("dO", dO, k.dtype)
    check_dtype("g", g, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("beta", beta, (torch.float32, k.dtype))
    if safe_gate:
        if a_log is not None:
            check_dtype("a_log", a_log, (torch.float32, torch.bfloat16, torch.float16))
        if dt_bias is not None:
            check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
    elif a_log is not None or dt_bias is not None:
        raise ValueError("kimi_delta_attention_summary_bwd: a_log/dt_bias require safe_gate=True")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention_summary_bwd: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention_summary_bwd: gate_domain='linear' cannot combine with safe_gate=True")
    if d_final_state is not None:
        check_dtype("d_final_state", d_final_state, torch.float32)
        if tuple(d_final_state.shape) != (N, HO, V, K):
            raise ValueError(f"kimi_delta_attention_summary_bwd: d_final_state must be [N, HO, V, K] = [{N}, {HO}, {V}, {K}]; got {tuple(d_final_state.shape)}")
    for tensor_name, tensor in (
        ("q", q),
        ("k", k),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu_seqlens),
        ("d_final_state", d_final_state),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"kimi_delta_attention_summary_bwd: {tensor_name} must be on dO's device ({device}); got {tensor.device}")

    cache_key = make_summary_cache_key(
        "kda_summary_bwd",
        total,
        N,
        HK,
        HV,
        HO,
        K,
        V,
        k.dtype,
        None,
        tuple(k.shape),
        None,
        cu_seqlens.dtype,
        g.dtype,
        beta.dtype,
        None,
        output_transition,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        a_log.dtype if a_log is not None else None,
        dt_bias.dtype if dt_bias is not None else None,
        device,
        plan_name,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
        scale=float(scale),
        d_final_state_dtype=d_final_state.dtype if d_final_state is not None else None,
        do_dtype=dO.dtype,
        do_shape=tuple(dO.shape),
        q_shape=tuple(q.shape),
    ) + (bool(output_intermediates), int(checkpoint_every_n_tokens))
    if cache_key not in summary_cache:
        summary_cache[cache_key] = build_kda_summary_bwd_graph(
            total,
            N,
            HQ,
            HK,
            HV,
            HO,
            K,
            V,
            torch_dtype_to_cudnn(k.dtype),
            torch_dtype_to_cudnn(g.dtype),
            torch_dtype_to_cudnn(beta.dtype),
            torch_dtype_to_cudnn(d_final_state.dtype) if d_final_state is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            float(scale),
            bool(use_qk_l2norm_in_kernel),
            bool(batch_invariant),
            bool(use_beta_sigmoid_in_kernel),
            bool(allow_neg_eigval),
            bool(safe_gate),
            float(gate_lower_bound) if gate_lower_bound is not None else None,
            str(gate_domain),
            output_transition=bool(output_transition),
            a_log_dtype=torch_dtype_to_cudnn(a_log.dtype) if a_log is not None else None,
            dt_bias_dtype=torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            output_intermediates=bool(output_intermediates),
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        )
        select_plan(summary_cache[cache_key][0], plan_name)

    graph, t = summary_cache[cache_key]

    d_initial_state = torch.empty(N, HO, V, K, dtype=torch.float32, device=device)
    variant_pack = {
        t["q"]: q,
        t["k"]: k,
        t["g"]: g,
        t["beta"]: beta,
        t["dO"]: dO,
        t["cu"]: cu,
        t["d_initial_state"]: d_initial_state,
    }
    if d_final_state is not None:
        variant_pack[t["d_final_state"]] = d_final_state
    if a_log is not None:
        variant_pack[t["a_log"]] = a_log
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
    transition = torch.empty(0, dtype=torch.float32, device=device)
    if output_transition:
        transition = torch.empty(N, HO, K, K, dtype=torch.float32, device=device)
        variant_pack[t["transition"]] = transition
    intermediates = backward_context_outputs({name: (list(t[name].dim) if t[name] is not None else None) for name in BACKWARD_CONTEXT_PORTS}, k.dtype, device)
    for name, tensor in intermediates.items():
        if t[name] is not None:
            variant_pack[t[name]] = tensor
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return (d_initial_state, transition, *(intermediates[name] for name in BACKWARD_CONTEXT_PORTS))


@kda_summary_bwd.register_fake
def kda_summary_bwd_fake(
    dO,
    q,
    k,
    g,
    beta,
    cu_seqlens,
    scale,
    d_final_state=None,
    output_transition=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
    output_intermediates=False,
    checkpoint_every_n_tokens=0,
):
    """Return FakeTensor-compatible outputs for a KDA backward summary."""
    HV, V = dO.shape[1], dO.shape[2]
    K = k.shape[2]
    HO = g.shape[1]
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"kimi_delta_attention_summary_bwd: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("kimi_delta_attention_summary_bwd: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if gate_domain not in ("log", "linear"):
        raise ValueError(f"kimi_delta_attention_summary_bwd: gate_domain must be 'log' or 'linear', got {gate_domain!r}")
    if gate_domain == "linear" and safe_gate:
        raise ValueError("kimi_delta_attention_summary_bwd: gate_domain='linear' cannot combine with safe_gate=True")
    if d_final_state is not None and tuple(d_final_state.shape) != (N, HO, V, K):
        raise ValueError(f"kimi_delta_attention_summary_bwd: d_final_state must be [N, HO, V, K] = [{N}, {HO}, {V}, {K}]; got {tuple(d_final_state.shape)}")
    if output_intermediates and not output_transition:
        raise ValueError("kimi_delta_attention_summary_bwd: output_intermediates requires output_transition")
    d_initial_state = dO.new_empty((N, HO, V, K), dtype=torch.float32)
    transition = dO.new_empty((N, HO, K, K) if output_transition else (0,), dtype=torch.float32)
    shapes = dict.fromkeys(BACKWARD_CONTEXT_PORTS)
    if output_intermediates:
        shapes = backward_context_shapes(
            "kda",
            total=q.shape[0],
            num_seqs=N,
            h_q=q.shape[1],
            h_k=k.shape[1],
            h_o=HO,
            d_k=K,
            d_v=V,
            use_qk_l2norm=use_qk_l2norm_in_kernel,
        )
    intermediates = backward_context_outputs(shapes, k.dtype, dO.device)
    return (d_initial_state, transition, *(intermediates[name] for name in BACKWARD_CONTEXT_PORTS))


# ---------------------------------------------------------------------------
# Summary autograd registration
# ---------------------------------------------------------------------------


torch.library.register_autograd(
    "cudnn::kimi_delta_attention_summary",
    summary_backward,
    setup_context=summary_setup_context,
)
torch.library.register_autograd(
    "cudnn::kimi_delta_attention_summary_bwd",
    summary_backward,
    setup_context=summary_setup_context,
)


# ---------------------------------------------------------------------------
# Summary public API
# ---------------------------------------------------------------------------


def kimi_delta_attention_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
):
    """Kimi Delta Attention (KDA) per-span summary for context parallelism: the state-only pass over each sequence, returning
    ``(H, M)`` float32 with ``H`` ``[N, HO, V, K]`` the final state under ``initial_state`` (zero when ``None``) and ``M``
    ``[N, HO, K, K]`` the span transition in stored domain (``M_buf = M^T``), so that across spans ``X_final = X_init @
    M_buf + X_H`` and ``X_dh0 = X_dht @ M_buf^T + X_G``.  Both outputs are non-differentiable.  THD layout and gate
    semantics match :func:`kimi_delta_attention` (k ``[total_tokens, HK, K]``, v ``[total_tokens, HV, V]``, g
    ``[total_tokens, HO, K]`` per-key-channel log decay, beta ``[total_tokens, HO]``, cu_seqlens ``[N+1]``); ``initial_state`` is float32.

    Args:
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` in the kernel; ``beta`` then holds raw logits.
        safe_gate: ``g`` is read as ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``; ``a_log`` ``[HO]`` and
            ``dt_bias`` ``[HO, K]`` (fp32/bf16/fp16) default to unit amplitude / zero bias and require ``safe_gate=True``.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        gate_domain: ``"log"`` (``g = ln(alpha)``) or ``"linear"`` (``g = alpha in (0, 1]^K``, 1e-10 floor, ``dG`` with respect
            to ``alpha``); not with ``safe_gate``.
        plan_name: pin one execution plan by name (e.g. ``kda_summary_frost``).
    """
    if k.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    final_state, transition = torch.ops.cudnn.kimi_delta_attention_summary(
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state=initial_state,
        output_transition=True,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        allow_neg_eigval=bool(allow_neg_eigval),
        safe_gate=bool(safe_gate),
        gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
        gate_domain=str(gate_domain),
        a_log=a_log,
        dt_bias=dt_bias,
        batch_invariant=bool(batch_invariant),
        plan_name=plan_name,
    )
    return final_state, transition


def kimi_delta_attention_summary_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dO: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    d_final_state: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
    output_intermediates: bool = False,
    checkpoint_every_n_tokens: int = 0,
):
    """Kimi Delta Attention (KDA) per-span backward summary for context parallelism: the reverse state-gradient recurrence
    over each sequence, returning ``G`` ``[N, HO, V, K]`` float32, the incoming state gradient ``d_initial_state`` under
    ``d_final_state`` (the pure summary term with ``None``, the full ``dh0`` otherwise), and with ``output_transition=True``
    also ``transition`` ``[N, HO, K, K]`` float32, the span transition in the backward's stored-domain orientation
    (``M_buf^T``) so that ``X_dh0 = X_dht @ transition + X_G``.  The outputs are non-differentiable.  THD layout and gate
    semantics match :func:`kimi_delta_attention` (q ``[total_tokens, HQ, K]``, k ``[total_tokens, HK, K]``, g
    ``[total_tokens, HO, K]``, beta ``[total_tokens, HO]``, dO ``[total_tokens, HO, V]`` in the io dtype, cu_seqlens
    ``[N+1]``); ``d_final_state`` is float32 ``[N, HO, V, K]``.

    Args:
        scale: attention scale applied to ``q``; defaults to ``1 / sqrt(K)``.
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` in the kernel; ``beta`` then holds raw logits.
        safe_gate: ``g`` is read as ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``; ``a_log`` ``[HO]`` and
            ``dt_bias`` ``[HO, K]`` (fp32/bf16/fp16) default to unit amplitude / zero bias and require ``safe_gate=True``.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        gate_domain: ``"log"`` (``g = ln(alpha)``) or ``"linear"`` (``g = alpha in (0, 1]^K``, 1e-10 floor, ``dG`` with respect
            to ``alpha``); not with ``safe_gate``.
        plan_name: pin one execution plan by name (e.g. ``kda_summary_frost``).
        output_intermediates: with ``output_transition``, also return the intermediates for the backward over the same tokens as
            a third value, a dict of the raw backward's keyword arguments (``q_normalized``, ``k_normalized``,
            ``q_inv_norm``, ``k_inv_norm``, ``chunk_factors``, ``piece_transitions``, ``piece_state_grads``): the
            intermediates the backward would otherwise recompute, so ``torch.ops.cudnn.kimi_delta_attention_bwd(...,
            **intermediates)`` skips its per-piece summary passes.  KDA emits the per-piece tables alone (its l2norm and
            chunk factor are in-kernel); the other entries are zero-size.  The tables are emitted for one local sequence
            cut at ``checkpoint_every_n_tokens``, which must equal the backward's.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    d_initial_state, transition, *intermediates = torch.ops.cudnn.kimi_delta_attention_summary_bwd(
        dO,
        q,
        k,
        g,
        beta,
        cu_seqlens,
        float(scale),
        d_final_state=d_final_state,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        allow_neg_eigval=bool(allow_neg_eigval),
        safe_gate=bool(safe_gate),
        gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
        gate_domain=str(gate_domain),
        a_log=a_log,
        dt_bias=dt_bias,
        batch_invariant=bool(batch_invariant),
        output_transition=bool(output_transition),
        plan_name=plan_name,
        output_intermediates=bool(output_intermediates),
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
    )
    if output_intermediates:
        return d_initial_state, transition, dict(zip(BACKWARD_CONTEXT_PORTS, intermediates))
    return (d_initial_state, transition) if output_transition else d_initial_state
