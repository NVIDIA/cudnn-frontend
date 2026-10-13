# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch custom operator for the Mamba-2 state space duality (SSD) layer.

Mamba-2 is the gated delta rule without the erase term, with one per-token scalar dt
playing both the decay and the write strength:

    dt_t  = clamp(softplus(dt_raw_t + dt_bias_h), dt_min, dt_max),
    S_t   = exp(A_h dt_t) S_{t-1} + dt_t B_t^T x_t,
    y_t   = C_t S_t + D_h x_t              (* z_t sigmoid(z_t) when the z gate is fused),

where ``S_t`` is the recurrent state ``[N, P]`` (``N`` the state dimension, ``P`` the
head dimension), ``A_h < 0`` a per-head decay rate, ``D`` a per-head or per-channel skip,
and ``B`` / ``C`` grouped over ``G`` groups of the ``H`` heads.

Layout follows the graph-API MAMBA2 node (THD), token-packed ``[total_tokens, heads, dim]``
tensors plus ``cu_seqlens`` sequence boundaries.  In the node's vocabulary ``C`` is ``q``,
``B`` is ``k``, ``x`` is ``v``, ``dt`` is ``g`` and ``A`` is ``a``.

The op is a thin adapter over the graph API; forward and backward execute cached
single-node ``MAMBA2`` / ``MAMBA2_BWD`` pygraphs.  Engine selection happens at graph
planning time over the manifest's python engines (``Mamba2FrostEngine`` on
SM100 / SM103 / SM107).  Registered through ``torch.library.custom_op`` so it composes
with autograd, ``torch.compile``, and DDP.

Graph caching ensures cuDNN graphs are built once per unique configuration and reused
across calls.
"""

from typing import Dict, Optional, Tuple

import torch
import cudnn

from .common import execute_graph, make_execution_resources, select_plan, torch_dtype_to_cudnn

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


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
        raise TypeError(f"mamba2: {name} must be {names} (kernel-native; callers convert), got {t.dtype}")


def make_fprop_cache_key(
    total,
    N,
    H,
    G,
    K,
    V,
    io_dtype,
    cu_dtype,
    g_dtype,
    a_dtype,
    dt_bias_dtype,
    d_dtype,
    d_per_channel,
    z_dtype,
    state_dtype,
    output_final_state,
    output_pre_gate,
    batch_invariant,
    checkpoint,
    dt_softplus,
    dt_min,
    dt_max,
    device,
    plan_name,
    overwrite_initial_state,
    state_pool_rows=None,
):
    return (
        "fprop",
        total,
        N,
        H,
        G,
        K,
        V,
        io_dtype,
        cu_dtype,
        g_dtype,
        a_dtype,
        dt_bias_dtype,
        d_dtype,
        bool(d_per_channel),
        z_dtype,
        state_dtype,
        bool(output_final_state),
        bool(output_pre_gate),
        bool(batch_invariant),
        checkpoint,
        bool(dt_softplus),
        float(dt_min),
        float(dt_max),
        device,
        plan_name,
        bool(overwrite_initial_state),
        state_pool_rows,
    )


# ---------------------------------------------------------------------------
# Forward graph builder
# ---------------------------------------------------------------------------


def build_fprop_graph(
    total,
    N,
    H,
    G,
    K,
    V,
    io_dtype,
    g_dtype,
    a_dtype,
    dt_bias_dtype,
    d_dtype,
    d_per_channel,
    z_dtype,
    state_dtype,
    cu_dtype,
    output_final_state,
    output_pre_gate,
    batch_invariant,
    checkpoint,
    dt_softplus,
    dt_min,
    dt_max,
    overwrite_initial_state=False,
    state_indices_dtype=None,
    state_pool_rows=None,
):
    graph = cudnn.pygraph()
    q_t = graph.tensor([total, G, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, G, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, H, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, H], data_type=g_dtype, name="g")
    a_t = graph.tensor([H], data_type=a_dtype, name="a")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    state0_t = None
    if state_dtype is not None:
        state_rows = N if state_pool_rows is None else state_pool_rows
        state0_t = graph.tensor([state_rows, H, V, K], data_type=state_dtype, name="initial_state")
    state_indices_t = None
    if state_indices_dtype is not None:
        state_indices_t = graph.tensor([N], data_type=state_indices_dtype, name="state_indices")
    pool_ports = {"state_indices": state_indices_t} if state_indices_t is not None else {}
    dt_bias_t = None
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([H], data_type=dt_bias_dtype, name="dt_bias")
    d_t = None
    if d_dtype is not None:
        d_t = graph.tensor([H, V] if d_per_channel else [H], data_type=d_dtype, name="d")
    z_t = None
    if z_dtype is not None:
        z_t = graph.tensor([total, H, V], data_type=z_dtype, name="z")
    O_t, fs_t, state_checkpoints_t, o_pre_t = graph.mamba2(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        **pool_ports,
        dt_bias=dt_bias_t,
        d=d_t,
        z=z_t,
        output_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=None if dt_max == float("inf") else dt_max,
        output_pre_gate=output_pre_gate or None,
        batch_invariant=batch_invariant,
        overwrite_initial_state=overwrite_initial_state or None,
        name="mamba2",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu=cu_t,
        state0=state0_t,
        state_indices=state_indices_t,
        dt_bias=dt_bias_t,
        d=d_t,
        z=z_t,
        O=O_t,
        fs=fs_t,
        state_checkpoints=state_checkpoints_t,
        O_pre=o_pre_t,
    )


# ---------------------------------------------------------------------------
# Forward custom op
# ---------------------------------------------------------------------------


def run_mamba2_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    output_pre_gate: bool = False,
    checkpoint_every_n_tokens: int = 0,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
    final_state_out: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal Mamba-2 forward over a cached single-node MAMBA2 pygraph in THD layout.

    Returns ``(y, final_state, state_checkpoints, y_pre)``; ``final_state`` / ``state_checkpoints`` / ``y_pre`` are
    zero-size tensors when ``output_final_state`` is ``False`` / ``checkpoint_every_n_tokens`` is ``0`` /
    ``output_pre_gate`` is ``False`` or ``z`` is absent.  ``final_state_out`` binds the final_state output to a caller
    buffer (the in-place op passes ``initial_state``); ``state_indices`` (int32 ``[N]``) makes ``initial_state`` a
    pool whose row ``state_indices[i]`` seeds sequence ``i`` and receives its final state.
    """
    total, H, V = x.shape
    G, K = C.shape[1], C.shape[2]
    if B.shape != C.shape:
        raise ValueError(f"mamba2: B and C must share one head count and shape; got {tuple(B.shape)} and {tuple(C.shape)}")
    if H % G != 0:
        raise ValueError(f"mamba2: the head count ({H}) must be a multiple of the B / C group count ({G})")
    if dt.shape != (total, H):
        raise ValueError(f"mamba2: dt must be [total_tokens, H] = [{total}, {H}]; got {tuple(dt.shape)}")
    if A.shape != (H,):
        raise ValueError(f"mamba2: A must be [H] = [{H}]; got {tuple(A.shape)}")
    N = cu_seqlens.shape[0] - 1
    device = x.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"mamba2: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    check_dtype("x", x, (torch.float16, torch.bfloat16))
    check_dtype("B", B, x.dtype)
    check_dtype("C", C, x.dtype)
    check_dtype("dt", dt, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("A", A, (torch.float32, torch.bfloat16, torch.float16))
    if dt_bias is not None:
        check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
        if dt_bias.shape != (H,):
            raise ValueError(f"mamba2: dt_bias must be [H] = [{H}]; got {tuple(dt_bias.shape)}")
    if D is not None:
        check_dtype("D", D, (torch.float32, torch.bfloat16, torch.float16))
        if D.shape not in ((H,), (H, V)):
            raise ValueError(f"mamba2: D must be [H] or [H, P] = [{H}] / [{H}, {V}]; got {tuple(D.shape)}")
    if z is not None:
        check_dtype("z", z, x.dtype)
        if z.shape != x.shape:
            raise ValueError(f"mamba2: z must match x's shape {tuple(x.shape)}; got {tuple(z.shape)}")
    dt_max_value = float("inf") if dt_max is None else float(dt_max)
    if float(dt_min) > dt_max_value:
        raise ValueError(f"mamba2: dt_min ({dt_min}) must not exceed dt_max ({dt_max_value})")
    state_pool_rows = None
    if state_indices is not None:
        if initial_state is None:
            raise ValueError("mamba2: state_indices selects rows of the initial_state pool, so initial_state is required")
        check_dtype("state_indices", state_indices, torch.int32)
        if state_indices.dim() != 1 or state_indices.shape[0] != N:
            raise ValueError(f"mamba2: state_indices must be [N] = [{N}]; got {tuple(state_indices.shape)}")
        if int(checkpoint_every_n_tokens):
            raise ValueError("mamba2: state_indices cannot be combined with checkpoint_every_n_tokens")
        if (
            tuple(initial_state.stride()[1:]) != (V * K, K, 1)
            or initial_state.stride(0) < H * V * K
            or initial_state.stride(0) % (16 // initial_state.element_size())
        ):
            raise ValueError("mamba2: state_indices requires a pool whose rows are dense [H, P, N] blocks on a 16-byte aligned slot stride")
        state_pool_rows = int(initial_state.shape[0])
    if initial_state is not None:
        check_dtype("initial_state", initial_state, (torch.float32, torch.bfloat16))
        if state_indices is None and initial_state.shape[0] != N:
            raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    for tensor_name, tensor in (
        ("dt", dt),
        ("A", A),
        ("B", B),
        ("C", C),
        ("cu_seqlens", cu_seqlens),
        ("D", D),
        ("dt_bias", dt_bias),
        ("z", z),
        ("initial_state", initial_state),
        ("state_indices", state_indices),
    ):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"mamba2: {tensor_name} must be on x's device ({device}); got {tensor.device}")
    state0 = initial_state if initial_state is not None else None
    state_out_dtype = state0.dtype if state0 is not None else torch.float32
    checkpoint = int(checkpoint_every_n_tokens)
    want_pre = bool(output_pre_gate) and z is not None
    cache_key = make_fprop_cache_key(
        total,
        N,
        H,
        G,
        K,
        V,
        x.dtype,
        cu_seqlens.dtype,
        dt.dtype,
        A.dtype,
        dt_bias.dtype if dt_bias is not None else None,
        D.dtype if D is not None else None,
        D is not None and D.dim() == 2,
        z.dtype if z is not None else None,
        state0.dtype if state0 is not None else None,
        output_final_state,
        want_pre,
        batch_invariant,
        checkpoint,
        dt_softplus,
        dt_min,
        dt_max_value,
        device,
        plan_name,
        final_state_out is not None,
        state_pool_rows,
    )
    if cache_key not in fprop_cache:
        fprop_cache[cache_key] = build_fprop_graph(
            total,
            N,
            H,
            G,
            K,
            V,
            torch_dtype_to_cudnn(x.dtype),
            torch_dtype_to_cudnn(dt.dtype),
            torch_dtype_to_cudnn(A.dtype),
            torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            torch_dtype_to_cudnn(D.dtype) if D is not None else None,
            D is not None and D.dim() == 2,
            torch_dtype_to_cudnn(z.dtype) if z is not None else None,
            torch_dtype_to_cudnn(state0.dtype) if state0 is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            bool(output_final_state),
            want_pre,
            bool(batch_invariant),
            checkpoint,
            bool(dt_softplus),
            float(dt_min),
            dt_max_value,
            overwrite_initial_state=final_state_out is not None,
            state_indices_dtype=torch_dtype_to_cudnn(state_indices.dtype) if state_indices is not None else None,
            state_pool_rows=state_pool_rows,
        )
        select_plan(fprop_cache[cache_key][0], plan_name)

    graph, t = fprop_cache[cache_key]

    y = torch.empty(total, H, V, dtype=x.dtype, device=device)
    variant_pack = {
        t["q"]: C,
        t["k"]: B,
        t["v"]: x,
        t["g"]: dt,
        t["a"]: A,
        t["cu"]: cu_seqlens,
        t["O"]: y,
    }
    if state0 is not None:
        variant_pack[t["state0"]] = state0
    if state_indices is not None:
        variant_pack[t["state_indices"]] = state_indices
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
    if D is not None:
        variant_pack[t["d"]] = D
    if z is not None:
        variant_pack[t["z"]] = z
    final_state = torch.empty(0, dtype=state_out_dtype, device=device)
    if final_state_out is not None:
        final_state = final_state_out
        variant_pack[t["fs"]] = final_state
    elif output_final_state:
        final_state = torch.empty(N, H, V, K, dtype=state_out_dtype, device=device)
        variant_pack[t["fs"]] = final_state
    state_checkpoints = torch.empty(0, dtype=x.dtype, device=device)
    if checkpoint > 0:
        total_checkpoints = max(total // checkpoint + N, 1)
        state_checkpoints = torch.empty(total_checkpoints, H, V, K, dtype=x.dtype, device=device)
        variant_pack[t["state_checkpoints"]] = state_checkpoints
    y_pre = torch.empty(0, dtype=x.dtype, device=device)
    if want_pre:
        y_pre = torch.empty(total, H, V, dtype=x.dtype, device=device)
        variant_pack[t["O_pre"]] = y_pre
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return y, final_state, state_checkpoints, y_pre


@torch.library.custom_op("cudnn::mamba2_fwd", mutates_args=())
def mamba2_fwd_op(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    output_pre_gate: bool = False,
    checkpoint_every_n_tokens: int = 0,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional Mamba-2 forward: :func:`run_mamba2_fwd` with fresh outputs; the autograd formula registers below."""
    return run_mamba2_fwd(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        cu_seqlens=cu_seqlens,
        D=D,
        dt_bias=dt_bias,
        z=z,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_pre_gate=output_pre_gate,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        batch_invariant=batch_invariant,
        plan_name=plan_name,
    )


@torch.library.custom_op("cudnn::mamba2_fwd_overwrite_state", mutates_args=("initial_state",))
def mamba2_fwd_overwrite_state_op(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
    state_indices: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-place Mamba-2 forward: the final state overwrites ``initial_state`` (or the ``state_indices`` rows of the
    pool it addresses).  Inference only: no autograd formula."""
    y, _, state_checkpoints, _ = run_mamba2_fwd(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        cu_seqlens=cu_seqlens,
        D=D,
        dt_bias=dt_bias,
        z=z,
        initial_state=initial_state,
        output_final_state=bool(output_final_state),
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        batch_invariant=batch_invariant,
        plan_name=plan_name,
        final_state_out=initial_state if output_final_state else None,
        state_indices=state_indices,
    )
    return y, state_checkpoints


@mamba2_fwd_op.register_fake
def mamba2_fwd_fake(
    x,
    dt,
    A,
    B,
    C,
    cu_seqlens,
    D=None,
    dt_bias=None,
    z=None,
    initial_state=None,
    output_final_state=False,
    output_pre_gate=False,
    checkpoint_every_n_tokens=0,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    total, H, V = x.shape
    K = C.shape[2]
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"mamba2: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences")
    y = x.new_empty(total, H, V)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    final = x.new_empty((N, H, V, K) if output_final_state else (0,), dtype=state_dtype)
    if checkpoint_every_n_tokens > 0:
        total_checkpoints = max(total // int(checkpoint_every_n_tokens) + N, 1)
        state_checkpoints = x.new_empty(total_checkpoints, H, V, K)
    else:
        state_checkpoints = x.new_empty(0)
    y_pre = x.new_empty((total, H, V) if (output_pre_gate and z is not None) else (0,))
    return y, final, state_checkpoints, y_pre


@mamba2_fwd_overwrite_state_op.register_fake
def mamba2_fwd_overwrite_state_fake(
    x,
    dt,
    A,
    B,
    C,
    cu_seqlens,
    initial_state,
    D=None,
    dt_bias=None,
    z=None,
    checkpoint_every_n_tokens=0,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
    plan_name=None,
    state_indices=None,
    output_final_state=True,
):
    y, _, state_checkpoints, _ = mamba2_fwd_fake(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        cu_seqlens=cu_seqlens,
        D=D,
        dt_bias=dt_bias,
        z=z,
        initial_state=initial_state if state_indices is None else None,
        output_final_state=bool(output_final_state),
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        batch_invariant=batch_invariant,
        plan_name=plan_name,
    )
    return y, state_checkpoints


# ---------------------------------------------------------------------------
# Backward graph + custom op
# ---------------------------------------------------------------------------


def make_bprop_cache_key(*parts):
    return tuple(parts)


def build_bprop_graph(
    total,
    N,
    H,
    G,
    K,
    V,
    io_dtype,
    g_dtype,
    a_dtype,
    dt_bias_dtype,
    d_dtype,
    d_per_channel,
    z_dtype,
    state_dtype,
    dstate_in_dtype,
    cu_dtype,
    checkpoint_rows,
    checkpoint,
    batch_invariant,
    dt_softplus,
    dt_min,
    dt_max,
):
    graph = cudnn.pygraph()
    q_t = graph.tensor([total, G, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, G, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, H, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, H], data_type=g_dtype, name="g")
    a_t = graph.tensor([H], data_type=a_dtype, name="a")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    dO_t = graph.tensor([total, H, V], data_type=io_dtype, name="dO")
    checkpoints_t = None
    if checkpoint_rows is not None:
        checkpoints_t = graph.tensor([checkpoint_rows, H, V, K], data_type=io_dtype, name="state_checkpoints")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, H, V, K], data_type=state_dtype, name="initial_state")
    dfs_t = None
    if dstate_in_dtype is not None:
        dfs_t = graph.tensor([N, H, V, K], data_type=dstate_in_dtype, name="d_final_state")
    dt_bias_t = None
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([H], data_type=dt_bias_dtype, name="dt_bias")
    d_t = None
    if d_dtype is not None:
        d_t = graph.tensor([H, V] if d_per_channel else [H], data_type=d_dtype, name="d")
    z_t = None
    o_pre_t = None
    if z_dtype is not None:
        z_t = graph.tensor([total, H, V], data_type=z_dtype, name="z")
        o_pre_t = graph.tensor([total, H, V], data_type=io_dtype, name="O_pre")
    dQ_t, dK_t, dV_t, dG_t, dA_t, dstate0_t, dDt_t, dD_t, dZ_t = graph.mamba2_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        dO=dO_t,
        state_checkpoints=checkpoints_t,
        initial_state=state0_t,
        d_final_state=dfs_t,
        dt_bias=dt_bias_t,
        d=d_t,
        z=z_t,
        O_pre=o_pre_t,
        checkpoint_every_n_tokens=checkpoint or None,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=None if dt_max == float("inf") else dt_max,
        batch_invariant=batch_invariant,
        name="mamba2_bwd",
    )
    return graph, dict(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu=cu_t,
        dO=dO_t,
        checkpoints=checkpoints_t,
        state0=state0_t,
        dfs=dfs_t,
        dt_bias=dt_bias_t,
        d=d_t,
        z=z_t,
        O_pre=o_pre_t,
        dQ=dQ_t,
        dK=dK_t,
        dV=dV_t,
        dG=dG_t,
        dA=dA_t,
        dstate0=dstate0_t,
        d_dt_bias=dDt_t,
        dD=dD_t,
        dZ=dZ_t,
    )


@torch.library.custom_op("cudnn::mamba2_bwd", mutates_args=())
def mamba2_bwd_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    d_final_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    y_pre: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal Mamba-2 backward over a cached single-node MAMBA2_BWD pygraph in THD layout.

    ``state_checkpoints`` is the forward's per-chunk state series (io dtype, chunk cadence); when given at the kernel
    chunk cadence the engine consumes it instead of regenerating it.  ``y_pre`` is the forward's pre-gate output,
    required with ``z``.  Returns ``(dx, ddt, dA, dB, dC, dD, ddt_bias, dz, d_initial_state)``; the gradients of
    absent inputs come back as zero-size tensors.
    """
    total, H, V = x.shape
    G, K = C.shape[1], C.shape[2]
    N = cu_seqlens.shape[0] - 1
    device = x.device
    if 0 in dy.stride():
        dy = dy.contiguous()
    check_dtype("x", x, (torch.float16, torch.bfloat16))
    check_dtype("B", B, x.dtype)
    check_dtype("C", C, x.dtype)
    check_dtype("dy", dy, x.dtype)
    if d_final_state is not None and 0 in d_final_state.stride():
        d_final_state = d_final_state.contiguous()
    if z is not None:
        if y_pre is None or y_pre.shape != x.shape:
            raise ValueError("mamba2: the backward of the z gate needs the forward's pre-gate output, shaped like x")
        check_dtype("y_pre", y_pre, x.dtype)
    if state_checkpoints is not None:
        check_dtype("state_checkpoints", state_checkpoints, x.dtype)
    dstate_dtype = initial_state.dtype if initial_state is not None else torch.float32
    if d_final_state is not None:
        check_dtype("d_final_state", d_final_state, (torch.float32, torch.bfloat16))
        if d_final_state.dtype != dstate_dtype:
            raise TypeError(f"mamba2: d_final_state must be {dstate_dtype} (one state dtype per kernel)")
    dt_max_value = float("inf") if dt_max is None else float(dt_max)
    checkpoint = int(checkpoint_every_n_tokens) if state_checkpoints is not None else 0
    cache_key = make_bprop_cache_key(
        total,
        N,
        H,
        G,
        K,
        V,
        x.dtype,
        cu_seqlens.dtype,
        dt.dtype,
        A.dtype,
        dt_bias.dtype if dt_bias is not None else None,
        D.dtype if D is not None else None,
        D is not None and D.dim() == 2,
        z.dtype if z is not None else None,
        initial_state.dtype if initial_state is not None else None,
        d_final_state.dtype if d_final_state is not None else None,
        state_checkpoints.shape[0] if state_checkpoints is not None else None,
        checkpoint,
        batch_invariant,
        dt_softplus,
        dt_min,
        dt_max_value,
        device,
        plan_name,
    )
    if cache_key not in bprop_cache:
        bprop_cache[cache_key] = build_bprop_graph(
            total,
            N,
            H,
            G,
            K,
            V,
            torch_dtype_to_cudnn(x.dtype),
            torch_dtype_to_cudnn(dt.dtype),
            torch_dtype_to_cudnn(A.dtype),
            torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            torch_dtype_to_cudnn(D.dtype) if D is not None else None,
            D is not None and D.dim() == 2,
            torch_dtype_to_cudnn(z.dtype) if z is not None else None,
            torch_dtype_to_cudnn(initial_state.dtype) if initial_state is not None else None,
            torch_dtype_to_cudnn(d_final_state.dtype) if d_final_state is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            state_checkpoints.shape[0] if state_checkpoints is not None else None,
            checkpoint,
            bool(batch_invariant),
            bool(dt_softplus),
            float(dt_min),
            dt_max_value,
        )
        select_plan(bprop_cache[cache_key][0], plan_name)

    graph, t = bprop_cache[cache_key]

    dx = torch.empty(total, H, V, dtype=x.dtype, device=device)
    ddt = torch.empty(total, H, dtype=dt.dtype, device=device)
    dB = torch.empty(total, G, K, dtype=x.dtype, device=device)
    dC = torch.empty(total, G, K, dtype=x.dtype, device=device)
    dA = torch.empty(H, dtype=A.dtype, device=device)
    variant_pack = {
        t["q"]: C,
        t["k"]: B,
        t["v"]: x,
        t["g"]: dt,
        t["a"]: A,
        t["cu"]: cu_seqlens,
        t["dO"]: dy,
        t["dQ"]: dC,
        t["dK"]: dB,
        t["dV"]: dx,
        t["dG"]: ddt,
        t["dA"]: dA,
    }
    if state_checkpoints is not None:
        variant_pack[t["checkpoints"]] = state_checkpoints
    dstate0 = torch.empty(0, dtype=dstate_dtype, device=device)
    if initial_state is not None:
        variant_pack[t["state0"]] = initial_state
        dstate0 = torch.empty(initial_state.shape, dtype=dstate_dtype, device=device)
        variant_pack[t["dstate0"]] = dstate0
    if d_final_state is not None:
        variant_pack[t["dfs"]] = d_final_state
    ddt_bias = torch.empty(0, dtype=torch.float32, device=device)
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
        ddt_bias = torch.empty_like(dt_bias)
        variant_pack[t["d_dt_bias"]] = ddt_bias
    dD = torch.empty(0, dtype=torch.float32, device=device)
    if D is not None:
        variant_pack[t["d"]] = D
        dD = torch.empty_like(D)
        variant_pack[t["dD"]] = dD
    dz = torch.empty(0, dtype=x.dtype, device=device)
    if z is not None:
        variant_pack[t["z"]] = z
        variant_pack[t["O_pre"]] = y_pre
        dz = torch.empty(total, H, V, dtype=z.dtype, device=device)
        variant_pack[t["dZ"]] = dz
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return dx, ddt, dA, dB, dC, dD, ddt_bias, dz, dstate0


@mamba2_bwd_op.register_fake
def mamba2_bwd_fake(
    dy,
    x,
    dt,
    A,
    B,
    C,
    cu_seqlens,
    D=None,
    dt_bias=None,
    z=None,
    initial_state=None,
    d_final_state=None,
    state_checkpoints=None,
    y_pre=None,
    checkpoint_every_n_tokens=0,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    total, H, V = x.shape
    G, K = C.shape[1], C.shape[2]
    dstate_dtype = initial_state.dtype if initial_state is not None else torch.float32
    dx = x.new_empty(total, H, V)
    ddt = dt.new_empty(total, H)
    dA = A.new_empty(H)
    dB = x.new_empty(total, G, K)
    dC = x.new_empty(total, G, K)
    dD = D.new_empty(D.shape) if D is not None else x.new_empty(0, dtype=torch.float32)
    ddt_bias = dt_bias.new_empty(H) if dt_bias is not None else x.new_empty(0, dtype=torch.float32)
    dz = z.new_empty(total, H, V) if z is not None else x.new_empty(0)
    dstate0 = initial_state.new_empty(initial_state.shape) if initial_state is not None else x.new_empty(0, dtype=dstate_dtype)
    return dx, ddt, dA, dB, dC, dD, ddt_bias, dz, dstate0


def mamba2_setup_context(ctx, inputs, output):
    (
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens,
        D,
        dt_bias,
        z,
        initial_state,
        output_final_state,
        output_pre_gate,
        checkpoint_every_n_tokens,
        dt_softplus,
        dt_min,
        dt_max,
        batch_invariant,
        plan_name,
    ) = inputs
    saved = [x, dt, A, B, C, cu_seqlens]
    ctx.checkpoint_reuse = checkpoint_every_n_tokens > 0 and checkpoint_every_n_tokens % 128 == 0 and output[2].numel() > 0
    ctx.checkpoint_every_n_tokens = checkpoint_every_n_tokens if ctx.checkpoint_reuse else 0
    if ctx.checkpoint_reuse:
        saved.append(output[2])
    ctx.has_d = D is not None
    ctx.has_dt_bias = dt_bias is not None
    ctx.has_z = z is not None
    ctx.has_initial_state = initial_state is not None
    for tensor in (D, dt_bias, z, initial_state):
        if tensor is not None:
            saved.append(tensor)
    if ctx.has_z:
        saved.append(output[3])
    ctx.save_for_backward(*saved)
    ctx.dt_softplus = bool(dt_softplus)
    ctx.dt_min = float(dt_min)
    ctx.dt_max = dt_max
    ctx.batch_invariant = bool(batch_invariant)
    ctx.plan_name = plan_name
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2], output[3])


def mamba2_backward(ctx, dy, dFinal, dstate_checkpoints, dy_pre):
    # read ctx.saved_tensors once
    saved_tensors = list(ctx.saved_tensors)
    x, dt, A, B, C, cu_seqlens = saved_tensors[:6]
    rest = saved_tensors[6:]
    state_checkpoints = rest.pop(0) if ctx.checkpoint_reuse else None
    D = rest.pop(0) if ctx.has_d else None
    dt_bias = rest.pop(0) if ctx.has_dt_bias else None
    z = rest.pop(0) if ctx.has_z else None
    initial_state = rest.pop(0) if ctx.has_initial_state else None
    y_pre = rest.pop(0) if ctx.has_z else None

    if dy is None:
        dy = torch.zeros_like(x)
    dstate_in = dFinal if (dFinal is not None and dFinal.numel() > 0) else None
    dx, ddt, dA, dB, dC, dD, ddt_bias, dz, dstate0 = torch.ops.cudnn.mamba2_bwd(
        dy,
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens,
        D=D,
        dt_bias=dt_bias,
        z=z,
        initial_state=initial_state,
        d_final_state=dstate_in,
        state_checkpoints=state_checkpoints,
        y_pre=y_pre,
        checkpoint_every_n_tokens=ctx.checkpoint_every_n_tokens,
        dt_softplus=ctx.dt_softplus,
        dt_min=ctx.dt_min,
        dt_max=ctx.dt_max,
        batch_invariant=ctx.batch_invariant,
        plan_name=ctx.plan_name,
    )
    return (
        dx,
        ddt,
        dA,
        dB,
        dC,
        None,
        dD if ctx.has_d else None,
        ddt_bias if ctx.has_dt_bias else None,
        dz if ctx.has_z else None,
        dstate0 if ctx.has_initial_state else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "cudnn::mamba2_fwd",
    mamba2_backward,
    setup_context=mamba2_setup_context,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mamba2(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    checkpoint_every_n_tokens: int = 0,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    state_indices: Optional[torch.Tensor] = None,
    overwrite_initial_state: bool = False,
    plan_name: Optional[str] = None,
):
    """Mamba-2 state space duality (SSD) layer, the chunked scan of ``mamba_chunk_scan_combined``.

    THD layout (matches the graph-API MAMBA2 node):

        x: ``[total_tokens, H, P]``; dt: ``[total_tokens, H]`` (raw, pre-softplus);
        A: ``[H]`` (negative); B, C: ``[total_tokens, G, N]`` with ``H`` a multiple of ``G``;
        cu_seqlens: ``[N_seq + 1]`` int32; D: ``[H]`` or ``[H, P]``; dt_bias: ``[H]``;
        z: ``[total_tokens, H, P]``; the states are ``[N_seq, H, P, N]``.

    The FROST kernels serve ``N = 128`` and ``P = 64``.  A dense batch of N equal-length
    sequences is expressed as ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Dtypes are kernel-native and strict (callers convert).  ``x``, ``B``, ``C`` and ``z`` share
    one io dtype (bfloat16 or float16); ``dt``, ``A``, ``dt_bias`` and ``D`` are float32,
    bfloat16 or float16 and their gradients come back in the same dtype; ``initial_state`` is
    float32 or bfloat16 and ``final_state`` follows it.

    Args:
        x: the SSM input (``v`` in the delta-rule vocabulary).
        dt: raw per-token, per-head step logits; ``dt_t = clamp(softplus(dt_t + dt_bias), dt_min, dt_max)``
            is the write strength and ``exp(A dt_t)`` the decay.
        A: per-head decay rate, negative.
        B: the write key (``k``), grouped over ``G``.
        C: the read key (``q``), grouped over ``G``.
        cu_seqlens: ``[N_seq+1]`` int32/int64 sequence boundaries over the packed tokens.
        D: optional skip connection ``y += D x`` per head (``[H]``) or per channel (``[H, P]``).
        dt_bias: optional per-head bias added to ``dt`` before the softplus.
        z: optional gate, ``y *= z sigmoid(z)``, fused in the output epilogue (the pre-gate
            output is kept for the backward).
        initial_state: optional recurrent state (otherwise zero); float32 or bfloat16, and
            ``final_state`` comes back in the same dtype.
        output_final_state: if ``True``, also return the per-sequence state after the last token.
        checkpoint_every_n_tokens: if ``> 0``, also return the per-chunk recurrent state series
            ``state_checkpoints`` (``[total_checkpoints, H, P, N]`` io dtype, V-major like
            ``final_state``, one entry per N tokens strictly before each sequence end; the FROST
            engine requires a positive multiple of the kernel chunk size, 128).  The series is a
            non-differentiable dump.
        dt_softplus: apply the softplus to ``dt + dt_bias`` (``False`` takes ``dt + dt_bias`` as is).
        dt_min, dt_max: clamp range for the transformed ``dt`` (``dt_max=None`` is unbounded); the
            gradient is zero where the clamp is active.
        batch_invariant: if ``True``, each sequence's results are bitwise independent of the batch
            composition (whole-sequence scheduling; disables split-K load balancing).
        state_indices: optional ``[N_seq]`` int32 table making ``initial_state`` a pool
            ``[N_pool, H, P, N]``: sequence ``i`` reads row ``state_indices[i]`` and, under
            ``output_final_state``, writes its final state back to that row in place.  Implies
            ``overwrite_initial_state``.  Forward only; cannot combine with
            ``checkpoint_every_n_tokens``.
        overwrite_initial_state: if ``True``, ``final_state`` is written into the ``initial_state``
            buffer, which is required and returned as the final state.  Inference only.
        plan_name: optionally pin one execution plan by name (``mamba2_frost``); a graph offering no
            such plan raises ``cudnnGraphNotSupportedError``.

    Returns:
        ``(y, final_state)`` with ``y`` shaped like ``x``, or ``(y, final_state, state_checkpoints)``
        when ``checkpoint_every_n_tokens > 0``.  ``final_state`` is empty unless
        ``output_final_state=True``; under ``overwrite_initial_state`` or ``state_indices`` it is
        the caller's buffer.
    """
    if x.dim() != 3 or B.dim() != 3 or C.dim() != 3 or dt.dim() != 2:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if state_indices is not None and initial_state is None:
        raise ValueError("mamba2: state_indices selects rows of the initial_state pool, so initial_state is required")
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, dt, A, B, C, D, dt_bias, z, initial_state))
    if overwrite_initial_state or state_indices is not None:
        if initial_state is None:
            raise ValueError("mamba2: overwrite_initial_state requires initial_state")
        y, state_checkpoints = torch.ops.cudnn.mamba2_fwd_overwrite_state(
            x,
            dt,
            A,
            B,
            C,
            cu_seqlens,
            initial_state=initial_state,
            D=D,
            dt_bias=dt_bias,
            z=z,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            dt_softplus=bool(dt_softplus),
            dt_min=float(dt_min),
            dt_max=None if dt_max is None else float(dt_max),
            batch_invariant=bool(batch_invariant),
            plan_name=plan_name,
            state_indices=state_indices,
            output_final_state=bool(output_final_state) or bool(overwrite_initial_state),
        )
        final_state = initial_state if (output_final_state or overwrite_initial_state) else initial_state.new_empty(0)
    else:
        y, final_state, state_checkpoints, _ = torch.ops.cudnn.mamba2_fwd(
            x,
            dt,
            A,
            B,
            C,
            cu_seqlens,
            D=D,
            dt_bias=dt_bias,
            z=z,
            initial_state=initial_state,
            output_final_state=bool(output_final_state),
            output_pre_gate=needs_grad and z is not None,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            dt_softplus=bool(dt_softplus),
            dt_min=float(dt_min),
            dt_max=None if dt_max is None else float(dt_max),
            batch_invariant=bool(batch_invariant),
            plan_name=plan_name,
        )
    if checkpoint_every_n_tokens > 0:
        return y, final_state, state_checkpoints
    return y, final_state


summary_cache = {}


def build_summary_graph(
    total,
    N,
    H,
    G,
    K,
    V,
    io_dtype,
    g_dtype,
    a_dtype,
    dt_bias_dtype,
    state_dtype,
    cu_dtype,
    output_transition,
    batch_invariant,
    dt_softplus,
    dt_min,
    dt_max,
):
    graph = cudnn.pygraph()
    k_t = graph.tensor([total, G, K], data_type=io_dtype, name="k")
    v_t = graph.tensor([total, H, V], data_type=io_dtype, name="v")
    g_t = graph.tensor([total, H], data_type=g_dtype, name="g")
    a_t = graph.tensor([H], data_type=a_dtype, name="a")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    state0_t = None
    if state_dtype is not None:
        state0_t = graph.tensor([N, H, V, K], data_type=state_dtype, name="initial_state")
    dt_bias_t = None
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([H], data_type=dt_bias_dtype, name="dt_bias")
    fs_t, transition_t = graph.mamba2_summary(
        k=k_t,
        v=v_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        initial_state=state0_t,
        dt_bias=dt_bias_t,
        output_transition=output_transition,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=None if dt_max == float("inf") else dt_max,
        batch_invariant=batch_invariant,
        name="mamba2_summary",
    )
    fs_t.set_data_type(state_dtype if state_dtype is not None else cudnn.data_type.FLOAT)
    if transition_t is not None:
        transition_t.set_data_type(cudnn.data_type.FLOAT)
    return graph, dict(k=k_t, v=v_t, g=g_t, a=a_t, cu=cu_t, state0=state0_t, dt_bias=dt_bias_t, fs=fs_t, transition=transition_t)


def build_summary_bwd_graph(
    total,
    N,
    H,
    G,
    K,
    V,
    io_dtype,
    g_dtype,
    a_dtype,
    dt_bias_dtype,
    dstate_dtype,
    cu_dtype,
    output_transition,
    batch_invariant,
    dt_softplus,
    dt_min,
    dt_max,
):
    graph = cudnn.pygraph()
    q_t = graph.tensor([total, G, K], data_type=io_dtype, name="q")
    k_t = graph.tensor([total, G, K], data_type=io_dtype, name="k")
    g_t = graph.tensor([total, H], data_type=g_dtype, name="g")
    a_t = graph.tensor([H], data_type=a_dtype, name="a")
    cu_t = graph.tensor([N + 1], data_type=cu_dtype, name="cu_seqlens")
    do_t = graph.tensor([total, H, V], data_type=io_dtype, name="dO")
    dfs_t = None
    if dstate_dtype is not None:
        dfs_t = graph.tensor([N, H, V, K], data_type=dstate_dtype, name="d_final_state")
    dt_bias_t = None
    if dt_bias_dtype is not None:
        dt_bias_t = graph.tensor([H], data_type=dt_bias_dtype, name="dt_bias")
    dstate0_t, transition_t = graph.mamba2_summary_bwd(
        q=q_t,
        k=k_t,
        g=g_t,
        a=a_t,
        cu_seqlens=cu_t,
        dO=do_t,
        d_final_state=dfs_t,
        dt_bias=dt_bias_t,
        output_transition=output_transition,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=None if dt_max == float("inf") else dt_max,
        batch_invariant=batch_invariant,
        name="mamba2_summary_bwd",
    )
    dstate0_t.set_data_type(dstate_dtype if dstate_dtype is not None else cudnn.data_type.FLOAT)
    if transition_t is not None:
        transition_t.set_data_type(cudnn.data_type.FLOAT)
    return graph, dict(q=q_t, k=k_t, g=g_t, a=a_t, cu=cu_t, dO=do_t, dfs=dfs_t, dt_bias=dt_bias_t, dstate0=dstate0_t, transition=transition_t)


def check_summary_args(op, dt, A, B, cu_seqlens, dt_bias, state, state_name, total, H, device):
    """The shape, dtype and device rules the two summary ops share."""
    if dt.shape != (total, H):
        raise ValueError(f"{op}: dt must be [total_tokens, H] = [{total}, {H}]; got {tuple(dt.shape)}")
    if A.shape != (H,):
        raise ValueError(f"{op}: A must be [H] = [{H}]; got {tuple(A.shape)}")
    if H % B.shape[1] != 0:
        raise ValueError(f"{op}: the head count ({H}) must be a multiple of the B / C group count ({B.shape[1]})")
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{op}: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}")
    check_dtype("dt", dt, (torch.float32, torch.bfloat16, torch.float16))
    check_dtype("A", A, (torch.float32, torch.bfloat16, torch.float16))
    if dt_bias is not None:
        check_dtype("dt_bias", dt_bias, (torch.float32, torch.bfloat16, torch.float16))
        if dt_bias.shape != (H,):
            raise ValueError(f"{op}: dt_bias must be [H] = [{H}]; got {tuple(dt_bias.shape)}")
    if state is not None:
        check_dtype(state_name, state, torch.float32)
        if state.shape[0] != N:
            raise ValueError(f"{state_name} must carry one state per sequence: got {state.shape[0]} for {N} sequences")
    for tensor_name, tensor in (("dt", dt), ("A", A), ("B / C", B), ("cu_seqlens", cu_seqlens), ("dt_bias", dt_bias), (state_name, state)):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"{op}: {tensor_name} must be on the same device ({device}); got {tensor.device}")
    return N


@torch.library.custom_op("cudnn::mamba2_summary", mutates_args=())
def mamba2_summary_op(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal Mamba-2 summary over a cached single-node MAMBA2_SUMMARY pygraph.  Returns ``(final_state,
    transition)``: the state after the last token of every sequence (float32, like
    ``initial_state``) and the log2 of every sequence's decay product, float32 ``[N, H]``, zero-size unless ``output_transition``."""
    total, H, V = x.shape
    G, K = B.shape[1], B.shape[2]
    device = x.device
    check_dtype("x", x, (torch.float16, torch.bfloat16))
    check_dtype("B", B, x.dtype)
    N = check_summary_args("mamba2_summary", dt, A, B, cu_seqlens, dt_bias, initial_state, "initial_state", total, H, device)
    dt_max_value = float("inf") if dt_max is None else float(dt_max)
    if float(dt_min) > dt_max_value:
        raise ValueError(f"mamba2_summary: dt_min ({dt_min}) must not exceed dt_max ({dt_max_value})")
    state0 = initial_state if initial_state is not None else None
    cache_key = make_bprop_cache_key(
        "mamba2_summary",
        total,
        N,
        H,
        G,
        K,
        V,
        x.dtype,
        cu_seqlens.dtype,
        dt.dtype,
        A.dtype,
        dt_bias.dtype if dt_bias is not None else None,
        state0.dtype if state0 is not None else None,
        bool(output_transition),
        bool(batch_invariant),
        bool(dt_softplus),
        float(dt_min),
        dt_max_value,
        device,
        plan_name,
    )
    if cache_key not in summary_cache:
        summary_cache[cache_key] = build_summary_graph(
            total,
            N,
            H,
            G,
            K,
            V,
            torch_dtype_to_cudnn(x.dtype),
            torch_dtype_to_cudnn(dt.dtype),
            torch_dtype_to_cudnn(A.dtype),
            torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            torch_dtype_to_cudnn(state0.dtype) if state0 is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            bool(output_transition),
            bool(batch_invariant),
            bool(dt_softplus),
            float(dt_min),
            dt_max_value,
        )
        select_plan(summary_cache[cache_key][0], plan_name)
    graph, t = summary_cache[cache_key]
    final_state = torch.empty(N, H, V, K, dtype=torch.float32, device=device)
    variant_pack = {t["k"]: B, t["v"]: x, t["g"]: dt, t["a"]: A, t["cu"]: cu_seqlens, t["fs"]: final_state}
    if state0 is not None:
        variant_pack[t["state0"]] = state0
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
    transition = torch.empty(0, dtype=torch.float32, device=device)
    if output_transition:
        transition = torch.empty(N, H, dtype=torch.float32, device=device)
        variant_pack[t["transition"]] = transition
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return final_state, transition


@mamba2_summary_op.register_fake
def mamba2_summary_fake(
    x,
    dt,
    A,
    B,
    cu_seqlens,
    initial_state=None,
    dt_bias=None,
    output_transition=False,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    total, H, V = x.shape
    K = B.shape[2]
    N = cu_seqlens.shape[0] - 1
    final_state = x.new_empty((N, H, V, K), dtype=torch.float32)
    transition = x.new_empty((N, H) if output_transition else (0,), dtype=torch.float32)
    return final_state, transition


@torch.library.custom_op("cudnn::mamba2_summary_bwd", mutates_args=())
def mamba2_summary_bwd_op(
    dy: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    d_final_state: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal Mamba-2 backward summary over a cached single-node MAMBA2_SUMMARY_BWD pygraph.  Returns
    ``(d_initial_state, transition)``: the state gradient at the start of every sequence under ``d_final_state`` (zero
    end gradient without one; float32) and the log2 of every sequence's decay product, float32
    ``[N, H]``, zero-size unless ``output_transition``."""
    total, H, V = dy.shape
    G, K = C.shape[1], C.shape[2]
    device = dy.device
    check_dtype("dy", dy, (torch.float16, torch.bfloat16))
    check_dtype("C", C, dy.dtype)
    N = check_summary_args("mamba2_summary_bwd", dt, A, C, cu_seqlens, dt_bias, d_final_state, "d_final_state", total, H, device)
    dt_max_value = float("inf") if dt_max is None else float(dt_max)
    if float(dt_min) > dt_max_value:
        raise ValueError(f"mamba2_summary_bwd: dt_min ({dt_min}) must not exceed dt_max ({dt_max_value})")
    if 0 in dy.stride():
        dy = dy.contiguous()
    cache_key = make_bprop_cache_key(
        "mamba2_summary_bwd",
        total,
        N,
        H,
        G,
        K,
        V,
        dy.dtype,
        cu_seqlens.dtype,
        dt.dtype,
        A.dtype,
        dt_bias.dtype if dt_bias is not None else None,
        d_final_state.dtype if d_final_state is not None else None,
        bool(output_transition),
        bool(batch_invariant),
        bool(dt_softplus),
        float(dt_min),
        dt_max_value,
        device,
        plan_name,
    )
    if cache_key not in summary_cache:
        summary_cache[cache_key] = build_summary_bwd_graph(
            total,
            N,
            H,
            G,
            K,
            V,
            torch_dtype_to_cudnn(dy.dtype),
            torch_dtype_to_cudnn(dt.dtype),
            torch_dtype_to_cudnn(A.dtype),
            torch_dtype_to_cudnn(dt_bias.dtype) if dt_bias is not None else None,
            torch_dtype_to_cudnn(d_final_state.dtype) if d_final_state is not None else None,
            torch_dtype_to_cudnn(cu_seqlens.dtype),
            bool(output_transition),
            bool(batch_invariant),
            bool(dt_softplus),
            float(dt_min),
            dt_max_value,
        )
        select_plan(summary_cache[cache_key][0], plan_name)
    graph, t = summary_cache[cache_key]
    d_initial_state = torch.empty(N, H, V, K, dtype=torch.float32, device=device)
    variant_pack = {t["q"]: C, t["k"]: C, t["g"]: dt, t["a"]: A, t["cu"]: cu_seqlens, t["dO"]: dy, t["dstate0"]: d_initial_state}
    if d_final_state is not None:
        variant_pack[t["dfs"]] = d_final_state
    if dt_bias is not None:
        variant_pack[t["dt_bias"]] = dt_bias
    transition = torch.empty(0, dtype=torch.float32, device=device)
    if output_transition:
        transition = torch.empty(N, H, dtype=torch.float32, device=device)
        variant_pack[t["transition"]] = transition
    workspace, handle = make_execution_resources(graph, device)
    execute_graph(graph, variant_pack, workspace, handle)
    return d_initial_state, transition


@mamba2_summary_bwd_op.register_fake
def mamba2_summary_bwd_fake(
    dy,
    dt,
    A,
    C,
    cu_seqlens,
    d_final_state=None,
    dt_bias=None,
    output_transition=False,
    dt_softplus=True,
    dt_min=0.0,
    dt_max=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    total, H, V = dy.shape
    K = C.shape[2]
    N = cu_seqlens.shape[0] - 1
    d_initial_state = dy.new_empty((N, H, V, K), dtype=torch.float32)
    transition = dy.new_empty((N, H) if output_transition else (0,), dtype=torch.float32)
    return d_initial_state, transition


def mamba2_summary(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
):
    """Mamba-2 per-span summary for context parallelism: the state chain alone over each sequence, returning
    ``(final_state, transition)`` with ``final_state`` ``[N, H, P, N_state]`` the state after the last token under
    ``initial_state`` (zero when ``None``; float32) and ``transition`` ``[N, H]`` float32 the
    log2 of the span's decay product ``gamma = exp(A sum_t dt_t)``, so that across spans ``S_final = gamma S_init +
    S_H`` (the transition is a scalar per head: no matrix).  Both outputs are non-differentiable.  THD layout and the
    dt transform match :func:`mamba2` (x ``[total_tokens, H, P]``, dt ``[total_tokens, H]`` raw, A ``[H]``, B
    ``[total_tokens, G, N_state]``, cu_seqlens ``[N+1]`` int32/int64).

    Args:
        plan_name: pin one execution plan by name (``mamba2_summary_frost``).
    """
    if x.dim() != 3 or B.dim() != 3 or dt.dim() != 2:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    final_state, transition = torch.ops.cudnn.mamba2_summary(
        x,
        dt,
        A,
        B,
        cu_seqlens,
        initial_state=initial_state,
        dt_bias=dt_bias,
        output_transition=True,
        dt_softplus=bool(dt_softplus),
        dt_min=float(dt_min),
        dt_max=None if dt_max is None else float(dt_max),
        batch_invariant=bool(batch_invariant),
        plan_name=plan_name,
    )
    return final_state, transition


def mamba2_summary_bwd(
    dy: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    d_final_state: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    dt_softplus: bool = True,
    dt_min: float = 0.0,
    dt_max: Optional[float] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
):
    """Mamba-2 per-span backward summary for context parallelism: the reverse state-gradient recurrence over each
    sequence, returning ``d_initial_state`` ``[N, H, P, N_state]`` (the incoming state gradient under ``d_final_state``;
    the pure summary term with ``None``; float32) and, with ``output_transition=True``, also
    ``transition`` ``[N, H]`` float32, the log2 of the span's decay product, so that across spans ``dS_init = gamma
    dS_final + dS_G``.  The outputs are non-differentiable.  THD layout and the dt transform match :func:`mamba2`
    (dy ``[total_tokens, H, P]`` in the io dtype, dt ``[total_tokens, H]`` raw, A ``[H]``, C ``[total_tokens, G,
    N_state]``, cu_seqlens ``[N+1]``); B and x play no part in the state-gradient recurrence.

    Args:
        plan_name: pin one execution plan by name (``mamba2_summary_frost``).
    """
    if dy.dim() != 3 or C.dim() != 3 or dt.dim() != 2:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    d_initial_state, transition = torch.ops.cudnn.mamba2_summary_bwd(
        dy,
        dt,
        A,
        C,
        cu_seqlens,
        d_final_state=d_final_state,
        dt_bias=dt_bias,
        output_transition=bool(output_transition),
        dt_softplus=bool(dt_softplus),
        dt_min=float(dt_min),
        dt_max=None if dt_max is None else float(dt_max),
        batch_invariant=bool(batch_invariant),
        plan_name=plan_name,
    )
    if output_transition:
        return d_initial_state, transition
    return d_initial_state
