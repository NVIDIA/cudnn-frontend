# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch adapter over the MAMBA2 / MAMBA2_BWD graph family.

Plans are cached by tensor declarations on each host thread. Outputs and
workspace belong to each call, so concurrent streams never share scratch.
The graph engine itself allocates nothing and compiles during plan building.
"""

from collections import OrderedDict
import threading
from typing import Optional, Tuple

import torch
import cudnn

from .common import select_plan, torch_dtype_to_cudnn

_local = threading.local()
_FWD_OUTPUTS = ("O", "final_state", "ungated_out", "state_checkpoints")
_GRADS = {"dX": "x", "dDt": "dt", "dA": "A", "dB": "B", "dC": "C", "dD": "D", "d_dt_bias": "dt_bias", "dZ": "z", "d_initial_state": "initial_state"}


def _signature(inputs):
    return tuple((name, tuple(t.shape), tuple(t.stride()), t.dtype) for name, t in inputs.items() if t is not None)


def _outputs(inputs, bwd, output_final_state=False, reuse_forward_states=False, intermediate_dtype="float32"):
    x = inputs["x"]
    if bwd:
        return {dst: torch.empty_like(inputs[src]) if inputs.get(src) is not None else x.new_empty(0) for dst, src in _GRADS.items()}
    batch, length, heads, dim = x.shape
    state = (batch, heads, dim, inputs["B"].shape[-1])
    return dict(
        O=torch.empty_like(x),
        final_state=x.new_empty(state if output_final_state else (0,), dtype=torch.float32),
        ungated_out=torch.empty_like(x) if inputs.get("z") is not None else x.new_empty(0),
        state_checkpoints=x.new_empty(
            (batch, heads, (length + 31) // 32, dim, state[-1]) if reuse_forward_states else (0,),
            dtype=torch.bfloat16 if intermediate_dtype == "bfloat16" else torch.float32,
        ),
    )


def _get_graph(inputs, bwd, chunk_size, intermediate_dtype, output_final_state=False, reuse_forward_states=False, plan_name=None):
    """Build from declarations; no tensor values or addresses enter the key."""
    device = inputs["x"].device
    key = (bwd, _signature(inputs), device.index, chunk_size, intermediate_dtype, output_final_state, reuse_forward_states, plan_name)
    cache = getattr(_local, "plans", None)
    if cache is None:
        cache = _local.plans = OrderedDict()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    handles = getattr(_local, "handles", None)
    if handles is None:
        handles = _local.handles = {}
    if device.index not in handles:
        handles[device.index] = cudnn.create_handle()
    handle = handles[device.index]
    graph = cudnn.pygraph(handle=handle)
    ports = {
        name: graph.tensor(list(t.shape), stride=list(t.stride()), data_type=torch_dtype_to_cudnn(t.dtype), name=name)
        for name, t in inputs.items()
        if t is not None
    }
    params = dict(chunk_size=chunk_size, dt_softplus=True, intermediate_dtype=intermediate_dtype)
    if bwd:
        result = graph.mamba2_bwd(**ports, **params)
        names = tuple(_GRADS)
    else:
        result = graph.mamba2(**ports, **params, output_final_state=output_final_state, save_state_checkpoints=reuse_forward_states)
        names = _FWD_OUTPUTS
    for name, tensor in zip(names, result):
        if tensor is not None:
            if name == "state_checkpoints":
                tensor.set_data_type(cudnn.data_type.BFLOAT16 if intermediate_dtype == "bfloat16" else cudnn.data_type.FLOAT)
            ports[name] = tensor
    # A thread-local handle avoids changing another host thread's launch stream.
    select_plan(graph, plan_name)
    graph.build()
    cache[key] = (graph, ports, handle)
    if len(cache) > 64:
        cache.popitem(last=False)
    return cache[key]


def _run(inputs, bwd, chunk_size, intermediate_dtype, output_final_state=False, reuse_forward_states=False, plan_name=None):
    x = inputs["x"]
    if x.device.type != "cuda":
        raise ValueError("mamba2 requires CUDA tensors")
    for name, tensor in inputs.items():
        if tensor is not None and tensor.device != x.device:
            raise ValueError(f"mamba2: {name} must be on {x.device}")
        if tensor is not None:
            bf16 = name in ("x", "dt", "B", "C", "z", "dO", "ungated_out") or (name == "state_checkpoints" and intermediate_dtype == "bfloat16")
            want = torch.bfloat16 if bf16 else torch.float32
            if tensor.dtype != want:
                raise ValueError(f"mamba2: {name} must be {want}, got {tensor.dtype}")
            if not tensor.is_contiguous():
                raise ValueError(f"mamba2: {name} must be contiguous, got strides {tensor.stride()}")
    with torch.cuda.device(x.device):
        graph, ports, handle = _get_graph(inputs, bwd, chunk_size, intermediate_dtype, output_final_state, reuse_forward_states, plan_name)
        outputs = _outputs(inputs, bwd, output_final_state, reuse_forward_states, intermediate_dtype)
        values = {**inputs, **outputs}
        # Caller allocation: released after launch, stream ordered by PyTorch's
        # allocator, and private to a CUDA graph's capture pool during capture.
        workspace = torch.empty(graph.get_workspace_size(), device=x.device, dtype=torch.uint8)
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(x.device).cuda_stream)
        graph.execute({port: values[name] for name, port in ports.items()}, workspace=workspace, handle=handle)
        return tuple(outputs.values())


@torch.library.custom_op("cudnn::mamba2_fwd", mutates_args=())
def mamba2_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    chunk_size: int,
    intermediate_dtype: str,
    reuse_forward_states: bool,
    plan_name: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run(
        dict(x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=dt_bias, z=z, initial_state=initial_state),
        False,
        chunk_size,
        intermediate_dtype,
        output_final_state,
        reuse_forward_states,
        plan_name,
    )


@mamba2_fwd.register_fake
def _fake_fwd(x, dt, A, B, C, D, dt_bias, z, initial_state, output_final_state, chunk_size, intermediate_dtype, reuse_forward_states, plan_name):
    return tuple(_outputs(dict(x=x, B=B, z=z), False, output_final_state, reuse_forward_states, intermediate_dtype).values())


@torch.library.custom_op("cudnn::mamba2_bwd", mutates_args=())
def mamba2_bwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dO: torch.Tensor,
    D: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    d_final_state: Optional[torch.Tensor],
    ungated_out: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    chunk_size: int,
    intermediate_dtype: str,
    plan_name: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _run(
        dict(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            dO=dO,
            D=D,
            dt_bias=dt_bias,
            z=z,
            initial_state=initial_state,
            d_final_state=d_final_state,
            ungated_out=ungated_out,
            state_checkpoints=state_checkpoints,
        ),
        True,
        chunk_size,
        intermediate_dtype,
        plan_name=plan_name,
    )


@mamba2_bwd.register_fake
def _fake_bwd(x, dt, A, B, C, dO, D, dt_bias, z, initial_state, d_final_state, ungated_out, state_checkpoints, chunk_size, intermediate_dtype, plan_name):
    return tuple(_outputs(dict(x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=dt_bias, z=z, initial_state=initial_state), True).values())


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:9], output[2], output[3])
    ctx.chunk_size, ctx.intermediate_dtype, ctx.reuse_forward_states, ctx.plan_name = inputs[10:]
    ctx.mark_non_differentiable(output[2], output[3])
    ctx.set_materialize_grads(False)


def _backward(ctx, dO, d_final_state, _ungated_grad, _checkpoint_grad):
    x, dt, A, B, C, D, dt_bias, z, initial_state, ungated, checkpoints = ctx.saved_tensors
    # Autograd can produce broadcast cotangents (e.g. sum()), unlike the dense
    # graph contract. Materialize only here, outside graph.execute().
    dO = torch.zeros_like(x) if dO is None else dO.contiguous()
    d_final_state = d_final_state.contiguous() if d_final_state is not None else None
    grads = mamba2_bwd(
        x,
        dt,
        A,
        B,
        C,
        dO,
        D,
        dt_bias,
        z,
        initial_state,
        d_final_state,
        ungated if z is not None else None,
        checkpoints if ctx.reuse_forward_states else None,
        ctx.chunk_size,
        ctx.intermediate_dtype,
        ctx.plan_name,
    )
    present = (x, dt, A, B, C, D, dt_bias, z, initial_state)
    return tuple(g if t is not None else None for g, t in zip(grads, present)) + (None,) * 5


torch.library.register_autograd("cudnn::mamba2_fwd", _backward, setup_context=_setup_context)


def mamba2(
    x,
    dt,
    A,
    B,
    C,
    D=None,
    dt_bias=None,
    z=None,
    initial_state=None,
    return_final_state=False,
    chunk_size=32,
    intermediate_dtype="float32",
    reuse_forward_states=False,
    plan_name=None,
):
    """Mamba-2 SSD forward and first-order backward on SM100.

    ``x,z``: BF16 [batch,length,heads,64]; ``dt``: BF16 [batch,length,heads];
    ``B,C``: BF16 [batch,length,groups,128]; ``A,D,dt_bias``: FP32 [heads].
    State is FP32 [batch,heads,64,128], with value then state axes. All inputs
    must be contiguous. Uses softplus(dt + dt_bias), exp(A * dt), D*x skip,
    and optional SiLU(z) gate. Pass A directly, not log(-A).

    ``chunk_size`` must be 32. ``intermediate_dtype`` controls checkpoints and
    partial gradients: float32 (default) or bfloat16 (less memory, different
    rounding, only without the optional SiLU gate). ``reuse_forward_states`` saves chunk-entry states for backward
    instead of recomputing them. Outputs are O or (O, FP32 final_state).
    Warm forward and backward before CUDA graph capture. Only first-order
    gradients are supported. Autograd materializes noncontiguous cotangents;
    direct graph calls require contiguous buffers and never repack them.
    """
    if chunk_size != 32 or isinstance(chunk_size, bool):
        raise ValueError("mamba2 currently requires chunk_size=32")
    if intermediate_dtype not in ("float32", "bfloat16"):
        raise ValueError("mamba2: intermediate_dtype must be float32 or bfloat16")
    if z is not None and intermediate_dtype != "float32":
        raise ValueError("mamba2: SiLU gate requires intermediate_dtype=float32")
    out, final, _, _ = mamba2_fwd(
        x, dt, A, B, C, D, dt_bias, z, initial_state, return_final_state, chunk_size, intermediate_dtype, reuse_forward_states, plan_name
    )
    return (out, final) if return_final_state else out
