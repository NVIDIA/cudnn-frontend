# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The operator API: ``cutemok_prepare``, ``cutemok_forward``, ``cutemok_backward`` and ``cutemok_quantize_weights``
over the input types ``CuteMoKConfig``, ``CuteMoKPlan`` and ``CuteMoKSplitWeight`` (the names cuDNN frontend exports
as ``cudnn.ops.*``).

The operator functions are thin wrappers over the plan / execute entries of ``functional``: ``prepare`` is
``make_plan``, ``forward`` is ``schedule_execute`` followed by ``forward_execute``, ``backward`` is
``backward_execute``.  The wrappers add two things: the decline of an input layout or dtype the execute steps do not
serve, raised as ``NotImplementedError`` naming the tensor before anything is launched, and the allocation of the
results.  ``prepare`` compiles the kernels and allocates the symmetric workspace, counters and scratch, which the
plan owns; ``forward`` and ``backward`` compile nothing and allocate their results on every call (``torch.empty``
on the plan's device, the current stream's caching allocator: a CUDA graph capture's private pool inside one).
``cutemok_quantize_weights`` turns BF16 routed weights into the MXFP8 forms the forward and backward take.
"""

import torch
import torch.distributed as dist

from cudnn._tupledict import TupleDict
from cudnn.moe_ep._cutemok import ops
from cudnn.moe_ep._cutemok.functional import (
    CuteMoKConfig,
    CuteMoKForwardContext,
    CuteMoKPlan,
    CuteMoKSchedule,
    CuteMoKSplitWeight,
    _combined_fc1,
    _empty_buffers,
    _require_backward_gate,
    _require_plan_gate,
    backward_execute,
    forward_execute,
    make_plan,
    schedule_execute,
)

#: The keys of the backward's result, in order.  The ninth is the shared output gate weight's gradient (None without
#: the gate).
_BACKWARD_RESULT_NAMES = (
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
    "d_w_output_gate",
)

__all__ = [
    "CuteMoKConfig",
    "CuteMoKPlan",
    "CuteMoKSplitWeight",
    "cutemok_backward",
    "cutemok_forward",
    "cutemok_prepare",
    "cutemok_quantize_weights",
]
# CuteMoKConfig: the runtime configuration of every call (communication SMs, macrobatch / minibatch sizes, schedule
# capacity).  CuteMoKPlan: what ``cutemok_prepare`` returns; it owns the workspace.
# CuteMoKSplitWeight: a routed weight held in per-expert allocations (``CuteMoKSplitWeight.from_experts``).


def _decline_layout(name: str, tensor, dtype: torch.dtype) -> None:
    """``NotImplementedError`` for a tensor the execute steps do not read natively: they take contiguous
    ``[num_local_tokens, ...]`` tensors of one dtype and never convert or copy them into another layout."""
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise NotImplementedError(f"{name} must be a torch.Tensor; got {type(tensor).__name__}")
    if tensor.dtype != dtype or tensor.ndim != 2 or not tensor.is_contiguous():
        raise NotImplementedError(
            f"{name}: only a contiguous 2-D {dtype} tensor is supported; got {tensor.dtype} of shape "
            f"{tuple(tensor.shape)} and strides {tuple(tensor.stride())}"
        )


def _decline_routed_weights(plan, weights) -> None:
    """The routed weights' form must be the plan's: BF16 tensors (or BF16 ``CuteMoKSplitWeight``) for a BF16 plan,
    MXFP8 tuples (or ``CuteMoKSplitWeight`` with scales) for an MXFP8 plan."""
    if not isinstance(plan, CuteMoKPlan):
        raise TypeError("plan must come from cutemok_prepare")
    for name, weight in zip(("routed_gate_weights", "routed_up_weights", "routed_down_weights"), weights):
        if isinstance(weight, tuple) or (isinstance(weight, CuteMoKSplitWeight) and weight.scale is not None):
            form = "mxfp8"
        elif isinstance(weight, CuteMoKSplitWeight) or (isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16):
            form = "bf16"
        else:
            raise NotImplementedError(
                f"{name}: a BF16 tensor, an MXFP8 (data, scale, ...) tuple or a CuteMoKSplitWeight is supported; "
                f"got {getattr(weight, 'dtype', type(weight).__name__)}"
            )
        if form != plan.routed_precision:
            raise NotImplementedError(f"{name}: {form} routed weights on a plan prepared for {plan.routed_precision} " "(cutemok_prepare's routed_precision)")


def cutemok_prepare(
    config: CuteMoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    routed_hidden_size: int | None = None,
    swiglu_clamped: bool = False,
    situ: bool = False,
    has_shared: bool = True,
    split_routed_weights: bool = False,
    shared_output_gate: bool = False,
    backward: bool = True,
    main_grad_dtype: torch.dtype | None = None,
    routed_precision: str = "bf16",
    native_columnwise: bool = False,
) -> CuteMoKPlan:
    """Prepare one layer configuration: compile every kernel the forward and backward launch and allocate the
    symmetric workspace, counters and scratch they use.  Collective over ``group``; outside CUDA graph capture.

    The arguments are ``cudnn.moe_ep._cutemok.functional.make_plan``'s, which this calls.  The token count, macrobatch /
    minibatch sizes and communication SMs are not part of the plan: they are runtime values of every call (token
    counts up to ``max_tokens_per_rank``; macrobatches up to ``config.macrobatch_size``).

    The workspace is the plan's own (not cached or shared with another plan): the symmetric buffers, dominated by
    the two BF16 [max_tokens_per_rank * topk, routed_hidden_size] buffers of the routed rows' results (combine) and
    input gradients, next to two [max_tokens_per_rank, routed_hidden_size] BF16 buffers (staged x and dY), the
    router weights and their gradients and the gathered routes.  One plan serves every layer of the same
    configuration: pass it to all of them.  Calls on a plan run in stream order and must not overlap.  Keep the
    plan until every call and graph replay that uses it has finished on every rank of the group; then drop it on
    every rank, outside graph capture, which frees the workspace.  Synchronize first (for example
    ``torch.cuda.synchronize()`` and a barrier over the group): the other ranks write into this rank's buffers.

    Returns:
        CuteMoKPlan.
    """
    return make_plan(
        config,
        group,
        device=device,
        max_tokens_per_rank=max_tokens_per_rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        topk=topk,
        routed_hidden_size=routed_hidden_size,
        swiglu_clamped=swiglu_clamped,
        situ=situ,
        has_shared=has_shared,
        split_routed_weights=split_routed_weights,
        shared_output_gate=shared_output_gate,
        backward=backward,
        main_grad_dtype=main_grad_dtype,
        routed_precision=routed_precision,
        native_columnwise=native_columnwise,
    )


def cutemok_forward(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple | CuteMoKSplitWeight,
    routed_up_weights: torch.Tensor | tuple | CuteMoKSplitWeight,
    routed_down_weights: torch.Tensor | tuple | CuteMoKSplitWeight,
    *,
    plan: CuteMoKPlan,
    config: CuteMoKConfig,
    swiglu_limit: float | None = None,
    routed_input: torch.Tensor | None = None,
    shared_output_gate_weight: torch.Tensor | None = None,
    situ_beta: tuple[float, float] | None = None,
) -> TupleDict:
    """Dispatch the tokens to their top-k experts, run the expert MLPs and combine the results.  Collective over
    the plan's group; every rank passes the same token count.

    Allocates its results (the output, the schedule and the saved activations, ``torch.empty``) and launches
    ``schedule_execute``'s kernels (route all-gather, barrier, three scheduler kernels) and then
    ``forward_execute``'s (staging copy, barrier, counter reset, forward megakernel, barrier, gated the GEMM and
    sigmoid of G, epilogue).  Every element of the results is written by the kernels (unused schedule capacity is
    never read), so nothing is zeroed.

    Inputs:
        x:                 bfloat16 [T, hidden_size], T a multiple of 256 in [512, max_tokens_per_rank]
        top_experts:       int32 [T, topk], global expert ids; -1 marks an unused route slot
        router_weights:    float32 [T, topk]
        shared_*_weights:  bfloat16 [I, H], [I, H], [H, I], or three None for a layer without a shared expert
        routed_*_weights:  bfloat16 [E_local, I, H], [E_local, I, H], [E_local, H, I]; MXFP8 (data, scale)
                           tuples (the forward forms of ``cutemok_quantize_weights``); or CuteMoKSplitWeight
                           (``CuteMoKSplitWeight.from_experts``).  One object as both gate and up: a combined FC1.
        plan:              from cutemok_prepare
        config:            CuteMoKConfig of this call
        shared_output_gate_weight: bfloat16 [1, hidden_size], given exactly when the plan was prepared with
                           ``shared_output_gate=True``: the shared output is scaled by G = sigmoid(x @ w^T)
        swiglu_limit, routed_input, situ_beta: the options of ``cudnn.moe_ep._cutemok.functional.forward_execute``

    Returns:
        TupleDict(output, schedule, forward_context): ``output`` is bfloat16 [T, routed_hidden_size] (with differing hiddens
        the pair (y_shared, y_routed)); ``schedule`` and ``forward_context`` are passed to the backward and hold
        its inputs (the schedule's int32 tensors and the saved activations).  Keep them, unmodified, until then.
    """
    for name, tensor, dtype in (
        ("x", x, torch.bfloat16),
        ("top_experts", top_experts, torch.int32),
        ("router_weights", router_weights, torch.float32),
        ("routed_input", routed_input, torch.bfloat16),
    ):
        _decline_layout(name, tensor, dtype)
    _decline_routed_weights(plan, (routed_gate_weights, routed_up_weights, routed_down_weights))
    _require_plan_gate(plan, shared_output_gate_weight)
    device = plan.workspace.device
    schedule_out = _empty_buffers(plan._schedule_shapes(int(top_experts.shape[0])), torch.int32, device)
    schedule = schedule_execute(plan, config, top_experts, out=schedule_out)
    context_shapes, out_shape = plan._forward_shapes(config, int(x.shape[0]))
    output, forward_context = forward_execute(
        plan,
        config,
        schedule,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        routed_input,
        context_out=_empty_buffers(context_shapes, torch.bfloat16, device),
        out=_empty_buffers(out_shape, torch.bfloat16, device),
        shared_output_gate_weight=shared_output_gate_weight,
        situ_beta=situ_beta,
    )
    return TupleDict(output=output, schedule=schedule, forward_context=forward_context)


def cutemok_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple | CuteMoKSplitWeight,
    routed_up_weights: torch.Tensor | tuple | CuteMoKSplitWeight,
    routed_down_weights: torch.Tensor | tuple | CuteMoKSplitWeight,
    schedule: CuteMoKSchedule,
    forward_context: CuteMoKForwardContext,
    *,
    plan: CuteMoKPlan,
    config: CuteMoKConfig,
    main_grads: tuple | None = None,
    main_grad_storage_tables: tuple | None = None,
    swiglu_limit: float | None = None,
    routed_grad_output: torch.Tensor | None = None,
    routed_input: torch.Tensor | None = None,
    zero_unrouted_router_grads: bool = True,
    shared_output_gate_weight: torch.Tensor | None = None,
    shared_output_gate_main_grad: torch.Tensor | None = None,
) -> TupleDict:
    """The gradients of ``cutemok_forward`` with respect to x, the router weights, the six weights and the shared
    output gate's weight.  Collective over the plan's group.  Allocates its results (``torch.empty``; every element
    is written by the kernels, so nothing is zeroed) and launches ``backward_execute``'s kernels (gated: the fused
    G * dY / dZ kernel; staging copy, barrier, counter reset, metadata builder or tensormap prologue, backward
    megakernel, barrier, epilogue, router-weight gradient copy and, with ``zero_unrouted_router_grads``, the
    masking of unrouted slots; gated: the gate weight's gradient GEMM and the fused merge into d_x).

    Inputs: the forward's, with the weights' backward forms (MXFP8: (e4m3, scale, e4m3_t, scale_t) for gate / up,
    (e4m3_t, scale_t) for down, the backward forms of ``cutemok_quantize_weights``), and:
        grad_output:          bfloat16 [T, hidden_size]
        schedule, forward_context: the forward's results
        main_grads:           the six buffers the weight gradients are accumulated into (FP32 or BF16, the plan's
                              ``main_grad_dtype``), ordered shared / routed per matrix (gate, up, down), None
                              where a routed-only layer has no shared weight; None (default): the gradients are
                              allocated and returned
        main_grad_storage_tables: with CuteMoKSplitWeight routed weights (which accumulate into ``main_grads``),
                              the routed gate / up / down main_grads' per-expert base addresses: three int64
                              [E_local] CUDA tensors, in expert order (a combined FC1 passes one table as both
                              gate and up); ``main_grads`` holds the first expert's buffers
        swiglu_limit, routed_grad_output, routed_input, zero_unrouted_router_grads:
                              the options of ``cudnn.moe_ep._cutemok.functional.backward_execute``
        shared_output_gate_weight: the forward's, unchanged (a gated forward context requires it)
        shared_output_gate_main_grad: optional BF16 or FP32 [1, hidden_size] buffer the gate weight's gradient
                              is added into

    Returns:
        TupleDict of ``backward_execute``'s nine entries, in its order: d_x (bfloat16 [T, hidden_size]; with
        differing hiddens the pair (d_x_shared, d_routed_input)), d_router_weights (float32 [T, topk]),
        d_w_routed_gate, d_w_routed_up, d_w_routed_down, d_w_shared_gate, d_w_shared_up and d_w_shared_down
        (bfloat16, shaped like the weights; a combined FC1's gate and up are the halves of one allocation; with
        ``main_grads`` the entries alias the buffers they were added into), and d_w_output_gate: the gate weight's
        float32 [1, hidden_size] gradient (the ``shared_output_gate_main_grad`` it was added into, if given; None
        without the gate).
    """
    for name, tensor, dtype in (
        ("grad_output", grad_output, torch.bfloat16),
        ("x", x, torch.bfloat16),
        ("router_weights", router_weights, torch.float32),
        ("routed_grad_output", routed_grad_output, torch.bfloat16),
        ("routed_input", routed_input, torch.bfloat16),
    ):
        _decline_layout(name, tensor, dtype)
    _decline_routed_weights(plan, (routed_gate_weights, routed_up_weights, routed_down_weights))
    _require_backward_gate(plan, forward_context, shared_output_gate_weight)
    device = plan.workspace.device
    out_shape, d_router_shape, d_weight_shapes = plan._backward_shapes(int(x.shape[0]))
    d_weights_out = None
    if main_grads is None:
        shapes = list(d_weight_shapes)
        combined = bool(_combined_fc1(routed_gate_weights, routed_up_weights)[2])
        if combined:  # a combined FC1: one [E_local, 2I, H] gradient under both gate and up
            experts, rows, columns = shapes[1]
            shapes[1], shapes[3] = (experts, 2 * rows, columns), None
        d_weights_out = list(_empty_buffers(tuple(shapes), torch.bfloat16, device))
        if combined:
            d_weights_out[3] = d_weights_out[1]
        d_weights_out = tuple(d_weights_out)
    d_w_output_gate_out = None
    if shared_output_gate_weight is not None and shared_output_gate_main_grad is None:
        d_w_output_gate_out = torch.empty(1, plan.workspace.hidden_size, dtype=torch.float32, device=device)
    gradients = backward_execute(
        plan,
        config,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        main_grads,
        main_grad_storage_tables,
        routed_grad_output,
        routed_input,
        out=_empty_buffers(out_shape, torch.bfloat16, device),
        d_router_weights_out=_empty_buffers(d_router_shape, torch.float32, device),
        d_weights_out=d_weights_out,
        zero_unrouted_router_grads=zero_unrouted_router_grads,
        shared_output_gate_weight=shared_output_gate_weight,
        shared_output_gate_main_grad=shared_output_gate_main_grad,
        d_w_output_gate_out=d_w_output_gate_out,
    )
    return TupleDict(zip(_BACKWARD_RESULT_NAMES, gradients, strict=True))


def _check_bf16_weight(name: str, weight) -> None:
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(weight).__name__}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16; got {weight.dtype}")
    if weight.ndim != 3:
        raise ValueError(f"{name} must be 3-D [E_local, rows, columns]; got shape {tuple(weight.shape)}")


def cutemok_quantize_weights(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> TupleDict:
    """Quantize the BF16 routed weights to the MXFP8 forms ``cutemok_forward`` and ``cutemok_backward`` take, for a
    plan prepared with ``routed_precision="mxfp8"``.  Call after every weight update; the shared weights stay BF16.

    MXFP8: an E4M3 payload with one E8M0 scale per 32 consecutive elements of a row, the scales in the tcgen05
    layout the kernels read.  Every matrix is quantized in both orientations, ``[N, K]`` (rowwise) and its
    transpose ``[K, N]`` (columnwise, read by the backward's data gradient).  Allocates the results on the weights'
    device; launches on the current stream.

    Inputs:
        gate, up: bfloat16 [E_local, I, H]; or one [E_local, 2I, H] tensor passed as both (a combined FC1,
                  quantized once)
        down:     bfloat16 [E_local, H, I]
        All three contiguous, 16-byte aligned, on one CUDA device; I and H multiples of 128.

    Returns:
        TupleDict(forward_weights, backward_weights), each a (gate, up, down) triple of tuples, splatted as the routed
        weights of the two calls:
            forward_weights:  (e4m3, scale) per matrix
            backward_weights: (e4m3, scale, e4m3_t, scale_t) for gate and up, (e4m3_t, scale_t) for down
        with e4m3 float8_e4m3fn [E_local, N, K], scale uint8 [E_local * N // 128, K // 128, 32, 16], e4m3_t
        float8_e4m3fn [E_local, K, N] and scale_t uint8 [E_local * K // 128, N // 128, 32, 16].  The two triples
        share their tensors; a combined FC1 is one tuple under both gate and up.
    """
    combined = up is gate
    for name, weight in (("gate", gate), ("up", up), ("down", down)):
        _check_bf16_weight(name, weight)
    experts, rows, hidden = gate.shape
    intermediate = rows // 2 if combined else rows
    if combined and rows % 2:
        raise ValueError(f"gate is up, a combined FC1 [E_local, 2I, H]: its {rows} rows are odd")
    if up.shape != gate.shape:
        raise ValueError(f"up must have gate's shape {tuple(gate.shape)}; got {tuple(up.shape)}")
    if down.shape != (experts, hidden, intermediate):
        raise ValueError(f"down must be [E_local, H, I] = {(experts, hidden, intermediate)} for gate " f"{tuple(gate.shape)}; got {tuple(down.shape)}")
    if experts < 1 or hidden % 128 or intermediate % 128 or not hidden or not intermediate:
        raise ValueError(f"MXFP8 routed weights need E_local >= 1 and I, H positive multiples of 128; got " f"E_local {experts}, I {intermediate}, H {hidden}")
    for name, weight in (("gate", gate), ("up", up), ("down", down)):
        if not weight.is_cuda or weight.device != gate.device:
            raise ValueError(f"{name} must be on gate's CUDA device; got {weight.device}")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous; got strides {tuple(weight.stride())}")
        if weight.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    with torch.cuda.device(gate.device):
        gate_q = tuple(ops.mxfp8_quantize(gate, True, True))
        up_q = gate_q if combined else tuple(ops.mxfp8_quantize(up, True, True))
        down_q = tuple(ops.mxfp8_quantize(down, True, True))
    gate_rowwise = gate_q[:2]
    up_rowwise = gate_rowwise if combined else up_q[:2]
    return TupleDict(forward_weights=(gate_rowwise, up_rowwise, down_q[:2]), backward_weights=(gate_q, up_q, down_q[2:]))
