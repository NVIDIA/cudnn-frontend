# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-routed BF16 SwiGLU Mixture-of-Experts operation.

The caller owns routing.  Tokens are already grouped by expert and
``first_token_offset[e]`` is the first row belonging to expert ``e``; the last
expert ends at ``routed_x.shape[1]``.  One fused FROST kernel computes the two
FC1 grouped GEMMs and SwiGLU, then a grouped GEMM computes the down projection.
The custom autograd function supplies gradients for the tokens and all three
expert weights.
"""

from __future__ import annotations

import torch

import cudnn

_BF16 = cudnn.data_type.BFLOAT16
_FP32 = cudnn.data_type.FLOAT

_FC1_CACHE = {}
_MM_CACHE = {}
_DSWIGLU_CACHE = {}
_WGRAD_CACHE = {}
_FROST_WORKSPACES = {}
_CUDNN_HANDLES = {}
_CUDNN_WORKSPACES = {}


def _device_key(tensor: torch.Tensor) -> tuple:
    device = tensor.device
    return device.index, torch.cuda.get_device_capability(device)


def _frost_plan(graph):
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan

    return build_gemm_plan(graph)


def _output_map(binding) -> dict[str, object]:
    return {tensor.get_name().split("::")[0]: tensor for tensor in binding.outputs}


def _frost_workspace(plan, tensor):
    """Per-plan/per-stream torch-owned MoE descriptor workspace."""
    stream = torch.cuda.current_stream(tensor.device).cuda_stream
    key = (id(plan), tensor.device.index, stream)
    workspace = _FROST_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=tensor.device)
        _FROST_WORKSPACES[key] = workspace
    return workspace, stream


def _frost_fc1(routed_x, Wg, Wu, offsets):
    """Fused ``gate/up grouped GEMMs + SwiGLU``; also materialize BF16 taps."""
    _, S, H = routed_x.shape
    E, interm, _ = Wg.shape
    key = ("fc1", S, H, interm, E, routed_x.dtype, *_device_key(routed_x))
    plan = _FC1_CACHE.get(key)
    if plan is None:
        graph = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=_FP32, compute_data_type=_FP32)
        X = graph.tensor(name="x", dim=[1, S, H], stride=[S * H, H, 1], data_type=_BF16)
        WG = graph.tensor(name="Wg", dim=[E, H, interm], stride=[H * interm, 1, H], data_type=_BF16)
        WU = graph.tensor(name="Wu", dim=[E, H, interm], stride=[H * interm, 1, H], data_type=_BF16)
        FTO = graph.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        gate_raw = graph.moe_grouped_matmul(X, WG, FTO, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=_FP32, name="gate")
        up_raw = graph.moe_grouped_matmul(X, WU, FTO, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=_FP32, name="up")
        # Match BF16 nn.Linear semantics before evaluating SiLU in FP32.
        gate_raw.set_data_type(_BF16)
        up_raw.set_data_type(_BF16)
        gate = graph.identity(input=gate_raw, name="gate_tap")
        up = graph.identity(input=up_raw, name="up_tap")
        gate.set_output(True).set_data_type(_BF16)
        up.set_output(True).set_data_type(_BF16)
        h = graph.mul(a=graph.swish(input=gate, name="silu"), b=up, name="h")
        h.set_output(True).set_data_type(_BF16)
        plan = _frost_plan(graph)
        _FC1_CACHE[key] = plan

    binding = plan.binding
    outputs = _output_map(binding)
    gate = torch.empty(1, S, interm, dtype=routed_x.dtype, device=routed_x.device)
    up = torch.empty_like(gate)
    h = torch.empty_like(gate)
    buffers = {"gate_tap": gate, "up_tap": up, "h": h}
    workspace, stream = _frost_workspace(plan, routed_x)
    plan(
        {
            binding.a_operands[0]: routed_x,
            binding.b_operands[0]: Wg,
            binding.b_operands[1]: Wu,
            binding.first_token_offset: offsets,
            **{tensor: buffers[name] for name, tensor in outputs.items()},
        },
        workspace=workspace,
        stream=stream,
    )
    return h, gate, up


def _frost_mm(token, weight, offsets):
    """One grouped GEMM; runtime weight follows FROST's physical ``[E,N,K]`` contract."""
    _, S, K = token.shape
    E, N, weight_k = weight.shape
    if weight_k != K:
        raise ValueError(f"internal grouped GEMM K mismatch: token K={K}, weight K={weight_k}")
    key = (
        "mm",
        S,
        K,
        N,
        E,
        tuple(weight.stride()),
        token.dtype,
        *_device_key(token),
    )
    plan = _MM_CACHE.get(key)
    if plan is None:
        graph = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=_FP32, compute_data_type=_FP32)
        A = graph.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=_BF16)
        logical_weight = weight.transpose(1, 2)
        B = graph.tensor(name="weight", dim=[E, K, N], stride=list(logical_weight.stride()), data_type=_BF16)
        FTO = graph.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        Y = graph.moe_grouped_matmul(A, B, FTO, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=_FP32, name="out")
        Y.set_output(True).set_data_type(_BF16)
        plan = _frost_plan(graph)
        _MM_CACHE[key] = plan

    binding = plan.binding
    output = torch.empty(1, S, N, dtype=token.dtype, device=token.device)
    workspace, stream = _frost_workspace(plan, token)
    plan(
        {
            binding.a_operands[0]: token,
            binding.b_operands[0]: weight,
            binding.first_token_offset: offsets,
            binding.outputs[0]: output,
        },
        workspace=workspace,
        stream=stream,
    )
    return output


def _frost_dswiglu(dout, Wd, gate, up, offsets):
    """Fuse the grouped down dgrad with both SwiGLU derivatives."""
    _, S, H = dout.shape
    E, weight_h, interm = Wd.shape
    if weight_h != H:
        raise ValueError(f"internal grouped dSwiGLU H mismatch: dout H={H}, Wd H={weight_h}")
    key = ("dswiglu", S, H, interm, E, dout.dtype, *_device_key(dout))
    plan = _DSWIGLU_CACHE.get(key)
    if plan is None:
        graph = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=_FP32, compute_data_type=_FP32)
        DY = graph.tensor(name="dout", dim=[1, S, H], stride=[S * H, H, 1], data_type=_BF16)
        WD = graph.tensor(name="Wd", dim=[E, H, interm], stride=[H * interm, interm, 1], data_type=_BF16)
        FTO = graph.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        G = graph.tensor(name="gate", dim=[1, S, interm], stride=[S * interm, interm, 1], data_type=_BF16)
        U = graph.tensor(name="up", dim=[1, S, interm], stride=[S * interm, interm, 1], data_type=_BF16)
        dh = graph.moe_grouped_matmul(DY, WD, FTO, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=_FP32, name="dh")
        dup = graph.mul(a=dh, b=graph.swish(input=G), name="dup")
        dgate = graph.mul(a=graph.swish_backward(loss=dh, input=G), b=U, name="dgate")
        dup.set_output(True).set_data_type(_BF16)
        dgate.set_output(True).set_data_type(_BF16)
        plan = _frost_plan(graph)
        _DSWIGLU_CACHE[key] = plan

    binding = plan.binding
    outputs = _output_map(binding)
    aux = {tensor.get_name(): tensor for tensor in binding.aux}
    dgate = torch.empty(1, S, interm, dtype=dout.dtype, device=dout.device)
    dup = torch.empty_like(dgate)
    buffers = {"dgate": dgate, "dup": dup}
    workspace, stream = _frost_workspace(plan, dout)
    plan(
        {
            binding.a_operands[0]: dout,
            # Physical [E,N=I,K=H], as a zero-copy N-major view of [E,H,I].
            binding.b_operands[0]: Wd.transpose(1, 2),
            binding.first_token_offset: offsets,
            aux["gate"]: gate,
            aux["up"]: up,
            **{tensor: buffers[name] for name, tensor in outputs.items()},
        },
        workspace=workspace,
        stream=stream,
    )
    return dgate, dup


def _cudnn_handle(tensor):
    device = tensor.device
    handle = _CUDNN_HANDLES.get(device.index)
    if handle is None:
        handle = cudnn.create_handle()
        _CUDNN_HANDLES[device.index] = handle
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


def _moe_wgrad(doutput, token, offsets):
    """Native cuDNN grouped wgrad with arbitrary expert token counts."""
    _, S, N = doutput.shape
    _, token_s, K = token.shape
    E = offsets.numel()
    if token_s != S:
        raise ValueError(f"internal grouped wgrad token mismatch: {S} and {token_s}")
    key = (S, N, K, E, doutput.dtype, *_device_key(doutput))
    cached = _WGRAD_CACHE.get(key)
    handle = _cudnn_handle(doutput)
    if cached is None:
        graph = cudnn.pygraph(intermediate_data_type=_FP32, compute_data_type=_FP32, handle=handle)
        DY = graph.tensor(name="doutput", dim=[1, S, N], stride=[S * N, N, 1], data_type=_BF16)
        X = graph.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=_BF16)
        FTO = graph.tensor(name="first_token_offset", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
        DW = graph.moe_grouped_matmul_bwd(DY, X, FTO, compute_data_type=_FP32, name="wgrad")
        # Logical [E,K,N], column-major inner dimensions.  Its physical storage
        # is a contiguous [E,N,K] tensor, exactly the public weight layout.
        DW.set_output(True).set_data_type(_BF16)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans()
        cached = (graph, DY, X, FTO, DW, graph.get_workspace_size())
        _WGRAD_CACHE[key] = cached

    graph, DY, X, FTO, DW, workspace_size = cached
    output = torch.empty(E, N, K, dtype=doutput.dtype, device=doutput.device)
    stream = torch.cuda.current_stream(doutput.device).cuda_stream
    workspace_key = (id(graph), doutput.device.index, stream)
    workspace = _CUDNN_WORKSPACES.get(workspace_key)
    if workspace is None:
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=doutput.device)
        _CUDNN_WORKSPACES[workspace_key] = workspace
    graph.execute({DY: doutput, X: token, FTO: offsets, DW: output}, workspace, handle=handle)
    return output


class _SwiGLUMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed_x, Wg, Wu, Wd, first_token_offset):
        with torch.cuda.device(routed_x.device):
            h, gate, up = _frost_fc1(routed_x, Wg, Wu, first_token_offset)
            # Wd already has FROST's physical [E,N=H,K=I] runtime layout.
            output = _frost_mm(h, Wd, first_token_offset)
        ctx.save_for_backward(routed_x, Wg, Wu, Wd, first_token_offset, h, gate, up)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        routed_x, Wg, Wu, Wd, offsets, h, gate, up = ctx.saved_tensors
        need_x, need_wg, need_wu, need_wd, _ = ctx.needs_input_grad
        with torch.cuda.device(dout.device):
            dgate = dup = None
            if need_x or need_wg or need_wu:
                dgate, dup = _frost_dswiglu(dout.contiguous(), Wd, gate, up, offsets)
            dx = None
            if need_x:
                dx = _frost_mm(dgate, Wg.transpose(1, 2), offsets) + _frost_mm(dup, Wu.transpose(1, 2), offsets)
            dWg = _moe_wgrad(dgate, routed_x, offsets) if need_wg else None
            dWu = _moe_wgrad(dup, routed_x, offsets) if need_wu else None
            dWd = _moe_wgrad(dout.contiguous(), h, offsets) if need_wd else None
        return dx, dWg, dWu, dWd, None


def swiglu_moe(routed_x, Wg, Wu, Wd, first_token_offset):
    """Pre-routed BF16 MoE layer.

    For expert ``e`` and rows ``b:e`` selected by ``first_token_offset``:

    ``out = (silu(x @ Wg[e].T) * (x @ Wu[e].T)) @ Wd[e].T``.

    Args:
        routed_x: Expert-grouped tokens ``[1,S,H]`` in row-major BF16.
        Wg, Wu: Contiguous expert gate/up weights ``[E,I,H]`` in BF16.
        Wd: Contiguous expert down weights ``[E,H,I]`` in BF16.
        first_token_offset: Contiguous INT32 starts ``[E]`` (``[E,1,1]`` is
            also accepted). The first value must be 0; all values must be in
            ``[0, S]`` and monotonically nondecreasing. The last expert ends at
            ``S``, so the final value is its start and need not equal ``S``.
            Tokens remain in routed order; routing and combine probabilities
            are intentionally outside this operation. These value invariants
            are a device-data contract and are not host-validated on the hot
            path.

    Returns:
        BF16 tensor ``[1,S,H]``, differentiable with respect to the tokens and
        all three expert weights.
    """
    prefix = "cudnn.gemm.swiglu_moe"
    tensors = (("routed_x", routed_x), ("Wg", Wg), ("Wu", Wu), ("Wd", Wd))
    for name, tensor in tensors:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{prefix}: {name} must be bfloat16, got {tensor.dtype}")
        if tensor.device.type != "cuda":
            raise ValueError(f"{prefix}: {name} must be a CUDA tensor, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{prefix}: {name} must be contiguous, got stride {tuple(tensor.stride())}")
    if routed_x.dim() != 3 or routed_x.shape[0] != 1 or Wg.dim() != 3 or Wu.dim() != 3 or Wd.dim() != 3:
        raise ValueError(
            f"{prefix}: expected routed_x[1,S,H], Wg/Wu[E,I,H], Wd[E,H,I]; "
            f"got {tuple(routed_x.shape)}, {tuple(Wg.shape)}, {tuple(Wu.shape)}, {tuple(Wd.shape)}"
        )
    if any(tensor.device != routed_x.device for _, tensor in tensors[1:]) or first_token_offset.device != routed_x.device:
        raise ValueError(f"{prefix}: inputs and first_token_offset must be on the same CUDA device")
    E, interm, H = Wg.shape
    if Wu.shape != Wg.shape or Wd.shape != (E, H, interm) or routed_x.shape[2] != H:
        raise ValueError(
            f"{prefix}: expected x[1,S,{H}], Wg/Wu[{E},{interm},{H}], Wd[{E},{H},{interm}]; "
            f"got x{tuple(routed_x.shape)}, Wg{tuple(Wg.shape)}, Wu{tuple(Wu.shape)}, Wd{tuple(Wd.shape)}"
        )
    if first_token_offset.dtype != torch.int32 or first_token_offset.numel() != E or not first_token_offset.is_contiguous():
        raise ValueError(f"{prefix}: first_token_offset must be contiguous int32 with E={E} elements")
    if H % 8 or interm % 8 or routed_x.shape[1] == 0:
        raise ValueError(f"{prefix}: S must be nonzero and H/I multiples of 8; " f"got S={routed_x.shape[1]}, H={H}, I={interm}")
    capability = torch.cuda.get_device_capability(routed_x.device)
    if not ((10, 0) <= capability < (12, 0)):
        raise RuntimeError(f"{prefix}: FROST MoE requires SM100-SM119, got SM{capability[0]}{capability[1]}")
    # The graph declares cuDNN's [E,1,1] descriptor, while FROST's runtime
    # contract consumes the same storage as a rank-1 offsets vector.
    offsets = first_token_offset.reshape(E)
    return _SwiGLUMoE.apply(routed_x, Wg, Wu, Wd, offsets)


__all__ = ["swiglu_moe"]
