# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private implementation of the dense bf16 GELU MLP cuDNN op.

The public contract matches two ordinary ``torch.nn.Linear`` layers separated
by ``GELU(approximate="tanh")``::

    h = gelu(x @ w1.T + b1, approximate="tanh")
    y = h @ w2.T + b2

The input must have rank two or greater. ``w1`` and ``w2`` use the natural ``nn.Linear`` layouts
``[intermediate, in_features]`` and ``[out_features, intermediate]``.  The
first graph fuses matmul, per-column bias and tanh-GELU; the second graph fuses
matmul and bias.  The post-bias first-layer value is explicitly bf16-rounded
before GELU, preserving the observable boundary of eager bf16 ``Linear``.

Backward uses cuDNN GEMMs for the input and weight gradients.  Its first stage
is one ``dout @ w2`` + tanh-GELU-backward graph, avoiding a materialized ``dh``;
bias gradients are reductions in PyTorch for now.

The implementation is intentionally narrow: dense contiguous bf16 CUDA tensors
on SM100.  Unsupported layouts, devices, architectures, or dtypes raise instead
of being silently copied or reinterpreted.
"""

from __future__ import annotations

import torch

import cudnn

_BF16 = cudnn.data_type.BFLOAT16
_FP32 = cudnn.data_type.FLOAT
_AUTOTUNE_ITERS = 20

# A cuDNN handle, execution plans, and scratch workspaces are private to one
# (device, stream).  A stream serializes reuse of its workspace, while distinct
# streams never race through one handle or allocation.
_HANDLES = {}
_LINEAR_CACHE = {}
_MM_CACHE = {}
_DGELU_CACHE = {}

_GRAD_X = 1 << 0
_GRAD_W1 = 1 << 1
_GRAD_B1 = 1 << 2
_GRAD_W2 = 1 << 3
_GRAD_B2 = 1 << 4
_GRAD_FC1 = _GRAD_X | _GRAD_W1 | _GRAD_B1


def _handle(device: torch.device):
    stream = torch.cuda.current_stream(device).cuda_stream
    key = (device.index, stream)
    handle = _HANDLES.get(key)
    if handle is None:
        with torch.cuda.device(device):
            handle = cudnn.create_handle()
            cudnn.set_stream(handle=handle, stream=stream)
        _HANDLES[key] = handle
    return handle, stream


def _autotune(graph, handle, variant_pack):
    """Build and time every viable graph plan, returning plan and workspace."""
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.ALL)
    count = graph.get_execution_plan_count()
    if count == 0:
        raise RuntimeError("cudnn.gemm.gelu_mlp: no execution plan was generated for this graph")

    device = next(iter(variant_pack.values())).device
    elapsed = [float("inf")] * count
    errors = {}
    with torch.cuda.device(device):
        workspace = torch.empty(
            max(graph.get_workspace_size_plan_at_index(i) for i in range(count)),
            device=device,
            dtype=torch.uint8,
        )
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for index in range(count):
            try:
                graph.execute_plan_at_index(variant_pack, workspace, index=index, handle=handle)
                torch.cuda.synchronize(device)
                start.record()
                for _ in range(_AUTOTUNE_ITERS):
                    graph.execute_plan_at_index(variant_pack, workspace, index=index, handle=handle)
                stop.record()
                stop.synchronize()
                elapsed[index] = start.elapsed_time(stop) / _AUTOTUNE_ITERS
            except Exception as exc:  # noqa: BLE001 -- one invalid plan must not suppress viable plans
                errors[index] = repr(exc)

    best = min(range(count), key=elapsed.__getitem__)
    if elapsed[best] == float("inf"):
        raise RuntimeError(f"cudnn.gemm.gelu_mlp: all {count} autotune plans failed to execute; errors: {errors}")
    return best, workspace


def _mm(a2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Execute ``[M,K] @ [K,N]`` as an autotuned cuDNN graph."""
    with torch.cuda.device(a2.device):
        handle, stream = _handle(a2.device)
        av = a2.unsqueeze(0)
        bv = b2.unsqueeze(0)
        key = (
            tuple(av.shape),
            tuple(av.stride()),
            tuple(bv.shape),
            tuple(bv.stride()),
            a2.dtype,
            a2.device.index,
            stream,
        )
        entry = _MM_CACHE.get(key)
        if entry is None:
            graph = cudnn.pygraph(handle=handle, compute_data_type=_FP32)
            A = graph.tensor(dim=list(av.shape), stride=list(av.stride()), data_type=_BF16)
            B = graph.tensor(dim=list(bv.shape), stride=list(bv.stride()), data_type=_BF16)
            C = graph.matmul(name="mm", A=A, B=B, compute_data_type=_FP32)
            C.set_output(True).set_data_type(_BF16)
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            output = torch.empty((1, av.shape[1], bv.shape[2]), device=a2.device, dtype=a2.dtype)
            best, workspace = _autotune(graph, handle, {A: av, B: bv, C: output})
            entry = (graph, A, B, C, best, workspace)
            _MM_CACHE[key] = entry

        graph, A, B, C, best, workspace = entry
        output = torch.empty((1, av.shape[1], bv.shape[2]), device=a2.device, dtype=a2.dtype)
        graph.execute_plan_at_index({A: av, B: bv, C: output}, workspace, index=best, handle=handle)
        return output.squeeze(0)


def _linear_bias(
    x2: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    gelu: bool,
    save_pre_activation: bool = False,
):
    """Run one fused linear epilogue and return ``(output, pre_activation)``.

    With ``gelu=True``, ``pre_activation`` is the bf16-rounded post-bias value
    and is emitted only when backward needs it.  The h-only and h+z graphs use
    distinct cache entries because output taps are part of a graph's contract.
    """
    if save_pre_activation and not gelu:
        raise ValueError("cudnn.gemm.gelu_mlp: a pre-activation tap is only valid for the GELU layer")

    with torch.cuda.device(x2.device):
        handle, stream = _handle(x2.device)
        xv = x2.unsqueeze(0)
        wv = weight.t().unsqueeze(0)
        bv = bias.view(1, 1, -1)
        key = (
            bool(gelu),
            bool(save_pre_activation),
            tuple(xv.shape),
            tuple(xv.stride()),
            tuple(wv.shape),
            tuple(wv.stride()),
            tuple(bv.stride()),
            x2.dtype,
            x2.device.index,
            stream,
        )
        entry = _LINEAR_CACHE.get(key)
        if entry is None:
            graph = cudnn.pygraph(handle=handle, compute_data_type=_FP32)
            X = graph.tensor(dim=list(xv.shape), stride=list(xv.stride()), data_type=_BF16)
            W = graph.tensor(dim=list(wv.shape), stride=list(wv.stride()), data_type=_BF16)
            BIAS = graph.tensor(dim=list(bv.shape), stride=list(bv.stride()), data_type=_BF16)
            mm = graph.matmul(name="linear", A=X, B=W, compute_data_type=_FP32)
            pre_activation = graph.bias(input=mm, bias=BIAS, name="bias")
            # Preserve eager bf16 Linear -> GELU semantics even while the two
            # operations stay in one graph and need not round-trip through HBM.
            pre_activation.set_data_type(_BF16)
            if gelu:
                output_tensor = graph.gelu_approx_tanh(input=pre_activation, name="gelu_tanh")
                output_tensor.set_output(True).set_data_type(_BF16)
                if save_pre_activation:
                    pre_activation.set_output(True)
            else:
                output_tensor = pre_activation
                output_tensor.set_output(True)

            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            output = torch.empty((1, xv.shape[1], wv.shape[2]), device=x2.device, dtype=x2.dtype)
            variant_pack = {X: xv, W: wv, BIAS: bv, output_tensor: output}
            if gelu and save_pre_activation:
                saved_pre_activation = torch.empty_like(output)
                variant_pack[pre_activation] = saved_pre_activation
            best, workspace = _autotune(graph, handle, variant_pack)
            entry = (
                graph,
                X,
                W,
                BIAS,
                output_tensor,
                pre_activation,
                best,
                workspace,
            )
            _LINEAR_CACHE[key] = entry

        graph, X, W, BIAS, output_tensor, pre_activation, best, workspace = entry
        output = torch.empty((1, xv.shape[1], wv.shape[2]), device=x2.device, dtype=x2.dtype)
        variant_pack = {X: xv, W: wv, BIAS: bv, output_tensor: output}
        if gelu and save_pre_activation:
            saved_pre_activation = torch.empty_like(output)
            variant_pack[pre_activation] = saved_pre_activation
        else:
            saved_pre_activation = None
        graph.execute_plan_at_index(variant_pack, workspace, index=best, handle=handle)
        return output.squeeze(0), None if saved_pre_activation is None else saved_pre_activation.squeeze(0)


def _linear_dgelu(dout2: torch.Tensor, w2: torch.Tensor, pre_activation: torch.Tensor):
    """Fuse ``dh = dout @ w2`` with tanh-GELU backward into one cuDNN graph."""
    with torch.cuda.device(dout2.device):
        handle, stream = _handle(dout2.device)
        dyv = dout2.unsqueeze(0)
        wv = w2.unsqueeze(0)
        zv = pre_activation.unsqueeze(0)
        key = (
            tuple(dyv.shape),
            tuple(dyv.stride()),
            tuple(wv.shape),
            tuple(wv.stride()),
            tuple(zv.stride()),
            dout2.dtype,
            dout2.device.index,
            stream,
        )
        entry = _DGELU_CACHE.get(key)
        if entry is None:
            graph = cudnn.pygraph(handle=handle, compute_data_type=_FP32)
            DY = graph.tensor(dim=list(dyv.shape), stride=list(dyv.stride()), data_type=_BF16)
            W2 = graph.tensor(dim=list(wv.shape), stride=list(wv.stride()), data_type=_BF16)
            Z = graph.tensor(dim=list(zv.shape), stride=list(zv.stride()), data_type=_BF16)
            dh = graph.matmul(name="dhidden", A=DY, B=W2, compute_data_type=_FP32)
            # Match the bf16 gradient crossing the eager second Linear boundary.
            dh.set_data_type(_BF16)
            DZ = graph.gelu_approx_tanh_backward(loss=dh, input=Z, name="dgelu_tanh")
            DZ.set_output(True).set_data_type(_BF16)
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            dz = torch.empty_like(zv)
            best, workspace = _autotune(graph, handle, {DY: dyv, W2: wv, Z: zv, DZ: dz})
            entry = (graph, DY, W2, Z, DZ, best, workspace)
            _DGELU_CACHE[key] = entry

        graph, DY, W2, Z, DZ, best, workspace = entry
        dz = torch.empty_like(zv)
        graph.execute_plan_at_index(
            {DY: dyv, W2: wv, Z: zv, DZ: dz},
            workspace,
            index=best,
            handle=handle,
        )
        return dz.squeeze(0)


class _GeluMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, grad_mask):
        input_shape = x.shape
        x2 = x.reshape(-1, input_shape[-1])
        need_fc1_grad = bool(grad_mask & _GRAD_FC1)
        hidden, pre_activation = _linear_bias(
            x2,
            w1,
            b1,
            gelu=True,
            save_pre_activation=need_fc1_grad,
        )
        output, _ = _linear_bias(hidden, w2, b2, gelu=False)

        saved_names = []
        saved_tensors = []

        def save(name, tensor):
            saved_names.append(name)
            saved_tensors.append(tensor)

        if grad_mask & _GRAD_W1:
            save("x2", x2)
        if grad_mask & _GRAD_X:
            save("w1", w1)
        if need_fc1_grad:
            save("w2", w2)
            save("pre_activation", pre_activation)
        if grad_mask & _GRAD_W2:
            save("hidden", hidden)

        ctx.save_for_backward(*saved_tensors)
        ctx.saved_names = tuple(saved_names)
        ctx.grad_mask = grad_mask
        ctx.input_shape = input_shape
        ctx.out_features = w2.shape[0]
        return output.reshape(*input_shape[:-1], w2.shape[0])

    @staticmethod
    def backward(ctx, dout):
        if torch.is_grad_enabled():
            raise NotImplementedError("cudnn.gemm.gelu_mlp: double backward is not supported")
        saved = dict(zip(ctx.saved_names, ctx.saved_tensors))
        grad_mask = ctx.grad_mask
        dout2 = dout.reshape(-1, ctx.out_features).contiguous()

        dw2 = _mm(dout2.t(), saved["hidden"]) if grad_mask & _GRAD_W2 else None
        db2 = dout2.sum(dim=0) if grad_mask & _GRAD_B2 else None

        if grad_mask & _GRAD_FC1:
            dz = _linear_dgelu(dout2, saved["w2"], saved["pre_activation"])
        else:
            dz = None

        dx = _mm(dz, saved["w1"]).reshape(ctx.input_shape) if grad_mask & _GRAD_X else None
        dw1 = _mm(dz.t(), saved["x2"]) if grad_mask & _GRAD_W1 else None
        db1 = dz.sum(dim=0) if grad_mask & _GRAD_B1 else None
        return dx, dw1, db1, dw2, db2, None


def _validate(x, w1, b1, w2, b2):
    operands = (("x", x), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2))
    for name, tensor in operands:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cudnn.gemm.gelu_mlp: {name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"cudnn.gemm.gelu_mlp: {name} must be bfloat16, got {tensor.dtype}")
        if tensor.device.type != "cuda":
            raise ValueError(f"cudnn.gemm.gelu_mlp: {name} must be a CUDA tensor, got device {tensor.device}")

    if any(tensor.device != x.device for _, tensor in operands[1:]):
        devices = ", ".join(f"{name}={tensor.device}" for name, tensor in operands)
        raise ValueError(f"cudnn.gemm.gelu_mlp: all operands must be on the same CUDA device; got {devices}")

    if x.dim() < 2 or w1.dim() != 2 or b1.dim() != 1 or w2.dim() != 2 or b2.dim() != 1:
        raise ValueError(
            "cudnn.gemm.gelu_mlp: expected x[...,H], w1[I,H], b1[I], "
            f"w2[O,I], b2[O]; got x{tuple(x.shape)}, w1{tuple(w1.shape)}, "
            f"b1{tuple(b1.shape)}, w2{tuple(w2.shape)}, b2{tuple(b2.shape)}"
        )

    in_features = x.shape[-1]
    intermediate = w1.shape[0]
    out_features = w2.shape[0]
    if w1.shape[1] != in_features or b1.shape[0] != intermediate or w2.shape[1] != intermediate or b2.shape[0] != out_features:
        raise ValueError(
            f"cudnn.gemm.gelu_mlp: shape mismatch for x[...,{in_features}], "
            f"w1{tuple(w1.shape)}, b1{tuple(b1.shape)}, w2{tuple(w2.shape)}, b2{tuple(b2.shape)}; "
            f"expected w1=[I,{in_features}], b1=[I], w2=[O,I], b2=[O]"
        )
    if x.numel() == 0 or in_features == 0 or intermediate == 0 or out_features == 0:
        raise ValueError("cudnn.gemm.gelu_mlp: zero-sized dimensions are not supported")

    noncontiguous = [name for name, tensor in operands if not tensor.is_contiguous()]
    if noncontiguous:
        raise ValueError("cudnn.gemm.gelu_mlp: operands must use dense contiguous nn.Linear layouts; " f"noncontiguous: {', '.join(noncontiguous)}")

    capability = torch.cuda.get_device_capability(x.device)
    if capability != (10, 0):
        raise NotImplementedError("cudnn.gemm.gelu_mlp: this implementation requires SM100; " f"got sm_{capability[0]}{capability[1]} on {x.device}")


def gelu_mlp(x, w1, b1, w2, b2):
    """Run a dense bf16 tanh-GELU MLP on cuDNN.

    Args:
        x: Dense rank-two-or-higher input activations ``[..., H]``.
        w1: First ``nn.Linear`` weight ``[I, H]``.
        b1: First ``nn.Linear`` bias ``[I]``.
        w2: Second ``nn.Linear`` weight ``[O, I]``.
        b2: Second ``nn.Linear`` bias ``[O]``.

    Returns:
        Dense bf16 output ``[..., O]``.  First-order gradients are supported
        with respect to all five tensor inputs; double backward is not.

    All inputs must be contiguous bf16 tensors on the same SM100 CUDA device.
    GELU always uses ``approximate="tanh"``.
    """
    _validate(x, w1, b1, w2, b2)

    grad_mask = 0
    if torch.is_grad_enabled():
        for bit, tensor in (
            (_GRAD_X, x),
            (_GRAD_W1, w1),
            (_GRAD_B1, b1),
            (_GRAD_W2, w2),
            (_GRAD_B2, b2),
        ):
            if tensor.requires_grad:
                grad_mask |= bit
    return _GeluMLP.apply(x, w1, b1, w2, b2, grad_mask)
