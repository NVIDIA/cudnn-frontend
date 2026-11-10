# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic BF16 RMSNorm(x) * weight * SiLU(z), with width 128."""

from __future__ import annotations

import math

from cudnn._tupledict import TupleDict


def _geometry(shape):
    shape = tuple(shape)
    if len(shape) not in (2, 3) or shape[-1] != 128 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("shape must be positive [rows, 128] or [tokens, heads, 128]")
    tokens, heads = (shape[0], 1) if len(shape) == 2 else shape[:2]
    if tokens * heads > 2**31 - 1:
        raise ValueError("row count exceeds signed int32 indexing")
    return shape, tokens, heads


def _tensor(tensor, name, shape, dtype, device, *, contiguous=False):
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {device}")
    if tensor.data_ptr() % 32:
        raise ValueError(f"{name} must be 32-byte aligned")
    if contiguous:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    elif tensor.stride(-1) != 1 or any(stride < 0 or stride % 16 for stride, size in zip(tensor.stride()[:-1], shape[:-1]) if size > 1):
        raise ValueError(f"{name} must have unit last stride and outer strides divisible by 16")


def _no_overlap(outputs, inputs):
    import torch

    for index, (name, tensor) in enumerate(outputs):
        for other_name, other in inputs + outputs[:index]:
            if torch._C._overlaps(tensor, other):
                raise ValueError(f"{name} must not overlap {other_name}")


class GatedRMSNormPlan:
    """Prepared forward/backward kernels; outputs and scratch belong to the caller."""

    def __init__(self, shape, *, device=None, eps=1e-6):
        import torch

        from cudnn._cake import loader

        self.shape, self.tokens, self.heads = _geometry(shape)
        self.rows = self.tokens * self.heads
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = float(eps)
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        arch = loader.device_arch(self.device)
        loader.check_toolkit(arch, "gated RMSNorm")
        package = loader.load_package("cake_rmsnorm_silu_gate_det_train")
        jit = package.cake_jit
        with torch.cuda.device(self.device):
            sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
            names = (jit.module_for_rows(arch, "fwd", self.rows, sm_count), jit.route_module(arch, "bwd"), jit.route_module(arch, "dw_f32"))
            self.chunks, self.blocks_per_cta = jit.backward_chunks(names[1], self.rows)
            self._launches = []
            for name in names:
                record = jit.MODULES[name]
                grid = jit.launch_grid(record, rows=self.rows, sm_count=sm_count, chunks=self.chunks)
                grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid))
                arguments = tuple((kind, grid_values[key] if kind == "grid" else record["logical"][key]) for kind, key in record["arg_plan"])
                entry = getattr(jit.load(name, arch), record["ffi_entry"])
                self._launches.append((entry, arguments))

    def get_workspace_size(self):
        return self.chunks * 128 * 4

    def _inputs(self, x, z, weight):
        import torch

        _tensor(x, "x", self.shape, torch.bfloat16, self.device)
        _tensor(z, "z", self.shape, torch.bfloat16, self.device)
        _tensor(weight, "weight", (128,), torch.bfloat16, self.device, contiguous=True)
        values = {"x": x, "z": z, "w": weight, "rows": self.rows, "heads": self.heads, "eps": self.eps}
        for name, tensor in (("x", x), ("z", z)):
            values[name + "_tok"] = tensor.stride(0) if self.tokens > 1 else 0
            values[name + "_head"] = tensor.stride(1) if len(self.shape) == 3 and self.heads > 1 else 128
        return values

    def _run(self, index, values):
        import tvm_ffi

        entry, arguments = self._launches[index]
        with tvm_ffi.use_torch_stream():
            entry(*(key if kind == "grid" else values[key] for kind, key in arguments))

    def forward(self, x, z, weight, out, *, stream=None):
        import torch

        from cudnn._torch_stream import stream_context

        values = self._inputs(x, z, weight)
        _tensor(out, "out", self.shape, torch.bfloat16, self.device, contiguous=True)
        _no_overlap([("out", out)], [("x", x), ("z", z), ("weight", weight)])
        values["y"] = out
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            self._run(0, values)

    def backward(self, grad_out, x, z, weight, grad_x, grad_z, grad_weight, workspace, *, stream=None):
        import torch

        from cudnn._torch_stream import stream_context

        values = self._inputs(x, z, weight)
        _tensor(grad_out, "grad_out", self.shape, torch.bfloat16, self.device)
        _tensor(grad_x, "grad_x", self.shape, torch.bfloat16, self.device, contiguous=True)
        _tensor(grad_z, "grad_z", self.shape, torch.bfloat16, self.device, contiguous=True)
        _tensor(grad_weight, "grad_weight", (128,), torch.float32, self.device, contiguous=True)
        if not isinstance(workspace, torch.Tensor) or workspace.numel() * workspace.element_size() < self.get_workspace_size():
            raise ValueError(f"workspace needs at least {self.get_workspace_size()} bytes")
        _tensor(workspace, "workspace", tuple(workspace.shape), torch.float32, self.device, contiguous=True)
        _no_overlap(
            [("grad_x", grad_x), ("grad_z", grad_z), ("grad_weight", grad_weight), ("workspace", workspace)],
            [("grad_out", grad_out), ("x", x), ("z", z), ("weight", weight)],
        )
        values.update(
            dy=grad_out,
            dx=grad_x,
            dz=grad_z,
            dw=grad_weight,
            partial=workspace,
            dy_tok=grad_out.stride(0) if self.tokens > 1 else 0,
            dy_head=grad_out.stride(1) if len(self.shape) == 3 and self.heads > 1 else 128,
            blocks_per_cta=self.blocks_per_cta,
            chunks=self.chunks,
            accumulate=0,
        )
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            self._run(1, values)
            self._run(2, values)


def gated_rmsnorm_forward(x, z, weight, eps=1e-6, *, stream=None):
    """Return BF16 ``output`` for RMSNorm(x) * weight * SiLU(z)."""
    import torch

    from cudnn._torch_stream import stream_context

    plan = GatedRMSNormPlan(tuple(x.shape), device=x.device, eps=eps)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        plan.forward(x, z, weight, out, stream=stream)
    return TupleDict(output=out)


def gated_rmsnorm_backward(grad_out, x, z, weight, eps=1e-6, *, stream=None):
    """Return BF16 ``grad_x``/``grad_z`` and deterministic FP32 ``grad_weight``."""
    import torch

    from cudnn._torch_stream import stream_context

    plan = GatedRMSNormPlan(tuple(x.shape), device=x.device, eps=eps)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        dx, dz = (torch.empty(x.shape, dtype=torch.bfloat16, device=x.device) for _ in range(2))
        dw = torch.empty(128, dtype=torch.float32, device=x.device)
        workspace = torch.empty(plan.get_workspace_size() // 4, dtype=torch.float32, device=x.device)
        plan.backward(grad_out, x, z, weight, dx, dz, dw, workspace, stream=stream)
    return TupleDict(grad_x=dx, grad_z=dz, grad_weight=dw)
