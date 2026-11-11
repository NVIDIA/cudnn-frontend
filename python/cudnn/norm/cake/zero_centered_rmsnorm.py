# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-centered RMSNorm with FP32 parameters and deterministic Cake gradients."""

from __future__ import annotations

import math

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._tupledict import TupleDict


def _geometry(shape):
    shape = tuple(shape)
    if len(shape) < 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("shape must be positive [..., hidden] with at least two dimensions")
    rows, hidden = math.prod(shape[:-1]), shape[-1]
    if rows * hidden > 2**31 - 1:
        raise ValueError("shape exceeds signed int32 indexing")
    return shape, rows, hidden


def _tensor(tensor, name, shape, dtype, device, alignment=32):
    import torch

    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


class ZeroCenteredRMSNormPlan:
    """Prepare the FP32 weight-offset stage and Cake forward/backward kernels."""

    def __init__(self, shape, *, device=None, eps=1e-6):
        import torch

        from cudnn._cake import loader

        self.shape, self.rows, self.hidden = _geometry(shape)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = float(eps)
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        arch = loader.device_arch(self.device)
        loader.check_toolkit(arch, "zero-centered RMSNorm")
        package = loader.load_package("cake_rmsnorm_det_train")
        jit = package.cake_jit
        if self.hidden not in jit.supported_hidden_sizes(arch):
            raise ValueError(f"hidden must be one of {jit.supported_hidden_sizes(arch)} on {arch}")
        if arch == "sm_107a" and self.hidden == 4096:
            raise ValueError("SM107a width 4096 needs oversized shared-memory support, which this API does not yet enable")
        with torch.cuda.device(self.device):
            selected = package._backward_selection_rows(arch, self.hidden, False, self.rows, "f32", self.device.index)
            forward = jit.selected_modules(arch, self.hidden, "fwd", False, rows=self.rows, w_dtype="f32")
            backward = jit.selected_modules(arch, self.hidden, "bwd", False, rows=selected, w_dtype="f32")
            owner = package._workspace_record(arch, self.hidden, False, self.rows, "f32")
            main = next(jit.MODULES[name] for name in backward if jit.MODULES[name].get("workspace_rule") is not None)
            self.layout = jit.workspace_layout(main, self.rows, device_index=self.device.index, capacity_record=owner)
            self._partial_dtype = torch.float64 if main["workspace_rule"]["kind"] == "claim_sum_f64" else torch.float32
            self._launches = []
            for names in (forward, backward):
                launches = []
                for name in names:
                    record = jit.MODULES[name]
                    grid = dict(zip(("grid_x", "grid_y", "grid_z"), jit.launch_grid(record, self.rows, device_index=self.device.index)))
                    arguments = tuple((kind, grid[key] if kind == "grid" else record["logical"][key]) for kind, key in record["arg_plan"])
                    launches.append((jit.load(name, arch).run, arguments))
                self._launches.append(launches)
        self._scale_offset = (self.layout["total_bytes"] + 255) // 256 * 256

    def get_workspace_size(self):
        return self._scale_offset + self.hidden * 4

    def _inputs(self, x, weight, workspace):
        import torch

        _tensor(x, "x", self.shape, torch.bfloat16, self.device)
        _tensor(weight, "weight", (self.hidden,), torch.float32, self.device)
        if not isinstance(workspace, torch.Tensor) or workspace.ndim != 1 or workspace.numel() < self.get_workspace_size():
            raise ValueError(f"workspace needs at least {self.get_workspace_size()} uint8 elements")
        _tensor(workspace, "workspace", tuple(workspace.shape), torch.uint8, self.device, 256)
        scale = workspace[self._scale_offset : self._scale_offset + 4 * self.hidden].view(torch.float32)
        return {"x": x.view(self.rows, self.hidden), "w": scale, "rows": self.rows, "x_stride": self.hidden}

    def _run(self, index, values):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, arguments in self._launches[index]:
                entry(*(key if kind == "grid" else values[key] for kind, key in arguments))

    def forward(self, x, weight, out, rstd, workspace, *, stream=None):
        import torch

        from cudnn._torch_stream import stream_context

        values = self._inputs(x, weight, workspace)
        _tensor(out, "out", self.shape, torch.bfloat16, self.device)
        _tensor(rstd, "rstd", (self.rows,), torch.float32, self.device)
        check_buffers_disjoint([("out", out), ("rstd", rstd), ("workspace", workspace)], [("x", x), ("weight", weight)])
        values.update(y=out.view(self.rows, self.hidden), r=rstd, eps=self.eps, u=values["x"], h_new=out.view(self.rows, self.hidden), u_stride=0)
        with torch.no_grad(), torch.cuda.device(self.device), stream_context(stream, self.device):
            torch.add(weight, 1.0, out=values["w"])
            self._run(0, values)

    def backward(self, grad_out, x, weight, rstd, grad_x, grad_weight, workspace, *, stream=None):
        import torch

        from cudnn._torch_stream import stream_context

        values = self._inputs(x, weight, workspace)
        _tensor(grad_out, "grad_out", self.shape, torch.bfloat16, self.device)
        _tensor(rstd, "rstd", (self.rows,), torch.float32, self.device)
        _tensor(grad_x, "grad_x", self.shape, torch.bfloat16, self.device)
        _tensor(grad_weight, "grad_weight", (self.hidden,), torch.float32, self.device)
        check_buffers_disjoint(
            [("grad_x", grad_x), ("grad_weight", grad_weight), ("workspace", workspace)],
            [("grad_out", grad_out), ("x", x), ("weight", weight), ("rstd", rstd)],
        )
        layout = self.layout
        start = layout["partial_offset"]
        partial = workspace[start : start + layout["partial_bytes"]].view(self._partial_dtype)
        if layout["partial_rows"]:
            partial = partial[: layout["n_chunks"] * layout["partial_rows"] * self.hidden].view(-1, self.hidden)
        counters = workspace[: 4 * max(layout["counters"], 1)].view(torch.uint32)
        grad = grad_out.view(self.rows, self.hidden)
        values.update(
            g=grad,
            r=rstd,
            dx=grad_x.view(self.rows, self.hidden),
            dw=grad_weight,
            partial=partial,
            counters=counters,
            g_h=grad,
            g_stride=self.hidden,
            g_h_stride=self.hidden,
            n_chunks=layout["n_chunks"],
            slots=layout["slots"],
        )
        with torch.no_grad(), torch.cuda.device(self.device), stream_context(stream, self.device):
            torch.add(weight, 1.0, out=values["w"])
            counters.zero_()
            self._run(1, values)


def zero_centered_rmsnorm_forward(x, weight, eps=1e-6, *, stream=None):
    """Return BF16 ``output`` and saved FP32 ``rstd`` for BF16 x and FP32 weight."""
    import torch

    from cudnn._torch_stream import stream_context

    plan = ZeroCenteredRMSNormPlan(tuple(x.shape), device=x.device, eps=eps)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        output = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        rstd = torch.empty(plan.rows, dtype=torch.float32, device=x.device)
        workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=x.device)
        plan.forward(x, weight, output, rstd, workspace)
        for tensor in (x, weight):
            tensor.record_stream(torch.cuda.current_stream(x.device))
    return TupleDict(output=output, rstd=rstd)


def zero_centered_rmsnorm_backward(grad_out, x, weight, rstd, *, stream=None):
    """Return deterministic BF16 ``grad_x`` and FP32 ``grad_weight`` using saved rstd."""
    import torch

    from cudnn._torch_stream import stream_context

    plan = ZeroCenteredRMSNormPlan(tuple(x.shape), device=x.device)
    with torch.cuda.device(x.device), stream_context(stream, x.device):
        grad_x = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        grad_weight = torch.empty(plan.hidden, dtype=torch.float32, device=x.device)
        workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=x.device)
        plan.backward(grad_out, x, weight, rstd, grad_x, grad_weight, workspace)
        for tensor in (grad_out, x, weight, rstd):
            tensor.record_stream(torch.cuda.current_stream(x.device))
    return TupleDict(grad_x=grad_x, grad_weight=grad_weight)
