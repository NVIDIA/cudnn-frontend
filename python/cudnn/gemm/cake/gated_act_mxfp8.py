# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused SwiGLU and MXFP8 quantization with RCEIL block scales."""

from __future__ import annotations

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._tupledict import TupleDict


def _geometry(shape, rowwise, colwise):
    if type(rowwise) is not bool or type(colwise) is not bool or not (rowwise or colwise):
        raise ValueError("rowwise and colwise must be booleans with at least one enabled")
    shape = tuple(shape)
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape) or shape[0] % 128 or shape[1] % 256:
        raise ValueError("gated_input must have shape [M, 2K] with positive M and K divisible by 128")
    if shape[0] * shape[1] > 2**31:
        raise ValueError("gated_input exceeds signed int32 indexing")
    return shape[0], shape[1] // 2


def _tensor(value, name, shape, stride, dtype, device):
    import torch

    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {device}")
    if tuple(value.stride()) != stride or value.data_ptr() % 32:
        raise ValueError(f"{name} needs strides {stride} and 32-byte alignment")


class GatedActMxfp8Plan:
    """A single compiled forward or backward kernel with caller-owned outputs."""

    def __init__(self, shape, *, backward=False, rowwise=True, colwise=False, device=None):
        import torch

        from cudnn._cake import loader

        self.m, self.k = _geometry(shape, rowwise, colwise)
        if type(backward) is not bool:
            raise ValueError("backward must be a boolean")
        self.backward, self.rowwise, self.colwise = backward, rowwise, colwise
        self.n = 2 * self.k if backward else self.k
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        arch = loader.device_arch(self.device)
        loader.check_toolkit(arch, "gated activation MXFP8")
        package = loader.load_package("cake_gated_act_mxfp8_backend")
        jit = package.cake_jit
        kind = jit.route_name(backward=backward, rowwise=rowwise, colwise=colwise)
        name = jit.selected_module(arch, kind, M=self.m, K=self.k)
        record = jit.MODULES[name]
        grid = jit.launch_grid(record, M=self.m, K=self.k)
        if grid[0] > 2**31 - 1 or grid[1] > 65535:
            raise NotImplementedError("selected Cake activation grid exceeds CUDA limits")
        grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid))
        self._arguments = tuple((kind, grid_values[key] if kind == "grid" else record["logical"][key]) for kind, key in record["arg_plan"])
        with torch.cuda.device(self.device):
            self._entry = getattr(jit.load(name, arch), record["ffi_entry"])

    def get_workspace_size(self):
        return 0

    def execute(self, gated_input, outputs, grad_output=None, *, stream=None):
        import torch
        import tvm_ffi

        from cudnn._torch_stream import stream_context

        _tensor(gated_input, "gated_input", (self.m, 2 * self.k), (2 * self.k, 1), torch.bfloat16, self.device)
        if self.backward:
            _tensor(grad_output, "grad_output", (self.m, self.k), (self.k, 1), torch.bfloat16, self.device)
        elif grad_output is not None:
            raise ValueError("a forward plan does not accept grad_output")
        if len(outputs) != 4:
            raise ValueError("outputs must contain row_output, col_output, row_scales, col_scales")
        row, col, row_sf, col_sf = outputs
        specifications = (
            ("row_output", row, self.rowwise, (self.m, self.n), (self.n, 1), torch.float8_e4m3fn),
            ("col_output", col, self.colwise, (self.m, self.n), (1, self.m), torch.float8_e4m3fn),
            ("row_scales", row_sf, self.rowwise, (self.m, self.n // 32), (self.n // 32, 1), torch.uint8),
            ("col_scales", col_sf, self.colwise, (self.n * self.m // 32,), (1,), torch.uint8),
        )
        active = []
        inputs = [("gated_input", gated_input)] + ([("grad_output", grad_output)] if self.backward else [])
        for name, tensor, enabled, shape, stride, dtype in specifications:
            _tensor(tensor, name, shape if enabled else (0,), stride if enabled else (1,), dtype, self.device)
            if enabled:
                active.append((name, tensor))
        check_buffers_disjoint(active, inputs, what="gated activation kernels")
        values = {
            "gated_input": gated_input,
            "grad_output": grad_output,
            "row_output": row.view(torch.uint8),
            "col_output": (col.t() if self.colwise else col).view(torch.uint8),
            "row_scales": row_sf,
            "col_scales": col_sf,
            "M": self.m,
            "K": self.k,
        }
        with torch.cuda.device(self.device), stream_context(stream, self.device), tvm_ffi.use_torch_stream():
            self._entry(*(key if kind == "grid" else values[key] for kind, key in self._arguments))


def _outputs(plan):
    import torch

    m, n, device = plan.m, plan.n, plan.device
    row = torch.empty((m, n) if plan.rowwise else (0,), device=device, dtype=torch.float8_e4m3fn)
    col = (
        torch.empty_strided((m, n), (1, m), device=device, dtype=torch.float8_e4m3fn)
        if plan.colwise
        else torch.empty(0, device=device, dtype=torch.float8_e4m3fn)
    )
    row_sf = torch.empty((m, n // 32) if plan.rowwise else (0,), device=device, dtype=torch.uint8)
    col_sf = torch.empty(n * m // 32 if plan.colwise else 0, device=device, dtype=torch.uint8)
    return row, col, row_sf, col_sf


def _call(gated_input, grad_output, rowwise, colwise, stream):
    import torch

    from cudnn._torch_stream import stream_context

    if not isinstance(gated_input, torch.Tensor) or gated_input.device.type != "cuda":
        raise ValueError("gated_input must be a CUDA tensor")
    m, k = _geometry(tuple(gated_input.shape), rowwise, colwise)
    _tensor(gated_input, "gated_input", (m, 2 * k), (2 * k, 1), torch.bfloat16, gated_input.device)
    if grad_output is not None:
        _tensor(grad_output, "grad_output", (m, k), (k, 1), torch.bfloat16, gated_input.device)
    plan = GatedActMxfp8Plan(tuple(gated_input.shape), backward=grad_output is not None, rowwise=rowwise, colwise=colwise, device=gated_input.device)
    with torch.cuda.device(plan.device), stream_context(stream, plan.device):
        outputs = _outputs(plan)
        plan.execute(gated_input, outputs, grad_output, stream=stream)
        gated_input.record_stream(torch.cuda.current_stream(plan.device))
        if grad_output is not None:
            grad_output.record_stream(torch.cuda.current_stream(plan.device))
    return TupleDict(zip(("row_output", "col_output", "row_scales", "col_scales"), outputs))


def gated_act_mxfp8_forward(gated_input, *, rowwise=True, colwise=False, stream=None):
    """Quantize SwiGLU to E4M3 values and swizzled E8M0 scale bytes."""
    return _call(gated_input, None, rowwise, colwise, stream)


def gated_act_mxfp8_backward(gated_input, grad_output, *, rowwise=True, colwise=False, stream=None):
    """Quantize the SwiGLU gate/up gradients to MXFP8."""
    if grad_output is None:
        raise ValueError("grad_output is required")
    return _call(gated_input, grad_output, rowwise, colwise, stream)
