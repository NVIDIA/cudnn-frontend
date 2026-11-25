# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace

import torch

import cudnn

FORMATS = ("e4m3", "e5m2", "groupwise", "mxfp8", "nvfp4", "mxfp4", "mxfp8_mxfp4")
TENSOR_SCALE = ("e4m3", "e5m2")


def _swizzle(scales):
    rows, blocks = scales.shape
    padded = scales.new_zeros(((rows + 127) // 128 * 128, (blocks + 3) // 4 * 4))
    padded[:rows, :blocks] = scales
    return padded.view(padded.size(0) // 128, 4, 32, padded.size(1) // 4, 4).permute(0, 3, 2, 1, 4).contiguous().flatten()


@dataclass
class QuantizedMatmul:
    fmt: str
    x: torch.Tensor
    weight: torch.Tensor
    sa: torch.Tensor
    sb: torch.Tensor
    alpha: torch.Tensor | None

    @property
    def block(self):
        return {"groupwise": 128, "nvfp4": 16}.get(self.fmt, 32)

    @property
    def k(self):
        return self.x.size(-1) * (2 if self.x.dtype == torch.uint8 else 1)

    def select_rows(self, rows):
        index = torch.tensor(rows, device=self.x.device)
        x = self.x.view(torch.uint8).index_select(-2, index).view(self.x.dtype)
        sa = self.sa if self.fmt in TENSOR_SCALE else self.sa.index_select(0, index)
        return replace(self, x=x, sa=sa)

    def inputs(self):
        a, b = self.x, self.weight.transpose(-1, -2)
        sa, sb = self.sa, self.sb
        if self.fmt not in (*TENSOR_SCALE, "groupwise"):
            sa, sb = _swizzle(sa), _swizzle(sb)
        return a, b, sa, sb

    def decoded(self):
        # Decode nibbles and scale bytes independently of the cuDNN implementations.
        def operand(data, scales, row_group):
            if data.dtype == torch.uint8:
                magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
                codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
                values = magnitude[codes & 7].double() * (1 - 2 * (codes >> 3))
            else:
                values = data.float().double()
            if self.fmt in TENSOR_SCALE:
                return values * scales
            if self.fmt == "nvfp4":
                scales = scales.view(torch.float8_e4m3fn).float()
            elif self.fmt != "groupwise":
                scales = torch.exp2(scales.cpu().float() - 127).to(data.device)
            return values * scales.repeat_interleave(row_group, 0)[: values.shape[0]].repeat_interleave(self.block, 1)

        return operand(self.x, self.sa, 1), operand(self.weight, self.sb, 128 if self.fmt == "groupwise" else 1)


def make_case(fmt, device, m=7, n=136, k=256, *, exact=False, batch=2):
    gen = torch.Generator().manual_seed(913)
    prefix = (batch,) if fmt in TENSOR_SCALE else ()

    def operand(rows, packed):
        shape = (*prefix, rows, k)
        if packed:
            codes = torch.randint(0, 16, shape, generator=gen, dtype=torch.uint8)
            return (codes[..., ::2] | (codes[..., 1::2] << 4)).to(device)
        dtype = torch.float8_e5m2 if fmt == "e5m2" else torch.float8_e4m3fn
        if exact:
            values = torch.randint(-3, 4, shape, generator=gen)
        else:
            values = torch.randn(shape, generator=gen) * torch.exp2(torch.randint(-5, 5, shape, generator=gen).float())
        return values.to(dtype=dtype, device=device)

    x = operand(m, fmt in ("nvfp4", "mxfp4"))
    weight = operand(n, fmt in ("nvfp4", "mxfp4", "mxfp8_mxfp4"))
    block = {"groupwise": 128, "nvfp4": 16}.get(fmt, 32)

    def scales(rows):
        power = torch.randint(-2, 2, (rows, k // block), generator=gen)
        values = torch.exp2(power.float())
        if fmt == "groupwise":
            return (values if exact else values * (torch.rand(values.shape, generator=gen) + 0.5)).to(device)
        if fmt == "nvfp4":
            return values.to(torch.float8_e4m3fn).view(torch.uint8).to(device)
        return (127 + power).to(torch.uint8).to(device)

    if fmt in TENSOR_SCALE:
        sa, sb = (torch.tensor([v], device=device) for v in (0.75, 0.5))
    else:
        sa, sb = scales(m), scales((n + 127) // 128 if fmt == "groupwise" else n)
    alpha = torch.tensor([0.75], device=device) if fmt in ("nvfp4", "mxfp4", "mxfp8_mxfp4") else None
    return QuantizedMatmul(fmt, x, weight, sa, sb, alpha)


def quantized_graph(case, inputs, out, *, rank_three=False):
    a, b, sa, sb = inputs
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bindings = {}

    def bind(desc, value):
        bindings[desc] = value
        return desc

    def bind_like(value):
        if rank_three and value.ndim < 3:
            leading = 3 - value.ndim
            value = value.as_strided((1,) * leading + value.shape, (value.numel(),) * leading + value.stride())
        return bind(graph.tensor_like(value), value)

    def matrix_desc(rows, cols, transpose, **kwargs):
        dims, strides = ((cols, rows), (1, cols)) if transpose else ((rows, cols), (cols, 1))
        if rank_three:
            dims, strides = (1, *dims), (rows * cols, *strides)
        return graph.tensor(dim=dims, stride=strides, **kwargs)

    def data(value, packed, rows, transpose):
        if not packed:
            return bind_like(value)
        return bind(matrix_desc(rows, case.k, transpose, data_type=cudnn.data_type.FP4_E2M1), value)

    x = data(a, a.dtype == torch.uint8, a.size(-2), False)
    w = data(b, b.dtype == torch.uint8, b.size(-1), True)
    if case.fmt in TENSOR_SCALE:
        scale = graph.mul(a=bind_like(sa), b=bind_like(sb))
        y = graph.mul(a=graph.matmul(A=x, B=w), b=scale)
    else:

        def scaled(desc, scale, rows, transpose):
            row_group = 128 if case.fmt == "groupwise" and transpose else 1
            if case.fmt == "groupwise":
                value = scale.T if transpose else scale
                sdesc = bind_like(value)
            else:
                r = (rows + 127) // 128 * 128
                k = (case.k // case.block + 3) // 4 * 4
                sf_type = cudnn.data_type.FP8_E4M3 if case.fmt == "nvfp4" else cudnn.data_type.FP8_E8M0
                sdesc = bind(matrix_desc(r, k, transpose, data_type=sf_type, reordering_type=cudnn.tensor_reordering.F8_128x4), scale)
            blocks = [case.block, row_group] if transpose else [row_group, case.block]
            return graph.block_scale_dequantize(input=desc, descale=sdesc, block_size=blocks)

        y = graph.matmul(A=scaled(x, sa, a.size(-2), False), B=scaled(w, sb, b.size(-1), True))
        if case.alpha is not None:
            y = graph.mul(a=y, b=bind_like(case.alpha))
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16 if out.dtype == torch.bfloat16 else cudnn.data_type.HALF)
    bindings[y] = out
    return graph, bindings


def build_quantized(case, dtype):
    inputs = case.inputs()
    a, b = inputs[:2]
    out = torch.empty((*a.shape[:-1], b.shape[-1]), device=a.device, dtype=dtype)
    graph, bindings = quantized_graph(case, inputs, out)
    graph.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
    graph.build()
    assert graph.selected_engine.name == "batch_invariant_gemm"
    work = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=a.device)

    def run():
        graph.execute(bindings, work)
        return {"out": out}

    return run
