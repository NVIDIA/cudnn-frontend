# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared BF16 tail RoPE with caller-owned output and constant FP32 tables."""

from dataclasses import replace
from functools import lru_cache

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old


def _require_cute():
    if cutedsl_too_old(cutedsl_state()[1]):
        raise RuntimeError(cutedsl_requirement_error("Frost tail RoPE"))


_require_cute()

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict


@lru_cache(maxsize=64)
def _compile(heads, dim, device_index, capability):
    _require_cute()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from .frost.tail import launch

    tokens = cute.sym_int(64)
    data = make_fake_tensor(cutlass.BFloat16, (tokens * heads * dim,), (1,), assumed_align=16)
    table = make_fake_tensor(cutlass.Float32, (tokens * 32,), (1,), assumed_align=16)
    with torch.cuda.device(device_index):
        return cute.compile(
            launch,
            data,
            table,
            table,
            data,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            cutlass.Int64(0),
            heads,
            dim,
            False,
            16 if (heads, dim) == (64, 512) else 8,
            options="--enable-tvm-ffi",
        )


class TailRoPEForward(APIBase):
    """Copy BF16 [T,D] or [T,H,D], rotating its last 64 channels on SM100.

    Cosine and sine are separate contiguous FP32 [T,32] prepared tables.
    Output is disjoint from every input; all tensors are 16-byte aligned.
    T may change after compile. This explicit forward API has no autograd rule.
    """

    def __init__(self, x, cosine, sine, out, *, backend="frost"):
        super().__init__()
        self._warn_experimental_api()
        declarations = dict(x=x, cosine=cosine, sine=sine, out=out)
        self._descs = {}
        for name, value in declarations.items():
            if not isinstance(value, (torch.Tensor, TensorDesc)):
                raise TypeError(f"{name} must be a torch tensor or TensorDesc")
            desc = self._make_tensor_desc(value, name=name)
            device = torch.device(desc.device.type, desc.device.index)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self._descs[name] = replace(desc, device=device)
        self._requires_grad = any(getattr(value, "requires_grad", False) for value in declarations.values())
        self._backend = backend
        desc = self._descs["x"]
        self._device, self._rank = desc.device, desc.ndim
        self._heads = desc.shape[1] if desc.ndim == 3 else 1
        self._dim = desc.shape[-1] if desc.ndim else 0

    def _validate(self, tensors, *, declaration):
        x = tensors["x"]
        if x.ndim != self._rank or x.ndim not in (2, 3):
            raise ValueError("x must preserve its declared rank: [T,D] or [T,H,D]")
        n = x.shape[0]
        if not isinstance(n, int) or n < 0:
            raise ValueError("T must be a non-negative concrete integer")
        shape = (n, self._heads, self._dim) if self._rank == 3 else (n, self._dim)
        vector = 16 if (self._heads, self._dim) == (64, 512) else 8
        if n * self._heads * self._dim > (2**31 - 1) * 128 * vector:
            raise ValueError("Tensor extent exceeds the CUDA grid limit")
        for name, expected, dtype in (
            ("x", shape, torch.bfloat16),
            ("cosine", (n, 32), torch.float32),
            ("sine", (n, 32), torch.float32),
            ("out", shape, torch.bfloat16),
        ):
            value = tensors[name]
            if tuple(value.shape) != expected or value.dtype != dtype or value.device != self._device:
                raise ValueError(f"{name} must be {dtype} {expected} on {self._device}")
            if not value.is_contiguous():
                strides = value.stride if isinstance(value, TensorDesc) else value.stride()
                error = NotImplementedError if declaration else ValueError
                raise error(f"{name} requires contiguous storage; received strides={strides}")
            if not declaration and value.numel() and value.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned")
        return n

    def check_support(self):
        _require_cute()
        if self._backend != "frost":
            raise ValueError("backend must be 'frost'")
        if self._rank not in (2, 3) or self._dim not in (128, 512) or not isinstance(self._heads, int) or self._heads <= 0 or self._heads * self._dim >= 2**31:
            raise NotImplementedError("Frost tail RoPE supports [T,D] or [T,H,D], H>0, D=128/512, H*D<2**31")
        if self._device.type != "cuda":
            raise ValueError("Frost tail RoPE requires CUDA tensors")
        if self._requires_grad:
            raise ValueError("Tail RoPE has no autograd rule; detach inputs explicitly")
        self._validate(self._descs, declaration=True)
        self._capability = torch.cuda.get_device_capability(self._device)
        if self._capability != (10, 0):
            raise NotImplementedError("Frost tail RoPE currently supports SM100")
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            self._compiled_kernel = _compile(self._heads, self._dim, self._device.index, self._capability)

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        return 0

    def execute(self, x, cosine, sine, out, *, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        tensors = dict(x=x, cosine=cosine, sine=sine, out=out)
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("execute arguments must be torch tensors")
        if any(value.requires_grad for value in tensors.values()):
            raise ValueError("Tail RoPE has no autograd rule; detach inputs explicitly")
        n = self._validate(tensors, declaration=False)
        if not n:
            return TupleDict(out=out)
        low, high = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for name, value in (("x", x), ("cosine", cosine), ("sine", sine)):
            start, end = value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()
            if low < end and start < high:
                raise ValueError(f"out must not overlap {name}")
        with torch.cuda.device(self._device):
            stream = cuda.CUstream(torch.cuda.current_stream(self._device).cuda_stream if current_stream is None else int(current_stream))
            self._compiled_kernel(x.view(-1), cosine.view(-1), sine.view(-1), out.view(-1), stream, x.numel())
        return TupleDict(out=out)


def tail_rope(x, cosine, sine, *, backend="frost", stream=None):
    """Allocate a disjoint output and return ``TupleDict(out=...)``; prewarm before capture."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch tensor")
    if x.device.type != "cuda":
        raise ValueError("Frost tail RoPE requires CUDA tensors")
    with torch.cuda.device(x.device):
        context = (
            torch.cuda.stream(torch.cuda.ExternalStream(int(stream), device=x.device))
            if stream is not None
            else torch.cuda.stream(torch.cuda.current_stream(x.device))
        )
        with context:
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            plan = TailRoPEForward(x, cosine, sine, out, backend=backend)
            plan.compile()
            return plan.execute(x, cosine, sine, out, current_stream=stream)
