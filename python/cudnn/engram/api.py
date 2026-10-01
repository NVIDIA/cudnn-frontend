# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit saved-state Engram gate APIs with native packed KV gradients."""

from dataclasses import replace
from functools import lru_cache
import math

from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old


def _require_cute():
    if cutedsl_too_old(cutedsl_state()[1]):
        raise RuntimeError(cutedsl_requirement_error("Engram saved-state gate"))


# Direct submodule imports must obey the same version boundary as lazy exports.
_require_cute()

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import torch
from cudnn.api_base import APIBase, TupleDict


@lru_cache(maxsize=1)
def _reductions():
    import triton
    from packaging.version import Version

    if Version(triton.__version__) < Version("3.7.0"):
        raise RuntimeError(f"Engram saved backward requires Triton >=3.7.0; found {triton.__version__}")
    from . import _reductions as kernels

    return triton, kernels


def _ptr(dtype, address):
    kind = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.bool: cutlass.Uint8}[dtype]
    return make_ptr(kind, address, cute.AddressSpace.gmem, assumed_align=1 if dtype == torch.bool else 16)


@lru_cache(maxsize=64)
def _compile_forward(d, eps, device_index, capability):
    _require_cute()
    from ._forward import launch_forward

    types = (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32, torch.bool, torch.bfloat16, torch.float32)
    with torch.cuda.device(device_index):
        return cute.compile(launch_forward, *(_ptr(t, 0) for t in types), cutlass.Int32(0), cuda.CUstream(0), d, eps, 5 * d, 5 * d)


@lru_cache(maxsize=64)
def _compile_backward(tokens, d, device_index, capability):
    _require_cute()
    triton, kernels = _reductions()
    from ._backward import launch_apply

    types = (torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16, torch.float32, torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32)
    splits = tokens // 64
    with torch.cuda.device(device_index):
        apply = cute.compile(launch_apply, *(_ptr(t, 0) for t in types), cutlass.Int32(0), cuda.CUstream(0), d, 5 * d, 5 * d, 5 * d)
        moments = kernels.cudnn_engram_gate_saved_moments.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            5 * d,
            d,
            8192,
            grid=(tokens * 4, 1, 1),
            num_warps=8,
            enable_fp_fusion=False,
        )
        reduce = kernels.cudnn_engram_gate_weight_reduce.warmup(
            torch.float32,
            torch.float32,
            splits,
            d,
            triton.next_power_of_2(splits),
            128,
            grid=(4 * d // 128, 1, 1),
            num_warps=8,
            enable_fp_fusion=False,
        )
        # Initialize device handles and launch adapters now, without executing.
        return apply, moments, reduce, moments[(tokens * 4, 1, 1)], reduce[(4 * d // 128, 1, 1)]


class _SavedGate(APIBase):
    def __init__(self, declarations, *, eps, backend):
        super().__init__()
        self._warn_experimental_api()
        self._descs = {}
        for name, tensor in declarations.items():
            desc = self._make_tensor_desc(tensor, name=name)
            device = torch.device(desc.device.type, desc.device.index)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            # Bind metadata-only declarations now, so later device selection
            # cannot redirect compilation or make concrete tensors mismatch.
            self._descs[name] = replace(desc, device=device)
        self._eps, self._backend = float(eps), backend
        x = self._descs["x"]
        self._tokens = x.shape[0] if x.ndim == 3 else 0
        self._device = x.device
        self._d = 5120

    def _check_common(self):
        _require_cute()
        self._value_error_if(self._backend != "frost", "backend must be 'frost'")
        self._value_error_if(not 0 < self._tokens <= 64 * 65535, "tokens must be in [1,4194240]")
        self._value_error_if(not math.isfinite(self._eps) or self._eps <= 0, "eps must be finite and positive")
        for name, shape, dtype in self._declarations():
            desc = self._descs[name]
            self._value_error_if(
                desc.shape != shape or desc.dtype != dtype or not desc.is_contiguous() or desc.device != self._device,
                f"{name} must be contiguous {dtype} {shape} on {self._device}",
            )
        self._runtime_error_if(self._device.type != "cuda", "Engram gate requires CUDA")
        self._capability = torch.cuda.get_device_capability(self._device)
        self._not_implemented_error_if(self._capability != (10, 0), "Engram saved-state gate currently supports SM100")

    def _validate(self, inputs, outputs, workspace=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        declarations = list(self._declarations()) + list(self._output_declarations())
        tensors = dict(inputs) | dict(outputs)
        for name, shape, dtype in declarations:
            tensor = tensors[name]
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self._device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % (1 if dtype == torch.bool else 16)
            ):
                raise ValueError(f"{name} must match planned shape, dtype, device, contiguous layout and alignment")
        destinations = list(outputs.values())
        if workspace is not None:
            if (
                workspace.dtype != torch.uint8
                or workspace.ndim != 1
                or not workspace.is_contiguous()
                or workspace.device != self._device
                or workspace.numel() < self._workspace_bytes
                or workspace.data_ptr() % 16
            ):
                raise ValueError(f"workspace must be aligned contiguous uint8 on {self._device}, at least {self._workspace_bytes} bytes")
            destinations.append(workspace)
        # Public destinations are packed parents, not overlapping bounding spans
        # of logically disjoint DK and DV views inside one packed gradient.
        for index, dest in enumerate(destinations):
            low, high = dest.data_ptr(), dest.data_ptr() + dest.numel() * dest.element_size()
            for other in (*inputs.values(), *destinations[:index]):
                start, end = other.data_ptr(), other.data_ptr() + other.numel() * other.element_size()
                if low < end and start < high:
                    raise ValueError("outputs and workspace must not overlap inputs or each other")

    def _stream(self, current_stream):
        return cuda.CUstream(torch.cuda.current_stream(self._device).cuda_stream if current_stream is None else int(current_stream))


class EngramGateSavedForward(_SavedGate):
    """FP32 gate/residual forward producing explicit saved state for backward.

    KV is contiguous [N,5*5120]: four keys followed by one shared value.
    The caller retains X, KV, weight and saved state until backward completes.
    """

    def __init__(self, x, kv, weight, token_mask, *, eps=1e-20, backend="frost"):
        super().__init__(dict(x=x, kv=kv, weight=weight, token_mask=token_mask), eps=eps, backend=backend)

    def _declarations(self):
        n, d = self._tokens, self._d
        return (("x", (n, 4, d), torch.bfloat16), ("kv", (n, 5 * d), torch.bfloat16), ("weight", (4, d), torch.float32), ("token_mask", (n,), torch.bool))

    def _output_declarations(self):
        return (("out", (self._tokens, 4, self._d), torch.bfloat16), ("saved", (self._tokens, 4, 4), torch.float32))

    def check_support(self):
        self._check_common()
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        self._compiled_kernel = _compile_forward(self._d, self._eps, self._device.index, self._capability)

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        return 0

    def execute(self, x, kv, weight, token_mask, out, saved, *, current_stream=None):
        self._validate(dict(x=x, kv=kv, weight=weight, token_mask=token_mask), dict(out=out, saved=saved))
        with torch.cuda.device(self._device):
            pointers = (
                _ptr(torch.bfloat16, x.data_ptr()),
                _ptr(torch.bfloat16, kv.data_ptr()),
                _ptr(torch.bfloat16, kv.data_ptr() + 4 * self._d * 2),
                _ptr(torch.float32, weight.data_ptr()),
                _ptr(torch.bool, token_mask.data_ptr()),
                _ptr(torch.bfloat16, out.data_ptr()),
                _ptr(torch.float32, saved.data_ptr()),
            )
            self._compiled_kernel(*pointers, cutlass.Int32(self._tokens), self._stream(current_stream))
        return TupleDict(out=out, saved=saved)


class EngramGateSavedBackward(_SavedGate):
    """Deterministic saved-state backward producing contiguous packed dKV.

    The supplied saved state must come from forward with the same X/KV/weight.
    N must be a multiple of 64. dWeight is FP32; dX and packed dKV are BF16.
    """

    def __init__(self, x, kv, weight, saved, grad_out, *, backend="frost"):
        super().__init__(dict(x=x, kv=kv, weight=weight, saved=saved, grad_out=grad_out), eps=1e-20, backend=backend)
        self._workspace_bytes = (self._tokens * 4 * 4 + (self._tokens // 64) * 4 * self._d) * 4

    def _declarations(self):
        n, d = self._tokens, self._d
        return (
            ("x", (n, 4, d), torch.bfloat16),
            ("kv", (n, 5 * d), torch.bfloat16),
            ("weight", (4, d), torch.float32),
            ("saved", (n, 4, 4), torch.float32),
            ("grad_out", (n, 4, d), torch.bfloat16),
        )

    def _output_declarations(self):
        return (
            ("grad_x", (self._tokens, 4, self._d), torch.bfloat16),
            ("grad_kv", (self._tokens, 5 * self._d), torch.bfloat16),
            ("grad_weight", (4, self._d), torch.float32),
        )

    def check_support(self):
        self._check_common()
        self._value_error_if(self._tokens % 64 != 0, "saved backward requires tokens divisible by 64")
        self._value_error_if(self._tokens > 8192, "saved backward currently supports at most 8192 tokens")
        _reductions()
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        self._compiled_kernel = _compile_backward(self._tokens, self._d, self._device.index, self._capability)

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        return self._workspace_bytes

    def allocate_workspace(self):
        """Allocate before capture; execute only binds this caller-owned storage."""
        return torch.empty(self.scratch_workspace_bytes(), dtype=torch.uint8, device=self._device)

    def execute(self, x, kv, weight, saved, grad_out, grad_x, grad_kv, grad_weight, workspace, *, current_stream=None):
        if workspace is None:
            raise ValueError("workspace is required")
        self._validate(
            dict(x=x, kv=kv, weight=weight, saved=saved, grad_out=grad_out), dict(grad_x=grad_x, grad_kv=grad_kv, grad_weight=grad_weight), workspace
        )
        apply, _, _, moments, reduce = self._compiled_kernel
        n, d = self._tokens, self._d
        coef = workspace.data_ptr()
        partial = coef + n * 4 * 4 * 4
        splits = n // 64
        bs = 1 << (splits - 1).bit_length()
        with torch.cuda.device(self._device):
            stream = self._stream(current_stream)
            moments(kv.data_ptr() + 4 * d * 2, grad_out, saved, coef, 5 * d, d, 8192, stream=int(stream))
            pointers = (
                _ptr(torch.bfloat16, x.data_ptr()),
                _ptr(torch.bfloat16, kv.data_ptr()),
                _ptr(torch.float32, weight.data_ptr()),
                _ptr(torch.bfloat16, grad_out.data_ptr()),
                _ptr(torch.float32, coef),
                _ptr(torch.bfloat16, grad_x.data_ptr()),
                _ptr(torch.bfloat16, grad_kv.data_ptr()),
                _ptr(torch.bfloat16, grad_kv.data_ptr() + 4 * d * 2),
                _ptr(torch.float32, partial),
            )
            apply(*pointers, cutlass.Int32(n), stream)
            reduce(partial, grad_weight, splits, d, bs, 128, stream=int(stream))
        return TupleDict(grad_x=grad_x, grad_kv=grad_kv, grad_weight=grad_weight)


def engram_gate_saved_forward(x, kv, weight, token_mask, *, eps=1e-20, backend="frost", current_stream=None):
    """Allocating convenience wrapper; keep returned saved state for backward."""
    plan = EngramGateSavedForward(x, kv, weight, token_mask, eps=eps, backend=backend)
    plan.compile()
    with (
        torch.cuda.device(x.device),
        torch.cuda.stream(
            torch.cuda.ExternalStream(int(current_stream), device=x.device) if current_stream is not None else torch.cuda.current_stream(x.device)
        ),
    ):
        out = torch.empty_like(x)
        saved = torch.empty((x.shape[0], 4, 4), dtype=torch.float32, device=x.device)
        return plan.execute(x, kv, weight, token_mask, out, saved, current_stream=current_stream)


def engram_gate_saved_backward(x, kv, weight, saved, grad_out, *, backend="frost", current_stream=None):
    """Allocating convenience wrapper returning dX, packed dKV and dWeight."""
    plan = EngramGateSavedBackward(x, kv, weight, saved, grad_out, backend=backend)
    plan.compile()
    with (
        torch.cuda.device(x.device),
        torch.cuda.stream(
            torch.cuda.ExternalStream(int(current_stream), device=x.device) if current_stream is not None else torch.cuda.current_stream(x.device)
        ),
    ):
        grad_x, grad_kv, grad_weight = torch.empty_like(x), torch.empty_like(kv), torch.empty_like(weight)
        workspace = plan.allocate_workspace()
        return plan.execute(x, kv, weight, saved, grad_out, grad_x, grad_kv, grad_weight, workspace, current_stream=current_stream)
