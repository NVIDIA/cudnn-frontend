# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4.1 vision RoPE backward with native attention-gradient layouts."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

_INPUTS = ("grad_q", "grad_k", "grad_v", "cosine", "sine")


def _require_dsl():
    if cutedsl_too_old(cutedsl_state()[1]):
        raise RuntimeError(cutedsl_requirement_error("FROST vision RoPE backward"))


@lru_cache(maxsize=16)
def _compile(device_index, capability):
    _require_dsl()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from .kernel import launch

    n = cute.sym_int(64)
    htd = make_fake_tensor(cutlass.BFloat16, (n, 16, 64), (64, n * 64, 1), assumed_align=16)
    hdt = make_fake_tensor(cutlass.BFloat16, (n, 16, 64), (1, n * 64, n), assumed_align=16)
    table = make_fake_tensor(cutlass.Float32, (n, 1, 32), (32, 32, 1), assumed_align=16)
    out = make_fake_tensor(cutlass.BFloat16, (n, 3072), (3072, 1), assumed_align=16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    with torch.cuda.device(device_index):
        return cute.compile(launch, htd, hdt, htd, table, table, out, stream, cutlass.Int64(0), options="--enable-tvm-ffi")


class VisionRoPEBackward(APIBase):
    """Fuse inverse Q/K rotation, native K-layout transpose, and packed QKV output.

    Inputs have logical shape ``(T, 16, 64)``. Q/V use HTD storage and K uses
    HDT storage; cosine/sine are constant FP32 ``(T, 1, 32)`` tables. The class
    writes caller-owned BF16 ``(T, 3072)`` output and supports changing T
    without recompilation. See the FE OSS API documentation for exact strides.
    """

    def __init__(self, grad_q, grad_k, grad_v, cosine, sine, *, backend="frost"):
        super().__init__()
        self.backend = backend
        tensors = (grad_q, grad_k, grad_v, cosine, sine)
        if not all(isinstance(t, (torch.Tensor, TensorDesc)) for t in tensors):
            raise TypeError("inputs must be torch tensors or TensorDesc metadata")
        self._descs = tuple(self._make_tensor_desc(t, name=name) for t, name in zip(tensors, _INPUTS))
        self._device = torch.device(self._descs[0].device.type, self._descs[0].device.index)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._initial_alignment = all(isinstance(t, TensorDesc) or t.data_ptr() % 16 == 0 for t in tensors)

    def _metadata(self, tensors, *, descriptors=False):
        n = tensors[0].shape[0] if tensors[0].ndim == 3 else 0
        if not isinstance(n, int) or not 0 < n <= 32 * (2**31 - 1):
            raise ValueError("T must be positive and ceil(T / 32) must fit the CUDA x grid")
        shapes = ((n, 16, 64),) * 3 + ((n, 1, 32),) * 2
        strides = ((64, n * 64, 1), (1, n * 64, n), (64, n * 64, 1), (32, 32, 1), (32, 32, 1))
        dtypes = (torch.bfloat16,) * 3 + (torch.float32,) * 2
        for name, tensor, shape, stride, dtype in zip(_INPUTS, tensors, shapes, strides, dtypes):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            actual_stride = tensor.stride if descriptors else tensor.stride()
            if tuple(actual_stride) != stride:
                raise ValueError(f"{name} requires native strides {stride}, got {tuple(actual_stride)}")
            device = torch.device(tensor.device.type, tensor.device.index)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device.type != "cuda" or device != self._device:
                raise ValueError(f"{name} must be on the planned CUDA device {self._device}")
        return n

    def check_support(self):
        _require_dsl()
        if self.backend != "frost":
            raise ValueError("backend must be 'frost'")
        self._metadata(self._descs, descriptors=True)
        if not self._initial_alignment:
            raise ValueError("all input addresses must be 16-byte aligned")
        self._capability = torch.cuda.get_device_capability(self._device)
        if self._capability != (10, 0):
            raise NotImplementedError("FROST vision RoPE backward currently supports SM100")
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        self._compiled_kernel = _compile(self._device.index, self._capability)

    def scratch_workspace_bytes(self):
        return 0

    def execute(self, grad_q, grad_k, grad_v, cosine, sine, grad_qkv, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        inputs = (grad_q, grad_k, grad_v, cosine, sine)
        if not all(isinstance(t, torch.Tensor) for t in (*inputs, grad_qkv)):
            raise TypeError("execute() requires torch tensors")
        n = self._metadata(inputs)
        if tuple(grad_qkv.shape) != (n, 3072) or grad_qkv.dtype != torch.bfloat16 or grad_qkv.device != self._device or grad_qkv.stride() != (3072, 1):
            raise ValueError("grad_qkv must be contiguous BF16 (T, 3072) on the planned CUDA device")
        if any(t.data_ptr() % 16 for t in (*inputs, grad_qkv)):
            raise ValueError("all runtime addresses must be 16-byte aligned")
        low, high = grad_qkv.data_ptr(), grad_qkv.data_ptr() + grad_qkv.numel() * grad_qkv.element_size()
        for tensor in inputs:
            start = tensor.data_ptr()
            if low < start + tensor.numel() * tensor.element_size() and start < high:
                raise ValueError("grad_qkv must not overlap any input")
        with torch.cuda.device(self._device):
            stream = cuda.CUstream(int(current_stream)) if current_stream is not None else self._get_default_stream(None)
            self._compiled_kernel(*inputs, grad_qkv, stream, n)
        return grad_qkv


def vision_rope_backward_wrapper(grad_q, grad_k, grad_v, cosine, sine, *, backend="frost", current_stream=None):
    """Allocate packed dQKV; use VisionRoPEBackward for allocation-free replay."""
    plan = VisionRoPEBackward(grad_q, grad_k, grad_v, cosine, sine, backend=backend)
    plan.compile()
    with torch.cuda.device(grad_q.device):
        stream = (
            torch.cuda.get_stream_from_external(int(current_stream), grad_q.device) if current_stream is not None else torch.cuda.current_stream(grad_q.device)
        )
        with torch.cuda.stream(stream):
            grad_qkv = torch.empty((grad_q.shape[0], 3072), dtype=torch.bfloat16, device=grad_q.device)
            plan.execute(grad_q, grad_k, grad_v, cosine, sine, grad_qkv, current_stream=current_stream)
    return TupleDict(grad_qkv=grad_qkv)
