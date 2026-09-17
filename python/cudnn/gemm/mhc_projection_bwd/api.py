# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared DSv4.1 mHC projection/RMS backward, with explicit TF32 permission."""

from functools import lru_cache
import importlib.metadata

import torch

from cudnn.api_base import APIBase, TupleDict

_M, _N, _K, _SPLITS = 4096, 24, 20480, 8
_WORKSPACE_BYTES = _SPLITS * _N * _K * 4
_PORTS = {
    "x": ((_M, _K), (_K, 1), torch.bfloat16),
    "weight": ((_N, _K), (_K, 1), torch.float32),
    "grad_proj": ((_M, 32), (32, 1), torch.float32),
    "grad_r": ((_M, 1), (1, 1), torch.float32),
    "r": ((_M, 1), (1, 1), torch.float32),
    "dx": ((_M, _K), (_K, 1), torch.bfloat16),
    "dweight": ((_N, _K), (_K, 1), torch.float32),
}


def _dependencies():
    try:
        version = importlib.metadata.version("cuda-tile")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("mHC projection backward requires cuda-tile>=1.5 and the cuDNN Frontend [cutile,triton] extras") from exc
    # cuda-tile currently uses numeric major.minor.patch releases. Compare the
    # release portion only, so local version suffixes do not affect the gate.
    parts = version.split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts[:2]) or tuple(map(int, parts[:2])) < (1, 5):
        raise ImportError(f"mHC projection backward requires cuda-tile>=1.5; found {version}")
    try:
        import cuda.tile.compilation
        import triton
    except ImportError as exc:
        raise ImportError(f"mHC projection backward requires the cuDNN Frontend [cutile,triton] extras and a system tileiras: {exc}") from exc


def _device_index(device):
    if device.type != "cuda":
        raise ValueError("mHC projection backward requires CUDA tensors")
    return torch.cuda.current_device() if device.index is None else device.index


class MhcProjectionBackward(APIBase):
    """Compute the projection/RMS stage's dX and dW for the DSv4.1 profile.

    ``dx = grad_proj[:, :24] @ weight + grad_r / (r * 20480) * x``
    ``dweight = grad_proj[:, :24].T @ x``

    Matrix operands use TF32 conversion and FP32 accumulation. dX is BF16;
    dW is FP32. This is one stage of mHC backward, not the whole mHC gradient.
    ``r`` contains positive forward RMS values. The last eight grad_proj
    columns are ignored. All tensor operands must be disjoint and 16-byte
    aligned. ``execute`` overwrites both outputs, and requires caller-owned
    workspace; it does not accumulate into dweight.
    """

    def __init__(self, x, weight, grad_proj, grad_r, r, dx, dweight, *, allow_tf32=False, backend="frost"):
        super().__init__()
        self._allow_tf32 = allow_tf32
        self._backend = backend
        tensors = dict(x=x, weight=weight, grad_proj=grad_proj, grad_r=grad_r, r=r, dx=dx, dweight=dweight)
        self._descriptors = {name: self._make_tensor_desc(tensor, name=name) for name, tensor in tensors.items()}
        self._device = None

    def check_support(self):
        self._is_supported = False
        if self._backend != "frost":
            raise ValueError("mHC projection backward supports backend='frost'")
        if self._allow_tf32 is not True:
            raise NotImplementedError("mHC projection backward requires explicit allow_tf32=True")
        device = None
        for name, (shape, stride, dtype) in _PORTS.items():
            desc = self._descriptors[name]
            if desc is None or desc.shape != shape or desc.stride != stride or desc.dtype != dtype:
                raise NotImplementedError(f"mHC projection backward: {name} requires shape={shape}, stride={stride}, dtype={dtype}")
            index = _device_index(desc.device)
            if device is not None and index != device:
                raise ValueError("mHC projection backward requires all operands on the same device")
            device = index
        if torch.cuda.get_device_capability(device) != (10, 0):
            raise NotImplementedError("mHC projection backward currently supports SM100")
        _dependencies()
        self._device = device
        self._is_supported = True
        return True

    def scratch_workspace_bytes(self):
        return _WORKSPACE_BYTES

    def compile(self):
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            from ._compile import compiled_projection

            with torch.cuda.device(self._device):
                self._compiled_kernel = compiled_projection(self._device)
        return self

    def execute(self, x, weight, grad_proj, grad_r, r, dx, dweight, workspace, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("mHC projection backward: call compile() before execute()")
        tensors = dict(x=x, weight=weight, grad_proj=grad_proj, grad_r=grad_r, r=r, dx=dx, dweight=dweight)
        ranges = []
        for name, tensor in tensors.items():
            shape, stride, dtype = _PORTS[name]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or tensor.device.index != self._device:
                raise ValueError(f"mHC projection backward: {name} must be a CUDA tensor on device {self._device}")
            if tuple(tensor.shape) != shape or tensor.stride() != stride or tensor.dtype != dtype:
                raise ValueError(f"mHC projection backward: {name} does not match the compiled descriptor")
            pointer = tensor.data_ptr()
            if pointer % 16:
                raise ValueError(f"mHC projection backward: {name} must be 16-byte aligned")
            ranges.append((pointer, pointer + tensor.numel() * tensor.element_size(), name))
        if (
            not isinstance(workspace, torch.Tensor)
            or not workspace.is_cuda
            or workspace.device.index != self._device
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < _WORKSPACE_BYTES
            or workspace.data_ptr() % 16
        ):
            raise ValueError(f"mHC projection backward: workspace requires at least {_WORKSPACE_BYTES} contiguous, 16-byte-aligned CUDA uint8 bytes")
        ranges.append((workspace.data_ptr(), workspace.data_ptr() + _WORKSPACE_BYTES, "workspace"))
        ordered = sorted(ranges)
        for left, right in zip(ordered, ordered[1:]):
            if left[1] > right[0]:
                raise ValueError(f"mHC projection backward: operands {left[2]} and {right[2]} overlap")
        if current_stream is None:
            stream = torch.cuda.current_stream(self._device).cuda_stream
        elif isinstance(current_stream, torch.cuda.Stream):
            if current_stream.device.index != self._device:
                raise ValueError("mHC projection backward: stream and tensors must use the same device")
            stream = current_stream.cuda_stream
        else:
            stream = int(current_stream)
        self._compiled_kernel.execute(x, weight, grad_proj, grad_r, r, dx, dweight, workspace, stream)


@lru_cache(maxsize=None)
def _wrapper_plan(device):
    from cudnn.api_base import TensorDesc

    descriptors = {
        name: TensorDesc(dtype, shape, stride, tuple(reversed(range(len(shape)))), torch.device("cuda", device), name=name)
        for name, (shape, stride, dtype) in _PORTS.items()
    }
    plan = MhcProjectionBackward(**descriptors, allow_tf32=True, backend="frost")
    plan.compile()
    return plan


def mhc_projection_backward(x, weight, grad_proj, grad_r, r, *, allow_tf32=False, backend="frost", current_stream=None):
    """Allocate dX/dW and scratch, then execute the prepared Frost stage.

    Return ``TupleDict(dx=..., dweight=...)``. Compile the class API and supply
    its buffers directly for allocation-free execution and CUDA Graph capture.
    """
    if backend != "frost":
        raise ValueError("mHC projection backward supports backend='frost'")
    if allow_tf32 is not True:
        raise NotImplementedError("mHC projection backward requires explicit allow_tf32=True")
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("mHC projection backward requires CUDA tensors")
    device = x.device.index
    with torch.cuda.device(device):
        if current_stream is None:
            stream = torch.cuda.current_stream(device)
        elif isinstance(current_stream, torch.cuda.Stream):
            stream = current_stream
            if stream.device.index != device:
                raise ValueError("mHC projection backward: stream and tensors must use the same device")
        else:
            stream = torch.cuda.ExternalStream(int(current_stream), device=device)
        with torch.cuda.stream(stream):
            plan = _wrapper_plan(device)
            dx = torch.empty((_M, _K), dtype=torch.bfloat16, device=device)
            dweight = torch.empty((_N, _K), dtype=torch.float32, device=device)
            workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
            plan.execute(x, weight, grad_proj, grad_r, r, dx, dweight, workspace, current_stream=stream)
    return TupleDict(dx=dx, dweight=dweight)
