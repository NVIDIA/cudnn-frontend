# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-geometry HCA backward with native rank-major KV and explicit scratch."""

from __future__ import annotations

import math
import threading

import cuda.bindings.driver as cuda
import torch

from cudnn.api_base import APIBase, TensorDesc, TupleDict

try:
    from ._plan import _HcaPlan
except ImportError as exc:
    raise ImportError("Aligned HCA requires torch and nvidia-cudnn-frontend[cutedsl,triton]") from exc


_SPECS = {
    "q": ((4096, 128, 512), (65536, 512, 1), torch.bfloat16),
    "kv": ((4752, 512), (512, 1), torch.bfloat16),
    "out": ((4096, 128, 512), (65536, 512, 1), torch.bfloat16),
    "dout": ((4096, 128, 512), (65536, 512, 1), torch.bfloat16),
    "lse": ((4096, 128), (128, 1), torch.float32),
    "attn_sink": ((128,), (1,), torch.float32),
    "dq": ((4096, 128, 512), (65536, 512, 1), torch.bfloat16),
    "dkv": ((4752, 512), (512, 1), torch.bfloat16),
    "d_sink": ((128,), (1,), torch.float32),
}


def _launch_stream(current_stream, device):
    if current_stream is None:
        return torch.cuda.current_stream(device)
    if isinstance(current_stream, torch.cuda.Stream):
        if current_stream.device != device:
            raise ValueError("The launch stream must belong to the tensor device")
        return current_stream
    if isinstance(current_stream, bool) or not isinstance(current_stream, (int, cuda.CUstream)):
        raise TypeError("current_stream must be a Torch CUDA stream, CUstream, integer handle, or None")
    handle = int(current_stream)
    if handle == 0:
        return torch.cuda.default_stream(device)
    if handle < 3:
        raise ValueError("Pass a CUDA stream object or concrete handle, not a legacy/PTDS sentinel")
    current = torch.cuda.current_stream(device)
    return current if current.cuda_stream == handle else torch.cuda.ExternalStream(handle, device=device)


class AlignedHCABackward(APIBase):
    """Aligned S65536/CP16/H128/D512 BF16 HCA backward on GB300.

    The declaration fixes window=128, compression=128, one unpadded sequence,
    contiguous CP partitioning and 33 physical compressed rows per rank.
    No arbitrary index tensor is accepted. See the FE OSS HCA documentation
    for physical row ownership and KV-only LSE semantics.

    Compile before capture, then supply every output and a sufficiently large
    uint8 workspace to execute. Keep the API and workspace alive for captured
    graphs. One API instance must not execute concurrently from host threads.
    """

    def __init__(
        self,
        sample_q,
        sample_kv,
        sample_out,
        sample_dout,
        sample_lse,
        sample_attn_sink,
        sample_dq,
        sample_dkv,
        sample_d_sink,
        *,
        cp_rank,
        softmax_scale=512**-0.5,
    ):
        super().__init__()
        samples = (sample_q, sample_kv, sample_out, sample_dout, sample_lse, sample_attn_sink, sample_dq, sample_dkv, sample_d_sink)
        if any(not isinstance(value, (torch.Tensor, TensorDesc)) for value in samples):
            raise TypeError("Aligned HCA accepts Torch tensors or TensorDesc metadata")
        self.descriptors = {name: self._make_tensor_desc(value, name=name) for name, value in zip(_SPECS, samples)}
        self.cp_rank = cp_rank
        self.softmax_scale = softmax_scale
        self._warn_experimental_api()

    def check_support(self):
        if type(self.cp_rank) is not int or not 0 <= self.cp_rank < 16:
            raise ValueError("cp_rank must be an integer in [0, 16)")
        if type(self.softmax_scale) not in (int, float) or not math.isfinite(self.softmax_scale):
            raise ValueError("softmax_scale must be a finite host scalar")
        device = self.descriptors["q"].device
        if not isinstance(device, torch.device) or device.type != "cuda" or device.index is None:
            raise ValueError("Descriptors must declare an indexed Torch CUDA device")
        for name, desc in self.descriptors.items():
            shape, stride, dtype = _SPECS[name]
            if desc.shape != shape or desc.stride != stride:
                raise NotImplementedError(f"{name}: unsupported shape {desc.shape} or strides {desc.stride}; expected {shape}, {stride}")
            self._check_dtype(desc, dtype, name=name)
            if desc.device != device:
                raise ValueError("All HCA tensors must be on the same device")
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise NotImplementedError("Aligned HCA backward currently supports GB300 (SM103) only")
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            plan = _HcaPlan(self.cp_rank, self.softmax_scale, self.descriptors["q"].device)
            self._compiled_kernel = plan.compile()

    def scratch_workspace_bytes(self):
        if self._compiled_kernel is None:
            raise RuntimeError("Compile before querying workspace size")
        return self._compiled_kernel.workspace_size

    def execute(self, q, kv, out, dout, lse, attn_sink, dq, dkv, d_sink, workspace, *, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("Compile AlignedHCABackward before execute")
        if any(not isinstance(t, torch.Tensor) for t in (q, kv, out, dout, lse, attn_sink, dq, dkv, d_sink, workspace)):
            raise TypeError("Execute requires Torch tensors")
        stream = _launch_stream(current_stream, self._compiled_kernel.device)
        result = self._compiled_kernel.execute(q, kv, out, dout, lse, attn_sink, dq, dkv, d_sink, workspace, stream=stream)
        return TupleDict(result)


# Per-host-thread plans avoid sharing a mutable cuDNN handle between callers.
# Retaining entries also retains compiled modules for wrapper-captured graphs.
_WRAPPER_APIS = {}


def aligned_hca_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, *, cp_rank, softmax_scale=512**-0.5, dq=None, dkv=None, d_sink=None, workspace=None, current_stream=None
):
    """Allocate missing outputs/scratch and return (dq, dkv, d_sink) as TupleDict.

    Warm this wrapper before CUDA graph capture for each metadata signature.
    Scratch allocated during capture belongs to that graph's private pool.
    Use AlignedHCABackward.execute for a caller-owned, allocation-free path.
    """
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("Q must be a CUDA Torch tensor")
    if type(cp_rank) is not int or not 0 <= cp_rank < 16:
        raise ValueError("cp_rank must be an integer in [0, 16)")
    if type(softmax_scale) not in (int, float) or not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be a finite host scalar")
    if any(not isinstance(t, torch.Tensor) for t in (kv, out, dout, lse, attn_sink)):
        raise TypeError("The wrapper requires Torch tensors")
    stream = _launch_stream(current_stream, q.device)
    with torch.cuda.device(q.device), torch.cuda.stream(stream):
        dq = torch.empty_like(q) if dq is None else dq
        dkv = torch.empty_like(kv) if dkv is None else dkv
        d_sink = torch.empty_like(attn_sink) if d_sink is None else d_sink
        tensors = (q, kv, out, dout, lse, attn_sink, dq, dkv, d_sink)
        if any(not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("The wrapper requires Torch tensors")
        key = (threading.get_ident(), cp_rank, float(softmax_scale), tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors))
        api = _WRAPPER_APIS.get(key)
        if api is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm aligned_hca_backward_wrapper before capture")
            api = AlignedHCABackward(*tensors, cp_rank=cp_rank, softmax_scale=softmax_scale)
            api.compile()
            _WRAPPER_APIS[key] = api
        if workspace is None:
            workspace = torch.empty(api.scratch_workspace_bytes(), device=q.device, dtype=torch.uint8)
        return api.execute(*tensors, workspace, current_stream=stream)
