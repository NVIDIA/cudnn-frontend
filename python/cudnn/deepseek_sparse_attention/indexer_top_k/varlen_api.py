# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared, indices-only BF16 variable-length Top-K with caller-owned output."""

from __future__ import annotations

from dataclasses import replace
import re

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state, cutedsl_too_old


def _require_dsl(capability=None):
    installed, version = cutedsl_state()
    if not installed:
        raise ImportError("IndexerTopKVarlen requires nvidia-cutlass-dsl >= 4.7.0")
    if cutedsl_too_old(version):
        raise RuntimeError(cutedsl_requirement_error("IndexerTopKVarlen"))
    if capability == (10, 7):
        if version is not None and version[0] == "nvidia-cutlass-dsl":
            release = tuple(int(x) for x in re.findall(r"\d+", version[1].split("+", 1)[0])[:2])
            if release < (4, 8):
                raise RuntimeError(f"IndexerTopKVarlen on SM107 requires nvidia-cutlass-dsl >= 4.8.0; found {version[1]}")
        error = cutedsl_arch_requirement_error(capability)
        if error:
            raise RuntimeError(error)


# Rule 7: gate before APIBase's DSL import; private kernel import is later,
# after the actual operand device's architecture gate in check_support().
_require_dsl()

import torch

from cudnn._torch_stream import as_torch_stream, stream_context
from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.compiled_cache import positional_entry


def _device(value):
    device = torch.device(value.type, value.index)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _scalars(top_k, next_n, compress_ratio):
    if any(type(value) is not int for value in (top_k, next_n, compress_ratio)):
        raise TypeError("top_k, next_n and compress_ratio must be Python integers")
    if top_k not in (512, 1024, 2048):
        raise ValueError("top_k must be 512, 1024 or 2048")
    if not 1 <= next_n <= 512 or not 1 <= compress_ratio <= 2**31 - 1:
        raise ValueError("next_n must be 1..512 and compress_ratio must be a positive Int32")


def _span(tensor, name):
    """Validate live storage without a read, conversion, descriptor or allocation."""
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError(f"{name} cannot have an unresolved negative or conjugate flag")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must use CUDA storage")
    pointer = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    offset = tensor.storage_offset() * tensor.element_size()
    storage = tensor.untyped_storage()
    if pointer <= 0 or pointer % 16:
        raise ValueError(f"{name} must have a nonzero 16-byte aligned address")
    if offset < 0 or offset + size > storage.nbytes() or pointer != storage.data_ptr() + offset:
        raise ValueError(f"{name} logical extent exceeds its observed storage")
    if pointer + size > 2**64 - 1:
        raise ValueError(f"{name} address extent overflows UInt64")
    return pointer, pointer + size


def _descriptor_metadata(value, *, name, shape, dtype, device):
    if tuple(value.shape) != shape or value.dtype != dtype or _device(value.device) != device:
        raise ValueError(f"{name} must be {dtype} {shape} on {device}")
    if not value.is_contiguous():
        strides = value.stride if isinstance(value, TensorDesc) else value.stride()
        raise ValueError(f"{name} requires contiguous storage; received strides {strides}")


class IndexerTopKVarlen(APIBase):
    """Prepared BF16 Top-K on SM103/SM107, producing Int32 indices only.

    ``input_values`` is contiguous ``[T,N]`` and ``seq_lens`` is Int32
    ``[T/next_n]``. Row r sees a prefix of length
    ``min(N, max(0, floor((seq_lens[r//next_n] - next_n + 1 + r%next_n)
    / compress_ratio)))`` using signed Int64 arithmetic. Select up to K
    largest values, followed by -1 padding. Ordering and tied-index choice
    are unspecified. NaNs within an eligible prefix are unsupported; scores
    outside the effective prefix are ignored, including an entirely empty row.

    This plan fixes tensor shapes/device and scalar configuration. Compiled
    code is shared across plans with different T. compile() is metadata-only;
    execute() takes a caller-owned contiguous Int32 [T,K] destination, never
    allocates GPU memory, and launches one kernel. Call compile() before capture.
    """

    def __init__(self, sample_input_values, sample_seq_lens, top_k, next_n=1, compress_ratio=1):
        _scalars(top_k, next_n, compress_ratio)
        if not all(isinstance(t, (torch.Tensor, TensorDesc)) for t in (sample_input_values, sample_seq_lens)):
            raise TypeError("samples must be torch tensors or TensorDesc metadata")
        super().__init__()
        self._warn_experimental_api()
        self._configuration = (top_k, next_n, compress_ratio)
        self._device = _device(sample_input_values.device)
        self._descs = tuple(
            replace(self._make_tensor_desc(t, name=name), device=_device(t.device))
            for t, name in ((sample_input_values, "input_values"), (sample_seq_lens, "seq_lens"))
        )
        self._validate_inputs(self._descs)
        # Samples establish metadata only; no tensors, views or pointers survive.
        for tensor, name in ((sample_input_values, "input_values"), (sample_seq_lens, "seq_lens")):
            if isinstance(tensor, torch.Tensor):
                _span(tensor, name)
        self._entry = None

    def _validate_inputs(self, tensors):
        values, lengths = tensors
        top_k, next_n, _ = self._configuration
        if values.ndim != 2:
            raise ValueError("input_values must have shape [T,N]")
        rows, cols = self._descs[0].shape
        if any(type(x) is not int for x in (rows, cols)) or not 1 <= rows <= 512 or not 1 <= cols <= 262144 or rows % next_n:
            raise ValueError("Require T1..512, N1..262144 and next_n dividing T")
        if self._device.type != "cuda":
            raise ValueError("IndexerTopKVarlen requires a CUDA device")
        _descriptor_metadata(values, name="input_values", shape=(rows, cols), dtype=torch.bfloat16, device=self._device)
        _descriptor_metadata(lengths, name="seq_lens", shape=(rows // next_n,), dtype=torch.int32, device=self._device)
        return rows, cols, top_k

    def check_support(self):
        self._validate_inputs(self._descs)
        self._capability = tuple(torch.cuda.get_device_capability(self._device))
        if self._capability not in ((10, 3), (10, 7)):
            raise NotImplementedError(f"IndexerTopKVarlen supports SM103 and SM107; found {self._capability} on {self._device}")
        _require_dsl(self._capability)
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Compile IndexerTopKVarlen before CUDA graph capture")
            from .varlen_kernel import compile_topk

            _, cols = self._descs[0].shape
            artifact = compile_topk(cols, *self._configuration, self._device.index, f"sm_{self._capability[0]}{self._capability[1]}a")
            entry = positional_entry(artifact)
            if entry is None:
                raise NotImplementedError(f"Installed CuTe DSL produced no executable entry for {self._device} / SM{self._capability}")
            self._compiled_kernel, self._entry = artifact, entry

    def scratch_workspace_bytes(self):
        return 0

    def execute(self, input_values, seq_lens, indices, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        tensors = (input_values, seq_lens, indices)
        if any(not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("execute() requires torch tensors")
        rows, _, top_k = self._validate_inputs(tensors[:2])
        _descriptor_metadata(indices, name="indices", shape=(rows, top_k), dtype=torch.int32, device=self._device)
        spans = tuple(_span(t, name) for t, name in zip(tensors, ("input_values", "seq_lens", "indices")))
        low, high = spans[2]
        if any(low < end and start < high for start, end in spans[:2]):
            raise ValueError("indices must not overlap input_values or seq_lens")
        with torch.cuda.device(self._device), stream_context(current_stream, self._device):
            launch_stream = torch.cuda.current_stream(self._device)
            self._entry(*(span[0] for span in spans), rows, launch_stream.cuda_stream)
            # Foreign launches are invisible to Torch's dispatcher. Preserve each
            # caller allocation through pending side-stream consumers, without
            # retaining tensors in the plan or owning CUDA memory.
            for tensor in tensors:
                tensor.record_stream(launch_stream)
        return None


_wrapper_plans = {}


def indexer_top_k_varlen_wrapper(input_values, seq_lens, top_k, next_n=1, compress_ratio=1, stream=None):
    """Allocate Int32 indices and return ``TupleDict(indices=...)``; prepare before capture."""
    _scalars(top_k, next_n, compress_ratio)
    if not all(isinstance(t, torch.Tensor) for t in (input_values, seq_lens)):
        raise TypeError("input_values and seq_lens must be torch tensors")
    key = (top_k, next_n, compress_ratio, tuple((str(t.device), str(t.dtype), tuple(t.shape), tuple(t.stride())) for t in (input_values, seq_lens)))
    plan = _wrapper_plans.get(key)
    if plan is None:
        plan = IndexerTopKVarlen(input_values, seq_lens, top_k, next_n, compress_ratio)
        plan.compile()
        _wrapper_plans[key] = plan
    plan._validate_inputs((input_values, seq_lens))
    for tensor, name in ((input_values, "input_values"), (seq_lens, "seq_lens")):
        _span(tensor, name)
    with torch.cuda.device(plan._device):
        launch_stream = torch.cuda.current_stream(plan._device) if stream is None else as_torch_stream(stream, plan._device)
        with stream_context(launch_stream, plan._device):
            indices = torch.empty((input_values.shape[0], top_k), dtype=torch.int32, device=plan._device)
            plan.execute(input_values, seq_lens, indices, current_stream=launch_stream)
    return TupleDict(indices=indices)
