# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit Kimi CP affine merge over caller-gathered FP32 summaries."""

from dataclasses import replace
from functools import lru_cache

import torch

from cudnn._torch_stream import as_torch_stream
from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from cudnn.tensor_adapter import get_compute_capability

__all__ = ["KimiCPMerge", "kimi_cp_merge"]


def _compact(shape, stride):
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size != 1 and step != expected:
            return False
        expected *= size
    return True


def _tensor_span(name, tensor, shape, device):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.data_ptr() % 16
    ):
        raise ValueError(f"{name} requires compact, 16-byte-aligned CUDA FP32 shape {shape} on {device}")
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise ValueError("KimiCPMerge has no autograd; merge backward summaries explicitly with reverse=True")
    return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * 4, name


def _check_disjoint(spans):
    ordered = sorted(spans)
    for a, b in zip(ordered, ordered[1:]):
        if a[1] > b[0]:
            raise ValueError(f"merge operands {a[2]} and {b[2]} overlap")


class KimiCPMerge(APIBase):
    """Prepared, allocation-free merge. No NCCL or summary computation is hidden.

    H: [pieces, heads, V, K], M: [pieces, heads, K, K]. Return every piece's
    incoming state and the chain's final state. In forward order:
    X[j] = S; S = S @ M[j] + H[j]. Reverse walks j=P-1..0 and uses M[j].T.
    All storage is compact CUDA FP32; K is 64/128, V is 64/128/256. Optional transition
    receives the composed transition in the same walk order. No autograd is
    registered: CP integrations explicitly merge backward G/M with reverse=True.

    The merge uses the existing IEEE FP32 FMA state chain; it does not consult
    ambient Torch matmul precision settings. Supported targets are SM100/SM103
    with CuTeDSL >= 4.7.0 and SM107 with a DSL build that contains the sm_107a
    target (first available in the public 4.8.0 wheel). The cudnn-frontend
    package requires CuTeDSL >= 4.8.0. All input/output buffers must be disjoint.
    initial_states has H's shape; final_state/initial_state are [1, heads, V, K];
    transition is [1, heads, K, K]. Optional initial_state defaults to zeros;
    absent transition disables product output. No device storage is plan-owned.
    Direction is fixed at compile time. Pass already gathered FP32 summaries:
    no implicit conversion, communication or fallback.
    The caller orders summary/seed production before the launch stream and
    prevents concurrent writes or overlapping executions on the same buffers.
    """

    def __init__(self, h, m, initial_states, final_state, initial_state=None, *, reverse=False, transition=None):
        super().__init__()
        self._warn_experimental_api()
        self.reverse = reverse
        declarations = dict(h=h, m=m, initial_states=initial_states, final_state=final_state, initial_state=initial_state, transition=transition)
        self._descriptors = {}
        for name, tensor in declarations.items():
            if tensor is None and name in ("initial_state", "transition"):
                self._descriptors[name] = None
                continue
            if not isinstance(tensor, (torch.Tensor, TensorDesc)):
                raise TypeError(f"{name} must be a torch tensor or TensorDesc")
            desc = self._make_tensor_desc(tensor, name=name)
            device = torch.device(desc.device.type, desc.device.index)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self._descriptors[name] = replace(desc, device=device)
        self._requires_grad = any(getattr(tensor, "requires_grad", False) for tensor in declarations.values() if tensor is not None)

    def check_support(self):
        self._is_supported = False
        if not isinstance(self.reverse, bool):
            raise ValueError("reverse must be a bool")
        h = self._descriptors["h"]
        if h is None or len(h.shape) != 4:
            raise ValueError("h must have shape [pieces, heads, V, K]")
        p, heads, v, k = h.shape
        if p < 1 or heads < 1 or k not in (64, 128) or v not in (64, 128, 256):
            raise ValueError("merge requires nonempty pieces/heads, K in (64, 128), V in (64, 128, 256)")
        if p > 2**31 - 1 or heads > 65535:
            raise NotImplementedError("merge requires pieces <= INT32_MAX and heads <= the CUDA grid-y limit (65535)")
        if torch.is_grad_enabled() and self._requires_grad:
            raise ValueError("KimiCPMerge has no autograd; merge backward summaries explicitly with reverse=True")
        expected = dict(
            h=h.shape, m=(p, heads, k, k), initial_states=h.shape, final_state=(1, heads, v, k), initial_state=(1, heads, v, k), transition=(1, heads, k, k)
        )
        for name, desc in self._descriptors.items():
            if desc is None:
                if name in ("initial_state", "transition"):
                    continue
                raise ValueError(f"{name} is required")
            if desc.shape != expected[name] or desc.dtype != torch.float32 or not _compact(desc.shape, desc.stride):
                raise ValueError(f"{name} requires compact FP32 shape {expected[name]}")
            if desc.device.type != "cuda" or desc.device != h.device:
                raise ValueError("merge operands must be on one CUDA device")
        error = cutedsl_requirement_error("Kimi CP merge")
        if error:
            raise NotImplementedError(error)
        cc = get_compute_capability(h.device.index)
        if cc not in ((10, 0), (10, 3), (10, 7)):
            raise NotImplementedError("Kimi CP merge currently supports FROST SM100/SM103/SM107 targets")
        error = cutedsl_arch_requirement_error(cc)
        if error:
            raise NotImplementedError(error)
        self._device = h.device.index
        self._shape = h.shape
        self._is_supported = True
        return True

    def compile(self):
        """Validate metadata and cache a kernel without allocating device storage."""
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            from .frost.common.piece_chain import build_state_chain

            p, heads, v, k = self._shape
            with torch.cuda.device(self._device):
                self._compiled_kernel = build_state_chain(
                    heads_out=heads,
                    dim_v=v,
                    dim_k=k,
                    pieces=p,
                    rows_per_cta=16 if (v == 256 and k == 128) else v // 8,
                    transpose=self.reverse,
                    has_seed=self._descriptors["initial_state"] is not None,
                    has_tail=True,
                    emit_summary=self._descriptors["transition"] is not None,
                    device=self._device,
                    opt_level=3,
                )
        return self

    def scratch_workspace_bytes(self):
        """Return zero: the prepared merge owns no storage and needs no workspace."""
        self._ensure_support_checked()
        return 0

    def execute(self, h, m, initial_states, final_state, initial_state=None, *, transition=None, current_stream=None):
        """Overwrite preallocated outputs with the compiled merge; allocate/sync nothing.

        Buffers must match the compiled CUDA FP32 shapes and optional slots,
        remain compact, 16-byte aligned and mutually disjoint. Unsupported or
        changed bindings raise before launch. current_stream accepts a Torch
        stream or raw CUDA handle, defaulting to the operand device's current
        stream. Buffers and current_stream must stay on the device declared at
        compilation; construct a separate plan for each device. No
        producer-stream wait is inserted: callers must order every input write
        before current_stream and keep inputs/outputs free of concurrent writes
        or overlapping executions until that stream completes.
        Recording storage on the consumer stream protects allocator lifetime
        only; it does not establish data dependencies. The caller's current
        device is restored. Warm calls support CUDA Graph capture.
        """
        if self._compiled_kernel is None:
            raise RuntimeError("Call compile() before execute()")
        if self.reverse != self._compiled_kernel.transpose:
            raise ValueError("reverse is fixed at compilation; construct a new KimiCPMerge plan")
        tensors = dict(h=h, m=m, initial_states=initial_states, final_state=final_state, initial_state=initial_state, transition=transition)
        spans = []
        for name, tensor in tensors.items():
            desc = self._descriptors[name]
            if desc is None:
                if tensor is not None:
                    raise ValueError(f"{name} was not compiled")
                continue
            spans.append(_tensor_span(name, tensor, desc.shape, desc.device))
        _check_disjoint(spans)
        from .frost.common.piece_chain import run_state_chain

        with torch.cuda.device(self._device):
            stream = torch.cuda.current_stream(self._device) if current_stream is None else as_torch_stream(current_stream, self._device)
            if stream.device.index != self._device:
                raise ValueError("merge stream and operands must use the same device")
            for tensor in tensors.values():
                if tensor is not None:
                    tensor.record_stream(stream)
            run_state_chain(self._compiled_kernel, 1, h, m, initial_states, initial_state, final_state, transition, stream.cuda_stream)


@lru_cache(maxsize=128)
def _plan(shape, device, has_seed, reverse, output_transition):
    p, heads, v, k = shape

    def desc(s):
        strides = []
        stride = 1
        for size in reversed(s):
            strides.append(stride)
            stride *= size
        return TensorDesc(torch.float32, s, tuple(reversed(strides)), (3, 2, 1, 0), device)

    plan = KimiCPMerge(
        desc(shape),
        desc((p, heads, k, k)),
        desc(shape),
        desc((1, heads, v, k)),
        desc((1, heads, v, k)) if has_seed else None,
        reverse=reverse,
        transition=desc((1, heads, k, k)) if output_transition else None,
    )
    return plan.compile()


def kimi_cp_merge(h, m, initial_state=None, *, reverse=False, output_transition=False, current_stream=None):
    """Allocate merge outputs; return TupleDict(initial_states, final_state, transition).

    Caller performs all-gather and supplies FP32 H/M in rank/piece order. The
    merge uses the existing IEEE FP32 FMA state chain for its multiplication,
    running state/product, additive H and outputs.
    Inputs use the compact CUDA FP32 row-state layout described by KimiCPMerge;
    return shapes are [pieces, heads, V, K], [1, heads, V, K] and optional
    [1, heads, K, K]. This convenience wrapper allocates on current_stream;
    the prepared class is the allocation-free entry. Autograd-enabled inputs,
    overlapping inputs and dtype/layout/direction mismatches raise before
    compilation or output allocation. No summary generation is performed.
    No producer-stream wait is inserted; the caller orders H/M/seed production
    before current_stream and serializes access to all merge buffers until the
    launch completes. Internal record_stream calls protect allocator lifetime,
    not cross-stream data visibility.
    """
    if not isinstance(h, torch.Tensor) or not h.is_cuda or h.ndim != 4:
        raise ValueError("h must be a CUDA tensor of shape [pieces, heads, V, K]")
    if not isinstance(reverse, bool) or not isinstance(output_transition, bool):
        raise ValueError("reverse and output_transition must be bools")
    p, heads, v, k = h.shape
    spans = [_tensor_span("h", h, tuple(h.shape), h.device), _tensor_span("m", m, (p, heads, k, k), h.device)]
    if initial_state is not None:
        spans.append(_tensor_span("initial_state", initial_state, (1, heads, v, k), h.device))
    _check_disjoint(spans)
    with torch.cuda.device(h.device):
        stream = torch.cuda.current_stream(h.device) if current_stream is None else as_torch_stream(current_stream, h.device)
        if stream.device != h.device:
            raise ValueError("merge stream and operands must use the same device")
        with torch.cuda.stream(stream):
            plan = _plan(tuple(h.shape), h.device, initial_state is not None, reverse, output_transition)
            p, heads, v, k = h.shape
            states = torch.empty((p, heads, v, k), device=h.device, dtype=torch.float32)
            tail = torch.empty((1, heads, v, k), device=h.device, dtype=torch.float32)
            transition = torch.empty((1, heads, k, k), device=h.device, dtype=torch.float32) if output_transition else None
            plan.execute(h, m, states, tail, initial_state, transition=transition, current_stream=stream)
    return TupleDict(initial_states=states, final_state=tail, transition=transition)
