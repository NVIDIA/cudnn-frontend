# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed 12:1 QSA from four-token groups, without a quadratic bitmap."""

from functools import lru_cache
import importlib
import math

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict, WorkspaceCarver, ws_align
from cudnn._torch_stream import stream_context
from cudnn.QSA._contract import check_tensor


def _validate(q, k, v, indices, scale, grad=None, out32=None, lse=None):
    if not isinstance(q, torch.Tensor) or not q.is_cuda or q.ndim != 3:
        raise ValueError("q must be a CUDA tensor with shape [T, 12, 256]")
    total = q.shape[0]
    if total < 1 or q.shape[1:] != (12, 256) or q.numel() > 2**31 - 1:
        raise NotImplementedError("Indexed QSA requires T > 0, 12Q/1KV, D256 and T * 12 * 256 <= INT32_MAX")
    for name, tensor, heads in (("q", q, 12), ("k", k, 1), ("v", v, 1)):
        check_tensor(name, tensor, (total, heads, 256), torch.bfloat16, q.device, 16)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite host scalar")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 2:
        raise ValueError("indices must have shape [T, W]")
    width = indices.shape[1]
    if width < 4 or width > 2052 or width % 4:
        raise NotImplementedError("Indexed QSA requires an index width divisible by four in [4, 2052]")
    check_tensor("indices", indices, (total, width), torch.int32, q.device)
    if grad is not None:
        check_tensor("grad_out", grad, q.shape, torch.bfloat16, q.device, 16)
        check_tensor("out_fp32", out32, q.shape, torch.float32, q.device, 16)
        check_tensor("lse", lse, (12, total), torch.float32, q.device)
    return total, width


class _QsaIndexedPlan(APIBase):
    """Immutable geometry and compiled kernels; execute uses caller storage only."""

    def __init__(self, total, width, device, direction):
        with torch.cuda.device(device):
            super().__init__()
        self.total, self.width, self.device, self.direction = total, width, device, direction
        self.nq = 1 if direction == "fprop" else 2
        groups = (total + self.nq - 1) // self.nq
        self.layouts = [
            ("ids", (groups, self.nq * (width // 4)), torch.int32),
            ("counts", (groups,), torch.int32),
            *((name, (total,), torch.int32) for name in ("lo", "hi", "first", "last")),
        ]
        if direction == "bprop":
            self.layouts.extend(
                [
                    ("delta", (total, 12), torch.float32),
                    ("dq_acc", (total, 12, 256), torch.float32),
                    ("dk_acc", (total, 1, 256), torch.float32),
                    ("dv_acc", (total, 1, 256), torch.float32),
                    ("flags", (((total * 256 + 1023) // 1024) * 4 + 1,), torch.int32),
                ]
            )
        self.workspace_bytes = sum(ws_align(math.prod(shape) * dtype.itemsize) for _, shape, dtype in self.layouts)

    def check_support(self):
        from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

        installed, version = cutedsl_state()
        if not installed or cutedsl_too_old(version):
            raise ImportError(
                cutedsl_requirement_error("Indexed QSA") or f"Indexed QSA requires nvidia-cutlass-dsl >= {'.'.join(str(x) for x in CUTEDSL_MIN_VERSION)}"
            )
        cc = torch.cuda.get_device_capability(self.device)
        if cc not in ((10, 0), (10, 3), (10, 7)):
            raise NotImplementedError(f"Indexed QSA requires SM100, SM103 or SM107, got SM{cc[0]}{cc[1]}")
        reason = cutedsl_arch_requirement_error(cc)
        if reason:
            raise NotImplementedError(reason)
        self.arch = cc[0] * 10 + cc[1]
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        import cutlass
        from . import _metadata

        with torch.cuda.device(self.device):
            module = f"{self.direction}_sm100"
            if self.direction == "fprop" and self.arch == 107 and self.total >= 32768:
                module = "fprop_n64_sm107"
            self.impl = importlib.import_module(f"cudnn.QSA.kernels.{module}")
            self._compiled_kernel = self.impl.compile_kernel(cutlass.BFloat16, 12, 1, self.arch)
            self._metadata = _metadata.compile_metadata(self.arch, self.nq, self.width // 4)

    def scratch_workspace_bytes(self):
        return self.workspace_bytes

    def execute(self, q, k, v, indices, outputs, scale, workspace, grad=None, out32=None, lse=None, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("Indexed QSA plan must be compiled before execute")
        if self.direction == "bprop" and grad is None:
            raise ValueError("grad_out, out_fp32 and lse are required for backward")
        if self.direction == "fprop" and any(value is not None for value in (grad, out32, lse)):
            raise ValueError("forward does not accept backward saved state")
        total, width = _validate(q, k, v, indices, scale, grad, out32, lse)
        if (total, width, q.device) != (self.total, self.width, self.device):
            raise ValueError("Inputs must match the compiled token count, index width and device")
        if len(outputs) != 3:
            raise ValueError("Indexed QSA requires three preallocated outputs")
        specifications = (
            ((q.shape, q.dtype), (q.shape, torch.float32), ((12, total), torch.float32))
            if self.direction == "fprop"
            else ((q.shape, q.dtype), (k.shape, k.dtype), (v.shape, v.dtype))
        )
        for index, (tensor, (shape, dtype)) in enumerate(zip(outputs, specifications)):
            check_tensor(f"output[{index}]", tensor, shape, dtype, q.device, 4 if self.direction == "fprop" and index == 2 else 16)
        if not isinstance(workspace, torch.Tensor) or workspace.device != q.device:
            raise ValueError(f"workspace must be a torch.Tensor on {q.device}")
        carver = WorkspaceCarver(workspace, self.workspace_bytes, "Indexed QSA")
        inputs = (q, k, v, indices) + (() if grad is None else (grad, out32, lse))
        span = lambda tensor: (tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
        overlaps = lambda a, b: a[0] < b[1] and b[0] < a[1]
        ws_span = (workspace.data_ptr(), workspace.data_ptr() + self.workspace_bytes)
        for tensor in (*inputs, *outputs):
            if overlaps(ws_span, span(tensor)):
                raise ValueError("workspace must not overlap an input or output")
        for index, tensor in enumerate(outputs):
            if any(overlaps(span(tensor), span(other)) for other in (*inputs, *outputs[:index])):
                raise ValueError("outputs must not overlap inputs or each other")
        scratch = {name: carver.take(math.prod(shape), dtype).view(shape) for name, shape, dtype in self.layouts}
        metadata = tuple(scratch[name] for name in ("ids", "counts", "lo", "hi", "first", "last"))
        with torch.cuda.device(q.device), stream_context(current_stream, q.device):
            stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
            self._metadata(indices, *metadata, stream)
            if self.direction == "fprop":
                self.impl.launch(self._compiled_kernel, q, k, v, *outputs, *metadata, float(scale), stream)
            else:
                self._compiled_kernel["dispatch"](
                    q,
                    k,
                    v,
                    grad,
                    out32,
                    lse,
                    scratch["delta"],
                    scratch["dq_acc"],
                    scratch["dk_acc"],
                    scratch["dv_acc"],
                    *metadata,
                    *outputs,
                    scratch["flags"],
                    float(scale),
                    self.sms,
                    stream,
                )
        return outputs


@lru_cache(maxsize=32)
def _plan(total, width, device, direction):
    plan = _QsaIndexedPlan(total, width, device, direction)
    plan.check_support()
    plan.compile()
    return plan


def qsa_indexed_forward(q, k, v, indices, scale=1 / 16):
    """Return ``(out, out_fp32, lse)`` from complete four-token groups and a tail."""
    total, width = _validate(q, k, v, indices, scale)
    plan = _plan(total, width, q.device, "fprop")
    with torch.cuda.device(q.device):
        outputs = (torch.empty_like(q), torch.empty_like(q, dtype=torch.float32), torch.empty((12, total), dtype=torch.float32, device=q.device))
        workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
        plan.execute(q, k, v, indices, outputs, scale, workspace)
    return TupleDict(zip(("out", "out_fp32", "lse"), outputs))


def qsa_indexed_backward(grad_out, q, k, v, out_fp32, lse, indices, scale=1 / 16, *, selection_mode="general", cu_seqlens=None, max_seq_len=None):
    """Return BF16 ``(dq, dk, dv)`` using forward's FP32 output and LSE.

    ``general`` preserves arbitrary group ordering, padding holes and empty rows.
    ``causal_topk`` declares the finite-input, top-512 complete four-token blocks
    plus causal-tail contract described in ``docs/operations/sparse_attention/qsa_indexed.md``.
    Its packed form requires device int32 ``cu_seqlens``; ``max_seq_len`` is an
    optional host upper bound. Values stay on the device and are caller contracts.
    """
    if not isinstance(selection_mode, str) or selection_mode not in ("general", "causal_topk"):
        raise ValueError("selection_mode must be 'general' or 'causal_topk'")
    if selection_mode == "general" and (cu_seqlens is not None or max_seq_len is not None):
        raise ValueError("cu_seqlens and max_seq_len require selection_mode='causal_topk'; general selection encodes its bounds in indices")
    if grad_out is None:
        raise ValueError("grad_out is required")
    total, width = _validate(q, k, v, indices, scale, grad_out, out_fp32, lse)
    if selection_mode == "causal_topk":
        if width != 2052 or scale != 1 / 16:
            raise NotImplementedError("causal_topk requires index width 2052 and scale=1/16")
        if max_seq_len is not None and (type(max_seq_len) is not int or not 1 <= max_seq_len <= total):
            raise ValueError(f"max_seq_len must be a host integer in [1, {total}]")
        if cu_seqlens is None and max_seq_len is not None and max_seq_len != total:
            raise ValueError("Without cu_seqlens, max_seq_len must equal the single sequence's token count")
        if cu_seqlens is not None:
            if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                raise ValueError("cu_seqlens must have shape [B + 1] with B >= 1")
            check_tensor("cu_seqlens", cu_seqlens, cu_seqlens.shape, torch.int32, q.device)
        # Explicit selection semantics admit this backend; tensor shape alone
        # must never route general selections (notably empty rows) into it.
        from cudnn.QSA.two_pass.api import _backward

        with torch.cuda.device(q.device):
            return _backward(q, k, v, out_fp32, grad_out, lse, indices, softmax_scale=scale, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)
    plan = _plan(total, width, q.device, "bprop")
    with torch.cuda.device(q.device):
        outputs = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
        workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
        plan.execute(q, k, v, indices, outputs, scale, workspace, grad_out, out_fp32, lse)
    return TupleDict(zip(("dq", "dk", "dv"), outputs))
