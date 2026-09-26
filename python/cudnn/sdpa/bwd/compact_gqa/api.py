# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compact backward for packed causal GQA batches."""

from __future__ import annotations

import ctypes
import operator
from numbers import Integral
import weakref
from contextlib import contextmanager

import torch
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old, cutedsl_requirement_error

_installed, _version = cutedsl_state()
if not _installed or cutedsl_too_old(_version):
    raise ImportError(cutedsl_requirement_error("CompactGqaBackward"))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict
from cudnn.sdpa import graph_analyzer

from ._blas import load_helper
from .dq import DirectCausalDq
from .launcher import FusedPipelined, FusedPacked
from .packed import PackedReduce
from .reduce import ReduceAndPointers, BandPointers, PackK, ClearDiagonal


class CompactGqaBackward(APIBase):
    """BF16 causal THD backward with runtime sequence lengths."""

    _producer_cls = FusedPipelined
    _reducer_cls = ReduceAndPointers
    _packed_reduce_cls = PackedReduce
    _packed_cls = FusedPacked
    _dq_cls = DirectCausalDq

    def __init__(self, q, k, v, o, do, lse, *, max_seqlen=None, query_rows=16384, groups=4, fast_store=True, ds_budget_bytes=8 * 1024**3, packed=True):
        super().__init__()
        assert query_rows >= 128 and query_rows % 128 == 0
        self.query_rows = query_rows
        self.packed = bool(packed)
        self._packed_batches = 0
        self._packed_sequences = 0
        self.ds_budget_bytes = operator.index(ds_budget_bytes)
        if self.ds_budget_bytes < 8 * 128 * 128 * 2:
            raise ValueError("dS budget must hold one 128-row tile.")
        self._ds_geometry = None
        assert groups in (1, 2, 4, 8)
        self.groups = groups
        self.fast_store = bool(fast_store)
        self._descs = tuple(self._make_tensor_desc(t, name=n) for n, t in zip(("q", "k", "v", "o", "do", "lse"), (q, k, v, o, do, lse)))
        self.device = q.device
        self.tokens = int(q.shape[0])
        self._stream = None
        self._handle = None
        self._workspace_ref = None
        self._capacity = self.tokens if max_seqlen is None else operator.index(max_seqlen)
        self._layout, self._scratch_bytes = self._workspace_layout(self._capacity)

    def _workspace_layout(self, capacity):
        if capacity < 1:
            raise ValueError("Workspace sequence capacity must be positive.")
        chunk_rows = min(self.query_rows, (capacity + 127) // 128 * 128)
        stats = (capacity + 127) // 128 * 128 + 256
        layout, scratch_bytes = {}, 0
        shapes = {
            "delta": ((8, stats + 128 * 128), torch.float32),
            "lse2": ((8, stats + 128 * 128), torch.float32),
            "ds": ((min(self.ds_budget_bytes // 2, 8 * chunk_rows * ((capacity + chunk_rows - 1) // chunk_rows * chunk_rows)),), torch.bfloat16),
            "dkp": ((capacity, self.groups, 256), torch.float32),
            "dvp": ((capacity, self.groups, 256), torch.float32),
            "kr": ((((chunk_rows + 4095) // 4096) * (stats - 256 + chunk_rows) * 256,), torch.bfloat16),
            "tables": ((128, 3, 8), torch.int64),
            "packed_meta": ((257,), torch.int32),
            "cu": ((2,), torch.int32),
            "blas": ((32 * 1024 * 1024,), torch.uint8),
        }
        for name, (shape, dtype) in shapes.items():
            size = 1
            for dim in shape:
                size *= dim
            count = size * dtype.itemsize
            offset = (scratch_bytes + 255) // 256 * 256
            layout[name] = (offset, count, shape, dtype)
            scratch_bytes = offset + count
        self._band_geometry(capacity, layout["ds"][1] // 2)
        return layout, scratch_bytes

    def _band_geometry(self, length, elements, query_start=0):
        # Fit only the keys reachable by this query band.
        end = (length + 127) // 128 * 128
        low, high = 1, min(self.query_rows, end - query_start) // 128
        best = 0
        while low <= high:
            middle = (low + high) // 2
            rows = middle * 128
            if 8 * rows * (query_start + rows) <= elements:
                best = rows
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            raise ValueError("dS budget is too small for this sequence capacity.")
        return best, query_start + best

    def _band_views(self, workspace_views, length, query_start=0):
        result = dict(workspace_views)
        rows, leading = self._band_geometry(max(1, length), result["ds"].numel(), query_start)
        result["ds"] = result["ds"].narrow(0, 0, 8 * rows * leading).view(1, 8, rows, leading)
        return result

    def _sequence_batches(self, offsets, lengths, spans, storage):
        pending = []
        maximum = 0

        def fits(items, width):
            count = len(items)
            total = items[-1][0] + items[-1][2] - items[0][0]
            return (
                count <= 128 and total <= self._capacity and count * 8 * width * width <= storage["ds"].numel() and count * width * 256 <= storage["kr"].numel()
            )

        for item in zip(offsets, lengths, spans):
            if item[2] == 0:
                continue
            if not self.packed or item[1] > 4096:
                if pending:
                    yield pending
                    pending, maximum = [], 0
                yield [item]
                continue
            width = max(maximum, (max(1, item[1]) + 127) // 128 * 128)
            if pending and not fits(pending + [item], width):
                yield pending
                pending, maximum = [], 0
                width = (max(1, item[1]) + 127) // 128 * 128
            pending.append(item)
            maximum = width
        if pending:
            yield pending

    def _execute_packed(self, inputs, outputs, storage, members):
        count = len(members)
        start = members[0][0]
        span = members[-1][0] + members[-1][2] - start
        lengths = [item[1] for item in members]
        rows = (max(1, max(lengths)) + 127) // 128 * 128
        offsets = [item[0] - start for item in members] + [span]
        w = dict(storage)
        w["ds"] = w["ds"].narrow(0, 0, count * 8 * rows * rows).view(count, 8, rows, rows)
        w["kr"] = w["kr"].narrow(0, 0, count * rows * 256).view(count, rows, 1, 256)
        w["tables"] = w["tables"].narrow(0, 0, count)
        metadata = storage["packed_meta"].narrow(0, 0, 2 * count + 1)
        host = torch.tensor(offsets + lengths, dtype=torch.int32, pin_memory=True)
        metadata.copy_(host, non_blocking=True)
        cu, lens = metadata[: count + 1], metadata[count + 1 :]
        data = [t.narrow(0, start, span) for t in inputs[:5]]
        li = inputs[5].narrow(0, start, span).view(span, 8)
        grads = [t.narrow(0, start, span) for t in outputs]
        partials = [w[name].narrow(0, 0, span) for name in ("dkp", "dvp")]
        if max(lengths):
            geometry = tuple(w["ds"].shape)
            if geometry != self._ds_geometry:
                w["ds"].zero_()
            self._ds_geometry = geometry
            values = (*data, li, w["lse2"], w["delta"], w["delta"], grads[0], *partials, cu, lens, w["ds"], w["kr"], w["tables"])
            args = tuple(from_dlpack(t, assumed_align=4 if i in (5, 12, 13) else 16, enable_tvm_ffi=True) for i, t in enumerate(values))
            self._packed_kernel(*args, None, max(lengths), 1 / 16, 0, self._cuda_stream)
            sizes = (ctypes.c_int * count)(*lengths)
            self._check_status(self._lib.gqa_packed_gemm(self._handle, w["tables"].data_ptr(), rows, sizes, count))
        values = (*partials, *grads, cu, lens)
        args = tuple(from_dlpack(t, assumed_align=4 if i >= 5 else 16, enable_tvm_ffi=True) for i, t in enumerate(values))
        self._packed_reduce(*args, max(item[2] for item in members), self._cuda_stream)
        self._packed_batches += 1
        self._packed_sequences += count

    def check_support(self):
        if torch.cuda.get_device_capability(self.device) != (10, 7):
            raise NotImplementedError("CompactGqaBackward requires SM107.")
        if self.tokens < 1:
            raise NotImplementedError("Expected a nonempty packed token buffer.")
        for desc, heads in zip(self._descs[:5], (8, 1, 1, 8, 8)):
            if desc.dtype != torch.bfloat16 or desc.shape != (self.tokens, heads, 256):
                raise NotImplementedError(f"{desc.name}: expected BF16 THD with H={heads}, D=256.")
            if desc.stride != (heads * 256, 256, 1) or desc.device != self.device:
                raise NotImplementedError(f"{desc.name}: expected contiguous input on {self.device}.")
        lse = self._descs[5]
        if lse.dtype != torch.float32 or lse.device != self.device:
            raise NotImplementedError("LSE must be FP32 on the input device.")
        if (lse.shape, lse.stride) not in (((self.tokens, 8), (8, 1)), ((self.tokens, 8, 1), (8, 1, 1))):
            raise NotImplementedError("LSE must use packed token-major storage.")
        # FE 1.29 uses the exact LSE metadata check above.
        classifier = getattr(graph_analyzer, "thd_stats_packing", None)
        if classifier is not None and classifier(lse.stride[1], lse.stride[0], 8) != "token_major":
            raise NotImplementedError("LSE must use packed token-major storage.")
        self._is_supported = True
        return True

    def scratch_workspace_bytes(self, max_seqlen=None):
        if max_seqlen is None:
            return self._scratch_bytes
        return self._workspace_layout(operator.index(max_seqlen))[1]

    @contextmanager
    def _context(self, current_stream):
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream if current_stream is None else int(current_stream)
            if self._stream is not None and stream != self._stream:
                raise ValueError("Use one plan per CUDA stream.")
            with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=self.device)):
                yield stream

    def compile(self, current_stream=None):
        self._ensure_support_checked()
        with self._context(current_stream) as stream:
            if self._compiled_kernel is not None:
                return self
            self._stream = stream

            def fake(dtype, shape, align=16):
                return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align)

            length = cute.sym_int(divisibility=1)
            span = cute.sym_int(divisibility=1)
            stats = cute.sym_int(divisibility=128)
            qf = fake(cutlass.BFloat16, (length, 8, 256))
            kf = fake(cutlass.BFloat16, (length, 1, 256))
            lsef = fake(cutlass.Float32, (length, 8), 4)
            df = fake(cutlass.Float32, (8, stats))
            pf = fake(cutlass.Float32, (length, self.groups, 256))
            dqf = fake(cutlass.BFloat16, (span, 8, 256))
            dkf = fake(cutlass.BFloat16, (span, 1, 256))
            cf = fake(cutlass.Int32, (2,), 4)
            chunk_rows = cute.sym_int(divisibility=128)
            dsf = fake(cutlass.BFloat16, (1, 8, chunk_rows, cute.sym_int(divisibility=128)))
            tf = fake(cutlass.Int64, (cute.sym_int(divisibility=1), 3, 8))
            krf = fake(cutlass.BFloat16, (cute.sym_int(divisibility=1), cute.sym_int(divisibility=128), 1, 256))
            self._cuda_stream = cuda.CUstream(stream)
            options = "--enable-tvm-ffi --opt-level 2 --gpu-arch sm_107a"
            args = (qf, kf, kf, qf, qf, lsef, df, df, df, qf, pf, pf, cf, dsf)
            compiled = cute.compile(
                self._producer_cls(self.groups, 4096, fast_store=self.fast_store),
                *args,
                None,
                cutlass.Int32(0),
                cutlass.Float32(1 / 16),
                cutlass.Int32(0),
                self._cuda_stream,
                options=options,
            )
            self._reduce = cute.compile(self._reducer_cls(4096, self.groups), kf, dqf, dsf, tf, pf, pf, dkf, dkf, self._cuda_stream, options=options)
            self._dq = cute.compile(
                self._dq_cls(),
                dsf,
                kf,
                dqf,
                cutlass.Int32(0),
                torch.cuda.get_device_properties(self.device).multi_processor_count // 2,
                self._cuda_stream,
                options=options,
            )
            if self.packed:
                batch = cute.sym_int(divisibility=1)
                cuf = fake(cutlass.Int32, (cute.sym_int(divisibility=1),), 4)
                lenf = fake(cutlass.Int32, (batch,), 4)
                short_rows = cute.sym_int(divisibility=128)
                packed_ds = fake(cutlass.BFloat16, (batch, 8, short_rows, short_rows))
                packed_k = fake(cutlass.BFloat16, (batch, short_rows, 1, 256))
                packed_tables = fake(cutlass.Int64, (batch, 3, 8))
                pargs = (*args[:12], cuf, lenf, packed_ds, packed_k, packed_tables)
                self._packed_kernel = cute.compile(
                    self._packed_cls(self.groups, 4096, fast_store=self.fast_store),
                    *pargs,
                    None,
                    cutlass.Int32(0),
                    cutlass.Float32(1 / 16),
                    cutlass.Int32(0),
                    self._cuda_stream,
                    options=options,
                )
                self._packed_reduce = cute.compile(
                    self._packed_reduce_cls(self.groups), pf, pf, dqf, dkf, dkf, cuf, lenf, cutlass.Int32(0), self._cuda_stream, options=options
                )
            self._lib = load_helper()
            status = ctypes.c_int()
            self._handle = self._lib.gqa_create(stream, None, 0, ctypes.byref(status))
            self._check_status(status.value)
            self._compiled_kernel = compiled
            return self

    @staticmethod
    def _check_status(status):
        if status:
            raise RuntimeError(f"cuBLAS status {status}")

    def _views(self, workspace):
        if (
            workspace.dtype != torch.uint8
            or workspace.device != self.device
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < self._scratch_bytes
            or workspace.data_ptr() % 256
        ):
            raise ValueError(f"Expected {self._scratch_bytes} aligned CUDA workspace bytes.")
        return {name: workspace.narrow(0, offset, size).view(dtype).view(shape) for name, (offset, size, shape, dtype) in self._layout.items()}

    def initialize_workspace(self, workspace, current_stream=None, *, max_seqlen=None):
        """Zero compact dS once per workspace allocation."""
        self.compile(current_stream)
        with self._context(current_stream):
            if max_seqlen is not None:
                self._capacity = operator.index(max_seqlen)
                self._layout, self._scratch_bytes = self._workspace_layout(self._capacity)
            self._views(workspace)["ds"].zero_()
            self._ds_geometry = None
            self._workspace_ref = weakref.ref(workspace)

    def _validate_runtime(self, inputs, outputs):
        tokens = inputs[0].shape[0]
        if tokens < 1:
            raise ValueError("Expected a nonempty packed token buffer.")
        for tensor, desc in zip(inputs, self._descs):
            if (
                tensor.dtype != desc.dtype
                or tuple(tensor.shape) != (tokens, *desc.shape[1:])
                or tuple(tensor.stride()) != desc.stride
                or tensor.device != desc.device
            ):
                raise ValueError(f"{desc.name}: runtime metadata differs from the plan.")
        for tensor, desc in zip(outputs, self._descs[:3]):
            if (
                tensor.dtype != desc.dtype
                or tuple(tensor.shape) != (tokens, *desc.shape[1:])
                or tuple(tensor.stride()) != desc.stride
                or tensor.device != desc.device
            ):
                raise ValueError(f"d{desc.name}: invalid output metadata.")
        for tensor in (*inputs[:5], *outputs):
            if tensor.data_ptr() % 16:
                raise ValueError("Input and gradient pointers require 16-byte alignment.")
        for i, output in enumerate(outputs):
            lo, hi = output.data_ptr(), output.data_ptr() + output.numel() * output.element_size()
            for other in (*inputs, *outputs[:i]):
                a, b = other.data_ptr(), other.data_ptr() + other.numel() * other.element_size()
                if lo < b and a < hi:
                    raise ValueError("Gradient outputs must not overlap inputs or each other.")

    def execute(self, q, k, v, o, do, lse, dq, dk, dv, workspace, current_stream=None, *, sequence_offsets=None, sequence_lengths=None):
        if self._compiled_kernel is None or not self._handle:
            raise RuntimeError("Compile the plan before execute.")
        if self._workspace_ref is None or self._workspace_ref() is not workspace:
            raise ValueError("Initialize this workspace before execute.")
        inputs, outputs = (q, k, v, o, do, lse), (dq, dk, dv)
        self._validate_runtime(inputs, outputs)
        if torch.is_tensor(sequence_offsets) or torch.is_tensor(sequence_lengths):
            raise TypeError("Packing metadata must be host integer sequences.")
        for values in (sequence_offsets, sequence_lengths):
            if values is not None and any(not isinstance(n, Integral) for n in values):
                raise TypeError("Packing metadata must contain host integers.")
        offsets = (0, q.shape[0]) if sequence_offsets is None else tuple(operator.index(n) for n in sequence_offsets)
        spans = tuple(b - a for a, b in zip(offsets, offsets[1:]))
        lengths = spans if sequence_lengths is None else tuple(operator.index(n) for n in sequence_lengths)
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != q.shape[0] or len(lengths) != len(spans):
            raise ValueError("Packing must cover the complete token buffer.")
        if any(span < length or length < 0 for span, length in zip(spans, lengths)):
            raise ValueError("Sequence lengths must fit their physical spans.")
        if max(lengths) > self._capacity:
            raise ValueError("Initialize a workspace with sufficient sequence capacity.")
        with self._context(current_stream):
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("CompactGqaBackward does not support CUDA graphs.")
            w = self._views(workspace)
            low, high = workspace.data_ptr(), workspace.data_ptr() + self._scratch_bytes
            for tensor in (*inputs, *outputs):
                a, b = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
                if low < b and a < high:
                    raise ValueError("Workspace must not overlap inputs or outputs.")
            self._check_status(self._lib.gqa_set_workspace(self._handle, w["blas"].data_ptr(), w["blas"].numel()))
            storage = w
            for members in self._sequence_batches(offsets, lengths, spans, storage):
                if len(members) > 1:
                    self._execute_packed(inputs, outputs, storage, members)
                    continue
                start, length, span = members[0]
                if span == 0:
                    continue
                w = self._band_views(storage, length)
                qi, ki, vi, oi, doi = [t.narrow(0, start, length) for t in inputs[:5]]
                li = lse.narrow(0, start, length).view(length, 8)
                dqi, dki, dvi = [t.narrow(0, start, span) for t in outputs]
                dkp, dvp = [w[name].narrow(0, 0, length) for name in ("dkp", "dvp")]
                if length:
                    query_start = 0
                    while query_start < length:
                        w = self._band_views(storage, length, query_start)
                        self._ds_geometry = ("natural", *w["ds"].shape)
                        values = (qi, ki, vi, oi, doi, li, w["lse2"], w["delta"], w["delta"], dqi.narrow(0, 0, length), dkp, dvp, w["cu"], w["ds"])
                        args = tuple(from_dlpack(t, assumed_align=4 if i in (5, 12) else 16, enable_tvm_ffi=True) for i, t in enumerate(values))
                        dq_args = tuple(from_dlpack(t, assumed_align=16, enable_tvm_ffi=True) for t in (w["ds"], ki, dqi))
                        self._compiled_kernel(*args, None, length, 1 / 16, query_start, self._cuda_stream)
                        self._dq(*dq_args, query_start, self._cuda_stream)
                        query_start += w["ds"].shape[2]
                values = (ki, dqi, w["ds"], w["tables"], dkp, dvp, dki, dvi)
                args = tuple(from_dlpack(t, assumed_align=16, enable_tvm_ffi=True) for t in values)
                self._reduce(*args, self._cuda_stream)

    def close(self):
        if self._handle:
            with torch.cuda.device(self.device):
                self._check_status(self._lib.gqa_destroy(self._handle))
            self._handle = None


def compact_gqa_backward(q, k, v, o, do, lse, *, plan, workspace, current_stream=None, sequence_offsets=None, sequence_lengths=None):
    """Execute a prepared plan and return newly allocated gradients."""
    with plan._context(current_stream):
        outputs = [torch.empty_like(t) for t in (q, k, v)]
        plan.execute(q, k, v, o, do, lse, *outputs, workspace, current_stream, sequence_offsets=sequence_offsets, sequence_lengths=sequence_lengths)
    return TupleDict(dq=outputs[0], dk=outputs[1], dv=outputs[2])
